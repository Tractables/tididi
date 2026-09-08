//! Recognizing a constant-true marginal level, and the bounded spine merge.
//!
//! Sibling of `apply_tests.rs`.

use super::*;

use crate::engine::Engine;
use super::sparse::is_self_conjunction;
use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::query::model_count;
use crate::diagram::{
    BigSide, ExtMulti, Tdd, TddLevel,
};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use num_bigint::BigUint;

#[test]
fn test_level_marginal_is_constant_true_small_subvars() {
    // subvars = 127: the last value on the `subvars < 128` fast path (target
    // 2^127 fits in u128 — u128::MAX = 2^128 - 1 — so no BigUint side table
    // is consulted).
    let mut ct = TddLevel::new();
    ct.make_marginal(vec![1u128 << 127], None);
    assert!(
        level_marginal_is_constant_true(&ct, 127),
        "counts[0] == 2^127 at subvars=127 must be recognised as constant-true"
    );

    // Same subvars, one below the target: NOT constant-true.
    let mut not_ct = TddLevel::new();
    not_ct.make_marginal(vec![(1u128 << 127) - 1], None);
    assert!(
        !level_marginal_is_constant_true(&not_ct, 127),
        "counts[0] one below 2^127 must not be recognised as constant-true"
    );
}

#[test]
fn test_level_marginal_is_constant_true_small_subvars_sentinel_never_true() {
    // subvars < 128 with the u128::MAX overflow sentinel: per the function's
    // doc comment, this can NEVER be constant-true (2^subvars <= 2^127 <
    // u128::MAX, so an overflowed real count is strictly greater than the
    // target) — decided directly by the `subvars < 128` branch, WITHOUT
    // consulting the big table, even when one is present and would
    // (incorrectly) "match" if this were misread as a `subvars >= 128` case.
    let mut level = TddLevel::new();
    level.make_marginal(
        vec![u128::MAX],
        // present but must be ignored
        Some([(0u32, BigUint::from(1u32) << 100usize)].into_iter().collect()),
    );
    assert!(
        !level_marginal_is_constant_true(&level, 100),
        "sentinel c0 == u128::MAX at subvars < 128 must never read as constant-true"
    );
}

#[test]
fn test_level_marginal_is_constant_true_large_subvars_matching_big() {
    // subvars = 128: target 2^128 exceeds u128::MAX, so this is on the
    // BigUint side-table branch. Sentinel present and the big-table value
    // matches 2^128 exactly -> constant-true.
    let target = BigUint::from(1u32) << 128usize;
    let mut level = TddLevel::new();
    level.make_marginal(vec![u128::MAX], Some([(0u32, target)].into_iter().collect()));
    assert!(
        level_marginal_is_constant_true(&level, 128),
        "matching big-table value at subvars=128 must be recognised as constant-true"
    );
}

#[test]
fn test_level_marginal_is_constant_true_large_subvars_disqualified() {
    // subvars = 128: every way the `subvars >= 128` branch can be
    // disqualified must report `false`, even with the sentinel present.
    let subvars = 128u32;

    // (1) big-table value present but doesn't match 2^128.
    let mut mismatched_big = TddLevel::new();
    mismatched_big.make_marginal(
        vec![u128::MAX],
        Some([(0u32, BigUint::from(5u32))].into_iter().collect()),
    );
    assert!(
        !level_marginal_is_constant_true(&mismatched_big, subvars),
        "big-table value != 2^subvars must not be constant-true"
    );

    // (2) no big table at all (`marginal_counts_big` is `None`).
    let mut no_big_table = TddLevel::new();
    no_big_table.make_marginal(vec![u128::MAX], None);
    assert!(
        !level_marginal_is_constant_true(&no_big_table, subvars),
        "missing big-table side entry must not be constant-true"
    );

    // (3) big table allocated but carrying no entry for slot 0. Sparse storage
    // makes "absent entry" the encoding of "fits the fast lane", so this must
    // read exactly like (2) — the `Some(empty)` shape is the one a caller can
    // still hand over after every overflow entry was taken back out.
    let mut none_entry = TddLevel::new();
    none_entry.make_marginal(vec![u128::MAX], Some(BigSide::default()));
    assert!(
        !level_marginal_is_constant_true(&none_entry, subvars),
        "missing big-table entry must not be constant-true"
    );

    // (4) c0 never overflowed at all (no sentinel): the real value fits in
    // u128, which is always < 2^128, so it can't reach the target —
    // disqualified before ever consulting the big table.
    let mut no_overflow = TddLevel::new();
    no_overflow.make_marginal(vec![12345u128], None);
    assert!(
        !level_marginal_is_constant_true(&no_overflow, subvars),
        "c0 != u128::MAX at subvars >= 128 must not be constant-true"
    );
}

// ── T7c: self-conjunction — shortcut vs. general-path branch coverage ──────

#[test]
fn test_apply_and_self_conjunction_shortcut_vs_general_path() {
    let eng = &crate::engine::Engine::new();
    // `apply_and` (via `apply_and_fallible_inner`) and `apply_and`
    // (via `conjoin_owned`) both gate the
    // `f ∧ f = f` structural shortcut on the SAME predicate,
    // `is_self_conjunction` (`conjoin/sparse.rs`). Calling it directly
    // on the exact operands then fed to `apply_and` is a genuine
    // observability hook — not a guess — for which of the two branches
    // (shortcut vs. general product construction) a given call takes.
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = vec![Literal::pos(VarId(0)), Literal::pos(VarId(2))];
    let c2 = vec![Literal::neg(VarId(1)), Literal::pos(VarId(3))];
    let mut tdd = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    tdd = apply_and(tdd, t2);
    minimize(&mut tdd);
    let expected_mc = model_count(&tdd);

    // ── Branch 1: MUST take the shortcut ────────────────────────────────
    // Byte-identical, non-marginal clones satisfy `is_self_conjunction` by
    // construction (equal output, equal per-level nodes/pairs/ext).
    let shortcut_lhs = tdd.clone();
    let shortcut_rhs = tdd.clone();
    assert!(
        is_self_conjunction(&shortcut_lhs, &shortcut_rhs),
        "byte-identical clones must satisfy the shortcut predicate"
    );
    let mut shortcut_result = apply_and(shortcut_lhs, shortcut_rhs);
    minimize(&mut shortcut_result);
    assert_eq!(
        model_count(&shortcut_result), expected_mc,
        "shortcut path (is_self_conjunction=true): f \u{2227} f must equal f"
    );

    // ── Branch 2: MUST take the general path ────────────────────────────
    // Same represented function, but one clone carries one extra,
    // completely UNREFERENCED `ext` side-table entry at the root level —
    // mirrors `conjoin::sparse::a4_self_conjunction_tests::
    // differing_ext_blocks_shortcut`, which pins that `is_self_conjunction`
    // treats a differing `ext` table as a structural difference even when
    // `nodes`/`pairs` agree. No node encodes a reference to the new entry
    // (append-only, past the end of the existing table), so the represented
    // function is completely unchanged — only the raw structural comparison
    // differs, which is exactly what forces the shortcut predicate to
    // `false` and routes `apply_and` through the general product-
    // construction path instead.
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let general_lhs = tdd.clone();
    let mut general_rhs = tdd.clone();
    general_rhs.levels[root.idx()].ext.push(ExtMulti { start: 0, len: 2 });
    assert!(
        !is_self_conjunction(&general_lhs, &general_rhs),
        "operand with a differing (unreferenced) ext entry must NOT satisfy the shortcut predicate"
    );
    let mut general_result = apply_and(general_lhs, general_rhs);
    minimize(&mut general_result);
    assert_eq!(
        model_count(&general_result), expected_mc,
        "general path (is_self_conjunction=false): f \u{2227} f must still equal f"
    );
}


// ── Spine-bounded merge: differential against the generic apply ──────────────

/// Assert two diagrams are bit-identical, level by level. The spine-bounded
/// merge promises exactly this (not merely the same function), so the
/// comparison is structural — every level field — with the level index and
/// the first differing field named on failure.
fn assert_tdds_identical(expected: &Tdd, got: &Tdd, what: &str) {
    assert_eq!(expected.output, got.output, "{what}: output node differs");
    assert_eq!(expected.levels.len(), got.levels.len(), "{what}: level count differs");
    for (i, (a, b)) in expected.levels.iter().zip(got.levels.iter()).enumerate() {
        assert_eq!(a.is_marginal(), b.is_marginal(), "{what}: level {i} is_marginal");
        assert_eq!(a.width(), b.width(), "{what}: level {i} width");
        assert!(a.nodes == b.nodes, "{what}: level {i} nodes differ");
        assert!(a.pairs == b.pairs, "{what}: level {i} pairs differ");
        assert!(a.ext == b.ext, "{what}: level {i} ext differ");
        assert_eq!(a.marg_flags, b.marg_flags, "{what}: level {i} marg_flags");
        assert_eq!(a.marginal_counts, b.marginal_counts, "{what}: level {i} marginal_counts");
        assert_eq!(a.marginal_counts_big, b.marginal_counts_big, "{what}: level {i} marginal_counts_big");
        assert_eq!(a.n_tombstones, b.n_tombstones, "{what}: level {i} n_tombstones");
        assert_eq!(a.weight_width, b.weight_width, "{what}: level {i} weight_width");
        assert_eq!(a.retired_marg_slots, b.retired_marg_slots, "{what}: level {i} retired_marg_slots");
    }
}

/// The spine-bounded merge (`conjoin_batch`) must produce the
/// bit-identical diagram the generic owned apply produces, on every batch it
/// accepts. This is the claim the restricted path rests on (its module doc
/// spells out the one deliberate FP1/FP2 divergence and why it is invisible in
/// the output); the test pins it on a run of merges into a growing accumulator,
/// each batch's spine derived exactly as the batch builder derives it
/// (`walk_mark_spine` over the folded clauses' variables).
#[test]
fn spine_bounded_merge_matches_generic_apply() {
    let eng = Engine::new();
    use crate::apply::conjoin::{
        conjoin_batch, conjoin_owned, BatchMerge,
    };
    use crate::apply::conjoin_clause::walk_mark_spine;

    let nvars = 20u32;
    let vtree = Arc::new(Vtree::balanced(nvars));
    let mut state: u64 = 0x5b1e_ba7c_4a5e_ed01;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let random_clause = |rng: &mut dyn FnMut() -> u64| -> Vec<Literal> {
        let len = 2 + (rng() % 2) as usize;
        let mut lits: Vec<Literal> = Vec::new();
        while lits.len() < len {
            let v = VarId((rng() % nvars as u64) as u32);
            if lits.iter().any(|l| l.var == v) {
                continue;
            }
            lits.push(Literal::new(v, rng() % 2 == 0));
        }
        lits
    };

    // A moderately sized accumulator: eight clauses folded generically, then
    // minimized — the state the batch builder's accumulator is in between
    // merges.
    let mut acc = constant_one(&eng, &vtree);
    for _ in 0..8 {
        let c = clause_to_tdd(&eng, &vtree, &random_clause(&mut rng));
        acc = apply_and(acc, c);
    }
    minimize(&mut acc);

    let widest_internal = |t: &Tdd| -> usize {
        (0..vtree.num_nodes())
            .filter(|&i| !vtree.node(VtreeIdx(i as u32)).is_leaf())
            .map(|i| t.levels[i].width())
            .max()
            .unwrap_or(0)
    };

    let mut merged = 0usize;
    for batch_no in 0..12 {
        // A small batch: two or three clauses, and its spine — the ancestor
        // closure of every clause variable's leaf.
        let mut batch = constant_one(&eng, &vtree);
        let mut on_spine = vec![false; vtree.num_nodes()];
        let mut spine: Vec<VtreeIdx> = Vec::new();
        for _ in 0..(2 + (rng() % 2)) {
            let clause = random_clause(&mut rng);
            walk_mark_spine(&vtree, &clause, &mut on_spine, Some(&mut spine));
            let c = clause_to_tdd(&eng, &vtree, &clause);
            batch = apply_and(batch, c);
        }
        if batch.is_zero() {
            continue;
        }

        let expected = conjoin_owned(&eng, acc.clone(), batch.clone(), None)
            .expect("generic merge must not run out of budget in this test");
        let restricted = conjoin_batch(
            &eng,
            acc.clone(),
            batch,
            &Spine {
                levels: &spine,
                marg_parents: &[],
                acc_max_width: acc.max_width(),
                acc_widest_internal: widest_internal(&acc),
            },
        )
        .expect("restricted merge must not run out of budget in this test");
        let got = match restricted {
            BatchMerge::Merged(t, _) => t,
            BatchMerge::Declined(..) => {
                panic!("batch {batch_no}: the restricted merge declined an accepted-shape batch")
            }
        };
        assert_tdds_identical(&expected, &got, &format!("batch {batch_no}"));
        assert_eq!(model_count(&expected), model_count(&got), "batch {batch_no}: model count");
        merged += 1;

        // Continue from the restricted result, minimized as the batch builder
        // would between merges.
        acc = got;
        minimize(&mut acc);
        if acc.is_zero() {
            break;
        }
    }
    assert!(merged >= 6, "too few batches exercised the restricted path ({merged})");
}
