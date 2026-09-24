//! Recognizing a constant-true marginal level.
//!
//! Sibling of `conjunction.rs`.

use super::*;

use super::sparse::is_self_conjunction;
use crate::test_helpers::clause_to_tdd;


use crate::diagram::{
    CountOverflow, PairRange, TddLevel,
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
    ct.become_marginal(vec![1u128 << 127], None);
    assert!(
        level_marginal_is_constant_true(&ct, 127),
        "counts[0] == 2^127 at subvars=127 must be recognised as constant-true"
    );

    // Same subvars, one below the target: not constant-true.
    let mut not_ct = TddLevel::new();
    not_ct.become_marginal(vec![(1u128 << 127) - 1], None);
    assert!(
        !level_marginal_is_constant_true(&not_ct, 127),
        "counts[0] one below 2^127 must not be recognised as constant-true"
    );
}

#[test]
fn test_level_marginal_is_constant_true_small_subvars_sentinel_never_true() {
    // subvars < 128 with the u128::MAX overflow sentinel: per the function's
    // doc comment, this can never be constant-true (2^subvars <= 2^127 <
    // u128::MAX, so an overflowed real count is strictly greater than the
    // target) — decided directly by the `subvars < 128` branch, WITHOUT
    // consulting the big table, even when one is present and would
    // (incorrectly) "match" if this were misread as a `subvars >= 128` case.
    let mut level = TddLevel::new();
    level.become_marginal(
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
    level.become_marginal(vec![u128::MAX], Some([(0u32, target)].into_iter().collect()));
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
    mismatched_big.become_marginal(
        vec![u128::MAX],
        Some([(0u32, BigUint::from(5u32))].into_iter().collect()),
    );
    assert!(
        !level_marginal_is_constant_true(&mismatched_big, subvars),
        "big-table value != 2^subvars must not be constant-true"
    );

    // (2) no big table at all (`marginal_counts_big` is `None`).
    let mut no_big_table = TddLevel::new();
    no_big_table.become_marginal(vec![u128::MAX], None);
    assert!(
        !level_marginal_is_constant_true(&no_big_table, subvars),
        "missing big-table side entry must not be constant-true"
    );

    // (3) big table allocated but carrying no entry for slot 0. Sparse storage
    // makes "absent entry" the encoding of "fits the fast lane", so this must
    // read exactly like (2) — the `Some(empty)` shape is the one a caller can
    // still hand over after every overflow entry was taken back out.
    let mut none_entry = TddLevel::new();
    none_entry.become_marginal(vec![u128::MAX], Some(CountOverflow::default()));
    assert!(
        !level_marginal_is_constant_true(&none_entry, subvars),
        "missing big-table entry must not be constant-true"
    );

    // (4) c0 never overflowed at all (no sentinel): the real value fits in
    // u128, which is always < 2^128, so it can't reach the target —
    // disqualified before ever consulting the big table.
    let mut no_overflow = TddLevel::new();
    no_overflow.become_marginal(vec![12345u128], None);
    assert!(
        !level_marginal_is_constant_true(&no_overflow, subvars),
        "c0 != u128::MAX at subvars >= 128 must not be constant-true"
    );
}

// ── T7c: self-conjunction — shortcut vs. general-path branch coverage ──────

#[test]
fn test_apply_and_self_conjunction_shortcut_vs_general_path() {
    let eng = &crate::Engine::new();
    // `apply_and` (via `conjoin_owned`) gates the `f ∧ f = f` structural
    // shortcut on `is_self_conjunction` (`conjoin/sparse/mod.rs`). Calling it directly
    // on the exact operands then fed to `apply_and` is a genuine
    // observability hook — not a guess — for which of the two branches
    // (shortcut vs. general product construction) a given call takes.
    let vtree = Arc::new(Vtree::balanced(4));
    let f = vec![Literal::pos(VarId(1)), Literal::pos(VarId(3))];
    let g = vec![Literal::neg(VarId(2)), Literal::pos(VarId(4))];
    let mut tdd = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    tdd = apply_and(tdd, t2);
    tdd.minimize().unwrap();
    let expected_mc = tdd.model_count().unwrap();

    // ── Branch 1: must take the shortcut ────────────────────────────────
    // Byte-identical, non-marginal clones satisfy `is_self_conjunction` by
    // construction (equal output, equal per-level nodes/pairs/ranges).
    let shortcut_lhs = tdd.clone();
    let shortcut_rhs = tdd.clone();
    assert!(
        is_self_conjunction(&shortcut_lhs, &shortcut_rhs),
        "byte-identical clones must satisfy the shortcut predicate"
    );
    let mut shortcut_result = apply_and(shortcut_lhs, shortcut_rhs);
    shortcut_result.minimize().unwrap();
    assert_eq!(
        shortcut_result.model_count().unwrap(), expected_mc,
        "shortcut path (is_self_conjunction=true): f \u{2227} f must equal f"
    );

    // ── Branch 2: must take the general path ────────────────────────────
    // same represented function, but one clone carries one extra,
    // completely UNREFERENCED `ranges` side-table entry at the root level —
    // mirrors `conjoin::sparse::tests::self_conjunction::
    // `differing_ext_blocks_shortcut``, which pins that ``is_self_conjunction``
    // treats a differing `ranges` table as a structural difference even when
    // `nodes`/`pairs` agree. No node encodes a reference to the new entry
    // (append-only, past the end of the existing table), so the represented
    // function is completely unchanged — only the raw structural comparison
    // differs, which is exactly what forces the shortcut predicate to
    // `false` and routes `apply_and` through the general product-
    // construction path instead.
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let general_lhs = tdd.clone();
    let mut general_rhs = tdd.clone();
    general_rhs.levels[root.idx()].ranges.push(PairRange { start: 0, len: 2 });
    assert!(
        !is_self_conjunction(&general_lhs, &general_rhs),
        "operand with a differing (unreferenced) ranges entry must NOT satisfy the shortcut predicate"
    );
    let mut general_result = apply_and(general_lhs, general_rhs);
    general_result.minimize().unwrap();
    assert_eq!(
        general_result.model_count().unwrap(), expected_mc,
        "general path (is_self_conjunction=false): f \u{2227} f must still equal f"
    );
}

