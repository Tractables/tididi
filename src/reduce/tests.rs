use super::*;
use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::query::model_count;
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};
use std::sync::Arc;

#[test]
fn test_minimize_constant_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let mut tdd = constant_one(eng, &vtree);
    minimize(&mut tdd);
    // Internal levels have width 1; leaf levels are marginal (width 3)
    for (t, _left, _right) in vtree.internal_bottomup() {
        assert_eq!(tdd.level(t).width(), 1);
    }
}

#[test]
fn test_minimize_single_clause() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let clause = vec![Literal::pos(VarId(0))];
    let mut tdd = clause_to_tdd(eng, &vtree, &clause);
    let count_before = model_count(&tdd);
    minimize(&mut tdd);
    let count_after = model_count(&tdd);
    assert_eq!(count_before, count_after);
}

#[test]
fn test_minimize_reduces_width_after_apply() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);

    let width_before = result.max_width();

    let count_before = model_count(&result);
    minimize(&mut result);
    let count_after = model_count(&result);

    // Width should not increase after minimization
    assert!(
        result.max_width() <= width_before,
        "Width should not increase after minimization ({} → {})",
        width_before, result.max_width()
    );
    // Model count preserved
    assert_eq!(count_before, count_after);
}

#[test]
fn test_minimize_preserves_unsat() {
    let eng = &crate::engine::Engine::new();
    // Single variable: x ∧ ¬x = UNSAT
    let vtree = Arc::new(Vtree::balanced(1));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);
    minimize(&mut result);

    assert_eq!(model_count(&result), 0u64.into());
}

#[test]
fn test_minimize_unsat_2vars_width() {
    let eng = &crate::engine::Engine::new();
    // 2 variables: (x0) AND (not-x0) = UNSAT
    // The canonical TDD for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(2));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);

    minimize(&mut result);

    assert_eq!(model_count(&result), 0u64.into());
    assert_eq!(
        result.max_width(),
        0,
        "Canonical TDD for false should have width 0 (ZERO sentinel), got {}",
        result.max_width()
    );
}

#[test]
fn test_minimize_unsat_3vars_width() {
    let eng = &crate::engine::Engine::new();
    // 3 variables: (x0) AND (not-x0) = UNSAT
    // The canonical TDD for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(3));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);

    minimize(&mut result);

    assert_eq!(model_count(&result), 0u64.into());
    assert_eq!(
        result.max_width(),
        0,
        "Canonical TDD for false should have width 0 (ZERO sentinel), got {}",
        result.max_width()
    );
}

#[test]
fn test_minimize_sat_2vars_reduces_width() {
    let eng = &crate::engine::Engine::new();
    // (x0) ∧ (x1) over 2 vars → 1 model (x0=1, x1=1)
    // After apply: width 4. After minimize: should have width < 4.
    let vtree = Arc::new(Vtree::balanced(2));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::pos(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);

    let count_before = model_count(&result);
    minimize(&mut result);
    let count_after = model_count(&result);

    assert_eq!(count_before, count_after);
    // With pruned clause TDDs, the product may already be minimal (width 1).
    // Just verify the result is correct and small.
    assert!(
        result.max_width() <= 2,
        "Minimized (x0) ∧ (x1) should have small width, got {}",
        result.max_width()
    );
}

// ── Leaf twin contraction regression tests ────────────────────────────────
//
// These guard the (Pos_x, S) + (Neg_x, S) → (One_x, S) rewrite in
// `contract_leaf::contract_leaf_twins`, and the leaf-mode determinism
// property the rewrite preserves.

// ── Marginal-level twin contraction ──────────────────────────────────────
//
// When a vtree level is made marginal (`TddLevel::make_marginal`), its
// node/pair structure is dropped and replaced with per-node model counts.
// Later parent conjunctions can reshape the parent's pair list so that two
// marginal entries end up in identical `(parent_idx, sibling_idx)`
// multisets — i.e. they become twins. Contracting those twins is sound
// under TDD determinism (twins are disjoint within shared parent contexts,
// so `m_{A∨B} = m_A + m_B`); at a marginal level, that's exactly summing
// `marginal_counts` (with u128-overflow promotion into the BigUint
// side-table) and deduping the parent pairs.

// ── Order-independent twin detection (regression) ─────────────────────────
//
// `find_twin_groups` identifies twins by each node's *signature* — the SET of
// `(parent_idx, sibling_idx)` contexts referencing it. The scatter that builds
// these signatures writes entries in parent-pair STORAGE order, which is
// arbitrary. Comparing signature slices with an order-sensitive `==` therefore
// misses two genuine twins whose parents stored their pairs in different orders
// — a silent under-contraction that leaves the diagram non-minimal. The
// comparison canonicalizes each signature (sorts the slice) before `==`, so
// detection is order-independent.
//
// Both tests construct that hazard: at the root, twin node A's pairs are listed
// in one sibling order and twin node B's in the reverse order, so A's and B's
// raw signature slices differ as *sequences* while being equal as *sets*. On
// the pre-fix order-sensitive `==` the twins go undetected (`v_left` width does
// not shrink); with the fix they contract. Both FAIL on the pre-fix commit and
// pass after — i.e. they pin the fixed behavior (the regression-test contract).
//
// Why width ≥ 3, not width 2: a level with exactly two scrambled twins A,B can
// always *self-canonicalize*. The two siblings A,B are paired with become twins
// themselves (each referenced by the same `{A,B}` set), collapse via the
// cascade, and that collapse rewrites A's and B's signatures down to a single
// shared sibling — which even the order-sensitive `==` then matches. To keep
// the bug observable we add a distinct node (C / C,D) that references one
// sibling and not the others, breaking the sibling symmetry so no cascade
// re-canonicalizes the signatures. The under-contraction is therefore a
// width-≥3 phenomenon, and these tests target exactly the hash-bucket exact
// comparison that handles `child_width >= 3`.
//
// The merge preserves `model_count` even though the hand-built TDDs are not
// deterministic: when A,B merge, the parent's now-duplicate `(AB, sibling)`
// pairs dedup, and `|AB| = |A| + |B|` makes the deduped term equal to the sum
// of the two originals — so the count is invariant. We assert it as a guard.

// ── OverBudget safety in contract_twins ───────────────────────────────────
//
// An `ApplyError::OverBudget` raised part-way through `contract_twins`' group-
// merge loop must never poison the model count. Two windows:
//  - Cross-group: group g's reserve fails after groups 0..g-1 already grew
//    their survivors and the parent hasn't been rewritten — a silent overcount.
//    The fix hoists ONE grand reserve before the loop, so a failure bails
//    transactionally (no mutation, count unchanged, `poisoned == false`).
//  - Mid parent-rewrite: a fallible push during the parent rewrite leaves
//    the diagram structurally inconsistent — irrecoverable, so it sets
//    `tdd.poisoned` and any later count extraction panics.

// ── Dirty-worklist restoration on Err ─────────────────────────────────────
//
// A top-down contraction sweep `mem::take`s `tdd.dirty.contract` into a
// topo-heap. If a mid-sweep `Err` fires (race-lane `Deadline` preemption, or
// `OverBudget` from `contract_twins`), every parent that had not yet been
// popped — plus the one being processed — must be restored to
// `tdd.dirty.contract`, or those levels keep stale contexts and are never
// re-contracted (a permanent canonicity/size leak; sound but a leak). On
// unfixed HEAD the taken worklist is dropped, so `dirty_contract` is empty
// after the Err — the assertions below fail.

// `test_contract_tolerates_tombstones` and
// `test_tombstone_tolerant_readers_and_prune_reclaim` moved to
// `tests/tdd_search_minimize_compile.rs`: both depend on CNF parsing and on a
// compile loop, neither of which lives in this crate.
// `minimize::contract`/`minimize::prune` were promoted `pub mod` (with
// `contract_all_twins_topdown`/`prune_unreachable` promoted `pub`) so the
// moved copies can reach them from the external test crate.

// ── Prune value-merge → twin mint regression ──────────────────────────────
//
// If two boundary-parent nodes p = [(X1, c1), (X2, d)] and
// q = [(X1, c2), (X2, d)] have c1 ≠ c2 as slot indices but equal stored
// values (only possible for BIG counts — small ones are inline post-tagger),
// `prune_marg_slots`'s value-dedup merges c1 and c2 onto one slot and
// rewrites both parent refs to it. That makes p and q raw-identical twins —
// but at this point contract has already run and won't run again (pre-fix).
// The no-twins postcondition at minimize exit is then violated. The fix
// iterates contract→prune until prune reports values_merged == 0.
//
// This test MUST FAIL on the unfixed code (verified by running the broken
// contract→prune sequence manually and asserting check_no_twins fails).

// ── Marg-sibling fold-allowed regression ──────────────────────────────────
//
// When two content-equal twin nodes Q1, Q2 live at a boundary-parent level
// `v_left`, and the grandparent `root` holds pairs (Q1, c) and (Q2, c) where
// `c` is a count-carrying slot at a MARGINAL sibling level `v_right`, the
// old cancellation pass unconditionally cancelled the Q2→Q1 redirect to avoid
// creating a duplicate (Q1,c),(Q1,c) pair at `root`. That was over-conservative:
// at a marg-flagged level (sibling side marginal) duplicate (Q,c),(Q,c) pairs
// are legal multiset entries that p-fusion soundly folds to (Q,2c). The fix
// exempts that configuration and lets the redirect proceed, then hands root to
// dirty_contract so p-fusion runs.
//
// TEST LIFECYCLE:
//   - FAILS before the fix: cancellation fires, Q1/Q2 stay as two nodes at
//     v_left (deferred, loop breaks on non-decreasing deferred), so v_left.width()
//     stays 2. The assertion `v_left.width() == 1` fails.
//   - PASSES after the fix: redirect allowed, root gets (Q1,c),(Q1,c), p-fusion
//     folds to (Q1, 2c), Q2 becomes unreferenced and is pruned → v_left.width()==1.
//     Model count is preserved throughout.

// ── Inline-ref twin merge regression ─────────────────────────────────────────
//
// Inline-encoded marg-side refs (bit-30 set) carry their count directly in the
// pair field — they are INVISIBLE to slot-prune, which only walks slot-index
// refs.  When two boundary-parent nodes P and Q are born already raw-identical
// (same explicit-side child X, same inline marg ref Inline(1)), slot-prune
// reports values_merged == 0 and the old values_merged-gated content-twin scan
// was never invoked.  Context-based twin contraction (T) also misses them when
// they have different grandparent contexts (different grandparent siblings).
//
// Fix (minimise/mod.rs): run the content-twin scan unconditionally BEFORE the
// values_merged loop.  After T has already run, any surviving content-equal
// twins necessarily have different context signatures (carriers — same sibling —
// would have been caught by T), so the unconditional scan finds only SHARABLE
// twins (different siblings) and merging them is count-preserving.  After the
// merge, p-fusion folds any duplicate (X, Inline(1)) pair entries in the
// survivor's pair list into (X, Inline(2)).
//
// Regression guard: model count must be UNCHANGED after try_minimize (any
// count change would flag the halving/doubling bug the spec warns about).

// ─────────────────────────────────────────────────────────────────────────────
// Content twins at PLAIN levels (contraction-leak closure)
//
// Before the fix the content merge scanned only `boundary_marginal_levels` —
// the parents of marginal levels — so content-identical nodes at a PLAIN level
// (both children explicit or leaves) referenced from DIFFERENT parent contexts
// were compared by nothing: context-based `contract_all_twins_topdown` groups by
// the multiset of `(parent_node, sibling)` contexts, which differ by
// construction here, and the content scan never looked at the level.
//
// TEST LIFECYCLE:
//   - FAILS before the fix: `sub_left_r` is not a boundary parent (both its
//     children are vtree leaves), so nothing ever compares B1 and B2 and the
//     level keeps width 2.
//   - PASSES after the fix: the scan covers every explicit level, merges B2 into
//     B1, rewrites v_left's refs, and prune GCs B2 → width 1. Model count and
//     the twin-canonicality checker are asserted throughout.
// ─────────────────────────────────────────────────────────────────────────────


#[path = "tests/marginal.rs"]
mod marginal;
#[path = "tests/prune.rs"]
mod prune;
#[path = "tests/twins.rs"]
mod twins;
#[path = "tests/twins_budget.rs"]
mod twins_budget;
#[path = "tests/twins_inline.rs"]
mod twins_inline;
