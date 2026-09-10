//! Minimize on whole compiled diagrams.
//!
//! Operands come from [`crate::test_helpers::compile_clauses`] where a whole
//! formula is wanted, and from `clause_to_tdd` plus `apply_and` where a
//! particular intermediate shape is. A driver that preprocesses the formula
//! before compiling it reaches these diagrams by other routes; that variety
//! belongs to the driver's own tests.

use super::*;
use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::query::model_count;
use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};
use std::sync::Arc;

#[test]
fn test_minimize_constant_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let mut tdd = constant_one(eng, &vtree);
    minimize(&mut tdd);
    assert_canonical(&tdd);
    // Internal levels each hold one node; leaf levels stay implicit.
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
    assert_canonical(&tdd);
    let count_after = model_count(&tdd);
    assert_eq!(count_before, count_after);
}

#[test]
fn test_minimize_reduces_width_after_apply() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = vec![Literal::pos(VarId(0))];
    let g = vec![Literal::neg(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);

    let width_before = result.max_width();

    let count_before = model_count(&result);
    minimize(&mut result);
    assert_canonical(&result);
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
    let f = vec![Literal::pos(VarId(0))];
    let g = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);
    minimize(&mut result);
    assert_canonical(&result);

    assert_eq!(model_count(&result), 0u64.into());
}

#[test]
fn test_minimize_unsat_2vars_width() {
    let eng = &crate::engine::Engine::new();
    // 2 variables: (x0) AND (not-x0) = UNSAT
    // The canonical diagram for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(2));
    let f = vec![Literal::pos(VarId(0))];
    let g = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);

    minimize(&mut result);
    assert_canonical(&result);

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
    // The canonical diagram for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(3));
    let f = vec![Literal::pos(VarId(0))];
    let g = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);

    minimize(&mut result);
    assert_canonical(&result);

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
    // after apply: width 4. After minimize: should have width < 4.
    let vtree = Arc::new(Vtree::balanced(2));
    let f = vec![Literal::pos(VarId(0))];
    let g = vec![Literal::pos(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);

    let count_before = model_count(&result);
    minimize(&mut result);
    assert_canonical(&result);
    let count_after = model_count(&result);

    assert_eq!(count_before, count_after);
    // With pruned clause diagrams, the product may already be minimal (width 1).
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
// When a vtree level is made marginal (`TddLevel::become_marginal`), its
// node/pair structure is dropped and replaced with per-node model counts.
// Later parent conjunctions can reshape the parent's pair list so that two
// marginal entries end up in identical `(parent_idx, sibling_idx)`
// multisets — i.e. they become twins. Contracting those twins is sound
// under diagram determinism (twins are disjoint within shared parent contexts,
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
// raw signature slices differ as *sequences* while being equal as *sets*, and
// assert that they contract.
//
// Why width ≥ 3, not width 2: a level with exactly two scrambled twins A,B can
// always *self-canonicalize*. The two siblings A,B are paired with become twins
// themselves (each referenced by the same `{A,B}` set), collapse via the
// cascade, and that collapse rewrites A's and B's signatures down to a single
// shared sibling — which even the order-sensitive `==` then matches. To keep
// the bug observable we add a distinct node (C / C,D) that references one
// sibling and not the others, breaking the sibling symmetry so no cascade
// re-canonicalizes the signatures. The hazard is therefore a width-≥3
// phenomenon, which is what the hash-bucket exact comparison for
// `child_width >= 3` handles.
//
// The merge preserves `model_count` even though the hand-built diagrams are not
// deterministic: when A,B merge, the parent's now-duplicate `(AB, sibling)`
// pairs dedup, and `|AB| = |A| + |B|` makes the deduped term equal to the sum
// of the two originals — so the count is invariant. We assert it as a guard.

// ── OverBudget safety in contract_twins ───────────────────────────────────
//
// An `ApplyError::OverBudget` raised part-way through `contract_twins`' group-
// merge loop must never corrupt the model count. Every reserve the pass needs
// — the survivors' pair growth AND the parent's `multi_pairs` growth — is taken in one
// grand reserve before the loop mutates anything, so a refusal bails with the
// diagram exactly as it was: count unchanged, worklist restored. The commit
// pass that follows the reserve pushes infallibly.

// ── Dirty-worklist restoration on Err ─────────────────────────────────────
//
// A top-down contraction sweep drains the twin-contraction worklist into a
// topo-heap. If a mid-sweep `Err` fires (`Deadline` or `OverBudget` from
// `contract_twins`), every parent that had not yet been popped — plus the one
// being processed — must be restored to the twin-contraction worklist, or
// those levels keep stale contexts and are never re-contracted: sound, but a
// permanent canonicity and size leak.

// ── Prune value-merge → twin mint regression ──────────────────────────────
//
// If two boundary-parent nodes p = [(X1, f), (X2, d)] and
// q = [(X1, g), (X2, d)] have f ≠ g as slot indices but equal stored
// values (only possible for BIG counts — small ones are inline post-tagger),
// `prune_value_slots`'s value-dedup merges f and g onto one slot and
// rewrites both parent refs to it. That makes p and q raw-identical twins, so
// contract must run again or the no-twins postcondition at minimize exit is
// violated: minimize iterates contract → prune until prune reports
// `values_merged == 0`.

// ── Marg-sibling fold-allowed regression ──────────────────────────────────
//
// When two content-equal twin nodes Q1, Q2 live at a boundary-parent level
// `v_left`, and the grandparent `root` holds pairs (Q1, c) and (Q2, c) where
// `c` is a count-carrying slot at a MARGINAL sibling level `v_right`, the
// Q2→Q1 redirect is allowed even though it leaves a duplicate (Q1,c),(Q1,c)
// pair at `root`: at a marginal sibling level such duplicates are legal
// multiset entries that pair fusion folds to (Q1, 2c). Cancelling the redirect
// instead would leave Q1 and Q2 as two nodes at `v_left` forever. `root` goes
// on the contraction worklist so the fusion runs, after which `v_left` has
// width 1 and the model count is unchanged.

// ── Inline-ref twin merge regression ─────────────────────────────────────────
//
// Inline-encoded marginal-side refs carry their value directly in the pair
// field, so the slot prune — which walks slot-index refs only — never sees
// them. Two boundary-parent nodes P and Q born already raw-identical (same
// structural-side child X, same inline ref Inline(1)) leave the prune with
// `values_merged == 0`, and context-based contraction misses them when their
// grandparent contexts differ.
//
// `minimize` therefore runs the content-twin scan unconditionally before the
// `values_merged` loop. Once context-based contraction has run, any surviving
// content-equal twins have different context signatures, so the scan finds
// only twins whose merge is count-preserving. Pair fusion then folds the
// duplicate (X, Inline(1)) entries in the survivor's pair list into
// (X, Inline(2)), leaving the model count unchanged.

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
//   - fails before the fix: `sub_left_r` is not a boundary parent (both its
//     children are vtree leaves), so nothing ever compares B1 and B2 and the
//     level keeps width 2.
//   - passes after the fix: the scan covers every explicit level, merges B2 into
//     B1, rewrites v_left's refs, and prune GCs B2 → width 1. Model count and
//     the twin-canonicality checker are asserted throughout.
// ─────────────────────────────────────────────────────────────────────────────


#[path = "tests/canonicity.rs"]
mod canonicity;
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

/// `(a ∨ b) ∧ (a ∨ ¬b)` is `a`: `b` is irrelevant, and minimize has to say so
/// at the diagram level. After the pass, no pair list anywhere may still name
/// the positive or negative label of the `b` leaf — every reference into that
/// leaf must be the don't-care one.
#[test]
fn a_variable_the_function_ignores_leaves_no_literal_references_behind() {
    use crate::diagram::LeafLabel;
    use crate::check::check_determinism;
    use crate::vtree::{VtreeIdx, VtreeNode};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![1, -2]]);
    minimize(&mut tdd);
    assert_canonical(&tdd);

    // `a` over two variables: `a` true, `b` free.
    assert_eq!(model_count(&tdd), 2u64.into());

    let b_leaf = vtree.leaf_of(VarId(1)).expect("the vtree carries this variable");
    for level_idx in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(level_idx as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        for (_, pairs) in tdd.levels()[level_idx].internal_inputs_iter() {
            for pair in pairs {
                if left == b_leaf {
                    assert_ne!(pair.left.idx(), LeafLabel::Pos as usize, "level {level_idx}");
                    assert_ne!(pair.left.idx(), LeafLabel::Neg as usize, "level {level_idx}");
                }
                if right == b_leaf {
                    assert_ne!(pair.right.idx(), LeafLabel::Pos as usize, "level {level_idx}");
                    assert_ne!(pair.right.idx(), LeafLabel::Neg as usize, "level {level_idx}");
                }
            }
        }
    }

    check_determinism(&tdd).expect("leaf-mode determinism holds after minimize");
}

/// Minimize is idempotent. A second pass that shrinks the diagram means the
/// first one missed something; a second pass that changes the count or breaks
/// determinism means the first one rewrote too much.
#[test]
fn a_second_minimize_changes_nothing() {
    use crate::check::check_determinism;

    let vtree = Arc::new(Vtree::balanced(3));
    let formulas: &[&[&[i32]]] = &[
        // Independent of the second variable.
        &[&[1, 2], &[1, -2]],
        // One satisfying assignment.
        &[&[1], &[2], &[3]],
        // Seven of eight assignments.
        &[&[1, 2, 3]],
        // Depends on all three, with an irrelevant variable inside.
        &[&[1, 2], &[1, -2], &[-3]],
    ];

    for (i, clauses) in formulas.iter().enumerate() {
        let clauses: Vec<Vec<i32>> = clauses.iter().map(|c| c.to_vec()).collect();
        let mut tdd = compile_clauses(&vtree, &clauses);
        minimize(&mut tdd);
        assert_canonical(&tdd);
        let size = tdd.size();
        let count = model_count(&tdd);
        check_determinism(&tdd)
            .unwrap_or_else(|e| panic!("formula {i}: determinism after the first minimize: {e}"));

        minimize(&mut tdd);
        assert_canonical(&tdd);
        assert_eq!(tdd.size(), size, "formula {i}: the second minimize shrank the diagram");
        assert_eq!(model_count(&tdd), count, "formula {i}: the model count moved");
        check_determinism(&tdd)
            .unwrap_or_else(|e| panic!("formula {i}: determinism after the second minimize: {e}"));
    }
}
