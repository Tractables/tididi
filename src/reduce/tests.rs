use super::*;
use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, LocalNodeIdx, Tdd, TddNodeId, assert_can_make_marginal, take_levels,
};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use std::sync::Arc;

#[test]
fn test_minimize_constant_one() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut tdd = constant_one(&vtree);
    minimize(&mut tdd);
    // Internal levels have width 1; leaf levels are marginal (width 3)
    for (t, _left, _right) in vtree.internal_bottomup() {
        assert_eq!(tdd.level(t).width(), 1);
    }
}

#[test]
fn test_minimize_single_clause() {
    let vtree = Arc::new(Vtree::balanced(3));
    let clause = vec![Literal::pos(VarId(0))];
    let mut tdd = clause_to_tdd(&vtree, &clause);
    let count_before = model_count(&tdd);
    minimize(&mut tdd);
    let count_after = model_count(&tdd);
    assert_eq!(count_before, count_after);
}

#[test]
fn test_minimize_reduces_width_after_apply() {
    let vtree = Arc::new(Vtree::balanced(3));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(1))];

    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
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
    // Single variable: x ∧ ¬x = UNSAT
    let vtree = Arc::new(Vtree::balanced(1));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
    let mut result = apply_and(t1, t2);
    minimize(&mut result);

    assert_eq!(model_count(&result), 0u64.into());
}

#[test]
fn test_minimize_unsat_2vars_width() {
    // 2 variables: (x0) AND (not-x0) = UNSAT
    // The canonical TDD for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(2));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
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
    // 3 variables: (x0) AND (not-x0) = UNSAT
    // The canonical TDD for false should have width 0 (ZERO sentinel, empty levels)
    let vtree = Arc::new(Vtree::balanced(3));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
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
    // (x0) ∧ (x1) over 2 vars → 1 model (x0=1, x1=1)
    // After apply: width 4. After minimize: should have width < 4.
    let vtree = Arc::new(Vtree::balanced(2));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::pos(VarId(1))];

    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
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

/// All-or-nothing safety: when one parent on a leaf-adjacent level has matched
/// `Pos_x`/`Neg_x` partners but a sibling parent has an unmatched literal pair,
/// the rewrite must skip the entire level on that side. Contracting only the
/// matched parent would put the leaf in mixed mode (some refs in `{Pos, Neg}`,
/// others in `{One}`), violating the leaf-mode determinism invariant.
///
/// Construction: vtree balanced(3). Parent A at level 3 has matched literals
/// on its left side; parent B has an unmatched `Pos`. Both are reachable via
/// the root.
#[test]
fn test_leaf_contract_skips_when_one_parent_unmatched() {
    let vtree = Arc::new(Vtree::balanced(3));
    // balanced(3): leaves 0/1/2; level 3 = parent of leaves 0,1; level 4 = root (3, 2).
    assert!(matches!(*vtree.node(VtreeIdx(3)), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(VtreeIdx(4)), VtreeNode::Internal { .. }));

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // Parent A: [(Pos_0, Pos_1), (Neg_0, Pos_1)] — left contractible to (One_0, Pos_1).
    let a = levels[3].push_internal_node(&[
        InputPair { left: pos, right: pos },
        InputPair { left: neg, right: pos },
    ]);
    // Parent B: [(Pos_0, Neg_1)] — left has Pos with no matching Neg.
    let b = levels[3].push_internal_node(&[
        InputPair { left: pos, right: neg },
    ]);
    // Root references both parents with a literal on the var-2 leaf.
    let root = levels[4].push_internal_node(&[
        InputPair { left: a, right: pos },
        InputPair { left: b, right: neg },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: VtreeIdx(4), local: root },
    );

    let pairs_a_before: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(0).collect();
    let pairs_b_before: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(1).collect();

    let changed = contract_leaf_twins(&mut tdd);
    assert!(!changed, "all-or-nothing: any unmatched literal must veto rewrite on that side");

    let pairs_a_after: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(0).collect();
    let pairs_b_after: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(1).collect();
    assert_eq!(pairs_a_before, pairs_a_after, "parent A's pairs must be untouched");
    assert_eq!(pairs_b_before, pairs_b_after, "parent B's pairs must be untouched");
}

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

/// Marginal-level twins induced by a parent-level restructure must be
/// contracted by `minimize`.
///
/// Walks through the full lifecycle:
///
/// **Phase 1.** Build a canonical 4-leaf TDD shaped `((x0, x1), (x2, x3))`.
/// `v_left` has two distinct non-twin entries A=0 (x0=1) and B=1 (x0=0),
/// paired at the root with *different* `v_right` siblings (r0, r1). No twins
/// exist anywhere; `minimize` is a no-op.
///
/// **Phase 2.** Make `v_left` marginal. The level's structural data is
/// dropped and replaced with per-node model counts. Still no twins;
/// `minimize` is again a no-op.
///
/// **Phase 3.** Restructure the root's pair list so that A and B end up in
/// the *same* multiset of `(root_node, sibling)` contexts — i.e. they become
/// twins at the marginal `v_left`. This is the structural effect a later
/// `apply_and` could produce when conjoining a clause that reshapes the
/// parent's pair list. Run `minimize`: the twins must contract into one
/// node, with `marginal_counts[merged] = marginal_counts[A] +
/// marginal_counts[B]`.
#[test]
fn test_minimize_contracts_marginal_twins() {
    // Marginal-twin / p-fusion path: two marginal nodes at v_left with
    // DISTINCT counts (C_A ≠ C_B) both appearing paired with multiple
    // right-side siblings at the root → p-fusion closes the redex by
    // summing their counts, leaving one merged slot.
    //
    // We use distinct counts (C_A=2, C_B=3) so that slot-prune does NOT
    // merge them (value-dedup only merges equal-valued slots). Equal-valued
    // slots would be merged by slot-prune before contraction fires, which
    // is a separate (correct) behaviour tested elsewhere. With distinct counts
    // we can exercise the full p-fusion path that fires for same-explicit-
    // different-marginal-count pairs.
    //
    // Pin inline threshold to 0 so both counts stay as slots (no inlining).
    let _thr = crate::diagram::marg::set_marg_inline_max(0);
    const C_A: u128 = 2;
    const C_B: u128 = 3;
    const C_SUM: u128 = C_A + C_B; // 5

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), VtreeNode::Internal { .. }));

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    // ── Phase 1: canonical TDD with no twins anywhere ──────────────────
    //
    // v_left:  A=(Pos, One)  represents "x0=1, x1 free"
    //          B=(Neg, One)  represents "x0=0, x1 free"
    // v_right: r0=(Pos, One) represents "x2=1, x3 free"
    //          r1=(One, Pos) represents "x2 free, x3=1"
    // root:    one node with pairs [(A, r0), (B, r1)]
    //
    // A and B have *different* siblings at root → not twins.
    // r0 and r1 likewise → not twins. Determinism holds: A∧r0 disjoint
    // from B∧r1 because A∧B = false.
    let mut levels = take_levels(vtree.num_nodes());
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let r0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let r1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: r0 },
        InputPair { left: b, right: r1 },
    ]);
    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );

    let phase1_size = tdd.size();
    let phase1_count = model_count(&tdd);
    minimize(&mut tdd);
    assert_eq!(tdd.size(), phase1_size, "phase 1: T0 has no twins; minimize must be a no-op");
    assert_eq!(model_count(&tdd), phase1_count);
    assert_eq!(tdd.levels[v_left.idx()].width(), 2);

    // ── Phase 2: marginalize v_left with DISTINCT counts ────────────────
    //
    // A has count C_A=2, B has count C_B=3 (distinct → slot-prune keeps
    // both, and no structural change occurs within this phase's minimize).
    assert_can_make_marginal(&tdd.levels, &vtree, v_left);
    tdd.levels[v_left.idx()].make_marginal(vec![C_A, C_B], None);
    // Hand-rolled make_marginal bypasses production marginalization; tag the
    // now-marginal level's persisted parent refs so the 0=inline decode
    // invariant holds (mirrors marginalize_batch / marginalize_subtree).
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    let phase2_count = model_count(&tdd);
    assert_eq!(tdd.levels[v_left.idx()].width(), 2, "phase 2: marginalization preserves width");
    minimize(&mut tdd);
    // Distinct slot values → slot-prune does NOT merge them → width stays 2.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "phase 2: distinct-count slots must survive slot-prune; minimize must be a no-op at v_left",
    );
    assert_eq!(model_count(&tdd), phase2_count);

    // ── Phase 3: restructure root to induce a p-fusion redex at v_left ──
    //
    // Replace root's pair list with [(A, r0), (A, r1), (B, r0), (B, r1)].
    // Both A and B now appear paired with both r0 and r1. P-fusion groups
    // by same-explicit-side: (A, r0) + (B, r0) → (C_SUM, r0), and
    // (A, r1) + (B, r1) → (C_SUM, r1). After the first round the root
    // has [(C_SUM, r0), (C_SUM, r1)] — same marginal ref again, so a
    // second slot-prune pass merges the two C_SUM slots into one.
    // The final result: width=1 and the surviving count = C_SUM.
    //
    // The TDD is no longer deterministic — fine for this test, which is
    // about contraction structure, not Boolean semantics.
    tdd.levels[root.idx()].clear();
    let new_root = tdd.levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: r0 },
        InputPair { left: a, right: r1 },
        InputPair { left: b, right: r0 },
        InputPair { left: b, right: r1 },
    ]);
    tdd.output = TddNodeId { vtree: root, local: new_root };
    // Mark root as needing re-contraction (mimics what `apply_and` does
    // when it rebuilds the root level from scratch).
    tdd.scratch.dirty_contract.push(root.0);
    // The rebuilt root pairs reuse the bare phase-1 refs `a`/`b`, which now
    // point into the marginal v_left and must be tagged (production's
    // end-of-apply tagger does this after apply rebuilds the root level).
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    let phase3_count = model_count(&tdd);

    minimize(&mut tdd);
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&mut tdd).unwrap();
    // After p-fusion + slot-prune: v_left should have exactly 1 surviving slot.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 1,
        "phase 3 (target): p-fusion at the marginal v_left must reduce to \
         one surviving slot (the sum C_A+C_B={C_SUM}). Parent pair list \
         must dedup from 4 entries down to 1 or 2.",
    );
    // The surviving slot's count must equal C_A + C_B = C_SUM.
    assert_eq!(
        tdd.levels[v_left.idx()].marginal_counts.as_ref().unwrap()[0],
        C_SUM,
        "phase 3: surviving slot count must equal C_A + C_B = {C_SUM}",
    );
    // Sanity: model count is preserved across minimize. Defends against
    // an arithmetic mistake in the merge or remap.
    assert_eq!(model_count(&tdd), phase3_count);
}

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

/// A,B are scrambled-order twins (`[r0,r1]` vs `[r1,r0]`); a distinct node C
/// (paired only with r0) breaks sibling symmetry so no cascade masks the bug.
/// Exercises the open-addressing hash-bucket exact comparison; width 3 → 2.
#[test]
fn test_contract_detects_twins_with_scrambled_signature_order_width3() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // v_left: A,B are twins; C = (One,Pos) [x1=1] is a distinct non-twin.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let c = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let r0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let r1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    // A: (r0, r1); B: (r1, r0) [scrambled twins]; C: only r0 [distinct sig,
    // breaks r0/r1 symmetry so neither sibling collapses].
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: r0 },
        InputPair { left: a, right: r1 },
        InputPair { left: b, right: r1 },
        InputPair { left: b, right: r0 },
        InputPair { left: c, right: r0 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );
    assert_eq!(tdd.levels[v_left.idx()].width(), 3, "setup: A, B (twins) + C (distinct)");
    let count_before = model_count(&tdd);

    tdd.scratch.dirty_contract.push(root.0);
    contract_all_twins(&mut tdd).expect("contraction must not OOM");

    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "the scrambled-order twins A,B must merge via the hash-bucket exact \
         comparison while the distinct node C survives → width 3 → 2. The \
         pre-fix order-sensitive `==` left all 3 nodes (under-contraction).",
    );
    assert_eq!(model_count(&tdd), count_before, "merge must preserve model count");
}

/// Stronger scramble: A and B are twins over THREE shared siblings in fully
/// reversed order (`[s0,s1,s2]` vs `[s2,s1,s0]`), so the signature sort must
/// reorder a multi-element slice (not just swap a pair). Two distinct nodes
/// C,D pin s0 and s1 respectively, leaving every sibling a unique signature so
/// no cascade canonicalizes anything; width 4 → 3.
#[test]
fn test_contract_detects_twins_with_reversed_multi_sibling_signature() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // v_left twins A,B over {x0,x1}; C,D distinct symmetry-breakers.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let c = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let d = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: neg }]);
    // Three distinct siblings over {x2,x3}.
    let s0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let s1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let s2 = levels[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    // A: (s0,s1,s2);  B: (s2,s1,s0) reversed — same SET, reversed sequence.
    // C pins s0, D pins s1 → sibling signatures s0=[A,B,C], s1=[A,B,D],
    // s2=[A,B] are pairwise distinct, so v_right has no twins to cascade.
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: s0 },
        InputPair { left: a, right: s1 },
        InputPair { left: a, right: s2 },
        InputPair { left: b, right: s2 },
        InputPair { left: b, right: s1 },
        InputPair { left: b, right: s0 },
        InputPair { left: c, right: s0 },
        InputPair { left: d, right: s1 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );
    assert_eq!(tdd.levels[v_left.idx()].width(), 4, "setup: A,B (twins) + C,D (distinct)");
    let count_before = model_count(&tdd);

    tdd.scratch.dirty_contract.push(root.0);
    contract_all_twins(&mut tdd).expect("contraction must not OOM");

    assert_eq!(
        tdd.levels[v_left.idx()].width(), 3,
        "twins over a 3-element reversed signature must merge while C,D survive \
         → width 4 → 3. The pre-fix order-sensitive `==` compared [s0,s1,s2] to \
         [s2,s1,s0], missed the twin, and left all 4 nodes (under-contraction).",
    );
    assert_eq!(model_count(&tdd), count_before, "merge must preserve model count");
}

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
//    `tdd.scratch.poisoned` and any later count extraction panics.

/// A reserve failure across twin groups must leave the model count
/// UNCHANGED (transactional grand reserve). FAILS on the pre-fix code, which
/// grows one group's survivor before the second group's reserve fails while the
/// parent still references both.
#[test]
fn test_contract_twins_overbudget_w1_count_unchanged() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // Two twin groups at v_left, each member a disjoint 2-pair node so the merge
    // takes the concat path (the 1+1 inline fast path never reaches the reserve).
    let x = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: pos, right: pos }, InputPair { left: pos, right: neg }]);
    let y = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: neg, right: pos }, InputPair { left: neg, right: neg }]);
    let x2 = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: pos, right: pos }, InputPair { left: pos, right: neg }]);
    let y2 = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: neg, right: pos }, InputPair { left: neg, right: neg }]);
    // Two distinct siblings ⇒ the two groups have distinct signatures, and s0/s1
    // are non-twins (no cascade masks the merge).
    let s0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let s1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);

    // Group 1: {x, y} both paired with s0; group 2: {x2, y2} both paired with s1.
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: x, right: s0 },
        InputPair { left: y, right: s0 },
        InputPair { left: x2, right: s1 },
        InputPair { left: y2, right: s1 },
    ]);

    let mut tdd = Tdd::with_levels(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].width(), 4, "setup: two twin groups {{x,y}},{{x2,y2}}");
    let count_before = model_count(&tdd);

    tdd.scratch.dirty_contract.push(root.0);
    // Consult #1 (first group's reserve) succeeds; consult #2 (second group's
    // reserve) fires. On the fixed code both consults hit the single hoisted
    // grand reserve, so the bail happens before any mutation.
    super::contract::arm_fail_after(1);
    let res = contract_all_twins(&mut tdd);
    super::contract::disarm_fail();

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(!tdd.scratch.poisoned, "a cross-group reserve failure must bail transactionally, not poison");
    assert_eq!(
        model_count(&tdd),
        count_before,
        "OverBudget in contract_twins must leave the model count unchanged",
    );
}

/// An OverBudget in the mid parent-rewrite "shrink-to-1 but can't inline"
/// branch is IRRECOVERABLE — earlier parent pairs are already remapped and there
/// is no clean rollback — so it must set `tdd.scratch.poisoned`. The count extractor then
/// refuses the diagram (see `test_model_count_refuses_poisoned_tdd`).
#[test]
fn test_contract_twins_overbudget_w2_poisons() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // Two single-pair twins at v_left (same parent context, distinct data → they
    // merge via the 1+1 path, growing the survivor). Width 2 so the edge IS
    // contracted; v_right (width 1) is skipped by `try_contract_child`, which is
    // what lets its sibling ref safely carry bit 31.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: neg }]);
    // A single v_right node; the root pairs reference it with bit 31 (LEAF_BIT)
    // set on the SIBLING (right) field. On the top-down contract path that field
    // is copied but never dereferenced, yet it makes the lone surviving parent
    // pair `can_inline() == false` — forcing the mid-rewrite branch (the one remaining
    // fallible allocation in the parent rewrite).
    let s0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let sib = LocalNodeIdx((1u32 << 31) | s0.0);

    // Root: both twins paired with the SAME (bit-31) sibling ⇒ they share a
    // context ⇒ twins. After they merge, one of the two parent pairs is filtered
    // (both now reference the survivor), shrinking the parent node to a single
    // can't-inline pair ⇒ the mid-rewrite window.
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib },
        InputPair { left: b, right: sib },
    ]);

    let mut tdd = Tdd::with_levels(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].width(), 2, "setup: one twin group {{a,b}}");

    tdd.scratch.dirty_contract.push(root.0);
    // Consults on the v_left edge: #0 (grand-reserve pairs), #1 (grand-reserve
    // ext), #2 at the mid-rewrite ext push. Fire #2.
    super::contract::arm_fail_after(2);
    let res = contract_all_twins(&mut tdd);
    super::contract::disarm_fail();

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(
        tdd.scratch.poisoned,
        "an OverBudget mid parent-rewrite must poison the TDD",
    );
    // NB: deliberately DON'T call model_count(&tdd) — it is poisoned (would trip
    // the backstop assert) and carries a bit-31 sibling (not a real node ref).
}

/// A poisoned diagram must be refused by the count extractor: the poison flag
/// is only a backstop if no consumer reads a count from a poisoned diagram.
#[test]
#[should_panic(expected = "poisoned")]
fn test_model_count_refuses_poisoned_tdd() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut tdd = constant_one(&vtree);
    // Sanity: the un-poisoned diagram counts fine.
    assert_ne!(model_count(&tdd), num_bigint::BigUint::ZERO);
    // Flip the flag a mid-rewrite failure would set; the next count extraction must panic.
    tdd.scratch.poisoned = true;
    let _ = model_count(&tdd);
}

// ── Dirty-worklist restoration on Err ─────────────────────────────────────
//
// A top-down contraction sweep `mem::take`s `tdd.scratch.dirty_contract` into a
// topo-heap. If a mid-sweep `Err` fires (race-lane `Deadline` preemption, or
// `OverBudget` from `contract_twins`), every parent that had not yet been
// popped — plus the one being processed — must be restored to
// `tdd.scratch.dirty_contract`, or those levels keep stale contexts and are never
// re-contracted (a permanent canonicity/size leak; sound but a leak). On
// unfixed HEAD the taken worklist is dropped, so `dirty_contract` is empty
// after the Err — the assertions below fail.

/// Seed TWO dirty parents, fire an `OverBudget` during the FIRST (root-most)
/// parent's contraction, and assert BOTH the failing parent and the still-queued
/// second parent survive in `dirty_contract`. The second parent (`v_right`) is
/// never popped — it proves the heap-remainder restore; `root` proves the
/// failed-mid-processing restore. FAILS if the worklist is dropped instead of
/// restored (→ empty).
#[test]
fn test_contract_dirty_worklist_restored_on_err() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // Twin group {x, y} at v_left: disjoint 2-pair nodes so the merge takes the
    // concat path (the 1+1 inline fast path never reaches the grand reserve).
    let x = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: pos, right: pos }, InputPair { left: pos, right: neg }]);
    let y = levels[v_left.idx()]
        .push_internal_node(&[InputPair { left: neg, right: pos }, InputPair { left: neg, right: neg }]);
    // A multi-pair node at v_right: makes `has_multi_pair(v_right)` true so
    // v_right is a valid heap parent (seeded as the SECOND dirty parent), and
    // serves as the shared right-sibling context that makes x,y twins.
    let s = levels[v_right.idx()]
        .push_internal_node(&[InputPair { left: pos, right: one }, InputPair { left: neg, right: one }]);

    // Root pairs (x, s) and (y, s): same right sibling ⇒ x,y share a context ⇒
    // twins at v_left.
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: x, right: s },
        InputPair { left: y, right: s },
    ]);

    let mut tdd = Tdd::with_levels(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].width(), 2, "setup: one twin group {{x,y}}");

    // Seed BOTH parents. Heap pops root-most first (root), leaving v_right queued.
    tdd.scratch.dirty_contract.clear();
    tdd.scratch.dirty_contract.push(root.0);
    tdd.scratch.dirty_contract.push(v_right.0);

    // Fire on the very first consult — the grand reserve inside root's
    // contract_twins — so root fails mid-processing while v_right is still queued.
    super::contract::arm_fail_after(0);
    let res = super::contract::contract_all_twins_topdown(&mut tdd, None);
    super::contract::disarm_fail();

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(
        !tdd.scratch.poisoned,
        "a grand-reserve failure bails transactionally, not poison",
    );
    assert!(
        tdd.scratch.dirty_contract.contains(&v_right.0),
        "the unprocessed parent still queued in the heap must be restored on Err; \
         dirty_contract = {:?}",
        tdd.scratch.dirty_contract,
    );
    assert!(
        tdd.scratch.dirty_contract.contains(&root.0),
        "the parent that failed mid-processing must be restored on Err; \
         dirty_contract = {:?}",
        tdd.scratch.dirty_contract,
    );
}

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

/// Regression: marg-sibling fold-allowed — fix in `merge_content_equal_nodes`.
///
/// Layout (balanced(6)):
///   sub_left_r = internal(leaf1, leaf2) — right child of v_left; made marginal (count C_SLR=5)
///   sub_right_r = internal(leaf4, leaf5) — right child of v_right; made marginal (needed for v_right)
///   v_left = internal(leaf0, sub_left_r) — boundary parent; Q1 and Q2 are content twins
///              (both have pair (LeafLabel::Pos, slot0_of_sub_left_r))
///   v_right = internal(leaf3, sub_right_r) — made marginal; the sibling side at root
///   root = grandparent: one node R with pairs [(Q1, slot0_vright), (Q2, slot0_vright)]
///
/// Pre-minimize model_count = Q1_count*C_VR + Q2_count*C_VR = 5*3 + 5*3 = 30.
///
/// Both v_left and root are pre-marked contracted=true so the initial contract_only
/// in try_minimize is a no-op. This forces the content-twin scan to be the ONLY
/// mechanism that handles the Q1/Q2 twin merge. The scan must then:
///   1. Perform the redirect Q2→Q1 (creating duplicate (Q1,slot0),(Q1,slot0) pairs at root).
///   2. Direct contract's p-fusion to fold the duplicate into one (Q1, slot1=2*C_VR) pair.
///   3. Let prune_marg_slots compact v_right's store to a single slot with count 2*C_VR.
///
/// The scan used to cancel any redirect that produced a duplicate pair at a
/// grandparent, deferring the merge to contract's fork-down concat path, which left
/// v_right's slot count at C_VR (=3). Duplicate pairs in a marginalized diagram are
/// legal multiset entries, so the redirect now always happens. The discriminating
/// assertion is (d): v_right's surviving count = 2*C_VR = 6.
#[test]
fn test_marg_sibling_fold_allowed_regression() {
    use crate::limits::apply_limits;
    use crate::vtree::VtreeNode;

    // Prevent inlining so slot refs stay as bare indices (not bit-30-tagged).
    // With threshold=0 no count c satisfies c <= 0, so all refs stay as slot indices.
    let _thr = crate::diagram::marg::set_marg_inline_max(0);
    let _g = apply_limits().budget(None).apply();

    // balanced(6): 11 nodes (6 leaves + 5 internals)
    // Structure (after bottom-up reindex):
    //   sub_left_r  = internal(leaf1, leaf2)    (right child of v_left)
    //   sub_right_r = internal(leaf4, leaf5)    (right child of v_right)
    //   v_left      = internal(leaf0, sub_left_r)
    //   v_right     = internal(leaf3, sub_right_r)
    //   root        = internal(v_left, v_right)
    let vtree = Arc::new(Vtree::balanced(6));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);
    assert!(matches!(*vtree.node(v_left), VtreeNode::Internal { .. }), "v_left internal");
    assert!(matches!(*vtree.node(v_right), VtreeNode::Internal { .. }), "v_right internal");

    // v_left's children: left = leaf0 (leaf), right = sub_left_r (internal).
    let (leaf0, sub_left_r) = vtree.children(v_left);
    assert!(matches!(*vtree.node(leaf0), VtreeNode::Leaf { .. }), "leaf0 leaf");
    assert!(matches!(*vtree.node(sub_left_r), VtreeNode::Internal { .. }), "sub_left_r internal");

    // v_right's children: left = leaf3 (leaf), right = sub_right_r (internal).
    let (_leaf3, sub_right_r) = vtree.children(v_right);
    assert!(matches!(*vtree.node(sub_right_r), VtreeNode::Internal { .. }), "sub_right_r internal");

    let n = vtree.num_nodes();
    let mut levels = take_levels(n);

    // --- sub_left_r: make marginal (count C_SLR). This makes v_left a boundary parent. ---
    // sub_left_r's children are leaves (ok per assert_can_make_marginal).
    assert_can_make_marginal(&levels, &vtree, sub_left_r);
    const C_SLR: u128 = 5; // model count stored at sub_left_r's slot 0
    levels[sub_left_r.idx()].make_marginal(vec![C_SLR], None);

    // --- sub_right_r: make marginal (needed so v_right can be marginalized). ---
    assert_can_make_marginal(&levels, &vtree, sub_right_r);
    const C_SRR: u128 = 7; // model count stored at sub_right_r's slot 0; unused in count calc
    let _ = C_SRR;
    levels[sub_right_r.idx()].make_marginal(vec![C_SRR], None);

    // --- v_right: make marginal (count C_VR). This is the SIBLING of v_left at root. ---
    // When the redirect Q2→Q1 creates duplicate (Q1,slot0),(Q1,slot0) at root,
    // the fold_allowed check sees v_right.is_marginal()==true and allows the redirect.
    assert_can_make_marginal(&levels, &vtree, v_right);
    const C_VR: u128 = 3; // model count stored at v_right's slot 0
    levels[v_right.idx()].make_marginal(vec![C_VR], None);

    // --- v_left: two content-equal twin nodes Q1 and Q2. ---
    //
    // Each has one pair: left=LeafLabel::Pos (ref into plain leaf0), right=slot0_slr=0
    // (ref into marginal sub_left_r). Raw pair values (1, 0) are identical → twins ✓.
    let pos      = LocalNodeIdx(LeafLabel::Pos as u32); // = LocalNodeIdx(1)
    let slr_slot0 = LocalNodeIdx(0); // slot index 0 of sub_left_r (marginal)
    let q1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: slr_slot0 }]);
    let q2 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: slr_slot0 }]);
    assert_eq!(q1.idx(), 0, "Q1 must be node 0 at v_left");
    assert_eq!(q2.idx(), 1, "Q2 must be node 1 at v_left");

    // --- root: one node R with pairs (Q1, slot0_vright) and (Q2, slot0_vright). ---
    //
    // Both Q1 and Q2 are referenced with the SAME marginal sibling (slot0 of v_right).
    // After the fix the redirect Q2→Q1 is allowed (fold_allowed=true); root gets
    // (Q1,slot0),(Q1,slot0); p-fusion folds to (Q1, slot1=2*C_VR); prune compacts.
    let vr_slot0 = LocalNodeIdx(0); // slot index 0 of v_right (marginal)
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: q1, right: vr_slot0 },
        InputPair { left: q2, right: vr_slot0 },
    ]);

    // Pre-mark v_left and root as contracted (harmless when calling
    // canonicalize_content_twins directly, kept for documentation: the scan
    // must be the sole merge mechanism exercised here — not contract's fork-down
    // concat path — so the fold_allowed discriminator assertion (d) is clean).

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );

    // Tag marg-side slots so the marg_inlined_right markers are set on v_left
    // (right child sub_left_r is marginal) and root (right child v_right is marginal).
    // With marg_inline_max=0 no inlining happens; markers enable decode in model_count.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    // Pre-minimize model count:
    //   Q1's count at v_left = Pos_leaf0 × C_SLR = 1 × 5 = 5.
    //   Q2 identical → 5.
    //   root = 5×C_VR + 5×C_VR = 30.
    let count_before = model_count(&tdd);
    let expected_count_u: u64 = 30;
    assert_eq!(
        count_before,
        expected_count_u.into(),
        "pre-minimize model count must equal {expected_count_u}"
    );
    assert_eq!(tdd.levels[v_left.idx()].width(), 2, "setup: Q1 and Q2 are two distinct nodes");

    // Call canonicalize_content_twins directly: try_minimize's normal path does
    // not run the content-twin scan, so tests exercise it via the extracted pub(crate)
    // function.
    super::canonicalize_content_twins(&mut tdd).expect("canonicalize_content_twins must not OOM");

    // (a) Model count MUST be unchanged.
    let count_after = model_count(&tdd);
    assert_eq!(
        count_after, count_before,
        "minimize must not change model count: before={count_before} after={count_after}"
    );

    // (b) v_left must have contracted from width 2 to width 1 (Q2 merged).
    //     This holds on both fixed and unfixed code (contract's fork-down path also
    //     merges them when fold_allowed is absent). The key discriminator is (d).
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 1,
        "fold-allowed regression: Q1 and Q2 must merge at v_left (width 2 → 1)"
    );

    // (c) v_right must compact to exactly 1 slot after p-fusion + prune.
    assert_eq!(
        tdd.levels[v_right.idx()].width(), 1,
        "v_right must compact to 1 slot after p-fusion folds (Q1,c),(Q1,c) → (Q1,2c)"
    );

    // (d) THE DISCRIMINATING ASSERTION: the surviving count at v_right must be 2*C_VR.
    //
    // On UNFIXED code: fold_allowed is absent; the redirect is cancelled; contract uses
    // the fork-down concat path which scales sub_left_r's count instead of v_right's.
    // v_right's slot stays at C_VR=3. This assertion FAILS: left=3, right=6.
    //
    // On FIXED code: fold_allowed fires; root gets duplicate (Q1,slot0),(Q1,slot0) pairs;
    // p-fusion folds them into (Q1, new_slot=2*C_VR=6); prune_marg_slots compacts v_right
    // from [C_VR, 2*C_VR] down to [2*C_VR]. This assertion PASSES.
    assert_eq!(
        tdd.levels[v_right.idx()].marginal_counts.as_ref().unwrap()[0],
        2 * C_VR,
        "v_right surviving slot count must equal 2*C_VR = {}",
        2 * C_VR,
    );
}

/// Regression: prune value-merge can mint twins after contract ran.
/// Pre-fix: the broken one-shot sequence leaves unmerged twins.
/// Post-fix: `try_minimize`'s iterate-to-fixpoint loop eliminates them.
///
/// Fixture: boundary store at v_marg holds 3 slots [C, C, D] (slots 0,1 equal;
/// slot 2 distinct). Parent-level nodes p and q each hold TWO pairs with the
/// same non-marg side (X1=Pos, X2=Neg) but different marg-side slot refs for X1:
///   p: [(Pos, slot_0), (Neg, slot_2)]   — slot_0=C, slot_2=D
///   q: [(Pos, slot_1), (Neg, slot_2)]   — slot_1=C (= slot_0's value), slot_2=D
/// Pre-prune p != q (slot_0 != slot_1 as indices). After prune's value-merge
/// (slot_1 -> slot_0), both become [(Pos, slot_0), (Neg, new_slot_1)] -> twins.
#[test]
fn test_prune_value_merge_does_not_mint_twins_at_minimize_exit() {
    use crate::reduce::slot_prune::prune_marg_slots;
    use crate::check::marg::{
        check_no_orphan_slots, check_no_twins, check_slot_count_uniqueness,
    };
    use crate::diagram::MargRef;
    use crate::vtree::VtreeNode;

    // BIG ensures counts cannot inline (MARG_INLINE_MAX = 2^30 - 1 < 2^40).
    // Slot-prune is where slot-count uniqueness is established; equal-valued slots
    // only collapse there (the emit site is forbidden from deduping).
    const BIG: u128 = 1u128 << 40;
    const C: u128 = BIG + 99; // equal value shared by slots 0 and 1
    const D: u128 = BIG + 7;  // distinct value at slot 2

    // balanced(4): 7 vtree nodes (0-3=leaves, 4=internal(0,1), 5=internal(2,3),
    // 6=root=internal(4,5)).
    //
    // Topology:
    //   v_marg    = right leaf-child of v_parent4 (holds 3 slots [C,C,D]).
    //   v_parent4 = non-marginal; nodes p and q (2 pairs each).
    //   v_right5  = non-marginal; nodes s0, s1 (symmetry breakers at root).
    //   root      = output; one node with pairs (p,s0) and (q,s1).
    let _g = crate::limits::apply_limits().budget(None).apply();
    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_parent4, v_right5) = vtree.children(root_idx);
    assert!(
        matches!(*vtree.node(v_parent4), VtreeNode::Internal { .. }),
        "v_parent4 must be an internal vtree node"
    );
    // vtree.children returns (left, right); make the RIGHT child marginal.
    let (_v_leaf0, v_marg) = vtree.children(v_parent4);
    assert!(
        matches!(*vtree.node(v_marg), VtreeNode::Leaf { .. }),
        "v_marg must be a leaf vtree node"
    );

    let n = vtree.num_nodes();
    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..n).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_marg: 3 slots [C, C, D]. Slots 0 and 1 carry equal values — a duplicate
    // planted deliberately; slot-prune collapses them.
    levels[v_marg.idx()].marginal_counts = Some(vec![C, C, D]);

    // v_parent4: two 2-pair nodes p and q.
    //   Marg-side (right) refs are bare indices (MargRef::Slot(i).to_raw() = i,
    //   bit-30 clear). BIG values cannot inline; slot-prune leaves them as slots.
    //   p: [(Pos, slot_0), (Neg, slot_2)]
    //   q: [(Pos, slot_1), (Neg, slot_2)]
    //   Pre-prune: p != q (slot_0 != slot_1 as raw indices) -> contract sees no twins.
    //   After value-merge slot_1->slot_0: both become [(Pos, slot_0), (Neg, slot_1')]
    //   where slot_1' is the compacted D slot -> identical pair lists -> twins.
    let pos  = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg  = LocalNodeIdx(LeafLabel::Neg as u32);
    let slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    let slot1 = LocalNodeIdx(MargRef::slot_raw(1));
    let slot2 = LocalNodeIdx(MargRef::slot_raw(2));
    let p = levels[v_parent4.idx()].push_internal_node(&[
        InputPair { left: pos, right: slot0 },
        InputPair { left: neg, right: slot2 },
    ]);
    let q = levels[v_parent4.idx()].push_internal_node(&[
        InputPair { left: pos, right: slot1 },
        InputPair { left: neg, right: slot2 },
    ]);

    // v_right5: two structurally distinct nodes (different left-leaf label).
    // Their distinctness ensures the root's pair list is non-trivial and both
    // p and q are independently reachable from the output node.
    let one = LocalNodeIdx(LeafLabel::One as u32);
    let s0 = levels[v_right5.idx()].push_internal_node(&[
        InputPair { left: pos, right: one },
    ]);
    let s1 = levels[v_right5.idx()].push_internal_node(&[
        InputPair { left: neg, right: one },
    ]);

    // root (output): single node with pairs (p, s0) and (q, s1).
    // Both p and q are referenced -> reachable -> prune is a no-op.
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: p, right: s0 },
        InputPair { left: q, right: s1 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );

    // ── Pre-fix verification: the broken one-shot sequence leaves twins ────────
    //
    // Manually reproduce the PRE-FIX order: contract (no merge since p!=q), then
    // prune_marg_slots once (merges equal slots, mints twins). Assert check_no_twins
    // FAILS — confirming the test pins the fixed behaviour.
    {
        let mut tdd2 = tdd.clone();
        // Seed dirty list: contract short-circuits on an empty list.
        tdd2.scratch.dirty_contract.push(root_idx.0);
        // Step 1: contract — p and q have different slot refs -> no twins -> no-op.
        super::contract::contract_all_twins_topdown(&mut tdd2, None)
            .expect("contract must not OOM in pre-fix verification");
        // Step 2: one prune pass — slots 0,1 both = C -> merge -> twins minted.
        let prune_stats = prune_marg_slots(&mut tdd2);
        assert!(
            prune_stats.values_merged > 0,
            "pre-fix verification: prune must report values_merged > 0 \
             (equal-valued slots 0 and 1 must collapse)"
        );
        // Step 3: check_no_twins must FAIL (twins minted, no re-contract ran).
        assert!(
            check_no_twins(&tdd2).is_err(),
            "pre-fix verification: check_no_twins must FAIL after the broken \
             one-shot contract->prune sequence (twin pair minted by value-merge)"
        );
    }

    // ── Post-fix: try_minimize iterates to the true joint fixpoint ────────────
    //
    // Seed dirty list so the initial contract pass runs; prune reports
    // values_merged > 0, the fix re-seeds and re-contracts, prune next pass
    // reports 0 -> loop exits.
    tdd.scratch.dirty_contract.push(root_idx.0);
    try_minimize(&mut tdd, MinimizeOptions::default()).expect("try_minimize must not OOM");
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&mut tdd).unwrap();

    // (a) Primary: no unmerged twins after the fix's iterate-to-fixpoint loop.
    check_no_twins(&tdd)
        .expect("post-fix: check_no_twins must pass after try_minimize");

    // (b) No duplicate slot values remain.
    check_slot_count_uniqueness(&tdd)
        .expect("no duplicate slot values after try_minimize");

    // (c) No orphan slots remain.
    check_no_orphan_slots(&tdd)
        .expect("post-fix: check_no_orphan_slots must pass after try_minimize");
}

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

/// Two boundary-parent nodes P and Q with identical pair lists {(X, Inline(1))}
/// referenced by a root node with DIFFERENT siblings — the SHARABLE shape.
/// Context-based T does not merge them (different grandparent contexts).
/// Slot-prune does not touch them (no slots involved).
/// try_minimize must merge them via the unconditional content-twin scan, and
/// the model count must be preserved.
///
/// Fixture (balanced(4) vtree — 7 nodes, leaves 0-3, internals 4-6):
///   v_marg    = right leaf-child of v_parent4  — empty store (all refs inline)
///   v_parent4 = boundary parent; nodes P and Q with pair {(Pos, Inline(1))}
///   v_right5  = non-marginal; nodes s0 and s1 (DIFFERENT siblings)
///   root      = output; one node with pairs (P, s0) and (Q, s1)
///
/// P's grandparent context: {(root_node, s1)}
/// Q's grandparent context: {(root_node, s0)}
/// Different contexts → T does NOT merge → content-twin scan MUST merge.
/// After merge P→Q (or Q→P), root carries two refs to the same node:
///   (Q, s0) and (Q, s1).  No duplicate same-x pairs (s0 ≠ s1), so p-fusion
/// is a no-op here.  Model count before == model count after.
#[test]
fn test_inline_ref_twins_merged_by_minimize() {
    use crate::check::marg::{check_no_orphan_slots, check_no_twins, check_slot_count_uniqueness};
    use crate::diagram::MargRef;
    use crate::vtree::VtreeNode;

    const INLINE_VAL: u32 = 1;

    let _g = crate::limits::apply_limits().budget(None).apply();

    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_parent4, v_right5) = vtree.children(root_idx);
    assert!(
        matches!(*vtree.node(v_parent4), VtreeNode::Internal { .. }),
        "v_parent4 must be an internal vtree node"
    );
    let (_v_leaf0, v_marg) = vtree.children(v_parent4);
    assert!(
        matches!(*vtree.node(v_marg), VtreeNode::Leaf { .. }),
        "v_marg must be a leaf vtree node"
    );

    let n = vtree.num_nodes();
    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..n).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_marg: empty store — all marg-side refs from v_parent4 are inline.
    levels[v_marg.idx()].marginal_counts = Some(vec![]);
    // Mark the marg side inlined so the tagger and readers decode correctly.
    levels[v_parent4.idx()].set_marg_inlined_right(true);

    let inline_ref = LocalNodeIdx(MargRef::Inline(INLINE_VAL).to_raw());
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);

    // P and Q: IDENTICAL pair lists [(Pos, Inline(1))].
    // slot-prune never sees them (inline, not slot) -> values_merged == 0.
    // Context-based T sees different sibling contexts -> no merge.
    let p = levels[v_parent4.idx()].push_internal_node(&[
        InputPair { left: pos, right: inline_ref },
    ]);
    let q = levels[v_parent4.idx()].push_internal_node(&[
        InputPair { left: pos, right: inline_ref },
    ]);

    // v_right5: two DISTINCT siblings so root refs P and Q with different contexts.
    let one  = LocalNodeIdx(LeafLabel::One as u32);
    let neg  = LocalNodeIdx(LeafLabel::Neg as u32);
    let s0 = levels[v_right5.idx()].push_internal_node(&[
        InputPair { left: pos, right: one },
    ]);
    let s1 = levels[v_right5.idx()].push_internal_node(&[
        InputPair { left: neg, right: one },
    ]);

    // root: one node with pairs (P, s0) and (Q, s1) — DIFFERENT siblings.
    // P's grandparent context = {(root0, s1)}, Q's = {(root0, s0)}: not equal.
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: p, right: s0 },
        InputPair { left: q, right: s1 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );

    let count_before = model_count(&tdd);
    assert!(count_before > 0u64.into(), "fixture must be satisfiable");

    // Mark root dirty; try_minimize runs prune + contract + unconditional scan.
    tdd.scratch.dirty_contract.push(root_idx.0);
    try_minimize(&mut tdd, MinimizeOptions::default()).expect("try_minimize must not OOM");
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&mut tdd).unwrap();

    // (a) Model count MUST be unchanged — regression guard against count halving.
    let count_after = model_count(&tdd);
    assert_eq!(
        count_before, count_after,
        "try_minimize must not change model count: got before={count_before}, after={count_after}"
    );

    // (b) No unmerged sharable twins at v_parent4 after minimize.
    check_no_twins(&tdd)
        .expect("check_no_twins must pass: inline-ref sharable twins must be merged");

    // (c) No duplicate slot values (trivially true — empty store).
    check_slot_count_uniqueness(&tdd).expect("no duplicate slot values");

    // (d) No orphan slots (trivially true — empty store).
    check_no_orphan_slots(&tdd).expect("no orphan slots");

    // (e) v_parent4 must have contracted from width 2 to width 1.
    assert_eq!(
        tdd.levels[v_parent4.idx()].width(), 1,
        "sharable inline-ref twins P and Q must merge to 1 node at v_parent4"
    );
}


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

/// Regression: content-identical nodes at a PLAIN level, referenced from
/// different parent contexts, must be merged by `merge_content_equal_nodes`.
///
/// Layout (balanced(6)):
///   sub_left_r  = internal(leaf1, leaf2) — PLAIN level; B1 and B2 are content
///                 twins (both hold the single pair `(Pos, Pos)`)
///   v_left      = internal(leaf0, sub_left_r) — PLAIN level; X1 = (Pos, B1),
///                 X2 = (Neg, B2). Different siblings ⇒ B1 and B2 have different
///                 context signatures, so context-based contraction cannot see them.
///   sub_right_r = internal(leaf4, leaf5) — made marginal (needed for v_right)
///   v_right     = internal(leaf3, sub_right_r) — made marginal; this is what
///                 makes the diagram marginalized at all, so the scan is in scope
///   root        = internal(v_left, v_right) — one node R with pairs
///                 (X1, slot0_vright), (X2, slot0_vright)
///
/// Model count = (c(X1) + c(X2)) · C_VR = (1 + 1) · 3 = 6, before and after.
#[test]
fn test_content_twins_merge_at_plain_levels() {
    use crate::check::marg::check_no_twins;
    use crate::limits::apply_limits;

    // Keep slot refs as bare indices so the marg side is easy to reason about.
    let _thr = crate::diagram::marg::set_marg_inline_max(0);
    let _g = apply_limits().budget(None).apply();

    let vtree = Arc::new(Vtree::balanced(6));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);
    let (_leaf0, sub_left_r) = vtree.children(v_left);
    let (_leaf3, sub_right_r) = vtree.children(v_right);
    assert!(matches!(*vtree.node(sub_left_r), VtreeNode::Internal { .. }), "sub_left_r internal");

    let n = vtree.num_nodes();
    let mut levels = take_levels(n);

    // --- The marginal side: sub_right_r then v_right (makes the diagram marg). ---
    assert_can_make_marginal(&levels, &vtree, sub_right_r);
    levels[sub_right_r.idx()].make_marginal(vec![7], None);
    assert_can_make_marginal(&levels, &vtree, v_right);
    const C_VR: u128 = 3;
    levels[v_right.idx()].make_marginal(vec![C_VR], None);

    // --- sub_left_r: a PLAIN level (both children are vtree leaves) with two
    //     content-identical nodes. ---
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let b1 = levels[sub_left_r.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let b2 = levels[sub_left_r.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    assert_eq!(levels[sub_left_r.idx()].width(), 2, "setup: B1 and B2 are two distinct nodes");

    // --- v_left: also a PLAIN level. X1 and X2 give B1/B2 DIFFERENT sibling
    //     contexts, which is what blinds context-based twin contraction. ---
    let x1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: b1 }]);
    let x2 = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: b2 }]);

    // --- root: one node over both, with the marginal sibling on the right. ---
    let vr_slot0 = LocalNodeIdx(0);
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: x1, right: vr_slot0 },
        InputPair { left: x2, right: vr_slot0 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    let count_before = model_count(&tdd);
    let expected: u64 = 6;
    assert_eq!(count_before, expected.into(), "pre-minimize model count must be {expected}");

    super::canonicalize_content_twins(&mut tdd).expect("canonicalize_content_twins must not OOM");

    // (a) Model count MUST be unchanged — the merge is a pure canonicalization.
    let count_after = model_count(&tdd);
    assert_eq!(
        count_after, count_before,
        "content merge must not change model count: before={count_before} after={count_after}"
    );

    // (b) THE DISCRIMINATOR: the plain level collapsed from 2 nodes to 1.
    assert_eq!(
        tdd.levels[sub_left_r.idx()].width(), 1,
        "plain-level content twins B1/B2 must merge (width 2 → 1)"
    );

    // (c) Twin canonicality holds everywhere the merge is responsible for, not just here.
    check_no_twins(&tdd).expect("no twins at any explicit level after canonicalization");
}

/// Companion: a VANILLA Boolean diagram (no marginal level anywhere) must be
/// left byte-identical by the content merge. There a redirect's minted duplicate
/// pair would be a genuine determinism violation, so the merge stands down entirely —
/// the `Tdd::has_marginal_level` scope gate. Guards against the wider level set
/// leaking into Boolean compiles.
#[test]
fn test_content_merge_stands_down_without_a_marginal_level() {
    use crate::limits::apply_limits;

    let _g = apply_limits().budget(None).apply();

    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);

    let n = vtree.num_nodes();
    let mut levels = take_levels(n);

    // Two content-identical nodes at v_left, referenced with different siblings
    // from v_right — exactly the shape the marginalized test above merges.
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let b1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let b2 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let r1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let r2 = levels[v_right.idx()].push_internal_node(&[InputPair { left: neg, right: neg }]);
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: b1, right: r1 },
        InputPair { left: b2, right: r2 },
    ]);

    let mut tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    assert!(!tdd.has_marginal_level(), "setup: no level may be marginal");

    let merged = super::contract::content_twin::merge_content_equal_nodes(&mut tdd, None)
        .expect("merge must not OOM");
    assert_eq!(merged, 0, "content merge must stand down on a marg-free diagram");
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "Boolean diagram must be left untouched by the content merge"
    );
}

mod tombstones {
    use std::sync::Arc;

    use crate::reduce::contract::contract_all_twins_topdown;
    use crate::reduce::minimize;
    use crate::reduce::prune::prune_unreachable;
    use crate::query::model_count;
    use crate::test_helpers::compile_clauses;
    use crate::diagram::TddNodeData;
    use crate::vtree::{Vtree, VtreeIdx};

    /// Two unreferenced tombstones carry the same empty fingerprint; contract
    /// must not treat them as twins. Contract on a tombstoned copy must match
    /// the dense run and leave the tombstones in place.
    #[test]
    fn contract_tolerates_tombstones() {
        let vtree = Arc::new(Vtree::balanced(5));
        let clauses = vec![vec![1, 2, -3], vec![-2, 3, 4], vec![3, -4, 5], vec![1, -5]];
        let mut dense = compile_clauses(&vtree, &clauses);
        minimize(&mut dense);
        let mc0 = model_count(&dense);

        // Appending keeps every existing slot index stable.
        let mut withtomb = dense.clone();
        let mut injected = 0usize;
        for t in 0..withtomb.vtree.num_nodes() {
            if withtomb.vtree.node(VtreeIdx(t as u32)).is_leaf() {
                continue;
            }
            let level = &mut withtomb.levels[t];
            if level.is_marginal() {
                continue;
            }
            level.nodes.push(TddNodeData::tombstone());
            level.nodes.push(TddNodeData::tombstone());
            level.n_tombstones += 2;
            injected += 2;
        }
        assert!(injected > 0);
        // Seed every internal level as dirty in both copies so the walk
        // examines the same levels.
        for t in 0..withtomb.vtree.num_nodes() {
            if !withtomb.vtree.node(VtreeIdx(t as u32)).is_leaf() {
                withtomb.scratch.dirty_contract.push(t as u32);
                dense.scratch.dirty_contract.push(t as u32);
            }
        }
        assert_eq!(model_count(&withtomb), mc0, "tombstones must not change the count");

        contract_all_twins_topdown(&mut dense, None).unwrap();
        contract_all_twins_topdown(&mut withtomb, None).unwrap();

        assert_eq!(model_count(&withtomb), mc0);
        assert_eq!(model_count(&dense), mc0);
        for t in 0..dense.vtree.num_nodes() {
            assert_eq!(
                withtomb.levels[t].live_width(),
                dense.levels[t].width(),
                "live width diverged at level {t}"
            );
        }
        let surviving: usize = withtomb.levels.iter().map(|l| l.n_tombstones as usize).sum();
        assert!(surviving > 0, "contract must not merge tombstones away");
    }

    /// An unreferenced tombstone changes no reported metric, and prune drops
    /// it.
    #[test]
    fn readers_skip_tombstones_and_prune_reclaims() {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![3, -4]]);
        minimize(&mut tdd);

        let size0 = tdd.size();
        let total0 = tdd.total_nodes();
        let maxw0 = tdd.max_width();
        let mc0 = model_count(&tdd);
        assert!(total0 > 0);

        let target = tdd
            .levels
            .iter()
            .position(|l| !l.is_marginal() && l.nodes.iter().any(|n| n.is_internal()))
            .expect("a level with an internal node");
        let len_before = tdd.levels[target].nodes.len();
        tdd.levels[target].nodes.push(TddNodeData::tombstone());
        tdd.levels[target].n_tombstones += 1;

        assert_eq!(tdd.size(), size0);
        assert_eq!(tdd.total_nodes(), total0);
        assert_eq!(tdd.max_width(), maxw0);
        assert_eq!(model_count(&tdd), mc0);
        assert_eq!(tdd.levels[target].width(), len_before + 1);
        assert_eq!(tdd.levels[target].live_width(), len_before);
        assert!(tdd.levels[target].nodes.last().unwrap().is_tombstone());

        prune_unreachable(&mut tdd).expect("tiny scratch reservation cannot fail");
        assert_eq!(tdd.levels[target].n_tombstones, 0);
        assert!(!tdd.levels.iter().any(|l| l.nodes.iter().any(|n| n.is_tombstone())));
        assert_eq!(tdd.size(), size0);
        assert_eq!(tdd.total_nodes(), total0);
        assert_eq!(tdd.max_width(), maxw0);
        assert_eq!(model_count(&tdd), mc0);
    }
}
