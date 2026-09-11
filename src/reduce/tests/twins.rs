//! Twin detection and contraction.
//!
//! The fixtures these read are in `mod.rs`.

use super::*;

use crate::engine::Engine;
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId, assert_can_make_marginal, take_levels,
};
use crate::test_helpers::assert_canonical;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use std::sync::Arc;


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
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    // balanced(3): leaves 0/1/2; level 3 = parent of leaves 0,1; level 4 = root (3, 2).
    assert!(matches!(*vtree.node(VtreeIdx(3)), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(VtreeIdx(4)), VtreeNode::Internal { .. }));

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: VtreeIdx(4), local: root },
    );

    let pairs_a_before: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(0).collect();
    let pairs_b_before: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(1).collect();

    let changed = contract_leaf_twins(&eng, &mut tdd).expect("nothing is armed");
    assert!(!changed, "all-or-nothing: any unmatched literal must veto rewrite on that side");

    let pairs_a_after: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(0).collect();
    let pairs_b_after: Vec<InputPair> = tdd.levels[3].pairs_iter_of_idx(1).collect();
    assert_eq!(pairs_a_before, pairs_a_after, "parent A's pairs must be untouched");
    assert_eq!(pairs_b_before, pairs_b_after, "parent B's pairs must be untouched");
    // No canonical-form assertion: the levels are hand-encoded and the leaf
    // levels are left implicit, which the structural checker reads as a
    // reference past the child's width. The claim here is that the two pair
    // lists are untouched.
}

/// Marginal-level twins induced by a parent-level restructure must be
/// contracted by `minimize`.
///
/// Walks through the full lifecycle:
///
/// **Phase 1.** Build a canonical 4-leaf diagram shaped `((x0, x1), (x2, x3))`.
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
    let eng = Engine::new();
    // Marginal-twin / pair fusion path: two marginal nodes at v_left with
    // DISTINCT counts (C_A ≠ C_B) both appearing paired with multiple
    // right-side siblings at the root → pair fusion closes the redex by
    // summing their counts, leaving one merged slot.
    //
    // We use distinct counts so that slot-prune does not merge them (value-dedup
    // only merges equal-valued slots). Equal-valued slots would be merged by
    // slot-prune before contraction fires, which is a separate (correct)
    // behaviour tested elsewhere. With distinct counts we can exercise the full
    // pair fusion path that fires for same-explicit-different-marginal-count
    // pairs. Both are too wide to fit a ref, so both stay slots.
    const C_A: u128 = (1u128 << 40) + 2;
    const C_B: u128 = (1u128 << 40) + 3;
    const C_SUM: u128 = C_A + C_B;

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), VtreeNode::Internal { .. }));

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    // ── Phase 1: canonical diagram with no twins anywhere ──────────────────
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
    let mut levels = take_levels(&eng, vtree.num_nodes());
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let r0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let r1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let root_node = levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: r0 },
        InputPair { left: b, right: r1 },
    ]);
    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );

    let phase1_size = tdd.size();
    let phase1_count = model_count(&tdd);
    minimize(&mut tdd);
    assert_eq!(tdd.size(), phase1_size, "phase 1: T0 has no twins; minimize must be a no-op");
    assert_canonical(&tdd);
    assert_eq!(model_count(&tdd), phase1_count);
    assert_eq!(tdd.levels[v_left.idx()].width(), 2);

    // ── Phase 2: marginalize v_left with DISTINCT counts ────────────────
    //
    // A has count C_A=2, B has count C_B=3 (distinct → slot-prune keeps
    // both, and no structural change occurs within this phase's minimize).
    assert_can_make_marginal(&tdd.levels, &vtree, v_left);
    tdd.levels[v_left.idx()].become_marginal(vec![C_A, C_B], None);
    // Hand-rolled become_marginal bypasses production marginalization; tag the
    // now-marginal level's persisted parent refs so the 0=inline decode
    // invariant holds (mirrors marginalize_batch / marginalize_subtree).
    crate::diagram::tag_all_marginal_side_slots(&mut tdd, None);

    let phase2_count = model_count(&tdd);
    assert_eq!(tdd.levels[v_left.idx()].width(), 2, "phase 2: marginalization preserves width");
    minimize(&mut tdd);
    // Distinct slot values → slot-prune does not merge them → width stays 2.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "phase 2: distinct-count slots must survive slot-prune; minimize must be a no-op at v_left",
    );
    assert_canonical(&tdd);
    assert_eq!(model_count(&tdd), phase2_count);

    // ── Phase 3: restructure root to induce a pair fusion redex at v_left ──
    //
    // Replace root's pair list with [(A, r0), (A, r1), (B, r0), (B, r1)].
    // Both A and B now appear paired with both r0 and r1. Pair fusion groups
    // by same-explicit-side: (A, r0) + (B, r0) → (C_SUM, r0), and
    // (A, r1) + (B, r1) → (C_SUM, r1). After the first round the root
    // has [(C_SUM, r0), (C_SUM, r1)] — same marginal ref again, so a
    // second slot-prune pass merges the two C_SUM slots into one.
    // The final result: width=1 and the surviving count = C_SUM.
    //
    // The diagram is no longer deterministic — fine for this test, which is
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
    tdd.seed_contract_worklist([root.0]);
    // The rebuilt root pairs reuse the bare phase-1 refs `a`/`b`, which now
    // point into the marginal v_left and must be tagged (production's
    // end-of-apply tagger does this after apply rebuilds the root level).
    crate::diagram::tag_all_marginal_side_slots(&mut tdd, None);

    let phase3_count = model_count(&tdd);

    minimize(&mut tdd);
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&eng, &mut tdd).unwrap();
    // After pair fusion + slot-prune: v_left should have exactly 1 surviving slot.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 1,
        "phase 3 (target): pair fusion at the marginal v_left must reduce to \
         one surviving slot (the sum C_A+C_B={C_SUM}). Parent pair list \
         must dedup from 4 entries down to 1 or 2.",
    );
    // The surviving slot's count must equal C_A + C_B = C_SUM.
    assert_eq!(
        tdd.levels[v_left.idx()].marginal_counts().unwrap()[0],
        C_SUM,
        "phase 3: surviving slot count must equal C_A + C_B = {C_SUM}",
    );
    // Sanity: model count is preserved across minimize. Defends against
    // an arithmetic mistake in the merge or remap.
    assert_eq!(model_count(&tdd), phase3_count);
    assert_canonical(&tdd);
}

/// A,B are scrambled-order twins (`[r0,r1]` vs `[r1,r0]`); a distinct node C
/// (paired only with r0) breaks sibling symmetry so no cascade masks the bug.
/// Exercises the open-addressing hash-bucket exact comparison; width 3 → 2.
#[test]
fn test_contract_detects_twins_with_scrambled_signature_order_width3() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );
    assert_eq!(tdd.levels[v_left.idx()].width(), 3, "setup: A, B (twins) + C (distinct)");
    let count_before = model_count(&tdd);

    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contraction must not OOM");

    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "the scrambled-order twins A,B must merge via the hash-bucket exact \
         comparison while the distinct node C survives → width 3 → 2",
    );
    assert_eq!(model_count(&tdd), count_before, "merge must preserve model count");
    assert_canonical(&tdd);
}

/// Stronger scramble: A and B are twins over THREE shared siblings in fully
/// reversed order (`[s0,s1,s2]` vs `[s2,s1,s0]`), so the signature sort must
/// reorder a multi-element slice (not just swap a pair). Two distinct nodes
/// C,D pin s0 and s1 respectively, leaving every sibling a unique signature so
/// no cascade canonicalizes anything; width 4 → 3.
#[test]
fn test_contract_detects_twins_with_reversed_multi_sibling_signature() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root, local: root_node },
    );
    assert_eq!(tdd.levels[v_left.idx()].width(), 4, "setup: A,B (twins) + C,D (distinct)");
    let count_before = model_count(&tdd);

    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contraction must not OOM");

    assert_eq!(
        tdd.levels[v_left.idx()].width(), 3,
        "twins over a 3-element reversed signature must merge while C,D survive \
         → width 4 → 3. The pre-fix order-sensitive `==` compared [s0,s1,s2] to \
         [s2,s1,s0], missed the twin, and left all 4 nodes (under-contraction).",
    );
    assert_eq!(model_count(&tdd), count_before, "merge must preserve model count");
    assert_canonical(&tdd);
}
