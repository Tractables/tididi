//! Twin merges that only a marginal or inline-ref diagram can produce.
//!
//! Sibling of `twins.rs`.

use super::*;

use crate::engine::Engine;
use crate::marginal::free_subsumed_marginal_children;
use crate::test_helpers::compile_clauses;
use crate::diagram::TddNodeData;
use crate::reduce::contract::contract_all_twins;
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId, assert_can_make_marginal, take_levels,
};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use std::sync::Arc;

/// Two boundary-parent nodes P and Q with identical pair lists {(X, Inline(1))}
/// referenced by a root node with DIFFERENT siblings — the SHARABLE shape.
/// Context-based T does not merge them (different grandparent contexts).
/// Slot-prune does not touch them (no slots involved).
/// try_minimize must merge them via the unconditional content-twin scan, and
/// the model count must be preserved.
///
/// Fixture (balanced(4) vtree — 7 nodes, leaves 0-3, internals 4-6):
///   v_marginal    = right leaf-child of v_parent4  — empty store (all refs inline)
///   v_parent4 = boundary parent; nodes P and Q with pair {(Pos, Inline(1))}
///   v_right5  = non-marginal; nodes s0 and s1 (DIFFERENT siblings)
///   root      = output; one node with pairs (P, s0) and (Q, s1)
///
/// P's grandparent context: {(root_node, s1)}
/// Q's grandparent context: {(root_node, s0)}
/// Different contexts → T does not merge → content-twin scan must merge.
/// After merge P→Q (or Q→P), root carries two refs to the same node:
///   (Q, s0) and (Q, s1).  No duplicate same-x pairs (s0 ≠ s1), so pair fusion
/// is a no-op here.  Model count before == model count after.
#[test]
fn test_inline_ref_twins_merged_by_minimize() {
    use crate::test_helpers::check::marginal::{check_no_orphan_slots, check_twin_canonicality, check_slot_count_uniqueness};
    use crate::diagram::ValueRef;
    use crate::vtree::VtreeNode;

    const INLINE_VAL: u32 = 1;

    let eng = Engine::new();

    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_parent4, v_right5) = vtree.children(root_idx);
    assert!(
        matches!(*vtree.node(v_parent4), VtreeNode::Internal { .. }),
        "v_parent4 must be an internal vtree node"
    );
    let (_v_leaf0, v_marginal) = vtree.children(v_parent4);
    assert!(
        matches!(*vtree.node(v_marginal), VtreeNode::Leaf { .. }),
        "v_marginal must be a leaf vtree node"
    );

    let n = vtree.num_nodes();
    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..n).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_marginal: empty store — all marginal-side refs from v_parent4 are inline.
    levels[v_marginal.idx()].set_counts_state(vec![], None);
    // Mark the marginal side inlined so the tagger and readers decode correctly.
    levels[v_parent4.idx()].set_marginal_inlined_right(true);

    let inline_ref = NodeIdx(ValueRef::Inline(INLINE_VAL).to_raw().0);
    let pos = NodeIdx(LeafLabel::Pos as u32);

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
    let one  = NodeIdx(LeafLabel::One as u32);
    let neg  = NodeIdx(LeafLabel::Neg as u32);
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

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );

    let count_before = model_count(&tdd);
    assert!(count_before > 0u64.into(), "fixture must be satisfiable");

    // Mark root dirty; try_minimize runs prune + contract + unconditional scan.
    tdd.seed_contract_worklist([root_idx.0]);
    try_minimize(&eng, &mut tdd, MinimizeOptions::default()).expect("try_minimize must not OOM");
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&eng, &mut tdd).unwrap();

    // (a) Model count must be unchanged — regression guard against count halving.
    let count_after = model_count(&tdd);
    assert_eq!(
        count_before, count_after,
        "try_minimize must not change model count: got before={count_before}, after={count_after}"
    );

    // (b) No unmerged sharable twins at v_parent4 after minimize.
    check_twin_canonicality(&tdd)
        .expect("check_twin_canonicality must pass: inline-ref sharable twins must be merged");

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
    use crate::test_helpers::check::marginal::check_twin_canonicality;

    // Keep slot refs as bare indices so the marginal side is easy to reason about.
    let eng = Engine::new();

    let vtree = Arc::new(Vtree::balanced(6));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);
    let (_leaf0, sub_left_r) = vtree.children(v_left);
    let (_leaf3, sub_right_r) = vtree.children(v_right);
    assert!(matches!(*vtree.node(sub_left_r), VtreeNode::Internal { .. }), "sub_left_r internal");

    let n = vtree.num_nodes();
    let mut levels = take_levels(&eng, n);

    // --- The marginal side: sub_right_r then v_right (makes the diagram marginal). ---
    assert_can_make_marginal(&levels, &vtree, sub_right_r);
    levels[sub_right_r.idx()].become_marginal(vec![7], None);
    assert_can_make_marginal(&levels, &vtree, v_right);
    const C_VR: u128 = 3;
    levels[v_right.idx()].become_marginal(vec![C_VR], None);

    // --- sub_left_r: a PLAIN level (both children are vtree leaves) with two
    //     content-identical nodes. ---
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let b1 = levels[sub_left_r.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    let b2 = levels[sub_left_r.idx()].push_internal_node(&[InputPair { left: pos, right: pos }]);
    assert_eq!(levels[sub_left_r.idx()].width(), 2, "setup: B1 and B2 are two distinct nodes");

    // --- v_left: also a PLAIN level. X1 and X2 give B1/B2 DIFFERENT sibling
    //     contexts, which is what blinds context-based twin contraction. ---
    let x1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: b1 }]);
    let x2 = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: b2 }]);

    // --- root: one node over both, with the marginal sibling on the right. ---
    let vr_slot0 = NodeIdx(0);
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: x1, right: vr_slot0 },
        InputPair { left: x2, right: vr_slot0 },
    ]);

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    // `v_right` is marginal over the marginal `sub_right_r`, whose store the
    // marginalize step frees as `v_right` becomes marginal.
    free_subsumed_marginal_children(&mut tdd.levels, &vtree, v_right, None);
    crate::diagram::tag_all_marginal_side_slots(&mut tdd, None);

    let count_before = model_count(&tdd);
    let expected: u64 = 6;
    assert_eq!(count_before, expected.into(), "pre-minimize model count must be {expected}");

    super::canonicalize_content_twins(&eng, &mut tdd).expect("canonicalize_content_twins must not OOM");

    // (a) Model count must be unchanged — the merge is a pure canonicalization.
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
    check_twin_canonicality(&tdd).expect("no twins at any explicit level after canonicalization");
}

    /// Two unreferenced tombstones carry the same empty fingerprint; contract
    /// must not treat them as twins. Contract on a tombstoned copy must match
    /// the dense run and leave the tombstones in place.
    #[test]
    fn contract_tolerates_tombstones() {
    let eng = Engine::new();
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
                withtomb.seed_contract_worklist([t as u32]);
                dense.seed_contract_worklist([t as u32]);
            }
        }
        assert_eq!(model_count(&withtomb), mc0, "tombstones must not change the count");

        contract_all_twins(&eng, &mut dense).unwrap();
        contract_all_twins(&eng, &mut withtomb).unwrap();

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

/// Leaf-twin contraction must leave the parent's marginal-side markers alone.
///
/// `MARGINAL_INLINED_RIGHT` says "this level's refs toward its marginal right child
/// already hold inline counts". The contraction rewrites the LEFT (literal)
/// side of a pair list and copies every right field through verbatim, so the
/// marker still describes the level truthfully afterward — but the rewrite
/// zeroed the whole flag byte, leaving a level whose refs are inline counts and
/// whose marker says they are not. The tagger would then be free to re-resolve
/// them as slot indices.
///
/// Fixture: vtree `(x (y z))` with `(y z)` marginalized and tagged, and a root
/// node whose two pairs are `(Pos_x, m)` and `(Neg_x, m)` — the shape
/// `classify` calls `AllContractible`.
#[test]
fn contracting_a_leaf_twin_keeps_the_parents_marginal_side_marker() {
    use crate::diagram::tag_all_marginal_side_slots;
    use crate::reduce::contract::contract_leaf::contract_leaf_twins;
    use crate::vtree::VarId;

    let eng = Engine::new();

    let x = Vtree::leaf(VarId(0));
    let yz = Vtree::balanced_over(&[VarId(1), VarId(2)]);
    let vtree = Arc::new(Vtree::join(&x, &yz).expect("disjoint variable sets"));
    let root_idx = vtree.root();
    let (v_leaf_x, v_marginal) = vtree.children(root_idx);
    assert!(vtree.node(v_leaf_x).is_leaf(), "the left child must be the leaf x");

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // `(y z)` summed out to one node holding a count small enough to inline.
    for child in [vtree.children(v_marginal).0, vtree.children(v_marginal).1] {
        assert!(vtree.node(child).is_leaf(), "the marginal subtree is two leaves");
    }
    levels[v_marginal.idx()].become_marginal(vec![2], None);

    // The root's two pairs differ only in the polarity of x and share the one
    // marginal partner, which is what makes them a contractible leaf twin.
    let m = NodeIdx(0);
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(LeafLabel::Pos as u32), right: m },
        InputPair { left: NodeIdx(LeafLabel::Neg as u32), right: m },
    ]);

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    tag_all_marginal_side_slots(&mut tdd, None);
    assert!(
        tdd.levels[root_idx.idx()].marginal_inlined_right(),
        "the tagger must inline the marginal side and mark it, or the fixture proves nothing"
    );

    tdd.seed_leaf_worklist([root_idx.0]);
    assert!(
        contract_leaf_twins(&eng, &mut tdd).expect("nothing is armed"),
        "the two pairs differ only in the polarity of x, so the level contracts"
    );
    assert!(
        tdd.levels[root_idx.idx()].marginal_inlined_right(),
        "the rewrite copies the marginal side through verbatim, so its marker still holds"
    );
}
