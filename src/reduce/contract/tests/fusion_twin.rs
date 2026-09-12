//! Where pair fusion and twin contraction meet: a fusion redex that creates
//! twins, and the fork-down content twins above a boundary parent.

use crate::diagram::MarginalSide;
use crate::diagram::*;
use crate::diagram::{NodeIdx, ValueRef};
use crate::engine::Engine;
use crate::vtree::Vtree;
use crate::vtree::VtreeIdx;
use std::sync::Arc;

use super::strategies::contract_all_twins;

// ── Change-C: directed fixture — fusion redex creates twins ─────────────

/// Directed joint-fixpoint fixture (change C).
///
/// Constructs a level where closing pair fusion redexes CREATES structural twins:
/// One root node holds four pairs referencing two explicit nodes A and B.
/// A's group uses slots 0 and 1 (COUNT_A + COUNT_B = COUNT_SUM); B's group
/// uses slots 2 and 3 (COUNT_C + COUNT_D = COUNT_SUM). Before fusion, A and B
/// have DIFFERENT parent contexts, so they are not twins. After fusion, both
/// groups yield the same summed count COUNT_SUM, and slot-count uniqueness maps them to the same
/// surviving slot — giving A and B identical contexts. Since A and B also have
/// identical child pairs, the joint fixpoint loop must detect and merge them.
/// The merge is a content-equal duplicate redirect: the root keeps both pairs
/// remapped onto the survivor and pair fusion folds them, leaving one root node
/// with one pair whose count is 2·COUNT_SUM — the denoted value is
/// MC(A)·(c_A+c_B) + MC(B)·(c_C+c_D) = MC(A)·2·COUNT_SUM, so multiplicity is
/// SUMMED into the count; set-dedup to COUNT_SUM would halve the model count.
///
/// Fixture (balanced(4)):
///   v_right = marginal; four slots: COUNT_A, COUNT_B, COUNT_C, COUNT_D
///             (all distinct; COUNT_A+COUNT_B = COUNT_C+COUNT_D = COUNT_SUM)
///   v_left  = explicit internal; two nodes A and B with identical child pairs
///   root    = one multi-pair node:
///               (A, slot_0), (A, slot_1), (B, slot_2), (B, slot_3)
///
/// Pre-fusion: A's context = {(root0,slot_0),(root0,slot_1)},
///             B's context = {(root0,slot_2),(root0,slot_3)} → DIFFERENT → not twins
/// Post-fusion: (A, slot_sum), (B, slot_sum) → A and B both context {(root0,slot_sum)}
///              → structural twins → duplicate redirect merge at v_left →
///              root: {(merged, slot_sum), (merged, slot_sum)} → pair fusion →
///              root: {(merged, slot_2sum)} with 2·COUNT_SUM
#[test]
fn fusion_creates_twin_both_closed_in_one_call() {
    let eng = Engine::new();

    // Four distinct counts; two pairs summing to the same total.
    const COUNT_A: u128 = 1_000_000_000_000u128;
    const COUNT_B: u128 = 2_000_000_000_000u128;
    const COUNT_C: u128 = 500_000_000_000u128;
    const COUNT_D: u128 = 2_500_000_000_000u128;
    const COUNT_SUM: u128 = COUNT_A + COUNT_B; // = COUNT_C + COUNT_D = 3e12

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(
        *vtree.node(v_left),
        crate::vtree::VtreeNode::Internal { .. }
    ));
    assert!(matches!(
        *vtree.node(v_right),
        crate::vtree::VtreeNode::Internal { .. }
    ));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::diagram::TddLevel> = (0..vtree.num_nodes())
        .map(|_| crate::diagram::TddLevel::new())
        .collect();

    // v_left: two nodes A and B with IDENTICAL child pairs (Pos, One).
    // Before fusion their contexts in root differ (different slots); after
    // fusion they share the same slot_sum context → structural twins.
    let a = levels[v_left.idx()].push_internal_node(&[ChildPair {
        left: pos,
        right: one,
    }]);
    let b = levels[v_left.idx()].push_internal_node(&[ChildPair {
        left: pos,
        right: one,
    }]);

    // Leaf children of v_left.
    levels[vl_left.idx()].nodes = vec![crate::diagram::EncodedNode::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::EncodedNode::leaf(LeafLabel::One)];

    // v_right: marginal sibling with FOUR distinct slots (all different counts).
    levels[v_right.idx()].become_marginal(vec![COUNT_A, COUNT_B, COUNT_C, COUNT_D], None);
    let slot_0 = NodeIdx(ValueRef::slot_raw(0)); // COUNT_A  }
    let slot_1 = NodeIdx(ValueRef::slot_raw(1)); // COUNT_B  } sum = COUNT_SUM
    let slot_2 = NodeIdx(ValueRef::slot_raw(2)); // COUNT_C  }
    let slot_3 = NodeIdx(ValueRef::slot_raw(3)); // COUNT_D  } sum = COUNT_SUM

    // root: one node with four pairs.
    //   A's group: (A, slot_0), (A, slot_1) → pair fusion redex → fuses to (A, slot_sum)
    //   B's group: (B, slot_2), (B, slot_3) → pair fusion redex → fuses to (B, slot_sum)
    //                                         (same count COUNT_SUM → same slot)
    // A's pre-fusion context  = {(root0, slot_0), (root0, slot_1)} ← different from B's
    // B's pre-fusion context  = {(root0, slot_2), (root0, slot_3)} → not twins yet
    // Post-fusion both become = {(root0, slot_sum)}                 → NOW twins
    levels[root.idx()].push_internal_node(&[
        ChildPair {
            left: a,
            right: slot_0,
        },
        ChildPair {
            left: a,
            right: slot_1,
        },
        ChildPair {
            left: b,
            right: slot_2,
        },
        ChildPair {
            left: b,
            right: slot_3,
        },
    ]);

    let output = crate::diagram::TddNodeId {
        vtree: root,
        local: NodeIdx(0),
    };
    let mut tdd = crate::diagram::Tdd::from_levels_unchecked(vtree, levels, output);

    // Tag marginal-side refs for consistent boundary decode.
    crate::diagram::tag_all_marginal_side_slots(&mut tdd, None);

    // Precondition A: fusion redexes are present.
    assert!(
        crate::test_helpers::check::marginal::check_pair_fusion_saturation(&tdd, None).is_err(),
        "fixture must start WITH pair fusion redexes"
    );

    // Mark root dirty and run the joint pipeline (change B joint fixpoint).
    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contract_all_twins");

    // Postcondition A: no fusion redexes remain.
    crate::test_helpers::check::marginal::check_pair_fusion_saturation(&tdd, None)
        .unwrap_or_else(|e| panic!("fusion redex survived after fixpoint: {e}"));

    // Postcondition B: no unmerged twins at the parent-of-marginal level (root).
    crate::test_helpers::check::marginal::check_twin_canonicality(&tdd)
        .unwrap_or_else(|e| panic!("twin pair survived after fixpoint: {e}"));

    // Postcondition C: v_left contracted from 2 nodes (A, B) to 1 (merged twin).
    let vl_width = tdd.levels[v_left.idx()].slot_count();
    assert_eq!(
        vl_width, 1,
        "v_left must have 1 node after twin-merge of A and B; got {vl_width}",
    );

    // Postcondition D: root has one pair with count = 2·COUNT_SUM. The fixture
    // denotes MC(A)·(c_A+c_B) + MC(B)·(c_C+c_D) = MC(A)·2·COUNT_SUM, so the
    // twin merge must SUM the multiplicity into the count (duplicate redirect +
    // pair fusion fold) — a set-dedup ending at COUNT_SUM would halve the count.
    let root_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(
        root_pairs.len(),
        1,
        "surviving root node must have 1 pair; got {}",
        root_pairs.len()
    );
    let marginal_raw = root_pairs[0].right.0;
    let marginal_counts = tdd.levels[v_right.idx()].marginal_counts().unwrap();
    let count = match ValueRef::from_raw(MarginalSide(marginal_raw)) {
        ValueRef::Slot(s) => marginal_counts[s as usize],
        ValueRef::Inline(c) => c as u128,
    };
    assert_eq!(
        count,
        2 * COUNT_SUM,
        "surviving pair must decode to 2*COUNT_SUM={}; got {count}",
        2 * COUNT_SUM,
    );
}

// ── Fork-down: content twins ABOVE the boundary parent (plain level) ────
