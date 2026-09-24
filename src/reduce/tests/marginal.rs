//! Reduction over diagrams carrying marginal levels.
//!
//! The fixtures these read are in `mod.rs`.

use crate::Engine;
use crate::marginal::transition::free_subsumed_marginal_children;

use crate::diagram::{
    EncodedChildRef, ChildPair, LeafLabel, NodeIdx, Tdd, TddNodeId, assert_can_make_marginal, take_levels,
};
use crate::vtree::Vtree;
use std::sync::Arc;


/// Regression: marginal-sibling fold-allowed — fix in `merge_content_equal_nodes`.
///
/// Layout (balanced(6)):
///   sub_left_r = internal(leaf1, leaf2) — right child of v_left; made marginal (count `C_SLR`=5)
///   sub_right_r = internal(leaf4, leaf5) — right child of v_right; made marginal (needed for v_right)
///   v_left = internal(leaf0, sub_left_r) — boundary parent; Q1 and Q2 are content twins
///              (both have pair (LeafLabel::Pos, slot0_of_sub_left_r))
///   v_right = internal(leaf3, sub_right_r) — made marginal; the sibling side at root
///   root = grandparent: one node R with pairs [(Q1, slot0_vright), (Q2, slot0_vright)]
///
/// Pre-minimize `model_count` = Q1_count*`C_VR` + Q2_count*`C_VR` = 5*3 + 5*3 = 30.
///
/// Both v_left and root are pre-marked contracted=true so the initial
/// `contract_all_twins` in Engine::reduce is a no-op. This forces the content-twin scan to be the only
/// mechanism that handles the Q1/Q2 twin merge. The scan must then:
///   1. Perform the redirect Q2→Q1 (creating duplicate (Q1,slot0),(Q1,slot0) pairs at root).
///   2. Direct contract's pair fusion to fold the duplicate into one (Q1, slot1=2*`C_VR`) pair.
///   3. Let `prune_value_slots` compact v_right's store to a single slot with count 2*`C_VR`.
///
/// Duplicate pairs in a marginalized diagram are legal multiset entries, so
/// the redirect happens even though it produces one at the grandparent;
/// cancelling it instead would leave v_right's slot count at `C_VR`. The
/// discriminating assertion is (d): v_right's surviving count = 2*`C_VR`.
#[test]
fn test_marginal_sibling_fold_allowed_regression() {
    use crate::vtree::VtreeNode;

    // Every count below is too wide to fit a ref, so the slot refs stay bare
    // indices (not bit-30-tagged) — which is the shape this test is about.
    let eng = Engine::new();

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
    let mut levels = take_levels(&eng, n);

    // --- sub_left_r: make marginal (count `C_SLR`). This makes v_left a boundary parent. ---
    // sub_left_r's children are leaves (ok per `assert_can_make_marginal`).
    assert_can_make_marginal(&levels, &vtree, sub_left_r);
    const C_SLR: u128 = (1u128 << 40) + 5; // model count stored at sub_left_r's slot 0
    levels[sub_left_r.idx()].become_marginal(vec![C_SLR], None);

    // --- sub_right_r: make marginal (needed so v_right can be marginalized). ---
    assert_can_make_marginal(&levels, &vtree, sub_right_r);
    const C_SRR: u128 = (1u128 << 40) + 7; // model count stored at sub_right_r's slot 0; unused in count calc
    let _ = C_SRR;
    levels[sub_right_r.idx()].become_marginal(vec![C_SRR], None);

    // --- v_right: make marginal (count `C_VR`). This is the SIBLING of v_left at root. ---
    // When the redirect Q2→Q1 creates duplicate (Q1,slot0),(Q1,slot0) at root,
    // the fold_allowed check sees `v_right.is_marginal() == true` and allows the redirect.
    assert_can_make_marginal(&levels, &vtree, v_right);
    const C_VR: u128 = (1u128 << 40) + 3; // model count stored at v_right's slot 0
    levels[v_right.idx()].become_marginal(vec![C_VR], None);

    // --- v_left: two content-equal twin nodes Q1 and Q2. ---
    //
    // Each has one pair: left=LeafLabel::Pos (ref into plain leaf0), right=slot0_slr=0
    // (ref into marginal sub_left_r). Raw pair values (1, 0) are identical → twins ✓.
    let pos      = NodeIdx(LeafLabel::Pos as u32); // = NodeIdx(1)
    let slr_slot0 = NodeIdx(0); // slot index 0 of sub_left_r (marginal)
    let q1 = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, slr_slot0)]);
    let q2 = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, slr_slot0)]);
    assert_eq!(q1.idx(), 0, "Q1 must be node 0 at v_left");
    assert_eq!(q2.idx(), 1, "Q2 must be node 1 at v_left");

    // --- root: one node R with pairs (Q1, slot0_vright) and (Q2, slot0_vright). ---
    //
    // Both Q1 and Q2 are referenced with the same marginal sibling (slot0 of v_right).
    // After the fix the redirect Q2→Q1 is allowed (fold_allowed=true); root gets
    // (Q1,slot0),(Q1,slot0); pair fusion folds to (Q1, slot1=2*`C_VR`); prune compacts.
    let vr_slot0 = NodeIdx(0); // slot index 0 of v_right (marginal)
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        ChildPair::new(q1, vr_slot0),
        ChildPair::new(q2, vr_slot0),
    ]);

    // Pre-mark v_left and root as contracted (harmless when calling
    // `canonicalize_content_twins` directly, kept for documentation: the scan
    // must be the sole merge mechanism exercised here — not contract's fork-down
    // concat path — so the fold_allowed discriminator assertion (d) is clean).

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    // `v_right` is marginal over the marginal `sub_right_r`, whose store the
    // marginalization step frees as `v_right` becomes marginal.
    free_subsumed_marginal_children(&mut tdd.levels, &vtree, v_right, None);

    // Tag marginal-side slots so the `has_value_refs` markers are set on v_left
    // (right child sub_left_r is marginal) and root (right child v_right is marginal).
    // No count here fits a ref, so nothing inlines; the markers enable decode in
    // `model_count`.
    crate::diagram::inline_small_marginal_refs(&mut tdd, None);

    // Pre-minimize model count:
    //   Q1's count at v_left = Pos_leaf0 × `C_SLR` = 1 × `C_SLR`.
    //   Q2 identical.
    //   root = `C_SLR`×`C_VR` + `C_SLR`×`C_VR`.
    let count_before = tdd.model_count().unwrap();
    let expected_count_u: u128 = 2 * C_SLR * C_VR;
    assert_eq!(
        count_before,
        expected_count_u.into(),
        "pre-minimize model count must equal {expected_count_u}"
    );
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), 2, "setup: Q1 and Q2 are two distinct nodes");

    // Call `canonicalize_content_twins` directly: Engine::reduce's normal path does
    // not run the content-twin scan, so tests exercise it via the extracted pub(crate)
    // function.
    super::canonicalize_content_twins(&eng, &mut tdd).expect("canonicalize_content_twins must not OOM");

    // (a) Model count must be unchanged.
    let count_after = tdd.model_count().unwrap();
    assert_eq!(
        count_after, count_before,
        "minimize must not change model count: before={count_before} after={count_after}"
    );

    // (b) v_left must have contracted from width 2 to width 1 (Q2 merged).
    //     This holds on both fixed and unfixed code (contract's fork-down path also
    //     merges them when fold_allowed is absent). The key discriminator is (d).
    assert_eq!(
        tdd.levels[v_left.idx()].slot_count(), 1,
        "fold-allowed regression: Q1 and Q2 must merge at v_left (width 2 → 1)"
    );

    // (c) v_right must compact to exactly 1 slot after pair fusion + prune.
    assert_eq!(
        tdd.levels[v_right.idx()].slot_count(), 1,
        "v_right must compact to 1 slot after pair fusion folds (Q1,c),(Q1,c) → (Q1,2c)"
    );

    // (d) THE DISCRIMINATING ASSERTION: the surviving count at v_right must be 2*`C_VR`.
    //
    // On UNFIXED code: fold_allowed is absent; the redirect is cancelled; contract uses
    // the fork-down concat path which scales sub_left_r's count instead of v_right's.
    // v_right's slot stays at `C_VR`=3. This assertion fails: left=3, right=6.
    //
    // On FIXED code: fold_allowed fires; root gets duplicate (Q1,slot0),(Q1,slot0) pairs;
    // pair fusion folds them into (Q1, new_slot=2*`C_VR`=6); `prune_value_slots` compacts v_right
    // from [C_VR, 2*C_VR] down to [2*C_VR]. This assertion passes.
    assert_eq!(
        tdd.levels[v_right.idx()].marginal_counts().unwrap()[0],
        2 * C_VR,
        "v_right surviving slot count must equal 2*C_VR = {}",
        2 * C_VR,
    );
}

/// Companion: a VANILLA Boolean diagram (no marginal level anywhere) must be
/// left byte-identical by the content merge. There a redirect's minted duplicate
/// pair would be a genuine determinism violation, so the merge stands down entirely —
/// the `Tdd::has_marginal_level` scope gate. Guards against the wider level set
/// leaking into Boolean compiles.
#[test]
fn test_content_merge_stands_down_without_a_marginal_level() {
    let eng = &crate::Engine::new();

    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);

    let n = vtree.num_nodes();
    let mut levels = take_levels(eng, n);

    // Two content-identical nodes at v_left, referenced with different siblings
    // from v_right — exactly the shape the marginalized test above merges.
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let b1 = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let b2 = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let r1 = levels[v_right.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let r2 = levels[v_right.idx()].push_internal_node(&[ChildPair::new(neg, neg)]);
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        ChildPair::new(b1, r1),
        ChildPair::new(b2, r2),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );
    assert!(!tdd.has_marginal_level(), "setup: no level may be marginal");

    let merged = super::contract::content_twin::merge_content_equal_nodes(eng, &mut tdd, None)
        .expect("merge must not OOM");
    assert_eq!(merged, 0, "content merge must stand down on a marginal-free diagram");
    assert_eq!(
        tdd.levels[v_left.idx()].slot_count(), 2,
        "Boolean diagram must be left untouched by the content merge"
    );
}

/// Reduction relocates a weight-marginal level's store rows along with its
/// slots.
///
/// A weighted level keeps its per-node values in the diagram's `WeightStore`,
/// keyed by vtree level and indexed by slot. Slot-prune compacts that column —
/// orphaned slots go and the survivors move down — and rewrites the parent
/// refs to match. If the column and the refs were ever compacted apart, a
/// parent would read some other node's weight and the diagram's value would
/// change under a reduction that is supposed to preserve it.
///
/// The fixture forces the compaction to be a real move rather than a
/// truncation: the root's first pair is dropped, which orphans the lowest slot
/// its sides named, so every surviving slot above must shift down.
#[test]
fn minimize_relocates_weight_store_rows_with_their_slots() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, ChildDecoder, WeightStore};


    use crate::reduce::{ReductionPlan};
    use crate::test_helpers::{assert_canonical, compile_clauses, exact_weight, rat};

    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = vtree.root();
    let (left, right) = vtree.children(root);

    let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![2, -3], vec![3, 4], vec![-1, 4]]);
    tdd.set_weights(WeightStore::new(
        RationalWeights::from_literals(&[
            LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) },
            LiteralWeights { negative: rat(2, 5), positive: rat(3, 7) },
            LiteralWeights { negative: rat(5, 11), positive: rat(2, 9) },
            LiteralWeights { negative: rat(1, 1), positive: rat(4, 9) },
        ]),
        Arithmetic::ExactRational,
    )).unwrap();
    eng.marginalize_levels(&mut tdd, &[left, right]).expect("no wall is installed in a test");
    assert!(
        tdd.levels[left.idx()].is_weight_marginal() && tdd.levels[right.idx()].is_weight_marginal(),
        "setup: both root children must be weight-marginal"
    );

    // Drop the output node's first pair, orphaning the slots only it named.
    let out = tdd.output.local;
    let kept: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(out.idx())[1..].to_vec();
    assert!(kept.len() >= 2, "setup: the output node must keep several pairs");
    tdd.levels[root.idx()].replace_node_pairs(out, &kept);

    let slot = |side: EncodedChildRef| ChildDecoder::marginal().child(side).index();
    let lowest = |sides: &dyn Fn(&ChildPair) -> EncodedChildRef| {
        kept.iter().filter_map(|p| slot(sides(p))).min()
    };
    assert!(
        lowest(&|p: &ChildPair| p.left) > Some(0) || lowest(&|p: &ChildPair| p.right) > Some(0),
        "setup: a surviving pair must name a slot above the orphaned one, or nothing moves"
    );

    let column = |t: &Tdd, v| t.weights().expect("weighted").level(v).expect("column").len();
    let left_before = column(&tdd, left.idx());
    let right_before = column(&tdd, right.idx());
    let exact = |t: &Tdd| exact_weight(&t.weighted_value().unwrap().expect("a weighted diagram has a value"));
    let value_before = exact(&tdd);

    eng.reduce(&mut tdd, ReductionPlan::default()).expect("no budget is armed");
    assert_canonical(&tdd);

    assert_eq!(
        exact(&tdd),
        value_before,
        "reduction changed the weighted value: store rows and slots moved apart"
    );
    assert!(
        column(&tdd, left.idx()) < left_before || column(&tdd, right.idx()) < right_before,
        "setup: the orphaned slot must have been pruned, or nothing was relocated"
    );
}
