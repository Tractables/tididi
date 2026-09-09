//! Reduction over diagrams carrying marginal levels.
//!
//! Sibling of `tests.rs`, which holds the fixtures these read.

use crate::engine::Engine;
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId, assert_can_make_marginal, take_levels,
};
use crate::vtree::Vtree;
use std::sync::Arc;


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
    use crate::vtree::VtreeNode;

    // Prevent inlining so slot refs stay as bare indices (not bit-30-tagged).
    // With threshold=0 no count c satisfies c <= 0, so all refs stay as slot indices.
    let _thr = crate::diagram::marg::set_marg_inline_max(0);
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
    let pos      = NodeIdx(LeafLabel::Pos as u32); // = NodeIdx(1)
    let slr_slot0 = NodeIdx(0); // slot index 0 of sub_left_r (marginal)
    let q1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: slr_slot0 }]);
    let q2 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: slr_slot0 }]);
    assert_eq!(q1.idx(), 0, "Q1 must be node 0 at v_left");
    assert_eq!(q2.idx(), 1, "Q2 must be node 1 at v_left");

    // --- root: one node R with pairs (Q1, slot0_vright) and (Q2, slot0_vright). ---
    //
    // Both Q1 and Q2 are referenced with the SAME marginal sibling (slot0 of v_right).
    // After the fix the redirect Q2→Q1 is allowed (fold_allowed=true); root gets
    // (Q1,slot0),(Q1,slot0); p-fusion folds to (Q1, slot1=2*C_VR); prune compacts.
    let vr_slot0 = NodeIdx(0); // slot index 0 of v_right (marginal)
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
    super::canonicalize_content_twins(&eng, &mut tdd).expect("canonicalize_content_twins must not OOM");

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
    let eng = &crate::engine::Engine::new();

    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_left, v_right) = vtree.children(root_idx);

    let n = vtree.num_nodes();
    let mut levels = take_levels(eng, n);

    // Two content-identical nodes at v_left, referenced with different siblings
    // from v_right — exactly the shape the marginalized test above merges.
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
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

    let merged = super::contract::content_twin::merge_content_equal_nodes(eng, &mut tdd, None)
        .expect("merge must not OOM");
    assert_eq!(merged, 0, "content merge must stand down on a marg-free diagram");
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "Boolean diagram must be left untouched by the content merge"
    );
}
