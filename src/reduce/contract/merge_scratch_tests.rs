//! Merge-group concatenation and the scratch buffers behind it.
//!
//! Sibling of `content_twin_tests.rs`.

use crate::engine::Engine;
use crate::diagram::*;
use crate::vtree::Vtree;
use std::sync::Arc;
use crate::vtree::VtreeIdx;
use super::strategies::contract_all_twins_topdown;

/// A twin group holding both disjoint-support members and a content-equal dup
/// member concatenates the disjoint members; the dup member is not redirected.
///
/// # Fixture
///
///   t1 (explicit, non-marg, marg_flags=0):  3 nodes A, B, C
///     A: pairs {(pos, one)}          — one pair
///     B: pairs {(pos, one)}          — SAME as A (content-equal)
///     C: pairs {(one, pos)}          — different from A (disjoint)
///   parent (explicit, marg_flags=MARG_INLINED_LEFT):
///     one node P with 3 pairs: (A, sib_slot0), (B, sib_slot0), (C, sib_slot0)
///   sib (marginal):  slot 0 → count 7
///
/// All three A, B, C are structural twins (same parent context: each appears
/// with sib_slot0 at parent node P).
///
/// `filtered = [A, C]` (disjoint pair sets), `dup_members = [B]`. Concat A+C;
/// B's `merge_target` stays B (identity) → B is canonical after compaction and
/// remains at t1. After concat A has `{(pos,one),(one,pos)}`, B has
/// `{(pos,one)}` — no longer content-equal → B stays unmerged.
///
/// Final: t1.width = 2 (A_merged and B); parent has 2 pairs (A_merged, slot0)
/// and (B, slot0).
#[test]
fn mixed_group_concats_disjoint_members_and_keeps_dup_member() {
    let eng = Engine::new();
    // Force all marg refs onto slots (no inlining) so sib_slot refs stay as
    // bare slot indices — the scenario the dup_members detection depends on.
    let _thr = crate::diagram::marg::set_marg_inline_max(0);

    // balanced(4):  root.left = v_left (internal), root.right = v_right (internal)
    // Use root as the parent, v_left as t1 (the target), v_right as the marginal sib.
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }),
        "v_left must be internal for this fixture");
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }),
        "v_right must be internal for this fixture");
    let (vl_left, vl_right) = vtree.children(v_left);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));

    // Helper: build the test fixture TDD.
    let build_fixture = || {
        let mut levels: Vec<crate::diagram::TddLevel> =
            (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

        // t1 (v_left): 3 internal nodes A(0), B(1), C(2).
        // A and B are content-equal (same pair), C is disjoint.
        let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
        let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
        let c = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
        assert_eq!(a.0, 0); assert_eq!(b.0, 1); assert_eq!(c.0, 2);

        // Leaf children of v_left — trivial leaf-label nodes.
        levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
        levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

        // sib (v_right): marginal with slot 0 → count 7.
        levels[v_right.idx()].make_marginal(vec![7u128], None);

        // parent (root): one multi-pair node P with 3 pairs — all t1 nodes share
        // sib_slot0.  This makes A, B, C structural twins (equal contexts).
        levels[root.idx()].push_internal_node(&[
            InputPair { left: a, right: sib_slot0 },    // A with sib slot0
            InputPair { left: b, right: sib_slot0 },    // B with sib slot0
            InputPair { left: c, right: sib_slot0 },    // C with sib slot0
        ]);
        // Mark parent as marg-flagged so parent_marg=true in contract_twins.
        // This is what enables the dup_members collection (content-equal twins
        // under a marg-flagged parent).
        levels[root.idx()].marg_flags = TddLevel::MARG_INLINED_LEFT;

        let output = crate::diagram::TddNodeId {
            vtree: root,
            local: LocalNodeIdx(0),
        };
        let mut tdd = crate::diagram::Tdd::with_levels(vtree.clone(), levels, output);
        // Tag marg-side refs for the boundary decode.
        crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
        tdd.dirty.contract.push(root.0);
        tdd
    };

    let mut tdd = build_fixture();
    contract_all_twins_topdown(&eng, &mut tdd, None).expect("contract_all_twins_topdown");

    // filtered=[A,C] → concat; B's merge_target stays B (canonical). A and B
    // are still twins after the first pass (both context={parent_node_0, slot0}), but
    // B overlaps A_merged (B's pair is a subset of A_merged's pairs) and is not
    // content-equal to it → B is in neither filtered nor dup_members → B never
    // merges. Final: t1 = {A_merged, B} → width 2.
    let t1_width = tdd.levels[v_left.idx()].width();
    assert_eq!(t1_width, 2, "B must remain as a separate node (width=2); got {t1_width}");

    // Parent has exactly 2 pairs: the C-referencing pair was dropped (C mapped
    // to A); the B-referencing pair was kept (B canonical).
    let parent_pairs = tdd.levels[root.idx()].pair_count_at(0);
    assert_eq!(parent_pairs, 2, "parent must have 2 pairs (A_merged and B); got {parent_pairs}");
}

// ── Scratch pooling: a checked-out buffer set is always empty ───────────────
//
// The pools carry CAPACITY across calls, never state. These pin the take-side
// clear for the buffer sets whose stale contents would be silently wrong rather
// than loud: a leftover `sel` / `group_plans` would make `contract_twins` commit
// a previous level's groups, and a leftover `remap` / `key_to_canonical` would
// redirect this level's refs onto a previous level's node indices.

#[test]
fn merge_buffers_are_cleared_on_take() {
    use super::merge::{GroupAction, GroupPlan};
    use super::scratch::{ContractScratch, MergeBuffers};

    let mut seen_pairs: rustc_hash::FxHashSet<(u32, u32)> = Default::default();
    seen_pairs.insert((5, 6));
    let mut scratch = ContractScratch::default();
    scratch.put_merge_buffers(MergeBuffers {
        resolve_keeps: vec![1],
        filtered: vec![2],
        dup_members: vec![3],
        keep_pairs_sorted: vec![(1, 2)],
        member_pairs: vec![(3, 4)],
        seen_pairs,
        sel: vec![7, 8],
        group_plans: vec![GroupPlan { action: GroupAction::Concat, start: 0, end: 2 }],
    });

    let b = scratch.take_merge_buffers();
    assert!(b.resolve_keeps.is_empty(), "resolve_keeps must be cleared on take");
    assert!(b.filtered.is_empty(), "filtered must be cleared on take");
    assert!(b.dup_members.is_empty(), "dup_members must be cleared on take");
    assert!(b.keep_pairs_sorted.is_empty(), "keep_pairs_sorted must be cleared on take");
    assert!(b.member_pairs.is_empty(), "member_pairs must be cleared on take");
    assert!(b.seen_pairs.is_empty(), "seen_pairs must be cleared on take");
    assert!(b.sel.is_empty(), "sel must be cleared on take");
    assert!(b.group_plans.is_empty(), "group_plans must be cleared on take");
}

#[test]
fn c2_scratch_is_cleared_on_take() {
    let eng = &crate::engine::Engine::new();
    use super::content_twin::{return_scratch, take_scratch, C2Scratch};

    let mut fp_counts: rustc_hash::FxHashMap<u64, u32> = Default::default();
    fp_counts.insert(11, 2);
    let mut key_to_canonical: rustc_hash::FxHashMap<Vec<(u32, u32)>, u32> = Default::default();
    key_to_canonical.insert(vec![(1, 2)], 3);
    return_scratch(eng, C2Scratch {
        node_fp: vec![11, 11],
        fp_counts,
        key_to_canonical,
        remap: vec![0, 0],
    });

    let s = take_scratch(eng);
    assert!(s.node_fp.is_empty(), "node_fp must be cleared on take");
    assert!(s.fp_counts.is_empty(), "fp_counts must be cleared on take");
    assert!(s.key_to_canonical.is_empty(), "key_to_canonical must be cleared on take");
    assert!(s.remap.is_empty(), "remap must be cleared on take");
}

// ── Budget: the contract-merge scratch buffers are charged ─────────────────

/// `contract_twins` grows three level-width scratch buffers (`merge_target`,
/// `dup_redirect`, `final_remap`). They must go through the budget-charged
/// `try_resize`, so a contraction that runs out of apply budget returns
/// `Err(OverBudget)` instead of allocating past the envelope — and it must do
/// so BEFORE any level is mutated.
///
/// Shape: a warm-up contraction over a level of the same width whose nodes are
/// NOT twins sizes every width-keyed scratch (the fingerprint tables), and the
/// group-keyed ones are grown directly, so the twin run below charges nothing
/// for those; it then finds its three merge buffers still empty, and a budget
/// smaller than the first of them (`merge_target`, `4 × width` bytes) must trip.
fn wide_twin_fixture(vtree: &Arc<Vtree>, width: usize, twins: bool) -> Tdd {
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);
    let kinds = [
        InputPair { left: pos, right: one },
        InputPair { left: one, right: pos },
        InputPair { left: neg, right: one },
        InputPair { left: one, right: neg },
    ];

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    // Arenas pre-reserved so the twin run's grand reserve and parent re-encode
    // charge nothing: only the three scratch buffers are left to be charged.
    levels[v_left.idx()].pairs.reserve(8 * width);
    levels[v_left.idx()].ext.reserve(width);
    levels[root.idx()].pairs.reserve(8 * width);
    levels[root.idx()].ext.reserve(width);

    let mut nodes = Vec::with_capacity(width);
    for i in 0..width {
        nodes.push(levels[v_left.idx()].push_internal_node(&[kinds[i % kinds.len()]]));
    }
    levels[vl_left.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::One)];

    // Marginal sibling: one slot shared by every parent pair (twins), or a
    // distinct slot per pair (not twins — raw slot index is the signature key).
    let slots = if twins { 1 } else { width };
    levels[v_right.idx()].make_marginal((1..=slots as u128).collect(), None);
    let pairs: Vec<InputPair> = nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| InputPair {
            left: n,
            right: LocalNodeIdx(MargRef::slot_raw(if twins { 0 } else { i as u32 })),
        })
        .collect();
    levels[root.idx()].push_internal_node(&pairs);

    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = Tdd::with_levels(vtree.clone(), levels, output);
    tag_all_marg_side_slots(&mut tdd, None);
    tdd.dirty.contract.push(root.0);
    tdd
}

#[test]
fn contract_merge_scratch_buffers_are_budget_charged() {
    let eng = Engine::new();
    let lim = eng.limits();
    use crate::error::ApplyError;
    let _thr = crate::diagram::marg::set_marg_inline_max(0);
    let vtree = Arc::new(Vtree::balanced(4));
    let width = 64usize;

    // Warm-up: same width, no twins. Sizes the width-keyed fingerprint scratch
    // on this thread without ever reaching `contract_twins`.
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, _) = vtree.children(root);
    let mut warm = wide_twin_fixture(&vtree, width, false);
    contract_all_twins_topdown(&eng, &mut warm, None).expect("warm-up contraction");
    assert_eq!(warm.levels[v_left.idx()].width(), width, "warm-up must not merge");
    // The group-keyed fingerprint buffers are sized by what the twin run finds,
    // which the twin-free warm-up cannot pre-size: grow them here, untracked and
    // generously, leaving the three merge buffers as the only cold scratch.
    {
        let mut s = super::scratch::take_scratch(&eng);
        let big = 64 * width;
        s.flat_groups.resize_with(big, Default::default);
        s.group_starts.resize_with(big, Default::default);
        s.entries.resize_with(big, Default::default);
        s.counts.resize_with(big, Default::default);
        s.cursors.resize_with(big, Default::default);
        s.slice_unsorted.resize_with(big, Default::default);
        assert!(s.merge_target.is_empty() && s.dup_redirect.is_empty() && s.final_remap.is_empty());
        super::scratch::return_scratch(&eng, s);
    }

    // Twin run under a budget smaller than `merge_target` alone.
    let mut tdd = wide_twin_fixture(&vtree, width, true);
    lim.reset_meters();
    let budget = (4 * width - 1) as u64;
    let out = {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(budget));
        contract_all_twins_topdown(&eng, &mut tdd, None)
    };
    assert!(
        matches!(out, Err(ApplyError::OverBudget)),
        "a contraction whose scratch buffers exceed the budget must return OverBudget, got {out:?}"
    );
    // The trip happened before any mutation: the level is untouched.
    assert_eq!(tdd.levels[v_left.idx()].width(), width, "the budget trip must precede the merge");
    assert_eq!(tdd.levels[root.idx()].pairs_of_idx(0).len(), width, "parent pairs untouched");
}
