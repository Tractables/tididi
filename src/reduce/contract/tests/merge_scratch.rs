//! Merge-group concatenation and the scratch buffers behind it.
//!
//! Sibling of `content_twin_forking.rs`.

use crate::Engine;
use crate::diagram::{ValueRef, NodeIdx};
use crate::diagram::*;
use crate::vtree::Vtree;
use std::sync::Arc;
use crate::vtree::VtreeIdx;
use super::sweep::contract_all_twins;

/// A twin group holding both disjoint-support members and a content-equal duplicate
/// member concatenates the disjoint members; the duplicate member is not redirected.
///
/// # Fixture
///
///   t1 (explicit, non-marginal, no inlined side):  3 nodes A, B, C
///     A: pairs {(pos, one)}          — one pair
///     B: pairs {(pos, one)}          — same as A (content-equal)
///     C: pairs {(one, pos)}          — different from A (disjoint)
///   parent (explicit, left side inlined):
///     One node P with 3 pairs: (A, sib_slot0), (B, sib_slot0), (C, sib_slot0)
///   sib (marginal):  slot 0 → count 7
///
/// All three A, B, C are structural twins (same parent context: each appears
/// with sib_slot0 at parent node P).
///
/// `filtered = [A, C]` (disjoint pair sets), `duplicate_members = [B]`. Concat A+C;
/// B's `merge_target` stays B (identity) → B is canonical after compaction and
/// remains at t1. After concat A has `{(pos,one),(one,pos)}`, B has
/// `{(pos,one)}` — no longer content-equal → B stays unmerged.
///
/// Final: `t1.slot_count()` = 2 (A_merged and B); parent has 2 pairs (A_merged, slot0)
/// and (B, slot0).
#[test]
fn mixed_group_concats_disjoint_members_and_keeps_dup_member() {
    let eng = Engine::new();
    // Force all marginal refs onto slots (no inlining) so sib_slot refs stay as
    // bare slot indices — the scenario the duplicate_members detection depends on.

    // balanced(4):  root.left = v_left (internal), root.right = v_right (internal)
    // Use root as the parent, v_left as t1 (the target), v_right as the marginal sib.
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }),
        "v_left must be internal for this fixture");
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }),
        "v_right must be internal for this fixture");

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let sib_slot0 = NodeIdx(ValueRef::slot_raw(0));

    // Helper: build the test fixture diagram.
    let build_fixture = || {
        let mut levels: Vec<crate::diagram::TddLevel> =
            (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

        // t1 (v_left): 3 internal nodes A(0), B(1), C(2).
        // A and B are content-equal (same pair), C is disjoint.
        let a = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
        let b = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
        let c = levels[v_left.idx()].push_internal_node(&[ChildPair::new(one, pos)]);
        assert_eq!(a.0, 0); assert_eq!(b.0, 1); assert_eq!(c.0, 2);


        // sib (v_right): marginal with slot 0 → count 7.
        levels[v_right.idx()].become_marginal(vec![7u128], None);

        // parent (root): one multi-pair node P with 3 pairs — all t1 nodes share
        // sib_slot0.  This makes A, B, C structural twins (equal contexts).
        levels[root.idx()].push_internal_node(&[
            ChildPair::new(a, sib_slot0),    // A with sib slot0
            ChildPair::new(b, sib_slot0),    // B with sib slot0
            ChildPair::new(c, sib_slot0),    // C with sib slot0
        ]);
        // Mark parent as marginal-flagged so parent_marginal=true in `contract_twins`.
        // This is what enables the duplicate_members collection (content-equal twins
        // under a marginal-flagged parent).
        levels[root.idx()].set_has_value_refs(ChildSide::Left, true);

        let output = crate::diagram::TddNodeId {
            vtree: root,
            local: NodeIdx(0),
        };
        let mut tdd = crate::diagram::Tdd::from_levels_unchecked(vtree.clone(), levels, output);
        // Tag marginal-side refs for the boundary decode.
        crate::diagram::inline_small_marginal_refs(&mut tdd, None);
        tdd.seed_contract_worklist([root.0]);
        tdd
    };

    let mut tdd = build_fixture();
    contract_all_twins(&eng, &mut tdd).expect("contract_all_twins");

    // filtered=[A,C] → concat; B's merge_target stays B (canonical). A and B
    // are still twins after the first pass (both context={parent_node_0, slot0}), but
    // B overlaps A_merged (B's pair is a subset of A_merged's pairs) and is not
    // content-equal to it → B is in neither filtered nor duplicate_members → B never
    // merges. Final: t1 = {A_merged, B} → width 2.
    let t1_width = tdd.levels[v_left.idx()].slot_count();
    assert_eq!(t1_width, 2, "B must remain as a separate node (width=2); got {t1_width}");

    // Parent has exactly 2 pairs: the C-referencing pair was dropped (C mapped
    // to A); the B-referencing pair was kept (B canonical).
    let parent_pairs = tdd.levels[root.idx()].pair_count_at(0);
    assert_eq!(parent_pairs, 2, "parent must have 2 pairs (A_merged and B); got {parent_pairs}");
}

// Merge buffers are cleared before planning each level. Leftover plans could
// replay another level's groups.

#[test]
fn merge_buffers_clear_retains_allocations() {
    use super::merge::{GroupAction, GroupPlan};
    use super::scratch::MergeBuffers;

    let mut seen_pairs: rustc_hash::FxHashMap<(u32, u32), ()> = Default::default();
    seen_pairs.insert((5, 6), ());
    let mut b = MergeBuffers {
        resolve_keeps: vec![1],
        filtered: vec![2],
        duplicate_members: vec![3],
        keep_pairs_sorted: vec![(1, 2)],
        member_pairs: vec![(3, 4)],
        seen_pairs,
        sel: vec![7, 8],
        group_plans: vec![GroupPlan { action: GroupAction::Concat, start: 0, end: 2 }],
    };

    let allocation = b.sel.as_ptr();
    b.clear();
    assert_eq!(b.sel.as_ptr(), allocation);
    assert!(b.resolve_keeps.is_empty(), "resolve_keeps must be cleared");
    assert!(b.filtered.is_empty(), "filtered must be cleared");
    assert!(b.duplicate_members.is_empty(), "duplicate_members must be cleared");
    assert!(b.keep_pairs_sorted.is_empty(), "keep_pairs_sorted must be cleared");
    assert!(b.member_pairs.is_empty(), "member_pairs must be cleared");
    assert!(b.seen_pairs.is_empty(), "seen_pairs must be cleared");
    assert!(b.sel.is_empty(), "sel must be cleared");
    assert!(b.group_plans.is_empty(), "group_plans must be cleared");
}

// ── Budget: the contract-merge scratch buffers are charged ─────────────────

/// `contract_twins` grows three level-width scratch buffers (`merge_target`,
/// `duplicate_redirect`, `final_remap`). They must go through the budget-charged
/// `try_resize`, so a contraction that runs out of apply budget returns
/// `Err(OverBudget)` instead of allocating past the envelope — and it must do
/// so before any level is mutated.
///
/// Shape: a warm-up contraction over a level of the same width whose nodes are
/// not twins sizes every width-keyed scratch (the fingerprint tables), and the
/// group-keyed ones are grown directly, so the twin run below charges nothing
/// for those; it then finds its three merge buffers still empty, and a budget
/// smaller than the first of them (`merge_target`, `4 × width` bytes) must trip.
fn wide_twin_fixture(vtree: &Arc<Vtree>, width: usize, twins: bool) -> Tdd {
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let kinds = [
        ChildPair::new(pos, one),
        ChildPair::new(one, pos),
        ChildPair::new(neg, one),
        ChildPair::new(one, neg),
    ];

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    // Arenas pre-reserved so the twin run's grand reserve and parent re-encode
    // charge nothing: only the three scratch buffers are left to be charged.
    levels[v_left.idx()].pairs.reserve(8 * width);
    levels[v_left.idx()].ranges.reserve(width);
    levels[root.idx()].pairs.reserve(8 * width);
    levels[root.idx()].ranges.reserve(width);

    let mut nodes = Vec::with_capacity(width);
    for i in 0..width {
        nodes.push(levels[v_left.idx()].push_internal_node(&[kinds[i % kinds.len()]]));
    }

    // Marginal sibling: one slot shared by every parent pair (twins), or a
    // distinct slot per pair (not twins — raw slot index is the signature key).
    let slots = if twins { 1 } else { width };
    levels[v_right.idx()].become_marginal((1..=slots as u128).collect(), None);
    let pairs: Vec<ChildPair> = nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| ChildPair::new(n, NodeIdx(ValueRef::slot_raw(if twins { 0 } else { i as u32 }))))
        .collect();
    levels[root.idx()].push_internal_node(&pairs);

    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, output);
    inline_small_marginal_refs(&mut tdd, None);
    tdd.seed_contract_worklist([root.0]);
    tdd
}

#[test]
fn contract_merge_scratch_buffers_are_budget_charged() {
    let eng = Engine::new();
    let lim = eng.limits();
    use crate::limits::OperationError;
    let vtree = Arc::new(Vtree::balanced(4));
    let width = 64usize;

    // Warm-up: same width, no twins. Sizes the width-keyed fingerprint scratch
    // on this thread without ever reaching `contract_twins`.
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, _) = vtree.children(root);
    let mut warm = wide_twin_fixture(&vtree, width, false);
    contract_all_twins(&eng, &mut warm).expect("warm-up contraction");
    assert_eq!(warm.levels[v_left.idx()].slot_count(), width, "warm-up must not merge");
    // The group-keyed fingerprint buffers are sized by what the twin run finds,
    // which the twin-free warm-up cannot pre-size: grow them here, untracked and
    // generously, leaving the three merge buffers as the only cold scratch.
    {
        let mut s = eng.reduce_scratch().contract.checkout(lim);
        let big = 64 * width;
        s.flat_groups.resize_with(big, Default::default);
        s.group_starts.resize_with(big, Default::default);
        s.entries.resize_with(big, Default::default);
        s.counts.resize_with(big, Default::default);
        s.cursors.resize_with(big, Default::default);
        s.slice_unsorted.resize_with(big, Default::default);
        assert!(s.remap.merge_target.is_empty() && s.remap.duplicate_redirect.is_empty() && s.remap.final_remap.is_empty());
        drop(s);
    }

    // Twin run under a budget smaller than `merge_target` alone.
    let mut tdd = wide_twin_fixture(&vtree, width, true);
    lim.reset_meters();
    let budget = (4 * width - 1) as u64;
    let out = {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(budget));
        contract_all_twins(&eng, &mut tdd)
    };
    assert!(
        matches!(out, Err(OperationError::OverBudget)),
        "a contraction whose scratch buffers exceed the budget must return OverBudget, got {out:?}"
    );
    // The trip happened before any mutation: the level is untouched.
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), width, "the budget trip must precede the merge");
    assert_eq!(tdd.levels[root.idx()].pairs_of_idx(0).len(), width, "parent pairs untouched");
}

#[test]
fn contract_checkout_invalidates_the_previous_diagrams_marginal_map() {
    let eng = crate::Engine::new();
    {
        let mut scratch = eng.reduce_scratch().contract.checkout(eng.limits());
        scratch.has_marginal_below = vec![true, false];
        scratch.has_marginal_below_valid = true;
    }
    let scratch = eng.reduce_scratch().contract.checkout(eng.limits());
    assert!(!scratch.has_marginal_below_valid);
    assert_eq!(scratch.has_marginal_below.capacity(), 2);
}
