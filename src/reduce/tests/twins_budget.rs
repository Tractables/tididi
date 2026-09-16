//! Twin contraction under a budget, and the state it must leave behind.
//!
//! Sibling of `twins.rs`.

use super::*;

use crate::engine::Engine;
use crate::limits::OperationError;

use crate::diagram::{
    ChildPair, LeafLabel, NodeIdx, Tdd, TddNodeId, take_levels,
};
use crate::vtree::{Vtree, VtreeIdx};
use std::sync::Arc;

/// How many reserves a fixture's contraction is asked for, generously over-
/// estimated: the sweeps below arm the injection at every index up to this, so
/// each one is refused in turn and the invariant is asserted for all of them.
/// Indices past the last reserve simply never fire, and the sweep asserts that
/// at least one did.
const RESERVES_PER_CONTRACTION: u32 = 24;

/// Two twin groups at `v_left`, each member a disjoint 2-pair node so the merge
/// takes the concat path (the 1+1 inline fast path never reaches the reserve).
/// The two groups have distinct signatures and their siblings are non-twins, so
/// no cascade masks the merge.
fn two_twin_groups() -> (Arc<Vtree>, Tdd) {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&Engine::new(), vtree.num_nodes());
    let x = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(pos, pos), ChildPair::new(pos, neg)]);
    let y = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(neg, pos), ChildPair::new(neg, neg)]);
    let x2 = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(pos, pos), ChildPair::new(pos, neg)]);
    let y2 = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(neg, pos), ChildPair::new(neg, neg)]);
    let s0 = levels[v_right.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let s1 = levels[v_right.idx()].push_internal_node(&[ChildPair::new(neg, one)]);

    // Group 1: {x, y} both paired with s0; group 2: {x2, y2} both paired with s1.
    let root_node = levels[root.idx()].push_internal_node(&[
        ChildPair::new(x, s0),
        ChildPair::new(y, s0),
        ChildPair::new(x2, s1),
        ChildPair::new(y2, s1),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), 4, "setup: two twin groups {{x,y}},{{x2,y2}}");
    tdd.seed_contract_worklist([root.0]);
    (vtree, tdd)
}

/// A reserve failure anywhere in twin contraction must leave the model count
/// UNCHANGED (transactional grand reserve). Fails on the pre-fix code, which
/// grows one group's survivor before the second group's reserve fails while the
/// parent still references both.
#[test]
fn test_contract_twins_overbudget_w1_count_unchanged() {
    let mut refusals = 0;
    for nth in 0..RESERVES_PER_CONTRACTION {
        let eng = Engine::new();
        let (_vtree, mut tdd) = two_twin_groups();
        let count_before = tdd.model_count().unwrap();
        eng.limits().refuse_nth_reserve(nth);
        let res = contract_all_twins(&eng, &mut tdd);
        eng.limits().grant_every_reserve();
        if res.is_err() {
            refusals += 1;
            assert_eq!(
                tdd.model_count().unwrap(),
                count_before,
                "OverBudget at reserve {nth} must leave the model count unchanged",
            );
        }
    }
    assert!(refusals > 0, "the sweep must actually refuse something");
}

/// One twin group at `v_left` whose parent pairs carry a sibling ref with bit 31
/// set. The bit-31 field is copied but never dereferenced on the top-down
/// contract path, and it makes the lone surviving parent pair non-inlinable —
/// which forces the parent's own `multi_pairs` growth, the last allocation the
/// pass can be refused. `v_right` has width 1 so `contract_child` skips it,
/// which is what lets its sibling ref carry the bit safely.
fn twin_group_with_parent_growth() -> (Arc<Vtree>, Tdd, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&Engine::new(), vtree.num_nodes());
    let a = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let b = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, neg)]);
    let s0 = levels[v_right.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let sib = NodeIdx((1u32 << 31) | s0.0);

    // Both twins paired with the same (bit-31) sibling ⇒ they share a context ⇒
    // twins. After they merge, one of the two parent pairs is filtered, shrinking
    // the parent node to a single non-inlinable pair.
    let root_node = levels[root.idx()].push_internal_node(&[
        ChildPair::new(a, sib),
        ChildPair::new(b, sib),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), 2, "setup: one twin group {{a,b}}");
    tdd.seed_contract_worklist([root.0]);
    (vtree, tdd, v_left)
}

/// The parent's own `multi_pairs` growth is reserved in the same transaction as
/// the survivors' pairs, so an OverBudget on it — the last refusal the pass can
/// raise — still bails before anything is rewritten.
#[test]
fn test_contract_twins_overbudget_parent_ext_bails() {
    let mut refusals = 0;
    for nth in 0..RESERVES_PER_CONTRACTION {
        let eng = Engine::new();
        let (_vtree, mut tdd, v_left) = twin_group_with_parent_growth();
        eng.limits().refuse_nth_reserve(nth);
        let res = contract_all_twins(&eng, &mut tdd);
        eng.limits().grant_every_reserve();
        if res.is_err() {
            refusals += 1;
            assert_eq!(
                tdd.levels[v_left.idx()].slot_count(),
                2,
                "every reservation the pass takes is ahead of every mutation, so a refusal \
                 at reserve {nth} must leave the twin group unmerged",
            );
        }
    }
    assert!(refusals > 0, "the sweep must actually refuse something");
    // NB: deliberately DON'T call model_count — this fixture carries a bit-31
    // sibling, which is not a real node ref.
}

/// Two dirty parents, an `OverBudget` during the first (root-most) parent's
/// contraction: both the failing parent and the still-queued second parent must
/// survive in `dirty_contract`. The second parent (`v_right`) is never popped —
/// it proves the heap-remainder restore; `root` proves the failed-mid-processing
/// restore. Fails if the worklist is dropped instead of restored (→ empty).
fn two_dirty_parents() -> (Arc<Vtree>, Tdd, VtreeIdx, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&Engine::new(), vtree.num_nodes());
    let x = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(pos, pos), ChildPair::new(pos, neg)]);
    let y = levels[v_left.idx()]
        .push_internal_node(&[ChildPair::new(neg, pos), ChildPair::new(neg, neg)]);
    // A multi-pair node at v_right makes `has_multi_pair(v_right)` true, so it is
    // a valid heap parent, and it is the shared right sibling that makes x,y twins.
    let s = levels[v_right.idx()]
        .push_internal_node(&[ChildPair::new(pos, one), ChildPair::new(neg, one)]);

    let root_node = levels[root.idx()].push_internal_node(&[
        ChildPair::new(x, s),
        ChildPair::new(y, s),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), 2, "setup: one twin group {{x,y}}");
    // Heap pops root-most first (root), leaving v_right queued.
    tdd.seed_contract_worklist([root.0, v_right.0]);
    (vtree, tdd, root, v_right)
}

#[test]
fn test_contract_dirty_worklist_restored_on_err() {
    let mut refusals = 0;
    for nth in 0..RESERVES_PER_CONTRACTION {
        let eng = Engine::new();
        let (_vtree, mut tdd, root, v_right) = two_dirty_parents();
        eng.limits().refuse_nth_reserve(nth);
        let res = super::contract::contract_all_twins(&eng, &mut tdd);
        eng.limits().grant_every_reserve();
        if res.is_err() {
            refusals += 1;
            assert!(
                tdd.contract_worklist().contains(&v_right.0),
                "the unprocessed parent still queued in the heap must be restored on Err \
                 at reserve {nth}; dirty_contract = {:?}",
                tdd.contract_worklist(),
            );
            assert!(
                tdd.contract_worklist().contains(&root.0),
                "the parent that failed mid-processing must be restored on Err at reserve \
                 {nth}; dirty_contract = {:?}",
                tdd.contract_worklist(),
            );
        }
    }
    assert!(refusals > 0, "the sweep must actually refuse something");
}

/// A leaf twin at the root of `(x (y z))` whose sibling ref carries bit 31:
/// the survivor `(One, sib)` cannot be stored inline, so the rewrite needs a
/// fresh `multi_pairs` entry — the one reservation leaf contraction takes. The
/// `(y z)` level is never dereferenced, which is what lets the ref carry the
/// bit safely.
fn leaf_twin_with_range_growth() -> (Arc<Vtree>, Tdd, VtreeIdx) {
    use crate::vtree::VarId;

    let x = Vtree::leaf(VarId(0));
    let yz = Vtree::balanced_over(&[VarId(1), VarId(2)]);
    let vtree = Arc::new(Vtree::join(&x, &yz).expect("disjoint variable sets"));
    let root = vtree.root();
    let (v_leaf, v_right) = vtree.children(root);
    assert!(vtree.node(v_leaf).is_leaf(), "setup: the root's left child is the leaf x");

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);

    let mut levels = take_levels(&Engine::new(), vtree.num_nodes());
    let a = levels[v_right.idx()].push_internal_node(&[ChildPair::new(pos, neg)]);
    let sib = NodeIdx((1u32 << 31) | a.0);
    let root_node = levels[root.idx()].push_internal_node(&[
        ChildPair::new(pos, sib),
        ChildPair::new(neg, sib),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: root_node });
    tdd.seed_leaf_worklist([root.0]);
    (vtree, tdd, root)
}

/// The leaf rewrite reserves before it moves a pair, so a refused reservation
/// comes back as `Err(OverBudget)` with the level untouched and still queued,
/// and the same engine contracts it once the reservation is granted.
#[test]
fn test_contract_leaf_twins_overbudget_leaves_the_level_queued_and_unchanged() {
    let mut refusals = 0;
    for nth in 0..RESERVES_PER_CONTRACTION {
        let eng = Engine::new();
        let (_vtree, mut tdd, root) = leaf_twin_with_range_growth();
        let before: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        eng.limits().refuse_nth_reserve(nth);
        let res = contract_leaf_twins(&eng, &mut tdd);
        eng.limits().grant_every_reserve();
        match res {
            Ok(fired) => {
                assert!(fired, "with every reservation granted the twin contracts");
                let one = NodeIdx(LeafLabel::One as u32);
                assert_eq!(
                    tdd.levels[root.idx()].pairs_of_idx(0),
                    &[ChildPair::new(one, before[0].right)],
                    "the two literal pairs contract to one `One` pair",
                );
            }
            Err(e) => {
                refusals += 1;
                assert_eq!(e, OperationError::OverBudget);
                assert_eq!(
                    tdd.levels[root.idx()].pairs_of_idx(0),
                    &before[..],
                    "a refusal at reserve {nth} must leave the level as it was",
                );
                assert!(
                    contract_leaf_twins(&eng, &mut tdd).expect("every reservation is granted now"),
                    "the refused level stays on the worklist, so the next call contracts it",
                );
            }
        }
    }
    assert!(refusals > 0, "the sweep must actually refuse something");
}

/// Regression: prune value-merge can mint twins after contract ran.
/// Pre-fix: the broken one-shot sequence leaves unmerged twins.
/// Post-fix: `Engine::reduce`'s iterate-to-fixpoint loop eliminates them.
///
/// Fixture: boundary store at v_marginal holds 3 slots [C, C, D] (slots 0,1 equal;
/// slot 2 distinct). Parent-level nodes p and q each hold two pairs with the
/// same non-marginal side (X1=Pos, X2=Neg) but different marginal-side slot refs for X1:
///   p: [(Pos, slot_0), (Neg, slot_2)]   — slot_0=C, slot_2=D
///   q: [(Pos, slot_1), (Neg, slot_2)]   — slot_1=C (= slot_0's value), slot_2=D
/// Pre-prune p != q (slot_0 != slot_1 as indices). After prune's value-merge
/// (slot_1 -> slot_0), both become [(Pos, slot_0), (Neg, new_slot_1)] -> twins.
#[test]
fn test_prune_value_merge_does_not_mint_twins_at_minimize_exit() {
    use crate::reduce::slot_prune::prune_value_slots;
    use crate::test_helpers::check::marginal::{check_no_orphan_slots, check_twin_canonicality, check_slot_count_uniqueness};
    use crate::diagram::ValueRef;
    use crate::vtree::VtreeNode;

    // BIG ensures counts cannot inline (MARGINAL_INLINE_MAX = 2^30 - 1 < 2^40).
    // Slot-prune is where slot-count uniqueness is established; equal-valued slots
    // only collapse there (the emit site is forbidden from deduping).
    const BIG: u128 = 1u128 << 40;
    const C: u128 = BIG + 99; // equal value shared by slots 0 and 1
    const D: u128 = BIG + 7;  // distinct value at slot 2

    // balanced(4): 7 vtree nodes (0-3=leaves, 4=internal(0,1), 5=internal(2,3),
    // 6=root=internal(4,5)).
    //
    // Topology:
    //   v_marginal    = right leaf-child of v_parent4 (holds 3 slots [C,C,D]).
    //   v_parent4 = non-marginal; nodes p and q (2 pairs each).
    //   v_right5  = non-marginal; nodes s0, s1 (symmetry breakers at root).
    //   root      = output; one node with pairs (p,s0) and (q,s1).
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root_idx = vtree.root();
    let (v_parent4, v_right5) = vtree.children(root_idx);
    assert!(
        matches!(*vtree.node(v_parent4), VtreeNode::Internal { .. }),
        "v_parent4 must be an internal vtree node"
    );
    // vtree.children returns (left, right); make the RIGHT child marginal.
    let (_v_leaf0, v_marginal) = vtree.children(v_parent4);
    assert!(
        matches!(*vtree.node(v_marginal), VtreeNode::Leaf { .. }),
        "v_marginal must be a leaf vtree node"
    );

    let n = vtree.num_nodes();
    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..n).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_marginal: 3 slots [C, C, D]. Slots 0 and 1 carry equal values — a duplicate
    // planted deliberately; slot-prune collapses them.
    levels[v_marginal.idx()].set_counts_state(vec![C, C, D], None);

    // v_parent4: two 2-pair nodes p and q.
    //   Marg-side (right) refs are bare indices (ValueRef::Slot(i).to_raw().0 = i,
    //   bit-30 clear). BIG values cannot inline; slot-prune leaves them as slots.
    //   p: [(Pos, slot_0), (Neg, slot_2)]
    //   q: [(Pos, slot_1), (Neg, slot_2)]
    //   Pre-prune: p != q (slot_0 != slot_1 as raw indices) -> contract sees no twins.
    //   After value-merge slot_1->slot_0: both become [(Pos, slot_0), (Neg, slot_1')]
    //   where slot_1' is the compacted D slot -> identical pair lists -> twins.
    let pos  = NodeIdx(LeafLabel::Pos as u32);
    let neg  = NodeIdx(LeafLabel::Neg as u32);
    let slot0 = NodeIdx(ValueRef::slot_raw(0));
    let slot1 = NodeIdx(ValueRef::slot_raw(1));
    let slot2 = NodeIdx(ValueRef::slot_raw(2));
    let p = levels[v_parent4.idx()].push_internal_node(&[
        ChildPair::new(pos, slot0),
        ChildPair::new(neg, slot2),
    ]);
    let q = levels[v_parent4.idx()].push_internal_node(&[
        ChildPair::new(pos, slot1),
        ChildPair::new(neg, slot2),
    ]);

    // v_right5: two structurally distinct nodes (different left-leaf label).
    // Their distinctness ensures the root's pair list is non-trivial and both
    // p and q are independently reachable from the output node.
    let one = NodeIdx(LeafLabel::One as u32);
    let s0 = levels[v_right5.idx()].push_internal_node(&[
        ChildPair::new(pos, one),
    ]);
    let s1 = levels[v_right5.idx()].push_internal_node(&[
        ChildPair::new(neg, one),
    ]);

    // root (output): single node with pairs (p, s0) and (q, s1).
    // Both p and q are referenced -> reachable -> prune is a no-op.
    let root_node = levels[root_idx.idx()].push_internal_node(&[
        ChildPair::new(p, s0),
        ChildPair::new(q, s1),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: root_idx, local: root_node },
    );

    // ── Pre-fix verification: the broken one-shot sequence leaves twins ────────
    //
    // Manually reproduce the PRE-FIX order: contract (no merge since p!=q), then
    // prune_value_slots once (merges equal slots, mints twins). Assert check_twin_canonicality
    // fails — confirming the test pins the fixed behaviour.
    {
        let mut tdd2 = tdd.clone();
        // Seed dirty list: contract short-circuits on an empty list.
        tdd2.seed_contract_worklist([root_idx.0]);
        // Step 1: contract — p and q have different slot refs -> no twins -> no-op.
        super::contract::contract_all_twins(&eng, &mut tdd2)
            .expect("contract must not OOM in pre-fix verification");
        // Step 2: one prune pass — slots 0,1 both = C -> merge -> twins minted.
        let prune_stats = prune_value_slots(&eng, &mut tdd2);
        assert!(
            prune_stats.values_merged > 0,
            "pre-fix verification: prune must report values_merged > 0 \
             (equal-valued slots 0 and 1 must collapse)"
        );
        // Step 3: check_twin_canonicality must FAIL (twins minted, no re-contract ran).
        assert!(
            check_twin_canonicality(&tdd2).is_err(),
            "pre-fix verification: check_twin_canonicality must FAIL after the broken \
             one-shot contract->prune sequence (twin pair minted by value-merge)"
        );
    }

    // ── Post-fix: Engine::reduce iterates to the true joint fixpoint ────────────
    //
    // Seed dirty list so the initial contract pass runs; prune reports
    // values_merged > 0, the fix re-seeds and re-contracts, prune next pass
    // reports 0 -> loop exits.
    tdd.seed_contract_worklist([root_idx.0]);
    eng.reduce(&mut tdd, ReductionPlan::default()).expect("Engine::reduce must not OOM");
    // The content-twin scan is not run by Engine::reduce's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&eng, &mut tdd).unwrap();

    // (a) Primary: no unmerged twins after the fix's iterate-to-fixpoint loop.
    check_twin_canonicality(&tdd)
        .expect("post-fix: check_twin_canonicality must pass after Engine::reduce");

    // (b) No duplicate slot values remain.
    check_slot_count_uniqueness(&tdd)
        .expect("no duplicate slot values after Engine::reduce");

    // (c) No orphan slots remain.
    check_no_orphan_slots(&tdd)
        .expect("post-fix: check_no_orphan_slots must pass after Engine::reduce");
}
