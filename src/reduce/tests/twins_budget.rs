//! Twin contraction under a budget, and the state it must leave behind.
//!
//! Sibling of `twins.rs`.

use super::*;

use crate::engine::Engine;
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId, take_levels,
};
use crate::vtree::{Vtree, VtreeIdx};
use std::sync::Arc;

/// A reserve failure across twin groups must leave the model count
/// UNCHANGED (transactional grand reserve). FAILS on the pre-fix code, which
/// grows one group's survivor before the second group's reserve fails while the
/// parent still references both.
#[test]
fn test_contract_twins_overbudget_w1_count_unchanged() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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

    tdd.seed_contract_worklist([root.0]);
    // Consult #1 (first group's reserve) succeeds; consult #2 (second group's
    // reserve) fires. On the fixed code both consults hit the single hoisted
    // grand reserve, so the bail happens before any mutation.
    super::contract::arm_fail_after(&eng, 1);
    let res = contract_all_twins(&eng, &mut tdd);
    super::contract::disarm_fail(&eng);

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(!tdd.poisoned, "a cross-group reserve failure must bail transactionally, not poison");
    assert_eq!(
        model_count(&tdd),
        count_before,
        "OverBudget in contract_twins must leave the model count unchanged",
    );
}

/// An OverBudget in the mid parent-rewrite "shrink-to-1 but can't inline"
/// branch is IRRECOVERABLE — earlier parent pairs are already remapped and there
/// is no clean rollback — so it must set `tdd.poisoned`. The count extractor then
/// refuses the diagram (see `test_model_count_refuses_poisoned_tdd`).
#[test]
fn test_contract_twins_overbudget_w2_poisons() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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
    let sib = NodeIdx((1u32 << 31) | s0.0);

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

    tdd.seed_contract_worklist([root.0]);
    // Consults on the v_left edge: #0 (grand-reserve pairs), #1 (grand-reserve
    // ext), #2 at the mid-rewrite ext push. Fire #2.
    super::contract::arm_fail_after(&eng, 2);
    let res = contract_all_twins(&eng, &mut tdd);
    super::contract::disarm_fail(&eng);

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(
        tdd.poisoned,
        "an OverBudget mid parent-rewrite must poison the TDD",
    );
    // NB: deliberately DON'T call model_count(&tdd) — it is poisoned (would trip
    // the backstop assert) and carries a bit-31 sibling (not a real node ref).
}

/// Seed TWO dirty parents, fire an `OverBudget` during the FIRST (root-most)
/// parent's contraction, and assert BOTH the failing parent and the still-queued
/// second parent survive in `dirty_contract`. The second parent (`v_right`) is
/// never popped — it proves the heap-remainder restore; `root` proves the
/// failed-mid-processing restore. FAILS if the worklist is dropped instead of
/// restored (→ empty).
#[test]
fn test_contract_dirty_worklist_restored_on_err() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels = take_levels(&eng, vtree.num_nodes());
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
    tdd.seed_contract_worklist([root.0, v_right.0]);

    // Fire on the very first consult — the grand reserve inside root's
    // contract_twins — so root fails mid-processing while v_right is still queued.
    super::contract::arm_fail_after(&eng, 0);
    let res = super::contract::contract_all_twins_topdown(&eng, &mut tdd, None);
    super::contract::disarm_fail(&eng);

    assert!(res.is_err(), "the injected OverBudget must surface as Err");
    assert!(
        !tdd.poisoned,
        "a grand-reserve failure bails transactionally, not poison",
    );
    assert!(
        tdd.contract_worklist().contains(&v_right.0),
        "the unprocessed parent still queued in the heap must be restored on Err; \
         dirty_contract = {:?}",
        tdd.contract_worklist(),
    );
    assert!(
        tdd.contract_worklist().contains(&root.0),
        "the parent that failed mid-processing must be restored on Err; \
         dirty_contract = {:?}",
        tdd.contract_worklist(),
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
    use crate::diagram::ValueRef;
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
    let eng = Engine::new();
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
    levels[v_marg.idx()].set_counts_state(vec![C, C, D], None);

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
    let one = NodeIdx(LeafLabel::One as u32);
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
        tdd2.seed_contract_worklist([root_idx.0]);
        // Step 1: contract — p and q have different slot refs -> no twins -> no-op.
        super::contract::contract_all_twins_topdown(&eng, &mut tdd2, None)
            .expect("contract must not OOM in pre-fix verification");
        // Step 2: one prune pass — slots 0,1 both = C -> merge -> twins minted.
        let prune_stats = prune_marg_slots(&eng, &mut tdd2);
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
    tdd.seed_contract_worklist([root_idx.0]);
    try_minimize(&eng, &mut tdd, MinimizeOptions::default()).expect("try_minimize must not OOM");
    // The content-twin scan is not run by try_minimize's normal path, so
    // call the canonicalization machinery directly so the assertions hold.
    canonicalize_content_twins(&eng, &mut tdd).unwrap();

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
