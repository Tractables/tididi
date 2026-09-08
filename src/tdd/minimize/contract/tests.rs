use crate::tdd::types::*;
use crate::vtree::Vtree;
use std::sync::Arc;
use crate::vtree::VtreeIdx;

use super::strategies::contract_all_twins_topdown;

// ── Test: explicit twin contraction when the SIBLING side is marginal ───

/// Two nodes at an explicit (non-marginal) target level are structural twins
/// when every parent pair referencing them uses the SAME sibling slot ref.
///
/// Fixture (balanced(4)):
///   v_left  = explicit internal vtree node; two TDD nodes A, B
///   v_right = marginal sibling; one slot, count 3
///   root    = one multi-pair node: pairs (A, sib_slot0), (B, sib_slot0)
///
/// Because both A and B appear with the SAME sibling raw (slot 0), they are
/// structurally identical in the parent's context — `find_twin_groups` must
/// detect them as twins and `minimize` must merge them into one node.
#[test]
fn twins_with_marginal_sibling_are_contracted() {
    // Force all marg refs onto slots (inline threshold = 0) so the sibling
    // side uses bare slot indices — the scenario this test is about.
    let _thr = crate::tdd::types::set_marg_inline_max(0);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    // v_left must be internal (so its TDD nodes can be explicit internal nodes).
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // Two explicit twin nodes A and B at v_left, each with one pair.
    // Their content differs but they will be twins because the parent
    // pairs them with the identical sibling ref.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    // Leaf children of v_left — give them trivial leaf-label nodes.
    levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // v_right: marginal sibling with a single slot carrying count 3.
    levels[v_right.idx()].make_marginal(vec![3u128], None);
    // Tag the marginal side so slot 0's raw ref is slot_raw(0).
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));

    // Root: one multi-pair node with TWO pairs — both A and B use the SAME
    // sibling slot 0. This makes A and B structural twins.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot0 },
    ]);

    let output = crate::tdd::types::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs so the boundary decode is consistent.
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
    // Declare root dirty so contract_all_twins_topdown picks it up.
    tdd.dirty_contract.push(root.0);

    // Run the full contraction pipeline.
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // A and B were structural twins (same parent context) → must merge to 1.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 1,
        "explicit twins with a common marginal sibling slot must contract to width 1",
    );
    // The parent must have had its duplicate pair removed (2 → 1).
    let parent_node_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(
        parent_node_pairs.len(), 1,
        "parent pair list must collapse from 2 to 1 after twin merge; got {} pairs",
        parent_node_pairs.len(),
    );
}

/// When two parent pairs reference different sibling SLOTS that happen to
/// carry EQUAL counts, the nodes they point to are NOT structural twins —
/// the signature key is the raw slot index, not the decoded count.
///
/// In production, C3 ensures two slots with equal counts never coexist, so
/// this scenario cannot arise via the normal pipeline. For compile_marginalize-
/// path stores C3 is enforced at birth via `dedup_fresh_store`; for apply-emit-
/// born stores it is established at post-tagger slot-prune (`prune_marg_slots`).
/// This test constructs the scenario directly to document and pin the
/// contraction logic's raw-ref semantics: distinct-slot refs prevent merge
/// regardless of count equality.
///
/// Fixture: same as `twins_with_marginal_sibling_are_contracted` but the
/// two parent pairs use DIFFERENT sibling slots (slot 0 and slot 1) both
/// carrying count 3. The nodes A and B are NOT contracted.
#[test]
fn twins_with_marginal_sibling_distinct_slots_not_contracted() {
    let _thr = crate::tdd::types::set_marg_inline_max(0);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // Two sibling slots with EQUAL counts (both 3) but DIFFERENT raw indices.
    levels[v_right.idx()].make_marginal(vec![3u128, 3u128], None);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    let sib_slot1 = LocalNodeIdx(MargRef::slot_raw(1));

    // Root: A paired with slot0, B paired with slot1 — different sibling raws.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot1 },
    ]);

    let output = crate::tdd::types::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
    tdd.dirty_contract.push(root.0);

    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // A and B have DIFFERENT sibling slot raws → different signatures → NOT twins.
    assert_eq!(
        tdd.levels[v_left.idx()].width(), 2,
        "nodes with different sibling slot raws (even if counts equal) must NOT merge; \
         canon must redirect equal-count slots first. width = {}",
        tdd.levels[v_left.idx()].width(),
    );
}

/// INLINE sibling refs encode the count VALUE in the raw, so equal counts
/// produce equal raws — explicit twins whose shared context is an inline
/// marginal count are detected with no canon pass needed. This is the
/// inline counterpart of the two tests above: the SAME fixture as
/// `twins_with_marginal_sibling_distinct_slots_not_contracted` (two
/// distinct slots, equal counts), but with the inline threshold raised so
/// the tagger rewrites both slot refs to `Inline(5)`. Where the slot form
/// blocked the merge (distinct raw indices), the inline form merges —
/// inlining acts as canonicalization-by-value.
#[test]
fn twins_with_equal_inline_sibling_counts_are_contracted() {
    // Inline threshold ABOVE the counts: tagger converts slot refs → inline.
    let _thr = crate::tdd::types::set_marg_inline_max(64);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // Two DISTINCT slots carrying EQUAL counts (5) — the configuration the
    // slot-form test proves is NOT contracted when refs stay bare slots.
    levels[v_right.idx()].make_marginal(vec![5u128, 5u128], None);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    let sib_slot1 = LocalNodeIdx(MargRef::slot_raw(1));

    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot1 },
    ]);

    let output = crate::tdd::types::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Tagger rewrites both small-count slot refs to Inline(5) — equal raws.
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
    for p in tdd.levels[root.idx()].pairs_of_idx(0) {
        match MargRef::from_raw(p.right.0) {
            MargRef::Inline(c) => assert_eq!(c, 5, "tagger must inline count 5"),
            other => panic!("sibling ref must be inline after tagging, got {other:?}"),
        }
    }
    tdd.dirty_contract.push(root.0);

    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    assert_eq!(
        tdd.levels[v_left.idx()].width(), 1,
        "explicit twins sharing an EQUAL inline sibling count must contract; \
         inline raws compare by value so no canon pass is required",
    );
    let parent_node_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(parent_node_pairs.len(), 1, "duplicate pair must be removed");
    match MargRef::from_raw(parent_node_pairs[0].right.0) {
        MargRef::Inline(c) => assert_eq!(c, 5, "merged pair keeps the inline count"),
        other => panic!("merged sibling must stay inline, got {other:?}"),
    }
}

// ── Test: overflow promotion in marginal p-fusion sum ────────────────────

/// Two marginal slots whose counts sum to > u128::MAX must be fused correctly
/// by p-fusion (the ONLY mechanism for marginal-side redexes after change C).
///
/// Fixture: the two slots share the same explicit sibling `n` — this is a
/// p-fusion redex. p-fusion sums the counts through the seeded SlotInterner,
/// which performs BigUint promotion when the sum overflows u128. The old
/// `merge_twin_marginal_counts` path (twin contraction on marginal levels) is
/// deleted; this test now goes through p-fusion alone.
///
/// p-fusion leaves the original slots (C0, C1) in the marginal level and
/// appends a NEW slot (index 2) holding the sum. The root pair collapses from
/// 2 to 1, referencing the new slot. The old slots become unreferenced
/// (compacted by a subsequent minimize pass); their presence here is expected.
///
/// Fixture (balanced(4)):
///   v_left  = marginal, two slots (C0, C1) — p-fusion redex at root
///   v_right = explicit internal, one node `n`
///   root    = one multi-pair node: pairs (slot0, n) and (slot1, n)
///             (same sibling n, different marginal refs → p-fusion redex)
#[test]
fn marginal_slot_twins_sum_with_overflow_promotion() {
    let _thr = crate::tdd::types::set_marg_inline_max(0); // force slot refs; no inlining

    const OVERFLOW: u128 = u128::MAX;
    // Two counts whose sum overflows u128: (u128::MAX - 2) + 10 = u128::MAX + 8
    const C0: u128 = u128::MAX - 2;
    const C1: u128 = 10u128;

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vr_left, vr_right) = vtree.children(v_right);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // v_left: marginal level with two slots — both referenced from root.
    levels[v_left.idx()].make_marginal(vec![C0, C1], None);

    // v_right: explicit internal with one node `n` (single pair, leaf children).
    let n = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_right — trivial leaf-label nodes.
    levels[vr_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vr_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // root: one multi-pair node with TWO pairs — both reference the same
    // explicit sibling `n` but different marginal refs (slot0, slot1). This is
    // a p-fusion redex: same-x-different-marg-ref pairs at the same node.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: LocalNodeIdx(MargRef::slot_raw(0)), right: n },
        InputPair { left: LocalNodeIdx(MargRef::slot_raw(1)), right: n },
    ]);

    let output = crate::tdd::types::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs and mark root dirty; the full pipeline closes the redex.
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
    tdd.dirty_contract.push(root.0);
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // The parent must have had its duplicate pair fused (2 → 1) by p-fusion.
    let root_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(
        root_pairs.len(), 1,
        "root pair list must collapse from 2 to 1 after p-fusion; got {} pairs",
        root_pairs.len(),
    );

    // p-fusion leaves old slots and appends a NEW slot for the sum.
    // v_left grows: [C0, C1] → [C0, C1, sum_slot]. Old slots stay (unreferenced,
    // to be compacted by a later minimize pass).
    let counts = tdd.levels[v_left.idx()].marginal_counts.clone().unwrap();
    assert_eq!(
        counts.len(), 3,
        "v_left must have 3 slots after p-fusion (C0, C1, sum_slot); got {}",
        counts.len(),
    );

    // The new slot (index 2) must carry the OVERFLOW sentinel.
    assert_eq!(
        counts[2], OVERFLOW,
        "new sum slot must hold OVERFLOW sentinel for (u128::MAX-2)+10; got {}",
        counts[2],
    );

    // The BigUint side-table must hold the true sum for the new slot.
    let big = tdd.levels[v_left.idx()].marginal_counts_big.as_ref()
        .expect("marginal_counts_big must be Some after overflow promotion");
    let big_val = big.get(2)
        .expect("the overflow table must carry an entry keyed by the new sum slot");
    use num_bigint::BigUint;
    let expected = BigUint::from(C0) + BigUint::from(C1);
    assert_eq!(
        *big_val, expected,
        "BigUint side-table entry must equal (u128::MAX-2)+10 = {expected}; got {big_val}",
    );

    // The surviving root pair must reference the new sum slot.
    let sum_slot_raw = root_pairs[0].left.0;
    match MargRef::from_raw(sum_slot_raw) {
        MargRef::Slot(s) => assert_eq!(
            s, 2,
            "surviving pair must reference new sum slot (index 2); got slot {s}",
        ),
        MargRef::Inline(c) => panic!(
            "surviving pair must be a slot ref, not inline({c})",
        ),
    }
}

// ── Test: p-fusion redex resolved within contract_all_twins_topdown ──────

/// A fixture where a parent node already holds two pairs with the SAME
/// explicit element but DIFFERENT marginal-side count refs — a p-fusion
/// redex — and assert that a single `contract_all_twins_topdown` call (with
/// the parent marked dirty) fuses it to ONE pair whose count is the sum.
///
/// This tests change B's wire-in: p-fusion now runs inside the per-parent
/// joint fixpoint loop of `contract_all_twins_topdown`, so the combined
/// pipeline closes the redex without a separate `apply_p_fusion` call.
///
/// Fixture (balanced(4)):
///   v_right = marginal; two slots: COUNT_A and COUNT_B (different, large)
///   v_left  = explicit internal; one node `n`
///   root    = one multi-pair node: pairs (n, slot_a), (n, slot_b)
///
/// Because both pairs share the same explicit side `n` with different marginal
/// refs, this is a p-fusion redex at root. The two slots also share the same
/// parent context {(root_node=0, sibling=n)}, so generic twin contraction
/// detects them as twins and `merge_twin_marginal_counts` sums the counts.
/// The combined pipeline (either path) must yield ONE pair at the root with
/// the summed count accessible via the surviving slot.
#[test]
fn p_fusion_redex_closed_within_contract_all_twins_topdown() {
    let _thr = crate::tdd::types::set_marg_inline_max(0); // force slot refs; no inlining

    // Choose counts large enough that they'll never be inlined.
    const COUNT_A: u128 = 1_000_000_000_000u128;
    const COUNT_B: u128 = 2_000_000_000_000u128;
    const COUNT_SUM: u128 = COUNT_A + COUNT_B;

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // v_left: explicit internal with one node `n` (single pair child nodes).
    let n = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_left — trivial leaf-label nodes.
    levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // v_right: marginal sibling with TWO slots carrying different large counts.
    levels[v_right.idx()].make_marginal(vec![COUNT_A, COUNT_B], None);
    let slot_a = LocalNodeIdx(MargRef::slot_raw(0));
    let slot_b = LocalNodeIdx(MargRef::slot_raw(1));

    // Root: one multi-pair node with TWO pairs — both use the same explicit
    // node `n` but different marginal refs (slot_a and slot_b). This is both
    // a p-fusion redex (same-x-different-marg at root) and a twin contraction
    // redex (slot_a and slot_b have identical parent context {(root_node, n)}).
    levels[root.idx()].push_internal_node(&[
        InputPair { left: n, right: slot_a },
        InputPair { left: n, right: slot_b },
    ]);

    let output = crate::tdd::types::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs so the boundary decode is consistent.
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
    // Mark root dirty so contract_all_twins_topdown picks it up.
    tdd.dirty_contract.push(root.0);

    // Run the full pipeline — must close the redex in one call.
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // The root node must have exactly ONE pair remaining.
    let root_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(
        root_pairs.len(), 1,
        "root must have 1 pair after fusing (n, slot_a) and (n, slot_b); got {} pairs",
        root_pairs.len(),
    );

    // The surviving marg-side ref must decode to the summed count COUNT_SUM.
    let surviving_raw = root_pairs[0].right.0;
    let marg_counts = tdd.levels[v_right.idx()].marginal_counts.as_ref().unwrap();
    let fused_count = match MargRef::from_raw(surviving_raw) {
        MargRef::Slot(s) => marg_counts[s as usize],
        MargRef::Inline(v) => v as u128,
    };
    assert_eq!(
        fused_count, COUNT_SUM,
        "surviving slot must hold COUNT_A + COUNT_B = {COUNT_SUM}; got {fused_count}",
    );

    // The explicit side must still be n (index 0 after any compaction).
    let surviving_explicit = root_pairs[0].left.0;
    assert_eq!(
        surviving_explicit, n.0,
        "explicit side of the surviving pair must be n; got {surviving_explicit}",
    );
}

// ── Change-C: directed fixture — fusion redex creates twins ─────────────

/// Directed joint-fixpoint fixture (change C).
///
/// Constructs a level where closing p-fusion redexes CREATES structural twins:
/// one root node holds four pairs referencing two explicit nodes A and B.
/// A's group uses slots 0 and 1 (COUNT_A + COUNT_B = COUNT_SUM); B's group
/// uses slots 2 and 3 (COUNT_C + COUNT_D = COUNT_SUM). Before fusion, A and B
/// have DIFFERENT parent contexts, so they are NOT twins. After fusion, both
/// groups yield the SAME summed count COUNT_SUM, and C3 maps them to the same
/// surviving slot — giving A and B identical contexts. Since A and B also have
/// identical child pairs, the joint fixpoint loop must detect and merge them.
/// The merge is a content-equal dup-redirect: the root keeps BOTH pairs
/// remapped onto the survivor and p-fusion folds them, leaving one root node
/// with one pair whose count is 2·COUNT_SUM — the denoted value is
/// MC(A)·(c_A+c_B) + MC(B)·(c_C+c_D) = MC(A)·2·COUNT_SUM, so multiplicity is
/// SUMMED into the count; set-dedup to COUNT_SUM would halve the model count.
///
/// Fixture (balanced(4)):
///   v_right = marginal; four slots: COUNT_A, COUNT_B, COUNT_C, COUNT_D
///             (all distinct; COUNT_A+COUNT_B = COUNT_C+COUNT_D = COUNT_SUM)
///   v_left  = explicit internal; two nodes A and B with identical child pairs
///   root    = ONE multi-pair node:
///               (A, slot_0), (A, slot_1), (B, slot_2), (B, slot_3)
///
/// Pre-fusion: A's context = {(root0,slot_0),(root0,slot_1)},
///             B's context = {(root0,slot_2),(root0,slot_3)} → DIFFERENT → not twins
/// Post-fusion: (A, slot_sum), (B, slot_sum) → A and B both context {(root0,slot_sum)}
///              → structural twins → dup-redirect merge at v_left →
///              root: {(merged, slot_sum), (merged, slot_sum)} → p-fusion →
///              root: {(merged, slot_2sum)} with 2·COUNT_SUM
#[test]
fn fusion_creates_twin_both_closed_in_one_call() {
    let _thr = crate::tdd::types::set_marg_inline_max(0); // force slot refs; no inlining

    // Four distinct counts; two pairs summing to the same total.
    const COUNT_A: u128 = 1_000_000_000_000u128;
    const COUNT_B: u128 = 2_000_000_000_000u128;
    const COUNT_C: u128 = 500_000_000_000u128;
    const COUNT_D: u128 = 2_500_000_000_000u128;
    const COUNT_SUM: u128 = COUNT_A + COUNT_B; // = COUNT_C + COUNT_D = 3e12

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // v_left: two nodes A and B with IDENTICAL child pairs (Pos, One).
    // Before fusion their contexts in root differ (different slots); after
    // fusion they share the same slot_sum context → structural twins.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_left.
    levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

    // v_right: marginal sibling with FOUR distinct slots (all different counts).
    levels[v_right.idx()].make_marginal(vec![COUNT_A, COUNT_B, COUNT_C, COUNT_D], None);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0)); // COUNT_A  }
    let slot_1 = LocalNodeIdx(MargRef::slot_raw(1)); // COUNT_B  } sum = COUNT_SUM
    let slot_2 = LocalNodeIdx(MargRef::slot_raw(2)); // COUNT_C  }
    let slot_3 = LocalNodeIdx(MargRef::slot_raw(3)); // COUNT_D  } sum = COUNT_SUM

    // root: ONE node with four pairs.
    //   A's group: (A, slot_0), (A, slot_1) → p-fusion redex → fuses to (A, slot_sum)
    //   B's group: (B, slot_2), (B, slot_3) → p-fusion redex → fuses to (B, slot_sum)
    //                                         (same count COUNT_SUM → same slot by C3)
    // A's pre-fusion context  = {(root0, slot_0), (root0, slot_1)} ← different from B's
    // B's pre-fusion context  = {(root0, slot_2), (root0, slot_3)} → NOT twins yet
    // Post-fusion both become = {(root0, slot_sum)}                 → NOW twins
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: slot_0 },
        InputPair { left: a, right: slot_1 },
        InputPair { left: b, right: slot_2 },
        InputPair { left: b, right: slot_3 },
    ]);

    let output = crate::tdd::types::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs for consistent boundary decode.
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);

    // Precondition A: fusion redexes are present.
    assert!(
        crate::tdd::validate::marg::check_no_fusion_redexes(&tdd).is_err(),
        "fixture must start WITH p-fusion redexes"
    );

    // Mark root dirty and run the joint pipeline (change B joint fixpoint).
    tdd.dirty_contract.push(root.0);
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // Postcondition A: no fusion redexes remain.
    crate::tdd::validate::marg::check_no_fusion_redexes(&tdd)
        .unwrap_or_else(|e| panic!("fusion redex survived after fixpoint: {e}"));

    // Postcondition B: no unmerged twins at the parent-of-marginal level (root).
    crate::tdd::validate::marg::check_no_twins(&tdd)
        .unwrap_or_else(|e| panic!("twin pair survived after fixpoint: {e}"));

    // Postcondition C: v_left contracted from 2 nodes (A, B) to 1 (merged twin).
    let vl_width = tdd.levels[v_left.idx()].width();
    assert_eq!(
        vl_width, 1,
        "v_left must have 1 node after twin-merge of A and B; got {vl_width}",
    );

    // Postcondition D: root has one pair with count = 2·COUNT_SUM. The fixture
    // denotes MC(A)·(c_A+c_B) + MC(B)·(c_C+c_D) = MC(A)·2·COUNT_SUM, so the
    // twin merge must SUM the multiplicity into the count (dup-redirect +
    // p-fusion fold) — a set-dedup ending at COUNT_SUM would halve the count.
    let root_pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(root_pairs.len(), 1, "surviving root node must have 1 pair; got {}", root_pairs.len());
    let marg_raw = root_pairs[0].right.0;
    let marg_counts = tdd.levels[v_right.idx()].marginal_counts.as_ref().unwrap();
    let count = match MargRef::from_raw(marg_raw) {
        MargRef::Slot(s) => marg_counts[s as usize],
        MargRef::Inline(c) => c as u128,
    };
    assert_eq!(
        count,
        2 * COUNT_SUM,
        "surviving pair must decode to 2*COUNT_SUM={}; got {count}",
        2 * COUNT_SUM,
    );
}

// ── Fork-down: content twins ABOVE the boundary parent (plain level) ────

/// Vtree shape shared by the fork-down fixtures below. Custom (not `balanced`)
/// so the marg-carrying level `m` sits at an INTERNAL vtree node: an integer
/// marginal LEAF keeps an EMPTY store (bare refs are leaf-LABELS, decoded by
/// `read_marginal_count`), so it is not a legal fork-down scale target and the
/// mint that these tests exercise would be unsound there (see B4 /
/// `dup_resolve.rs` `scale_leaf_marg_label`). An internal marg level exercises
/// the multiplicity-fork-down mechanics identically, with a real store to mint
/// into. Shape (left spine root → gp → bp; each 2-leaf subtree on the right):
///   root → (gp, σ);  gp → (bp, s);  bp → (x [leaf], m [INTERNAL]);
///   m → (m_l, m_r);  s → (s_l, s_r);  σ → (sig_l, sig_r).
fn boundary_internal_marg_vtree() -> Vtree {
    // 7 vars; node ids reindexed bottom-up by `from_vtree_text` (root last),
    // so callers navigate via `children()` exactly as with `balanced`.
    //   x=0(leaf)  m=(1,2)  s=(3,4)  σ=(5,6);  bp=(x,m) gp=(bp,s) root=(gp,σ)
    Vtree::from_vtree_text(
        "vtree 13\n\
         L 0 1\nL 1 2\nL 2 3\nL 3 4\nL 4 5\nL 5 6\nL 6 7\n\
         I 7 1 2\nI 8 0 7\nI 9 3 4\nI 10 8 9\nI 11 5 6\nI 12 10 11\n",
    )
    .expect("boundary_internal_marg_vtree parse")
}

/// Directed fixture for duplicate-pair resolution by fork-down scaling
/// (dup_resolve): content-equal context-twins at a PLAIN level whose merge
/// mints a duplicate pair, resolved by scaling the marg-carrying child.
///
/// Fixture (`boundary_internal_marg_vtree`), left spine root → gp → bp:
///   m       = bp's right child (INTERNAL level) made MARGINAL; one slot, count 5
///   bp      = boundary parent; one node P = {(Pos, slot_0)}
///   gp      = PLAIN level; two nodes A = B = {(P, s)} — content-equal
///   s       = one node {(Pos, One)} at gp's right child (plain sibling)
///   root    = one node {(A, σ), (B, σ)} — A and B share context {(root0, σ)}
///   σ       = one node {(Pos, One)} at root's right child
///
/// Denoted count through root: MC(A)·MC(σ) + MC(B)·MC(σ) = 2·5·(…) — the twin
/// merge must preserve the factor 2. Expected: A, B merge; the survivor's
/// concat {(P,s), (P,s)} is KEPT as two multiset terms summing to 2·5·MC(s) —
/// post-bd433a75d: no O(1) absorber in this fixture (gp's own children `bp` and
/// `s` are both plain; the marginal level `m` sits a level lower, under `bp`),
/// so the duplicates legally remain uncollapsed. What still must NOT happen is
/// set-dedup, which would drop a term and halve the total to 5.
#[test]
fn plain_level_content_twins_fork_multiplicity_down() {
    let _thr = crate::tdd::types::set_marg_inline_max(0); // force slot refs

    const COUNT: u128 = 5;

    let vtree = Arc::new(boundary_internal_marg_vtree());
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, sigma_v) = vtree.children(root);
    assert!(matches!(*vtree.node(gp), crate::vtree::VtreeNode::Internal { .. }));
    let (bp, s_v) = vtree.children(gp);
    assert!(matches!(*vtree.node(bp), crate::vtree::VtreeNode::Internal { .. }));
    let (x_v, m_v) = vtree.children(bp);
    // m must be INTERNAL (the B4 invariant): a leaf marg store cannot hold a slot.
    assert!(matches!(*vtree.node(m_v), crate::vtree::VtreeNode::Internal { .. }));
    let (s_l, s_r) = vtree.children(s_v);
    let (sig_l, sig_r) = vtree.children(sigma_v);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // m: marginal leaf-side level with one slot of count 5.
    levels[m_v.idx()].make_marginal(vec![COUNT], None);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0));

    // bp: one node P = {(Pos, slot_0)}.
    levels[x_v.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_0 }]);

    // s: one plain node {(Pos, One)} at gp's right child.
    levels[s_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // gp: A and B, identical pair lists {(P, s)} — content-equal twins.
    let a = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);
    let b = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);

    // σ: one plain node at root's right child.
    levels[sig_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // root: {(A, σ), (B, σ)} — gives A and B the same context.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::tdd::types::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);

    tdd.dirty_contract.push(root.0);
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // Root: one pair (survivor, σ).
    assert_eq!(tdd.levels[root.idx()].pair_count_at(0), 1, "root must end with 1 pair");
    let surv = tdd.levels[root.idx()].pairs_of_idx(0)[0].left.0 as usize;

    // Survivor at gp: BOTH duplicate terms remain — the multiplicity is carried
    // by the pair list itself, not set-dedup'd away.
    let surv_pairs: Vec<_> = tdd.levels[gp.idx()].pairs_of_idx(surv).to_vec();
    assert_eq!(surv_pairs.len(), 2, "gp survivor must keep both duplicate terms");
    assert_eq!(
        surv_pairs[0].left.0, surv_pairs[1].left.0,
        "the kept run is two copies of ONE pair — same left child"
    );
    for pr in &surv_pairs {
        assert_eq!(pr.right.0, s.0, "plain sibling side must be untouched");
    }

    // Count soundness, unchanged in strength: the terms SUM to 2·COUNT = 10 —
    // the exact total the collapsed single pair P₂ = {(Pos, count 10)} used to
    // carry. Set-dedup would leave one term and halve it to 5.
    let marg_counts = tdd.levels[m_v.idx()].marginal_counts.as_ref().unwrap();
    let total: u128 = surv_pairs
        .iter()
        .map(|pr| {
            let p_pair = tdd.levels[bp.idx()].pairs_of_idx(pr.left.0 as usize)[0];
            match MargRef::from_raw(p_pair.right.0) {
                MargRef::Slot(sl) => marg_counts[sl as usize],
                MargRef::Inline(c) => c as u128,
            }
        })
        .sum();
    assert_eq!(total, 2 * COUNT, "the kept run must still total 2*COUNT, got {total}");

    // No twins left anywhere.
    crate::tdd::validate::marg::check_no_twins(&tdd)
        .unwrap_or_else(|e| panic!("twin survived fork-down: {e}"));
}

/// WEIGHTED analogue of `plain_level_content_twins_fork_multiplicity_down`.
///
/// Same fixture, but the marg child `m` is a WEIGHT-marginal level: its per-slot
/// value lives in the external `WeightStore` (a `BigRational`), and
/// `marginal_counts` is `None`. With a weight context installed
/// (a weight store attached) the contraction concat-merges the two
/// content-equal twins A,B, leaving the survivor with the duplicate pair
/// `(P, s),(P, s)`.
///
/// post-bd433a75d: no O(1) absorber in this fixture — gp's own children `bp` and
/// `s` are both plain (the weight-marginal `m` sits one level lower, under
/// `bp`), so `resolve_duplicate_pairs_in_node` early-outs and the duplicates
/// legally remain uncollapsed. What this pins is that the weighted twin-fold
/// completes without erroring or touching the `WeightStore`, and that the
/// multiplicity survives as two multiset terms: their values SUM to
/// 2·(3/7) = 6/7, exactly the total the scaled single pair used to carry, where
/// a set-dedup would leave 3/7.
///
/// NOTE: the weighted scale dispatch itself (`scale_marg_ref`'s
/// `is_weight_marginal()` branch → `scale_weight_ref`, added in cd23bda4d) is no
/// longer reached from this geometry — under the cost policy it needs a
/// duplicate run at a plain level whose OWN child is the weight-marginal one.
#[test]
fn weighted_plain_level_content_twins_fork_multiplicity_down() {
    use crate::tdd::query::semiring::RationalSemiring;
    use crate::tdd::weight_store::Precision;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    let _thr = crate::tdd::types::set_marg_inline_max(0); // force slot refs

    // The slot value to be scaled. A non-trivial rational so a missing ×2 (or a
    // set-dedup that drops multiplicity) is unmistakable.
    let v = BigRational::new(BigInt::from(3), BigInt::from(7)); // 3/7
    let two = BigRational::from_integer(BigInt::from(2));

    let vtree = Arc::new(Vtree::balanced(8));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, sigma_v) = vtree.children(root);
    assert!(matches!(*vtree.node(gp), crate::vtree::VtreeNode::Internal { .. }));
    let (bp, s_v) = vtree.children(gp);
    assert!(matches!(*vtree.node(bp), crate::vtree::VtreeNode::Internal { .. }));
    let (x_v, m_v) = vtree.children(bp);
    let (s_l, s_r) = vtree.children(s_v);
    let (sig_l, sig_r) = vtree.children(sigma_v);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    // m: WEIGHT-marginal leaf-side level with one slot holding value 3/7.
    levels[m_v.idx()].make_marginal_weighted_with_slots(1);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0));

    // bp: one node P = {(Pos, slot_0)}.
    levels[x_v.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_0 }]);

    // s: one plain node {(Pos, One)} at gp's right child.
    levels[s_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // gp: A and B, identical pair lists {(P, s)} — content-equal twins.
    let a = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);
    let b = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);

    // σ: one plain node at root's right child.
    levels[sig_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // root: {(A, σ), (B, σ)} — gives A and B the same context.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::tdd::types::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);

    // Attach the store AFTER building the diagram (mirrors toy_weighted's
    // contract) and write the slot's value into it, so the C2 twin-fold takes
    // the weighted scaling path.
    let mut ws = crate::tdd::weight_store::WeightStore::new(
        RationalSemiring::from_weights(&[(v.clone(), v.clone())]),
        Precision::Exact,
    );
    ws.set_level(m_v.idx(), vec![crate::tdd::query::semiring::WeightVal::exact(v.clone())]);
    tdd.attach_weights(ws);

    tdd.dirty_contract.push(root.0);

    // Run the contraction (this is the call that would PANIC on unfixed code).
    let result = contract_all_twins_topdown(&mut tdd, None);

    let captured: Option<(usize, BigRational, bool, usize, BigRational)> =
        result.as_ref().ok().map(|_| {
            // Root: one pair (survivor, σ).
            let surv = tdd.levels[root.idx()].pairs_of_idx(0)[0].left.0 as usize;
            // Survivor at gp: BOTH duplicate terms (P, s) remain.
            let surv_pairs: Vec<(u32, u32)> = tdd.levels[gp.idx()]
                .pairs_of_idx(surv)
                .iter()
                .map(|pr| (pr.left.0, pr.right.0))
                .collect();
            let sibling_ok = surv_pairs.iter().all(|&(_, r)| r == s.0);
            // Sum the weighted values the terms carry, and record the store's
            // slot count — nothing may have been minted into it.
            let (total, n_slots) = {
                let level =
                    tdd.weights().unwrap().level(m_v.idx()).expect("weight store level");
                let mut acc = BigRational::from_integer(BigInt::from(0));
                for &(l, _) in &surv_pairs {
                    let p_pair = tdd.levels[bp.idx()].pairs_of_idx(l as usize)[0];
                    let slot = match MargRef::from_raw(p_pair.right.0) {
                        MargRef::Slot(sl) => sl as usize,
                        MargRef::Inline(_) => unreachable!("weighted marg ref is never inline"),
                    };
                    // Non-exhaustive on purpose: the Exact domain has two
                    // representations (`Exact`/`ExactSmall`), and
                    // `as_rational` is the one canonical read of either.
                    acc = acc
                        + match &level[slot] {
                            crate::tdd::query::semiring::WeightVal::Log(_) => {
                                panic!("test expects exact mode")
                            }
                            v => v.as_rational().into_owned(),
                        };
                }
                (acc, level.len())
            };
            (surv_pairs.len(), total, sibling_ok, n_slots, BigRational::clone(&v))
        });

    let result = result.expect("contract_all_twins_topdown (weighted twin-fold)");
    let _ = result;
    let (surv_pairs, total, sibling_ok, n_slots, orig_v) =
        captured.expect("captured assertion inputs");

    assert_eq!(surv_pairs, 2, "gp survivor must keep both duplicate terms");
    assert!(sibling_ok, "plain sibling side must be untouched");
    assert_eq!(n_slots, 1, "nothing absorbed the factor — no fresh WeightStore slot");
    // The kept run's values must SUM to 2·(3/7) = 6/7 — multiplicity carried by
    // the pair list, NOT set-dedup'd (which would leave the total at 3/7).
    assert_eq!(
        total,
        &orig_v * &two,
        "the kept run must still total 2·(3/7) = 6/7, got {total}",
    );
}

/// Partial-overlap variant: context twins sharing ONE pair (not all). The
/// shared pair keeps its multiplicity; the disjoint remainder concats.
///   A = {(P, s), (Q, t)},  B = {(P, s), (R, u)}  →
///   survivor = {(P, s), (P, s), (Q, t), (R, u)}.
/// post-bd433a75d: no O(1) absorber in this fixture (gp's own children `bp` and
/// `s` are both plain; the marginal `m` sits under `bp`), so the shared pair's
/// duplicates legally remain uncollapsed instead of folding into P₂ = count 10.
/// The multiset total is what count soundness rests on, and it is unchanged.
/// Uses `boundary_internal_marg_vtree` so `m` is an INTERNAL marg level (B4).
#[test]
fn plain_level_partial_overlap_twins_fork_shared_pair_down() {
    let _thr = crate::tdd::types::set_marg_inline_max(0);

    const COUNT_P: u128 = 5;
    const COUNT_Q: u128 = 7;
    const COUNT_R: u128 = 11;

    let vtree = Arc::new(boundary_internal_marg_vtree());
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, sigma_v) = vtree.children(root);
    let (bp, s_v) = vtree.children(gp);
    let (x_v, m_v) = vtree.children(bp);
    assert!(matches!(*vtree.node(m_v), crate::vtree::VtreeNode::Internal { .. }));
    let (s_l, s_r) = vtree.children(s_v);
    let (sig_l, sig_r) = vtree.children(sigma_v);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::tdd::types::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

    levels[m_v.idx()].make_marginal(vec![COUNT_P, COUNT_Q, COUNT_R], None);
    let slot_p = LocalNodeIdx(MargRef::slot_raw(0));
    let slot_q = LocalNodeIdx(MargRef::slot_raw(1));
    let slot_r = LocalNodeIdx(MargRef::slot_raw(2));

    levels[x_v.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    // bp: P, Q, R — distinct structural lefts so gp pairs stay distinct.
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_p }]);
    let q = levels[bp.idx()].push_internal_node(&[InputPair { left: neg, right: slot_q }]);
    let r = levels[bp.idx()].push_internal_node(&[InputPair { left: one, right: slot_r }]);

    levels[s_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let t = levels[s_v.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let u = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: neg }]);

    // gp: A = {(P,s),(Q,t)}, B = {(P,s),(R,u)} — shared pair (P,s).
    let a = levels[gp.idx()].push_internal_node(&[
        InputPair { left: p, right: s },
        InputPair { left: q, right: t },
    ]);
    let b = levels[gp.idx()].push_internal_node(&[
        InputPair { left: p, right: s },
        InputPair { left: r, right: u },
    ]);

    levels[sig_l.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::tdd::types::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::tdd::types::Tdd::with_levels(vtree, levels, output);
    crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);

    tdd.dirty_contract.push(root.0);
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    assert_eq!(tdd.levels[root.idx()].pair_count_at(0), 1, "root must end with 1 pair");
    let surv = tdd.levels[root.idx()].pairs_of_idx(0)[0].left.0 as usize;

    // Survivor: 4 pairs — the shared (P,s) kept TWICE plus the disjoint
    // (Q,t), (R,u).
    let surv_pairs: Vec<_> = tdd.levels[gp.idx()].pairs_of_idx(surv).to_vec();
    assert_eq!(surv_pairs.len(), 4, "survivor must hold 4 pairs, got {}", surv_pairs.len());

    let marg_counts = tdd.levels[m_v.idx()].marginal_counts.as_ref().unwrap();
    let decode = |raw: u32| -> u128 {
        match MargRef::from_raw(raw) {
            MargRef::Slot(sl) => marg_counts[sl as usize],
            MargRef::Inline(c) => c as u128,
        }
    };
    // Collect the multiset of decoded counts of the survivor's left children.
    let mut counts: Vec<u128> = surv_pairs
        .iter()
        .map(|pr| decode(tdd.levels[bp.idx()].pairs_of_idx(pr.left.0 as usize)[0].right.0))
        .collect();
    counts.sort_unstable();
    assert_eq!(
        counts,
        vec![COUNT_P, COUNT_P, COUNT_Q, COUNT_R],
        "shared pair must keep multiplicity 2; disjoint pairs unchanged"
    );
    // Count soundness, unchanged in strength: the multiset still totals
    // 2·COUNT_P + COUNT_Q + COUNT_R — exactly what the collapsed form
    // {(P₂ = 2·COUNT_P, s), (Q,t), (R,u)} carried.
    assert_eq!(
        counts.iter().sum::<u128>(),
        2 * COUNT_P + COUNT_Q + COUNT_R,
        "the kept run must preserve the survivor's total"
    );
}

// ── B4 regression: fork-down scaling must be leaf-aware ────────────────────
//
// A marginalized LEAF keeps an EMPTY integer store: the production decoder
// (`read_marginal_count`, compile_marginalize.rs ~1441) reads a bare marg-side
// ref at a leaf as a leaf-LABEL (fixed count), never indexing the store. So a
// leaf store is NOT a legal fork-down mint target: indexing it panics (hazard
// b), and minting a slot into it produces a ref that is silently re-decoded as
// a label — a wrong count (hazard a). `try_scale_child` is leaf-aware
// (`scale_leaf_marg_label`); these two tests pin both hazards. Both drive
// `resolve_duplicate_pairs_in_node` directly (isolating the scale from twin
// detection / slot tagging) on `balanced(8)`, at the PLAIN level `bp` whose
// right child `m_v` IS an integer-marginal leaf — the O(1)-absorber geometry
// the cost policy admits, and the only one that still reaches the leaf scale.

/// Build the shared hazard fixture: a PLAIN node at boundary parent `bp` holding
/// the DUPLICATE pair `(Pos, marg_ref)` twice, where `bp`'s right child `m_v` is
/// an integer-marginal LEAF with an EMPTY store. Returns `(tdd, gp, bp, m_v)`.
/// `marg_ref` is caller-chosen to select the hazard: a bare leaf-label (hazard
/// b) or an inline count (hazard a).
fn b4_leaf_hazard_fixture(marg_ref: u32) -> (Tdd, VtreeIdx, VtreeIdx, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(8));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, _sigma_v) = vtree.children(root);
    let (bp, s_v) = vtree.children(gp);
    let (x_v, m_v) = vtree.children(bp);
    // The hazard geometry: m_v is a LEAF (balanced(8) bottoms out here), and it
    // is `bp`'s own child — so `bp` HAS an O(1) absorber and fork-down runs.
    assert!(matches!(*vtree.node(m_v), crate::vtree::VtreeNode::Leaf { .. }));
    let (s_l, s_r) = vtree.children(s_v);

    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<TddLevel> =
        (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();

    // m_v: integer-marginal LEAF with an EMPTY store — the projection case in
    // which bare refs are leaf-LABELS, not store slots.
    levels[m_v.idx()].make_marginal(vec![], None);

    levels[x_v.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::Pos)];
    // bp: one PLAIN node holding the duplicate pair (Pos, marg_ref) twice. The
    // marg side is the leaf `m_v`, so this is exactly the run fork-down folds.
    let p = levels[bp.idx()].push_internal_node(&[
        InputPair { left: pos, right: LocalNodeIdx(marg_ref) },
        InputPair { left: pos, right: LocalNodeIdx(marg_ref) },
    ]);

    levels[s_l.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // gp: a plain node over (P, s), so the diagram is well-formed above `bp`.
    levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);

    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    (Tdd::with_levels(vtree, levels, output), gp, bp, m_v)
}

/// Hazard (b): a bare leaf-LABEL ref on the duplicated pair's marg side. Without
/// the leaf branch, fork-down routes it into `scale_marg_ref`'s `Slot` arm,
/// which indexes `counts[label_idx]` on the EMPTY leaf store — index-OOB PANIC.
/// With it, the label decodes (Pos → 1), scales by k=2 → 2, and inlines (no
/// store touched).
#[test]
fn b4_fork_down_leaf_label_ref_no_oob() {
    // No inline-max override: the doubled label count (2) must fit inline.

    // A bare "Pos" leaf-label ref (raw = label index), NOT a store slot.
    let pos_label_ref = MargRef::slot_raw(LeafLabel::Pos as u32);
    let (mut tdd, _gp, bp, m_v) = b4_leaf_hazard_fixture(pos_label_ref);

    // The fork-down scratch is caller-owned and REUSED across the survivors of
    // one pass, so it arrives holding the previous node's buffers. Hand it over
    // dirty: the resolver must clear all three at entry, or it would resolve
    // this node against another node's pair list / counts.
    let mut scratch = super::scratch::DupScratch::default();
    scratch.pairs.push((7, 7));
    scratch.counts.insert((7, 7), 5);
    scratch.out.push(InputPair { left: LocalNodeIdx(7), right: LocalNodeIdx(7) });

    // PANICS without the leaf branch (counts[label] on empty leaf store).
    let changed =
        super::dup_resolve::resolve_duplicate_pairs_in_node(&mut tdd, bp, 0, &mut scratch)
            .expect("resolve must not error");
    assert!(changed, "duplicate pair must be resolved");

    // Survivor: one pair whose marg ref decodes to Pos(1)·2 = 2.
    assert_eq!(tdd.levels[bp.idx()].pair_count_at(0), 1, "duplicate must collapse to 1 pair");
    let scaled_ref = tdd.levels[bp.idx()].pairs_of_idx(0)[0].right.0;
    let count = match MargRef::from_raw(scaled_ref) {
        MargRef::Inline(c) => c as u128,
        MargRef::Slot(_) => panic!("leaf scale must inline, never mint a leaf slot"),
    };
    assert_eq!(count, 2, "Pos leaf label (count 1) must double to 2");
    // The leaf store must remain EMPTY — nothing was minted into it.
    assert!(
        tdd.levels[m_v.idx()].marginal_counts.as_ref().is_none_or(|c| c.is_empty()),
        "leaf marg store must stay empty (no slot minted)"
    );
}

/// Hazard (a): an INLINE count on the marg side whose ×k product overflows the
/// inline cap. Without the leaf branch, `scale_marg_ref` mints a fresh slot into
/// the EMPTY leaf store and returns a bare slot ref — which the decoder re-reads
/// as a leaf LABEL (slot 0 → label One), silently miscounting. With it the leaf
/// side refuses (`None`); the other side is structural, which the O(1)-absorber
/// cost policy never descends into, so nothing absorbs and the run is KEPT as
/// two legal multiset terms — same count, no mint.
#[test]
fn b4_fork_down_leaf_inline_overflow_keeps_run() {

    // An inline count at the cap; ×2 overflows the inline range → cannot re-inline.
    let big_inline = MargRef::inline_raw(crate::tdd::types::MARG_INLINE_MAX as u128)
        .expect("cap value inlines");
    let (mut tdd, _gp, bp, m_v) = b4_leaf_hazard_fixture(big_inline);

    let mut scratch = super::scratch::DupScratch::default();
    let changed =
        super::dup_resolve::resolve_duplicate_pairs_in_node(&mut tdd, bp, 0, &mut scratch)
            .expect("keeping the run is not an error");
    assert!(!changed, "nothing can absorb the factor — the run must be kept as-is");

    // Both terms survive, unscaled: the multiset still sums to the same count.
    assert_eq!(tdd.levels[bp.idx()].pair_count_at(0), 2, "the duplicate run must be kept");
    for pair in tdd.levels[bp.idx()].pairs_of_idx(0) {
        assert_eq!(pair.right.0, big_inline, "kept terms must be the ORIGINAL ref");
    }
    // The regression this pins: no slot was minted into the leaf store, which
    // the decoder would have re-read as a leaf LABEL (slot 0 → One = 2).
    assert!(
        tdd.levels[m_v.idx()].marginal_counts.as_ref().is_none_or(|c| c.is_empty()),
        "leaf marg store must stay empty (no slot minted)"
    );
}


// ── Mixed-group dup-first round (Change 2) ─────────────────────────────────

/// Tests the gated mixed-group dup-first behavior in `contract_twins`.
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
/// # Gate OFF behavior (existing)
///
/// `filtered = [A, C]` (disjoint pair sets), `dup_members = [B]`.
/// The `filtered.len() >= 2` arm fires WITHOUT the dup-first guard: concat A+C,
/// B's `merge_target` stays B (identity, never updated) → B is canonical after
/// compaction → B remains at t1. After concat A has `{(pos,one),(one,pos)}`, B
/// has `{(pos,one)}` — they're no longer content-equal → B stays unmerged.
///
/// Final gate OFF: t1.width = 2 (A_merged and B).
///         parent has 2 pairs (A_merged, slot0) and (B, slot0).
///
/// # Gate ON behavior (new dup-first)
///
/// Round 1: mixed guard fires → process ONLY dup_members=[B].
///   `merge_target[B] = A`, `dup_redirect[B] = true`.
///   concat is deferred (filtered=[A,C] untouched).
///   In parent rewrite: (A,slot0) kept, (B,slot0) KEPT via dup_redirect and
///   remapped to A — creating a second (A,slot0) pair; (C,slot0) kept unchanged.
///   After compaction: t1 = {A, C} (B removed, merge_target[B]=A).
///   Parent now has 3 pairs: (A,slot0), (A,slot0), (C,slot0).
///
/// Round 2: A's context = {(0,slot0), (0,slot0)} (appears twice at parent_node_0);
///   C's context = {(0,slot0)} (appears once). Different lengths → NOT twins.
///   No further merge fires.
///
/// Final gate ON: t1.width = 2 (A and C unchanged), parent has 3 pairs.
///
/// The key difference: gate OFF silently drops B from the group (its contribution
/// is lost — B stays as a structurally separate node, and the B pair survives in
/// the parent, but B's function is never joined with A). Gate ON preserves B's
/// contribution via the dup_redirect: B's parent pair is redirected to A as a
/// duplicate (A,slot0) entry, whose multiplicity is load-bearing (p-fusion sums
/// it). B is soundly absorbed into A; C remains a separate twin.
///
/// Discriminant: gate ON → parent has 3 pairs (the extra (A,slot0) from B's
///   dup_redirect); gate OFF → parent has 2 pairs (A_merged and B).
///   Also: gate ON → t1 contains original-A and C (neither grown);
///         gate OFF → t1 contains grown A+C and original B.
#[test]
fn mixed_group_dup_first_gate_on_keeps_b_contribution_gate_off_drops_b() {
    // Force all marg refs onto slots (no inlining) so sib_slot refs stay as
    // bare slot indices — the scenario the dup_members detection depends on.
    let _thr = crate::tdd::types::set_marg_inline_max(0);

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
        let mut levels: Vec<crate::tdd::types::TddLevel> =
            (0..vtree.num_nodes()).map(|_| crate::tdd::types::TddLevel::new()).collect();

        // t1 (v_left): 3 internal nodes A(0), B(1), C(2).
        // A and B are content-equal (same pair), C is disjoint.
        let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
        let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
        let c = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
        assert_eq!(a.0, 0); assert_eq!(b.0, 1); assert_eq!(c.0, 2);

        // Leaf children of v_left — trivial leaf-label nodes.
        levels[vl_left.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::Pos)];
        levels[vl_right.idx()].nodes = vec![crate::tdd::types::TddNodeData::leaf(LeafLabel::One)];

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

        let output = crate::tdd::types::TddNodeId {
            vtree: root,
            local: LocalNodeIdx(0),
        };
        let mut tdd = crate::tdd::types::Tdd::with_levels(vtree.clone(), levels, output);
        // Tag marg-side refs for the boundary decode.
        crate::tdd::types::tag_all_marg_side_slots(&mut tdd, None);
        tdd.dirty_contract.push(root.0);
        tdd
    };

    // ── Gate OFF: concat fires first, B silently stays separate ──────────────
    {
        let _no_fold = super::super::set_c2_fold_allow(false);
        let mut tdd = build_fixture();
        contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown (gate off)");

        // Gate OFF: filtered=[A,C] → concat; B's merge_target stays B (canonical).
        // A and B are still twins after round 1 (both context={parent_node_0, slot0}).
        // BUT B overlaps A_merged (B's pair is a subset of A_merged's pairs) and B is
        // NOT content-equal to A_merged → B ends up in neither filtered nor dup_members
        // → B never merges. Final: t1 = {A_merged, B} → width 2.
        let t1_width = tdd.levels[v_left.idx()].width();
        assert_eq!(
            t1_width, 2,
            "gate OFF: B must remain as a separate node (width=2); got {t1_width}"
        );

        // Parent has exactly 2 pairs: the C-referencing pair was dropped (C mapped
        // to A); the B-referencing pair was kept (B canonical).
        let parent_pairs = tdd.levels[root.idx()].pair_count_at(0);
        assert_eq!(
            parent_pairs, 2,
            "gate OFF: parent must have 2 pairs (A_merged and B); got {parent_pairs}"
        );
    }

    // ── Gate ON: dup fires first, B's contribution preserved via p-fusion ────────
    {
        let _fold = super::super::set_c2_fold_allow(true);
        let mut tdd = build_fixture();
        contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown (gate on)");

        // Gate ON round 1: dup_members=[B] → B dup_redirect to A; A and C untouched.
        //   Parent (intermediate): (B,slot0) remapped to (A,slot0) via dup_redirect →
        //   parent temporarily holds (A,slot0),(A,slot0),(C,slot0) = 3 pairs.
        //   t1 after compaction = {A, C} (B removed).
        // p-fusion (same fixpoint iteration): the two (A,slot0) pairs are same-explicit
        //   (same A) same-count-ref (same slot0) → p-fusion redex → fused to (A,slot=14)
        //   where 14 = 7+7 (count doubled). Parent now has 2 pairs: (A,slot=14),(C,slot0=7).
        // Final: t1 = {A, C} → width 2; parent has 2 pairs.
        let t1_width = tdd.levels[v_left.idx()].width();
        assert_eq!(
            t1_width, 2,
            "gate ON: A and C must remain (width=2) — different contexts after B dup_redirect; got {t1_width}"
        );

        // A's pair count in t1 must be 1 (A was never concat-merged with C).
        // Gate OFF would have A_merged with 2 pairs; gate ON keeps A at 1 pair.
        let a_pair_count = tdd.levels[v_left.idx()].pair_count_at(0); // A is at index 0
        assert_eq!(
            a_pair_count, 1,
            "gate ON: A must have 1 pair (no concat with C); got {a_pair_count}"
        );

        // Parent: p-fusion fused the two (A,slot0) pairs from the dup_redirect into
        // (A, count=14). B's contribution is preserved in the doubled count.
        // Total pairs = 2: the fused (A, 14) and the untouched (C, 7).
        let parent_pairs = tdd.levels[root.idx()].pair_count_at(0);
        assert_eq!(
            parent_pairs, 2,
            "gate ON: parent must have 2 pairs after p-fusion fuses the duplicate (A,slot0); got {parent_pairs}"
        );

        // The A-referencing parent pair must carry count 14 (7+7 from B's dup_redirect).
        let sib_counts = tdd.levels[v_right.idx()].marginal_counts.as_ref().unwrap();
        let decode = |raw: u32| -> u128 {
            match MargRef::from_raw(raw) {
                MargRef::Slot(s) => sib_counts[s as usize],
                MargRef::Inline(c) => c as u128,
            }
        };
        let pairs = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        // Find the pair that references A (node index 0) on the left.
        let a_pair = pairs.iter().find(|p| p.left.0 == 0)
            .expect("parent must have a pair referencing A (index 0)");
        let a_count = decode(a_pair.right.0);
        assert_eq!(
            a_count, 14u128,
            "gate ON: A's paired count must be 14 (7+7 from B's dup_redirect via p-fusion); got {a_count}"
        );
    }
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
    use super::content_twin::{return_scratch, take_scratch, C2Scratch};

    let mut fp_counts: rustc_hash::FxHashMap<u64, u32> = Default::default();
    fp_counts.insert(11, 2);
    let mut key_to_canonical: rustc_hash::FxHashMap<Vec<(u32, u32)>, u32> = Default::default();
    key_to_canonical.insert(vec![(1, 2)], 3);
    return_scratch(C2Scratch {
        node_fp: vec![11, 11],
        fp_counts,
        key_to_canonical,
        remap: vec![0, 0],
    });

    let s = take_scratch();
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
    tdd.dirty_contract.push(root.0);
    tdd
}

#[test]
fn contract_merge_scratch_buffers_are_budget_charged() {
    use crate::tdd::transform::pairwise::conjoin::{
        apply_limits, reset_apply_in_flight, ApplyError,
    };
    let _thr = crate::tdd::types::set_marg_inline_max(0);
    let vtree = Arc::new(Vtree::balanced(4));
    let width = 64usize;

    // Warm-up: same width, no twins. Sizes the width-keyed fingerprint scratch
    // on this thread without ever reaching `contract_twins`.
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, _) = vtree.children(root);
    let mut warm = wide_twin_fixture(&vtree, width, false);
    contract_all_twins_topdown(&mut warm, None).expect("warm-up contraction");
    assert_eq!(warm.levels[v_left.idx()].width(), width, "warm-up must not merge");
    // The group-keyed fingerprint buffers are sized by what the twin run finds,
    // which the twin-free warm-up cannot pre-size: grow them here, untracked and
    // generously, leaving the three merge buffers as the only cold scratch.
    {
        let mut s = super::scratch::take_scratch();
        let big = 64 * width;
        s.flat_groups.resize_with(big, Default::default);
        s.group_starts.resize_with(big, Default::default);
        s.entries.resize_with(big, Default::default);
        s.counts.resize_with(big, Default::default);
        s.cursors.resize_with(big, Default::default);
        s.slice_unsorted.resize_with(big, Default::default);
        assert!(s.merge_target.is_empty() && s.dup_redirect.is_empty() && s.final_remap.is_empty());
        super::scratch::return_scratch(s);
    }

    // Twin run under a budget smaller than `merge_target` alone.
    let mut tdd = wide_twin_fixture(&vtree, width, true);
    reset_apply_in_flight();
    let budget = (4 * width - 1) as u64;
    let out = {
        let _limits = apply_limits().budget(Some(budget)).apply();
        contract_all_twins_topdown(&mut tdd, None)
    };
    assert!(
        matches!(out, Err(ApplyError::OverBudget)),
        "a contraction whose scratch buffers exceed the budget must return OverBudget, got {out:?}"
    );
    // The trip happened before any mutation: the level is untouched.
    assert_eq!(tdd.levels[v_left.idx()].width(), width, "the budget trip must precede the merge");
    assert_eq!(tdd.levels[root.idx()].pairs_of_idx(0).len(), width, "parent pairs untouched");
}
