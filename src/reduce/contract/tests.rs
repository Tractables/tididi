use crate::diagram::*;
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
    let _thr = crate::diagram::marg::set_marg_inline_max(0);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    // v_left must be internal (so its TDD nodes can be explicit internal nodes).
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // Two explicit twin nodes A and B at v_left, each with one pair.
    // Their content differs but they will be twins because the parent
    // pairs them with the identical sibling ref.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    // Leaf children of v_left — give them trivial leaf-label nodes.
    levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

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

    let output = crate::diagram::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs so the boundary decode is consistent.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
    // Declare root dirty so contract_all_twins_topdown picks it up.
    tdd.scratch.dirty_contract.push(root.0);

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
/// In production, slot-count uniqueness ensures two slots with equal counts never coexist, so
/// this scenario cannot arise via the normal pipeline. For compile_marginalize-
/// path stores it is enforced at birth via `dedup_fresh_store`; for apply-emit-
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
    let _thr = crate::diagram::marg::set_marg_inline_max(0);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), crate::vtree::VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }));

    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

    // Two sibling slots with EQUAL counts (both 3) but DIFFERENT raw indices.
    levels[v_right.idx()].make_marginal(vec![3u128, 3u128], None);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    let sib_slot1 = LocalNodeIdx(MargRef::slot_raw(1));

    // Root: A paired with slot0, B paired with slot1 — different sibling raws.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot1 },
    ]);

    let output = crate::diagram::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
    tdd.scratch.dirty_contract.push(root.0);

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
    let _thr = crate::diagram::marg::set_marg_inline_max(64);

    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);

    levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

    // Two DISTINCT slots carrying EQUAL counts (5) — the configuration the
    // slot-form test proves is NOT contracted when refs stay bare slots.
    levels[v_right.idx()].make_marginal(vec![5u128, 5u128], None);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    let sib_slot1 = LocalNodeIdx(MargRef::slot_raw(1));

    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot1 },
    ]);

    let output = crate::diagram::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Tagger rewrites both small-count slot refs to Inline(5) — equal raws.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
    for p in tdd.levels[root.idx()].pairs_of_idx(0) {
        match MargRef::from_raw(p.right.0) {
            MargRef::Inline(c) => assert_eq!(c, 5, "tagger must inline count 5"),
            other => panic!("sibling ref must be inline after tagging, got {other:?}"),
        }
    }
    tdd.scratch.dirty_contract.push(root.0);

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
/// which performs BigUint promotion when the sum overflows u128. Twin
/// contraction on marginal levels does not participate; this test goes through
/// p-fusion alone.
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
    let _thr = crate::diagram::marg::set_marg_inline_max(0); // force slot refs; no inlining

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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_left: marginal level with two slots — both referenced from root.
    levels[v_left.idx()].make_marginal(vec![C0, C1], None);

    // v_right: explicit internal with one node `n` (single pair, leaf children).
    let n = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_right — trivial leaf-label nodes.
    levels[vr_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vr_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

    // root: one multi-pair node with TWO pairs — both reference the same
    // explicit sibling `n` but different marginal refs (slot0, slot1). This is
    // a p-fusion redex: same-x-different-marg-ref pairs at the same node.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: LocalNodeIdx(MargRef::slot_raw(0)), right: n },
        InputPair { left: LocalNodeIdx(MargRef::slot_raw(1)), right: n },
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs and mark root dirty; the full pipeline closes the redex.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
    tdd.scratch.dirty_contract.push(root.0);
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
/// detects them as twins and sums the counts.
/// The combined pipeline (either path) must yield ONE pair at the root with
/// the summed count accessible via the surviving slot.
#[test]
fn p_fusion_redex_closed_within_contract_all_twins_topdown() {
    let _thr = crate::diagram::marg::set_marg_inline_max(0); // force slot refs; no inlining

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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_left: explicit internal with one node `n` (single pair child nodes).
    let n = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_left — trivial leaf-label nodes.
    levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

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

    let output = crate::diagram::TddNodeId {
        vtree: root,
        local: LocalNodeIdx(0),
    };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs so the boundary decode is consistent.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);
    // Mark root dirty so contract_all_twins_topdown picks it up.
    tdd.scratch.dirty_contract.push(root.0);

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
/// groups yield the SAME summed count COUNT_SUM, and slot-count uniqueness maps them to the same
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
    let _thr = crate::diagram::marg::set_marg_inline_max(0); // force slot refs; no inlining

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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // v_left: two nodes A and B with IDENTICAL child pairs (Pos, One).
    // Before fusion their contexts in root differ (different slots); after
    // fusion they share the same slot_sum context → structural twins.
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // Leaf children of v_left.
    levels[vl_left.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];

    // v_right: marginal sibling with FOUR distinct slots (all different counts).
    levels[v_right.idx()].make_marginal(vec![COUNT_A, COUNT_B, COUNT_C, COUNT_D], None);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0)); // COUNT_A  }
    let slot_1 = LocalNodeIdx(MargRef::slot_raw(1)); // COUNT_B  } sum = COUNT_SUM
    let slot_2 = LocalNodeIdx(MargRef::slot_raw(2)); // COUNT_C  }
    let slot_3 = LocalNodeIdx(MargRef::slot_raw(3)); // COUNT_D  } sum = COUNT_SUM

    // root: ONE node with four pairs.
    //   A's group: (A, slot_0), (A, slot_1) → p-fusion redex → fuses to (A, slot_sum)
    //   B's group: (B, slot_2), (B, slot_3) → p-fusion redex → fuses to (B, slot_sum)
    //                                         (same count COUNT_SUM → same slot)
    // A's pre-fusion context  = {(root0, slot_0), (root0, slot_1)} ← different from B's
    // B's pre-fusion context  = {(root0, slot_2), (root0, slot_3)} → NOT twins yet
    // Post-fusion both become = {(root0, slot_sum)}                 → NOW twins
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: slot_0 },
        InputPair { left: a, right: slot_1 },
        InputPair { left: b, right: slot_2 },
        InputPair { left: b, right: slot_3 },
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Tag marg-side refs for consistent boundary decode.
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    // Precondition A: fusion redexes are present.
    assert!(
        crate::check::marg::check_no_fusion_redexes(&tdd).is_err(),
        "fixture must start WITH p-fusion redexes"
    );

    // Mark root dirty and run the joint pipeline (change B joint fixpoint).
    tdd.scratch.dirty_contract.push(root.0);
    contract_all_twins_topdown(&mut tdd, None).expect("contract_all_twins_topdown");

    // Postcondition A: no fusion redexes remain.
    crate::check::marg::check_no_fusion_redexes(&tdd)
        .unwrap_or_else(|e| panic!("fusion redex survived after fixpoint: {e}"));

    // Postcondition B: no unmerged twins at the parent-of-marginal level (root).
    crate::check::marg::check_no_twins(&tdd)
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

