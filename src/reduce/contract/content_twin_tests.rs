//! Content-twin forking at plain levels.
//!
//! Sibling of `tests.rs`.

use crate::engine::Limits;
use crate::diagram::*;
use crate::vtree::Vtree;
use std::sync::Arc;
use crate::vtree::VtreeIdx;
use super::strategies::contract_all_twins_topdown;

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
    let lim = Limits::new();
    let _thr = crate::diagram::marg::set_marg_inline_max(0); // force slot refs

    const COUNT: u128 = 5;

    let vtree = Arc::new(crate::test_helpers::boundary_internal_marg_vtree());
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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // m: marginal leaf-side level with one slot of count 5.
    levels[m_v.idx()].make_marginal(vec![COUNT], None);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0));

    // bp: one node P = {(Pos, slot_0)}.
    levels[x_v.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_0 }]);

    // s: one plain node {(Pos, One)} at gp's right child.
    levels[s_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // gp: A and B, identical pair lists {(P, s)} — content-equal twins.
    let a = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);
    let b = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);

    // σ: one plain node at root's right child.
    levels[sig_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // root: {(A, σ), (B, σ)} — gives A and B the same context.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    tdd.scratch.dirty_contract.push(root.0);
    contract_all_twins_topdown(&lim, &mut tdd, None).expect("contract_all_twins_topdown");

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
    crate::check::marg::check_no_twins(&tdd)
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
    let lim = Limits::new();
    use crate::query::RationalWeights;
    use crate::weight_store::Precision;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    let _thr = crate::diagram::marg::set_marg_inline_max(0); // force slot refs

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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    // m: WEIGHT-marginal leaf-side level with one slot holding value 3/7.
    levels[m_v.idx()].make_marginal_weighted_with_slots(1);
    let slot_0 = LocalNodeIdx(MargRef::slot_raw(0));

    // bp: one node P = {(Pos, slot_0)}.
    levels[x_v.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_0 }]);

    // s: one plain node {(Pos, One)} at gp's right child.
    levels[s_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
    let s = levels[s_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // gp: A and B, identical pair lists {(P, s)} — content-equal twins.
    let a = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);
    let b = levels[gp.idx()].push_internal_node(&[InputPair { left: p, right: s }]);

    // σ: one plain node at root's right child.
    levels[sig_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    // root: {(A, σ), (B, σ)} — gives A and B the same context.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Attach the store AFTER building the diagram (mirrors toy_weighted's
    // contract) and write the slot's value into it, so the content-twin fold takes
    // the weighted scaling path.
    let mut ws = crate::weight_store::WeightStore::new(
        RationalWeights::from_weights(&[(v.clone(), v.clone())]),
        Precision::Exact,
    );
    ws.set_level(m_v.idx(), vec![crate::query::WeightVal::exact(v.clone())]);
    tdd.attach_weights(ws);

    tdd.scratch.dirty_contract.push(root.0);

    // Run the contraction (this is the call that would PANIC on unfixed code).
    let result = contract_all_twins_topdown(&lim, &mut tdd, None);

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
                            crate::query::WeightVal::Log(_) => {
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
    let lim = Limits::new();
    let _thr = crate::diagram::marg::set_marg_inline_max(0);

    const COUNT_P: u128 = 5;
    const COUNT_Q: u128 = 7;
    const COUNT_R: u128 = 11;

    let vtree = Arc::new(crate::test_helpers::boundary_internal_marg_vtree());
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

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    levels[m_v.idx()].make_marginal(vec![COUNT_P, COUNT_Q, COUNT_R], None);
    let slot_p = LocalNodeIdx(MargRef::slot_raw(0));
    let slot_q = LocalNodeIdx(MargRef::slot_raw(1));
    let slot_r = LocalNodeIdx(MargRef::slot_raw(2));

    levels[x_v.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    // bp: P, Q, R — distinct structural lefts so gp pairs stay distinct.
    let p = levels[bp.idx()].push_internal_node(&[InputPair { left: pos, right: slot_p }]);
    let q = levels[bp.idx()].push_internal_node(&[InputPair { left: neg, right: slot_q }]);
    let r = levels[bp.idx()].push_internal_node(&[InputPair { left: one, right: slot_r }]);

    levels[s_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[s_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
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

    levels[sig_l.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::Pos)];
    levels[sig_r.idx()].nodes = vec![crate::diagram::TddNodeData::leaf(LeafLabel::One)];
    let sigma = levels[sigma_v.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);

    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sigma },
        InputPair { left: b, right: sigma },
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);
    crate::diagram::tag_all_marg_side_slots(&mut tdd, None);

    tdd.scratch.dirty_contract.push(root.0);
    contract_all_twins_topdown(&lim, &mut tdd, None).expect("contract_all_twins_topdown");

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
    let lim = Limits::new();
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
        super::dup_resolve::resolve_duplicate_pairs_in_node(&lim, &mut tdd, bp, 0, &mut scratch)
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
    let lim = Limits::new();

    // An inline count at the cap; ×2 overflows the inline range → cannot re-inline.
    let big_inline = MargRef::inline_raw(crate::diagram::MARG_INLINE_MAX as u128)
        .expect("cap value inlines");
    let (mut tdd, _gp, bp, m_v) = b4_leaf_hazard_fixture(big_inline);

    let mut scratch = super::scratch::DupScratch::default();
    let changed =
        super::dup_resolve::resolve_duplicate_pairs_in_node(&lim, &mut tdd, bp, 0, &mut scratch)
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


// ── Mixed twin group: concat wins, the dup member stays ─────────────────────
