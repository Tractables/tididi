//! Content-twin forking at plain levels.
//!
//! Sibling of `inline_denorm.rs`.

use crate::Engine;
use crate::diagram::{ValueRef, NodeIdx};
use crate::diagram::*;
use std::sync::Arc;
use crate::vtree::VtreeIdx;
use super::sweep::contract_all_twins;

/// Build the `boundary_internal_marginal_vtree` fixture with content-equal
/// twins `A` and `B` at `gp` and `m_v` initialized by `init_m`.
fn boundary_content_twins_fixture(
    init_m: impl FnOnce(&mut TddLevel),
) -> (Tdd, VtreeIdx, VtreeIdx, VtreeIdx, VtreeIdx, NodeIdx) {
    let vtree = Arc::new(crate::test_helpers::boundary_internal_marginal_vtree());
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, sigma_v) = vtree.children(root);
    assert!(matches!(*vtree.node(gp), crate::vtree::VtreeNode::Internal { .. }));
    let (bp, s_v) = vtree.children(gp);
    assert!(matches!(*vtree.node(bp), crate::vtree::VtreeNode::Internal { .. }));
    let (_, m_v) = vtree.children(bp);
    assert!(matches!(*vtree.node(m_v), crate::vtree::VtreeNode::Internal { .. }));

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    init_m(&mut levels[m_v.idx()]);
    let slot_0 = NodeIdx(ValueRef::slot_raw(0));
    let p = levels[bp.idx()].push_internal_node(&[ChildPair::new(pos, slot_0)]);
    let s = levels[s_v.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let a = levels[gp.idx()].push_internal_node(&[ChildPair::new(p, s)]);
    let b = levels[gp.idx()].push_internal_node(&[ChildPair::new(p, s)]);
    let sigma = levels[sigma_v.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    levels[root.idx()].push_internal_node(&[ChildPair::new(a, sigma), ChildPair::new(b, sigma)]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    (tdd, root, gp, bp, m_v, s)
}

/// Directed fixture for the multiplicity of a duplicate pair: content-equal
/// context-twins at a PLAIN level whose merge mints a duplicate pair.
///
/// Fixture (`boundary_internal_marginal_vtree`), left spine root → gp → bp:
///   m       = bp's right child (INTERNAL level) made MARGINAL; one slot, count 5
///   bp      = boundary parent; one node P = {(Pos, slot_0)}
///   gp      = PLAIN level; two nodes A = B = {(P, s)} — content-equal
///   s       = one node {(Pos, One)} at gp's right child (plain sibling)
///   root    = one node {(A, σ), (B, σ)} — A and B share context {(root0, σ)}
///   σ       = one node {(Pos, One)} at root's right child
///
/// Denoted count through root: MC(A)·MC(σ) + MC(B)·MC(σ) = 2·5·(…) — the twin
/// merge must preserve the factor 2. Expected: A, B merge; the survivor's
/// concat {(P,s), (P,s)} is kept as two multiset terms summing to 2·5·MC(s).
/// What must not happen is set-dedup, which would drop a term and halve the
/// total to 5.
#[test]
fn plain_level_content_twins_fork_multiplicity_down() {
    let eng = Engine::new();
    const COUNT: u128 = 5;

    let (mut tdd, root, gp, bp, m_v, s) =
        boundary_content_twins_fixture(|m| m.become_marginal(vec![COUNT], None));
    crate::diagram::inline_small_marginal_refs(&crate::limits::Limits::new(), &mut tdd, None).unwrap();

    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contract_all_twins");

    // Root: one pair (survivor, σ).
    assert_eq!(tdd.levels[root.idx()].pair_count_at(0), 1, "root must end with 1 pair");
    let surv = tdd.levels[root.idx()].pairs_vec(0)[0].left.0 as usize;

    // Survivor at gp: both duplicate terms remain — the multiplicity is carried
    // by the pair list itself, not set-dedup'd away.
    let surv_pairs: Vec<_> = tdd.levels[gp.idx()].pairs_vec(surv).to_vec();
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
    let marginal_counts = tdd.levels[m_v.idx()].marginal_counts().unwrap();
    let total: u128 = surv_pairs
        .iter()
        .map(|pr| {
            let p_pair = tdd.levels[bp.idx()].pairs_vec(pr.left.0 as usize)[0];
            match ChildDecoder::marginal().value(p_pair.right) {
                ValueRef::Slot(sl) => marginal_counts[sl as usize],
                ValueRef::Inline(c) => c as u128,
            }
        })
        .sum();
    assert_eq!(total, 2 * COUNT, "the kept run must still total 2*COUNT, got {total}");

    // No twins left anywhere.
    crate::test_helpers::check::marginal::check_twin_canonicality(&tdd)
        .unwrap_or_else(|e| panic!("twin survived contraction: {e}"));
}

/// WEIGHTED analogue of `plain_level_content_twins_fork_multiplicity_down`.
///
/// Same fixture, but the marginal child `m` is a WEIGHT-marginal level: its per-slot
/// value lives in the level's weight column (a `BigRational`), and
/// `marginal_counts` is `None`. On a weighted vtree the contraction
/// concat-merges the two
/// content-equal twins A,B, leaving the survivor with the duplicate pair
/// `(P, s),(P, s)`.
///
/// What this pins is that the weighted twin-fold completes without erroring
/// or adding a weight value, and that the multiplicity survives as two
/// multiset terms: their values SUM to 2·(3/7) = 6/7, where a set-dedup would
/// leave 3/7.
#[test]
fn weighted_plain_level_content_twins_fork_multiplicity_down() {
    let eng = Engine::new();
    use crate::diagram::{LiteralWeights, RationalWeights};
    use crate::diagram::Arithmetic;
    use num_bigint::BigInt;
    use num_rational::BigRational;


    // The slot value to be scaled. A non-trivial rational so a missing ×2 (or a
    // set-dedup that drops multiplicity) is unmistakable.
    let v = BigRational::new(BigInt::from(3), BigInt::from(7)); // 3/7
    let two = BigRational::from_integer(BigInt::from(2));

    let (tdd, root, gp, bp, m_v, s) =
        boundary_content_twins_fixture(|m| m.become_marginal_weighted(vec![crate::diagram::WeightValue::exact(BigRational::new(BigInt::from(3), BigInt::from(7)))]));

    // The fixture's levels on a weighted twin of its vtree, so the content-twin
    // fold takes the weighted scaling path.
    let ws = crate::diagram::Weights::new(
        RationalWeights::from_literals(&vec![LiteralWeights { negative: v.clone(), positive: v.clone() }; tdd.vtree.num_vars() as usize]),
        Arithmetic::ExactRational,
    );
    let weighted = Arc::new(crate::vtree::Vtree::clone(&tdd.vtree).with_weights(ws).unwrap());
    let mut tdd = Tdd::from_levels_unchecked(weighted, tdd.levels.to_vec(), tdd.output);

    tdd.seed_contract_worklist([root.0]);

    // Run the contraction (this is the call that would PANIC on unfixed code).
    let result = contract_all_twins(&eng, &mut tdd);

    let captured: Option<(usize, BigRational, bool, usize, BigRational)> =
        result.as_ref().ok().map(|_| {
            // Root: one pair (survivor, σ).
            let surv = tdd.levels[root.idx()].pairs_vec(0)[0].left.0 as usize;
            // Survivor at gp: both duplicate terms (P, s) remain.
            let surv_pairs: Vec<(u32, u32)> = tdd.levels[gp.idx()]
                .pairs_vec(surv)
                .iter()
                .map(|pr| (pr.left.0, pr.right.0))
                .collect();
            let sibling_ok = surv_pairs.iter().all(|&(_, r)| r == s.0);
            // Sum the weighted values the terms carry, and record the store's
            // slot count — nothing may have been minted into it.
            let (total, n_slots) = {
                let level =
                    tdd.levels[m_v.idx()].weight_values().expect("weight store level");
                let mut acc = BigRational::from_integer(BigInt::from(0));
                for &(l, _) in &surv_pairs {
                    let p_pair = tdd.levels[bp.idx()].pairs_vec(l as usize)[0];
                    let slot = match ChildDecoder::marginal().value(p_pair.right) {
                        ValueRef::Slot(sl) => sl as usize,
                        ValueRef::Inline(_) => unreachable!("weighted marginal ref is never inline"),
                    };
                    // Non-exhaustive on purpose: the Exact domain has two
                    // representations (`Exact`/`ExactSmall`), and
                    // `as_rational` is the one canonical read of either.
                    acc += match &level[slot] {
                            crate::diagram::WeightValue::Log(_) => {
                                panic!("test expects exact mode")
                            }
                            v => v.as_rational().into_owned(),
                        };
                }
                (acc, level.len())
            };
            (surv_pairs.len(), total, sibling_ok, n_slots, BigRational::clone(&v))
        });

    result.expect("contract_all_twins (weighted twin-fold)");
    let (surv_pairs, total, sibling_ok, n_slots, orig_v) =
        captured.expect("captured assertion inputs");

    assert_eq!(surv_pairs, 2, "gp survivor must keep both duplicate terms");
    assert!(sibling_ok, "plain sibling side must be untouched");
    assert_eq!(n_slots, 1, "nothing absorbed the factor — no fresh weight slot");
    // The kept run's values must SUM to 2·(3/7) = 6/7 — multiplicity carried by
    // the pair list, not set-dedup'd (which would leave the total at 3/7).
    assert_eq!(
        total,
        &orig_v * &two,
        "the kept run must still total 2·(3/7) = 6/7, got {total}",
    );
}

/// Partial-overlap variant: context twins sharing one pair (not all). The
/// shared pair keeps its multiplicity; the disjoint remainder concats.
///   A = {(P, s), (Q, t)},  B = {(P, s), (R, u)}  →
///   survivor = {(P, s), (P, s), (Q, t), (R, u)}.
/// The shared pair's duplicates remain as two multiset terms. The multiset total is what count soundness rests on, and it is unchanged.
/// Uses `boundary_internal_marginal_vtree` so `m` is an internal marginal level.
#[test]
fn plain_level_partial_overlap_twins_fork_shared_pair_down() {
    let eng = Engine::new();

    const COUNT_P: u128 = 5;
    const COUNT_Q: u128 = 7;
    const COUNT_R: u128 = 11;

    let vtree = Arc::new(crate::test_helpers::boundary_internal_marginal_vtree());
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (gp, sigma_v) = vtree.children(root);
    let (bp, s_v) = vtree.children(gp);
    let (_, m_v) = vtree.children(bp);
    assert!(matches!(*vtree.node(m_v), crate::vtree::VtreeNode::Internal { .. }));

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<crate::diagram::TddLevel> =
        (0..vtree.num_nodes()).map(|_| crate::diagram::TddLevel::new()).collect();

    levels[m_v.idx()].become_marginal(vec![COUNT_P, COUNT_Q, COUNT_R], None);
    let slot_p = NodeIdx(ValueRef::slot_raw(0));
    let slot_q = NodeIdx(ValueRef::slot_raw(1));
    let slot_r = NodeIdx(ValueRef::slot_raw(2));

    // bp: P, Q, R — distinct structural lefts so gp pairs stay distinct.
    let p = levels[bp.idx()].push_internal_node(&[ChildPair::new(pos, slot_p)]);
    let q = levels[bp.idx()].push_internal_node(&[ChildPair::new(neg, slot_q)]);
    let r = levels[bp.idx()].push_internal_node(&[ChildPair::new(one, slot_r)]);

    let s = levels[s_v.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let t = levels[s_v.idx()].push_internal_node(&[ChildPair::new(neg, one)]);
    let u = levels[s_v.idx()].push_internal_node(&[ChildPair::new(pos, neg)]);

    // gp: A = {(P,s),(Q,t)}, B = {(P,s),(R,u)} — shared pair (P,s).
    let a = levels[gp.idx()].push_internal_node(&[
        ChildPair::new(p, s),
        ChildPair::new(q, t),
    ]);
    let b = levels[gp.idx()].push_internal_node(&[
        ChildPair::new(p, s),
        ChildPair::new(r, u),
    ]);

    let sigma = levels[sigma_v.idx()].push_internal_node(&[ChildPair::new(pos, one)]);

    levels[root.idx()].push_internal_node(&[
        ChildPair::new(a, sigma),
        ChildPair::new(b, sigma),
    ]);

    let output = crate::diagram::TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::from_levels_unchecked(vtree, levels, output);
    crate::diagram::inline_small_marginal_refs(&crate::limits::Limits::new(), &mut tdd, None).unwrap();

    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contract_all_twins");

    assert_eq!(tdd.levels[root.idx()].pair_count_at(0), 1, "root must end with 1 pair");
    let surv = tdd.levels[root.idx()].pairs_vec(0)[0].left.0 as usize;

    // Survivor: 4 pairs — the shared (P,s) kept TWICE plus the disjoint
    // (Q,t), (R,u).
    let surv_pairs: Vec<_> = tdd.levels[gp.idx()].pairs_vec(surv).to_vec();
    assert_eq!(surv_pairs.len(), 4, "survivor must hold 4 pairs, got {}", surv_pairs.len());

    let marginal_counts = tdd.levels[m_v.idx()].marginal_counts().unwrap();
    let decode = |raw: u32| -> u128 {
        match ChildDecoder::marginal().value(EncodedChildRef::from_raw(raw)) {
            ValueRef::Slot(sl) => marginal_counts[sl as usize],
            ValueRef::Inline(c) => c as u128,
        }
    };
    // Collect the multiset of decoded counts of the survivor's left children.
    let mut counts: Vec<u128> = surv_pairs
        .iter()
        .map(|pr| decode(tdd.levels[bp.idx()].pairs_vec(pr.left.0 as usize)[0].right.0))
        .collect();
    counts.sort_unstable();
    assert_eq!(
        counts,
        vec![COUNT_P, COUNT_P, COUNT_Q, COUNT_R],
        "shared pair must keep multiplicity 2; disjoint pairs unchanged"
    );
    // Count soundness, unchanged in strength: the multiset still totals
    // 2·`COUNT_P` + `COUNT_Q` + `COUNT_R` — exactly what the collapsed form
    // {(P₂ = 2·`COUNT_P`, s), (Q,t), (R,u)} carried.
    assert_eq!(
        counts.iter().sum::<u128>(),
        2 * COUNT_P + COUNT_Q + COUNT_R,
        "the kept run must preserve the survivor's total"
    );
}

// ── Mixed twin group: concat wins, the duplicate member stays ─────────────────────
