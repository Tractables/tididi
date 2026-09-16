//! Weighted fusion at the leaf levels.
//!
//! Child of `weighted.rs`, whose fixtures these read.

use super::*;

use crate::Engine;
use num_rational::BigRational;
use crate::diagram::LeafLabel;
use crate::vtree::VtreeNode;

/// Asymmetric weights (w⁺ ≠ w⁻, both nonzero): the pinned column's three values
/// are distinct, so equal-value ref canonicalization does nothing here and the
/// integer arm's twin bonus has no analogue. The group still folds, because its
/// sum is `w⁺ + w⁻`, which is the One slot by definition, for every weight
/// table. That identity is the whole lever: the fold is a parent-pair rewrite
/// only, with no minted slot, no column write and no width bump (a leaf column is
/// pinned compile-wide and aliased by bare leaf-label refs from every other
/// `Tdd`).
#[test]
fn weighted_leaf_fusion_folds_pos_plus_neg_onto_the_pinned_one_slot() {
    let eng = Engine::new();
    let weights = fixture_weights();
    let (mut tdd, root, leaf) = weighted_leaf_fixture(
        &weights,
        &[vec![
            (LeafLabel::Pos as u32, LeafLabel::Pos as u32),
            (LeafLabel::Pos as u32, LeafLabel::Neg as u32),
        ]],
    );
    let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(leaf) else {
        panic!("the fixture's marginal side must be a vtree leaf")
    };
    let LiteralWeights { negative: wn, positive: wp } = weights[var.idx()].clone();

    assert!(crate::test_helpers::check::marginal::check_pair_fusion_saturation(&tdd, None).is_err());
    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, leaf, 0));
    let (slots_before, size_before) = (store_len(&tdd, leaf), tdd.pair_count());
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let pairs: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
    assert_eq!(pairs.len(), 1, "the two pairs must collapse to one");
    let (fused, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, leaf);
        assert_leaf_column_pinned(&tdd, ws, leaf);
        (marginal_value(ws, leaf, pairs[0].right.0), node_value(&tdd, ws, root, leaf, 0))
    });

    assert_eq!(stats.fusion_groups, 1, "the Pos/Neg pair pair is one fusion group");
    assert_eq!(size_before - tdd.pair_count(), 1);
    assert_eq!(store_len(&tdd, leaf), slots_before, "a leaf fold must never mint a slot");
    assert_eq!(
        pairs[0].right.0,
        LeafLabel::One as u32,
        "the fused ref must name the One slot — the canonical slot holding w⁺+w⁻"
    );
    assert_eq!(fused, wp.clone() + wn.clone(), "the fused value must be exactly w⁺ + w⁻");
    assert_eq!(before, after, "the leaf fold must preserve the diagram's semiring value");
    crate::test_helpers::oracle::assert_canonical(&tdd);
}

// ── T7: a leaf group whose sum is not in the column stays unfused ────────────

/// `(x,One), (x,Pos)` sums to `2w⁺ + w⁻`, a value these weights do not put in the
/// pinned column — and a leaf column can never grow to hold it. The plan is
/// dropped: the node's pairs are left byte-identical (an un-fused fusion redex is a
/// size residual, never a wrong value), nothing is minted, and the stats report
/// no fusion — which is what keeps the contract fixpoint from looping forever on
/// a rewrite that never happened.
#[test]
fn weighted_leaf_fusion_declines_a_sum_the_pinned_column_cannot_hold() {
    let eng = Engine::new();
    let weights = fixture_weights();
    let (mut tdd, root, leaf) = weighted_leaf_fixture(
        &weights,
        &[vec![
            (LeafLabel::Pos as u32, LeafLabel::One as u32),
            (LeafLabel::Pos as u32, LeafLabel::Pos as u32),
        ]],
    );
    let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(leaf) else {
        panic!("the fixture's marginal side must be a vtree leaf")
    };
    let LiteralWeights { negative: wn, positive: wp } = weights[var.idx()].clone();
    let want = wp.clone() + wp.clone() + wn.clone(); // (w⁺+w⁻) + w⁺

    let before: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
    let before_val = with_ws(&tdd, |ws| {
        // The premise of the test: this sum really is outside the column, so the
        // fold has no slot to land on. (If a weight change ever made it land,
        // this fires instead of the test silently asserting the wrong thing.)
        let col: Vec<BigRational> = ws
            .level(leaf.idx())
            .expect("pinned leaf column")
            .iter()
            .map(|v| v.clone().into_rational_opt().expect("fixture is exact-domain"))
            .collect();
        assert!(
            !col.contains(&want),
            "fixture premise broken: the pinned column {col:?} already holds 2w⁺+w⁻ = {want}"
        );
        node_value(&tdd, ws, root, leaf, 0)
    });

    let slots_before = store_len(&tdd, leaf);
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let after: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
    let after_val = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, leaf);
        assert_leaf_column_pinned(&tdd, ws, leaf);
        node_value(&tdd, ws, root, leaf, 0)
    });

    assert_eq!(stats.fusion_groups, 0, "a dropped leaf plan is not a fusion");
    assert_eq!(store_len(&tdd, leaf), slots_before, "declining must not mint anything either");
    assert_eq!(before, after, "an unrepresentable group must be left exactly as it was");
    assert_eq!(before_val, after_val, "declining must preserve the diagram's semiring value");
    crate::test_helpers::oracle::assert_canonical(&tdd);
}

// ── T8: equal weights — the post-canon duplicate run folds on either route ───

/// At `w⁺ = w⁻` the leaf-marginal pass canonicalizes `(x,Neg)` onto `(x,Pos)`, so the
/// parent holds a duplicate run. Two rewrites can reach it — pair fusion's group sum
/// (`w⁺ + w⁺`) and duplicate resolution's multiplicity scale (`2·w⁺`) — and both compute
/// the same number, `2w⁺ = w⁺+w⁻ = One`. Whichever runs first must therefore land
/// on the same pinned slot, mint nothing, and keep the value exact.
#[test]
fn weighted_leaf_equal_weight_duplicate_run_folds_to_one_on_either_route() {
    let eng = Engine::new();
    let weights = equal_leaf_weights();
    let node = vec![
        (LeafLabel::Pos as u32, LeafLabel::Pos as u32),
        (LeafLabel::Pos as u32, LeafLabel::Neg as u32),
    ];

    // Route A: pair fusion's sum lookup.
    let (pairs_a, before_a, after_a) = {
            let (mut tdd, root, leaf) = weighted_leaf_fixture(&weights, std::slice::from_ref(&node));
        let canon: Vec<u32> =
            tdd.levels[root.idx()].pairs_of_idx(0).iter().map(|p| p.right.0).collect();
        assert_eq!(
            canon,
            vec![LeafLabel::Pos as u32, LeafLabel::Pos as u32],
            "at w⁺ = w⁻ the leaf-marginal canon pass must rewrite Neg onto Pos"
        );
        let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, leaf, 0));
        let slots_before = store_len(&tdd, leaf);
        fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
        assert_eq!(store_len(&tdd, leaf), slots_before, "a leaf fold must never mint a slot");
        let pairs: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        let after = with_ws(&tdd, |ws| {
            assert_refs_and_width_in_sync(&tdd, ws, root, leaf);
            assert_leaf_column_pinned(&tdd, ws, leaf);
            node_value(&tdd, ws, root, leaf, 0)
        });
        (pairs, before, after)
    };

    // Route B: duplicate resolution's k-scale lookup on the same duplicate run.
    let (pairs_b, before_b, after_b) = {
            let (mut tdd, root, leaf) = weighted_leaf_fixture(&weights, std::slice::from_ref(&node));
        let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, leaf, 0));
        // A fresh bundle: production hands one down from the contract loop and the
        // callee clears it per node, so a default one is the same starting state.
        let mut scratch = crate::reduce::contract::scratch::DuplicateScratch::default();
        let changed = crate::reduce::contract::duplicate_pair_resolve::resolve_duplicate_pairs_in_node(
            &eng,
            &mut tdd,
            root,
            0,
            &mut scratch,
        )
        .expect("no budget → must not over-budget");
        assert!(changed, "the duplicate run must be absorbed by the marginal leaf side");
        let pairs: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        let after = with_ws(&tdd, |ws| {
            assert_refs_and_width_in_sync(&tdd, ws, root, leaf);
            assert_leaf_column_pinned(&tdd, ws, leaf);
            node_value(&tdd, ws, root, leaf, 0)
        });
        (pairs, before, after)
    };

    assert_eq!(pairs_a.len(), 1, "pair fusion must collapse the duplicate run to one pair");
    assert_eq!(
        pairs_a[0].right.0,
        LeafLabel::One as u32,
        "2w⁺ = w⁺+w⁻ must fold onto the One slot"
    );
    assert_eq!(pairs_a, pairs_b, "both routes must produce the identical pair list");
    assert_eq!(before_a, after_a, "pair fusion's fold must preserve the semiring value");
    assert_eq!(before_b, after_b, "duplicate resolution's fold must preserve the semiring value");
}
