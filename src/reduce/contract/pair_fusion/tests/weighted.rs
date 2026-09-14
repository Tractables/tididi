//! Weighted same-left pair fusion: soundness pins for the slot-mode port.
//!
//! The integer fusion tests (`fallible.rs`) cover the count
//! arithmetic. These cover what is genuinely different once the fused value is a
//! SIGNED semiring element read out of the external `WeightStore`:
//!
//!   * a group can cancel to exactly ZERO, which is a real value and must never
//!     be confused with the bit-31 structural-FALSE sentinel;
//!   * fusion must not disturb the slots it read (other parents still reference
//!     them with their original values);
//!   * Two groups that fuse to EQUAL values share one slot, and the
//!     parent's pair MULTISET must survive that sharing;
//!   * the level's live width must still cover every slot ref the parent holds;
//!   * the bounded-precision Log domain is excluded and must behave exactly like
//!     the fusion-off path.
//!
//! These pin the fusion-on, Exact-domain behavior (there is no opt-out).

use super::*;
use crate::diagram::{ValueRef, NodeIdx};

use crate::engine::Engine;
use crate::diagram::*;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

use crate::diagram::{LiteralWeights, RationalWeights, SignedLog, WeightValue};
use crate::marginal::marginalize_leaf_weighted;
use crate::diagram::{MarginalSide, LeafLabel, TddLevel, TddNodeId, LEAF_WIDTH};
use crate::diagram::{Arithmetic, WeightStore};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use crate::test_helpers::{rat, toy_weighted};
use std::sync::Arc;

/// Slots the attached store holds for `level`.
fn store_len(tdd: &Tdd, level: VtreeIdx) -> usize {
    with_ws(tdd, |ws| ws.level(level.idx()).map_or(0, |s| s.len()))
}

/// Read the store the fixture attached to `tdd`.
fn with_ws<R>(tdd: &Tdd, f: impl FnOnce(&WeightStore) -> R) -> R {
    f(tdd.weights().expect("the fixture attaches a weight store"))
}

/// Two variables' `(w⁻, w⁺)` literal weights. Deliberately awkward rationals so
/// any dropped or duplicated factor is unmistakable.
///
/// Var 1 — the one carried by the marginal-side LEAF in [`weighted_leaf_fixture`] —
/// is ASYMMETRIC (`w⁺ = −4/9 ≠ 1/3 = w⁻`, both nonzero, `w⁺+w⁻ ≠ 0`), so its
/// pinned column holds three DISTINCT values and `leaf_canon_map` is the
/// identity. That is the regime equal-value ref canonicalization cannot touch and
/// only the sum lookup reaches.
fn fixture_weights() -> Vec<LiteralWeights<BigRational>> {
    vec![LiteralWeights { negative: rat(2, 5), positive: rat(3, 11) }, LiteralWeights { negative: rat(1, 3), positive: rat(-4, 9) }]
}

/// Same shape as [`fixture_weights`] but with `w⁺ = w⁻` on var 1, the case where
/// `leaf_canon_map` is `[0, 1, 1]` (Neg → Pos) and a leaf group is therefore a
/// post-canon DUPLICATE run.
fn equal_leaf_weights() -> Vec<LiteralWeights<BigRational>> {
    vec![LiteralWeights { negative: rat(2, 5), positive: rat(3, 11) }, LiteralWeights { negative: rat(2, 7), positive: rat(2, 7) }]
}

/// Build a `balanced(3)` diagram whose RIGHT child level is WEIGHT-marginal with one
/// slot per entry of `values`, and whose root holds one internal node per entry of
/// `nodes` (each a list of `(x_idx, slot_idx)` pairs; `x_idx` is a leaf-label
/// index on the explicit left side, which is a LEAF). Installs the weight context
/// holding `values`.
///
/// `balanced(3)` puts an INTERNAL node on the marginal (right) side while keeping
/// the explicit (left) side a leaf — the boundary class where the fused value is
/// MINTED as a fresh appended slot. A weight-marginal LEAF's column is PINNED to
/// the label-ordered 3-slot `leaf_val` cache and admits no mint, so that boundary
/// folds by sum-lookup instead and has its own fixture
/// ([`weighted_leaf_fixture`]) and its own tests (T6–T8).
///
/// Returns `(tdd, root, marginal_level)`.
fn weighted_fixture(
    values: &[BigRational],
    nodes: &[Vec<(u32, u32)>],
) -> (Tdd, VtreeIdx, VtreeIdx) {
    let ws = WeightStore::new(
        RationalWeights::from_literals(&fixture_weights()),
        Arithmetic::ExactRational,
    );
    // A bare marginal-side ref IS its slot index, which is the polarity
    // `toy_weighted` reads `nodes` in.
    let pair_lists: Vec<&[(u32, u32)]> = nodes.iter().map(|n| n.as_slice()).collect();
    let tdd = toy_weighted(ws, values.to_vec(), &pair_lists);
    let root = tdd.vtree.root();
    let (_, right) = tdd.vtree.children(root);
    (tdd, root, right)
}

/// Build a `balanced(2)` diagram whose RIGHT child is a weight-marginal LEAF, and
/// whose root holds one internal node per entry of `nodes` (each a list of
/// `(x_label, marginal_label)` pairs — the explicit LEFT side is a leaf, so `x_label`
/// is a leaf-label index, and a bare marginal-side ref into a weight-marginal leaf IS
/// a leaf label too, aliasing the pinned column slot of the same index).
///
/// The leaf level is made marginal by the PRODUCTION path
/// (`marginalize_leaf_weighted`) rather than by hand, so the installed column is
/// the real pinned `leaf_val` triple and the parent's refs have already been
/// through equal-value canonicalization — exactly the state pair fusion meets at a
/// leaf boundary in a weighted compile.
///
/// Returns `(tdd, root, leaf_level)`.
fn weighted_leaf_fixture(
    weights: &[LiteralWeights<BigRational>],
    nodes: &[Vec<(u32, u32)>],
) -> (Tdd, VtreeIdx, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let (left, right) = vtree.children(root);
    assert!(
        vtree.node(left).is_leaf() && vtree.node(right).is_leaf(),
        "balanced(2) must put a LEAF on both sides of the root"
    );
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    for node in nodes {
        let ps: Vec<ChildPair> = node
            .iter()
            .map(|&(x, s)| ChildPair::new(NodeIdx(x), NodeIdx(ValueRef::slot_raw(s))))
            .collect();
        levels[root.idx()].push_internal_node(&ps);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    let mut ws = WeightStore::new(
        RationalWeights::from_literals(weights),
        Arithmetic::ExactRational,
    );
    // `marginalize_leaf_weighted` borrows the vtree while mutating the diagram.
    let vt = Arc::clone(&tdd.vtree);
    marginalize_leaf_weighted(&mut tdd, right, &vt, &mut ws);
    tdd.set_weights(ws).unwrap();
    assert!(
        tdd.levels[right.idx()].is_weight_marginal(),
        "leaf fixture: the marginal-side leaf level must end WEIGHT-marginal"
    );
    (tdd, root, right)
}

/// THE PIN INVARIANT, asserted from the outside: the leaf's column is still the
/// exactly-`LEAF_WIDTH`, label-ordered `leaf_val` cache, and the level still
/// advertises that width. A leaf fold that minted (or rewrote) a slot fails here.
fn assert_leaf_column_pinned(tdd: &Tdd, ws: &WeightStore, leaf: VtreeIdx) {
    let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(leaf) else {
        panic!("assert_leaf_column_pinned: not a vtree leaf")
    };
    let col = ws.level(leaf.idx()).expect("pinned leaf column");
    assert_eq!(col.len(), LEAF_WIDTH, "the pinned column must keep exactly LEAF_WIDTH slots");
    assert_eq!(
        tdd.levels[leaf.idx()].slot_count(),
        LEAF_WIDTH,
        "the leaf level's live width must stay pinned at LEAF_WIDTH"
    );
    for (i, slot_val) in col.iter().enumerate() {
        assert_eq!(
            slot_val.clone().into_rational_opt().expect("fixture is exact-domain"),
            ws.leaf_val(var, LeafLabel::from_idx(i))
                .into_rational_opt()
                .expect("fixture is exact-domain"),
            "leaf column slot {i} is no longer the label-ordered leaf_val cache"
        );
    }
}

/// Resolve a marginal-side ref to its exact value.
fn marginal_value(ws: &WeightStore, marginal: VtreeIdx, raw: u32) -> BigRational {
    let ValueRef::Slot(s) = ValueRef::from_raw(MarginalSide(raw)) else {
        panic!("weighted marginal-side refs are bare slots")
    };
    ws.level(marginal.idx()).expect("weighted level")[s as usize]
        .clone()
        .into_rational_opt()
        .expect("fixture is exact-domain")
}

/// The semiring value of root node `n`: `Σ over pairs W(x)·W(m)` — the whole
/// diagram's value for this two-level fixture. This is the quantity every
/// count-neutral rewrite must preserve.
fn node_value(
    tdd: &Tdd,
    ws: &WeightStore,
    root: VtreeIdx,
    marginal: VtreeIdx,
    n: usize,
) -> BigRational {
    let (left, _) = tdd.vtree.children(root);
    let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(left) else {
        panic!("balanced(3) left child must be a leaf")
    };
    let mut acc = BigRational::zero();
    for p in tdd.levels[root.idx()].pairs_of_idx(n) {
        let xw = ws
            .leaf_val(var, LeafLabel::from_idx(p.left.0 as usize))
            .into_rational_opt()
            .expect("fixture is exact-domain");
        acc += xw * marginal_value(ws, marginal, p.right.0);
    }
    acc
}

/// Every marginal-side ref the parent still holds must resolve in bounds, and the
/// level's live width (`weight_width` on a weight-marginal level, which is
/// what apply sizes its buffers from) must cover the whole WeightStore vec.
fn assert_refs_and_width_in_sync(tdd: &Tdd, ws: &WeightStore, root: VtreeIdx, marginal: VtreeIdx) {
    let store_len = ws.level(marginal.idx()).expect("weighted level").len();
    assert_eq!(
        tdd.levels[marginal.idx()].slot_count(),
        store_len,
        "weight-marginal level width must track the WeightStore length \
         (a missed weight_width bump mis-sizes apply buffers)",
    );
    for n in 0..tdd.levels[root.idx()].nodes.len() {
        if tdd.levels[root.idx()].nodes[n].is_leaf() {
            continue;
        }
        for p in tdd.levels[root.idx()].pairs_of_idx(n) {
            let raw = p.right.0;
            assert!(
                !MarginalSide(raw).is_zero_sentinel(),
                "marginal ref {raw} aliases the ZERO sentinel"
            );
            match ValueRef::from_raw(MarginalSide(raw)) {
                ValueRef::Slot(s) => assert!(
                    (s as usize) < store_len,
                    "slot ref {s} out of range for a store of {store_len}",
                ),
                ValueRef::Inline(g) => panic!("weighted marginal-side ref must be a slot, got Inline({g})"),
            }
        }
    }
}

// ── T1: signed cancellation to exactly zero ──────────────────────────────────

/// Two pairs at the same `(node, x)` whose slot values are `+3/7` and `−3/7`.
/// Fusion must collapse them into one pair whose marginal ref resolves to a real
/// zero value — never the bit-31 ZERO sentinel (which denotes the structural
/// FALSE node; conflating the two corrupts the Boolean structure). The whole
/// diagram's semiring value is unchanged (it was zero at this node's x, and
/// stays zero).
#[test]
fn weighted_fusion_cancels_to_a_real_zero_value() {
    let eng = Engine::new();
    let a = rat(3, 7);
    let (mut tdd, root, marginal) = weighted_fixture(
        &[a.clone(), -a.clone()],
        &[vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marginal, 0));
    let size_before = tdd.pair_count();
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs_len, fused_val, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marginal);
        let ps = tdd.levels[root.idx()].pairs_of_idx(0);
        (ps.len(), marginal_value(ws, marginal, ps[0].right.0), node_value(&tdd, ws, root, marginal, 0))
    });

    assert_eq!(stats.fusion_groups, 1, "the +a/−a pair pair is one fusion group");
    assert_eq!(size_before - tdd.pair_count(), 1);
    assert_eq!(pairs_len, 1, "fusion must collapse the two pairs to one");
    assert!(fused_val.is_zero(), "fused value must be exactly 0; got {fused_val}");
    assert_eq!(before, after, "fusion must preserve the diagram's semiring value");
    assert!(before.is_zero(), "the fixture's value is zero by construction");
}

// ── T2: the read slots survive for their other contexts ──────────────────────

/// The same two slots are ALSO referenced from a second root node, there under
/// DIFFERENT explicit children (so nothing fuses at that node). Fusing the first
/// node must leave both the second node's pair list and the slots' original
/// `±a` values intact.
#[test]
fn weighted_fusion_leaves_other_contexts_untouched() {
    let eng = Engine::new();
    let a = rat(3, 7);
    let (mut tdd, root, marginal) = weighted_fixture(
        &[a.clone(), -a.clone()],
        &[
            // node 0: fusable (same x on both pairs)
            vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)],
            // node 1: not fusable (distinct x per pair)
            vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Neg as u32, 1)],
        ],
    );

    let before_other = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marginal, 1));
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (other_pairs, other_vals, after_other, slot0, slot1) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marginal);
        let ps: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(1).to_vec();
        let values: Vec<BigRational> =
            ps.iter().map(|p| marginal_value(ws, marginal, p.right.0)).collect();
        let store = ws.level(marginal.idx()).expect("weighted level");
        let unwrap = |i: usize| {
            store[i].clone().into_rational_opt().expect("fixture is exact-domain")
        };
        (ps, values, node_value(&tdd, ws, root, marginal, 1), unwrap(0), unwrap(1))
    });

    assert_eq!(stats.fusion_groups, 1, "only node 0 has a same-x group");
    assert_eq!(other_pairs.len(), 2, "the non-fusable node keeps both pairs");
    assert_eq!(other_vals, vec![a.clone(), -a.clone()], "its refs keep their original values");
    assert_eq!(slot0, a, "slot 0 must not be rewritten by the fusion at node 0");
    assert_eq!(slot1, -a, "slot 1 must not be rewritten by the fusion at node 0");
    assert_eq!(before_other, after_other, "the other node's value is unchanged");
}

// ── T3: two groups fusing to EQUAL values share a ref, multiset survives ─────

/// One node carrying two groups whose sums collide: `{1/2, 1/2}` and
/// `{1/3, 2/3}` both fuse to `1`. Interning makes both fused refs the same
/// value (and, being equal, the same intern index) — so the pair list must keep
/// both occurrences. A pair-list dedup anywhere here would silently drop one
/// group's `W(x)·1` contribution.
#[test]
fn weighted_fusion_keeps_both_occurrences_on_an_equal_sum_collision() {
    let eng = Engine::new();
    let (mut tdd, root, marginal) = weighted_fixture(
        &[rat(1, 2), rat(1, 2), rat(1, 3), rat(2, 3)],
        &[vec![
            (LeafLabel::Pos as u32, 0),
            (LeafLabel::Neg as u32, 2),
            (LeafLabel::Pos as u32, 1),
            (LeafLabel::Neg as u32, 3),
        ]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marginal, 0));
    let size_before = tdd.pair_count();
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs, values, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marginal);
        let ps: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        let values: Vec<BigRational> =
            ps.iter().map(|p| marginal_value(ws, marginal, p.right.0)).collect();
        (ps, values, node_value(&tdd, ws, root, marginal, 0))
    });

    assert_eq!(stats.fusion_groups, 2, "x=Pos and x=Neg are two independent groups");
    assert_eq!(size_before - tdd.pair_count(), 2, "each group of 2 removes one pair");
    assert_eq!(pairs.len(), 2, "both fused occurrences must be retained");
    let one = BigRational::from_integer(BigInt::from(1));
    assert_eq!(values, vec![one.clone(), one], "both groups fuse to exactly 1");
    // Distinct x sides survive: the collision is on the VALUE, not the pair.
    let mut xs: Vec<u32> = pairs.iter().map(|p| p.left.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![LeafLabel::Pos as u32, LeafLabel::Neg as u32]);
    assert_eq!(before, after, "fusion must preserve the diagram's semiring value");
}

// ── T4: width / ref-range sync after fusion ──────────────────────────────────

/// The width pin on its own, over a group large enough to exercise the >2
/// accumulate: every surviving marginal ref resolves in bounds and the weight-
/// marginal level's `slot_count()` still equals the WeightStore length. (The
/// intern-table-full SLOT fallback — the one branch that bumps
/// `weight_width` — needs >2^30 distinct values to reach and cannot be
/// provoked from a test; this asserts the invariant it exists to maintain.)
#[test]
fn weighted_fusion_keeps_width_and_refs_in_sync() {
    let eng = Engine::new();
    let values = [rat(1, 2), rat(-1, 3), rat(5, 7), rat(2, 9)];
    let (mut tdd, root, marginal) = weighted_fixture(
        &values,
        &[vec![
            (LeafLabel::Pos as u32, 0),
            (LeafLabel::Pos as u32, 1),
            (LeafLabel::Pos as u32, 2),
            (LeafLabel::Pos as u32, 3),
        ]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marginal, 0));
    let size_before = tdd.pair_count();
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs_len, fused, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marginal);
        let ps = tdd.levels[root.idx()].pairs_of_idx(0);
        (ps.len(), marginal_value(ws, marginal, ps[0].right.0), node_value(&tdd, ws, root, marginal, 0))
    });

    assert_eq!(stats.fusion_groups, 1);
    assert_eq!(size_before - tdd.pair_count(), 3, "a group of 4 removes three pairs");
    assert_eq!(pairs_len, 1);
    let expect: BigRational = values.iter().cloned().sum();
    assert_eq!(fused, expect, "fused value must be the exact sum of all four slots");
    assert_eq!(before, after, "fusion must preserve the diagram's semiring value");
}

// ── T5: the Log domain is excluded ───────────────────────────────────────────

/// The bounded-precision Log domain is excluded by the fusion gate,
/// so the sweep must return default stats and leave the diagram byte-identical
/// to the fusion-off path — repeated signed `add_assign` on a signed-log value
/// is order-dependent and cancellation-prone, so summing there is not sound.
#[test]
fn weighted_fusion_does_not_run_in_the_log_domain() {
    let eng = Engine::new();
    let a = rat(3, 7);
    // Build the fixture (attaching an Exact store), then REPLACE it with a
    // Log-domain store carrying the same values.
    let (mut tdd, root, marginal) = weighted_fixture(
        &[a.clone(), rat(5, 7)],
        &[vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)]],
    );
    let mut ws = WeightStore::new(
        tdd.weights().unwrap().algebra().clone(),
        Arithmetic::SignedLog,
    );
    ws.set_level(
        marginal.idx(),
        vec![
            WeightValue::Log(SignedLog::from_rational(&a)),
            WeightValue::Log(SignedLog::from_rational(&rat(5, 7))),
        ],
    );
    tdd.weights = Some(ws);

    let before: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
    let slots_before = store_len(&tdd, marginal);
    let stats = fuse_pairs(&eng, &mut tdd).expect("the log-domain gate must not error");
    let after: Vec<ChildPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();

    assert_eq!(stats.fusion_groups, 0, "log domain must not fuse");
    assert_eq!(store_len(&tdd, marginal), slots_before, "log domain must mint nothing");
    assert_eq!(before, after, "log domain must leave the pair list untouched");
    crate::test_helpers::oracle::assert_canonical(&tdd);
}

// ── T6: a LEAF boundary folds (x,Pos)+(x,Neg) onto the pinned One slot ───────

mod leaf;
