//! Weighted same-left pair fusion: soundness pins for the slot-mode port.
//!
//! The integer fusion tests (`p_fusion_fallible_tests.rs`) cover the count
//! arithmetic. These cover what is genuinely different once the fused value is a
//! SIGNED semiring element read out of the external `WeightStore`:
//!
//!   * a group can cancel to EXACTLY ZERO, which is a real value and must never
//!     be confused with the bit-31 structural-FALSE sentinel;
//!   * fusion must not disturb the slots it read (other parents still reference
//!     them with their original values);
//!   * two groups that fuse to EQUAL values share one slot, and the
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

use crate::query::{RationalWeights, SignedLog, WeightVal};
use crate::marginal::marginalize_leaf_weighted;
use crate::diagram::{MargSide, LeafLabel, TddLevel, TddNodeId, LEAF_WIDTH};
use crate::weight_store::{Precision, WeightStore};
use crate::vtree::{Vtree, VtreeNode};
use std::sync::Arc;

/// Read the store the fixture attached to `tdd`.
fn with_ws<R>(tdd: &Tdd, f: impl FnOnce(&WeightStore) -> R) -> R {
    f(tdd.weights().expect("the fixture attaches a weight store"))
}

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

/// Two variables' `(w⁻, w⁺)` literal weights. Deliberately awkward rationals so
/// any dropped or duplicated factor is unmistakable.
///
/// Var 1 — the one carried by the marg-side LEAF in [`weighted_leaf_fixture`] —
/// is ASYMMETRIC (`w⁺ = −4/9 ≠ 1/3 = w⁻`, both nonzero, `w⁺+w⁻ ≠ 0`), so its
/// pinned column holds three DISTINCT values and `leaf_canon_map` is the
/// identity. That is the regime equal-value ref canonicalization cannot touch and
/// only the sum lookup reaches.
fn fixture_weights() -> Vec<(BigRational, BigRational)> {
    vec![(rat(2, 5), rat(3, 11)), (rat(1, 3), rat(-4, 9))]
}

/// Same shape as [`fixture_weights`] but with `w⁺ = w⁻` on var 1, the case where
/// `leaf_canon_map` is `[0, 1, 1]` (Neg → Pos) and a leaf group is therefore a
/// post-canon DUPLICATE run.
fn equal_leaf_weights() -> Vec<(BigRational, BigRational)> {
    vec![(rat(2, 5), rat(3, 11)), (rat(2, 7), rat(2, 7))]
}

/// Build a `balanced(3)` TDD whose RIGHT child level is WEIGHT-marginal with one
/// slot per entry of `vals`, and whose root holds one internal node per entry of
/// `nodes` (each a list of `(x_idx, slot_idx)` pairs; `x_idx` is a leaf-label
/// index on the explicit left side, which is a LEAF). Installs the weight context
/// holding `vals`.
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
    vals: &[BigRational],
    nodes: &[Vec<(u32, u32)>],
) -> (Tdd, VtreeIdx, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(3));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(3) root must be internal"),
    };
    assert!(
        !vtree.node(right).is_leaf(),
        "weighted p-fusion fixture needs an INTERNAL marginal side"
    );
    let mut levels: Vec<TddLevel> =
        (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    // Weight-marginal: structure cleared, `marginal_counts` stays None, the
    // per-slot values live in the WeightStore installed below.
    levels[right.idx()].make_marginal_weighted_with_slots(vals.len() as u32);
    for node in nodes {
        let ps: Vec<InputPair> = node
            .iter()
            .map(|&(x, s)| InputPair {
                left: NodeIdx(x),
                right: NodeIdx(ValueRef::slot_raw(s)),
            })
            .collect();
        levels[root.idx()].push_internal_node(&ps);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::with_levels(vtree, levels, output);

    let mut ws = WeightStore::new(
        RationalWeights::from_weights(&fixture_weights()),
        Precision::Exact,
    );
    ws.set_level(right.idx(), vals.iter().cloned().map(WeightVal::exact).collect());
    tdd.attach_weights(ws);
    (tdd, root, right)
}

/// Build a `balanced(2)` TDD whose RIGHT child is a weight-marginal LEAF, and
/// whose root holds one internal node per entry of `nodes` (each a list of
/// `(x_label, marg_label)` pairs — the explicit LEFT side is a leaf, so `x_label`
/// is a leaf-label index, and a bare marg-side ref into a weight-marginal leaf IS
/// a leaf label too, aliasing the pinned column slot of the same index).
///
/// The leaf level is made marginal by the PRODUCTION path
/// (`marginalize_leaf_weighted`) rather than by hand, so the installed column is
/// the real pinned `leaf_val` triple and the parent's refs have already been
/// through equal-value canonicalization — exactly the state p-fusion meets at a
/// leaf boundary in a weighted compile.
///
/// Returns `(tdd, root, leaf_level)`.
fn weighted_leaf_fixture(
    weights: &[(BigRational, BigRational)],
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
        let ps: Vec<InputPair> = node
            .iter()
            .map(|&(x, s)| InputPair {
                left: NodeIdx(x),
                right: NodeIdx(ValueRef::slot_raw(s)),
            })
            .collect();
        levels[root.idx()].push_internal_node(&ps);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::with_levels(vtree, levels, output);

    let mut ws = WeightStore::new(
        RationalWeights::from_weights(weights),
        Precision::Exact,
    );
    // `marginalize_leaf_weighted` borrows the vtree while mutating the TDD.
    let vt = Arc::clone(&tdd.vtree);
    marginalize_leaf_weighted(&crate::engine::Engine::new(), &mut tdd, right, &vt, &mut ws);
    tdd.attach_weights(ws);
    assert!(
        tdd.levels[right.idx()].is_weight_marginal(),
        "leaf fixture: the marg-side leaf level must end WEIGHT-marginal"
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
        tdd.levels[leaf.idx()].width(),
        LEAF_WIDTH,
        "the leaf level's live width must stay pinned at LEAF_WIDTH"
    );
    for i in 0..LEAF_WIDTH {
        assert_eq!(
            col[i].clone().into_rational_opt().expect("fixture is exact-domain"),
            ws.leaf_val(var, LeafLabel::from_idx(i))
                .into_rational_opt()
                .expect("fixture is exact-domain"),
            "leaf column slot {i} is no longer the label-ordered leaf_val cache"
        );
    }
}

/// Resolve a marg-side ref to its exact value.
fn marg_value(ws: &WeightStore, marg: VtreeIdx, raw: u32) -> BigRational {
    let ValueRef::Slot(s) = ValueRef::from_raw(MargSide(raw)) else {
        panic!("weighted marg-side refs are bare slots")
    };
    ws.level(marg.idx()).expect("weighted level")[s as usize]
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
    marg: VtreeIdx,
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
        acc += xw * marg_value(ws, marg, p.right.0);
    }
    acc
}

/// Every marg-side ref the parent still holds must resolve in bounds, and the
/// level's live width (`weight_width` on a weight-marginal level, which is
/// what apply sizes its buffers from) must cover the whole WeightStore vec.
fn assert_refs_and_width_in_sync(tdd: &Tdd, ws: &WeightStore, root: VtreeIdx, marg: VtreeIdx) {
    let store_len = ws.level(marg.idx()).expect("weighted level").len();
    assert_eq!(
        tdd.levels[marg.idx()].width(),
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
            assert_eq!(raw & (1u32 << 31), 0, "marg ref {raw} aliases the ZERO sentinel");
            match ValueRef::from_raw(MargSide(raw)) {
                ValueRef::Slot(s) => assert!(
                    (s as usize) < store_len,
                    "slot ref {s} out of range for a store of {store_len}",
                ),
                ValueRef::Inline(g) => panic!("weighted marg-side ref must be a slot, got Inline({g})"),
            }
        }
    }
}

// ── T1: signed cancellation to exactly zero ──────────────────────────────────

/// Two pairs at the same `(node, x)` whose slot values are `+3/7` and `−3/7`.
/// Fusion must collapse them into ONE pair whose marg ref resolves to a real
/// zero value — never the bit-31 ZERO sentinel (which denotes the structural
/// FALSE node; conflating the two corrupts the Boolean structure). The whole
/// diagram's semiring value is unchanged (it was zero at this node's x, and
/// stays zero).
#[test]
fn weighted_fusion_cancels_to_a_real_zero_value() {
    let eng = Engine::new();
    let a = rat(3, 7);
    let (mut tdd, root, marg) = weighted_fixture(
        &[a.clone(), -a.clone()],
        &[vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marg, 0));
    let stats = apply_p_fusion(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs_len, fused_val, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marg);
        let ps = tdd.levels[root.idx()].pairs_of_idx(0);
        (ps.len(), marg_value(ws, marg, ps[0].right.0), node_value(&tdd, ws, root, marg, 0))
    });

    assert_eq!(stats.fusion_groups, 1, "the +a/−a pair pair is one fusion group");
    assert_eq!(stats.pairs_eliminated, 1);
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
    let (mut tdd, root, marg) = weighted_fixture(
        &[a.clone(), -a.clone()],
        &[
            // node 0: fusable (same x on both pairs)
            vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)],
            // node 1: NOT fusable (distinct x per pair)
            vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Neg as u32, 1)],
        ],
    );

    let before_other = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marg, 1));
    let stats = apply_p_fusion(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (other_pairs, other_vals, after_other, slot0, slot1) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marg);
        let ps: Vec<InputPair> = tdd.levels[root.idx()].pairs_of_idx(1).to_vec();
        let vals: Vec<BigRational> =
            ps.iter().map(|p| marg_value(ws, marg, p.right.0)).collect();
        let store = ws.level(marg.idx()).expect("weighted level");
        let unwrap = |i: usize| {
            store[i].clone().into_rational_opt().expect("fixture is exact-domain")
        };
        (ps, vals, node_value(&tdd, ws, root, marg, 1), unwrap(0), unwrap(1))
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
/// `{1/3, 2/3}` both fuse to `1`. Interning makes both fused refs the SAME
/// value (and, being equal, the same intern index) — so the pair list must keep
/// BOTH occurrences. A pair-list dedup anywhere here would silently drop one
/// group's `W(x)·1` contribution.
#[test]
fn weighted_fusion_keeps_both_occurrences_on_an_equal_sum_collision() {
    let eng = Engine::new();
    let (mut tdd, root, marg) = weighted_fixture(
        &[rat(1, 2), rat(1, 2), rat(1, 3), rat(2, 3)],
        &[vec![
            (LeafLabel::Pos as u32, 0),
            (LeafLabel::Neg as u32, 2),
            (LeafLabel::Pos as u32, 1),
            (LeafLabel::Neg as u32, 3),
        ]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marg, 0));
    let stats = apply_p_fusion(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs, vals, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marg);
        let ps: Vec<InputPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
        let vals: Vec<BigRational> =
            ps.iter().map(|p| marg_value(ws, marg, p.right.0)).collect();
        (ps, vals, node_value(&tdd, ws, root, marg, 0))
    });

    assert_eq!(stats.fusion_groups, 2, "x=Pos and x=Neg are two independent groups");
    assert_eq!(stats.pairs_eliminated, 2, "each group of 2 removes one pair");
    assert_eq!(pairs.len(), 2, "both fused occurrences must be retained");
    let one = BigRational::from_integer(BigInt::from(1));
    assert_eq!(vals, vec![one.clone(), one], "both groups fuse to exactly 1");
    // Distinct x sides survive: the collision is on the VALUE, not the pair.
    let mut xs: Vec<u32> = pairs.iter().map(|p| p.left.0).collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![LeafLabel::Pos as u32, LeafLabel::Neg as u32]);
    assert_eq!(before, after, "fusion must preserve the diagram's semiring value");
}

// ── T4: width / ref-range sync after fusion ──────────────────────────────────

/// The width pin on its own, over a group large enough to exercise the >2
/// accumulate: every surviving marg ref resolves in bounds and the weight-
/// marginal level's `width()` still equals the WeightStore length. (The
/// intern-table-full SLOT fallback — the one branch that bumps
/// `weight_width` — needs >2^30 distinct values to reach and cannot be
/// provoked from a test; this asserts the invariant it exists to maintain.)
#[test]
fn weighted_fusion_keeps_width_and_refs_in_sync() {
    let eng = Engine::new();
    let vals = [rat(1, 2), rat(-1, 3), rat(5, 7), rat(2, 9)];
    let (mut tdd, root, marg) = weighted_fixture(
        &vals,
        &[vec![
            (LeafLabel::Pos as u32, 0),
            (LeafLabel::Pos as u32, 1),
            (LeafLabel::Pos as u32, 2),
            (LeafLabel::Pos as u32, 3),
        ]],
    );

    let before = with_ws(&tdd, |ws| node_value(&tdd, ws, root, marg, 0));
    let stats = apply_p_fusion(&eng, &mut tdd).expect("no budget → must not over-budget");
    let (pairs_len, fused, after) = with_ws(&tdd, |ws| {
        assert_refs_and_width_in_sync(&tdd, ws, root, marg);
        let ps = tdd.levels[root.idx()].pairs_of_idx(0);
        (ps.len(), marg_value(ws, marg, ps[0].right.0), node_value(&tdd, ws, root, marg, 0))
    });

    assert_eq!(stats.fusion_groups, 1);
    assert_eq!(stats.pairs_eliminated, 3, "a group of 4 removes three pairs");
    assert_eq!(pairs_len, 1);
    let expect: BigRational = vals.iter().cloned().sum();
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
    let (mut tdd, root, marg) = weighted_fixture(
        &[a.clone(), rat(5, 7)],
        &[vec![(LeafLabel::Pos as u32, 0), (LeafLabel::Pos as u32, 1)]],
    );
    let mut ws = WeightStore::new(
        RationalWeights::from_weights(&fixture_weights()),
        Precision::Log,
    );
    ws.set_level(
        marg.idx(),
        vec![
            WeightVal::Log(SignedLog::from_rational(&a)),
            WeightVal::Log(SignedLog::from_rational(&rat(5, 7))),
        ],
    );
    tdd.attach_weights(ws);

    let before: Vec<InputPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();
    let stats = apply_p_fusion(&eng, &mut tdd).expect("the log-domain gate must not error");
    let after: Vec<InputPair> = tdd.levels[root.idx()].pairs_of_idx(0).to_vec();

    assert_eq!(stats.fusion_groups, 0, "log domain must not fuse");
    assert_eq!(stats.pairs_eliminated, 0);
    assert_eq!(stats.slots_added, 0);
    assert_eq!(before, after, "log domain must leave the pair list untouched");
}

// ── T6: a LEAF boundary folds (x,Pos)+(x,Neg) onto the pinned One slot ───────

#[path = "p_fusion_weighted_tests/leaf.rs"]
mod leaf;
