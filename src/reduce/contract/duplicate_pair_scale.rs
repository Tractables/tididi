//! Multiplying a duplicate pair's marginal side by its run length.

use crate::engine::Engine;
use crate::diagram::{ValueRef, NodeIdx};
use crate::diagram::MarginalSide;

use crate::limits::OperationError;
use crate::diagram::ChildSide;
use crate::value::slots::{mint_ref, scaled_weight, SlotValues};
use crate::value::{IntFold, WeightFold};
use crate::diagram::*;
use crate::vtree::VtreeIdx;

/// Scale the marginal-side ref `raw` (into the internal marginal level `mv`) by
/// `k` in the value domain `D`: an inline ref where the domain has one, else a
/// fresh slot. No value interning here — the slot pruner merges equal-valued
/// slots on the next prune.
fn scale_ref<D: SlotValues>(eng: &Engine, tdd: &mut Tdd, mv: VtreeIdx, raw: u32, k: u32) -> Result<u32, OperationError> {
    debug_assert!(k >= 2);
    // A bare marginal-side ref at a leaf is a leaf label, never a store index,
    // so a slot minted into a leaf store would be re-read as a label; the leaf
    // branch of `try_scale_child` absorbs by label or lookup before this.
    debug_assert!(
        !tdd.vtree.node(mv).is_leaf(),
        "refusing to mint a scaled slot into a leaf store: leaf refs are labels"
    );
    let scaled = D::scaled(tdd, mv, raw, k);
    mint_ref::<D>(eng, tdd, mv, scaled)
}

/// Scale an integer-marginal leaf ref by `k` without touching the (empty) leaf
/// store. A bare `Slot(s)` ref is a leaf-label index — decoded with the same
/// fixed-count mapping as `read_marginal_count`;
/// an `Inline(c)` ref carries the count directly. Returns the scaled value as an
/// inline ref (`Some(Ok(..))`), or `None` when the scaled value cannot inline:
/// the leaf side cannot absorb the factor, and we must never mint a slot into a
/// leaf store, so the caller routes the factor to the other side / bails.
fn scale_leaf_marginal_label(raw: u32, k: u32) -> Option<Result<u32, OperationError>> {
    let base: u128 = match ValueRef::from_raw(MarginalSide(raw)) {
        ValueRef::Inline(c) => c as u128,
        // Same fixed-count mapping as `read_marginal_count`: a bare leaf ref is a
        // `LeafLabel` index (Zero→0, One→2, Pos|Neg→1), not a store slot.
        ValueRef::Slot(s) => match LeafLabel::from_idx(s as usize) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        },
    };
    // base ≤ 2 (label) or ≤ `MARGINAL_INLINE_MAX` (inline), k ≤ 2^32−1 ⇒ product fits u128.
    let scaled = base * k as u128;
    // `None` (can't inline) ⇒ this side can't absorb — never mint into a leaf store.
    ValueRef::inline_raw(scaled).map(Ok)
}

/// Scale a ref into a weight-marginal leaf by `k` without minting: compute
/// `k · column[slot(raw)]` and look that value up among the pinned column's own
/// three slots (`diagram::find_leaf_slot_by_value`), returning the slot ref if
/// it is there and `None` if it is not.
///
/// This is the weighted counterpart of `scale_leaf_marginal_label`'s inline
/// absorb. The integer arm can encode any scaled count in the ref itself; a
/// weighted `ValueRef::Inline(gidx)` indexes the store's process-wide intern
/// table, so the only representable results here are the column's existing
/// values. After `marginal::canonicalize_leaf_refs_at_parent` the duplicate
/// runs that reach this path have equal values, and for `w⁺ = w⁻` the product
/// lands on the column: `2·Pos = w⁺+w⁻ = One`.
///
/// Exact domain only: in log mode `weight_key` equality compares `f64` bit
/// patterns, so a hit would be a rounding coincidence, and we decline.
fn scale_weight_leaf_by_lookup(
    ws: &crate::diagram::WeightStore,
    cv: VtreeIdx,
    raw: u32,
    k: u32,
) -> Option<u32> {
    use crate::diagram::find_leaf_slot_by_value;
    debug_assert!(k >= 2);
    // The zero sentinel (bit 31) names no slot. A weighted leaf side carries no
    // inline (bit-30) ref either — nothing mints one — so both decline rather
    // than being decoded as a column index.
    if MarginalSide(raw).is_zero_sentinel() || ValueRef::is_inline_raw(raw) {
        return None;
    }
    let slot = raw as usize;
    {
        if ws.is_log() {
            return None;
        }
        let base = ws.level(cv.idx())?.get(slot)?;
        let want = scaled_weight(ws, base, k);
        // The shared search scans in ascending slot order, so the hit is the
        // canonical slot for that value (`leaf_canon_map`'s min-index rule). Folding
        // onto anything else would re-introduce exactly the non-canonical ref the
        // canon pass exists to remove.
        find_leaf_slot_by_value(ws, cv.idx(), &want).map(ValueRef::slot_raw)
    }
}

/// Try to scale the O(1)-absorbing side of a pair: the ref `raw` into the
/// marginal child level `cv`. Returns `None` when this side declines the factor —
/// an integer-marginal leaf label whose scaled value will not inline, or a
/// weight-marginal leaf whose scaled value is not one the pinned column already
/// carries. Only marginal `cv` reaches here (`scale_pair_one_side`).
fn try_scale_child(
    eng: &Engine,
    tdd: &mut Tdd,
    cv: VtreeIdx,
    raw: u32,
    k: u32,
) -> Option<Result<u32, OperationError>> {
    debug_assert!(
        tdd.levels[cv.idx()].is_marginal(),
        "try_scale_child: only a marginal child is an O(1) absorber",
    );
    {
        // At a leaf nothing is minted. An integer-marginal leaf has an empty
        // store: a bare ref is a leaf label (decoded as `read_marginal_count`
        // does), so the label is scaled and inlined, or `None` says this side
        // cannot absorb. A weight-marginal leaf's column is pinned
        // (`test_helpers::check::marginal::check_leaf_columns_pinned`), so the
        // scaled value is looked up in the column and `None` returned when it
        // is absent. On `None` the caller tries the other side or keeps the
        // run as k multiset terms; nothing has been mutated at that point.
        if tdd.vtree.node(cv).is_leaf() {
            if tdd.levels[cv.idx()].is_weight_marginal() {
                let ws = tdd.weight_store();
                return scale_weight_leaf_by_lookup(ws, cv, raw, k).map(Ok);
            }
            return scale_leaf_marginal_label(raw, k);
        }
        Some(if tdd.levels[cv.idx()].is_weight_marginal() {
            scale_ref::<WeightFold>(eng, tdd, cv, raw, k)
        } else {
            scale_ref::<IntFold>(eng, tdd, cv, raw, k)
        })
    }
}

/// True when a duplicate run at plain level `pv` has an O(1) absorber: one of
/// `pv`'s children is a marginal level, so the multiplicity can be folded into
/// one count.
pub(super) fn has_o1_absorber(tdd: &Tdd, pv: VtreeIdx) -> bool {
    let (lv, rv) = tdd.vtree.children(pv);
    tdd.levels[lv.idx()].is_marginal() || tdd.levels[rv.idx()].is_marginal()
}

/// One pair with exactly one side scaled by k, plus which side (if any) now
/// carries an inline marginal ref — the caller must raise that side's
/// `MARGINAL_INLINED_*` marker on the level it writes the pair into, or the apply
/// reader misdecodes the bit-30-tagged count as a grid coordinate.
pub(super) struct ScaledPair {
    pub(super) pair: ChildPair,
    pub(super) inlined: Option<ChildSide>,
}

/// Scale exactly one side of the pair `(l, r)` held at level `pv` by `k`.
/// `None` = neither side is an O(1) absorber, so the caller keeps the run.
///
/// Either side would be sound, `k·(L⊗R) = L⊗(k·R) = (k·L)⊗R`, and so would not
/// scaling at all; only a marginal child is cheap, since the factor multiplies
/// one count (the cost policy in the module doc of `duplicate_pair_resolve`).
/// So only marginal sides are considered, the right one first when both are.
pub(super) fn scale_pair_one_side(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    l: u32,
    r: u32,
    k: u32,
) -> Option<Result<ScaledPair, OperationError>> {
    let (lv, rv) = tdd.vtree.children(pv);
    // Right first unless the left side is the only O(1) absorber.
    let left_first =
        tdd.levels[lv.idx()].is_marginal() && !tdd.levels[rv.idx()].is_marginal();
    let order = if left_first {
        [ChildSide::Left, ChildSide::Right]
    } else {
        [ChildSide::Right, ChildSide::Left]
    };
    for side in order {
        let (cv, raw) = match side {
            ChildSide::Left => (lv, l),
            ChildSide::Right => (rv, r),
        };
        // Cost policy: a structural side would cost a scaled subtree clone.
        if !tdd.levels[cv.idx()].is_marginal() {
            continue;
        }
        let Some(res) = try_scale_child(eng, tdd, cv, raw, k) else {
            continue; // even this marginal side cannot absorb — try the other
        };
        let new_raw = match res {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let inlined = if tdd.levels[cv.idx()].is_marginal()
            && matches!(ValueRef::from_raw(MarginalSide(new_raw)), ValueRef::Inline(_))
        {
            Some(side)
        } else {
            None
        };
        let pair = match side {
            ChildSide::Left => ChildPair { left: NodeIdx(new_raw), right: NodeIdx(r) },
            ChildSide::Right => ChildPair { left: NodeIdx(l), right: NodeIdx(new_raw) },
        };
        return Some(Ok(ScaledPair { pair, inlined }));
    }
    None
}
