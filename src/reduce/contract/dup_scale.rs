//! Multiplying a duplicate pair's marginal side by its run length.

use crate::engine::Engine;
use crate::diagram::{ValueRef, NodeIdx};
use crate::diagram::MargSide;
use num_bigint::BigUint;

use crate::error::ApplyError;
use crate::marg_slots::{push_count_key, ChildSide, CountKey};
use crate::diagram::*;
use crate::vtree::VtreeIdx;

/// Append a new count slot to marginal level `mv`, mirroring p-fusion's mint
/// path (u128 with `u128::MAX` overflow sentinel + BigUint side table). No
/// count-keyed interning here — slot-prune value-merge dedups equal values on
/// the next prune pass.
fn push_count_slot(eng: &Engine, tdd: &mut Tdd, mv: VtreeIdx, val: CountKey) -> Result<u32, ApplyError> {
    // A minted slot index is only meaningful at an INTERNAL marginal level: the
    // production decoder (`marginal::store::read_marginal_count`)
    // reads a bare marg-side ref at a LEAF as a leaf-LABEL (fixed count), never
    // indexing the store — so a slot minted here into a leaf store would be
    // silently re-decoded as a label (slot 0 → label One), miscounting. The
    // integer-leaf scale path (`try_scale_child`) intercepts leaves before they
    // reach here, so this invariant is now enforceable.
    debug_assert!(
        !tdd.vtree.node(mv).is_leaf(),
        "push_count_slot: refusing to mint a slot into a LEAF marginal store — \
         integer leaf marg refs are labels, not slots (see try_scale_child leaf \
         branch / read_marginal_count)"
    );
    let level = &mut tdd.levels[mv.idx()];
    let counts = level
        .marginal_counts
        .as_mut()
        .expect("push_count_slot: level is not marginal");
    let new_idx = push_count_key(eng, counts, &mut level.marginal_counts_big, &val)?;
    Ok(ValueRef::slot_raw(new_idx))
}

/// Scale a marg-side ref (into marginal level `mv`) by k: count ×= k.
/// Inline result if it fits, otherwise a fresh slot.
fn scale_marg_ref(eng: &Engine, tdd: &mut Tdd, mv: VtreeIdx, raw: u32, k: u32) -> Result<u32, ApplyError> {
    debug_assert!(k >= 2);
    // Weighted mode: the value lives in the external WeightStore (not
    // `marginal_counts`), so scale the BigRational by k and mint a fresh slot.
    // Multiplicity fold for the content-twin merge: k twins of value V → one slot k·V.
    if tdd.levels[mv.idx()].is_weight_marginal() {
        return scale_weight_ref(tdd, mv, raw, k);
    }
    match ValueRef::from_raw(MargSide(raw)) {
        ValueRef::Inline(c) => {
            // c ≤ 2^30−1, k ≤ 2^32−1 → product fits u128 with room to spare.
            let scaled = c as u128 * k as u128;
            if let Some(r) = ValueRef::inline_raw(scaled) {
                return Ok(r);
            }
            push_count_slot(eng, tdd, mv, CountKey::Small(scaled))
        }
        ValueRef::Slot(s) => {
            let level = &tdd.levels[mv.idx()];
            let counts = level
                .marginal_counts
                .as_ref()
                .expect("scale_marg_ref: slot ref into non-marginal level");
            let c = counts[s as usize];
            if c == u128::MAX {
                // Overflow sentinel: true value lives in the big side table.
                let b = level
                    .marginal_counts_big
                    .as_ref()
                    .and_then(|v| v.get(s as usize))
                    .expect("scale_marg_ref: overflow sentinel without big entry")
                    .clone();
                return push_count_slot(eng, tdd, mv, CountKey::Big(b * k));
            }
            match c.checked_mul(k as u128) {
                Some(v) if v != u128::MAX => {
                    if let Some(r) = ValueRef::inline_raw(v) {
                        Ok(r)
                    } else {
                        push_count_slot(eng, tdd, mv, CountKey::Small(v))
                    }
                }
                _ => push_count_slot(eng, tdd, mv, CountKey::Big(BigUint::from(c) * k)),
            }
        }
    }
}

/// `k · v` in the `WeightStore`'s active mode — the ONE place a multiplicity
/// becomes a weighted factor, shared by `scale_weight_ref`'s mint and
/// `scale_weight_leaf_by_lookup`'s pinned-column lookup. Building `k` as a
/// same-mode `WeightVal` keeps the scale a same-variant `WeightVal::mul`.
fn scaled_weight(
    ws: &crate::weight_store::WeightStore,
    v: &crate::query::WeightVal,
    k: u32,
) -> crate::query::WeightVal {
    use crate::query::{SignedLog, WeightVal};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let k_w = if ws.is_log() {
        WeightVal::Log(SignedLog::from_rational(&BigRational::from_integer(BigInt::from(k))))
    } else {
        // A `u32` multiplicity is always in the small exact representation.
        WeightVal::ExactSmall(i128::from(k))
    };
    v.mul(&k_w)
}

/// Weighted analogue of `scale_marg_ref`: the marginal value lives in the
/// external `WeightStore`, so multiply slot `s`'s `BigRational` by k and append a
/// fresh slot. Bumps `weight_width` (the weighted level's live slot count,
/// what `width()` reads) to cover the new slot. The slot-prune value-merge dedups
/// equal-valued slots on the next pass.
fn scale_weight_ref(tdd: &mut Tdd, mv: VtreeIdx, raw: u32, k: u32) -> Result<u32, ApplyError> {
    use crate::query::WeightVal;

    // Weighted marg-side refs reaching here are always Slot — nothing mints a
    // weighted `Inline` — and the arm below only holds the match exhaustive.
    // ZERO sentinels carry no value and are not scaled here.
    match ValueRef::from_raw(MargSide(raw)) {
        ValueRef::Slot(s) => {
            let s = s as usize;
            let ws = tdd
                .weights
                .as_mut()
                .expect("scale_weight_ref: diagram carries no weight store");
            let scaled: WeightVal = {
                let vals = ws
                    .level(mv.idx())
                    .expect("scale_weight_ref: weighted level has no store");
                scaled_weight(ws, &vals[s], k)
            };
            let new_idx = ws.push_value(mv.idx(), scaled);
            // Keep the level's live slot count in sync with the store length.
            tdd.levels[mv.idx()].weight_width = (new_idx + 1) as u32;
            Ok(ValueRef::slot_raw(new_idx as u32))
        }
        ValueRef::Inline(_) => {
            unreachable!(
                "weighted Inline marg refs are never minted (since a1c7876e08); \
                 an Inline ref here would dangle across component graft (store \
                 rebuild drops the intern table)"
            )
        }
    }
}

/// Scale an INTEGER-marginal LEAF ref by `k` without touching the (empty) leaf
/// store. A bare `Slot(s)` ref is a leaf-LABEL index — decoded with the SAME
/// fixed-count mapping as `marginal::store::read_marginal_count`;
/// an `Inline(c)` ref carries the count directly. Returns the scaled value as an
/// inline ref (`Some(Ok(..))`), or `None` when the scaled value cannot inline:
/// the leaf side cannot absorb the factor, and we must NEVER mint a slot into a
/// leaf store, so the caller routes the factor to the other side / bails.
fn scale_leaf_marg_label(raw: u32, k: u32) -> Option<Result<u32, ApplyError>> {
    let base: u128 = match ValueRef::from_raw(MargSide(raw)) {
        ValueRef::Inline(c) => c as u128,
        // Same fixed-count mapping as `read_marginal_count`: a bare leaf ref is a
        // `LeafLabel` index (Zero→0, One→2, Pos|Neg→1), not a store slot.
        ValueRef::Slot(s) => match LeafLabel::from_idx(s as usize) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        },
    };
    // base ≤ 2 (label) or ≤ MARG_INLINE_MAX (inline), k ≤ 2^32−1 ⇒ product fits u128.
    let scaled = base * k as u128;
    // `None` (can't inline) ⇒ this side can't absorb — never mint into a leaf store.
    ValueRef::inline_raw(scaled).map(Ok)
}

/// Scale a ref into a WEIGHT-marginal LEAF by `k` **without minting**: compute
/// `k · column[slot(raw)]` and look that value up among the pinned column's own
/// three slots (`marginalize::find_leaf_slot_by_value`, the shared pinned-column
/// value search), returning the slot ref if it is there and `None` if it is not.
///
/// This is the weighted counterpart of `scale_leaf_marg_label`'s inline absorb.
/// The integer arm can encode any scaled count in the ref itself; a weighted
/// `ValueRef::Inline(gidx)` indexes the store's GLOBAL intern table (rebuilt at
/// every component graft), so the only representable results here are the column's
/// existing values — hence a lookup, not an encode.
///
/// It is not a narrow special case. After
/// `marginalize::canonicalize_leaf_refs_at_parent` the duplicate runs that reach
/// this path are the equal-value ones, and for `w⁺ = w⁻` the arithmetic lands
/// exactly on the column: `2·Pos = 2w⁺ = w⁺+w⁻ = One`. That is the integer arm's
/// `Inline(2)` fold, reached with no new slot and no column write — the pin
/// (`marginalize::marginalize_leaf_weighted`, THE PIN INVARIANT) stands untouched.
///
/// Exact domain only: in log mode `weight_key` equality compares `f64` bit
/// patterns, so a "hit" would be a rounding coincidence rather than a value
/// identity, and we decline.
fn scale_weight_leaf_by_lookup(
    ws: &crate::weight_store::WeightStore,
    cv: VtreeIdx,
    raw: u32,
    k: u32,
) -> Option<u32> {
    use crate::marginal::find_leaf_slot_by_value;
    debug_assert!(k >= 2);
    // ZERO sentinel (bit 31) names no slot. A weighted leaf side carries no
    // inline (bit-30) ref either — nothing mints one — so both decline rather
    // than being decoded as a column index.
    if raw & (1u32 << 31) != 0 || raw & MARG_OVERFLOW_TAG != 0 {
        return None;
    }
    let slot = raw as usize;
    {
        if ws.is_log() {
            return None;
        }
        let base = ws.level(cv.idx())?.get(slot)?;
        let want = scaled_weight(ws, base, k);
        // The shared search scans ASCENDING, so the hit is the CANONICAL slot for
        // that value (`marginalize::leaf_canon_map`'s min-index rule). Folding
        // onto anything else would re-introduce exactly the non-canonical ref the
        // canon pass exists to remove.
        find_leaf_slot_by_value(ws, cv.idx(), &want).map(ValueRef::slot_raw)
    }
}

/// Try to scale the O(1)-absorbing side of a pair: the ref `raw` into the
/// MARGINAL child level `cv`. Returns `None` when this side declines the factor —
/// an integer-marginal leaf label whose scaled value will not inline, or a
/// weight-marginal leaf whose scaled value is not one the pinned column already
/// carries (see below). Only marginal `cv` reaches here — see the cost policy in
/// the module doc and in `scale_pair_one_side`.
fn try_scale_child(
    eng: &Engine,
    tdd: &mut Tdd,
    cv: VtreeIdx,
    raw: u32,
    k: u32,
) -> Option<Result<u32, ApplyError>> {
    debug_assert!(
        tdd.levels[cv.idx()].is_marginal(),
        "try_scale_child: only a marginal child is an O(1) absorber",
    );
    {
        // Integer-marginal LEAF: the store is EMPTY (all counts live inline at the
        // parent), so a bare marg-side ref here is a leaf-LABEL, not a store index
        // — exactly how the production decoder reads it
        // (`marginal::store::read_marginal_count`). Scaling must not index the (empty) store
        // and must not `push_count_slot` a fresh slot: a minted slot index would be
        // re-decoded as a leaf label (slot 0 → label One), silently miscounting,
        // and indexing the empty store panics (OOB). Scale the decoded label
        // directly, inline the result, or return `None` (the leaf side cannot
        // absorb) so the OTHER side takes the factor / the no-absorbing-side path
        // bails soundly.
        //
        // WEIGHT-marginal leaves absorb the factor only by LOOKUP, never by mint.
        // Their `WeightStore` column is NOT ordinary per-level slot storage: it is
        // the pinned, shared, label-ordered 3-slot `leaf_val` cache
        // (`marginalize_leaf_weighted`), and a bare leaf-side ref is a leaf LABEL
        // that aliases a slot by position. Appending a scaled 4th slot would (a)
        // mint a `Slot(3)` ref that `Tdd::effective_width` — which hardcodes
        // LEAF_WIDTH for leaf levels — sizes no remap window for, so
        // `prune_unreachable` would index the NEIGHBOURING level's remap region,
        // and (b) break the pin for every other `Tdd` sharing the column. Nor is
        // there a weighted analogue of `scale_leaf_marg_label`'s inline absorb: a
        // an inline payload is an integer count, which a weighted value has no
        // encoding for (same reason `scale_weight_ref` refuses to mint one).
        //
        // What IS available — and is the whole point of the equal-value ref
        // canonicalization the leaf-marg pass now runs — is the column itself:
        // `scale_weight_leaf_by_lookup` takes the scaled value's slot when the
        // column already carries it (`2·Pos = w⁺+w⁻ = One` whenever w⁺ = w⁻, the
        // ~common case). Nothing is minted and the column is not written, so the
        // pin is untouched. DECLINING is now the FALLBACK, for a scaled value the
        // column does not hold (asymmetric weights, k ≥ 3, log mode) — and it stays
        // free of correctness cost: the caller keeps the duplicate run as `k` legal
        // multiset terms (pair lists are multisets).
        //
        // Both outcomes are decided HERE, before any structural mutation —
        // `scale_pair_one_side` only tries the other side on `None`, and
        // `resolve_duplicate_pairs_in_node` re-emits the untouched run.
        if tdd.vtree.node(cv).is_leaf() {
            if tdd.levels[cv.idx()].is_weight_marginal() {
                let ws = tdd
                    .weights
                    .as_ref()
                    .expect("weighted leaf scale: diagram carries no weight store");
                return scale_weight_leaf_by_lookup(ws, cv, raw, k).map(Ok);
            }
            return scale_leaf_marg_label(raw, k);
        }
        Some(scale_marg_ref(eng, tdd, cv, raw, k))
    }
}

/// True when a duplicate run at plain level `pv` has an O(1) absorber: one of
/// `pv`'s children is a marginal level, so the multiplicity can be folded into
/// one count. Single source of truth for the cost policy's level test — read by
/// `resolve_duplicate_pairs_in_node`'s early-out and by `scale_pair_one_side`'s
/// side choice (which re-derives it per side).
pub(super) fn has_o1_absorber(tdd: &Tdd, pv: VtreeIdx) -> bool {
    let (lv, rv) = tdd.vtree.children(pv);
    tdd.levels[lv.idx()].is_marginal() || tdd.levels[rv.idx()].is_marginal()
}

/// One pair with exactly one side scaled by k, plus which side (if any) now
/// carries an INLINE marg ref — the caller must raise that side's
/// `MARG_INLINED_*` marker on the level it writes the pair into, or the apply
/// reader misdecodes the bit-30-tagged count as a grid coordinate.
pub(super) struct ScaledPair {
    pub(super) pair: InputPair,
    pub(super) inlined: Option<ChildSide>,
}

/// Scale exactly one side of the pair `(l, r)` held at level `pv` by `k`.
/// `None` = neither side is an O(1) absorber, so the caller keeps the run.
///
/// Single source of truth for "which side absorbs the factor" — the one entry
/// `resolve_duplicate_pairs_in_node` uses to fold a duplicate run.
///
/// ## Side choice is a cost decision, not a correctness one
///
/// Either side would be sound — a pair is an independent additive term,
/// `k·(L⊗R) = L⊗(k·R) = (k·L)⊗R` — and so is not scaling at all (k copies of
/// `(L,R)` already sum to `k·c(L)·c(R)`). What differs is cost, and only a
/// MARGINAL child is cheap: the factor multiplies one count. A structural child
/// would have to be cloned and re-scaled all the way down to the nearest count,
/// which grows the diagram; that descent is not taken (see the module doc).
///
/// So: consider only marginal sides. The historical order (right, then left) is
/// kept when both are marginal, so diagrams where the choice is a tie are
/// rewritten exactly as before.
pub(super) fn scale_pair_one_side(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    l: u32,
    r: u32,
    k: u32,
) -> Option<Result<ScaledPair, ApplyError>> {
    let (lv, rv) = tdd.vtree.children(pv);
    // Right first unless the LEFT side is the only O(1) absorber.
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
            && matches!(ValueRef::from_raw(MargSide(new_raw)), ValueRef::Inline(_))
        {
            Some(side)
        } else {
            None
        };
        let pair = match side {
            ChildSide::Left => InputPair { left: NodeIdx(new_raw), right: NodeIdx(r) },
            ChildSide::Right => InputPair { left: NodeIdx(l), right: NodeIdx(new_raw) },
        };
        return Some(Ok(ScaledPair { pair, inlined }));
    }
    None
}
