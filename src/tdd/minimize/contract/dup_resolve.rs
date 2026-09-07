//! Duplicate-pair resolution by multiplicity fork-down ("scale" rewrite).
//!
//! Twin contraction above a marginal boundary can produce duplicate `(L, R)`
//! entries in a plain (`marg_flags == 0`) node's pair list: merging content-
//! equal or partially-overlapping context-twins folds k disjoint upstream
//! plan families onto the same structural pair, and at plain levels there is
//! no count field to carry the multiplicity k.
//!
//! Leaving the duplicates in place is not *count*-wrong — the count recurrences
//! sum over a node's pairs, so k copies of `(L, R)` already contribute
//! `k·c(L)·c(R)` (maintainer ruling 2026-07-27), and since the
//! content-twin merge was widened to every explicit level the whole plain-level
//! machinery is stated for multisets (`contract/merge.rs`'s duplicate check and
//! `conjoin/sparse.rs`'s Phase F check are both diagram-scoped now). So the
//! rewrite below is a SIZE optimization — k pair slots become one — never a
//! correctness obligation.
//!
//! Resolution (user-ratified): replace the k copies of
//! `(L, R)` with a single pair whose marg-carrying side is *scaled by k* — a
//! fresh value denoting k times the original's. Scaling a marg-side ref
//! multiplies one count (inline re-encode, or one fresh slot) — except at a
//! weight-marginal LEAF, whose 3-slot column is pinned and admits no mint: there
//! the fold succeeds only when the scaled value is one the column already carries
//! (`scale_weight_leaf_by_lookup`), which after equal-value ref canonicalization
//! is the common `2·Pos = One` case.
//!
//! ## Cost policy: only an O(1) absorber is taken (2026-07-27)
//!
//! A pair has an absorbing side only where marginalization lies below it, and
//! the two kinds of absorber are not comparable:
//!
//! * a **marginal child** absorbs in O(1) — the factor multiplies one count;
//! * a **structural child** would have to be CLONED with one of *its* marg-side
//!   children scaled, recursing down the vtree until some count absorbs the
//!   factor — minting a scaled copy of every node on the way down.
//!
//! The structural descent used to run whenever no marginal side was available.
//! Measured on `mc2024_track1_062` it entered ~43 M times and minted
//! ~6.3 M cloned nodes in 11 s — trading ~17 M pair slots
//! for ~6.3 M new nodes plus the work every later pass then does on the bigger
//! diagram — and it accounted for ~37–41 % of the compile's self time. It never
//! paid: it *grows* the diagram to shrink pair lists.
//!
//! So the resolution now takes the O(1) absorber and nothing else. When neither
//! child of the plain level is marginal, the duplicate run is left in place as k
//! legal multiset terms — the twin merge that produced it still stands (it is the
//! merge that removed k−1 NODES), only its pair-list representation is left
//! un-collapsed. `resolve_duplicate_pairs_in_node` early-outs on that level
//! shape, so a level with no O(1) absorber pays nothing at all.
//!
//! Scaled counts are fresh slots; slot-prune value-merge later shares them with
//! existing equal-valued slots (C3 is restored by that pass, not by construction
//! here).

use num_bigint::BigUint;

use super::scratch::DupScratch;
use crate::tdd::marg_slots::ChildSide;
use crate::tdd::transform::pairwise::conjoin::ApplyError;
use crate::tdd::transform::pairwise::conjoin::try_push;
use crate::tdd::types::*;
use crate::tdd::marg_slots::{push_count_key, CountKey};
use crate::vtree::VtreeIdx;

/// `has_marg_below[v]` — v's vtree subtree (including v itself) contains a
/// marginal level. Stable across one minimize pass: marginalization converts
/// levels between compile phases, never during contraction. Fills `below` in
/// place (buffer reused across contract runs via ContractScratch).
///
/// O(vtree nodes), so the caller fills it lazily — on the first merge a sweep
/// attempts, never on a sweep that finds no twins. See
/// `ContractScratch::has_marg_below_valid`.
pub(crate) fn compute_has_marg_below_into(tdd: &Tdd, below: &mut Vec<bool>) {
    let n = tdd.vtree.num_nodes();
    below.clear();
    below.resize(n, false);
    for i in 0..n.min(tdd.levels.len()) {
        if tdd.levels[i].is_marginal() {
            let mut v = Some(VtreeIdx(i as u32));
            while let Some(x) = v {
                if below[x.idx()] {
                    break; // ancestors above already marked by an earlier walk
                }
                below[x.idx()] = true;
                v = tdd.vtree.node(x).parent();
            }
        }
    }
}


/// Append a new count slot to marginal level `mv`, mirroring p-fusion's mint
/// path (u128 with `u128::MAX` overflow sentinel + BigUint side table). No
/// count-keyed interning here — slot-prune value-merge dedups equal values on
/// the next prune pass.
fn push_count_slot(tdd: &mut Tdd, mv: VtreeIdx, val: CountKey) -> Result<u32, ApplyError> {
    // A minted slot index is only meaningful at an INTERNAL marginal level: the
    // production decoder (`read_marginal_count`, compile_marginalize.rs ~1441)
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
    let new_idx = push_count_key(counts, &mut level.marginal_counts_big, &val)?;
    Ok(MargRef::slot_raw(new_idx))
}

/// Scale a marg-side ref (into marginal level `mv`) by k: count ×= k.
/// Inline result if it fits, otherwise a fresh slot.
fn scale_marg_ref(tdd: &mut Tdd, mv: VtreeIdx, raw: u32, k: u32) -> Result<u32, ApplyError> {
    debug_assert!(k >= 2);
    // Weighted mode: the value lives in the external WeightStore (not
    // `marginal_counts`), so scale the BigRational by k and mint a fresh slot.
    // Multiplicity fold for the C2 twin merge: k twins of value V → one slot k·V.
    if tdd.levels[mv.idx()].is_weight_marginal() {
        return scale_weight_ref(tdd, mv, raw, k);
    }
    match MargRef::from_raw(raw) {
        MargRef::Inline(c) => {
            // c ≤ 2^30−1, k ≤ 2^32−1 → product fits u128 with room to spare.
            let scaled = c as u128 * k as u128;
            if let Some(r) = MargRef::inline_raw(scaled) {
                return Ok(r);
            }
            push_count_slot(tdd, mv, CountKey::Small(scaled))
        }
        MargRef::Slot(s) => {
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
                return push_count_slot(tdd, mv, CountKey::Big(b * k));
            }
            match c.checked_mul(k as u128) {
                Some(v) if v != u128::MAX => {
                    if let Some(r) = MargRef::inline_raw(v) {
                        Ok(r)
                    } else {
                        push_count_slot(tdd, mv, CountKey::Small(v))
                    }
                }
                _ => push_count_slot(tdd, mv, CountKey::Big(BigUint::from(c) * k)),
            }
        }
    }
}

/// `k · v` in the `WeightStore`'s active mode — the ONE place a multiplicity
/// becomes a weighted factor, shared by `scale_weight_ref`'s mint and
/// `scale_weight_leaf_by_lookup`'s pinned-column lookup. Building `k` as a
/// same-mode `WeightVal` keeps the scale a same-variant `WeightVal::mul`.
fn scaled_weight(
    ws: &crate::tdd::weight_store::WeightStore,
    v: &crate::tdd::query::semiring::WeightVal,
    k: u32,
) -> crate::tdd::query::semiring::WeightVal {
    use crate::tdd::query::semiring::{SignedLog, WeightVal};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let k_w = if ws.log_mode {
        WeightVal::Log(SignedLog::from_rational(&BigRational::from_integer(BigInt::from(k))))
    } else {
        // A `u32` multiplicity is always in the small exact representation.
        WeightVal::ExactSmall(i128::from(k))
    };
    v.mul(&k_w)
}

/// Weighted analogue of `scale_marg_ref`: the marginal value lives in the
/// external `WeightStore`, so multiply slot `s`'s `BigRational` by k and append a
/// fresh slot. Bumps `retired_marg_width` (the weighted level's live slot count,
/// what `width()` reads) to cover the new slot. The slot-prune value-merge dedups
/// equal-valued slots on the next pass.
fn scale_weight_ref(tdd: &mut Tdd, mv: VtreeIdx, raw: u32, k: u32) -> Result<u32, ApplyError> {
    use crate::tdd::transform::unary::marginalize::{with_weight_ctx, with_weight_ctx_mut};
    use crate::tdd::query::semiring::WeightVal;

    // Weighted marg-side refs reaching here are always Slot. A weighted `Inline`
    // ref is never minted (since a1c7876e08 — `allocate_fusion_slots_weighted` in
    // p_fusion.rs emits Slot only, and this function's own `Inline` arm below is
    // the sole production caller of `WeightStore::intern`, so it can't be the
    // first mint); the arm is kept only to hold the match exhaustive and fail
    // fast if that ever changes. ZERO sentinels carry no value and are not scaled
    // here.
    match MargRef::from_raw(raw) {
        MargRef::Slot(s) => {
            let s = s as usize;
            let scaled: WeightVal = with_weight_ctx(|ws| {
                let vals = ws
                    .level(mv.idx())
                    .expect("scale_weight_ref: weighted level has no store");
                scaled_weight(ws, &vals[s], k)
            });
            let new_idx = with_weight_ctx_mut(|ws| ws.push_value(mv.idx(), scaled));
            // Keep the level's live slot count in sync with the store length.
            tdd.levels[mv.idx()].retired_marg_width = (new_idx + 1) as u32;
            Ok(MargRef::slot_raw(new_idx as u32))
        }
        MargRef::Inline(_) => {
            // Unreachable in practice (see comment above): an Inline ref persisted
            // in a pair list would dangle across the next component graft, which
            // rebuilds the WeightStore and drops its intern table.
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
/// fixed-count mapping as `read_marginal_count` (compile_marginalize.rs ~1441);
/// an `Inline(c)` ref carries the count directly. Returns the scaled value as an
/// inline ref (`Some(Ok(..))`), or `None` when the scaled value cannot inline:
/// the leaf side cannot absorb the factor, and we must NEVER mint a slot into a
/// leaf store, so the caller routes the factor to the other side / bails.
fn scale_leaf_marg_label(raw: u32, k: u32) -> Option<Result<u32, ApplyError>> {
    let base: u128 = match MargRef::from_raw(raw) {
        MargRef::Inline(c) => c as u128,
        // Same fixed-count mapping as `read_marginal_count`: a bare leaf ref is a
        // `LeafLabel` index (Zero→0, One→2, Pos|Neg→1), not a store slot.
        MargRef::Slot(s) => match LeafLabel::from_idx(s as usize) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        },
    };
    // base ≤ 2 (label) or ≤ MARG_INLINE_MAX (inline), k ≤ 2^32−1 ⇒ product fits u128.
    let scaled = base * k as u128;
    // `None` (can't inline) ⇒ this side can't absorb — never mint into a leaf store.
    MargRef::inline_raw(scaled).map(Ok)
}

/// Scale a ref into a WEIGHT-marginal LEAF by `k` **without minting**: compute
/// `k · column[slot(raw)]` and look that value up among the pinned column's own
/// three slots (`marginalize::find_leaf_slot_by_value`, the shared pinned-column
/// value search), returning the slot ref if it is there and `None` if it is not.
///
/// This is the weighted counterpart of `scale_leaf_marg_label`'s inline absorb.
/// The integer arm can encode any scaled count in the ref itself; a weighted
/// `MargRef::Inline(gidx)` indexes the store's GLOBAL intern table (rebuilt at
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
fn scale_weight_leaf_by_lookup(cv: VtreeIdx, raw: u32, k: u32) -> Option<u32> {
    use crate::tdd::transform::unary::marginalize::{find_leaf_slot_by_value, with_weight_ctx};
    debug_assert!(k >= 2);
    // ZERO sentinel (bit 31) names no slot. A weighted leaf side carries no
    // inline (bit-30) ref either — nothing mints one — so both decline rather
    // than being decoded as a column index.
    if raw & (1u32 << 31) != 0 || raw & MARG_OVERFLOW_TAG != 0 {
        return None;
    }
    let slot = raw as usize;
    with_weight_ctx(|ws| {
        if ws.log_mode {
            return None;
        }
        let base = ws.level(cv.idx())?.get(slot)?;
        let want = scaled_weight(ws, base, k);
        // The shared search scans ASCENDING, so the hit is the CANONICAL slot for
        // that value (`marginalize::leaf_canon_map`'s min-index rule). Folding
        // onto anything else would re-introduce exactly the non-canonical ref the
        // canon pass exists to remove.
        find_leaf_slot_by_value(ws, cv.idx(), &want).map(MargRef::slot_raw)
    })
}

/// Try to scale the O(1)-absorbing side of a pair: the ref `raw` into the
/// MARGINAL child level `cv`. Returns `None` when this side declines the factor —
/// an integer-marginal leaf label whose scaled value will not inline, or a
/// weight-marginal leaf whose scaled value is not one the pinned column already
/// carries (see below). Only marginal `cv` reaches here — see the cost policy in
/// the module doc and in `scale_pair_one_side`.
fn try_scale_child(
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
        // — exactly how the production decoder reads it (`read_marginal_count`,
        // compile_marginalize.rs ~1441). Scaling must not index the (empty) store
        // and must not `push_count_slot` a fresh slot: a minted slot index would be
        // re-decoded as a leaf label (slot 0 → label One), silently miscounting,
        // and indexing the empty store panics (OOB). Scale the decoded label
        // directly, inline the result, or return `None` (the leaf side cannot
        // absorb) so the OTHER side takes the factor / the no-absorbing-side path
        // bails soundly.
        //
        // WEIGHT-marginal leaves absorb the factor only by LOOKUP, never by mint.
        // Their `WeightStore` column is NOT ordinary per-level slot storage: it is
        // the pinned, compile-global, label-ordered 3-slot `leaf_val` cache
        // (`marginalize_leaf_weighted`), and a bare leaf-side ref is a leaf LABEL
        // that aliases a slot by position. Appending a scaled 4th slot would (a)
        // mint a `Slot(3)` ref that `Tdd::effective_width` — which hardcodes
        // LEAF_WIDTH for leaf levels — sizes no remap window for, so
        // `prune_unreachable` would index the NEIGHBOURING level's remap region,
        // and (b) break the pin for every other `Tdd` sharing the column. Nor is
        // there a weighted analogue of `scale_leaf_marg_label`'s inline absorb: a
        // weighted `MargRef::Inline` indexes the store's GLOBAL intern table, which
        // is rebuilt and re-indexed at every component graft, so an inline ref
        // persisted in a pair list dangles (same reason `scale_weight_ref` refuses
        // to mint one).
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
                return scale_weight_leaf_by_lookup(cv, raw, k).map(Ok);
            }
            return scale_leaf_marg_label(raw, k);
        }
        Some(scale_marg_ref(tdd, cv, raw, k))
    }
}

/// True when a duplicate run at plain level `pv` has an O(1) absorber: one of
/// `pv`'s children is a marginal level, so the multiplicity can be folded into
/// one count. Single source of truth for the cost policy's level test — read by
/// `resolve_duplicate_pairs_in_node`'s early-out and by `scale_pair_one_side`'s
/// side choice (which re-derives it per side).
fn has_o1_absorber(tdd: &Tdd, pv: VtreeIdx) -> bool {
    let (lv, rv) = tdd.vtree.children(pv);
    tdd.levels[lv.idx()].is_marginal() || tdd.levels[rv.idx()].is_marginal()
}

/// One pair with exactly one side scaled by k, plus which side (if any) now
/// carries an INLINE marg ref — the caller must raise that side's
/// `MARG_INLINED_*` marker on the level it writes the pair into, or the apply
/// reader misdecodes the bit-30-tagged count as a grid coordinate (marg-canon
/// #63).
struct ScaledPair {
    pair: InputPair,
    inlined: Option<ChildSide>,
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
/// which grows the diagram; that descent is not taken (module doc, round 9).
///
/// So: consider only marginal sides. The historical order (right, then left) is
/// kept when both are marginal, so diagrams where the choice is a tie are
/// rewritten exactly as before.
fn scale_pair_one_side(
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
        let Some(res) = try_scale_child(tdd, cv, raw, k) else {
            continue; // even this marginal side cannot absorb — try the other
        };
        let new_raw = match res {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let inlined = if tdd.levels[cv.idx()].is_marginal()
            && matches!(MargRef::from_raw(new_raw), MargRef::Inline(_))
        {
            Some(side)
        } else {
            None
        };
        let pair = match side {
            ChildSide::Left => InputPair { left: LocalNodeIdx(new_raw), right: LocalNodeIdx(r) },
            ChildSide::Right => InputPair { left: LocalNodeIdx(l), right: LocalNodeIdx(new_raw) },
        };
        return Some(Ok(ScaledPair { pair, inlined }));
    }
    None
}

/// Resolve duplicate pairs in node `idx` at plain structural level `pv`:
/// each run of k > 1 equal `(L, R)` pairs whose level has an O(1) absorber is
/// replaced by a single pair with the marginal side scaled by k. Returns whether
/// the node changed. No-op (`Ok(false)`) when the pair list is duplicate-free,
/// and — by the cost policy — when neither child of `pv` is marginal, in which
/// case the duplicates stay as legal multiset terms.
///
/// `scratch` is caller-owned and reused across the survivors of one fork-down
/// pass (`merge::compact_and_fork_down`); every buffer is cleared here on
/// entry, so it carries capacity across nodes and nothing else — including out
/// of the `?` bails below, which leave it dirty by design.
pub(super) fn resolve_duplicate_pairs_in_node(
    tdd: &mut Tdd,
    pv: VtreeIdx,
    idx: usize,
    scratch: &mut DupScratch,
) -> Result<bool, ApplyError> {
    // `pv` is a plain (non-marginal) level — marginal levels are p-fusion's
    // domain. Its `marg_flags` are NOT asserted zero: scaling a marginal child
    // ref can mint an INLINE marg ref into `pv`'s pairs, which raises `pv`'s
    // `MARG_INLINED_*` marker (below). The caller resolves several survivors per
    // pass, so the second and later calls legitimately see the marker already up.
    debug_assert!(
        !tdd.levels[pv.idx()].is_marginal(),
        "resolve_duplicate_pairs_in_node: marginal levels are p-fusion's domain"
    );
    // Cost policy early-out: with no marginal child there is no O(1) absorber,
    // so every run would be kept anyway — don't pay the collect + hash-count.
    // This is the whole cost on levels the widened content merge made duplicate-
    // rich (millions of calls over tens of millions of pairs).
    if !has_o1_absorber(tdd, pv) {
        return Ok(false);
    }
    scratch.clear();
    let DupScratch { pairs, counts, out } = scratch;
    pairs.extend(
        tdd.levels[pv.idx()]
            .pairs_of_idx(idx)
            .iter()
            .map(|p| (p.left.0, p.right.0)),
    );
    if pairs.len() < 2 {
        return Ok(false);
    }
    // Pair lists are unordered sets — group duplicates with a
    // hash count in O(p) instead of an O(p·log p) sort. The output multiset
    // {(distinct pair, multiplicity k)} is identical to the former sort+run-length
    // form; `out` is written back in arbitrary (hash) order, which §6 explicitly
    // allows (twin contraction is order-independent), and `try_scale_child(child, k)`
    // is order-independent (distinct pairs scale distinct children). This is the
    // fork-down dup-resolution hot path on contraction-bound instances (e.g.
    // mc2025_051), where the survivor list can grow large.
    //
    // The map is REUSED across the pass's nodes (cleared above), so its table
    // can be wider than a fresh `reserve` would make it and the hash order —
    // hence `out`'s order — need not match a cold call's. That is exactly the
    // freedom §6 grants above; nothing downstream reads a pair list positionally.
    counts.reserve(pairs.len());
    for &p in pairs.iter() {
        *counts.entry(p).or_insert(0) += 1;
    }
    if counts.len() == pairs.len() {
        // No duplicate pairs.
        return Ok(false);
    }

    // Worst case (nothing absorbs) `out` is the input multiset verbatim.
    out.reserve(pairs.len());
    let mut inl_left = false;
    let mut inl_right = false;
    for (&(l, r), &k) in counts.iter() {
        let pair = InputPair { left: LocalNodeIdx(l), right: LocalNodeIdx(r) };
        if k == 1 {
            out.push(pair);
            continue;
        }
        let Some(res) = scale_pair_one_side(tdd, pv, l, r, k) else {
            // The marginal side exists (checked at entry) but declined this
            // particular ref — an integer-marginal leaf label whose scaled value
            // will not inline (minting a slot into that leaf store would be
            // decoded back as a label), or a weight-marginal leaf whose pinned
            // global `leaf_val` column does not already carry the scaled value
            // (and admits no minted slot to hold it). Keep the run as k legal
            // multiset terms.
            for _ in 0..k {
                out.push(pair);
            }
            continue;
        };
        let scaled = res?;
        match scaled.inlined {
            Some(ChildSide::Left) => inl_left = true,
            Some(ChildSide::Right) => inl_right = true,
            None => {}
        }
        out.push(scaled.pair);
    }
    debug_assert!(out.len() <= pairs.len());
    if out.len() == pairs.len() {
        // Nothing absorbed — the pair list is unchanged as a multiset.
        return Ok(false);
    }

    // Write back: overwrite the prefix in place and shrink.
    let level = &mut tdd.levels[pv.idx()];
    // A scaled marg ref may have come back INLINE (bit-30 tagged). Raise the
    // side's marker or the apply reader decodes the tagged count as a grid
    // coordinate (marg-canon #63).
    if inl_left {
        level.marg_flags |= TddLevel::MARG_INLINED_LEFT;
    }
    if inl_right {
        level.marg_flags |= TddLevel::MARG_INLINED_RIGHT;
    }
    if level.nodes[idx].is_inline() {
        unreachable!("inline single-pair node cannot hold duplicates");
    }
    let new_len = out.len();
    // Slots the shrink leaves behind are unreferenced arena; the two `new_len == 1`
    // re-encodes abandon the whole old range (inline word, or a fresh tail slot).
    let abandoned = if new_len == 1 { pairs.len() } else { pairs.len() - new_len };
    if new_len == 1 {
        let surviving = out[0];
        if surviving.can_inline() {
            level.nodes[idx] = TddNodeData::inline(surviving);
        } else {
            // Single pair that can't inline: extended multi with len=1, whose
            // pair is PUSHED as a fresh tail slot rather than aliased in place —
            // `out` is a rewritten pair, not necessarily one already sitting in
            // this node's range. That is why `abandoned` above is the WHOLE old
            // range: the node stops referencing every one of its former slots.
            let pair_start = level.pairs.len();
            try_push(&mut level.pairs, surviving)?;
            let ext_idx = level.ext.len();
            try_push(&mut level.ext, ExtMulti { start: pair_start as u64, len: 1 })?;
            level.nodes[idx] = TddNodeData::multi_extended(ext_idx as u32);
        }
    } else {
        let dst = level.pairs_mut(idx);
        dst[..new_len].copy_from_slice(&out[..]);
        level.set_pair_len(idx, new_len as u32);
    }
    level.note_dead_pairs(abandoned);
    Ok(true)
}
