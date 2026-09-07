//! Marginal-side ref encoding, marg consts, and associated helpers.

use num_bigint::BigUint;
use rustc_hash::FxHashMap;

use crate::tdd::transform::pairwise::conjoin::ApplyError;
use super::level::TddLevel;
use super::tdd::Tdd;

// ── Marginal-side ref encoding (bit-tagged u32) ──────────────────────────────
//
// At marg-boundary parent levels, the marg-side of each pair encodes either an
// inline u30 model count OR a 30-bit overflow slot index into the marg-level's
// `marginal_counts` / `marginal_counts_big` vecs. Bit 31 is reserved (must be 0)
// to preserve the `TddNodeData` single-pair-inline encoding — `right.0 & LEAF_BIT`
// and `left.0 & MULTI_BIT` both occupy bit 31. Bit 30 is the inline/overflow tag.
//
//   bits     | meaning
//   ---------|---------------------------------------------------------------
//   31       | always 0 (LEAF_BIT/MULTI_BIT-safe)
//   30       | tag: 0 = inline count, 1 = overflow slot
//   29..0    | 30-bit payload (count value or slot index, range 0..2^30)
//
// This encoding is context-dependent: bit 30 is interpreted as a tag *only* for
// marg-side reads at boundary levels (callers branch via `t1_is_left` /
// `ChildSide`). At non-boundary levels and on the non-marg side of boundary
// pairs, bit 30 is just a regular index bit (slot range up to 2^31).
//
// The ZERO sentinel (`u32::MAX` = 0xFFFFFFFF) has bit 31 set and so falls
// outside this encoding entirely — invariant guarantees ZERO never appears in
// pair lists, so the collision with the otherwise-unused high bit is harmless.

/// Bit 30 of a pair side whose child level is marginal: clear means the value
/// is an index into the child's `marginal_counts`, set means the low 30 bits
/// are the model count itself. Readers use [`resolve_marg_ref`] instead of
/// testing this bit.
///
/// This "bare-is-slot, tag-the-inline" polarity makes the encoding failure-SAFE
/// and tagging-free on the common path. After a child level is marginalized,
/// slot index ≡ node index in `marginal_counts`, so a parent's child-ref — a
/// bare node index left over from before marginalization — is *already* a valid
/// slot reference. Nothing has to be re-tagged when a child marginalizes
/// (including late, by an ancestor's streaming). Only the optional inline
/// OPTIMISATION (store a small count in the ref itself, saving a heap load) sets
/// bit 30, and it does so explicitly.
///
/// A missed inline-write therefore reads back as a (correct) bare slot index,
/// never a wrong count — the inverse of the old "tag-the-slot" polarity, whose
/// bit-30-clear value was ambiguous (untagged-slot vs inline count) and cost a
/// silent ×N overcount when a slot reached a count-decode before being tagged.
pub const MARG_OVERFLOW_TAG: u32 = 1 << 30;
/// Mask for the 30-bit payload (count value or slot index).
pub const MARG_VALUE_MASK: u32 = MARG_OVERFLOW_TAG - 1;
/// Largest model count a pair side stores inline; larger counts are held in
/// the child's `marginal_counts` and referenced by index.
pub const MARG_INLINE_MAX: u32 = MARG_OVERFLOW_TAG - 1;

/// The encoding of a pair side whose child level is marginal, for writers
/// building such a pair by hand ([`to_raw`](Self::to_raw)). Readers use
/// [`resolve_marg_ref`], which yields [`MargResolved`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum MargRef {
    /// The model count itself, at most [`MARG_INLINE_MAX`].
    Inline(u32),
    /// An index into the child level's `marginal_counts`.
    Slot(u32),
}

impl MargRef {
    /// Decode a pair side whose child level is marginal.
    #[inline(always)]
    pub fn from_raw(r: u32) -> Self {
        debug_assert!(r & (1u32 << 31) == 0, "marg-side ref must have bit 31 unset");
        if r & MARG_OVERFLOW_TAG != 0 {
            MargRef::Inline(r & MARG_VALUE_MASK)
        } else {
            MargRef::Slot(r)
        }
    }

    /// The `u32` to store in the pair side (`LocalNodeIdx(raw)`).
    #[inline(always)]
    pub fn to_raw(self) -> u32 {
        match self {
            MargRef::Inline(c) => {
                debug_assert!(c <= MARG_INLINE_MAX, "inline count overflow: {} > {}", c, MARG_INLINE_MAX);
                c | MARG_OVERFLOW_TAG
            }
            MargRef::Slot(s) => {
                debug_assert!(s & !MARG_VALUE_MASK == 0, "slot index overflow: {} >= {}", s, MARG_OVERFLOW_TAG);
                s
            }
        }
    }

    /// Convenience: encode a slot index as a raw u32 marg-side ref.
    #[inline(always)]
    pub(crate) fn slot_raw(slot_idx: u32) -> u32 {
        MargRef::Slot(slot_idx).to_raw()
    }

    /// Convenience: encode an inline count as a raw u32 marg-side ref.
    /// Returns `None` if the count doesn't fit (caller should allocate a slot).
    #[inline(always)]
    pub(crate) fn inline_raw(count: u128) -> Option<u32> {
        if count <= marg_inline_max() as u128 {
            Some(MargRef::Inline(count as u32).to_raw())
        } else {
            None
        }
    }
}

// ── Overflow side table (sparse) ─────────────────────────────────────────────

/// The exact `BigUint` value of every count slot whose fast `u128` cell holds
/// the `u128::MAX` OVERFLOW sentinel — the overflow half of a marginal count
/// store (`TddLevel::marginal_counts` / `marginal_counts_big`) and of the
/// scratch column that builds one (`counts::CountVec`). A reader needs only
/// [`get`](Self::get).
///
/// **Keyed by slot index, not parallel to the fast column.** Overflow is sparse
/// by construction: a slot lands here only when its model count exceeds
/// `u128::MAX`, i.e. the sub-function has more than 2^128 models, while the
/// store itself can be millions of slots wide. The dense `Vec<Option<BigUint>>`
/// this replaces cost `size_of::<Option<BigUint>>()` (24 B) per *slot* on every
/// marginal level that overflowed even once — the first `Big` write sized the
/// side table to the full column width. Keying by slot makes the cost
/// proportional to the overflow set instead, and makes the no-overflow case
/// free: an empty `BigSide` owns no heap at all.
///
/// Representation: `(slot, value)` pairs sorted by `slot`, strictly ascending,
/// no duplicate slots. Chosen over a hash map because every write path appends
/// at a slot larger than any already stored — `CountVec::set`/`push` fill a
/// column left to right, `marg_slots::push_count_key` and
/// `resolve_swapped_marg_side` mint at the store's end, and the two compaction
/// passes (`dedup_fresh_store`, `slot_prune`'s `IntFold::compact_store`) rebuild
/// by draining this table in ascending order. So insertion is an O(1) amortized
/// push on the common path and an in-place overwrite otherwise; reads
/// binary-search a handful of entries, which beats hashing and keeps the
/// per-entry footprint to one `(u32, BigUint)` with no control bytes or
/// load-factor slack. Slot indices are ≤ 30 bits wherever a parent ref can name
/// them (see `MARG_VALUE_MASK`), so `u32` keys are ample.
///
/// A slot with no entry means "the value fits the fast `u128` lane" — the same
/// convention the dense `None` carried.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BigSide {
    /// Sorted by slot, strictly ascending, slots unique. Every method below
    /// preserves that; nothing outside this module can break it.
    entries: Vec<(u32, BigUint)>,
}

impl BigSide {
    /// Number of slots carrying an exact `BigUint` — NOT the store width.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no slot has overflowed (the common case, and the one that
    /// owns no heap).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Exact value of `slot`, or `None` when its `marginal_counts` cell is
    /// not the `u128::MAX` sentinel.
    #[inline]
    pub fn get(&self, slot: usize) -> Option<&BigUint> {
        let slot = u32::try_from(slot).ok()?;
        match self.entries.binary_search_by_key(&slot, |&(s, _)| s) {
            Ok(pos) => Some(&self.entries[pos].1),
            Err(_) => None,
        }
    }

    /// Store `v` at `slot`, replacing any value already there. The ONE insert
    /// path; the fallible wrapper [`try_insert`](Self::try_insert) reserves
    /// through a budget policy and then calls this.
    #[inline]
    pub(crate) fn insert(&mut self, slot: usize, v: BigUint) {
        let slot = u32::try_from(slot).expect("marginal slot index must fit u32");
        match self.entries.binary_search_by_key(&slot, |&(s, _)| s) {
            Ok(pos) => self.entries[pos].1 = v,
            // Ascending appends (the common path) land at `pos == len`, where
            // `Vec::insert` is a plain push. An out-of-order write only ever
            // arrives from a compaction rekeying a merged slot onto a canonical
            // one, which is an already-present key and so takes the `Ok` arm —
            // no mid-vector shift on any production path.
            Err(pos) => self.entries.insert(pos, (slot, v)),
        }
    }

    /// Budget-tracked [`insert`](Self::insert): charges the one-entry growth
    /// against `R` before committing it, so an over-budget store push surfaces
    /// as the policy's error instead of an infallible allocator abort.
    #[inline]
    pub(crate) fn try_insert<R: crate::tdd::counts::ReservePolicy>(
        &mut self,
        slot: usize,
        v: BigUint,
    ) -> Result<(), R::Err> {
        R::reserve(&mut self.entries, 1)?;
        self.insert(slot, v);
        Ok(())
    }

    /// Bulk twin of [`try_insert`](Self::try_insert): reserve room for
    /// `additional` entries through the same policy, so a caller that has
    /// already begun mutating the store — and therefore must not fail
    /// part-way — can front-load its allocation and then [`insert`](Self::insert)
    /// infallibly. `resolve_swapped_marg_side` is that caller.
    #[inline]
    pub(crate) fn try_reserve<R: crate::tdd::counts::ReservePolicy>(
        &mut self,
        additional: usize,
    ) -> Result<(), R::Err> {
        R::reserve(&mut self.entries, additional)
    }

    /// Remove `slot`'s value and hand it back, so no stale `BigUint` is left
    /// behind under a key whose fast cell no longer holds the sentinel. `None`
    /// when the slot carried no exact value. Point mutation only — a pass that
    /// relocates MANY slots must drain and rebuild ([`IntoIterator`]) instead,
    /// since removing survivors one at a time shifts the tail each time.
    #[inline]
    pub(crate) fn take(&mut self, slot: usize) -> Option<BigUint> {
        let slot = u32::try_from(slot).ok()?;
        match self.entries.binary_search_by_key(&slot, |&(s, _)| s) {
            Ok(pos) => Some(self.entries.remove(pos).1),
            Err(_) => None,
        }
    }

    /// Drop every entry and return the backing allocation to the allocator —
    /// the "this store is dead" path (`slot_prune`'s deep-store clear), which
    /// must leave a zero-footprint table behind.
    #[inline]
    pub(crate) fn clear_and_free(&mut self) {
        self.entries.clear();
        self.entries.shrink_to_fit();
    }

    /// Heap bytes the table itself holds (`BigUint` contents excluded).
    #[cfg(test)]
    #[inline]
    pub(crate) fn bytes(&self) -> u64 {
        (self.entries.capacity() * std::mem::size_of::<(u32, BigUint)>()) as u64
    }

    /// Budget-tracked clone: reserves the entry count exactly before copying,
    /// mirroring [`try_insert`](Self::try_insert)'s accounting discipline.
    /// Test-only since the borrowed-view rewrite removed production column
    /// duplication (sole caller: `CountVec::try_clone`).
    #[cfg(test)]
    pub(crate) fn try_clone<R: crate::tdd::counts::ReservePolicy>(
        &self,
    ) -> Result<Self, R::Err> {
        let mut entries: Vec<(u32, BigUint)> = Vec::new();
        R::reserve_exact(&mut entries, self.entries.len())?;
        entries.extend(self.entries.iter().cloned());
        Ok(BigSide { entries })
    }

}

impl FromIterator<(u32, BigUint)> for BigSide {
    /// Build from `(slot, value)` pairs in any order; later values win for a
    /// repeated slot. Goes through `insert` so the sorted
    /// invariant has exactly one enforcer.
    fn from_iter<I: IntoIterator<Item = (u32, BigUint)>>(iter: I) -> Self {
        let mut out = BigSide::default();
        for (slot, v) in iter {
            out.insert(slot as usize, v);
        }
        out
    }
}

impl IntoIterator for BigSide {
    type Item = (u32, BigUint);
    type IntoIter = std::vec::IntoIter<(u32, BigUint)>;

    /// Consume the table into its `(slot, value)` pairs in ASCENDING slot
    /// order, moving each `BigUint` out (never cloning — one can be megabytes).
    ///
    /// This is how a compaction pass rekeys a table: consume, map each old slot
    /// through the pass's remap, and `collect()` back. Doing it in one drain is
    /// what keeps compaction linear — removing survivors one at a time from the
    /// front instead would memmove the whole tail per entry, which is quadratic
    /// on a level where most slots overflowed (counts above 2^128 are ordinary
    /// on large instances, so that is not a corner case).
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// What a pair side refers to, as decoded by [`resolve_marg_ref`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum MargResolved {
    /// The child's model count itself (at most [`MARG_INLINE_MAX`]); no
    /// child node is referenced.
    Inline(u32),
    /// An index into the child level: into `nodes` when the child is
    /// structural, into `marginal_counts` when it is marginal.
    Index(usize),
}

// Test-only runtime override of the inline-vs-slot count threshold (#63 shrink).
// Lowering it forces small counts onto the tagged-slot path so a *toy* CNF
// exercises the same regime as the giant m139 reproducer. Set to `0` ⇒ every
// count ≥ 1 becomes a slot, faithfully matching m139's all-huge-count levels
// (where every parent ref is a slot). Checked before the production const, like
// the gate overrides. Compiled out of release builds.
#[cfg(any(test, debug_assertions))]
thread_local! {
    static MARG_INLINE_MAX_OVERRIDE: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard restoring the previous inline-max override on drop (panic-safe).
#[cfg(any(test, debug_assertions))]
#[doc(hidden)]
pub struct MargInlineMaxGuard(Option<u32>);

#[cfg(any(test, debug_assertions))]
impl Drop for MargInlineMaxGuard {
    fn drop(&mut self) {
        MARG_INLINE_MAX_OVERRIDE.with(|c| c.set(self.0));
    }
}

/// Force the inline-vs-slot threshold for the lifetime of the returned guard.
/// Test-only. See `MARG_INLINE_MAX_OVERRIDE`.
#[cfg(any(test, debug_assertions))]
#[doc(hidden)]
pub fn set_marg_inline_max(v: u32) -> MargInlineMaxGuard {
    MargInlineMaxGuard(MARG_INLINE_MAX_OVERRIDE.with(|c| c.replace(Some(v))))
}

/// Effective inline-vs-slot threshold: counts `<=` this are referenced inline,
/// counts above become tagged slots. Production: [`MARG_INLINE_MAX`] (full
/// 30-bit range). Tests may lower it via [`set_marg_inline_max`] to reproduce
/// the all-slots regime on a small CNF. Every inline-vs-slot DECISION site funnels
/// through this one accessor so a lowered threshold is applied consistently.
#[inline(always)]
pub(crate) fn marg_inline_max() -> u32 {
    #[cfg(any(test, debug_assertions))]
    if let Some(v) = MARG_INLINE_MAX_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    MARG_INLINE_MAX
}

/// Decode one side of a pair: `raw` is `pair.left.0` or `pair.right.0`, and
/// `child_is_marginal` is `is_marginal()` of the child level on that side.
///
/// For a structural child the value is a plain index and comes back as
/// `Index` unchanged; for a marginal child bit 30 distinguishes an inline
/// count from an index into `marginal_counts`. Every reader of a stored pair
/// goes through this; passing the wrong `child_is_marginal` misreads a count
/// as an index or vice versa.
///
/// ```
/// use tididi::tdd::types::{MargResolved, resolve_marg_ref, MargRef};
/// // A structural child: any value is an index.
/// assert_eq!(resolve_marg_ref(7, false), MargResolved::Index(7));
/// // A marginal child: the same bits are an index...
/// assert_eq!(resolve_marg_ref(MargRef::Slot(7).to_raw(), true), MargResolved::Index(7));
/// // ...or a count, per the tag bit.
/// assert_eq!(resolve_marg_ref(MargRef::Inline(7).to_raw(), true), MargResolved::Inline(7));
/// ```
#[inline(always)]
pub fn resolve_marg_ref(raw: u32, child_is_marginal: bool) -> MargResolved {
    let is_marg = child_is_marginal;
    if is_marg {
        // Polarity flip (bit-30-clear == slot): a bare ref is a valid slot index,
        // not a misdecode hazard, so no tagged-ness assert is needed — `from_raw`
        // disambiguates purely on bit 30.
        match MargRef::from_raw(raw) {
            MargRef::Inline(c) => MargResolved::Inline(c),
            MargRef::Slot(s) => MargResolved::Index(s as usize),
        }
    } else {
        MargResolved::Index(raw as usize)
    }
}

/// Decode a tagged marg-side ref back to its bare slot index for *structural*
/// use (a grid/width coordinate or array index in apply), NOT a count decode.
///
/// `mask` is loop-invariant per (operand, child-side): `MARG_VALUE_MASK` when
/// the child level is marginal (strip the bit-30 slot tag), `u32::MAX` (identity)
/// otherwise — so the common non-marginal path is a no-op AND. Inverse of
/// `tag_marg_slot`: bit-31-set sentinels (ZERO = `u32::MAX`) pass through
/// untouched so a dead-node ref round-trips exactly as the untagged path saw it.
///
/// Unlike `resolve_marg_ref`, this carries NO bit-30-SET strict assert: it is
/// the entry decode for structural reads, where the value is consumed as a raw
/// coordinate, never interpreted as inline-vs-slot. Phase B: when canon emits
/// inline refs (bit-30 clear), the inline-vs-slot branch lives at the ~3 call
/// sites that feed this helper, gated on the invariant "a canon-inlined level is
/// never again an apply operand" (see Phase B / task #45).
#[inline(always)]
pub(crate) fn decode_marg_coord(raw: u32, mask: u32) -> u32 {
    if raw & (1 << 31) != 0 {
        raw
    } else {
        raw & mask
    }
}

/// End-of-apply production tagger (Phase A): set the slot tag on every persisted
/// marg-side ref across the whole TDD whose child level is marginal.
///
/// Called once after each apply completes, before minimize/canon/query — the
/// boundary at which all intra-apply structural reads (which use raw indices)
/// are done and the first persisted count-decode is about to happen. Path-
/// independent (covers every level-build fast path) and idempotent across the
/// repeated applies of an accumulating compile. This is the single writer
/// chokepoint the strict decode assert in `resolve_marg_ref` audits.
pub(crate) fn tag_all_marg_side_slots(
    tdd: &mut Tdd,
    // #63 no-reexpand: `Some(snapshot)` where `snapshot[i]` is whether level `i`
    // was ALREADY marginal at the enclosing `marginalize_batch` entry. When
    // present, it replaces the lossy `marg_inlined_left/right` marker as the
    // discriminator for which child sides to (re)emit: only sides whose child
    // became marginal *in this batch* hold bare-coord refs needing resolution;
    // already-marginal children carry inline counts from a prior end-sweep and
    // must be skipped (re-resolving an inline value as a slot index → OOB). The
    // reexpand baseline passes `None` and keeps the marker (byte-identical).
    was_marginal: Option<&[bool]>,
) {
    tag_all_marg_side_slots_at(tdd, was_marginal, None);
}

/// [`tag_all_marg_side_slots`] over a caller-chosen subset of internal levels.
///
/// `only` is the spine-bounded apply's rebuild set. Restricting the sweep is
/// result-identical there, not merely sound: the body below does work ONLY at a
/// structural level with at least one marginal child, every such level is in the
/// rebuild set by construction, and off the set neither the level nor its
/// children were touched by the apply — so the skipped iterations would
/// re-derive tags the accumulator already carries. `None` = every internal level
/// (the unrestricted end-of-apply sweep).
pub(crate) fn tag_all_marg_side_slots_at(
    tdd: &mut Tdd,
    was_marginal: Option<&[bool]>,
    only: Option<&[crate::vtree::VtreeIdx]>,
) {
    // Disjoint-field borrow: vtree (shape) immutable, levels (data) mutable.
    let vtree = &tdd.vtree;
    let levels = &mut tdd.levels;
    match only {
        Some(sel) => for &t in sel {
            let (left, right) = vtree.children(t);
            tag_marg_side_slots_at_level(levels, was_marginal, t, left, right);
        },
        None => for (t, left, right) in vtree.internal_bottomup() {
            tag_marg_side_slots_at_level(levels, was_marginal, t, left, right);
        },
    }
}

/// One internal level's share of [`tag_all_marg_side_slots_at`] — the whole
/// per-level body, so the full and restricted sweeps run the identical code.
fn tag_marg_side_slots_at_level(
    levels: &mut [TddLevel],
    was_marginal: Option<&[bool]>,
    t: crate::vtree::VtreeIdx,
    left: crate::vtree::VtreeIdx,
    right: crate::vtree::VtreeIdx,
) {
    {
        let ti = t.idx();
        if levels[ti].is_marginal() {
            return;
        }
        let li = left.idx();
        let ri = right.idx();
        let tag_left = levels[li].is_marginal();
        let tag_right = levels[ri].is_marginal();
        if !tag_left && !tag_right {
            return;
        }
        // Inline-emit: skip any side already holding inline counts — set either
        // by the apply pass-through path (carrier field carried through verbatim)
        // or by a prior emit on this same level (e.g. a fast-path-swapped level
        // carrying its producing apply's marker). Re-running `emit_or_tag` on an
        // inline count C would misread it as slot index C → `counts[C]`
        // corruption; the marker is the only discriminator since inline counts
        // and fresh slots are both bit-30 clear.
        // Discriminator: process (resolve bare coords / emit inline) a side iff
        // its child became marginal in THIS batch. Prefer the reliable
        // `was_marginal` snapshot (the marker is clobbered by rebuilds);
        // otherwise fall back to the per-level marker.
        let do_left = tag_left
            && match was_marginal {
                Some(wm) => !wm[li],
                None => !levels[ti].marg_inlined_left(),
            };
        let do_right = tag_right
            && match was_marginal {
                Some(wm) => !wm[ri],
                None => !levels[ti].marg_inlined_right(),
            };
        match (do_left, do_right) {
            (true, true) => {
                let [p, l, r] = levels.get_disjoint_mut([ti, li, ri]).expect("distinct");
                p.emit_marg_side_slots(
                    l.marginal_counts.as_deref(),
                    r.marginal_counts.as_deref(),
                );
            }
            (true, false) => {
                let [p, l] = levels.get_disjoint_mut([ti, li]).expect("distinct");
                p.emit_marg_side_slots(l.marginal_counts.as_deref(), None);
            }
            (false, true) => {
                let [p, r] = levels.get_disjoint_mut([ti, ri]).expect("distinct");
                p.emit_marg_side_slots(None, r.marginal_counts.as_deref());
            }
            (false, false) => {} // both sides already inline — nothing to emit
        }
        // Mark the sides now carrying inline counts. Keyed off tag_left/tag_right
        // (the marginal-child predicate), not do_*: an already-inline side stays
        // marked so a later re-tag still skips it.
        if tag_left { levels[ti].set_marg_inlined_left(true); }
        if tag_right { levels[ti].set_marg_inlined_right(true); }
    }
}

/// Slot value in [`resolve_swapped_marg_side`]'s interners meaning "this count
/// has no dst slot yet" — the pre-scan collected the key, and the dst seed pass
/// found no existing slot carrying it. Real slot indices are `< MARG_OVERFLOW_TAG`
/// (2^30, asserted by `MargRef::to_raw`), so `u32::MAX` cannot collide with one.
const SLOT_UNSEEDED: u32 = u32::MAX;

/// What [`resolve_swapped_marg_side`] must do with one marg-side ref of the
/// swapped-in parent.
///
/// The SINGLE classification point: the pre-scan and the rewrite pass both
/// branch on this, so "needs a dst slot" in the pre-scan is *definitionally*
/// the condition the rewrite hits — the two cannot drift apart.
enum SwapRef {
    /// Store-independent: a ZERO sentinel (bit 31) or an already-inline count
    /// (bit 30). Passes through untouched.
    Keep,
    /// Bare slot whose source count fits inline: rewritten to an inline ref,
    /// which is store-independent. Touches no store.
    Inline(u32),
    /// Bare slot whose source count is above the inline threshold (or is the
    /// `u128::MAX` BigUint sentinel): must be interned into the dst store.
    /// Carries the SOURCE slot index and its source count.
    Mint(usize, u128),
}

/// Classify one marg-side ref of a swapped-in parent. `inline_max` is read once
/// by the caller (not per ref) so both passes use the same threshold — the
/// test-only override is a thread-local cell that a re-read could observe
/// differently.
///
/// Panics (index OOB) if a bare ref points outside the source store, exactly as
/// the rewrite would; the pre-scan runs first, so that panic now precedes any
/// mutation instead of landing half-way through one.
#[inline]
fn classify_swap_ref(raw: u32, src_counts: &[u128], inline_max: u128) -> SwapRef {
    if raw & (1 << 31) != 0 {
        return SwapRef::Keep; // ZERO sentinel
    }
    if raw & MARG_OVERFLOW_TAG != 0 {
        return SwapRef::Keep; // already an inline count (bit-30 set)
    }
    // Bare slot (bit-30 clear): store-relative index into the source store.
    let s = (raw & MARG_VALUE_MASK) as usize;
    let c = src_counts[s];
    if c != u128::MAX && c <= inline_max {
        SwapRef::Inline(c as u32) // store-independent once written
    } else {
        SwapRef::Mint(s, c)
    }
}

/// Every marg-side ref of `level` on the given side, in the order the rewrite
/// loops at the end of [`resolve_swapped_marg_side`] visit them: the inline-node
/// home (`node.a`/`node.b`) first, then the pairs arena (which includes pairs no
/// live node references — contraction leaves those behind, and the rewrite
/// remaps them too). Read-only twin of those loops: the pre-scan sees exactly
/// the refs the rewrite will.
fn marg_side_refs(level: &TddLevel, is_left: bool) -> impl Iterator<Item = u32> + '_ {
    level
        .nodes
        .iter()
        .filter(|n| n.is_inline())
        .map(move |n| if is_left { n.a } else { n.b })
        .chain(level.pairs.iter().map(move |p| if is_left { p.left.0 } else { p.right.0 }))
}

/// marg-canon #63 (no-reexpand): re-resolve a swapped-in parent level's marginal
/// refs from a SOURCE child store-space into the OUTPUT child store-space.
///
/// The identity fast-paths in `apply_inner` (`SWAP_FP1`/`SWAP_FP2`) `mem::swap` a
/// parent level out of an operand's store into the apply output. A bare ref
/// (bit-30 clear) on a marginal-child side is a *store-relative* slot index into
/// the operand's child `marginal_counts`; after the swap it must point into the
/// OUTPUT child store (`levels[ci]`) instead. For each bare slot ref: read the
/// source count, then either inline it (≤ `MARG_INLINE_MAX` ⇒ store-independent,
/// bit-30 set) or re-mint a fresh slot in the output child store (recording the
/// exact value under the new slot key in [`BigSide`] when it overflowed). Inline
/// refs (bit-30 set) and ZERO sentinels (bit-31) are store-independent and pass
/// through untouched.
///
/// `ti` (swapped-in parent, in `levels`) and `ci` (output child, in `levels`) are
/// distinct; `src_child` is the operand's corresponding child level (a *different*
/// `Tdd`'s store), borrowed immutably.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` when an interner entry or the destination
/// store's growth cannot be reserved. Every allocation is front-loaded by the
/// pre-scan BEFORE the rewrite touches a single ref, and the rewrite pass is
/// infallible by construction — so an over-budget swap leaves the TDD exactly as
/// it found it. A half-remapped level would not merely be large: its unrewritten
/// refs still index the SOURCE store, which miscounts silently.
pub(crate) fn resolve_swapped_marg_side(
    levels: &mut [TddLevel],
    ti: usize,
    ci: usize,
    src_child: &TddLevel,
    is_left: bool,
) -> Result<(), ApplyError> {
    use crate::tdd::counts::{ApplyBudget, ReservePolicy};

    debug_assert_ne!(ti, ci);
    // WEIGHTED (`--weighted`): nothing to re-resolve, by construction. The whole
    // remap exists because the integer marginal store is PER-`Tdd`, so a swapped-in
    // parent's bare slot refs are relative to the operand's store and must be
    // re-minted into the output's. The weighted store is not per-`Tdd`: there is
    // ONE compile-global `WeightStore` keyed by vtree level, so the source child
    // level and the output child level at `ci` share the very same column and a
    // bare slot ref is already in the destination store-space. Returning here is
    // the correct no-op — and the only correct action, since a weight-marginal
    // level's `marginal_counts` is `None` (the two representations are mutually
    // exclusive within a compile) and both `expect`s below would fire.
    if src_child.is_weight_marginal() || levels[ci].is_weight_marginal() {
        debug_assert!(
            src_child.is_weight_marginal() && levels[ci].is_weight_marginal(),
            "mixed marginal representations at level {ci}: integer on one side, \
             weighted on the other"
        );
        return Ok(());
    }
    let src_counts = src_child
        .marginal_counts
        .as_deref()
        .expect("resolve_swapped_marg_side: src child missing marginal_counts");
    let src_big = src_child.marginal_counts_big.as_ref();
    // Read the inline threshold ONCE, not per ref: the pre-scan and the rewrite
    // must classify every ref identically, and the test-only override backing
    // `marg_inline_max` is a thread-local cell a re-read could observe changed.
    let inline_max = marg_inline_max() as u128;
    // Disjoint &mut borrows of the parent (ti) and output child (ci) levels.
    let (parent, dst_child) = if ti < ci {
        let (a, b) = levels.split_at_mut(ci);
        (&mut a[ti], &mut b[0])
    } else {
        let (a, b) = levels.split_at_mut(ti);
        (&mut b[0], &mut a[ci])
    };
    // The destination side table is SPARSE (`BigSide`), so it needs no
    // pre-alignment to the destination store's width — a re-minted overflow
    // slot simply records its own key. Disjoint field borrows of `dst_child`.
    let dst_big = &mut dst_child.marginal_counts_big;
    let dst_counts = dst_child
        .marginal_counts
        .as_mut()
        .expect("resolve_swapped_marg_side: dst child missing marginal_counts");

    // ── Pre-scan: the counts that actually need a destination slot ──────────
    //
    // Only a `Mint` ref reaches the dst store at all — `Keep` and `Inline` refs
    // are store-independent — so a swap carrying none of them needs no interner,
    // no store growth and no side table, and must allocate NOTHING. The
    // interners are therefore keyed by what this scan finds (bounded by the
    // parent's ref count), not seeded from the whole dst store: the store-sized
    // seed cost ~1.5-2× the store in hash entries plus one `BigUint` clone per
    // dst overflow slot, built before knowing whether one ref needed re-minting.
    //
    // Store is born C3: no duplicate count values; enforced here, not by a later
    // canon pass. Key: `u128` for above-threshold counts, `BigUint` for
    // OVERFLOW-sentinel counts (so two numerically equal BigUints share one dst
    // slot). Counts ≤ `inline_max` ride inline at the ref and never become
    // slots, so they need no entry. Values stay `SLOT_UNSEEDED` until a dst slot
    // is found (seed pass below) or minted (rewrite pass).
    let mut small_to_slot: FxHashMap<u128, u32> = FxHashMap::default();
    let mut big_to_slot: FxHashMap<BigUint, u32> = FxHashMap::default();
    // OVERFLOW-sentinel source slots carrying no exact value to key on. The
    // rewrite's debug_assert rejects them; in release each one pushes its own
    // dst slot and cannot dedup, so each needs its own reservation.
    let mut orphan_overflow = 0usize;
    for raw in marg_side_refs(parent, is_left) {
        let SwapRef::Mint(s, c) = classify_swap_ref(raw, src_counts, inline_max) else {
            continue;
        };
        if c == u128::MAX {
            match src_big.and_then(|sb| sb.get(s)) {
                Some(b) if !big_to_slot.contains_key(b) => {
                    big_to_slot.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                    big_to_slot.insert(b.clone(), SLOT_UNSEEDED);
                }
                Some(_) => {} // key already interned by an earlier ref
                None => orphan_overflow += 1,
            }
        } else if !small_to_slot.contains_key(&c) {
            small_to_slot.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
            small_to_slot.insert(c, SLOT_UNSEEDED);
        }
    }
    // Upper bound on the slots the rewrite can mint: one per distinct interned
    // count (a second ref carrying it dedups onto the first) plus one per
    // un-keyable overflow.
    let new_slots = small_to_slot.len() + big_to_slot.len() + orphan_overflow;

    if new_slots > 0 {
        // Front-load every allocation the rewrite can need — it must not fail
        // part-way (see the doc comment's error contract). `reserve` (doubling)
        // matches the growth the `push`es below would have taken on their own,
        // and routes through the ONE apply budget accounting path.
        ApplyBudget::reserve(dst_counts, new_slots)?;
        if !big_to_slot.is_empty() {
            // A keyed overflow mint always lands here, so the table exists by
            // the time the rewrite inserts into it. Creating it is not a new
            // side effect: with no table there is nothing for an overflow ref to
            // dedup against, so the first such ref minted — and created it —
            // before this change too.
            dst_big
                .get_or_insert_with(BigSide::default)
                .try_reserve::<ApplyBudget>(big_to_slot.len())?;
        }
        // Seed the interners from the dst slots already carrying a wanted count,
        // so an equal count reuses its slot instead of pushing a duplicate.
        // Lowest index wins, as the `or_insert` seed this replaced did. Slots
        // holding a count nothing asked for are skipped — that is what keeps the
        // interners sized by the scan rather than by the store.
        for i in 0..dst_counts.len() {
            let c = dst_counts[i];
            if c == u128::MAX {
                if let Some(b) = dst_big.as_ref().and_then(|b| b.get(i)) {
                    if let Some(slot) = big_to_slot.get_mut(b) {
                        if *slot == SLOT_UNSEEDED {
                            *slot = i as u32;
                        }
                    }
                }
            } else if let Some(slot) = small_to_slot.get_mut(&c) {
                if *slot == SLOT_UNSEEDED {
                    *slot = i as u32;
                }
            }
        }
    }

    // Rewrite pass — INFALLIBLE: every push below has reserved capacity above.
    let mut remap = |raw: u32| -> u32 {
        let (s, c) = match classify_swap_ref(raw, src_counts, inline_max) {
            // ZERO sentinel or already-inline count: store-independent.
            SwapRef::Keep => return raw,
            // Small enough to carry in the ref (bit-30 set): store-independent.
            SwapRef::Inline(c) => return MargRef::Inline(c).to_raw(),
            SwapRef::Mint(s, c) => (s, c),
        };
        // Big (`u128::MAX` sentinel) or large-but-u128 count: re-mint into dst store,
        // deduplicating via the inline interner so equal large counts share one slot.
        if c == u128::MAX {
            // BigUint path: look up in big_to_slot first.
            let big_val = src_big.and_then(|sb| sb.get(s));
            debug_assert!(
                big_val.is_some(),
                "resolve_swapped_marg_side: src slot {s} is the u128::MAX sentinel \
                 but has no BigUint entry"
            );
            if let Some(b) = big_val {
                match big_to_slot.get(b) {
                    Some(&existing) if existing != SLOT_UNSEEDED => {
                        return MargRef::slot_raw(existing);
                    }
                    _ => {}
                }
            }
            let new_idx = dst_counts.len() as u32;
            dst_counts.push(u128::MAX);
            if let Some(b) = big_val {
                dst_big
                    .as_mut()
                    .expect("keyed overflow ⇒ pre-scan created the side table")
                    .insert(new_idx as usize, b.clone());
                // The key was interned by the pre-scan; record the slot it just got.
                if let Some(slot) = big_to_slot.get_mut(b) {
                    *slot = new_idx;
                }
            }
            MargRef::slot_raw(new_idx)
        } else {
            // Small (but above inline threshold) count: look up in small_to_slot.
            // Nothing to mirror into the sparse side table — a non-overflow slot
            // simply has no entry there.
            match small_to_slot.get(&c) {
                Some(&existing) if existing != SLOT_UNSEEDED => {
                    return MargRef::slot_raw(existing);
                }
                _ => {}
            }
            let new_idx = dst_counts.len() as u32;
            dst_counts.push(c);
            if let Some(slot) = small_to_slot.get_mut(&c) {
                *slot = new_idx;
            }
            MargRef::slot_raw(new_idx)
        }
    };
    for node in &mut parent.nodes {
        if node.is_inline() {
            if is_left {
                node.a = remap(node.a);
            } else {
                node.b = remap(node.b);
            }
        }
    }
    for p in &mut parent.pairs {
        if is_left {
            p.left.0 = remap(p.left.0);
        } else {
            p.right.0 = remap(p.right.0);
        }
    }
    Ok(())
}


/// Soundness precondition for [`TddLevel::make_marginal`]: both children
/// of the target vtree node `t` must already be marginal. Leaves count as
/// already-marginal — a leaf's per-node model counts are fixed by its
/// label (Pos→1, Neg→1, One→2), so there is no pair structure to discard
/// and no precondition to enforce. For an internal target with leaf
/// children the check therefore reduces to "leaves are fine"; for a leaf
/// target the precondition is vacuously true.
///
/// Panics if the
/// precondition is violated — callers should arrange their work so the
/// precondition holds naturally (e.g. `marginalize_batch` processes its
/// targets in bottom-up topo order).
///
/// O(1): one branch + one array index per child.
#[inline]
pub(crate) fn assert_can_make_marginal(levels: &[TddLevel], vtree: &crate::vtree::Vtree, t: crate::vtree::VtreeIdx) {
    use crate::vtree::VtreeNode;
    let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
        // Leaf target: no children to check; marginalizing a leaf is a
        // semantic no-op (fixed-label counts), so nothing to enforce.
        return;
    };
    for child in [left, right] {
        let is_leaf = matches!(*vtree.node(child), VtreeNode::Leaf { .. });
        if !is_leaf && !levels[child.idx()].is_marginal() {
            panic!(
                "make_marginal({}) precondition violated: child {} is internal \
                 but not yet marginal. Process marginalize targets bottom-up so \
                 children are marginalized before parents.",
                t.idx(), child.idx(),
            );
        }
    }
}

/// Direct contract tests for [`resolve_swapped_marg_side`]. Integration-level
/// coverage cannot discriminate this fixup: on every quickset instance that
/// both solves and traverses it under plain `portfolio2`, and across the full
/// test suite's in-process traversals, no-op'ing the function changes no
/// count — the identity-fast-path chains that trigger it produce output child
/// stores content-identical to the source store, so the remap is semantically
/// idempotent there. The fixup is load-bearing exactly when the stores
/// DIVERGE (different slot order / absent counts / different lengths), which
/// these tests construct directly.
#[cfg(test)]
#[path = "marg_resolve_swap_tests.rs"]
mod resolve_swap_tests;

