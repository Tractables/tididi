//! Marginal-side ref encoding, marg consts, and associated helpers.

use crate::engine::Engine;
use num_bigint::BigUint;

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
/// never a wrong count. The opposite polarity — tag the slot, leave the inline
/// count bare — has no such safe failure: a bit-30-clear value would be
/// ambiguous between an untagged slot and an inline count, and reading a slot
/// index as a count silently multiplies the answer.
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
/// store itself can be millions of slots wide. A dense `Vec<Option<BigUint>>`
/// would cost 24 B per *slot* on every marginal level that overflowed even
/// once; keying by slot makes the cost proportional to the overflow set, and
/// makes the no-overflow case free — an empty `BigSide` owns no heap at all.
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
    pub(crate) fn try_insert<R: crate::counts::ReservePolicy>(
        &mut self, eng: &Engine, slot: usize,
        v: BigUint,
    ) -> Result<(), R::Err> {
        R::reserve(eng, &mut self.entries, 1)?;
        self.insert(slot, v);
        Ok(())
    }

    /// Bulk twin of [`try_insert`](Self::try_insert): reserve room for
    /// `additional` entries through the same policy, so a caller that has
    /// already begun mutating the store — and therefore must not fail
    /// part-way — can front-load its allocation and then [`insert`](Self::insert)
    /// infallibly. `resolve_swapped_marg_side` is that caller.
    #[inline]
    pub(crate) fn try_reserve<R: crate::counts::ReservePolicy>(
        &mut self, eng: &Engine, additional: usize,
    ) -> Result<(), R::Err> {
        R::reserve(eng, &mut self.entries, additional)
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
    pub(crate) fn try_clone<R: crate::counts::ReservePolicy>(
        &self, eng: &Engine) -> Result<Self, R::Err> {
        let mut entries: Vec<(u32, BigUint)> = Vec::new();
        R::reserve_exact(&eng, &mut entries, self.entries.len())?;
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

// Test-only runtime override of the inline-vs-slot count threshold.
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

/// Force the inline-vs-slot threshold for the lifetime of the returned guard.
/// Test-only. See `MARG_INLINE_MAX_OVERRIDE`.
#[cfg(any(test, debug_assertions))]
pub fn set_marg_inline_max(v: u32) -> crate::scoped::Scoped<std::cell::Cell<Option<u32>>> {
    crate::scoped::Scoped::install(&MARG_INLINE_MAX_OVERRIDE, Some(v))
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
/// use tididi::diagram::{MargResolved, resolve_marg_ref, MargRef};
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
/// `tag_all_marg_side_slots`: bit-31-set sentinels (ZERO = `u32::MAX`) pass through
/// untouched so a dead-node ref round-trips exactly as the untagged path saw it.
///
/// Unlike `resolve_marg_ref`, this carries NO bit-30-SET strict assert: it is
/// the entry decode for structural reads, where the value is consumed as a raw
/// coordinate, never interpreted as inline-vs-slot. Where canonicalization emits
/// inline refs (bit-30 clear), the inline-vs-slot branch lives at the few call
/// sites that feed this helper, resting on the invariant "a level whose refs
/// were inlined by canonicalization is never again an apply operand".
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
    // No-re-expand: `Some(snapshot)` where `snapshot[i]` is whether level `i`
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

#[path = "marg_swap.rs"]
mod swap;
pub(crate) use swap::resolve_swapped_marg_side;
