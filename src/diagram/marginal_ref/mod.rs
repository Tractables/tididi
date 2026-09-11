//! Marginal-side ref encoding, marginal consts, and associated helpers.

use crate::engine::Engine;
use num_bigint::BigUint;

use super::level::TddLevel;
use super::primitives::NodeIdx;

/// Bit 30 of a pair side whose child level is marginal: clear means the value
/// is an index into the child's `marginal_counts`, set means the low 30 bits
/// are the model count itself. Readers use [`SideView`] instead of testing
/// this bit.
///
/// This "bare-is-slot, tag-the-inline" polarity makes the encoding fail safe
/// and tagging-free on the common path. After a child level is marginalized,
/// slot index ≡ node index in `marginal_counts`, so a parent's child-ref — a
/// bare node index left over from before marginalization — is *already* a valid
/// slot reference. Nothing has to be re-tagged when a child marginalizes
/// (including late, by an ancestor's streaming). Only the optional inline
/// optimization (store a small count in the ref itself, saving a heap load) sets
/// bit 30, and it does so explicitly.
///
/// A missed inline-write therefore reads back as a (correct) bare slot index,
/// never a wrong count. The opposite polarity — tag the slot, leave the inline
/// count bare — has no such safe failure: a bit-30-clear value would be
/// ambiguous between an untagged slot and an inline count, and reading a slot
/// index as a count silently multiplies the answer.
pub(super) const MARGINAL_OVERFLOW_TAG: u32 = 1 << 30;
/// Mask for the 30-bit payload (count value or slot index).
pub(super) const MARGINAL_VALUE_MASK: u32 = MARGINAL_OVERFLOW_TAG - 1;
/// Largest model count a pair side stores inline; larger counts are held in
/// the child's `marginal_counts` and referenced by index.
pub(crate) const MARGINAL_INLINE_MAX: u32 = MARGINAL_OVERFLOW_TAG - 1;

/// A pair side whose child level is marginal: the stored word, before decode.
///
/// The bits are laid out as
///
/// ```text
///   bit 31    | always 0 — the `TddNodeData` inline-pair encoding claims it
///   bit 30    | tag: 0 = slot index, 1 = inline count
///   bits 29..0| payload (the count, or the index into `marginal_counts`)
/// ```
///
/// Bit 30 is a tag only here — on a side whose child level is structural it is
/// an ordinary index bit, which is why the decode needs the child's kind and
/// why [`SideView`] carries it. The [`super::ZERO`] sentinel
/// (`u32::MAX`) has bit 31 set and so lies outside the encoding entirely; it
/// never appears in a pair list.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]
pub(crate) struct MarginalSide(pub u32);

impl MarginalSide {
    /// The word as it is stored in a pair side.
    #[inline(always)]
    pub(crate) fn side(self) -> NodeIdx {
        NodeIdx(self.0)
    }

    /// True when the word is the [`super::ZERO`] sentinel rather than a
    /// reference into the child level. The sentinel never appears in a stored
    /// pair; a scratch array being swept can still hold one, and every decode
    /// tests this before interpreting the payload.
    #[inline(always)]
    pub(crate) fn is_zero_sentinel(self) -> bool {
        self.side().is_reserved()
    }
}

/// The value a pair side denotes when its child level is marginal: either the
/// count itself or the slot that holds it.
///
/// Readers decode a whole level's sides through [`SideView`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ValueRef {
    /// The model count itself, at most the 30-bit payload a pair side holds.
    Inline(u32),
    /// An index into the child level's `marginal_counts`.
    Slot(u32),
}

impl ValueRef {
    /// Decode a pair side whose child level is marginal.
    #[inline(always)]
    pub(crate) fn from_raw(r: MarginalSide) -> Self {
        debug_assert!(
            !r.is_zero_sentinel(),
            "marginal-side ref must not be the zero sentinel"
        );
        if r.0 & MARGINAL_OVERFLOW_TAG != 0 {
            ValueRef::Inline(r.0 & MARGINAL_VALUE_MASK)
        } else {
            ValueRef::Slot(r.0)
        }
    }

    /// The word a pair side stores for this value — the writer half of the
    /// decode [`SideView::child`] performs. A caller assembling a level by hand
    /// encodes through this and reads back through the view.
    #[inline(always)]
    pub fn side(self) -> NodeIdx {
        self.to_raw().side()
    }

    /// The word to store in the pair side.
    #[inline(always)]
    pub(crate) fn to_raw(self) -> MarginalSide {
        match self {
            ValueRef::Inline(c) => {
                debug_assert!(
                    c <= MARGINAL_INLINE_MAX,
                    "inline count overflow: {} > {}",
                    c,
                    MARGINAL_INLINE_MAX
                );
                MarginalSide(c | MARGINAL_OVERFLOW_TAG)
            }
            ValueRef::Slot(s) => {
                debug_assert!(
                    s & !MARGINAL_VALUE_MASK == 0,
                    "slot index overflow: {} >= {}",
                    s,
                    MARGINAL_OVERFLOW_TAG
                );
                MarginalSide(s)
            }
        }
    }

    /// The count a marginal-side word carries inline, or `None` when it is a slot
    /// reference. The two-instruction decode the counting fold wants, without
    /// building a `ValueRef` it would immediately match on.
    #[inline(always)]
    pub(crate) fn inline_count(raw: u32) -> Option<u32> {
        if raw & MARGINAL_OVERFLOW_TAG != 0 {
            Some(raw & MARGINAL_VALUE_MASK)
        } else {
            None
        }
    }

    /// Whether a marginal-side word carries its count inline — the predicate the
    /// invariant checks want, with no payload.
    #[inline(always)]
    pub(crate) fn is_inline_raw(raw: u32) -> bool {
        raw & MARGINAL_OVERFLOW_TAG != 0
    }

    /// Whether `slot_idx` fits the payload a pair side can hold. A store that
    /// outgrows it cannot be referenced at all, so the caller that minted the
    /// slot must fail rather than truncate.
    #[inline(always)]
    pub(crate) fn slot_is_referenceable(slot_idx: u32) -> bool {
        slot_idx & !MARGINAL_VALUE_MASK == 0
    }

    /// Convenience: encode a slot index as a raw u32 marginal-side ref.
    #[inline(always)]
    pub(crate) fn slot_raw(slot_idx: u32) -> u32 {
        ValueRef::Slot(slot_idx).to_raw().0
    }

    /// Convenience: encode an inline count as a raw u32 marginal-side ref.
    /// Returns `None` if the count doesn't fit (caller should allocate a slot).
    #[inline(always)]
    pub(crate) fn inline_raw(count: u128) -> Option<u32> {
        if count <= MARGINAL_INLINE_MAX as u128 {
            Some(ValueRef::Inline(count as u32).to_raw().0)
        } else {
            None
        }
    }
}

// ── Overflow side table (sparse) ─────────────────────────────────────────────

/// The exact `BigUint` value of every count slot whose fast `u128` cell holds
/// the `u128::MAX` overflow sentinel — the overflow half of a marginal count
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
/// column left to right, `value::slots::push_count_key` and
/// `resolve_swapped_marginal_side` mint at the store's end, and the two compaction
/// passes (`dedup_fresh_store`, `slot_prune`'s `IntFold::compact_store`) rebuild
/// by draining this table in ascending order. So insertion is an O(1) amortized
/// push on the common path and an in-place overwrite otherwise; reads
/// binary-search a handful of entries, which beats hashing and keeps the
/// per-entry footprint to one `(u32, BigUint)` with no control bytes or
/// load-factor slack. Slot indices are ≤ 30 bits wherever a parent ref can name
/// them (see `MARGINAL_VALUE_MASK`), so `u32` keys are ample.
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
    /// Number of slots carrying an exact `BigUint` — not the store width.
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

    /// Store `v` at `slot`, replacing any value already there. The one insert
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
    pub(crate) fn try_insert<R: crate::limits::ReservePolicy>(
        &mut self,
        eng: &Engine,
        slot: usize,
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
    /// infallibly. `resolve_swapped_marginal_side` is that caller.
    #[inline]
    pub(crate) fn try_reserve<R: crate::limits::ReservePolicy>(
        &mut self,
        eng: &Engine,
        additional: usize,
    ) -> Result<(), R::Err> {
        R::reserve(eng, &mut self.entries, additional)
    }

    /// Remove `slot`'s value and hand it back, so no stale `BigUint` is left
    /// behind under a key whose fast cell no longer holds the sentinel. `None`
    /// when the slot carried no exact value. Point mutation only — a pass that
    /// relocates many slots must drain and rebuild ([`IntoIterator`]) instead,
    /// since removing survivors one at a time shifts the tail each time.
    #[inline]
    pub(crate) fn take(&mut self, slot: usize) -> Option<BigUint> {
        let slot = u32::try_from(slot).ok()?;
        match self.entries.binary_search_by_key(&slot, |&(s, _)| s) {
            Ok(pos) => Some(self.entries.remove(pos).1),
            Err(_) => None,
        }
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

    /// Consume the table into its `(slot, value)` pairs in ascending slot
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

/// What one pair side refers to in its child level.
///
/// A side of a structural child names a node; a side of a marginal child names
/// a value — a slot of the child's `marginal_counts`, or the count itself when
/// it is small enough to ride in the side. [`SideView`] produces this.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ChildRef {
    /// A node of the child level, at this index.
    Node(NodeIdx),
    /// A value of a marginal child level.
    Value(ValueRef),
}

impl ChildRef {
    /// The cell of the child level this side indexes — `nodes` for a node,
    /// `marginal_counts` for a slot. `None` for an inline value, which names
    /// no cell of the child at all.
    #[inline(always)]
    pub fn index(self) -> Option<usize> {
        match self {
            ChildRef::Node(NodeIdx(i)) | ChildRef::Value(ValueRef::Slot(i)) => Some(i as usize),
            ChildRef::Value(ValueRef::Inline(_)) => None,
        }
    }
}

/// How to read the pair sides that point at one child level.
///
/// The same 32 bits mean different things depending on the child: a plain node
/// index under a structural child, a tagged [`ValueRef`] under a marginal one.
/// Build the view once per level visit — [`TddLevel::side_view`] — and decode
/// every side of that level through it, rather than re-deciding per side.
///
/// Two decodes, and a reader wants exactly one of them:
///
/// - [`child`](Self::child) — what the side *denotes*, for counting and
///   traversal.
/// - [`coord`](Self::coord) — where the side *sits* in the child level, for a
///   width-sized array index or grid coordinate. It strips the tag and never
///   interprets it, so an inline value comes back as its own bits.
///
/// ```
/// use tididi::diagram::{ChildRef, NodeIdx, SideView, ValueRef};
/// // A structural child: any word is a node index.
/// assert_eq!(SideView::structural().child(NodeIdx(7)), ChildRef::Node(NodeIdx(7)));
/// // A marginal child: a bare word is a slot...
/// assert_eq!(
///     SideView::marginal().child(NodeIdx(7)),
///     ChildRef::Value(ValueRef::Slot(7))
/// );
/// // ...and a tagged one is the count itself.
/// assert_eq!(
///     SideView::marginal().child(ValueRef::Inline(7).side()),
///     ChildRef::Value(ValueRef::Inline(7))
/// );
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct SideView {
    valued: bool,
}

impl SideView {
    /// Sides pointing at a structural or leaf level: every word is a node index.
    #[inline(always)]
    pub const fn structural() -> Self {
        SideView { valued: false }
    }

    /// Sides pointing at a marginal level: every word is a [`ValueRef`].
    #[inline(always)]
    pub const fn marginal() -> Self {
        SideView { valued: true }
    }

    /// Whether the child level is marginal — whether a side of it carries a
    /// [`ValueRef`] rather than a node index.
    #[inline(always)]
    pub const fn is_marginal(self) -> bool {
        self.valued
    }

    /// What `side` denotes in the child level.
    #[inline(always)]
    pub fn child(self, side: NodeIdx) -> ChildRef {
        if self.valued {
            // Bare-is-slot: a side left over from before the child marginalized
            // is already a valid slot, so nothing needs re-tagging and only the
            // inline optimisation sets bit 30.
            ChildRef::Value(ValueRef::from_raw(MarginalSide(side.0)))
        } else {
            ChildRef::Node(side)
        }
    }

    /// Rewrite `side` through a `remap` indexed by the child level's cells —
    /// the child was compacted and every cell moved to `remap[cell]`.
    ///
    /// An inline value names no cell of the child, so it passes through
    /// unchanged; a slot comes back re-tagged.
    #[inline]
    pub fn remap(self, side: NodeIdx, remap: &[u32]) -> NodeIdx {
        // A bit-31 sentinel (the `ZERO` ref) names no cell either. It never
        // appears in a stored pair, so this only guards a caller sweeping a
        // scratch array that still holds one.
        if side.is_reserved() {
            return side;
        }
        debug_assert!(
            self.child(side).index().is_none_or(|c| remap[c] != u32::MAX),
            "a referenced cell must survive the compaction it is remapped through",
        );
        match self.child(side) {
            ChildRef::Node(NodeIdx(i)) => NodeIdx(remap[i as usize]),
            ChildRef::Value(ValueRef::Slot(s)) => ValueRef::Slot(remap[s as usize]).to_raw().side(),
            ChildRef::Value(ValueRef::Inline(_)) => side,
        }
    }

    /// Where `side` sits in the child level, as a bare coordinate.
    ///
    /// Unlike [`child`](Self::child) this never interprets the tag: it is the
    /// entry decode for structural reads, which consume the value as a grid or
    /// array coordinate. Bit-31 sentinels (a dead-node ref) pass through
    /// untouched, so such a ref round-trips exactly as an untagged read saw it.
    #[inline(always)]
    pub fn coord(self, side: NodeIdx) -> NodeIdx {
        if self.valued && !side.is_reserved() {
            NodeIdx(side.0 & MARGINAL_VALUE_MASK)
        } else {
            side
        }
    }
}

/// Soundness precondition for [`TddLevel::become_marginal`]: both children
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
pub(crate) fn assert_can_make_marginal(
    levels: &[TddLevel],
    vtree: &crate::vtree::Vtree,
    t: crate::vtree::VtreeIdx,
) {
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
                "become_marginal({}) precondition violated: child {} is internal \
                 but not yet marginal. Process marginalize targets bottom-up so \
                 children are marginalized before parents.",
                t.idx(),
                child.idx(),
            );
        }
    }
}

pub(crate) mod refs;
mod swap;
mod tag;

pub(crate) use refs::{
    boundary_marginal_levels, boundary_marginal_levels_into, boundary_marginal_levels_of,
    for_each_side_ref_mut, remap_refs_into, ChildSide,
};
pub(crate) use swap::resolve_swapped_marginal_side;
pub(crate) use tag::tag_all_marginal_side_slots;

// Test support.
impl BigSide {
    /// Budget-tracked clone: reserves the entry count exactly before copying,
    /// mirroring [`try_insert`](Self::try_insert)'s accounting discipline.
    /// Test-only since the borrowed-view rewrite removed production column
    /// duplication (sole caller: `CountVec::try_clone`).
    #[cfg(test)]
    pub(crate) fn try_clone<R: crate::limits::ReservePolicy>(
        &self,
        eng: &Engine,
    ) -> Result<Self, R::Err> {
        let mut entries: Vec<(u32, BigUint)> = Vec::new();
        R::reserve_exact(eng, &mut entries, self.entries.len())?;
        entries.extend(self.entries.iter().cloned());
        Ok(BigSide { entries })
    }
}
