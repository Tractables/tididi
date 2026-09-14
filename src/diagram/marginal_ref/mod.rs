//! Marginal-side ref encoding, marginal consts, and associated helpers.

use crate::limits::OperationError;
use crate::engine::Engine;
use num_bigint::BigUint;

use super::level::TddLevel;
use super::primitives::{EncodedChildRef, NodeIdx};

/// Bit 30 of a pair side whose child level is marginal: clear means the value
/// is an index into the child's `marginal_counts`, set means the low 30 bits
/// are the model count itself. Readers use [`ChildDecoder`] instead of testing
/// this bit.
///
/// After a child level is marginalized, slot index equals node index in
/// `marginal_counts`, so a parent's bare node index is already a valid slot
/// reference and nothing has to be re-tagged; only the inline optimization
/// sets bit 30.
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
///   bit 31    | always 0 — the `EncodedNode` inline-pair encoding claims it
///   bit 30    | tag: 0 = slot index, 1 = inline count
///   bits 29..0| payload (the count, or the index into `marginal_counts`)
/// ```
///
/// Bit 30 is a tag only here — on a side whose child level is structural it is
/// an ordinary index bit, which is why the decode needs the child's kind and
/// why [`ChildDecoder`] carries it. The [`super::ZERO`] sentinel
/// (`u32::MAX`) has bit 31 set and so lies outside the encoding entirely; it
/// never appears in a pair list.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]
pub(crate) struct MarginalSide(pub u32);

impl MarginalSide {
    /// The word as it is stored in a pair side.
    #[inline(always)]
    pub(crate) fn side(self) -> EncodedChildRef {
        EncodedChildRef(self.0)
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
/// Readers decode a whole level's sides through [`ChildDecoder`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ValueRef {
    /// The model count itself, at most the 30-bit payload a pair side holds.
    Inline(u32),
    /// An index into the child level's `marginal_counts`.
    Slot(u32),
}

/// A value reference whose payload does not fit the 30-bit pair-side encoding.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ValueRefError {
    /// The inline count or slot index that could not be encoded.
    pub reference: ValueRef,
}

impl std::fmt::Display for ValueRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.reference {
            ValueRef::Inline(count) => write!(f, "inline count {count} exceeds {MARGINAL_VALUE_MASK}"),
            ValueRef::Slot(slot) => write!(f, "value slot {slot} exceeds {MARGINAL_VALUE_MASK}"),
        }
    }
}

impl std::error::Error for ValueRefError {}

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

    /// Encode this value reference as a pair side, for decoding with [`ChildDecoder::child`].
    ///
    /// # Errors
    ///
    /// Returns [`ValueRefError`] if the inline count or slot index exceeds
    /// `2^30 - 1`. Larger counts must be stored in a slot.
    #[inline(always)]
    pub fn side(self) -> Result<EncodedChildRef, ValueRefError> {
        let (payload, tag) = match self {
            ValueRef::Inline(count) => (count, MARGINAL_OVERFLOW_TAG),
            ValueRef::Slot(slot) => (slot, 0),
        };
        if payload > MARGINAL_VALUE_MASK {
            return Err(ValueRefError { reference: self });
        }
        Ok(EncodedChildRef(payload | tag))
    }

    /// Encode an internally constructed reference whose payload fits in 30 bits.
    #[inline(always)]
    pub(crate) fn to_raw(self) -> MarginalSide {
        MarginalSide(self.side().expect("internal marginal reference must fit in 30 bits").raw())
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
/// store (`TddLevel::marginal_counts` / `marginal_counts_big`). A reader
/// decoding a marginal level reads it through [`get`](Self::get), and
/// [`len`](Self::len) / [`is_empty`](Self::is_empty) say how many slots
/// overflowed at all.
///
/// Keyed by slot index, not parallel to the fast column: a slot lands here
/// only when its count exceeds `u128::MAX`, so the cost is proportional to the
/// overflow set and an empty `CountOverflow` owns no heap. A slot with no entry
/// means the value fits the fast `u128` lane.
///
/// Representation: `(slot, value)` pairs sorted by `slot`, strictly ascending,
/// no duplicate slots. Every write path appends at a slot larger than any
/// stored, so insertion is an amortized O(1) push and reads binary-search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CountOverflow {
    /// Sorted by slot, strictly ascending, slots unique. Every method below
    /// preserves that; nothing outside this module can break it.
    entries: Vec<(u32, BigUint)>,
}

impl CountOverflow {
    /// Reserved entry storage, excluding each big integer's numeric payload.
    pub(crate) fn buffer_bytes(&self) -> u64 {
        (self.entries.capacity() * std::mem::size_of::<(u32, BigUint)>()) as u64
    }

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
    /// through the engine and then calls this.
    #[inline]
    pub(crate) fn insert(&mut self, slot: usize, v: BigUint) {
        let slot = u32::try_from(slot).expect("marginal slot index must fit u32");
        match self.entries.binary_search_by_key(&slot, |&(s, _)| s) {
            Ok(pos) => self.entries[pos].1 = v,
            // Ascending appends land at `pos == len`, a plain push.
            Err(pos) => self.entries.insert(pos, (slot, v)),
        }
    }

    /// Reserve room for one entry through the engine before calling
    /// [`insert`](Self::insert), returning [`OperationError`] on refusal.
    #[inline]
    pub(crate) fn try_insert(
        &mut self,
        eng: &Engine,
        slot: usize,
        v: BigUint,
    ) -> Result<(), OperationError> {
        eng.limits().reserve(&mut self.entries, 1)?;
        self.insert(slot, v);
        Ok(())
    }

    /// Bulk twin of [`try_insert`](Self::try_insert): reserve room for
    /// `additional` entries through the engine, so a caller that must not
    /// fail part-way can then [`insert`](Self::insert) infallibly.
    #[inline]
    pub(crate) fn try_reserve(
        &mut self,
        eng: &Engine,
        additional: usize,
    ) -> Result<(), OperationError> {
        eng.limits().reserve(&mut self.entries, additional)
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

impl FromIterator<(u32, BigUint)> for CountOverflow {
    /// Build from `(slot, value)` pairs in any order; later values win for a
    /// repeated slot. Goes through `insert` so the sorted
    /// invariant has exactly one enforcer.
    fn from_iter<I: IntoIterator<Item = (u32, BigUint)>>(iter: I) -> Self {
        let mut out = CountOverflow::default();
        for (slot, v) in iter {
            out.insert(slot as usize, v);
        }
        out
    }
}

impl IntoIterator for CountOverflow {
    type Item = (u32, BigUint);
    type IntoIter = std::vec::IntoIter<(u32, BigUint)>;

    /// Consume the table into its `(slot, value)` pairs in ascending slot
    /// order, moving each `BigUint` out. A compaction pass rekeys a table by
    /// consuming it, remapping each slot, and collecting back.
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// What one pair side refers to in its child level.
///
/// A side of a structural child names a node; a side of a marginal child names
/// a value — a slot of the child's `marginal_counts`, or the count itself when
/// it is small enough to ride in the side. [`ChildDecoder`] produces this.
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
    pub(crate) fn index(self) -> Option<usize> {
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
/// Build the view once per level visit — [`TddLevel::child_decoder`] — and decode
/// every side of that level through it, rather than re-deciding per side.
///
/// ```
/// use tididi::diagram::{ChildRef, EncodedChildRef, NodeIdx, ChildDecoder, ValueRef};
/// // A structural child: any word is a node index.
/// assert_eq!(ChildDecoder::structural().child(EncodedChildRef::from_raw(7)), ChildRef::Node(NodeIdx(7)));
/// // A marginal child: a bare word is a slot...
/// assert_eq!(
///     ChildDecoder::marginal().child(EncodedChildRef::from_raw(7)),
///     ChildRef::Value(ValueRef::Slot(7))
/// );
/// // ...and a tagged one is the count itself.
/// assert_eq!(
///     ChildDecoder::marginal().child(ValueRef::Inline(7).side().unwrap()),
///     ChildRef::Value(ValueRef::Inline(7))
/// );
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ChildDecoder {
    valued: bool,
}

impl ChildDecoder {
    /// Sides pointing at a structural or leaf level: every word is a node index.
    #[inline(always)]
    pub const fn structural() -> Self {
        ChildDecoder { valued: false }
    }

    /// Sides pointing at a marginal level: every word is a [`ValueRef`].
    #[inline(always)]
    pub const fn marginal() -> Self {
        ChildDecoder { valued: true }
    }

    /// Whether the child level is marginal — whether a side of it carries a
    /// [`ValueRef`] rather than a node index.
    #[inline(always)]
    pub const fn is_marginal(self) -> bool {
        self.valued
    }

    /// What `side` denotes in the child level.
    #[inline(always)]
    pub fn child(self, side: EncodedChildRef) -> ChildRef {
        if self.valued {
            // Bare-is-slot: a side left over from before the child marginalized
            // is already a valid slot, so nothing needs re-tagging and only the
            // inline optimisation sets bit 30.
            ChildRef::Value(ValueRef::from_raw(MarginalSide(side.0)))
        } else {
            ChildRef::Node(NodeIdx(side.0))
        }
    }

    /// Decode a side known to point at a structural child.
    #[inline(always)]
    pub(crate) fn node(self, side: EncodedChildRef) -> NodeIdx {
        match self.child(side) {
            ChildRef::Node(index) => index,
            ChildRef::Value(_) => panic!("expected a structural child"),
        }
    }

    /// Rewrite `side` through a `remap` indexed by the child level's cells —
    /// the child was compacted and every cell moved to `remap[cell]`.
    ///
    /// An inline value names no cell of the child, so it passes through
    /// unchanged; a slot comes back re-tagged.
    #[inline]
    pub(crate) fn remap(self, side: EncodedChildRef, remap: &[u32]) -> EncodedChildRef {
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
            ChildRef::Node(NodeIdx(i)) => EncodedChildRef(remap[i as usize]),
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
    pub(crate) fn coord(self, side: EncodedChildRef) -> u32 {
        if self.valued && !side.is_reserved() {
            side.0 & MARGINAL_VALUE_MASK
        } else {
            side.0
        }
    }
}

/// Soundness precondition for [`TddLevel::become_marginal`]: both children
/// of the target vtree node `t` must already be marginal. Leaves count as
/// marginal (their counts are fixed by label), so a leaf target passes
/// vacuously.
///
/// # Panics
///
/// Panics if an internal child of `t` is not yet marginal; process targets
/// bottom-up so the precondition holds.
#[inline]
pub(crate) fn assert_can_make_marginal(
    levels: &[TddLevel],
    vtree: &crate::vtree::Vtree,
    t: crate::vtree::VtreeIdx,
) {
    use crate::vtree::VtreeNode;
    let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
        return;
    };
    for child in [left, right] {
        let is_leaf = matches!(*vtree.node(child), VtreeNode::Leaf { .. });
        if !is_leaf && !levels[child.idx()].is_marginal() {
            panic!(
                "become_marginal({}) precondition violated: child {} is internal \
                 but not yet marginal. Process marginalization targets bottom-up so \
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
    for_each_side_ref_mut, remap_refs_into, ChildSide, Sides,
};
pub(crate) use swap::resolve_swapped_marginal_side;
pub(crate) use tag::tag_all_marginal_side_slots;

#[cfg(test)]
mod tests;
