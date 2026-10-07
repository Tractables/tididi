//! The overflow half of a count-marginal level's store.

use num_bigint::BigUint;

use crate::Engine;
use crate::limits::OperationError;

/// The exact `BigUint` value of every count slot whose fast `u128` cell holds
/// the `u128::MAX` overflow sentinel — the overflow half of a marginal count
/// store (`TddLevel::marginal_counts` / `marginal_counts_big`). A reader
/// decoding a marginal level reads it through [`get`](Self::get), and
/// [`len`](Self::len) / [`is_empty`](Self::is_empty) say how many slots
/// overflowed at all.
///
/// Keyed by slot index, not parallel to the fast column: a slot lands here
/// only when its count reaches `u128::MAX`, so the cost is proportional to the
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

#[cfg(test)]
#[path = "tests/count_overflow.rs"]
mod tests;
