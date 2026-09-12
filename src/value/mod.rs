//! Folding a level's values, in either domain the crate counts in.
//!
//! Both domains share one shape — `Σ over pairs (left × right)`, walked
//! bottom-up — and both are folded from two places: the marginal cascade after
//! a compile, and the streaming column inside an apply. [`IntFold`] and
//! [`WeightFold`] are the two arithmetics, [`MarginalFold`] the column contract
//! they share, and [`walk_bottom_up`] the walk that drives either one.
//!
//! The integer domain also needs a representation, which the weighted one does
//! not: a model count outgrows `u128`, and paying `BigUint` for every node
//! would be far more expensive than the rare overflow. So a count is a fast
//! `u128` with a sentinel value meaning "the real value is in the side table",
//! and the side table ([`CountOverflow`], which lives beside the marginal-ref
//! encoding it shares a slot space with) is sparse and built on the first
//! overflow. [`Count`] is one fold result, [`CountRead`] a borrowed read of a
//! stored slot, and [`CountVec`] the column — so the sentinel and its promotion
//! rule are written once.
//!
//! What is here is the scratch a fold works in, and the vocabulary a stored
//! column is described by — a hashable key for a slot, minting, interning, the
//! referenced set. The storage itself is elsewhere:
//! `TddLevel::marginal_counts` and the `WeightStore` are where a finished
//! column lands.

use crate::engine::Engine;
use crate::limits::{RecoveryPanic, ReservePolicy};
use std::marker::PhantomData;

use num_bigint::BigUint;

use crate::diagram::CountOverflow;

/// The "u128 fold overflowed" sentinel. A fold total equal to this value is
/// ambiguous between a true count of `u128::MAX` and an overflow whose real
/// value lives in the side table; [`Count::from_u128`] resolves it.
pub(crate) const COUNT_OVERFLOW: u128 = u128::MAX;

/// The scalar result of one integer count fold: either it fit in a `u128`, or
/// it overflowed into an exact arbitrary-precision count.
///
/// Also the key a marginal store is deduped by, which is the same two cases
/// asking the same question of a value: does it fit the fast lane, or does it
/// live in the side table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Count {
    Fast(u128),
    Big(BigUint),
}

impl Count {
    /// Build a `Count` from a raw `u128` fold total, applying the one
    /// exact-max promotion rule: a total that lands exactly on
    /// [`COUNT_OVERFLOW`] (`u128::MAX`) is indistinguishable from the
    /// overflow sentinel itself, so it is promoted to `Big` even though it
    /// numerically fits in a `u128`. No fold site outside this constructor
    /// may re-derive the `v == COUNT_OVERFLOW` check.
    pub(crate) fn from_u128(v: u128) -> Count {
        if v == COUNT_OVERFLOW {
            Count::Big(BigUint::from(u128::MAX))
        } else {
            Count::Fast(v)
        }
    }
}

/// A borrowed read of one stored [`CountVec`] slot — the decoded counterpart
/// of [`Count`] (which owns its `Big` value; this borrows it).
pub(crate) enum CountRead<'a> {
    Fast(u128),
    Big(&'a BigUint),
}

impl<'a> CountRead<'a> {
    /// Decode slot `i` of a `(fast, big)` column: a fast value equal to
    /// [`COUNT_OVERFLOW`] reads through to the big table, which holds an
    /// entry for every such slot. Every reader of a stored count decodes
    /// through here.
    #[inline(always)]
    pub(crate) fn from_slot(fast: &[u128], big: Option<&'a CountOverflow>, i: usize) -> Self {
        let raw = fast[i];
        if raw == COUNT_OVERFLOW {
            CountRead::Big(
                big.and_then(|b| b.get(i))
                    .expect("the overflow sentinel requires a big entry for the slot"),
            )
        } else {
            CountRead::Fast(raw)
        }
    }

    /// The owned value.
    #[inline]
    pub(crate) fn to_count(&self) -> Count {
        match *self {
            CountRead::Fast(c) => Count::Fast(c),
            CountRead::Big(b) => Count::Big(b.clone()),
        }
    }
}

/// The count column: a `width`-indexed sequence of [`Count`] values, stored as
/// a dense `u128` fast array plus a sparse, slot-keyed [`CountOverflow`] holding the
/// exact value of the slots that overflowed.
///
/// Invariants:
/// - `fast[i] == COUNT_OVERFLOW` ⇔ `big` holds an entry for slot `i`.
/// - The side table carries one entry per overflowing slot, never one per
///   slot, so a column with no overflow owns no side-table heap. See
///   [`CountOverflow`].
/// - `all_u64` is true only if every stored value fits in `u64`. It is
///   maintained incrementally and monotonically: a `Big` value or a `Fast`
///   value `> u64::MAX` clears it for good, even if that slot is later
///   overwritten with a small value.
///
/// `R: ReservePolicy` is the reservation policy every allocation goes through.
pub(crate) struct CountVec<R: ReservePolicy> {
    fast: Vec<u128>,
    big: Option<CountOverflow>,
    all_u64: bool,
    _res: PhantomData<R>,
}

impl<R: ReservePolicy> CountVec<R> {
    /// A fresh `width`-element column, all zeroed (0 fits `u64`, so
    /// `all_u64` starts `true`). Reserves exactly `width` before filling.
    pub(crate) fn try_with_width(eng: &Engine, width: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, width)?;
        fast.resize(width, 0u128);
        Ok(CountVec {
            fast,
            big: None,
            all_u64: true,
            _res: PhantomData,
        })
    }

    /// An empty column with `cap` slots reserved exactly up front (the
    /// streaming output column pre-reserves `left_width.max(right_width)` and then grows
    /// fallibly via [`Self::push`]).
    pub(crate) fn try_with_capacity(eng: &Engine, cap: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, cap)?;
        Ok(CountVec {
            fast,
            big: None,
            all_u64: true,
            _res: PhantomData,
        })
    }

    /// Borrow this column as a [`CountRef`], carrying the certificate rather
    /// than re-deriving it (a re-scan could disagree with the incrementally
    /// maintained flag on a column whose overflow slot was later overwritten).
    #[inline(always)]
    pub(crate) fn as_count_ref(&self) -> CountRef<'_> {
        CountRef {
            fast: &self.fast,
            big: self.big.as_ref(),
            all_u64: self.all_u64,
        }
    }

    /// Overwrite slot `i` (pre-sized fill; see [`Self::try_with_width`]).
    #[inline(always)]
    pub(crate) fn set(&mut self, eng: &Engine, i: usize, c: Count) -> Result<(), R::Err> {
        match c {
            Count::Fast(v) => {
                debug_assert!(
                    v != COUNT_OVERFLOW,
                    "Count::Fast carrying the overflow sentinel — Count::from_u128 should have promoted this to Big"
                );
                // A slot that stops overflowing must lose its side-table entry,
                // or the sentinel ⇔ entry invariant breaks; only a Big→Fast
                // transition touches the table.
                if std::mem::replace(&mut self.fast[i], v) == COUNT_OVERFLOW
                    && let Some(big) = self.big.as_mut() {
                        big.take(i);
                    }
                if v > u64::MAX as u128 {
                    self.all_u64 = false;
                }
            }
            Count::Big(b) => {
                self.fast[i] = COUNT_OVERFLOW;
                self.big
                    .get_or_insert_with(CountOverflow::default)
                    .try_insert::<R>(eng, i, b)?;
                self.all_u64 = false;
            }
        }
        Ok(())
    }

    /// Append one value, growing by amortized doubling. A `Big` append records
    /// one side-table entry under the new slot's index; a `Fast` append leaves
    /// the side table untouched.
    #[inline(always)]
    pub(crate) fn push(&mut self, eng: &Engine, c: Count) -> Result<(), R::Err> {
        R::reserve(eng, &mut self.fast, 1)?;
        match c {
            Count::Fast(v) => {
                debug_assert!(
                    v != COUNT_OVERFLOW,
                    "Count::Fast carrying the overflow sentinel — Count::from_u128 should have promoted this to Big"
                );
                self.fast.push(v);
                if v > u64::MAX as u128 {
                    self.all_u64 = false;
                }
            }
            Count::Big(b) => {
                self.fast.push(COUNT_OVERFLOW);
                let idx = self.fast.len() - 1;
                // Strictly ascending key ⇒ an O(1) amortized push inside `CountOverflow`.
                self.big
                    .get_or_insert_with(CountOverflow::default)
                    .try_insert::<R>(eng, idx, b)?;
                self.all_u64 = false;
            }
        }
        Ok(())
    }

    /// Decode slot `i`: a sentinel fast value reads through to the big table.
    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> CountRead<'_> {
        CountRead::from_slot(&self.fast, self.big.as_ref(), i)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.fast.len()
    }

    /// Storage-handoff escape hatch: unwrap into the raw `(fast, big)` pair at
    /// the `dedup_fresh_store`/`become_marginal` boundary. Both halves are exactly
    /// what `TddLevel` stores, so the handoff is a move — no re-shaping.
    pub(crate) fn into_parts(self) -> (Vec<u128>, Option<CountOverflow>) {
        (self.fast, self.big)
    }
}

/// The `all_u64` certificate of a raw fast column: every stored value fits in
/// `u64`. A sentinel (`COUNT_OVERFLOW`) or any value `> u64::MAX` fails it, so
/// `all_u64 ⇒ no overflow slot present`.
#[inline]
fn certify_all_u64(fast: &[u128]) -> bool {
    fast.iter().all(|&c| c <= u64::MAX as u128)
}

/// Borrowed twin of [`CountVec`]: a read-only view over raw `(fast, big)`
/// arrays plus their certificate, with the same decode rules and no ownership.
///
/// Lets a fold read a column that lives elsewhere (a `TddLevel`'s
/// `marginal_counts` storage, or another `CountVec`) without copying it; the
/// apply-side streaming child columns are wide marginal stores a copy would
/// double.
#[derive(Clone, Copy)]
pub(crate) struct CountRef<'a> {
    fast: &'a [u128],
    big: Option<&'a CountOverflow>,
    all_u64: bool,
}

impl<'a> CountRef<'a> {
    /// View raw arrays that are not a `CountVec` (a level's marginal storage,
    /// or the fixed leaf-label slots). The certificate is scanned
    /// ([`certify_all_u64`]).
    #[inline]
    pub(crate) fn from_parts_scanned(fast: &'a [u128], big: Option<&'a CountOverflow>) -> Self {
        CountRef {
            fast,
            big,
            all_u64: certify_all_u64(fast),
        }
    }

    /// Raw view of the fast column (sentinels included) — for the
    /// monomorphized unchecked-read fold fast path (`streaming_marginal::read_fast`).
    #[inline(always)]
    pub(crate) fn fast_slice(&self) -> &'a [u128] {
        self.fast
    }

    /// Raw fast-slot read (sentinel included, no big-table decode).
    #[inline(always)]
    pub(crate) fn fast_val(&self, i: usize) -> u128 {
        self.fast[i]
    }

    /// Raw big-table read; `None` when slot `i` has no overflow value.
    #[inline(always)]
    pub(crate) fn big_val(&self, i: usize) -> Option<&'a BigUint> {
        self.big.and_then(|v| v.get(i))
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.fast.len()
    }

    #[inline(always)]
    pub(crate) fn all_u64(&self) -> bool {
        self.all_u64
    }
}

mod domain;
pub(crate) mod read;
mod fold;
pub(crate) mod slots;
mod stream_cache;

pub use fold::*;
pub(crate) use domain::{Column, FoldInput, FoldScope, SlotStore, StreamChild, ValueDomain};
pub(crate) use stream_cache::StreamCache;

impl CountVec<RecoveryPanic> {
    /// Infallible convenience wrapper (`RecoveryPanic::Err = Infallible`, so
    /// the fallible form can never actually return `Err` — it panics first).
    pub(crate) fn with_width(eng: &Engine, width: usize) -> Self {
        unwrap_infallible(Self::try_with_width(eng, width))
    }

    pub(crate) fn set_i(&mut self, eng: &Engine, i: usize, c: Count) {
        unwrap_infallible(self.set(eng, i, c))
    }
}

#[cfg(test)]
mod tests;
