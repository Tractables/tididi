//! Counting a one-product level's candidates instead of storing them.
//!
//! On a level where f and g have one node each, every candidate the scatter
//! finds is a pair of the level's one product (see [`Collect::Direct`]). When
//! that product is the diagram's output and only its count is wanted, the
//! pairs need not be kept: the count is `Σ left × right` over them, with each
//! side read from its child's count column, which is the fold a model count
//! would run over the stored node.
//!
//! The general arm groups the sum further (`count_general_arm`): under one
//! outer key the candidates of an inner product contribute its count times
//! one sum per inner-g child, so no candidate is ever listed.

use num_bigint::BigUint;

use super::ChildPair;
use crate::limits::{Limits, OperationError};
use crate::value::{IntFold, StreamChild, ValueDomain};

/// How many candidates [`CandidateFold::push`] holds before folding them.
const BATCH: usize = 1024;

/// The running count of a one-product level's candidates.
///
/// The two views are the level's child count columns. When both are
/// structural and carry the `u64` certificate the grouped path of the
/// general arm is open ([`Self::grouped`]) and reads them as plain `u64`
/// arrays; otherwise every candidate goes through [`Self::push`], whose
/// batches fold through the exact cell fold.
pub(crate) struct CandidateFold<'a> {
    left: StreamChild<'a, IntFold>,
    right: StreamChild<'a, IntFold>,
    batch: Vec<ChildPair>,
    /// The two count columns, left then right, as raw slices: set exactly
    /// when the grouped path is open, so each read is one load.
    cols: Option<[&'a [u128]; 2]>,
    /// Per inner-g child, the summed counts of the live outer products its
    /// g pairs reach under the current outer key, stamped with the round
    /// that summed it ([`Stamped`]).
    sums: Vec<u128>,
    /// Per outer-g child, the count of its live product under the current
    /// outer key, stamped likewise: what a sum by inner child reads.
    outer: Vec<u128>,
    /// The current round; slots stamped with another hold nothing.
    round: u32,
    fast: u128,
    big: BigUint,
}

/// A sum slot: the round that wrote it in the top 32 bits, the sum in the
/// low 96. One word, so marking a key and summing into it touch one place.
///
/// A sum never reaches the stamp: each term is a `u64` count
/// ([`CandidateFold::grouped`]) and a key sums at most one term per g pair,
/// of which there are fewer than `2^32` (the reverse indexes that list them
/// refuse more), so every sum is below `2^96`.
struct Stamped;

impl Stamped {
    const SUM_BITS: u32 = 96;
    const SUM: u128 = (1 << Self::SUM_BITS) - 1;

    #[inline(always)]
    fn round(slot: u128) -> u32 {
        (slot >> Self::SUM_BITS) as u32
    }

    #[inline(always)]
    fn slot(round: u32, sum: u128) -> u128 {
        debug_assert!(sum <= Self::SUM, "a bucket sum reached its stamp");
        (round as u128) << Self::SUM_BITS | sum
    }
}

impl<'a> CandidateFold<'a> {
    /// A zero count over the given child columns, with its batch reserved.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the batch reservation is refused.
    pub(crate) fn new(
        lim: &Limits,
        left: StreamChild<'a, IntFold>,
        right: StreamChild<'a, IntFold>,
    ) -> Result<Self, OperationError> {
        let mut batch = Vec::new();
        lim.reserve_exact(&mut batch, BATCH)?;
        // A structural view reads a reference as the slot it names, and the
        // certificate rules out the overflow sentinel, so a certified
        // structural column is its raw slice.
        let raw = |side: &StreamChild<'a, IntFold>| {
            (!side.view.is_marginal() && side.col.all_u64()).then(|| side.col.fast_slice())
        };
        let cols = raw(&left).zip(raw(&right)).map(|(l, r)| [l, r]);
        Ok(CandidateFold {
            left, right, batch, cols, sums: Vec::new(), outer: Vec::new(), round: 0, fast: 0, big: BigUint::ZERO,
        })
    }

    /// Fold one candidate in.
    #[inline(always)]
    pub(crate) fn push(&mut self, pair: ChildPair) {
        // The batch was reserved at `BATCH` and is folded when it fills, so
        // this push never grows it.
        self.batch.push(pair);
        if self.batch.len() == BATCH {
            self.flush();
        }
    }

    #[inline(never)]
    fn flush(&mut self) {
        let count = IntFold::fold_cell(&self.batch, &self.left, &self.right, &());
        self.batch.clear();
        match count {
            crate::value::Count::Fast(v) => self.add(v),
            crate::value::Count::Big(v) => self.big += v,
        }
    }

    #[inline(always)]
    fn add(&mut self, v: u128) {
        match self.fast.checked_add(v) {
            Some(sum) => self.fast = sum,
            None => {
                self.big += self.fast;
                self.fast = v;
            }
        }
    }

    /// Whether the general arm may sum per inner-g child before multiplying:
    /// both columns are structural and every count on them fits `u64`.
    pub(crate) fn grouped(&self) -> bool {
        self.cols.is_some()
    }

    /// Size the bucket sums for `sums` inner-g children and the outer
    /// counts for `outer` outer-g children, all unstamped.
    pub(crate) fn prepare_sums(&mut self, lim: &Limits, sums: usize, outer: usize) -> Result<(), OperationError> {
        self.sums.clear();
        self.outer.clear();
        self.round = 0;
        lim.try_resize(&mut self.sums, sums, 0u128)?;
        lim.try_resize(&mut self.outer, outer, 0u128)
    }

    /// Start a round: every bucket sum and outer count is empty again.
    #[inline]
    pub(crate) fn begin_round(&mut self) {
        self.round = self.round.wrapping_add(1);
        if self.round == 0 {
            // The stamp wrapped, so a slot left by an older round could read
            // as this one's. One pass per 2^32 rounds.
            self.sums.fill(0);
            self.outer.fill(0);
            self.round = 1;
        }
    }

    /// Record `count` as outer-g child `key`'s live product count this round.
    #[inline(always)]
    pub(crate) fn set_outer(&mut self, key: u32, count: u64) {
        self.outer[key as usize] = Stamped::slot(self.round, u128::from(count));
    }

    /// Outer-g child `key`'s live product count this round, 0 when it has
    /// no live product: what it adds to a sum.
    #[inline(always)]
    pub(crate) fn outer_or_zero(&self, key: u32) -> u64 {
        let slot = self.outer[key as usize];
        if Stamped::round(slot) == self.round { slot as u64 } else { 0 }
    }

    /// The count of product `prod` on the inner side (the left child, or the
    /// right one when `SWAPPED`); the grouped path only.
    #[inline(always)]
    pub(crate) fn inner_count<const SWAPPED: bool>(&self, prod: u32) -> u64 {
        self.column(usize::from(SWAPPED))[prod as usize] as u64
    }

    /// The count of product `prod` on the outer side; the grouped path only.
    #[inline(always)]
    pub(crate) fn outer_count<const SWAPPED: bool>(&self, prod: u32) -> u64 {
        self.column(usize::from(!SWAPPED))[prod as usize] as u64
    }

    #[inline(always)]
    fn column(&self, side: usize) -> &'a [u128] {
        self.cols.expect("the grouped path reads raw columns")[side]
    }

    /// Whether inner-g child `key` has a sum this round.
    #[inline(always)]
    pub(crate) fn has_sum(&self, key: u32) -> bool {
        Stamped::round(self.sums[key as usize]) == self.round
    }

    /// Record the summed count of inner-g child `key`'s bucket this round.
    #[inline(always)]
    pub(crate) fn set_sum(&mut self, key: u32, sum: u128) {
        self.sums[key as usize] = Stamped::slot(self.round, sum);
    }

    /// Add `count` to the sum of inner-g child `key`, which starts this
    /// round's sum when the key has none yet.
    #[inline(always)]
    pub(crate) fn add_to_sum(&mut self, key: u32, count: u64) {
        let slot = &mut self.sums[key as usize];
        if Stamped::round(*slot) == self.round {
            *slot += count as u128;
        } else {
            *slot = Stamped::slot(self.round, count as u128);
        }
    }

    /// Give inner-g child `key` an empty sum this round, unless it has one.
    #[inline(always)]
    pub(crate) fn open_sum(&mut self, key: u32) {
        let slot = &mut self.sums[key as usize];
        if Stamped::round(*slot) != self.round {
            *slot = Stamped::slot(self.round, 0);
        }
    }

    /// Add `count` to the sum of inner-g child `key` when it has one this
    /// round ([`Self::open_sum`]); a key nobody opened is not read.
    #[inline(always)]
    pub(crate) fn add_to_open(&mut self, key: u32, count: u64) {
        let slot = &mut self.sums[key as usize];
        if Stamped::round(*slot) == self.round {
            *slot += u128::from(count);
        }
    }

    /// Add `count × sum(key)` when `key` has a sum this round: an inner
    /// product's candidates against a non-empty bucket.
    #[inline(always)]
    pub(crate) fn add_grouped(&mut self, count: u64, key: u32) {
        let slot = self.sums[key as usize];
        if Stamped::round(slot) == self.round {
            self.add_product(count, slot & Stamped::SUM);
        }
    }

    /// Add `count × sum`, a sum below `2^96`.
    #[inline(always)]
    fn add_product(&mut self, count: u64, sum: u128) {
        if sum >> 64 == 0 {
            // One widening multiply; the product fits `u128`.
            self.add(count as u128 * sum);
        } else {
            match (count as u128).checked_mul(sum) {
                Some(v) => self.add(v),
                None => self.big += BigUint::from(count) * sum,
            }
        }
    }

    /// The count of every candidate folded in.
    pub(crate) fn finish(mut self) -> BigUint {
        if !self.batch.is_empty() {
            self.flush();
        }
        self.big + self.fast
    }
}
