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
    /// The grouped path's state for the current outer key.
    round: Round,
    fast: u128,
    big: BigUint,
}

/// What the grouped path keeps for one outer key: the inner-g children the
/// round has opened, each with its weight and sum, and the outer-g children
/// with a live product under the key, each with that product's count.
///
/// Membership is a bit per child. The builds test it for every g pair they
/// walk, and most of those tests miss, so a miss reads one bit of a dense
/// set rather than a wide slot of its own. An opened child's weight and sum
/// sit in a slot numbered in the order the round opened it, which is the
/// order the round's closing sum reads them in. A round clears exactly the
/// bits it set.
#[derive(Default)]
struct Round {
    /// A bit per inner-g child: opened this round.
    opened: Vec<u64>,
    /// Per inner-g child opened this round, its slot.
    slot_of: Vec<u32>,
    /// The inner-g children opened this round, in slot order.
    keys: Vec<u32>,
    /// Per slot, the opened child's weight and sum.
    slots: Vec<Slot>,
    /// A bit per outer-g child: it has a live product this round.
    live: Vec<u64>,
    /// Per outer-g child with a live product this round, that product's count.
    live_count: Vec<u64>,
    /// The outer-g children set live this round.
    live_keys: Vec<u32>,
}

/// An opened inner-g child: the summed counts of the inner products that
/// read it (its weight), and the summed counts of the live outer products
/// its g pairs reach (its sum).
///
/// Each sums at most one `u64` per inner product or g pair, of which there
/// are fewer than `2^32` (the reverse indexes that list them refuse more), so
/// neither reaches `2^96`.
#[derive(Clone, Copy)]
struct Slot {
    weight: u128,
    sum: u128,
}

#[inline(always)]
fn has(set: &[u64], key: u32) -> bool {
    set[(key >> 6) as usize] >> (key & 63) & 1 != 0
}

#[inline(always)]
fn insert(set: &mut [u64], key: u32) {
    set[(key >> 6) as usize] |= 1 << (key & 63);
}

#[inline(always)]
fn remove(set: &mut [u64], key: u32) {
    set[(key >> 6) as usize] &= !(1 << (key & 63));
}

impl Round {
    /// The slot of inner-g child `key`, when this round has opened it.
    #[inline(always)]
    fn slot(&self, key: u32) -> Option<usize> {
        has(&self.opened, key).then(|| self.slot_of[key as usize] as usize)
    }

    /// Open inner-g child `key` with the given weight and sum.
    #[inline(always)]
    fn open(&mut self, lim: &Limits, key: u32, weight: u128, sum: u128) -> Result<(), OperationError> {
        insert(&mut self.opened, key);
        self.slot_of[key as usize] = self.keys.len() as u32;
        lim.try_push(&mut self.keys, key)?;
        lim.try_push(&mut self.slots, Slot { weight, sum })
    }

    /// Empty the round: every child it opened or set live is cleared.
    fn clear(&mut self) {
        for &key in &self.keys {
            remove(&mut self.opened, key);
        }
        for &key in &self.live_keys {
            remove(&mut self.live, key);
        }
        self.keys.clear();
        self.slots.clear();
        self.live_keys.clear();
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
            left, right, batch, cols, round: Round::default(), fast: 0,
            big: BigUint::ZERO,
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

    /// Size the round for `inner` inner-g children and `outer` outer-g
    /// children, with none opened or live.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when a reservation is refused.
    pub(crate) fn prepare_sums(&mut self, lim: &Limits, inner: usize, outer: usize) -> Result<(), OperationError> {
        let r = &mut self.round;
        r.opened.clear();
        r.live.clear();
        r.keys.clear();
        r.slots.clear();
        r.live_keys.clear();
        lim.try_resize(&mut r.opened, inner.div_ceil(64), 0u64)?;
        lim.try_resize(&mut r.slot_of, inner, 0u32)?;
        lim.try_resize(&mut r.live, outer.div_ceil(64), 0u64)?;
        lim.try_resize(&mut r.live_count, outer, 0u64)
    }

    /// Start a round: every inner-g child is closed and every outer-g child
    /// dead again.
    #[inline]
    pub(crate) fn begin_round(&mut self) {
        self.round.clear();
    }

    /// Record `count` as outer-g child `key`'s live product count this round.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the round's list cannot grow.
    #[inline(always)]
    pub(crate) fn set_outer(&mut self, lim: &Limits, key: u32, count: u64) -> Result<(), OperationError> {
        let r = &mut self.round;
        if !has(&r.live, key) {
            insert(&mut r.live, key);
            lim.try_push(&mut r.live_keys, key)?;
        }
        r.live_count[key as usize] = count;
        Ok(())
    }

    /// Outer-g child `key`'s live product count this round, 0 when it has
    /// no live product: what it adds to a sum.
    #[inline(always)]
    pub(crate) fn outer_or_zero(&self, key: u32) -> u64 {
        if has(&self.round.live, key) { self.round.live_count[key as usize] } else { 0 }
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

    /// How many inner-g children this round has opened.
    #[inline(always)]
    pub(crate) fn opened_len(&self) -> usize {
        self.round.keys.len()
    }

    /// The `i`th inner-g child this round opened.
    #[inline(always)]
    pub(crate) fn opened_key(&self, i: usize) -> u32 {
        self.round.keys[i]
    }

    /// Record the sum of the `i`th inner-g child this round opened.
    #[inline(always)]
    pub(crate) fn set_sum_at(&mut self, i: usize, sum: u128) {
        self.round.slots[i].sum = sum;
    }

    /// Add `count` to the sum of inner-g child `key`, opening it with that
    /// sum and no weight when this round has not.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the round's lists cannot grow.
    #[inline(always)]
    pub(crate) fn add_to_sum(&mut self, lim: &Limits, key: u32, count: u64) -> Result<(), OperationError> {
        match self.round.slot(key) {
            Some(i) => {
                self.round.slots[i].sum += u128::from(count);
                Ok(())
            }
            None => self.round.open(lim, key, 0, u128::from(count)),
        }
    }

    /// Weigh inner-g child `key` by one more inner product's `count`,
    /// opening it with an empty sum when this round has not; returns whether
    /// it was opened here.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the round's lists cannot grow.
    #[inline(always)]
    pub(crate) fn open_weighted(&mut self, lim: &Limits, key: u32, count: u64) -> Result<bool, OperationError> {
        match self.round.slot(key) {
            Some(i) => {
                self.round.slots[i].weight += u128::from(count);
                Ok(false)
            }
            None => {
                self.round.open(lim, key, u128::from(count), 0)?;
                Ok(true)
            }
        }
    }

    /// Add each opened inner-g child's weight times its sum: the candidates
    /// of the inner products that read it under the current outer key.
    #[inline]
    pub(crate) fn add_weighted(&mut self) {
        for i in 0..self.round.slots.len() {
            let Slot { weight, sum } = self.round.slots[i];
            match weight.checked_mul(sum) {
                Some(v) => self.add(v),
                None => self.big += BigUint::from(weight) * sum,
            }
        }
    }

    /// Add `count` to the sum of inner-g child `key` when this round has
    /// opened it ([`Self::open_weighted`]); a key nobody opened is not read.
    #[inline(always)]
    pub(crate) fn add_to_open(&mut self, key: u32, count: u64) {
        if let Some(i) = self.round.slot(key) {
            self.round.slots[i].sum += u128::from(count);
        }
    }

    /// Add inner product `prod`'s count times the sum of inner-g child `key`
    /// when this round has opened it: the product's candidates against a
    /// non-empty bucket. Most keys a walk tests are not opened, so the count
    /// is read only for one that is.
    #[inline(always)]
    pub(crate) fn add_grouped<const SWAPPED: bool>(&mut self, prod: u32, key: u32) {
        if let Some(i) = self.round.slot(key) {
            let count = self.inner_count::<SWAPPED>(prod);
            let sum = self.round.slots[i].sum;
            self.add_product(count, sum);
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
