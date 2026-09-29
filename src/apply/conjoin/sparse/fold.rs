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

use super::{prefetch_at, ChildPair, GroupedView, ProductEntry, RevEntry, WALK_AHEAD};
use crate::limits::{Limits, OperationError, PollGate};
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
    /// The keys this level's filtered walks have tested, and how many of
    /// them passed; see [`Self::walk_grouped`].
    filter_tested: u64,
    filter_passed: u64,
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
    /// A bit per hash of an opened inner-g child, set for every child the
    /// round has opened when its walk runs filtered ([`Round::build_filter`]):
    /// a clear bit says the child is not opened, from a set small enough to
    /// stay in the nearest cache.
    filter: Vec<u64>,
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

/// Most inner-g children a round may have opened for its walk to test each
/// key against the round's filter first ([`Round::build_filter`]); a round
/// with more tests the bit set alone.
const FILTER_KEYS: usize = 1 << 15;

/// Filter bits per opened child, as a power of two: with 64, a key the round
/// has not opened passes the filter about once in 64 walks.
const FILTER_BITS_PER_KEY_LOG: u32 = 6;

/// The fewest filter bits a round uses: eight words.
const FILTER_MIN_BITS_LOG: u32 = 9;

/// How many keys a level's filtered walks test before the share that passes
/// decides whether the rest of the level still filters.
const FILTER_SAMPLE: u64 = 1 << 16;

/// A level whose filtered walks pass more than one key in this many walks
/// the rest unfiltered: most of its keys are opened, so the filter's test
/// adds to the exact one rather than saving it.
const FILTER_PASS_SHARE: u64 = 8;

/// The fewest bytes a level's count column spans for its walks to filter:
/// below it a hit's exact reads mostly find the nearest caches, so the
/// filter's test and the ring of held hits only add to them.
const FILTER_MIN_BYTES: usize = 16 << 20;

/// When a round filters its walk: the most opened children it filters over,
/// the filter's size, how many keys a level tests before its pass share is
/// read, and the fewest bytes its count column spans, the last three as a
/// test pins them.
#[inline(always)]
fn filter_policy() -> (usize, Option<u32>, u64, usize) {
    forced_filter().unwrap_or((FILTER_KEYS, None, FILTER_SAMPLE, FILTER_MIN_BYTES))
}

// Tests pin the filter and count the walks; production always sizes it.
#[cfg(test)]
use super::tests::{forced_filter, note_walk};

#[cfg(not(test))]
#[inline(always)]
fn forced_filter() -> Option<(usize, Option<u32>, u64, usize)> {
    None
}

#[cfg(not(test))]
#[inline(always)]
fn note_walk(_kind: usize) {}

/// The filter bit of `key` in a filter of `2^bits` bits: the top bits of a
/// multiplicative hash, which spreads runs of consecutive keys.
#[inline(always)]
fn filter_bit(key: u32, bits: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> (32 - bits)) as usize
}

/// How many bits the filter over `keys` opened children takes, as a power
/// of two, for `1 <= keys <= FILTER_KEYS`.
fn filter_bits_log(keys: usize) -> u32 {
    let ceil_log = usize::BITS - (keys - 1).leading_zeros();
    (ceil_log + FILTER_BITS_PER_KEY_LOG).max(FILTER_MIN_BITS_LOG)
}

/// How many hits a filtered walk holds before it reads them
/// ([`CandidateFold::walk_grouped`]): enough that the reads each one asked
/// the cache for have arrived by the time it is taken.
const HITS_HELD: usize = 32;

/// The keys a filtered walk found to pass the filter, with their products,
/// in a ring: each is read once `HITS_HELD` more have passed, or when the
/// walk ends.
struct Held {
    keys: [u32; HITS_HELD],
    prods: [u32; HITS_HELD],
    len: usize,
    head: usize,
}

/// A running total, `fast` spilling into `big` as [`CandidateFold::add`]
/// does.
#[derive(Default)]
struct Total {
    fast: u128,
    big: BigUint,
}

impl Total {
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

    /// Add `count × sum`, a sum below `2^96`, as
    /// [`CandidateFold::add_product`] does.
    #[inline(always)]
    fn add_product(&mut self, count: u64, sum: u128) {
        if sum >> 64 == 0 {
            self.add(count as u128 * sum);
        } else {
            match (count as u128).checked_mul(sum) {
                Some(v) => self.add(v),
                None => self.big += BigUint::from(count) * sum,
            }
        }
    }
}

impl Round {
    /// Set the filter's bits for every child this round has opened, in a
    /// filter of `2^bits` bits, clearing the rest.
    fn build_filter(&mut self, bits: u32) {
        let words = 1usize << (bits - 6);
        self.filter[..words].fill(0);
        for &key in &self.keys {
            let bit = filter_bit(key, bits);
            self.filter[bit >> 6] |= 1 << (bit & 63);
        }
    }

    /// Take a hit a filtered walk held: inner product `prod`, read from
    /// `col`, against child `key` when the round opened it, which the
    /// filter could not tell.
    #[inline(always)]
    fn take_hit(&self, col: &[u128], key: u32, prod: u32, total: &mut Total) {
        if let Some(i) = self.slot(key) {
            total.add_product(col[prod as usize] as u64, self.slots[i].sum);
        }
    }

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
            big: BigUint::ZERO, filter_tested: 0, filter_passed: 0,
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

    /// Whether a level whose inner side is the right child when `SWAPPED`
    /// runs its walks through [`Self::walk_grouped`]: its count column
    /// spans at least [`FILTER_MIN_BYTES`]. Below that a hit's exact reads
    /// mostly find the nearest caches, so the filter's test and the ring of
    /// held hits would only add to them. The grouped path only.
    pub(crate) fn filters_walks<const SWAPPED: bool>(&self) -> bool {
        let wide = std::mem::size_of_val(self.column(usize::from(SWAPPED))) >= filter_policy().3;
        if !wide {
            note_walk(4);
        }
        wide
    }

    /// Size the filter of a level whose walks run through
    /// [`Self::walk_grouped`], over `inner` inner-g children, for a level
    /// whose walks have tested no key yet.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the reservation is refused.
    pub(crate) fn prepare_filter(&mut self, lim: &Limits, inner: usize) -> Result<(), OperationError> {
        let most = inner.clamp(1, filter_policy().0.max(1));
        lim.try_resize(&mut self.round.filter, 1usize << (filter_bits_log(most) - 6), 0u64)?;
        self.filter_tested = 0;
        self.filter_passed = 0;
        Ok(())
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

    /// The walk of one outer key's candidates once every opened child's sum
    /// is final: for each f pair in `under`, every product of its inner
    /// child in `inner` adds its count times its inner-g child's sum when
    /// the round opened that child ([`Self::add_grouped`]), asking the cache
    /// for the bucket of the pair [`WALK_AHEAD`] on.
    ///
    /// Most of the keys it tests are usually not opened, and the bit set
    /// that says so is as wide as the level. A round that opened few
    /// children first tests each key against a filter over its own opened
    /// set, which stays in the nearest cache, and holds each key that
    /// passes, asking for what testing it exactly reads, until
    /// [`HITS_HELD`] more have passed. A key the filter stops is not opened,
    /// so it would have added nothing, and the sums are final, so a held hit
    /// adds what it would have when found. Once a level's filtered walks
    /// have tested [`FILTER_SAMPLE`] keys, a level that passed more than one
    /// in [`FILTER_PASS_SHARE`] walks the rest unfiltered. Only a level
    /// whose count column is wide enough walks here
    /// ([`Self::filters_walks`]).
    ///
    /// # Errors
    ///
    /// What the ticker's poll returns.
    pub(super) fn walk_grouped<const SWAPPED: bool>(
        &mut self,
        under: &[RevEntry],
        inner: GroupedView<'_, ProductEntry>,
        ticker: &mut PollGate<'_>,
    ) -> Result<(), OperationError> {
        let opened = self.round.keys.len();
        if opened == 0 {
            return Ok(());
        }
        let (most, pinned_bits, sample, _) = filter_policy();
        let col = self.column(usize::from(SWAPPED));
        let passing = self.filter_tested >= sample
            && self.filter_passed.saturating_mul(FILTER_PASS_SHARE) > self.filter_tested;
        if opened > most || passing {
            note_walk(if passing { 3 } else { 1 });
            for (j, &RevEntry { other: inner1, .. }) in under.iter().enumerate() {
                if let Some(ahead) = under.get(j + WALK_AHEAD) {
                    inner.prefetch_bucket(ahead.other as usize);
                }
                let products = inner.bucket(inner1 as usize);
                for e in products {
                    self.add_grouped::<SWAPPED>(e.prod_idx.0, e.g_idx.0);
                }
                ticker.poll(products.len() as u64)?;
            }
            return Ok(());
        }
        note_walk(0);
        let bits = pinned_bits.unwrap_or_else(|| filter_bits_log(opened));
        self.round.build_filter(bits);
        let round = &self.round;
        let filter = &round.filter[..1usize << (bits - 6)];
        let mut held = Held { keys: [0; HITS_HELD], prods: [0; HITS_HELD], len: 0, head: 0 };
        let mut total = Total::default();
        let (mut tested, mut passed) = (0u64, 0u64);
        for (j, &RevEntry { other: inner1, .. }) in under.iter().enumerate() {
            if let Some(ahead) = under.get(j + WALK_AHEAD) {
                inner.prefetch_bucket(ahead.other as usize);
            }
            let products = inner.bucket(inner1 as usize);
            tested += products.len() as u64;
            for e in products {
                let key = e.g_idx.0;
                let bit = filter_bit(key, bits);
                if filter[bit >> 6] >> (bit & 63) & 1 == 0 {
                    continue;
                }
                passed += 1;
                let prod = e.prod_idx.0;
                prefetch_at(&round.opened, (key >> 6) as usize);
                prefetch_at(&round.slot_of, key as usize);
                prefetch_at(col, prod as usize);
                let at = held.head;
                if held.len == HITS_HELD {
                    if at == 0 {
                        note_walk(2);
                    }
                    round.take_hit(col, held.keys[at], held.prods[at], &mut total);
                } else {
                    held.len += 1;
                }
                held.keys[at] = key;
                held.prods[at] = prod;
                held.head = (at + 1) % HITS_HELD;
            }
            ticker.poll(products.len() as u64)?;
        }
        for i in 0..held.len {
            round.take_hit(col, held.keys[i], held.prods[i], &mut total);
        }
        self.filter_tested += tested;
        self.filter_passed += passed;
        self.add(total.fast);
        self.big += total.big;
        Ok(())
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
