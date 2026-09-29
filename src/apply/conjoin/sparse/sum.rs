//! Summing one child of a one-product root out as the root's pairs are found.
//!
//! [`Engine::and_marginalizing`](crate::Engine::and_marginalizing) with one
//! target `c`, a child of the root, can build the root in full — one pair
//! `(s, n)` per candidate the scatter finds, `s` a product of the root's
//! other child and `n` one of `c` — and then, once the sweep is over, sum `c`
//! out and fuse the pairs that share an `s` into one pair carrying the sum of
//! their counts. On a large join most of the root's candidates are fused away
//! again: the wide level is built only to be folded. [`ChildSum`] builds the
//! folded level instead. `c`'s column is folded before the scatter, each
//! candidate adds its `n`'s count to its `s` as the scatter finds it, and the
//! root is written from the sums, in the two-step path's order and with its
//! references (see [`ChildSum::write_root`]).

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rustc_hash::FxHashMap;

use super::{finish_direct, try_push_pair_into, ChildPair, OperationError, ProductEntry, TddLevel};
use crate::diagram::{ChildSide, EncodedChildRef, ValueRef};
use crate::limits::{Limits, Transient};
use crate::value::slots::{push_count_slot, seed_count_slots, SlotValues};
use crate::value::{Count, CountRead, CountVec, IntFold};
use crate::Engine;

/// A kept product no candidate has reached ([`ChildSum::acc`]).
const UNREACHED: u64 = 0;

/// A kept product whose sum is held exactly in [`ChildSum::exact`]; in
/// [`ChildSum::fast`], a node whose count is read from the column instead.
const EXACT: u64 = u64::MAX;

/// Set in a kept product's word once a second candidate has reached it.
const MULTI: u64 = 1 << 63;

/// The sum bits of a kept product's word.
const SUM: u64 = MULTI - 1;

/// The largest sum a word carries: two of them add without reaching
/// [`MULTI`], so an add never carries into the flag.
const FAST_MAX: u64 = (1 << 62) - 1;

/// The running sums of a one-product root's candidates, one per product of
/// the child that is kept, each over the counts of the summed child's nodes
/// the candidates pair it with.
///
/// Every candidate the scatter finds is `(s, n)`, with `s` a product of the
/// kept child and `n` a node of the summed one; [`Self::push`] adds `n`'s
/// count to `s`'s sum. A sum and whether more than one candidate reached it
/// share one word of [`Self::acc`] while the sum stays at most [`FAST_MAX`],
/// which covers any count of a diagram that fits in memory many times over;
/// past it, or for a count the column holds beyond it, the sum moves to
/// [`Self::exact`], so no sum is ever rounded.
pub(crate) struct ChildSum {
    /// The summed child's side of the root's pairs.
    side: ChildSide,
    /// Per node of the summed child, its count; [`EXACT`] where the count is
    /// 0 or above [`FAST_MAX`] and is read from [`Self::col`] instead.
    fast: Vec<u64>,
    /// The summed child's column.
    col: CountVec,
    /// Per product of the kept child: [`UNREACHED`], [`EXACT`], or the sum
    /// with [`MULTI`] set once a second candidate reached it.
    acc: Vec<u64>,
    /// The kept products in the order candidates first reached them.
    order: Vec<u32>,
    /// The sums held exactly, each with whether more than one candidate
    /// reached it.
    exact: FxHashMap<u32, (BigUint, bool)>,
}

impl ChildSum {
    /// Sums over the `kept_width` products of the kept child, of the counts
    /// in `col`, the summed child's column, which sits on `side` of the
    /// root's pairs.
    pub(crate) fn new(lim: &Limits, side: ChildSide, col: CountVec, kept_width: usize) -> Result<ChildSum, OperationError> {
        let mut fast = Vec::new();
        lim.try_resize(&mut fast, col.len(), EXACT)?;
        for (n, slot) in fast.iter_mut().enumerate() {
            if let CountRead::Fast(c) = col.get(n)
                && (1..=u128::from(FAST_MAX)).contains(&c)
            {
                *slot = c as u64;
            }
        }
        let mut acc = Vec::new();
        lim.try_resize(&mut acc, kept_width, UNREACHED)?;
        Ok(ChildSum { side, fast, col, acc, order: Vec::new(), exact: FxHashMap::default() })
    }

    /// Add one candidate: its summed-side node's count to its kept product's sum.
    #[inline(always)]
    pub(super) fn push(&mut self, lim: &Limits, pair: ChildPair) -> Result<(), OperationError> {
        let (kept, summed) = match self.side {
            ChildSide::Right => (pair.left.0, pair.right.0),
            ChildSide::Left => (pair.right.0, pair.left.0),
        };
        let count = self.fast[summed as usize];
        let word = self.acc[kept as usize];
        if count != EXACT {
            if word == UNREACHED {
                self.acc[kept as usize] = count;
                return lim.try_push(&mut self.order, kept);
            }
            if word != EXACT {
                let sum = (word & SUM) + count;
                if sum <= FAST_MAX {
                    self.acc[kept as usize] = sum | MULTI;
                    return Ok(());
                }
            }
        }
        self.push_exact(lim, kept, summed)
    }

    /// [`Self::push`] where the count or the sum leaves the word: the sum
    /// continues in [`Self::exact`].
    #[cold]
    #[inline(never)]
    fn push_exact(&mut self, lim: &Limits, kept: u32, summed: u32) -> Result<(), OperationError> {
        let count = match self.col.get(summed as usize) {
            CountRead::Fast(c) => BigUint::from(c),
            CountRead::Big(b) => b.clone(),
        };
        let word = self.acc[kept as usize];
        let (prior, reached) = match word {
            UNREACHED => {
                lim.try_push(&mut self.order, kept)?;
                (BigUint::default(), false)
            }
            EXACT => self.exact.remove(&kept).map(|(sum, _)| (sum, true)).expect("an exact sum is held"),
            word => (BigUint::from(word & SUM), true),
        };
        lim.reserve_map(&mut self.exact, 1)?;
        self.exact.insert(kept, (prior + count, reached));
        self.acc[kept as usize] = EXACT;
        Ok(())
    }

    /// Whether no candidate was found: the root's one product is false.
    pub(crate) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The summed child's column, handed back once the scatter is done.
    pub(crate) fn take_column(&mut self) -> CountVec {
        std::mem::take(&mut self.col)
    }

    /// Kept product `s`'s sum, as the two-step path's pair fusion sums it
    /// ([`Count::from_u128`] promotion included), and whether more than one
    /// candidate reached it.
    fn sum_of(&self, s: u32) -> (Count, bool) {
        match self.acc[s as usize] {
            EXACT => {
                let (sum, reached) = &self.exact[&s];
                let count = match sum.to_u128() {
                    Some(v) => Count::from_u128(v),
                    None => Count::Big(sum.clone()),
                };
                (count, *reached)
            }
            word => (Count::Fast(u128::from(word & SUM)), word & MULTI != 0),
        }
    }

    /// Write the root's one node from the sums, with `summed` the summed
    /// child's level, installed marginal with its deduplicated column; `base`
    /// is where the node's pairs start in `root`'s arena.
    ///
    /// The pairs are the ones the two-step path leaves, in its order and with
    /// its references. Built in full, the root holds one pair per candidate,
    /// in the order the scatter finds them, and every candidate is a
    /// distinct pair (distinct products of the two operands' disjoint
    /// nodes), so each group of pairs sharing a kept product has distinct
    /// nodes of the summed child. Summing that child out gives each pair its
    /// node's count; pair fusion keeps a kept product that one pair names
    /// where it was and replaces each other group by one pair carrying the
    /// group's sum, appended in the order the pairs first name the groups.
    /// So the pairs are the kept products one candidate reached, in the
    /// order candidates first reached them, and then the others, in the same
    /// order. A reference is the one the two-step path gives a value: inline
    /// where it fits one, else the slot of the column holding it, which is
    /// unique since the column is deduplicated, else a slot appended to the
    /// column, in the order the fused groups are written. A lone pair's
    /// value is its node's count, which the column holds, so only a fused
    /// group ever appends one, as in the two-step path.
    pub(crate) fn write_root(
        &self,
        eng: &Engine,
        root: &mut TddLevel,
        summed: &mut TddLevel,
        base: usize,
        pl_output: &mut Vec<ProductEntry>,
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        // Charged as it grows and handed back when the node is written, as
        // pair fusion charges the map it resolves values by.
        let mut by_value: Transient<'_, FxHashMap<Count, u32>> = Transient::new(lim, FxHashMap::default());
        let mut seeded = false;
        let mut stride = 0u64;
        for fused in [false, true] {
            for &s in &self.order {
                let (value, reached) = self.sum_of(s);
                if reached != fused {
                    continue;
                }
                let raw = match <IntFold as SlotValues>::inline_ref(&value) {
                    Some(raw) => raw,
                    None => {
                        if !seeded {
                            seed_count_slots(lim, summed, &mut by_value)?;
                            seeded = true;
                        }
                        match by_value.get(&value) {
                            Some(&slot) => ValueRef::slot_raw(slot),
                            None => {
                                let slot = push_count_slot(eng, summed, value.clone())?;
                                lim.reserve_map(&mut by_value, 1)?;
                                by_value.insert(value, slot);
                                ValueRef::slot_raw(slot)
                            }
                        }
                    }
                };
                let pair = match self.side {
                    ChildSide::Right => ChildPair::new(EncodedChildRef::from_raw(s), EncodedChildRef::from_raw(raw)),
                    ChildSide::Left => ChildPair::new(EncodedChildRef::from_raw(raw), EncodedChildRef::from_raw(s)),
                };
                try_push_pair_into(eng, root, pair)?;
                stride += 1;
                if stride == 1 << 20 {
                    stride = 0;
                    lim.check_stop()?;
                }
            }
        }
        finish_direct(eng, root, base, pl_output, false)?;
        // Every reference on the summed side is a value, as the two-step
        // path's end-of-pass tagger marks it.
        root.set_has_value_refs(self.side, true);
        Ok(())
    }
}
