use super::*;

use std::cell::Cell;
use std::sync::Arc;

use crate::vtree::{VarId, Vtree};

thread_local! {
    /// The thresholds a [`ForcedThresholds`] guard on this thread has installed.
    static FORCED: Cell<Option<SparseThresholds>> = const { Cell::new(None) };
}

/// The thresholds a guard on this thread has installed, if any.
pub(super) fn forced_thresholds() -> Option<SparseThresholds> {
    FORCED.with(Cell::get)
}

/// Thresholds in force on this thread until the guard drops, which restores
/// whatever was in force before it.
pub(super) struct ForcedThresholds {
    prior: Option<SparseThresholds>,
}

impl ForcedThresholds {
    pub(super) fn install(thresholds: SparseThresholds) -> ForcedThresholds {
        ForcedThresholds { prior: FORCED.with(|c| c.replace(Some(thresholds))) }
    }
}

impl Drop for ForcedThresholds {
    fn drop(&mut self) {
        FORCED.with(|c| c.set(self.prior));
    }
}

thread_local! {
    /// The prefetch gate a [`ForcedPrefetch`] guard on this thread has
    /// installed.
    static PREFETCH: Cell<Option<bool>> = const { Cell::new(None) };
    /// How many f pairs on this thread had their reads asked for ahead, per
    /// walk (see [`prefetches_run`]).
    static PREFETCHED: Cell<[u64; 2]> = const { Cell::new([0; 2]) };
}

/// The pinned prefetch gate, if any: `Some(true)` has every scatter walk ask
/// the cache ahead whatever its level spans, `Some(false)` none.
pub(super) fn forced_prefetch() -> Option<bool> {
    PREFETCH.with(Cell::get)
}

/// The scatter's prefetch gate pinned on this thread until the guard drops;
/// `None` leaves it to the level's size.
pub(super) struct ForcedPrefetch {
    prior: Option<bool>,
}

impl ForcedPrefetch {
    pub(super) fn install(gate: Option<bool>) -> ForcedPrefetch {
        ForcedPrefetch { prior: PREFETCH.with(|c| c.replace(gate)) }
    }
}

impl Drop for ForcedPrefetch {
    fn drop(&mut self) {
        PREFETCH.with(|c| c.set(self.prior));
    }
}

/// Count one read asked for ahead at `site` (see [`prefetches_run`]).
pub(super) fn note_prefetch(site: usize) {
    PREFETCHED.with(|c| {
        let mut seen = c.get();
        seen[site] += 1;
        c.set(seen);
    });
}

/// How many reads the scatter on this thread asked for ahead: per f pair of
/// a counted fold's walk and of a walk writing a one-product level directly.
pub(super) fn prefetches_run() -> [u64; 2] {
    PREFETCHED.with(Cell::get)
}

/// The filter policy a [`ForcedFilter`] guard on this thread has installed:
/// the most opened children a round may filter its walk over, the filter's
/// size, as a power of two in bits, when pinned, how many keys a level
/// tests before its pass share is read, and the fewest bytes a level's count
/// column spans for it to filter.
#[derive(Clone, Copy, Debug)]
pub(super) struct FilterPin {
    pub(super) keys: usize,
    pub(super) bits_log: Option<u32>,
    pub(super) sample: u64,
    pub(super) min_bytes: usize,
}

thread_local! {
    static FILTER: Cell<Option<FilterPin>> = const { Cell::new(None) };
    /// How many counted walks on this thread ran filtered, ran unfiltered,
    /// held more hits than the filtered walk's ring holds, and ran
    /// unfiltered because their level's filter passed too many keys, and
    /// how many counted levels walked without the filter because their
    /// count column was too narrow.
    static WALKS: Cell<[u64; 5]> = const { Cell::new([0; 5]) };
}

/// The pinned filter policy, if any.
pub(super) fn forced_filter() -> Option<(usize, Option<u32>, u64, usize)> {
    FILTER.with(Cell::get).map(|pin| (pin.keys, pin.bits_log, pin.sample, pin.min_bytes))
}

/// A counted walk's filter policy pinned on this thread until the guard
/// drops.
pub(super) struct ForcedFilter {
    prior: Option<FilterPin>,
}

impl ForcedFilter {
    pub(super) fn install(pin: Option<FilterPin>) -> ForcedFilter {
        ForcedFilter { prior: FILTER.with(|c| c.replace(pin)) }
    }
}

impl Drop for ForcedFilter {
    fn drop(&mut self) {
        FILTER.with(|c| c.set(self.prior));
    }
}

/// Count one counted walk of kind `kind` (see [`walks_run`]).
pub(super) fn note_walk(kind: usize) {
    WALKS.with(|c| {
        let mut seen = c.get();
        seen[kind] += 1;
        c.set(seen);
    });
}

/// How many counted walks on this thread ran filtered, ran unfiltered,
/// wrapped the filtered walk's ring of held hits, and ran unfiltered on a
/// level whose filter passed too many keys, and how many counted levels
/// walked without the filter, their count column too narrow.
pub(super) fn walks_run() -> [u64; 5] {
    WALKS.with(Cell::get)
}

/// `n` rows of values below `2^bits[i]` in column `i`, from a fixed seed. With
/// `skew`, three rows in four instead start with a value below 4.
fn random_rows(seed: u64, n: usize, bits: &[u32], skew: bool) -> Vec<Vec<u64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|_| {
            let mut row: Vec<u64> = bits.iter().map(|&b| next() % (1u64 << b)).collect();
            if skew && next() % 4 != 0 {
                row[0] = next() % 4;
            }
            row
        })
        .collect()
}

/// `n` rows of `cols` values below `2^bits`, from a fixed seed.
fn rows(seed: u64, n: usize, cols: usize, bits: u32) -> Vec<Vec<u64>> {
    random_rows(seed, n, &vec![bits; cols], false)
}

/// The vtrees the joins run on: three subtrees paired every way under the root.
fn paired(x: &Vtree, y: &Vtree, z: &Vtree) -> Vec<Arc<Vtree>> {
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    vec![
        Arc::new(join(&join(x, y), z)),
        Arc::new(join(x, &join(y, z))),
        Arc::new(join(&join(x, z), y)),
    ]
}

/// [`paired`] over three blocks, each a balanced subtree.
fn vtrees(a: &[VarId], b: &[VarId], c: &[VarId]) -> Vec<Arc<Vtree>> {
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    paired(&bal(a), &bal(b), &bal(c))
}

mod count_root;
mod counting_sort;
mod direction;
mod emit;
mod many_parents;
mod held_bits;
mod root_gate;
mod inner_index;
mod passthrough;
mod projection;
mod regression;
mod reset_ws;
mod scatter_direction_pool;
mod prefetch_gate;
mod walk_filter;
mod walks;
