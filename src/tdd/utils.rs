//! General-purpose utilities used across the TDD implementation.

use std::cell::Cell;
use std::thread::LocalKey;

use super::types::InputPair;

/// Borrow the contents of a thread-local `Cell` pool, leaving `Default::default()` behind.
///
/// The pooling pattern used throughout `tdd/` for scratch buffers: `Cell::take()`
/// moves the value out (replacing it with `Default`), caller uses it, then
/// `pool_put` writes it back. Generic over any `Default` payload.
#[inline]
pub(crate) fn pool_take<T: Default + 'static>(cell: &'static LocalKey<Cell<T>>) -> T {
    cell.with(|c| c.take())
}

/// Return a value to a thread-local `Cell` pool, replacing whatever is currently there.
#[inline]
pub(crate) fn pool_put<T: 'static>(cell: &'static LocalKey<Cell<T>>, value: T) {
    cell.with(|c| c.set(value));
}

/// Drop `v`'s allocation (leaving it empty) if its retained capacity in bytes
/// exceeds `max_bytes`. Keeps small buffers warm while preventing rare large
/// calls from leaking GiB-scale scratch into RSS.
///
/// The single implementation of the scratch-retention rule. `pool_put_bounded`
/// applies it to `Cell`-pooled buffers; scratch held as struct fields (which
/// cannot round-trip through a pool per buffer) applies it field by field.
///
/// Companion to the per-arena cap in `reset_level` (`types::MAX_LEVEL_ARENA_BYTES`):
/// scratch pools are not part of any diagram's retained-capacity accounting, so
/// they can't trip the caller's step budget, but they DO inflate real RSS —
/// invisible to the soft apply budget's predictive checks.
#[inline]
pub(crate) fn release_if_oversized<T>(v: &mut Vec<T>, max_bytes: usize) {
    if v.capacity().saturating_mul(std::mem::size_of::<T>()) > max_bytes {
        *v = Vec::new();
    }
}

/// Return a `Vec<T>` to a thread-local `Cell` pool, dropping its allocation
/// first if it is oversized (see [`release_if_oversized`]).
#[inline]
pub(crate) fn pool_put_bounded<T: 'static>(
    cell: &'static LocalKey<Cell<Vec<T>>>,
    mut v: Vec<T>,
    max_bytes: usize,
) {
    release_if_oversized(&mut v, max_bytes);
    cell.with(|c| c.set(v));
}

/// Sorting network for 3..=8 elements (optimal compare-swap counts).
///
/// Works on any indexable + swappable container. Each case is a hardcoded
/// sequence of conditional swaps (`cswap!(a, b)` = "if s[a] > s[b], swap
/// them"). Faster than general-purpose sort for small n because the comparison
/// sequence is known at compile time, enabling branch-free code generation.
macro_rules! sorting_network {
    ($s:expr, $n:expr) => {{
        macro_rules! cswap {
            ($a:expr, $b:expr) => { if $s[$a] > $s[$b] { $s.swap($a, $b); } };
        }
        match $n {
            3 => { cswap!(0,1); cswap!(1,2); cswap!(0,1); }
            4 => { cswap!(0,1); cswap!(2,3); cswap!(0,2); cswap!(1,3); cswap!(1,2); }
            5 => {
                cswap!(0,1); cswap!(3,4); cswap!(2,4); cswap!(2,3);
                cswap!(0,3); cswap!(0,2); cswap!(1,4); cswap!(1,3); cswap!(1,2);
            }
            6 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(0,2); cswap!(1,3); cswap!(0,4); cswap!(1,5);
                cswap!(1,2); cswap!(3,5); cswap!(2,4); cswap!(3,4); cswap!(1,2);
            }
            7 => {
                // Green's construction (Knuth TAOCP Vol 3, 16 comparators).
                cswap!(0,4); cswap!(1,5); cswap!(2,6);
                cswap!(0,2); cswap!(1,3); cswap!(4,6);
                cswap!(2,4); cswap!(3,5);
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(1,4); cswap!(3,6);
                cswap!(1,2); cswap!(3,4); cswap!(5,6);
            }
            8 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5); cswap!(6,7);
                cswap!(0,2); cswap!(1,3); cswap!(4,6); cswap!(5,7);
                cswap!(1,2); cswap!(5,6);
                cswap!(0,4); cswap!(3,7); cswap!(1,5); cswap!(2,6);
                cswap!(1,4); cswap!(3,6); cswap!(2,4); cswap!(3,5); cswap!(3,4);
            }
            _ => unreachable!("sorting_network called with n={}, expected 3..=8", $n),
        }
    }};
}

/// Sort a slice of input pairs in-place into ascending `(left, right)` order.
///
/// Optimized for the small pair counts typical in TDD nodes: uses sorting
/// networks for ≤8 pairs, insertion sort for ≤24, and pdqsort for larger.
/// Checks if already sorted first (common after apply_and).
///
/// This is a localized helper for the specific node-construction paths that
/// build a pair list in arbitrary order and must canonicalize it before pushing
/// the node — projection, conditioning and restriction. It is NOT part of the general
/// pair-storage contract: pair lists carry no globally-maintained sorted
/// invariant, the apply/conjoin hot path never calls this, and no data layout
/// assumes sorted order (see the NOTE at the bottom of `types.rs`).
#[inline]
pub(crate) fn sort_pairs(pairs: &mut [InputPair]) {
    let n = pairs.len();
    if n < 2 { return; }
    if n == 2 {
        if pairs[0] > pairs[1] { pairs.swap(0, 1); }
        return;
    }
    // Check if already sorted (common after apply_and which builds sorted pairs).
    let mut sorted = true;
    for i in 1..n {
        if pairs[i - 1] > pairs[i] { sorted = false; break; }
    }
    if sorted { return; }
    match n {
        3..=8 => { sorting_network!(pairs, n); }
        9..=24 => {
            // Insertion sort for small-medium lists: O(n²) but low constant
            // factor, no recursion overhead, excellent cache behavior.
            // Faster than pdqsort for n ≤ ~24 (pdqsort has partition overhead).
            for i in 1..n {
                let key = pairs[i];
                let mut j = i;
                while j > 0 && pairs[j - 1] > key {
                    pairs[j] = pairs[j - 1];
                    j -= 1;
                }
                pairs[j] = key;
            }
        }
        _ => {
            // Sort via explicit u64 key `(left << 32) | right` instead of the
            // derived field-by-field Ord. The derive expands to a branchy
            // `left.cmp(&right) else right.cmp(...)`; the u64 form is a single
            // unsigned compare and lets pdqsort's branchless partition kick in.
            pairs.sort_unstable_by_key(|p| ((p.left.0 as u64) << 32) | (p.right.0 as u64));
        }
    }
}

#[cfg(test)]
#[path = "utils_tests.rs"]
mod tests;
