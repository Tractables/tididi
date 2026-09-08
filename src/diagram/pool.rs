//! Thread-local recycling pool for `Vec<TddLevel>` allocations.

use std::cell::Cell;

use super::level::TddLevel;
use super::primitives::{ExtMulti, TddNodeData};

// ── Level allocation pool ────────────────────────────────────────────────────
//
// Recycling pool for Vec<TddLevel> allocations. apply_and and clause_to_tdd
// create and discard level arrays frequently; pooling avoids repeated heap
// allocation. Two pool slots exist so that `apply_and` can recycle
// both of its consumed operands' level arrays simultaneously.
//
// Pattern: `Cell::take()` moves the value out of thread-local storage (leaving
// None behind), and `Cell::set()` puts it back when done. This is preferred
// over RefCell because it gives exclusive ownership to the caller (no runtime
// borrow tracking needed) and avoids the risk of panicking on double borrow.

thread_local! {
    #[cfg(not(test))]
    static LEVELS_POOL: Cell<Option<Vec<TddLevel>>> = const { Cell::new(None) };
    #[cfg(not(test))]
    static LEVELS_POOL2: Cell<Option<Vec<TddLevel>>> = const { Cell::new(None) };
    #[cfg(test)]
    pub(crate) static LEVELS_POOL: Cell<Option<Vec<TddLevel>>> = const { Cell::new(None) };
    #[cfg(test)]
    pub(crate) static LEVELS_POOL2: Cell<Option<Vec<TddLevel>>> = const { Cell::new(None) };
}

/// Try to take a recycled `Vec<TddLevel>` from the given pool slot, sized to
/// `num_nodes`.
///
/// A parked entry whose length differs from the request is RESIZED, not
/// discarded. Discarding threw away every warm arena the moment two different
/// level counts alternated — successive components, and successive compiles,
/// routinely differ in variable count, so the pool went cold and each consumer
/// regrew every level's `nodes`/`pairs` from capacity 0. Resizing keeps the
/// first `min(old, new)` levels warm.
///
/// Reset semantics are unchanged. `return_levels_to` is the only writer of a
/// slot and runs `reset_level` over every level of the entry before parking it, so every
/// level that survives the resize has already been through that barrier, and
/// every level the resize *adds* is a fresh `TddLevel::new()` — byte-identical
/// to what the fresh-allocation path in `take_levels` produces. Neither
/// direction can hand out a level that skipped its reset.
fn try_take_from(slot: &Cell<Option<Vec<TddLevel>>>, num_nodes: usize) -> Option<Vec<TddLevel>> {
    use std::mem::size_of;
    let mut pool = slot.take()?;   // Cell::take() leaves None in the cell
    if pool.len() != num_nodes {
        let truncating = pool.len() > num_nodes;
        // Covers both directions: truncates when the entry is longer (releasing
        // the surplus levels' arenas), appends empty levels when it is shorter.
        pool.resize_with(num_nodes, TddLevel::new);
        // The level array itself is an arena too, and truncation leaves its
        // capacity at the high-water mark of every size this slot has ever
        // served. Hold it to the same per-arena byte cap the levels are held to,
        // so recycling across a big-then-small size change cannot carry an
        // unbounded spine forward. `shrink_to_fit` relocates the `TddLevel`
        // structs but not their `nodes`/`pairs`/`ext` buffers, so the arenas
        // that survived the truncation stay warm.
        if truncating
            && pool.capacity().saturating_mul(size_of::<TddLevel>()) > MAX_LEVEL_ARENA_BYTES
        {
            pool.shrink_to_fit();
        }
    }
    Some(pool)
}

/// Per-arena capacity cap on pooled levels, in bytes.
///
/// A level's arena (`nodes`/`pairs`/`ext`) survives pool recycle only if its
/// allocated capacity is under this cap. Larger arenas are replaced with a
/// fresh empty `Vec` at the moment the levels are handed back, so a parked
/// entry never holds more than this per arena and the bytes are back with the
/// allocator before the next consumer runs.
///
/// Without this cap, an apply intermediate that grew `pairs` to hundreds of
/// MB and then minimized down to a few nodes would pass the
/// `POOL_NODE_CAP_LIMIT` gate (which inspects only `nodes.capacity()`) and
/// be retained with the giant pair arena intact. The next `take_levels`
/// consumer (e.g. `Tdd::clause`) would then build a small TDD on those
/// levels, be charged for the retained capacity, and — with the soft apply
/// budget armed — trip the budget on a step that holds kilobytes of real data.
/// A one-clause diagram built on a retained level has been seen holding a
/// multi-GiB pair capacity at a single level.
pub(crate) const MAX_LEVEL_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Reset one recycled level to empty state.
///
/// Beyond clearing content, also enforces the per-arena capacity cap
/// (`MAX_LEVEL_ARENA_BYTES`): any arena (`nodes`/`pairs`/`ext`) whose
/// `.capacity()` exceeds the cap is replaced with a fresh empty `Vec`.
///
/// Runs on the return path (`return_levels_to`), which is the only writer of a
/// pool slot: everything parked is already in this state, so `take_levels`
/// hands out clean levels without a second pass. Per-LEVEL rather than
/// per-array so that path can fuse the reset into the same visit that tallies
/// the retained capacity its gate reads.
#[inline]
pub(crate) fn reset_level(level: &mut TddLevel) {
    use std::mem::size_of;
    level.nodes.clear();
    level.pairs.clear();
    level.ext.clear();
    level.n_tombstones = 0;
    // The recycled arena is empty, so its garbage accounting must be too.
    level.dead_pairs = 0;
    // Marginal state must be cleared too: otherwise a pooled level that
    // was previously marginalized retains `marginal_counts`, and the
    // next consumer sees `is_marginal() == true` even after pushing
    // fresh nodes into `nodes`. That mismatch crashes pairs_of_idx /
    // apply_and's marginal-schedule invariant on an unrelated TDD.
    level.marginal_counts = None;
    level.marginal_counts_big = None;
    // The inline markers are part of the marginal state and MUST be cleared
    // too. The no-reexpand (NR) path sets `marg_inlined_left/right = true`
    // and — unlike reexpand — never clears them, so a recycled NR level
    // leaks a stale `true` into the next compile. The next NR compile then
    // reads it as the marginal-count decode-mode flag and misdecodes a
    // bare slot as an inline count → wrong/zero count. (emit-off never sets
    // these; reexpand clears them — which is why only a prior NR compile
    // contaminated the pool. Mirrors the per-level `clear()` reset.)
    level.marg_flags = 0;
    level.retired_marg_width = 0;
    // Drop oversized arenas — keep small ones warm. See
    // `MAX_LEVEL_ARENA_BYTES` doc for the underlying bug.
    if level.nodes.capacity().saturating_mul(size_of::<TddNodeData>()) > MAX_LEVEL_ARENA_BYTES {
        level.nodes = Vec::new();
    }
    if level.pairs.capacity().saturating_mul(super::INPUT_PAIR_BYTES) > MAX_LEVEL_ARENA_BYTES {
        level.pairs = Vec::new();
    }
    if level.ext.capacity().saturating_mul(size_of::<ExtMulti>()) > MAX_LEVEL_ARENA_BYTES {
        level.ext = Vec::new();
    }
}

/// Take a pre-allocated `Vec<TddLevel>` from the pool (resized to `num_nodes` by
/// `try_take_from` if a slot has one), or allocate a fresh one. All levels are
/// guaranteed to be empty — a pooled entry was reset by `return_levels_to`
/// before it was parked, a level added by the resize is fresh, and a
/// fresh array is empty by construction.
pub fn take_levels(num_nodes: usize) -> Vec<TddLevel> {
    // Try primary pool, then secondary, then allocate fresh.
    let recycled = LEVELS_POOL.with(|cell| try_take_from(cell, num_nodes))
        .or_else(|| LEVELS_POOL2.with(|cell| try_take_from(cell, num_nodes)));
    if let Some(levels) = recycled {
        return levels;
    }
    (0..num_nodes).map(|_| TddLevel::new()).collect()
}

/// Maximum total node capacity (across all levels) to retain in the pool.
/// Levels exceeding this limit are dropped rather than pooled, to avoid
/// retaining the capacity of large intermediate TDDs indefinitely.
/// 4M nodes × 8 bytes/node = 32 MB per pool slot.
const POOL_NODE_CAP_LIMIT: usize = 4_000_000;

/// Return a `Vec<TddLevel>` to a pool slot for reuse. Drops the levels when
/// their total node capacity exceeds `POOL_NODE_CAP_LIMIT` so we don't
/// retain peak memory from rare giant intermediate TDDs.
///
/// The retention gate reads the arenas as they arrive — resetting first would
/// hide a giant `nodes` arena from it and park a Vec the limit exists to drop.
/// What survives the gate is then reset here rather than at the next
/// `take_levels`: a `pairs`/`ext` arena over `MAX_LEVEL_ARENA_BYTES` (which the
/// node-capacity gate does not see) goes back to the allocator now instead of
/// sitting in the thread-local for the gap between return and take.
#[inline]
fn return_levels_to(slot: &'static std::thread::LocalKey<Cell<Option<Vec<TddLevel>>>>,
                    mut levels: Vec<TddLevel>) {
    // ONE pass over the levels: tally the capacity the retention gate reads and
    // reset each level in the same visit. Two passes over a level array with
    // hundreds of thousands of entries is two streams of the whole array —
    // the second was pure repetition.
    //
    // Resetting BEFORE the gate decides is state-equivalent to the sum-then-
    // reset order: the gate's two outcomes are "reset and park" and "drop", and
    // a dropped Vec releases exactly the arenas a reset had kept warm. Only the
    // TALLY must see pre-reset capacities (reset zeroes an oversized arena), so
    // it is taken from each level before that level is reset.
    let mut node_capacity = 0usize;
    for level in &mut levels {
        node_capacity += level.nodes.capacity();
        reset_level(level);
    }
    if node_capacity <= POOL_NODE_CAP_LIMIT {
        slot.with(|cell| cell.set(Some(levels)));
    }
    // else: drop levels, releasing the retained capacity
}

/// Return a `Vec<TddLevel>` to the primary pool slot (used for the first operand
/// in `apply_and` — the slot that most callers fetch from).
pub fn return_levels(levels: Vec<TddLevel>) {
    return_levels_to(&LEVELS_POOL, levels)
}

/// Return a Vec<TddLevel> to the secondary pool slot. `apply_and`
/// consumes two operands, and a second slot lets it recycle both without
/// dropping either's capacity.
pub(crate) fn return_levels2(levels: Vec<TddLevel>) {
    return_levels_to(&LEVELS_POOL2, levels)
}

/// Empty both level-pool slots, releasing any recycled `Vec<TddLevel>` capacity
/// (up to `POOL_NODE_CAP_LIMIT` per slot) back to the allocator.
///
/// Called from `reset_apply_scratch` at an inter-compile recovery boundary so a
/// failed compile's pooled levels don't carry into the child compiles. NOT on
/// any hot path — the normal recycle path is `return_levels`/`take_levels`.
pub(crate) fn drop_pools() {
    LEVELS_POOL.with(|c| c.set(None));
    LEVELS_POOL2.with(|c| c.set(None));
}

