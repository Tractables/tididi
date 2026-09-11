//! Recycling pool for `Vec<TddLevel>` allocations, owned by the engine.

use crate::engine::Engine;
use std::cell::Cell;

use super::level::{LevelState, TddLevel};
use super::primitives::{MultiPairRange, TddNodeData};

// ── Level allocation pool ────────────────────────────────────────────────────
//
// Conjunction and clause construction create and discard level arrays
// frequently; pooling avoids repeated heap allocation. Two pool slots exist so
// that a conjunction can recycle both of its consumed operands' level arrays
// simultaneously.
//
// Each slot is a `Cell`: `take()` moves the value out, leaving `None` behind,
// and `set()` puts it back when the caller is done. A `Cell` hands the caller
// exclusive ownership with no runtime borrow tracking and no double-borrow
// panic.

/// The engine's two recycled level arrays.
///
/// Two slots because a conjunction consumes two operands and one slot would
/// drop the second's arenas. `take_levels` prefers the primary.
#[derive(Default)]
pub(crate) struct LevelPool {
    primary: Cell<Option<Vec<TddLevel>>>,
    secondary: Cell<Option<Vec<TddLevel>>>,
}

impl LevelPool {
    /// How many of the two slots currently hold a recycled vector.
    pub(crate) fn occupancy(&self) -> usize {
        let primary = self.primary.take();
        let secondary = self.secondary.take();
        let held = usize::from(primary.is_some()) + usize::from(secondary.is_some());
        self.primary.set(primary);
        self.secondary.set(secondary);
        held
    }

    /// Empty both slots, releasing the recycled capacity to the allocator.
    pub(crate) fn drain(&self) {
        self.primary.set(None);
        self.secondary.set(None);
    }
}

/// Try to take a recycled `Vec<TddLevel>` from the given pool slot, sized to
/// `num_nodes`.
///
/// A parked entry whose length differs from the request is resized rather
/// than discarded, keeping the first `min(old, new)` levels warm. Discarding
/// would empty the pool as soon as two different level counts alternated —
/// successive components, and successive compiles, routinely differ in
/// variable count — leaving each consumer to regrow every level's
/// `nodes`/`pairs` from capacity 0.
///
/// `return_levels_to` is the only writer of a slot and runs `reset_level`
/// over every level of the entry before parking it, so every level that
/// survives the resize has already been through that barrier, and every
/// level the resize *adds* is a fresh `TddLevel::new()` — byte-identical to
/// what the fresh-allocation path in `take_levels` produces. Neither
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
        // structs but not their `nodes`/`pairs`/`multi_pairs` buffers, so the arenas
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
/// A level's arena (`nodes`/`pairs`/`multi_pairs`) survives pool recycle only if its
/// allocated capacity is under this cap. Larger arenas are replaced with a
/// fresh empty `Vec` at the moment the levels are handed back, so a parked
/// entry never holds more than this per arena and the bytes are back with the
/// allocator before the next consumer runs.
///
/// Without this cap, an apply intermediate that grew `pairs` to hundreds of
/// MB and then minimized down to a few nodes would pass the
/// `POOL_NODE_CAP_LIMIT` gate (which inspects only `nodes.capacity()`) and
/// be retained with the giant pair arena intact. The next `take_levels`
/// consumer (e.g. `Tdd::clause`) would then build a small diagram on those
/// levels, be charged for the retained capacity, and — with the soft apply
/// budget armed — trip the budget on a step that holds kilobytes of real data.
/// A one-clause diagram built on a retained level has been seen holding a
/// multi-GiB pair capacity at a single level.
pub(crate) const MAX_LEVEL_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Reset one recycled level to empty state.
///
/// Beyond clearing content, also enforces the per-arena capacity cap
/// (`MAX_LEVEL_ARENA_BYTES`): any arena (`nodes`/`pairs`/`multi_pairs`) whose
/// `.capacity()` exceeds the cap is replaced with a fresh empty `Vec`.
///
/// Runs on the return path (`return_levels_to`), which is the only writer of a
/// pool slot: everything parked is already in this state, so `take_levels`
/// hands out clean levels without a second pass. Reset is per level rather
/// than per array so that path can fuse the reset into the same visit that
/// tallies the retained capacity its gate reads.
#[inline]
pub(crate) fn reset_level(level: &mut TddLevel) {
    use std::mem::size_of;
    level.nodes.clear();
    level.pairs.clear();
    level.multi_pairs.clear();
    level.n_tombstones = 0;
    // The recycled arena is empty, so its garbage accounting must be too.
    level.dead_pairs = 0;
    // Marginal state must be cleared too: otherwise a pooled level that was
    // previously marginalized comes back valued, and the next consumer sees
    // `is_marginal() == true` even after pushing fresh nodes into `nodes`.
    // That mismatch crashes pairs_of_idx / apply_and's marginal-schedule
    // invariant on an unrelated diagram.
    //
    // The inline markers go with it. The no-reexpand (NR) path sets them and —
    // unlike reexpand — never clears them, so a recycled NR level would leak a
    // stale marker into the next compile, which would then read it as the
    // marginal-count decode mode and misdecode a bare slot as an inline count
    // → wrong/zero count. (emit-off never sets these; reexpand clears them —
    // which is why only a prior NR compile contaminated the pool.)
    level.inlined_sides = 0;
    level.state = LevelState::Structural;
    // Drop oversized arenas — keep small ones warm. See
    // `MAX_LEVEL_ARENA_BYTES` doc for the underlying bug.
    if level.nodes.capacity().saturating_mul(size_of::<TddNodeData>()) > MAX_LEVEL_ARENA_BYTES {
        level.nodes = Vec::new();
    }
    if level.pairs.capacity().saturating_mul(super::INPUT_PAIR_BYTES) > MAX_LEVEL_ARENA_BYTES {
        level.pairs = Vec::new();
    }
    if level.multi_pairs.capacity().saturating_mul(size_of::<MultiPairRange>()) > MAX_LEVEL_ARENA_BYTES {
        level.multi_pairs = Vec::new();
    }
}

/// Take a pre-allocated `Vec<TddLevel>` from the pool (resized to `num_nodes` by
/// `try_take_from` if a slot has one), or allocate a fresh one. All levels are
/// guaranteed to be empty — a pooled entry was reset by `return_levels_to`
/// before it was parked, a level added by the resize is fresh, and a
/// fresh array is empty by construction.
pub(crate) fn take_levels(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    // Try primary pool, then secondary, then allocate fresh.
    let pool = eng.levels();
    let recycled = try_take_from(&pool.primary, num_nodes)
        .or_else(|| try_take_from(&pool.secondary, num_nodes));
    if let Some(levels) = recycled {
        return levels;
    }
    (0..num_nodes).map(|_| TddLevel::new()).collect()
}

/// Maximum total node capacity (across all levels) to retain in the pool.
/// Levels exceeding this limit are dropped rather than pooled, to avoid
/// retaining the capacity of large intermediate diagrams indefinitely.
/// 4M nodes × 8 bytes/node = 32 MB per pool slot.
const POOL_NODE_CAP_LIMIT: usize = 4_000_000;

/// Return a `Vec<TddLevel>` to a pool slot for reuse. Drops the levels when
/// their total node capacity exceeds `POOL_NODE_CAP_LIMIT` so we don't
/// retain peak memory from rare giant intermediate diagrams.
///
/// The retention gate reads the arenas as they arrive — resetting first would
/// hide a giant `nodes` arena from it and park a Vec the limit exists to drop.
/// What survives the gate is then reset here rather than at the next
/// `take_levels`: a `pairs`/`multi_pairs` arena over `MAX_LEVEL_ARENA_BYTES` (which the
/// node-capacity gate does not see) goes back to the allocator now instead of
/// sitting in the pool for the gap between return and take.
#[inline]
fn return_levels_to(slot: &Cell<Option<Vec<TddLevel>>>, mut levels: Vec<TddLevel>) {
    // One pass over the levels: tally the capacity the retention gate reads and
    // reset each level in the same visit. Two passes over a level array with
    // hundreds of thousands of entries would stream the whole array twice for
    // no added information.
    //
    // Resetting before the gate decides is state-equivalent to the sum-then-
    // reset order: the gate's two outcomes are "reset and park" and "drop", and
    // a dropped Vec releases exactly the arenas a reset had kept warm. Only the
    // tally must see pre-reset capacities (reset zeroes an oversized arena), so
    // it is taken from each level before that level is reset.
    let mut node_capacity = 0usize;
    for level in &mut levels {
        node_capacity += level.nodes.capacity();
        reset_level(level);
    }
    if node_capacity <= POOL_NODE_CAP_LIMIT {
        slot.set(Some(levels));
    }
    // else: drop levels, releasing the retained capacity
}

/// Which of the two pool slots a level array goes back to.
///
/// A conjunction consumes two operands; returning both to one slot would drop
/// the second's arenas, so the caller says which is which. [`take_levels`]
/// prefers [`PoolSlot::First`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PoolSlot {
    /// The first operand's slot — where most callers fetch from.
    First,
    /// The second operand's slot.
    Second,
}

/// Return a `Vec<TddLevel>` to one of the pool slots for reuse.
pub(crate) fn return_levels(eng: &Engine, slot: PoolSlot, levels: Vec<TddLevel>) {
    let pool = eng.levels();
    let cell = match slot {
        PoolSlot::First => &pool.primary,
        PoolSlot::Second => &pool.secondary,
    };
    return_levels_to(cell, levels)
}

/// Empty both level-pool slots, releasing any recycled `Vec<TddLevel>` capacity
/// (up to `POOL_NODE_CAP_LIMIT` per slot) back to the allocator.
///
/// Called from `Engine::reset` at an inter-compile recovery boundary so a
/// failed compile's pooled levels don't carry into the child compiles. Not on
/// any hot path — the normal recycle path is `return_levels`/`take_levels`.
pub(crate) fn drop_pools(eng: &Engine) {
    eng.levels().drain();
}

