//! Recycling pool for `Vec<TddLevel>` allocations, owned by the engine.

use crate::engine::Engine;
use std::cell::Cell;

use super::level::TddLevel;
use super::primitives::{MultiPairRange, EncodedNode};

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
/// A parked entry of a different length is resized, keeping the first
/// `min(old, new)` levels warm. Every level handed out is empty: parked levels
/// were reset by `return_levels_to`, and added ones are fresh.
fn try_take_from(slot: &Cell<Option<Vec<TddLevel>>>, num_nodes: usize) -> Option<Vec<TddLevel>> {
    use std::mem::size_of;
    let mut pool = slot.take()?;   // Cell::take() leaves None in the cell
    if pool.len() != num_nodes {
        let truncating = pool.len() > num_nodes;
        // Covers both directions: truncates when the entry is longer (releasing
        // the surplus levels' arenas), appends empty levels when it is shorter.
        pool.resize_with(num_nodes, TddLevel::new);
        // The level array is an arena too; hold its capacity to the same byte
        // cap as the levels. `shrink_to_fit` moves the `TddLevel` structs, not
        // their buffers, so the surviving arenas stay warm.
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
/// allocated capacity is under this cap; a larger one is replaced with a fresh
/// empty `Vec` when the levels are handed back. The `POOL_NODE_CAP_LIMIT` gate
/// reads only `nodes.capacity()`, so without this cap a minimized intermediate
/// could park a huge `pairs` arena that the next small diagram is then charged for.
pub(crate) const MAX_LEVEL_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Reset one recycled level to empty state.
///
/// Beyond clearing content, also enforces the per-arena capacity cap
/// (`MAX_LEVEL_ARENA_BYTES`): any arena (`nodes`/`pairs`/`multi_pairs`) whose
/// `.capacity()` exceeds the cap is replaced with a fresh empty `Vec`.
///
/// Runs on the return path, so everything parked is already in this state.
#[inline]
pub(crate) fn reset_level(level: &mut TddLevel) {
    use std::mem::size_of;
    level.clear();
    // Drop oversized arenas; keep small ones warm.
    if level.nodes.capacity().saturating_mul(size_of::<EncodedNode>()) > MAX_LEVEL_ARENA_BYTES {
        level.nodes = Vec::new();
    }
    if level.pairs.capacity().saturating_mul(super::CHILD_PAIR_BYTES) > MAX_LEVEL_ARENA_BYTES {
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
/// The gate reads `nodes.capacity()` before the reset, which would zero an
/// oversized arena; the reset happens here so a parked entry is already clean.
#[inline]
fn return_levels_to(slot: &Cell<Option<Vec<TddLevel>>>, mut levels: Vec<TddLevel>) {
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
/// For a recovery boundary, so a failed compile's pooled levels do not carry over.
pub(crate) fn drop_pools(eng: &Engine) {
    eng.levels().drain();
}

