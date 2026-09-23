//! Recycling pool for `Vec<TddLevel>` allocations, owned by the engine.

use crate::Engine;
use crate::limits::{Limits, pool::{Pool, PooledScratch, capacity_bytes}};

use super::level::TddLevel;

/// The engine's two recycled level arrays.
///
/// Two slots because a conjunction consumes two operands and one slot would
/// drop the second's arenas. `take_levels` prefers the primary.
#[derive(Default)]
pub(crate) struct LevelPool {
    primary: Pool<LevelBuffer>,
    secondary: Pool<LevelBuffer>,
}

impl LevelPool {
    /// How many of the two slots currently hold a recycled vector.
    pub(crate) fn occupancy(&self) -> usize {
        usize::from(self.primary.occupied()) + usize::from(self.secondary.occupied())
    }

    /// Empty both slots, releasing the recycled capacity to the allocator.
    pub(crate) fn drain(&self, lim: &Limits) {
        self.primary.drain(lim);
        self.secondary.drain(lim);
    }
}

/// Trim oversized individual arenas before applying the engine's shared ceiling.
pub(crate) const MAX_LEVEL_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Reset one recycled level to empty state.
///
/// Beyond clearing content, also enforces the per-arena capacity cap
/// (`MAX_LEVEL_ARENA_BYTES`): any arena (`nodes`/`pairs`/`multi_pairs`) whose
/// `.capacity()` exceeds the cap is replaced with a fresh empty `Vec`.
///
/// Runs on the return path, so everything parked is already in this state.
#[inline]
pub(crate) fn reset_level(level: &mut TddLevel) -> usize {
    fn retain<T>(arena: &mut Vec<T>) -> usize {
        let bytes = arena.capacity() * std::mem::size_of::<T>();
        if bytes > MAX_LEVEL_ARENA_BYTES { *arena = Vec::new(); 0 } else { bytes }
    }
    level.clear();
    retain(&mut level.nodes) + retain(&mut level.pairs) + retain(&mut level.multi_pairs)
}

/// Take a pre-allocated `Vec<TddLevel>` from the pool (resized to `num_nodes` by
/// the pool if a slot has one), or allocate a fresh one. All levels are
/// guaranteed to be empty — a pooled entry was reset by `return_levels`
/// before it was parked, a level added by the resize is fresh, and a
/// fresh array is empty by construction.
pub(crate) fn take_levels(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    take_levels_with(eng, num_nodes, false).expect("an untracked take cannot be refused")
}

/// Take empty levels from the pool, charging any array growth to the engine.
pub(crate) fn try_take_levels(eng: &Engine, num_nodes: usize) -> Result<Vec<TddLevel>, crate::limits::OperationError> {
    take_levels_with(eng, num_nodes, true)
}

/// Resize the first available level array. `charged` grows it through the
/// engine's budget, which may refuse; otherwise it grows through `Vec`.
fn take_levels_with(
    eng: &Engine,
    num_nodes: usize,
    charged: bool,
) -> Result<Vec<TddLevel>, crate::limits::OperationError> {
    let pool = eng.levels();
    let mut levels = if pool.primary.occupied() { pool.primary.take(eng.limits()) }
        else { pool.secondary.take(eng.limits()) }.levels;
    if levels.len() < num_nodes {
        let additional = num_nodes - levels.len();
        if charged {
            eng.limits().reserve_exact(&mut levels, additional)?;
        } else {
            levels.reserve_exact(additional);
        }
        levels.resize_with(num_nodes, TddLevel::new);
    } else if levels.len() > num_nodes {
        levels.truncate(num_nodes);
        if levels.capacity().saturating_mul(std::mem::size_of::<TddLevel>()) > MAX_LEVEL_ARENA_BYTES {
            levels.shrink_to_fit();
        }
    }
    Ok(levels)
}

/// Recycled structural arrays. Clearing a level drops its marginal values.
#[derive(Default)]
struct LevelBuffer {
    levels: Vec<TddLevel>,
    bytes: usize,
}

impl PooledScratch for LevelBuffer {
    fn prepare(&mut self) {}
    fn retain(&mut self, _lim: &Limits) {
        self.bytes = capacity_bytes(&self.levels);
        for level in &mut self.levels { self.bytes += reset_level(level); }
    }
    fn retained_bytes(&self) -> usize { self.bytes }
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
    cell.put(eng.limits(), LevelBuffer { levels, bytes: 0 })
}

