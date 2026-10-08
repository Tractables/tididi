//! Recycling pool for `Vec<TddLevel>` allocations, owned by the engine.

use crate::Engine;
use crate::limits::{Charged, Limits};
use crate::execution::pool::{Buffers, Drain, Pool, Pools, PooledScratch, Scratch};

use super::level::TddLevel;

/// The engine's two recycled level arrays.
///
/// Two slots because a conjunction consumes two operands and one slot would
/// drop the second's arenas. A take uses the parked array nearest in length
/// to the levels it wants, and a fresh array when every parked one is as
/// far from that as an empty one: cutting a long array down to a short take
/// drops levels a later long take would use. A return keeps the two longest
/// arrays: it parks an array in a vacant slot or in place of a shorter
/// parked one, and drops an array no longer than either parked one as it
/// is, without resetting levels that would not be kept.
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
}

impl Pools for LevelPool {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        visit(&self.primary);
        visit(&self.secondary);
    }
}

/// Trim oversized individual arenas before applying the engine's shared ceiling.
pub(crate) const MAX_LEVEL_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Reset one recycled level to empty state, and return the bytes its arenas
/// keep: what [`LevelBuffer`] lists for it.
///
/// Beyond clearing content, also enforces the per-arena capacity cap
/// (`MAX_LEVEL_ARENA_BYTES`): any arena (`nodes`/`pairs`/`ranges`) whose
/// `.capacity()` exceeds the cap is replaced with a fresh empty `Vec`.
///
/// Runs on the return path, so everything parked is already in this state.
#[inline]
pub(crate) fn reset_level(level: &mut TddLevel) -> u64 {
    // A kept arena is at most `MAX_LEVEL_ARENA_BYTES`, so the sum of three
    // does not overflow.
    #[inline(always)]
    fn retain<T>(arena: &mut Vec<T>) -> u64 {
        let bytes = arena.capacity() * std::mem::size_of::<T>();
        if bytes > MAX_LEVEL_ARENA_BYTES {
            *arena = Vec::new();
            return 0;
        }
        bytes as u64
    }
    level.clear();
    retain(&mut level.nodes) + retain(level.pairs.stored_mut()) + retain(&mut level.ranges)
}

/// Take `num_nodes` empty levels from the pool, growing the array through
/// the allocator when the pooled one is shorter. For the infallible
/// constructors and tests; an operation takes its levels through
/// [`try_take_levels`]. Every level is empty: a pooled entry was reset by
/// `return_levels` before it was parked, and a level added here is fresh.
///
/// An array grows as a `Vec` does, to twice its capacity at least, so that
/// the arrays of a run whose vtrees grow a few nodes at a time are not
/// reallocated at every take.
pub(crate) fn take_levels(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    let mut levels = take_level_array(eng, num_nodes);
    if levels.len() < num_nodes {
        levels.reserve(num_nodes - levels.len());
        levels.resize_with(num_nodes, TddLevel::new);
    }
    levels
}

/// [`take_levels`] charging the array's growth to the engine.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when the growth is refused; the pooled
/// array is dropped with it.
pub(crate) fn try_take_levels(eng: &Engine, num_nodes: usize) -> Result<Vec<TddLevel>, crate::limits::OperationError> {
    let mut levels = take_level_array(eng, num_nodes);
    let missing = num_nodes - levels.len();
    if missing > 0 {
        eng.limits().reserve(&mut levels, missing)?;
        levels.resize_with(num_nodes, TddLevel::new);
    }
    Ok(levels)
}

/// The parked level array nearest `num_nodes` levels in length, the
/// primary's on a tie, cut down to at most `num_nodes` levels; an empty
/// array when every parked one is `num_nodes` levels away or more.
fn take_level_array(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    let pool = &eng.scratch.levels;
    let gap = |slot: &Pool<LevelBuffer>| slot.parked(|parked| parked.levels.len().abs_diff(num_nodes));
    let slot = match (gap(&pool.primary), gap(&pool.secondary)) {
        (first, Some(second)) if second < num_nodes && first.is_none_or(|first| second < first) => &pool.secondary,
        (Some(first), _) if first < num_nodes => &pool.primary,
        _ => return Vec::new(),
    };
    let mut levels = slot.take(eng).levels;
    if levels.len() > num_nodes {
        levels.truncate(num_nodes);
        if levels.capacity().saturating_mul(std::mem::size_of::<TddLevel>()) > MAX_LEVEL_ARENA_BYTES {
            levels.shrink_to_fit();
        }
    }
    levels
}

/// Recycled structural arrays. Clearing a level drops its marginal values.
#[derive(Default)]
struct LevelBuffer {
    levels: Vec<TddLevel>,
}

impl Buffers for LevelBuffer {
    /// A parked level was reset ([`reset_level`]), which leaves its arena
    /// stored, so its pairs are the vector of a stored arena.
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.levels);
        for level in &mut self.levels {
            visit(&mut level.nodes);
            visit(level.pairs.stored_mut());
            visit(&mut level.ranges);
        }
    }
}

impl PooledScratch for LevelBuffer {
    fn prepare(&mut self) {}
    /// Levels follow their own per-arena cap, [`MAX_LEVEL_ARENA_BYTES`]; the
    /// bytes kept are counted as each level is reset, in one walk.
    fn retain(&mut self, _lim: &Limits) -> usize {
        let arenas = self.levels.iter_mut().fold(0u64, |bytes, level| bytes.saturating_add(reset_level(level)));
        usize::try_from(arenas.saturating_add(self.levels.charged_bytes())).unwrap_or(usize::MAX)
    }
}

/// Return a `Vec<TddLevel>` to the pool for reuse: to a vacant slot, the
/// primary first, or in place of the shorter parked array, the primary's on
/// a tie, when it is longer. An array no longer than either parked one is
/// dropped as it is.
pub(crate) fn return_levels(eng: &Engine, levels: Vec<TddLevel>) {
    let pool = &eng.scratch.levels;
    let len = |slot: &Pool<LevelBuffer>| slot.parked(|parked| parked.levels.len());
    let cell = match (len(&pool.primary), len(&pool.secondary)) {
        (None, _) => &pool.primary,
        (_, None) => &pool.secondary,
        (Some(first), Some(second)) => {
            let (cell, shorter) = if second < first { (&pool.secondary, second) } else { (&pool.primary, first) };
            if levels.len() <= shorter {
                return;
            }
            cell
        }
    };
    cell.put(eng, LevelBuffer { levels })
}

