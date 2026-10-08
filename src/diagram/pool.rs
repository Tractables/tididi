//! Recycling pool for `Vec<TddLevel>` allocations, owned by the engine.

use crate::Engine;
use crate::limits::{Charged, Limits};
use crate::execution::pool::{Buffers, Drain, Pool, Pools, PooledScratch, Scratch};

use super::level::TddLevel;

/// The engine's two recycled level arrays.
///
/// Two slots because a conjunction consumes two operands and one slot would
/// drop the second's arenas. `take_levels` takes the array with the most
/// levels it can use, the primary on a tie.
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
pub(crate) fn take_levels(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    let mut levels = take_level_array(eng, num_nodes);
    if levels.len() < num_nodes {
        levels.reserve_exact(num_nodes - levels.len());
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
        eng.limits().reserve_exact(&mut levels, missing)?;
        levels.resize_with(num_nodes, TddLevel::new);
    }
    Ok(levels)
}

/// The parked level array with the most of the `num_nodes` levels wanted,
/// the primary's on a tie, cut down to at most `num_nodes` levels.
fn take_level_array(eng: &Engine, num_nodes: usize) -> Vec<TddLevel> {
    let pool = &eng.scratch.levels;
    let usable = |slot: &Pool<LevelBuffer>| match slot.occupied() {
        true => slot.parked(|parked| parked.levels.len().min(num_nodes)),
        false => None,
    };
    let slot = match (usable(&pool.primary), usable(&pool.secondary)) {
        (Some(first), Some(second)) if second > first => &pool.secondary,
        (Some(_), _) => &pool.primary,
        (None, _) => &pool.secondary,
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

/// Which of the two pool slots a level array goes back to.
///
/// A conjunction consumes two operands; returning both to one slot would drop
/// the second's arenas, so the caller says which is which. [`take_levels`]
/// prefers [`PoolSlot::First`] on a tie. A slot that is occupied drops what
/// it held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PoolSlot {
    /// The first operand's slot — where most callers fetch from.
    First,
    /// The second operand's slot.
    Second,
    /// The first slot if it is vacant, the second otherwise: for an array
    /// returned between the takes of one operation, which should not drop
    /// an array parked for a later take.
    Vacant,
}

/// Return a `Vec<TddLevel>` to one of the pool slots for reuse.
pub(crate) fn return_levels(eng: &Engine, slot: PoolSlot, levels: Vec<TddLevel>) {
    let pool = &eng.scratch.levels;
    let cell = match slot {
        PoolSlot::First => &pool.primary,
        PoolSlot::Second => &pool.secondary,
        PoolSlot::Vacant if !pool.primary.occupied() => &pool.primary,
        PoolSlot::Vacant => &pool.secondary,
    };
    cell.put(eng, LevelBuffer { levels })
}

