//! Core TDD data types and level-allocation pool.
//!
//! Defines the types that make up a Tree Decision Diagram: nodes, levels,
//! input pairs, and the TDD struct itself. Also provides a thread-local
//! recycling pool for level arrays to avoid repeated heap allocation.
//!
//! **Node encoding**: each `TddNodeData` is 8 bytes (two u32 fields).
//! Three-way encoding: leaf label, inline single pair, or multi-pair arena reference.
//! Single-pair nodes (60–95% of nodes in practice) store the pair directly in the
//! node itself — no arena entry, no indirection. Multi-pair nodes reference a
//! contiguous slice in the level's `pairs` arena via `(pair_start, pair_len)`.

mod primitives;
mod packed;
mod marg;
mod level;
mod pool;
mod tdd;

// ── Public re-exports (mirror all previously visible items) ──────────────────

// primitives
pub use primitives::{
    InputPair, LeafLabel, LocalNodeIdx, TddNodeData, TddNodeId,
    LEAF_WIDTH, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX, ZERO,
};
// crate-internal only (no downstream consumer)
pub(crate) use primitives::{ExtMulti, INPUT_PAIR_BYTES};

// (packed::PairsIter has no re-export consumer — internal users import it
// directly from `super::packed`.)

// marg consts + types
pub use marg::{
    BigSide, MARG_INLINE_MAX, MARG_OVERFLOW_TAG, MARG_VALUE_MASK,
    MargRef, marg_skip_minimize_on,
};
// crate-internal only (no downstream consumer)
pub(crate) use marg::{
    MargResolved, decode_marg_coord, marg_inline_max,
    resolve_marg_ref, tag_all_marg_side_slots, tag_all_marg_side_slots_at,
    assert_can_make_marginal,
};

// marg pub(crate) items
pub(crate) use marg::resolve_swapped_marg_side;

// marg test-only override hooks — dev/test profiles only (compiled out of
// plain release); `pub` so the downstream compiler crate's tests can reach
// them across the crate boundary. (public-release P3a)
#[cfg(any(test, debug_assertions))]
pub use marg::{set_marg_gates, set_marg_inline_max};

// level
pub use level::TddLevel;

// pool
pub use pool::{return_levels, take_levels};
// crate-internal only (no downstream consumer)
pub(crate) use pool::{MAX_LEVEL_ARENA_BYTES, drop_pools, return_levels2};

// pool test-only internals (used by types_tests.rs via `use super::*`)
#[cfg(test)]
pub(crate) use pool::{reset_level, LEVELS_POOL, LEVELS_POOL2};

// tdd
pub use tdd::{C2Probe, Tdd};

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "../types_tests.rs"]
mod tests;
