//! The pair arena's growth policy: the emit choke points, their bounded
//! increments, and the poll strides the cell loops run at.

use crate::engine::{Limits, PAIR_ELEM_BYTES};
use crate::error::ApplyError;

/// Sentinel for dead product cells: c1[i] ∧ c2[j] = ⊥ (no output node created).
///
/// Same bit pattern as `ZERO` in the diagram types but semantically distinct:
/// `ZERO` marks a diagram whose output is UNSAT, while this marks a single
/// product grid cell that produced no live pairs.
pub(crate) const DEAD: u32 = u32::MAX;

/// Amortization stride for the sparse-scatter and collapse-collector poll gates
/// — one poll per ~1 M units of inner work.
pub(super) const APPLY_POLL_STRIDE: u64 = 1 << 20;

/// Amortization stride for the dense between-cell poll gates — one poll per
/// 65536 cell iterations.
pub(super) const DENSE_CELL_POLL_STRIDE: u64 = 1 << 16;

/// Grow `v` up to `new_len`, filling with [`DEAD`].
#[inline]
pub(super) fn try_resize_dead(
    lim: &Limits,
    v: &mut Vec<u32>,
    new_len: usize,
) -> Result<(), ApplyError> {
    lim.try_resize(v, new_len, DEAD)
}

/// Fallible pair push: stores into `level.pairs`, routing growth through the
/// level's growth mode.
///
/// The single choke point for `level.pairs` growth on the dense emit walk. When
/// [`Limits::begin_level`] flagged the level as near-cap, a growth event routes
/// through [`grow_pairs_bounded`] — bounded, headroom-aware increments instead
/// of `Vec`'s doubling — so the reallocation transient stays `current +
/// increment` rather than doubling's three times current.
///
/// Split like [`Limits::try_push`]: a bare `len < capacity` store here,
/// everything else in [`push_pair_grow`], which is what lets LLVM keep `len`,
/// `capacity` and the arena base in registers across the emit walk's pushes.
#[inline(always)]
pub(super) fn try_push_pair_into(
    lim: &Limits,
    level: &mut crate::diagram::TddLevel,
    pair: crate::diagram::InputPair,
) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if v.len() < v.capacity() {
        v.push(pair);
        return Ok(());
    }
    push_pair_grow(lim, v, pair)
}

/// Growth arm of [`try_push_pair_into`]. Reached only when `len == capacity`, so
/// once per growth event.
#[cold]
#[inline(never)]
fn push_pair_grow(
    lim: &Limits,
    v: &mut Vec<crate::diagram::InputPair>,
    pair: crate::diagram::InputPair,
) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    if lim.bounded_growth() {
        grow_pairs_bounded(lim, v)?;
    }
    let out = lim.try_push(v, pair);
    // The output-pair meter is charged here rather than at the choke point
    // because this is the arm a growth event reaches, and growth is the only
    // thing that moves capacity.
    lim.charge_output_pairs(v.capacity().saturating_sub(pre_cap));
    out
}

/// Minimum bounded-growth increment for `level.pairs`, in bytes (1 M pairs at
/// 8 B). Floors the increment so growth never degenerates to per-push
/// reallocation. It cannot cause quadratic copying in practice: the floor only
/// binds when half the remaining headroom is below it, which is within one or
/// two growth events of `OverBudget` — everywhere else the increment is
/// half-headroom (geometric) or full doubling.
const PAIRS_GROW_MIN_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// Pure increment policy for [`grow_pairs_bounded`] (factored out for unit
/// tests): grow a full `cap`-capacity Vec by
/// `next_cap = min(2 × cap, cap + max(min_chunk, half-headroom))`.
///
/// - Plentiful headroom (`headroom/2 ≥ cap × elem_bytes`): increment = `cap` —
///   plain doubling, amortized linear pushes.
/// - Shrinking headroom: increment = half the remaining headroom, so the
///   reallocation transient (`old cap + increment`) always fits, and successive
///   increments halve geometrically.
/// - Near exhaustion: the `min_chunk` floor; at most a couple of events before
///   the budget or the allocator trips `OverBudget`.
fn bounded_grow_increment(cap: usize, headroom_bytes: u64, elem_bytes: u64) -> usize {
    let eb = elem_bytes.max(1);
    let half_room = usize::try_from(headroom_bytes / 2 / eb).unwrap_or(usize::MAX);
    let min_chunk = (PAIRS_GROW_MIN_CHUNK_BYTES / eb) as usize;
    half_room.max(min_chunk).min(cap)
}

/// Bounded-growth increment for a `cap`-capacity `level.pairs`: the ONE place
/// that feeds the pair element size and the current headroom into
/// [`bounded_grow_increment`], shared by the per-push choke point and the bulk
/// twin so both grow by the same policy.
#[inline]
fn bounded_pairs_increment(lim: &Limits, cap: usize) -> usize {
    bounded_grow_increment(cap, lim.headroom(), PAIR_ELEM_BYTES)
}

/// Bounded, headroom-aware growth for a full `level.pairs`. Reserves the
/// increment through the ONE accounting path, so there is no second budget
/// mechanism. Cold: once per growth event, never per push.
#[cold]
#[inline(never)]
fn grow_pairs_bounded(
    lim: &Limits,
    v: &mut Vec<crate::diagram::InputPair>,
) -> Result<(), ApplyError> {
    let cap = v.capacity();
    if cap == 0 {
        // Fresh vec: doubling from empty is trivially transient-safe.
        return Ok(());
    }
    lim.reserve_exact(v, bounded_pairs_increment(lim, cap))
}

/// Bulk twin of [`try_push_pair_into`]: guarantee room for `additional` more
/// pairs so the caller can emit them with plain pushes instead of a per-pair
/// reserve. Growth obeys the SAME per-level mode as the per-push choke point, so
/// `level.pairs` has one growth policy and not two.
///
/// The one intentional divergence from the accounted reserves is the budget
/// CHARGE: this serves the clause conjunction, which runs outside the
/// per-operation meter reset, so charging its pair arena would accumulate across
/// clauses and trip the soft budget spuriously. The allocator preflight and the
/// `OverBudget` refusal channel are the same.
#[inline]
pub(crate) fn reserve_pairs_for_emit(
    lim: &Limits,
    level: &mut crate::diagram::TddLevel,
    additional: usize,
) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if additional <= v.capacity() - v.len() {
        return Ok(());
    }
    if lim.bounded_growth() {
        let inc = bounded_pairs_increment(lim, v.capacity()).max(additional);
        lim.preflight_alloc((inc as u64).saturating_mul(PAIR_ELEM_BYTES));
        return v.try_reserve_exact(inc).map_err(|_| ApplyError::OverBudget);
    }
    lim.preflight_alloc((v.capacity().max(additional) as u64).saturating_mul(PAIR_ELEM_BYTES));
    v.try_reserve(additional).map_err(|_| ApplyError::OverBudget)
}

#[cfg(test)]
#[path = "budget_bounded_growth_tests.rs"]
mod bounded_growth_tests;
