//! Test-oracle flags and census hooks for the conjunction sweep.
//!
//! In `cfg(test)` builds, [`super`] imports the live thread-local hooks from
//! [`super::tests`]; in non-test builds, the no-op stubs here inline away.

#[cfg(not(test))]
use super::{ChildPair, PlainLookups, Route, VtreeIdx};

/// What a test counts of the levels read by g-pair vectors.
#[derive(Clone, Copy)]
pub(super) enum WordNote {
    /// A level of two grids.
    Grids,
    /// A level with a complete side.
    Complete,
    /// A cell walked in the grouped order.
    Grouped,
    /// A row read by vectors.
    Row,
    /// A row read by vectors that walks its live cells alone.
    Cells,
    /// A row read by vectors that writes its live cells off the sorted
    /// list of its live candidates.
    Sorted,
    /// A streaming target's level, and one read through marginal lookups.
    Stream,
    StreamMarginal,
    /// A sparse-marginal level.
    SparseMarginal,
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn taken_levels_settled_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn grid_lookups_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_lookups(_lookups: PlainLookups) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_scheduled_charge() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_lone_pair() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_lone_columns() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_whole_row() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn lone_columns_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn one_sided_masks_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_one_sided_masks() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn reach_columns_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn row_columns_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_reach_columns_row() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn single_rows_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_single_rows() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn reach_tables_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_reach_tables() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn relay_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_relayed() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_row_columns_row(_whole: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn ahead_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn ahead_after_forced() -> Option<usize> {
    None
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_ahead(_sparse: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn seed_pairs_forced() -> Option<usize> {
    None
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn pair_rows_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn ands_buffered_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_pair_rows(_complete: bool, _one_pair: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_pair_row_unrolled() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn word_cells_forced() -> Option<bool> {
    None
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn word_rows_sorted_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn dense_poll_stride_forced() -> Option<u64> {
    None
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_word_cells(_what: WordNote) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_pairs_grown() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn lend_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_lent() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_implied_walk(_dense: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_as_operand(_t: usize, _of_f: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_held_tight() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_route(_t: VtreeIdx, _route: Route) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn relabel_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_relabelled(_moved: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_read_through() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_written_pairs(_pairs: &[ChildPair]) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn complete_sides_listed() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_complete_side(_side: crate::diagram::ChildSide) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn builds_past_cap_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_refused_past_cap() {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn note_closed_in_sweep(_holes: bool) {}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn close_in_sweep_pairs() -> usize {
    super::output::CLOSE_IN_SWEEP_PAIRS
}
