//! The seam the CNF compiler compiles against.
//!
//! Everything re-exported here is an implementation detail of this crate that
//! the compiler in the surrounding workspace reaches for: scratch-pool hooks,
//! the bounded rotation primitives, and the diagnostic counters. None of it is covered by
//! the crate's compatibility promise, and it is hidden from the documented
//! API — a caller outside that compiler wants the modules in the module map
//! instead.
//!
//! The re-exports are the whole seam: an item is reachable from outside this
//! crate either through the documented modules or through here, never both.

pub use crate::apply::conjoin_clause::walk_mark_spine;
pub use crate::diagram::pool::{return_levels, take_levels};
pub use crate::query::count::{compute_node_counts, node_counts_u128};
pub use crate::reduce::contract::p_fusion::{apply_p_fusion_at_parents, PFusionStats};
pub use crate::reduce::slot_prune::{prune_marg_slots, MargSlotPruneStats};
pub use crate::restructure::graft::graft_over;
pub use crate::restructure::relevel::{
    restructure_after_left_rotation_bounded, restructure_after_right_rotation_bounded,
};
pub use crate::restructure::scratch::RestructureScratch;
pub use crate::restructure::search::cluster::cluster_marginal_rotations_in_subtree;
pub use crate::vtree::graft::GraftLayout;
pub use crate::vtree::rotate::{rotate_left, rotate_right};

#[cfg(any(test, debug_assertions))]
pub use crate::diagram::marg::set_marg_inline_max;

use crate::vtree::{Vtree, VtreeIdx};

/// Monotone tally of marginal-count slots collected by `prune_marg_slots`
/// across all levels. Strictly non-decreasing over a compile; resets only when
/// a level is cleared or reset (at a component boundary, say).
///
/// A caller that gates on diagram size records the retired total at its
/// baseline instant; at comparison time,
/// `collected_since = retired_marg_total(t).saturating_sub(baseline_retired)`
/// is added to `node_count()` so that slot-pruning does not silently deflate
/// the metric.
pub fn retired_marg_total(t: &crate::Tdd) -> usize {
    t.levels.iter().map(|l| l.retired_marg_slots as usize).sum()
}

/// The internal vtree nodes in the order [`Vtree::internal_bottomup`] yields.
pub fn vtree_internal_topo_slice(vtree: &Vtree) -> &[VtreeIdx] {
    vtree.internal_topo_slice()
}

/// Every vtree node, leaves included, in bottom-up topological order.
pub fn vtree_bottomup_topo(vtree: &Vtree) -> &[VtreeIdx] {
    vtree.bottomup_topo()
}

/// The position of `idx` in the bottom-up topological order.
pub fn vtree_topo_pos(vtree: &Vtree, idx: VtreeIdx) -> u32 {
    vtree.topo_pos(idx)
}
