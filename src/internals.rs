//! The seam the CNF compiler compiles against.
//!
//! Everything re-exported here is an implementation detail of this crate that
//! the compiler in the surrounding workspace reaches for: scratch-pool hooks,
//! the scoped projection registers the compile loop drives, the bounded
//! rotation primitives, and the diagnostic counters. None of it is covered by
//! the crate's compatibility promise, and it is hidden from the documented
//! API — a caller outside that compiler wants the modules in the module map
//! instead.
//!
//! The re-exports are the whole seam: an item is reachable from outside this
//! crate either through the documented modules or through here, never both.

pub use crate::apply::conjoin_clause::{try_apply_and_clause, walk_mark_spine};
pub use crate::apply::project::{
    caller_projection_active, free_nonprojected_count, project_var_scoped, project_vars_gated,
    project_vars_scoped, ScopedProjectLeaves, ScopedProjectionGuard, PROJECT_APPLIED_COUNT,
    PROJECT_FORGOTTEN_SCOPED, PROJECT_LEAF_IDXS_SCOPED,
};
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
pub use crate::write::save::write_tdd;

#[cfg(any(test, debug_assertions))]
pub use crate::diagram::marg::set_marg_inline_max;

use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// Assemble a vtree from nodes laid out by the caller.
///
/// See [`Vtree::from_nodes`](crate::vtree::Vtree).
pub fn vtree_from_nodes(nodes: Vec<VtreeNode>, root: VtreeIdx, num_vars: u32) -> Vtree {
    Vtree::from_nodes(nodes, root, num_vars)
}

/// Append a balanced subtree over `vars` to `nodes` and return its root.
pub fn vtree_build_balanced_recursive(vars: &[VarId], nodes: &mut Vec<VtreeNode>) -> VtreeIdx {
    Vtree::build_balanced_recursive(vars, nodes)
}

/// Graft `subtrees` under one root and report where every node landed.
pub fn vtree_graft_over(
    subtrees: &[&Vtree],
    rename: impl Fn(usize, VarId) -> VarId,
    spine_vars: &[VarId],
    num_vars: u32,
) -> Result<(Vtree, GraftLayout), crate::vtree::VtreeError> {
    Vtree::graft_over(subtrees, rename, spine_vars, num_vars)
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
