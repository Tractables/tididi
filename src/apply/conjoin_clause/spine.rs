//! The clause spine: which vtree levels a clause touches, and their map layout.

use super::*;

/// Per-level marks over the vtree, with a rollback log so clearing them
/// visits only the levels marked.
///
/// The flags are all false whenever the buffer is parked in its pool: the
/// log is replayed when the buffer goes back, on an error exit as much as on
/// success, so a walk that stopped part-way leaves nothing behind for the
/// next checkout.
#[derive(Default)]
pub(super) struct MarkBuffer {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
}

impl MarkBuffer {
    /// Grow the flags to cover `num_nodes` levels, charging `lim`.
    pub(super) fn cover(&mut self, lim: &crate::limits::Limits, num_nodes: usize) -> Result<(), crate::limits::OperationError> {
        lim.try_resize(&mut self.flags, num_nodes, false)?;
        lim.reserve_exact(&mut self.set, num_nodes)
    }

    /// Mark level `t` and report whether it was newly marked.
    #[inline]
    pub(super) fn mark(&mut self, t: VtreeIdx) -> bool {
        if self.flags[t.idx()] { return false; }
        self.flags[t.idx()] = true;
        self.set.push(t);
        true
    }
}

impl std::ops::Deref for MarkBuffer {
    type Target = [bool];
    #[inline]
    fn deref(&self) -> &[bool] {
        &self.flags
    }
}

impl crate::limits::pool::Buffers for MarkBuffer {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::limits::pool::Scratch)) {
        visit(&mut self.flags);
        visit(&mut self.set);
    }
}

impl crate::limits::pool::PooledScratch for MarkBuffer {
    fn prepare(&mut self) {
        debug_assert!(self.set.is_empty() && self.flags.iter().all(|&b| !b), "clause-spine marks were not cleared");
    }

    /// Replay the log so the parked buffer is all false, then apply the
    /// retention cap to both arrays.
    fn retain(&mut self, lim: &crate::limits::Limits) {
        use crate::limits::pool::Buffers;
        for &t in &self.set {
            self.flags[t.idx()] = false;
        }
        self.set.clear();
        self.release_oversized(lim);
    }
}

/// Build the clause spine: mark every ancestor (inclusive) of each clause
/// leaf in `on_spine`, then collect the spine's internal levels bottom-up
/// (post-order) into `spine_internal`.
pub(super) fn build_clause_spine(
    lim: &crate::limits::Limits,
    vtree: &crate::vtree::Vtree,
    clause: &[(Literal, VtreeIdx)],
    on_spine: &mut MarkBuffer,
    spine_internal: &mut Vec<VtreeIdx>,
    dfs_stack: &mut Vec<(VtreeIdx, bool)>,
) -> Result<(), OperationError> {
    let mut gate = lim.gate();
    for &(_, leaf) in clause {
        let mut cur = leaf;
        loop {
            gate.poll(1)?;
            if !on_spine.mark(cur) { break; }
            match vtree.node(cur).parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }

    // The marked set is ancestor-closed, so it is a connected subtree
    // containing the root; the DFS descends only into marked children.
    spine_internal.clear();
    dfs_stack.clear();
    let root = vtree.root();
    if on_spine[root.idx()] && !vtree.node(root).is_leaf() {
        lim.try_push(dfs_stack, (root, false))?;
    }
    while let Some((t, processed)) = dfs_stack.pop() {
        gate.poll(1)?;
        if processed {
            lim.try_push(spine_internal, t)?;
        } else {
            lim.try_push(dfs_stack, (t, true))?;
            let (l, r) = vtree.children(t);
            if on_spine[l.idx()] && !vtree.node(l).is_leaf() { lim.try_push(dfs_stack, (l, false))?; }
            if on_spine[r.idx()] && !vtree.node(r).is_leaf() { lim.try_push(dfs_stack, (r, false))?; }
        }
    }
    gate.flush()
}

/// Propagate `need_dt` top-down over the spine: a level needs the complement
/// conjunction (acc × `d_t`) iff its parent does or both its children are on
/// the spine (the both-relevant `c_t` contains `(d_L, c_R)` and `(c_L, d_R)`).
/// Only spine levels are written.
pub(super) fn propagate_need_dt(
    vtree: &crate::vtree::Vtree,
    spine_internal: &[VtreeIdx],
    on_spine: &[bool],
    need_dt: &mut MarkBuffer,
) {
    for &t in spine_internal.iter().rev() {
        let (l, r) = vtree.children(t);
        let both = on_spine[l.idx()] && on_spine[r.idx()];
        let inherited = need_dt[t.idx()] || both;
        if inherited {
            if on_spine[l.idx()] { need_dt.mark(l); }
            if on_spine[r.idx()] { need_dt.mark(r); }
        }
    }
}

/// Lay out the `cd_map` blocks for the spine: one `LEAF_WIDTH` block per clause
/// leaf and one width-sized block per spine internal level, in that order.
/// Returns the total number of map entries.
///
/// Returns [`OperationError::MarginalLevel`] when the clause needs structure already summed out.
pub(super) fn plan_cd_map_bases(
    clause: &[(Literal, VtreeIdx)],
    spine_internal: &[VtreeIdx],
    levels: &[TddLevel],
    level_base: &mut [usize],
) -> Result<usize, OperationError> {
    let mut total = 0usize;
    for &(_, leaf) in clause {
        level_base[leaf.idx()] = total;
        total += LEAF_WIDTH;
    }
    for &t in spine_internal {
        let ti = t.idx();
        if levels[ti].is_marginal() {
            return Err(OperationError::MarginalLevel(t));
        }
        level_base[ti] = total;
        total += levels[ti].slot_count();
    }
    Ok(total)
}

/// Fill the `cd_map` blocks of the clause's own leaves.
///
/// A leaf level stores no nodes: under the leaf encoding (One, Pos, Neg) the
/// literal picks the `c_t` column and its complement the `d_t` one, and
/// `CONJOIN_GRID` gives both directly.
pub(super) fn fill_leaf_maps(
    clause: &[(Literal, VtreeIdx)],
    level_base: &[usize],
    need_dt: &[bool],
    cd_map: &mut [[u32; 2]],
) {
    for &(lit, t) in clause {
        let base = level_base[t.idx()];
        let compute_dt = need_dt[t.idx()];
        let (clause_idx, compl_idx) = if lit.sign {
            (POS_LEAF_IDX.0 as usize, NEG_LEAF_IDX.0 as usize)
        } else {
            (NEG_LEAF_IDX.0 as usize, POS_LEAF_IDX.0 as usize)
        };
        for i in 0..LEAF_WIDTH {
            let dt = if compute_dt { CONJOIN_GRID[i][compl_idx] } else { NO_PRODUCT };
            cd_map[base + i] = [CONJOIN_GRID[i][clause_idx], dt];
        }
    }
}


#[cfg(test)]
#[path = "tests/spine.rs"]
mod tests;
