//! The clause spine: which vtree levels a clause touches, and their map layout.

use super::*;

/// Clause-spine marks, cleared and returned to their pool on every exit.
/// Reset visits only marked levels, including during error unwinding.
pub(super) struct SpineMarks<'a> {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
    pool: &'a Pool<MarkBuffer>,
    /// Held for `Drop`, which is where the retain cap frees the buffers and so
    /// where their bytes go back to the meter.
    lim: &'a crate::limits::Limits,
}

/// The flags and their rollback log reuse capacity together.
#[derive(Default)]
pub(super) struct MarkBuffer {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
}

impl<'a> SpineMarks<'a> {
    /// Take the pooled array, grown to cover `num_nodes` levels.
    pub(super) fn take(lim: &'a crate::limits::Limits, pool: &'a Pool<MarkBuffer>, num_nodes: usize) -> Result<Self, crate::limits::OperationError> {
        let MarkBuffer { mut flags, mut set } = pool.take();
        lim.try_resize(&mut flags, num_nodes, false)?;
        lim.reserve_exact(&mut set, num_nodes)?;
        Ok(SpineMarks { flags, set, pool, lim })
    }

    /// Mark level `t` and report whether it was newly marked.
    #[inline]
    pub(super) fn set(&mut self, t: VtreeIdx) -> bool {
        if self.flags[t.idx()] { return false; }
        self.flags[t.idx()] = true;
        self.set.push(t);
        true
    }
}

impl std::ops::Deref for SpineMarks<'_> {
    type Target = [bool];
    #[inline]
    fn deref(&self) -> &[bool] {
        &self.flags
    }
}

impl Drop for SpineMarks<'_> {
    fn drop(&mut self) {
        for &t in &self.set {
            self.flags[t.idx()] = false;
        }
        debug_assert!(
            self.flags.iter().all(|&b| !b),
            "clause-spine marks were not cleared",
        );
        self.set.clear();
        crate::limits::pool::release_if_oversized(self.lim, &mut self.set);
        crate::limits::pool::release_if_oversized(self.lim, &mut self.flags);
        self.pool.put(MarkBuffer { flags: std::mem::take(&mut self.flags), set: std::mem::take(&mut self.set) });
    }
}

/// Build the clause spine: mark every ancestor (inclusive) of each clause
/// leaf in `on_spine`, then collect the spine's internal levels bottom-up
/// (post-order) into `spine_internal`.
pub(super) fn build_clause_spine(
    lim: &crate::limits::Limits,
    vtree: &crate::vtree::Vtree,
    clause: &[Literal],
    on_spine: &mut SpineMarks<'_>,
    spine_internal: &mut Vec<VtreeIdx>,
    dfs_stack: &mut Vec<(VtreeIdx, bool)>,
) -> Result<(), OperationError> {
    let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
    for lit in clause {
        let mut cur = vtree.leaf_of(lit.var).expect("the vtree carries this variable");
        loop {
            lim.poll(&mut gate, 1)?;
            if !on_spine.set(cur) { break; }
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
        lim.poll(&mut gate, 1)?;
        if processed {
            lim.try_push(spine_internal, t)?;
        } else {
            lim.try_push(dfs_stack, (t, true))?;
            let (l, r) = vtree.children(t);
            if on_spine[l.idx()] && !vtree.node(l).is_leaf() { lim.try_push(dfs_stack, (l, false))?; }
            if on_spine[r.idx()] && !vtree.node(r).is_leaf() { lim.try_push(dfs_stack, (r, false))?; }
        }
    }
    lim.flush_poll(&mut gate)
}

/// Propagate `need_dt` top-down over the spine: a level needs the complement
/// conjunction (acc × `d_t`) iff its parent does or both its children are on
/// the spine (the both-relevant `c_t` contains `(d_L, c_R)` and `(c_L, d_R)`).
/// Only spine levels are written.
#[inline(always)]
pub(super) fn propagate_need_dt(
    vtree: &crate::vtree::Vtree,
    spine_internal: &[VtreeIdx],
    on_spine: &[bool],
    need_dt: &mut SpineMarks<'_>,
) {
    for &t in spine_internal.iter().rev() {
        let (l, r) = vtree.children(t);
        let both = on_spine[l.idx()] && on_spine[r.idx()];
        let inherited = need_dt[t.idx()] || both;
        if inherited {
            if on_spine[l.idx()] { need_dt.set(l); }
            if on_spine[r.idx()] { need_dt.set(r); }
        }
    }
}

/// Lay out the `cd_map` blocks for the spine: one `LEAF_WIDTH` block per clause
/// leaf and one width-sized block per spine internal level, in that order.
/// Returns the total number of map entries.
///
/// Returns [`OperationError::MarginalLevel`] when the clause needs structure already summed out.
pub(super) fn plan_cd_map_bases(
    vtree: &Vtree,
    clause: &[Literal],
    spine_internal: &[VtreeIdx],
    levels: &[TddLevel],
    level_base: &mut [usize],
) -> Result<usize, OperationError> {
    let mut total = 0usize;
    for lit in clause {
        let ti = vtree.leaf_of(lit.var).expect("the vtree carries this variable").idx();
        level_base[ti] = total;
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
    vtree: &Vtree,
    clause: &[Literal],
    level_base: &[usize],
    need_dt: &[bool],
    cd_map: &mut [[u32; 2]],
) {
    for lit in clause {
        let t = vtree.leaf_of(lit.var).expect("the vtree carries this variable");
        let base = level_base[t.idx()];
        let compute_dt = need_dt[t.idx()];
        let (clause_idx, compl_idx) = if lit.positive {
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
