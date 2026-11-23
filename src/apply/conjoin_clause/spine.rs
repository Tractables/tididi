//! The clause spine: which vtree levels a clause touches, and their map layout.

use super::*;

/// Per-level marks over the vtree, with a rollback log so clearing them
/// visits only the levels marked.
///
/// The flags are all false whenever the buffer is parked with its working set
/// ([`ClauseWork`]): the log is replayed when the set goes back, on an error exit as much as on
/// success, so a walk that stopped part-way leaves nothing behind for the
/// next checkout.
#[derive(Default)]
pub(super) struct MarkBuffer {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
}

impl MarkBuffer {
    /// Grow the flags to cover `num_nodes` levels, charging `lim`.
    pub(super) fn cover(
        &mut self,
        lim: &crate::limits::Limits,
        num_nodes: usize,
    ) -> Result<(), crate::limits::OperationError> {
        if self.flags.len() >= num_nodes && self.set.capacity() >= num_nodes {
            return Ok(());
        }
        lim.try_resize(&mut self.flags, num_nodes, false)?;
        lim.reserve_exact(&mut self.set, num_nodes)
    }

    /// Mark level `t` and report whether it was newly marked.
    #[inline]
    pub(super) fn mark(&mut self, t: VtreeIdx) -> bool {
        if self.flags[t.idx()] {
            return false;
        }
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

impl crate::execution::pool::Buffers for MarkBuffer {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.flags);
        visit(&mut self.set);
    }
}

impl MarkBuffer {
    pub(super) fn debug_assert_clear(&self) {
        debug_assert!(
            self.set.is_empty() && self.flags.iter().all(|&b| !b),
            "clause-spine marks were not cleared"
        );
    }

    /// Replay the log, so every flag is false again.
    pub(super) fn unmark_all(&mut self) {
        for &t in &self.set {
            self.flags[t.idx()] = false;
        }
        self.set.clear();
    }
}

/// Build the clause spine: mark every ancestor (inclusive) of each clause
/// leaf in `on_spine`, and list the spine's internal levels children first in
/// `spine_internal`, which starts empty. Returns the topmost level with both
/// children on the spine: each such level is where a later leaf's path meets
/// the spine marked so far.
pub(super) fn build_clause_spine(
    lim: &crate::limits::Limits,
    vtree: &crate::vtree::Vtree,
    clause: &[(Literal, VtreeIdx)],
    on_spine: &mut MarkBuffer,
    spine_internal: &mut Vec<VtreeIdx>,
) -> Result<Option<VtreeIdx>, OperationError> {
    let mut gate = lim.gate();
    let mut top_fork: Option<VtreeIdx> = None;
    for &(_, leaf) in clause {
        let mut cur = leaf;
        loop {
            gate.poll(1)?;
            if !on_spine.mark(cur) {
                if top_fork.is_none_or(|f| vtree.topo_pos(f) < vtree.topo_pos(cur)) {
                    top_fork = Some(cur);
                }
                break;
            }
            if !vtree.node(cur).is_leaf() {
                lim.try_push(spine_internal, cur)?;
            }
            match vtree.node(cur).parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
    vtree.sort_bottom_up(spine_internal);
    // Two units a level, what the order walk this sort replaced charged, so
    // work budgets callers tuned against it stay where they were.
    gate.poll(2 * spine_internal.len() as u64)?;
    gate.flush()?;
    Ok(top_fork)
}

/// Propagate `need_dt` top-down over the spine: a level needs the complement
/// conjunction (acc × `d_t`) iff its parent needs it or has both children on
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
            if on_spine[l.idx()] {
                need_dt.mark(l);
            }
            if on_spine[r.idx()] {
                need_dt.mark(r);
            }
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
            let dt = if compute_dt {
                CONJOIN_GRID[i][compl_idx]
            } else {
                NO_PRODUCT
            };
            cd_map[base + i] = [CONJOIN_GRID[i][clause_idx], dt];
        }
    }
}

#[cfg(test)]
#[path = "tests/spine.rs"]
mod tests;
