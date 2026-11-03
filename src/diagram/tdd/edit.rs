//! Coordinated edits to a diagram's levels and the references that use them.

use crate::diagram::{ChildDecoder, MarginalStorage, TddLevel, finish_refs_remap, plan_refs_remap, remap_refs_into};
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;
use super::Tdd;

impl Tdd {
    /// Rewrite pair storage without changing node indices. Reserve fallible
    /// scratch before entering, or have the closure refuse only before it
    /// changes the level; the dirty notification cannot stop, and on a level
    /// left as it was it costs only a rescan.
    #[inline]
    pub(crate) fn rewrite_level<R>(
        &mut self, v: VtreeIdx, rewrite: impl FnOnce(&mut TddLevel) -> R,
    ) -> R {
        let result = rewrite(&mut self.levels[v.idx()]);
        self.levels.mark_changed(v);
        self.invalidate(v);
        result
    }

    /// Compact a value column to the slots `referenced` names, merging equal
    /// values unless `merge` is off on a bounded column
    /// ([`MarginalStorage::compact`]), and redirect its references, as one
    /// edit: `remap` is filled with each referenced slot's new one, and the
    /// number merged is returned. This preserves every referenced value; the
    /// caller schedules any extra content-twin work caused by equal values
    /// acquiring the same slot.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses the map that finds
    /// equal values or the move of an implicit parent. Both are had before any
    /// slot moves, so the column and its references are then as they were.
    pub(crate) fn reindex_level(
        &mut self, lim: &Limits, v: VtreeIdx, referenced: &[u32], remap: &mut [u32], merge: bool,
    ) -> Result<usize, OperationError> {
        let plan = MarginalStorage::new(&mut self.levels[v.idx()], self.weights.as_mut(), v.idx())
            .plan_compact(lim, referenced, remap, merge)?;
        let refs = match referenced.is_empty() {
            true => None,
            false => {
                let view = self.levels[v.idx()].child_decoder();
                Some(plan_refs_remap(lim, self, v, view, remap)?)
            }
        };
        let merged = MarginalStorage::new(&mut self.levels[v.idx()], self.weights.as_mut(), v.idx())
            .compact(referenced, remap, plan);
        if let Some(refs) = refs {
            finish_refs_remap(self, refs, remap);
        }
        Ok(merged)
    }

    /// Redirect merged nodes and schedule their parent for reduction. Return
    /// that parent so an ongoing bottom-up pass can extend its own worklist.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
    /// implicit parent; nothing is redirected then.
    #[inline]
    pub(crate) fn merge_level_nodes(&mut self, lim: &Limits, v: VtreeIdx, remap: &[u32]) -> Result<Option<VtreeIdx>, OperationError> {
        remap_refs_into(lim, self, v, remap)?;
        let Some(parent) = self.vtree.node(v).parent() else { return Ok(None) };
        self.invalidate(parent);
        Ok(Some(parent))
    }

    /// Replace structural nodes by values. The domain supplies the column,
    /// deduplicated, and `remap`, the slot each node's value took when the
    /// dedup moved any ([`MarginalDomain::dedup`]); this boundary owns the
    /// parent's references and work.
    ///
    /// [`MarginalDomain::dedup`]: crate::marginal::transition::MarginalDomain::dedup
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
    /// implicit parent. It is planned before `install` runs, so the level and
    /// its parent are then as they were.
    pub(crate) fn install_marginal_level(
        &mut self, lim: &Limits, v: VtreeIdx, remap: Option<&[u32]>,
        install: impl FnOnce(&mut TddLevel),
    ) -> Result<(), OperationError> {
        let parent = self.vtree.node(v).parent();
        // The parent's sides name the level's nodes, as a marginal child's
        // bare slots do: the decoder the level has once installed reads them.
        let refs = match (remap, parent) {
            (Some(remap), Some(parent)) if !self.levels[parent.idx()].is_marginal() => {
                Some(plan_refs_remap(lim, self, v, ChildDecoder::marginal(), remap)?)
            }
            _ => None,
        };
        if let Some(parent) = parent { self.invalidate(parent); }
        install(&mut self.levels[v.idx()]);
        if let (Some(refs), Some(remap)) = (refs, remap) {
            finish_refs_remap(self, refs, remap);
        }
        Ok(())
    }

    /// Install a rotation only after both levels have been built. Their old
    /// arenas are returned together for a search that may undo the rotation.
    pub(crate) fn replace_level_pair(
        &mut self, outer: (VtreeIdx, TddLevel), inner: (VtreeIdx, TddLevel),
    ) -> (TddLevel, TddLevel) {
        // Rotation can create twins in the inner level. Contraction visits it
        // through the outer level, whose pair lists now describe those nodes.
        self.invalidate(outer.0);
        self.dirty.forget_fused(inner.0.0);
        let old_inner = std::mem::replace(&mut self.levels[inner.0.idx()], inner.1);
        let old_outer = std::mem::replace(&mut self.levels[outer.0.idx()], outer.1);
        self.levels.mark_changed(inner.0);
        self.levels.mark_changed(outer.0);
        (old_outer, old_inner)
    }
}
