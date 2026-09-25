//! Coordinated edits to a diagram's levels and the references that use them.

use crate::diagram::{TddLevel, WeightStore, remap_refs_into};
use crate::vtree::VtreeIdx;
use super::Tdd;

impl Tdd {
    /// Rewrite pair storage without changing node indices. Reserve fallible
    /// scratch before entering; the closure and dirty notification cannot stop.
    #[inline]
    pub(crate) fn rewrite_level<R>(
        &mut self, v: VtreeIdx, rewrite: impl FnOnce(&mut TddLevel) -> R,
    ) -> R {
        let result = rewrite(&mut self.levels[v.idx()]);
        self.invalidate(v);
        result
    }

    /// Compact a value column and redirect its references as one edit. The
    /// closure updates both column storage and level metadata and fills `remap`.
    /// This preserves every referenced value; the caller schedules any extra
    /// content-twin work caused by equal values acquiring the same slot.
    #[inline]
    pub(crate) fn reindex_level<R>(
        &mut self, v: VtreeIdx, referenced: &[u32], remap: &mut [u32],
        compact: impl FnOnce(&mut TddLevel, Option<&mut WeightStore>, &[u32], &mut [u32]) -> R,
    ) -> R {
        let result = compact(&mut self.levels[v.idx()], self.weights.as_mut(), referenced, remap);
        if !referenced.is_empty() { remap_refs_into(self, v, remap); }
        result
    }

    /// Redirect merged nodes and schedule their parent for reduction. Return
    /// that parent so an ongoing bottom-up pass can extend its own worklist.
    #[inline]
    pub(crate) fn merge_level_nodes(&mut self, v: VtreeIdx, remap: &[u32]) -> Option<VtreeIdx> {
        remap_refs_into(self, v, remap);
        let parent = self.vtree.node(v).parent()?;
        self.invalidate(parent);
        Some(parent)
    }

    /// Replace structural nodes by values. The domain supplies the column and
    /// optional slot remap; this boundary owns the parent's references and work.
    pub(crate) fn install_marginal_level(
        &mut self, v: VtreeIdx,
        install: impl FnOnce(&mut TddLevel) -> Option<Vec<u32>>,
    ) {
        let parent = self.vtree.node(v).parent();
        if let Some(parent) = parent { self.invalidate(parent); }
        let remap = install(&mut self.levels[v.idx()]);
        if let (Some(remap), Some(parent)) = (remap, parent)
            && !self.levels[parent.idx()].is_marginal() {
                remap_refs_into(self, v, &remap);
            }
    }

    /// Install a rotation only after both levels have been built. Their old
    /// arenas are returned together for a search that may undo the rotation.
    pub(crate) fn replace_level_pair(
        &mut self, outer: (VtreeIdx, TddLevel), inner: (VtreeIdx, TddLevel),
    ) -> (TddLevel, TddLevel) {
        // Rotation can create twins in the inner level. Contraction visits it
        // through the outer level, whose pair lists now describe those nodes.
        self.invalidate(outer.0);
        let old_inner = std::mem::replace(&mut self.levels[inner.0.idx()], inner.1);
        let old_outer = std::mem::replace(&mut self.levels[outer.0.idx()], outer.1);
        (old_outer, old_inner)
    }
}
