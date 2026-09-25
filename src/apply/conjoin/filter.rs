//! Filtering intermediate products before a conjunction builds their parents.
use super::*;

impl Engine {
    /// Conjoin structural diagrams, omitting intermediate products rejected by `keep`.
    ///
    /// The callback receives a vtree level and the node indices of `f` and `g`
    /// there. It visits live internal products after their level is built,
    /// before any parent reads them. Leaves are retained. Returning false can
    /// only remove models; the caller supplies the justification for removals.
    /// If every rejected product is impossible under a common care constraint,
    /// conjoining that constraint with the result preserves the original
    /// conjunction under the same constraint.
    /// The result may retain unreachable nodes and needs minimization for
    /// canonical form. Both operands are consumed on every outcome.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for a summed-out level, and
    /// the vtree, weight and resource errors of [`Engine::and`]. Callback work
    /// is the caller's responsibility.
    pub fn and_filter_products(
        &self, mut f: Tdd, mut g: Tdd,
        mut keep: impl FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
        for t in f.vtree().bottomup() {
            if f.level(t).is_marginal() || g.level(t).is_marginal() {
                return Err(OperationError::MarginalLevel(t));
            }
        }
        conjoin_recycling(self, f, g, VtreeMask::default(), VtreeMask::default(), Some(&mut keep))
    }
}

#[cfg(test)]
#[path = "tests/filter.rs"]
mod tests;
