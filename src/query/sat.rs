//! Satisfiability of compiled diagrams.

use crate::diagram::Tdd;
use crate::Engine;

impl Engine {
    /// Run [`Tdd::is_sat`](crate::Tdd::is_sat) under this engine's limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::is_sat`](crate::Tdd::is_sat).
    pub fn is_sat(&self, f: &Tdd) -> Result<bool, crate::OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if f.is_zero() { return Ok(false); }
        let out_vtree = f.output.vtree;
        let out_level = &f.levels[out_vtree.idx()];
        if out_level.is_weight_marginal() {
            return Err(crate::OperationError::IncompatibleWeights);
        }
        if f.vtree.node(out_vtree).is_leaf() || out_level.is_marginal() { return Ok(true); }
        // With every marginal value read as satisfiable, no stored node is
        // unsatisfiable, so the output node decides. Conditioning asks before
        // its reduction, when a node it emptied may still be stored.
        let out_i = f.output.local.idx();
        Ok(out_level.pairs_iter_of(&out_level.node(out_i)).next().is_some())
    }
}
