//! The value of a weighted diagram.

use crate::diagram::{LeafLabel, Tdd, WeightStore, WeightValue};
use crate::value::{ColumnRetention, FoldInput, ValueDomain, WeightFold};
use crate::limits::{OperationError, PollGate};
use crate::vtree::{VtreeIdx, VtreeNode};
use crate::engine::Engine;

/// The diagram's value under its attached
/// [`WeightStore`](crate::diagram::WeightStore), or `None` when the diagram
/// carries no store. ⊥ has the store's zero.
///
/// A weighted marginalization usually leaves the output level explicit and
/// marginalizes only levels below it, so this folds the explicit levels above the
/// marginal ones on demand from the store's values and leaf weights; when the
/// output level is itself marginal it reads the stored value directly. The
/// diagram is borrowed and unchanged. [`Engine::weighted_value`] uses a caller's
/// limits; this convenience form uses a fresh, unarmed engine.
///
/// # Panics
///
/// Panics if the fold's allocation is refused.
pub fn weighted_value(tdd: &Tdd) -> Option<WeightValue> {
    Engine::new().weighted_value(tdd).expect("weighted_value: allocation refused")
}

impl Engine {
    /// Fold the attached weights under this engine's allocation and stop rules.
    ///
    /// Returns `Ok(None)` without a weight store. Stop rules are checked at
    /// entry, at amortized node boundaries, and before returning the result.
    /// Numeric payload allocations remain outside the best-effort byte budget.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] for a refused scratch reservation and
    /// [`OperationError::Stopped`] for a stop decision. The diagram is unchanged.
    ///
    /// # Examples
    ///
    /// Independent fair Boolean variables give an exact probability:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_rational::BigRational;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    ///
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let mut f = Tdd::clause(&tree, [1, 2]);
    /// let half = BigRational::new(1.into(), 2.into());
    /// let weights = vec![LiteralWeights { negative: half.clone(), positive: half }; 2];
    /// let algebra = RationalWeights::from_literals(&weights);
    /// f.set_weights(WeightStore::new(algebra, Arithmetic::ExactRational))?;
    /// let probability = Engine::new().weighted_value(&f)?.unwrap().into_rational();
    /// assert_eq!(probability, BigRational::new(3.into(), 4.into()));
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn weighted_value(&self, tdd: &Tdd) -> Result<Option<WeightValue>, OperationError> {
        let _op = self.limits().begin_operation();
        let Some(ws) = tdd.weights.as_ref() else { return Ok(None); };
        if self.limits().should_stop() { return Err(OperationError::Stopped); }
        let mut gate = PollGate::new(self.limits().reduce_poll_stride());
        let value = weighted_output_value(self, tdd, ws, &mut gate)?;
        self.limits().poll(&mut gate, 1)?;
        self.limits().flush_poll(&mut gate)?;
        Ok(Some(value))
    }
}

/// The weighted value of `tdd`'s output node under `ws`; `tdd` must have been
/// weighted with `ws`.
fn weighted_output_value(eng: &Engine, tdd: &Tdd, ws: &WeightStore, gate: &mut PollGate) -> Result<WeightValue, OperationError> {
    let vtree = &tdd.vtree;
    // UNSAT / constant-false output: the `ZERO` sentinel carries no level slot
    // (`output.local` is the `ZERO` idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return Ok(ws.wzero());
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return Ok(ws.level(out_t).expect("output level weight-marginalized")[out_i].clone());
    }
    // Leaf output level: the fold below stores nothing for leaves (their values
    // come from the semiring on demand), so read the leaf value directly.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return Ok(ws.leaf_val(var, LeafLabel::from_idx(out_i)));
    }
    let mut computed: Vec<Option<Vec<WeightValue>>> = Vec::new();
    eng.limits().try_resize(&mut computed, vtree.num_nodes(), None)?;
    // Only the root value is read, so child columns are released as their
    // parent completes (`ColumnRetention::Frontier`). The "already stored" test
    // is this diagram's own marginality rather than `WeightStore::is_set`: the
    // store is shared, so a column at this index may belong to another live
    // `Tdd` while this diagram's level is still structural.
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    WeightFold::ensure(
        eng,
        VtreeIdx(out_t as u32),
        FoldInput { vtree, levels: &tdd.levels, store: ws },
        &mut computed,
        &marginal,
        ColumnRetention::Frontier,
        |work| eng.limits().poll(gate, work),
    )?;
    Ok(computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone())
}
