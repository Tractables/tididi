//! The value of a weighted diagram.

use crate::diagram::{LeafLabel, Tdd, WeightStore, WeightValue};
use crate::value::{ColumnRetention, FoldInput, ValueDomain, WeightFold};
use crate::limits::{OperationError, PollGate};
use crate::vtree::{VtreeIdx, VtreeNode};
use crate::engine::Engine;

impl Engine {
    /// Run [`Tdd::weighted_value`](crate::Tdd::weighted_value) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Stops are checked at entry, at amortized node boundaries and before return.
    /// Numeric payload allocations are outside the best-effort byte budget.
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
