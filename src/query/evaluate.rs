//! Numeric evaluation with a supplied algebra or the diagram's attached weights.

use crate::value::{ColumnRetention, FoldInput, ValueDomain, WeightFold};
use crate::diagram::EvalAlgebra;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::Engine;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

use super::fold::{fold_bottom_up, LevelFold, PairAlgebra, Side};
use crate::limits::{OperationError, PollGate};

impl Engine {
    /// Run [`Tdd::evaluate`](crate::Tdd::evaluate) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Column buffers are charged to the byte budget and released after their
    /// parent consumes them; allocations inside algebra values are not charged.
    /// Stops are checked at entry, at amortized node boundaries and before return.
    /// An individual algebra callback or node fold cannot be interrupted.
    /// Caller algebra panics propagate as described on the diagram method.
    pub fn evaluate<S: EvalAlgebra>(&self, tdd: &Tdd, algebra: &S) -> Result<S::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        tdd.require_structure()?;
        if lim.should_stop() { return Err(OperationError::Stopped); }
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        let result = if tdd.is_zero() {
            algebra.zero()
        } else {
            let fold = Evaluate(algebra);
            let mut cols = Vec::new();
            lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
            cols.resize_with(tdd.vtree.num_nodes(), Vec::new);
            fold_bottom_up(&fold, self, tdd, &mut cols, ColumnRetention::Frontier,
                Some(&mut gate), |cols, ti| {
                    cols[ti] = fold.alloc(self, tdd.reference_slot_count(VtreeIdx(ti as u32)))?;
                    Ok(())
                })?;
            cols[tdd.output.vtree.idx()].swap_remove(tdd.output.local.idx())
        };
        lim.poll(&mut gate, 1)?;
        lim.flush_poll(&mut gate)?;
        Ok(result)
    }
}

/// [`evaluate`] as an instance of the shared bottom-up walk.
struct Evaluate<'a, S>(&'a S);

impl<S: EvalAlgebra> LevelFold for Evaluate<'_, S> {
    type Value = S::Value;
    type Col = Vec<S::Value>;

    fn alloc(&self, eng: &Engine, width: usize) -> Result<Vec<S::Value>, OperationError> {
        let mut col = Vec::new();
        eng.limits().reserve_exact(&mut col, width)?;
        col.resize(width, self.0.zero());
        Ok(col)
    }

    fn release(&self, eng: &Engine, col: &mut Self::Col) {
        let bytes = (col.capacity() * std::mem::size_of::<S::Value>()) as u64;
        *col = Vec::new();
        eng.limits().release_bytes(bytes);
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<S::Value>, i: usize, v: S::Value) -> Result<(), OperationError> {
        col[i] = v;
        Ok(())
    }

    fn leaf(&self, _leaf: VtreeIdx, var: VarId, label: LeafLabel) -> S::Value {
        match label {
            LeafLabel::Zero => self.0.zero(),
            _ => self.0.leaf(var, label),
        }
    }

    /// Unreachable under the precondition. A marginal level stores model counts,
    /// and an arbitrary algebra has no way to say what a count is worth: the
    /// algebra promises a value per leaf, not an embedding of ℕ. Weighted
    /// evaluation of a marginal diagram is `Tdd::weighted_value`, which
    /// reads the store the weighted marginalization wrote.
    fn marginal_column(&self, _eng: &Engine, _tdd: &Tdd, t: VtreeIdx, _col: &mut Vec<S::Value>) -> Result<(), OperationError> {
        unreachable!(
            "evaluate: level {t:?} is marginal, which this traversal cannot read \
             (see the precondition on `evaluate`)"
        )
    }

    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<S::Value>>,
        right: Side<'_, Vec<S::Value>>,
    ) -> S::Value {
        self.sum_over_pairs(pairs, left, right)
    }
}

impl<S: EvalAlgebra> PairAlgebra for Evaluate<'_, S> {
    fn zero(&self) -> S::Value {
        self.0.zero()
    }
    fn read(&self, col: &Vec<S::Value>, i: usize) -> S::Value {
        col[i].clone()
    }
    fn inline(&self, _count: u32) -> S::Value {
        unreachable!("evaluate: a marginal level's inline ref (see the precondition)")
    }
    fn add_assign(&self, acc: &mut S::Value, v: &S::Value) {
        self.0.add_assign(acc, v);
    }
    fn mul(&self, a: &S::Value, b: &S::Value) -> S::Value {
        self.0.mul(a, b)
    }
}

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
