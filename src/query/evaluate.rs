//! Bottom-up evaluation of a whole diagram in a caller's algebra.
//!
//! [`Tdd::evaluate`] walks the diagram the way `node_counts` does
//! and delegates every arithmetic step to an
//! [`EvalAlgebra`] impl. The algebra and its
//! exact-rational instance live beside the diagram they value; only the walk
//! is here.
//!
//! `model_count` does not go through this trait: its per-node overflow
//! detection is not expressible in an arbitrary algebra.

use crate::value::ColumnRetention;
use crate::diagram::EvalAlgebra;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::engine::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up, LevelFold, PairAlgebra, Side};
use crate::limits::{OperationError, PollGate};

impl Engine {
    /// Evaluate a structural diagram in a caller-supplied algebra.
    ///
    /// For literal weights, evaluation sums the weight of every satisfying
    /// assignment; an assignment's weight is the product of its literal weights.
    /// Nonnegative weights that sum to one for each variable define independent
    /// Bernoulli probabilities. Other weight tables give an unnormalized weighted
    /// sum, not necessarily a probability.
    ///
    /// The exact-weight example uses `BigRational` from the `num-rational` crate.
    /// Add it as a direct dependency to use that type in your application:
    ///
    /// ```sh
    /// cargo add num-rational@0.4
    /// ```
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_rational::BigRational;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{LiteralWeights, RationalWeights};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let f = engine.clause(&tree, [1, 2])?;
    /// let half = BigRational::new(1.into(), 2.into());
    /// let weights = RationalWeights::from_literals(&vec![
    ///     LiteralWeights { negative: half.clone(), positive: half }; 2
    /// ]);
    /// assert_eq!(engine.evaluate(&f, &weights)?, BigRational::new(3.into(), 4.into()));
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// The diagram is borrowed and need not be minimized. Every vtree variable is
    /// folded, including free variables through its `One` value. All values come
    /// from `algebra`, independently of an attached weight store; another call can
    /// use another table without rebuilding the diagram. A [`RationalWeights`]
    /// table must have an entry for every variable named by the tree.
    ///
    /// For a conditional probability, evaluate the conjunction of query and evidence
    /// and divide by the evidence's value using the same weights; zero evidence
    /// value leaves the conditional probability undefined. [`EvalAlgebra`] shows
    /// how to supply other arithmetic. A zero value alone does not establish
    /// unsatisfiability, since weights can be zero or cancel.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] for a summed-out level,
    /// [`OperationError::OverBudget`] for a refused buffer reservation, or
    /// [`OperationError::Stopped`] for an armed stop. The diagram is unchanged.
    ///
    /// Column buffers are charged to the best-effort byte budget and released once
    /// the parent consumes them; allocations inside algebra values are not charged.
    /// Stops are checked at entry, at amortized node boundaries, and before return;
    /// an individual algebra callback or node fold cannot be interrupted.
    ///
    /// # Panics
    ///
    /// Panics from the caller's algebra propagate, including an out-of-range lookup
    /// when a literal weight table does not cover a vtree variable.
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
    /// evaluation of a marginal diagram is `query::weighted_value`, which
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
