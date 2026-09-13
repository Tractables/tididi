//! Bottom-up evaluation of a whole diagram in a caller's algebra.
//!
//! `evaluate(&tdd, &algebra)` walks the diagram the way `node_counts` does
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

/// Bottom-up evaluate the diagram in `algebra`. Returns the value of the
/// output node (or `algebra.zero()` for the constant-zero diagram).
///
/// Uses [`Engine::evaluate`] on a fresh engine; see that method for the
/// algebra, counting-domain and memory contracts.
///
/// # Panics
///
/// Panics if a level is marginal or a scratch reservation is refused.
/// Panics from the caller's algebra propagate unchanged.
///
/// ```
/// use std::sync::Arc;
/// use num_rational::BigRational;
/// use tididi::Tdd;
/// use tididi::diagram::{LiteralWeights, RationalWeights};
/// use tididi::query::evaluate;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);   // x1 ∧ x2
///
/// // Unit weights reproduce the model count.
/// let unit = RationalWeights::unit(3);
/// assert_eq!(evaluate(&f, &unit), BigRational::from_integer(2.into()).into());
///
/// // A half on every literal weights each of the eight assignments by 1/8.
/// let half = BigRational::new(1.into(), 2.into());
/// let weights: Vec<_> = (0..3).map(|_| LiteralWeights { negative: half.clone(), positive: half.clone() }).collect();
/// let algebra = RationalWeights::from_literals(&weights);
/// assert_eq!(evaluate(&f, &algebra), BigRational::new(1.into(), 4.into()).into());
/// ```
pub fn evaluate<S: EvalAlgebra>(tdd: &Tdd, algebra: &S) -> S::Value {
    Engine::new().evaluate(tdd, algebra).expect("evaluate: operation refused")
}

impl Engine {
    /// Evaluate a structural diagram in the caller's algebra under this engine's limits.
    ///
    /// Every vtree variable is folded, including free variables through their
    /// `One` value; the diagram need not be minimized. The algebra supplies all
    /// values, independently of any attached weight store. Completed child
    /// columns are released after their parent consumes them.
    ///
    /// Library-owned column buffers are charged to the best-effort byte budget.
    /// Allocations inside algebra values and callbacks are outside that budget.
    /// Stops are checked at entry, at amortized node boundaries and before return;
    /// an individual algebra callback or node fold cannot be interrupted.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] for a summed-out level,
    /// [`OperationError::OverBudget`] for a refused buffer reservation, or
    /// [`OperationError::Stopped`] for an armed stop. The diagram is unchanged.
    /// Panics from the caller's algebra propagate unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::diagram::RationalWeights;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&tree, [1, 2]);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let value = engine.evaluate(&f, &RationalWeights::unit(3))?;
    /// assert_eq!(value.to_integer(), 6.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
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

    fn leaf(&self, var: VarId, label: LeafLabel) -> S::Value {
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
