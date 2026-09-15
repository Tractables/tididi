//! Structural satisfiability queries on compiled diagrams.

use crate::value::ColumnRetention;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::engine::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

impl Engine {
    /// Whether a structural diagram has at least one satisfying assignment.
    ///
    /// Borrows the diagram and accepts nonminimal input. Literal weights are
    /// ignored, so a satisfiable function remains satisfiable even when its
    /// weighted value is zero. For an assignment itself, use
    /// [`Engine::satisfying_assignment`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let either = engine.clause(&tree, [1, 2])?;
    /// assert!(engine.is_sat(&either)?);
    /// let neither = engine.cube(&tree, [-1, -2])?;
    /// let impossible = engine.and(either, neither)?;
    /// assert!(!engine.is_sat(&impossible)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// Checks that all levels are structural, then reads the false sentinel.
    /// No scratch buffers or minimization are needed: every live structural node
    /// has a nonempty pair list whose children are satisfiable on disjoint variables.
    /// The structural check takes time proportional to the vtree's size.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel) for
    /// discarded structure, or
    /// [`OperationError::Stopped`](crate::OperationError::Stopped) for an armed stop.
    /// The borrowed diagram is unchanged.
    pub fn is_sat(&self, f: &Tdd) -> Result<bool, crate::OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() { return Err(crate::OperationError::Stopped); }
        let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
        for t in f.vtree().bottomup() {
            lim.poll(&mut gate, 1)?;
            f.require_structure_at(t)?;
        }
        lim.flush_poll(&mut gate)?;
        Ok(!f.is_zero())
    }
}

/// Check whether a diagram is satisfiable (has at least one model).
///
/// Structural diagrams answer from the false sentinel without minimization.
/// For a diagram containing count-marginal levels, minimize first so zero-count
/// contributions have been removed; a count-marginal root answers directly from
/// its stored count. Literal weights do not affect structural satisfiability.
/// Use [`Engine::is_sat`] for a checked query restricted to structural diagrams.
///
/// # Panics
///
/// Panics if the output level is weight-marginal: its per-node values are
/// semiring weights, and a weight of zero does not mean the node has no model.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::query::is_sat_minimized;
/// use tididi::reduce::minimize;
///
/// let tree = Arc::new(Vtree::balanced(2));
/// let mut f = Tdd::clause(&tree, [1]) & Tdd::clause(&tree, [-1]);
/// minimize(&mut f);
/// assert!(!is_sat_minimized(&f));
/// # tididi::test_helpers::assert_canonical(&f);
/// ```
pub fn is_sat_minimized(f: &Tdd) -> bool {
    // `ZERO` sentinel means the diagram computes the constant-false function.
    if f.is_zero() {
        return false;
    }
    let out_vtree = f.output.vtree;
    if f.vtree.node(out_vtree).is_leaf() {
        // Implicit leaf: any index in {One=0, Pos=1, Neg=2} is satisfiable.
        return true;
    }
    let out_level = &f.levels[out_vtree.idx()];
    let out_i = f.output.local.idx();
    if let Some(counts) = out_level.marginal_counts() {
        // A count of `u128::MAX` stands for a larger exact count, still > 0.
        return counts[out_i] > 0;
    }
    assert!(
        !out_level.is_weight_marginal(),
        "is_sat_minimized: the output level {:?} is weight-marginal; \
         its values are weights, which do not decide satisfiability",
        out_vtree
    );
    let out_node = &out_level.nodes[out_i];
    out_level.pairs_iter_of(out_node).next().is_some()
}

/// True iff the diagram's output node is satisfiable, computed by a full
/// Boolean bottom-up pass: the model counter's traversal with every count
/// collapsed to `> 0`, so it agrees with `model_count(f) > 0` on every input,
/// including a non-canonical diagram whose output node's pairs all bottom out
/// in zero-count children. [`is_sat_minimized`] is the O(1) form for a
/// minimized diagram.
pub(crate) fn is_sat_structural(f: &Tdd) -> bool {
    if f.is_zero() {
        return false;
    }
    let eng = Engine::new();
    let fold = SatBits;
    let mut cols: Vec<Vec<bool>> = (0..f.vtree.num_nodes())
        .map(|i| fold.alloc(&eng, f.reference_slot_count(VtreeIdx(i as u32))).expect("query column allocation"))
        .collect();
    fold_bottom_up_unpolled(&fold, &eng, f, &mut cols, ColumnRetention::Frontier, |_, _| {});
    let (out_t, out_i) = (f.output.vtree.idx(), f.output.local.idx());
    cols[out_t][out_i]
}

/// The counting fold with every count collapsed to a bit: `+` is disjunction,
/// `×` is conjunction, and a node that already has a model cannot lose it, which
/// is what makes the pair loop stoppable.
struct SatBits;

impl LevelFold for SatBits {
    type Value = bool;
    type Col = Vec<bool>;

    fn alloc(&self, _eng: &Engine, width: usize) -> Result<Vec<bool>, crate::OperationError> {
        Ok(vec![false; width])
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<bool>, i: usize, v: bool) -> Result<(), crate::OperationError> {
        col[i] = v;
        Ok(())
    }

    /// The counter's leaf seeds, thresholded: only `Zero` has no model (and
    /// `Zero` is never stored at an implicit leaf level).
    fn leaf(&self, _leaf: VtreeIdx, _var: VarId, label: LeafLabel) -> bool {
        !matches!(label, LeafLabel::Zero)
    }

    /// A marginal slot has a model iff its summed count is nonzero. The overflow
    /// sentinel is `u128::MAX`, itself nonzero, so an overflowed — hence huge —
    /// count reads as satisfiable without consulting the side table.
    fn marginal_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<bool>) -> Result<(), crate::OperationError> {
        let counts = tdd.levels[t.idx()]
            .marginal_counts()
            .expect("a marginal level carries counts");
        for (i, &c) in counts.iter().enumerate() {
            col[i] = c != 0;
        }
        Ok(())
    }

    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<bool>>,
        right: Side<'_, Vec<bool>>,
    ) -> bool {
        self.sum_over_pairs(pairs, left, right)
    }
}

impl PairAlgebra for SatBits {
    fn zero(&self) -> bool {
        false
    }
    fn read(&self, col: &Vec<bool>, i: usize) -> bool {
        col[i]
    }
    fn inline(&self, count: u32) -> bool {
        count != 0
    }
    fn add_assign(&self, acc: &mut bool, v: &bool) {
        *acc |= *v;
    }
    fn mul(&self, a: &bool, b: &bool) -> bool {
        *a && *b
    }
    fn short_circuit(&self, acc: &bool) -> bool {
        *acc
    }
}
