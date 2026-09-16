//! Structural satisfiability queries on compiled diagrams.

use crate::value::ColumnRetention;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

impl Engine {
    /// Run [`Tdd::is_sat`](crate::Tdd::is_sat) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`](crate::OperationError::Stopped)
    /// on cancellation.
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

impl Tdd {
    /// Check satisfiability from the output of a minimized diagram.
    ///
    /// Structural diagrams need no minimization. Diagrams with count-marginal
    /// levels must be minimized first so zero-count contributions are removed;
    /// a count-marginal root answers from its stored count. Literal weights on
    /// structural levels do not affect the answer. This query allocates no scratch.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`](crate::OperationError::IncompatibleWeights)
    /// for a non-false weight-marginal output: a zero weight does not establish
    /// unsatisfiability. False diagrams always return `Ok(false)`.
    pub fn is_sat_minimized(&self) -> Result<bool, crate::OperationError> {
        if self.is_zero() { return Ok(false); }
        let out_vtree = self.output.vtree;
        let out_level = &self.levels[out_vtree.idx()];
        if out_level.is_weight_marginal() {
            return Err(crate::OperationError::IncompatibleWeights);
        }
        if self.vtree.node(out_vtree).is_leaf() { return Ok(true); }
        let out_i = self.output.local.idx();
        if let Some(counts) = out_level.marginal_counts() { return Ok(counts[out_i] > 0); }
        Ok(out_level.pairs_iter_of(&out_level.nodes[out_i]).next().is_some())
    }
}

impl Engine {
    /// Run [`Tdd::is_sat_minimized`] after checking this batch's stop condition.
    ///
    /// Returns the query's errors or [`OperationError::Stopped`](crate::OperationError::Stopped).
    pub fn is_sat_minimized(&self, f: &Tdd) -> Result<bool, crate::OperationError> {
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() { return Err(crate::OperationError::Stopped); }
        f.is_sat_minimized()
    }
}

/// True iff the diagram's output node is satisfiable, computed by a full
/// Boolean bottom-up pass: the model counter's traversal with every count
/// collapsed to `> 0`, so it agrees with `f.model_count()? > 0` on every input,
/// including a non-canonical diagram whose output node's pairs all bottom out
/// in zero-count children. [`Tdd::is_sat_minimized`] is the O(1) form for a
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
