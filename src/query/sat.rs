//! Structural satisfiability queries on compiled diagrams.

use crate::value::Retention;
use crate::diagram::{LeafLabel, PairsIter, Tdd};
use crate::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up, LevelFold, PairAlgebra, Side};

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
        lim.check_stop()?;
        if f.is_zero() { return Ok(false); }
        let out_vtree = f.output.vtree;
        let out_level = &f.levels[out_vtree.idx()];
        if out_level.is_weight_marginal() {
            return Err(crate::OperationError::IncompatibleWeights);
        }
        if f.vtree.node(out_vtree).is_leaf() { return Ok(true); }
        let out_i = f.output.local.idx();
        if let Some(counts) = out_level.marginal_counts() { return Ok(counts[out_i] > 0); }
        // A structural diagram stores no unsatisfiable node, and neither does
        // a marginal one whose reduction worklists are drained: the output
        // node then decides. Weighted values never stand for an unsatisfiable
        // function either (a weight store rules out count-marginal levels).
        // An edited count-marginal diagram may still hold structural nodes
        // over zero-count values, so it is walked.
        if f.dirty.is_empty() || f.weights.is_some() || !f.has_marginal_level() {
            return Ok(out_level.pairs_iter_of(&out_level.nodes[out_i]).next().is_some());
        }
        is_sat_structural(self, f)
    }
}

/// True iff the diagram's output node is satisfiable, computed by a full
/// Boolean bottom-up pass under the caller's limits: the model counter's
/// traversal with every count collapsed to `> 0`, so it agrees with
/// `f.model_count()? > 0` on every input, including a non-canonical diagram
/// whose output node's pairs all bottom out in zero-count children.
pub(crate) fn is_sat_structural(eng: &Engine, f: &Tdd) -> Result<bool, crate::OperationError> {
    if f.is_zero() {
        return Ok(false);
    }
    let fold = SatBits;
    let mut cols: Vec<Vec<bool>> = Vec::new();
    eng.limits().reserve(&mut cols, f.vtree.num_nodes())?;
    for i in 0..f.vtree.num_nodes() {
        cols.push(fold.alloc(eng, f.reference_slot_count(VtreeIdx(i as u32)))?);
    }
    let mut poll = eng.limits().gate();
    fold_bottom_up(&fold, eng, f, &mut cols, Retention::Frontier, Some(&mut poll), |_, _| Ok(()))?;
    let (out_t, out_i) = (f.output.vtree.idx(), f.output.local.idx());
    Ok(cols[out_t][out_i])
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

    /// A count slot has a model iff its summed count is nonzero. The overflow
    /// sentinel is `u128::MAX`, itself nonzero, so an overflowed — hence huge —
    /// count reads as satisfiable without consulting the side table. A weighted
    /// slot always has one: a zero weight does not establish unsatisfiability.
    fn marginal_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<bool>) -> Result<(), crate::OperationError> {
        match tdd.levels[t.idx()].marginal_counts() {
            Some(counts) => for (i, &c) in counts.iter().enumerate() { col[i] = c != 0; },
            None => col.fill(true),
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
