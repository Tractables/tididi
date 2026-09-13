//! Structural satisfiability queries on compiled diagrams.

use crate::value::ColumnRetention;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::engine::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

// ---------------------------------------------------------------------------

/// Check whether a diagram is satisfiable (has at least one model).
///
/// Requires a minimized diagram. After minimization, dead input pairs (pairs where
/// a child computes zero) have been removed, so an internal node with a
/// non-empty input set is guaranteed to have at least one satisfying assignment.
/// Checking the output node structurally is therefore O(1) and avoids the
/// O(size × `BigUint`) cost of `model_count`. On an unminimized diagram the
/// answer can be true for a function with no model. ⊥ is unsatisfiable. A
/// count-marginal output level answers from its output node's count, which is
/// exact on any diagram.
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
        .map(|i| fold.alloc(&eng, f.reference_slot_count(VtreeIdx(i as u32))))
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

    fn alloc(&self, _eng: &Engine, width: usize) -> Vec<bool> {
        vec![false; width]
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<bool>, i: usize, v: bool) {
        col[i] = v;
    }

    /// The counter's leaf seeds, thresholded: only `Zero` has no model (and
    /// `Zero` is never stored at an implicit leaf level).
    fn leaf(&self, _var: VarId, label: LeafLabel) -> bool {
        !matches!(label, LeafLabel::Zero)
    }

    /// A marginal slot has a model iff its summed count is nonzero. The overflow
    /// sentinel is `u128::MAX`, itself nonzero, so an overflowed — hence huge —
    /// count reads as satisfiable without consulting the side table.
    fn marginal_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<bool>) {
        let counts = tdd.levels[t.idx()]
            .marginal_counts()
            .expect("a marginal level carries counts");
        for (i, &c) in counts.iter().enumerate() {
            col[i] = c != 0;
        }
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
