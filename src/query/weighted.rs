//! The value of a weighted diagram.

use crate::diagram::{LeafLabel, Tdd, WeightStore, WeightVal};
use crate::value::{ColumnRetention, unwrap_infallible, FoldInput, ValueDomain, WeightFold};
use crate::limits::RecoveryPanic;
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
/// allocation policy; this convenience form uses a fresh, unarmed engine.
///
/// # Panics
///
/// Panics if the fold's allocation is refused.
pub fn weighted_value(tdd: &Tdd) -> Option<WeightVal> {
    Engine::new().weighted_value(tdd)
}

impl Engine {
    /// Fold the diagram's attached weights using this engine's allocation policy.
    ///
    /// Returns `None` without a weight store. This read does not poll stop rules.
    ///
    /// # Panics
    ///
    /// Panics if the fold's allocation is refused.
    pub fn weighted_value(&self, tdd: &Tdd) -> Option<WeightVal> {
        let _op = self.limits().begin_operation();
        Some(weighted_output_value(self, tdd, tdd.weights.as_ref()?))
    }
}

/// The weighted value of `tdd`'s output node under `ws`; `tdd` must have been
/// weighted with `ws`.
fn weighted_output_value(eng: &Engine, tdd: &Tdd, ws: &WeightStore) -> WeightVal {
    let vtree = &tdd.vtree;
    // UNSAT / constant-false output: the `ZERO` sentinel carries no level slot
    // (`output.local` is the `ZERO` idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return ws.wzero();
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return ws.level(out_t).expect("output level weight-marginalized")[out_i].clone();
    }
    // Leaf output level: the fold below stores nothing for leaves (their values
    // come from the semiring on demand), so read the leaf value directly.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return ws.leaf_val(var, LeafLabel::from_idx(out_i));
    }
    let mut computed: Vec<Option<Vec<WeightVal>>> = vec![None; vtree.num_nodes()];
    // Only the root value is read, so child columns are released as their
    // parent completes (`ColumnRetention::Frontier`). The "already stored" test
    // is this diagram's own marginality rather than `WeightStore::is_set`: the
    // store is shared, so a column at this index may belong to another live
    // `Tdd` while this diagram's level is still structural.
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    unwrap_infallible(WeightFold::ensure::<RecoveryPanic>(
        eng,
        VtreeIdx(out_t as u32),
        FoldInput { vtree, levels: &tdd.levels, store: ws },
        &mut computed,
        &marginal,
        ColumnRetention::Frontier,
    ));
    computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone()
}
