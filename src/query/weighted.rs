//! The value of a weighted diagram.

use crate::diagram::{Tdd, WeightVal};
use crate::engine::Engine;

/// The diagram's value under its attached
/// [`WeightStore`](crate::diagram::WeightStore), or `None` when the diagram
/// carries no store. ⊥ has the store's zero.
///
/// A weighted marginalization usually leaves the output level explicit and
/// marginalizes only levels below it, so this folds the explicit levels above the
/// marginal ones on demand from the store's values and leaf weights; when the
/// output level is itself marginal it reads the stored value directly. The
/// diagram is borrowed and unchanged, and the fold runs on a transient
/// engine, so nothing is charged to a limit.
///
/// # Panics
///
/// Panics if the output level is marginal but its value is absent from the
/// store, and if the fold's allocation is refused.
pub fn weighted_value(tdd: &Tdd) -> Option<WeightVal> {
    let eng = Engine::new();
    let ws = tdd.weights.as_ref()?;
    Some(crate::marginal::weighted_output_value(&eng, tdd, ws))
}
