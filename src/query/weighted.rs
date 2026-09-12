//! The value of a weighted diagram.

use crate::diagram::{Tdd, WeightVal};
use crate::engine::Engine;

/// The diagram's value under its attached
/// [`WeightStore`](crate::diagram::WeightStore), or `None` in integer mode.
///
/// A weighted marginalization usually leaves the output level explicit and
/// marginalizes only levels below it, so this folds the explicit levels above the
/// marginal ones on demand from the store's values and leaf weights; when the
/// output level is itself marginal it reads the stored value directly.
///
/// # Panics
///
/// Panics if the output level is marginal but its value is absent from the
/// store.
pub fn weighted_value(tdd: &Tdd) -> Option<WeightVal> {
    let eng = Engine::new();
    let ws = tdd.weights.as_ref()?;
    Some(crate::marginal::weighted_output_value(&eng, tdd, ws))
}
