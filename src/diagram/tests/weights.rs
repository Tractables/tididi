//! Attaching a weight store: refused once a level holds integer counts.

use std::sync::Arc;

use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::marginal::marginalize;
use crate::test_helpers::rat;
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

#[test]
#[should_panic(expected = "Tdd::set_weights: level")]
fn a_store_is_refused_after_a_level_was_summed_out_as_counts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    marginalize(&eng, &mut f, &[left]).unwrap();
    let weights: Vec<_> = (0..4).map(|_| (rat(1, 2), rat(1, 3))).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_weights(&weights), Arithmetic::ExactRational));
}
