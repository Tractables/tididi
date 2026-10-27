//! A streamed sum-out level whose column outgrows the slots a marginal
//! reference can name is refused, whatever its value domain.

use std::sync::Arc;

use crate::{Engine, OperationError, Tdd};
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::test_helpers::{rand_cnf, CnfShape, Lcg};
use crate::vtree::{Vtree, VtreeIdx};

/// The conjunction of `clauses` over `tree`, weighted by `store` if given.
fn conjoin_all(eng: &Engine, tree: &Arc<Vtree>, clauses: &[Vec<i32>], store: Option<&WeightStore>) -> Tdd {
    let mut d = eng.one(tree);
    if let Some(store) = store {
        d.set_weights(store.empty_like()).unwrap();
    }
    for clause in clauses {
        let mut c = eng.clause(tree, clause.iter().copied()).unwrap();
        if let Some(store) = store {
            c.set_weights(store.empty_like()).unwrap();
        }
        d = eng.and(d, c).unwrap();
    }
    d
}

/// The value of a summed-out diagram, as text: a count, or a weight.
fn value(eng: &Engine, d: &Tdd) -> String {
    match d.weights.is_some() {
        true => format!("{:?}", eng.weighted_value(d).unwrap()),
        false => eng.model_count(d).unwrap().to_string(),
    }
}

#[test]
fn a_streamed_column_past_the_slot_cap_is_refused() {
    let eng = Engine::new();
    let num_vars = 10;
    let tree = Arc::new(Vtree::balanced(num_vars));
    let mut rng = Lcg::new(23);
    let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 10, width: 3 });
    let (a, b) = clauses.split_at(clauses.len() / 2);
    let targets: Vec<VtreeIdx> = tree.bottomup().filter(|&t| !tree.node(t).is_leaf()).collect();
    let stores = [
        None,
        Some(WeightStore::new(RationalWeights::unit(num_vars as usize), Arithmetic::ExactRational)),
        Some(WeightStore::new(RationalWeights::unit(num_vars as usize), Arithmetic::SignedLog)),
        Some(WeightStore::new(RationalWeights::unit(num_vars as usize), Arithmetic::ScaledFloat)),
    ];
    for store in &stores {
        let (f, g) = (conjoin_all(&eng, &tree, a, store.as_ref()), conjoin_all(&eng, &tree, b, store.as_ref()));
        let expected = value(&eng, &eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap());
        eng.limits().pin_slot_cap(Some(1));
        let refused = eng.and_marginalizing(f.clone(), g.clone(), &targets);
        eng.limits().pin_slot_cap(None);
        assert_eq!(refused.err(), Some(OperationError::IndexOverflow), "{:?}", store.as_ref().map(WeightStore::arithmetic));
        let again = eng.and_marginalizing(f, g, &targets).unwrap();
        assert_eq!(value(&eng, &again), expected);
    }
}
