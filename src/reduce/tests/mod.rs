//! The fixtures every file here reads through `use super::*`.

use super::*;
use super::contract::{contract_all_twins, contract_leaf::contract_leaf_twins};
fn canonicalize_content_twins(eng: &Engine, tdd: &mut crate::Tdd) -> Result<(), OperationError> {
    super::driver::Reduction::new(eng, tdd).content_twins()
}
use crate::apply::apply_and;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;

use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use std::sync::Arc;

/// A count-marginal diagram straight out of a conjunction, still owing every
/// pass.
fn edited_marginal_diagram(eng: &Engine) -> crate::Tdd {
    let vtree = Arc::new(Vtree::balanced(8));
    let mut g = eng.clause(&vtree, [1, 5]).unwrap();
    let (left, _) = vtree.children(vtree.root());
    let summed: Vec<VtreeIdx> = vtree.internal_bottomup_slice().iter().copied()
        .filter(|&t| { let mut cur = t; loop {
            if cur == left { break true; }
            match vtree.node(cur).parent() { Some(p) => cur = p, None => break false }
        } })
        .collect();
    eng.marginalize_levels(&mut g, &summed).unwrap();
    let f = eng.and(g, eng.literal(&vtree, 8).unwrap()).unwrap();
    assert!(f.has_marginal_level() && !f.dirty.is_empty());
    f
}

mod canonicity;
mod fixpoint;
mod marginal;
mod pair_arena;
mod plans;
mod pruning;
mod twins;
mod twins_budget;
mod twins_inline;
mod whole_diagrams;
