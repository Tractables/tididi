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
use crate::vtree::{VarId, Vtree};
use std::sync::Arc;

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
