//! The fixtures every file here reads through `use super::*`.

use super::*;
use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::query::model_count;
use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};
use std::sync::Arc;

mod canonicity;
mod marginal;
mod pruning;
mod twins;
mod twins_budget;
mod twins_inline;
mod whole_diagrams;
