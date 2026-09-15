//! The unary transform family — projection, conditioning, restriction —
//! together with the structural support queries.
//!
//! One file per subject. Every fixture they run on lives in
//! [`crate::test_helpers`], imported here so a sibling picks the whole set up
//! with `use super::*`.

use crate::test_helpers::*;

use std::sync::Arc;

use num_bigint::BigUint;

use crate::apply::condition_var;
use crate::apply::project::{exists_var, exists_var_with_strategy, exists_vars, exists_vars_with_strategy, QuantificationStrategy};
use crate::apply::restrict_to_care::restrict_to_care;
use crate::apply::{apply_and, apply_or};
use crate::test_helpers::clause_to_tdd;
use crate::build::{constant_one, constant_zero};
use crate::diagram::Tdd;
use crate::engine::Engine;
use crate::query::model_count;
use crate::vtree::{VarId, Vtree};

mod project;
mod restrict_to_care;
mod restrict_marginal;
mod restrict_marginal_gate;
mod restrict_scaling;
mod support;

mod operands;

mod weights;
mod inputs;

mod everyday;

mod shortcuts;
