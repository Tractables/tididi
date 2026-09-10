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
use crate::apply::project::{project_var, project_vars, Projection};
use crate::apply::restrict::{restrict, CareCanonical};
use crate::apply::{apply_and, apply_or};
use crate::build::{clause_to_tdd, constant_one, constant_zero};
use crate::diagram::Tdd;
use crate::engine::Engine;
use crate::query::model_count;
use crate::vtree::{VarId, Vtree};

mod project;
mod restrict;
mod restrict_marginal;
mod restrict_marginal_gate;
mod restrict_scaling;
mod support;
