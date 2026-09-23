//! Read counts, assignments, and other properties of a diagram.
//!
//! Start with [`Tdd::is_sat`](crate::Tdd::is_sat) to test for a solution,
//! [`Tdd::model_count`](crate::Tdd::model_count) for an exact
//! count or [`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment)
//! for a witness. These queries borrow the diagram and accept nonminimal structural inputs.
//! [`ModelCounter`] retains counts and [`Evaluator`] retains algebra values for repeated evidence updates.
//!
//! [`Tdd::equivalent`](crate::Tdd::equivalent) compares Boolean functions;
//! [`Tdd::implies`](crate::Tdd::implies) tests entailment.
//! [`Tdd::support`](crate::Tdd::support) returns variables that affect the
//! function, while [`Tdd::implied_literals`](crate::Tdd::implied_literals) finds literals true in every model.
//! Each item states its structural or minimization requirements.
//!
//! [`Tdd::evaluate`](crate::Tdd::evaluate) accepts an
//! [`EvalAlgebra`](crate::diagram::EvalAlgebra), such as
//! [`RationalWeights`](crate::diagram::RationalWeights), for a fresh evaluation.
//! [`Tdd::weighted_value`](crate::Tdd::weighted_value) instead reads the
//! weights attached to a diagram, including stored marginal values, using the
//! [`Arithmetic`](crate::diagram::Arithmetic) chosen for its store.
//!
//! Diagram queries return errors and use their vtree context automatically.

pub(crate) mod count;
pub(crate) mod fold;
pub(crate) mod sat;
pub(crate) mod evaluate;
mod boolean;
mod cache;
mod evaluator;
pub use evaluator::{Evaluation, Evaluator, OwnedEvaluator, BoundEvaluation, BoundEvaluator};

pub use count::{Counter, ModelCounter, OwnedModelCounter, BoundCounter, BoundModelCounter, Retention, PinSemantics};

#[cfg(test)]
mod tests;
