//! Read counts, assignments, and other properties of a diagram.
//!
//! Start with [`Tdd::is_sat`](crate::Tdd::is_sat) to test for a solution,
//! [`Tdd::model_count`](crate::Tdd::model_count) for an exact
//! count or [`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment)
//! for a witness. These queries borrow the diagram and accept nonminimal structural inputs.
//! [`ModelCounter`] retains counting state for repeated evidence updates.
//!
//! [`Tdd::equivalent`](crate::Tdd::equivalent) compares Boolean functions;
//! [`Tdd::implies`](crate::Tdd::implies) tests entailment.
//! [`Tdd::support`](crate::Tdd::support) returns variables that affect the
//! function, while [`implied_literals`] finds literals true in every model.
//! Each item states its structural or minimization requirements.
//!
//! [`Tdd::evaluate`](crate::Tdd::evaluate) accepts an
//! [`EvalAlgebra`](crate::diagram::EvalAlgebra), such as
//! [`RationalWeights`](crate::diagram::RationalWeights), for a fresh evaluation.
//! [`Tdd::weighted_value`](crate::Tdd::weighted_value) instead reads the
//! weights attached to a diagram, including stored marginal values, using the
//! [`Arithmetic`](crate::diagram::Arithmetic) chosen for its store.
//!
//! Checked diagram methods return errors; convenience functions
//! use the vtree context and document their panic behavior.

pub(crate) mod count;
pub(crate) mod fold;
pub(crate) mod sat;
pub(crate) mod evaluate;
pub(crate) mod support;
pub(crate) mod weighted;
mod boolean;

pub use count::{
    node_counts_u128, KeepAllColumns, ColumnRetention,
    KeepFrontier, ModelCounter, Retention, PinSemantics,
};
pub(crate) use count::model_count;
pub use sat::is_sat_minimized;
pub use evaluate::evaluate;
pub use support::implied_literals;
pub use weighted::weighted_value;

#[cfg(test)]
mod tests;
