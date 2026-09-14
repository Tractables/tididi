//! Read counts, assignments, and other properties of a diagram.
//!
//! Start with [`Engine::model_count`](crate::Engine::model_count) for an exact
//! count or [`Engine::satisfying_assignment`](crate::Engine::satisfying_assignment)
//! for a witness. Both borrow the diagram and accept nonminimal structural inputs.
//! [`ModelCounter`] retains counting state for repeated evidence updates.
//!
//! [`Engine::equivalent`](crate::Engine::equivalent) compares Boolean functions;
//! [`Engine::implies`](crate::Engine::implies) tests entailment.
//! [`Engine::support`](crate::Engine::support) returns variables that affect the
//! function, while [`implied_literals`] finds literals true in every model.
//! Each item states its structural or minimization requirements.
//!
//! [`Engine::evaluate`](crate::Engine::evaluate) accepts an
//! [`EvalAlgebra`](crate::diagram::EvalAlgebra), such as
//! [`RationalWeights`](crate::diagram::RationalWeights), for a fresh evaluation.
//! [`Engine::weighted_value`](crate::Engine::weighted_value) instead reads the
//! weights attached to a diagram, including stored marginal values, using the
//! [`Arithmetic`](crate::diagram::Arithmetic) chosen for its store.
//!
//! Engine methods return resource errors to the caller; convenience functions
//! use a temporary engine and document their panic behavior.

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
