//! Model counting, satisfiability, algebra evaluation.
//!
//! Everything here reads a finished diagram and returns a number, a verdict or a
//! set; nothing mutates. Building and transforming a diagram is [`crate::apply`],
//! [`crate::marginal`] and [`crate::reduce`].
//!
//! Entry points:
//!
//! - Counting: [`Tdd::model_count`](crate::Tdd::model_count), and
//!   [`ModelCounter`] for a count under a partial assignment that
//!   re-folds only the levels a pin change reaches. [`Engine::model_count`](crate::Engine::model_count) is the
//!   same count under the caller's limits.
//! - Satisfiability and implied literals: [`is_sat_minimized`], [`implied_literals`].
//! - Algebra: [`evaluate()`] folds any
//!   [`EvalAlgebra`](crate::diagram::EvalAlgebra) bottom-up;
//!   [`RationalWeights`](crate::diagram::RationalWeights) and
//!   [`SignedLog`](crate::diagram::SignedLog) are the two supplied domains.
//! - Weighted: [`weighted_value`] folds a diagram carrying a
//!   [`WeightStore`](crate::diagram::WeightStore) down to its value.
//!
//! - Boolean queries: [`Engine::equivalent`](crate::Engine::equivalent),
//!   [`Engine::implies`](crate::Engine::implies),
//!   [`Engine::support`](crate::Engine::support) and
//!   [`Engine::satisfying_assignment`](crate::Engine::satisfying_assignment).
//!
//! Engine methods provide checked queries; free functions describe their own
//! resource behavior. The submodules are an implementation layout.

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
