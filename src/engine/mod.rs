//! The session object every operation runs on: the limits it is held to and the
//! scratch it reuses.
//!
//! An [`Engine`] is what a caller keeps between operations. Holding the scratch
//! makes the reuse explicit — two engines never share a buffer, and dropping one
//! frees everything it warmed up — and holding the limits makes the arming
//! explicit: what a conjunction is allowed to spend is a field a caller sets,
//! not ambient state it inherits.
//!
//! Every operation has one real form — an `Engine` method, or a [`crate::query`]
//! function for a read — and at most one sugar, which is the spelling a doc
//! example writes. A sugar is a one-line forward that builds a transient engine
//! and panics on failure; it is never a second implementation.
//!
//! | Operation | Real form | Sugar |
//! |---|---|---|
//! | build | [`Engine::clause`], [`Engine::one`], [`Engine::zero`] | [`Tdd::clause`](crate::Tdd::clause), [`Tdd::one`](crate::Tdd::one), [`Tdd::zero`](crate::Tdd::zero) |
//! | conjunction, disjunction, negation | [`Engine::and`], [`Engine::or`], [`crate::negate`] | `&`, `\|`, `!` |
//! | model count | [`crate::query::model_count`] | [`Tdd::model_count`](crate::Tdd::model_count) |
//!
//! Marginalization, reduction, projection, conditioning, restriction and
//! rotation search have a real form and no sugar.

mod limits;
pub(crate) mod pool;
mod memory;
mod meters;
mod poll;
mod stop;

pub use limits::{LimitScope, LimitSet, Limits, WorkMark};
pub use memory::MemPressure;
pub use meters::{ApplyMeters, MergeProgress};
pub use stop::{Scheduled, Stop, StopAt};

pub(crate) use limits::{ByteCharge, PollGate, PAIR_ELEM_BYTES};
pub(crate) use limits::policy::{ApplyBudget, RecoveryPanic, ReservePolicy};

#[cfg(test)]
pub(crate) use limits::DENSE_GROWTH_DECISION_THRESHOLD;

#[cfg(test)]
pub(crate) use memory::{vas_headroom_with_margin, SOFT_HEADROOM_MARGIN_BYTES};

#[cfg(test)]
#[path = "headroom_tests.rs"]
mod headroom_tests;

#[cfg(test)]
#[path = "limits_tests.rs"]
mod limits_tests;

pub use crate::session::Engine;
