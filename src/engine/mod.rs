//! The session: limits, memory probes, meters, scratch pools.
//!
//! An engine holds what an operation runs under and what it reuses, never what
//! it produces: the diagram's contents belong to [`crate::diagram`], and the
//! operations themselves to [`crate::apply`], [`crate::marginal`],
//! [`crate::reduce`] and [`crate::restructure`].
//!
//! Entry points: [`Engine::new`] opens a session; [`Engine::limits`] reaches the
//! armed [`Limits`], which [`LimitSet`] describes and
//! [`Limits::install`]/[`Limits::scope`] arm; [`Limits::meters`] reads what the
//! last operation spent, and [`MemPressure`] installs the host's memory probes.
//! Every operation is a method on the engine.
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
//! | conjunction, disjunction, negation | [`Engine::and`], [`Engine::or`], [`negate`](crate::apply::negate) | `&`, `\|`, `!` |
//! | model count | [`Engine::model_count`] | [`Tdd::model_count`](crate::Tdd::model_count) |
//!
//! Marginalization, reduction, projection, conditioning, restriction and
//! rotation search have a real form and no sugar.

mod limits;
pub(crate) mod pool;
mod memory;
mod meters;
mod poll;
mod stop;
mod tuning;

pub use limits::{LimitScope, LimitSet, Limits, WorkMark};
pub use memory::MemPressure;
pub use meters::{ApplyMeters, MergeProgress};
pub use stop::{Scheduled, Stop, StopAt};

pub(crate) use limits::{ByteCharge, PollGate, PAIR_ELEM_BYTES};
pub(crate) use tuning::Tuning;
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
