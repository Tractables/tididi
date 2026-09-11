//! Rotation search and graft over a compiled diagram.
//!
//! Both change the diagram's shape and neither changes the function it denotes.
//! The vtree operations they follow are [`crate::vtree`]; the counting fold that
//! reads the result is [`crate::query`].
//!
//! Entry points:
//!
//! - [`Engine::rotation_search`](crate::Engine::rotation_search) descends an
//!   objective over rotations under the caller's limits;
//!   [`search::RotationObjective`] is the trait a caller implements to descend
//!   something other than size.
//! - [`Tdd::graft`](crate::Tdd::graft) conjoins diagrams over pairwise-disjoint
//!   variable sets into one diagram on a grafted vtree, structurally and with no
//!   apply.
//!
//! `relevel` applies one vtree rotation to an existing diagram and is what the
//! search moves with.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod graft;
