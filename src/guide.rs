//! The prose guides of `docs/`, included as documentation so their examples and
//! their identifiers are checked by the build.
//!
//! The module holds no code and no behaviour; everything a guide describes lives
//! in the module it is about.
//!
//! Entry points: [`api`] is the capability-by-capability reference, [`model`]
//! the data model, and [`architecture`] the module boundaries and the numbered
//! invariants. Each is one file of `docs/`, included verbatim; every code fence
//! is a doctest and every item named is a link, so a guide that has drifted from
//! the API fails the build instead of misleading a reader.

#[doc = include_str!("../docs/api-guide.md")]
pub mod api {}

#[doc = include_str!("../docs/tdd.md")]
pub mod model {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}
