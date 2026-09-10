//! The prose guides, compiled with the crate.
//!
//! Each submodule is one file of `docs/`, included verbatim. Every code fence
//! in a guide is a doctest and every item a guide names is a link, so a guide
//! that has drifted from the API fails the build instead of misleading a
//! reader.

#[doc = include_str!("../docs/api-guide.md")]
pub mod api {}

#[doc = include_str!("../docs/tdd.md")]
pub mod model {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}
