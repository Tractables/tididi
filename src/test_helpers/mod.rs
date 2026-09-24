//! The generators, oracles and invariant checkers the crate's tests share.
//!
//! `check` compiles in every debug build, because the library's own
//! `debug_assert` sites call it. Everything else needs the `testing` feature,
//! which is how a consumer's tests, and this crate's integration tests, reach
//! it: the seeded `Lcg` / `rand_cnf` pair under a `CnfShape`, the
//! enumeration oracles, the apply-free evaluator `eval`, canonicity by
//! `assert_canonical`, structural equality by `assert_same_shape`, and
//! `assert_restrict_ok`. The modules under `cfg(test)` need crate-private
//! state:
//!
//! - `compile` — turning a formula into a diagram, and the small constructors
//!   (`pair`, `rat`, `exact_weight`) hand-built fixtures need.
//! - `toy` — hand-encoded marginal diagrams too small to reach by compiling,
//!   and `marginalize_subtree`.
//! - `access` — the infallible `CountVecExt` fixture builders,
//!   `stopping_engine`, and the whole-tree `rotate_left` / `rotate_right`.
//!
//! The canonicity oracle has a test of its own:
//! `reduce::tests::canonicity::every_route_to_one_function_minimizes_to_the_same_diagram`
//! reaches one function by three routes and demands one diagram back.

pub mod check;
#[cfg(test)]
mod access;
#[cfg(test)]
mod compile;
#[cfg(any(test, feature = "testing"))]
pub(crate) mod r#gen;
#[cfg(any(test, feature = "testing"))]
pub(crate) mod oracle;
#[cfg(test)]
mod toy;

#[cfg(test)]
pub(crate) use access::*;
#[cfg(test)]
pub(crate) use compile::*;
#[cfg(any(test, feature = "testing"))]
pub use r#gen::*;
#[cfg(any(test, feature = "testing"))]
pub use oracle::*;
#[cfg(test)]
pub(crate) use toy::*;
