//! The generators, oracles and invariant checkers the crate's tests share.
//!
//! `check` compiles in every debug build, because the library's own
//! `debug_assert` sites call it. Everything else needs the `testing` feature,
//! which is how a consumer's tests, and this crate's integration tests, reach
//! it: the seeded `Lcg` / `rand_cnf` pair under a `CnfShape`, the
//! enumeration oracles, the apply-free evaluator `eval`, canonicity by
//! `assert_canonical`, structural equality by `assert_same_shape`, and
//! `assert_restrict_ok`; `compile` turns a formula or a truth table into a
//! diagram and holds the small constructors (`pair`, `rat`, `exact_weight`)
//! hand-built fixtures need. Two modules need crate-private state and are
//! `cfg(test)`:
//!
//! - `toy` — hand-encoded marginal diagrams too small to reach by compiling,
//!   and `marginalize_subtree`.
//! - `access` — the infallible `CountVecExt` fixture builders,
//!   `stopping_engine`, the whole-tree `rotate_left` / `rotate_right`, and
//!   `reroot_to_child`.
//!
//! The canonicity oracle has a test of its own:
//! `reduce::tests::canonicity::every_route_to_one_function_minimizes_to_the_same_diagram`
//! reaches one function by three routes and demands one diagram back.

pub mod check;
#[cfg(test)]
mod access;
#[cfg(any(test, feature = "testing"))]
mod compile;
#[cfg(any(test, feature = "testing"))]
pub(crate) mod r#gen;
#[cfg(any(test, feature = "testing"))]
pub(crate) mod oracle;
#[cfg(test)]
mod toy;

#[cfg(test)]
pub(crate) use access::*;
#[cfg(any(test, feature = "testing"))]
pub use compile::*;
#[cfg(any(test, feature = "testing"))]
pub use r#gen::*;
#[cfg(any(test, feature = "testing"))]
pub use oracle::*;
#[cfg(test)]
pub(crate) use toy::*;
