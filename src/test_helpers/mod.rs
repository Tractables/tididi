//! The generators, oracles and invariant checkers the crate's tests share.
//!
//! `check` compiles in every debug build, because the reduction's
//! `debug_assertions` sites call it. Everything else needs the `testing` feature,
//! which is how a consumer's tests, and this crate's integration tests, reach
//! it: the seeded `Lcg` / `rand_cnf` pair under a `CnfShape`, the
//! enumeration oracles, the apply-free evaluator `eval`, canonicity by
//! `assert_canonical`, structural equality by `assert_same_shape`, and
//! `assert_restrict_ok`, and `ClauseStore`, which decides CNF encodings and
//! probes without a solver; `compile` turns a formula or a truth table into a
//! diagram and holds the small constructors (`pair`, `rat`, `exact_weight`)
//! hand-built fixtures need. Three modules are `cfg(test)` throughout:
//!
//! - `toy` — hand-encoded marginal diagrams too small to reach by compiling,
//!   and `marginalize_subtree`.
//! - `diagrams` — the hand-built `chain`, `inline_marginal` and
//!   `marginal_boundary` with the side constructors they use, seeded diagrams
//!   with unreachable nodes or summed-out levels, and `node_value`, which
//!   evaluates one node of them.
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
mod cnf;
#[cfg(any(test, feature = "testing"))]
mod compile;
#[cfg(test)]
mod diagrams;
#[cfg(any(test, feature = "testing"))]
mod r#gen;
#[cfg(any(test, feature = "testing"))]
mod oracle;
#[cfg(test)]
mod toy;

#[cfg(test)]
pub(crate) use access::*;
#[cfg(any(test, feature = "testing"))]
pub use cnf::ClauseStore;
#[cfg(any(test, feature = "testing"))]
pub use compile::*;
#[cfg(test)]
pub(crate) use diagrams::*;
#[cfg(any(test, feature = "testing"))]
pub use r#gen::*;
#[cfg(any(test, feature = "testing"))]
pub use oracle::*;
#[cfg(test)]
pub(crate) use toy::*;
