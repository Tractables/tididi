//! The invariant checkers, one per numbered invariant. Compiled in every
//! debug build, because the reduction's `debug_assertions` sites call them,
//! and under the `testing` feature in any profile.
//!
//! A checker reports and never repairs: restoring an invariant belongs to the
//! pass that broke it, in [`crate::reduce`] or [`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels). Every
//! checker walks the whole diagram, which is why a consumer's release build
//! does not carry them. A test suite built on the crate reaches them through
//! `test_helpers`.
//!
//! Every checker returns `Ok(())` or `Err(String)` naming the violation. The
//! invariant table in `docs/architecture.md` names the checker that decides
//! each invariant; the marginal-form checks are reached through the
//! `marginal` submodule.

mod canonicity;
mod rotation;
pub(crate) mod signature;
mod soundness;
mod structure;

pub use canonicity::check_canonicity;
pub use rotation::assert_rotation_locality;
pub use soundness::check_determinism;
pub use structure::{check_no_false_nodes, validate_vtree_structure};

use crate::diagram::Tdd;

/// Panic naming the checker and the caller's label, or carry on.
fn require(label: &str, checker: &str, r: Result<(), String>) {
    if let Err(e) = r {
        panic!("{label}: {checker}: {e}");
    }
}

/// Run all fast invariant checks (structure + no_false_nodes + canonicity).
///
/// Convenience wrapper that runs the three cheapest checks in sequence.
/// Suitable for a diagram of any size.
///
/// Cost: O(diagram size).
pub fn check_all_fast(tdd: &Tdd, label: &str) {
    require(label, "vtree structure", validate_vtree_structure(tdd));
    require(label, "no_false_nodes", check_no_false_nodes(tdd));
    require(label, "canonicity", check_canonicity(tdd, 3));
}

pub mod marginal;
mod marginal_counts;

#[cfg(test)]
mod tests;
