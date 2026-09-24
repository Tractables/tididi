//! The invariant checkers, one per numbered invariant. Compiled in every
//! debug build, because the library's own `debug_assert` sites call them, and
//! under the `testing` feature in any profile.
//!
//! A checker reports and never repairs: restoring an invariant belongs to the
//! pass that broke it, in [`crate::reduce`] or [`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels). Every
//! checker walks the whole diagram, which is why a consumer's release build
//! does not carry them. A test suite built on the crate reaches them through
//! `test_helpers`.
//!
//! Every checker returns `Ok(())` or `Err(String)` naming the violation. The
//! marginal-canonical-form checks and the model-count localizer are reached
//! through the `marginal` submodule, which is the one path to them.
//!
//! ## Available checks
//!
//! | Checker | Cost | When to use |
//! |---------|------|-------------|
//! | `validate_vtree_structure` | O(size) | Always |
//! | `check_no_false_nodes` | O(nodes) | Always |
//! | `check_no_false_nodes_in_levels` | O(nodes) | Pre-minimize |
//! | `check_canonicity` | O(size × rounds) | After minimize |
//! | `check_determinism` | O(width² × apply / level + size) | Small diagrams only (≤5 vars). Includes leaf-level label-mode consistency. |

mod canonicity;
mod rotation;
pub(crate) mod signature;
mod soundness;
mod structure;

pub use canonicity::check_canonicity;
pub use rotation::debug_assert_rotation_locality;
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
