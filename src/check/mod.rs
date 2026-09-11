//! The invariant checkers, one per numbered invariant, compiled only under
//! `cfg(test)` or `debug_assertions`.
//!
//! A checker reports and never repairs: restoring an invariant belongs to the
//! pass that broke it, in [`crate::reduce`] or [`crate::marginal`]. Every
//! checker walks the whole diagram, which is why a release build does not carry
//! them, and the module is hidden from the documented API.
//!
//! Every checker returns `Ok(())` or `Err(String)` naming the violation. The
//! marginal-canonical-form checks and the model-count localizer are reached
//! through the `marginal` submodule, which is the one path to them.
//!
//! ## Available checks
//!
//! | Checker | Cost | When to use |
//! |---------|------|-------------|
//! | [`validate_vtree_structure`] | O(size) | Always |
//! | [`check_no_false_nodes`] | O(nodes) | Always |
//! | [`check_no_false_nodes_in_levels`] | O(nodes) | Pre-minimize |
//! | [`check_canonicity`] | O(size × rounds) | After minimize |
//! | [`check_minimize_soundness`] | O(size × rounds) + minimize | Before + after minimize |
//! | [`check_determinism`] | O(width² × apply / level + size) | Small diagrams only (≤5 vars). Includes leaf-level label-mode consistency. |

mod canonicity;
mod rotation;
mod signature;
mod soundness;
mod structure;

pub use canonicity::{check_canonicity, check_minimize_soundness};
pub(crate) use rotation::debug_assert_rotation_locality;
pub use soundness::check_determinism;
pub use structure::{check_no_false_nodes, check_no_false_nodes_in_levels, validate_vtree_structure};

use crate::diagram::*;

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
