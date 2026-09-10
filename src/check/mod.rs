//! Invariant checkers for diagrams — test infrastructure, hidden from the
//! documented API.
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
//! | [`check_reduced_size_sanity`] | O(size) + BigUint | After minimize |
//! | [`check_determinism`] | O(width² × apply / level + size) | Small diagrams only (≤5 vars). Includes leaf-level label-mode consistency. |

mod canonicity;
mod signature;
mod soundness;
mod structure;

pub use canonicity::*;
pub use soundness::*;
pub use structure::*;

use crate::diagram::*;
use crate::query::{reduced_size, ReductionRule};

/// Panic naming the checker and the caller's label, or carry on.
fn require(label: &str, checker: &str, r: Result<(), String>) {
    if let Err(e) = r {
        panic!("{label}: {checker}: {e}");
    }
}

/// Run all fast invariant checks (structure + no_false_nodes + canonicity).
///
/// Convenience wrapper that runs the three cheapest checks in sequence.
/// Suitable for use on any compiled diagram, including large easy benchmarks.
///
/// Cost: O(diagram size).
pub fn check_all_fast(tdd: &Tdd, label: &str) {
    require(label, "vtree structure", validate_vtree_structure(tdd));
    require(label, "no_false_nodes", check_no_false_nodes(tdd));
    require(label, "canonicity", check_canonicity(tdd, 3));
}

/// Run all invariant checks including minimize soundness and reduced size sanity.
///
/// **Mutates `tdd`** (calls minimize once via `check_minimize_soundness`).
/// Suitable only for moderately-sized diagrams — see individual checker docs for costs.
pub fn check_all_deep(tdd: &mut Tdd, label: &str) {
    check_all_fast(tdd, label);
    require(label, "minimize_soundness", check_minimize_soundness(tdd, 3));
    require(label, "reduced_size_sanity", check_reduced_size_sanity(tdd));
    // Also call reduced_size to trigger its inline debug_assert!s
    let _ = reduced_size(tdd, ReductionRule::R1Sdd);
}

pub mod marginal;
mod marginal_counts;

#[cfg(test)]
mod invariants_tests;
