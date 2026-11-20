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
mod implicit;
mod rotation;
pub(crate) mod signature;
mod soundness;
mod structure;
mod trimmed;

pub use canonicity::{check_canonicity, check_distinct_functions};
pub use trimmed::{check_certified, check_trimmed, reachable_true_labels};
pub use implicit::{check_implicit_levels, check_left_pairs_taken};
pub use rotation::assert_rotation_locality;
pub use soundness::{assert_no_mixed_leaf_side, check_determinism, labels_at};
pub use structure::{check_loose_levels, check_no_false_nodes, check_no_false_nodes_in_levels, validate_vtree_structure};

use crate::diagram::{LEAF_WIDTH, Tdd};
use crate::vtree::VtreeIdx;

/// [`Tdd::reference_slot_count`], but [`LEAF_WIDTH`] at a region not summed
/// out: an operand placed beside another operand's region names `One` there,
/// slot 0 of a level with no slots.
fn checked_width(tdd: &Tdd, t: VtreeIdx) -> usize {
    if tdd.vtree.node(t).is_region() && !tdd.level(t).is_marginal() {
        LEAF_WIDTH
    } else {
        tdd.reference_slot_count(t)
    }
}

/// Panic naming the checker and the caller's label, or carry on.
fn require(label: &str, checker: &str, r: Result<(), String>) {
    if let Err(e) = r {
        panic!("{label}: {checker}: {e}");
    }
}

/// Run the fast checks of what holds after any operation, minimized or not:
/// vtree structure, no false nodes, implicit levels and, on a diagram with no
/// summed-out level, distinct functions within a level
/// ([`check_distinct_functions`], 3 rounds); on a certified diagram canonical
/// form ([`check_canonicity`]) instead. Panics naming the first that fails.
///
/// On an uncertified diagram it does not decide canonical form: an
/// unminimized result with unreachable nodes or twins passes, and so does a
/// summed one whose sum-out made two nodes equal.
///
/// Cost: O(diagram size).
pub fn check_all_fast(tdd: &Tdd, label: &str) {
    require(label, "vtree structure", validate_vtree_structure(tdd));
    require(label, "no_false_nodes", check_no_false_nodes(tdd));
    match (tdd.is_canonical(), tdd.has_marginal_level()) {
        (true, _) => require(label, "canonicity", check_canonicity(tdd, 3)),
        (false, false) => require(label, "distinct functions", check_distinct_functions(tdd, 3)),
        (false, true) => {}
    }
    require(label, "implicit levels", check_implicit_levels(tdd));
}

pub mod marginal;
mod marginal_counts;

#[cfg(test)]
mod tests;
