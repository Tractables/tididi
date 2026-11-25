//! The invariant checkers, one per numbered invariant. Compiled in every
//! debug build, because the reduction's `debug_assertions` sites call them,
//! and under the `testing` feature in any profile.
//!
//! A checker reports and never repairs: restoring an invariant belongs to the
//! pass that broke it, in [`crate::reduce`] or [`Tdd::marginalize`](crate::Tdd::marginalize). Every
//! checker walks the whole diagram, which is why a consumer's release build
//! does not carry them. A test suite built on the crate reaches them through
//! `test_helpers`.
//!
//! Every checker returns `Ok(())` or `Err(String)` naming the violation.
//! `docs/invariants.md` (the `guide::invariants` page) states each invariant
//! once, with its number, the checker that decides it and where the checkers
//! run; the comments here say how a check works. The marginal-form checks are
//! reached through the `marginal` submodule.

mod canonicity;
mod implicit;
mod rotation;
pub(crate) mod signature;
mod soundness;
mod structure;
mod trimmed;

pub use canonicity::{check_canonicity, check_distinct_functions};
pub use trimmed::{check_certified, check_no_true_node, check_padded_support, check_trimmed, reachable_true_labels};
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
/// vtree structure; the storage checks of
/// [`TddBuilder::finish`](crate::diagram::TddBuilder::finish) (determinism as
/// no repeated pair at every level with no summed-out level under it, and the
/// leaf form); no
/// false nodes; on a home the trimmed form ([`check_trimmed`]: no stored pair
/// names a variable leaf `One`, and a certified diagram with no summed-out
/// level has the support as its leaves and no node that denotes every
/// assignment of its variables, [`check_no_true_node`]; on a plain root a
/// certified diagram names `One` exactly the leaves outside its support,
/// [`check_padded_support`]);
/// implicit levels and, on a diagram with no summed-out level, distinct
/// functions within a level ([`check_distinct_functions`], 3 rounds); on a
/// certified diagram canonical form ([`check_canonicity`]) instead. Panics
/// naming the first that fails.
///
/// On an uncertified diagram it does not decide canonical form: an
/// unminimized result with unreachable nodes or twins passes, and so does a
/// summed one whose sum-out made two nodes equal.
///
/// Cost: O(diagram size).
pub fn check_all_fast(tdd: &Tdd, label: &str) {
    require(label, "vtree structure", validate_vtree_structure(tdd));
    require(label, "storage", crate::diagram::check_stored(tdd).map_err(|e| e.to_string()));
    require(label, "no_false_nodes", check_no_false_nodes(tdd));
    if tdd.vtree.on_home() {
        require(label, "trimmed", check_trimmed(tdd, tdd.is_canonical()));
        if tdd.is_canonical() {
            require(label, "no true node", check_no_true_node(tdd));
        }
    } else if tdd.is_canonical() {
        require(label, "padded support", check_padded_support(tdd));
    }
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
