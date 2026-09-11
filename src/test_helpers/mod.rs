//! Test-only helpers shared across the crate test modules.
//!
//! Four submodules, by what a test needs them for:
//!
//! - `gen` — the formulas a test runs on: the fixed [`test_cases`] corpus,
//!   the [`vtree_shapes`] a formula is compiled against, [`queens_clauses`],
//!   and the seeded [`Lcg`] / [`rand_cnf`] pair every randomized sweep draws
//!   from.
//! - [`compile`] — turning a formula into a diagram: [`literals`], [`clause`],
//!   [`compile_clauses`] and its engine-taking form, [`and2`] and [`cube`],
//!   the seeded [`rand_conj`] / [`rand_conj_over`] pair, [`reroot_to_child`]
//!   for the low-rooted operand shape, plus the small constructors
//!   ([`pair`], [`rat`], [`exact_weight`]) that hand-built fixtures need.
//! - [`oracle`] — the ways a test decides a diagram is right: brute-force
//!   enumeration ([`brute_force_count`] and the projected [`brute_force_pmc`]),
//!   canonicity ([`assert_canonical`]), structural equality
//!   ([`assert_same_shape`] over [`normalized_levels`]), the apply-free
//!   evaluator [`eval`] and what is built on it ([`equiv`], [`equiv_nf`],
//!   [`count_is_zero`], [`assert_restrict_ok`]), the support oracles, and
//!   `deadline_probe` for the cut-at-a-deadline family.
//! - [`toy`] — hand-encoded marginal diagrams too small to reach by
//!   compiling, and [`marginalize_subtree`], the pass a test applies to one
//!   subtree of a compiled one.
//! - `access` — what reaches crate-private state, compiled only under
//!   `cfg(test)`: the infallible `CountVecExt` fixture builders,
//!   `stopping_engine`, and the whole-tree `rotate_left` / `rotate_right`.
//!
//! The canonicity oracle has a test of its own:
//! `reduce::tests::canonicity::every_route_to_one_function_minimizes_to_the_same_diagram`
//! reaches one function by three routes and demands one diagram back.

#[cfg(test)]
mod access;
pub mod compile;
pub mod r#gen;
pub mod oracle;
pub mod toy;

#[cfg(test)]
pub(crate) use access::*;
pub use compile::*;
pub use r#gen::*;
pub use oracle::*;
pub use toy::*;
