//! The generators and oracles the crate's tests share.
//!
//! Compiled under the `testing` feature, and in a debug build regardless,
//! because the library's own `debug_assert` sites call the checkers in `check`.
//! A consumer that wants these in its own tests turns the feature on there.
//!
//! What every build carries is what the randomized differential suite in
//! `tests/` draws on: the seeded [`Lcg`] / [`rand_cnf`] pair under a
//! [`CnfShape`], enumeration by [`brute_force_count`], the apply-free
//! evaluator [`eval`], canonicity by [`assert_canonical`] and
//! [`assert_marginal_canonical`], structural equality by
//! [`assert_same_shape`], and [`assert_restrict_ok`]. The invariant checkers
//! in `check`, one per numbered invariant of `docs/architecture.md`, run only
//! under `cfg(test)` or `debug_assertions`, since each walks the whole
//! diagram. The rest is compiled only under `cfg(test)`, by what a test needs
//! it for:
//!
//! - `gen` — the fixed `test_cases` corpus, the `vtree_shapes` a formula is
//!   compiled against, and `queens_clauses`.
//! - `compile` — turning a formula into a diagram: `literals`, `clause`,
//!   `compile_clauses` and its engine-taking form, `and2` and `cube`, the
//!   seeded `rand_conj` / `rand_conj_over` pair, `reroot_to_child` for the
//!   low-rooted operand shape, plus the small constructors (`pair`, `rat`,
//!   `exact_weight`) that hand-built fixtures need.
//! - `oracle` — the projected `brute_force_pmc`, what is built on the
//!   evaluator (`equiv`, `equiv_nf`, `count_is_zero`), the support oracles,
//!   and `deadline_probe` for the cut-at-a-deadline family.
//! - `toy` — hand-encoded marginal diagrams too small to reach by compiling,
//!   and `marginalize_subtree`, the pass a test applies to one subtree of a
//!   compiled one.
//! - `access` — what reaches crate-private state: the infallible
//!   `CountVecExt` fixture builders, `stopping_engine`, and the whole-tree
//!   `rotate_left` / `rotate_right`.
//!
//! The canonicity oracle has a test of its own:
//! `reduce::tests::canonicity::every_route_to_one_function_minimizes_to_the_same_diagram`
//! reaches one function by three routes and demands one diagram back.

#[cfg(any(test, debug_assertions))]
pub mod check;
#[cfg(test)]
mod access;
#[cfg(test)]
mod compile;
pub mod r#gen;
pub mod oracle;
#[cfg(test)]
mod toy;

#[cfg(test)]
pub(crate) use access::*;
#[cfg(test)]
pub(crate) use compile::*;
pub use r#gen::*;
pub use oracle::*;
#[cfg(test)]
pub(crate) use toy::*;
