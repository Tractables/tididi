//! Twin contraction phase: merge TDD nodes with identical parent contexts.
//!
//! Two nodes at the same vtree level are "twins" if they appear in exactly
//! the same positions in parent input pairs (same parent node, same sibling partner).
//! Since twins always co-occur, their functions can be disjoined into a single
//! node without changing the TDD's overall function.
//!
//! **Important:** Contraction must always run unconditionally — never skip it based
//! on node count or TDD size. A size-based limit only optimizes for one timeout
//! horizon: it speeds up instances near the current timeout but leaves the TDD
//! bloated, so at a higher timeout those instances would be slower or unsolvable.
//!
//! **Interface to the prune phase.** This phase is decoupled from prune
//! (`minimize/prune.rs`) except through the `Tdd` dirty-contract worklists:
//! prune (and the content-twin merge in `content_twin.rs`) seed
//! `dirty_contract`/`dirty_leaf_contract` via `Tdd::mark_contract_dirty`, and the
//! incremental strategies here drain them. The phase orchestration — including the
//! `canonicalize_content_twins` fixpoint that drives `content_twin.rs` — lives in
//! `minimize/mod.rs`.

mod scratch;
mod fingerprint;
mod strategies;
mod merge;
mod dup_resolve; // duplicate-pair scaling/resolution (contract-internal: merge.rs, strategies.rs)
pub(crate) mod contract_leaf; // leaf-side twin specialization (orchestrated by minimize::mod's content-twin loop)
pub(crate) mod content_twin; // content-twin merge over every explicit level (driven by minimize::canonicalize_content_twins)
pub(crate) mod p_fusion; // same-left pair fusion (production caller: strategies.rs; binary: compile/step.rs)

pub(crate) use strategies::contract_all_twins;

#[cfg(test)]
pub(crate) use strategies::contract_all_twins_topdown;

// Allocation-failure injection seam for the OverBudget-safety regression tests
// (minimize/tests.rs). See `scratch::fail_point`.
#[cfg(test)]
pub(crate) use scratch::{arm_fail_after, disarm_fail};

#[cfg(debug_assertions)]
pub(crate) use strategies::contract_all_twins_with_locality;

#[cfg(test)]
#[path = "tests.rs"]
mod inline_denorm_tests;
