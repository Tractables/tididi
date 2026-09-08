//! Two-TDD operations.
//!
//! - **conjoin** — inner-node conjunction (AND) via the compacting product
//!   construction (`apply_and` and friends).
//! - **`conjoin_clause`** — specialized TDD × clause conjunction.
//! - **leaf** — leaf-level apply (`CONJOIN_GRID` + leaf label operations).
//! - **grid** — per-level grid descriptor for the product construction.
//! - **disjoin** — disjunction (`apply_or` / `try_apply_or`).

pub mod conjoin;
pub mod conjoin_clause;
pub mod leaf;
mod grid;
pub mod disjoin;
