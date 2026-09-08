//! Vtree-restructuring of a compiled TDD (structure-changing, count-preserving).
//!
//! - **relevel** — apply a vtree rotation to an existing TDD (re-level the diagram).
//! - **search** — size-driven rotation search (greedy + dependent pairs + iterated local search).
//! - **graft** — `Tdd::graft`: the conjunction of TDDs over disjoint variable sets, built structurally on a grafted vtree.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod graft;
