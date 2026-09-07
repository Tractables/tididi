//! Vtree-restructuring of a compiled TDD (structure-changing, count-preserving).
//!
//! - **rotate** — apply a vtree rotation to an existing TDD (re-level the diagram).
//! - **search** — size-driven rotation search (greedy + dependent pairs + ILS).
//! - **graft** — `Tdd::graft`: the conjunction of TDDs over disjoint variable sets, built structurally on a grafted vtree.

pub mod rotate;
pub mod search;
pub mod graft;
