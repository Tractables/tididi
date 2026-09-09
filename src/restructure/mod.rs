//! Vtree-restructuring of a compiled diagram (structure-changing, count-preserving).
//!
//! - **relevel** — apply a vtree rotation to an existing diagram (re-level the diagram).
//! - **search** — size-driven rotation search (greedy + dependent pairs + iterated local search).
//! - **graft** — `Tdd::graft`: the conjunction of diagrams over disjoint variable sets, built structurally on a grafted vtree.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod graft;
