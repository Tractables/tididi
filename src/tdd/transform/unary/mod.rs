//! One-TDD transforms.
//!
//! Each takes a single TDD and produces a transformed TDD:
//! - **negate** — complement (`negate`, `make_full`)
//! - **condition** — literal conditioning (`condition_var`)
//! - **project** — existential projection (`project_var`, `project_vars`, …)
//! - **restrict** — restrict-to-care (`restrict`)
//! - **marginalize** — summing vtree levels out into per-node counts or weights

pub mod negate;
pub mod marginalize;
pub mod project;
pub mod condition;
pub mod restrict;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
