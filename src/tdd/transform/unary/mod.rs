//! One-TDD transforms.
//!
//! Each takes a single TDD and produces a transformed TDD:
//! - **negate** — complement (`negate`, `negate_tdd`, `make_full`)
//! - **condition** — literal conditioning (`condition_var`)
//! - **project** — existential projection (`project_var`, `project_vars`, …)
//! - **restrict** — restrict-to-care (`restrict`)
//! - **demarginalize** — marginal → indicator (`demarginalize_to_indicator`)
//! - **marginalize** — weight context + marginalization primitives

pub mod negate;
pub mod marginalize;
pub mod project;
pub mod condition;
pub mod restrict;
pub mod demarginalize;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
