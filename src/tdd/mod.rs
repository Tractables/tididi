//! Tree Decision Diagram (TDD) implementation.
//!
//! A TDD is a canonical representation of a Boolean function, structured
//! according to a vtree (variable tree). Each vtree node `t` has a set of
//! **t-nodes** — sub-functions over `t`'s variables. Internal t-nodes have
//! **input pairs** (`left_child`, `right_child`) representing their decomposition.
//! The TDD's output node at the vtree root represents the overall function.
//!
//! See `docs/tdd.md` for the data structure and `docs/api-guide.md` for a
//! task-oriented tour of the API.

pub(crate) mod utils; // General-purpose utilities (sorting networks, etc.)
pub(crate) mod counts; // u128-sentinel + BigUint-side-table count discipline: Count/CountRead/CountVec
pub mod types;       // Core types: Tdd, TddLevel, TddNodeData, InputPair, etc.
pub mod build;       // TDD construction: clause_to_tdd, constant_one, constant_zero
pub mod transform;   // TDD→TDD transformations: pairwise (conjoin/disjoin) + unary (negate/condition/project/restrict/…)
pub mod minimize;    // Canonicalization: prune → twin contraction
pub mod restructure; // Vtree-restructuring of a compiled TDD: rotate, size-driven search, graft
pub mod query;       // Read-only queries: model counting, SAT check, semiring eval, reduction metrics, invariant checkers
pub mod ops;         // std::ops operator sugar for Tdd (&, |, !) — thin delegations to transform::*
pub mod weight_store; // External side-table of weighted marginal values (--weighted)
pub mod io;          // TDD serialization: DOT/Graphviz rendering + .tdd text format
pub(crate) mod marg_slots; // Shared marginal-slot primitives (ChildSide/CountKey/SlotInterner/…)
pub mod mem_pressure;   // Installed memory-pressure interface: fn-pointer table installed by the compiler (decouples tdd from jemalloc mem.rs)
pub mod config;         // Installed runtime tuning knobs (TIDIDI_* config as data): decouples tdd from process env (public-release P2c)

// `negate_tdd` — the TDD complement op — is DEFINED in `transform::unary::negate`
// (it lives next to `make_full`, which it uses, and `apply_or` in the sibling
// `pairwise::disjoin` module, which it powers), but readers look for a core op
// like this at the module root alongside the other TDD operations. Re-export it
// here so `tdd::negate_tdd` resolves; the definition stays in
// `transform::unary::negate::negate_tdd`.
pub use transform::unary::negate::negate_tdd;

// Re-export the core `Tdd` type at the module root so `tididi::tdd::Tdd`
// resolves — the natural place a reader looks for the crate's central type
// (its definition stays in `types::tdd`). Mirrors the `negate_tdd` re-export.
pub use types::Tdd;

// Shared test-only helpers for tdd::* tests AND the downstream compiler crate's
// tests (which reach them across the crate boundary — dependency crates never
// compile with `cfg(test)`, so this can't be `#[cfg(test)]`-gated). `doc(hidden)`
// keeps them out of the published API surface. (public-release P3a)
#[doc(hidden)]
pub mod test_helpers;
