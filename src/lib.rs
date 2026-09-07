//! `TiDiDi` core: the pure Tree Decision Diagram (TDD) data model.
//!
//! This crate holds the TDD itself — nodes, `apply`, `minimize`/reduce,
//! counting/inference, semirings, the weight/marginalization primitives — plus
//! the Vtree *structure* (nodes, rotations, hoist). It has NO cargo features,
//! reads NO environment variables, installs NO process globals (memory probes
//! arrive as data through the scoped apply-limits install), and does not depend
//! on a global allocator. Vtree *construction* (treewidth/partition heuristics, CNF-driven
//! refinement) and the CNF/preprocess pipeline are not part of this crate. They
//! live in companion projects: [vitri](https://github.com/Tractables/vitri) is
//! the CNF front end, and the `tididi-cnf` solver drives the two together.
//!
//! See `docs/tdd.md` for the data structure — vtrees, semantics, reduction
//! rules, canonicity, size guarantees — and `docs/api-guide.md` for a
//! task-oriented tour of building, combining, transforming, and querying TDDs.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use num_bigint::BigUint;
//! use tididi::tdd::Tdd;
//! use tididi::vtree::Vtree;
//!
//! // Build (x1 ∧ x2) ∨ x3 over a 3-variable vtree with the operator API,
//! // then count its models. Integers are DIMACS literals (`1` → x1).
//! let vtree = Arc::new(Vtree::balanced(3));
//! let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
//! assert_eq!(f.model_count(), BigUint::from(5u32));
//! ```

// Guards the public-release doc surface: an undocumented public item warns.
#![warn(missing_docs)]

// Tier-0 invariant assertion: O(1) cost, compiled into **every** build —
// including `--release`. Unlike `debug_assert!` (tier-2, debug-only) this fires
// in optimized benchmark binaries, so reserve it for genuinely O(1) checks whose
// value justifies a hot-path branch. For everything-on-at-near-release-speed
// use `cargo build --profile release-checked`.
//
// CRATE-INTERNAL (0.1 API freeze): it guards TiDiDi's own invariants and a
// downstream user has no reason to call it, so it is deliberately NOT
// `#[macro_export]`ed. `macro_rules!` textual scoping alone makes it visible to
// every module declared below, which is how all call sites reach it — no
// `pub(crate) use` re-export, since an unused one is a `-D warnings` error in
// the standalone release build.
macro_rules! cheap_assert {
    ($($arg:tt)*) => { ::std::assert!($($arg)*) };
}

pub mod vtree; // Variable tree (vtree): structure that governs TDD decomposition
pub mod tdd;   // Tree Decision Diagram: nodes, apply, minimize, query
