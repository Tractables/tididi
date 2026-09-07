# tididi — Tree Decision Diagrams [![Rust](https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg)](https://github.com/Tractables/tididi/actions/workflows/rust.yml) [![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A pure-Rust library for **Tree Decision Diagrams (TDDs)** — a canonical
knowledge-compilation language that generalizes Ordered Binary Decision Diagrams
(OBDDs) by replacing the linear variable order with a **vtree** (a binary tree
over the variables). TDDs keep the tractable queries that make OBDDs useful —
model counting, SAT, conditioning — and offer canonicity guarantees with
efficient apply, at sizes that scale with the **treewidth** of the function
rather than its pathwidth. TDDs were introduced in Capelli, Choi, Mengel,
Muñoz & Van den Broeck, [*A Canonical Generalization of
OBDD*](https://arxiv.org/abs/2604.05537).

**Construct** TDDs from constants, literals, and clauses, and combine them with
the pairwise `apply` operations (conjoin, disjoin) and unary transformations:
negation, conditioning, restriction, existential projection, and
marginalization. **Minimize** reduces any TDD to its canonical minimal form,
and vtree rotation/restructuring primitives (with local-search drivers) reshape
the vtree under a live diagram. **Query** for satisfiability, exact model
counts (arbitrary precision), weighted model counts and general semiring
evaluations, implied literals, and support. The crate has **no cargo features,
reads no environment variables, and has no C/C++ dependencies** — it is
portable, ordinary Rust. Vtree *construction* heuristics and the DIMACS-CNF
compilation / model counting driver are separate concerns and live in companion
projects (see below).

## Install

```sh
cargo add tididi
```

The count-query examples below return `num-bigint` types (e.g. `num_bigint::BigUint`),
so add that crate too:

```sh
cargo add num-bigint
```

## Usage

Build TDDs over three variables and combine, negate, and forget:

```rust
use std::sync::Arc;
use num_bigint::BigUint;
use tididi::tdd::Tdd;
use tididi::tdd::transform::unary::project::project_var;
use tididi::vtree::{VarId, Vtree};

let vtree = Arc::new(Vtree::balanced(3));           // variables x1, x2, x3

// Arbitrary Boolean combinations — conjunction, disjunction:
let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
assert_eq!(f.model_count(), BigUint::from(5u32));   // (x1 ∧ x2) ∨ x3

// Exact canonical negation:
let g = !f;
assert_eq!(g.model_count(), BigUint::from(3u32));

// Forgetting (existential projection): ∃x1. ¬f
let h = project_var(&g, VarId(0));                  // VarId is 0-based: x1
assert_eq!(h.model_count(), BigUint::from(4u32));   // x1 now free: 2 × 2 don't-care assignments
```

`Vtree::balanced(n)` builds a balanced vtree over `n` variables and `Tdd::clause`
compiles a single clause into a canonical TDD (integer literals use the 1-based
DIMACS sign convention). The `&` and `|` operators conjoin and disjoin two TDDs,
`!` negates canonically, and `project_var` existentially forgets a variable
(`VarId` is 0-based) — every result is a fully reduced TDD, not just a formula.
`model_count` returns the exact unweighted model count over the vtree's variables
as a `num_bigint::BigUint` (a forgotten variable stays as a free don't-care, so it
still multiplies the count).

Vtrees are built without a CNF: `Vtree::leaf`, `Vtree::join`, `balanced_over`
and `linear_from_order` over a variable order, `random`, `from_vtree_text` /
`to_vtree_text` (the SDD `.vtree` format), `graft` (independent subtrees under
one spine), `project_to_vars`, and `validate` for hand-built input. `Tdd::graft`
conjoins TDDs over disjoint variable sets structurally, without running apply —
the way separately compiled pieces become one diagram.

## Visualization

Export a TDD and its vtree as [Graphviz](https://graphviz.org/) DOT:

```rust
use tididi::tdd::io::dot::{tdd_to_dot, vtree_to_dot};

std::fs::write("tdd.dot", tdd_to_dot(&g).unwrap()).unwrap();
std::fs::write("vtree.dot", vtree_to_dot(&vtree, Some(&g))).unwrap();
// render: dot -Tsvg tdd.dot -o tdd.svg
```

Below: the annotated vtree (left) and TDD circuit (right) for the 14-variable
majority function under a random vtree. `vtree_to_dot` colors each internal
vtree node light yellow → dark red by its input-pair count `s` and annotates it
with `w`, the number of distinct subfunctions at that level — when the TDD is
too dense to inspect directly, the vtree still shows at a glance where the
complexity is concentrated.

<p align="center">
  <!-- Absolute URLs: this README ships inside the crates.io package, where the
       repo's docs/ directory does not exist — relative links would 404 there. -->
  <img src="https://raw.githubusercontent.com/Tractables/tididi/main/docs/vtree_example.svg" alt="Annotated vtree" height="350">
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Tractables/tididi/main/docs/tdd_example.svg" alt="TDD circuit" height="350">
</p>

## Compiling CNF

This crate is the TDD *core*: it operates on the TDDs and vtrees you hand it and
knows nothing about CNF. Two companion projects cover the rest of that pipeline.
[**vitri**](https://github.com/Tractables/vitri) is the CNF front end — DIMACS
parsing, preprocessing, and the vtree-construction heuristics that decide which
vtree a formula should be compiled under. **`tididi-cnf`** is the solver: it
drives the two together into a CNF-to-TDD compiler and an exact model counter.

## Documentation

API docs: [docs.rs/tididi](https://docs.rs/tididi).

Guides bundled with the source:

- [`docs/tdd.md`](https://github.com/Tractables/tididi/blob/main/docs/tdd.md) — the data structure: vtrees, semantics, reduction rules, canonicity, size guarantees.
- [`docs/api-guide.md`](https://github.com/Tractables/tididi/blob/main/docs/api-guide.md) — task-oriented tour: building, combining, transforming, and querying TDDs.

The stored diagram is the traversal contract: algorithm writers read levels and
pairs directly (see the `tdd::types` module docs and `examples/`).

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
