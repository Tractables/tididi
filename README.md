<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="480">
</p>

[![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A Rust library for Tree Decision Diagrams (TDDs): represent Boolean functions,
combine and transform them, count satisfying assignments, and evaluate weights.
A vtree groups the variables; a right-linear vtree gives the OBDD special case.
TDDs were introduced in
[*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537).

## Install

```sh
cargo add tididi
```

Requires Rust 1.88 or later.

## Build a function and ask for an answer

```rust
use std::sync::Arc;
use tididi::{Engine, Vtree};

let engine = Engine::new();
let tree = Arc::new(Vtree::balanced(3));
let x = engine.literal(&tree, 1)?;
let y = engine.literal(&tree, 2)?;
let z = engine.literal(&tree, 3)?;

// (x AND y) OR z
let both = engine.and(x, y)?;
let f = engine.or(both, z)?;
assert_eq!(engine.model_count(&f)?, 5u32.into());

let model = engine.satisfying_assignment(&f)?.unwrap();
assert_eq!(model.len(), 3); // one literal for every variable
# Ok::<(), tididi::OperationError>(())
```

Integers name signed, 1-based literals: `1` means `x`, `-2` means `NOT y`.
An `Engine` owns reusable scratch and resource limits; each diagram owns its
structure and shares its vtree through `Arc`. See [`Tdd`] for retaining a
function across multiple transformations and [`Engine`] for handling limits.

## Choose your next task

The [task guide](docs/api-guide.md) links short examples for:

- [Building Boolean functions](docs/api-guide.md#build-a-boolean-function).
- [Counting, finding assignments and comparing functions](docs/api-guide.md#ask-questions-about-a-function).
- [Conditioning, quantifying and substituting variables](docs/api-guide.md#change-assignments-or-variables).
- [Evaluating probabilities and repeated observations](docs/api-guide.md#evaluate-probabilities-or-repeated-observations).
- [Controlling resources and minimizing diagrams](docs/api-guide.md#control-resources-and-diagram-size).
- [Saving, drawing and traversing diagrams](docs/api-guide.md#save-or-inspect-a-diagram).

[`minimize`] finds the canonical minimal TDD under its current vtree;
[`rotation_search`] searches different vtree shapes. You can construct a vtree
with the library's constructors or load its `.vtree` format; the
[`vitri`](https://github.com/Tractables/vitri) crate can derive one from CNF structure.

## Run complete examples

- `cargo run --example build_minimize_count`: construct, minimize and count.
- `cargo run --example dimacs_count -- examples/tiny.cnf 5 6 --check`: a caller
  that parses DIMACS, compiles clauses, projects and evaluates weights.
- `cargo run --example statistic`: inspect the stored representation.

## Reference

[API documentation](https://docs.rs/tididi),
[TDD data model](docs/tdd.md), and
[implementation architecture](docs/architecture.md).

## Citing

TDDs were introduced in the paper below; [`CITATION.cff`](CITATION.cff) carries
the same reference in machine-readable form.

```bibtex
@article{capelli2026canonical,
  title   = {A Canonical Generalization of {OBDD}},
  author  = {Capelli, Florent and Choi, YooJung and Mengel, Stefan and
             Mu{\~n}oz, Mart{\'i}n and Van den Broeck, Guy},
  journal = {arXiv preprint arXiv:2604.05537},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.05537}
}
```

## License

Apache License, Version 2.0 ([LICENSE](./LICENSE)).

[`minimize`]: https://docs.rs/tididi/latest/tididi/reduce/fn.minimize.html
[`Tdd`]: https://docs.rs/tididi/latest/tididi/diagram/struct.Tdd.html
[`Engine`]: https://docs.rs/tididi/latest/tididi/engine/struct.Engine.html
[`rotation_search`]: https://docs.rs/tididi/latest/tididi/engine/struct.Engine.html#method.rotation_search
