<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="400">
</p>

[![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A Rust library for Tree Decision Diagrams (TDDs): representations of Boolean
functions that you can build once and query repeatedly. Encode constraints to
count valid configurations or find a solution, evaluate probabilities under
changing assumptions, or compute reachable states in a transition system.

A TDD decomposes a function along a **vtree**, a binary tree over its variables.
TDDs can be strictly more succinct than ordered binary decision diagrams (OBDDs).
The representation and its minimization algorithm are described in [*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537).

## Start with a function

Add the crate to your project:

```sh
cargo add tididi
```

This example builds `(x ∧ y) ∨ z` and counts its satisfying assignments:

```rust
use std::sync::Arc;
use tididi::{and, literal, or, Vtree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vtree = Arc::new(Vtree::balanced(3));
    let x = literal(&vtree, 1)?;
    let y = literal(&vtree, 2)?;
    let z = literal(&vtree, 3)?;

    let f = or(and(x, y)?, z)?;
    let count = u64::try_from(f.model_count()?)?;
    assert_eq!(count, 5);
    Ok(())
}
```

There are four models with `z` true, and one more with `z` false and both
`x` and `y` true. Counts range over every variable in the vtree, including
variables the function leaves free. The conversion to `u64` returns an error
if the exact count does not fit.

Signed integers name literals: `1` means `x`, `-2` means `¬y`, and zero is
invalid. Use the same `Arc<Vtree>` for functions you intend to combine.

Each [`Tdd`] owns its diagram. Boolean operations consume their operands; clone an
operand first if you need to keep it. Cloning copies diagram storage and shares
the vtree; queries such as `model_count` borrow the diagram.
The shared vtree retains reusable working buffers, used automatically by these
operations. Constructors, transformations and queries return `Result`; `?`
propagates an operation error. Constants and storage accessors return directly.

For bounded work, the context lends a batch engine through
[`Context::with_limits`]. Diagrams own their results; the engine only supplies
working buffers and execution controls.

## Continue with your task

Start with [the configuration walkthrough](docs/examples/configurations.md).
It encodes rules for a backup application, counts its valid configurations,
finds one solution, and counts the configurations that enable remote backups.

Then use the [task guide] to find operations for your own application, or
continue with another walkthrough. Execution controls and representation
specialization follow the basic workflows. Each page explains the program in steps
and links to the full runnable source.

| Walkthrough | What it shows |
| --- | --- |
| [Conditional probability](docs/examples/probability.md) | Compute the probability of rain given wet grass, then change the priors. |
| [Reachable states](docs/examples/reachability.md) | Find reachable states and check that a forbidden state cannot be reached. |
| [Save and reload diagrams](docs/examples/persistence.md) | Restore two rules onto one shared tree, then combine them. |
| [Execution controls](docs/examples/execution.md) | Bound a batch and release idle working buffers. |
| [Variable grouping](docs/examples/vtrees.md) | Compare the same function under two vtrees. |
| [A custom statistic](docs/examples/statistics.md) | Traverse the stored nodes and pairs. |

For the concepts behind the API, read the [TDD data model]. The
[API reference] documents each operation's input requirements, result, and
errors; the [architecture reference] describes the implementation for contributors.
For local documentation, run `cargo doc --no-deps` and open
`target/doc/tididi/index.html`.

## Citing

TDDs were introduced in the following paper:

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

[`Literal`]: https://docs.rs/tididi/latest/tididi/diagram/struct.Literal.html
[`Tdd`]: https://docs.rs/tididi/latest/tididi/diagram/struct.Tdd.html
[`Context::with_limits`]: https://docs.rs/tididi/latest/tididi/engine/struct.Context.html#method.with_limits
[task guide]: https://docs.rs/tididi/latest/tididi/guide/api/index.html
[TDD data model]: https://docs.rs/tididi/latest/tididi/guide/model/index.html
[API reference]: https://docs.rs/tididi
[architecture reference]: https://docs.rs/tididi/latest/tididi/guide/architecture/index.html
