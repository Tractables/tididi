<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="400">
</p>

[![Rust](https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg)](https://github.com/Tractables/tididi/actions/workflows/rust.yml)

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
cargo add tididi --git https://github.com/Tractables/tididi
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

## Continue with your task

Start with [the configuration walkthrough](https://tractables.github.io/tididi/tididi/guide/examples/configurations/index.html).
It encodes rules for a backup application, counts its valid configurations,
finds one solution, and counts the configurations that enable remote backups.

Use the [task guide] to find an operation, or continue with another walkthrough:

| Walkthrough | What it shows |
| --- | --- |
| [Conditional probability](https://tractables.github.io/tididi/tididi/guide/examples/probability/index.html) | Compute the probability of rain given wet grass, then change the priors. |
| [Reachable states](https://tractables.github.io/tididi/tididi/guide/examples/reachability/index.html) | Find reachable states and check that a forbidden state cannot be reached. |
| [Save and reload diagrams](https://tractables.github.io/tididi/tididi/guide/examples/persistence/index.html) | Restore two rules onto one shared vtree, then combine them. |
| [Execution controls](https://tractables.github.io/tididi/tididi/guide/examples/execution/index.html) | Bound a batch and release idle working buffers. |
| [Variable grouping](https://tractables.github.io/tididi/tididi/guide/examples/vtrees/index.html) | Compare the same function under two vtrees. |
| [A custom statistic](https://tractables.github.io/tididi/tididi/guide/examples/statistics/index.html) | Traverse the stored nodes and pairs. |

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

[`Tdd`]: https://tractables.github.io/tididi/tididi/diagram/struct.Tdd.html
[task guide]: https://tractables.github.io/tididi/tididi/guide/api/index.html
[TDD data model]: https://tractables.github.io/tididi/tididi/guide/model/index.html
[API reference]: https://tractables.github.io/tididi/tididi/
[architecture reference]: https://tractables.github.io/tididi/tididi/guide/architecture/index.html
