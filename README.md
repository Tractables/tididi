<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="480">
</p>

[![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A Rust library for Tree Decision Diagrams (TDDs). Build a Boolean function
once, then combine it with other functions, count its satisfying assignments,
or evaluate it under different literal weights.

A TDD decomposes a function along a **vtree**, a binary tree over its variables.
This generalizes an ordered binary decision diagram: a right-linear vtree
corresponds to a variable order. The representation and its minimization
algorithm are described in [*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537).

## Start with a function

Add the crate to a Rust project; Rust 1.88 or later is required:

```sh
cargo add tididi
```

This example builds `(x ∧ y) ∨ z` and counts its satisfying assignments:

```rust
use std::sync::Arc;
use tididi::{Engine, Vtree};

fn main() -> Result<(), tididi::OperationError> {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let x = engine.literal(&tree, 1)?;
    let y = engine.literal(&tree, 2)?;
    let z = engine.literal(&tree, 3)?;

    let both = engine.and(x, y)?;
    let f = engine.or(both, z)?;
    assert_eq!(engine.model_count(&f)?, 5u32.into());
    Ok(())
}
```

There are four models with `z` true, and one more with `z` false and both
`x` and `y` true. Counts range over every variable in the vtree, including
variables the function leaves free.

Signed integers name literals: `1` means `x`, `-2` means `¬y`, and zero is
invalid. The typed form, [`Literal`], uses zero-based variable identifiers.
Use the same `Arc<Vtree>` for functions you intend to combine.

The [`Engine`] holds reusable working memory and resource limits; a [`Tdd`]
owns the resulting diagram and can outlive it. Queries borrow their inputs.
Transformations taking `Tdd` consume it; the `Tdd` example shows how to keep
an original for several transformations.

## Continue with your task

The [task guide] walks through construction, queries, conditioning and
quantification, probabilities, resource limits, and persistence, with links
to the relevant API examples. For a complete program, run one of these from
a source checkout:

| Example | What it shows |
| --- | --- |
| [build_minimize_count](examples/build_minimize_count.rs)<br>`cargo run --example build_minimize_count` | Add constraints, minimize, count, and take a cofactor. |
| [probabilistic_query](examples/probabilistic_query.rs)<br>`cargo run --example probabilistic_query` | Reuse a query under changing probabilities and compute a conditional probability. |
| [symbolic_reachability](examples/symbolic_reachability.rs)<br>`cargo run --example symbolic_reachability` | Compute reachable states to a fixed point and obtain a witness. |
| [statistic](examples/statistic.rs)<br>`cargo run --example statistic` | Traverse the stored nodes and pairs. |
| [dimacs_count](examples/dimacs_count.rs)<br>`cargo run --example dimacs_count -- examples/tiny.cnf 5 6 --check` | Supply a DIMACS reader, compile its clauses, and count a projection. |

For the concepts behind the API, read the [TDD data model]. The
[API reference] documents each operation's input requirements, result, and
errors; the [architecture reference] describes the implementation for contributors.
For local documentation, run `cargo doc --no-deps` and open
`target/doc/tididi/index.html`.

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

[`Literal`]: https://docs.rs/tididi/latest/tididi/diagram/struct.Literal.html
[`Tdd`]: https://docs.rs/tididi/latest/tididi/diagram/struct.Tdd.html
[`Engine`]: https://docs.rs/tididi/latest/tididi/engine/struct.Engine.html
[task guide]: https://docs.rs/tididi/latest/tididi/guide/api/index.html
[TDD data model]: https://docs.rs/tididi/latest/tididi/guide/model/index.html
[API reference]: https://docs.rs/tididi
[architecture reference]: https://docs.rs/tididi/latest/tididi/guide/architecture/index.html
