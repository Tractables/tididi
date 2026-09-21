<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="400">
</p>

[![Rust](https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg)](https://github.com/Tractables/tididi/actions/workflows/rust.yml)
[![Documentation](https://img.shields.io/badge/docs-main-blue)](https://tractables.github.io/tididi/tididi/)

tididi is a Rust library for **Tree Decision Diagrams (TDDs)**. A TDD represents
a Boolean function as a circuit, so you can work with its satisfying assignments
without listing them. Build a circuit once, then combine it with other circuits
or query it as your inputs change.

For Python, start with the [Python guide](https://tractables.github.io/tididi/python/)
and [installation instructions](bindings/python/README.rst).

TDDs can be much smaller in practice than widely used binary decision diagrams.
They can be made canonical, so equivalent functions have the same diagram.
They can also be combined efficiently using Boolean operations. Their
representation and minimization algorithm are described in
[*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537).

## Working with circuits

**Build and transform Boolean functions.** Combine smaller circuits with Boolean
operations, fix variables to chosen values, or eliminate variables with
quantification. The [reachability example] shows how to use conjunction,
quantification, and variable renaming to explore a transition system.

**Find solutions and count possibilities.** Check whether constraints can be
satisfied, find a solution, or count all solutions exactly. Compare functions
for equivalence or implication. The
[configuration example] builds rules for a backup application, counts the valid
configurations, and reuses the circuit as a user changes their choices.

**Evaluate weighted models.** Assign weights to literals to compute weighted
sums, including probabilities, and change those weights without rebuilding the
circuit. The [probability example] combines events and evaluates their weights
to answer a conditional-probability query. Custom evaluation algebras let the
same circuit compute other quantities, such as the [minimum configuration cost].

## Getting started

With `tididi = "0.1"` in your dependencies, this example builds `(x ∧ y) ∨ z`
and counts its satisfying assignments:

```rust
use std::sync::Arc;
use tididi::{literal, Vtree};

let vtree = Arc::new(Vtree::balanced(3));
let x = literal(&vtree, 1)?;
let y = literal(&vtree, 2)?;
let z = literal(&vtree, 3)?;
let f = (x & y) | z;
println!("Satisfying assignments: {}", f.model_count()?);
```

Output:

```text
Satisfying assignments: 5
```

There are four models with `z` true, and one more with `z` false and both
`x` and `y` true. Counts range over every variable in the vtree, including
variables the function leaves free.

A **vtree** is a binary tree that groups the circuit's variables. Use the same
`Arc<Vtree>` for circuits you intend to combine. Signed integers name literals:
`1` means `x`, `-2` means `¬y`, and zero is invalid.

Each [`Tdd`] owns its circuit. Boolean operations consume their operands; clone
an operand first if you need to keep it. Cloning copies diagram storage and shares
the vtree; queries such as `model_count` borrow the circuit.

## Representation and integration

Minimization removes redundancy, and the choice of vtree affects circuit size.
The [variable-grouping example] compares the same function under two vtrees;
the [TDD data model] explains the representation.

For applications that keep circuits between runs, the [persistence example]
shows how to save and reload them. The [execution example] adds resource limits
to a batch of operations. You can also inspect the stored nodes and pairs, as
shown in the [custom-statistic example].

Use the [API overview] to find an operation and the [API reference] for its
contract. Contributors can start with the [architecture reference].
To browse the documentation locally, run `cargo doc --no-deps`, then
`python3 scripts/prepare_docs.py target/doc`, and open `target/doc/tididi/index.html`.
This bundles the matching full programs and figures with the pages.
[GitHub releases](https://github.com/Tractables/tididi/releases) include a ZIP
of these pages; unpack it and open `index.html` to read offline.

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
[configuration example]: https://tractables.github.io/tididi/tididi/guide/examples/configurations/index.html
[reachability example]: https://tractables.github.io/tididi/tididi/guide/examples/reachability/index.html
[probability example]: https://tractables.github.io/tididi/tididi/guide/examples/probability/index.html
[variable-grouping example]: https://tractables.github.io/tididi/tididi/guide/examples/vtrees/index.html
[persistence example]: https://tractables.github.io/tididi/tididi/guide/examples/persistence/index.html
[execution example]: https://tractables.github.io/tididi/tididi/guide/examples/execution/index.html
[custom-statistic example]: https://tractables.github.io/tididi/tididi/guide/examples/statistics/index.html
[API overview]: https://tractables.github.io/tididi/tididi/guide/api/index.html
[TDD data model]: https://tractables.github.io/tididi/tididi/guide/model/index.html
[API reference]: https://tractables.github.io/tididi/tididi/
[architecture reference]: https://tractables.github.io/tididi/tididi/guide/architecture/index.html

[minimum configuration cost]: https://tractables.github.io/tididi/tididi/guide/examples/optimization/index.html
