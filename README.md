<!-- scenario: docs/scenarios.md#first-circuit -->

<p align="center">
  <img src="docs/logo.svg" alt="tididi" width="400">
</p>

<p align="center">
  <a href="https://crates.io/crates/tididi"><img
     src="https://img.shields.io/crates/v/tididi.svg" alt="crates.io"></a>
  <a href="https://github.com/Tractables/tididi/actions/workflows/rust.yml"><img
     src="https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg" alt="CI"></a>
  <a href="https://tractables.github.io/tididi/tididi/"><img
     src="https://img.shields.io/badge/docs-main-blue" alt="Documentation"></a>
  <a href="https://tractables.github.io/tididi/python/"><img
     src="https://img.shields.io/badge/guide-Python-blue" alt="Python guide"></a>
  <a href="https://tractables.github.io/tididi/c/"><img
     src="https://img.shields.io/badge/guide-C-blue" alt="C guide"></a>
  <a href="https://arxiv.org/abs/2604.05537"><img
     src="https://img.shields.io/badge/arXiv-2604.05537-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img
     src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

tididi is a Rust library for **Tree Decision Diagrams (TDDs)**. A TDD represents
a Boolean function as a circuit, so you can work with its satisfying assignments
without listing them. Build a circuit once, then combine it with other circuits
or query it as your inputs change.

Like binary decision diagrams, minimized TDDs are canonical and combine
efficiently under Boolean operations; in practice they can be much smaller.

From Python, start with the [Python guide](https://tractables.github.io/tididi/python/)
([installation](bindings/python/README.rst)); from C or C++, with the
[C guide](https://tractables.github.io/tididi/c/) ([build instructions](bindings/c/README.rst)).

## Working with circuits

**Build and transform Boolean functions.** Combine circuits with Boolean
operations, fix variables to chosen values, or eliminate variables by
quantification. The [reachability example] explores a transition system this
way.

**Find solutions and count possibilities.** Check satisfiability, find a
solution, count solutions exactly, or test equivalence and implication. The
[configuration example] counts the valid configurations of a backup application
and updates the count as a user changes their choices.

**Evaluate weighted models.** Assign weights to literals to compute weighted
sums such as probabilities, and change the weights without rebuilding the
circuit, as in the [probability example]. Custom algebras compute other
quantities, such as a [minimum configuration cost].

## Applications built on tididi

- [tididi-cnf](https://github.com/Tractables/tididi-cnf) compiles CNF formulas
  into TDDs for model counting, weighted model counting and projected model
  counting.
- [tddlog](https://github.com/Tractables/tddlog) performs exact probabilistic
  inference for higher-level languages: ProbLog programs, Bayesian and Markov
  networks, and discrete probabilistic programs.

## Getting started

This example builds `(x ∧ y) ∨ z` and counts its satisfying assignments:

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

Four assignments have `z` true, and one more has `z` false with `x` and `y` true.

A **vtree** is a binary tree that groups the variables; circuits you combine
must share the same `Arc<Vtree>`. Literals are signed variable numbers: `1` is
`x` and `-2` is `¬y`. Boolean operations consume their operands, so clone an
operand first to keep it; queries such as `model_count` borrow the circuit.

The [worked examples] continue with tables, saving and loading, variable
grouping and resource limits. The [API overview] finds the operation for a task,
the [API reference] gives its contract, and the [data model] explains the
representation. Contributors can start with the [architecture reference].

## Citing

Please cite [*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537),
which introduced TDDs.

## Acknowledgements

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Agreement No. HR0011269E224.

[configuration example]: https://tractables.github.io/tididi/tididi/guide/examples/configurations/index.html
[reachability example]: https://tractables.github.io/tididi/tididi/guide/examples/reachability/index.html
[probability example]: https://tractables.github.io/tididi/tididi/guide/examples/probability/index.html
[minimum configuration cost]: https://tractables.github.io/tididi/tididi/guide/examples/optimization/index.html
[worked examples]: https://tractables.github.io/tididi/tididi/guide/examples/index.html
[API overview]: https://tractables.github.io/tididi/tididi/guide/api/index.html
[API reference]: https://tractables.github.io/tididi/tididi/
[data model]: https://tractables.github.io/tididi/tididi/guide/model/index.html
[architecture reference]: https://tractables.github.io/tididi/tididi/guide/architecture/index.html
