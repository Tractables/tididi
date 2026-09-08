# tididi

[![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A Rust library for Tree Decision Diagrams (TDDs). A TDD represents a Boolean
function as a decision diagram shaped by a vtree, a binary tree over the
variables; an OBDD is the special case where the vtree is a chain. Diagrams
combine by conjunction, disjunction, and negation, and transform by
conditioning, quantification, restriction, and grafting. `minimize` reduces a
diagram to the canonical form for its vtree, so two diagrams of one function
over one vtree are identical. Model counts, weighted counts, and other
semiring evaluations fold bottom-up over the diagram, and a level whose
structure is no longer needed can be summed out into per-node counts (a
marginal level) to bound memory on large counts. TDDs were introduced in
Capelli, Choi, Mengel, Muñoz and Van den Broeck,
[*A Canonical Generalization of OBDD*](https://arxiv.org/abs/2604.05537).

## Install

```sh
cargo add tididi num-bigint
```

Counts are returned as `num_bigint::BigUint`.

## Example

```rust
use std::sync::Arc;
use num_bigint::BigUint;
use tididi::Tdd;
use tididi::io::save_tdd;
use tididi::reduce::minimize;
use tididi::apply::apply_and_clause;
use tididi::vtree::{VarId, Vtree};

// A vtree over x1..x4 that groups {x1, x2} and {x3, x4}.
let left = Vtree::balanced_over(&[VarId(0), VarId(1)]);
let right = Vtree::balanced_over(&[VarId(2), VarId(3)]);
let vtree = Arc::new(Vtree::join(&left, &right).unwrap());

// (x1 ∨ ¬x2) ∧ (x2 ∨ x3) ∧ (¬x3 ∨ x4), one clause at a time; integers are
// DIMACS literals (1 → x1, -2 → ¬x2).
let mut f = Tdd::one(&vtree);
for clause in [[1, -2], [2, 3], [-3, 4]] {
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    f = apply_and_clause(&mut f, &lits);
}
minimize(&mut f); // canonical form for this vtree
assert_eq!(f.model_count(), BigUint::from(5u32));

// Conjoin with x1 ⊕ x4, built from clauses with the operators.
let g = Tdd::clause(&vtree, [1, 4]) & Tdd::clause(&vtree, [-1, -4]);
let h = f & g; // apply results are already canonical
assert_eq!(h.model_count(), BigUint::from(2u32));

save_tdd(&h, "h.tdd").unwrap();
```

`tests/readme_example.rs` compiles and checks this example.

## Capabilities

Each line links to its section of the [API guide](docs/api-guide.md).

- [Diagrams and vtrees](docs/api-guide.md#diagrams-and-vtrees): `Tdd` over an `Arc<Vtree>`; vtrees from `leaf` and `join`, balanced, linear, random, the `.vtree` text format, `graft`, `project_to_vars`.
- [Base diagrams](docs/api-guide.md#base-diagrams): `Tdd::one`, `Tdd::zero`, `Tdd::clause`, `Tdd::clause`.
- [Boolean combination](docs/api-guide.md#boolean-combination): `apply_and`, `apply_or`, `negate` and the `&`, `|`, `!` operators; `apply_and_clause` for a clause stream; `engine.and_batch` for a small batch into a large accumulator.
- [Conditioning](docs/api-guide.md#conditioning): `condition_var`, `condition_vars`.
- [Quantification](docs/api-guide.md#quantification): `project_var`, `project_vars`.
- [Restrict-to-care](docs/api-guide.md#restrict-to-care): `restrict`.
- [Graft](docs/api-guide.md#graft): `Tdd::graft` over `Vtree::graft`.
- [Marginalization](docs/api-guide.md#marginalization): `marginalize`, `marginalize_schedule`, `WeightStore`.
- [Model counting](docs/api-guide.md#model-counting): `model_count`, `IncrementalPinnedCounter`.
- [Weighted and semiring evaluation](docs/api-guide.md#weighted-and-semiring-evaluation): `evaluate`, `EvalAlgebra`, `RationalWeights`, `SignedLog`.
- [Reduction](docs/api-guide.md#reduction): `minimize`, `try_minimize`, `MinimizeOptions`.
- [Restructuring](docs/api-guide.md#restructuring): `rotation_search`, `search_to_local_min`, `RotationObjective`.
- [Engine and limits](docs/api-guide.md#engine-and-limits): `Engine`, `LimitSet`, `Stop`, `ApplyError`, `MemPressure`.
- [Introspection](docs/api-guide.md#introspection): `size`, `max_width`, `node_count`, `is_sat_minimized`, `implied_literals`, `reduced_size`.
- [Serialization and rendering](docs/api-guide.md#serialization-and-rendering): `save_tdd`, `tdd_to_dot`, `vtree_to_dot`, `to_text`.
- [Traversing a diagram](docs/api-guide.md#traversing-a-diagram): the stored encoding, `Tdd::try_from_levels`.

## Vtrees

The library operates on the vtree it is given and contains no vtree
construction heuristics. The [`vitri`](https://crates.io/crates/vitri) crate
builds vtrees from CNF structure and emits the `.vtree` text format this
library reads.

## Traversal contract

The stored encoding is public: a reader walks `Tdd::levels` and their pairs
directly, and the `diagram` module documentation states what a reader may
rely on. `examples/statistic.rs` is a complete walk, and
`examples/build_minimize_count.rs` the shortest path from clauses to a
count.

## Documentation

API reference: [docs.rs/tididi](https://docs.rs/tididi). Guides:
[`docs/api-guide.md`](docs/api-guide.md), one section per capability, and
[`docs/tdd.md`](docs/tdd.md), the data model.

## License

Apache License, Version 2.0 ([LICENSE](LICENSE)).
