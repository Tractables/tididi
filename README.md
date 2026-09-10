# tididi

[![crates.io](https://img.shields.io/crates/v/tididi.svg)](https://crates.io/crates/tididi) [![docs.rs](https://img.shields.io/docsrs/tididi)](https://docs.rs/tididi)

A Rust library for Tree Decision Diagrams (TDDs). A TDD represents a Boolean
function as a decision diagram shaped by a vtree, a binary tree over the
variables; an OBDD is the special case where the vtree is a chain. Diagrams
combine by conjunction, disjunction, and negation, and transform by
conditioning, quantification, restriction, and grafting. [`minimize`] reduces a
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

Counts are returned as [`num_bigint::BigUint`]. The crate builds on Rust 1.88
and later, and uses the 2024 edition.

## Example

```rust
use std::env::temp_dir;
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

let path = temp_dir().join("h.tdd");
save_tdd(&h, path.to_str().unwrap()).unwrap();
```

## Capabilities

Each line links to its section of the [API guide](docs/api-guide.md).

- [Diagrams and vtrees](docs/api-guide.md#diagrams-and-vtrees): [`Tdd`] over an `Arc<Vtree>`; vtrees from [`leaf`] and [`join`], balanced, linear, random, the `.vtree` text format, [`graft`], [`project_to_vars`].
- [Base diagrams](docs/api-guide.md#base-diagrams): [`Tdd::one`], [`Tdd::zero`], [`Tdd::clause`].
- [Boolean combination](docs/api-guide.md#boolean-combination): the `&`, `|`, `!` operators and [`negate`]; [`engine.and`] and [`engine.or`] for the fallible forms; [`apply_and_clause`] for a clause stream; [`engine.and_batch`] for a small batch into a large accumulator.
- [Conditioning](docs/api-guide.md#conditioning): [`condition_var`], [`condition_vars`].
- [Quantification](docs/api-guide.md#quantification): [`project_var`], [`project_vars`].
- [Restrict-to-care](docs/api-guide.md#restrict-to-care): [`restrict`].
- [Graft](docs/api-guide.md#graft): [`Tdd::graft`] over [`Vtree::graft`].
- [Marginalization](docs/api-guide.md#marginalization): [`marginalize`], [`marginalize_schedule`], [`WeightStore`].
- [Model counting](docs/api-guide.md#model-counting): [`model_count`], [`IncrementalCounter`].
- [Weighted and semiring evaluation](docs/api-guide.md#weighted-and-semiring-evaluation): [`evaluate`], [`EvalAlgebra`], [`RationalWeights`], [`SignedLog`].
- [Reduction](docs/api-guide.md#reduction): [`minimize`], [`try_minimize`], [`MinimizeOptions`].
- [Restructuring](docs/api-guide.md#restructuring): [`rotation_search`], [`search_to_local_min`], [`RotationObjective`].
- [Engine and limits](docs/api-guide.md#engine-and-limits): [`Engine`], [`LimitSet`], [`Stop`], [`ApplyError`], [`MemPressure`].
- [Introspection](docs/api-guide.md#introspection): [`size`], [`max_width`], [`node_count`], `is_sat_minimized`, `implied_literals`, [`reduced_size`].
- [Serialization and rendering](docs/api-guide.md#serialization-and-rendering): [`save_tdd`], [`load_tdd`], [`tdd_to_dot`], [`vtree_to_dot`], [`to_text`].
- [Traversing a diagram](docs/api-guide.md#traversing-a-diagram): the stored encoding, [`TddBuilder`].

## Vtrees

The library operates on the vtree it is given and contains no vtree
construction heuristics. The [`vitri`](https://crates.io/crates/vitri) crate
builds vtrees from CNF structure and emits the `.vtree` text format this
library reads.

## Traversal contract

The stored encoding is public: a reader walks [`Tdd::levels`] and their pairs
directly, and the [`diagram`] module documentation states what a reader may
rely on. Its own example is the worked walk: a bottom-up model count over a
diagram with a summed-out subtree, so every kind of pair side is decoded.
`examples/statistic.rs` reads one statistic off the same encoding, and
`examples/build_minimize_count.rs` is the shortest path from clauses to a
count.

## Documentation

API reference: [docs.rs/tididi](https://docs.rs/tididi). Guides:
[`docs/api-guide.md`](docs/api-guide.md), one section per capability;
[`docs/tdd.md`](docs/tdd.md), the data model; and
[`docs/architecture.md`](docs/architecture.md), the module map and the
numbered invariants.

## License

Apache License, Version 2.0 ([LICENSE](./LICENSE)).

[`ApplyError`]: crate::ApplyError
[`Engine`]: crate::Engine
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`IncrementalCounter`]: crate::query::IncrementalCounter
[`LimitSet`]: crate::engine::LimitSet
[`MemPressure`]: crate::engine::MemPressure
[`MinimizeOptions`]: crate::reduce::MinimizeOptions
[`RationalWeights`]: crate::diagram::RationalWeights
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`SignedLog`]: crate::diagram::SignedLog
[`Stop`]: crate::engine::Stop
[`Tdd`]: crate::Tdd
[`Tdd::clause`]: crate::Tdd::clause
[`Tdd::graft`]: crate::Tdd::graft
[`Tdd::levels`]: crate::Tdd::levels
[`Tdd::one`]: crate::Tdd::one
[`TddBuilder`]: crate::diagram::TddBuilder
[`Tdd::zero`]: crate::Tdd::zero
[`Vtree::graft`]: crate::Vtree::graft
[`WeightStore`]: crate::diagram::WeightStore
[`apply_and_clause`]: crate::apply::apply_and_clause
[`condition_var`]: crate::apply::condition_var
[`condition_vars`]: crate::apply::condition_vars
[`diagram`]: crate::diagram
[`engine.and`]: crate::Engine::and
[`engine.and_batch`]: crate::Engine::and_batch
[`engine.or`]: crate::Engine::or
[`evaluate`]: crate::query::evaluate
[`graft`]: crate::Vtree::graft
[`join`]: crate::Vtree::join
[`leaf`]: crate::Vtree::leaf
[`load_tdd`]: crate::io::load_tdd
[`marginalize`]: crate::marginal::marginalize
[`marginalize_schedule`]: crate::marginal::marginalize_schedule
[`max_width`]: crate::Tdd::max_width
[`minimize`]: crate::reduce::minimize
[`model_count`]: crate::query::model_count
[`negate`]: crate::negate
[`node_count`]: crate::Tdd::node_count
[`num_bigint::BigUint`]: num_bigint::BigUint
[`project_to_vars`]: crate::Vtree::project_to_vars
[`project_var`]: crate::apply::project_var
[`project_vars`]: crate::apply::project_vars
[`reduced_size`]: crate::query::reduced_size
[`restrict`]: crate::apply::restrict
[`rotation_search`]: crate::restructure::search::rotation_search
[`save_tdd`]: crate::io::save_tdd
[`search_to_local_min`]: crate::restructure::search::search_to_local_min
[`size`]: crate::Tdd::size
[`tdd_to_dot`]: crate::io::tdd_to_dot
[`to_text`]: crate::Vtree::to_text
[`try_minimize`]: crate::reduce::try_minimize
[`vtree_to_dot`]: crate::io::vtree_to_dot
