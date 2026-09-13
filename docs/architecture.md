# Architecture

This document is the maintainer's boundary reference: what each module owns,
what it may not touch, and the numbered invariants every checker and comment
cites. A reader who wants to use the library wants
[`docs/api-guide.md`](https://docs.rs/tididi/latest/tididi/guide/api/index.html) instead.

## Model

A diagram ([`Tdd`]) is an `Arc<Vtree>`, one [`TddLevel`] per vtree node, and an
output reference naming a node at the vtree root. A level is a leaf level
(implicit: it stores nothing), a structural level (its nodes are lists of
`(left, right)` pairs), or a marginal level (its structure has been summed out
and it stores one value per node instead). Every operation reads and writes
levels; nothing outside `diagram/` decodes a reference by hand.

## Glossary

| Term | Meaning |
|---|---|
| **diagram** | A [`Tdd`] value. The word used in prose; [`Tdd`] appears only as a type. |
| **level** | One vtree node's storage in a diagram ([`TddLevel`]). |
| **node** | One function at a level, addressed by [`NodeIdx`]. |
| **pair** | One `(left, right)` element of a node's decomposition, holding two [`EncodedChildRef`] words decoded through [`ChildDecoder`]. |
| **slot count** | The number of stored node or value entries at a level, including tombstones; structural leaves store no slots. |
| **marginal level** | A level whose structure was summed out into per-node values. |
| **marginalize** | Replace a level's structure by per-node values. |
| **value** | The per-node payload of a marginal level: a count, or a weight. |
| **value slot** | An index into a marginal level's value store; nodes may share one. |
| **algebra** | The domain a fold computes in: the zero, the leaf values, the sum and the product an [`EvalAlgebra`] supplies. |
| **arithmetic** | Which numeric representation a store's values use ([`Arithmetic`]): exact rationals, or the signed log domain. |
| **cell** | A position in the apply product grid. That use only. |
| **column** | A per-node value array for one level during a bottom-up fold. |
| **fold** | One bottom-up pass computing a value per node. |
| **frontier** | The levels a fold currently holds columns for. |
| **output** | The result diagram, or its root reference ([`Tdd::output`]). |
| **limit** | A bound an operation runs under, installed on the engine. |

## Invariants

The numbered list. Every checker and every comment cites these numbers.

| # | Statement | Established by | Transiently broken by | Decided by |
|---|---|---|---|---|
| 1 | Determinism: distinct nodes at one level compute disjoint functions. | apply's emit | — | `test_helpers::check::check_determinism` |
| 2 | No node computes ⊥; ⊥ is the output sentinel only. | apply's emit; conditioning's falsity sweep | conditioning's leaf rewrite, within one call | `test_helpers::check::check_no_false_nodes` |
| 3 | Canonicity: no two nodes at one level are content-equal. | [`reduce::minimize`] | any apply or marginalization | `test_helpers::check::check_canonicity` |
| 4 | Reachability: every stored node is reachable from the output. | [`reduce::minimize`] | conditioning, restriction | `test_helpers::check_minimize_soundness` |
| 5 | Marginality is permanent and downward-closed: a marginal level never becomes structural, and every descendant of a marginal level is marginal. | [`marginal::marginalize_levels`] | — | [`reduce`]'s demarginalization guard |
| 6 | Every reference into a marginal child decodes through [`ChildDecoder`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | — | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds an inline-eligible value. | the reference tagger, then the slot prune | apply's emit, before tagging | `test_helpers::check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: no eligible same-structural-child group remains (exact arithmetic; weighted leaf sums must fit the pinned column). | [`marginalize_levels`]'s fusion sweep | a later twin merge | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 9 | Twin canonicality: no two nodes at one level have equal pair multisets. | twin contraction | pair fusion | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 10 | Value-slot uniqueness: at a marginal level all stored values are pairwise distinct. | mint-time dedup, then the slot prune | apply's emit | `test_helpers::check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are an immutable, label-ordered cache of `WeightStore::leaf_val`. No pass compacts, erases, reorders or appends to the column, and every reader re-derives it through `diagram::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | — | `test_helpers::check::marginal::check_leaf_columns_pinned` |

Invariants 3, 4, 8, 9 and 10 are post-pass properties, not properties of every
intermediate state; each row says which pass establishes it.

## Modules

Four layers, one hub, and two seams that are not a layer. The tables below run
in the order `lib.rs` declares the modules in, and a module uses only its own
layer or a layer above it. **Uses** names the crate modules a module's own code
reads, which is the layering rule as it can be checked.

**Ground** — what everything else reads.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`vtree`] | The variable tree, its orders, its text format, rotation and graft of the tree itself. | Nothing. | Diagram storage. |
| [`diagram`] | Levels, nodes, pairs, the reference encodings, the level pool, weights. | `vtree`, `limits`. | Any operation's algorithm. |
| [`limits`] | What an operation runs under and what it parks between calls: the budget, the output cap, the stop axis, the memory hooks, the meters, the scratch pools, and [`OperationError`], returned when a limit trips. | `vtree`, `diagram`. | The diagram's contents; any operation's algorithm. |
| `value` | The working form of a value: the count representation and its overflow sentinel, the one bottom-up fold walk, the two domains folded over it, the streaming fold's cache, and the vocabulary a stored column is described by — slot key, minting, interning, the referenced set. Internal to the crate. | `vtree`, `diagram`, `limits`. | Which levels to fold; where a finished column is stored. |

**Operations** — the verbs.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`build`] | Constants and cubes as diagrams. | `vtree`, `diagram`. | Reduction. |
| [`apply`] | Conjunction, disjunction, negation, conditioning, projection, restriction, a clause as a diagram, and the `&`, `\|`, `!` impls. | `vtree`, `diagram`, `limits`, `value`, `build`, `marginal`, `query`, `reduce`. | Reference decoding by hand; reduction policy. |
| [`marginal`] | Marginal-column installation, reference remapping, child reclamation, and summing levels out. | `vtree`, `diagram`, `limits`, `value`, `reduce`, and `test_helpers::check` in a debug build. | The reduction passes' internals. |
| [`reduce`] | Canonical form: pruning, twin contraction, pair fusion, slot pruning. | `vtree`, `diagram`, `limits`, `value`, and `test_helpers::check` in a debug build. | Apply; marginalization. |
| [`restructure`] | Rotation search and graft over a compiled diagram. | `vtree`, `diagram`, `limits`, `marginal`, `reduce`, and `test_helpers::check` in a debug build. | The counting fold. |
| [`query`] | Model counting, satisfiability, algebra evaluation, a weighted diagram's value. | `vtree`, `diagram`, `limits`, `value`. | Mutation of a diagram. |

**Session** — the hub.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`engine`] | The hub: the scratch every operation reuses and the limits armed on it. Every operation is a method on it. | `diagram`, `limits`, `apply`, `reduce`, `restructure`. | The diagram's contents. |

**Edges** — reading a finished diagram.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`io`] | The `.tdd` text format, both directions, and Graphviz rendering. | `vtree`, `diagram`. | Anything but reading a finished diagram. |
| [`guide`] | The prose guides of `docs/`, included as documentation so their examples and their identifiers are checked by the build. | Nothing; it holds no code. | Any behaviour. |

**Seams** — the ways in from outside, which are not a layer.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| `compiler_seam` | Every entry point a driver that builds a diagram clause by clause reaches the crate through: clause-spine marking, mid-compile clustering, the marginalization schedule and its intra-batch refinement, a hand-built marginal level, and the two whole-diagram edits that splice a subtree or reseat a diagram on another tree. The driver-facing module, outside the compatibility promise. | `vtree`, `diagram`, `apply`, `restructure`. | The documented modules' jobs; it holds entry points, not operations. |
| `test_helpers` | The generators every randomized sweep draws from, the oracles a test decides a diagram by (enumeration, canonicity, structural equality, the apply-free evaluator), and in `test_helpers::check` the invariant checkers, one per numbered invariant, compiled only under `cfg(test)` or `debug_assertions`. The test-facing module. | `vtree`, `diagram`, `limits`, `value`, `build`, `apply`, `reduce`, `query`. | Any behaviour the library ships; a test reads a diagram through it, and a checker reports and never repairs. |

No **Uses** cell names [`engine`]: every operation, `diagram`, `value` and
both seams use it, and it uses the scratch of `apply`, `reduce` and
`restructure` in return, the crate's one two-way edge.

`test_helpers::check` is compiled only under `cfg(test)` or
`debug_assertions`, so `assert_canonical` is a no-op elsewhere and the
differential suite in `tests/` is run in both configurations.

## One conjunction

[`Engine::and(f, g)`] plans the operation, walks the vtree bottom-up (children
before parents), and at each level runs the product-grid kernel over the two
operands' nodes, emitting the pairs that survive. The emit establishes
invariants 1 and 2, so the result needs no dedup pass. Marginal sides are
tagged after the walk, which is what invariant 7 is stated against. Reduction
is a separate call.

## Extension points: public

Implementable from outside the crate, against the published API:

- A new read-only value domain: implement [`EvalAlgebra`].
- A new rotation objective: implement [`RotationObjective`].
- A new stopping rule the caller decides: the schedule hook on [`LimitConfig`],
  answered at every poll the running operation reaches.

## Internal seams

For a contributor working inside the crate. None of these is a published
extension point, and none is reachable from outside:

- A new reduction rule: add it beside the rule it resembles — twin
  contraction in `reduce/contract/`, content twins in
  `reduce/content_twins.rs`, pair fusion in `reduce/contract/pair_fusion/`,
  leaf twins in `reduce/contract/contract_leaf.rs` — mark its dirty levels,
  and add a checker for the invariant it claims.
- A new marginalizable value domain: `ValueDomain` for arithmetic and `marginal::transition::MarginalDomain` for storage transitions.
- A new fold: implement `ValueDomain` and use the shared walk.
- A new order for the contraction pass to visit dirty levels in: a walk
  beside the ones in `reduce/contract/strategies.rs`.
- A new limit: a field on [`LimitConfig`] and the poll site that reads it.

## Oracles

| Oracle | Path | When |
|---|---|---|
| Fast invariants | `test_helpers::check::check_all_fast` | After any operation, on any size. |
| Minimize round-trip | `test_helpers::check_minimize_soundness` | On small structural diagrams; it minimizes. |
| Marginal invariants | `test_helpers::check::marginal` | After marginalization or a reduction pass. |
| Brute-force count | `test_helpers::brute_force_count` | Small formulas, to confirm a count. |
| Round trip | [`io`] | To confirm a diagram survives text. |
| Differential fold | [`query`] | Fast and exact counts must agree. |

## Constraints

No cargo features, no `build.rs`, no environment reads, no threads, no
process-wide state, no C or C++ code built.

[`Engine::and(f, g)`]: crate::Engine::and
[`Arithmetic`]: crate::diagram::Arithmetic
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`OperationError`]: crate::OperationError
[`LimitConfig`]: crate::limits::LimitConfig
[`NodeIdx`]: crate::diagram::NodeIdx
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`ChildDecoder`]: crate::diagram::ChildDecoder
[`Tdd`]: crate::Tdd
[`Tdd::output`]: crate::Tdd::output
[`TddLevel`]: crate::diagram::TddLevel
[`WeightStore`]: crate::diagram::WeightStore
[`apply`]: crate::apply
[`build`]: crate::build
[`diagram`]: crate::diagram
[`engine`]: crate::engine
[`guide`]: crate::guide
[`io`]: crate::io
[`limits`]: crate::limits
[`marginal`]: crate::marginal
[`marginal::marginalize_levels`]: crate::marginal::marginalize_levels
[`marginalize_levels`]: crate::marginal::marginalize_levels
[`query`]: crate::query
[`reduce`]: crate::reduce
[`reduce::minimize`]: crate::reduce::minimize
[`restructure`]: crate::restructure
[`vtree`]: crate::vtree

[`EncodedChildRef`]: crate::diagram::EncodedChildRef
