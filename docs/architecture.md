# Architecture

This reference describes the storage, passes, and invariants used by the
implementation. For library use, start with the [task guide]; for the meaning
of levels and pairs, read the [data model].

## Storage and operations

A diagram owns its level arenas and shares a vtree. Operations consume or
borrow those diagrams according to their signatures and use an engine for
scratch and limits. The diagram's dirty worklists record which levels need
reduction after an edit; they do not contribute to its Boolean meaning.

Structural levels hold pair lists. Marginal levels hold counts or fixed
weighted values, and their references may carry an inline value instead of
a node index. The `diagram` module owns that encoding; other modules read it
through `ChildDecoder`.

## Glossary

| Term | Meaning |
|---|---|
| **diagram** | A [`Tdd`] value, owning level storage and an output reference. |
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

The numbers below are used by the invariant checkers and source comments.
Structural determinism assumes valid TDD input; storage validation in the
builder and reader does not establish it for an arbitrary circuit.

| # | Statement | Established by | Transiently broken by | Decided by |
|---|---|---|---|---|
| 1 | Structural determinism: distinct nodes at one level compute disjoint functions; each child pair belongs to at most one node. | apply's emit | — | `test_helpers::check::check_determinism` |
| 2 | Every live node in a structural diagram is satisfiable; ⊥ is the output sentinel only. | checked builder; apply's emit; conditioning's falsity sweep | conditioning's leaf rewrite, within one call | `test_helpers::check::check_no_false_nodes` |
| 3 | Content uniqueness: no two stored nodes at one level have equal pair multisets. | [`reduce::minimize`] | any apply or marginalization | `test_helpers::check::check_canonicity` |
| 4 | Reachability: every live stored node is reachable from the output. | [`reduce::minimize`] | conditioning, restriction | `test_helpers::check_minimize_soundness` |
| 5 | Marginality is permanent and downward-closed: a marginal level never becomes structural, and its descendants are marginal or leaves whose contribution is absorbed. | [`marginal::marginalize_levels`] | — | [`reduce`]'s demarginalization guard |
| 6 | Every reference into a marginal child decodes through [`ChildDecoder`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | — | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds an inline-eligible value. | the reference tagger, then the slot prune | apply's emit, before tagging | `test_helpers::check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: no eligible same-structural-child group remains (exact arithmetic; weighted leaf sums must fit the pinned column). | [`marginalize_levels`]'s fusion sweep | a later twin merge | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 9 | Twin canonicality: no two nodes at one level have equal pair multisets. | twin contraction | pair fusion | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 10 | Value-slot uniqueness: at a marginal level all stored values are pairwise distinct. | mint-time dedup, then the slot prune | apply's emit | `test_helpers::check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are an immutable, label-ordered cache of `WeightStore::leaf_val`. No pass compacts, erases, reorders or appends to the column, and every reader re-derives it through `diagram::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | — | `test_helpers::check::marginal::check_leaf_columns_pinned` |

Invariants 3, 4, 8, 9 and 10 are post-pass properties; each row identifies the
pass that establishes it. Canonical structural form additionally requires
that no context twins remain: these are nodes with identical uses by their
parents, rather than identical pair lists. The distinction matters because
twin contraction unions their functions, while content deduplication merges
identical representations.

## Modules

The tables group modules by responsibility. **Uses** summarizes their main
dependencies; operations also use the engine for scratch and limits. These
are ownership boundaries, not a claim that every module dependency is acyclic.

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
| `build` | Constants and cubes as diagrams. | `vtree`, `diagram`, `limits`. | Reduction. |
| [`apply`] | Conjunction, disjunction, negation, conditioning, projection, restriction, a clause as a diagram, and the `&`, `\|`, `!` impls. | `vtree`, `diagram`, `limits`, `value`, `build`, `marginal`, `query`, `reduce`. | Reference decoding by hand; reduction policy. |
| `marginal` | Marginal-column installation, reference remapping, child reclamation, and summing levels out. | `vtree`, `diagram`, `limits`, `value`, `reduce`, and `test_helpers::check` in a debug build. | The reduction passes' internals. |
| [`reduce`] | Canonical form: pruning, twin contraction, pair fusion, slot pruning. | `vtree`, `diagram`, `limits`, `value`, and `test_helpers::check` in a debug build. | Apply; marginalization. |
| [`restructure`] | Rotation search and graft over a compiled diagram. | `vtree`, `diagram`, `limits`, `marginal`, `reduce`, and `test_helpers::check` in a debug build. | The counting fold. |
| [`query`] | Model counting, satisfiability, algebra evaluation, a weighted diagram's value. | `vtree`, `diagram`, `limits`, `value`, `apply`, `reduce`. | Mutation of a borrowed input diagram. |

**Session** — the hub.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`engine`] | The scratch and limits shared by checked operations; method implementations live with the operations. | `diagram`, `limits`, `apply`, `reduce`, `restructure`. | The operations' algorithms. |

**Edges** — reading a finished diagram.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| [`io`] | The `.tdd` text format, both directions, and Graphviz rendering. | `vtree`, `diagram`. | Apply or reduction policy. |
| [`guide`] | The prose guides of `docs/`, included as documentation so examples are doctested and the rendered pages share their source. | Nothing; it holds no code. | Any behaviour. |

**Testing** — generators and structural checks.

| Module | Owns | Uses | May not touch |
|---|---|---|---|
| `test_helpers` | The generators every randomized sweep draws from, the oracles a test decides a diagram by (enumeration, canonicity, structural equality, the apply-free evaluator), and in `test_helpers::check` the invariant checkers, one per numbered invariant, compiled only under `cfg(test)` or `debug_assertions`. The test-facing module. | `vtree`, `diagram`, `limits`, `value`, `build`, `apply`, `reduce`, `query`. | Any behaviour the library ships; a test reads a diagram through it, and a checker reports and never repairs. |

`test_helpers::check` is compiled under `cfg(test)` or `debug_assertions`;
`assert_canonical` is a no-op in other builds. Run the differential suite in
both debug and release configurations to exercise the structural checks and
the optimized algorithms.

## One conjunction

[`and(f, g)`] checks out the shared context, then validates the shared vtree and weight interpretation,
then walks levels bottom-up. At each structural level it combines operand
nodes in a product grid and emits surviving child pairs. Consumed level
arenas return to the engine's pools for reuse.

The result has the correct function and count but may retain unreachable
nodes and context twins. A full reduction first prunes, then contracts inner
and leaf twins. For marginal diagrams it also runs eligible content-twin and
value-slot cleanup. Edits register their effects through `Tdd::invalidate`;
reduction drains the corresponding dirty worklists.

Resource refusal is not an implicit rollback of a whole operation. Consuming
operations return an error without their operands; in-place passes document
which completed edits remain valid. Reserve-before-mutation boundaries must
preserve those contracts.

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
- A new query fold: implement `query::fold::LevelFold` and use its shared traversal.
- A new stored-value domain: implement `ValueDomain` for the value walk.
- A new contraction strategy: extend the dispatch and implementations in
  `reduce/contract/strategies.rs`.
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

## Shared execution state

Each vtree carries an `Arc<Context>` separately from its shape and traversal
tables. Public Boolean functions and diagram methods clone that context
handle, borrow an engine from its idle pool, and call the operation kernels. Kernels pass the
borrowed engine through nested work so one operation retains its limits and
metering. The pool moves a boxed engine and holds no mutex during computation.
Nested or concurrent checkouts use separate engines; only one idle engine is
retained. A completed checkout clears configuration and callback references,
and an unwinding checkout discards its scratch.

Tree clones and projections retain their context. Grafts retain a context
shared by all source trees; otherwise they start fresh. Binary compatibility
still compares vtree allocations. A rotation wrapper retains only the context
handle so it does not force extra copy-on-write clones of the tree. Serialized
trees contain shape alone and receive fresh execution state when loaded.

## Constraints

No cargo features, no `build.rs`, no environment reads, no threads, no
process-wide state, no C or C++ code built.

[`and(f, g)`]: crate::and
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
[`diagram`]: crate::diagram
[`engine`]: crate::engine
[`guide`]: crate::guide
[`io`]: crate::io
[`limits`]: crate::limits
[`marginal::marginalize_levels`]: crate::Tdd::marginalize_levels
[`marginalize_levels`]: crate::Tdd::marginalize_levels
[`query`]: crate::query
[`reduce`]: crate::reduce
[`reduce::minimize`]: crate::Tdd::minimize
[`restructure`]: crate::restructure
[`vtree`]: crate::vtree

[`EncodedChildRef`]: crate::diagram::EncodedChildRef

[task guide]: crate::guide::api
[data model]: crate::guide::model
