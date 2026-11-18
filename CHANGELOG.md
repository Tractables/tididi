# Changelog

## Unreleased

Breaking changes:

- `Engine::and_filter_products` is removed. Conjoin, then restrict to the
  care constraint with `Engine::restrict_to_care`: the result agrees with
  the filtered product under that constraint and is canonical.
- `Engine::from_models_direct` is removed. `Engine::from_models` builds the
  same diagram.
- `Tdd::restrict_to_care` and `Engine::restrict_to_care` borrow `care` and
  no longer reduce it first. The walk's cost follows `care` as written, so
  reduce an unreduced care before the call.
- `Engine::restrict_to_care_bounded` is removed. `Engine::restrict_to_care`
  runs the same walk; the engine's limits bound its work.
- `Quantification::FusedSubtrees` is removed. `and_exists_with` takes
  `Quantification::Fused` (the default) or `Quantification::Product`.
- `Tdd::embed_mirrored`, `EmbeddingPlan::new_mirrored`,
  `Engine::embed_moving_mirrored` and `Engine::and_onto_mirrored` are
  removed. `Engine::embed_mirrored` and `Engine::embedding_plan_mirrored`
  remain as hidden aliases of `Engine::embed` and `Engine::embedding_plan`
  and will be removed in a later release. `Tdd::embed`, `Engine::embed`,
  `Engine::embedding_plan`, `EmbeddingPlan::new`, `Engine::embed_moving`
  and `Engine::and_onto` match the vtrees up to mirrors and give the
  diagrams the removed forms gave; a destination with some node's children
  in the other order is no longer refused. `Engine::embed_over` does so for
  a structural diagram, and a diagram with a marginal level still needs
  every node's children in their order.

Fixed:

- `Tdd::marginalize_levels` and `Engine::marginalize_levels` on the root
  level of a diagram with several root nodes, as `TddBuilder` can build,
  could report another node's count when two root nodes had equal counts;
  the output now follows the merge.
- `exists_var`, `exists_vars` and `exists_vars_with` (on `Tdd` and
  `Engine`) could answer wrong on a diagram with a marginal level away
  from the quantified leaves; they now return
  `OperationError::MarginalLevel` whenever any level is marginal.
- `TddBuilder::finish` and `read_tdd` accepted a level holding a pair twice,
  within one node or across two, and `negate` and the model count were then
  wrong on the diagram. They now return `TddBuildError::RepeatedPair`,
  unless a level under it is marginal, where a repeated pair is a legal
  multiplicity. The binary reader trusts the levels it reads and does not
  check.

## 0.2.0

Breaking changes:

- `and`, `Engine::and`, `Engine::and_restoring`, `Engine::and_onto`,
  `Engine::and_onto_mirrored`, `Engine::and_marginalizing`, `Engine::negate`
  and `Engine::negate_with` take each operand as `impl Into<Cow<'_, Tdd>>`.
  A `Tdd` is consumed as before; a `&Tdd` is read in place and left
  unchanged, so a caller that keeps a diagram no longer clones it first.
  `AndRefused` and `AndOntoRefused` gain a lifetime and hand each operand
  back as it was given, owned or borrowed.
- `PoolSearchConfig`, `PoolRestarts`, `PoolSearchStats`, `StopRules` and
  `SparseRoute` are `#[non_exhaustive]`. Build them from `Default` and their
  `with_*` setters, or `SparseRoute::new` for a constant.
  `StopRules::with_after_pairs` takes an `Option` and replaces
  `StopRules::after_pairs`.

Added:

- `Tdd::forall_vars` and `Engine::forall_vars` quantify variables
  universally; Python has `Circuit.forall` and C has `tididi_forall`.
- `Engine::project_over` removes from a diagram's vtree the variables its
  marginal levels have summed out, keeping its weighted value.
- `PoolSearchConfig::with_restarts` restarts a pool search from random moves
  at its local minimum (`PoolRestarts`, off by default); `PoolSearchStats`
  counts the restarts and those kept.
- `Arithmetic::ScaledFloat` holds weighted values as an `f64` mantissa with
  an exponent of its own: one `f64` operation per product or sum, without
  the signed log domain's `exp` and `ln_1p`.

Fixed:

- An allocation refused while moving an implicit level's references
  panicked; it now returns `OperationError::OverBudget`.
- A streamed sum-out level with 2^30 columns or more wrote references its
  parent misread; it now returns `OperationError::IndexOverflow`.
- On 32-bit targets such as wasm32, two size computations wrapped and an
  operation could index past a table; they now decide as on a 64-bit target.

Performance: conjunction, negation, minimization and summing out are faster,
and more of their allocations are charged to the operation's `Limits`. On
x86-64, summing out under `Arithmetic::ScaledFloat` uses AVX2 where the
processor has it.

## 0.1.1

Documentation and packaging: Python wheels on PyPI, prebuilt C packages on
the GitHub release, and shorter READMEs. The library code is unchanged from
0.1.0.

## 0.1.0

First release.
