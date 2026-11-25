# Changelog

## Unreleased

Breaking changes:

- The placements onto a caller's vtree are removed: `Tdd::embed`,
  `Engine::embed`, `Engine::embed_mirrored`, `Engine::embed_moving` with
  `restructure::EmbedRefused`, `Engine::embed_with`,
  `Engine::embedding_plan`, `Engine::embedding_plan_mirrored`,
  `Engine::compose_embeddings`,
  `restructure::EmbeddingPlan`, `Engine::embed_over`, `Engine::and_onto`
  with `AndOntoRefused`, and `Tdd::graft_over` with
  `restructure::GraftError`. Diagrams live on projections of a home:
  `Engine::transport` moves one onto a home under a renaming, and `and`
  joins operands on different projections of one home. `EmbedError` loses
  the variants `SourceWeights`, `DestinationWeights` and `Unsummed`.
- `Engine::project_over`, `Engine::project_to_vars` and
  `Tdd::project_to_vars` are removed. On a home, `Tdd::exists_vars`
  followed by `Tdd::minimize` leaves a diagram on the projection of its
  support. `Vtree::project_to_vars` stays.
- `Tdd::marginalize_levels`, `Engine::marginalize_levels`,
  `Engine::and_marginalizing` and `Vtree::with_weights` are removed: sums
  are named by variables with `marginalize` and `and_marginalizing_where`,
  and weights are attached with `Home::weighted`.
  `Engine::and_model_count` takes a variable predicate in place of level
  targets.
- `Tdd::splice_subtree_unchecked` and `Tdd::reseat_vtree_unchecked` are
  removed.
- No operation minimizes its result by default. Each one ends with its own
  kernel and the close it needs (levels in closed form, a constant on the
  empty projection, the leaves it named true dropped); its documentation
  says whether it sets, keeps or gives up the certificate
  (`Tdd::is_canonical`); `forall_vars` returns a certified result,
  because it minimizes the projection it complements. A caller that wants
  a canonical result asks for it: in Rust on the result, with
  `Engine::minimize`, or `Engine::reduce` with a plan for part of the
  work; in Python with the keyword `canonical=True` on every operation
  that returns a new circuit, readers included (`Circuit.minimize`
  stays); in C with a new last field `canonical` of `TididiLimits`, which
  every call that returns a circuit reads and every other call rejects
  when set (`tididi_minimize` stays). The struct's layout differs from
  0.2.0; initialize it with `tididi_limits_default`. Either binding form
  runs the operation, then the minimization, under the same limits. No setting
  minimizes every result, and no query minimizes its intermediates.
  `Tdd::equivalent`, `Tdd::implies` and
  `Tdd::is_zero` are exact either way; `Tdd::is_one` is exact on a
  minimized diagram.
- Python `Circuit.update` and C `tididi_update` no longer minimize their
  result; pass `canonical=True` (C: `TididiLimits.canonical`) for the
  previous behavior.
- `Engine::negate_with`, `Tdd::negate_with`,
  `Engine::and_exists_with_reduction` and `Engine::filter_nodes_with` are
  removed. Call the operation, then `Engine::reduce` with the plan.
  `Engine::exists_vars_with` stays: it runs its plan before the close, which
  costs less than `exists_vars` followed by `Engine::reduce`.

- `Engine::and_filter_products` is removed. Conjoin, then restrict to the
  care constraint with `Engine::restrict_to_care`: the result agrees with
  the filtered product under that constraint, and `Engine::minimize` makes
  it canonical.
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
- `RotationProbe::diagram` is removed. On a projection the probed diagram's
  vtree still carried the home link of the shape before the rotations, so a
  clone of it conjoined with a diagram on the same leaves gave a wrong
  function. `RotationProbe::before`, `after` and `live_pairs_delta` read
  the rebuilt levels.
- `exists_var`, `exists_vars` and the operations built on them refuse a
  diagram with any marginal level with `OperationError::MarginalLevel`.
  Before, a marginal level off the quantified variables' paths was
  accepted, and the count could be wrong: after a sum-out, merged nodes no
  longer tell which events their values came from.
- On a diagram on a projection, `Engine::at_least_distinct` and
  `Engine::project_to_subtree` take `key` as a node of the home, count over
  every home variable under its sibling, and return a diagram on the
  projection of the same home on the diagram's leaves under `key`. Both
  take `key` as `impl Into<DistinctKey>`: a `VtreeIdx` as before, or the
  variables under the node, which name the same node on a home and on a
  projection of it; variables that are not exactly one node's leaves give
  `DistinctError::Key`. `Engine::tag_level` into a vtree on a home returns a diagram on the
  projection of that home on the tagged and code variables, the code
  subtree on either side of its root. Diagrams on root vtrees are
  unchanged.
- On a home or a projection of one, a constant is on the home's empty
  projection: `Tdd::one`, `Tdd::zero`, `Engine::one`, `Engine::zero` and
  every operation that returns true or false. `literal`, `Tdd::cube` and
  `Tdd::clause` return a diagram on the projection of their variables,
  which need only be in the home. `exists_var`, `exists_vars`,
  `forall_vars`, `condition` and `and_exists` return a diagram on the
  projection of the variables the result still depends on, and
  `from_models`, `from_columns` and `from_cubes` leave out a variable every
  model leaves free. `rename_vars` accepts a target variable of the home
  that the diagram's vtree does not hold.
- On a vtree on a home, `TddBuilder::finish`, `read_tdd`, `load_tdd`,
  `read_tdd_binary` and `load_tdd_binary` return a diagram on the
  projection of the home on the leaves its pairs use, and false on the
  empty projection; a diagram with a marginal level keeps the builder's
  vtree. The readers take the supplied vtree as naming the home only and
  accept a file over any projection of it; another tree, or a vtree with
  a summed-out region, is `IoError::Format`. On a home the binary reader
  refuses a pair repeated in a level, as the text reader does everywhere.
  `finish` reports a refused allocation in that trim as the new
  `TddBuildError::OverBudget`, and the readers as the new
  `IoError::OverBudget`. `TddBuilder::finish_unchecked` does not
  trim. Diagrams on root vtrees are unchanged.
- On a home, `Maintenance` updates accept a variable of the home the
  diagram has no leaf for, and a rebuild or the removal of the last model
  can move the diagram to another projection of the home.
  `Tdd::reseat_vtree_unchecked` and `Tdd::splice_subtree_unchecked` state
  their contracts on a home.

Added:

- `Vtree::renamed`: a root vtree of the same shape with each leaf's
  variable renamed.
- `Vtree::is_on` is public.

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
