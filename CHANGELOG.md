# Changelog

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
