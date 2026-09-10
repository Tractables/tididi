# API guide

One section per capability, in the order of the README capability list. Item
documentation is on [docs.rs](https://docs.rs/tididi); the data model is in
[tdd.md](tdd.md).

Every diagram is tied to a vtree, shared as an `Arc<Vtree>`. The operands of
a binary operation must share the same `Arc`.

The snippets below are fragments: each one continues from the `vtree` and `f`
bindings of the sections before it, so none compiles on its own. The compiled
examples are `tests/readme_example.rs`, `examples/statistic.rs`, and
`examples/build_minimize_count.rs`.

The documented API is the modules below. `compiler_seam` and `check` are
hidden: `compiler_seam` holds the hooks a clause-by-clause driver compiles
against, and `check` the invariant checkers; neither is covered by any
compatibility promise.

## Diagrams and vtrees

`Tdd` is the diagram: one `TddLevel` per vtree node, an `output` node, and
the `Arc<Vtree>`. `Vtree` is a binary tree whose leaves are variables.
Variables are `VarId(0..n)`, 0-based; `Literal` pairs a `VarId` with a
polarity (`Literal::pos`, `Literal::neg`, or `Literal::from(i32)` with the
1-based DIMACS sign convention).

```rust
use std::sync::Arc;
use tididi::vtree::{VarId, Vtree};

let vtree = Arc::new(Vtree::balanced(4)); // x1..x4, balanced shape
```

Constructors: `Vtree::leaf(var)`; `Vtree::join(&l, &r)` for a new root over
two vtrees with disjoint variables; `Vtree::balanced(n)` and
`Vtree::balanced_over(&order)`; `Vtree::linear(n)` and
`Vtree::linear_over(&order)` for a right-linear chain, which is an OBDD
variable order; `Vtree::random(n, seed)`; `Vtree::graft(&subtrees,
&spine_vars)` (see [Graft](#graft)); and `project_to_vars` for the vtree
induced on a subset of the variables. `Vtree::from_text` parses the
`.vtree` text format (`vtree N`, then `L <id> <var>` and `I <id> <left>
<right>` lines, last node the root) and `to_text` writes it. A vtree may
skip variable ids: `num_vars()` is the id space and `num_leaves()` the
variables carried. `validate()` checks the invariants of a hand-built tree.
Construction and parsing errors are `VtreeError` (`Text`,
`OverlappingVariable`, `Invalid`).

Read a tree with `root()`, `node()`, `children()`, `sibling()`, `leaf_of()` (`None` for a variable no leaf carries),
`leaf_var()`, `lca()`, and the traversal orders `bottomup()`,
`leaf_bottomup()`, `internal_bottomup()`. `same_tree()` compares shape and
variables; node numbering is not identity.

## Base diagrams

```rust
use tididi::Tdd;

let top = Tdd::one(&vtree);          // ⊤
let bot = Tdd::zero(&vtree);         // ⊥: the ZERO sentinel, no nodes
let c = Tdd::clause(&vtree, [1, -2]);    // x1 ∨ ¬x2
```

`Tdd::clause` accepts anything convertible to `Literal`, so a `&[i32]` of
DIMACS literals and a `&[Literal]` both work. It builds the canonical diagram
of the clause directly.

## Boolean combination

```rust
use tididi::negate;

let conj = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let disj = Tdd::clause(&vtree, [1]) | Tdd::clause(&vtree, [2]);
let neg = !Tdd::clause(&vtree, [1, 2]);
```

`&` and `|` are conjunction and disjunction; `!` forwards to `negate`. All three
consume their operands and recycle their storage into the result; clone an
operand first to keep it. Apply results are canonical. Negation is exact but
must first fill every level with the pairs it lacks, which can grow the
diagram; when only the count of `¬f` is needed, use `2ⁿ − count(f)`.
`engine.and(f, g)` and `engine.or(f, g)` are the same operations run on a
caller's engine: they return `ApplyError` instead of aborting under a limit
([Engine and limits](#engine-and-limits)), and reuse the engine's buffers
across calls. `engine.and_marginalizing(f, g, &targets)` takes a flag per vtree
level, marking the levels to emit as marginal ([Marginalization](#marginalization)).

`apply_and_clause(&mut acc, &lits)` conjoins one clause into an accumulator
without building the clause as a diagram; `engine.and_clause(acc, &lits)` is
the fallible form. The accumulator is count-correct after every clause and
canonical after `minimize`.

```rust
use tididi::apply::apply_and_clause;

let mut acc = Tdd::one(&vtree);
for clause in [[1, -2], [2, 3], [-1, 3]] {
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    acc = apply_and_clause(&mut acc, &lits);
}
```

`engine.and_batch(acc, batch, &spine)` conjoins a small diagram into a large
accumulator visiting only the vtree levels the batch can change, and returns
`BatchMergeOutcome::Merged` or `BatchMergeOutcome::Declined` with both operands intact when
the restricted walk is not provably exact; a decline means "run `engine.and`".
The `MergeScope` argument carries the levels the batch may touch together with the
accumulator measurements the walk needs; its rustdoc states the contract each
field must satisfy.

## Conditioning

`condition_var(&f, x, value)` returns the cofactor `f|x=value` with `x` removed
from the diagram; `condition_vars(&f, &vars, value)` conditions many variables
with one final `minimize`. Conditioning only shrinks the diagram and is sound
when other levels are marginal. The kept side of `x` becomes free, so the
model count of the result still carries a factor of two per conditioned
variable; divide by `2^k` for the count of the cofactor itself.

## Quantification

`project_var(&f, x, how)` returns `∃x. f`; `project_vars(&f, &vars, how)`
forgets a set. The result keeps the vtree, so a forgotten variable still ranges
over both values in `model_count`.

`how` picks the rewrite. `Projection::Automatic` computes `f|x=⊤ ∨ f|x=⊥`
where that is sound and switches to an in-place leaf-to-root rewrite where it is
not — the cofactor form disjoins by negation, which a marginal level cannot
survive. `Projection::Structural` asks for the in-place rewrite outright: it is
slower, and it never clones the diagram to negate it, which is what a caller
wants when the diagram is large enough for that clone to be the risk.

```rust
use tididi::apply::{project_var, Projection};

let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
let g = project_var(&f, VarId(1), Projection::Automatic);     // ∃x2: x1, with x2 free, twice the models
```

## Restrict-to-care

`restrict(&f, care, CareCanonical::{Yes, No})` prunes `f` to the pairs and
nodes that produce a model under `care`, returning `g` with `g ∧ care == f ∧
care` and `g` no larger than `f`. Pass `CareCanonical::Yes` when `care` is
already minimized to skip its reduction. The result is `Restricted::Unchanged`
when nothing died, `Restricted::Shrunk(g)` with a non-canonical `g`, or
`Restricted::Unsatisfiable(⊥)`; `into_tdd(&f)` collapses the three to a diagram.

```rust
use tididi::apply::{restrict, CareCanonical};

let f = Tdd::clause(&vtree, [1, 2]);
let care = Tdd::clause(&vtree, [1]);
let g = restrict(&f, care, CareCanonical::No).into_tdd(&f);
```

## Graft

`Tdd::graft(parts, &spine_vars)` is the conjunction of diagrams over
pairwise-disjoint variable sets, each on its own vtree, built structurally on
`Vtree::graft` of their vtrees: the parts' levels move into place and one
width-1 level per join ties them together, so no apply runs and the result is
canonical when the parts are. `spine_vars` are variables no part mentions;
the result is unconstrained in them. The error is `VtreeError`.

```rust
let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
let fg = Tdd::graft(vec![Tdd::clause(&a, [1, 2]), Tdd::clause(&b, [3, -4])], &[VarId(4)]).unwrap();
assert_eq!(fg.model_count(), 18u32.into()); // 3 · 3 · 2
```

## Marginalization

`marginalize(engine, &mut f, &levels)` sums the named vtree levels out of the
diagram: each becomes a marginal level holding one value per node instead of
pairs, and the storage below it is released. `levels` is sorted bottom-up, and a
level may be summed out only once its children are marginal or are leaves. The
value of a node is the number of assignments to the level's subtree that reach
it; counting folds `Σ count(left) × count(right)` over pairs and stops at a
marginal level, and a free variable contributes a factor of two. A marginal
level is permanent, and no conjunction may touch it, so sum a level out only
once every clause over its variables is in. On return the diagram's marginal
invariants hold again: no value slot is orphaned or duplicated, and no parent
node carries two pairs the fold would double-count. The error is
`ApplyError::Deadline`; the levels summed out before the cut keep their
values.

`marginalize_schedule(&clauses, &vtree, &clauses_at, &keep_explicit,
&defer_nodes)` computes, for a clause-by-clause build, the levels that may be
summed out after each step; `intra_batch_completions` refines one step's group to
the clauses of a batch. `Tdd::has_marginal_level` reports whether any level
is marginal.

With a `WeightStore` attached, the same operation stores each node's
semiring value in the store instead of a count:

```rust
use tididi::query::RationalWeights;
use tididi::diagram::{Arithmetic, WeightStore};
use tididi::marginal::{marginalize, weighted_value};

let sr = RationalWeights::from_weights(&weights); // (w_neg, w_pos) per variable
f.set_weights(WeightStore::new(sr, Arithmetic::ExactRational));
marginalize(&engine, &mut f, &levels).unwrap();
let total = weighted_value(&f);                    // Option<WeightVal>
```

`Arithmetic::ExactRational` folds in `BigRational`; `Arithmetic::SignedLog` folds in the
bounded-precision `SignedLog` domain. Attach the store before the first
marginalize; a conjunction moves it to its result. `Tdd::weights` reads the store,
`Tdd::take_weights` detaches it, and `WeightStore::level(t)` reads a marginal
level's values. `weighted_value` folds whatever is still explicit above the
marginal levels and returns the diagram's value.

## Model counting

```rust
use tididi::query::model_count;

let n = f.model_count();   // BigUint; sugar for model_count(&f)
```

The count is over all variables of the vtree: a variable the function does
not mention contributes a factor of two. `engine.model_count(&f)` is the same
count under the engine's limits, returning `Err(ApplyError)` where an armed
stop cuts the pass.

`IncrementalCounter` counts under a partial assignment and updates the count when
pins change without a full pass. Two type parameters say what a given counter
can do. The first is the column-lifetime policy: `KeepAllColumns` keeps a column per
level, `KeepFrontier` frees each column as its parent completes and offers the
output count alone. The second is whether a pass has run: `IncrementalCounter::new(eng,
&f, n_pins, convention)` allocates the columns, `set_pin(var, Some(value))` pins
a variable, and `compute(eng, &f)` consumes the counter and returns one in the
`Evaluated` state, where `output_count(&f)` reads the count. `SeedConvention::Fixed`
counts a pinned variable once; `SeedConvention::Free` leaves the factor of two.

Under `KeepAllColumns` a computed counter also has `recompute_dirty(eng, &f,
&levels)`, which recomputes only the levels between the changed leaves and the
root. `levels` is a `BottomUpSubset`, minted by `vtree.bottom_up_subset(...)`
from levels named in any order, so a level can never be recomputed before its
children.

## Weighted and semiring evaluation

`evaluate(&f, &sr)` folds any `EvalAlgebra` bottom-up over an explicit diagram:
implement `zero`, `leaf(var, label)`, `add_assign`, and `mul`.
`RationalWeights::from_weights(&[(w_neg, w_pos)])` is exact weighted model
counting in `BigRational`; `RationalWeights::unit(n)` reproduces the model
count. A weighted value of zero is a cancellation, not unsatisfiability.

```rust
use tididi::query::{evaluate, RationalWeights};

let sr = RationalWeights::from_weights(&weights);
let wmc = evaluate(&f, &sr);
```

`SignedLog` is a signed log-domain value with `mul`, `add_assign`, and
`from_rational`. `WeightVal` is the per-node value a `WeightStore` holds; it
is `#[non_exhaustive]`, so build values with `WeightVal::exact` and read
them with `as_rational`, `into_rational`, `into_rational_opt`, or `as_log`
for the log-domain form; its variants are not constructible from outside the
crate, so the representation stays free to change.

## Reduction

```rust
use tididi::engine::Engine;
use tididi::reduce::{minimize, try_minimize, MinimizeOptions, MinimizeScope};

let engine = Engine::new();
minimize(&mut t);
try_minimize(&engine, &mut t, MinimizeOptions { passes: MinimizeScope::PruneOnly, ..Default::default() })?;
```

`minimize` prunes unreachable nodes and contracts twins until the diagram is
the canonical form for its vtree ([tdd.md](tdd.md)). Apply, `Tdd::clause`,
and `Tdd::graft` return canonical diagrams; `apply_and_clause` accumulators,
`restrict` results, and hand-built diagrams need it. `try_minimize` returns
`ApplyError` instead of exiting on an allocation refusal or a deadline;
`MinimizeOptions` selects `MinimizeScope::{Full, PruneOnly, ContractOnly}`,
skips the content-twin scan, or carries a `ContentTwinProbe` across calls.
On `Err` the diagram is exactly as it was at the last pass boundary.
`minimize` itself panics on a refusal, so a caller that must survive one uses
`try_minimize`.

## Restructuring

Rotating a bare vtree is not a public operation here: a rotation is only
meaningful against the diagram built over the vtree, and the levels have to be
relinked with it. On a compiled diagram,
`search_to_local_min(&mut t)` rotates the vtree under the diagram to a local
minimum of its size, and `rotation_search(&mut t, &mut objective, &config)`
does the same for any `RotationObjective` (`delta(before, after) -> i64`,
negative to accept; `search_to_local_min` is that call with the size objective).
`RotationSearchConfig` bounds the rebuilt level size and the sweep count;
both entries return `RotationSearchStats { probes, accepts, sweeps }`. Each
rotation rewrites only the two affected levels and re-minimizes them, and
the model count is preserved.

`engine.rotation_search(&mut t, &mut objective, &config)` is the same search on
a caller's engine: it polls the armed stop once per pivot and returns
`Err(ApplyError::Deadline)` rather than running to the local minimum, leaving
the diagram canonical and count-correct wherever it stopped.

```rust
use tididi::restructure::search::{rotation_search, RotationObjective, RotationSearchConfig};
use tididi::diagram::TddLevel;

struct MinPeak;
impl RotationObjective for MinPeak {
    fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
        a.0.width().max(a.1.width()) as i64 - b.0.width().max(b.1.width()) as i64
    }
}
let stats = rotation_search(&mut t, &mut MinPeak, &RotationSearchConfig::default());
```

## Engine and limits

An `Engine` is the session an operation runs in: it owns the limits the
operation runs under and every buffer the operation reuses between calls.
Nothing is per-thread, and nothing is armed over an operation the caller did
not arm it over:

```rust
use std::time::{Duration, Instant};
use tididi::engine::{Engine, LimitSet, MemPressure, Scheduled, Stop, StopAt};
use tididi::ApplyError;

let engine = Engine::new();
engine.limits().install(
    LimitSet::none()
        .deadline(Some(Instant::now() + Duration::from_secs(30)))
        .budget(Some(4 << 30))        // bytes one operation may grow its storage by
        .output_cap(Some(50_000_000)) // output nodes one conjunction may build
        .schedule(Some(|_meters, _now| Scheduled::Carry))
        .mem_pressure(MemPressure::NONE)
        .watch(true),
);
match engine.and(f, g) {
    Ok(h) => { /* ... */ }
    Err(ApplyError::OverBudget | ApplyError::Deadline | ApplyError::OutputCap) => { /* cut short */ }
}
```

`LimitSet` is a plain `Copy` value and installing one replaces every axis.
`engine.limits().install(set)` returns what was armed before, so a caller that
wants one axis changed for a scope reads the armed set, edits the field, and
installs what it found again when the scope ends. `LimitSet::uncut()` clears
the whole stop axis, which `deadline(None)` does not: that clears the
unconditional wall and leaves a size-conditional bound in force.

The axes: `budget`, a soft byte budget for one operation's storage;
`output_cap`, a cap on the nodes one conjunction may build; `stop`, when the
operation gives up; `schedule`, a callback the in-operation polls ask;
`mem_pressure`, the host's memory probes (`MemPressure` holds four function
pointers: `mapped_bytes`, `address_space_limit`, `preflight_alloc`,
`eager_reclaim`; `MemPressure::NONE` is the default); and `watch`, which makes
conjunctions publish where they stand.

A `Stop` carries two bounds in one axis. `wall` is unconditional — past it the
operation stops whatever it has built. `after` is `(pairs, at)`: it applies once
the operation has built that many output pairs, so a caller can cut a step for
spending too long on a big diagram and leave a small one alone. Each bound falls
either at an instant (`StopAt::Wall`) or at a reading of the engine's own work
clock (`StopAt::Work`), which is reproducible across machines where a wall is
not. `LimitSet::deadline(Some(t))` is the common case, and `Stop::by(t)` spells
the same thing.

The schedule callback is asked on every poll, handed the meters and the clock
reading the poll had already taken. It answers `Scheduled::Carry`,
`Scheduled::Stop`, or `Scheduled::Replace(stop)` — a commitment that replaces
the stop the operation was running under. The library holds no view on when a
decision is due: a caller with decision points of its own tests them and carries
until one arrives.

`engine.reset()` releases every buffer the engine retains, keeping the armed
limits. Call it between a failed operation and whatever recovers from it, so
the recovery starts on a clean allocator slate instead of inheriting the peak
the failure parked. It is sound only between operations.

`ApplyError` has three variants: `OverBudget` (an allocation refused or the
budget exceeded), `Deadline` (a stop fell, or a schedule said so), and
`OutputCap`. It implements `Display` and `std::error::Error`, so it propagates
with `?` into `Box<dyn Error>`. An `Err` from an owned entry point spends both
operands. A caller may also return one for a resource failure of its own.

`engine.limits().meters()` snapshots the meters (`ApplyMeters`:
`in_flight_bytes`, `pairs_in_flight`, `work_units`, `refused_reserve_bytes`, and
`merge` as a `MergeProgress`); `engine.limits().armed()` reads back what is
armed; `reset_meters()` zeroes the per-operation meters
at the start of an independent compile. The infallible entries — `apply_and_clause`,
`minimize`, `Tdd::model_count`, `project_var`, `restrict`, `condition_var`,
`Tdd::clause`, `Tdd::one`, `Tdd::zero`, `rotation_search`, the operators — run
on an engine of their own with nothing armed, so no caller's deadline can cut
one short. Every one of them has an engine-owned form (`engine.and`,
`engine.or`, `engine.and_clause`, `engine.project_var`, `engine.restrict`, `engine.condition_var`,
`engine.clause`, `engine.one`, `engine.zero`, `engine.rotation_search`, …) that
computes the same thing under the caller's limits and keeps the buffers warm
for the next call; the free function is that method on a transient engine. The library reads no
environment variables and holds no process-wide state.

## Introspection

`Tdd::size()` is the total pair count, the size measure of the paper;
`size_at_most(cap)` answers the threshold question without counting past `cap`. `node_count()`, `max_width()`,
`width_at(t)`, `effective_width(t)`, `is_zero()`, `has_marginal_level()`, and
`retired_marginal_slots()` read the diagram's shape and state. `is_sat_minimized(&f)` is a
constant-time check on a minimized diagram; `implied_literals(&f)` returns
the literals true in every model of a minimized diagram;
`reduced_size(&f, ReductionRule::R1Sdd)` reports the size after the non-smooth reduction of
[tdd.md](tdd.md) without applying it.

## Serialization and rendering

`save_tdd(&f, path)` writes the `.tdd` text format (a header, one `L` line
per leaf, one `I` line per stored node with its pairs, bottom-up), and
`load_tdd(path, &vtree)` reads it back. The format records the diagram, not
the vtree, so the reader takes the vtree it belongs to and validates the file
against it. `tdd_to_dot(&f)` and `vtree_to_dot(&vtree, Some(&f))` render
Graphviz DOT; the vtree render colors each internal node by its pair count.
All of these return `io::Result` or `Result<_, IoError>`, where `IoError` is
either an underlying `std::io::Error` or a `Format` message naming what the
file or diagram violated — a marginal level is refused that way.
`Vtree::to_text()` writes the `.vtree` format and `Vtree::from_text()`
reads it; `Display` and `FromStr` are the same two. `vtree_example.svg` and `tdd_example.svg` in this
directory are renders of one diagram.

## Traversing a diagram

The stored encoding is the traversal contract. The `diagram` module
documentation states it: how each kind of level is read, and the invariants a
reader may rely on.

```rust
use tididi::diagram::{ChildRef, ValueRef};

for (t, left, right) in f.vtree.internal_bottomup() {
    let level = f.level(t);
    if level.is_marginal() { continue; }
    let (lv, rv) = (f.level(left).side_view(), f.level(right).side_view());
    for (i, pairs) in level.internal_inputs_iter() {
        for p in pairs {
            let l = lv.child(p.left);   // Node(index), or Value(Slot | Inline)
            let r = rv.child(p.right);
        }
    }
}
```

`examples/statistic.rs` is a custom statistic read straight off the stored
encoding; run it with `cargo run --example statistic`, and
`examples/build_minimize_count.rs` for the shortest path from clauses to a
count. `Tdd::try_from_levels(vtree, levels, output)` assembles a diagram from
levels you filled; `TddBuildError` names what it checks.
