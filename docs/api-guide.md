# API guide

One section per capability, in the order of the README capability list. Item
documentation is on [docs.rs](https://docs.rs/tididi); the data model is in
[tdd.md](tdd.md).

Every diagram is tied to a vtree, shared as an `Arc<Vtree>`. The operands of
a binary operation must share the same `Arc`.

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
`Vtree::linear_from_order(&order)` for a right-linear chain, which is an OBDD
variable order; `Vtree::random(n, seed)`; `Vtree::graft(&subtrees,
&spine_vars)` (see [Graft](#graft)); and `project_to_vars` for the vtree
induced on a subset of the variables. `Vtree::from_vtree_text` parses the
`.vtree` text format (`vtree N`, then `L <id> <var>` and `I <id> <left>
<right>` lines, last node the root) and `to_vtree_text` writes it. A vtree may
skip variable ids: `num_vars()` is the id space and `num_leaves()` the
variables carried. `validate()` checks the invariants of a hand-built tree.
Construction and parsing errors are `VtreeError` (`Text`,
`OverlappingVariable`, `Invalid`).

Read a tree with `root()`, `node()`, `children()`, `sibling()`, `leaf_of()`,
`leaf_var()`, `lca()`, and the traversal orders `bottomup()`,
`leaf_bottomup()`, `internal_bottomup()`. `same_tree()` compares shape and
variables; node numbering is not identity.

## Base diagrams

```rust
use tididi::Tdd;
use tididi::build::{constant_one, constant_zero, clause_to_tdd};

let top = constant_one(&vtree);          // ⊤
let bot = constant_zero(&vtree);         // ⊥: the ZERO sentinel, no nodes
let c = Tdd::clause(&vtree, [1, -2]);    // x1 ∨ ¬x2
```

`Tdd::clause` accepts anything convertible to `Literal`; `clause_to_tdd(&vtree,
&[Literal])` is the underlying function. Both build the canonical diagram of
the clause directly.

## Boolean combination

```rust
use tididi::apply::{apply_and, try_apply_and};
use tididi::apply::disjoin::{apply_or, try_apply_or};
use tididi::negate;

let conj = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let disj = Tdd::clause(&vtree, [1]) | Tdd::clause(&vtree, [2]);
let neg = !Tdd::clause(&vtree, [1, 2]);
```

`&`, `|`, `!` forward to `apply_and`, `apply_or`, `negate`. `apply_and`
consumes its operands and recycles their storage into the result; clone an
operand first to keep it. Apply results are canonical. Negation is exact but
must first fill every level with the pairs it lacks, which can grow the
diagram; when only the count of `¬f` is needed, use `2ⁿ − count(f)`.
`try_apply_and` and `try_apply_or` return `ApplyError` instead of aborting
under a limit ([Limits and memory](#limits-and-memory)); `try_apply_and` also
takes the levels to emit as marginal ([Marginalization](#marginalization)).

`apply_and_clause(&mut acc, &lits)` conjoins one clause into an accumulator
without building the clause as a diagram; `try_apply_and_clause_owned` is the
fallible form. The accumulator is count-correct after every clause and
canonical after `minimize`.

```rust
use tididi::apply::apply_and_clause;

let mut acc = constant_one(&vtree);
for clause in [[1, -2], [2, 3], [-1, 3]] {
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    acc = apply_and_clause(&mut acc, &lits);
}
```

`try_apply_and_batch(acc, batch, &spine, &marg_parents, ..)` conjoins a small
diagram into a large accumulator visiting only the vtree levels the batch can
change, and returns `BatchMerge::Merged` or `BatchMerge::Declined` with both
operands intact when the restricted walk is not provably exact; a decline
means "run `try_apply_and`". Its rustdoc states the `spine` contract.

## Conditioning

`condition_var(&f, x, value)` returns the cofactor `f|x=value` with `x` removed
from the diagram; `condition_vars(&f, &vars, value)` conditions many variables
with one final `minimize`. Conditioning only shrinks the diagram and is sound
when other levels are marginal. The kept side of `x` becomes free, so the
model count of the result still carries a factor of two per conditioned
variable; divide by `2^k` for the count of the cofactor itself.

## Quantification

`project_var(&f, x)` returns `∃x. f`, computed as `f|x=⊤ ∨ f|x=⊥`;
`project_vars(&f, &vars)` forgets a set. The result keeps the vtree, so a
forgotten variable still ranges over both values in `model_count`. Call on a
diagram with no marginal levels.

```rust
use tididi::apply::project_var;

let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
let g = project_var(&f, VarId(1));                            // ∃x2: x1, with x2 free, twice the models
```

## Restrict-to-care

`restrict(&f, care, CareCanonical::{Yes, No})` prunes `f` to the pairs and
nodes that produce a model under `care`, returning `g` with `g ∧ care == f ∧
care` and `g` no larger than `f`. Pass `CareCanonical::Yes` when `care` is
already minimized to skip its reduction. The result is `Restricted::Unchanged`
when nothing died, `Restricted::Shrunk(g)` with a non-canonical `g`, or
`Restricted::False(⊥)`; `into_tdd(&f)` collapses the three to a diagram.

```rust
use tididi::apply::restrict::{restrict, CareCanonical};

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

`marginalize(&mut f, &levels)` sums the named vtree levels out of the
diagram: each becomes a marginal level holding one value per node instead of
pairs, and the storage below it is released. `levels` is sorted bottom-up, and
a level is frozen only once its children are frozen or are leaves. The value
of a node is the number of assignments to the level's subtree that reach it;
counting folds `Σ count(left) × count(right)` over pairs and stops at a
marginal level, and a free variable contributes a factor of two. After a
level is frozen no conjunction may touch it, so freeze a level only once
every clause over its variables is in. The error is `ApplyError::Deadline`;
the levels frozen before the cut keep their values.

`marginalize_schedule(&clauses, &vtree, &clauses_at, &keep_explicit,
&defer_nodes)` computes, for a clause-by-clause build, the levels that may be
frozen after each step; `intra_batch_completions` refines one step's group to
the clauses of a batch. `Tdd::has_marginal_level` reports whether any level
is marginal.

With a `WeightStore` attached, the same operation stores each node's
semiring value in the store instead of a count:

```rust
use tididi::query::RationalSemiring;
use tididi::weight_store::{Precision, WeightStore};
use tididi::marginal::{marginalize, weighted_value};

let sr = RationalSemiring::from_weights(&weights); // (w_neg, w_pos) per variable
f.attach_weights(WeightStore::new(sr, Precision::Exact));
marginalize(&mut f, &levels).unwrap();
let total = weighted_value(&f);                    // Option<WeightVal>
```

`Precision::Exact` folds in `BigRational`; `Precision::Log` folds in the
bounded-precision `SignedLog` domain. Attach the store before the first
freeze; a conjunction moves it to its result. `Tdd::weights` reads the store,
`Tdd::take_weights` detaches it, and `WeightStore::level(t)` reads a frozen
level's values. `weighted_value` folds whatever is still explicit above the
frozen levels and returns the diagram's value.

## Model counting

```rust
use tididi::query::model_count;

let n = f.model_count();   // BigUint; sugar for model_count(&f)
```

The count is over all variables of the vtree: a variable the function does
not mention contributes a factor of two. `model_count` panics on a diagram
that a budget abort left inconsistent (`Tdd::is_poisoned`).

`IncrementalPinnedCounter` counts under a partial assignment and updates the
count when pins change without a full pass: `new_with_fix(&f, n_pins, fix,
ColumnRetention::All)` allocates the per-level count columns, `set_pin(var,
Some(value))` pins a variable, `full_recompute(&f)` runs one pass,
`recompute_levels(&f, &levels)` recomputes only the levels between the
changed leaves and the root (children before parents), and `root_count(&f)`
reads the count. `fix = true` counts a pinned variable once; `fix = false`
leaves the factor of two. `ColumnRetention::Frontier` frees each column as
its parent completes and allows only `full_recompute` and `root_count`.

## Weighted and semiring evaluation

`evaluate(&f, &sr)` folds any `Semiring` bottom-up over an explicit diagram:
implement `zero`, `leaf(var, label)`, `add_assign`, and `mul`.
`RationalSemiring::from_weights(&[(w_neg, w_pos)])` is exact weighted model
counting in `BigRational`; `RationalSemiring::unit(n)` reproduces the model
count. A weighted value of zero is a cancellation, not unsatisfiability.

```rust
use tididi::query::{evaluate, RationalSemiring};

let sr = RationalSemiring::from_weights(&weights);
let wmc = evaluate(&f, &sr);
```

`SignedLog` is a signed log-domain value with `mul`, `add_assign`, and
`from_rational`. `WeightVal` is the per-node value a `WeightStore` holds; it
is `#[non_exhaustive]`, so build values with `WeightVal::exact` and read
them with `as_rational`, `into_rational`, or `into_rational_opt`.

## Reduction

```rust
use tididi::reduce::{minimize, try_minimize, MinimizeOptions, MinimizePasses};

minimize(&mut t);
try_minimize(&mut t, MinimizeOptions { passes: MinimizePasses::PruneOnly, ..Default::default() })?;
```

`minimize` prunes unreachable nodes and contracts twins until the diagram is
the canonical form for its vtree ([tdd.md](tdd.md)). Apply, `clause_to_tdd`,
and `Tdd::graft` return canonical diagrams; `apply_and_clause` accumulators,
`restrict` results, and hand-built diagrams need it. `try_minimize` returns
`ApplyError` instead of exiting on an allocation refusal or a deadline;
`MinimizeOptions` selects `MinimizePasses::{Full, PruneOnly, ContractOnly}`,
skips the content-twin scan, or carries a `ContentTwinProbe` across calls.
On `Err` the diagram is sound unless `Tdd::is_poisoned`, in which case drop
it. `minimize_oom_exit` reports an allocation refusal and exits as
`minimize` would.

## Restructuring

`rotate_left(&mut vtree, v)` and `rotate_right(&mut vtree, v)` rotate a vtree
in place and return a `RotationInfo`. On a compiled diagram,
`search_to_local_min(&mut t)` rotates the vtree under the diagram to a local
minimum of its size, and `rotation_search(&mut t, &mut objective, &config)`
does the same for any `RotationObjective` (`delta(before, after) -> i64`,
negative to accept; `SizeDelta` is the default objective).
`RotationSearchConfig` bounds the rebuilt level size and the sweep count;
both entries return `RotationSearchStats { probes, accepts, sweeps }`. Each
rotation rewrites only the two affected levels and re-minimizes them, and
the model count is preserved.

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

## Limits and memory

Limits are per-thread state installed for a lexical scope:

```rust
use std::time::{Duration, Instant};
use tididi::limits::{apply_limits, apply_meters, ApplyError, MemPressure, RopeLimit, Scheduled};
use tididi::apply::try_apply_and;

let _guard = apply_limits()
    .deadline(Some(Instant::now() + Duration::from_secs(30)))
    .budget(Some(4 << 30))          // bytes one apply may grow its scratch by
    .output_cap(Some(50_000_000))   // output nodes one apply may build
    .stall_rope(Some((1 << 20, RopeLimit::Wall(Instant::now() + Duration::from_secs(10)))))
    .schedule(Some(|_now| Scheduled::Carry))
    .mem_pressure(MemPressure::NONE)
    .watch(true)
    .apply();                       // restored when the guard drops
match try_apply_and(f, g, None) {
    Ok(h) => { /* ... */ }
    Err(ApplyError::OverBudget | ApplyError::Deadline | ApplyError::OutputCap) => { /* cut short */ }
}
```

`apply()` snapshots each named axis and returns a guard that restores it on
drop; axes not named are untouched, and naming an axis with `None` clears it
for the scope. The axes: `deadline`, a wall-clock instant; `budget`, a soft
byte budget for one apply's scratch (`set_apply_budget` writes the same cell
open-endedly); `output_cap`, a cap on output nodes; `stall_rope`, a deadline
that comes into force once the apply has built a floor of output pairs, with
`RopeLimit::Wall(instant)` or `RopeLimit::Work(units)` on the work clock;
`schedule`, a callback every in-operation poll asks, answering
`Scheduled::Carry`, `Scheduled::Stop`, or `Scheduled::Until(instant)`;
`mem_pressure`, the host's memory probes (`MemPressure` holds four function
pointers: `mapped_bytes`, `address_space_limit`, `preflight_alloc`,
`eager_reclaim`; `MemPressure::NONE` is the default); and `watch`, which
makes applies publish their position.

`ApplyError` has three variants: `OverBudget` (an allocation refused or the
budget exceeded), `Deadline` (the deadline, rope, or a `Scheduled::Stop`),
and `OutputCap`. It implements `Display` and `std::error::Error`, so it
propagates with `?` into `Box<dyn Error>`. An `Err` from an owned entry point
spends both operands.
`apply_meters()` snapshots the armed limits and the meters (`ApplyMeters`:
`in_flight_bytes`, `pairs_in_flight`, `work_units`, `refused_reserve_bytes`,
`merge` as a `MergePosition`, and every armed axis); `reset_apply_meters()`
zeroes the per-apply meters at the start of an independent compile. The
library reads no environment variables and installs no process-wide state;
every limit lives on the thread that installed it.

## Introspection

`Tdd::size()` is the total pair count, the size measure of the paper;
`size_capped(cap)` stops counting at `cap`. `total_nodes()`, `max_width()`,
`width_at(t)`, `effective_width(t)`, `is_zero()`, `has_marginal_level()`, and
`is_poisoned()` read the diagram's shape and state. `is_sat(&f)` is a
constant-time check on a minimized diagram; `implied_literals(&f)` returns
the literals true in every model of a minimized diagram;
`reduced_tdd_size(&f)` reports the size after the non-smooth reduction of
[tdd.md](tdd.md) without applying it.

## Serialization and rendering

`save_tdd(&f, path)` writes the `.tdd` text format (a header, one `L` line
per leaf, one `I` line per stored node with its pairs, bottom-up). `tdd_to_dot(&f)`
and `vtree_to_dot(&vtree, Some(&f))` render Graphviz DOT; the vtree render
colors each internal node by its pair count. Both refuse a diagram with a marginal
level with `std::io::ErrorKind::InvalidInput`. `Vtree::to_vtree_text()` writes
the `.vtree` format. `vtree_example.svg` and `tdd_example.svg` in this
directory are renders of one diagram.

## Traversing a diagram

The stored encoding is the traversal contract. Read `Tdd::levels` directly,
children before parents, with `vtree.internal_bottomup()`; a level is
leaf (nothing stored), structural (`nodes` and `pairs`), or marginal
(`marginal_counts`). `TddLevel::internal_inputs_iter` yields each live node
with its pairs, and `resolve_marg_ref` decodes a pair side whose child level
is marginal. The `diagram` module documentation lists the invariants a
reader may rely on.

```rust
use tididi::diagram::{MargResolved, resolve_marg_ref};

for (t, left, right) in f.vtree.internal_bottomup() {
    let level = f.level(t);
    if level.is_marginal() { continue; }
    let (lm, rm) = (f.level(left).is_marginal(), f.level(right).is_marginal());
    for (i, pairs) in level.internal_inputs_iter() {
        for p in pairs {
            let l = resolve_marg_ref(p.left.0, lm);   // Inline(count) or Index(node)
            let r = resolve_marg_ref(p.right.0, rm);
        }
    }
}
```

`examples/traverse_count.rs` is a complete model count against this contract
and `examples/statistic.rs` a custom statistic; run them with `cargo run
--example traverse_count`. `Tdd::try_from_levels(vtree, levels, output)`
assembles a diagram from levels you filled, checking the invariants and
returning `TddBuildError` on the first violation; the result is well-formed
but not canonical until `minimize` runs.
