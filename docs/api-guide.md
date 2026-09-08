# Using the library

A task-oriented tour of the `tididi` crate: build a vtree, build TDDs, combine
and transform them, and query them. For the data model see [tdd.md](tdd.md).
Full item docs are on [docs.rs](https://docs.rs/tididi).

Every TDD is tied to a vtree, shared cheaply as an `Arc<Vtree>`. Operands to a
binary op must share the same vtree.

## Building a vtree

```rust
use std::sync::Arc;
use tididi::vtree::{VarId, Vtree};

let vtree = Arc::new(Vtree::balanced(4)); // 4 variables, balanced tree
```

Variables are `VarId(0..n)` — 0-based internally. The constructors, all
without a CNF: `Vtree::leaf(var)` (one variable), `Vtree::join(&l, &r)` (a new
root over two vtrees with disjoint variables — the composition primitive),
`Vtree::balanced(n)` / `Vtree::balanced_over(&order)` (balanced shape, in
natural or the given left-to-right order), `Vtree::linear(n)` /
`Vtree::linear_from_order(&order)` (right-linear, an OBDD variable order),
`Vtree::random(n, seed)`, `Vtree::from_vtree_text(s)` / `to_vtree_text()` (the
SDD `.vtree` text format, also what vitri emits), `Vtree::graft(&subtrees,
&spine_vars)` (independent subtrees and single variables hung under one
right-linear spine), and `project_to_vars` (restrict to a subset, keeping the
grouping). A vtree may skip variable ids: `num_vars()` is the id space,
`num_leaves()` the variables it carries. `validate()` checks the structural
invariants of anything you built by hand; the combining constructors return
`VtreeError` on overlapping variables. Read a tree through `root()`, `node()`,
`children()`, `leaf_of()`, `leaf_var()`, `bottomup()` and `lca()`, and compare
trees with `same_tree()` (node numbering is not identity).

## Build TDDs: constants and clauses

```rust
use tididi::tdd::Tdd;
use tididi::tdd::build::{constant_one, constant_zero, clause_to_tdd};

let top = constant_one(&vtree);     // ⊤
let bot = constant_zero(&vtree);    // ⊥ (output is the ZERO sentinel; no nodes)

// A single clause. Integers use the 1-based DIMACS sign convention:
// 1 → x1, -2 → ¬x2. So this is (x1 ∨ ¬x2):
let c = Tdd::clause(&vtree, [1, -2]);
```

`Tdd::clause` is ergonomic sugar; the underlying API is the free function
`clause_to_tdd(&vtree, &[Literal])`, where `Literal` is built from a `VarId` and
polarity (`Literal::pos`, `Literal::neg`) or via `Literal::from(i32)` using the
same DIMACS convention. `clause_to_tdd` builds the minimal canonical clause TDD
directly, with no build-then-minimize round trip.

## Combine TDDs

```rust
use tididi::tdd::transform::pairwise::conjoin::apply_and;
use tididi::tdd::transform::pairwise::disjoin::apply_or;

// Preferred ergonomic form (consumes both operands):
let conj = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let disj = Tdd::clause(&vtree, [1]) | Tdd::clause(&vtree, [2]);

// Underlying API:
let a = Tdd::clause(&vtree, [1, -2]);
let b = Tdd::clause(&vtree, [2, 3]);
let conj2 = apply_and(a, b);
```

`apply_and` / `&` compute conjunction; `apply_or` / `|` compute disjunction; `!f`
computes negation. `apply_and` consumes its operands: it drains their level
arenas as it goes and recycles the storage into the result, so clone one first if
you need to keep it. `try_apply_and` is the fallible form, and also takes the
marginalization targets. The output of apply is
already canonical (minimized in-line), so you rarely need to call `minimize`
afterward. Apply runs a compacting product over child levels; its cost scales
with the product of the operand widths at each level.

For conjoining a stream of clauses (e.g. compiling a CNF one clause at a time),
prefer `apply_and_clause`, which conjoins a clause into an accumulator without
first materializing the clause as a separate TDD:

```rust
use tididi::tdd::transform::pairwise::conjoin_clause::apply_and_clause;

let cnf: Vec<Vec<i32>> = vec![vec![1, -2], vec![2, 3], vec![-1, 3]];
let mut acc = constant_one(&vtree);
for clause in &cnf {
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    acc = apply_and_clause(&mut acc, &lits);
}
```

## Unary transforms

- **negate** — `!f`, or `tididi::tdd::negate(&f)`, returns the canonical `¬f`.
  Negation is exact but can grow the diagram, sometimes sharply: a compiled TDD stores only the pair structure of its *satisfying*
  assignments, so negation must first **fill** each level with the pairs no
  existing node holds (making the level cover the whole assignment space) before
  complementing at the root — and the fill typically dominates. When only the
  count of `¬f` is needed, `count(¬f) = 2ⁿ − count(f)` avoids building it at all.

- **condition** — `condition_var(&f, x, value)` returns the cofactor `f|x=value`
  with `x` *removed* from the result. `condition_vars(&f, &vars, value)`
  conditions many variables with a single final minimize. Conditioning is
  monotone non-increasing in size — it can only shrink the diagram — and is safe
  in counting mode.

- **restrict** — `restrict(&f, care, care_canonical)` is a generalized cofactor:
  it returns a `g` with `g ∧ care == f ∧ care` and `g` no larger than `f`. `g`
  agrees with `f` wherever `care` holds and is free to differ (as don't-cares)
  where `care` is false; the caller typically discards the don't-cares with a
  later `∧ care`. Use it to grow a conjunction cheaply when the care does not
  collapse `f`.

  ```rust
  use tididi::tdd::transform::unary::restrict::{restrict, CareCanonical};

  let f = Tdd::clause(&vtree, [1, 2]);
  let care = Tdd::clause(&vtree, [1]);
  let g = restrict(&f, care, CareCanonical::No).into_tdd(&f);
  ```

- **project (existential)** — `project_var(&f, x)` returns `∃x. f`, computed as
  `apply_or(f|x=⊤, f|x=⊥)`. `project_vars(&f, &vars)` forgets a set of variables.
  Call on a fully structural (non-counting-mode) TDD.

- **marginalize** — the counting-mode primitive. `marginalize(&mut f, &levels)`
  sums each named vtree level out of the diagram, replacing its Boolean
  structure with per-node values and freeing the arenas below it. The values are
  integer model counts, one per node, unless a `WeightStore` is attached to the
  diagram (`f.attach_weights(store)`), in which case each level is summed in
  that store's semiring instead and the results live in the store;
  `weighted_value(&f)` then reads the diagram's total. A level may be
  marginalized only once every clause over its variables has been conjoined in;
  after that no further conjunction may touch it. Most callers never touch this
  directly.

## Minimize

```rust
use tididi::tdd::minimize::minimize;

let mut t = /* some non-canonical TDD */;
minimize(&mut t); // prune unreachable + twin contraction → canonical form
```

`minimize` yields the canonical reduced form for the TDD's vtree: no false nodes,
no unreachable nodes, no duplicate functions at a level. Apply and `clause_to_tdd`
already return canonical results, so call `minimize` only after operations that
may leave a diagram non-canonical (e.g. building one by hand).

`try_minimize(&mut t, MinimizeOptions { .. })` is the fallible form: it reports an
allocation refusal or a deadline cut as `ApplyError` instead of exiting, and its
options select a cheaper subset of the passes (`MinimizePasses::PruneOnly`,
`MinimizePasses::ContractOnly`), skip the content-twin canonicalization, or carry a
`ContentTwinProbe` scan schedule across calls.

## Queries

```rust
use tididi::tdd::query::is_sat;

let f = Tdd::clause(&vtree, [1, -2]);
let n = f.model_count();     // num_bigint::BigUint — exact, arbitrary precision
let sat = is_sat(&f);        // bool
```

`Tdd::model_count()` is sugar for the free function
`tididi::tdd::query::model_count(&f)`; both return a `BigUint`.

**Weighted / algebraic counting** goes through the `Semiring` trait and the
generic `evaluate` traversal:

```rust
use tididi::tdd::query::semiring::{evaluate, RationalSemiring};

// One literal-weight pair (w_pos, w_neg) per variable:
let sr = RationalSemiring::from_weights(&weights);
let wmc = evaluate(&f, &sr); // exact rational weighted model count
```

`Semiring` has `zero`, `leaf(var, label)`, `add_assign`, and `mul`; implement it
to fold any commutative semiring bottom-up over the diagram.

**Structural queries.** `implied_literals(&f)` returns the literals forced true in
every model. `reachable_pairs(&f)` and `Tdd::size()` report the input-pair count
(the standard TDD size metric); `Tdd::max_width()` gives the widest level.
`reduced_tdd_size(&f)` measures how much a non-smooth reduction *could* save
without modifying the diagram.

## Traversing a diagram

The stored encoding is the traversal contract: read `Tdd::levels` directly,
visiting children before parents with `vtree.internal_bottomup()`. The
`tididi::tdd::types` module documentation lists the level states (leaf,
structural, marginal), how a pair side is decoded when its child level is
marginal (`resolve_marg_ref`), and the invariants a reader may rely on.

```rust
use tididi::tdd::types::{MargResolved, resolve_marg_ref};

for (t, left, right) in f.vtree.internal_bottomup() {
    let level = f.level(t);
    if level.is_marginal() { /* one count per node in level.marginal_counts */ continue; }
    let (lm, rm) = (f.level(left).is_marginal(), f.level(right).is_marginal());
    for (i, pairs) in level.internal_inputs_iter() {
        for p in pairs {
            let l = resolve_marg_ref(p.left.0, lm);   // Inline(count) or Index(node)
            let r = resolve_marg_ref(p.right.0, rm);
            // ...
        }
    }
}
```

Two complete walks ship as examples: `examples/traverse_count.rs` (a model
count, checked against `model_count`) and `examples/statistic.rs` (the widest
node). Run them with `cargo run --example traverse_count`.

To build a diagram from levels you filled yourself, use `Tdd::try_from_levels`;
it checks the invariants and returns a `TddBuildError` naming the first
violation. The result is well-formed but not canonical until `minimize` runs.

## Vtree restructuring

The vtree strongly affects TDD size, and you can improve it *after* compiling by
rotating the live diagram. `search_to_local_min(&mut t)` drives a size-reducing
rotation search to a local minimum. Each rotation rewrites only the two affected
levels and re-minimizes locally, so the search is cheap per step, and it always
preserves the model count. Reach for it when a compiled TDD is larger than you
want and you are willing to spend time shrinking it.

The search is objective-generic: `rotation_search(&mut t, &mut obj, &cfg)` takes
any `RotationObjective` (implement `delta(before, after) -> i64`, negative to
accept) so you can descend a metric other than size. `search_to_local_min` is the
convenience form with the built-in `SizeDelta` objective and default config.

```rust
use tididi::tdd::restructure::search::{rotation_search, search_to_local_min, RotationObjective, RotationSearchConfig};
use tididi::tdd::types::TddLevel;

// Custom objective: only accept a rotation that strictly shrinks the *wider*
// of the two affected levels (delta < 0 accepts).
struct MinPeak;
impl RotationObjective for MinPeak {
    fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
        a.0.width().max(a.1.width()) as i64 - b.0.width().max(b.1.width()) as i64
    }
}
let stats = rotation_search(&mut t, &mut MinPeak, &RotationSearchConfig::default());
// or simply: let stats = search_to_local_min(&mut t);  // size objective
```

Both entry points return `RotationSearchStats { probes, accepts, sweeps }` (it is
`Debug`-printable), tallying rotations scored, rotations accepted, and full sweeps run.

## Graft

`Tdd::graft(parts, &spine_vars)` is the conjunction of TDDs over
pairwise-disjoint variable sets, each on its own vtree, built structurally on
`Vtree::graft` of their vtrees: the parts' levels are moved into place and one
width-1 level per spine join ties them together, so no apply runs and the
result is canonical when the parts are. `spine_vars` are variables no part
mentions; the result is unconstrained in them, so each doubles the model count.
This is how independent pieces of a function (the components of a CNF, say)
compiled separately become one TDD.

```rust
let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
let fg = Tdd::graft(vec![Tdd::clause(&a, [1, 2]), Tdd::clause(&b, [3, -4])], &[VarId(4)])?;
assert_eq!(fg.model_count(), 18u32.into()); // 3 · 3 · 2
```

## Limits and memory

```rust
use tididi::tdd::limits::{apply_limits, ApplyError, MemPressure};
use tididi::tdd::transform::pairwise::conjoin::try_apply_and;

let _limits = apply_limits()
    .deadline(Some(std::time::Instant::now() + std::time::Duration::from_secs(30)))
    .budget(Some(4 << 30))        // bytes one apply may grow its scratch by
    .output_cap(Some(50_000_000)) // output nodes one apply may build
    .apply();                     // restored when the guard drops
match try_apply_and(f, g, None) {
    Ok(h) => { /* ... */ }
    Err(ApplyError::Deadline | ApplyError::OverBudget | ApplyError::OutputCap) => { /* cut short */ }
}
```

`apply_limits()` also takes a `.schedule(..)` callback, which the in-operation
polls ask alongside the deadline: it is handed the clock reading the poll already
took, and answers `Scheduled::Carry`, `Scheduled::Stop`, or `Scheduled::Until(t)`
to replace the deadline the operation runs under. That is a place to stand inside
an apply that would otherwise run to completion before the caller is asked
anything.

Every limit is scoped: `apply_limits()` names the axes to install, `apply()`
installs them for the current thread and returns a guard that restores the
previous values when dropped, so nested scopes tighten and release cleanly.
Axes not named are untouched. `apply_meters()` snapshots what is armed and
what the engine has metered against it:

```rust
use tididi::tdd::limits::{apply_limits, apply_meters};

let _watch = apply_limits().watch(true).apply();
// ... an apply runs ...
let m = apply_meters();
// m.in_flight_bytes, m.pairs_in_flight: what the apply in flight has charged and built
// m.work_units: a monotone work clock, so an interval is a difference of two reads
// m.refused_reserve_bytes: the size the allocator refused, if it did (vs. the soft budget)
// m.merge: where a watched apply stands (began, level, levels)
// m.deadline, m.budget_remaining, m.output_node_cap, m.stall_rope, m.schedule: what is armed
```

`reset_apply_meters()` zeroes the in-flight charge and the recorded refusal;
a loop that compiles several diagrams on one thread calls it at each entry so
one compile never inherits another's charge. `mem_pressure(MemPressure { .. })` is the one
axis a host process usually installs once, around everything it compiles: four
plain function pointers through which the engine learns the mapped high-water
bytes and the address-space ceiling, announces a growth allocation before
making it, and nudges the host to reclaim once per apply. The default is
`MemPressure::NONE` — no ceiling, plain doubling growth — and the crate never
reads the environment or installs a process global on its own.

## Memory behavior

TDDs share their vtree via `Arc` and are cloned cheaply only in that respect —
level storage is owned per TDD. Apply drains its operands' storage, so pass
throwaway operands by value where possible (`f & g`) rather than keeping copies
alive. In counting mode, marginalizing frozen levels frees their Boolean arenas
and is the main lever for keeping peak memory down on large counts.

---

See [tdd.md](tdd.md) for the data model and the paper
(<https://arxiv.org/abs/2604.05537>) for the theory.
