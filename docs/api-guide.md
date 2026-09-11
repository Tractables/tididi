# API guide

The first section is the whole library in three snippets, and the table after
it places every operation by cost. The sections below those stand on their
own: read the one for the capability you need. Item documentation is on
[docs.rs](https://docs.rs/tididi); the data model is in [tdd.md](tdd.md).

Every diagram is tied to a vtree, shared as an `Arc<Vtree>`. The operands of
a binary operation must share the same [`Arc`].

The documented API is the modules below. `compiler_seam` and `check` are
hidden: `compiler_seam` holds every entry point a clause-by-clause driver
reaches the crate through, and `check` the invariant checkers; neither is
covered by any compatibility promise.

## First five minutes

Build a diagram one clause at a time, reduce it to the canonical form for its
vtree, and count its models.

```rust
use std::sync::Arc;
use num_bigint::BigUint;
use tididi::Tdd;
use tididi::apply::apply_and_clause;
use tididi::reduce::minimize;
use tididi::vtree::Vtree;

let vtree = Arc::new(Vtree::balanced(4));   // x1..x4
let mut f = Tdd::one(&vtree);
for clause in [[1, -2], [2, 3], [-3, 4]] {  // DIMACS literals
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    f = apply_and_clause(f, &lits);
}
minimize(&mut f);
assert_eq!(f.model_count(), BigUint::from(5u32));
```

Conjoin two diagrams under a caller's limits. An operation that runs past a
limit returns [`ApplyError`] rather than a partial answer, and an owned entry
point spends both operands on the way out.

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
use std::time::Instant;
use tididi::{ApplyError, Engine, Tdd};
use tididi::engine::LimitSet;

let vtree = Arc::new(Vtree::balanced(4));
let engine = Engine::new();
engine.limits().install(LimitSet::none().deadline(Some(Instant::now())));

let f = Tdd::clause(&vtree, [1, -2]);
let g = Tdd::clause(&vtree, [2, 3]);
match engine.and(f, g) {
    Ok(h) => assert!(!h.is_zero()),          // it finished before a poll
    Err(ApplyError::Deadline) => {}          // the deadline had already passed
    Err(e) => panic!("unexpected refusal: {e}"),
}
```

Sum a vtree level out. The diagram keeps its count and loses the storage below
that level, which is what bounds the memory a count needs.

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
use tididi::{Engine, Tdd};
use tididi::marginal::marginalize;

let vtree = Arc::new(Vtree::balanced(4));
let engine = Engine::new();
let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let before = f.model_count();

// The root's left child: its own children are leaves, so it may go first.
let (left, _right) = vtree.children(vtree.root());
marginalize(&engine, &mut f, &[left]).unwrap();

assert!(f.has_marginal_level());
assert_eq!(f.model_count(), before);
```

## Operations by cost

Cost is in the diagram's own terms: what the operation walks, folds over, or
rebuilds. Refusal is available only on a caller's [`Engine`] — the free
functions run on an engine with nothing armed, so no limit cuts one short —
except where the table says otherwise.

| Operation | Cost | Refuses under limits | Canonical form |
|---|---|---|---|
| [`Tdd::one`], [`Tdd::zero`], [`Tdd::clause`], [`engine.cube`] | linear in the vtree | on an engine | canonical |
| [`Tdd::graft`] | linear in the parts, and runs no apply | no | canonical when the parts are |
| [`engine.and`], [`engine.or`] and the operator forms | product of the operand levels | on an engine | canonical |
| [`apply_and_clause`], [`engine.and_clause`] | walks the accumulator bottom-up | on an engine | count-correct; canonical after [`minimize`] |
| [`engine.and_batch`] | visits only the levels the batch may touch | yes | canonical, or declines with both operands intact |
| [`negate`] | fills every level, then complements | no | canonical |
| [`condition_var`], [`condition_vars`] | walks the diagram, and never grows it | on an engine | canonical |
| [`project_var`], [`project_vars`] | disjoins the cofactors of the variable, or rewrites the levels above it | on an engine | canonical |
| [`restrict`] | walks the diagram, and never grows it | on an engine | not canonical; run [`minimize`] |
| [`minimize`], [`try_minimize`] | walks the diagram | [`try_minimize`] only | establishes it |
| [`marginalize`] | walks the levels named, and frees the storage below them | yes | preserved |
| [`Tdd::model_count`], [`engine.model_count`], [`evaluate`] | folds over the diagram | on an engine | unchanged |
| [`IncrementalCounter`] | folds over the diagram, then over the levels between the changed leaves and the root | yes | unchanged |
| [`rotation_search`] | rebuilds the levels each pivot touches | on an engine | canonical |
| [`save_tdd`], [`load_tdd`], [`tdd_to_dot`] | walks the diagram | no | unchanged |
| [`Tdd::size`], [`node_count()`], [`max_width()`] | walks the diagram | no | unchanged |

## Building

### Diagrams and vtrees

[`Tdd`] is the diagram: one [`TddLevel`] per vtree node, an `output` node, and
the `Arc<Vtree>`. [`Vtree`] is a binary tree whose leaves are variables.
Variables are [`VarId(0..n)`], 0-based; [`Literal`] pairs a [`VarId`] with a
polarity ([`Literal::pos`], [`Literal::neg`], or [`Literal::from(i32)`] with the
1-based DIMACS sign convention).

```rust
use std::sync::Arc;
use tididi::Literal;
use tididi::vtree::{VarId, Vtree};

let vtree = Arc::new(Vtree::balanced(4)); // x1..x4, balanced shape
let x1 = Literal::pos(VarId(0));          // the literal x1, 0-based
assert_eq!(x1, Literal::from(1));         // the same literal, DIMACS-signed
```

Constructors: [`Vtree::leaf(var)`]; [`Vtree::join(&l, &r)`] for a new root over
two vtrees with disjoint variables; [`Vtree::balanced(n)`] and
[`Vtree::balanced_over(&order)`]; [`Vtree::linear(n)`] and
[`Vtree::linear_over(&order)`] for a right-linear chain, which is an OBDD
variable order; [`Vtree::random(n, seed)`]; `Vtree::graft(&subtrees,
&spine_vars)` (see [Graft](#graft)); and `project_to_vars` for the vtree
induced on a subset of the variables. [`Vtree::from_text`] parses the
`.vtree` text format (`vtree N`, then `L <id> <var>` and `I <id> <left>
<right>` lines, last node the root) and `to_text` writes it. A vtree may
skip variable ids: [`num_vars()`] is the id space and [`num_leaves()`] the
variables carried. [`validate()`] checks the invariants of a hand-built tree.
Construction and parsing errors are [`VtreeError`] ([`Text`],
[`OverlappingVariable`], [`Invalid`]).

Read a tree with [`root()`], [`node()`], [`children()`], [`sibling()`], [`leaf_of()`] (`None` for a variable no leaf carries),
[`leaf_var()`], [`lca()`], and the traversal orders [`bottomup()`],
[`leaf_bottomup()`], [`internal_bottomup()`]. [`same_tree()`] compares shape and
variables; node numbering is not identity.

### Base diagrams

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::Tdd;

let top = Tdd::one(&vtree);          // ⊤
let bot = Tdd::zero(&vtree);         // ⊥: the ZERO sentinel, no nodes
let c = Tdd::clause(&vtree, [1, -2]);    // x1 ∨ ¬x2
```

[`Tdd::clause`] accepts anything convertible to [`Literal`], so a `&[i32]` of
DIMACS literals and a `&[Literal]` both work. It builds the canonical diagram
of the clause directly. [`engine.cube`] is the conjunction of literals rather
than their disjunction, built the same way: one width-1 node per internal
vtree node, so the whole diagram is one path. A variable no literal mentions
is free in both.

### Graft

[`Tdd::graft(parts, &spine_vars)`] is the conjunction of diagrams over
pairwise-disjoint variable sets, each on its own vtree, built structurally on
[`Vtree::graft`] of their vtrees: the parts' levels move into place and one
width-1 level per join ties them together, so no apply runs and the result is
canonical when the parts are. `spine_vars` are variables no part mentions;
the result is unconstrained in them. The error is [`VtreeError`].

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
let fg = Tdd::graft(vec![Tdd::clause(&a, [1, 2]), Tdd::clause(&b, [3, -4])], &[VarId(4)]).unwrap();
assert_eq!(fg.model_count(), 18u32.into()); // 3 · 3 · 2
```

## Combining

### Boolean combination

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::negate;

let conj = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let disj = Tdd::clause(&vtree, [1]) | Tdd::clause(&vtree, [2]);
let neg = !Tdd::clause(&vtree, [1, 2]);
let same = negate(Tdd::clause(&vtree, [1, 2]));  // what `!` forwards to
assert_eq!(neg.model_count(), same.model_count());
```

`&` and `|` are conjunction and disjunction; `!` forwards to [`negate`]. All three
consume their operands and recycle their storage into the result; clone an
operand first to keep it. Apply results are canonical. Negation is exact but
must first fill every level with the pairs it lacks, which can grow the
diagram; when only the count of `¬f` is needed, use `2ⁿ − count(f)`.
[`engine.and(f, g)`] and [`engine.or(f, g)`] are the same operations run on a
caller's engine: they return [`ApplyError`] instead of aborting under a limit
([Limits and refusal](#limits-and-refusal)), and reuse the engine's buffers
across calls. [`engine.and_marginalizing(f, g, &targets)`] names the vtree
levels to emit as marginal ([Marginal levels](#marginal-levels)).

[`apply_and_clause(acc, &lits)`] conjoins one clause into an accumulator
without building the clause as a diagram; [`engine.and_clause(acc, &lits)`] is
the fallible form. Both consume the accumulator and hand back the new one, which
is count-correct after every clause and canonical after [`minimize`], as in the
first snippet above.

[`engine.and_batch(acc, batch, &levels)`] conjoins a small diagram into a large
accumulator visiting only the vtree levels the batch can change, and returns
[`BatchMergeOutcome::Merged`] or [`BatchMergeOutcome::Declined`] with both operands intact when
the restricted walk is not provably exact; a decline means "run [`engine.and`]".
`levels` names the levels the batch may touch; the method's rustdoc states the
contract that set must satisfy. Everything else the walk needs is a property of
the accumulator, which carries it.

### Conditioning

[`condition_var(&f, x, value)`] returns the cofactor `f|x=value` with `x` removed
from the diagram; [`condition_vars(&f, &vars, value)`] conditions many variables
with one final [`minimize`]. Conditioning only shrinks the diagram and is sound
when other levels are marginal. The kept side of `x` becomes free, so the
model count of the result still carries a factor of two per conditioned
variable; divide by `2^k` for the count of the cofactor itself.

### Quantification

[`project_var(&f, x, how)`] returns `∃x. f`; [`project_vars(&f, &vars, how)`]
forgets a set. The result keeps the vtree, so a forgotten variable still ranges
over both values in [`Tdd::model_count`].

`how` picks the rewrite. [`Projection::Automatic`] computes `f|x=⊤ ∨ f|x=⊥`
where that is sound and switches to an in-place leaf-to-root rewrite where it is
not — the cofactor form disjoins by negation, which a marginal level cannot
survive. [`Projection::Structural`] asks for the in-place rewrite outright: it is
slower, and it never clones the diagram to negate it, which is what a caller
wants when the diagram is large enough for that clone to be the risk.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::{project_var, Projection};

let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
let g = project_var(&f, VarId(1), Projection::Automatic);     // ∃x2: x1, with x2 free, twice the models
assert_eq!(g.model_count(), f.model_count() * 2u32);
```

### Restrict-to-care

[`restrict(f, care, CareCanonical::{Yes, No})`] prunes `f` to the pairs and
nodes that produce a model under `care`, returning `g` with `g ∧ care == f ∧
care` and `g` no larger than `f`. Pass `CareCanonical::Yes` when `care` is
already minimized to skip its reduction. The result is [`Restricted::Unchanged`]
when nothing died, [`Restricted::Shrunk(g)`] with a non-canonical `g`, or
[`Restricted::Unsatisfiable(⊥)`]; [`into_tdd()`] collapses the three to a
diagram. Both operands are consumed, and the unchanged arm hands `f` straight
back. [`engine.restrict`] is the same operation under the caller's limits: the
rebuild ends in a prune that an armed stop can cut.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::{restrict, CareCanonical};

let f = Tdd::clause(&vtree, [1, 2]);
let care = Tdd::clause(&vtree, [1]);
let g = restrict(f.clone(), care.clone(), CareCanonical::No).into_tdd();
assert_eq!((g & care.clone()).model_count(), (f & care).model_count());
```

### Reduction

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# fn main() -> Result<(), tididi::ApplyError> {
# let vtree = Arc::new(Vtree::balanced(4));
# let mut t = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
use tididi::engine::Engine;
use tididi::reduce::{minimize, try_minimize, MinimizeOptions, MinimizeScope};

let engine = Engine::new();
minimize(&mut t);
try_minimize(&engine, &mut t, MinimizeOptions { passes: MinimizeScope::PruneOnly, ..Default::default() })?;
# Ok(())
# }
```

[`minimize`] prunes unreachable nodes and contracts twins until the diagram is
the canonical form for its vtree ([tdd.md](tdd.md)). Apply, [`Tdd::clause`],
and [`Tdd::graft`] return canonical diagrams; [`apply_and_clause`] accumulators,
[`restrict`] results, and hand-built diagrams need it. [`try_minimize`] returns
[`ApplyError`] instead of exiting on an allocation refusal or a deadline;
[`MinimizeOptions`] selects [`MinimizeScope::{Full, PruneOnly, ContractOnly}`],
skips the content-twin scan, or carries a [`ContentTwinProbe`] across calls.
On `Err` the diagram is exactly as it was at the last pass boundary.
[`minimize`] itself panics on a refusal, so a caller that must survive one uses
[`try_minimize`].

## Limits and refusal

An [`Engine`] is the session an operation runs in: it owns the limits the
operation runs under and every buffer the operation reuses between calls.
Nothing is per-thread, and nothing is armed over an operation the caller did
not arm it over:

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
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

[`LimitSet`] is a plain `Copy` value and installing one replaces every axis.
[`engine.limits().install(set)`] returns what was armed before, so a caller that
wants one axis changed for a scope reads the armed set, edits the field, and
installs what it found again when the scope ends. [`LimitSet::uncut()`] clears
the whole stop axis, which [`deadline(None)`] does not: that clears the
unconditional wall and leaves a size-conditional bound in force.

The axes: [`budget`], a soft byte budget for one operation's storage;
[`output_cap`], a cap on the nodes one conjunction may build; [`stop`], when the
operation gives up; [`schedule`], a callback the in-operation polls ask;
[`mem_pressure`], the host's memory probes ([`MemPressure`] holds four function
pointers: [`mapped_bytes`], [`address_space_limit`], [`preflight_alloc`],
[`eager_reclaim`]; [`MemPressure::NONE`] is the default); and [`watch`], which makes
conjunctions publish where they stand.

A [`Stop`] carries two bounds in one axis. [`wall`] is unconditional — past it the
operation stops whatever it has built. [`after`] is `(pairs, at)`: it applies once
the operation has built that many output pairs, so a caller can cut a step for
spending too long on a big diagram and leave a small one alone. Each bound falls
either at an instant ([`StopAt::Wall`]) or at a reading of the engine's own work
clock ([`StopAt::Work`]), which is reproducible across machines where a wall is
not. [`LimitSet::deadline(Some(t))`] is the common case, and [`Stop::by(t)`] spells
the same thing.

The schedule callback is asked on every poll, handed the meters and the clock
reading the poll had already taken. It answers [`Scheduled::Carry`],
[`Scheduled::Stop`], or [`Scheduled::Replace(stop)`] — a commitment that replaces
the stop the operation was running under. The library holds no view on when a
decision is due: a caller with decision points of its own tests them and carries
until one arrives.

[`engine.reset()`] releases every buffer the engine retains, keeping the armed
limits. Call it between a failed operation and whatever recovers from it, so
the recovery starts on a clean allocator slate instead of inheriting the peak
the failure parked. It is sound only between operations.

[`ApplyError`] has three variants: [`OverBudget`] (an allocation refused or the
budget exceeded), [`Deadline`] (a stop fell, or a schedule said so), and
[`OutputCap`]. It implements [`Display`] and [`std::error::Error`], so it propagates
with `?` into `Box<dyn Error>`. An `Err` from an owned entry point spends both
operands. A caller may also return one for a resource failure of its own.

[`engine.limits().meters()`] snapshots the meters ([`ApplyMeters`]:
[`in_flight_bytes`], [`pairs_in_flight`], [`work_units`], [`refused_reserve_bytes`], and
[`merge`] as a [`MergeProgress`]); [`engine.limits().armed()`] reads back what is
armed; [`reset_meters()`] zeroes the per-operation meters
at the start of an independent compile. The infallible entries — [`apply_and_clause`],
[`minimize`], [`Tdd::model_count`], [`project_var`], [`restrict`], [`condition_var`],
[`Tdd::clause`], [`Tdd::one`], [`Tdd::zero`], [`rotation_search`], the operators — run
on an engine of their own with nothing armed, so no caller's deadline can cut
one short. Each has a form that computes the same thing under the caller's
limits and keeps the buffers warm for the next call ([`engine.and`],
[`engine.or`], [`engine.and_clause`], [`engine.project_var`],
[`engine.restrict`], [`engine.condition_var`], [`engine.clause`],
[`engine.one`], [`engine.zero`], [`engine.rotation_search`],
[`try_minimize`]); the free function is that form on a transient engine. The
split is deliberate: a free function that borrows its operand is the infallible
convenience, and the engine form that owns it is the one that can refuse.
Negation is the exception: [`negate`] has no such form and always runs with
nothing armed. The library reads no environment variables and holds no
process-wide state.

## Counting and semirings

### Model counting

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let n = f.model_count();   // BigUint
# assert_eq!(n, tididi::Engine::new().model_count(&f).unwrap());
```

The count is over all variables of the vtree: a variable the function does
not mention contributes a factor of two. [`engine.model_count(&f)`] is the same
count under the engine's limits, returning `Err(ApplyError)` where an armed
stop cuts the pass.

[`IncrementalCounter`] counts under a partial assignment and updates the count when
pins change without a full pass. Two type parameters say what a given counter
can do. The first is the column-lifetime policy: [`KeepAllColumns`] keeps a column per
level, [`KeepFrontier`] frees each column as its parent completes and offers the
output count alone. The second is whether a pass has run:
[`IncrementalCounter::new(eng, &f, n_pins, convention)`] allocates the columns,
[`set_pin(var, Some(value))`] pins a variable, and [`compute(eng, &f)`] consumes the counter and returns one in the
[`Evaluated`] state, where [`output_count(&f)`] reads the count. [`SeedConvention::Fixed`]
counts a pinned variable once; [`SeedConvention::Free`] leaves the factor of two.

Under [`KeepAllColumns`] a computed counter also has
[`recompute_dirty(eng, &f, &levels)`], which recomputes only the levels between the changed leaves and the
root. `levels` is a [`BottomUpSubset`], minted by [`vtree.bottom_up_subset(...)`]
from levels named in any order, so a level can never be recomputed before its
children.

### Weighted and semiring evaluation

[`evaluate(&f, &sr)`] folds any [`EvalAlgebra`] bottom-up over an explicit diagram:
implement `zero`, `leaf(var, label)`, `add_assign`, and `mul`.
[`RationalWeights::from_weights(&[(w_neg, w_pos)])`](crate::diagram::RationalWeights::from_weights) is exact weighted model
counting in [`BigRational`]; [`RationalWeights::unit(n)`] reproduces the model
count. A weighted value of zero is a cancellation, not unsatisfiability.

```rust
# use std::sync::Arc;
# use num_rational::BigRational;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
# let half = BigRational::new(1.into(), 2.into());
# let weights: Vec<(BigRational, BigRational)> =
#     (0..4).map(|_| (half.clone(), half.clone())).collect();
use tididi::diagram::RationalWeights;
use tididi::query::evaluate;

let sr = RationalWeights::from_weights(&weights);
let wmc = evaluate(&f, &sr);
```

[`SignedLog`] is a signed log-domain value with `mul`, `add_assign`, and
[`from_rational`]. [`WeightVal`] is the per-node value a [`WeightStore`] holds; it
is `#[non_exhaustive]`, so build values with [`WeightVal::exact`] and read
them with [`as_rational`], [`into_rational`], [`into_rational_opt`], or [`as_log`]
for the log-domain form; its variants are not constructible from outside the
crate, so the representation stays free to change.

### Shape and status

[`Tdd::size()`] is the total pair count, the size measure of the paper;
[`size_at_most(cap)`] answers the threshold question without counting past `cap`. [`node_count()`], [`max_width()`],
[`width_at(t)`], [`effective_width(t)`], [`is_zero()`], [`has_marginal_level()`], and
[`retired_marginal_slots()`] read the diagram's shape and state. [`is_sat_minimized(&f)`] is a
constant-time check on a minimized diagram; [`implied_literals(&f)`] returns
the literals true in every model of a minimized diagram;
[`reduced_size(&f, ReductionRule::R1Sdd)`] reports the size after the non-smooth reduction of
[tdd.md](tdd.md) without applying it.

## Marginal levels

[`marginalize(engine, &mut f, &levels)`] sums the named vtree levels out of the
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
[`ApplyError::Deadline`]; the levels summed out before the cut keep their
values.

Summing out a vtree leaf inlines the leaf's fixed count into its parent's
references by default, which is what makes the parent's two branches over that
variable twins for contraction; [`Engine::set_leaf_marginalize_inlines`] turns
that off for a caller that still needs to read the leaf's labels afterwards.

[`Tdd::has_marginal_level`] reports whether any level is marginal.

With a [`WeightStore`] attached, the same operation stores each node's
semiring value in the store instead of a count:

```rust
# use std::sync::Arc;
# use num_rational::BigRational;
# use tididi::{Engine, Tdd};
# use tididi::vtree::{Vtree, VtreeNode};
# let vtree = Arc::new(Vtree::balanced(4));
# let engine = Engine::new();
# let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
# let half = BigRational::new(1.into(), 2.into());
# let weights: Vec<(BigRational, BigRational)> =
#     (0..4).map(|_| (half.clone(), half.clone())).collect();
# // The root's left child: an internal level whose own children are leaves,
# // so it is the first level that may be summed out.
# let VtreeNode::Internal { left, .. } = *vtree.node(vtree.root()) else { unreachable!() };
# let levels = [left];
use tididi::diagram::RationalWeights;
use tididi::diagram::{Arithmetic, WeightStore};
use tididi::marginal::{marginalize, weighted_value};

let sr = RationalWeights::from_weights(&weights); // (w_neg, w_pos) per variable
f.set_weights(WeightStore::new(sr, Arithmetic::ExactRational));
marginalize(&engine, &mut f, &levels).unwrap();
let total = weighted_value(&f);                    // Option<WeightVal>
# assert!(total.is_some());
```

[`Arithmetic::ExactRational`] folds in [`BigRational`]; [`Arithmetic::SignedLog`] folds in the
bounded-precision [`SignedLog`] domain. Attach the store before the first
marginalize; a conjunction moves it to its result. [`Tdd::weights`] reads the store,
[`Tdd::take_weights`] detaches it, and [`WeightStore::level(t)`] reads a marginal
level's values. [`weighted_value`] folds whatever is still explicit above the
marginal levels and returns the diagram's value.

## Traversal contract

The stored encoding is the traversal contract. The [`diagram`] module
documentation states it: how each kind of level is read, and the invariants a
reader may rely on.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
use tididi::diagram::{ChildRef, ValueRef};

for (t, left, right) in f.vtree().internal_bottomup() {
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
count. [`Tdd::build(&eng, &vtree)`] opens a [`TddBuilder`], which appends levels bottom-up
and hands back the diagram from [`finish(output)`]; [`TddBuildError`] names what it
checks.

## Persistence

[`save_tdd(&f, path)`] writes the `.tdd` text format (a header, one `L` line
per leaf, one `I` line per stored node with its pairs, bottom-up), and
[`load_tdd(path, &vtree)`] reads it back. The format records the diagram, not
the vtree, so the reader takes the vtree it belongs to and validates the file
against it. [`tdd_to_dot(&f)`] and [`vtree_to_dot(&vtree, Some(&f))`] render
Graphviz DOT; the vtree render colors each internal node by its pair count.
All of these return `io::Result` or `Result<_, IoError>`, where [`IoError`] is
either an underlying [`std::io::Error`] or a [`Format`] message naming what the
file or diagram violated — a marginal level is refused that way.
[`Vtree::to_text()`] writes the `.vtree` format and [`Vtree::from_text()`]
reads it; [`Display`] and [`FromStr`] are the same two. `vtree_example.svg` and `tdd_example.svg` in this
directory are renders of one diagram.

## Restructuring

Rotating a bare vtree is not a public operation here: a rotation is only
meaningful against the diagram built over the vtree, and the levels have to be
relinked with it. On a compiled diagram,
[`rotation_search(&mut t, &mut objective, &config)`] rotates the vtree under the
diagram to a local minimum of any [`RotationObjective`]
([`delta(before, after) -> i64`], negative to accept — an objective that returns
the change in size descends to a size local minimum).
[`RotationSearchConfig`] bounds the rebuilt level size and the sweep count;
both entries return [`RotationSearchStats { probes, accepts, sweeps }`]. Each
rotation rewrites only the two affected levels and re-minimizes them, and
the model count is preserved.

[`engine.rotation_search(&mut t, &mut objective, &config)`] is the same search on
a caller's engine: it polls the armed stop once per pivot and returns
[`Err(ApplyError::Deadline)`] rather than running to the local minimum, leaving
the diagram canonical and count-correct wherever it stopped.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let mut t = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
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

[`ApplyError::Deadline`]: crate::ApplyError::Deadline
[`ApplyError`]: crate::ApplyError
[`ApplyMeters`]: crate::engine::ApplyMeters
[`Arc`]: std::sync::Arc
[`Arithmetic::ExactRational`]: crate::diagram::Arithmetic::ExactRational
[`Arithmetic::SignedLog`]: crate::diagram::Arithmetic::SignedLog
[`BatchMergeOutcome::Declined`]: crate::apply::BatchMergeOutcome::Declined
[`BatchMergeOutcome::Merged`]: crate::apply::BatchMergeOutcome::Merged
[`BigRational`]: num_rational::BigRational
[`BottomUpSubset`]: crate::vtree::BottomUpSubset
[`ContentTwinProbe`]: crate::reduce::ContentTwinProbe
[`Deadline`]: crate::ApplyError::Deadline
[`Display`]: std::fmt::Display
[`Engine`]: crate::Engine
[`Err(ApplyError::Deadline)`]: crate::ApplyError::Deadline
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`Evaluated`]: crate::query::Evaluated
[`Format`]: crate::io::IoError::Format
[`FromStr`]: std::str::FromStr
[`IncrementalCounter::new(eng, &f, n_pins, convention)`]: crate::query::IncrementalCounter::new
[`IncrementalCounter`]: crate::query::IncrementalCounter
[`Invalid`]: crate::vtree::VtreeError::Invalid
[`IoError`]: crate::io::IoError
[`KeepAllColumns`]: crate::query::KeepAllColumns
[`KeepFrontier`]: crate::query::KeepFrontier
[`LimitSet::deadline(Some(t))`]: crate::engine::LimitSet::deadline
[`LimitSet::uncut()`]: crate::engine::LimitSet::uncut
[`LimitSet`]: crate::engine::LimitSet
[`Literal::from(i32)`]: crate::Literal
[`Literal::neg`]: crate::Literal::neg
[`Literal::pos`]: crate::Literal::pos
[`Literal`]: crate::Literal
[`MemPressure::NONE`]: crate::engine::MemPressure::NONE
[`MemPressure`]: crate::engine::MemPressure
[`MergeProgress`]: crate::engine::MergeProgress
[`MinimizeOptions`]: crate::reduce::MinimizeOptions
[`MinimizeScope::{Full, PruneOnly, ContractOnly}`]: crate::reduce::MinimizeScope
[`OutputCap`]: crate::ApplyError::OutputCap
[`OverBudget`]: crate::ApplyError::OverBudget
[`OverlappingVariable`]: crate::vtree::VtreeError::OverlappingVariable
[`Projection::Automatic`]: crate::apply::Projection::Automatic
[`Projection::Structural`]: crate::apply::Projection::Structural
[`RationalWeights::unit(n)`]: crate::diagram::RationalWeights::unit
[`Restricted::Shrunk(g)`]: crate::apply::Restricted::Shrunk
[`Restricted::Unchanged`]: crate::apply::Restricted::Unchanged
[`Restricted::Unsatisfiable(⊥)`]: crate::apply::Restricted::Unsatisfiable
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`RotationSearchConfig`]: crate::restructure::search::RotationSearchConfig
[`RotationSearchStats { probes, accepts, sweeps }`]: crate::restructure::search::RotationSearchStats
[`Scheduled::Carry`]: crate::engine::Scheduled::Carry
[`Scheduled::Replace(stop)`]: crate::engine::Scheduled::Replace
[`Scheduled::Stop`]: crate::engine::Scheduled::Stop
[`SeedConvention::Fixed`]: crate::query::SeedConvention::Fixed
[`SeedConvention::Free`]: crate::query::SeedConvention::Free
[`SignedLog`]: crate::diagram::SignedLog
[`Stop::by(t)`]: crate::engine::Stop::by
[`StopAt::Wall`]: crate::engine::StopAt::Wall
[`StopAt::Work`]: crate::engine::StopAt::Work
[`Stop`]: crate::engine::Stop
[`Tdd::build(&eng, &vtree)`]: crate::Tdd::build
[`Tdd::clause`]: crate::Tdd::clause
[`Tdd::graft(parts, &spine_vars)`]: crate::Tdd::graft
[`Tdd::graft`]: crate::Tdd::graft
[`Engine::set_leaf_marginalize_inlines`]: crate::Engine::set_leaf_marginalize_inlines
[`Tdd::has_marginal_level`]: crate::Tdd::has_marginal_level
[`Tdd::model_count`]: crate::Tdd::model_count
[`Tdd::one`]: crate::Tdd::one
[`Tdd::size()`]: crate::Tdd::size
[`Tdd::size`]: crate::Tdd::size
[`Tdd::take_weights`]: crate::Tdd::take_weights
[`Tdd::weights`]: crate::Tdd::weights
[`Tdd::zero`]: crate::Tdd::zero
[`TddBuildError`]: crate::diagram::TddBuildError
[`TddBuilder`]: crate::diagram::TddBuilder
[`TddLevel`]: crate::diagram::TddLevel
[`Tdd`]: crate::Tdd
[`Text`]: crate::vtree::VtreeError::Text
[`VarId(0..n)`]: crate::vtree::VarId
[`VarId`]: crate::vtree::VarId
[`Vtree::balanced(n)`]: crate::Vtree::balanced
[`Vtree::balanced_over(&order)`]: crate::Vtree::balanced_over
[`Vtree::from_text()`]: crate::Vtree::from_text
[`Vtree::from_text`]: crate::Vtree::from_text
[`Vtree::graft`]: crate::Vtree::graft
[`Vtree::join(&l, &r)`]: crate::Vtree::join
[`Vtree::leaf(var)`]: crate::Vtree::leaf
[`Vtree::linear(n)`]: crate::Vtree::linear
[`Vtree::linear_over(&order)`]: crate::Vtree::linear_over
[`Vtree::random(n, seed)`]: crate::Vtree::random
[`Vtree::to_text()`]: crate::Vtree::to_text
[`VtreeError`]: crate::vtree::VtreeError
[`Vtree`]: crate::Vtree
[`WeightStore::level(t)`]: crate::diagram::WeightStore::level
[`WeightStore`]: crate::diagram::WeightStore
[`WeightVal::exact`]: crate::diagram::WeightVal::exact
[`WeightVal`]: crate::diagram::WeightVal
[`address_space_limit`]: crate::engine::MemPressure::address_space_limit
[`after`]: crate::engine::Stop::after
[`apply_and_clause(acc, &lits)`]: crate::apply::apply_and_clause
[`apply_and_clause`]: crate::apply::apply_and_clause
[`as_log`]: crate::diagram::WeightVal::as_log
[`as_rational`]: crate::diagram::WeightVal::as_rational
[`bottomup()`]: crate::Vtree::bottomup
[`budget`]: crate::engine::LimitSet::budget
[`children()`]: crate::Vtree::children
[`compute(eng, &f)`]: crate::query::IncrementalCounter::compute
[`condition_var(&f, x, value)`]: crate::apply::condition_var
[`condition_var`]: crate::apply::condition_var
[`condition_vars(&f, &vars, value)`]: crate::apply::condition_vars
[`condition_vars`]: crate::apply::condition_vars
[`deadline(None)`]: crate::engine::LimitSet::deadline
[`delta(before, after) -> i64`]: crate::restructure::search::RotationObjective::delta
[`diagram`]: crate::diagram
[`eager_reclaim`]: crate::engine::MemPressure::eager_reclaim
[`effective_width(t)`]: crate::Tdd::effective_width
[`engine.and(f, g)`]: crate::Engine::and
[`engine.and_batch(acc, batch, &levels)`]: crate::Engine::and_batch
[`engine.and_batch`]: crate::Engine::and_batch
[`engine.and_clause(acc, &lits)`]: crate::Engine::and_clause
[`engine.and_clause`]: crate::Engine::and_clause
[`engine.and_marginalizing(f, g, &targets)`]: crate::Engine::and_marginalizing
[`engine.and`]: crate::Engine::and
[`engine.clause`]: crate::Engine::clause
[`engine.condition_var`]: crate::Engine::condition_var
[`engine.cube`]: crate::Engine::cube
[`engine.limits().armed()`]: crate::engine::Limits::armed
[`engine.limits().install(set)`]: crate::engine::Limits::install
[`engine.limits().meters()`]: crate::engine::Limits::meters
[`engine.model_count(&f)`]: crate::Engine::model_count
[`engine.model_count`]: crate::Engine::model_count
[`engine.one`]: crate::Engine::one
[`engine.or(f, g)`]: crate::Engine::or
[`engine.or`]: crate::Engine::or
[`engine.project_var`]: crate::Engine::project_var
[`engine.reset()`]: crate::Engine::reset
[`engine.restrict`]: crate::Engine::restrict
[`engine.rotation_search(&mut t, &mut objective, &config)`]: crate::Engine::rotation_search
[`engine.rotation_search`]: crate::Engine::rotation_search
[`engine.zero`]: crate::Engine::zero
[`evaluate(&f, &sr)`]: crate::query::evaluate
[`evaluate`]: crate::query::evaluate
[`finish(output)`]: crate::diagram::TddBuilder::finish
[`from_rational`]: crate::diagram::SignedLog::from_rational
[`has_marginal_level()`]: crate::Tdd::has_marginal_level
[`implied_literals(&f)`]: crate::query::implied_literals
[`in_flight_bytes`]: crate::engine::ApplyMeters::in_flight_bytes
[`internal_bottomup()`]: crate::Vtree::internal_bottomup
[`into_rational_opt`]: crate::diagram::WeightVal::into_rational_opt
[`into_rational`]: crate::diagram::WeightVal::into_rational
[`into_tdd()`]: crate::apply::Restricted::into_tdd
[`is_sat_minimized(&f)`]: crate::query::is_sat_minimized
[`is_zero()`]: crate::Tdd::is_zero
[`lca()`]: crate::Vtree::lca
[`leaf_bottomup()`]: crate::Vtree::leaf_bottomup
[`leaf_of()`]: crate::Vtree::leaf_of
[`leaf_var()`]: crate::Vtree::leaf_var
[`load_tdd(path, &vtree)`]: crate::io::load_tdd
[`load_tdd`]: crate::io::load_tdd
[`mapped_bytes`]: crate::engine::MemPressure::mapped_bytes
[`marginalize(engine, &mut f, &levels)`]: crate::marginal::marginalize
[`marginalize`]: crate::marginal::marginalize
[`max_width()`]: crate::Tdd::max_width
[`mem_pressure`]: crate::engine::LimitSet::mem_pressure
[`merge`]: crate::engine::ApplyMeters::merge
[`minimize`]: crate::reduce::minimize
[`negate`]: crate::apply::negate
[`node()`]: crate::Vtree::node
[`node_count()`]: crate::Tdd::node_count
[`num_leaves()`]: crate::Vtree::num_leaves
[`num_vars()`]: crate::Vtree::num_vars
[`output_cap`]: crate::engine::LimitSet::output_cap
[`output_count(&f)`]: crate::query::IncrementalCounter::output_count
[`pairs_in_flight`]: crate::engine::ApplyMeters::pairs_in_flight
[`preflight_alloc`]: crate::engine::MemPressure::preflight_alloc
[`project_var(&f, x, how)`]: crate::apply::project_var
[`project_var`]: crate::apply::project_var
[`project_vars(&f, &vars, how)`]: crate::apply::project_vars
[`project_vars`]: crate::apply::project_vars
[`recompute_dirty(eng, &f, &levels)`]: crate::query::IncrementalCounter::recompute_dirty
[`reduced_size(&f, ReductionRule::R1Sdd)`]: crate::query::reduced_size
[`refused_reserve_bytes`]: crate::engine::ApplyMeters::refused_reserve_bytes
[`reset_meters()`]: crate::engine::Limits::reset_meters
[`restrict(f, care, CareCanonical::{Yes, No})`]: crate::apply::restrict
[`restrict`]: crate::apply::restrict
[`engine.restrict`]: crate::Engine::restrict
[`retired_marginal_slots()`]: crate::Tdd::retired_marginal_slots
[`root()`]: crate::Vtree::root
[`rotation_search(&mut t, &mut objective, &config)`]: crate::restructure::search::rotation_search
[`rotation_search`]: crate::restructure::search::rotation_search
[`same_tree()`]: crate::Vtree::same_tree
[`save_tdd(&f, path)`]: crate::io::save_tdd
[`save_tdd`]: crate::io::save_tdd
[`schedule`]: crate::engine::LimitSet::schedule
[`set_pin(var, Some(value))`]: crate::query::IncrementalCounter::set_pin
[`sibling()`]: crate::Vtree::sibling
[`size_at_most(cap)`]: crate::Tdd::size_at_most
[`std::error::Error`]: std::error::Error
[`std::io::Error`]: std::io::Error
[`stop`]: crate::engine::LimitSet::stop
[`tdd_to_dot(&f)`]: crate::io::tdd_to_dot
[`tdd_to_dot`]: crate::io::tdd_to_dot
[`try_minimize`]: crate::reduce::try_minimize
[`validate()`]: crate::Vtree::validate
[`vtree.bottom_up_subset(...)`]: crate::Vtree::bottom_up_subset
[`vtree_to_dot(&vtree, Some(&f))`]: crate::io::vtree_to_dot
[`wall`]: crate::engine::Stop::wall
[`watch`]: crate::engine::LimitSet::watch
[`weighted_value`]: crate::marginal::weighted_value
[`width_at(t)`]: crate::Tdd::width_at
[`work_units`]: crate::engine::ApplyMeters::work_units
