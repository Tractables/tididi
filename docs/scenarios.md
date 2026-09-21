# Shared teaching scenarios

These scripts describe the model and teaching sequence behind the Rust and
Python documentation. Each lists its guides, runnable programs and figures.
Hidden `scenario:` comments in those files point back here.

When editing an instance, update its scenario as needed and review every listed
instance. Keep language-specific code and explanations natural; record intentional
differences here. Run `python3 scripts/check_scenarios.py --review <scenario-id>`
after that review, then run the checker without arguments. It records content
fingerprints in hidden comments below; a change to an instance or its scenario
requires review again. It checks links and review freshness, not semantic truth;
executed examples and human review still establish that.

For Rust API source files, only doc comments enter the fingerprint; changing
runtime code alone does not invalidate a documentation review. The generated
navigation in `src/guide.rs` is checked as a whole. Standalone lessons are
automatically discovered, so a new page cannot silently omit its scenario.

## First circuit

Introduce circuits as Boolean functions that can be combined and queried
without enumerating assignments. Build `(x ∧ y) ∨ z` on a balanced vtree over
variables 1, 2 and 3, then count: four models have z true and one more has
x and y true, giving 5. Its complement has 3 models. Explain the vtree and
counting universe where they first appear, then explicit copies for reuse.

Rust uses `Tdd`, `!` and `clone()`; Python uses `Circuit`, `~` and `copy()`.
Rust's checked calls return `Result`; Python operators raise exceptions.
Keep installation instructions specific to the language and publication state.

### Instances

- [Rust README](../README.md) <!-- reviewed: e180a3b9533c7519063a76866d2f032e4cb7f46c12ff57e28674015f9735e994 -->
- [Rust crate introduction](../src/lib.rs) <!-- reviewed: aa1145e31d14be778e9f79a73deefe543341203a5631d4c0608750c2e5d0e7c6 -->
- [Python README](../bindings/python/README.rst) <!-- reviewed: c1cb68756607b2850e7bf9433faa319bb1219b21c2da5d7ebde94465e1e51b67 -->
- [Python getting started](../bindings/python/docs/getting_started.rst) <!-- reviewed: b7b67a6af0114e3f28b50731fc9fdbddb12d78b142d58cedb977f7f57f26103a -->

## Reading order

Lead with what a reader can build and query, then connect capabilities to
applications. The worked examples progress through configurations,
probabilities, reachability, tables, persistence, vtree grouping, execution
limits, minimum costs and storage statistics. Keep the index and sidebar in
that order. Link reference material separately; do not make beginners choose
an API category before they have seen a circuit.

### Instances

- [Rust guide and example navigation](../src/guide.rs) <!-- reviewed: 17795aa04c6fbd91188af399fffa242441bbbe0ecce2fd9b6b44a6ad3c6feb58 -->
- [Python guide index](../bindings/python/docs/index.rst) <!-- reviewed: bd95b43d37e627891d9149f842387051e2b7d99596d30f3229784610bd8bbcf5 -->
- [Python gallery introduction](../bindings/python/examples/GALLERY_HEADER.rst) <!-- reviewed: 5670a051f6c14a681d384f870bbed2b7caefda7ea0ec7230a8f2513337c2430f -->

## Configurations

Model four backup options: local L, remote R, encryption E and notifications N.
The rule is `(L ∨ R) ∧ (¬R ∨ E)`; N is free. Build named literal values and
circuits, introduce Boolean operators, then count and inspect user choices.
There are 8 configurations, 4 with notifications, and 4 with remote backups.
Remote forces encryption; remote without encryption is unsatisfiable.

A witness describes one complete valid assignment; conjoining its cube leaves
1 configuration. Observe R, then ¬N, then replace R and E by ¬R and ¬E:
the counter reports 4, 2, 1; clearing observations restores 8. Minimization
preserves those models. Do not require one particular witness or storage ID.

Rust's counter borrows its diagram; Python's owns it until `finish()`.
Teach checked calls after operators in Rust. Python operators already raise
exceptions. The Rust program also supplies the execution-limits lesson.

### Instances

- [Rust walkthrough](examples/configurations.md) <!-- reviewed: 7dc8c3a3accf2084eb53e2d7008429907749fb821bfc31489294e81b3b5a7647 -->
- [Rust program](../examples/build_minimize_count.rs) <!-- reviewed: 814befb94204e5fecf0da2b4cdc2b79c146b38e078765a659b942c8daf120b39 -->
- [Python walkthrough and program](../bindings/python/examples/01_configurations.py) <!-- reviewed: f3880bd2807224bcdbe555018f79d97f508e0fe5c2121a281803dc6b9bd68fd5 -->

## Probability

Rain R or sprinkler S makes the ground wet: `W = R ∨ S`. Wind is a third,
free variable. Build W and R ∧ W once. Use independent priors with P(S)=1/10,
P(wind)=2/5, and P(R)=1/5 followed by 3/5. Each variable's false/true weights
sum to one; a free variable therefore contributes one.

Evaluate exact fractions. The two scenarios give P(W)=7/25 and 16/25,
and P(R|W)=5/7 and 15/16. Evidence has positive mass in both. Introduce the
conditional-probability formula before the calculation; reuse the circuits
when weights change. Unit weights give 6 models of W over all three variables.
Rust uses BigRational; Python uses the standard-library Fraction.

### Instances

- [Rust walkthrough](examples/probability.md) <!-- reviewed: cb40fb5e14fb8a0f7c429327764090ebf1fa0eab1167f35888648bb0ff709366 -->
- [Rust program](../examples/probabilistic_query.rs) <!-- reviewed: 4e17e03373c7f4f03043eb75f8f760d01730622a8b8a9dc031d2c93de21f427b -->
- [Python walkthrough and program](../bindings/python/examples/02_probability.py) <!-- reviewed: 6f442fa45ef018a70a4c2c3ffec4767153a743d08d385720982975f0c6998de4 -->

## Reachability

Start at node 0 in a directed graph with nodes 0 through 15. Use one-hot
state indicators, current variables 1–16 and next variables 17–32, with a
balanced vtree. State i sets its indicator and negates the other 15 indicators.
The relation is `T(x,x′) = ⋁(Aᵢ(x) ∧ Aⱼ(x′))` over these 25 edges:

```text
(0,1) (1,2) (3,2) (4,5) (5,6) (6,7) (8,9) (9,10) (10,11)
(0,4) (1,5) (2,6) (3,7) (4,8) (5,9) (6,10) (7,11)
(4,0) (5,1) (10,6) (11,7) (12,13) (13,15) (15,14) (14,12)
```

Show the graph and formulas before code. Construct state indicators and the
relation programmatically. First conjoin, quantify current variables, and
rename next variables back to current ones. Then introduce `and_exists` as
conjunction and quantification combined. The first image contains nodes 1 and 4.

Union reached states with their successors until semantic equivalence holds.
Projected state counts are 3, 6, 8, 10, 11, 11. Report nodes and pairs too;
the final two iterations have equal storage sizes. Exact sizes are observed
implementation results, not a promise about the representation.

Node 3 points into the grid but has no incoming edge; 12–15 form a separate
component. All five are unreachable. A target query finds node 11. Its witness
is a state assignment, not a path. Ordinary model counts also include free
next-state variables; use projection for the number of states.

### Instances

- [Rust walkthrough](examples/reachability.md) <!-- reviewed: ed07ea316ae735f30c1b7ac20967afe465ae5be40b0a8ed7728de09cc1f1ddc5 -->
- [Rust program](../examples/symbolic_reachability.rs) <!-- reviewed: 71aa0435ef7d54a050186df2995e93cffaecb31534be9a65f159538b4123ff58 -->
- [Python walkthrough and program](../bindings/python/examples/03_reachability.py) <!-- reviewed: ef05ed7b9e779c4af2f463393005de5b6cfdffbe1df2d233436810e58e6d83f5 -->
- [Shared directed graph](reachability.svg) <!-- reviewed: 156e5685631817dd39d6b34389cbd1d76dda693bc141ea1de06846184f2d4a6d -->

## Tables

Model read, write and share permissions with variables 1, 2 and 3. Load rows
100, 110, 101, 110; duplicates describe one assignment, so the count is 3.
Insert 111 and remove 110: the count stays 3, and requiring share leaves 2.
Explain column order, free omitted variables, and signed-literal updates.
A partial cube edits all matching assignments; an empty cube matches all.

Rust uses packed rows and a maintenance batch that edits a diagram in place.
Python accepts Boolean rows and `update(insert=..., remove=...)`, consuming
the old wrapper and returning its minimized replacement. The language-specific
storage and ownership explanations must match these different interfaces.

### Instances

- [Rust walkthrough](examples/tables.md) <!-- reviewed: 95ef30a3554a345ca48d7806b7dc0daa86b220e12620176df5028546f63bad31 -->
- [Rust program](../examples/table_updates.rs) <!-- reviewed: 07c298527ea7664fcbe0756273e58c7ce35fc6d1a3ec361b4f2ddf1a9c4b039f -->
- [Python walkthrough and program](../bindings/python/examples/04_tables.py) <!-- reviewed: ef2b854203464afb300d8196c835452452d088edddb5b6234ced503fb4a4e17a -->

## Persistence

Use three backup variables: local, remote and encryption. Construct the two
rules L ∨ R and ¬R ∨ E separately. Serialize one vtree and both circuits,
restore one shared vtree object, and load both circuits onto it. Conjoining
the restored circuits gives 4 configurations and agrees with freshly built
rules on that same restored vtree.

Explain why separately reconstructed vtrees are not interchangeable domains.
Bytes and vtree text can be stored independently. Rust demonstrates readers
and writers; Python demonstrates strings and bytes and points to path helpers.

### Instances

- [Rust walkthrough](examples/persistence.md) <!-- reviewed: 2ef400c0cb389f79a174fce970c323a727c239a0766aad79787212480a78b79a -->
- [Rust program](../examples/save_reload.rs) <!-- reviewed: f8a6b1df8dc3e9efc6c194d1f5cefadee649b37ef5db79e2cb89ad0e708d71da -->
- [Python walkthrough and program](../bindings/python/examples/05_persistence.py) <!-- reviewed: 3b04acd5e833af63df54b62087672e0e54765e7bafd59d0db9e68ae7fff9996e -->

## Vtrees

Compare `(x₁ ↔ x₃) ∧ (x₂ ↔ x₄)` under balanced leaf orders [1,3,2,4]
and [1,2,3,4]. The first groups each equality together; the second separates
its variables. Both functions have 4 models, while minimized pair counts are
5 and 12 in the demonstrated representation. Show both groupings in a figure
and construct the same function for each. Introduce join and linear vtrees
only after the comparison. Explain size as a consequence of grouping, without
claiming that any heuristic guarantees small circuits.

### Instances

- [Rust walkthrough](examples/vtrees.md) <!-- reviewed: 5620d581df647cdad5f226ff679c68fd71b736dfa79a7d8acc229b319ba3f98b -->
- [Rust program](../examples/vtree_grouping.rs) <!-- reviewed: 79c4a7b229fb91c3535e73082ba99250e070fd50b04eedf1a49e8d7b17d89f57 -->
- [Python walkthrough and program](../bindings/python/examples/06_vtrees.py) <!-- reviewed: ee729b8a4257f4ac57677740eb960c4af8914fbcdce60d99c0136228b0c0bd8e -->
- [Shared grouping figure](vtree-grouping.svg) <!-- reviewed: b55861dc4606d7267949e4b2a7ef5b53c5a4bdca8d5d720293e0badc7840646c -->

## Execution

Continue the four-variable backup model. A zero-byte operation budget refuses
construction of L ∨ R; a later unrestricted call succeeds and counts 12.
Limits do not remain installed on unrelated calls. Show a bounded query and
releasing idle scratch while keeping circuits usable. Budgets cover charged
operation storage, not the entire process; deadlines are polled cooperatively.

Rust uses a context-supplied engine for bounded batches, minimizes the complete
backup rule, and counts its 8 models. Python uses per-call `limits=`, shows
copies preserving operands for retry, and counts 6 models of `(L ∨ R) ∧ E`.
These query counts differ because the examples deliberately constrain different
functions. Preserve that distinction when editing either lesson.

### Instances

- [Rust walkthrough](examples/execution.md) <!-- reviewed: bffcba21a82218dbfd12d65eb545f490d108eef4e221c6dd6932a4954405454d -->
- [Rust program](../examples/build_minimize_count.rs) <!-- reviewed: d6bd5453066abf1ee6bf17f6c8abc3d251c1091e16957d863a3f63d178630708 -->
- [Python walkthrough and program](../bindings/python/examples/07_execution.py) <!-- reviewed: 05879013dc86c3c1f9c5a024264a1b89226d4410a7cd014f3024d28b2afba3ad -->

## Minimum cost

Reuse the [backup configuration model](#configurations). Enabling local,
remote and encryption costs 5, 2 and 1; notifications and disabled options
cost zero. Explain the algebra: false is infeasible, OR chooses the cheaper
alternative, and AND adds costs over disjoint variable groups. A free variable
chooses its cheaper setting. Define the algebra immediately before using it.

The cheapest valid configuration costs 3. Discount local storage to 1 and the
minimum becomes 1; requiring remote still costs 3. Remote without encryption
is infeasible. Rust represents infeasibility with None and feasible costs
with Some; Python uses infinity. Python callbacks must not mutate their
arguments. Evaluation borrows circuits, so changing prices needs no rebuild.

### Instances

- [Rust walkthrough](examples/optimization.md) <!-- reviewed: a4fa4a960687d63fb6ffc8c8dd071cf5cb7c4f3e70f7a3f0f675ccc4353a6cbd -->
- [Rust program](../examples/minimum_cost.rs) <!-- reviewed: 140f72939fa780ee26c9e59c3b7df3bb96f1d07940d4c3786c692ea83a7144c4 -->
- [Python walkthrough and program](../bindings/python/examples/08_minimum_cost.py) <!-- reviewed: 6aefd6fc047e39d986bf5ca5ec3a434f130af708b7f6b48734550ee29ce6d065 -->

## Statistics

Construct x₁ XOR x₂ over a balanced four-variable vtree and compare it with
literal x₁. Find the stored internal node with the most pairs: XOR has a
maximum of 2; a literal has 1. Distinguish per-node width, total pair count,
and model count (8 for XOR with the other two variables free).

Rust traverses level storage directly and uses (root,0) for an empty traversal.
Python traverses a `node_sizes()` snapshot and uses None when it is empty.
Storage IDs and the chosen maximum on ties are not semantic identities.
Python may use its XOR operator while Rust demonstrates its Boolean expansion.

### Instances

- [Rust walkthrough](examples/statistics.md) <!-- reviewed: 364a391f218e32e5f60c0bd16547aaa373d8bc529f01c683d95f5fb341b30011 -->
- [Rust program](../examples/statistic.rs) <!-- reviewed: 21f033fba1922a0b58288c80979efdd2cf741182d17c0e3a376cce88bbe7caf6 -->
- [Python walkthrough and program](../bindings/python/examples/09_statistics.py) <!-- reviewed: ed8e5506d68db2c74cb39af3ec8c8f0e92e051d437eb6a7f084b6292bb965871 -->

## Ownership

Circuits own their diagram storage and share their vtree. Queries borrow;
transformations whose inputs are owned consume them. Copies duplicate storage
and share the vtree. Introduce copying as providing an operand while retaining
the original; do not suggest that assignment creates a copy.

Rust enforces moves statically and documents in-place methods through mutable
references. Python retains a wrapper marked consumed, raises ConsumedCircuitError
on reuse, and rejects duplicate consuming operands before taking anything.
Its counter owns the diagram until finish(); Rust's counter borrows it.
Python documents aliasing, retry after execution failure, and caller threads;
these mechanisms need no artificial Rust equivalent.

### Instances

- [Rust circuit documentation](../src/diagram/tdd/mod.rs) <!-- reviewed: 37042a15836ba02e5eb38f7eff75dd9c9f87e7ca154230b88b0b007a41e4fc16 -->
- [Python ownership guide](../bindings/python/docs/ownership.rst) <!-- reviewed: 00fc7987d60bef7cc0c71ec744439043860ba74d07d02c82002701d123aba310 -->

## API overview

Group operations by what a reader wants to do: construct and combine rules,
query solutions, condition/quantify/rename, evaluate weights or costs, save and
inspect, and control representation or resource use. Link the relevant
application and authoritative API specification rather than copying contracts.

The Python reference is generated from binding docstrings. The Rust overview
also links specialized operations not yet exposed in Python. An operation
appearing in one reference does not imply that the other language exposes it.

### Instances

- [Rust API overview](api-guide.md) <!-- reviewed: 2f528a2a54100bc71caece17a90538c60bcd4cd1f839fc814f46db31379cec0b -->
- [Python API reference](../bindings/python/docs/api.rst) <!-- reviewed: a2fa21a16b4f8d4ded0682a1200f27bbc01cd2ff83c9ea5305eb5ee030c8382f -->

## Representation

Explain the vtree before levels and pairs. Each pair conjoins disjoint left
and right variable groups; a node disjoins its alternatives. Use the two-leaf
XOR example `(x ∧ ¬y) ∨ (¬x ∧ y)` and its figure. Structural determinism makes
alternatives disjoint, which permits counting by combining their counts.

The vtree defines the whole counting universe, including unconstrained
variables. Minimization is canonical for a fixed vtree, up to storage numbering;
semantic equivalence is the appropriate comparison. Circuit nodes and pairs
measure stored structure, not models. Rust additionally explains representation
invariants and marginal levels; Python concentrates on using shared vtrees.

### Instances

- [Rust data model](tdd.md) <!-- reviewed: a88d58885452192b8bbbed7178bbb2d81af322e5217d371e4d1faf691cbf8ae3 -->
- [Python data model](../bindings/python/docs/representation.rst) <!-- reviewed: ed52b71f88c82caf64b54c6227acbe5de4215ce34886e327f1b102cdd338b20c -->
- [Shared decomposition figure](tdd-basics.svg) <!-- reviewed: 5f2ab1677522a6a53385289841534c2a0096087bc65c8ce7f6c51d3be0e83c12 -->

## Architecture

Explain the Rust implementation through storage ownership, level arenas,
child-reference decoding, reduction passes, scratch and limits. Define terms
once, map modules to responsibilities, and identify which operations establish
or temporarily break each invariant. Distinguish validation of stored data
from proving structural determinism. This is a Rust implementation reference;
there is no separate Python implementation to describe.

### Instances

- [Rust architecture reference](architecture.md) <!-- reviewed: c9fd514f97aad9abed3618f51f21cdea0db569025309fb4826bc167895976b90 -->
