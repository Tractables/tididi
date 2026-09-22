# Shared teaching scenarios

These scripts describe the model and teaching sequence behind the Rust, Python and
C documentation. Each lists its guides, runnable programs and figures.
Hidden `scenario:` comments in those files point back here.

When editing an instance, update its scenario as needed and review every listed
instance. Keep language-specific code and explanations natural; record intentional
differences here. Run `python3 scripts/check_scenarios.py --review <scenario-id>`
after that review, then run the checker without arguments. It records content
fingerprints in hidden comments below; a change to an instance or its scenario
requires review again. It checks links and review freshness, not semantic truth;
executed examples and human review still establish that.

For Rust API source files, including binding implementations, only doc comments
enter the fingerprint; changing
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

C uses opaque handles, output slots initialized to NULL, and explicit error and
cleanup functions. Its shared example helper wraps checking, copying, counting
and freeing without changing their ownership rules.

### Instances

- [Rust README](../README.md) <!-- reviewed: ff8cb955dded726fc9de884c0236de0c6511aa162eac4bb0f39c3b03c05ed7e4 -->
- [Rust crate introduction](../src/lib.rs) <!-- reviewed: 9c14ea28438b319153d89020cbb32df4b6f0c1757d839ddf5465874e73f3be2a -->
- [Python README](../bindings/python/README.rst) <!-- reviewed: 42cb63be6010d5beef49526d85187050c634956301696846c96e5d5f4708357b -->
- [Python getting started](../bindings/python/docs/getting_started.rst) <!-- reviewed: c6ead730e1436ba112d8bc518610d932e3f72180e516b274c1daff51c24f651a -->
- [C getting started](../bindings/c/docs/getting_started.rst) <!-- reviewed: fbcc2bf0db46b0e1845f76c8fe9f00f3675645dc41f8b80beb905da887ecac25 -->
- [C program first_circuit](../bindings/c/examples/first_circuit.c) <!-- reviewed: 3039126c0a5c738421a4eedf9848b5401d3e1657eefa9f2ec8e19f7e794a3894 -->
- [C example](../bindings/c/examples/example.h) <!-- reviewed: 1019207b1a680a883f8df2e9b09bb850465a9a4f8b0fd345169d89200f299a70 -->
- [C README](../bindings/c/README.rst) <!-- reviewed: c2505b3c85690814ce3c911401bb3554ffff1a9ec08b4bff9cd813e029a7b987 -->

## Reading order

Lead with what a reader can build and query, then connect capabilities to
applications. The worked examples progress through configurations,
counting choices, probabilities, reachability, tables, persistence, vtree grouping, execution
limits, minimum costs and storage statistics. Rust adds reusable components
after vtree grouping; embedding is currently exposed only in Rust. Keep the
index and sidebar in that order. Link reference material separately; do not make beginners choose
an API category before they have seen a circuit.

The counting lesson follows configurations in every language. Python uses a
short doctested page for this comparison instead of a gallery application.

The C guide adds a first-circuit chapter and places its ownership chapter before
resource limits; the application chapters retain the same order.

### Instances

- [Rust guide and example navigation](../src/guide.rs) <!-- reviewed: 99b9fb7554b6c096d6c86626988cf74f6529cb87a9c18fc2d9411fa23b6522e5 -->
- [Python guide index](../bindings/python/docs/index.rst) <!-- reviewed: f41367362e458fe17f8c608e980ef6cdf7169a0be66b6c616e99613069d1c2ac -->
- [Python gallery introduction](../bindings/python/examples/GALLERY_HEADER.rst) <!-- reviewed: dfda32e6316ede7503e373ad4274201737d32aef4c092c1c5efb4674a4a9f665 -->
- [C index](../bindings/c/docs/index.rst) <!-- reviewed: 4650f2722c043bcb88c47bad60d7266460fa75e9a854db83a6445aded68cbb1d -->

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

C expresses clauses through named signed integer constants. Its shorter lesson
shows counts 8 and 4, forced remote/encryption, then counter counts 4, 2 and 8.
It omits the witness, replacement observation and minimization demonstrations.

### Instances

- [Rust walkthrough](examples/configurations.md) <!-- reviewed: 1d1e2eede05e5159c8bbd9ee4c33a80c11015361604f13d7423b841d11c71bb7 -->
- [Rust program](../examples/build_minimize_count.rs) <!-- reviewed: 8608014d21d1284454cc9220ee95b049aeb15d0b1e529de69df7ffe98f3e0b7d -->
- [Python walkthrough and program](../bindings/python/examples/01_configurations.py) <!-- reviewed: 0aa18e286e024c73704ecdfa5c9716bdd0f044119deb6baf6f6291add69d4993 -->
- [C configurations](../bindings/c/docs/configurations.rst) <!-- reviewed: afa91c98a64dec891370050699f07dc3d2aaa8ede6f07039aba5ff676c090e3b -->
- [C program configurations](../bindings/c/examples/configurations.c) <!-- reviewed: 3a3557b4369cf8a7ffa7f2252570c4a695940095870e69e476a4ebef24e35c54 -->

## Counting choices

Use the four-option backup rule `(L ∨ R) ∧ (¬R ∨ E)`, with N free.
It has 8 configurations. Conjoining R leaves 4; observing R in a counter
also gives 4. Substituting R=true yields E, whose ordinary count is 8 because
L, R and N are free in the unchanged vtree. Projecting that residual onto
L, E and N yields the 4 distinct remaining choices.

The original rules projected onto L,R have 3 choices: local only, remote only,
or both. Quantifying E,N produces L∨R. Its full-vtree count is 12; projecting
onto L,R returns 3. Explain each changed Boolean function before its code and
show output beside it. Projection takes variables to keep; exists takes
variables to eliminate. No operation here shrinks the vtree. The conditioning contract agrees in all
languages: repeat assignments are ignored, contradictions produce false, and
invalid variables are still errors even in a contradictory assignment.

Rust and C use standalone programs with checked excerpts. Python uses a short
interactive doctested lesson. Counters borrow the circuit in Rust and own an
explicit copy in Python/C. All queries have identical counts across languages.

### Instances

- [Rust walkthrough](examples/counting.md) <!-- reviewed: 6df899ab956bc8bab21e39f81a71435648f1494cc5add94ee1d45f0c0eaa6f34 -->
- [Rust program](../examples/counting_choices.rs) <!-- reviewed: bb1cf3ae7ed7e7982e1cb32090d196796737e81b378da3c1e6e53d086726687f -->
- [Python lesson](../bindings/python/docs/counting.rst) <!-- reviewed: 240a243498376cfae2200bd33b3900fbba718b34db9917317dfee8f8d1b1ca0b -->
- [C walkthrough](../bindings/c/docs/counting.rst) <!-- reviewed: 7c8e8e8daff3c97e986125d470a0efed7dd42464b01e5f21293555639ce1d0ac -->
- [C program](../bindings/c/examples/counting.c) <!-- reviewed: 4f2d04d3f0373f4b26b264100f83ab71f183d579d5027b1ee81d9bff738038af -->

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

C passes exact weights as fraction strings and returns owned fraction strings.
Its weighted-ratio call divides the two event masses; it omits the unit-weight
comparison to keep arithmetic and allocation details out of the narrative.

### Instances

- [Rust walkthrough](examples/probability.md) <!-- reviewed: 90d6c93d0927c51e41a32bd036ecae851af5393d3451d1d8ccd1fac128a16d04 -->
- [Rust program](../examples/probabilistic_query.rs) <!-- reviewed: d22223b12cf6725aa5c100ba833eb3f3cdb4388ba64d6e5640d67c309c30fa3f -->
- [Python walkthrough and program](../bindings/python/examples/02_probability.py) <!-- reviewed: 43ebc3fefabcb732dbad5b04b78a9171aded6421a1a2a4f7eaf05bf9c81ada86 -->
- [C probability](../bindings/c/docs/probability.rst) <!-- reviewed: 5d30ac6f3b39409e65615580882a2bd742bbaf1c3da847116dde7ed8237d4931 -->
- [C program probability](../bindings/c/examples/probability.c) <!-- reviewed: f63e88702fd6a8afe3aa25132c2fe78197512554315d01ca33ef391454c3ee15 -->

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

C uses the same graph, formula and fixed-point sequence, with explicit release
of consumed handles. It lists unreachable nodes instead of printing a target
witness; the reachable-set semantics are unchanged.

### Instances

- [Rust walkthrough](examples/reachability.md) <!-- reviewed: e1bccd77d6ae7f2a945c071fab535bae147d0052350f6752a11f606ac93ed968 -->
- [Rust program](../examples/symbolic_reachability.rs) <!-- reviewed: 1256ee4c4e8f749de730a6e94e724db978b87c9cb976d7b56cdc8375bdfd2cd2 -->
- [Python walkthrough and program](../bindings/python/examples/03_reachability.py) <!-- reviewed: 959d0a9aea4d72b273a51ff60a2124ed1740a3ca0531376773705f423e8755ad -->
- [Shared directed graph](reachability.svg) <!-- reviewed: b578209ba2f5bf62b32b3ba8100d7efefe03ccb8f10745da60d5af30cda85cdf -->
- [C reachability](../bindings/c/docs/reachability.rst) <!-- reviewed: 2399b28271eb37b09b22d0955b4fefec4fdd910a5c92cc79f9e9727439ab77b4 -->
- [C program reachability](../bindings/c/examples/reachability.c) <!-- reviewed: 8bf234fbfbed5a58bf1f034478670ddb75efea33ed66112b311ae91b851699b4 -->

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

C takes row-major Boolean bytes and arrays of signed-literal cubes. Like Python,
its update consumes the old circuit and returns a minimized replacement.

### Instances

- [Rust walkthrough](examples/tables.md) <!-- reviewed: f8919277261cd75a8f3275df82ae8a659b281a971c8d7e16829a85b6136dbc19 -->
- [Rust program](../examples/table_updates.rs) <!-- reviewed: 3dfb36347e323a687c86e6d0bf1eb19955e6d305f24f6eb7b9520c2d8dd3d973 -->
- [Python walkthrough and program](../bindings/python/examples/04_tables.py) <!-- reviewed: 332bfeafd76deb93ec93dfb50d94d0b3b0d7a6ec71e85a2eea2433205b0128f1 -->
- [C tables](../bindings/c/docs/tables.rst) <!-- reviewed: b1eb59b11462e80ade4b6474c01f84bf8c61163ffa67dccbc76e0928cfc50b45 -->
- [C program tables](../bindings/c/examples/tables.c) <!-- reviewed: 1d1951432101bcb405b36cdea1285704f0e91fd0979bc72c64feb4fd63b6b308 -->

## Persistence

Use three backup variables: local, remote and encryption. Construct the two
rules L ∨ R and ¬R ∨ E separately. Serialize one vtree and both circuits,
restore one shared vtree object, and load both circuits onto it. Conjoining
the restored circuits gives 4 configurations and agrees with freshly built
rules on that same restored vtree.

Explain why separately reconstructed vtrees are not interchangeable domains.
Bytes and vtree text can be stored independently. Rust demonstrates readers
and writers; Python demonstrates strings and bytes and points to path helpers.

C returns owned byte buffers and UTF-8 vtree text, leaving file I/O to the
application. The tutorial restores both rules and counts their conjunction;
equivalence and malformed input are checked separately by the C consumer tests.

### Instances

- [Rust walkthrough](examples/persistence.md) <!-- reviewed: 8c3005e5acccf6749cd7233b42802f698448dd1d90de2bf99fb05eb9510b018d -->
- [Rust program](../examples/save_reload.rs) <!-- reviewed: a17fb99fe7fa1d3faf36f7137fac700566ae2a417761a49e08c7353947f51ee5 -->
- [Python walkthrough and program](../bindings/python/examples/05_persistence.py) <!-- reviewed: 2495a36faff122077a3e3a4135ec0f4ee83b46c86c2c05c920835b5766c85d42 -->
- [C persistence](../bindings/c/docs/persistence.rst) <!-- reviewed: 89fc3a373f111bc21b4d2489890849bc6c8dbfd9e00b40267721c75de23c3bd0 -->
- [C program persistence](../bindings/c/examples/persistence.c) <!-- reviewed: 3874298127872ef693e984c146532ff7c381b857fcb7825c2ef7fbe5772c649e -->

## Vtrees

Compare `(x₁ ↔ x₃) ∧ (x₂ ↔ x₄)` under balanced leaf orders [1,3,2,4]
and [1,2,3,4]. The first groups each equality together; the second separates
its variables. Both functions have 4 models, while minimized pair counts are
5 and 12 in the demonstrated representation. Show both groupings in a figure
and construct the same function for each. Introduce join and linear vtrees
only after the comparison. Explain size as a consequence of grouping, without
claiming that any heuristic guarantees small circuits.

C focuses on the two balanced groupings. Its reference documents join and linear
constructors without adding another construction to the introductory comparison.

### Instances

- [Rust walkthrough](examples/vtrees.md) <!-- reviewed: 3fc349d63f3cfd1c7a88123c21205a6eab6ed4496e5bb62efd495ed5b991c705 -->
- [Rust program](../examples/vtree_grouping.rs) <!-- reviewed: 72aba010a8335d636e15482c780aed00ffb2e416adb376b4b80aeb7abad1e058 -->
- [Python walkthrough and program](../bindings/python/examples/06_vtrees.py) <!-- reviewed: 6123259fa237f8a8c8430b51f61cdd913f5c6a0b820061dfcc5c2624d12e0610 -->
- [Shared grouping figure](vtree-grouping.svg) <!-- reviewed: aa5a790a0ab8d92ed87f6fcf143fe99f96ca68bc5db321bf2645b621a596c356 -->
- [C vtrees](../bindings/c/docs/vtrees.rst) <!-- reviewed: 8c08faf87c6e40a63d44c30825b6cd19238d7f398906b0f472840468910ef085 -->
- [C program vtrees](../bindings/c/examples/vtrees.c) <!-- reviewed: 18e0b69174b5fe04d81e9c5b27a2ec6c3e51093a23851d8d4643340fdf9d5b3e -->

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

C demonstrates the same retry and six-model function as Python, using a
TididiLimits pointer and an owned error. It goes directly to the consuming
operation failure rather than repeating the constructor-budget demonstration.

### Instances

- [Rust walkthrough](examples/execution.md) <!-- reviewed: 985edbc1da2ce69cb439e0fb9cd1457106ae8a01eb9bcd1c0c507b942797b1e5 -->
- [Rust program](../examples/build_minimize_count.rs) <!-- reviewed: 62e42d9dde24b58fd50b01a19d13c6af7518793736328cb2d1257ce34679ef3b -->
- [Python walkthrough and program](../bindings/python/examples/07_execution.py) <!-- reviewed: 9e28b5649d1aae35713b715fce2114e670ee34bd98b02e56ce33b160888d46e2 -->
- [C execution](../bindings/c/docs/execution.rst) <!-- reviewed: b68e839a2b3f124840b656b65fb9ccd20218c4f5c79eb740da2de26ea64b39eb -->
- [C program execution](../bindings/c/examples/execution.c) <!-- reviewed: 4bd5a512dad7d406d0ed60ef382d3900dfff386fbd53efcd3f649b1baecde176 -->

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

Every language documents the algebraic laws and the free-variable value on its
API item. A free leaf combines both signs; it is not automatically a unit value.
Floating-point calculations can approximate these laws but exact rational
weights have their own evaluation path.

C uses a double-valued callback table and infinity for infeasibility. It shows
minimum costs 3 and 1 and the remote-without-encryption conflict; the remote-only
comparison remains in the Rust and Python lessons.

### Instances

- [Rust walkthrough](examples/optimization.md) <!-- reviewed: b840639bfb25f1c5816b73da3b9090f010efe0ba738fc54b7eed4229ae96a627 -->
- [Rust program](../examples/minimum_cost.rs) <!-- reviewed: f3acca4df6cfe8655f3bd1180b79d3e42e7d6c1183f32bcb5d0079041f9d56be -->
- [Python walkthrough and program](../bindings/python/examples/08_minimum_cost.py) <!-- reviewed: 79dce3065ddab786f40085239b1e037fa83dbb734744bb7f262e6184f5e49638 -->
- [C minimum cost](../bindings/c/docs/minimum_cost.rst) <!-- reviewed: 31066073b902872f8b5ee0ed058b91001fe74c245575b5ce7462e0ebf4d99a05 -->
- [C program minimum_cost](../bindings/c/examples/minimum_cost.c) <!-- reviewed: d376304fe91f8dfee13b0efe4d45afcc69743b2d0082fdccafae3ef828303b8b -->
- [Python algebra contract](../bindings/python/tididi/__init__.py) <!-- reviewed: 0c18b270daece6e0970726e34b88e130d84ab7c208569a754c358b54ced8258e -->
- [C algebra contract](../bindings/c/src/evaluation.rs) <!-- reviewed: 290593cd8b5102135539193b1242d9cb2159140a95ccb263c3b2f652ffee9983 -->

## Statistics

Construct x₁ XOR x₂ over a balanced four-variable vtree and compare it with
literal x₁. Find the stored internal node with the most pairs: XOR has a
maximum of 2; a literal has 1. Distinguish per-node width, total pair count,
and model count (8 for XOR with the other two variables free).

Rust traverses level storage directly and uses (root,0) for an empty traversal.
Python traverses a `node_sizes()` snapshot and uses None when it is empty.
Storage IDs and the chosen maximum on ties are not semantic identities.
Python may use its XOR operator while Rust demonstrates its Boolean expansion.

C reports XOR totals and the maximum pair count from an owned snapshot. It
omits the literal comparison and represents an empty maximum by zero pairs.

### Instances

- [Rust walkthrough](examples/statistics.md) <!-- reviewed: 524e408c83bd1802cbe11e2bc611033d7b7c97eb2563a7f5f43273621ccf6250 -->
- [Rust program](../examples/statistic.rs) <!-- reviewed: c3ced7fe8b01876068a251673521a905290535a4ba172665d70ee285ddf3b004 -->
- [Python walkthrough and program](../bindings/python/examples/09_statistics.py) <!-- reviewed: 7b12de2327edc2c955b8bac8cd33dca31802b9bac47b5510fc454cdec7d1ef81 -->
- [C statistics](../bindings/c/docs/statistics.rst) <!-- reviewed: c80ccb0be03a8d51492c35eeaddfc0bac5012d440dbcc36a30d0d1c87c92633a -->
- [C program statistics](../bindings/c/examples/statistics.c) <!-- reviewed: cd12646e33b210286fc0e79ab6a38375e267409a656ac44960c4ef19bda62d64 -->

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

C retains a consumed handle until explicit free and reports a typed owned error
on reuse. It documents pointer validity, output initialization, borrowed arrays,
cleanup, callback reentry and caller synchronization. Like Python, its counter
owns the circuit until finish; unlike Python, freeing the handle ends all checks.

### Instances

- [Rust circuit documentation](../src/diagram/tdd/mod.rs) <!-- reviewed: 53443f07dc2f9e75af49bac56ed05003a96c15808cf5708dda4adfd41fab2a6c -->
- [Python ownership guide](../bindings/python/docs/ownership.rst) <!-- reviewed: 49e98fe18f2ea5b0d769f3b2301737e040ae22238e00b4dd9ac5296e493e94a0 -->
- [C ownership](../bindings/c/docs/ownership.rst) <!-- reviewed: 6fa3921cd9c76acae60b9bb7d58e71544dd85f15c8a54c55815cf910628e80c4 -->

## API overview

Group operations by what a reader wants to do: construct and combine rules,
query solutions, condition/quantify/rename, evaluate weights or costs, save and
inspect, and control representation or resource use. Link the relevant
application and authoritative API specification rather than copying contracts.

The Python reference is generated from binding docstrings. The Rust overview
also links specialized operations not yet exposed in Python. An operation
appearing in one reference does not imply that the other language exposes it.

The C header and reference are generated from the Rust binding declarations and
their contracts; the reference groups functions by purpose. Its numeric callback algebra uses doubles; exact weighted
evaluation has its own rational-string interface.

### Instances

- [Rust API overview](api-guide.md) <!-- reviewed: e33abb878c6a514ac28c20846d2c1c37401eca77e5cc5b1d7ecfc1dfb6daae49 -->
- [Python API reference](../bindings/python/docs/api.rst) <!-- reviewed: cf89774fb5f52ca207dd8cfb414e648661e4769736be8b529bf80ebe6eb7aeb2 -->
- [C api](../bindings/c/docs/api.rst) <!-- reviewed: a4cc0ac9db6cd02097704bc1e4d3dd2fed3acb087decd2073142fbaf614ea9cc -->

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

## Reusable components

Build one backup rule L∨R on a two-leaf vtree: 3 choices. Embed it twice in
a balanced four-leaf destination, mapping (1,2) to (1,2) and (3,4). Each
copy alone has 12 models because the other server's variables are free;
their conjunction has 9. Add ¬(R_A∧R_B) to represent shared remote capacity,
leaving 5 choices. The borrowed source still has 3 models.

Introduce the application, the local rule, the variable mapping, then the
connecting constraint. Explain matching variable grouping and destination
identity at placement, and attached-weight loss before suggesting evaluation.
Distinguish graft for components already using disjoint variable identifiers.
This lesson is Rust-only: the bindings do not yet expose embedding or grafting.
Do not invent a language counterpart that rebuilds the rule instead of reusing it.

### Instances

- [Rust walkthrough](examples/composition.md) <!-- reviewed: 9ba8bcd2b2328164bbfb3e67f9eeaff4fb0e5ed270a55f74087428ba2c493280 -->
- [Rust program](../examples/reusable_components.rs) <!-- reviewed: fd0b1dfed09cde22d73d5eacbdb9f39b32f32e31b41a929594d165bd6492e37b -->
