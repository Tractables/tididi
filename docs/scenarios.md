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
- [Rust crate introduction](../src/lib.rs) <!-- reviewed: 73ea6799948cda4653f6b6e2f39d92c611b091e9d8aa0d1b1b57c7514a513af5 -->
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
limits, minimum costs and storage statistics. Rust adds reusable components and care-set simplification
after vtree grouping, and marginalization after minimum costs; these operations
are currently exposed only in Rust. Keep the
index and sidebar in that order. Link reference material separately; do not make beginners choose
an API category before they have seen a circuit.

The counting lesson follows configurations in every language. Python uses a
short doctested page for this comparison instead of a gallery application.

The C guide adds a first-circuit chapter and places its ownership chapter before
resource limits; the application chapters retain the same order.

### Instances

- [Rust guide and example navigation](../src/guide.rs) <!-- reviewed: a659b4c3434b3dd1f96b690f00f9da9bd078896c1c58fc88e795b1287bcfaab9 -->
- [Python guide index](../bindings/python/docs/index.rst) <!-- reviewed: f9cb128a9cbd5b76db5695320ace78dd852ce38a7cdbc4ca2a453cdfbef4bce6 -->
- [Python gallery introduction](../bindings/python/examples/GALLERY_HEADER.rst) <!-- reviewed: 836d9d8f1ff4c5feb0b5f766281cc25372579e2d299f6a993cd825dfa81a7850 -->
- [C index](../bindings/c/docs/index.rst) <!-- reviewed: 0e6c564d98419e17db63d947b0488f27c319d95fdd65f20f4e3ee1677a8326fa -->

## Configurations

Model four backup options: local L, remote R, encryption E and notifications N.
The rule is `(L ∨ R) ∧ (¬R ∨ E)`; N is free. Build named literal values and
circuits, introduce Boolean operators, then count and inspect user choices.
There are 8 configurations, 4 with notifications, and 4 with remote backups.
Remote forces encryption; remote without encryption is unsatisfiable.

A witness describes one complete valid assignment; conjoining its cube leaves
1 configuration. Observe R, then ¬N, then replace R and E by ¬R and ¬E:
the counter reports 4, 2, 1; clearing observations restores 8. Do not require
one particular witness or storage ID.

Rust's counter borrows its diagram; Python's owns it until `finish()`.
Teach checked calls after operators in Rust. Python operators already raise
exceptions. Python finishes its owning counter and minimizes the returned
diagram, preserving the same models; Rust needs no ownership handoff.

C expresses clauses through named signed integer constants. Its shorter lesson
shows counts 8 and 4, forced remote/encryption, then counter counts 4, 2 and 8.
It omits the witness, replacement observation and minimization demonstrations.

### Instances

- [Rust walkthrough](examples/configurations.md) <!-- reviewed: 3d2f49ea5b1910c45d050759f5a07e88c735e16e72aa53ce705dbfa68fbf7b02 -->
- [Rust program](../examples/configurations.rs) <!-- reviewed: 79ec2d3fa05da95e1360b51129c1de3d8090c146a1a6aead105ef16c08c11dd4 -->
- [Python walkthrough and program](../bindings/python/examples/01_configurations.py) <!-- reviewed: 6a369c0d5f4dd63db5fe776ae2f5699ac0d370d32925b2a9bc6a9e9c6ae67006 -->
- [C configurations](../bindings/c/docs/configurations.rst) <!-- reviewed: ac03abce92815601a4e6410435e1724f471e5ffcd9be87a436fae77a9fd2e392 -->
- [C program configurations](../bindings/c/examples/configurations.c) <!-- reviewed: 2030cee871c5dd5621f1f7e144c0319450e8bab6709470e436581bba30702e19 -->

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

- [Rust walkthrough](examples/counting.md) <!-- reviewed: 9c1bdc805c696a75f82eb405a96d1aa11333cc7ddfa981682d5f12ef4b6ef98e -->
- [Rust program](../examples/counting.rs) <!-- reviewed: bb1cf3ae7ed7e7982e1cb32090d196796737e81b378da3c1e6e53d086726687f -->
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

With the second set of priors, retain an evaluator and observe no rain.
Its joint mass is P(W ∧ ¬R)=1/25; clearing observations restores P(W)=16/25.
Rust borrows the circuit; Python and C consume it and return it on finish.

C passes exact weights as fraction strings and returns owned fraction strings.
Its weighted-ratio call divides the two event masses; it omits the unit-weight
comparison to keep arithmetic and allocation details out of the narrative.

### Instances

- [Rust walkthrough](examples/probability.md) <!-- reviewed: cd36add9507b1bdce74931d33c3fa02e2520ad9611c61832c481e2099842cb9b -->
- [Rust program](../examples/probability.rs) <!-- reviewed: d8e5ecbbb0908c05c2de3229b1bcf6c352346d7eee9089bd51826ae0127a8132 -->
- [Python walkthrough and program](../bindings/python/examples/02_probability.py) <!-- reviewed: 2cb09a4d22d3c587ca0055ee88a9b3d6b72bb5c5a8f049485a9c77d0a47e26a0 -->
- [C probability](../bindings/c/docs/probability.rst) <!-- reviewed: 70407f3617cf030796406fe1a00da39d51e104f851b1401f59c6106075ab888c -->
- [C program probability](../bindings/c/examples/probability.c) <!-- reviewed: b0fbc98b3e3f91980df886ed3e078f16ce313ac8d70c387c330ea3420adac282 -->

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

- [Rust walkthrough](examples/reachability.md) <!-- reviewed: 2f1d93b86eb973430d6b7ef73239445dadccfeff8ac7d1c380fff6467efc8bd0 -->
- [Rust program](../examples/reachability.rs) <!-- reviewed: 0888b766b278776e1e14f9102d2a500314d098104cffd82892530a85ce339ee3 -->
- [Python walkthrough and program](../bindings/python/examples/03_reachability.py) <!-- reviewed: 959d0a9aea4d72b273a51ff60a2124ed1740a3ca0531376773705f423e8755ad -->
- [Shared directed graph](reachability.svg) <!-- reviewed: b578209ba2f5bf62b32b3ba8100d7efefe03ccb8f10745da60d5af30cda85cdf -->
- [C reachability](../bindings/c/docs/reachability.rst) <!-- reviewed: 2399b28271eb37b09b22d0955b4fefec4fdd910a5c92cc79f9e9727439ab77b4 -->
- [C program reachability](../bindings/c/examples/reachability.c) <!-- reviewed: 8bf234fbfbed5a58bf1f034478670ddb75efea33ed66112b311ae91b851699b4 -->

## Tables

Model read, write and share permissions with variables 1, 2 and 3. Load rows
100, 110, 101, 110; duplicates describe one assignment, so the count is 3.
Insert 111 and remove 110: the count stays 3, but the rows become 100, 101, 111.
Filter a copy by share to count 2, preserving the table for the next update.
Then remove the partial assignment [share]: every sharing row disappears,
leaving only 100 and a count of 1. This deletes matching rows rather than
setting their share column to false.
Explain column order, free omitted variables, and signed-literal updates.
A partial cube edits all matching assignments; an empty cube matches all.

Rust uses packed rows and a maintenance batch that edits a diagram in place.
Python accepts Boolean rows and `update(insert=..., remove=...)`, consuming
the old wrapper and returning its minimized replacement. The language-specific
storage and ownership explanations must match these different interfaces.

C takes row-major Boolean bytes and arrays of signed-literal cubes. Like Python,
its update consumes the old circuit and returns a minimized replacement.

### Instances

- [Rust walkthrough](examples/tables.md) <!-- reviewed: d6f01c54b3b7f6662706b0178f50da146d6b848fc2a4b60d35a7d51beb36390e -->
- [Rust program](../examples/tables.rs) <!-- reviewed: 96b6a3d9ae912ebd88700613c7eefc5844b27b0b13b2d036ddc4959af8b79cb8 -->
- [Python walkthrough and program](../bindings/python/examples/04_tables.py) <!-- reviewed: dbc98d8e11b63d057e7342fa6749cf9c52cb09fab96d416f932b5b47d652b5d6 -->
- [C tables](../bindings/c/docs/tables.rst) <!-- reviewed: 218aea5f39680c7479c6526ad8416a3ea0e00f25a1928dc72644bd91d6a0b1b5 -->
- [C program tables](../bindings/c/examples/tables.c) <!-- reviewed: 740e508e413b467b7b10d527d4808f3b5ea5aa1e153278afcf480fbc8a347d91 -->

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

- [Rust walkthrough](examples/persistence.md) <!-- reviewed: 6f071aae8f7fbe2fab875aa0511aafba256ec46927ad23b179ee5133aae5b4b1 -->
- [Rust program](../examples/persistence.rs) <!-- reviewed: f415e622866f274ee20a4433d56b95e04d94ed00e77fc7f11df1a8e2e2268671 -->
- [Python walkthrough and program](../bindings/python/examples/05_persistence.py) <!-- reviewed: 2495a36faff122077a3e3a4135ec0f4ee83b46c86c2c05c920835b5766c85d42 -->
- [C persistence](../bindings/c/docs/persistence.rst) <!-- reviewed: 89fc3a373f111bc21b4d2489890849bc6c8dbfd9e00b40267721c75de23c3bd0 -->
- [C program persistence](../bindings/c/examples/persistence.c) <!-- reviewed: 3874298127872ef693e984c146532ff7c381b857fcb7825c2ef7fbe5772c649e -->

## Vtrees

Compare `(x₁ ↔ x₃) ∧ (x₂ ↔ x₄)` under balanced leaf orders [1,3,2,4]
and [1,2,3,4]. The first groups each equality together; the second separates
its variables. Both functions have 4 models, while minimized pair counts are
5 and 12 in the demonstrated representation. Show both groupings in a figure
and construct the same function for each using negated XOR for equality.
Introduce join and linear vtrees
only after the comparison. Explain size as a consequence of grouping, without
claiming that any heuristic guarantees small circuits.

C focuses on the two balanced groupings. Its reference documents join and linear
constructors without adding another construction to the introductory comparison.

### Instances

- [Rust walkthrough](examples/vtrees.md) <!-- reviewed: f22f8696a7bef7e7d5cbfde2b6863e3c9eca224ad251968ee5e7ecc0ce41f6a5 -->
- [Rust program](../examples/vtrees.rs) <!-- reviewed: e229b28ec67bd124ddd7c544edf8ceb043b013db3b89eee73580f016e6c19531 -->
- [Python walkthrough and program](../bindings/python/examples/06_vtrees.py) <!-- reviewed: eb7e3d5cdcd211eee86994d058fa5e1ea745bc8dc15ade88141a5c963eb02113 -->
- [Shared grouping figure](vtree-grouping.svg) <!-- reviewed: e9c0efa139f3728e27c1073aeef91573670de9ded84b519b2a2264f4d1a87e9c -->
- [C vtrees](../bindings/c/docs/vtrees.rst) <!-- reviewed: ca64314bd76037516607c6dfdb4afae61e993b23963d52bb4ce77771431541f7 -->
- [C program vtrees](../bindings/c/examples/vtrees.c) <!-- reviewed: 838d836a9d4e6c44d01333d8642df1a3ca7a4b0f1e46fbdfe32a9686181b3ee4 -->

## Execution

Use the four backup variables L, R, E and N. A zero-byte operation budget refuses
construction of L ∨ R; a later unrestricted call succeeds and counts 12.
Limits do not remain installed on unrelated calls. Show a bounded query and
releasing idle scratch while keeping circuits usable. Budgets cover charged
operation storage, not the entire process; deadlines are polled cooperatively.

Conjoin L ∨ R with E under a zero-byte budget, preserving the original operands
by passing copies. On failure, retry without a limit and count the 6 models of
`(L ∨ R) ∧ E`. Rust uses a context-supplied engine for bounded batches; Python
uses per-call `limits=`. Both show a bounded count of 6 and a satisfiability
query after releasing idle scratch. Rust retries inside the error branch;
Python demonstrates retaining its objects, then combines the originals.

C demonstrates the same retry and six-model function as Python, using a
TididiLimits pointer and an owned error. It goes directly to the consuming
operation failure rather than repeating the constructor-budget demonstration.

### Instances

- [Rust walkthrough](examples/execution.md) <!-- reviewed: 620150cf58e3bd7e19f5698d5c8405fcdaf18a5b54c27ae964ca82d52f148c07 -->
- [Rust program](../examples/execution.rs) <!-- reviewed: fa7d43d4cd84aa0f0a1e797dc501dccbb52d20e8d3465af9cef52e26ef4b630f -->
- [Python walkthrough and program](../bindings/python/examples/07_execution.py) <!-- reviewed: b49d78efdd305db3f28a88755b9796ffc53f3d2765436b9e269f6e9ce58d6625 -->
- [C execution](../bindings/c/docs/execution.rst) <!-- reviewed: fcbcae5fd17cc60cdedbcca19811e88043792a963803773f5b29f1bd30df430a -->
- [C program execution](../bindings/c/examples/execution.c) <!-- reviewed: 16932faf09936f0e2a42cc7bdf15f7294b5e0447dfdca84ed256f6fb35c8f5a1 -->

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

- [Rust walkthrough](examples/optimization.md) <!-- reviewed: 981340698852070bf476d766587cbb0b1fa6bf66f1cd0a6e4550b390c5b10a98 -->
- [Rust program](../examples/optimization.rs) <!-- reviewed: 9997b1b1bc408a5523373cd7fbf832ad7ce7282bf1aa259689fcfe586360b6a8 -->
- [Python walkthrough and program](../bindings/python/examples/08_minimum_cost.py) <!-- reviewed: 79dce3065ddab786f40085239b1e037fa83dbb734744bb7f262e6184f5e49638 -->
- [C minimum cost](../bindings/c/docs/minimum_cost.rst) <!-- reviewed: 31066073b902872f8b5ee0ed058b91001fe74c245575b5ce7462e0ebf4d99a05 -->
- [C program minimum_cost](../bindings/c/examples/minimum_cost.c) <!-- reviewed: d376304fe91f8dfee13b0efe4d45afcc69743b2d0082fdccafae3ef828303b8b -->
- [Python algebra contract](../bindings/python/tididi/__init__.py) <!-- reviewed: e78c4042d15680740ab229fed8a25880e95f436177a0c00c28f1fceef7f674a2 -->
- [C algebra contract](../bindings/c/src/evaluation.rs) <!-- reviewed: 290593cd8b5102135539193b1242d9cb2159140a95ccb263c3b2f652ffee9983 -->

## Statistics

Construct x₁ XOR x₂ over a balanced four-variable vtree and compare it with
literal x₁. Find the stored internal node with the most pairs: XOR has a
maximum of 2; a literal has 1. Distinguish per-node width, total pair count,
and model count (8 for XOR with the other two variables free).

Rust traverses level storage directly and uses (root,0) for an empty traversal.
Python traverses a `node_sizes()` snapshot and uses None when it is empty.
Storage IDs and the chosen maximum on ties are not semantic identities.
Use the direct XOR operation to construct the circuit. Rust passes a copy
of x₁ to XOR so the original stays available for its literal comparison;
Python uses its XOR operator.

C reports XOR totals and the maximum pair count from an owned snapshot. It
omits the literal comparison and represents an empty maximum by zero pairs.

### Instances

- [Rust walkthrough](examples/statistics.md) <!-- reviewed: 9589ae6f53391622df6fb0e5722b2bc4d3f9128cd2dba6fa4d8b9dfb322e36dd -->
- [Rust program](../examples/statistics.rs) <!-- reviewed: af452250f58ee6f6a2ddacdeb1380ba0efc6405a902214ab6df305d3973ea70a -->
- [Python walkthrough and program](../bindings/python/examples/09_statistics.py) <!-- reviewed: d405decff7aa771faa37c7c8694e017388052b40060de82be72e3180fee835e3 -->
- [C statistics](../bindings/c/docs/statistics.rst) <!-- reviewed: 9445ca2077f5955c6a34013c4abec4f0132b827e6c32738aa84bfb7cee1d367c -->
- [C program statistics](../bindings/c/examples/statistics.c) <!-- reviewed: d4b8534d0c233aa6b18754eb98bb3f6be219cdeb5009eb252f34da77531c044d -->

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

- [Rust circuit documentation](../src/diagram/tdd/mod.rs) <!-- reviewed: bb4cf5efd63a25e2d0aa5f23efb7dabee5c593e367ec84d58474ec199ebc68d3 -->
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

- [Rust API overview](api-guide.md) <!-- reviewed: 6009637173dfb72d1b92cfc35f4b6e4aa073b1f240f3f586ccc5c0c566c5907e -->
- [Python API reference](../bindings/python/docs/api.rst) <!-- reviewed: 619c8cf6bbe3d4b720da766850817f34b4fdf92c866f3c6cd9828cd555fdcd39 -->
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

- [Rust data model](tdd.md) <!-- reviewed: 18e89ce1fbc9df16446520cf22f37e8752b56068d9f7f1045d083408f06630dc -->
- [Python data model](../bindings/python/docs/representation.rst) <!-- reviewed: ed52b71f88c82caf64b54c6227acbe5de4215ce34886e327f1b102cdd338b20c -->
- [Shared decomposition figure](tdd-basics.svg) <!-- reviewed: 5f2ab1677522a6a53385289841534c2a0096087bc65c8ce7f6c51d3be0e83c12 -->

## Architecture

Explain the Rust implementation through storage ownership, level arenas,
child-reference decoding, reduction passes, scratch and limits. Include unfinished
output ownership, intermediate product storage, coordinated marginal payloads,
shared incremental query scheduling for borrowed and owned circuits, reusable
embedding plans and shared placement assembly, pooled operation workspaces
with shared retained-capacity accounting,
reusable relation layouts, reduction scheduling, checked
vtree construction and traversal
validation, and the boundary that coordinates level edits with references
and worklists. Define terms
once, map modules to responsibilities, and identify which operations establish
or temporarily break each invariant. Distinguish validation of stored data
from proving structural determinism. This is a Rust implementation reference;
there is no separate Python implementation to describe.

### Instances

- [Rust architecture reference](architecture.md) <!-- reviewed: 30fe61129efd5731970b1c8f1a3ed95cb4cfac4c11ba822400666c6c4bc1158b -->

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
Then reuse a prepared placement for server B, comparing L∨R (12 models over
the destination) with L∧¬R (4); server A remains free in both.
This lesson is Rust-only: the bindings do not yet expose embedding or grafting.
Do not invent a language counterpart that rebuilds the rule instead of reusing it.

### Instances

- [Rust walkthrough](examples/composition.md) <!-- reviewed: f8ed30d4316d3fd6af576bb1dbfa3ecc40a56ca5e5a512bfa6f6bb8e9a85a510 -->
- [Rust program](../examples/composition.rs) <!-- reviewed: ab8256a1de3317d4f53841dec9361268a0ecd155a6a9791e554c4f5bc6a7e91e -->

## Care sets

A backup rule F=L∨R has three models. The deployment guarantees
C=(R→L)=¬R∨L. Under this assumption the rule simplifies to L. Build F and C
on the same two-leaf vtree, restrict a copy of F, extract the result and minimize.
The example reduces three pairs to one. Compare G∧C with F∧C: equivalent,
with two models (local only and both). Compare F and G without C: not equivalent.
The remote-only assignment (L=false,R=true) satisfies F but not G.

State the Boolean functions before the code. Explain that C is enforced
elsewhere, and that changing the deployment invalidates the simplification.
Contrast with conjunction for the exact constrained set and substitution for
fixed variable values. Do not promise a particular formula or maximal shrink
for arbitrary inputs. This is a Rust-only lesson; neither binding exposes
care-set restriction yet.

### Instances

- [Rust walkthrough](examples/care.md) <!-- reviewed: 730a40c4db6c0e629c0d201bb2481bd99d6ae247d2f26d586733ef98e65fd282 -->
- [Rust program](../examples/care.rs) <!-- reviewed: 9a1bf5d9f30981a960eb4b67fce5a675b6d6af75c342a2c197383dfcd0ba2d57 -->

## Keeping values

Use two servers with F=(L_A∨R_A)∧(L_B∨R_B), grouped by server in a balanced
four-leaf vtree. There are 9 models. Summarize A's subtree on a clone: its count
stays 9, while the original projected onto B has 3 choices. Observing R_B in
the summarized circuit gives 6; observing R_A raises MarginalLevel. Keeping the
original allows observing both remote options, giving 4 configurations.

For the weighted version, start from a structural copy and attach independent
probability-1/2 weights in exact rational arithmetic before marginalizing A.
The weighted value stays 9/16. Change every probability to 1/3 and evaluate the
original: 25/81. Evaluating the summarized circuit with a new algebra is refused
because assignments were discarded. Do not suggest that changing a weight table
can reconstruct a stored sum, or that projection and marginalization count the
same thing. Select a vtree subtree, not variable IDs. Rust-only: the bindings
currently expose structural evaluation, not marginalization or attached stores.

### Instances

- [Rust walkthrough](examples/marginalization.md) <!-- reviewed: e893ff37d2727a29896efa6d1f5764f96d963596a980898177dcbab7663cab2a -->
- [Rust program](../examples/marginalization.rs) <!-- reviewed: 14f8dea082246f892802cf21e9b1a3a8de4e9fbf507a908cde17780b7d2f4590 -->
