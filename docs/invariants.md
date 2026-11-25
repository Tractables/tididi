<!-- scenario: docs/scenarios.md#invariants -->

# Invariants

This page states what holds of a [`Tdd`]: of every diagram an operation
returns, of a diagram that carries the canonical certificate, and of a diagram
with a summed-out level. Each item names the code that keeps it and the checker
that decides it. Source comments and checker messages refer to the items by
number, so the numbers are fixed and are not consecutive within a section. The
[data model] explains vtrees, levels, nodes and pairs; the [architecture
reference] explains the passes named here.

## Terms

- A level is **structural** when it stores nodes made of pairs, and
  **summed out** (marginal) when [`Tdd::marginalize`] replaced its nodes
  by values.
- A leaf level offers the labels `Pos` (the variable), `Neg` (its negation)
  and `One` (true). `Pos` and `Neg` are the **literals**.
- A node is **dead** when the output does not reach it. Unless an item says
  "reachable", it speaks of every stored node, dead ones included.
- A node `m` is **named at the place** `(n, b)` when node `n` of the parent
  level holds a pair of `m` with partner `b` on the other side. Two nodes of
  one level are **twins** when they are named at the same set of places.
- A [`Home`] is a root vtree with an index over its leaves. Its
  **projection** on a set of variables has those variables as leaves and, as
  internal nodes, the home's nodes that have one of them under each child
  ([`Vtree::projection`]). A diagram is **on a home** when its vtree is a
  home's root or one of its projections ([`Vtree::home`] is `Some`), and **on
  a plain root** otherwise: a root vtree no `Home` was made of. Every diagram
  a public constructor, reader or operation returns is on a home, and
  [`Tdd::builder`], [`Tdd::restructure_to`] and [`Engine::tag_level`] refuse a
  plain root; a diagram is on a plain root only inside an operation, as its
  working form, and in the crate's own test fixtures.
- A diagram is **certified** when [`Tdd::is_canonical`] is true.

The two kinds of vtree give two layouts. On a plain root a diagram stores a
level for every vtree node, and a variable the function does not read keeps
its leaf, which a reduction names `One`. On a home a diagram sits on the
projection on the variables its pairs name (item 15), so it stores nothing
for the rest of the home.

## Every diagram

These hold of every diagram a public operation returns, reduced or not, and
every operation may assume them of its operands. They hold at operation
boundaries; the third column names where one is briefly false inside an
operation.

| # | Statement | Kept by | Decided by |
|---|---|---|---|
| 13 | Storage agrees with the vtree: one level per vtree node, at the node's index; a leaf level stores no node; every pair side names a slot of the child level (a label at a leaf, a node below the level's slot count, a value at a summed-out child); the output is a node of the root level or the false sentinel, and on the vtree with no node the false or the true sentinel. | every producer; [`TddBuilder::finish`] and the readers in [`io`] check it in every build profile | `test_helpers::check::validate_vtree_structure` (the output, empty leaf levels, pair sides in range); `diagram::check_stored`, the storage check of [`TddBuilder::finish`] on any diagram, the level count included |
| 1 | Determinism: distinct nodes of a structural level denote disjoint sets of assignments. So a pair of child nodes belongs to at most one node of its level and occurs once in it, and a node's count is the sum over its pairs of the product of the two child counts. | apply's emit; projection's owner-set regroup. [`TddBuilder::finish`] and the readers refuse a diagram that breaks it at a level with no summed-out level under it (see below), and leave the levels above one to the author | `test_helpers::check::check_determinism` on small diagrams; the repeated-pair check of [`TddBuilder::finish`] (see below), which `diagram::check_stored` runs on any diagram |
| 14 | Leaf form: a structural level names a leaf child that is not summed out in one form, as `One` in every stored pair or as a literal in every stored pair. It is the leaf case of item 1: `One` overlaps both literals. | the three writers of `One` (leaf twin contraction, conditioning, quantification) rewrite a whole side of a level or none of it | [`TddBuilder::finish`] and `diagram::check_stored`; `test_helpers::check::assert_no_mixed_leaf_side`; `check_determinism` |
| 2 | No false node: every stored node of a structural level holds a pair, and no pair names the false sentinel, so every stored node is satisfiable. False is the output sentinel only, and a false diagram stores no node. | checked builder; apply's emit; conditioning's falsity sweep. Broken by conditioning's leaf rewrite, within one call | `test_helpers::check::check_no_false_nodes` |
| 3 | Distinct functions, with no summed-out level: no two stored nodes of a level denote the same function, so no two have equal pair sets. It follows from items 1 and 2: equal and disjoint, the function would be false. | items 1 and 2 | `test_helpers::check::check_distinct_functions` (random signatures) |
| 15 | Projection form, on a home: no stored pair names a variable leaf `One`, and the output of a one-leaf diagram is a literal; a constant is on the home's projection with no node ([`Vtree::empty_projection`]), and only a constant is. So every leaf of the vtree is named by a literal in every pair of its parent level, and the vtree is the home's projection on those variables. | the close that ends every operation drops the leaves a pass named `One` (`reduce::settle`); [`TddBuilder::finish`] and the readers trim ([`Tdd::trim`]), a diagram with a summed-out level included. `finish` returns a diagram as built when the trim would have to rebuild a summed-out variable leaf, and this item can then fail | `test_helpers::check::check_trimmed` |
| 16 | Closed levels: a structural level that can be held as the description of its pairs ([`ImplicitLevel`], "Canonical form") is held so, in normal form, and every other level is stored; a count of live pairs kept on a level is its number of live pairs, and no level holds live pairs left for a close. | the close of each level an operation changed (`TddLevel::close`). False inside an operation, where a pass holds a changed level stored | `test_helpers::check::check_implicit_levels`, `check_left_pairs_taken` |
| 17 | Loose levels: when a diagram knows its loose levels, every structural level below the root that is not listed has each of its nodes named by a pair of its parent level. A diagram may not know them. | an edit keeps the list only by handing it on, as the conjunction, the prune, the contraction and the constructors do; any other edit forgets it | `test_helpers::check::check_loose_levels` |
| 18 | Worklists: a level that is on no worklist of a contraction pass is at that pass's fixpoint. A list may hold stale or repeated entries; it may not omit a level whose pairs changed. | `Tdd::invalidate`, called by every edit for each level it changed; a pass that is refused hands back what it did not reach | on a certified diagram `test_helpers::check::check_certified`; otherwise no checker |

**Dead nodes.** A diagram returned without a full reduction may hold dead
nodes and twins. Items 1, 2, 14 and 15 hold over every stored node all the
same, because some readers take a whole level: `trim` reads a level's form
off its first node, and negation's fill, the variable expansion and the
maintenance index read every node. At the root, which has no parent, the
nodes other than the output are dead and disjoint from it; an operation that
grows the output in place or builds its complement beside it drops them
first. A reader that needs reachability requires the certificate, prunes, or
numbers the reachable nodes.

**Determinism in linear time.** Given items 13, 14 and 2, a diagram with no
summed-out level is deterministic exactly when no structural level holds a
pair twice, within a node or across nodes. Induct from the leaves: two
different pairs differ on a side, where the two children are disjoint, so
their products are disjoint, and every product is nonempty; a repeated pair
is one nonempty product counted twice. [`TddBuilder::finish`] runs this
check ([`TddBuildError::RepeatedPair`]) on every level with no summed-out
level under it. It decides determinism of the stored circuit, not that the
circuit is the function its author meant.

**What item 15 excludes.** On a plain root a level can pass a child through:
every pair names true on one side, which is `One` at a leaf and, above the
leaves, a node whose pairs reach the leaves only through `One`. On a home no
pair names `One`, so item 15 excludes every pass-through spelled with `One`.
It does not exclude a node that denotes every assignment of its variables
through literals, as `{(Pos, Pos), (Pos, Neg), (Neg, Pos), (Neg, Neg)}` does,
nor a leaf read alike through both literals. Such a node is the only node of
its level (item 1), so its parent level passes only its other child through.
Item 15 together with "no node denotes every assignment of its variables"
(item 21) excludes every pass-through, internal sides included; only a
certified diagram is bound by item 21, and an uncertified result can hold
such a node until a reduction contracts its leaf twins.

## A certified diagram

[`Tdd::is_canonical`] reports a certificate the diagram carries. A certified
diagram satisfies the items above and these. The library certifies only
diagrams with no summed-out level.

| # | Statement | Kept by | Decided by |
|---|---|---|---|
| 4 | Reachability: the output reaches every stored node. The root level holds the output alone. | the prune | `test_helpers::check::check_canonicity` (rule 1); `check_certified` |
| 19 | No twins: no two nodes of a structural level are named at the same set of places. | twin contraction, which replaces twins by their union | `check_canonicity` (rule 2) |
| 20 | No contractible leaf side: no level names a variable leaf by literals with every node holding `(Pos, b)` exactly when it holds `(Neg, b)`. Such a leaf is one the function does not read. | leaf twin contraction, which on a plain root rewrites the side to `One` and on a home drops the leaf | `check_canonicity` (rule 3); `check_certified`; on a home `check_trimmed`; on a plain root, at the leaves, `test_helpers::check::check_padded_support` (a leaf is named `One` exactly when its variable is outside the support) |
| 21 | Support, on a home: the variables at the leaves are the [support](crate::Tdd::support) of the function. So the vtree is the home's projection on the support, and its internal nodes are exactly the home's nodes with a support variable under each child: the function depends on both sides of every vtree node that has a level, and of no other home node. No node denotes every assignment of its variables. | items 15 and 20 (proof below) | `check_trimmed` with `minimized` (the leaves); `test_helpers::check::check_no_true_node` (no node denotes every assignment) |

**Uniqueness.** Two certified diagrams on one plain root, or on projections
of one home, denote the same function over the root's or the home's
variables exactly when they are equal up to the order of nodes within a
level and of pairs within a node. On a home they are then on the same
projection. Weights attached to the vtree do not enter: the form is canonical
for the Boolean function. For a plain root this is the minimization result of
[the TDD paper, Section 5].

*Proof.* Let `f` be the function of a certified diagram and `L` the
variables at its leaves. For a node `m` at vtree node `t`, write `⟦m⟧` for
the assignments of `t`'s variables it denotes, and define its context
`C(m)`, a set of assignments of the variables of `L` outside `t`: the
context of the output is the empty assignment alone, and `C(m)` is the union
of `⟦b⟧ × C(n)` over the places `(n, b)` of `m`. Going down from the root:

1. `f`, as a set of assignments of `L`, is the disjoint union of
   `⟦m⟧ × C(m)` over the nodes `m` of `t` (item 1), and every `C(m)` is
   nonempty (items 4 and 2).
2. Distinct nodes of a level have distinct contexts, by induction from the
   root level, which holds one node. By item 1 a pair `(m, b)` lies in at
   most one node `n` of the parent level, and the nodes of the sibling
   level denote disjoint sets (at a leaf, by item 14), so the slice of
   `C(m)` at an assignment of `⟦b⟧` is `C(n)` for the node `n` holding
   `(m, b)`, and empty when no node holds it. If `C(m) = C(m′)`, then for
   every `b` the pairs `(m, b)` and `(m′, b)` both exist or neither does,
   and the nodes `n` and `n′` holding them have `C(n) = C(n′)`; contexts
   at the parent level are distinct, so `n = n′`. Then `m` and `m′` are
   named at the same places, against item 19.
3. So `⟦m⟧` is the set of assignments of `t`'s variables whose residual in
   `f` is `C(m)`: the nodes of `t` are the classes of assignments with one
   nonempty residual, and node `n` holds the pair `(a, b)` exactly when
   class `a` times class `b` lies in class `n`. The diagram is a function of
   `f`, `L` and the vtree, up to node and pair order.
4. The same slice at a leaf `x` shows that `f` ignores `x` exactly when its
   parent level names it `One`, or every node there holds `(Pos, b)` exactly
   when it holds `(Neg, b)`. Item 20 excludes the second. On a plain root
   the leaves named `One` are therefore the variables `f` ignores, and `L`
   is all the root's variables. On a home item 15 excludes the first, so `L`
   is the support of `f`, the vtree is the home's projection on it, and by
   the definition of a projection each internal node has a support variable
   under each child.
5. If a node `m` denoted every assignment of `t`'s variables, step 1 would
   give `f = ⟦m⟧ × C(m)`, which ignores them, against step 4. ∎

**Who sets the certificate.** [`Tdd::minimize`], and [`Tdd::reduce`] with a
full [`ReductionPlan`], certify a diagram with no summed-out level. Constants
and literals are built certified. Every other operation states in its
documentation whether it sets the certificate, keeps that of certified
operands or gives it up; a result without it may be canonical, unknown to the
library. A
write to any level forgets the certificate. `minimize` returns at once on a
certified diagram whose worklists are empty.

## A diagram with a summed-out level

A summed-out level holds one value per node: a model count, or a weighted
value when the vtree or home carries [`Weights`].
[`Tdd::marginalize`] sums out variables, and each summed part becomes a
**region**, a childless vtree node that stands for the variables summed into
it ([`VtreeNode::is_region`]).

What changes:

- A node's pairs are a multiset: a pair that occurs `k` times counts `k`
  times. The diagram denotes a value for each assignment of the variables
  that are left, the sum over the summed-out ones, so item 1 is not defined
  at a level with a summed-out level under it, and `check_determinism` does
  not apply to the diagram.
- Item 3 does not hold: two pairs over disjoint children and one summed
  value compute what one pair over their disjunction computes, and no pass
  merges the two.
- There is no certificate and no canonical form. `minimize` reaches the
  fixpoint of its passes, at which items 7 to 10 and 12 hold, but two
  diagrams of one summed function can differ in structure, depending on the
  operations that built them and on where content-twin scans ran
  ([`ContentTwinPolicy`]). Exact answers (counts, exact rationals) depend
  only on the function and the summed variables; float answers move by
  rounding. [`Tdd::equivalent`] refuses such a diagram.
- Items 13, 2, 14, 15, 16, 17 and 18 hold at the structural levels as stated.

| # | Statement | Kept by | Decided by |
|---|---|---|---|
| 5 | Marginality is permanent and downward-closed: a summed-out level never becomes structural, and the levels below it are summed out or leaves whose contribution is absorbed. | [`Tdd::marginalize`] | `diagram::check_stored` (downward closure); [`reduce`]'s demarginalization guard |
| 22 | Regions, on a home: a summed-out level stands only at a region or a variable leaf, and every region holds one. | [`Tdd::marginalize`]; [`Tdd::trim`]; [`TddBuilder::finish`], which makes a region of each internal summed-out level that is not under another (with the exception of item 15) | `check_trimmed` |
| 23 | Extents, on a home: the extents of the regions are pairwise disjoint and hold no variable that is at a leaf. | [`Tdd::marginalize`]; the conjunction refuses an operand with a variable leaf inside the other operand's extent | `check_trimmed` |
| 6 | Every reference into a summed-out child decodes through [`ChildDecoder`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds a value that fits inline in the reference. | the reference tagger, then the slot prune. Broken by apply's emit, before tagging | `test_helpers::check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: no node of a structural parent holds two pairs over the same structural child that pair fusion would add up (exact arithmetic; at a weighted leaf or region the sum must already be in the column, where fusion mints no value). | the fusion sweep of [`Tdd::marginalize`]. Broken by a later twin merge | `test_helpers::check::marginal::check_pair_fusion_saturation` |
| 9 | Content twins: no two nodes of a level have equal pair multisets. | the content-twin merge. Broken by pair fusion and by the slot prune's value merge | `test_helpers::check::marginal::check_twin_canonicality` |
| 10 | Value-slot uniqueness: the counts stored at a summed-out level are pairwise distinct. A weighted level instead holds one value per node, equal values included. | mint-time dedup, then the slot prune. Broken by apply's emit | `test_helpers::check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are a fixed, label-ordered copy of the literal weights. No pass compacts, erases, reorders or appends to the column, and every reader derives it through `diagram::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | `test_helpers::check::marginal::check_leaf_columns_pinned` |
| 12 | Slot reachability: at a summed-out child of a structural parent, other than the output's level, every count slot is referenced from the parent; in a weighted column every reference names a slot the column holds. | [`reduce`]'s slot prune. Broken by pair fusion and twin contraction, until the prune runs | `test_helpers::check::marginal::check_no_orphan_slots` |
| 24 | Subsumed stores: a summed-out level whose parent is summed out holds no value store; a weight-marginal leaf keeps its column (item 11). | marginalization and the marginalizing conjunction, which free a child's store as its parent is summed out | `test_helpers::check::marginal::check_subsumed_stores_empty` |

Items 5, 6, 11, 22, 23 and 24 hold throughout. Items 7 to 10 and 12 hold after
the pass that establishes them, and together at the end of a full reduction:
`check_canonicity` (rule 4) decides them there in place of item 3.

## Where the checks run

`test_helpers::check` is compiled in every debug build and under the
`testing` feature in any profile. A checker reports and never repairs.

| Entry | Checks | Run by |
|---|---|---|
| `test_helpers::assert_canonical` | `check_all_fast` and `check_certified`; item 11 on a diagram with a summed-out level | every test and documentation example that produces a diagram |
| `check::check_all_fast` | items 13, 14, 5's downward closure and 1 at every level with no summed-out level under it (`diagram::check_stored`); 2 and 16; on a home 15, 17, 22, 23, and when certified 21 (`check_trimmed`, `check_no_true_node`); on a plain root, when certified, 20 at the leaves (`check_padded_support`); when certified 4, 19, 20; with no summed-out level 3 | `assert_canonical`; the operation tests, after each operation |
| `diagram::check_stored` | the checks of [`TddBuilder::finish`] with the repeated-pair check and the diagram's weights: items 13, 14, 5's downward closure, and 1 as no repeated pair | `check_all_fast` |
| `check::check_certified` | on a certified diagram items 4, 19, 20, and that a copy with every level put back on the worklists minimizes to as many nodes and pairs (item 18) | `assert_canonical` |
| `check::check_canonicity` | items 4, 19, 20, then 3, or 7 to 10, 24 and 12 on a diagram with a summed-out level | `check_all_fast` on a certified diagram; `test_helpers::assert_marginal_canonical` after `minimize` |
| `check::check_determinism` | items 1 and 14 by pairwise conjunction; diagrams of a few variables with no summed-out level | the operation tests on small inputs; the randomized differential suite |
| `check::check_trimmed` | that the vtree is a home's root or a projection; item 15; on a home 22 and 23; 17; on a certified diagram 4 and 20; with its `minimized` argument and no summed-out level 21 | `check_all_fast` on a home; the tests of the trim, the close, the moves and the readers |
| `check::check_no_true_node` | the last sentence of item 21: no node of an internal level over at most 126 variables denotes every assignment of them; no summed-out level | `check_all_fast` on a certified diagram on a home; the producer audit after `minimize` |
| `check::check_padded_support` | on a plain root with no summed-out level, a leaf is named `One` exactly when its variable is outside the support: item 20 at the leaves, and step 4 of the proof | `check_all_fast` on a certified diagram on a plain root (a test fixture) |
| `check::check_loose_levels` | item 17 | `check_trimmed`; the conjunction tests |
| `check::marginal` | items 7 to 12 and 24, one function each; `check_marginal_canonical_form` runs 7 to 10 and 24 | the tests of marginalization and of each reduction pass |
| the tests `reduce::tests::canonicity` and `test_helpers::check::tests::same_function_same_diagram` | uniqueness: one function reached by several routes minimizes to one diagram; `reduce::tests::close` compares a minimized result on a home with a rebuild on the support | the test suite |
| the producer audit, `diagram::tdd::tests::audit_i1` | `check_all_fast` on the result of every public producer on projections of a home, then item 21 (`check_trimmed` with `minimized`, `check_no_true_node`) after `minimize` | the test suite |
| debug assertions | item 16 at every close; item 4, and on a home item 20, where a certificate is set; item 8 after each fusion sweep; item 11 in the slot prune | every debug build |

What the checkers leave open:

- Items 1 and 14 are decided on an operation result only where a test runs
  `check_all_fast` (the linear check of "Determinism in linear time") or
  `check_determinism` on it.
- That a summed-out level stays summed out (item 5) is checked only by
  [`reduce`]'s demarginalization guard.
- Item 6 is kept by review.
- Item 18 is decided only through a certified diagram: an uncertified result
  whose worklists omit a level shows only when a later `minimize` stops
  short of the form `check_canonicity` accepts.
- On a plain root item 17 is checked only by the tests that call
  `check_loose_levels`.

[the TDD paper, Section 5]: https://arxiv.org/html/2604.05537v1#S5
[`ChildDecoder`]: crate::diagram::ChildDecoder
[`ContentTwinPolicy`]: crate::reduce::ContentTwinPolicy
[`Engine::tag_level`]: crate::Engine::tag_level
[`Home`]: crate::vtree::Home
[`ImplicitLevel`]: crate::diagram::ImplicitLevel
[`ReductionPlan`]: crate::reduce::ReductionPlan
[`Tdd`]: crate::Tdd
[`Tdd::equivalent`]: crate::Tdd::equivalent
[`Tdd::builder`]: crate::Tdd::builder
[`Tdd::is_canonical`]: crate::Tdd::is_canonical
[`Tdd::marginalize`]: crate::Tdd::marginalize
[`Tdd::minimize`]: crate::Tdd::minimize
[`Tdd::reduce`]: crate::Tdd::reduce
[`Tdd::restructure_to`]: crate::Tdd::restructure_to
[`Tdd::trim`]: crate::Tdd::trim
[`TddBuildError::RepeatedPair`]: crate::diagram::TddBuildError::RepeatedPair
[`TddBuilder`]: crate::diagram::TddBuilder
[`TddBuilder::finish`]: crate::diagram::TddBuilder::finish
[`Vtree::empty_projection`]: crate::Vtree::empty_projection
[`Vtree::home`]: crate::Vtree::home
[`Vtree::projection`]: crate::Vtree::projection
[`VtreeNode::is_region`]: crate::vtree::VtreeNode::is_region
[`Weights`]: crate::diagram::Weights
[`io`]: crate::io
[`reduce`]: crate::reduce
[architecture reference]: crate::guide::architecture
[data model]: crate::guide::model
