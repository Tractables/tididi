//! Restrict-to-care: prune `f` to the subgraph that survives under a "care" diagram.
//!
//! `f.restrict_to_care(care)?` returns `g`, a **structural subgraph of `f`** — every node
//! of `g` is a node of `f` keeping a subset of its pairs — with
//! `g ∧ care == f ∧ care`. It never grows the diagram (`g.pair_count() ≤ f.pair_count()`):
//! it deletes pairs and nodes that produce no model under `care`, nothing else.
//! The result is orphan-free but otherwise non-canonical, so a caller that
//! needs a reduced diagram runs `minimize` on it.
//!
//! Algorithm — a top-down walk over the products of `f × care` that a model
//! of both uses, then an in-place drop of every `f` pair it did not mark:
//! 1. Start at `r = lca(root(f), root(care))`. An operand not rooted at `r` is
//!    `⊤` there (`None`) and becomes its root node once the walk reaches that
//!    vtree node. Incomparable roots cover disjoint variables, so `care` cannot
//!    constrain `f`: `Unchanged`.
//! 2. A product `(v, f_node, care_node)` is *live* iff some `f`-pair × `care`-pair
//!    has both child products live. A `ZERO` child is dead; a leaf is dead only
//!    for `{Pos, Neg}`; a level that is marginal in `f` is a count (always live,
//!    no descent); a level that is marginal in `care` is `⊤` for liveness, so
//!    `f` is walked under `⊤` below it. Below the lower operand root, liveness
//!    is decided bottom-up first, each level's live products joined from its
//!    children's through the two operands' pairs grouped by those children, so
//!    the work follows the live products. The walk then descends from the root
//!    along live pairs only and marks their `f` pairs: those are exactly the
//!    pairs some model of `f ∧ care` uses. Where every `f` node there has at
//!    most one live product, the listings are the marks and nothing is walked
//!    down: a live product the root does not reach keeps pairs of nodes no
//!    kept pair names, which the prune drops. A `care` node that is `⊤` over
//!    its level's variables (one pair of `⊤` children, how a reduced diagram
//!    holds it) is read as `⊤`: a level where every `care` node is one is
//!    joined on as a marginal one, never listed, and an `f` node a live product
//!    pairs with `⊤` is kept whole with everything below it, never walked,
//!    since every node of a diagram has a model. A care that is a projection
//!    onto some of the variables (`∃` of the others) is `⊤` over the rest of
//!    `f`'s variables, so the walk pays for `f`'s pairs over the projected
//!    variables and their ancestors only. Where `care` is rooted below `f`,
//!    every product on the path between the two roots is an `f` node beside a
//!    `care` not rooted yet: their liveness is one pass up the path's levels
//!    and their marks one pass down, on arrays by node, with nothing
//!    discovered or hashed. Where `f` is rooted below `care`, `f` has no pair
//!    above its root, and off the path a product is a `care` node beside
//!    `⊤`, live unless `ZERO`: one pass down the path over `care`'s nodes
//!    finds the products at `f`'s root that the root reaches.
//! 3. If the root product is dead, `f ∧ care ≡ ⊥` → `Unsatisfiable`. If every node
//!    and pair reachable from `f`'s root is marked → `Unchanged`. Otherwise the
//!    unmarked pairs are dropped in place (marginal levels untouched) and the
//!    orphan prune reclaims children stranded by a collapsed partner → `Shrunk`.
//!    A kept pair names live nodes only and a live node keeps a pair, so no
//!    falsity is left to propagate unless marks say otherwise.
//!
//! The walk is stack-driven; its tables, marking rows and work stacks use the
//! caller's limits.

mod mark;
mod rebuild;
mod pairs;

use std::borrow::Cow;

use crate::Engine;
use crate::limits::OperationError;

use crate::diagram::Tdd;

/// Outcome of [`Tdd::restrict_to_care`], so a caller can tell a no-op from a shrink
/// without comparing diagrams.
#[derive(Debug)]
pub enum RestrictionOutcome {
    /// The original circuit, returned without rebuilding it.
    Unchanged(Tdd),
    /// A smaller circuit that agrees with the original on the care set.
    /// Call [`Tdd::minimize`] if a canonical result is needed.
    Shrunk(Tdd),
    /// No assignments satisfy both the original circuit and the care set.
    /// Contains false on the original vtree, retaining its weight configuration.
    Unsatisfiable(Tdd),
}

impl RestrictionOutcome {
    /// Take the resulting circuit from any outcome.
    #[must_use]
    pub fn into_tdd(self) -> Tdd {
        match self {
            RestrictionOutcome::Unchanged(g) | RestrictionOutcome::Shrunk(g) | RestrictionOutcome::Unsatisfiable(g) => g,
        }
    }
}

/// [`restrict_prepared`] of operands on different projections of one home:
/// a copy of `f` placed with `care`, and `f` itself when nothing is removed.
#[cold]
fn restrict_across(eng: &Engine, f: Tdd, care: &Tdd) -> Result<RestrictionOutcome, OperationError> {
    let [placed, care] = crate::apply::extend::on_one(eng, [Cow::Borrowed(&f), Cow::Borrowed(care)])?;
    if f.is_zero() {
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    let placed = crate::apply::extend::into_owned(eng, placed)?;
    match restrict_prepared(eng, placed, &care)? {
        RestrictionOutcome::Unchanged(_) => Ok(RestrictionOutcome::Unchanged(f)),
        outcome => outcome.trimmed(eng),
    }
}

impl RestrictionOutcome {
    /// The outcome with a new diagram on a projection trimmed.
    fn trimmed(self, eng: &Engine) -> Result<Self, OperationError> {
        use crate::apply::extend::trimmed;
        Ok(match self {
            RestrictionOutcome::Unchanged(f) => RestrictionOutcome::Unchanged(f),
            RestrictionOutcome::Shrunk(f) => RestrictionOutcome::Shrunk(trimmed(eng, f)?),
            RestrictionOutcome::Unsatisfiable(f) => RestrictionOutcome::Unsatisfiable(trimmed(eng, f)?),
        })
    }

    fn exit(self, eng: &Engine) -> Result<Self, OperationError> {
        use crate::apply::exit::outcome;
        Ok(match self {
            RestrictionOutcome::Unchanged(f) => RestrictionOutcome::Unchanged(outcome(eng, Ok(f))?),
            RestrictionOutcome::Shrunk(f) => RestrictionOutcome::Shrunk(outcome(eng, Ok(f))?),
            RestrictionOutcome::Unsatisfiable(f) => RestrictionOutcome::Unsatisfiable(outcome(eng, Ok(f))?),
        })
    }
}

/// Restrict a non-false `f` to `care`.
fn restrict_prepared(eng: &Engine, f: Tdd, care: &Tdd) -> Result<RestrictionOutcome, OperationError> {
    if care.is_zero() {
        return Ok(RestrictionOutcome::Unsatisfiable(crate::build::constant_like(eng, &f, false)?));
    }
    let v0 = f.output.vtree;
    // Both operands must share vtree structure; the walk reads indices in `f.vtree`.
    let r = f.vtree.lca(v0, care.output.vtree);
    if r != v0 && r != care.output.vtree {
        // Incomparable roots ⇒ disjoint variable regions ⇒ care can't constrain f.
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    Ok(match Marking::walk(eng, &f, care, r)? {
        Walked::Unchanged => RestrictionOutcome::Unchanged(f),
        Walked::Unsatisfiable => RestrictionOutcome::Unsatisfiable(crate::build::constant_like(eng, &f, false)?),
        Walked::Marks(marks) if marks.nothing_reachable_died(eng, &f)? => RestrictionOutcome::Unchanged(f),
        Walked::Marks(marks) => RestrictionOutcome::Shrunk(marks.rebuild(eng, f)?),
    })
}

/// What [`Marking::walk`] found.
enum Walked {
    /// Nothing can die: the root product is a compatible leaf or a marginal
    /// scalar.
    Unchanged,
    /// The root product is dead: `f ∧ care ≡ ⊥`.
    Unsatisfiable,
    Marks(Marking),
}

/// Liveness marks over `f` produced by the `f × care` walk.
struct Marking {
    /// `[v.idx()][f-local]` — does this f-node survive under care?
    alive: Vec<Vec<bool>>,
    /// `[v.idx()][f-local]` — an f-node a live product pairs with a `care`
    /// that is `⊤` over the level's variables: the node and everything below
    /// it survive whole, so the walk never enters it.
    keep: Vec<Vec<bool>>,
    /// Pair liveness, including nodes wider than a machine word.
    pair_alive: pairs::PairMarks,
}

impl crate::Engine {
    /// Run [`Tdd::restrict_to_care`](crate::Tdd::restrict_to_care) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn restrict_to_care(&self, f: Tdd, care: &Tdd) -> Result<RestrictionOutcome, OperationError> {
        let _op = self.limits().enter()?;
        let outcome = if crate::apply::extend::needed([&f, care]) {
            restrict_across(self, f, care)?
        } else if f.is_zero() {
            RestrictionOutcome::Unchanged(f)
        } else {
            restrict_prepared(self, f, care)?.trimmed(self)?
        };
        outcome.exit(self)
    }
}
