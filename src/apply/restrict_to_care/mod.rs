//! Restrict-to-care: prune `f` to the subgraph that survives under a "care" diagram.
//!
//! `f.restrict_to_care(care)?` returns `g`, a **structural subgraph of `f`** — every node
//! of `g` is a node of `f` keeping a subset of its pairs — with
//! `g ∧ care == f ∧ care`. It never grows the diagram (`g.pair_count() ≤ f.pair_count()`):
//! it deletes pairs and nodes that produce no model under `care`, nothing else.
//! The result is orphan-free but otherwise non-canonical, so a caller that
//! needs a reduced diagram runs `minimize` on it.
//!
//! Algorithm — a memoized top-down walk over node pairs of `f × care`, then a
//! rebuild:
//! 1. Start at `r = lca(root(f), root(care))`. An operand not rooted at `r` is
//!    `⊤` there (`None`) and becomes its root node once the walk reaches that
//!    vtree node. Incomparable roots cover disjoint variables, so `care` cannot
//!    constrain `f`: `Unchanged`.
//! 2. A pair `(v, f_node, care_node)` is *live* iff some `f`-pair × `care`-pair
//!    has both child pairs live. A `ZERO` child is dead; a leaf is dead only for
//!    `{Pos, Neg}`; a level that is marginal in `f` is a count (always live, no
//!    descent); a level that is marginal in `care` is `⊤` for liveness, so `f` is
//!    walked under `⊤` below it. Every pair is scanned (no early exit): an
//!    `f`-pair is marked live when it is live against *some* care pair, and an
//!    `f`-node when some pair of it is.
//! 3. If the root pair is dead, `f ∧ care ≡ ⊥` → `False`. If every node and pair
//!    reachable from `f`'s root is live → `Unchanged`. Otherwise `DeadRebuilder`
//!    re-emits the live subgraph (marginal levels verbatim) and the orphan prune
//!    reclaims children stranded by a collapsed partner → `Shrunk`.
//!
//! The walk is stack-driven and visits at most `|f| · |care|` node pairs;
//! its discovery tables, marking rows, rebuild arenas and work stacks use the caller's limits.

mod mark;
mod rebuild;

use crate::Engine;
use crate::limits::OperationError;

use crate::diagram::Tdd;

/// Outcome of [`Tdd::restrict_to_care`], so a caller can tell a no-op from a shrink
/// without comparing diagrams.
#[derive(Debug)]
pub enum RestrictionOutcome {
    /// Provably `g == f` (nothing reachable died, zero-/leaf-f early-out, or
    /// incomparable roots). No new diagram was built: the operand rides back
    /// unchanged.
    Unchanged(Tdd),
    /// A strict subgraph `g ⊊ f` (some pair died and the rebuild produced a
    /// smaller, count-correct-but-non-canonical `g`; caller canonicalizes).
    Shrunk(Tdd),
    /// `care` killed every model of `f` (`care ≡ ⊥` or `f ∧ care = ∅`): the
    /// canonical `⊥` over the operand's vtree, retaining its weight configuration.
    Unsatisfiable(Tdd),
}

impl RestrictionOutcome {
    /// Collapse to a concrete `g`, whichever arm it is.
    #[must_use]
    pub fn into_tdd(self) -> Tdd {
        match self {
            RestrictionOutcome::Unchanged(g) | RestrictionOutcome::Shrunk(g) | RestrictionOutcome::Unsatisfiable(g) => g,
        }
    }
}

/// The implementation behind [`Engine::restrict_to_care`](crate::Engine::restrict_to_care).
fn restrict_to_care_on(eng: &Engine, f: Tdd, mut care: Tdd) -> Result<RestrictionOutcome, OperationError> {
    crate::apply::check_vtree(&f, &care)?;
    let _op = eng.limits().begin_operation();
    if eng.limits().should_stop() { return Err(OperationError::Stopped); }
    if f.is_zero() {
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    // Sound for any representation of `care`, since `g ∧ care == f ∧ care`
    // does not depend on it; the reduced one gives the walk fewer pairs.
    eng.reduce(&mut care, crate::reduce::ReductionPlan::default())?;
    if care.is_zero() {
        // care ≡ ∅ ⇒ f ∧ care = ∅ ⇒ ⊥ is the smallest sound representative.
        return Ok(RestrictionOutcome::Unsatisfiable(crate::build::constant_like(eng, &f, false)));
    }
    let v0 = f.output.vtree;
    if f.vtree.node(v0).is_leaf() {
        // A literal has no internal pairs to drop.
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    // Both operands must share vtree structure; the walk reads indices in `f.vtree`.
    let r = f.vtree.lca(v0, care.output.vtree);
    if r != v0 && r != care.output.vtree {
        // Incomparable roots ⇒ disjoint variable regions ⇒ care can't constrain f.
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    let marks = Marking::walk(eng, &f, &care, r)?;
    if !marks.root_live {
        // care killed every model of f ⇒ f ∧ care = ∅.
        return Ok(RestrictionOutcome::Unsatisfiable(crate::build::constant_like(eng, &f, false)));
    }
    if marks.nothing_reachable_died(eng, &f)? {
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    Ok(RestrictionOutcome::Shrunk(marks.rebuild(eng, f)?))
}

/// Liveness marks over `f` produced by the `f × care` walk.
struct Marking {
    /// `[v.idx()][f-local]` — does this f-node survive under care?
    alive: Vec<Vec<bool>>,
    /// `[v.idx()][f-local]` — bit `k` set iff pair `k` of the f-node is live
    /// against some care pair; `u64::MAX` for a live node with more than 64
    /// pairs (no pair info: the rebuild keeps every pair of it).
    pair_alive: Vec<Vec<u64>>,
    /// Is the root pair live, i.e. is `f ∧ care` structurally non-false?
    root_live: bool,
}

/// The restriction entry point on a caller's engine.
impl crate::Engine {
    /// Run [`Tdd::restrict_to_care`](crate::Tdd::restrict_to_care) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn restrict_to_care(&self, f: Tdd, care: Tdd) -> Result<RestrictionOutcome, OperationError> {
        crate::apply::restrict_to_care::restrict_to_care_on(self, f, care)
    }
}
