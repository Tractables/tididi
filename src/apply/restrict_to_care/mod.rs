//! Restrict-to-care: prune `f` to the subgraph that survives under a "care" diagram.
//!
//! `restrict_to_care(f, care)` returns `g`, a **structural subgraph of `f`** — every node
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
//! The walk is stack-driven, visits at most `|f| · |care|` node pairs, and is
//! its discovery tables, marking rows and work stacks use the caller's limits.

mod mark;
mod rebuild;

use std::sync::Arc;

use crate::engine::Engine;
use crate::limits::OperationError;

use crate::diagram::Tdd;
use crate::vtree::Vtree;

/// Outcome of [`restrict_to_care`], so a caller can tell a no-op from a shrink
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
    /// canonical `⊥` over the operands' vtree is the smallest sound
    /// representative, and [`into_tdd`](Self::into_tdd) builds it. This is a
    /// change, not a no-op.
    Unsatisfiable(Arc<Vtree>),
}

impl RestrictionOutcome {
    /// Collapse to a concrete `g`, whichever arm it is.
    #[must_use]
    pub fn into_tdd(self) -> Tdd {
        match self {
            RestrictionOutcome::Unchanged(g) | RestrictionOutcome::Shrunk(g) => g,
            RestrictionOutcome::Unsatisfiable(vtree) => Tdd::zero(&vtree),
        }
    }
}

/// The implementation behind [`Engine::restrict_to_care`](crate::Engine::restrict_to_care).
fn restrict_to_care_on(eng: &Engine, f: Tdd, mut care: Tdd) -> Result<RestrictionOutcome, OperationError> {
    let _op = eng.limits().begin_operation();
    if f.is_zero() {
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    // Sound for any representation of `care`, since `g ∧ care == f ∧ care`
    // does not depend on it; the reduced one gives the walk fewer pairs.
    crate::reduce::try_reduce(eng, &mut care, crate::reduce::ReductionPlan::default())?;
    if care.is_zero() {
        // care ≡ ∅ ⇒ f ∧ care = ∅ ⇒ ⊥ is the smallest sound representative.
        return Ok(RestrictionOutcome::Unsatisfiable(Arc::clone(&f.vtree)));
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
        return Ok(RestrictionOutcome::Unsatisfiable(Arc::clone(&f.vtree)));
    }
    if marks.nothing_reachable_died(eng, &f)? {
        return Ok(RestrictionOutcome::Unchanged(f));
    }
    Ok(RestrictionOutcome::Shrunk(marks.rebuild(eng, &f)?))
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

/// Restrict `f` to the region `care` names, on a transient engine with no
/// limits armed.
///
/// Both operands are taken by value, as [`Engine::restrict_to_care`] takes them: `f`
/// rides back in whichever arm of the result it belongs to, so the
/// [`RestrictionOutcome::Unchanged`] arm hands the operand over rather than copying it,
/// and `care` is minimized before the walk. [`Engine::restrict_to_care`] is the same
/// operation on a caller's engine — it keeps the per-level buffers warm
/// between calls and reports a refusal rather than panicking on it — and
/// states the contract.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::apply::{restrict_to_care, RestrictionOutcome};
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, 2]);   // x1 ∨ x2
/// let care = Tdd::clause(&vtree, [1]);   // only x1 matters
///
/// // Outside the care region the result may differ from `f`; inside it agrees.
/// let g = restrict_to_care(f.clone(), care.clone()).into_tdd();
/// assert_eq!((g.clone() & care.clone()).model_count(), (f & care).model_count());
///
/// // Care that no model satisfies collapses the result.
/// let nothing = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [-1]);
/// let out = restrict_to_care(g, nothing);
/// assert!(matches!(out, RestrictionOutcome::Unsatisfiable(_)));
/// ```
///
/// # Panics
///
/// Panics if the orphan prune inside the rebuild is refused. Nothing is armed
/// on the transient engine, so the only refusal left is the allocator's.
#[must_use]
pub fn restrict_to_care(f: Tdd, care: Tdd) -> RestrictionOutcome {
    restrict_to_care_on(&Engine::new(), f, care).expect("restrict_to_care: refused with no limits armed")
}

/// The restriction entry point on a caller's engine.
impl crate::engine::Engine {
    /// Restriction (generalized cofactor) by dead-marking: a diagram `g` no
    /// larger than `f` with `g ∧ care == f ∧ care`. The module doc of
    /// [`restrict_to_care()`](crate::apply::restrict_to_care()) gives the algorithm.
    ///
    /// Both operands must be over the same vtree; this is not checked. Takes
    /// both by value. `f` rides back in whichever arm of the result it belongs
    /// to, so a caller that only wants the diagram calls
    /// [`RestrictionOutcome::into_tdd`] and one that wants to skip the epilogue matches
    /// on [`RestrictionOutcome::Unchanged`] — neither copies `f`. `care` is minimized
    /// before the walk on this engine, and dropped. A ⊥ `f` is [`RestrictionOutcome::Unchanged`]; a `care` with no
    /// model is [`RestrictionOutcome::Unsatisfiable`]. A [`RestrictionOutcome::Shrunk`] result
    /// counts correctly but is not canonical. Marginal levels are allowed in
    /// both operands: one of `f` is carried through as it is, one of `care`
    /// constrains nothing below it.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when a care reduction, walk, or prune
    /// reservation is refused, [`OperationError::Stopped`] on the armed
    /// deadline or a stop decision, and `f` is spent.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    /// use tididi::Engine;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2]); // x1 ∨ x2
    /// let care = Tdd::clause(&vtree, [1]); // x1
    /// let g = eng.restrict_to_care(f, care).unwrap().into_tdd();
    /// // Contract: g agrees with f wherever care holds, i.e. g ∧ x1 == f ∧ x1.
    /// let lhs = g & Tdd::clause(&vtree, [1]);
    /// let rhs = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [1]);
    /// assert_eq!(lhs.model_count(), rhs.model_count());
    /// ```
    pub fn restrict_to_care(&self, f: Tdd, care: Tdd) -> Result<RestrictionOutcome, OperationError> {
        crate::apply::restrict_to_care::restrict_to_care_on(self, f, care)
    }
}
