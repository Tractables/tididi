//! Conjunction of two diagrams via the compacting product construction.
//!
//! Given two diagrams over the same vtree, produces a diagram for their conjunction.
//! The product construction pairs every node from f with every node from g
//! at each vtree level, computes the conjunction of their input pairs, and
//! omits dead nodes (compaction). See `docs/tdd.md` for details.

use std::borrow::{Borrow, Cow};
use std::sync::Arc;

use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::restructure::EmbedError;
use crate::restructure::embed::{place, Free, Plan};
use crate::restructure::placement::push_free_level;
use super::CONJOIN_GRID;
use crate::diagram::{self, *};

pub(crate) mod budget;
mod child_lookup; // Representation-specialized child lookups (sparse-conjunction kernels)
use crate::Engine;
use crate::limits::OperationError;
use budget::*;

mod cell;
use cell::{
    RightColumns, CellCtx, ChildPlan, ColumnSlice,
    run_level_described_stream, run_level_rows_complete, run_level_rows_marginal, run_level_rows_marginal_sparse, run_level_rows_plain,
    run_level_rows_pairs, run_level_rows_single, run_level_rows_stream_count, run_level_rows_stream_words, run_level_rows_words, words_pay,
    run_level_rows_marginal_sparse_words, words_pay_marginal,
    PlainLookups, RowLoop, RowScratch, WordScratch,
};

mod sparse;
pub(crate) use sparse::SparseWorkspace;
use sparse::{apply_sparse_level, count_sparse_level, sum_sparse_level, CandidateFold, ChildSum, Complete, Made, Passthrough};

// An operand as the sweep holds it, owned or borrowed.
mod held;
use held::Held;

// Identity/constant-true detection and the per-level identity fast paths.
mod identity;
pub(crate) use identity::init_leaf_identity;
use identity::take_level_fast_path;

// The relabelling route, where one operand is one node with one pair.
mod relabel;
use relabel::{single_pair, take_relabel_level};

// Apply setup → `ApplyRun`.
mod setup;
use setup::{apply_and_setup, ApplyRun, LevelShape, Operands};

// Per-level marginal classification plan and dead-pair masks.
mod marginal_plan;
use marginal_plan::{ChildGrid, MarginalPlan, SidePlan, plan_marginal_level, build_side_masks};

// The leaf levels, before the bottom-up loop.
mod leaf_seed;
use leaf_seed::apply_leaf_levels;

mod scratch;
pub(crate) use scratch::ApplyScratch;
pub(crate) use setup::VtreeMask;
mod route;
use route::*;
mod grid_arena;
mod price;
mod products;
pub(in crate::apply::conjoin) use grid_arena::GridBase;
mod output;
use output::*;
mod quantify;
mod drive;
use drive::{apply_and_core, Conjoined, ConjoinMode};
use drive::Sweep;
mod filter;

mod liveness;

pub(crate) mod streaming_marginal;
use crate::value::StreamCache;
use streaming_marginal::{StreamEnv, StreamLevelState, build_stream_state, commit_stream_state};

/// Panicking conjunction used by `BitAnd` and test fixtures.
/// The checked entry point is [`and`].
pub(crate) fn apply_and(f: Tdd, g: Tdd) -> Tdd {
    and(f, g)
        .expect("apply_and: operation refused; use tididi::and to handle errors")
}

/// Conjoin owned operands, recycling their levels as the bottom-up walk
/// proceeds, and collapsing every subtree in `quantified` to the constant-true
/// node instead of building it — see [`quantify`].
///
/// The flag says whether the sweep ran, which is what decides whether the
/// caller's quantification has collapsed levels to account for: the shortcuts
/// for a false operand and for `f ∧ f` return without collapsing anything.
/// Operand validation and allocation, cancellation and output-cap errors follow
/// [`Engine::and`]. Both inputs are consumed on every outcome.
pub(crate) fn conjoin_on(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    quantified: VtreeMask<'_>,
) -> Result<(Tdd, bool), OperationError> {
    // Checked before the swap and the self-conjunction shortcut, both of which
    // can return without ever reaching the conjunction sweep.
    crate::apply::check_vtree(&f, &g)?;
    crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
    conjoin_checked(eng, f, g, VtreeMask::default(), quantified)
}

/// True when `f` and `g` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, ranges)` on every level is
/// sufficient. This is structural equality, not pointer identity — but it is
/// only sound when no level is marginal, since a marginal level hides its
/// content outside `nodes`/`pairs` where the structural test cannot see it.
fn is_self_conjunction(f: &Tdd, g: &Tdd) -> bool {
    same_levels(f, g, Operands::default())
}

/// [`is_self_conjunction`] for operands with free levels ([`ApplyRun::cone`]):
/// a free level compares as the level it stands for.
fn same_levels(f: &Tdd, g: &Tdd, free: Operands<VtreeMask<'_>>) -> bool {
    // The shortcut lets `f ∧ g` return `f.clone()` when the operands are the
    // same function. It is a pure perf optimization, never needed for
    // correctness. A marginal level clears `nodes`/`pairs` (integer-marginal) or
    // `pairs` (weight-marginal) and moves its real content into
    // `marginal_counts`/the external weight store — which this structural test
    // does not compare. Two operands agreeing on every explicit level but
    // differing in marginal mass (or holding a marginal×marginal unsound
    // schedule the callers debug-assert against) would compare equal and
    // silently drop one side's content. Bail whenever either operand carries any
    // marginal level.
    if f.levels.iter().any(|l| l.is_marginal()) || g.levels.iter().any(|l| l.is_marginal()) {
        return false;
    }
    same_structure(f, g, free)
}

/// [`same_levels`] for operands known to have no marginal level.
///
/// The levels are compared from the last index down, the internal levels
/// before the leaves, where operands that differ usually differ first.
fn same_structure(f: &Tdd, g: &Tdd, free: Operands<VtreeMask<'_>>) -> bool {
    // `ranges` too: equal nodes+pairs with a differently-arranged `ranges` table
    // is a different function.
    let same = |l1: &TddLevel, l2: &TddLevel| l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ranges == l2.ranges;
    let vtree = &f.vtree;
    let same_at = |t: usize| {
        let (l1, l2) = (&f.levels[t], &g.levels[t]);
        let at = VtreeIdx(t as u32);
        let internal = !vtree.node(at).is_leaf();
        match (internal && free.f.contains(t), internal && free.g.contains(t)) {
            (false, false) => same(l1, l2),
            (true, true) => true,
            (free_f, _) => {
                // The level a free one stands for has one node.
                let built = if free_f { l2 } else { l1 };
                built.nodes().len() == 1 && {
                    let mut standing = TddLevel::new();
                    push_free_level(&mut standing, vtree, at);
                    same(&standing, built)
                }
            }
        }
    };
    f.output == g.output && (0..f.levels.len().min(g.levels.len())).rev().all(same_at)
}

/// Build the free levels ([`ApplyRun::cone`]) of `tdd` that are still empty,
/// so that it is the diagram they stand for.
fn fill_free(tdd: &mut Tdd, free: VtreeMask<'_>) {
    if free.is_empty() {
        return;
    }
    let vtree = Arc::clone(&tdd.vtree);
    for (t, _, _) in vtree.internal_bottomup() {
        if free.contains(t.idx()) && tdd.levels[t.idx()].slot_count() == 0 {
            push_free_level(&mut tdd.levels[t.idx()], &vtree, t);
        }
    }
}

/// [`fill_free`] of an owned operand; a borrowed one has no free level.
fn fill_free_cow(tdd: &mut Cow<'_, Tdd>, free: VtreeMask<'_>) {
    match tdd {
        Cow::Owned(tdd) => fill_free(tdd, free),
        Cow::Borrowed(_) => debug_assert!(free.is_empty(), "a borrowed operand has no free level"),
    }
}

/// Hand an owned operand's level array back to the pool once the sweep is
/// done with it; a borrowed operand is the caller's.
fn recycle(eng: &Engine, tdd: Cow<'_, Tdd>) {
    if let Cow::Owned(mut tdd) = tdd {
        diagram::return_levels(eng, std::mem::take(&mut tdd.levels).into_vec());
    }
}

/// [`Engine::and`] after its entry, for operands each owned or borrowed
/// ([`conjoin_cow_as`]).
fn conjoin_either(eng: &Engine, mut f: Cow<'_, Tdd>, mut g: Cow<'_, Tdd>) -> Result<Tdd, OperationError> {
    crate::apply::check_vtree(&f, &g)?;
    prepare_owned_weights(&mut f, &mut g)?;
    conjoin_cow_as(eng, f, g, VtreeMask::default(), VtreeMask::default(), ConjoinMode::Build).map(|(out, _)| out.diagram())
}

/// Check that the operands share one weight interpretation, and give an
/// owned operand without weights beside a weighted one an empty store, as
/// [`prepare_weights`](crate::apply::prepare_weights) does; a borrowed one
/// is read as it is.
fn prepare_owned_weights(f: &mut Cow<'_, Tdd>, g: &mut Cow<'_, Tdd>) -> Result<(), OperationError> {
    let Some(empty) = crate::apply::shared_weights([&**f, &**g].into_iter())? else { return Ok(()) };
    fn owned<'a>(op: &'a mut Cow<'_, Tdd>) -> Option<&'a mut Tdd> {
        match op {
            Cow::Owned(op) => Some(op),
            Cow::Borrowed(_) => None,
        }
    }
    for op in [owned(f), owned(g)].into_iter().flatten() {
        op.weights.get_or_insert_with(|| empty.empty_like());
    }
    Ok(())
}

/// Make `g` the narrower operand, and say whether the two were swapped; the
/// free levels in `free` ([`ApplyRun::cone`]) follow their operands and are
/// one node wide.
///
/// The identity fast path tests `right_width == 1` first, so the narrower side
/// on the right takes it at more levels, and grid rows (width `right_width`)
/// get shorter. Only the owned entries swap; borrowed callers
/// track operands by side.
///
/// `widths` holds each operand's [`Tdd::max_width`].
fn narrower_right<T: Borrow<Tdd>>(f: &mut T, g: &mut T, free: &mut Operands<VtreeMask<'_>>, widths: Operands<usize>) -> bool {
    let width = |d: &Tdd, free: VtreeMask<'_>, built: usize| match built {
        0 => usize::from(!free.is_empty() && d.vtree.internal_bottomup().any(|(t, _, _)| free.contains(t.idx()))),
        built => built,
    };
    let swap = width((*g).borrow(), free.g, widths.g) > width((*f).borrow(), free.f, widths.f);
    if swap {
        std::mem::swap(f, g);
        std::mem::swap(&mut free.f, &mut free.g);
    }
    swap
}

/// [`conjoin_on`] after its operand checks, summing out the levels in
/// `targets`: for a caller that has already validated and weight-aligned the
/// operands.
pub(crate) fn conjoin_checked(
    eng: &Engine,
    f: Tdd,
    g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
) -> Result<(Tdd, bool), OperationError> {
    conjoin_checked_as(eng, f, g, targets, quantified, ConjoinMode::Build)
        .map(|(out, nonzero)| (out.diagram(), nonzero))
}

/// [`conjoin_checked`] with the sweep run in `mode`, [`ConjoinMode::Build`]
/// or [`ConjoinMode::Sum`]; the self-conjunction shortcut returns
/// [`Conjoined::Built`].
fn conjoin_checked_as(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    mode: ConjoinMode,
) -> Result<(Conjoined, bool), OperationError> {
    let widths = Operands { f: f.max_width(), g: g.max_width() };
    narrower_right(&mut f, &mut g, &mut Operands::default(), widths);
    // Self-conjunction: f ∧ f = f. The test is structural equality of every
    // explicit level, not pointer identity, and it declines on any marginal
    // level — see `is_self_conjunction`, where the soundness of both choices
    // is stated.
    if is_self_conjunction(&f, &g) {
        let _op = eng.limits().enter()?;
        diagram::return_levels(eng, std::mem::take(&mut g.levels).into_vec());
        return Ok((Conjoined::Built(f), false));
    }
    let zero = f.is_zero() || g.is_zero();
    let result = conjoin_recycling_as(eng, f, g, targets, quantified, None, mode);
    result.map(|out| (out, !zero))
}

/// [`conjoin_checked_as`] of operands each owned or borrowed: two owned ones
/// take it; a borrowed one is read in place, weights and marginal levels
/// included, and left as it was, for the result and work of the owned call.
fn conjoin_cow_as<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    g: Cow<'a, Tdd>,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    mode: ConjoinMode,
) -> Result<(Conjoined, bool), OperationError> {
    let (mut f, mut g) = match (f, g) {
        (Cow::Owned(f), Cow::Owned(g)) => return conjoin_checked_as(eng, f, g, targets, quantified, mode),
        operands => operands,
    };
    let widths = Operands { f: f.max_width(), g: g.max_width() };
    narrower_right(&mut f, &mut g, &mut Operands::default(), widths);
    if is_self_conjunction(&f, &g) {
        // f ∧ f = f, as two owned operands return it: f itself where owned,
        // a copy of it where borrowed.
        let _op = eng.limits().enter()?;
        let out = match f {
            Cow::Owned(out) => out,
            Cow::Borrowed(f) => f.try_clone_on(eng)?,
        };
        recycle(eng, g);
        return Ok((Conjoined::Built(out), false));
    }
    let zero = f.is_zero() || g.is_zero();
    let result = drive::apply_and_either(eng, &mut Held::of(&mut f), &mut Held::of(&mut g), targets, quantified, mode);
    recycle(eng, f);
    recycle(eng, g);
    result.map(|out| (out, !zero))
}

/// Run the sweep over `f` and `g`, then hand both operands' level arrays back
/// to the pool whatever the outcome.
fn conjoin_recycling(
    eng: &Engine,
    f: Tdd,
    g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
) -> Result<Tdd, OperationError> {
    conjoin_recycling_as(eng, f, g, targets, quantified, filter, ConjoinMode::Build)
        .map(Conjoined::diagram)
}

/// [`conjoin_recycling`] with the sweep run in `mode`.
fn conjoin_recycling_as(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
    mode: ConjoinMode,
) -> Result<Conjoined, OperationError> {
    let result = apply_and_core(eng, &mut f, &mut g, targets, quantified, filter, mode, Operands::default());
    diagram::return_levels(eng, std::mem::take(&mut f.levels).into_vec());
    diagram::return_levels(eng, std::mem::take(&mut g.levels).into_vec());
    result
}

/// Return the conjunction of two diagrams sharing the same vtree allocation.
///
/// Uses the shared vtree's execution context automatically.
///
/// Each operand is a [`Tdd`], consumed on success and on error, or a
/// `&Tdd`, read in place and left as it is. Where the result takes a level
/// of an operand as it stands, such as one the other operand is constant
/// true under, an owned operand's level is moved and a borrowed one's
/// copied: borrowing costs those levels and a borrowed operand's weight
/// store, where cloning first copies every level. Owned or borrowed, the
/// result is the same diagram for the same work. It is correct for counting
/// but may retain unreachable nodes and twins; [`Tdd::minimize`] establishes
/// canonical form when required.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, literal, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let either = Tdd::clause(&vtree, [1, 2])?;
/// let not_third = literal(&vtree, -3)?;
/// let f = and(&either, not_third)?;
/// assert_eq!(f.model_count()?, 3u32.into());
/// assert_eq!(either.model_count()?, 6u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// # Weights and marginal levels
///
/// Attached weight tables and arithmetic must agree. A structural operand
/// without weights inherits the other operand's table; stored integer counts
/// cannot be reweighted. Marginal levels remain marginal, and conjunction is
/// valid there only when the other operand imposes no further constraint on
/// the summed-out variables. When both operands are marginal at a level, the
/// caller must ensure one represents the constant-true function on that subtree;
/// this condition is checked in debug builds.
///
/// # Errors
///
/// [`OperationError::VtreeMismatch`] for different vtree allocations,
/// [`OperationError::IncompatibleWeights`] for different weight interpretations,
/// or [`OperationError::MarginalLevel`] when a structural operand constrains
/// variables the other has summed out.
///
/// Allocation refusal is reported as [`OperationError::OverBudget`].
/// For explicit resource limits, use [`Context::with_limits`](crate::Context::with_limits)
/// and the supplied engine's operations.
pub fn and<'f, 'g>(f: impl Into<Cow<'f, Tdd>>, g: impl Into<Cow<'g, Tdd>>) -> Result<Tdd, OperationError> {
    let (f, g) = (f.into(), g.into());
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.and(f, g))
}

/// A conjunction [`Engine::and_restoring`](crate::Engine::and_restoring)
/// refused, with both operands as they were given: an owned one unchanged,
/// a borrowed one as the borrow.
#[derive(Debug)]
pub struct AndRefused<'a> {
    /// Why the conjunction was refused.
    pub error: OperationError,
    /// The first operand, unchanged.
    pub f: Cow<'a, Tdd>,
    /// The second operand, unchanged.
    pub g: Cow<'a, Tdd>,
}

/// A conjunction [`Engine::and_onto`](crate::Engine::and_onto) refused, with
/// each owned operand on the destination where it was placed there before
/// the refusal and as it was given otherwise, and each borrowed one as the
/// borrow.
#[derive(Debug)]
pub struct AndOntoRefused<'a> {
    /// Why the placement or the conjunction was refused.
    pub error: EmbedError,
    /// The first operand.
    pub f: Cow<'a, Tdd>,
    /// The second operand.
    pub g: Cow<'a, Tdd>,
}

/// [`Engine::and_restoring`](crate::Engine::and_restoring) after its entry
/// and vtree check, for operands whose free levels ([`ApplyRun::cone`]) are
/// in `free`: refused, both come back as they were given, every free level
/// built.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn conjoin_kept<'a>(
    eng: &Engine,
    mut f: Cow<'a, Tdd>,
    mut g: Cow<'a, Tdd>,
    mut free: Operands<VtreeMask<'_>>,
) -> Result<Tdd, AndRefused<'a>> {
    // Whether a level is marginal, and the widest level, in one pass.
    let scan = |t: &Tdd| t.levels.iter().fold((false, 0), |(marginal, width), l| {
        (marginal | l.is_marginal(), width.max(l.slot_count()))
    });
    let ((f_marginal, f_width), (g_marginal, g_width)) = (scan(&f), scan(&g));
    let weighted_or_marginal = f.weights.is_some() || f_marginal || g.weights.is_some() || g_marginal;
    let swapped = narrower_right(&mut f, &mut g, &mut free, Operands { f: f_width, g: g_width });
    let give_back = |error, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>| {
        fill_free_cow(&mut f, free.f);
        fill_free_cow(&mut g, free.g);
        if swapped {
            std::mem::swap(&mut f, &mut g);
        }
        AndRefused { error, f, g }
    };
    if weighted_or_marginal && let Err(error) = crate::apply::shared_weights([&*f, &*g].into_iter()) {
        return Err(give_back(error, f, g));
    }
    if !f_marginal && !g_marginal && same_structure(&f, &g, free) {
        // f ∧ f = f, as two owned operands return it: f itself where owned,
        // its free levels built, a copy of it where borrowed.
        let mut out = match f {
            Cow::Owned(mut out) => {
                fill_free(&mut out, free.f);
                out
            }
            Cow::Borrowed(borrowed) => match borrowed.try_clone_on(eng) {
                Ok(out) => out,
                Err(error) => return Err(give_back(error, Cow::Borrowed(borrowed), g)),
            },
        };
        if out.weights.is_none() {
            out.weights = g.weights.as_ref().map(crate::diagram::WeightStore::empty_like);
        }
        recycle(eng, g);
        return Ok(out);
    }
    // A refusal puts an owned operand's levels back as they were, through
    // accesses that forget its certificate, closed form and level counts:
    // they are read here and taken back with the levels. An operand placed
    // with free levels comes back with them built, a diagram of its own.
    let known = |tdd: &Cow<'_, Tdd>, free: VtreeMask<'_>| match tdd {
        Cow::Owned(tdd) if free.is_empty() => Some(tdd.levels.known()),
        _ => None,
    };
    let (f_known, g_known) = (known(&f, free.f), known(&g, free.g));
    let cone_free = if weighted_or_marginal {
        fill_free_cow(&mut f, free.f);
        fill_free_cow(&mut g, free.g);
        Operands::default()
    } else {
        free
    };
    match drive::apply_and_kept(eng, &mut Held::of(&mut f), &mut Held::of(&mut g), cone_free) {
        Ok(out) => {
            recycle(eng, f);
            recycle(eng, g);
            Ok(out)
        }
        Err(error) => {
            reinstate_known(&mut f, f_known);
            reinstate_known(&mut g, g_known);
            Err(give_back(error, f, g))
        }
    }
}

/// Take back what an owned operand given whole was known to be before a
/// refused conjunction put its levels back ([`conjoin_kept`]).
fn reinstate_known(tdd: &mut Cow<'_, Tdd>, known: Option<diagram::Known>) {
    if let (Cow::Owned(tdd), Some(known)) = (tdd, known) {
        tdd.levels.reinstate_known(known);
    }
}

/// An operand [`place_onto`] placed, with the plan it was placed by, or why
/// it was refused, with the operand as it was given.
type Placed<'a> = Result<(Cow<'a, Tdd>, Option<Plan>), (EmbedError, Cow<'a, Tdd>)>;

/// Place `tdd` on `into` for [`Engine::and_onto`](crate::Engine::and_onto),
/// free levels left empty, with the plan it was placed by; as it is, with no
/// plan, when it is on `into` already. A borrowed operand not on `into` is
/// copied into the placement ([`place`]). `begun` says whether a phase of
/// the operation has begun, and is set. With `mirror`, up to mirrors
/// ([`Engine::and_onto_mirrored`](crate::Engine::and_onto_mirrored)).
/// Refused, the operand comes back as it was given.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn place_onto<'a>(
    eng: &Engine,
    tdd: Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    begun: &mut bool,
    mirror: bool,
) -> Placed<'a> {
    if Arc::ptr_eq(tdd.vtree(), into) {
        return Ok((tdd, None));
    }
    if let Err(error) = begin_phase(eng, *begun) {
        return Err((error.into(), tdd));
    }
    *begun = true;
    place(eng, tdd, into, map, Free::Leave, mirror).map(|(tdd, plan)| (Cow::Owned(tdd), Some(plan)))
}

/// [`Engine::and_onto`](crate::Engine::and_onto), and with `mirror`
/// [`Engine::and_onto_mirrored`](crate::Engine::and_onto_mirrored).
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn conjoin_onto<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    f_map: impl Fn(VarId) -> VarId,
    g: Cow<'a, Tdd>,
    g_map: impl Fn(VarId) -> VarId,
    into: &Arc<Vtree>,
    mirror: bool,
) -> Result<Tdd, AndOntoRefused<'a>> {
    let _op = match eng.limits().enter() {
        Ok(op) => op,
        Err(error) => return Err(AndOntoRefused { error: error.into(), f, g }),
    };
    // A borrowed operand comes back as the borrow, whatever its placed
    // copy went through.
    let given = |tdd: &Cow<'a, Tdd>| match tdd {
        Cow::Borrowed(tdd) => Some(*tdd),
        Cow::Owned(_) => None,
    };
    let (f_given, g_given) = (given(&f), given(&g));
    let back = |tdd: Cow<'a, Tdd>, given: Option<&'a Tdd>| given.map_or(tdd, Cow::Borrowed);
    let mut begun = false;
    let (f, f_plan) = match place_onto(eng, f, into, f_map, &mut begun, mirror) {
        Ok(placed) => placed,
        Err((error, f)) => return Err(AndOntoRefused { error, f, g }),
    };
    let free_f = free_levels(&f, f_plan.as_ref());
    let (g, g_plan) = match place_onto(eng, g, into, g_map, &mut begun, mirror) {
        Ok(placed) => placed,
        Err((error, g)) => {
            let mut f = f;
            fill_free_cow(&mut f, free_f);
            return Err(AndOntoRefused { error, f: back(f, f_given), g });
        }
    };
    let free = Operands { f: free_f, g: free_levels(&g, g_plan.as_ref()) };
    let (mut f, mut g) = (f, g);
    if let Err(error) = begin_phase(eng, begun) {
        fill_free_cow(&mut f, free.f);
        fill_free_cow(&mut g, free.g);
        return Err(AndOntoRefused { error: error.into(), f: back(f, f_given), g: back(g, g_given) });
    }
    let conjoined = conjoin_kept(eng, f, g, free)
        .map_err(|r| AndOntoRefused { error: r.error.into(), f: back(r.f, f_given), g: back(r.g, g_given) });
    for plan in [f_plan, g_plan].into_iter().flatten() {
        plan.recycle(eng);
    }
    conjoined
}

/// Begin a phase of [`Engine::and_onto`](crate::Engine::and_onto) as the
/// operation it stands for begins, after the first: the meters start from
/// zero and the stop is tested. The first is the operation's own entry.
fn begin_phase(eng: &Engine, begun: bool) -> Result<(), OperationError> {
    match begun {
        true => eng.limits().next_phase(),
        false => Ok(()),
    }
}

/// The free levels of a diagram [`place_onto`] placed by `plan`: none for
/// one it did not place, or for the false diagram, which has no levels.
fn free_levels<'a>(tdd: &Tdd, plan: Option<&'a Plan>) -> VtreeMask<'a> {
    VtreeMask::new(plan.filter(|_| !tdd.is_zero()).map(|plan| &plan.free[..]))
}

impl crate::Engine {
    /// Run [`and`] using this batch's scratch and resource limits; each
    /// operand owned or borrowed, as [`and`] says.
    ///
    /// A copy that a borrowed operand costs (a level the result takes from
    /// it, and its weight store) is charged to the memory budget, not to the
    /// work clock.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and<'f, 'g>(&self, f: impl Into<Cow<'f, Tdd>>, g: impl Into<Cow<'g, Tdd>>) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        conjoin_either(self, f.into(), g.into())
    }

    /// Run [`Engine::and`], giving both operands back unchanged when it is
    /// refused.
    ///
    /// A caller that retries after a stop would otherwise clone both operands
    /// before each conjunction. Here the sweep keeps the operands whole until
    /// the result exists, which holds their storage as long as a clone would,
    /// without the copy. Operands with a weight table or a marginal level are
    /// read in place. Each operand is owned or borrowed, at the costs
    /// [`and`] states; a borrowed one comes back as the borrow. An owned one
    /// comes back as reduced as it was given and with the level counts it
    /// kept ([`Engine::attach_level_counts`]), so what is done with it next
    /// costs what it would have.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::limits::{LimitConfig, StopCallback, StopDecision};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [-2, 3])?;
    /// let refused = {
    ///     let stop = StopCallback::new(|_, _| StopDecision::Stop);
    ///     let _stopped = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(stop)));
    ///     engine.and_restoring(f, g).expect_err("every poll stops")
    /// };
    /// assert_eq!(engine.model_count(&refused.f)?, 12u32.into());
    /// let both = engine.and_restoring(refused.f, refused.g).map_err(|r| r.error)?;
    /// assert_eq!(engine.model_count(&both)?, 8u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`Engine::and`], each with both operands.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn and_restoring<'a>(
        &self,
        f: impl Into<Cow<'a, Tdd>>,
        g: impl Into<Cow<'a, Tdd>>,
    ) -> Result<Tdd, AndRefused<'a>> {
        let (f, g) = (f.into(), g.into());
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(error) => return Err(AndRefused { error, f, g }),
        };
        if let Err(error) = crate::apply::check_vtree(&f, &g) {
            return Err(AndRefused { error, f, g });
        }
        conjoin_kept(self, f, g, Operands::default())
    }

    /// [`Engine::and_restoring`] of `f` and `g` placed on `into` as
    /// [`Engine::embed_moving`] places them, each renamed by its own map,
    /// without building the levels a placement adds where none of its
    /// variables is under them.
    ///
    /// Such a level is the constant true level, one node true on both sides,
    /// so the conjunction takes the other operand's level there; it builds
    /// one only where the other operand is the constant true level too. The
    /// result is the diagram `and_restoring` returns on the two embeddings,
    /// node for node. An operand already on `into` is taken as it is, and
    /// its map is not read. Each operand is owned or borrowed, at the costs
    /// [`and`] states; a borrowed operand on `into` is read in place, and
    /// one that is not is copied whole and the copy placed.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let pair = Arc::new(Vtree::balanced(2));
    /// let wide = Arc::new(Vtree::balanced(4));
    /// let low = engine.clause(&pair, [1, 2])?;
    /// let high = engine.clause(&pair, [-1, -2])?;
    /// let both = engine
    ///     .and_onto(low, |v| v, high, |v| VarId(v.0 + 2), &wide)
    ///     .map_err(|r| r.error)?;
    /// assert_eq!(engine.model_count(&both)?, 9u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Those of [`Engine::embed_moving`] and [`Engine::and_restoring`]. Each
    /// owned operand comes back on `into` when it was placed there before
    /// the refusal and as it was given otherwise; its [`Tdd::vtree`] tells
    /// which. A borrowed operand comes back as the borrow.
    /// The placements and the conjunction are metered as the operations they
    /// stand for are, each from zero, with the stop tested as each begins.
    /// The work clock is charged as for the two embeddings and the
    /// conjunction, less one unit for each level a placement leaves free:
    /// the conjunction builds such a level only where it carries it as an
    /// identity level, which it charges nothing.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn and_onto<'a>(
        &self,
        f: impl Into<Cow<'a, Tdd>>,
        f_map: impl Fn(VarId) -> VarId,
        g: impl Into<Cow<'a, Tdd>>,
        g_map: impl Fn(VarId) -> VarId,
        into: &Arc<Vtree>,
    ) -> Result<Tdd, AndOntoRefused<'a>> {
        conjoin_onto(self, f.into(), f_map, g.into(), g_map, into, false)
    }

    /// [`Engine::and_onto`], where `into` may hold either operand's vtree
    /// shape up to mirrors: [`Engine::and_restoring`] of `f` and `g` placed
    /// as [`Engine::embed_moving_mirrored`] places them, free levels left
    /// as `and_onto` leaves them. A level whose image has its children the
    /// other way round has the sides of its pairs exchanged in place.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // x1 ∨ x2 on (x1 x2), placed with x1 on the right leaf of (x3 x4).
    /// let engine = Engine::new();
    /// let pair = Arc::new(Vtree::linear(2));
    /// let wide = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&pair, [1, 2])?;
    /// let g = engine.clause(&pair, [-1, -2])?;
    /// assert!(engine.and_onto(f.clone(), |v| VarId(5 - v.0), g.clone(), |v| v, &wide).is_err());
    /// let both = engine
    ///     .and_onto_mirrored(f, |v| VarId(5 - v.0), g, |v| v, &wide)
    ///     .map_err(|r| r.error)?;
    /// assert_eq!(engine.model_count(&both)?, 9u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Engine::and_onto`], with [`Engine::embed_moving_mirrored`]'s for
    /// the placements.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn and_onto_mirrored<'a>(
        &self,
        f: impl Into<Cow<'a, Tdd>>,
        f_map: impl Fn(VarId) -> VarId,
        g: impl Into<Cow<'a, Tdd>>,
        g_map: impl Fn(VarId) -> VarId,
        into: &Arc<Vtree>,
    ) -> Result<Tdd, AndOntoRefused<'a>> {
        conjoin_onto(self, f.into(), f_map, g.into(), g_map, into, true)
    }

    /// Conjoin two diagrams and replace selected subtrees with marginal values.
    ///
    /// Each operand is owned or borrowed, as [`and`] says: a borrowed one is
    /// read in place, weights and summed-out levels included, and costs the
    /// levels the result takes from it and a copy of its weight store.
    ///
    /// `targets` names vtree nodes; order and duplicates do not matter. Internal
    /// targets can be summed out during product construction; leaf targets are
    /// handled afterward. The result preserves the conjunction's count, or its
    /// fixed weighted value when weights are attached.
    ///
    /// Every target's remaining structure is discarded, including after identity
    /// and self-conjunction shortcuts; the false diagram stays false. See
    /// [`Tdd::marginalize_levels`] for the operations
    /// that remain valid after structure is discarded.
    ///
    /// A target built as structure is summed out once the conjunction is
    /// built, and the pairs above it that then name the same node on their
    /// other side are fused into one. With a single target on an unweighted
    /// conjunction, where each operand has one node at the target's parent
    /// and the sparse route builds that one product, the parent instead adds
    /// each pair's count into its fused pair as the pair is found, so the
    /// pairs the fusion would remove are never stored; the diagram is the
    /// one the other order leaves.
    ///
    /// Under [`Arithmetic::ScaledFloat`](crate::diagram::Arithmetic::ScaledFloat)
    /// a target summed out during product construction holds its values in
    /// 8 bytes each while they lie within an `f64`'s range of one power of
    /// two, 16 otherwise, and the weight store receives only the values of
    /// targets whose parent is not also summed out: naming every finished
    /// level of a weighted sum as a target keeps the levels below the
    /// topmost out of the store.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = vtree.children(vtree.root());
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// let counted = engine.and_marginalizing(&f, g, &[left])?;
    /// assert!(counted.level(left).is_marginal());
    /// assert_eq!(engine.model_count(&counted)?, 9u32.into());
    /// assert_eq!(engine.model_count(&f)?, 12u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The operand and resource errors of [`Engine::and`], plus
    /// [`OperationError::LevelNotInVtree`] for an invalid target, checked before
    /// product construction. An owned operand is consumed on every outcome,
    /// and a borrowed one is left as it was.
    pub fn and_marginalizing<'f, 'g>(
        &self,
        f: impl Into<Cow<'f, Tdd>>,
        g: impl Into<Cow<'g, Tdd>>,
        targets: &[VtreeIdx],
    ) -> Result<Tdd, OperationError> {
        let (mut f, mut g) = (f.into(), g.into());
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        f.check_level_indices(targets)?;
        prepare_owned_weights(&mut f, &mut g)?;
        // The apply core asks "is level `t` a target?" once per level it emits,
        // so the membership array is derived here, once, at the cost the caller
        // would pay to build it.
        let vtree = Arc::clone(f.vtree());
        let mut mask = Vec::new();
        self.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf();
        }
        collapse_target_subtrees(self, &vtree, &f, &g, &mut mask);
        // One internal target, on an unweighted conjunction: where each
        // operand has one node at the target's parent, that level may sum
        // the target out as it is built (`ConjoinMode::Sum`).
        let summable = if f.weights.is_none() && g.weights.is_none() && !two_step_forced() {
            let mut it = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| m.then_some(VtreeIdx(i as u32)));
            match (it.next(), it.next()) {
                (Some(c), None)
                    if vtree.node(c).parent().is_some()
                        && targets.iter().all(|&t| {
                            !vtree.node(t).is_leaf()
                                || std::iter::successors(vtree.node(t).parent(), |&x| {
                                    vtree.node(x).parent()
                                })
                                .any(|x| x == c)
                        }) =>
                {
                    Some(c)
                }
                _ => None,
            }
        } else {
            None
        };
        let mode = match summable {
            Some(c) => ConjoinMode::Sum(c),
            None => ConjoinMode::Build,
        };
        let summed = if mask.contains(&true) { VtreeMask::new(Some(&mask)) } else { VtreeMask::default() };
        let (out, _) = conjoin_cow_as(self, f, g, summed, VtreeMask::default(), mode)?;
        let mut out = match out {
            // The root summed the target out as it found its pairs, leaving
            // what the pass below and its pair fusion leave but for the
            // value slots no reference names any more (see
            // `sparse::ChildSum::write_root`).
            Conjoined::Summed(mut out) => {
                note_summed();
                for &t in targets {
                    if vtree.node(t).is_leaf() && !out.level(t).is_marginal() {
                        crate::marginal::marginalize_leaf_inline(&mut out, t, &vtree);
                    }
                }
                if let Some(parent) = summable.and_then(|c| vtree.node(c).parent()) {
                    #[cfg(debug_assertions)]
                    crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(
                        &out, Some(&[parent]), "and_marginalizing",
                    );
                    crate::reduce::contract::pair_fusion::record_fused(self, &mut out, parent);
                }
                crate::reduce::slot_prune::prune_value_slots(self, &mut out)?;
                return Ok(out);
            }
            other => other.diagram(),
        };
        // Streaming can finish every target; only retained structure needs the pass.
        if !out.is_zero() && targets.iter().any(|&t| !out.level(t).is_marginal()) {
            self.marginalize_levels(&mut out, targets)?;
        }
        Ok(out)
    }

    /// Count the models of `f ∧ g` without keeping the conjunction.
    ///
    /// The result is `model_count(and_marginalizing(f, g, targets))`:
    /// `targets` are summed out during product construction as there, which
    /// changes how much structure is built, never the count. Where the root
    /// of the conjunction is one product built by the sparse route, its
    /// pairs are folded into the count as they are found instead of being
    /// stored, so the largest level of a join that ends in a count is never
    /// held. Where both operands are one node at the root, the count may
    /// also skip one child of the root: it is summed from that child's own
    /// children, pair by pair against the other child's counts, when that
    /// walk is priced below building the child. Otherwise the conjunction is
    /// built and counted.
    ///
    /// Integer counts only: with weights attached, the result is the count
    /// of the conjunction built with them, as [`Engine::model_count`] gives it.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// assert_eq!(engine.and_model_count(f, g, &[])?, 9u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`Engine::and_marginalizing`] and of
    /// [`Engine::model_count`]. Both operands are consumed on every outcome.
    pub fn and_model_count(
        &self,
        mut f: Tdd,
        mut g: Tdd,
        targets: &[VtreeIdx],
    ) -> Result<num_bigint::BigUint, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        f.check_level_indices(targets)?;
        if f.weights.is_some() || g.weights.is_some() {
            let out = self.and_marginalizing(f, g, targets)?;
            return self.model_count(&out);
        }
        crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
        let vtree = Arc::clone(f.vtree());
        let mut mask = Vec::new();
        self.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
        // A target at the root sums the whole conjunction into its count,
        // which is what is returned anyway: it is dropped, so the root can be
        // counted instead of built.
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf() && t != vtree.root();
        }
        collapse_target_subtrees(self, &vtree, &f, &g, &mut mask);
        // With no target left the sweep is given none, so no route reads a
        // set mask that names nothing as one that sums levels out.
        let summed = if mask.contains(&true) { VtreeMask::new(Some(&mask)) } else { VtreeMask::default() };
        // The swap and the self-conjunction shortcut of `conjoin_checked`.
        if g.max_width() > f.max_width() {
            std::mem::swap(&mut f, &mut g);
        }
        if is_self_conjunction(&f, &g) {
            let count = self.model_count(&f);
            diagram::return_levels(self, std::mem::take(&mut g.levels).into_vec());
            return count;
        }
        let result = apply_and_core(
            self, &mut f, &mut g, summed, VtreeMask::default(), None, ConjoinMode::Count,
            Operands::default(),
        );
        diagram::return_levels(self, std::mem::take(&mut f.levels).into_vec());
        diagram::return_levels(self, std::mem::take(&mut g.levels).into_vec());
        match result? {
            Conjoined::Counted(count) => Ok(count),
            Conjoined::Built(out) | Conjoined::Summed(out) => self.model_count(&out),
        }
    }
}

/// For each maximal internal target `c` whose subtree is scheduled in `mask`
/// (or already marginal in an operand), clear `mask` on `c`'s strict
/// descendants — and on `c` itself when the subtree already holds a marginal
/// level — when `c`'s subtree has a structural level above the sparse floor or
/// `c`'s parent is a one-node root that can sum `c` out: the subtree can then
/// use the relabelling and sparse routes on the way up to `c`, and `c`'s
/// parent can sum `c` out at the root (`ConjoinMode::Sum`).
fn collapse_target_subtrees(
    eng: &Engine,
    vtree: &Vtree,
    f: &Tdd,
    g: &Tdd,
    mask: &mut [bool],
) {
    let weighted = f.weights.is_some() || g.weights.is_some();
    if weighted && mask[vtree.root().idx()] {
        return;
    }
    let min_grid = sparse::sparse_thresholds(eng.limits()).min_grid as u64;
    let is_big = |u: VtreeIdx| {
        let (fl, gl) = (f.level(u), g.level(u));
        (fl.slot_count() as u64) * (gl.slot_count() as u64) > min_grid
            || price::combinations_bound(fl, gl) > min_grid as u128
    };
    let is_scheduled = |c: VtreeIdx, mask: &[bool]| {
        mask[c.idx()]
            && vtree.subtree(c).all(|u| {
                vtree.node(u).is_leaf()
                    || mask[u.idx()]
                    || f.level(u).is_marginal()
                    || g.level(u).is_marginal()
            })
    };
    for c in vtree.bottomup().rev() {
        if c == vtree.root() || !is_scheduled(c, mask) {
            continue;
        }
        let is_relabel = |u: VtreeIdx| {
            single_pair(f, u.idx(), f.level(u).slot_count()).is_some()
                || single_pair(g, u.idx(), g.level(u).slot_count()).is_some()
        };
        let whole_structural = vtree
            .subtree(c)
            .all(|u| !f.level(u).is_marginal() && !g.level(u).is_marginal());
        let marginal_spine_relabels = whole_structural
            || vtree.subtree(c).all(|u| {
                f.level(u).is_marginal()
                    || g.level(u).is_marginal()
                    || is_relabel(u)
                    || vtree
                        .subtree(u)
                        .all(|x| !f.level(x).is_marginal() && !g.level(x).is_marginal())
            });
        if weighted && !marginal_spine_relabels {
            continue;
        }
        let sibling_marginal = vtree.node(c).parent().is_some_and(|p| {
            let (l, r) = vtree.children(p);
            let s = if l == c { r } else { l };
            f.level(s).is_marginal() || g.level(s).is_marginal() || mask[s.idx()]
        });
        let can_sum_at_parent = !weighted
            && whole_structural
            && vtree.node(c).parent().is_some_and(|p| {
                f.level(p).slot_count() == 1
                    && g.level(p).slot_count() == 1
                    && (vtree.subtree(c).all(|u| vtree.node(u).is_leaf() || f.level(u).slot_count() == 1)
                        || vtree.subtree(c).all(|u| vtree.node(u).is_leaf() || g.level(u).slot_count() == 1))
                    && (is_big(p) || {
                        let (l, r) = vtree.children(p);
                        let kept = if l == c { r } else { l };
                        vtree.subtree(kept).any(|u| !vtree.node(u).is_leaf() && is_big(u))
                    })
            });
        let has_big = vtree.subtree(c).any(|u| {
            !vtree.node(u).is_leaf()
                && !f.level(u).is_marginal()
                && !g.level(u).is_marginal()
                && (!weighted || (f.level(u).implicit().is_none() && g.level(u).implicit().is_none()))
                && is_big(u)
        });
        let relabels_c = !can_sum_at_parent && is_relabel(c);
        let keep_root = whole_structural && !sibling_marginal && !relabels_c;
        if has_big || can_sum_at_parent {
            for d in vtree.subtree(c) {
                if d != c || !keep_root {
                    mask[d.idx()] = false;
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;

// A test sends `and_marginalizing` down the two-step path to compare the
// root that sums its child out against it, and counts the roots that did.
#[cfg(test)]
use tests::{note_summed, two_step_forced};

#[cfg(not(test))]
#[inline(always)]
fn two_step_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_summed() {}

// A test reads every child side from the grid, as the oracle for the
// arithmetic lookups on complete children, and counts the levels that read a
// side by arithmetic, the meter charges of their reserved arenas, the levels
// that read a lone g pair in every cell, and those that read one pair a
// column; and reads each cell's column, as the oracle for the last.
#[cfg(test)]
use tests::{grid_lookups_forced, lone_columns_forced_off, note_lone_columns, note_lone_pair, note_lookups, note_scheduled_charge, note_whole_row};

#[cfg(not(test))]
#[inline(always)]
fn grid_lookups_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_lookups(_lookups: PlainLookups) {}

#[cfg(not(test))]
#[inline(always)]
fn note_scheduled_charge() {}

#[cfg(not(test))]
#[inline(always)]
fn note_lone_pair() {}

#[cfg(not(test))]
#[inline(always)]
fn note_lone_columns() {}

#[cfg(not(test))]
#[inline(always)]
fn note_whole_row() {}

#[cfg(not(test))]
#[inline(always)]
fn lone_columns_forced_off() -> bool {
    false
}

// A test builds no dead-pair masks with one multi-pair operand, as the oracle
// for those masks, and counts the levels that build them.
#[cfg(test)]
use tests::{note_one_sided_masks, one_sided_masks_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn one_sided_masks_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_one_sided_masks() {}

// A test builds no reach or row columns, as the oracle for the rows that
// read them, or no row columns, and counts those rows.
#[cfg(test)]
use tests::{note_reach_columns_row, note_row_columns_row, reach_columns_forced_off, row_columns_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn reach_columns_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn row_columns_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_reach_columns_row() {}

// A test runs every level of one pair a node through the row loop of any
// level, as the oracle for the loop of one-pair cells, and counts the
// levels that loop built.
#[cfg(test)]
use tests::{note_single_rows, single_rows_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn single_rows_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_single_rows() {}

// A test reads the columns of every level read by pairs of many-pair
// columns off the tables of their bytes' columns, and counts those levels.
#[cfg(test)]
use tests::{note_reach_tables, reach_tables_forced};

#[cfg(not(test))]
#[inline(always)]
fn reach_tables_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_reach_tables() {}

// A test builds every level of one-pair cells on their row loop, as the
// oracle for the levels written as a description of relayed right
// children, and counts those levels.
#[cfg(test)]
use tests::{note_relayed, relay_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn relay_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_relayed() {}

#[cfg(not(test))]
#[inline(always)]
fn note_row_columns_row(_whole: bool) {}

// A test builds levels with their arenas left to grow, as the oracle for
// the levels sized ahead from the first share of their walk, sizes small
// levels ahead from a larger share, and counts the levels sized ahead by
// route; it seeds the dense route's pair arenas small, so that the levels
// of small diagrams outgrow them.
#[cfg(test)]
use tests::{ahead_after_forced, ahead_forced_off, note_ahead, seed_pairs_forced};

#[cfg(not(test))]
#[inline(always)]
fn ahead_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn ahead_after_forced() -> Option<usize> {
    None
}

#[cfg(not(test))]
#[inline(always)]
fn note_ahead(_sparse: bool) {}

#[cfg(not(test))]
#[inline(always)]
fn seed_pairs_forced() -> Option<usize> {
    None
}

// A test runs every masked level on the row loop of any level, as the
// oracle for the levels read by pairs, and counts those levels.
#[cfg(test)]
use tests::{note_pair_row_unrolled, note_pair_rows, pair_rows_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn pair_rows_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_pair_rows(_complete: bool, _one_pair: bool) {}

#[cfg(not(test))]
#[inline(always)]
fn note_pair_row_unrolled() {}

// A test reads the cells of every level that can be read by g-pair vectors
// by them, or of none, as their oracle, writes every such row's live cells
// off the sorted list of its live candidates, and counts those levels.
#[cfg(test)]
use tests::{note_word_cells, word_cells_forced, word_rows_sorted_forced};

#[cfg(not(test))]
#[inline(always)]
fn word_cells_forced() -> Option<bool> {
    None
}

#[cfg(not(test))]
#[inline(always)]
fn word_rows_sorted_forced() -> bool {
    false
}

// A test lowers the stride of the dense levels' poll gates.
#[cfg(test)]
use tests::dense_poll_stride_forced;

#[cfg(not(test))]
#[inline(always)]
fn dense_poll_stride_forced() -> Option<u64> {
    None
}

#[cfg(not(test))]
#[inline(always)]
fn note_word_cells(_what: WordNote) {}

/// What a test counts of the levels read by g-pair vectors.
#[derive(Clone, Copy)]
enum WordNote {
    /// A level of two grids.
    Grids,
    /// A level with a complete side.
    Complete,
    /// A cell walked in the grouped order.
    Grouped,
    /// A row read by vectors.
    Row,
    /// A row read by vectors that walks its live cells alone.
    Cells,
    /// A row read by vectors that writes its live cells off the sorted
    /// list of its live candidates.
    Sorted,
    /// A streaming target's level, and one read through marginal lookups.
    Stream,
    StreamMarginal,
    /// A sparse-marginal level.
    SparseMarginal,
}

// A test counts the times a level's pairs arena grew.
#[cfg(test)]
use tests::note_pairs_grown;

#[cfg(not(test))]
#[inline(always)]
fn note_pairs_grown() {}

// A test sends every level a sparse walk would take from f through
// the walk's writes, as the oracle the move is checked against, and counts
// the levels it took.
#[cfg(test)]
use tests::{lend_forced_off, note_lent};

#[cfg(not(test))]
#[inline(always)]
fn lend_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_lent() {}

// A test counts the levels the sparse leaf arm's owner walk read from f's
// implied nodes.
#[cfg(test)]
use tests::note_implied_walk;

#[cfg(not(test))]
#[inline(always)]
fn note_implied_walk(_dense: bool) {}

// A test lists the levels a conjunction built as an operand holds them.
#[cfg(test)]
use tests::note_as_operand;

#[cfg(not(test))]
#[inline(always)]
fn note_as_operand(_t: usize, _of_f: bool) {}

// A test counts the levels only the leaf arm's held products show tight.
#[cfg(test)]
use tests::note_held_tight;

#[cfg(not(test))]
#[inline(always)]
fn note_held_tight() {}

// A test lists the route each level took.
#[cfg(test)]
use tests::note_route;

#[cfg(not(test))]
#[inline(always)]
fn note_route(_t: VtreeIdx, _route: Route) {}

// A test sends every level past the relabelling route, as the oracle the
// route is checked against, and counts the levels it took: moved whole, and
// rebuilt; those of them that read through a marginal child; and the nodes
// it wrote holding a pair twice.
#[cfg(test)]
use tests::{note_read_through, note_relabelled, note_written_pairs, relabel_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn relabel_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_relabelled(_moved: bool) {}

#[cfg(not(test))]
#[inline(always)]
fn note_read_through() {}

#[cfg(not(test))]
#[inline(always)]
fn note_written_pairs(_pairs: &[ChildPair]) {}

// A test lists every child a sparse level joins, the oracle the arithmetic
// reads of a complete child are checked against, and counts the levels that
// read one.
#[cfg(test)]
use tests::{complete_sides_listed, note_complete_side, note_summed_complete_root};

#[cfg(not(test))]
#[inline(always)]
fn complete_sides_listed() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_complete_side(_side: crate::diagram::ChildSide) {}

#[cfg(not(test))]
#[inline(always)]
fn note_summed_complete_root() {}

// A test counts the levels a count read from an operand's kept counts
// instead of folding them.
#[cfg(test)]
use tests::note_kept_counts;

#[cfg(not(test))]
#[inline(always)]
fn note_kept_counts(_levels: usize) {}

/// Note for the tests' census that a count outside the conjunction read
/// `levels` kept columns; nothing outside the tests.
#[inline(always)]
pub(crate) fn note_kept_counts_read(levels: usize) {
    note_kept_counts(levels);
}
