//! Bottom-up conjunction of two diagrams over a shared vtree.
//!
//! Entry points validate the operands, align weight tables, and check the
//! constant and self-conjunction (`f ∧ f = f`) shortcuts before entering the
//! level sweep ([`drive`]). Operands may be owned or borrowed (`Cow<Tdd>`),
//! placed onto a wider projection with free levels left unbuilt
//! ([`conjoin_onto`]), or restored intact on refusal
//! ([`Engine::and_restoring`]).
//!
//! Within the active [`setup::Cone`], leaf levels are seeded from the constant
//! conjunction table ([`leaf_seed`]) and each internal level is routed once
//! ([`route::route_level`]):
//! - [`identity`] moves or copies the other operand's level when one operand is
//!   constant-true over the subtree;
//! - [`relabel`] carries or rewrites one operand's pairs when the other operand
//!   has a single node with a single pair;
//! - [`sparse`] scatters, filters, and deduplicates over live child products
//!   (or probes one `f` node at a time) when child or level grids are wide and
//!   sparse;
//! - [`streaming_marginal`] folds each live cell directly into a count or weight
//!   column on levels scheduled for marginalization;
//! - [`cell`] walks the dense product grid, using dead-pair liveness masks
//!   ([`liveness`]), arithmetic lookups on complete children, or implicit
//!   affine product descriptions (`drive::compose`).
//!
//! A count ([`ConjoinMode::Count`]) folds a one-product sparse root without
//! materializing its pairs.

use std::borrow::{Borrow, Cow};
use std::sync::Arc;

use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::restructure::embed::{place, place_by, Free, Plan};
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
use sparse::{apply_sparse_level, count_sparse_level, CandidateFold, Complete, Made, Passthrough};

// An operand as the sweep holds it, owned or borrowed.
mod held;
pub(crate) mod view;
use held::{Held, Here};
use view::View;

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
mod projected;
mod union;
use crate::apply::extend;
use output::*;
mod drive;
use drive::{apply_and_core, Conjoined, ConjoinMode};
use drive::Sweep;

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
    // marginal level, as each operand's kept first marginal level says, once
    // the outputs, which most operands that differ differ in, agree.
    if f.output != g.output || f.has_marginal_level() || g.has_marginal_level() {
        return false;
    }
    same_structure(f, g, free)
}

/// [`same_levels`] for operands known to have no marginal level.
///
/// The levels are compared from the last index down, the internal levels
/// before the leaves, where operands that differ usually differ first.
fn same_structure(f: &Tdd, g: &Tdd, free: Operands<VtreeMask<'_>>) -> bool {
    f.output == g.output
        && same_structure_by(&f.vtree, |t| &f.levels[t], |t| &g.levels[t], f.levels.len().min(g.levels.len()), free)
}

/// [`same_structure`] of the operands whose level at node `t` of `vtree`
/// is `f_level(t)` and `g_level(t)`, on the first `len` nodes, their
/// outputs being the same.
fn same_structure_by<'a>(
    vtree: &Vtree,
    f_level: impl Fn(usize) -> &'a TddLevel,
    g_level: impl Fn(usize) -> &'a TddLevel,
    len: usize,
    free: Operands<VtreeMask<'_>>,
) -> bool {
    // `ranges` too: equal nodes+pairs with a differently-arranged `ranges` table
    // is a different function.
    let same = |l1: &TddLevel, l2: &TddLevel| l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ranges == l2.ranges;
    let same_at = |t: usize| {
        let (l1, l2) = (f_level(t), g_level(t));
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
    (0..len).rev().all(same_at)
}

/// Whether `f` and `g`, of which one at most is read through its view in
/// `views`, are the same diagram on the sweep's vtree ([`same_structure`]).
fn same_viewed<'a>(f: &'a Tdd, g: &'a Tdd, views: Operands<Option<&'a View>>, free: Operands<VtreeMask<'_>>) -> bool {
    match (views.f, views.g) {
        (None, None) => same_structure(f, g, free),
        (Some(view), None) | (None, Some(view)) => {
            let (viewed, other) = if views.f.is_some() { (f, g) } else { (g, f) };
            if view.output() != other.output {
                return false;
            }
            let viewed_level = |t: usize| view.level(viewed, t);
            let other_level = |t: usize| &other.levels[t];
            let len = other.levels.len();
            match views.f.is_some() {
                true => same_structure_by(view.vtree(), viewed_level, other_level, len, free),
                false => same_structure_by(view.vtree(), other_level, viewed_level, len, free),
            }
        }
        (Some(_), Some(_)) => false,
    }
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
        return_levels_of(eng, &mut tdd);
    }
}

/// Return `tdd`'s level array to the pool, unless a conjunction that
/// adopted it left none.
fn return_levels_of(eng: &Engine, tdd: &mut Tdd) {
    let levels = std::mem::take(&mut tdd.levels).into_vec();
    if !levels.is_empty() {
        diagram::return_levels(eng, levels);
    }
}

/// [`Engine::and`] after its entry, for operands each owned or borrowed
/// ([`conjoin_cow`]).
pub(crate) fn conjoin_either(eng: &Engine, f: Cow<'_, Tdd>, g: Cow<'_, Tdd>) -> Result<Tdd, OperationError> {
    if !Arc::ptr_eq(f.vtree(), g.vtree()) || f.vtree().num_nodes() == 0 {
        return projected::conjoin_projections(eng, f, g);
    }
    if f.vtree().has_regions() {
        projected::check_shared_regions(&f, &g)?;
    }
    let mut out = conjoin_cow(eng, f, g, VtreeMask::default())?;
    crate::reduce::close_constant(eng, &mut out);
    Ok(out)
}

/// Make `g` the narrower operand, and say whether the two were swapped; the
/// free levels in `free` ([`ApplyRun::cone`]) follow their operands and are
/// one node wide.
///
/// The identity fast path tests `g_width == 1` first, so the narrower side
/// on the right takes it at more levels, and grid rows (width `g_width`)
/// get shorter. Only the owned entries swap; borrowed callers
/// track operands by side.
///
/// `widths` holds each operand's [`Tdd::max_width`].
fn narrower_right<T: Borrow<Tdd>>(
    f: &mut T,
    g: &mut T,
    free: &mut Operands<VtreeMask<'_>>,
    widths: Operands<usize>,
    vtree: Option<&Vtree>,
) -> bool {
    let width = |d: &Tdd, free: VtreeMask<'_>, built: usize| match built {
        0 => usize::from(!free.is_empty() && vtree.unwrap_or(&d.vtree).internal_bottomup().any(|(t, _, _)| free.contains(t.idx()))),
        built => built,
    };
    let swap = width((*g).borrow(), free.g, widths.g) > width((*f).borrow(), free.f, widths.f);
    if swap {
        std::mem::swap(f, g);
        std::mem::swap(&mut free.f, &mut free.g);
    }
    swap
}

/// Conjoin validated, weight-aligned operands, summing out the levels in
/// `targets`.
pub(crate) fn conjoin_checked(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
) -> Result<Tdd, OperationError> {
    let widths = Operands { f: f.max_width(), g: g.max_width() };
    narrower_right(&mut f, &mut g, &mut Operands::default(), widths, None);
    // Self-conjunction: f ∧ f = f. The test is structural equality of every
    // explicit level, not pointer identity, and it declines on any marginal
    // level — see `is_self_conjunction`, where the soundness of both choices
    // is stated.
    if is_self_conjunction(&f, &g) {
        let _op = eng.limits().enter()?;
        diagram::return_levels(eng, std::mem::take(&mut g.levels).into_vec());
        return Ok(f);
    }
    conjoin_recycling(eng, f, g, targets)
}

/// [`conjoin_checked`] of operands each owned or borrowed: two owned ones
/// take it; a borrowed one is read in place, weights and marginal levels
/// included, and left as it was, for the result and work of the owned call.
fn conjoin_cow<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    g: Cow<'a, Tdd>,
    targets: VtreeMask<'_>,
) -> Result<Tdd, OperationError> {
    let (mut f, mut g) = match (f, g) {
        (Cow::Owned(f), Cow::Owned(g)) => return conjoin_checked(eng, f, g, targets),
        operands => operands,
    };
    let widths = Operands { f: f.max_width(), g: g.max_width() };
    narrower_right(&mut f, &mut g, &mut Operands::default(), widths, None);
    if is_self_conjunction(&f, &g) {
        // f ∧ f = f, as two owned operands return it: f itself where owned,
        // a copy of it where borrowed.
        let _op = eng.limits().enter()?;
        let out = match f {
            Cow::Owned(out) => out,
            Cow::Borrowed(f) => f.try_clone_on(eng)?,
        };
        recycle(eng, g);
        return Ok(out);
    }
    let result = drive::apply_and_either(eng, &mut Held::of(&mut f, None), &mut Held::of(&mut g, None), targets);
    recycle(eng, f);
    recycle(eng, g);
    result
}

/// Run the sweep over `f` and `g`, then hand both operands' level arrays back
/// to the pool whatever the outcome.
fn conjoin_recycling(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
) -> Result<Tdd, OperationError> {
    let result = apply_and_core(eng, &mut f, &mut g, targets, ConjoinMode::Build);
    return_levels_of(eng, &mut f);
    return_levels_of(eng, &mut g);
    result.map(Conjoined::diagram)
}

/// Return the conjunction of two diagrams sharing the same vtree allocation,
/// or on two projections of one home ([`Vtree::projection`]).
///
/// Uses the shared vtree's execution context automatically. The result of
/// operands on two projections lives on the larger operand's vtree when its
/// leaves hold the other's, and on the projection of both leaf sets
/// otherwise; an operand with a summed-out level is refused there
/// ([`OperationError::MarginalLevel`]).
///
/// Each operand is a [`Tdd`], consumed on success and on error, or a
/// `&Tdd`, read in place and left as it is. Where the result takes a level
/// of an operand as it stands, such as one the other operand is constant
/// true under, an owned operand's level is moved and a borrowed one's
/// copied: borrowing costs those levels and a borrowed operand's weight
/// store, where cloning first copies every level. Owned or borrowed, the
/// result is the same diagram for the same work. It is correct for counting
/// but may retain unreachable nodes and twins; [`Tdd::minimize`], or an
/// engine with [`canonical_results`](crate::limits::LimitConfig::with_canonical_results), certifies it.
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
/// the summed-out variables. On one vtree, when both operands are marginal at
/// a level, the caller must ensure one represents the constant-true function
/// on that subtree; this condition is checked in debug builds. On projections
/// of one home it is checked in every build: operands of which one has summed
/// out a variable the other has a leaf for or has summed out too are refused.
///
/// # Errors
///
/// [`OperationError::VtreeMismatch`] for different vtree allocations that
/// are not projections of one home,
/// [`OperationError::MarginalLevel`] when a structural operand constrains
/// variables the other has summed out, both have summed out one variable, or
/// an operand on a projection has an internal summed-out level that is not a
/// region, or [`OperationError::IndexOverflow`]
/// when a level of the result would hold more nodes than a node index
/// addresses.
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

/// [`Engine::and_restoring`](crate::Engine::and_restoring) after its entry
/// and vtree check, for operands whose free levels ([`ApplyRun::cone`]) are
/// in `free`: refused, both come back as they were given, every free level
/// built. An operand with a view has its plan in `plans`, and the nodes its
/// view passes through in `passes` ([`Held::passing`]).
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn conjoin_kept<'a>(
    eng: &Engine,
    mut f: Cow<'a, Tdd>,
    mut g: Cow<'a, Tdd>,
    mut free: Operands<VtreeMask<'_>>,
    mut views: Operands<Option<&mut View>>,
    plans: Operands<Option<&Plan>>,
    mut passes: Operands<&[bool]>,
) -> Result<Tdd, AndRefused<'a>> {
    // Whether a level is marginal, and the widest level, in one pass.
    let scan = |t: &Tdd| t.levels.iter().fold((false, 0), |(marginal, width), l| {
        (marginal | l.is_marginal(), width.max(l.slot_count()))
    });
    let ((f_marginal, f_width), (g_marginal, g_width)) = (scan(&f), scan(&g));
    if f_marginal || g_marginal {
        if views.f.is_some() || views.g.is_some() {
            let viewed = Operands { f: views.f.as_deref(), g: views.g.as_deref() };
            (f, g) = place_viewed(eng, f, g, free, viewed, plans)?;
        }
        fill_free_cow(&mut f, free.f);
        fill_free_cow(&mut g, free.g);
        return eng.and(Cow::Borrowed(&*f), Cow::Borrowed(&*g)).map_err(|error| AndRefused { error, f, g });
    }
    let vtree = views.f.as_deref().or(views.g.as_deref()).map(|view| Arc::clone(view.vtree()));
    let width = |own: usize, view: &Option<&mut View>| own.max(view.as_deref().map_or(0, View::built_width));
    let widths = Operands { f: width(f_width, &views.f), g: width(g_width, &views.g) };
    let swapped = narrower_right(&mut f, &mut g, &mut free, widths, vtree.as_deref());
    if swapped {
        std::mem::swap(&mut views.f, &mut views.g);
        std::mem::swap(&mut passes.f, &mut passes.g);
    }
    // A viewed operand was not placed, and has no free level of its own.
    let placed_free = Operands {
        f: if views.f.is_some() { VtreeMask::default() } else { free.f },
        g: if views.g.is_some() { VtreeMask::default() } else { free.g },
    };
    let give_back = |error, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>| {
        fill_free_cow(&mut f, placed_free.f);
        fill_free_cow(&mut g, placed_free.g);
        if swapped {
            std::mem::swap(&mut f, &mut g);
        }
        AndRefused { error, f, g }
    };
    if same_viewed(&f, &g, Operands { f: views.f.as_deref(), g: views.g.as_deref() }, free) {
        // f ∧ f = f, as two owned operands return it: f itself where owned,
        // its free levels built, a copy of it where borrowed; g where f is
        // viewed, and g is not.
        let (keep, drop, keep_free) = match views.f.is_some() {
            false => (f, g, free.f),
            true => (g, f, free.g),
        };
        let out = match keep {
            Cow::Owned(mut out) => {
                fill_free(&mut out, keep_free);
                out
            }
            Cow::Borrowed(borrowed) => match borrowed.try_clone_on(eng) {
                Ok(out) => out,
                Err(error) => {
                    let (f, g) = match views.f.is_some() {
                        false => (Cow::Borrowed(borrowed), drop),
                        true => (drop, Cow::Borrowed(borrowed)),
                    };
                    return Err(give_back(error, f, g));
                }
            },
        };
        recycle(eng, drop);
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
    let (f_known, g_known) = (known(&f, placed_free.f), known(&g, placed_free.g));
    let conjoined = {
        let (mut f, mut g) = (Held::passing(&mut f, views.f, passes.f), Held::passing(&mut g, views.g, passes.g));
        drive::apply_and_restoring(eng, &mut f, &mut g, free)
    };
    match conjoined {
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

/// Place each operand of [`conjoin_kept`] that has a view by its plan, free
/// levels left empty, for the conjunction of summed-out operands, which
/// reads no view. Refused, an operand not yet placed comes back as it was
/// given and a placed one with its free levels built.
#[cold]
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn place_viewed<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    g: Cow<'a, Tdd>,
    free: Operands<VtreeMask<'_>>,
    views: Operands<Option<&View>>,
    plans: Operands<Option<&Plan>>,
) -> Result<(Cow<'a, Tdd>, Cow<'a, Tdd>), AndRefused<'a>> {
    let place =|tdd: Cow<'a, Tdd>, view: Option<&View>, plan: Option<&Plan>| match view {
        None => Ok(tdd),
        Some(view) => place_by(eng, tdd, view.vtree(), plan.expect("a viewed operand has its plan"), Free::Leave)
            .map(Cow::Owned)
            .map_err(|(error, tdd)| (crate::apply::extend::operation_error(error), tdd)),
    };
    let built = |mut tdd: Cow<'a, Tdd>, free: VtreeMask<'_>| {
        fill_free_cow(&mut tdd, free);
        tdd
    };
    let f = match place(f, views.f, plans.f) {
        Ok(f) => f,
        Err((error, f)) => {
            let g = if views.g.is_some() { g } else { built(g, free.g) };
            return Err(AndRefused { error, f, g });
        }
    };
    match place(g, views.g, plans.g) {
        Ok(g) => Ok((f, g)),
        Err((error, g)) => Err(AndRefused { error, f: built(f, free.f), g }),
    }
}

/// Take back what an owned operand given whole was known to be before a
/// refused conjunction put its levels back ([`conjoin_kept`]).
fn reinstate_known(tdd: &mut Cow<'_, Tdd>, known: Option<diagram::Known>) {
    if let (Cow::Owned(tdd), Some(known)) = (tdd, known) {
        tdd.levels.reinstate_known(known);
    }
}

/// An operand [`place_onto`] placed, or to be viewed, with the plan it was
/// placed or is to be viewed by, or why it was refused, with the operand as
/// it was given.
type Placed<'a> = Result<(Cow<'a, Tdd>, Option<Plan>, bool), (OperationError, Cow<'a, Tdd>)>;

/// Make `tdd`, on a projection of `into`'s home holding none of `into`'s
/// leaves it lacks, an operand on `into` for [`conjoin_onto`], with the plan
/// that embeds it: as it is, with no plan, when it is on `into` already; with
/// `viewing`, left as it is for a [`View`] where it is structural,
/// unweighted, not false, its plan mirrors no image and, owned, it is
/// cheaper to view than to place ([`views_cheaper`]), which the flag says;
/// placed otherwise, free levels left empty, a borrowed one copied into the
/// placement ([`place`]). `begun` says whether a phase of the operation has
/// begun, and is set. Refused, the operand comes back as it was given.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn place_onto<'a>(
    eng: &Engine,
    tdd: Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    begun: &mut bool,
    viewing: bool,
) -> Placed<'a> {
    if Arc::ptr_eq(tdd.vtree(), into) {
        return Ok((tdd, None, false));
    }
    if let Err(error) = begin_phase(eng, *begun) {
        return Err((error, tdd));
    }
    *begun = true;
    if tdd.vtree().num_nodes() == 0 {
        return match crate::build::constant_padded_on(eng, into, tdd.is_one()) {
            Ok(constant) => Ok((Cow::Owned(constant), None, false)),
            Err(error) => Err((error, tdd)),
        };
    }
    if viewing && !tdd.is_zero() && tdd.require_structure().is_ok() {
        let plan = match Plan::build(eng, tdd.vtree(), into, |v| v, true, Free::Leave) {
            Ok(plan) => plan,
            Err(error) => return Err((extend::operation_error(error), tdd)),
        };
        let viewed = match tdd {
            Cow::Borrowed(_) => true,
            Cow::Owned(_) => views_cheaper(&plan, into),
        };
        if viewed && plan.mirrored_images().next().is_none() {
            return Ok((tdd, Some(plan), true));
        }
        return place_by(eng, tdd, into, &plan, Free::Leave)
            .map(|tdd| (Cow::Owned(tdd), Some(plan), false))
            .map_err(|(error, tdd)| (extend::operation_error(error), tdd));
    }
    place(eng, tdd, into, |v| v, Free::Leave)
        .map(|(tdd, plan)| (Cow::Owned(tdd), Some(plan), false))
        .map_err(|(error, tdd)| (extend::operation_error(error), tdd))
}

/// Whether an owned operand embedded by `plan` into `into` costs less read
/// through a view than placed. A pass-through level costs more built and
/// read in a view than in a placement, and a placement costs more than a
/// view for each node of `into`; measured, views lose where a quarter of
/// `into`'s nodes are pass-throughs and win where a tenth are, and the cut
/// is at a quarter. A borrowed operand is viewed whatever this says:
/// placing it copies every level.
#[inline]
fn views_cheaper(plan: &Plan, into: &Vtree) -> bool {
    plan.pass_throughs * 4 < into.num_nodes()
}

/// The view of `tdd` by `plan` where `viewed`, charging the work clock
/// one unit a pass-through level, as a placement does.
fn view_of(eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: Option<&Plan>, viewed: bool) -> Result<Option<View>, OperationError> {
    match plan {
        Some(plan) if viewed => View::build(eng, tdd, into, plan).map(Some),
        _ => Ok(None),
    }
}

/// `f ∧ g` on `into`, a projection of their home holding both leaf sets,
/// before the close: each operand not on `into` read through a view where
/// `viewing` and it is cheaper, placed otherwise.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
pub(crate) fn conjoin_onto<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    g: Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    viewing: bool,
) -> Result<Tdd, AndRefused<'a>> {
    let _op = match eng.limits().enter() {
        Ok(op) => op,
        Err(error) => return Err(AndRefused { error, f, g }),
    };
    // A borrowed operand comes back as the borrow, whatever its placed
    // copy went through.
    let given = |tdd: &Cow<'a, Tdd>| match tdd {
        Cow::Borrowed(tdd) => Some(*tdd),
        Cow::Owned(_) => None,
    };
    let (f_given, g_given) = (given(&f), given(&g));
    let back = |tdd: Cow<'a, Tdd>, given: Option<&'a Tdd>| given.map_or(tdd, Cow::Borrowed);
    // Off projections a summed-out operand is never placed, so it is on
    // `into` already, where the conjunction leaves this check to the caller.
    if (f.vtree().on_home() || g.vtree().on_home())
        && !Arc::ptr_eq(f.vtree(), g.vtree())
        && let Err(error) = projected::check_summed_leaves(&f, &g)
    {
        return Err(AndRefused { error, f, g });
    }
    let mut begun = false;
    let (f, f_plan, f_viewed) = match place_onto(eng, f, into, &mut begun, viewing) {
        Ok(placed) => placed,
        Err((error, f)) => return Err(AndRefused { error, f, g }),
    };
    let free_f = free_levels(&f, f_plan.as_ref());
    // The free levels a placement left empty, which a refusal builds.
    let placed_f = if f_viewed { VtreeMask::default() } else { free_f };
    let refuse_f = |error: OperationError, f: Cow<'a, Tdd>, g| {
        let mut f = f;
        fill_free_cow(&mut f, placed_f);
        Err(AndRefused { error, f: back(f, f_given), g })
    };
    let (g, g_plan, g_viewed) = match place_onto(eng, g, into, &mut begun, viewing) {
        Ok(placed) => placed,
        Err((error, g)) => return refuse_f(error, f, g),
    };
    let free = Operands { f: free_f, g: free_levels(&g, g_plan.as_ref()) };
    let placed_g = if g_viewed { VtreeMask::default() } else { free.g };
    let refuse = |error: OperationError, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>| {
        fill_free_cow(&mut f, placed_f);
        fill_free_cow(&mut g, placed_g);
        Err(AndRefused { error, f: back(f, f_given), g: back(g, g_given) })
    };
    // f ∧ f = f, placed as a conjunction of placed operands returns it.
    if let (Some(fp), Some(gp)) = (&f_plan, &g_plan)
        && f_viewed
        && g_viewed
        && fp.embedding.levels == gp.embedding.levels
        && same_levels(&f, &g, Operands::default())
    {
        if let Err(error) = eng.limits().gate().poll_each(gp.pass_throughs as u64) {
            return refuse(error, f, g);
        }
        let placed = place_by(eng, f, into, fp, Free::Leave).map(|mut out| {
            fill_free(&mut out, VtreeMask::free_of(fp));
            out
        });
        return placed.map_err(|(error, f)| AndRefused { error: extend::operation_error(error), f: back(f, f_given), g });
    }
    let mut f_view = match view_of(eng, &f, into, f_plan.as_ref(), f_viewed) {
        Ok(view) => view,
        Err(error) => return refuse(error, f, g),
    };
    let mut g_view = match view_of(eng, &g, into, g_plan.as_ref(), g_viewed) {
        Ok(view) => view,
        Err(error) => return refuse(error, f, g),
    };
    match begin_phase(eng, begun) {
        Err(error) => refuse(error, f, g),
        Ok(()) => conjoin_kept(
            eng,
            f,
            g,
            free,
            Operands { f: f_view.as_mut(), g: g_view.as_mut() },
            Operands { f: f_plan.as_ref(), g: g_plan.as_ref() },
            Operands::default(),
        )
            .map_err(|r| AndRefused { error: r.error, f: back(r.f, f_given), g: back(r.g, g_given) }),
    }
}

/// Begin a phase of [`conjoin_onto`], a placement or the conjunction, as the
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
    plan.filter(|_| !tdd.is_zero()).map_or_else(VtreeMask::default, VtreeMask::free_of)
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
        crate::apply::exit::outcome(self, conjoin_either(self, f.into(), g.into()))
    }

    /// Run [`Engine::and`], giving both operands back unchanged when it is
    /// refused.
    ///
    /// A caller that retries after a stop would otherwise clone both operands
    /// before each conjunction. Here the sweep keeps the operands whole until
    /// the result exists, which holds their storage as long as a clone would,
    /// without the copy. Operands with a marginal level are both read as
    /// borrowed ones. Each operand is owned or borrowed, at the costs
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
        if !Arc::ptr_eq(f.vtree(), g.vtree()) || f.vtree().num_nodes() == 0 {
            return projected::conjoin_projections_kept(self, f, g);
        }
        conjoin_kept(self, f, g, Operands::default(), Operands::default(), Operands::default(), Operands::default()).map(|mut out| {
            crate::reduce::close_constant(self, &mut out);
            out
        })
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
    /// fixed weighted value when weights are attached. Operands on projections
    /// take [`Engine::and_marginalizing_where`] instead.
    ///
    /// Every target's remaining structure is discarded, including after identity
    /// and self-conjunction shortcuts; the false diagram stays false. See
    /// [`Tdd::marginalize_levels`] for the operations
    /// that remain valid after structure is discarded.
    ///
    /// A target built as structure is summed out once the conjunction is
    /// built, and the pairs above it that then name the same node on their
    /// other side are fused into one.
    ///
    /// On an unweighted conjunction, a subtree whose internal levels are all
    /// targets is built as structure and summed at its top, as it is when
    /// only the top is named, where its products are large enough for the
    /// sparse route and its operands wide enough for its grids to be sparse.
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
    /// [`OperationError::LevelNotInVtree`] for an invalid target and
    /// [`OperationError::InertOption`] for an operand on a home (a projection
    /// or a home's root), checked before product construction. An owned operand is consumed on every
    /// outcome, and a borrowed one is left as it was.
    pub fn and_marginalizing<'f, 'g>(
        &self,
        f: impl Into<Cow<'f, Tdd>>,
        g: impl Into<Cow<'g, Tdd>>,
        targets: &[VtreeIdx],
    ) -> Result<Tdd, OperationError> {
        let (f, g) = (f.into(), g.into());
        let _op = self.limits().enter()?;
        if f.vtree().on_home() || g.vtree().on_home() {
            return Err(targets_on_projection());
        }
        f.check_level_indices(targets)?;
        let [f, g] = extend::on_one(self, [f, g])?;
        // The apply core asks "is level `t` a target?" once per level it emits,
        // so the membership array is derived here, once, at the cost the caller
        // would pay to build it.
        let vtree = Arc::clone(f.vtree());
        let mut mask = target_mask(self, &vtree, targets, true)?;
        let unweighted = f.weight_table().is_none() && g.weight_table().is_none();
        if unweighted {
            collapse_target_subtrees(self, &vtree, &f, &g, &mut mask)?;
        }
        // An unweighted sweep left no level to sum is given none, as
        // `and_model_count` gives it none: the pass below sums the targets.
        let summed = match unweighted && !mask.contains(&true) {
            true => VtreeMask::default(),
            false => VtreeMask::new(Some(&mask)),
        };
        let mut out = conjoin_cow(self, f, g, summed)?;
        // Streaming can finish every target; only retained structure needs the pass.
        if !out.is_zero() && targets.iter().any(|&t| !out.level(t).is_marginal()) {
            self.marginalize_levels(&mut out, targets)?;
        }
        crate::reduce::close_constant(self, &mut out);
        crate::apply::exit::outcome(self, Ok(out))
    }

    /// Conjoin two diagrams and sum out the variables `summable` names.
    ///
    /// Each operand is owned or borrowed, as [`and`] says. The conjunction is
    /// built on the vtree holding both operands' leaves, and every node of it
    /// whose variable leaves all satisfy `summable` is summed out, an
    /// internal one as the product is built, a region or a leaf summed out in an operand
    /// counting as satisfied; the result is then what
    /// [`Engine::marginalize`] gives on the conjunction, regions on a
    /// projection included. A node over regions only is therefore always
    /// summed into one. With no such node this is [`Engine::and`].
    ///
    /// `summable(v)` must be true only if no diagram the result will meet has
    /// `v` as a leaf: the caller takes both operands out of its record of
    /// who holds what before it asks. It is asked at most once per variable
    /// leaf of the conjunction's vtree, in no stated order. A later operand
    /// with a leaf in a region is refused [`OperationError::MarginalLevel`].
    /// Home variables neither operand uses stay free, also under a summed
    /// node, and [`Engine::evaluate`] applies its algebra to them.
    ///
    /// ```
    /// use tididi::{Engine, Tdd};
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&Vtree::projection(&home, [VarId(1), VarId(2)])?, [1, 2])?;
    /// let g = Tdd::clause(&Vtree::projection(&home, [VarId(2), VarId(3)])?, [-2, 3])?;
    /// let summed = engine.and_marginalizing_where(&f, &g, |v| v != VarId(3))?;
    /// assert_eq!(summed.vtree().num_leaves(), 1);
    /// assert_eq!(summed.model_count()?, engine.and(&f, &g)?.model_count()?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The operand and resource errors of [`Engine::and`]. An owned operand
    /// is consumed on every outcome, and a borrowed one is left as it was.
    pub fn and_marginalizing_where<'f, 'g>(
        &self,
        f: impl Into<Cow<'f, Tdd>>,
        g: impl Into<Cow<'g, Tdd>>,
        summable: impl Fn(VarId) -> bool,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::exit::outcome(self, marginalizing_where(self, f.into(), g.into(), &summable))
    }

    /// Count the models of `f ∧ g` without keeping the conjunction.
    ///
    /// The result is `model_count(and_marginalizing(f, g, targets))`:
    /// `targets` are summed out during product construction as there, which
    /// changes how much structure is built, never the count. Where the root
    /// of the conjunction is one product built by the sparse route, its
    /// pairs are folded into the count as they are found instead of being
    /// stored, so the largest level of a join that ends in a count is never
    /// held. Otherwise the conjunction is built and counted.
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
        f: Tdd,
        g: Tdd,
        targets: &[VtreeIdx],
    ) -> Result<num_bigint::BigUint, OperationError> {
        let _op = self.limits().enter()?;
        f.check_level_indices(targets)?;
        let from = Arc::clone(f.vtree());
        let [mut f, mut g] = extend::on_one(self, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
        let translated = extend::nodes_on(self, &from, f.vtree(), targets)?;
        let targets = translated.as_deref().unwrap_or(targets);
        if f.weight_table().is_some() || g.weight_table().is_some() {
            let out = match f.vtree().on_home() {
                true => self.and_marginalizing_where(f, g, |_| true)?,
                false => self.and_marginalizing(f, g, targets)?,
            };
            return self.model_count(&out);
        }
        let vtree = Arc::clone(f.vtree());
        // A target at the root sums the whole conjunction into its count,
        // which is what is returned anyway: it is dropped, so the root can be
        // counted instead of built.
        let mut mask = target_mask(self, &vtree, targets, false)?;
        collapse_target_subtrees(self, &vtree, &f, &g, &mut mask)?;
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
        let result = apply_and_core(self, &mut f, &mut g, summed, ConjoinMode::Count);
        diagram::return_levels(self, std::mem::take(&mut f.levels).into_vec());
        diagram::return_levels(self, std::mem::take(&mut g.levels).into_vec());
        match result? {
            Conjoined::Counted(count) => Ok(match vtree.free_leaves() {
                0 => count,
                free => count << free,
            }),
            Conjoined::Built(out) => self.model_count(&out),
        }
    }
}

#[cold]
fn targets_on_projection() -> OperationError {
    OperationError::InertOption { option: "targets", needs: "operands on a root vtree that is not a home; on a home, `and_marginalizing_where`" }
}

/// [`Engine::and_marginalizing_where`]: the nodes
/// [`finished`](crate::marginal::finished) names on the conjunction's vtree
/// are the sweep's targets, and [`sum_out`](crate::marginal::sum_out)
/// finishes them.
fn marginalizing_where(eng: &Engine, f: Cow<'_, Tdd>, g: Cow<'_, Tdd>, summable: &dyn Fn(VarId) -> bool) -> Result<Tdd, OperationError> {
    let projected = f.vtree().on_home() || g.vtree().on_home();
    if projected {
        if (f.has_marginal_level() || g.has_marginal_level()) && !Arc::ptr_eq(f.vtree(), g.vtree()) {
            Vtree::leaf_sets(f.vtree(), g.vtree())?;
            projected::check_summed_leaves(&f, &g)?;
        }
        if f.vtree().num_nodes() == 0 || g.vtree().num_nodes() == 0 {
            let mut out = projected::conjoin_constant(eng, f, g)?;
            if !out.is_zero() {
                crate::marginal::marginalize(eng, &mut out, summable)?;
            }
            return extend::trimmed(eng, out);
        }
    }
    let [f, g] = extend::on_one(eng, [f, g])?;
    if projected {
        projected::check_shared_regions(&f, &g)?;
    }
    let vtree = Arc::clone(f.vtree());
    let summed = |t: VtreeIdx| f.level(t).is_marginal() || g.level(t).is_marginal();
    let (mut mask, maximal) = crate::marginal::finished(eng, &vtree, summed, summable)?;
    // A root summed in the sweep keeps one slot per product node; the fold
    // after it merges equal values and renumbers the output.
    mask[vtree.root().idx()] = false;
    let unweighted = f.weight_table().is_none() && g.weight_table().is_none();
    if unweighted {
        collapse_target_subtrees(eng, &vtree, &f, &g, &mut mask)?;
    }
    let summed = match mask.contains(&true) {
        true => VtreeMask::new(Some(&mask)),
        false => VtreeMask::default(),
    };
    let mut out = conjoin_cow(eng, f, g, summed)?;
    if !out.is_zero() && !maximal.is_empty() {
        debug_assert!(Arc::ptr_eq(out.vtree(), &vtree), "the conjunction is on its operands' vtree");
        #[cfg(test)]
        tests::region::note_swept(maximal.iter().filter(|&&t| out.level(t).is_marginal()).count());
        crate::marginal::sum_out(eng, &mut out, &maximal)?;
    }
    extend::trimmed(eng, out)
}

/// Membership mask of internal vtree nodes in `targets`, optionally including
/// the vtree root.
fn target_mask(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    targets: &[VtreeIdx],
    include_root: bool,
) -> Result<Vec<bool>, OperationError> {
    let mut mask = Vec::new();
    eng.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
    for &t in targets {
        mask[t.idx()] = !vtree.node(t).is_leaf() && (include_root || t != vtree.root());
    }
    Ok(mask)
}

/// Whether a subtree's product grids can be sparse by the sparse routes'
/// factor `sparsity`, from its grid cells and those cells weighted by the
/// narrower operand's node count, both summed over the subtree's structural
/// levels.
///
/// Where each node of the wider operand meets some node of the other, as
/// the operands of a conjunction mostly do, a product level holds at least
/// as many nodes as the wider operand has there, and so at least `1/w` of
/// its grid, `w` the narrower operand's node count. Folded, the subtree
/// visits every cell of its grids and keeps one value per live product;
/// built as structure, it visits the live products only, but stores each
/// with its pairs, and the pass after the sweep reads them again. Only
/// where the cell-weighted `w` reaches `sparsity` can the live products fall
/// to the share of the grid at which the sparse routes pay.
fn may_be_sparse((cells, weighted): (u128, u128), sparsity: u128) -> bool {
    weighted >= cells.saturating_mul(sparsity)
}

/// Take out of `mask`, the levels an unweighted sweep sums out, the levels
/// of each closed target subtree that are better built as structure and
/// summed once the subtree's top is.
///
/// The sweep folds a level `mask` names over its children's whole grid
/// (`Route::Stream`) and builds no level under it by the sparse or
/// relabelling routes. A target `c` below the root is the top of a closed
/// subtree when every internal level under it is in `mask` or marginal in an
/// operand. The levels under such a `c` leave `mask` where
///
/// - some structural level of the subtree is over the sparse floor
///   ([`price::over_floor`]) and the subtree's grids can be as sparse as
///   the sparse routes need ([`may_be_sparse`]), so the subtree's products are worth finding by the
///   sparse routes before `c` sums them, or
/// - `c`'s parent is one product over a structural subtree: no level
///   under `c` is marginal in an operand, one operand has one node at every
///   level of the subtree, both have one node at the parent, and the parent
///   or a level under `c`'s sibling is over the floor. This case was written
///   for a parent that summed `c` as it was built, which is gone; the
///   two-step path it now takes was measured equal with the case kept.
///
/// Summing `c` makes the levels under it marginal, as it does for a caller
/// that names `c` alone. `c` itself leaves `mask` as well in the first case
/// where the fold over its children would not pay: a level under it is
/// marginal in an operand, its sibling is summed too, or `c` is a level the
/// relabelling route can take. The pass that follows the sweep
/// ([`Engine::marginalize_levels`](crate::Engine::marginalize_levels)) then
/// sums it.
///
/// Which levels `mask` names changes the work of the sweep and never the
/// diagram or count the caller gets. The pass reads two levels' headers per
/// vtree node and holds one byte and two 128-bit sums per node.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the per-node flags are refused.
fn collapse_target_subtrees(
    eng: &Engine,
    vtree: &Vtree,
    f: &Tdd,
    g: &Tdd,
    mask: &mut [bool],
) -> Result<(), OperationError> {
    /// Every internal level of the subtree is in `mask` or marginal in an operand.
    const CLOSED: u8 = 1 << 0;
    /// No level of the subtree is marginal in an operand.
    const STRUCTURAL: u8 = 1 << 1;
    /// `f` has one node at every internal level of the subtree.
    const F_UNIT: u8 = 1 << 2;
    /// `g`, likewise.
    const G_UNIT: u8 = 1 << 3;
    /// A structural level of the subtree is over the sparse floor.
    const BIG: u8 = 1 << 4;
    /// A level of the subtree, structural or not, is over the sparse floor.
    const ANY_BIG: u8 = 1 << 5;
    /// A level above summed the subtree's levels out of `mask`.
    const CLEARED: u8 = 1 << 6;

    if f.is_zero() || g.is_zero() || !mask.contains(&true) {
        return Ok(());
    }
    let thresholds = sparse::sparse_thresholds(eng.limits());
    let (min_grid, sparsity) = (thresholds.min_grid as u64, thresholds.sparsity_factor);
    let is_big = |u: VtreeIdx| {
        let (fl, gl) = (f.level(u), g.level(u));
        price::over_floor(fl.slot_count() as u64 * gl.slot_count() as u64, fl, gl, min_grid)
    };
    let is_marginal = |u: VtreeIdx| f.level(u).is_marginal() || g.level(u).is_marginal();
    let relabels = |u: VtreeIdx| {
        single_pair(&Held::reading(f), u.idx(), f.level(u).slot_count()).is_some()
            || single_pair(&Held::reading(g), u.idx(), g.level(u).slot_count()).is_some()
    };

    let mut flags: Vec<u8> = Vec::new();
    eng.limits().try_resize(&mut flags, vtree.num_nodes(), 0)?;
    // Per subtree, over its levels marginal in neither operand: the grid
    // cells, and the cells weighted by the narrower operand's node count.
    let mut grids: Vec<(u128, u128)> = Vec::new();
    eng.limits().try_resize(&mut grids, vtree.num_nodes(), (0, 0))?;
    for u in vtree.bottomup() {
        let marginal = is_marginal(u);
        if vtree.node(u).is_leaf() {
            flags[u.idx()] = CLOSED | F_UNIT | G_UNIT | if marginal { 0 } else { STRUCTURAL };
            continue;
        }
        let (l, r) = vtree.children(u);
        let (fw, gw) = (f.level(u).slot_count() as u128, g.level(u).slot_count() as u128);
        let own = if marginal { (0, 0) } else { (fw * gw, fw * gw * fw.min(gw)) };
        let (gl, gr) = (grids[l.idx()], grids[r.idx()]);
        grids[u.idx()] = (
            own.0.saturating_add(gl.0).saturating_add(gr.0),
            own.1.saturating_add(gl.1).saturating_add(gr.1),
        );
        let (both, either) = (flags[l.idx()] & flags[r.idx()], flags[l.idx()] | flags[r.idx()]);
        let big = is_big(u);
        let mut here = either & (BIG | ANY_BIG);
        if mask[u.idx()] || marginal {
            here |= both & CLOSED;
        }
        if !marginal {
            here |= both & STRUCTURAL;
        }
        if f.level(u).slot_count() == 1 {
            here |= both & F_UNIT;
        }
        if g.level(u).slot_count() == 1 {
            here |= both & G_UNIT;
        }
        if big {
            here |= ANY_BIG;
        }
        if big && !marginal {
            here |= BIG;
        }
        flags[u.idx()] = here;
    }

    for c in vtree.bottomup().rev() {
        let here = flags[c.idx()];
        if here & CLEARED != 0 {
            mask[c.idx()] = false;
            if !vtree.node(c).is_leaf() {
                let (l, r) = vtree.children(c);
                flags[l.idx()] |= CLEARED;
                flags[r.idx()] |= CLEARED;
            }
            continue;
        }
        let Some(parent) = vtree.node(c).parent() else {
            continue;
        };
        if !mask[c.idx()] || here & CLOSED == 0 {
            continue;
        }
        let sibling = match vtree.children(parent) {
            (l, r) if l == c => r,
            (l, _) => l,
        };
        let structural = here & STRUCTURAL != 0;
        let parent_sums = structural
            && here & (F_UNIT | G_UNIT) != 0
            && f.level(parent).slot_count() == 1
            && g.level(parent).slot_count() == 1
            && (is_big(parent) || flags[sibling.idx()] & ANY_BIG != 0);
        let sized = here & BIG != 0 && may_be_sparse(grids[c.idx()], sparsity);
        if !sized && !parent_sums {
            continue;
        }
        let sibling_summed = is_marginal(sibling) || mask[sibling.idx()];
        if !structural || sibling_summed || (!parent_sums && relabels(c)) {
            mask[c.idx()] = false;
        }
        let (l, r) = vtree.children(c);
        flags[l.idx()] |= CLEARED;
        flags[r.idx()] |= CLEARED;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
use tests::{
    ahead_after_forced, ahead_forced_off, ands_buffered_forced, close_in_sweep_pairs, complete_sides_listed,
    dense_poll_stride_forced, grid_lookups_forced, lend_forced_off, lone_columns_forced_off,
    note_ahead, note_as_operand, note_closed_in_sweep, note_complete_side, note_held_tight,
    note_implied_walk, note_lent, note_lone_columns, note_lookups,
    note_one_sided_masks, note_pair_row_unrolled, note_pair_rows, note_pairs_grown,
    note_reach_columns_row, note_reach_tables, note_read_through, note_relabelled, note_relayed,
    note_route, note_row_columns_row, note_scheduled_charge, note_single_rows,
    note_whole_row, note_word_cells,
    one_sided_masks_forced_off, pair_rows_forced_off, reach_columns_forced_off, reach_tables_forced,
    relabel_forced_off, relay_forced_off, row_columns_forced_off, seed_pairs_forced,
    single_rows_forced_off, taken_levels_settled_forced, word_cells_forced,
    word_rows_sorted_forced,
    builds_past_cap_forced, note_refused_past_cap,
};

mod probes;
use probes::WordNote;
#[cfg(not(test))]
use probes::*;
