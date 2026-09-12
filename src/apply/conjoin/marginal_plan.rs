//! Per-level marginal classification plan (`MarginalPlan` + `plan_marginal_level`) and the
//! dead-pair liveness masks (`build_side_masks`) for the apply product
//! construction.
//!
//! A level has exactly two child sides, and everything the product walk needs
//! to know about a side has the same shape on both. [`Sides<T>`] is that pair,
//! and every per-side quantity below is stored in one — so the left and right
//! halves of a computation are written once and applied twice, rather than
//! mirrored by hand. The liveness bitmask kernels live in `super::liveness`.

use crate::diagram::ChildDecoder;
use crate::diagram::*;
use super::OperationError;
use crate::engine::Engine;
use super::liveness::{bucket_shift, build_live_cols_bitmask, build_reach_masks, PrefilterSideMasks};
use super::setup::{ApplyRun, LevelShape};

/// Which operand supplies a pass-through side's per-pair field.
///
/// See [`carrier`] for what a carrier is; the walk copies the named operand's
/// field verbatim (`F` ⇒ the f pair's field, `G` ⇒ the g pair's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Carrier {
    F,
    G,
}

/// How one child side of a level is read: pass-through carrier (if any) and
/// the decode the side's refs need.
#[derive(Clone, Copy, Debug)]
pub(super) struct SidePlan {
    /// `Some` ⇒ this side is a pass-through carrier and has no product grid.
    pub(crate) carrier: Option<Carrier>,
    /// Decode for this side's pair fields: `MARGINAL_VALUE_MASK` when the child is
    /// marginal, so a bit-30 inline tag is stripped and the remaining payload
    /// is read as a coordinate; an identity view otherwise. See
    /// `MARGINAL_OVERFLOW_TAG` for the encoding.
    pub(crate) view: ChildDecoder,
}

impl SidePlan {
    /// True when this side carries an operand field through instead of
    /// gridding it.
    #[inline(always)]
    pub(super) fn is_passthrough(&self) -> bool {
        self.carrier.is_some()
    }
}

/// Which levels of each operand were marginal when the apply began.
///
/// An identity fast path can steal a marginal level out of an operand and into
/// the output mid-sweep, after which the operand's own level no longer reads as
/// marginal — but its pair fields are still marginal refs. The carrier test
/// needs to see that, so the two operands' marginality is snapshotted before
/// the sweep starts.
///
/// `None` means no operand level was marginal at entry, so no snapshot was
/// taken and every lookup is `false`. That is the dominant case — pure Boolean
/// and plain model counting — and it is why this is an `Option` rather than two
/// all-false vectors: the absence of a snapshot is itself the skip.
pub(super) struct EntryMarginality(Option<Sides<Vec<bool>>>);

impl EntryMarginality {
    /// Snapshot both operands' per-level marginality, or nothing when neither
    /// carries a marginal level.
    pub(super) fn snapshot(f: &Tdd, g: &Tdd, num_nodes: usize, any: bool) -> Self {
        if !any {
            return EntryMarginality(None);
        }
        EntryMarginality(Some(Sides {
            left: (0..num_nodes).map(|i| f.levels[i].is_marginal()).collect(),
            right: (0..num_nodes).map(|i| g.levels[i].is_marginal()).collect(),
        }))
    }

    /// Whether operand `carrier`'s level `idx` was marginal at entry.
    #[inline(always)]
    fn was_marginal(&self, carrier: Carrier, idx: usize) -> bool {
        let Some(sides) = &self.0 else { return false };
        let side = match carrier { Carrier::F => &sides.left, Carrier::G => &sides.right };
        side.get(idx).copied().unwrap_or(false)
    }
}

/// Per-level marginal classification plan produced by [`plan_marginal_level`].
#[derive(Clone, Copy)]
pub(super) struct MarginalPlan {
    /// How each child side is read.
    pub(crate) sides: Sides<SidePlan>,
    /// True when both operands have multi-pair nodes at this level, so the
    /// dead-pair pre-filter applies and its masks are worth building.
    pub(crate) both_multi_pair: bool,
}

/// Whether one child side is a pass-through carrier, and which operand carries it.
///
/// A side is a carrier when the level's refs into that child are marginal-encoded
/// and the opposite operand is the identity there, so the level's output can
/// carry the marginal child's refs across unchanged instead of re-deriving
/// them. Both operands are tested because either may be the identity, and
/// the entry snapshot lets the test see a child that was marginal at entry
/// but has since been stolen into the output by an identity swap.
#[inline(always)]
fn carrier(
    f: &Tdd,
    g: &Tdd,
    t_idx: usize,
    child_idx: usize,
    side: ChildSide,
    run: &ApplyRun,
) -> Option<Carrier> {
    let ApplyRun { left_identity, right_identity, entry_marginality: entry, .. } = run;
    // The carrier's per-pair field is an inline count or a tagged slot, never a
    // grid coordinate, and is copied into the output pair verbatim. Two facts
    // make that sound, and each side is tested on its own:
    //   (1) the opposite operand is the identity at the child, so the product
    //       there is the carrier's content unchanged; and
    //   (2) the carrier's field is a marginal ref: its child is marginal now,
    //       or was at entry and has since been swapped into the output by the
    //       identity fast path (the swap moves the store, so the slots stay
    //       valid), or the level's `marginal_inlined_*` marker says the side's
    //       fields were already inlined.
    // Testing (1) alone would carry a structural node index into slot space;
    // testing (2) alone would read a slot as a grid coordinate. The output
    // child's own marginality is not tested here: its snapshot predates the
    // mid-loop cascade, and reads stale-false in the cells where the child
    // marginalizes mid-loop.
    let inlined = |f: &Tdd| match side {
        ChildSide::Left => f.levels[t_idx].marginal_inlined_left(),
        ChildSide::Right => f.levels[t_idx].marginal_inlined_right(),
    };
    let left_ref = f.levels[child_idx].is_marginal()
        || entry.was_marginal(Carrier::F, child_idx) || inlined(f);
    if right_identity[child_idx] && left_ref {
        return Some(Carrier::F);
    }
    let right_ref = g.levels[child_idx].is_marginal()
        || entry.was_marginal(Carrier::G, child_idx) || inlined(g);
    if left_identity[child_idx] && right_ref {
        return Some(Carrier::G);
    }
    None
}


/// Refuse the forbidden marginal×marginal operand product.
///
/// Conjoining two marginal nodes is undefined: |f ∧ g| is not a function of |f|
/// and |g|, so there is no correct way to combine them in the product grid. A
/// marginal child must always be conjoined against an identity on the other
/// operand, since a marginalized scope is never re-constrained. Both operands
/// carrying a non-identity marginal level at the same child means the logic
/// deciding when to marginalize_levels is broken — a scope was summed out while a later
/// conjunction still constrained it — so this panics rather than silently
/// computing a wrong count.
///
/// The check is a debug assertion and is compiled out of a release build.
// The assertions are written as the negation of the forbidden shape so the
// condition reads as the invariant it guards; De Morgan's form does not.
#[allow(clippy::nonminimal_bool)]
fn debug_assert_no_marginal_products(
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    run: &ApplyRun,
    passthrough: Sides<bool>,
) {
    let (t, t_idx, left_idx, right_idx) = (shape.t, shape.t.idx(), shape.left.idx(), shape.right.idx());
    let ApplyRun { left_identity, right_identity, .. } = run;
    let Sides { left: left_passthrough, right: right_passthrough } = passthrough;
    debug_assert!(
        !(f.levels[left_idx].is_marginal() && g.levels[left_idx].is_marginal()
            && !left_identity[left_idx] && !right_identity[left_idx]),
        "marginal×marginal product at left child {left_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize_levels scheduling is unsound \
         (a marginalized scope was re-constrained)"
    );
    debug_assert!(
        !(f.levels[right_idx].is_marginal() && g.levels[right_idx].is_marginal()
            && !left_identity[right_idx] && !right_identity[right_idx]),
        "marginal×marginal product at right child {right_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize_levels scheduling is unsound \
         (a marginalized scope was re-constrained)"
    );
    // Hard case: two genuinely marginal sides with neither identity. This is
    // same-left pair fusion territory and must never reach the clause/child-merge apply.
    // Fires loud in debug if the disjoint-subtree assumption is ever violated.
    debug_assert!(
        !(f.levels[left_idx].is_marginal() && g.levels[left_idx].is_marginal()) || left_passthrough,
        "two marginal left operands, neither identity — unexpected outside same-left pair fusion (t={t:?})");
    debug_assert!(
        !(f.levels[right_idx].is_marginal() && g.levels[right_idx].is_marginal()) || right_passthrough,
        "two marginal right operands, neither identity — unexpected outside same-left pair fusion (t={t:?})");
}

/// Classify one level's two child sides: which are marginal, which are
/// pass-through carriers, how each side's refs decode, and whether the
/// dead-pair pre-filter applies.
///
/// This half reads no grid, so the caller can pick the route before
/// materializing any child grid. The liveness masks the `both_multi_pair` flag enables are
/// filled separately by `build_level_prefilter_masks`, which does read the grids.
pub(super) fn plan_marginal_level(
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    run: &ApplyRun,
) -> MarginalPlan {
    let (t, t_idx, left_idx, right_idx) = (shape.t, shape.t.idx(), shape.left.idx(), shape.right.idx());
    let levels = &run.levels[..];
    // ── Marg-side structural decode masks (per-child-side) ──
    // A child level that is marginal stores its parent's refs to it as
    // bit-30-tagged slot indices (the end-of-apply tagger). Every place that
    // consumes such a ref as a *structural* coordinate (grid stride/column,
    // reach/liveness array index) must strip the tag first. The masks are
    // loop-invariant per level: `MARGINAL_VALUE_MASK` strips the tag for a marginal
    // child, `u32::MAX` is an identity no-op otherwise.
    //
    // A tagged ref appears whenever the child level it points into is
    // marginal. That marginal status can live in three places, and we must
    // strip if any holds:
    //   1. the output child level (`levels[..]`) — when a marginal child was
    //      processed earlier this apply, the identity fast-path swapped it
    //      out of the operand and into `levels[child_idx]`;
    //   2/3. an operand child level (`f/g.levels[..]`) — when a genuinely
    //      marginal operand level is consumed directly (no identity swap),
    //      e.g. a streaming accumulator that a prior step already
    //      marginalized + tagged, while the output level is not marked
    //      marginal until the post-step `marginalize_batch`. The end-of-apply
    //      tagger keys on exactly this operand-child marginal status
    //      (`tag_all_marginal_side_slots`), so the decode mask must mirror it.
    // One mask per side serves both operands: masking a bare ref is a no-op,
    // since real node indices never set bit 30 and the zero sentinel is bit 31.
    let left_marginal = levels[left_idx].is_marginal()
        || f.levels[left_idx].is_marginal()
        || g.levels[left_idx].is_marginal();
    let right_marginal = levels[right_idx].is_marginal()
        || f.levels[right_idx].is_marginal()
        || g.levels[right_idx].is_marginal();
    let carriers = Sides { left: left_idx, right: right_idx }
        .map(|side, child_idx| carrier(f, g, t_idx, child_idx, side, run));
    debug_assert_no_marginal_products(
        f, g, shape, run,
        Sides { left: carriers.left.is_some(), right: carriers.right.is_some() },
    );

    // A pass-through side reads structurally even when its child is marginal:
    // the carrier field's tag bit (inline count vs big-count slot) must survive
    // verbatim, and stripping it would corrupt a big slot into a misread count.
    let child_decoder = |carrier: Option<Carrier>, marginal: bool| {
        if carrier.is_none() && marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() }
    };
    let sides = Sides { left: (carriers.left, left_marginal), right: (carriers.right, right_marginal) }
        .map(|_, (carrier, marginal)| SidePlan { carrier, view: child_decoder(carrier, marginal) });
    // ── dead-pair pre-filter (per-level setup) ────────────────
    // Masks are bit-exact for child widths ≤ 128 and bucketed (shift > 0,
    // sound-with-false-positives) above — see liveness.rs.
    let both_multi_pair = f.level(t).has_multi_pair() && g.level(t).has_multi_pair();
    // A pass-through side has no product grid to filter against, so it is
    // treated as alive throughout. The liveness masks, which read the child
    // grids, are built by `build_level_prefilter_masks` once the grids exist and
    // only when `both_multi_pair` holds.

    MarginalPlan { sides, both_multi_pair }
}

/// One child side's materialized product grid, as the liveness masks read it.
#[derive(Clone, Copy)]
pub(super) struct ChildGrid {
    /// How the side is read.
    pub(super) plan: SidePlan,
    /// f's width at the child: the grid's row count.
    pub(super) f_width: usize,
    /// g's width at the child: the grid's column count.
    pub(super) g_width: usize,
    /// Flat base offset of the grid in `node_idx`.
    pub(super) base: usize,
}

/// One child side's dead-pair liveness masks.
///
/// This is the grid-reading half of the marginal plan: it fills the side's
/// liveness scratch (`live_cols`, `reach`) by scanning the materialized child
/// grid through `node_idx`. Call only when [`MarginalPlan::both_multi_pair`] holds and after
/// the child grid exists. Masks are bit-exact for child widths ≤ 128 and
/// bucketed (shift > 0, sound-with-false-positives) above — see liveness.rs.
///
/// `RIGHT` names the side, so the per-pair field the reach masks read is
/// picked at compile time. A pass-through side is skipped entirely: the
/// marginal child has no product grid, and its field is a model count, not a
/// row/column index, so the builders below would index out of bounds.
#[inline(always)]
pub(super) fn build_side_masks<const RIGHT: bool>(
    eng: &Engine,
    right_level: &TddLevel,
    right_width: usize,
    child: ChildGrid,
    node_idx: &[u32],
    out: &mut PrefilterSideMasks,
) -> Result<(), OperationError> {
    let ChildGrid { plan, f_width: k1_child, g_width: k2_child, base } = child;
    if plan.is_passthrough() {
        return Ok(());
    }
    let shift = bucket_shift(k2_child);
    build_live_cols_bitmask(eng, k1_child, k2_child, base, node_idx, &mut out.live_cols, shift)?;
    let view = plan.view;
    build_reach_masks(eng, right_level, right_width, &mut out.reach,
        |p| view.coord(if RIGHT { p.right } else { p.left }).idx(), shift)
}
