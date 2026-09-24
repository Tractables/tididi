//! Classify marginal children and prepare their decoders and liveness masks.
//!
//! [`Sides<T>`] holds the symmetric plans for the two children. Liveness masks
//! are populated only after the selected route materializes its child grids.

use crate::diagram::{ChildDecoder, ChildSide, Sides, Tdd, TddLevel};
use super::OperationError;
use crate::Engine;
use super::liveness::{bucket_shift, build_live_cols_bitmask, build_reach_masks, PrefilterSideMasks};
use super::setup::{ApplyRun, LevelShape, Operands};
use super::route::LevelMarg;

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
    /// `INLINE_VALUE_BIT` for the encoding.
    pub(crate) view: ChildDecoder,
}

impl SidePlan {
    /// True when this side carries an operand field through instead of
    /// gridding it.
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
pub(super) struct EntryMarginality(Option<Operands<Vec<bool>>>);

impl EntryMarginality {
    /// Snapshot both operands' per-level marginality, or nothing when neither
    /// carries a marginal level.
    pub(super) fn snapshot(f: &Tdd, g: &Tdd, num_nodes: usize, any: bool) -> Self {
        if !any {
            return EntryMarginality(None);
        }
        EntryMarginality(Some(Operands {
            f: (0..num_nodes).map(|i| f.levels[i].is_marginal()).collect(),
            g: (0..num_nodes).map(|i| g.levels[i].is_marginal()).collect(),
        }))
    }

    /// Whether operand `carrier`'s level `idx` was marginal at entry.
    fn was_marginal(&self, carrier: Carrier, idx: usize) -> bool {
        let Some(operands) = &self.0 else { return false };
        let operand = match carrier { Carrier::F => &operands.f, Carrier::G => &operands.g };
        operand[idx]
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
fn carrier(
    f: &Tdd,
    g: &Tdd,
    t_idx: usize,
    child_idx: usize,
    side: ChildSide,
    run: &ApplyRun,
) -> Option<Carrier> {
    let ApplyRun { f_identity, g_identity, entry_marginality: entry, .. } = run;
    // The carrier's per-pair field is an inline count or a tagged slot, never a
    // grid coordinate, and is copied into the output pair verbatim. Two facts
    // make that sound, and each side is tested on its own:
    //   (1) the opposite operand is the identity at the child, so the product
    //       there is the carrier's content unchanged; and
    //   (2) the carrier's field is a marginal ref: its child is marginal now,
    //       or was at entry and has since been swapped into the output by the
    //       identity fast path (the swap moves the store, so the slots stay
    //       valid), or the level's `has_value_refs` marker says the side's
    //       fields were already inlined.
    // Testing (1) alone would carry a structural node index into slot space;
    // testing (2) alone would read a slot as a grid coordinate. The output
    // child's own marginality is not tested here: its snapshot predates the
    // mid-loop cascade, and reads stale-false in the cells where the child
    // marginalizes mid-loop.
    let inlined = |f: &Tdd| f.levels[t_idx].has_value_refs(side);
    let left_ref = f.levels[child_idx].is_marginal()
        || entry.was_marginal(Carrier::F, child_idx) || inlined(f);
    if g_identity[child_idx] && left_ref {
        return Some(Carrier::F);
    }
    let right_ref = g.levels[child_idx].is_marginal()
        || entry.was_marginal(Carrier::G, child_idx) || inlined(g);
    if f_identity[child_idx] && right_ref {
        return Some(Carrier::G);
    }
    None
}


/// Refuse the forbidden marginal×marginal operand product.
///
/// Conjoining two marginal nodes is undefined: |f ∧ g| is not a function of |f|
/// and |g|, so there is no correct way to combine them in the product grid. A
/// marginal child must always be conjoined against an identity on the other
/// operand, since a marginalized scope is never re-constrained, and the
/// identity is what makes the child a pass-through carrier. Both operands
/// carrying a marginal level at the same child with no carrier means the logic
/// deciding when to marginalize is broken — a scope was summed out while a later
/// conjunction still constrained it — so this panics rather than silently
/// computing a wrong count.
///
/// The check is a debug assertion and is compiled out of a release build.
fn debug_assert_no_marginal_products(
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    passthrough: Sides<bool>,
) {
    let t_idx = shape.t.idx();
    let both_marginal = |child_idx: usize| f.levels[child_idx].is_marginal() && g.levels[child_idx].is_marginal();
    debug_assert!(
        !both_marginal(shape.left.idx()) || passthrough.left,
        "marginal×marginal product at left child {} (vtree {t_idx}): both operands carry \
         marginal counts and neither is the identity — a marginalized scope was re-constrained",
        shape.left.idx()
    );
    debug_assert!(
        !both_marginal(shape.right.idx()) || passthrough.right,
        "marginal×marginal product at right child {} (vtree {t_idx}): both operands carry \
         marginal counts and neither is the identity — a marginalized scope was re-constrained",
        shape.right.idx()
    );
}

/// Classify one level's two child sides: which are marginal, which are
/// pass-through carriers, how each side's refs decode, and whether the
/// dead-pair pre-filter applies.
///
/// This half reads no grid, so the caller can pick the route before
/// materializing any child grid. The liveness masks the `both_multi_pair` flag enables are
/// filled separately by `build_level_prefilter_masks`, which does read the grids.
///
/// A child is marginal in `marginal`'s wider sense, in the output or in
/// either operand: an identity shortcut can move a marginal child from an
/// operand to the output, and the remaining parent references still use that
/// child's marginal encoding.
pub(super) fn plan_marginal_level(
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    run: &ApplyRun,
    marginal: &LevelMarg,
) -> MarginalPlan {
    let (t, t_idx, left_idx, right_idx) = (shape.t, shape.t.idx(), shape.left.idx(), shape.right.idx());
    let (left_marginal, right_marginal) = (marginal.left_any, marginal.right_any);
    let carriers = Sides { left: left_idx, right: right_idx }
        .map(|side, child_idx| carrier(f, g, t_idx, child_idx, side, run));
    debug_assert_no_marginal_products(
        f, g, shape,
        Sides { left: carriers.left.is_some(), right: carriers.right.is_some() },
    );

    // Pass-through copies the encoded value verbatim. Stripping its inline
    // marker would turn an inline count into a slot reference.
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
pub(super) fn build_side_masks<const RIGHT: bool>(
    eng: &Engine,
    right_level: &TddLevel,
    right_width: usize,
    child: ChildGrid,
    node_idx: &[u32],
    out: &mut PrefilterSideMasks,
) -> Result<(), OperationError> {
    let ChildGrid { plan, f_width: f_child_width, g_width: g_child_width, base } = child;
    if plan.is_passthrough() {
        return Ok(());
    }
    let shift = bucket_shift(g_child_width);
    build_live_cols_bitmask(eng, f_child_width, g_child_width, base, node_idx, &mut out.live_cols, shift)?;
    let view = plan.view;
    build_reach_masks(eng, right_level, right_width, &mut out.reach,
        |p| view.coord(if RIGHT { p.right } else { p.left }) as usize, shift)
}
