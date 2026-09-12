//! Per-level marginal classification plan (`MarginalPlan` + `plan_marginal_level`) and the
//! dead-pair liveness masks (`build_side_masks`) for the apply product
//! construction.
//!
//! A level has exactly two child sides, and everything the product walk needs
//! to know about a side has the same shape on both. [`Sides<T>`] is that pair,
//! and every per-side quantity below is stored in one — so the left and right
//! halves of a computation are written once and applied twice, rather than
//! mirrored by hand. `MARGINAL_ENTRY_*` stay in `mod.rs` and are reached via
//! `super::`; the liveness bitmask kernels live in `super::liveness`.

use crate::vtree::VtreeIdx;
use crate::diagram::SideView;
use crate::diagram::*;
use super::ApplyError;
use crate::engine::Engine;
use super::liveness::{bucket_shift, build_live_cols_bitmask, build_reach_masks, PrefilterSideMasks};

/// One of a level's two child sides.
///
/// Only per-level code names a side at runtime; the cell kernel reaches the
/// halves of a [`Sides`] as `.left` / `.right` so the choice is made at compile
/// time (indexing by a runtime `Side` inside the walk would be a branch per
/// access).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    Left,
    Right,
}

/// One value per child side.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sides<T> {
    pub(crate) left: T,
    pub(crate) right: T,
}

impl<T> Sides<T> {
    /// Apply `f` to each side, told which side it is.
    pub(super) fn map<U>(self, mut f: impl FnMut(Side, T) -> U) -> Sides<U> {
        Sides { left: f(Side::Left, self.left), right: f(Side::Right, self.right) }
    }
}

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
    pub(crate) view: SideView,
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
#[allow(clippy::too_many_arguments)]
fn carrier(
    f: &Tdd,
    g: &Tdd,
    t_idx: usize,
    child_idx: usize,
    side: Side,
    left_identity: &[bool],
    right_identity: &[bool],
    entry: &EntryMarginality,
) -> Option<Carrier> {
    // ── Pass-through: a marginal child meets an identity operand ──
    // Carrier: on a pass-through side, one operand holds the marginal child and
    // the other is identity there; the operand holding it is that side's
    // *carrier*, and its raw per-pair field is copied into the output verbatim.
    // The carrier may hold the child in its own level, or the level may already
    // have been swapped into the output accumulator by the identity fast path —
    // either way the field is an inline count or a tagged slot, never a grid
    // coordinate. This is the one definition of the term.
    //
    // When a child side's level is marginal in one operand while the other
    // operand is constant-true (identity) at that subtree, the marginal
    // operand's per-pair field is its inline model count (or a tagged
    // big-count slot), not a structural grid coordinate. Using it to index
    // the child product grid would read far out of bounds. Instead we copy
    // the marginal ("carrier") operand's raw field straight into the output
    // pair — the child grid is never consulted on that side, so the inline
    // count survives the apply verbatim.
    //
    // This is exactly the situation throughout CNF compilation: a clause
    // never mentions variables under a marginalized vtree subtree, so the
    // clause's function there is constant-true (a single width-1 One node);
    // and when conjoining two child sub-diagrams over disjoint variable sets,
    // each is identity on the other's subtree. The only place two genuinely
    // marginal sides meet is same-left pair fusion, which has its own inner and never
    // reaches this apply.
    //
    // Sides are independent: a level can be left-passthrough and right-real,
    // or both.
    //
    // Pass-through carries the carrier operand's raw per-pair marginal field
    // straight into the output, where the streaming sum reads it as a
    // slot/inline-count against the output child store. Two things must both
    // hold for that to be sound:
    //   (1) output child marginal — else the carried value is read as a
    //       structural node index, not a marginal slot; and
    //   (2) the carrier operand's field is itself a marginal ref — its child is
    //       marginal now, or was marginal at entry (`MARGINAL_ENTRY_*`) and got
    //       stolen into the output store earlier in this apply (an FP1/FP2
    //       mem::swap moves the store verbatim, so the carrier's slots stay
    //       valid against the output store), OR the parent-level marker
    //       (`marginal_inlined_*`, on t_idx — survives a child swap) says this
    //       side's pair fields were already inlined.
    // Both conjuncts are load-bearing, and either one alone segfaults: keying
    // on (1) only carries a genuinely structural node index into slot space,
    // and testing the carrier only drops the stolen-marginal case so a slot is
    // grid-read as a coordinate.
    //
    // Do not `&&` an `levels[child_idx].is_marginal()` conjunct here: that
    // snapshot predates the mid-loop cascade (it is recomputed post-cascade
    // further down), so it reads stale-false in exactly the cells where the
    // child marginalizes mid-loop, forcing them onto the grid path — an inline
    // overcount. It is redundant anyway: carrying marginal content through an
    // identity side always yields a marginal output.
    //
    // Requiring the `*_ref` conjunct keeps a genuinely structural carrier
    // (operands structural, output marginalized mid-loop by the cascade) on the
    // grid path, where its refs are structural indices and grid-safe.
    let inlined = |f: &Tdd| match side {
        Side::Left => f.levels[t_idx].marginal_inlined_left(),
        Side::Right => f.levels[t_idx].marginal_inlined_right(),
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
/// deciding when to marginalize is broken — a scope was summed out while a later
/// conjunction still constrained it — so this panics rather than silently
/// computing a wrong count.
///
/// The check is a debug assertion and is compiled out of a release build.
// The assertions are written as the negation of the forbidden shape so the
// condition reads as the invariant it guards; De Morgan's form does not.
#[allow(clippy::nonminimal_bool)]
#[allow(clippy::too_many_arguments)]
fn debug_assert_no_marginal_products(
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    left_identity: &[bool],
    right_identity: &[bool],
    left_passthrough: bool,
    right_passthrough: bool,
) {
    debug_assert!(
        !(f.levels[left_idx].is_marginal() && g.levels[left_idx].is_marginal()
            && !left_identity[left_idx] && !right_identity[left_idx]),
        "marginal×marginal product at left child {left_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize scheduling is unsound \
         (a marginalized scope was re-constrained)"
    );
    debug_assert!(
        !(f.levels[right_idx].is_marginal() && g.levels[right_idx].is_marginal()
            && !left_identity[right_idx] && !right_identity[right_idx]),
        "marginal×marginal product at right child {right_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize scheduling is unsound \
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
/// filled separately by `build_prefilter_masks`, which does read the grids.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_marginal_level(
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    levels: &[TddLevel],
    left_identity: &[bool],
    right_identity: &[bool],
    entry: &EntryMarginality,
) -> MarginalPlan {
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
    // Checking only the output level would miss cases 2/3: an operand's
    // bit-30-tagged ref would reach the grid lookup raw as `(1<<30)+base`, far
    // past `node_idx.len()`, and read out of bounds. A single shared
    // mask per side decodes both operands: `decode_marginal_coord(.., MARGINAL_VALUE_MASK)`
    // is a harmless no-op on a bare ref (real node indices never set bit-30;
    // the zero sentinel is bit-31 and is preserved), so over-masking the
    // non-marginal operand costs nothing. This per-level (not per-cell) check
    // adds two `Option::is_some` reads — negligible.
    let left_marginal = levels[left_idx].is_marginal()
        || f.levels[left_idx].is_marginal()
        || g.levels[left_idx].is_marginal();
    let right_marginal = levels[right_idx].is_marginal()
        || f.levels[right_idx].is_marginal()
        || g.levels[right_idx].is_marginal();
    let carriers = Sides { left: left_idx, right: right_idx }.map(|side, child_idx| {
        carrier(f, g, t_idx, child_idx, side, left_identity, right_identity, entry)
    });
    debug_assert_no_marginal_products(
        f, g, t, t_idx, left_idx, right_idx, left_identity, right_identity,
        carriers.left.is_some(), carriers.right.is_some(),
    );

    // A pass-through side reads structurally even when its child is marginal:
    // the carrier field's tag bit (inline count vs big-count slot) must survive
    // verbatim, and stripping it would corrupt a big slot into a misread count.
    let side_view = |carrier: Option<Carrier>, marginal: bool| {
        if carrier.is_none() && marginal { SideView::marginal() } else { SideView::structural() }
    };
    let sides = Sides { left: (carriers.left, left_marginal), right: (carriers.right, right_marginal) }
        .map(|_, (carrier, marginal)| SidePlan { carrier, view: side_view(carrier, marginal) });
    // ── dead-pair pre-filter (per-level setup) ────────────────
    // Masks are bit-exact for child widths ≤ 128 and bucketed (shift > 0,
    // sound-with-false-positives) above — see liveness.rs.
    let both_multi_pair = f.level(t).has_multi_pair() && g.level(t).has_multi_pair();
    // A pass-through side has no product grid to filter against; no
    // structures are built for it and every consumer below guards with
    // !*_passthrough, treating that side as unconditionally alive.
    //
    // The dead-pair liveness masks (which read the materialized child
    // grids via `node_idx`) are built separately in `build_prefilter_masks`, called
    // only when `both_multi_pair` holds and after the child grids exist. Splitting that grid read
    // out of the flags lets the caller compute the route (plain-dense vs not)
    // before materializing — so a sparse child under a dense parent on the
    // plain-dense route can skip `ensure_grid` entirely. `both_multi_pair` implies the
    // general (non-plain-dense) path, so the grids are always materialized by
    // the time `build_prefilter_masks` runs.

    MarginalPlan { sides, both_multi_pair }
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
#[allow(clippy::too_many_arguments)]
pub(super) fn build_side_masks<const RIGHT: bool>(
    eng: &Engine,
    right_level: &TddLevel,
    right_width: usize,
    plan: SidePlan,
    k1_child: usize,
    k2_child: usize,
    base: usize,
    node_idx: &[u32],
    out: &mut PrefilterSideMasks,
) -> Result<(), ApplyError> {
    if plan.is_passthrough() {
        return Ok(());
    }
    let shift = bucket_shift(k2_child);
    build_live_cols_bitmask(eng, k1_child, k2_child, base, node_idx, &mut out.live_cols, shift)?;
    let view = plan.view;
    build_reach_masks(eng, right_level, right_width, &mut out.reach,
        |p| view.coord(if RIGHT { p.right } else { p.left }).idx(), shift)
}
