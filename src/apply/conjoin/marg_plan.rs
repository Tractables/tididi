//! Per-level marg classification plan (`MargPlan` + `plan_marg_level`) and the
//! NxM dead-pair liveness masks (`build_side_masks`) for the apply product
//! construction.
//!
//! A level has exactly two child sides, and everything the product walk needs
//! to know about a side has the same shape on both. [`Sides<T>`] is that pair,
//! and every per-side quantity below is stored in one — so the left and right
//! halves of a computation are written once and applied twice, rather than
//! mirrored by hand. `MARG_ENTRY_*` stay in `mod.rs` and are reached via
//! `super::`; the liveness bitmask kernels live in `super::liveness`.

use crate::vtree::VtreeIdx;
use crate::diagram::SideView;
use crate::diagram::*;
use super::ApplyError;
use crate::engine::Engine;
use super::liveness::{bucket_shift, build_live_cols_bitmask, build_reach_masks, NxmSideMasks};

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
    pub left: T,
    pub right: T,
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
/// field verbatim (`C1` ⇒ the c1 pair's field, `C2` ⇒ the c2 pair's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Carrier {
    C1,
    C2,
}

/// How one child side of a level is read: pass-through carrier (if any) and
/// the decode the side's refs need.
#[derive(Clone, Copy, Debug)]
pub(super) struct SidePlan {
    /// `Some` ⇒ this side is a pass-through carrier and has no product grid.
    pub carrier: Option<Carrier>,
    /// Decode for this side's pair fields: `MARG_VALUE_MASK` when the child is
    /// marginal, so a bit-30 inline tag is stripped and the remaining payload
    /// is read as a coordinate; an identity view otherwise. See
    /// `MARG_OVERFLOW_TAG` for the encoding.
    pub view: SideView,
}

impl SidePlan {
    /// True when this side carries an operand field through instead of
    /// gridding it.
    #[inline(always)]
    pub(super) fn is_passthrough(&self) -> bool {
        self.carrier.is_some()
    }
}

/// Per-level marg classification plan produced by [`plan_marg_level`].
pub(super) struct MargPlan {
    /// How each child side is read.
    pub sides: Sides<SidePlan>,
    /// True when both operands have multi-pair nodes at this level, so the NxM
    /// dead-pair pre-filter applies and its masks are worth building.
    pub nxm: bool,
}

/// Whether one child side is a pass-through carrier, and which operand carries it.
///
/// A side is a carrier when the level's refs into that child are marg-encoded
/// and the opposite operand is the identity there, so the level's output can
/// carry the marginal child's refs across unchanged instead of re-deriving
/// them. Both operands are tested because either may be the identity, and
/// `any_entry_marginal` lets the test see a child that WAS marginal at entry
/// but has since been stolen into the output by an identity swap.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn carrier(
    eng: &Engine,
    c1: &Tdd,
    c2: &Tdd,
    t_idx: usize,
    child_idx: usize,
    side: Side,
    c1_identity: &[bool],
    c2_identity: &[bool],
    any_entry_marginal: bool,
) -> Option<Carrier> {
    // ── Pass-through: a marginal child meets an identity operand ──
    // CARRIER. On a pass-through side, one operand holds the marginal child and
    // the other is identity there; the operand holding it is that side's
    // *carrier*, and its raw per-pair field is copied into the output verbatim.
    // The carrier may hold the child in its own level, or the level may already
    // have been swapped into the OUTPUT accumulator by the identity fast path —
    // either way the field is an inline count or a tagged slot, never a grid
    // coordinate. This is the one definition of the term.
    //
    // When a child side's level is marginal in one operand AND the other
    // operand is constant-true (identity) at that subtree, the marginal
    // operand's per-pair field is its INLINE MODEL COUNT (or a tagged
    // big-count slot), NOT a structural grid coordinate. Using it to index
    // the child product grid would read far out of bounds. Instead we copy
    // the marginal ("carrier") operand's raw field straight into the output
    // pair — the child grid is never consulted on that side, so the inline
    // count survives the apply verbatim.
    //
    // This is exactly the situation throughout CNF compilation: a clause
    // never mentions variables under a marginalized vtree subtree, so the
    // clause's function there is constant-true (a single width-1 One node);
    // and when conjoining two child sub-TDDs over disjoint variable sets,
    // each is identity on the other's subtree. The only place two genuinely
    // marginal sides meet is same-left pair fusion, which has its own inner and never
    // reaches this apply.
    //
    // Sides are independent: a level can be left-passthrough and right-real,
    // or both.
    //
    // Pass-through carries the carrier operand's raw per-pair marg field
    // straight into the output, where the streaming sum reads it as a
    // slot/inline-count against the OUTPUT child store. Two things must BOTH
    // hold for that to be sound:
    //   (1) output child marginal — else the carried value is read as a
    //       structural node index, not a marginal slot; and
    //   (2) the carrier operand's field IS a marginal ref — its child is
    //       marginal now, OR was marginal AT ENTRY (`MARG_ENTRY_*`) and got
    //       stolen into the output store earlier THIS apply (an FP1/FP2
    //       mem::swap moves the store verbatim, so the carrier's slots stay
    //       valid against the output store), OR the parent-level marker
    //       (`marg_inlined_*`, on t_idx — survives a child swap) says this
    //       side's pair fields were already inlined.
    // BOTH conjuncts are load-bearing, and either one alone segfaults: keying
    // on (1) only carries a genuinely structural node index into slot space,
    // and testing the carrier only drops the stolen-marginal case so a slot is
    // grid-read as a coordinate. The reexpand baseline is the one exception —
    // it keys on the output alone, which is safe there because reexpand
    // un-inlines at apply entry.
    // `any_entry_marginal == false` ⇒ `apply_and_setup` CLEARED both snapshots
    // instead of filling them (setup.rs), so `get(idx)` is `None` at every index
    // and both lookups are constantly `false`. Short-circuiting on the flag is
    // therefore value-identical and skips two `RefCell` borrows per side on the
    // dominant pure-Boolean / MC fold path, where no operand level is marginal.
    //
    // Do NOT `&&` an `levels[child_idx].is_marginal()` conjunct here: that
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
        Side::Left => f.levels[t_idx].marg_inlined_left(),
        Side::Right => f.levels[t_idx].marg_inlined_right(),
    };
    let entry = |snapshot: &std::cell::RefCell<Vec<bool>>| any_entry_marginal
        && snapshot.borrow().get(child_idx).copied().unwrap_or(false);
    let c1_ref = c1.levels[child_idx].is_marginal()
        || entry(&eng.apply().marg_entry_c1) || inlined(c1);
    if c2_identity[child_idx] && c1_ref {
        return Some(Carrier::C1);
    }
    let c2_ref = c2.levels[child_idx].is_marginal()
        || entry(&eng.apply().marg_entry_c2) || inlined(c2);
    if c1_identity[child_idx] && c2_ref {
        return Some(Carrier::C2);
    }
    None
}


/// Refuse the forbidden marginal×marginal operand product.
///
/// Conjoining two marginal nodes is undefined: |f ∧ g| is not a function of |f|
/// and |g|, so there is no correct way to combine them in the product grid. A
/// marginal child must always be conjoined against IDENTITY on the other
/// operand, since a marginalized scope is never re-constrained. Both operands
/// carrying a non-identity marginal level at the same child means the logic
/// deciding WHEN to marginalize is broken — a scope was summed out while a later
/// conjunction still constrained it — so this panics rather than silently
/// computing a wrong count.
///
/// Debug-only: the case has been confirmed not to occur even with the check
/// armed as a release assert. To run it in an optimized binary, build
/// `--profile release-checked`.
// The assertions are written as the negation of the forbidden shape so the
// condition reads as the invariant it guards; De Morgan's form does not.
#[allow(clippy::nonminimal_bool)]
#[allow(clippy::too_many_arguments)]
fn debug_assert_no_marginal_products(
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    c1_identity: &[bool],
    c2_identity: &[bool],
    left_passthrough: bool,
    right_passthrough: bool,
) {
    debug_assert!(
        !(c1.levels[left_idx].is_marginal() && c2.levels[left_idx].is_marginal()
            && !c1_identity[left_idx] && !c2_identity[left_idx]),
        "marginal×marginal product at left child {left_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize scheduling is unsound \
         (a marginalized scope was re-constrained)"
    );
    debug_assert!(
        !(c1.levels[right_idx].is_marginal() && c2.levels[right_idx].is_marginal()
            && !c1_identity[right_idx] && !c2_identity[right_idx]),
        "marginal×marginal product at right child {right_idx} (vtree {t_idx}): both \
         operands carry non-identity marginal counts — marginalize scheduling is unsound \
         (a marginalized scope was re-constrained)"
    );
    // Hard case: two genuinely marginal sides with neither identity. This is
    // same-left pair fusion territory and must never reach the clause/child-merge apply.
    // Fires loud in debug if the disjoint-subtree assumption is ever violated.
    debug_assert!(
        !(c1.levels[left_idx].is_marginal() && c2.levels[left_idx].is_marginal()) || left_passthrough,
        "two marginal left operands, neither identity — unexpected outside same-left pair fusion (t={t:?})");
    debug_assert!(
        !(c1.levels[right_idx].is_marginal() && c2.levels[right_idx].is_marginal()) || right_passthrough,
        "two marginal right operands, neither identity — unexpected outside same-left pair fusion (t={t:?})");
}

/// Classify one level's two child sides: which are marginal, which are
/// pass-through carriers, how each side's refs decode, and whether the NxM
/// dead-pair pre-filter applies.
///
/// This half reads no grid, so the caller can pick the route before
/// materializing any child grid. The liveness masks the `nxm` flag enables are
/// filled separately by `build_nxm_masks`, which does read the grids.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_marg_level(
    eng: &Engine,
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    levels: &[TddLevel],
    c1_identity: &[bool],
    c2_identity: &[bool],
    any_entry_marginal: bool,
) -> MargPlan {
    // ── Marg-side structural decode masks (per-child-side) ──
    // A child level that is marginal stores its parent's refs to it as
    // bit-30-tagged slot indices (the end-of-apply tagger). Every place that
    // consumes such a ref as a *structural* coordinate (grid stride/column,
    // reach/liveness array index) must strip the tag first. The masks are
    // loop-invariant per level: MARG_VALUE_MASK strips the tag for a marginal
    // child, u32::MAX is an identity no-op otherwise.
    //
    // A tagged ref appears whenever the child level it points INTO is
    // marginal. That marginal status can live in THREE places, and we must
    // strip if ANY holds:
    //   1. the OUTPUT child level (`levels[..]`) — when a marginal child was
    //      processed earlier this apply, the identity fast-path swapped it
    //      OUT of the operand and INTO `levels[child_idx]`;
    //   2/3. an OPERAND child level (`c1/c2.levels[..]`) — when a genuinely
    //      marginal operand level is consumed DIRECTLY (no identity swap),
    //      e.g. a streaming accumulator that a prior step already
    //      marginalized + tagged, while the output level is not marked
    //      marginal until the post-step `marginalize_batch`. The end-of-apply
    //      tagger keys on exactly this operand-child marginal status
    //      (`tag_all_marg_side_slots`), so the decode mask must mirror it.
    // The output-only check missed cases 2/3 → an operand's bit-30-tagged
    // ref reached the grid lookup raw as `(1<<30)+base` ≫ node_idx.len() → OOB
    // segfault. A single shared
    // mask per side decodes both operands: `decode_marg_coord(.., MARG_VALUE_MASK)`
    // is a harmless no-op on a bare ref (real node indices never set bit-30;
    // the ZERO sentinel is bit-31 and is preserved), so over-masking the
    // non-marginal operand costs nothing. This per-level (not per-cell) check
    // adds two `Option::is_some` reads — negligible.
    let left_marg = levels[left_idx].is_marginal()
        || c1.levels[left_idx].is_marginal()
        || c2.levels[left_idx].is_marginal();
    let right_marg = levels[right_idx].is_marginal()
        || c1.levels[right_idx].is_marginal()
        || c2.levels[right_idx].is_marginal();
    let carriers = Sides { left: left_idx, right: right_idx }.map(|side, child_idx| {
        carrier(eng, c1, c2, t_idx, child_idx, side,
            c1_identity, c2_identity, any_entry_marginal)
    });
    debug_assert_no_marginal_products(
        c1, c2, t, t_idx, left_idx, right_idx, c1_identity, c2_identity,
        carriers.left.is_some(), carriers.right.is_some(),
    );

    // A pass-through side reads structurally even when its child is marginal:
    // the carrier field's tag bit (inline count vs big-count slot) must survive
    // verbatim, and stripping it would corrupt a big slot into a misread count.
    let side_view = |carrier: Option<Carrier>, marg: bool| {
        if carrier.is_none() && marg { SideView::valued() } else { SideView::structural() }
    };
    let sides = Sides { left: (carriers.left, left_marg), right: (carriers.right, right_marg) }
        .map(|_, (carrier, marg)| SidePlan { carrier, view: side_view(carrier, marg) });
    // ── NxM dead-pair pre-filter (per-level setup) ────────────────
    // Masks are bit-exact for child widths ≤ 128 and bucketed (shift > 0,
    // sound-with-false-positives) above — see liveness.rs.
    let nxm = c1.level(t).has_multi_pair() && c2.level(t).has_multi_pair();
    // A pass-through side has no product grid to filter against; no
    // structures are built for it and every consumer below guards with
    // !*_passthrough, treating that side as unconditionally alive.
    //
    // The NxM dead-pair liveness masks (which read the materialized child
    // grids via `node_idx`) are built separately in `build_nxm_masks`, called
    // only when `nxm` AND after the child grids exist. Splitting that grid read
    // out of the flags lets the caller compute the route (plain-dense vs not)
    // BEFORE materializing — so a sparse child under a dense parent on the
    // plain-dense route can skip `ensure_grid` entirely. `nxm` implies the
    // general (non-plain-dense) path, so the grids are always materialized by
    // the time `build_nxm_masks` runs.

    MargPlan { sides, nxm }
}

/// One child side's NxM dead-pair liveness masks.
///
/// This is the grid-reading half of the marg plan: it fills the side's
/// liveness scratch (`live_cols`, `reach`) by scanning the materialized child
/// grid through `node_idx`. Call ONLY when [`MargPlan::nxm`] holds and after
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
    c2_level: &TddLevel,
    k2: usize,
    plan: SidePlan,
    k1_child: usize,
    k2_child: usize,
    base: usize,
    node_idx: &[u32],
    out: &mut NxmSideMasks,
) -> Result<(), ApplyError> {
    if plan.is_passthrough() {
        return Ok(());
    }
    let shift = bucket_shift(k2_child);
    build_live_cols_bitmask(eng, k1_child, k2_child, base, node_idx, &mut out.live_cols, shift)?;
    let view = plan.view;
    build_reach_masks(eng, c2_level, k2, &mut out.reach,
        |p| view.coord(if RIGHT { p.right } else { p.left }).idx(), shift)
}
