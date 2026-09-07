//! Per-level marg classification plan (`MargPlan` + `plan_marg_level`) and the
//! NxM dead-pair liveness masks (`build_nxm_masks`) for the apply product
//! construction. Pure code motion out of `conjoin/mod.rs`; the driver
//! destructures `MargPlan` at the call site. `MARG_ENTRY_*` stay in `mod.rs`
//! and are reached via `super::`; the liveness bitmask kernels live in
//! `super::liveness`.

use crate::vtree::VtreeIdx;
use crate::tdd::types::*;
use super::{ApplyError, MARG_ENTRY_C1, MARG_ENTRY_C2};
use super::liveness::{bucket_shift, build_live_cols_bitmask, build_reach_masks};

/// Per-level marg classification plan produced by [`plan_marg_level`] (extraction 3).
///
/// Fields correspond exactly to the same-named locals in the main loop body.
/// Destructure at the call site to preserve identical downstream names.
pub(super) struct MargPlan {
    pub left_pt_c1:       bool,
    pub right_pt_c1:      bool,
    pub left_passthrough: bool,
    pub right_passthrough: bool,
    pub left_mask:        u32,
    pub right_mask:       u32,
    pub nxm:              bool,
}

/// Per-level marg classification + mask/pass-through-carrier setup (extraction 3).
///
/// Computes `left_marg`/`right_marg` (three-way ORs over output/c1/c2 child
/// levels), `left_pt_c1`/`left_pt_c2`/`right_pt_c1`/`right_pt_c2`,
/// `left_passthrough`/`right_passthrough`, `left_mask`/`right_mask`, `nxm`, and
/// fills the NxM dead-pair liveness scratch buffers (`live_left_cols`,
/// `reach_c2_left`, `live_right_cols`, `reach_c2_right`) for levels where `nxm`
/// fires. The MARG_ENTRY_C1/MARG_ENTRY_C2 thread-local reads move with this code.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_marg_level(
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
    // ref reached grid_read! raw as `(1<<30)+base` ≫ node_idx.len() → OOB
    // segfault on previously-solved CNFs (regression #43). A single shared
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
    // ── Pass-through: a marginal child meets an identity operand ──
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
    // marginal sides meet is (P)-fusion, which has its own inner and never
    // reaches this apply.
    //
    //   left_pt_c1 — c1 is the marginal carrier (c2 identity at left)  → carry p1.left
    //   left_pt_c2 — c2 is the marginal carrier (c1 identity at left)  → carry p2.left
    // (right mirrors). Sides are independent: a level can be left-passthrough
    // and right-real, or both.
    // The carrier child may be marginal in the operand itself OR already
    // swapped into the OUTPUT accumulator as a frozen frontier (the
    // identity-swap fast-path moved one operand's marginal child into
    // `levels[child]`, leaving the operand's own child level empty). In
    // both cases the carrier operand's per-pair field is still its inline
    // model count / big-count slot — never a grid coordinate — so carry it
    // verbatim.
    // Pass-through carrier selection. Pass-through carries the carrier
    // operand's raw per-pair marg field straight into the output, where the
    // streaming sum reads it as a slot/inline-count against the OUTPUT child
    // store. Two things must BOTH hold for that to be sound:
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
    // therefore value-identical and skips four `RefCell` borrows per level on the
    // dominant pure-Boolean / MC fold path, where no operand level is marginal.
    let ent_c1 = |idx: usize| any_entry_marginal
        && MARG_ENTRY_C1.with(|v: &std::cell::RefCell<Vec<bool>>| v.borrow().get(idx).copied().unwrap_or(false));
    let ent_c2 = |idx: usize| any_entry_marginal
        && MARG_ENTRY_C2.with(|v: &std::cell::RefCell<Vec<bool>>| v.borrow().get(idx).copied().unwrap_or(false));
    let (left_pt_c1, left_pt_c2) = {
        // #2 no-grid identity conjunction: a marginal child is always
        // conjoined against identity on the other operand (the invariant
        // asserted below), so carry the marginal operand's ref through
        // verbatim — never grid it (an inline ref is a count, not a
        // coordinate). Carrier = the operand that holds (or held, at batch
        // entry) marginal content; requiring `c{1,2}_ref` keeps a genuinely
        // structural carrier (case B: operands structural, output
        // marginalized mid-loop by the cascade) on the grid path, where its
        // refs are structural indices and grid-safe.
        //
        // Do NOT `&&` an `out_l = levels[left_idx].is_marginal()` conjunct
        // here: that snapshot predates the mid-loop cascade (it is recomputed
        // post-cascade further down), so it reads stale-false in exactly the
        // cells where the child marginalizes mid-loop, forcing them onto the
        // grid path — an inline overcount. It is redundant anyway: carrying
        // marginal content through an identity side always yields a marginal
        // output.
        let c1_ref = c1.levels[left_idx].is_marginal() || ent_c1(left_idx)
            || c1.levels[t_idx].marg_inlined_left();
        let c2_ref = c2.levels[left_idx].is_marginal() || ent_c2(left_idx)
            || c2.levels[t_idx].marg_inlined_left();
        (c2_identity[left_idx] && c1_ref,
         c1_identity[left_idx] && c2_ref)
    };
    let (right_pt_c1, right_pt_c2) = {
        // See left-side note: `out_r` dropped (stale pre-cascade snapshot).
        let c1_ref = c1.levels[right_idx].is_marginal() || ent_c1(right_idx)
            || c1.levels[t_idx].marg_inlined_right();
        let c2_ref = c2.levels[right_idx].is_marginal() || ent_c2(right_idx)
            || c2.levels[t_idx].marg_inlined_right();
        (c2_identity[right_idx] && c1_ref,
         c1_identity[right_idx] && c2_ref)
    };
    let left_passthrough = left_pt_c1 || left_pt_c2;
    let right_passthrough = right_pt_c1 || right_pt_c2;
    // INVARIANT — no marginal×marginal product. Conjoining two marginal nodes
    // is undefined: |f ∧ g| is not a function of |f| and |g|, so there is no
    // correct way to combine them in the product grid. A marginal child must
    // always be conjoined against IDENTITY on the other operand (the
    // marginalized scope is never re-constrained). The only way both operands
    // can carry a non-identity marginal level at the same child is if the
    // logic deciding WHEN to marginalize (marginalize-target scheduling /
    // `cascade_marginalize_in_apply`'s "no future references" condition) is
    // broken — i.e. a scope was summed out while a later conjunction still
    // constrained it. Panic here rather than silently compute a wrong count.
    // Tier-2 (debug-only) guard on the forbidden marginal×marginal operand
    // product. Confirmed NOT to fire on mc007 even as a hard release assert —
    // case (B) genuinely does not occur — so it stays a debug_assert!. To run
    // it in an optimized binary, build `--profile release-checked`.
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
    // (P)-fusion territory and must never reach the clause/child-merge apply.
    // Fires loud in debug if the disjoint-subtree assumption is ever violated.
    debug_assert!(
        !(c1.levels[left_idx].is_marginal() && c2.levels[left_idx].is_marginal()) || left_passthrough,
        "two marginal left operands, neither identity — unexpected outside (P)-fusion (t={t:?})");
    debug_assert!(
        !(c1.levels[right_idx].is_marginal() && c2.levels[right_idx].is_marginal()) || right_passthrough,
        "two marginal right operands, neither identity — unexpected outside (P)-fusion (t={t:?})");

    // On a pass-through side, decode raw (u32::MAX): we must preserve the
    // carrier field's tag bit (inline count vs big-count slot). Masking it to
    // a bare slot would corrupt a big slot into a misread inline count.
    let left_mask = if left_passthrough {
        u32::MAX
    } else if left_marg {
        crate::tdd::types::MARG_VALUE_MASK
    } else {
        u32::MAX
    };
    let right_mask = if right_passthrough {
        u32::MAX
    } else if right_marg {
        crate::tdd::types::MARG_VALUE_MASK
    } else {
        u32::MAX
    };
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

    MargPlan {
        left_pt_c1, right_pt_c1,
        left_passthrough, right_passthrough,
        left_mask, right_mask,
        nxm,
    }
}

/// NxM dead-pair liveness masks (extraction 3b, split from [`plan_marg_level`]).
///
/// This is the grid-reading half of the marg plan: it fills the per-side
/// liveness scratch buffers (`live_*_cols`, `reach_c2_*`) by scanning the
/// materialized child grids through `node_idx`. Call ONLY when
/// [`MargPlan::nxm`] holds and after both child grids exist. Masks are
/// bit-exact for child widths ≤ 128 and bucketed (shift > 0,
/// sound-with-false-positives) above — see liveness.rs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_nxm_masks(
    c2: &Tdd,
    t: VtreeIdx,
    k2: usize,
    k1_left: usize,
    k2_left_usize: usize,
    k2_right_usize: usize,
    left_base: usize,
    right_base: usize,
    right_idx: usize,
    left_passthrough: bool,
    right_passthrough: bool,
    left_mask: u32,
    right_mask: u32,
    node_idx: &[u32],
    c1_widths: &[usize],
    live_left_cols: &mut Vec<u128>,
    reach_c2_left: &mut Vec<u128>,
    live_right_cols: &mut Vec<u128>,
    reach_c2_right: &mut Vec<u128>,
) -> Result<(), ApplyError> {
    let k2l = k2_left_usize;
    let k2r = k2_right_usize;
    let shift_left = bucket_shift(k2l);
    let shift_right = bucket_shift(k2r);
    let c2_level = c2.level(t);

    // Skipped entirely on a pass-through side: the marginal child has
    // no product grid, and its field is a model count, not a
    // row/column index. build_* would index out of bounds.
    if !left_passthrough {
        build_live_cols_bitmask(k1_left, k2l, left_base, node_idx, live_left_cols, shift_left)?;
        build_reach_masks(c2_level, k2, reach_c2_left,
            |p| crate::tdd::types::decode_marg_coord(p.left.0, left_mask) as usize, shift_left)?;
    }
    if !right_passthrough {
        let k1_right = c1_widths[right_idx];
        build_live_cols_bitmask(k1_right, k2r, right_base, node_idx, live_right_cols, shift_right)?;
        build_reach_masks(c2_level, k2, reach_c2_right,
            |p| crate::tdd::types::decode_marg_coord(p.right.0, right_mask) as usize, shift_right)?;
    }
    Ok(())
}
