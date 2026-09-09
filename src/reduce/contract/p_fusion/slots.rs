//! Phase 2: turning each fusion plan's summed value into a marg-side ref.

use crate::engine::Engine;
use rustc_hash::FxHashMap;

use crate::error::ApplyError;
use crate::diagram::WeightVal;
use crate::diagram::{MargSide, ValueRef, Tdd};
use crate::vtree::VtreeIdx;

use crate::marg_slots::{push_count_key, CountKey, SlotInterner};

use super::PlanEntry;

/// Phase 2: allocate a count-keyed marginal slot per plan; fill `plan.new_ref`.
///
/// Returns `true` if at least one plan emitted an inline ref (the parent
/// level's marg-side inline marker must then be raised in Phase 3).
/// Increments `slots_added` for each newly-allocated slot.
#[inline(always)]
pub(super) fn allocate_fusion_slots(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut [PlanEntry],
    slots_added: &mut usize,
) -> Result<bool, ApplyError> {
    let mut any_inline = false;
    let level = &mut tdd.levels[v.idx()];
    // Seed the interner with existing slot counts so a plan whose
    // c_new equals an existing slot reuses it (slot-count uniqueness preserved on
    // every extension — no duplicate count values are introduced).
    let mut interner = SlotInterner::new();
    {
        let counts = level.marginal_counts.as_ref().unwrap();
        let big = level.marginal_counts_big.as_ref();
        interner.seed(counts, big);
    }
    for plan in plans.iter_mut() {
        // Inline small fused counts: the summed result lives in the pair
        // itself, no slot allocated. Skips count-keyed slot sharing —
        // an inline ref is cheaper than a shared slot.
        if let CountKey::Small(c) = &plan.c_new
            && let Some(raw) = ValueRef::inline_raw(*c) {
                plan.new_ref = raw;
                any_inline = true;
                continue;
            }
        // `interner` checks the map first (read-only); only on miss do we
        // need to push. The hit branch just reads `interner.map` and returns
        // the existing slot.
        if let Some(&existing) = interner.map.get(&plan.c_new) {
            plan.new_ref = ValueRef::slot_raw(existing);
            continue;
        }
        // Miss: mint a new slot (`counts` and, for a Big value, the lazily
        // allocated big side-table) via the shared store-push primitive.
        let new_idx = push_count_key(
            eng,
            level.marginal_counts.as_mut().unwrap(),
            &mut level.marginal_counts_big,
            &plan.c_new,
        )?;
        interner.map.insert(plan.c_new.clone(), new_idx);
        plan.new_ref = ValueRef::slot_raw(new_idx);
        *slots_added += 1;
    }
    Ok(any_inline)
}

/// Weighted Phase 1 helper: sum the semiring values of a marg-side occurrence
/// multiset. Mirrors [`sum_marginal_counts`] minus the u128→`BigUint` overflow
/// two-pass — a `BigRational` cannot overflow, so one clean accumulate suffices.
///
///
/// SOUNDNESS. Slots at a marginal level carry pairwise-disjoint model sets
/// (partition invariant), so the values of a group's
/// members are values of disjoint sets and add. Finite additivity over a
/// disjoint union holds for SIGNED measures, so a negative literal weight is not
/// an obstacle; the parent's contribution `Σᵢ W(x)·W(mᵢ) = W(x)·Σᵢ W(mᵢ)` then
/// follows from distributivity in ℚ. This is EXACT-domain reasoning only — the
/// caller's Log-domain decline keeps this exact-domain reasoning honest.
pub(super) fn sum_marginal_weights(ws: &crate::diagram::WeightStore, v: VtreeIdx, margs: &[u32]) -> WeightVal {
    {
        let vals = ws.level(v.idx());
        let mut acc = ws.wzero();
        for &raw in margs {
            // The ZERO sentinel (bit 31) never appears in a pair list (I-invariant;
            // `ValueRef::from_raw` debug-asserts the same). Defend anyway: a ZERO
            // child contributes the additive identity, so skipping it is the
            // value-preserving reading — and it keeps `from_raw`'s assert unreached.
            debug_assert!(raw & (1u32 << 31) == 0, "ZERO sentinel must not reach a marg-side pair ref");
            if raw & (1u32 << 31) != 0 {
                continue;
            }
            match ValueRef::from_raw(MargSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
                ValueRef::Slot(s) => {
                    let v = &vals.expect("weighted p-fusion: marg level has no WeightStore")
                        [s as usize];
                    acc.add_assign(v);
                }
            }
        }
        acc
    }
}

/// Weighted Phase 2: encode each plan's fused value as a marg-side ref.
///
/// Emission is the per-level `WeightStore` SLOT form — the same
/// `push_value`-then-bump-`weight_width` shape as `scale_weight_ref`'s
/// `Slot` arm (`dup_resolve.rs`), which is the weighted mint path that ships
/// today. On a weight-marginal level `weight_width` IS the live width read
/// by `TddLevel::width()`, and apply sizes its buffers from it, so a missed bump
/// is an out-of-bounds waiting to happen.
///
/// NOT the `ValueRef::Inline` form: an inline payload is an integer count, and a
/// weighted value has no self-describing encoding. Value-sharing is deferred
/// instead: `slot_prune`'s value-merge collapses equal-valued slots WITHIN a
/// level on the next prune, which is the sharing the boundary parent's twin
/// merge needs.
///
/// Plans in one sweep that fuse to EQUAL values share a single new slot
/// (`by_value`), so a sweep adds at most one slot per distinct fused value. Sound
/// for the same reason the integer count-keyed sharing is: pair lists are
/// multisets, and each shared-slot pair occurrence carries one plan's
/// contribution.
///
/// ZERO VALUES: signed weights make a fused sum of exactly 0 reachable (e.g.
/// `+a` and `−a`). That is a REAL value and gets a slot like any other — it must
/// NEVER become the bit-31 ZERO sentinel, which denotes the structural FALSE node
/// and would corrupt the Boolean structure. `slot_raw` keeps bit 31 clear by
/// construction; the assert pins it.
///
/// Returns `false`: the parent's marg-side inline marker is never raised, both
/// because this emits no inline ref at all and because `scale_weight_ref` — the
/// precedent — leaves the markers alone. They are an INTEGER-path discriminator
/// (`tag_all_marg_side_slots`, and the grouping scatter's guard).
#[inline(always)]
pub(super) fn allocate_fusion_slots_weighted(
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut [PlanEntry],
    slots_added: &mut usize,
) -> Result<bool, ApplyError> {
    use crate::diagram::semiring::{weight_key, WeightKey};
    // Never a vtree LEAF: its column is pinned to the 3-slot `leaf_val` cache and
    // this function's `push_value` would append a 4th. Phase 2 in
    // `apply_p_fusion_inner` routes every leaf boundary to the mint-free
    // `resolve_leaf_fusion_refs_by_lookup` instead; this pins that contract at the
    // mint site.
    debug_assert!(
        !tdd.vtree.node(v).is_leaf(),
        "weighted p-fusion must never mint into a pinned leaf column (level {})",
        v.0
    );
    let mut by_value: FxHashMap<WeightKey, u32> = FxHashMap::default();
    for plan in plans.iter_mut() {
        let val: &WeightVal = plan
            .c_new_w
            .as_deref()
            .expect("weighted p-fusion plan missing fused value");
        let key = weight_key(val);
        if let Some(&existing) = by_value.get(&key) {
            plan.new_ref = ValueRef::slot_raw(existing);
            continue;
        }
        let s = tdd.weight_store_mut().push_value(v.idx(), val.clone());
        let s = u32::try_from(s).map_err(|_| ApplyError::OverBudget)?;
        if s > crate::diagram::MARG_INLINE_MAX {
            // A slot index that would not fit the 30-bit marg-ref payload cannot
            // be referenced at all — surface it as OverBudget (routed to
            // recovery) rather than truncate a ref.
            return Err(ApplyError::OverBudget);
        }
        // Keep the weighted level's live width in sync with the store length
        // (the same bump `scale_weight_ref` performs after `push_value`).
        tdd.levels[v.idx()].weight_width = s + 1;
        by_value.insert(key, s);
        plan.new_ref = ValueRef::slot_raw(s);
        *slots_added += 1;
        debug_assert!(
            plan.new_ref & (1u32 << 31) == 0,
            "fused weighted marg ref must never alias the ZERO sentinel",
        );
    }
    Ok(false)
}
