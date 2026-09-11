//! Slot-prune of orphaned marginal-count slots.
//!
//! Slots become garbage at a boundary marginal level (a marginal child of a
//! structural parent), and no other pass collects them: the end-of-apply
//! tagger converts small-count slot refs to inline refs, and canon / pair
//! fusion / twin-merge redirect refs onto canonical or fused slots. The
//! abandoned slots stay in the store: canon never shrinks (its no-shrink doc),
//! and `prune_unreachable` deliberately keeps marginal stores at full length
//! (identity remap — see prune.rs's "store-relative" comment). A marginal level
//! under a marginal parent has no store to collect: the marginalize step frees
//! it as the parent becomes marginal (`free_subsumed_marginal_children`).
//!
//! `prune_value_slots` compacts each boundary store to exactly the slots
//! referenced from its parent's marginal-side refs (remapping those refs).
//! The output level's store is exempt — it holds the
//! result (the final count, or a component sub-diagram's count). Compaction is
//! sound here where canon's would not be, because every surviving parent ref
//! is rewritten through the composed remap in the same pass.
//!
//! The boundary compaction also merges equal-valued surviving slots, so this is
//! where slot-count uniqueness is established for stores born at an apply emit
//! site — the conjoin apply engine (`apply::conjoin`) skips
//! value-dedup when emitting.
//!
//! **Precondition:** parent levels must be in post-tagger form (marginal-side refs
//! decodable with `ValueRef::from_raw`) — never mid-apply.
//!
//! Each freed slot is tallied into `TddLevel::retired_marginal_slots` (summed by
//! `Tdd::retired_marginal_slots`), while `Tdd::node_count()` is the honest
//! surviving-circuit count and so decreases across a prune. A caller gating on
//! `node_count()` can add the slots retired since its own baseline back in and
//! keep a trigger cadence that collection does not shift.
//!
//! Wiring mirrors node-prune: at the end of `try_minimize` (after contract,
//! whose inline→slot redirects mint refs post-node-prune), and after each
//! `run_marginalize_at*` fusion sweep. Inlining only happens at marginalize
//! time (counts only grow afterwards, and post-tagger slot counts already
//! exceed the inline threshold), so slots die when node-prune kills the pairs
//! referencing them. The contract-only path (`MinimizeScope::ContractOnly`,
//! rotation-hot) is skipped: it kills no pairs.
//!
//! # One skeleton, two value kinds
//!
//! Integer and weighted marginal levels share one
//! prune skeleton, `prune_marginal_slots_generic`, monomorphized at the single
//! runtime branch in [`prune_value_slots`]. The traversal and the whole
//! `ValueSlotPruneStats` tally are written once; only where the per-slot values
//! live differs, and that is the `SlotStore` trait.

use crate::engine::Engine;


use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

use crate::value::{IntFold, WeightFold, SlotStore};
use crate::value::slots::{RefSlotScratch, referenced_marginal_slots};
use crate::diagram::{boundary_marginal_levels, remap_refs_into};
use crate::value::slots::{compact_slots, count_key_at, rekey_big, truncate_with_slack};

// ── Sweep scratch ───────────────────────────────────────────────────────────
//
// `prune_marginal_slots_generic` built its two sweep-lifetime buffers fresh on
// every call, and the sweep itself is per-merge: a caller that compiles very
// many tiny diagrams runs this a few dozen times each, so the ref-collector's
// `Vec`+`FxHashSet` and the slot remap were pure allocator churn. Pooled
// exactly like prune's own buffers: one engine-owned `Cell` each, cleared on
// take, capacity-capped on return.
/// Take the engine's sweep buffers, cleared and ready to use. A fresh (empty)
/// pair when the pool is cold or a nested sweep already holds them.
fn take_sweep_scratch(eng: &Engine) -> (RefSlotScratch, Vec<u32>) {
    let pool = eng.reduce();
    let mut slots = pool.slot_prune_slots.take().unwrap_or_default();
    slots.clear();
    let mut remap = pool.slot_prune_remap.take();
    remap.clear();
    (slots, remap)
}

/// Park the sweep buffers for the next `prune_value_slots`, each
/// released independently if its retained capacity exceeds the byte cap.
/// Skipping this (an early bail) costs only the buffers' capacity.
fn return_sweep_scratch(eng: &Engine, mut slots: RefSlotScratch, remap: Vec<u32>) {
    let pool = eng.reduce();
    slots.release_oversized();
    pool.slot_prune_slots.put(Some(slots));
    pool.slot_prune_remap.put_bounded(remap);
}

/// What a `prune_value_slots` sweep reclaimed.
#[derive(Debug, Default, Clone)]
pub(crate) struct ValueSlotPruneStats {
    /// Referenced slots eliminated specifically by value-dedup:
    /// a referenced slot that mapped onto an earlier equal-valued slot.
    /// Distinct from unreferenced-orphan drops.
    ///
    /// When `values_merged > 0`, the boundary compaction pass merged two or more
    /// referenced slots with equal values — remapping all parent refs to the
    /// surviving slot. This can make previously-distinct parent nodes raw-identical
    /// (new content twins). Value merges happen often (pair fusion routinely
    /// mints sum slots with colliding values), while actual twin minting is
    /// rare, so `try_minimize` checks only the affected boundary parents for
    /// content twins and pays a full prune+contract round when one is found.
    /// Gating the round on `values_merged` alone pays that round on every
    /// fusion-heavy sweep.
    pub values_merged: usize,
    /// Marginal vtree levels where `values_merged` fired this sweep — the
    /// content-twin scan in `try_minimize` is restricted to their boundary
    /// parents.
    pub value_merged_levels: Vec<u32>,
}


/// Collect orphaned marginal-count slots diagram-wide. See module doc for the
/// garbage source and the post-tagger precondition.
///
/// The one runtime value-kind branch: everything downstream is statically
/// monomorphized over `SlotStore`.
pub(crate) fn prune_value_slots(eng: &Engine, tdd: &mut Tdd) -> ValueSlotPruneStats {
    if tdd.weights.is_some() {
        prune_marginal_slots_generic::<WeightFold>(eng, tdd)
    } else {
        prune_marginal_slots_generic::<IntFold>(eng, tdd)
    }
}


/// Integer: values are u128 counts in `TddLevel::marginal_counts`,
/// with exact `BigUint` overflow entries in the `marginal_counts_big` side
/// table.
impl SlotStore for IntFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.levels[v.idx()].marginal_counts().map_or(0, |c| c.len())
    }

    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        // The fast `counts` column is compacted in place — no second
        // full-length store beside the old one. `referenced` is strictly
        // ascending and duplicate-free (its only producer,
        // `referenced_marginal_slots`, sorts it — the out-of-range guard at the
        // call site reads `referenced.last()` as the max on the same
        // assumption), which is what `compact_slots` needs.
        //
        // The sparse overflow table is rekeyed rather than compacted in place:
        // its keys are the old slot indices, and a survivor's key changes. It is
        // moved out here, left untouched for the duration of the loop (which
        // only reads it, through `count_key_at`), and rebuilt in one drain once
        // `remap` is complete.
        let (counts, big) = tdd.levels[v.idx()].marginal_store_mut().unwrap();
        let old_big = big.take();
        let (new_len, values_merged) = compact_slots(
            counts,
            referenced.iter().map(|&old| old as usize),
            |counts, old| count_key_at(counts, old_big.as_ref(), old),
            |counts, dst, src| counts[dst] = counts[src],
            remap,
        );
        *big = rekey_big(old_big, remap);
        truncate_with_slack(counts, new_len);
        (new_len, values_merged)
    }

    /// Integer semantics: `retired_marginal_slots` is a monotone retirement
    /// tally, not a width — `Tdd::retired_marginal_slots()` sums it so the
    /// adaptive-minimize gates can add back the slots this pass removed. This
    /// increments it by `freed`; the live width lives in `marginal_counts.len()`
    /// and was already committed by the caller's compaction.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize) {
        let level = &mut tdd.levels[v.idx()];
        let before = level.retired_marginal_slots();
        level.retire_marginal_slots(freed as u32);
        debug_assert!(
            level.retired_marginal_slots() >= before,
            "the retirement tally only grows — it is never a live width"
        );
        debug_assert_eq!(
            level.width(),
            new_len,
            "integer live width is marginal_counts.len(), committed before update_width"
        );
    }
}

/// Weighted: values are `BigRational`/log semiring values in the
/// external `WeightStore`, indexed by level. `TddLevel` is at its size cap and
/// carries no `marginal_counts` of its own, hence the `weight_width` field
/// below.
///
/// Stores must be compacted even when marginal-side refs are being inlined:
/// `weight_width` is what `width()` returns for a weight-marginal level,
/// so leaving it at the un-compacted width sizes the streaming/apply buffers
/// far too large. Both ref-walkers (`referenced_marginal_slots`,
/// `remap_slot_ref`) skip bit-31 sentinels and only touch `ValueRef::Slot`, so
/// `Inline` refs pass through verbatim.
impl SlotStore for WeightFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.weights
            .as_ref()
            .and_then(|ws| ws.level(v.idx()))
            .map_or(0, |s| s.len())
    }

    /// Value-dedup keys on the semiring value directly — one uniform key type,
    /// no Small/Big `Count` split. A marginalized node is fully represented
    /// by its value, so two referenced slots with equal value are
    /// interchangeable upward and merge to one (first occurrence wins).
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        use crate::diagram::semiring::weight_key;
        // Compacted in place, like the integer impl — no second full-length
        // store beside the old one at peak. Worth more here than on the integer
        // side: a `WeightVal` is never smaller than a `u128` and is usually a
        // multi-limb `BigRational`, so a second store would duplicate every
        // surviving rational's heap payload as well. `WeightVal` is not `Copy`,
        // so the move down is a swap; the displaced value is dead from that
        // moment on and dropped by the closing truncate.
        let ws = tdd.weight_store_mut();
        if !ws.is_set(v.idx()) {
            // Boundary level flagged marginal with no store allocated: leave
            // an empty-but-present store. `Some(empty)` is the
            // "marginal, zero slots" state `ensure_weights` reads as
            // "already weight-marginal". Only reachable with an empty
            // `referenced` — the caller's out-of-range guard rejects any ref into a
            // zero-length store.
            ws.set_level(v.idx(), Vec::new());
        }
        let values = ws.level_vals_mut(v.idx()).expect("store present: ensured just above");
        let (new_len, values_merged) = compact_slots(
            values,
            referenced.iter().map(|&old| old as usize),
            |values, old| weight_key(&values[old]),
            |values, dst, src| values.swap(dst, src),
            remap,
        );
        // Drops the orphans, the merged-away duplicates, and the values
        // swapped up out of the prefix — the point of the pass.
        truncate_with_slack(values, new_len);
        (new_len, values_merged)
    }

    /// Weighted semantics: `weight_width` is itself the live slot count —
    /// `TddLevel::width()` returns it for a weight-marginal level (set by
    /// `become_marginal_weighted`), and the apply/streaming buffers are
    /// sized from that. This assigns `new_len`; `freed` is stats-only here and must
    /// not be added, or the width drifts up and re-opens the oversized-buffer
    /// blowup described on the impl above.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, _freed: usize, new_len: usize) {
        tdd.levels[v.idx()].set_weight_width(new_len as u32);
        debug_assert_eq!(
            tdd.weights.as_ref().and_then(|ws| ws.level(v.idx())).map_or(0, |s| s.len()),
            new_len,
            "weight_width must equal the live WeightStore length"
        );
    }
}

/// The one prune skeleton, generic over where the values live. See the module
/// comment's table for what stays per-kind.
fn prune_marginal_slots_generic<S: SlotStore>(eng: &Engine, tdd: &mut Tdd) -> ValueSlotPruneStats {
    // Central pin-invariant check (debug builds, weighted mode only): a
    // weight-marginal leaf's column is the immutable label-ordered `leaf_val`
    // triple. This pass runs tens of times per compile, so a regression in any of
    // the passes that could break it lands here immediately.
    #[cfg(debug_assertions)]
    if let Err(e) = crate::check::marginal::check_leaf_columns_pinned(tdd) {
        panic!("leaf column pin: {e}");
    }
    let mut stats = ValueSlotPruneStats::default();
    // Sweep-lifetime scratch: the ref-collector's result/dedup buffers and the
    // old→new slot map. Reused across boundary levels — and, via the pool,
    // across sweeps (the sweep is per-merge, so a fresh allocation per level or
    // per call shows up as pure allocator churn on many-small-diagram
    // workloads). Both are refilled per level below (`referenced_marginal_slots`
    // clears its own buffers; `remap` is cleared and resized), so a pooled pair
    // differs from a fresh one only in capacity.
    let (mut slots, mut remap) = take_sweep_scratch(eng);
    // The output level's store is the result (a marginal output level has no
    // parent refs at all — e.g. a fully-marginalized component sub-diagram whose
    // output sits at the subtree root under an empty parent level). Never
    // touch it.
    let out_v = tdd.output.vtree;


    compact_boundary_stores::<S>(tdd, out_v, &mut stats, &mut slots, &mut remap);

    return_sweep_scratch(eng, slots, remap);
    stats
}

#[path = "slot_prune_stores.rs"]
mod stores;
use stores::compact_boundary_stores;

#[cfg(test)]
mod tests;
