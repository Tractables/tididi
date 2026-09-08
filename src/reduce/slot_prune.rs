//! Slot-prune of orphaned marginal-count slots.
//!
//! Slots become garbage two ways, and no other pass collects them:
//!
//! 1. **Boundary orphans** — the end-of-apply tagger converts small-count slot
//!    refs to inline refs, and canon / p-fusion / twin-merge redirect refs onto
//!    canonical or fused slots. The abandoned slots stay in the store: canon
//!    never shrinks (its no-shrink doc), and `prune_unreachable` deliberately
//!    keeps marginal stores at full length (identity remap — see prune.rs's
//!    "STORE-relative" comment).
//! 2. **Dead deep stores** — when marginalization cascades, a marginal parent
//!    consumes its marginal child's counts; from then on the child store is
//!    unreachable (the parent has no pairs, `model_count_hybrid`'s bottom-up
//!    walk shadows it, and no reexpand snapshot reads it).
//!
//! `prune_marg_slots` compacts each boundary store to exactly the slots
//! referenced from its parent's marg-side refs (remapping those refs), and
//! clears dead deep stores. The output level's store is exempt — it holds the
//! result (the final count, or a component sub-TDD's count). Compaction is
//! sound here where canon's would not be, because every surviving parent ref
//! is rewritten through the composed remap in the same pass.
//!
//! The boundary compaction also merges equal-valued surviving slots, so this is
//! where slot-count uniqueness is established for stores born at an apply emit
//! site — the conjoin apply engine (`transform::pairwise::conjoin`) skips
//! value-dedup when emitting.
//!
//! **Precondition:** parent levels must be in POST-TAGGER form (marg-side refs
//! decodable with `MargRef::from_raw`) — never mid-apply.
//!
//! Each freed slot is tallied into `TddLevel::retired_marg_width` (summed by
//! `Tdd::retired_marg_total()`), while `Tdd::total_nodes()` is the honest
//! surviving-circuit count and so decreases across a prune. A caller gating on
//! `total_nodes()` can add the slots retired since its own baseline back in and
//! keep a trigger cadence that collection does not shift.
//!
//! Wiring mirrors node-prune: at the end of `try_minimize` (after contract,
//! whose inline→slot redirects mint refs post-node-prune), and after each
//! `run_marginalize_at*` fusion sweep. Inlining only happens at marginalize
//! time (counts only grow afterwards, and post-tagger slot counts already
//! exceed the inline threshold), so slots die when node-prune kills the pairs
//! referencing them. The contract-only path (`MinimizePasses::ContractOnly`,
//! rotation-hot) is skipped: it kills no pairs.
//!
//! # One skeleton, two value kinds
//!
//! Integer and weighted marginal levels share ONE
//! prune skeleton, `prune_marg_slots_generic`, monomorphized at the single
//! runtime branch in [`prune_marg_slots`]. The traversal and the whole
//! `MargSlotPruneStats` tally are written once; only where the per-slot VALUES
//! live differs, and that is the `SlotStore` trait.

use std::cell::Cell;

use rustc_hash::FxHashMap;

use crate::diagram::{BigSide, MargRef, Tdd, MAX_LEVEL_ARENA_BYTES};
use crate::vtree::VtreeIdx;

use crate::counts::{IntFold, WeightFold};
use crate::marg_slots::{referenced_marg_slots, RefSlotScratch};
use crate::marg_slots::{boundary_marginal_levels, count_key_at, for_each_side_ref_mut, SlotInterner};
use crate::utils::{pool_put, pool_put_bounded, pool_take};

// ── Thread-local sweep scratch ──────────────────────────────────────────────
//
// `prune_marg_slots_generic` built its two sweep-lifetime buffers fresh on
// every call, and the sweep itself is per-merge: a caller that compiles very
// many tiny diagrams runs this a few dozen times each, so the ref-collector's
// `Vec`+`FxHashSet` and the slot remap were pure allocator churn. Pooled exactly like `minimize::prune`'s
// `SCRATCH_REMAP`/`SCRATCH_OFF`: one thread-local `Cell` each, cleared on take,
// capacity-capped on return.
thread_local! {
    static SCRATCH_SLOTS: Cell<Option<RefSlotScratch>> = const { Cell::new(None) };
    static SCRATCH_REMAP: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
}

/// Take the thread's sweep buffers, cleared and ready to use. A fresh (empty)
/// pair when the pool is cold or a nested sweep already holds them.
fn take_sweep_scratch() -> (RefSlotScratch, Vec<u32>) {
    let mut slots = pool_take(&SCRATCH_SLOTS).unwrap_or_default();
    slots.clear();
    let mut remap = pool_take(&SCRATCH_REMAP);
    remap.clear();
    (slots, remap)
}

/// Park the sweep buffers for the next `prune_marg_slots` on this thread, each
/// released independently if its retained capacity exceeds the byte cap.
/// Skipping this (an early bail) costs only the buffers' capacity.
fn return_sweep_scratch(mut slots: RefSlotScratch, remap: Vec<u32>) {
    slots.release_oversized(MAX_LEVEL_ARENA_BYTES);
    pool_put(&SCRATCH_SLOTS, Some(slots));
    pool_put_bounded(&SCRATCH_REMAP, remap, MAX_LEVEL_ARENA_BYTES);
}

/// What a `prune_marg_slots` sweep reclaimed.
#[derive(Debug, Default, Clone)]
pub struct MargSlotPruneStats {
    /// Slots dropped across all stores (boundary compaction + deep clears).
    pub slots_freed: usize,
    /// Dead deep stores cleared outright.
    pub stores_cleared: usize,
    /// Referenced slots eliminated specifically by value-dedup:
    /// a referenced slot that mapped onto an earlier equal-valued slot.
    /// Distinct from unreferenced-orphan drops (those are in `slots_freed` only).
    ///
    /// When `values_merged > 0`, the boundary compaction pass merged two or more
    /// referenced slots with equal values — remapping all parent refs to the
    /// surviving slot. This can make previously-distinct parent nodes raw-identical
    /// (new content twins). NOTE: value merges are COMMON (p-fusion routinely
    /// mints sum slots with colliding values), while actual twin minting is
    /// rare — `try_minimize` therefore only checks the affected boundary
    /// parents for content twins and pays a full prune+contract round when one
    /// is found. Gating the round on `values_merged` alone is a large measured
    /// regression on fusion-heavy CNFs.
    pub values_merged: usize,
    /// Marginal vtree levels where `values_merged` fired this sweep — the
    /// content-twin scan in `try_minimize` is restricted to their boundary
    /// parents.
    pub value_merged_levels: Vec<u32>,
}


/// Collect orphaned marginal-count slots TDD-wide. See module doc for the
/// garbage classes and the post-tagger precondition.
///
/// The ONE runtime value-kind branch: everything downstream is statically
/// monomorphized over `SlotStore`.
pub fn prune_marg_slots(tdd: &mut Tdd) -> MargSlotPruneStats {
    if tdd.weights.is_some() {
        prune_marg_slots_generic::<WeightFold>(tdd)
    } else {
        prune_marg_slots_generic::<IntFold>(tdd)
    }
}

/// Where a marginal level's per-slot VALUES live, for the one prune skeleton
/// (`prune_marg_slots_generic`). Implemented on the crate's value-kind
/// markers (`IntFold` / `WeightFold`, `counts.rs`) so slot STORAGE sits on the
/// same axis as the marginalization fold. Four hooks, each a place where the
/// two kinds genuinely differ.
trait SlotStore {
    /// Slot count of level `v`'s store: the domain of the remap that
    /// `compact_store` fills, and the pre-compaction width.
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize;

    /// Free level `v`'s DEAD DEEP store (its marginal parent already consumed
    /// these values), returning the slot count freed. Returns 0 — touching
    /// nothing — when the store is already empty. The level stays in marginal
    /// mode; only the payload goes.
    fn clear_dead_store(tdd: &mut Tdd, v: VtreeIdx) -> usize;

    /// Compact level `v`'s store to `referenced` with value-dedup, write
    /// the composed `old_slot → new_slot` map into `remap`, and commit the
    /// compacted store. Returns `(new_len, values_merged)`; `values_merged`
    /// counts referenced slots that landed on an earlier equal-valued slot.
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize);

    /// Fold a completed compaction of level `v` into `retired_marg_width`.
    /// **The two impls are INVERTED and must stay that way** (increment vs
    /// assign) — see each impl's comment.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize);
}

/// Integer: values are u128 counts in `TddLevel::marginal_counts`,
/// with exact `BigUint` overflow entries in the `marginal_counts_big` side
/// table.
impl SlotStore for IntFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.levels[v.idx()].marginal_counts.as_ref().map_or(0, |c| c.len())
    }

    fn clear_dead_store(tdd: &mut Tdd, v: VtreeIdx) -> usize {
        let level = &mut tdd.levels[v.idx()];
        let counts = level.marginal_counts.as_mut().unwrap();
        if counts.is_empty() {
            return 0;
        }
        let freed = counts.len();
        // Keep `Some(empty)` so the level stays in marginal mode.
        counts.clear();
        counts.shrink_to_fit();
        if let Some(big) = &mut level.marginal_counts_big {
            big.clear_and_free();
        }
        freed
    }

    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        // The fast `counts` column is compacted IN PLACE — no second
        // full-length store beside the old one.
        //
        // SOUNDNESS (why the moves can't clobber a slot still to be read):
        // `referenced` is strictly ASCENDING and duplicate-free (its only
        // producer, `referenced_marg_slots`, sorts it — the OOB guard at the
        // call site reads `referenced.last()` as the max on the same
        // assumption). So at step i the source is `old >= i` and the
        // destination is `new_len <= i <= old`: every write lands at or below a
        // slot already consumed, and every later read is at a strictly larger
        // index than any write done so far.
        let level = &mut tdd.levels[v.idx()];
        // The interner's map is what makes the value-dedup key discipline —
        // `CountKey`'s Small/Big split — the same one every other slot path uses.
        let mut interner = SlotInterner::new();
        let mut values_merged = 0usize;
        let mut new_len = 0usize;
        // The sparse overflow table is REKEYED rather than compacted in place:
        // its keys are the old slot indices, and a survivor's key changes. It is
        // moved out here, left untouched for the duration of the loop (which
        // only reads it, through `count_key_at`), and rebuilt in ONE drain once
        // `remap` is complete — see below.
        let old_big = level.marginal_counts_big.take();
        let counts = level.marginal_counts.as_deref_mut().unwrap();
        for &old in referenced {
            let old = old as usize;
            let key = count_key_at(&*counts, old_big.as_ref(), old);
            // A hit means `old` mapped onto an equal-valued surviving slot
            // (value-dedup merge); a miss mints the next compacted slot.
            if let Some(&hit) = interner.map.get(&key) {
                remap[old] = hit;
                values_merged += 1;
                continue;
            }
            interner.map.insert(key, new_len as u32);
            let moved_count = counts[old];
            counts[new_len] = moved_count;
            remap[old] = new_len as u32;
            new_len += 1;
        }
        // Rekey: consume the old table in one ascending drain and re-file each
        // value under its slot's compacted index. Values MOVE — a `BigUint` here
        // can be megabytes and is never cloned. An unreferenced slot keeps the
        // `u32::MAX` sentinel and its value is dropped as the drain passes it; a
        // merged slot maps onto its canonical's index and writes an EQUAL value
        // over it (equality is what made them merge), so either order yields the
        // same table. Draining once is what keeps this linear: taking survivors
        // one at a time out of the front would memmove the tail per entry, which
        // is quadratic on a level where most slots overflowed.
        let remap_ro: &[u32] = remap;
        let new_big = old_big.map(|b| {
            b.into_iter()
                .filter_map(|(slot, v)| {
                    let new = remap_ro[slot as usize];
                    if new == u32::MAX { None } else { Some((new, v)) }
                })
                .collect::<BigSide>()
        });
        // Slack ceiling: `counts` is compacted IN PLACE, so the capacity observed
        // here is the PRE-compaction one. Shrinking at 2× therefore reclaims
        // exactly when the store more than halved — the effective ceiling on
        // retained slack. The rebuilt overflow table needs no such policy: its
        // slack is bounded by the surviving overflow set, not by the width.
        let counts = level.marginal_counts.as_mut().unwrap();
        counts.truncate(new_len);
        if counts.capacity() > 64 && counts.capacity() > 2 * counts.len() {
            counts.shrink_to_fit();
        }
        level.marginal_counts_big = new_big;
        (new_len, values_merged)
    }

    /// INTEGER SEMANTIC: `retired_marg_width` is a monotone RETIREMENT TALLY,
    /// not a width — `Tdd::retired_marg_total()` sums it so the
    /// adaptive-minimize gates can add back the slots this pass removed.
    /// INCREMENT it by `freed`; the live width lives in `marginal_counts.len()`
    /// and was already committed by the caller's compaction.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize) {
        let level = &mut tdd.levels[v.idx()];
        let before = level.retired_marg_width;
        level.retired_marg_width = before.saturating_add(freed as u32);
        debug_assert!(
            level.retired_marg_width >= before,
            "integer retired_marg_width only grows — it is a retirement tally, never the live width"
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
/// carries no width field of its own, hence the `retired_marg_width` inversion
/// below.
///
/// Stores must be compacted even when marg-side refs are being inlined:
/// `retired_marg_width` is what `width()` returns for a weight-marginal level,
/// so leaving it at the un-compacted width sizes the streaming/apply buffers
/// (stream.rs) far too large. Both ref-walkers (`referenced_marg_slots`,
/// `remap_slot_ref`) skip bit-31 sentinels and only touch `MargRef::Slot`, so
/// `Inline` refs pass through verbatim.
/// The attached store, for the weighted impl below: reaching it means the
/// diagram is in weighted mode.
fn weights_mut(tdd: &mut Tdd) -> &mut crate::weight_store::WeightStore {
    tdd.weights.as_mut().expect("weighted slot prune without a weight store")
}

impl SlotStore for WeightFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.weights
            .as_ref()
            .and_then(|ws| ws.level(v.idx()))
            .map_or(0, |s| s.len())
    }

    /// The freed count comes from `retired_marg_width` (the live width on a
    /// weight-marginal level), not from the `WeightStore` vec: zeroing a stale
    /// width is the point — `width()` reads it and sizes apply buffers from it.
    fn clear_dead_store(tdd: &mut Tdd, v: VtreeIdx) -> usize {
        let freed = tdd.levels[v.idx()].retired_marg_width as usize;
        if freed == 0 {
            return 0;
        }
        // Keep the MARG_WEIGHTED flag so the level stays in marginal mode.
        weights_mut(tdd).set_level(v.idx(), Vec::new());
        freed
    }

    /// Value-dedup keys on the semiring value directly — one uniform key type,
    /// no Small/Big `CountKey` split. A marginalized node is fully represented
    /// by its value, so two referenced slots with equal value are
    /// interchangeable upward and merge to one (first occurrence wins).
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        use crate::query::semiring::{weight_key, WeightMap};
        // Compacted IN PLACE, like `IntFold::compact_store` — no second
        // full-length store beside the old one at peak. Worth more here than on
        // the integer side: a `WeightVal` is never smaller than a `u128` and is
        // usually a multi-limb `BigRational`, so the clone-then-replace form it
        // replaces duplicated every surviving rational's heap payload as well.
        //
        // SOUNDNESS (why a move can't clobber a slot still to be read) — the
        // same ASCENDING-`referenced` argument as the integer impl, re-checked
        // for the different move this store needs. `referenced` is strictly
        // ascending and duplicate-free (its only producer,
        // `referenced_marg_slots`, sorts it — the OOB guard at the call site
        // reads `referenced.last()` as the max on the same assumption). At step
        // `i` the source is `old_i >= i` and the destination is
        // `new_len <= i <= old_i`.
        //
        // `WeightVal` is not `Copy`, so the move down is a SWAP, not an
        // assignment: step `i` therefore writes TWO slots, `new_len` and
        // `old_i`. Both are `<= old_i`, and every later read is at
        // `old_j > old_i` (strict ascent), so neither write can land on a slot
        // a later step still reads. The displaced value swapped UP to `old_i`
        // is dead from that moment on — never read again, and dropped by the
        // closing `truncate`. (The integer impl gets away with a plain
        // assignment because `u128` is `Copy`; its `_big` side table uses
        // `mem::take` for the same "leave nothing stale behind" reason the swap
        // gives us for free.)
        {
            let ws = weights_mut(tdd);
            if !ws.is_set(v.idx()) {
                // Boundary level flagged marginal with no store allocated: leave
                // an empty-but-present store, as the clone-then-replace form
                // did. `Some(empty)` is the "marginal, zero slots" state
                // `clear_dead_store` also writes, and `ensure_weights` reads it
                // as "already weight-marginal". Only reachable with an empty
                // `referenced` — the caller's OOB guard rejects any ref into a
                // zero-length store.
                ws.set_level(v.idx(), Vec::new());
            }
            let vals = ws.level_vals_mut(v.idx()).expect("store present: ensured just above");
            let mut interner: WeightMap = FxHashMap::default();
            let mut values_merged = 0usize;
            let mut new_len = 0usize;
            for &old in referenced {
                let old = old as usize;
                let key = weight_key(&vals[old]);
                // A hit means `old` mapped onto an equal-valued surviving slot
                // (value-dedup merge); a miss mints the next compacted slot.
                if let Some(&slot) = interner.get(&key) {
                    remap[old] = slot;
                    values_merged += 1;
                    continue;
                }
                interner.insert(key, new_len as u32);
                vals.swap(new_len, old);
                remap[old] = new_len as u32;
                new_len += 1;
            }
            // Drops the orphans, the merged-away duplicates, and the values
            // swapped up out of the prefix — the point of the pass.
            vals.truncate(new_len);
            // Slack ceiling: compaction is IN PLACE, so the capacity observed
            // here is the PRE-compaction one. Shrinking at 2× therefore
            // reclaims exactly when the store more than halved — the effective
            // ceiling on retained slack, matching the integer impl.
            if vals.capacity() > 64 && vals.capacity() > 2 * vals.len() {
                vals.shrink_to_fit();
            }
            (new_len, values_merged)
        }
    }

    /// WEIGHTED SEMANTIC: `retired_marg_width` IS the live slot count —
    /// `TddLevel::width()` returns it for a weight-marginal level (set by
    /// `make_marginal_weighted_with_slots`), and the apply/streaming buffers are
    /// sized from that. ASSIGN `new_len`; `freed` is stats-only here and must
    /// NOT be added, or the width drifts up and re-opens the oversized-buffer
    /// blowup described on the impl above.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, _freed: usize, new_len: usize) {
        tdd.levels[v.idx()].retired_marg_width = new_len as u32;
        debug_assert_eq!(
            tdd.weights.as_ref().and_then(|ws| ws.level(v.idx())).map_or(0, |s| s.len()),
            new_len,
            "weighted retired_marg_width must equal the live WeightStore length"
        );
    }
}

/// The one prune skeleton, generic over where the values live. See the module
/// comment's table for what stays per-kind.
fn prune_marg_slots_generic<S: SlotStore>(tdd: &mut Tdd) -> MargSlotPruneStats {
    // Central pin-invariant check (debug builds, weighted mode only): a
    // weight-marginal LEAF's column is the immutable label-ordered `leaf_val`
    // triple. This pass runs tens of times per compile, so a regression in ANY of
    // the passes that could break it lands here immediately.
    crate::marginal::debug_check_leaf_columns_pinned(tdd);
    let mut stats = MargSlotPruneStats::default();
    // Sweep-lifetime scratch: the ref-collector's result/dedup buffers and the
    // old→new slot map. Reused across boundary levels — and, via the pool,
    // across sweeps (the sweep is per-merge, so a fresh allocation per level or
    // per call shows up as pure allocator churn on many-small-diagram
    // workloads). Both are refilled per level below (`referenced_marg_slots`
    // clears its own buffers; `remap` is cleared and resized), so a pooled pair
    // differs from a fresh one only in capacity.
    let (mut slots, mut remap) = take_sweep_scratch();
    // The output level's store is the result (a marginal output level has no
    // parent refs at all — e.g. a fully-marginalized component sub-TDD whose
    // output sits at the subtree root under an empty parent level). Never
    // touch it.
    let out_v = tdd.output.vtree;


    clear_dead_deep_stores::<S>(tdd, out_v, &mut stats);
    compact_boundary_stores::<S>(tdd, out_v, &mut stats, &mut slots, &mut remap);

    return_sweep_scratch(slots, remap);
    stats
}

/// Rewrite a marg-side slot ref through `remap`. ZERO sentinels (bit 31) and
/// inline refs pass through verbatim.
#[inline]
fn remap_slot_ref(raw: &mut u32, remap: &[u32]) {
    if *raw & (1u32 << 31) != 0 {
        return;
    }
    if let MargRef::Slot(s) = MargRef::from_raw(*raw) {
        let new = remap[s as usize];
        debug_assert_ne!(new, u32::MAX, "referenced slot must survive slot-prune");
        *raw = MargRef::slot_raw(new);
    }
}

#[path = "slot_prune_stores.rs"]
mod stores;
use stores::{clear_dead_deep_stores, compact_boundary_stores};

#[cfg(test)]
#[path = "slot_prune_tests.rs"]
mod slot_prune_tests;
