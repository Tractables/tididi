use crate::diagram::{ChildDecoder, ChildSide, EncodedChildRef, Tdd, TddLevel};
use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;

use super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};

/// Prefetch the twin-table slot a future iteration will probe first. The
/// probe target is a random index into a table that typically misses L2;
/// `fingerprints[]` is read sequentially, so the slot for iteration i+D is
/// known D iterations ahead. No-op on non-x86_64, and under Miri, which does
/// not implement the intrinsic.
fn prefetch_slot(p: *const TwinSlot, slot: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            p.add(slot) as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = (p, slot);
}

/// Iterate parent pairs and yield `(parent_i, target, sibling)` to `f`,
/// where `target` is the child index at level `t1` (left or right of each pair
/// depending on `t1_side`) and `sibling` is the other child.
///
/// Shared by the scatter passes of `find_twin_groups`.
#[inline]
pub(super) fn for_each_target_sibling(
    parent_level: &TddLevel,
    t1_side: ChildSide,
    target: ChildDecoder,
    mut f: impl FnMut(u32, u32, u32),
) {
    // `target` indexes child-width-sized scratch arrays, so it is the cell the
    // ref names. A side carrying an inline value names no cell (it is a count,
    // not a child node), so it never joins twin grouping; the parent rewrite
    // leaves such a ref verbatim. `sibling` is passed raw: it is only hashed
    // and packed, never indexed.
    let resolve_target = |side: EncodedChildRef| target.child(side).index().map(|c| c as u32);
    // `pairs_of` slice iteration (compiler-vectorizable).
    for (parent_i, parent_node) in parent_level.nodes.iter().enumerate() {
        let pi = parent_i as u32;
        for pair in parent_level.pairs_of(parent_node) {
            if t1_side == ChildSide::Left {
                if let Some(t) = resolve_target(pair.left) {
                    f(pi, t, pair.right.0);
                }
            } else {
                if let Some(t) = resolve_target(pair.right) {
                    f(pi, t, pair.left.0);
                }
            }
        }
    }
}

/// The splitmix64 finalizer (Steele et al., 2014) — the shared bit-diffusion
/// step behind every fingerprint in the contract module.
///
/// Constants and shift schedule are the original SplitMix64 ones, chosen for
/// their avalanche behaviour.
///
/// Callers own their own prelude (how the inputs are packed into the u64, and
/// whether a golden-ratio increment is added first) — that prelude is what
/// makes each rule's fingerprint distribution distinct, so do not fold one
/// caller's prelude in here.
pub(super) fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Hash a (parent_idx, sibling_idx) context pair to a u64 fingerprint.
///
/// The wrapping sum of these hashes over a multiset of contexts is a cheap
/// pre-screen for twin detection (~1/2^64 false-positive probability) before
/// doing exact signature comparison. Using wrapping_add rather than exclusive-or means
/// duplicate contexts contribute 2h rather than cancelling; removal of a
/// contribution uses wrapping_sub. Order-independence holds because addition
/// commutes.
pub(super) fn context_hash(parent_i: u32, sibling_j: u32) -> u64 {
    // Prelude: pack two 32-bit values; no increment.
    mix64((parent_i as u64) << 32 | sibling_j as u64)
}

mod groups;

#[cfg(test)]
mod tests;

/// Find nodes with identical parent-context multisets at the explicit child level.
///
/// First compute additive fingerprints and discard nodes with unique fingerprints.
/// Only collision candidates receive full sorted signatures; equal signatures
/// form a twin group. Marginal children are handled by pair fusion instead.
/// Groups are stored in `scratch.flat_groups`, indexed by `scratch.group_starts`.
/// Returns whether any group contains at least two nodes.
pub(super) fn find_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    t: VtreeIdx,
    t1_side: ChildSide,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    if child_width == 0 {
        return Ok(false);
    }
    // Node indices at this level are written u32-wide below (`cursors`,
    // `flat_groups`). That is the same invariant the merge side already relies
    // on (`merge_target[i] = i as u32`, `final_remap: Vec<NodeIdx>`): every
    // ref into a level is a `NodeIdx(u32)`, so a level wider than 2^32
    // slots could not be referenced at all.
    debug_assert!(
        child_width <= u32::MAX as usize,
        "level width {child_width} exceeds the u32 node-index range",
    );

    let parent_level = &tdd.levels[t.idx()];
    // How to read the parent refs that point at the contracted child (t1).
    let (left_c, right_c) = tdd.vtree.children(t);
    let t1_node = if t1_side == ChildSide::Left { left_c } else { right_c };
    let t1_view = tdd.levels[t1_node.idx()].child_decoder();

    // ── Pre-test: fingerprint-only scatter ────────────────────────────────────
    //
    // The common case is "no twins at this level", so only the additive
    // fingerprint is written here (no counts), touching half the cache lines
    // per parent pair. Accumulation is `wrapping_add`, which commutes, so the
    // result is order-independent; duplicate (parent, sibling) pairs contribute
    // 2h rather than cancelling as exclusive-or would, so even-multiplicity
    // duplicates (legal at marginal boundary levels) cannot collapse a
    // fingerprint to 0. Counts are computed in a second pass only after a
    // collision.
    lim.try_resize(&mut scratch.fingerprints, child_width, 0u64)?;
    scratch.fingerprints[..child_width].fill(0);

    for_each_target_sibling(parent_level, t1_side, t1_view, |pi, target, sibling| {
        scratch.fingerprints[target as usize] =
            scratch.fingerprints[target as usize].wrapping_add(context_hash(pi, sibling));
    });

    // ── Fingerprint collision check + candidate marking ───────────────────────
    //
    // Open-addressing table keyed by fingerprint. Every node that shares its
    // fingerprint with an earlier one is marked a twin candidate; no collision
    // means no twins, the common case. Marking is folded into this pass and
    // scans the full width (no early exit on the first collision) so that
    // `build_twin_groups_after_collision` can skip the unique-fingerprint
    // nodes in its scatters.
    if !mark_candidates(eng, scratch, child_width)? {
        return Ok(false);
    }

    build_twin_groups_after_collision(
        eng,
        parent_level,
        t1_side,
        t1_view,
        child_width,
        scratch,
    )
}

/// Size the open-addressing twin table for a pass that inserts at most
/// `max_occupancy` entries. The one sizing rule for `scratch.twin_hash_table`:
/// Both probe loops (`mark_candidates` and Pass 1 of
/// `build_twin_groups_after_collision`) call it.
///
/// # Soundness
///
/// Both loops probe linearly and never delete, so a probe for a fingerprint
/// meets every equal-fingerprint entry before an empty slot whatever the size.
/// The size must exceed `max_occupancy`, or a probe for an absent fingerprint
/// wraps forever; the `div_ceil` term is at least 1, so it does.
///
/// Rounding to `4/3 · max_occupancy` caps the load factor at 3/4.
#[inline]
fn twin_table_size(max_occupancy: usize) -> usize {
    (max_occupancy + max_occupancy.div_ceil(3))
        .next_power_of_two()
        .max(4)
}

/// Mark twin candidates among `scratch.fingerprints[..width]`: every node whose
/// fingerprint is shared with ≥1 other node is flagged in
/// `scratch.is_candidate`. Returns whether any node was flagged; `false` means
/// all fingerprints are distinct, hence no twins. The fingerprint is stored in
/// the slot beside the occupant index, so each probe is one random load.
#[inline]
fn mark_candidates(
    eng: &Engine,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    lim.try_resize(&mut scratch.is_candidate, width, false)?;
    scratch.is_candidate[..width].fill(false);
    // One insert at most per `0..width` iteration ⇒ occupancy ≤ width.
    let table_size = twin_table_size(width);
    let mask = table_size - 1;
    let ht = &mut scratch.twin_hash_table;
    lim.try_resize(ht, table_size, EMPTY_SLOT)?;
    ht[..table_size].fill(EMPTY_SLOT);
    let mut found = false;
    const PF_DIST: usize = 8;
    for i in 0..width {
        // Prefetch the twin-table slot that iteration `i + PF_DIST` will first probe.
        // `fingerprints[]` is sequential so the slot address is known ahead of time;
        // the ht probe is a random access into a table that typically misses L2.
        //
        // The pointer is re-derived from `ht` each iteration rather than taken
        // once before the loop: the probe below writes through `ht`, which
        // invalidates any raw pointer derived from it earlier, so a hoisted
        // one would be used after it went stale. `as_ptr` is a field read, and
        // the prefetch is a hint, so nothing here is a real load.
        if i + PF_DIST < width {
            let a = i + PF_DIST;
            prefetch_slot(ht.as_ptr(), (scratch.fingerprints[a] as usize) & mask);
        }
        let fp = scratch.fingerprints[i];
        let mut slot = (fp as usize) & mask;
        loop {
            let s = ht[slot];
            if s.idx == u64::MAX {
                ht[slot] = TwinSlot { fp, idx: i as u64 };
                break;
            }
            if s.fp == fp {
                // fingerprint match: both nodes are twin candidates. Idempotent
                // stores — re-flagging an already-flagged node is a redundant
                // write, not a miscount, so the probe loop needs no guards.
                let occ = s.idx as usize;
                scratch.is_candidate[i] = true;
                scratch.is_candidate[occ] = true;
                found = true;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
    Ok(found)
}

use groups::build_twin_groups_after_collision;
