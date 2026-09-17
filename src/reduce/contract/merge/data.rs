//! Unioning twin pair lists into the survivor and compacting the level after.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;
use crate::diagram::*;

use super::super::scratch::{DuplicateScratch, MergeRemap};

/// Compact the t1 level after twin contraction, sweep its arena, then run
/// fork-down duplicate resolution on the survivors merged via the concat-all
/// path. Fork-down runs after compaction so survivor indices are final; it
/// folds a run of k equal pairs into one pair with its marginal side scaled by
/// k only where t1 has a marginal child, and otherwise leaves the run (see the
/// module doc of `duplicate_pair_resolve`).
#[inline(always)]
pub(super) fn compact_and_fork_down(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    resolve_keeps: &[u32],
    remap: &MergeRemap,
    duplicate: &mut DuplicateScratch,
) -> Result<(), OperationError> {
    // Step 3: Compact the level in-place (keep only alive nodes). t1 is never
    // marginal here (see the guard note in `contract_twins`), so only the
    // explicit-level compaction is reachable.
    compact_explicit_level(&mut tdd.levels[t1.idx()], &remap.merge_target);

    // Sweep t1's merge garbage now: every union was appended at the arena tail
    // and every absorbed node has just been dropped, so up to half the arena is
    // unreferenced, and fork-down below grows the arenas again. Legal here: the
    // caller obligation on `compact_pairs_if_stale` is to hold no pair-arena
    // offset across the call, and everything live at this point
    // (`merge_target`, `final_remap`, `resolve_keeps`, `tdd.output.local`) is a
    // node index. What fork-down leaves behind is charged to `dead_pairs` and
    // waits for the next sweep.
    tdd.levels[t1.idx()].compact_pairs_if_stale();

    // Update output if it points to t1
    if tdd.output.vtree == t1 {
        tdd.output.local = remap.final_remap[tdd.output.local.idx()];
    }

    // Fork-down resolution, after compaction so survivor indices are final.
    // One scratch for the whole loop, cleared per node inside the callee.
    for &old_keep in resolve_keeps {
        let new_idx = remap.final_remap[old_keep as usize].idx();
        super::super::duplicate_pair_resolve::resolve_duplicate_pairs_in_node(eng, tdd, t1, new_idx, duplicate)?;
    }
    Ok(())
}

/// Merge a group of twin nodes' data into the first node (the "kept" node):
/// concatenate the members' pair lists at the arena tail and point the kept
/// node at the result. Pair lists are unordered sets and `find_twin_groups`
/// canonicalizes each signature slice before comparing, so concatenation is
/// the union with no sort. A repeated `(L, R)` entry across or within the
/// inputs is a legitimate multiset entry at a marginal-child level and is
/// preserved; at a fully non-marginal level determinism (invariant 1) makes
/// the supports disjoint, which `concat_twin_pairs` checks in debug builds.
///
/// Only called on internal nodes (leaf levels are marginal and never contracted).
pub(super) fn merge_twin_data(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    group: &[u32],
    allow_dups: bool,
) {
    let keep = group[0] as usize;
    // The debug set-ness check in `concat_twin_pairs` holds only for a purely
    // Boolean diagram: once any level is marginal, every count consumer folds
    // `Σ_pairs c(l)·c(r)` and the content-twin merge (`content_twin.rs`)
    // rewrites refs at plain levels too, so a plain-level node can arrive here
    // already holding the same pair twice, and concatenating it with a disjoint
    // twin carries that duplicate through. `cfg!` is a compile-time constant,
    // so the level scan is dead code in release.
    let allow_dups = allow_dups || (cfg!(debug_assertions) && tdd.has_marginal_level());
    let level = &mut tdd.levels[t1.idx()];

    // Leaf levels are marginal — `contract_all_twins` never calls this for leaves.
    debug_assert!(
        level.nodes[keep].is_internal(),
        "merge_twin_data called on leaf node — leaf levels should be skipped"
    );

    let total: usize = group.iter().map(|&idx| level.pair_count_at(idx as usize)).sum();
    concat_twin_pairs(level, keep, group, total, allow_dups);
}

/// Concatenate the pair lists of `group`'s nodes at the arena tail and point
/// `keep` at the result. `total` must be the exact summed pair count.
///
/// Sources are ranges of the arena itself (or inline node data), so
/// `extend_from_within` copies arena to arena with no temp buffer. Infallible:
/// `reserve_transactional` charged the whole merge loop's growth before any
/// mutation, so the pushes below cannot reallocate.
///
/// Every source range is left behind as dead arena and counted into
/// `TddLevel::dead_pairs`: here for the survivor, in `compact_explicit_level`
/// for the absorbed members.
pub(super) fn concat_twin_pairs(
    level: &mut TddLevel,
    keep: usize,
    group: &[u32],
    total: usize,
    allow_dups: bool,
) {
    let new_start = level.pairs.len();
    debug_assert!(
        level.pairs.capacity() - level.pairs.len() >= total,
        "concat_twin_pairs: hoisted grand reserve under-sized pairs capacity"
    );
    for &idx in group {
        let d = level.nodes[idx as usize];
        if d.is_inline() {
            level.pairs.push(d.inline_pair());
        } else {
            // `pair_range_at` handles both normal and extended multi encodings;
            // the range is owned, so the immutable borrow ends before the extend.
            let r = level.pair_range_at(idx as usize);
            level.pairs.extend_from_within(r);
        }
    }
    debug_assert_eq!(level.pairs.len() - new_start, total);
    // At fully non-marginal levels determinism (invariant 1) makes twin
    // supports pairwise disjoint, so the concatenation has no duplicates; a
    // debug-only full check. Skipped where the level carries marginal markers,
    // or where `allow_dups` says the caller resolves the duplicates right after
    // compaction (`duplicate_pair_resolve`).
    #[cfg(debug_assertions)]
    if !level.any_inlined_side() && !allow_dups {
        let mut chk: Vec<ChildPair> = level.pairs[new_start..].to_vec();
        chk.sort_unstable();
        debug_assert!(
            chk.windows(2).all(|w| w[0] != w[1]),
            "twin contraction (concat merge): duplicate pair across twin supports — invariant 1 violation"
        );
    }
    #[cfg(not(debug_assertions))]
    let _ = allow_dups;
    // The survivor is about to point at the tail copy, abandoning its own source
    // range; the absorbed members' ranges are accounted when compaction drops
    // their nodes (`compact_explicit_level`).
    let abandoned = level.arena_pairs_at(keep);
    finalize_merged_node(level, keep, new_start, total);
    level.note_dead_pairs(abandoned);
}

/// Finalize node at `level.nodes[keep]` from a merged pair sequence already
/// written to `level.pairs[new_start..new_start + new_len]`.
///
/// Encoding picks:
/// - `new_len == 1` + pair fits inline: pop the speculative pair from the arena
///   and store the pair directly in the node (no heap traffic).
/// - `new_len == 1` + pair cannot inline (e.g. high bit set): keep the pair in
///   the arena, record a 1-element `MultiPairRange` side-entry, and tag the node as
///   `multi_ranged`. (Required because the packed single-pair encoding aliases
///   leaf or multi-extended encodings when the high bits are set.)
/// - `new_len >= 2`: delegate to `encode_multi`, which picks packed vs extended
///   based on whether `new_start`/`new_len` fit in the packed bit-budget.
#[inline]
fn finalize_merged_node(
    level: &mut TddLevel,
    keep: usize,
    new_start: usize,
    new_len: usize,
) {
    if new_len == 1 {
        let pair = level.pairs[new_start];
        if pair.can_inline() {
            level.pairs.pop();
            level.nodes[keep] = EncodedNode::inline(pair);
        } else {
            // The grand reserve charged one `MultiPairRange` per group on
            // `level.multi_pairs`, so this push cannot reallocate — plain push.
            let multi_pairs_idx = level.multi_pairs.len();
            debug_assert!(
                level.multi_pairs.capacity() > level.multi_pairs.len(),
                "finalize_merged_node: hoisted grand reserve under-sized multi_pairs capacity"
            );
            level.multi_pairs.push(MultiPairRange { start: new_start as u64, len: 1 });
            level.nodes[keep] = EncodedNode::multi_ranged(multi_pairs_idx as u32);
        }
    } else {
        level.nodes[keep] = level.encode_multi(new_start, new_len);
    }
}

/// Compact a non-marginal (explicit) level in-place after twin contraction.
///
/// Walks `level.nodes`, keeping only entries whose `merge_target[read] == read`
/// (i.e. canonical survivors; absorbed twins are skipped). Survivors are shifted
/// left in-place via `swap` and the vec is truncated. Mirrors `compact_levels`
/// in `reduce/prune.rs` for the non-marginal case.
///
/// Also the accounting point for absorbed twins' pair ranges: dropping the node
/// is what makes its range unreferenced (whether the merge copied the content to
/// the survivor's tail range, or a duplicate redirect left the pair list untouched).
#[inline(always)]
pub(super) fn compact_explicit_level(level: &mut TddLevel, merge_target: &[u32]) {
    let n = level.nodes.len();
    let mut write = 0usize;
    // Accumulated and noted once after the walk: the counter's only reader
    // (`compact_pairs_if_stale`) runs later, so per-drop adds bought nothing.
    let mut dead_acc = 0usize;
    // In-place compaction: the read index advances independently of the write cursor.
    #[expect(clippy::needless_range_loop)]
    for read in 0..n {
        if merge_target[read] == read as u32 {
            if write < read {
                level.nodes.swap(write, read);
            }
            write += 1;
        } else {
            // `swap` only ever writes to positions ≤ the current `read`, so
            // `nodes[read]` is still this slot's original node here.
            dead_acc += level.arena_pairs_at(read);
        }
    }
    level.nodes.truncate(write);
    level.note_dead_pairs(dead_acc);
}

