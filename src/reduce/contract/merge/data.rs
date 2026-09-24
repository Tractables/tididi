//! Unioning twin pair lists into the survivor and compacting the level after.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;
use crate::diagram::{EncodedNode, NodeKind, Tdd, TddLevel};

use super::super::scratch::{DuplicateScratch, MergeRemap};

/// Compact the t1 level after twin contraction, sweep its arena, then run
/// fork-down duplicate resolution on the survivors merged via the concat-all
/// path. Fork-down runs after compaction so survivor indices are final; it
/// folds a run of k equal pairs into one pair with its marginal side scaled by
/// k only where t1 has a marginal child, and otherwise leaves the run (see the
/// module doc of `duplicate_pair_resolve`).
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
/// Only called on internal nodes: the sweep never contracts a leaf level.
pub(super) fn merge_twin_data(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    group: &[u32],
    allow_dups: bool,
) {
    let keep = group[0] as usize;
    // Read only for the debug check in `concat_twin_pairs`; `cfg!` is a
    // constant, so the level scan is compiled out of release builds.
    let diagram_marginal = cfg!(debug_assertions) && tdd.has_marginal_level();
    let level = &mut tdd.levels[t1.idx()];
    debug_assert!(
        level.nodes[keep].is_internal(),
        "merge_twin_data: the sweep never contracts a leaf level"
    );
    let total: usize = group.iter().map(|&idx| level.pair_count_at(idx as usize)).sum();
    concat_twin_pairs(level, keep, group, total, allow_dups, diagram_marginal);
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
///
/// `allow_dups` and `diagram_marginal` only decide whether the debug check
/// for duplicate pairs applies; release builds ignore them.
pub(super) fn concat_twin_pairs(
    level: &mut TddLevel,
    keep: usize,
    group: &[u32],
    total: usize,
    allow_dups: bool,
    diagram_marginal: bool,
) {
    let new_start = level.pairs.len();
    debug_assert!(
        level.pairs.capacity() - level.pairs.len() >= total,
        "concat_twin_pairs: hoisted grand reserve under-sized pairs capacity"
    );
    for &idx in group {
        let d = level.nodes[idx as usize];
        if let NodeKind::Inline(pair) = d.kind() {
            level.pairs.push(pair);
        } else {
            // `pair_range_at` handles both normal and extended multi encodings;
            // the range is owned, so the immutable borrow ends before the extend.
            let r = level.pair_range_at(idx as usize);
            level.pairs.extend_from_within(r);
        }
    }
    debug_assert_eq!(level.pairs.len() - new_start, total);
    // In a purely Boolean diagram determinism (invariant 1) makes twin
    // supports pairwise disjoint, so the concatenation has no duplicates; a
    // debug-only full check. A duplicate is legal, and the check skipped,
    // where the level carries marginal markers, where any level of the
    // diagram is marginal (every count consumer folds `Σ_pairs c(l)·c(r)`,
    // and the content-twin merge can leave a plain-level node holding the
    // same pair twice), and where `allow_dups` says the caller resolves the
    // duplicates right after compaction (`duplicate_pair_resolve`).
    if cfg!(debug_assertions) && !allow_dups && !diagram_marginal && !level.any_value_ref_side() {
        let mut chk = level.pairs[new_start..].to_vec();
        chk.sort_unstable();
        assert!(
            chk.windows(2).all(|w| w[0] != w[1]),
            "twin contraction (concat merge): duplicate pair across twin supports — invariant 1 violation"
        );
    }
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
/// A single pair is popped off the arena again and stored inline; a longer
/// list goes through `encode_multi`, which picks packed or extended by
/// whether `new_start` and `new_len` fit the packed bit budget.
#[inline]
fn finalize_merged_node(
    level: &mut TddLevel,
    keep: usize,
    new_start: usize,
    new_len: usize,
) {
    if new_len == 1 {
        let pair = level.pairs[new_start];
        level.pairs.pop();
        level.nodes[keep] = EncodedNode::inline(pair);
    } else {
        level.nodes[keep] = level.encode_multi(new_start, new_len);
    }
}

/// Compact a non-marginal (explicit) level in-place after twin contraction.
///
/// Walks `level.nodes`, keeping only entries whose `merge_target[read] == read`
/// (i.e. canonical survivors; absorbed twins are skipped). Survivors are shifted
/// left in-place via `swap` and the vec is truncated.
///
/// Also the accounting point for absorbed twins' pair ranges: dropping the node
/// is what makes its range unreferenced (whether the merge copied the content to
/// the survivor's tail range, or a duplicate redirect left the pair list untouched).
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

