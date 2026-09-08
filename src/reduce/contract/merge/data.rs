//! Unioning twin pair lists into the survivor and compacting the level after.

use crate::vtree::VtreeIdx;

use crate::limits::ApplyError;
use crate::diagram::*;

use super::super::scratch::ContractScratch;

/// Compact the t1 level after twin contraction and run fork-down duplicate
/// resolution for any survivors that were merged via the concat-all path
/// (Step 3b of `contract_twins`).
///
/// Compaction removes absorbed twins in-place; the freed arena is then swept
/// (see the sweep note inline) before fork-down resolution replaces each run of
/// k equal pairs in a survivor with one pair whose marginal side is scaled by k
/// — multiplicity is preserved in counts, never set-dedup'd. It only fires where
/// that scale is O(1) (t1 has a marginal child); elsewhere the run stays as k
/// legal multiset terms, which sum to the same count (see `dup_resolve`'s cost
/// policy). Fork-down runs after compaction so survivor indices are final.
#[inline(always)]
pub(super) fn compact_and_fork_down(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    resolve_keeps: &[u32],
    scratch: &mut ContractScratch,
) -> Result<(), ApplyError> {
    // Step 3: Compact the level in-place (keep only alive nodes). t1 is never
    // marginal here (see the guard note in `contract_twins`), so only the
    // explicit-level compaction is reachable.
    compact_explicit_level(&mut tdd.levels[t1.idx()], &scratch.merge_target);

    // Reclaim t1's merge garbage HERE, at the moment its dead fraction is
    // maximal and known: every merged union has been appended at the arena tail
    // and every absorbed member's node has just been dropped, so up to half the
    // arena is unreferenced — and fork-down below is what GROWS the arenas
    // again (scaled clones at t1's children, a re-encoded survivor here).
    // Sweeping before that growth is what keeps the two from being resident
    // together at the peak; the sweep only slides live ranges down, preserving
    // every node's pair slice and its order byte-for-byte.
    //
    // Legal here: the caller obligation on `compact_pairs_if_stale`
    // (types/level.rs) is to hold no pair-arena offset across the call, and
    // nothing live at this point is one — `merge_target`/`final_remap`/
    // `resolve_keeps`/`tdd.output.local` are all NODE indices, and fork-down
    // resolves each node to its slice through `pairs_of_idx` at use time.
    // What fork-down leaves behind (shrunk survivor tails) is charged to
    // `dead_pairs` and waits for the next contraction's sweep, exactly as the
    // counter is designed for.
    tdd.levels[t1.idx()].compact_pairs_if_stale();

    // Update output if it points to t1
    if tdd.output.vtree == t1 {
        tdd.output.local = scratch.final_remap[tdd.output.local.idx()];
    }

    // Fork-down resolution: survivors merged on the concat-all path may hold
    // duplicate pairs (overlapping twin supports). Fold each run of k equal
    // pairs into one pair whose marginal side carries the factor k, where that
    // is an O(1) count scale; otherwise leave the run — multiplicity is
    // preserved either way, never set-dedup'd. Runs after compaction so
    // survivor indices are final.
    // One scratch for the whole loop (cleared per node inside the callee): the
    // resolver runs once per survivor, so its three working buffers were three
    // fresh allocations per NODE — the finest granularity on this path.
    for &old_keep in resolve_keeps {
        let new_idx = scratch.final_remap[old_keep as usize].idx();
        super::super::dup_resolve::resolve_duplicate_pairs_in_node(tdd, t1, new_idx, &mut scratch.dup)?;
    }
    Ok(())
}

/// Merge a group of twin nodes' data into the first node (the "kept" node).
///
/// Only called on internal nodes (leaf levels are marginal and never contracted).
/// Unions input pair sets of twin nodes. Paths by group/pair count:
///   - 2 twins with 1 pair each: inline, no allocation
///   - otherwise: arena-internal concatenation (`concat_twin_pairs`) —
///     concatenation IS the union since pair lists are unordered sets
pub(super) fn merge_twin_data(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    group: &[u32],
    allow_dups: bool,
) {
    let keep = group[0] as usize;
    let level = &mut tdd.levels[t1.idx()];

    // Leaf levels are marginal — contract_all_twins never calls this for leaves.
    debug_assert!(
        level.nodes[keep].is_internal(),
        "merge_twin_data called on leaf node — leaf levels should be skipped"
    );

    // Internal twins: union input pair sets.
    if group.len() == 2 {
        merge_two_internal_twins(level, keep, group[1] as usize, allow_dups);
    } else {
        merge_many_internal_twins(level, keep, group, allow_dups);
    }
}

/// Merge two internal twin nodes — the most common case.
///
/// Concatenates both nodes' pair lists at the arena tail via
/// `extend_from_within` (no temp buffers), then updates the kept node's
/// pair_start/pair_len. Pair lists are unordered sets and
/// `find_twin_groups` canonicalizes each signature slice before comparing, so
/// no consumer needs the union sorted — plain concatenation IS the union.
/// Duplicate `(L, R)` entries across (and within) the inputs are legitimate
/// multiset entries at marginal-child levels — count-keyed slot sharing
/// (`apply_p_fusion`) lets each occurrence carry one historical plan's
/// `c(L)·c(R)` contribution — and concatenation
/// preserves them by construction. At fully non-marginal levels determinism
/// (Invariant 2) guarantees the supports are disjoint (checked debug-only in
/// `concat_twin_pairs`).
///
/// Do NOT reintroduce an ordered merge through temp buffers: on pathological
/// nodes the two transient copies land at exactly the moment memory is
/// tightest.
pub(super) fn merge_two_internal_twins(
    level: &mut TddLevel,
    keep: usize,
    other: usize,
    allow_dups: bool,
) {
    // 1+1 fast path: merge two single-pair nodes without allocation.
    // After the inline encoding, single-pair nodes are inline (pair in the node itself).
    let keep_len = level.pair_count_at(keep);
    let other_len = level.pair_count_at(other);
    if keep_len == 1 && other_len == 1 {
        let pa = level.pairs_of_idx(keep)[0];
        let pb = level.pairs_of_idx(other)[0];
        // pa == pb is permitted and both copies must survive — see this
        // function's doc for why duplicates are legitimate multiset entries.
        // The grand reserve charged these two pairs (keep_len + other_len), so
        // the pushes cannot reallocate — plain push.
        let new_start = level.pairs.len();
        debug_assert!(
            level.pairs.capacity() - level.pairs.len() >= 2,
            "merge_two_internal_twins: hoisted grand reserve under-sized pairs capacity"
        );
        // A 1-pair node is normally inline (owning no arena slot), but the
        // extended encoding also carries len-1 nodes; if `keep` was one, the
        // re-encode below abandons its slot. `other`'s is accounted when
        // compaction drops its node.
        let abandoned = level.arena_pairs_at(keep);
        level.pairs.push(pa);
        level.pairs.push(pb);
        let data = level.encode_multi(new_start, 2);
        level.nodes[keep] = data;
        level.note_dead_pairs(abandoned);
        return;
    }

    concat_twin_pairs(
        level,
        keep,
        &[keep as u32, other as u32],
        keep_len + other_len,
        allow_dups,
    );
}

/// Merge 3+ internal twin nodes. Rare in practice — most twin groups have
/// exactly 2 members. Same concatenation-is-union argument as
/// `merge_two_internal_twins` (the previous sort here existed only to support
/// a windows-based duplicate assert, now done debug-only in
/// `concat_twin_pairs`).
fn merge_many_internal_twins(
    level: &mut TddLevel,
    keep: usize,
    group: &[u32],
    allow_dups: bool,
) {
    let total: usize = group.iter().map(|&idx| level.pair_count_at(idx as usize)).sum();
    concat_twin_pairs(level, keep, group, total, allow_dups);
}

/// Concatenate the pair lists of `group`'s nodes at the arena tail and point
/// `keep` at the result. `total` must be the exact summed pair count.
///
/// Sources are ranges of the arena itself (or inline node data), so
/// `extend_from_within` copies arena→arena with no temp buffer. Infallible: the
/// arena growth of the WHOLE merge loop (a group can total ~1B pairs ≈ 8 GiB of
/// `InputPair`, asserted 8 bytes in types.rs) is charged ONCE up front by the
/// hoisted grand reserve in `contract_twins`, which bails before any
/// mutation on OverBudget. By the time we get here the capacity is guaranteed,
/// so the extends/pushes below cannot reallocate — hence plain `push`/`extend`.
///
/// Every source range is left behind as dead arena (the union is a tail copy).
/// Those slots are counted into `TddLevel::dead_pairs` — here for the survivor,
/// in `compact_explicit_level` for the absorbed members — and reclaimed by the
/// sweep at the end of `contract_twins`.
fn concat_twin_pairs(
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
    // At fully non-marginal levels, Invariant 2 (determinism) guarantees twin
    // supports are pairwise disjoint, so the concatenation has no duplicates.
    // Debug-only full check — stronger than an adjacency-only test, since
    // concatenation can place equal pairs anywhere. Skipped when the level carries marg
    // markers — there duplicate `(L, R)` entries are legitimate multiset
    // entries (see `merge_two_internal_twins`).
    // `allow_dups`: the caller is on the concat-then-fork-down path (plain
    // scalable level) and resolves the duplicates immediately after
    // compaction (dup_resolve) — transient duplicates are expected there.
    #[cfg(debug_assertions)]
    if level.marg_flags == 0 && !allow_dups {
        let mut chk: Vec<InputPair> = level.pairs[new_start..].to_vec();
        chk.sort_unstable();
        debug_assert!(
            chk.windows(2).all(|w| w[0] != w[1]),
            "twin contraction (concat merge): duplicate pair across twin supports — Invariant 2 violation"
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
/// written to `level.pairs[new_start..new_start + new_len]`. Used by the two-twin
/// and many-twin merge paths after they've appended the deduplicated pairs.
///
/// Encoding picks:
/// - `new_len == 1` + pair fits inline: pop the speculative pair from the arena
///   and store the pair directly in the node (no heap traffic).
/// - `new_len == 1` + pair cannot inline (e.g. high bit set): keep the pair in
///   the arena, record a 1-element `ExtMulti` side-entry, and tag the node as
///   `multi_extended`. (Required because the packed single-pair encoding aliases
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
            level.nodes[keep] = TddNodeData::inline(pair);
        } else {
            // The grand reserve charged one `ExtMulti` per group on
            // `level.ext`, so this push cannot reallocate — plain push.
            let ext_idx = level.ext.len();
            debug_assert!(
                level.ext.capacity() > level.ext.len(),
                "finalize_merged_node: hoisted grand reserve under-sized ext capacity"
            );
            level.ext.push(ExtMulti { start: new_start as u64, len: 1 });
            level.nodes[keep] = TddNodeData::multi_extended(ext_idx as u32);
        }
    } else {
        level.nodes[keep] = level.encode_multi(new_start, new_len);
    }
}

/// Compact a non-marginal (explicit) level in-place after twin contraction.
///
/// Walks `level.nodes`, keeping only entries whose `merge_target[read] == read`
/// (i.e. canonical survivors; absorbed twins are skipped). Survivors are shifted
/// left in-place via `swap` and the vec is truncated. Mirrors the prune-side
/// compaction in `prune.rs:154` for the non-marginal case.
///
/// Also the accounting point for absorbed twins' pair ranges: dropping the node
/// is what makes its range unreferenced (whether the merge copied the content to
/// the survivor's tail range, or a dup-redirect left the pair list untouched).
#[inline(always)]
pub(super) fn compact_explicit_level(level: &mut TddLevel, merge_target: &[u32]) {
    let n = level.nodes.len();
    let mut write = 0usize;
    // Accumulated and noted once after the walk: the counter's only reader
    // (`compact_pairs_if_stale`) runs later, so per-drop adds bought nothing.
    let mut dead_acc = 0usize;
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

