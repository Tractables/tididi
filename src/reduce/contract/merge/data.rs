//! Unioning twin pair lists into the survivor and compacting the level, in
//! one rebuild of the level's arenas.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};
use crate::diagram::{ChildPair, EncodedNode, NodeIdx, NodeKind, PairRange, RangeTable, Tdd, TddLevel};

use super::super::scratch::{DuplicateScratch, MergeBuffers, MergeRemap};
use super::plan::{GroupAction, GroupPlan};
use super::rewrite::{survivor, DISCARDED, DROPPED};

/// The arenas the merged level is rebuilt into, had before the merge
/// changes anything ([`remap_and_reserve`]).
pub(super) struct RebuiltArenas {
    pairs: Vec<ChildPair>,
    ranges: RangeTable,
}

/// The first arena offset or length the packed node encoding cannot hold;
/// see [`TddLevel::encode_multi`].
const PACKED_LIMIT: usize = 1 << 31;

/// How many members ahead of the one the rebuild copies it asks the cache
/// for a member's node word, and how many for the member's pairs: the
/// members of a level's groups are scattered over it, and the pairs'
/// address is read from the word, so each copy would wait on two misses in
/// turn.
const NODE_AHEAD: usize = 16;
const PAIRS_AHEAD: usize = 8;

/// Ask the cache for the line holding `at`. A hint only: a prefetch reads
/// nothing the program sees and never faults, so any address is sound. A
/// no-op off `x86_64` and under Miri, which lacks the intrinsic.
#[inline(always)]
fn prefetch_line<T>(at: *const T) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        // Sound whatever the address: a prefetch reads nothing the program
        // sees and never faults.
        unsafe { core::arch::x86_64::_mm_prefetch(at.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0) };
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = at;
}

/// Ask the cache for the node word of the member [`NODE_AHEAD`] places
/// after `sel[k]`, and for the first pairs of the one [`PAIRS_AHEAD`]
/// after it, whose word an earlier call asked for.
#[inline(always)]
fn prefetch_members(sel: &[u32], k: usize, nodes: &[EncodedNode], old: &[ChildPair]) {
    if let Some(&m) = sel.get(k + NODE_AHEAD) {
        prefetch_line(nodes.as_ptr().wrapping_add(m as usize));
    }
    if let Some(&m) = sel.get(k + PAIRS_AHEAD)
        && let Some(NodeKind::Multi { start, .. }) = nodes.get(m as usize).map(|n| n.kind())
    {
        prefetch_line(old.as_ptr().wrapping_add(start as usize));
    }
}

/// Whether node `i` is merged away into a survivor.
#[inline]
fn absorbed(bits: &[u64], i: usize) -> bool {
    (bits[i / 64] >> (i % 64)) & 1 != 0
}

/// In one pass in node order: number the survivors in `remap.final_remap`
/// and point each absorbed member's entry at its survivor's new number,
/// keeping its [`DROPPED`] bit; and reserve the arenas of the level `t1`
/// will be after the committed merge: exactly its pairs, every survivor's
/// own and each concat group's members' at its survivor, and a range entry
/// for each node whose range does not fit the packed encoding. Room past
/// the old arena's capacity is charged, as growth of that arena would be;
/// the old arena is freed when the rebuild swaps the new one in.
///
/// `remap` is as [`mark_group_members`](super::mark_group_members) leaves
/// it, and `bufs.group_plans` are sorted by survivor
/// ([`plan_groups`](super::plan::plan_groups)). A member follows its
/// survivor, the lowest of its group, so the survivor is numbered first.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation is refused; only
/// the scratch has changed at that point.
pub(super) fn remap_and_reserve(
    lim: &Limits,
    level: &TddLevel,
    remap: &mut MergeRemap,
    bufs: &MergeBuffers,
) -> Result<RebuiltArenas, OperationError> {
    // A survivor keeps its own range, an absorbed concat member's range
    // moves to its survivor, and an inline node takes an arena slot only
    // when a concat group unions it.
    let MergeRemap { final_remap, absorbed: bits, .. } = remap;
    let mut plans = concat_plans(bufs);
    let (mut total, mut next) = (0usize, 0u32);
    for i in 0..level.nodes().len() {
        let unioned = if absorbed(bits, i) {
            let to = final_remap[i].0;
            if to == DISCARDED {
                // A node no pair names: dropped with its pairs.
                continue;
            }
            let keep = (to & !DROPPED) as usize;
            debug_assert!(keep < i, "a group's survivor is its lowest member");
            final_remap[i] = NodeIdx(final_remap[keep].0 | (to & DROPPED));
            if to & DROPPED == 0 {
                // A content-equal redirect: its pairs are dropped.
                continue;
            }
            true
        } else {
            final_remap[i] = NodeIdx(next);
            next += 1;
            plans.next_if(|p| bufs.sel[p.start as usize] == i as u32).is_some()
        };
        let node = level.node(i);
        total += match level.arena_range(node.kind()) {
            Some(range) => range.len(),
            None => usize::from(unioned),
        };
    }
    let ranged = if total < PACKED_LIMIT { 0 } else { ranged_nodes(level, bits, bufs) };
    let mut pairs = Vec::new();
    lim.reserve_counted_exact(&mut pairs, total)?;
    let grown = total.saturating_sub(level.pairs.capacity()) as u64;
    lim.charge_bytes(grown.saturating_mul(std::mem::size_of::<ChildPair>() as u64))?;
    let mut ranges = RangeTable::default();
    if ranged > 0 {
        ranges.reserve_exact(lim, ranged)?;
    }
    Ok(RebuiltArenas { pairs, ranges })
}

/// The concat plans, in survivor order.
fn concat_plans(bufs: &MergeBuffers) -> std::iter::Peekable<impl Iterator<Item = &GroupPlan>> {
    bufs.group_plans.iter().filter(|p| p.action == GroupAction::Concat).peekable()
}

/// How many nodes of the rebuilt level need a range entry: those whose start
/// or length reaches [`PACKED_LIMIT`]. Read only for an arena of 2^31 pairs
/// or more, where it lays the level out as [`rebuild_level`] will.
#[cold]
fn ranged_nodes(level: &TddLevel, bits: &[u64], bufs: &MergeBuffers) -> usize {
    let mut plans = concat_plans(bufs);
    let (mut start, mut ranged) = (0usize, 0usize);
    for i in 0..level.nodes().len() {
        if absorbed(bits, i) {
            continue;
        }
        let len = match plans.next_if(|p| bufs.sel[p.start as usize] == i as u32) {
            Some(p) => bufs.sel[p.start as usize..p.end as usize].iter().map(|&m| level.pair_count_at(m as usize)).sum(),
            None => match level.arena_range(level.node(i).kind()) {
                Some(range) => range.len(),
                None => continue,
            },
        };
        if len == 1 {
            continue;
        }
        if start >= PACKED_LIMIT || len >= PACKED_LIMIT {
            ranged += 1;
        }
        start += len;
    }
    ranged
}

/// Rebuild the merged level `level` into `arenas` in one pass in node order:
/// drop the absorbed members, union each concat group's members' pairs into
/// its survivor, copy every other survivor's pairs, and compact the nodes in
/// place. The level is left with no dead arena, its pairs in node order.
///
/// Members of a group ascend from its survivor, so the in-place node
/// compaction, which writes at or below the node it reads, has not yet
/// overwritten any member when its survivor is rebuilt.
///
/// Pair lists are unordered sets and the twin search compares contexts,
/// never pair order, so concatenation is the union with no sort. A repeated
/// `(L, R)` entry across the members is a legitimate multiset entry at a
/// marginal-child level and is kept; where `check_disjoint` says no pair may
/// repeat (no marginal level in the diagram and no fork-down after), debug
/// builds check that the supports are disjoint (determinism, invariant 1).
pub(super) fn rebuild_level(
    level: &mut TddLevel,
    arenas: RebuiltArenas,
    remap: &MergeRemap,
    bufs: &MergeBuffers,
    check_disjoint: bool,
) {
    let RebuiltArenas { pairs: mut fresh, ranges: mut fresh_ranges } = arenas;
    let reserved = fresh.capacity();
    let mut plans = concat_plans(bufs);
    let TddLevel { nodes, pairs, ranges, .. } = &mut *level;
    let old = pairs.stored().expect("a merged level is stored");
    let old_ranges: &RangeTable = ranges;
    let source = |node: EncodedNode| -> std::ops::Range<usize> {
        match node.kind() {
            NodeKind::Multi { start, len } => start as usize..(start + len) as usize,
            NodeKind::MultiRanged(idx) => {
                let e = &old_ranges[idx as usize];
                e.start as usize..(e.start + e.len) as usize
            }
            NodeKind::Inline(_) => unreachable!("an inline node owns no range"),
        }
    };
    // A node of `len` pairs at `start` in the rebuilt arena.
    let mut place = |start: usize, len: usize| -> EncodedNode {
        if start < PACKED_LIMIT && len < PACKED_LIMIT {
            return EncodedNode::multi_pair(start as u32, len as u32);
        }
        let idx = fresh_ranges.len();
        debug_assert!(idx < fresh_ranges.capacity(), "ranged entries reserved");
        fresh_ranges.push(PairRange { start: start as u64, len: len as u64 });
        EncodedNode::multi_ranged(idx as u32)
    };
    let nodes = nodes.stored_mut();
    let mut write = 0usize;
    // A survivor outside every group keeps its pairs as they are. The
    // ranges of such survivors that follow one another in the old arena
    // are one run, copied whole when the next range does not continue it,
    // each node placed at its offset in the run.
    let mut run = 0..0usize;
    for read in 0..nodes.len() {
        if absorbed(&remap.absorbed, read) {
            continue;
        }
        let node = nodes[read];
        let rebuilt = match plans.next_if(|p| bufs.sel[p.start as usize] == read as u32) {
            Some(p) => {
                fresh.extend_from_slice(&old[run.clone()]);
                run = run.end..run.end;
                let start = fresh.len();
                for k in p.start as usize..p.end as usize {
                    prefetch_members(&bufs.sel, k, nodes, old);
                    let m = bufs.sel[k];
                    debug_assert!(m as usize >= read, "a group's survivor is its lowest member");
                    let member = nodes[m as usize];
                    match member.kind() {
                        NodeKind::Inline(pair) => fresh.push(pair),
                        _ => fresh.extend_from_slice(&old[source(member)]),
                    }
                }
                if cfg!(debug_assertions) && check_disjoint {
                    let mut chk = fresh[start..].to_vec();
                    chk.sort_unstable();
                    assert!(
                        chk.windows(2).all(|w| w[0] != w[1]),
                        "twin contraction: duplicate pair across twin supports — invariant 1 violation"
                    );
                }
                match fresh.len() - start {
                    1 => EncodedNode::inline(fresh.pop().expect("the one pair")),
                    len => place(start, len),
                }
            }
            None => match node.kind() {
                NodeKind::Inline(_) => node,
                _ => {
                    let src = source(node);
                    if src.start != run.end {
                        fresh.extend_from_slice(&old[run.clone()]);
                        run = src.start..src.start;
                    }
                    let start = fresh.len() + (src.start - run.start);
                    run.end = src.end;
                    place(start, src.len())
                }
            },
        };
        nodes[write] = rebuilt;
        write += 1;
    }
    fresh.extend_from_slice(&old[run]);
    debug_assert!(plans.next().is_none(), "every concat group met its survivor");
    debug_assert_eq!(fresh.capacity(), reserved, "the rebuild fits its reserved arena");
    nodes.truncate(write);
    *pairs.stored_mut() = fresh;
    *ranges = fresh_ranges;
    level.dead_pairs = 0;
    level.shrink_arrays();
}

/// Point the output at its survivor if it is at `t1`, then run fork-down
/// duplicate resolution on the survivors merged via the concat-all path.
/// Fork-down runs after the rebuild so survivor indices are final; it folds
/// a run of k equal pairs into one pair with its marginal side scaled by k
/// only where t1 has a marginal child, and otherwise leaves the run (see the
/// module doc of `duplicate_pair`).
pub(super) fn fork_down(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    resolve_keeps: &[u32],
    remap: &MergeRemap,
    duplicate: &mut DuplicateScratch,
) -> Result<(), OperationError> {
    if tdd.output.vtree == t1 {
        tdd.output.local = survivor(&remap.final_remap, tdd.output.local.idx());
    }
    // One scratch for the whole loop, cleared per node inside the callee.
    for &old_keep in resolve_keeps {
        let new_idx = remap.final_remap[old_keep as usize].idx();
        super::super::duplicate_pair::resolve_duplicate_pairs_in_node(eng, tdd, t1, new_idx, duplicate)?;
    }
    Ok(())
}
