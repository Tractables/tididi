//! Unioning twin pair lists into the survivor and compacting the level, in
//! one rebuild of the level's arenas.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};
use crate::diagram::{ChildPair, EncodedNode, NodeKind, PairRange, RangeTable, Tdd, TddLevel};

use super::super::scratch::{DuplicateScratch, MergeBuffers, MergeRemap};
use super::plan::{GroupAction, GroupPlan};

/// The arenas the merged level is rebuilt into, had before the merge
/// changes anything ([`reserve_rebuilt_level`]).
pub(super) struct RebuiltArenas {
    pairs: Vec<ChildPair>,
    ranges: RangeTable,
}

/// The first arena offset or length the packed node encoding cannot hold;
/// see [`TddLevel::encode_multi`].
const PACKED_LIMIT: usize = 1 << 31;

/// Reserve the arenas of the level `t1` will be after the committed merge:
/// exactly its pairs, every survivor's own and each concat group's members'
/// at its survivor, and a range entry for each node whose range does not fit
/// the packed encoding. Room past the old arena's capacity is charged, as
/// growth of that arena would be; the old arena is freed when the rebuild
/// swaps the new one in.
///
/// `remap.merge_target` and `remap.duplicate_redirect` mark the absorbed
/// members, and `bufs.group_plans` are sorted by survivor
/// ([`plan_groups`](super::plan::plan_groups)).
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation is refused; nothing
/// has changed at that point.
pub(super) fn reserve_rebuilt_level(
    lim: &Limits,
    level: &TddLevel,
    remap: &MergeRemap,
    bufs: &MergeBuffers,
) -> Result<RebuiltArenas, OperationError> {
    // One pass in node order over the node words: a survivor keeps its own
    // range, an absorbed concat member's range moves to its survivor, and an
    // inline node takes an arena slot only when a concat group unions it.
    let mut plans = concat_plans(bufs);
    let mut total = 0usize;
    for i in 0..level.nodes().len() {
        let absorbed = remap.merge_target[i] != i as u32;
        let unioned = if absorbed {
            if remap.duplicate_redirect[i] {
                continue;
            }
            true
        } else {
            plans.next_if(|p| bufs.sel[p.start as usize] == i as u32).is_some()
        };
        let node = level.node(i);
        total += match level.arena_range(node.kind()) {
            Some(range) => range.len(),
            None => usize::from(unioned),
        };
    }
    let ranged = if total < PACKED_LIMIT { 0 } else { ranged_nodes(level, remap, bufs) };
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
fn ranged_nodes(level: &TddLevel, remap: &MergeRemap, bufs: &MergeBuffers) -> usize {
    let mut plans = concat_plans(bufs);
    let (mut start, mut ranged) = (0usize, 0usize);
    for i in 0..level.nodes().len() {
        if remap.merge_target[i] != i as u32 {
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
    let mut encode = |fresh: &mut Vec<ChildPair>, start: usize| -> EncodedNode {
        let len = fresh.len() - start;
        if len == 1 {
            return EncodedNode::inline(fresh.pop().expect("the one pair"));
        }
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
    for read in 0..nodes.len() {
        if remap.merge_target[read] != read as u32 {
            continue;
        }
        let node = nodes[read];
        let start = fresh.len();
        let rebuilt = match plans.next_if(|p| bufs.sel[p.start as usize] == read as u32) {
            Some(p) => {
                let members = &bufs.sel[p.start as usize..p.end as usize];
                for &m in members {
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
                encode(&mut fresh, start)
            }
            None => match node.kind() {
                NodeKind::Inline(_) => node,
                _ => {
                    fresh.extend_from_slice(&old[source(node)]);
                    encode(&mut fresh, start)
                }
            },
        };
        nodes[write] = rebuilt;
        write += 1;
    }
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
        tdd.output.local = remap.final_remap[tdd.output.local.idx()];
    }
    // One scratch for the whole loop, cleared per node inside the callee.
    for &old_keep in resolve_keeps {
        let new_idx = remap.final_remap[old_keep as usize].idx();
        super::super::duplicate_pair::resolve_duplicate_pairs_in_node(eng, tdd, t1, new_idx, duplicate)?;
    }
    Ok(())
}
