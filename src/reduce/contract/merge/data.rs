//! Unioning twin pair lists into the survivor and compacting the level, in
//! one rebuild of the level's arenas.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};
use crate::diagram::{ChildPair, EncodedNode, NodeIdx, NodeKind, PairRange, RangeTable, Tdd, TddLevel};

use super::super::scratch::{ClassInfo, DuplicateScratch, Marks, MergeBuffers, MergeRemap};
use super::plan::{GroupAction, GroupPlan};
use super::rewrite::{survivor, DISCARDED, DROPPED};

/// The arenas the merged level is rebuilt into, had before the merge
/// changes anything ([`reserve_rebuilt`]).
pub(super) struct RebuiltArenas {
    pairs: Vec<ChildPair>,
    ranges: RangeTable,
    /// The live pairs of the rebuilt level, which the merge leaves for the
    /// close.
    pub(super) live: usize,
}

/// The pairs of the level `t1` will be after a merge: those of its arena,
/// which the rebuild reserves, and its live pairs, an inline node's one
/// counted.
#[derive(Clone, Copy, Debug)]
pub(super) struct RebuiltPairs {
    arena: usize,
    live: usize,
}

/// The first arena offset or length the packed node encoding cannot hold;
/// see [`TddLevel::encode_multi`].
const PACKED_LIMIT: usize = 1 << 31;

/// How many members ahead of the one the rebuild copies it asks the cache
/// for the member's pairs: the members of a level's groups are scattered
/// over it, and each copy would wait on a miss. Their words are read in
/// order ([`MergeBuffers::words`]).
const PAIRS_AHEAD: usize = 16;

/// The group of a member [`rebuild_in_order`] drops.
const NO_GROUP: u32 = u32::MAX;

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

/// Ask the cache for the first pairs of the member [`PAIRS_AHEAD`] places
/// after the `k`th, off its word in `words`.
#[inline(always)]
fn prefetch_members(words: &[EncodedNode], k: usize, old: &[ChildPair]) {
    if let Some(NodeKind::Multi { start, .. }) = words.get(k + PAIRS_AHEAD).map(|n| n.kind()) {
        prefetch_line(old.as_ptr().wrapping_add(start as usize));
    }
}

/// Append `src` to `fresh`, within its reserved room: four pairs or fewer,
/// as a node or a run the merge copies mostly is, by a copy of that many,
/// written out in place rather than through a call to copy them.
#[inline(always)]
fn append(fresh: &mut Vec<ChildPair>, src: &[ChildPair]) {
    match *src {
        [] => {}
        [a] => fresh.push(a),
        [a, b] => fresh.extend_from_slice(&[a, b]),
        [a, b, c] => fresh.extend_from_slice(&[a, b, c]),
        [a, b, c, d] => fresh.extend_from_slice(&[a, b, c, d]),
        _ => fresh.extend_from_slice(src),
    }
}

/// Write `src` into `room` from its start: four pairs or fewer, as a member
/// the merge copies mostly is, written out in place rather than through a
/// call to copy them. Returns how many it wrote.
#[inline(always)]
fn put(room: &mut [ChildPair], src: &[ChildPair]) -> usize {
    match *src {
        [a] => room[0] = a,
        [a, b] => room[..2].copy_from_slice(&[a, b]),
        [a, b, c] => room[..3].copy_from_slice(&[a, b, c]),
        [a, b, c, d] => room[..4].copy_from_slice(&[a, b, c, d]),
        _ => room[..src.len()].copy_from_slice(src),
    }
    src.len()
}

/// Whether node `i` is merged away into a survivor.
#[inline]
fn absorbed(bits: &[u64], i: usize) -> bool {
    (bits[i / 64] >> (i % 64)) & 1 != 0
}

/// Reserve the arenas of the level `t1` will be after the committed merge:
/// exactly its pairs, every survivor's own and each concat group's
/// members' at its survivor, and a range entry for each node whose range
/// does not fit the packed encoding. Room past the old arena's capacity is
/// charged, as growth of that arena would be; the old arena is freed when
/// the rebuild swaps the new one in.
///
/// The count is the level's arena pairs, less those of the nodes the
/// rebuild drops with their pairs (the unnamed ones and the content-equal
/// redirects), plus one for each inline node a concat group unions: a node
/// of one pair is inline, so the level's arena pairs are its pairs less its
/// nodes of one pair, read off the node words in one branch-free pass. The
/// rest reads only the groups' members and the unnamed nodes. A level with
/// ranged nodes is counted node by node. The live pairs are the level's
/// less those of the nodes dropped with their pairs, a concat group's
/// survivor holding its members' pairs.
///
/// `remap` is as [`mark_group_members`](super::mark_group_members) leaves
/// it, with the unnamed nodes marked [`DISCARDED`]; the survivors are
/// numbered by [`rebuild_level`]. Where the level is rebuilt in node order,
/// `in_order` holds the counts, from [`in_order_pairs`].
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation is refused; only
/// the scratch has changed at that point.
pub(super) fn reserve_rebuilt(
    lim: &Limits,
    level: &TddLevel,
    remap: &MergeRemap,
    bufs: &MergeBuffers,
    unnamed: &[u32],
    in_order: Option<RebuiltPairs>,
) -> Result<RebuiltArenas, OperationError> {
    let RebuiltPairs { arena: total, live } = if let Some(counts) = in_order {
        counts
    } else if level.ranges.is_empty() {
        let (pairs, _, single) = level.pair_census();
        let mut total = (pairs - single) as usize;
        let mut live = pairs as usize;
        for &u in unnamed {
            total -= level.arena_pairs_at(u as usize);
            live -= level.pair_count_at(u as usize);
        }
        for p in &bufs.group_plans {
            let members = &bufs.sel[p.start as usize..p.end as usize];
            match p.action {
                GroupAction::DupRedirect => {
                    for &m in &members[1..] {
                        total -= level.arena_pairs_at(m as usize);
                        live -= level.pair_count_at(m as usize);
                    }
                }
                GroupAction::Concat => {
                    let words = &bufs.words[p.start as usize..p.end as usize];
                    total += words.iter().filter(|w| !w.kind().pairs_in_arena()).count();
                }
            }
        }
        RebuiltPairs { arena: total, live }
    } else {
        rebuilt_pairs(level, remap, bufs)
    };
    let ranged = if total < PACKED_LIMIT { 0 } else { ranged_nodes(level, &remap.absorbed, bufs) };
    let mut pairs = Vec::new();
    lim.reserve_counted_exact(&mut pairs, total)?;
    let grown = total.saturating_sub(level.pairs.capacity()) as u64;
    lim.charge_bytes(grown.saturating_mul(std::mem::size_of::<ChildPair>() as u64))?;
    let mut ranges = RangeTable::default();
    if ranged > 0 {
        ranges.reserve_exact(lim, ranged)?;
    }
    Ok(RebuiltArenas { pairs, ranges, live })
}

/// The pairs of the level `t1` will be after a merge whose search marked
/// the members ([`Marks`]), where [`rebuild_in_order`] can rebuild it: the
/// level has no ranged node, the marks say every group holds two pairs or
/// more, and the rebuilt level fits the packed encoding, so that it has no
/// ranged node either. The counts are the ones [`reserve_rebuilt`] makes,
/// the marks having counted the groups' inline members.
pub(super) fn in_order_pairs(level: &TddLevel, marks: Marks, unnamed: &[u32]) -> Option<RebuiltPairs> {
    if !marks.in_order || !level.ranges.is_empty() {
        return None;
    }
    let (pairs, _, single) = level.pair_census();
    let mut total = (pairs - single) as usize;
    let mut live = pairs as usize;
    for &u in unnamed {
        total -= level.arena_pairs_at(u as usize);
        live -= level.pair_count_at(u as usize);
    }
    total += marks.inline;
    (total < PACKED_LIMIT).then_some(RebuiltPairs { arena: total, live })
}

/// The pairs of the level `t1` will be after the merge, node by node: a
/// survivor keeps its own range, an absorbed concat member's range moves
/// to its survivor, and an inline node takes an arena slot only when a
/// concat group unions it; every node the rebuild keeps or concatenates
/// keeps its pairs live. For a level with ranged nodes.
#[cold]
fn rebuilt_pairs(level: &TddLevel, remap: &MergeRemap, bufs: &MergeBuffers) -> RebuiltPairs {
    let MergeRemap { final_remap, absorbed: bits, .. } = remap;
    let mut plans = concat_plans(bufs);
    let (mut total, mut live) = (0usize, 0usize);
    for (i, (node, to)) in level.nodes().iter().zip(final_remap).enumerate() {
        let unioned = if absorbed(bits, i) {
            // A node no pair names, or a content-equal redirect: dropped
            // with its pairs.
            if to.0 == DISCARDED || to.0 & DROPPED == 0 {
                continue;
            }
            true
        } else {
            plans.next_if(|p| bufs.sel[p.start as usize] == i as u32).is_some()
        };
        let range = level.arena_range(node.kind());
        live += range.as_ref().map_or(1, |range| range.len());
        total += match range {
            Some(range) => range.len(),
            None => usize::from(unioned),
        };
    }
    RebuiltPairs { arena: total, live }
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
/// The same pass numbers the survivors in `remap.final_remap`, in node
/// order, reading only the survivors: the absorbed nodes are skipped a word
/// of bits at a time. A member whose pairs the parent rewrite drops keeps
/// its entry, its survivor's old index under [`DROPPED`], from which
/// [`survivor`] reads the new one; a content-equal redirect's entry is set
/// to its survivor's new number after the pass, and a node no pair names
/// keeps its [`DISCARDED`] entry.
///
/// A group's members are read off their words in `bufs.words`, copied
/// where they were marked, and ascend from its survivor: the in-place node
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
    remap: &mut MergeRemap,
    bufs: &MergeBuffers,
    check_disjoint: bool,
) {
    let RebuiltArenas { pairs: mut fresh, ranges: mut fresh_ranges, .. } = arenas;
    let reserved = fresh.capacity();
    let TddLevel { nodes, pairs, ranges, .. } = &mut *level;
    let old = pairs.stored().expect("a merged level is stored");
    let old_ranges: &RangeTable = ranges;
    // A node's range in the old arena, read off its word once.
    let source = |kind: NodeKind| -> std::ops::Range<usize> {
        match kind {
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
    let MergeRemap { final_remap, absorbed: bits, .. } = remap;
    // The next concat group, in survivor order, and its survivor (its
    // lowest member), or past every node once none is left.
    let mut plans = bufs.group_plans.iter().filter(|p| p.action == GroupAction::Concat);
    let survivor_of = |plan: Option<&GroupPlan>| plan.map_or(usize::MAX, |p| bufs.sel[p.start as usize] as usize);
    let mut plan = plans.next();
    let mut next = survivor_of(plan);
    let len = nodes.len();
    for (w, &word) in bits[..len.div_ceil(64)].iter().enumerate() {
        let base = w * 64;
        let mut live = !word;
        if len - base < 64 {
            live &= (1u64 << (len - base)) - 1;
        }
        while live != 0 {
            let read = base + live.trailing_zeros() as usize;
            live &= live - 1;
            final_remap[read] = NodeIdx(write as u32);
            let node = nodes[read];
            let rebuilt = if read == next {
                let p = plan.expect("a concat group's survivor");
                plan = plans.next();
                next = survivor_of(plan);
                append(&mut fresh, &old[run.clone()]);
                run = run.end..run.end;
                let start = fresh.len();
                for k in p.start as usize..p.end as usize {
                    prefetch_members(&bufs.words, k, old);
                    debug_assert!(bufs.sel[k] as usize >= read, "a group's survivor is its lowest member");
                    debug_assert!(bufs.words[k] == nodes[bufs.sel[k] as usize], "a member's word is its node's");
                    match bufs.words[k].kind() {
                        NodeKind::Inline(pair) => fresh.push(pair),
                        kind => append(&mut fresh, &old[source(kind)]),
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
            } else {
                match node.kind() {
                    NodeKind::Inline(_) => node,
                    kind => {
                        let src = source(kind);
                        if src.start != run.end {
                            append(&mut fresh, &old[run.clone()]);
                            run = src.start..src.start;
                        }
                        let start = fresh.len() + (src.start - run.start);
                        run.end = src.end;
                        place(start, src.len())
                    }
                }
            };
            nodes[write] = rebuilt;
            write += 1;
        }
    }
    fresh.extend_from_slice(&old[run]);
    debug_assert!(plan.is_none(), "every concat group met its survivor");
    // A content-equal redirect keeps its parent pairs, renamed onto its
    // survivor's new number.
    for p in bufs.group_plans.iter().filter(|p| p.action == GroupAction::DupRedirect) {
        let members = &bufs.sel[p.start as usize..p.end as usize];
        let to = final_remap[members[0] as usize];
        for &m in &members[1..] {
            final_remap[m as usize] = to;
        }
    }
    debug_assert_eq!(fresh.capacity(), reserved, "the rebuild fits its reserved arena");
    nodes.truncate(write);
    *pairs.stored_mut() = fresh;
    *ranges = fresh_ranges;
    level.dead_pairs = 0;
    level.shrink_arrays();
}

/// What a search that marked the members leaves for [`rebuild_in_order`]:
/// each node's class; the classes, with each group's pairs at its class's
/// [`ClassInfo::hits`]; the groups as listed; each member's group, in node
/// order; room for one place a group; and whether the merge drops a node
/// no pair names.
pub(super) struct InOrder<'a> {
    pub(super) class: &'a [u32],
    pub(super) classes: &'a [ClassInfo],
    pub(super) group_starts: &'a [u32],
    pub(super) flat_groups: &'a [u32],
    pub(super) member_groups: &'a [u32],
    pub(super) rooms: &'a mut [u32],
    pub(super) unnamed: bool,
}

/// [`rebuild_level`] for a merge whose search marked the members
/// ([`in_order_pairs`] says when): one pass over every node in node order.
/// A group's survivor, its lowest member, takes the room of the group's
/// pairs where it is met, its own first, and each member after it puts
/// its pairs at the group's next place in that room, where it is met. So
/// the old arena is read in order, where [`rebuild_level`] reads each
/// group's members scattered over it; the writes into the groups' rooms
/// are scattered instead. The rebuilt level is the one [`rebuild_level`]
/// builds, its nodes, pairs and order included: a group's members put their
/// pairs in ascending order, the order its listing holds them in.
///
/// A member reads its group off `member_groups`, in order, and the group's
/// next place off `rooms`. The members of 64 nodes are read before the
/// survivors among them are compacted, which writes at or below the node it
/// reads, and put after; the class of the first group's survivor among
/// them, and each member's place in its room, are asked of the cache
/// first.
pub(super) fn rebuild_in_order(
    level: &mut TddLevel,
    arenas: RebuiltArenas,
    remap: &mut MergeRemap,
    groups: InOrder<'_>,
    check_disjoint: bool,
) {
    let RebuiltArenas { pairs: mut fresh, ranges: fresh_ranges, .. } = arenas;
    debug_assert!(fresh_ranges.is_empty() && level.ranges.is_empty(), "a level rebuilt in node order has no ranged node");
    let reserved = fresh.capacity();
    let TddLevel { nodes, pairs, .. } = &mut *level;
    let old = pairs.stored().expect("a merged level is stored");
    let nodes = nodes.stored_mut();
    let InOrder { class, classes, group_starts, flat_groups, member_groups, rooms, unnamed } = groups;
    let mut write = 0usize;
    // As in `rebuild_level`: a run of survivors outside every group whose
    // ranges follow one another in the old arena is copied whole.
    let mut run = 0..0usize;
    let MergeRemap { final_remap, absorbed: bits, .. } = remap;
    // The groups' survivors, ascending, and the next, or past every node.
    let mut keeps = group_starts.iter().map(|&s| flat_groups[s as usize] as usize);
    let mut next = keeps.next().unwrap_or(usize::MAX);
    // The group of the next survivor met, and where the next member's
    // group is listed.
    let (mut group, mut ahead) = (0usize, 0usize);
    // The members of 64 nodes, each with its group, or `NO_GROUP` where
    // the merge drops it.
    let mut held = [(EncodedNode::multi_pair(0, 0), NO_GROUP); 64];
    let len = nodes.len();
    // Slices of the one length, so that a node's index is checked against
    // it once and neither length is read again.
    let (slots, remapped) = (&mut nodes[..], &mut final_remap[..len]);
    for (w, &word) in bits[..len.div_ceil(64)].iter().enumerate() {
        let base = w * 64;
        let valid = if len - base < 64 { (1u64 << (len - base)) - 1 } else { u64::MAX };
        let merged = word & valid;
        if next < base + 64 {
            prefetch_line(classes.as_ptr().wrapping_add(class[next] as usize));
        }
        // Each member's place in its room is asked of the cache as the
        // member is read, to have arrived when the member is put there.
        let mut m = merged;
        let mut h = 0;
        while m != 0 {
            let t = base + m.trailing_zeros() as usize;
            m &= m - 1;
            // A node no pair names is dropped with its pairs.
            let g = if unnamed && remapped[t].0 == DISCARDED {
                NO_GROUP
            } else {
                let g = member_groups[ahead];
                ahead += 1;
                prefetch_line(fresh.as_ptr().wrapping_add(rooms[g as usize] as usize));
                g
            };
            held[h] = (slots[t], g);
            h += 1;
        }
        let mut live = !word & valid;
        loop {
            // A run of the nodes kept that are no group's survivor and
            // whose pairs, if in the arena, follow the run's: each takes
            // its word or its new offset, and nothing is appended, so the
            // loop holds no more than it reads.
            let delta = fresh.len().wrapping_sub(run.start);
            let mut end = run.end;
            while live != 0 {
                let read = base + live.trailing_zeros() as usize;
                if read == next {
                    break;
                }
                let node = slots[read];
                let rebuilt = match node.kind() {
                    NodeKind::Inline(_) => node,
                    NodeKind::Multi { start, len } if start as usize == end => {
                        end += len as usize;
                        EncodedNode::multi_pair((start as usize).wrapping_add(delta) as u32, len)
                    }
                    _ => break,
                };
                live &= live - 1;
                remapped[read] = NodeIdx(write as u32);
                slots[write] = rebuilt;
                write += 1;
            }
            run.end = end;
            if live == 0 {
                break;
            }
            let read = base + live.trailing_zeros() as usize;
            live &= live - 1;
            remapped[read] = NodeIdx(write as u32);
            let node = slots[read];
            let rebuilt = if read == next {
                next = keeps.next().unwrap_or(usize::MAX);
                append(&mut fresh, &old[run.clone()]);
                run = run.end..run.end;
                let start = fresh.len();
                match node.kind() {
                    NodeKind::Inline(pair) => fresh.push(pair),
                    NodeKind::Multi { start, len } => append(&mut fresh, &old[start as usize..(start + len) as usize]),
                    NodeKind::MultiRanged(_) => unreachable!("a level rebuilt in node order has no ranged node"),
                }
                let total = classes[class[read] as usize].hits as usize;
                debug_assert!(total >= 2 && fresh.len() <= start + total, "a group holds two pairs or more");
                rooms[group] = fresh.len() as u32;
                group += 1;
                let filler = fresh[start];
                fresh.resize(start + total, filler);
                EncodedNode::multi_pair(start as u32, total as u32)
            } else {
                match node.kind() {
                    NodeKind::Inline(_) => node,
                    NodeKind::Multi { start, len } => {
                        let src = start as usize..(start + len) as usize;
                        if src.start != run.end {
                            append(&mut fresh, &old[run.clone()]);
                            run = src.start..src.start;
                        }
                        let at = fresh.len() + (src.start - run.start);
                        run.end = src.end;
                        EncodedNode::multi_pair(at as u32, len)
                    }
                    NodeKind::MultiRanged(_) => unreachable!("a level rebuilt in node order has no ranged node"),
                }
            };
            slots[write] = rebuilt;
            write += 1;
        }
        for &(node, g) in &held[..h] {
            if g == NO_GROUP {
                continue;
            }
            let room = &mut rooms[g as usize];
            let at = *room as usize;
            *room += match node.kind() {
                NodeKind::Inline(pair) => {
                    fresh[at] = pair;
                    1
                }
                NodeKind::Multi { start, len } => put(&mut fresh[at..], &old[start as usize..(start + len) as usize]) as u32,
                NodeKind::MultiRanged(_) => unreachable!("a level rebuilt in node order has no ranged node"),
            };
        }
    }
    fresh.extend_from_slice(&old[run]);
    debug_assert_eq!(next, usize::MAX, "every group met its survivor");
    if cfg!(debug_assertions) {
        debug_assert_eq!(ahead, member_groups.len(), "every member met");
        for (g, &s) in group_starts.iter().enumerate() {
            let keep = flat_groups[s as usize];
            let NodeKind::Multi { start, len } = nodes[final_remap[keep as usize].idx()].kind() else {
                unreachable!("a group holds two pairs or more");
            };
            let room = start as usize..(start + len) as usize;
            assert_eq!(rooms[g] as usize, room.end, "every member put its pairs");
            if check_disjoint {
                let mut chk = fresh[room].to_vec();
                chk.sort_unstable();
                assert!(
                    chk.windows(2).all(|w| w[0] != w[1]),
                    "twin contraction: duplicate pair across twin supports — invariant 1 violation"
                );
            }
        }
    }
    debug_assert_eq!(fresh.capacity(), reserved, "the rebuild fits its reserved arena");
    nodes.truncate(write);
    *pairs.stored_mut() = fresh;
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
