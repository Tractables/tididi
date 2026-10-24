//! Twin groups by refining one partition of the child level, parent node by
//! parent node.

use crate::diagram::{ChildPair, ChildSide, TddLevel};
use crate::limits::{Limits, OperationError};

use super::super::scratch::ContractScratch;
use crate::sort::Radix;
use super::{mix64, pack, split_pair};

/// The class of a node known to have no twin.
const ALONE: u32 = u32::MAX;

/// An empty cell of the table a wide parent node's runs are grouped in,
/// and the end of a group's chain of runs.
const NO_RUN: u32 = u32::MAX;

/// The scratch a wide parent node is split in ([`Partition::split_wide`]).
struct Wide<'a> {
    keys: &'a mut Vec<u64>,
    runs: &'a mut Vec<(u64, u32, u32)>,
    links: &'a mut Vec<(u32, u32)>,
    table: &'a mut Vec<u32>,
    radix: &'a mut Radix,
    degree: &'a mut Vec<u64>,
    column: &'a mut Vec<u64>,
    reach: &'a mut Vec<u32>,
    rows: &'a mut Vec<u32>,
    stamp: &'a mut u32,
    /// No parent node repeats a pair: the counts of
    /// [`Partition::split_by_degree`] are of sets.
    simple: bool,
}

/// The low half of a stamped count, the count.
const COUNT: u64 = 0xFFFF_FFFF;

/// The most pairs a parent node holds where its nodes are grouped on the
/// stack, by comparing every two; a wider one sorts them in scratch.
const SMALL_PAIRS: usize = 8;

/// The twin groups of the `width` nodes the pairs of `level` name on side
/// `side`, written to `scratch.flat_groups` and `scratch.group_starts` as
/// [`super::group_twins_by_entries`] writes them; returns whether there is
/// one.
///
/// Two nodes are twins when every parent node pairs them with the same
/// siblings. The nodes start in one class, and each parent node in turn
/// splits every class by the siblings it pairs each member with: after
/// parent node `p`, two nodes share a class exactly when parent nodes
/// `0..=p` pair them alike, so the classes left after the last are the twin
/// groups. A node alone in its group at some parent node has no twin (math
/// §2.8 is the case of a sibling no other node has there) and leaves the
/// partition, so that later parent nodes skip it. A class whose members a
/// parent node all names, in one group, stays whole; else each of its
/// groups of two or more there becomes a class of its own, one of which
/// may keep the class's id where every member is named there. The nodes no
/// pair names stay in the first class, and leave it at the end: a node of
/// no context is a node nothing reaches, whose merge the prune makes moot,
/// and it has no twin.
///
/// The work is one read of the class of each node a parent node names, and
/// one write for each node that changes class, with each parent node's pairs
/// sorted in cache; a node in the first class, or without a twin, is read
/// off two bits. With `simple`, no parent node repeats a pair, and a wide
/// parent node whose nodes' sibling sets are nested is split by degree,
/// without a sort ([`Partition::split_by_degree`]). No fingerprint is formed
/// and nothing is verified after: the classes are exact. A pair repeated
/// within a parent node counts twice, so the classes are those of context
/// multisets.
pub(super) fn refine_context_groups(
    level: &TddLevel,
    side: ChildSide,
    lim: &Limits,
    scratch: &mut ContractScratch,
    width: usize,
    simple: bool,
) -> Result<bool, OperationError> {
    let ContractScratch {
        twin_class, twin_named, twin_alone, class_size, class_pos, refine_keys, refine_runs, refine_links, refine_table,
        refine_wide, refine_radix, refine_degree, refine_column, refine_reach, refine_rows, refine_stamp, flat_groups,
        group_starts, ..
    } = scratch;
    let mut wide = Wide {
        keys: refine_keys,
        runs: refine_runs,
        links: refine_links,
        table: refine_table,
        radix: refine_radix,
        degree: refine_degree,
        column: refine_column,
        reach: refine_reach,
        rows: refine_rows,
        stamp: refine_stamp,
        simple,
    };
    flat_groups.clear();
    group_starts.clear();
    let words = width.div_ceil(64);
    lim.try_resize(twin_class, width, 0u32)?;
    lim.try_resize(twin_named, words, 0u64)?;
    lim.try_resize(twin_alone, words, 0u64)?;
    let (moved, alone) = (&mut twin_named[..words], &mut twin_alone[..words]);
    moved.fill(0);
    alone.fill(0);
    // The classes made are nested sets of two or more nodes each, so there
    // are fewer than `width` besides the first.
    class_size.clear();
    lim.reserve_exact(class_size, width + 1)?;
    class_size.push(width as u32);
    let twinable = if width >= 2 { width } else { 0 };
    let mut part = Partition { class: &mut twin_class[..width], moved, alone, size: class_size, twinable };
    // A stored level's wide parent nodes are refined after the narrow ones,
    // narrowest first: by then most nodes have left the partition, and a
    // wide node sorts only the pairs of the nodes still in it. The
    // refinement stops once no node can have a twin.
    let stored = level.stored();
    refine_wide.clear();
    let (mut buf, mut cursor) = (Vec::new(), None);
    for i in 0..level.node_count() {
        if part.twinable == 0 {
            break;
        }
        let pairs = match stored {
            Some(stored) => stored.of_idx(i),
            None => level.described_read_next(&mut cursor, i, &mut buf),
        };
        match pairs.len() {
            0 => {}
            1 => part.leave(split_pair(&pairs[0], side).0),
            2 => part.split_two(pairs, side),
            n if n <= SMALL_PAIRS => part.split_small(pairs, side),
            n if stored.is_some() => lim.try_push(refine_wide, ((n.min(u32::MAX as usize) as u64) << 32) | i as u64)?,
            _ => part.split_wide(lim, pairs, side, &mut wide)?,
        }
    }
    if let Some(stored) = stored {
        refine_wide.sort_unstable();
        for &w in refine_wide.iter() {
            if part.twinable == 0 {
                break;
            }
            part.split_wide(lim, stored.of_idx(w as u32 as usize), side, &mut wide)?;
        }
    }
    part.drop_unnamed(width);
    debug_assert_eq!(
        part.twinable,
        part.size.iter().filter(|&&members| members >= 2).map(|&members| members as usize).sum::<usize>(),
        "the count of nodes that can have a twin"
    );
    if part.twinable == 0 {
        return Ok(false);
    }
    let Partition { class, moved, alone, size, twinable } = part;
    // The classes of two or more, in the order of their lowest members,
    // each listed ascending: one pass over the nodes not alone, a word of
    // bits at a time, so the nodes without a twin are skipped 64 at once.
    lim.try_resize(class_pos, size.len(), 0u32)?;
    let pos = &mut class_pos[..size.len()];
    pos.fill(u32::MAX);
    lim.try_resize(flat_groups, twinable, 0u32)?;
    lim.reserve_exact(group_starts, twinable / 2)?;
    let mut next = 0u32;
    for (w, (&moved, &alone)) in moved.iter().zip(alone.iter()).enumerate() {
        let mut live = !alone;
        if (w + 1) * 64 > width {
            live &= (1u64 << (width % 64)) - 1;
        }
        while live != 0 {
            let b = live.trailing_zeros() as usize;
            live &= live - 1;
            let t = w * 64 + b;
            let c = if (moved >> b) & 1 == 0 { 0 } else { class[t] as usize };
            if size[c] < 2 {
                continue;
            }
            if pos[c] == u32::MAX {
                pos[c] = next;
                group_starts.push(next);
                next += size[c];
            }
            flat_groups[pos[c] as usize] = t as u32;
            pos[c] += 1;
        }
    }
    debug_assert_eq!(next as usize, twinable, "every node that can have a twin is listed");
    Ok(true)
}

/// The partition of the child level's nodes. A node is in the first class
/// until its bit in `moved` is set; then it has no twin if its bit in
/// `alone` is set, and is in class `class[t]` if not. The bits stay in
/// cache where the classes would not, so a node that leaves the first class
/// alone, as most do on a level without twins, costs two bits. A node a
/// parent node names has its bit set even where it stays in the first
/// class, so the nodes whose bit is clear are the nodes no pair names.
struct Partition<'a> {
    class: &'a mut [u32],
    moved: &'a mut [u64],
    alone: &'a mut [u64],
    size: &'a mut Vec<u32>,
    /// The nodes in classes of two or more: those that can still have a
    /// twin.
    twinable: usize,
}

impl Partition<'_> {
    /// Node `t`'s class, or [`ALONE`].
    #[inline]
    fn of(&self, t: u32) -> u32 {
        let (w, b) = ((t / 64) as usize, 1u64 << (t % 64));
        if self.moved[w] & b == 0 {
            0
        } else if self.alone[w] & b != 0 {
            ALONE
        } else {
            self.class[t as usize]
        }
    }

    /// Whether a node of class `c` can still have a twin.
    #[inline]
    fn can_twin(&self, c: u32) -> bool {
        c != ALONE && self.size[c as usize] >= 2
    }

    /// Take node `t` out of its class `c`: it has no twin, nor has the last
    /// node left in `c`.
    #[inline]
    fn alone(&mut self, t: u32, c: u32) {
        let (w, b) = ((t / 64) as usize, 1u64 << (t % 64));
        let members = self.size[c as usize];
        self.twinable -= match members {
            1 => 0,
            2 => 2,
            _ => 1,
        };
        self.size[c as usize] = members - 1;
        self.moved[w] |= b;
        self.alone[w] |= b;
    }

    /// Take node `t` out of its class, if it has one.
    #[inline]
    fn leave(&mut self, t: u32) {
        let c = self.of(t);
        if c != ALONE {
            self.alone(t, c);
        }
    }

    /// Put node `t` in class `id`.
    #[inline]
    fn put(&mut self, t: u32, id: u32) {
        self.class[t as usize] = id;
        self.moved[(t / 64) as usize] |= 1 << (t % 64);
    }

    /// A new class for `members` nodes taken out of class `c`, which the
    /// caller puts there.
    #[inline]
    fn fresh(&mut self, c: u32, members: u32) -> u32 {
        self.size[c as usize] -= members;
        if self.size[c as usize] == 1 {
            self.twinable -= 1;
        }
        let id = self.size.len() as u32;
        // Within the reserve made for every class there can be.
        self.size.push(members);
        id
    }

    /// A parent node of two pairs: two nodes of one class with one sibling
    /// each, the same, are a group; every other node is alone in its group.
    #[inline]
    fn split_two(&mut self, pairs: &[ChildPair], side: ChildSide) {
        let ((t0, s0), (t1, s1)) = (split_pair(&pairs[0], side), split_pair(&pairs[1], side));
        let c0 = self.of(t0);
        if t0 == t1 {
            if c0 != ALONE {
                self.alone(t0, c0);
            }
            return;
        }
        let c1 = self.of(t1);
        if s0 != s1 || c0 != c1 || c0 == ALONE {
            if c0 != ALONE {
                self.alone(t0, c0);
            }
            if c1 != ALONE {
                self.alone(t1, c1);
            }
            return;
        }
        let id = if self.size[c0 as usize] > 2 { self.fresh(c0, 2) } else { c0 };
        self.put(t0, id);
        self.put(t1, id);
    }

    /// At the end of the refinement: the nodes no pair names, all of them
    /// in the first class, leave the partition, which they would otherwise
    /// end as one group.
    fn drop_unnamed(&mut self, width: usize) {
        let words = width.div_ceil(64);
        let mut unnamed = 0;
        for w in 0..words {
            let mut clear = !self.moved[w];
            if (w + 1) * 64 > width {
                clear &= (1u64 << (width % 64)) - 1;
            }
            unnamed += clear.count_ones() as usize;
        }
        if unnamed < 2 {
            return;
        }
        debug_assert_eq!(self.size[0] as usize, unnamed, "the first class is the nodes no pair names");
        for w in 0..words {
            self.alone[w] |= !self.moved[w];
            self.moved[w] = u64::MAX;
        }
        self.size[0] = 0;
        self.twinable -= unnamed;
    }

    /// A parent node of three to [`SMALL_PAIRS`] pairs, grouped on the
    /// stack: its pairs sorted, so that each node's siblings are a run in
    /// ascending order, and each run compared with the runs before it.
    #[inline]
    fn split_small(&mut self, pairs: &[ChildPair], side: ChildSide) {
        let n = pairs.len();
        let mut key = [0u64; SMALL_PAIRS];
        for (k, pair) in pairs.iter().enumerate() {
            let (t, s) = split_pair(pair, side);
            key[k] = pack(t, s);
        }
        for k in 1..n {
            let x = key[k];
            let mut j = k;
            while j > 0 && key[j - 1] > x {
                key[j] = key[j - 1];
                j -= 1;
            }
            key[j] = x;
        }
        let key = &key[..n];
        // Run r: node `node[r]`, of class `of[r]`, its pairs
        // `key[from[r]..to[r]]`; the runs of nodes already alone are left out.
        let (mut from, mut to) = ([0u8; SMALL_PAIRS], [0u8; SMALL_PAIRS]);
        let (mut node, mut of) = ([0u32; SMALL_PAIRS], [0u32; SMALL_PAIRS]);
        let mut runs = 0;
        let mut k = 0;
        while k < n {
            let t = (key[k] >> 32) as u32;
            let mut end = k + 1;
            while end < n && (key[end] >> 32) as u32 == t {
                end += 1;
            }
            let c = self.of(t);
            if c != ALONE {
                (from[runs], to[runs], node[runs], of[runs]) = (k as u8, end as u8, t, c);
                runs += 1;
            }
            k = end;
        }
        let siblings = |r: usize| key[from[r] as usize..to[r] as usize].iter().map(|&e| e as u32);
        // `lead[r]`: the first run of r's group, its class and siblings;
        // `first[r]`: the first run of its class.
        let (mut lead, mut first) = ([0u8; SMALL_PAIRS], [0u8; SMALL_PAIRS]);
        let (mut members, mut named) = ([0u8; SMALL_PAIRS], [0u8; SMALL_PAIRS]);
        for r in 0..runs {
            let mut f = r;
            let mut g = r;
            for q in 0..r {
                if of[q] == of[r] {
                    f = f.min(q);
                    if lead[q] as usize == q && to[q] - from[q] == to[r] - from[r] && siblings(q).eq(siblings(r)) {
                        g = q;
                        break;
                    }
                }
            }
            (lead[r], first[r]) = (g as u8, f as u8);
            members[g] += 1;
            named[f] += 1;
        }
        // A class all of whose members are named here keeps its first group
        // of two or more.
        let mut keep = [false; SMALL_PAIRS];
        for r in 0..runs {
            if first[r] as usize == r {
                keep[r] = u32::from(named[r]) == self.size[of[r] as usize];
            }
        }
        for g in 0..runs {
            if lead[g] as usize != g {
                continue;
            }
            let f = first[g] as usize;
            if members[g] == 1 {
                self.alone(node[g], of[g]);
            } else if keep[f] {
                keep[f] = false;
                if of[g] == 0 {
                    for r in g..runs {
                        if lead[r] as usize == g {
                            self.put(node[r], 0);
                        }
                    }
                }
            } else {
                let id = self.fresh(of[g], u32::from(members[g]));
                for r in g..runs {
                    if lead[r] as usize == g {
                        self.put(node[r], id);
                    }
                }
            }
        }
    }

    /// A parent node of more than [`SMALL_PAIRS`] pairs. Where no parent
    /// node repeats a pair it is first split by degree
    /// ([`Self::split_by_degree`]), which settles it where its nodes'
    /// sibling sets are nested. Else its pairs are sorted in `keys`, so that
    /// each node's siblings are a run in ascending order; each run keyed by
    /// its node's class and a hash of its siblings, and grouped with the
    /// runs of its key in a table, by comparing their siblings. A group of
    /// one run leaves the partition; a group that is its whole class keeps
    /// it; any other group is a class of its own.
    fn split_wide(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        if wide.simple && self.split_by_degree(lim, pairs, side, wide)? {
            return Ok(());
        }
        let Wide { keys, runs, links, table, radix, .. } = wide;
        keys.clear();
        lim.reserve_exact(keys, pairs.len())?;
        // Each pair a key of its sibling's bits and its node's above them,
        // so that the sort places only the bits there are.
        let mut top = 0u32;
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            if self.can_twin(self.of(t)) {
                keys.push(pack(t, s));
                top |= s;
            }
        }
        if keys.is_empty() {
            return Ok(());
        }
        let sb = u32::BITS - top.leading_zeros();
        let tb = u32::BITS - (self.class.len() as u32 - 1).leading_zeros();
        let low = (1u64 << sb) - 1;
        for key in keys.iter_mut() {
            *key = ((*key >> 32) << sb) | (*key & low);
        }
        if !keys.is_sorted() {
            radix.sort(lim, keys, 0, (sb + tb) as usize)?;
        }
        runs.clear();
        lim.reserve_exact(runs, keys.len())?;
        let mut k = 0;
        while k < keys.len() {
            let t = (keys[k] >> sb) as u32;
            let end = k + keys[k..].iter().take_while(|&&e| (e >> sb) as u32 == t).count();
            let c = self.of(t);
            let h = keys[k..end].iter().fold((end - k) as u64, |h, &e| mix64(h ^ (e & low)));
            runs.push(((u64::from(c) << 32) | (h >> 32), k as u32, end as u32));
            k = end;
        }
        let keys = &keys[..];
        let siblings = |r: (u64, u32, u32)| keys[r.1 as usize..r.2 as usize].iter().map(|&e| e & low);
        group_runs(lim, runs, links, table, |first, run| {
            first.2 - first.1 == run.2 - run.1 && siblings(first).eq(siblings(run))
        })?;
        self.apply_groups(runs, links, |r| (keys[r.1 as usize] >> sb) as u32);
        Ok(())
    }

    /// A wide parent node split by how many siblings it pairs each node
    /// with, its degree there, where no parent node repeats a pair (math
    /// §4.24). The degrees certify that the nodes' sibling sets there are
    /// nested, totally ordered by inclusion, exactly when Γ, the sum over
    /// every two nodes of the smaller degree less the sum over siblings of
    /// the pairs of nodes each is paired with, is zero; then two nodes of a
    /// degree have the same siblings, and the classes split by degree alone,
    /// with no pair sorted. Both sums are kept as the pairs are read: a
    /// node's degree rising past d adds the nodes already past d, and a
    /// sibling's count rising adds that count. Where they differ, a node
    /// whose degree no other node of its class has there leaves the
    /// partition, and the caller splits the rest by their siblings. Says
    /// whether the parent node is split. Counted only over the nodes that
    /// can have a twin, whose sibling sets are those Γ reads; the counts of
    /// nodes and siblings live under a stamp per parent node, so neither
    /// array is cleared.
    fn split_by_degree(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<bool, OperationError> {
        let Wide { runs, links, table, degree, column, reach, rows, stamp, .. } = wide;
        let stamp = &mut **stamp;
        *stamp = match stamp.checked_add(1) {
            Some(next) => next,
            None => {
                // 0 is the stamp of a count never written.
                degree.fill(0);
                column.fill(0);
                1
            }
        };
        let tag = u64::from(*stamp) << 32;
        lim.try_resize(degree, self.class.len(), 0u64)?;
        lim.try_resize(reach, pairs.len(), 0u32)?;
        rows.clear();
        lim.reserve_exact(rows, pairs.len())?;
        let (mut within, mut shared, mut most) = (0u64, 0u64, 0usize);
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            if !self.can_twin(self.of(t)) {
                continue;
            }
            let cell = &mut degree[t as usize];
            let d = if *cell & !COUNT == tag {
                (*cell & COUNT) as usize
            } else {
                rows.push(t);
                0
            };
            *cell = tag | (d as u64 + 1);
            // `reach[d]`: the nodes of degree past d so far.
            within += u64::from(reach[d]);
            reach[d] += 1;
            most = most.max(d + 1);
            if s as usize >= column.len() {
                lim.try_resize(column, (s as usize + 1).next_power_of_two(), 0u64)?;
            }
            let cell = &mut column[s as usize];
            let e = if *cell & !COUNT == tag { *cell & COUNT } else { 0 };
            *cell = tag | (e + 1);
            shared += e;
        }
        reach[..most].fill(0);
        if rows.is_empty() {
            return Ok(true);
        }
        debug_assert!(within >= shared, "Γ is a sum of nonnegative terms over sets");
        runs.clear();
        lim.reserve_exact(runs, rows.len())?;
        for &t in rows.iter() {
            runs.push(((u64::from(self.of(t)) << 32) | (degree[t as usize] & COUNT), t, 0));
        }
        group_runs(lim, runs, links, table, |_, _| true)?;
        if within == shared {
            self.apply_groups(runs, links, |r| r.1);
            return Ok(true);
        }
        for (r, &run) in runs.iter().enumerate() {
            if links[r].1 == 1 {
                self.alone(run.1, (run.0 >> 32) as u32);
            }
        }
        Ok(false)
    }

    /// Split the classes by the groups [`group_runs`] left in `links`, of
    /// the runs `runs`, each run naming its node through `node` and its
    /// class in the high half of its key: a group of one leaves the
    /// partition, a group that is all that is left of its class keeps the
    /// class's id, and any other group of two or more is a class of its own.
    fn apply_groups(&mut self, runs: &[(u64, u32, u32)], links: &[(u32, u32)], node: impl Fn((u64, u32, u32)) -> u32) {
        for (r, &run) in runs.iter().enumerate() {
            let (mut next, members) = links[r];
            if members == 0 {
                continue;
            }
            let c = (run.0 >> 32) as u32;
            if members == 1 {
                self.alone(node(run), c);
            } else if members < self.size[c as usize] || c == 0 {
                let id = if members < self.size[c as usize] { self.fresh(c, members) } else { c };
                self.put(node(run), id);
                while next != NO_RUN {
                    self.put(node(runs[next as usize]), id);
                    next = links[next as usize].0;
                }
            }
        }
    }
}

/// Group `runs` through `table`: each run joins the group of the first run
/// of its key that `same` also matches, chained after that run in `links`,
/// whose entry counts the group's runs (0 in the entries of the others).
fn group_runs(
    lim: &Limits,
    runs: &[(u64, u32, u32)],
    links: &mut Vec<(u32, u32)>,
    table: &mut Vec<u32>,
    same: impl Fn((u64, u32, u32), (u64, u32, u32)) -> bool,
) -> Result<(), OperationError> {
    let cells = (2 * runs.len()).next_power_of_two();
    table.clear();
    lim.try_resize(table, cells, NO_RUN)?;
    links.clear();
    lim.reserve_exact(links, runs.len())?;
    let shift = u64::BITS - cells.trailing_zeros();
    for (r, &run) in runs.iter().enumerate() {
        links.push((NO_RUN, 1));
        let mut cell = (run.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize;
        loop {
            let lead = table[cell];
            if lead == NO_RUN {
                table[cell] = r as u32;
                break;
            }
            let first = runs[lead as usize];
            if first.0 == run.0 && same(first, run) {
                links[r] = (links[lead as usize].0, 0);
                links[lead as usize].0 = r as u32;
                links[lead as usize].1 += 1;
                break;
            }
            cell = (cell + 1) & (cells - 1);
        }
    }
    Ok(())
}
