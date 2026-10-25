//! Twin groups by refining one partition of the child level, parent node by
//! parent node.

use crate::diagram::{ChildPair, ChildSide, TddLevel};
use crate::limits::{Limits, OperationError};

use super::super::scratch::{ClassInfo, ContractScratch};
use crate::sort::Radix;
use super::{mix64, pack, split_pair};

/// The class of a node known to have no twin.
const ALONE: u32 = u32::MAX;

/// The bit of a class's `dest` that marks it, in a split by codes, as
/// meeting two codes or more, above where its row starts.
const MIXED: u32 = 1 << 31;

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
    column: &'a mut Vec<u32>,
    reach: &'a mut Vec<u32>,
    touched: &'a mut Vec<u32>,
    bucket: &'a mut Vec<u32>,
    met: &'a mut Vec<u32>,
    seen: &'a mut Vec<u32>,
    code: &'a mut [u8],
    rows: &'a mut Vec<u32>,
}

/// The most pairs a parent node holds where its nodes are grouped on the
/// stack, by comparing every two; a wider one sorts them in scratch.
pub(super) const SMALL_PAIRS: usize = 8;

/// The nodes a refinement starts with in its first class
/// ([`refine_context_groups`]).
pub(super) enum Start<'a> {
    /// Every node of the child level.
    Whole,
    /// These nodes, ascending, which hold every node with a twin
    /// ([`super::listed`]).
    Listed(&'a [u32]),
}

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
/// no context is a node nothing reaches and has no twin. A refinement from
/// [`Start::Whole`] lists them in `scratch.unnamed` where there are two or
/// more, for the merge to drop.
///
/// The work is one read of the class of each node a parent node names, and
/// one write for each node that changes class, with each parent node's pairs
/// sorted in cache; a node in the first class, or without a twin, is read
/// off two bits. A wide parent node is split without a sort: sibling by
/// sibling ([`Partition::split_by_siblings`]), or by degree where its pairs
/// come in node order and its nodes' sibling sets are nested
/// ([`Partition::split_by_degree`]). Both count sets: no parent node may
/// repeat a pair, as none does in a diagram without a marginal level. No
/// fingerprint is formed and nothing is verified after: the classes are
/// exact.
///
/// Where `start` lists nodes that hold every node with a twin, every other
/// node is out of the partition from the start, as a node found to have no
/// twin is, and the groups are those among the listed nodes.
pub(super) fn refine_context_groups(
    level: &TddLevel,
    side: ChildSide,
    lim: &Limits,
    scratch: &mut ContractScratch,
    width: usize,
    start: Start<'_>,
) -> Result<bool, OperationError> {
    let ContractScratch {
        twin_class, twin_bits, classes, refine_keys, refine_runs, refine_links, refine_table,
        refine_wide, refine_radix, refine_column, refine_reach, refine_touched, refine_bucket, refine_met,
        refine_seen, refine_code, refine_rows, flat_groups, group_starts, unnamed, ..
    } = scratch;
    // Zero between parent nodes, so it only grows.
    lim.try_resize(refine_code, width, 0u8)?;
    let mut wide = Wide {
        keys: refine_keys,
        runs: refine_runs,
        links: refine_links,
        table: refine_table,
        radix: refine_radix,
        column: refine_column,
        reach: refine_reach,
        touched: refine_touched,
        bucket: refine_bucket,
        met: refine_met,
        seen: refine_seen,
        code: &mut refine_code[..width],
        rows: refine_rows,
    };
    flat_groups.clear();
    group_starts.clear();
    let words = width.div_ceil(64);
    lim.try_resize(twin_class, width, 0u32)?;
    lim.try_resize(twin_bits, words, [0u64; 2])?;
    let bits = &mut twin_bits[..words];
    // The first class: every node, or the listed ones.
    let whole = matches!(start, Start::Whole);
    let first = match start {
        Start::Whole => {
            bits.fill([0; 2]);
            width
        }
        Start::Listed(listed) => {
            bits.fill([u64::MAX; 2]);
            for &t in listed {
                let word = &mut bits[(t / 64) as usize];
                word[0] &= !(1 << (t % 64));
                word[1] &= !(1 << (t % 64));
            }
            listed.len()
        }
    };
    // The classes made are nested sets of two or more nodes each, so there
    // are fewer than `width` besides the first.
    classes.clear();
    lim.reserve_exact(classes, width + 1)?;
    classes.push(ClassInfo::new(first as u32));
    let twinable = if first >= 2 { first } else { 0 };
    let mut part = Partition { class: &mut twin_class[..width], bits, classes, twinable };
    // A stored level's wide parent nodes are refined after the narrow ones,
    // narrowest first by the power of two of their widths, and in arena
    // order within one, so that their pairs are read forward: by then most
    // nodes have left the partition, and a wide node sorts only the pairs
    // of the nodes still in it. The refinement stops once no node can have
    // a twin.
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
            n if stored.is_some() => lim.try_push(refine_wide, (u64::from(usize::BITS - n.leading_zeros()) << 32) | i as u64)?,
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
    part.drop_unnamed(width, whole.then_some(unnamed), lim)?;
    debug_assert_eq!(
        part.twinable,
        part.classes.iter().filter(|k| k.size >= 2).map(|k| k.size as usize).sum::<usize>(),
        "the count of nodes that can have a twin"
    );
    if part.twinable == 0 {
        return Ok(false);
    }
    let Partition { class, bits, classes, twinable } = part;
    // The classes of two or more, in the order of their lowest members,
    // each listed ascending: one pass over the nodes not alone, a word of
    // bits at a time, so the nodes without a twin are skipped 64 at once.
    lim.try_resize(flat_groups, twinable, 0u32)?;
    lim.reserve_exact(group_starts, twinable / 2)?;
    let mut next = 0u32;
    for (w, &[moved, alone]) in bits.iter().enumerate() {
        let mut live = !alone;
        if (w + 1) * 64 > width {
            live &= (1u64 << (width % 64)) - 1;
        }
        while live != 0 {
            let b = live.trailing_zeros() as usize;
            live &= live - 1;
            let t = w * 64 + b;
            let c = if (moved >> b) & 1 == 0 { 0 } else { class[t] as usize };
            let k = &mut classes[c];
            if k.size < 2 {
                continue;
            }
            if k.pos == u32::MAX {
                k.pos = next;
                group_starts.push(next);
                next += k.size;
            }
            flat_groups[k.pos as usize] = t as u32;
            k.pos += 1;
        }
    }
    debug_assert_eq!(next as usize, twinable, "every node that can have a twin is listed");
    Ok(true)
}

/// The partition of the child level's nodes. A node is in the first class
/// until its moved bit is set; then it has no twin if its alone bit is set,
/// and is in class `class[t]` if not. The bits stay in cache where the
/// classes would not, so a node that leaves the first class alone, as most
/// do on a level without twins, costs two bits; each word of moved bits
/// sits beside the word of alone bits for the same nodes, in one cache
/// line. A node a parent node names has its moved bit set even where it
/// stays in the first class, so the nodes whose bit is clear are the nodes
/// no pair names.
struct Partition<'a> {
    class: &'a mut [u32],
    /// Per 64 nodes, their moved bits and their alone bits.
    bits: &'a mut [[u64; 2]],
    classes: &'a mut Vec<ClassInfo>,
    /// The nodes in classes of two or more: those that can still have a
    /// twin.
    twinable: usize,
}

impl Partition<'_> {
    /// Node `t`'s class, or [`ALONE`].
    #[inline]
    fn of(&self, t: u32) -> u32 {
        let ([moved, alone], b) = (self.bits[(t / 64) as usize], 1u64 << (t % 64));
        if moved & b == 0 {
            0
        } else if alone & b != 0 {
            ALONE
        } else {
            self.class[t as usize]
        }
    }

    /// Whether a node of class `c` can still have a twin.
    #[inline]
    fn can_twin(&self, c: u32) -> bool {
        c != ALONE && self.classes[c as usize].size >= 2
    }

    /// Take node `t` out of its class `c`: it has no twin, nor has the last
    /// node left in `c`.
    #[inline]
    fn alone(&mut self, t: u32, c: u32) {
        let (w, b) = ((t / 64) as usize, 1u64 << (t % 64));
        let members = self.classes[c as usize].size;
        self.twinable -= match members {
            1 => 0,
            2 => 2,
            _ => 1,
        };
        self.classes[c as usize].size = members - 1;
        let word = &mut self.bits[w];
        word[0] |= b;
        word[1] |= b;
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
        self.bits[(t / 64) as usize][0] |= 1 << (t % 64);
    }

    /// A new class for `members` nodes taken out of class `c`, which the
    /// caller puts there.
    #[inline]
    fn fresh(&mut self, c: u32, members: u32) -> u32 {
        self.classes[c as usize].size -= members;
        if self.classes[c as usize].size == 1 {
            self.twinable -= 1;
        }
        let id = self.classes.len() as u32;
        // Within the reserve made for every class there can be.
        self.classes.push(ClassInfo::new(members));
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
        // A class of these two alone stays as it is, unless it is the first,
        // whose nodes are moved to read as named.
        if self.classes[c0 as usize].size > 2 {
            let id = self.fresh(c0, 2);
            self.put(t0, id);
            self.put(t1, id);
        } else if c0 == 0 {
            self.put(t0, 0);
            self.put(t1, 0);
        }
    }

    /// At the end of the refinement: the nodes no pair names, all of them
    /// in the first class, leave the partition, which they would otherwise
    /// end as one group; where there are two or more, they are put in
    /// `out`, ascending. A refinement that stopped early leaves at most one.
    fn drop_unnamed(&mut self, width: usize, out: Option<&mut Vec<u32>>, lim: &Limits) -> Result<(), OperationError> {
        let words = width.div_ceil(64);
        let mut unnamed = 0;
        for w in 0..words {
            let mut clear = !self.bits[w][0];
            if (w + 1) * 64 > width {
                clear &= (1u64 << (width % 64)) - 1;
            }
            unnamed += clear.count_ones() as usize;
        }
        if unnamed < 2 {
            return Ok(());
        }
        debug_assert_eq!(self.classes[0].size as usize, unnamed, "the first class is the nodes no pair names");
        if let Some(out) = out {
            lim.reserve_exact(out, unnamed)?;
            for (w, word) in self.bits.iter().enumerate() {
                let mut clear = !word[0];
                if (w + 1) * 64 > width {
                    clear &= (1u64 << (width % 64)) - 1;
                }
                while clear != 0 {
                    out.push((w * 64) as u32 + clear.trailing_zeros());
                    clear &= clear - 1;
                }
            }
        }
        for w in 0..words {
            let [moved, alone] = &mut self.bits[w];
            *alone |= !*moved;
            *moved = u64::MAX;
        }
        self.classes[0].size = 0;
        self.twinable -= unnamed;
        Ok(())
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
                keep[r] = u32::from(named[r]) == self.classes[of[r] as usize].size;
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

    /// A parent node of more than [`SMALL_PAIRS`] pairs. One whose pairs
    /// come in node order is first split by degree
    /// ([`Self::split_by_degree`]), which settles it where its nodes'
    /// sibling sets are nested, and one whose pairs do not is split sibling
    /// by sibling ([`Self::split_by_siblings`]). Else the pairs
    /// of its nodes not alone are sorted in `keys`, so that each node's
    /// siblings are a run in ascending order; each run of a node that can
    /// have a twin keyed by its node's class and a hash of its siblings,
    /// and grouped with the runs of its key in a table, by
    /// comparing their siblings. A group of one run leaves the partition; a
    /// group that is its whole class keeps it; any other group is a class
    /// of its own.
    fn split_wide(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        match self.split_by_degree(lim, pairs, side, wide)? {
            Degrees::Split => return Ok(()),
            Degrees::Unordered => return self.split_by_siblings(lim, pairs, side, wide),
            Degrees::Unsettled => {}
        }
        let Wide { keys, runs, links, table, radix, .. } = wide;
        // Each pair of a node not alone a key of its sibling's bits and its
        // node's above them, so that the sort places only the bits there
        // are. Most pairs of a wide node name a node already alone, which
        // its alone bit tells in cache: every pair's key is written, and
        // kept only where the bit is clear, with no branch and no read of
        // the node's class.
        keys.clear();
        lim.try_resize(keys, pairs.len(), 0u64)?;
        let (mut kept, mut top) = (0usize, 0u32);
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            let live = !(self.bits[(t / 64) as usize][1] >> (t % 64)) & 1;
            keys[kept] = pack(t, s);
            kept += live as usize;
            top |= s & 0u32.wrapping_sub(live as u32);
        }
        keys.truncate(kept);
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
            if self.can_twin(c) {
                let h = keys[k..end].iter().fold((end - k) as u64, |h, &e| mix64(h ^ (e & low)));
                runs.push(((u64::from(c) << 32) | (h >> 32), k as u32, end as u32));
            } else if c != ALONE {
                // The last node of its class: alone from here, so that the
                // later parent nodes drop its pairs by the bit.
                self.alone(t, c);
            }
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
    /// with no pair sorted or sibling compared. Where Γ is not zero, a node
    /// whose degree no other node of its class has there leaves the
    /// partition, and the caller splits the rest by their siblings.
    ///
    /// Where the pairs come in node order a node's degree is the length of
    /// its run. A first pass reads only where each run ends, and keeps the
    /// runs of the nodes that can have a twin, whose sibling sets are those
    /// Γ reads: the run of a node already alone costs a compare a pair, and
    /// a node below the one before ends the pass with nothing changed, the
    /// parent node left [`Degrees::Unordered`]. The kept runs' siblings are
    /// then counted. The first sum is read off the degrees' histogram (the
    /// nodes of degree at least v, for each v, taken two at a time), the
    /// second added as each sibling's count rises. `column` and `reach` are
    /// zero between calls: every count written is cleared before the call
    /// returns, and nothing fallible runs while one is held. A parent node
    /// holds fewer than 2^32 pairs, as a run's degree in the low half of its
    /// key needs.
    fn split_by_degree(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<Degrees, OperationError> {
        let Wide { runs, links, table, column, reach, touched, .. } = wide;
        runs.clear();
        lim.reserve_exact(runs, pairs.len())?;
        let (mut last, mut top) = (None, 0u32);
        let mut k = 0;
        while k < pairs.len() {
            let t = split_pair(&pairs[k], side).0;
            if last.is_some_and(|last| t < last) {
                return Ok(Degrees::Unordered);
            }
            last = Some(t);
            let end = k + 1 + pairs[k + 1..].iter().take_while(|pair| split_pair(pair, side).0 == t).count();
            let c = self.of(t);
            if self.can_twin(c) {
                // The run's start in the third field, which the grouping
                // does not read.
                runs.push(((u64::from(c) << 32) | (end - k) as u64, t, k as u32));
                top = pairs[k..end].iter().fold(top, |top, pair| top | split_pair(pair, side).1);
            }
            k = end;
        }
        if runs.is_empty() {
            return Ok(Degrees::Split);
        }
        lim.try_resize(column, top as usize + 1, 0u32)?;
        lim.try_resize(reach, pairs.len() + 1, 0u32)?;
        touched.clear();
        lim.reserve_exact(touched, pairs.len())?;
        let (mut shared, mut most) = (0u64, 0usize);
        for run in runs.iter() {
            let (k, d) = (run.2 as usize, run.0 as u32 as usize);
            reach[d] += 1;
            most = most.max(d);
            for pair in &pairs[k..k + d] {
                let s = split_pair(pair, side).1;
                let e = &mut column[s as usize];
                if *e == 0 {
                    touched.push(s);
                }
                shared += u64::from(*e);
                *e += 1;
            }
        }
        for &s in touched.iter() {
            column[s as usize] = 0;
        }
        let (mut within, mut at_least) = (0u64, 0u64);
        for v in (1..=most).rev() {
            at_least += u64::from(std::mem::take(&mut reach[v]));
            within += at_least * (at_least - 1) / 2;
        }
        debug_assert!(within >= shared, "Γ is a sum of nonnegative terms over sets");
        group_runs(lim, runs, links, table, |_, _| true)?;
        if within == shared {
            self.apply_groups(runs, links, |r| r.1);
            return Ok(Degrees::Split);
        }
        for (r, &run) in runs.iter().enumerate() {
            if links[r].1 == 1 {
                self.alone(run.1, (run.0 >> 32) as u32);
            }
        }
        Ok(Degrees::Unsettled)
    }

    /// A wide parent node split sibling by sibling, where no parent node
    /// repeats a pair: the nodes each sibling is paired with split every
    /// class they meet into its members paired with that sibling and the
    /// rest ([`Self::split_sibling`]). After the last sibling two nodes share
    /// a class exactly when they shared one before and the parent node pairs
    /// them with the same siblings, as a split by sibling sets leaves them,
    /// with no sort, hash or compare. The nodes not alone are first
    /// bucketed by sibling in `bucket`: where the pairs come in sibling
    /// order, each run's as it is read, in one pass that also finds the
    /// order; else in two passes over the pairs that read only siblings,
    /// after that one stops, unless the pairs name at most
    /// [`FEW_SIBLINGS`] siblings and the nodes are split by their codes
    /// instead ([`Self::split_by_codes`]). `column` is zero between calls,
    /// as [`Self::split_by_degree`] reads it.
    fn split_by_siblings(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        let Wide { column, touched, bucket, met, seen, code, rows, .. } = wide;
        // Every class there can be, as `refine_context_groups` reserves.
        lim.try_resize(seen, pairs.len(), 0u32)?;
        met.clear();
        lim.reserve_exact(met, pairs.len())?;
        lim.try_resize(bucket, pairs.len(), 0u32)?;
        touched.clear();
        lim.reserve_exact(touched, pairs.len())?;
        // A pair's node is kept in the bucket where its alone bit is clear:
        // a node already alone is in no split. On a level whose wide parent
        // nodes each set one node apart most pairs name one; on others most
        // name a node not alone, and the branch is as seldom mispredicted.
        // The end of each run of a sibling that keeps a node is pushed to
        // `touched`; a sibling below the one before ends the pass, with
        // nothing split.
        let (mut kept, mut from, mut siblings) = (0usize, 0usize, 1usize);
        let mut sibling = split_pair(&pairs[0], side).1;
        let mut ordered = true;
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            if s != sibling {
                if s < sibling {
                    ordered = false;
                    break;
                }
                if kept > from {
                    touched.push(kept as u32);
                    from = kept;
                }
                sibling = s;
                siblings += 1;
            }
            if (self.bits[(t / 64) as usize][1] >> (t % 64)) & 1 == 0 {
                bucket[kept] = t;
                kept += 1;
            }
        }
        if ordered {
            if kept > from {
                touched.push(kept as u32);
            }
            // Two or more siblings of few that keep a node: their nodes'
            // classes are read once, by code. With no fifth sibling, the
            // split by codes refuses only before it reads a pair.
            if siblings <= FEW_SIBLINGS
                && touched.len() >= 2
                && self.split_by_codes(lim, pairs, side, bucket, seen, met, code, touched, rows)?
            {
                return Ok(());
            }
            let mut from = 0;
            for &to in touched.iter() {
                self.split_sibling(bucket[from..to as usize].iter().copied(), met, seen);
                from = to as usize;
            }
            return Ok(());
        }
        if self.split_by_codes(lim, pairs, side, bucket, seen, met, code, touched, rows)? {
            return Ok(());
        }
        touched.clear();
        // `column` grows to the siblings as they are counted; on a refusal
        // the counts made are cleared first.
        for pair in pairs {
            let s = split_pair(pair, side).1;
            if s as usize >= column.len()
                && let Err(e) = lim.try_resize(column, (s as usize + 1).next_power_of_two(), 0u32)
            {
                for &s in touched.iter() {
                    column[s as usize] = 0;
                }
                return Err(e);
            }
            let e = &mut column[s as usize];
            if *e == 0 {
                touched.push(s);
            }
            *e += 1;
        }
        // Each sibling's count becomes where its bucket starts, and where it
        // ends once filled.
        let mut at = 0u32;
        for &s in touched.iter() {
            let e = &mut column[s as usize];
            (*e, at) = (at, at + *e);
        }
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            let e = &mut column[s as usize];
            bucket[*e as usize] = t;
            *e += 1;
        }
        let mut from = 0;
        for &s in touched.iter() {
            let to = std::mem::take(&mut column[s as usize]) as usize;
            self.split_sibling(bucket[from..to].iter().copied(), met, seen);
            from = to;
        }
        Ok(())
    }

    /// A wide parent node whose pairs name at most [`FEW_SIBLINGS`]
    /// siblings, split in one step by each node's code there: the set of
    /// those siblings it is paired with, one bit each in `code`. Every class
    /// splits into its members of each code and its members the node does
    /// not name, as one split per sibling leaves it, with each named node's
    /// class read once rather than once per sibling. Returns `false`, with
    /// nothing changed, where a fifth sibling shows up or the parent node
    /// holds 2^27 pairs or more.
    ///
    /// The first pass reads each pair's sibling against the siblings met so
    /// far, held in registers, ORs its bit into the node's code and lists
    /// the node at its first pair, in `bucket`. The second takes each listed
    /// node's code, into `codes`, and reads its class: a class met keeps the
    /// first code it meets in `dest` and counts its members in `hits`, and
    /// only a class that meets a second code counts its members by code, in
    /// a row of `rows`. Each class met then gets the class its members of
    /// each code move to, as [`Partition::split_sibling`] gives them, and
    /// where some node moves the last pass moves the nodes.
    #[allow(clippy::too_many_arguments)]
    fn split_by_codes(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        bucket: &mut [u32],
        seen: &mut [u32],
        met: &mut Vec<u32>,
        code: &mut [u8],
        codes: &mut Vec<u32>,
        rows: &mut Vec<u32>,
    ) -> Result<bool, OperationError> {
        // Where every row starts is below `MIXED`.
        if pairs.len() >= 1 << 27 {
            return Ok(false);
        }
        let mut sib = [split_pair(&pairs[0], side).1; FEW_SIBLINGS];
        // The slots in use, one bit each.
        let (mut d, mut used) = (1usize, 1u8);
        let mut listed = 0;
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            let hit = u8::from(s == sib[0])
                | u8::from(s == sib[1]) << 1
                | u8::from(s == sib[2]) << 2
                | u8::from(s == sib[3]) << 3;
            let mut bit = hit & used;
            if bit == 0 {
                if d == FEW_SIBLINGS {
                    for &t in &bucket[..listed] {
                        code[t as usize] = 0;
                    }
                    return Ok(false);
                }
                sib[d] = s;
                bit = 1 << d;
                used |= bit;
                d += 1;
            }
            let e = &mut code[t as usize];
            bucket[listed] = t;
            listed += usize::from(*e == 0);
            *e |= bit;
        }
        let nodes = &bucket[..listed];
        // A class that meets two codes has two members or more here.
        let stride = 1usize << d;
        rows.clear();
        if let Err(e) = lim.reserve_exact(rows, listed / 2 * stride) {
            for &t in nodes {
                code[t as usize] = 0;
            }
            return Err(e);
        }
        // Within the reserve of one entry a pair.
        codes.clear();
        met.clear();
        for (&t, seen) in nodes.iter().zip(seen.iter_mut()) {
            let e = u32::from(std::mem::take(&mut code[t as usize]));
            codes.push(e);
            let c = self.of(t);
            *seen = ALONE;
            if !self.can_twin(c) {
                continue;
            }
            *seen = c;
            let k = &mut self.classes[c as usize];
            if k.hits == 0 {
                met.push(c);
                k.dest = e;
            } else if k.dest != e {
                if k.dest & MIXED == 0 {
                    let row = rows.len();
                    rows.resize(row + stride, 0);
                    rows[row + k.dest as usize] = k.hits;
                    k.dest = MIXED | row as u32;
                }
                rows[(k.dest ^ MIXED) as usize + e as usize] += 1;
            }
            k.hits += 1;
        }
        // A class of one code is split as by one sibling, and keeps its
        // `hits` no more; one of two codes or more moves some member, and
        // keeps in `hits` one past where its row starts.
        let mut moves = false;
        for &c in met.iter() {
            let k = self.classes[c as usize];
            if k.dest & MIXED == 0 {
                let dest = if k.hits == k.size {
                    c
                } else if k.hits == 1 {
                    ALONE
                } else {
                    self.fresh(c, k.hits)
                };
                let k = &mut self.classes[c as usize];
                (k.dest, k.hits) = (dest, 0);
                moves |= dest != c || c == 0;
                continue;
            }
            let row = (k.dest ^ MIXED) as usize;
            self.classes[c as usize].hits = row as u32 + 1;
            let mut keep = k.hits == k.size;
            for cell in &mut rows[row..row + stride] {
                let members = *cell;
                if members == 0 {
                    continue;
                }
                *cell = if members == 1 {
                    ALONE
                } else if keep {
                    keep = false;
                    c
                } else {
                    self.fresh(c, members)
                };
            }
            moves = true;
        }
        if moves {
            for ((&t, &c), &e) in nodes.iter().zip(seen.iter()).zip(codes.iter()) {
                if c == ALONE {
                    continue;
                }
                let k = self.classes[c as usize];
                let dest = if k.hits == 0 { k.dest } else { rows[(k.hits - 1) as usize + e as usize] };
                match dest {
                    ALONE => self.alone(t, c),
                    // A node of the first class it names is moved there
                    // anyway, so that it reads as named.
                    dest if dest != c || c == 0 => self.put(t, dest),
                    _ => {}
                }
            }
        }
        for &c in met.iter() {
            self.classes[c as usize].hits = 0;
        }
        Ok(true)
    }

    /// Split every class by the `nodes` one sibling is paired with: a class
    /// they all are is left whole; one member of a larger class leaves the
    /// partition alone; two or more members of a larger class become a class
    /// of their own, so the classes made still nest and hold two or more
    /// nodes each. Each class's `hits` is zero on entry and on return;
    /// `seen` keeps each node's class as the count read it, [`ALONE`] for a
    /// node that cannot have a twin, so the second pass reads no class.
    /// Where every class met is named whole and none is the first, as when
    /// a node's second context confirms the class its first one made, no
    /// node changes class and the second pass is left out.
    #[inline]
    fn split_sibling(&mut self, nodes: impl Iterator<Item = u32> + Clone, met: &mut Vec<u32>, seen: &mut [u32]) {
        met.clear();
        for (t, seen) in nodes.clone().zip(seen.iter_mut()) {
            let c = self.of(t);
            *seen = ALONE;
            if c == ALONE {
                continue;
            }
            let k = &mut self.classes[c as usize];
            if k.size < 2 {
                continue;
            }
            if k.hits == 0 {
                // Within the reserve of one class a pair.
                met.push(c);
            }
            k.hits += 1;
            *seen = c;
        }
        // Whether some node changes class: where every class met is named
        // whole, and none is the first, the second pass would change
        // nothing.
        let mut moves = false;
        for &c in met.iter() {
            let k = self.classes[c as usize];
            let dest = if k.hits == k.size {
                c
            } else if k.hits == 1 {
                ALONE
            } else {
                self.fresh(c, k.hits)
            };
            self.classes[c as usize].dest = dest;
            moves |= dest != c || c == 0;
        }
        if moves {
            for (t, &c) in nodes.zip(seen.iter()) {
                if c == ALONE {
                    continue;
                }
                match self.classes[c as usize].dest {
                    ALONE => self.alone(t, c),
                    // A node of the first class it names is moved there
                    // anyway, so that it reads as named.
                    d if d != c || c == 0 => self.put(t, d),
                    _ => {}
                }
            }
        }
        for &c in met.iter() {
            self.classes[c as usize].hits = 0;
        }
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
            } else if members < self.classes[c as usize].size || c == 0 {
                let id = if members < self.classes[c as usize].size { self.fresh(c, members) } else { c };
                self.put(node(run), id);
                while next != NO_RUN {
                    self.put(node(runs[next as usize]), id);
                    next = links[next as usize].0;
                }
            }
        }
    }
}

/// What a degree split of a wide parent node did
/// ([`Partition::split_by_degree`]).
enum Degrees {
    /// The parent node is split.
    Split,
    /// Its pairs come in node order, but its nodes' sibling sets are not
    /// nested: the nodes of a degree no other node of their class has there
    /// are out of the partition, and the rest are to be split by siblings.
    Unsettled,
    /// Its pairs do not come in node order, and nothing is changed.
    Unordered,
}

/// The most siblings a wide parent node's pairs name for its nodes to be
/// split by their codes ([`Partition::split_by_codes`]).
const FEW_SIBLINGS: usize = 4;

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
