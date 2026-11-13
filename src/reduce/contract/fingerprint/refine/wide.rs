//! The splits of one wide parent node.

use crate::diagram::{ChildPair, ChildSide};
use crate::limits::{Limits, OperationError};
use crate::sort::Radix;

use super::super::{mix64, pack, split_pair};
use super::partition::{Partition, ALONE};

/// The bit of a class's `dest` that marks it, in a split by codes, as
/// meeting two codes or more, above where its row starts.
const MIXED: u32 = 1 << 31;

/// An empty cell of the table a wide parent node's runs are grouped in,
/// and the end of a group's chain of runs.
const NO_RUN: u32 = u32::MAX;

/// The scratch a wide parent node is split in ([`Partition::split_wide`]).
pub(super) struct Wide<'a> {
    pub(super) keys: &'a mut Vec<u64>,
    pub(super) runs: &'a mut Vec<(u64, u32, u32)>,
    pub(super) links: &'a mut Vec<(u32, u32)>,
    pub(super) table: &'a mut Vec<u32>,
    pub(super) radix: &'a mut Radix,
    pub(super) column: &'a mut Vec<u32>,
    pub(super) reach: &'a mut Vec<u32>,
    pub(super) touched: &'a mut Vec<u32>,
    pub(super) bucket: &'a mut Vec<u32>,
    pub(super) met: &'a mut Vec<u32>,
    pub(super) seen: &'a mut Vec<u32>,
    pub(super) code: &'a mut [u8],
    pub(super) rows: &'a mut Vec<u32>,
}

impl Partition<'_> {
    /// A parent node of more than [`super::SMALL_PAIRS`] pairs. One whose pairs
    /// come in node order is split by its runs' codes where it names at
    /// most [`FEW_SIBLINGS`] siblings ([`Self::split_by_run_codes`]), and
    /// else first split by degree ([`Self::split_by_degree`]), which settles
    /// it where its nodes' sibling sets are nested; one whose pairs do not
    /// come in node order is split sibling by sibling
    /// ([`Self::split_by_siblings`]). Else the pairs
    /// of its nodes not alone are sorted in `wide.keys`, so that each node's
    /// siblings are a run in ascending order; each run of a node that can
    /// have a twin keyed by its node's class and a hash of its siblings,
    /// and grouped with the runs of its key in a table, by
    /// comparing their siblings. A group of one run leaves the partition; a
    /// group that is its whole class keeps it; any other group is a class
    /// of its own.
    pub(super) fn split_wide(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        match self.split_by_run_codes(lim, pairs, side, wide)? {
            Degrees::Split => return Ok(()),
            Degrees::Unordered => return self.split_by_siblings(lim, pairs, side, wide),
            Degrees::Unsettled | Degrees::Many => {}
        }
        match self.split_by_degree(lim, pairs, side, wide)? {
            Degrees::Split => return Ok(()),
            Degrees::Unordered => return self.split_by_siblings(lim, pairs, side, wide),
            Degrees::Unsettled | Degrees::Many => {}
        }
        let Wide {
            keys,
            runs,
            links,
            table,
            radix,
            ..
        } = wide;
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
            let end = k + keys[k..]
                .iter()
                .take_while(|&&e| (e >> sb) as u32 == t)
                .count();
            let c = self.of(t);
            if self.can_twin(c) {
                let h = keys[k..end]
                    .iter()
                    .fold((end - k) as u64, |h, &e| mix64(h ^ (e & low)));
                runs.push(((u64::from(c) << 32) | (h >> 32), k as u32, end as u32));
            } else if c != ALONE {
                // The last node of its class: alone from here, so that the
                // later parent nodes drop its pairs by the bit.
                self.alone(t, c);
            }
            k = end;
        }
        let keys = &keys[..];
        let siblings =
            |r: (u64, u32, u32)| keys[r.1 as usize..r.2 as usize].iter().map(|&e| e & low);
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
    /// second added as each sibling's count rises. `wide.column` and `wide.reach` are
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
        let Wide {
            runs,
            links,
            table,
            column,
            reach,
            touched,
            ..
        } = wide;
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
            let end = k
                + 1
                + pairs[k + 1..]
                    .iter()
                    .take_while(|pair| split_pair(pair, side).0 == t)
                    .count();
            let c = self.of(t);
            if self.can_twin(c) {
                // The run's start in the third field, which the grouping
                // does not read.
                runs.push(((u64::from(c) << 32) | (end - k) as u64, t, k as u32));
                top = pairs[k..end]
                    .iter()
                    .fold(top, |top, pair| top | split_pair(pair, side).1);
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
        debug_assert!(
            within >= shared,
            "Γ is a sum of nonnegative terms over sets"
        );
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
    /// bucketed by sibling in `wide.bucket`: where the pairs come in sibling
    /// order, each run's as it is read, in one pass that also finds the
    /// order; else in two passes over the pairs after that one stops
    /// ([`Self::split_scattered`]), unless the pairs name at most
    /// [`FEW_SIBLINGS`] siblings and the nodes are split by their codes
    /// instead ([`Self::split_by_codes`]). `wide.column` is zero between calls,
    /// as [`Self::split_by_degree`] reads it.
    fn split_by_siblings(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        // Every class there can be, as `refine_context_groups` reserves.
        lim.try_resize(wide.seen, pairs.len(), 0u32)?;
        wide.met.clear();
        lim.reserve_exact(wide.met, pairs.len())?;
        lim.try_resize(wide.bucket, pairs.len(), 0u32)?;
        wide.touched.clear();
        lim.reserve_exact(wide.touched, pairs.len())?;
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
                    wide.touched.push(kept as u32);
                    from = kept;
                }
                sibling = s;
                siblings += 1;
            }
            if (self.bits[(t / 64) as usize][1] >> (t % 64)) & 1 == 0 {
                wide.bucket[kept] = t;
                kept += 1;
            }
        }
        if ordered {
            if kept > from {
                wide.touched.push(kept as u32);
            }
            // Two or more siblings of few that keep a node: their nodes'
            // classes are read once, by code. With no fifth sibling, the
            // split by codes refuses only before it reads a pair.
            if siblings <= FEW_SIBLINGS
                && wide.touched.len() >= 2
                && self.split_by_codes(lim, pairs, side, wide, false)?
            {
                return Ok(());
            }
            let mut from = 0;
            for &to in wide.touched.iter() {
                self.split_bucket(&wide.bucket[from..to as usize], wide.met, wide.seen);
                from = to as usize;
            }
            return Ok(());
        }
        // As in the split by buckets, the pairs of nodes already alone are
        // left out where most of the level's nodes are alone.
        let skip = self.twinable * 2 < self.class.len();
        if self.split_by_codes(lim, pairs, side, wide, skip)? {
            return Ok(());
        }
        // Where most of the level's nodes are alone, the pairs of those
        // already alone are left out; elsewhere reading every pair's alone
        // bit costs more than the pairs it leaves out.
        if skip {
            self.split_scattered::<true>(lim, pairs, side, wide)
        } else {
            self.split_scattered::<false>(lim, pairs, side, wide)
        }
    }

    /// The nodes of a wide parent node whose pairs come in no order of
    /// their siblings bucketed by sibling in `wide.bucket`, in two passes over
    /// the pairs: one counts each sibling's nodes and one puts them. Each
    /// bucket is then split ([`Self::split_bucket`]). With `SKIP_ALONE`
    /// both passes leave out the pairs of nodes already alone.
    fn split_scattered<const SKIP_ALONE: bool>(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        let Wide {
            column,
            touched,
            bucket,
            met,
            seen,
            ..
        } = wide;
        touched.clear();
        // `column` grows to the siblings as they are counted; on a refusal
        // the counts made are cleared first.
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            if SKIP_ALONE && (self.bits[(t / 64) as usize][1] >> (t % 64)) & 1 != 0 {
                continue;
            }
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
            if SKIP_ALONE && (self.bits[(t / 64) as usize][1] >> (t % 64)) & 1 != 0 {
                continue;
            }
            let e = &mut column[s as usize];
            bucket[*e as usize] = t;
            *e += 1;
        }
        let mut from = 0;
        for &s in touched.iter() {
            let to = std::mem::take(&mut column[s as usize]) as usize;
            self.split_bucket(&bucket[from..to], met, seen);
            from = to;
        }
        Ok(())
    }

    /// A wide parent node whose pairs name at most [`FEW_SIBLINGS`]
    /// siblings, split in one step by each node's code there: the set of
    /// those siblings it is paired with, one bit each in `wide.code`. Every class
    /// splits into its members of each code and its members the node does
    /// not name, as one split per sibling leaves it, with each named node's
    /// class read once rather than once per sibling. Returns `false`, with
    /// nothing changed, where a fifth sibling shows up or the parent node
    /// holds 2^27 pairs or more. With `skip_alone` the pairs of nodes
    /// already alone are left out, so that only the siblings of the nodes
    /// not alone count against the four.
    ///
    /// The first pass reads each pair's sibling against the siblings met so
    /// far, held in registers, ORs its bit into the node's code and lists
    /// the node at its first pair, in `wide.bucket`. The second takes each listed
    /// node's code, into `wide.touched`, and reads its class: a class met keeps the
    /// first code it meets in `wide.dest` and counts its members in `wide.hits`, and
    /// only a class that meets a second code counts its members by code, in
    /// a row of `wide.rows`. Each class met then gets the class its members of
    /// each code move to, as [`Partition::split_sibling`] gives them, and
    /// where some node moves the last pass moves the nodes.
    fn split_by_codes(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
        skip_alone: bool,
    ) -> Result<bool, OperationError> {
        // Where every row starts is below `MIXED`.
        if pairs.len() >= 1 << 27 {
            return Ok(false);
        }
        let mut sib = [split_pair(&pairs[0], side).1; FEW_SIBLINGS];
        let mut d = 1usize;
        let mut listed = 0;
        for pair in pairs {
            let (t, s) = split_pair(pair, side);
            if skip_alone && (self.bits[(t / 64) as usize][1] >> (t % 64)) & 1 != 0 {
                continue;
            }
            let Some(bit) = sibling_bit(&mut sib, &mut d, s) else {
                for &t in &wide.bucket[..listed] {
                    wide.code[t as usize] = 0;
                }
                return Ok(false);
            };
            let e = &mut wide.code[t as usize];
            wide.bucket[listed] = t;
            listed += usize::from(*e == 0);
            *e |= bit;
        }
        // Within the reserve of one entry a pair.
        wide.touched.clear();
        for &t in &wide.bucket[..listed] {
            wide.touched
                .push(u32::from(std::mem::take(&mut wide.code[t as usize])));
        }
        self.split_listed_codes(lim, listed, d, wide)?;
        Ok(true)
    }

    /// A wide parent node whose pairs come in node order and name at most
    /// [`FEW_SIBLINGS`] siblings, split as [`Self::split_by_codes`] splits
    /// one, with each run's code formed as it is read: one pass over the
    /// pairs finds where each run ends and the siblings' bits it ORs, holds
    /// the siblings met in registers, and lists each node whose alone bit
    /// is clear with its run's code beside it, so that no node's code is
    /// formed in memory a pair at a time. Where every node listed is in the
    /// first class and unnamed before, as a parent node's private nodes
    /// are, the split reads no class ([`Self::split_first_codes`]).
    /// Returns [`Degrees::Unordered`] where a node
    /// comes below the one before, and [`Degrees::Many`] where a fifth
    /// sibling shows up or the parent node holds 2^27 pairs or more, in
    /// both cases with nothing changed.
    fn split_by_run_codes(
        &mut self,
        lim: &Limits,
        pairs: &[ChildPair],
        side: ChildSide,
        wide: &mut Wide<'_>,
    ) -> Result<Degrees, OperationError> {
        if pairs.len() >= 1 << 27 {
            return Ok(Degrees::Many);
        }
        lim.try_resize(wide.seen, pairs.len(), 0u32)?;
        lim.try_resize(wide.bucket, pairs.len(), 0u32)?;
        wide.met.clear();
        lim.reserve_exact(wide.met, pairs.len())?;
        wide.touched.clear();
        lim.reserve_exact(wide.touched, pairs.len())?;
        let mut sib = [split_pair(&pairs[0], side).1; FEW_SIBLINGS];
        let mut d = 1usize;
        let (mut listed, mut last) = (0usize, None);
        // Whether every node listed is in the first class with no pair
        // read before naming it.
        let mut first = true;
        let mut k = 0;
        while k < pairs.len() {
            let t = split_pair(&pairs[k], side).0;
            if last.is_some_and(|last| t < last) {
                return Ok(Degrees::Unordered);
            }
            last = Some(t);
            let mut e = 0u8;
            while k < pairs.len() {
                let (u, s) = split_pair(&pairs[k], side);
                if u != t {
                    break;
                }
                let Some(bit) = sibling_bit(&mut sib, &mut d, s) else {
                    return Ok(Degrees::Many);
                };
                e |= bit;
                k += 1;
            }
            let [moved, alone] = self.bits[(t / 64) as usize];
            if (alone >> (t % 64)) & 1 == 0 {
                first &= (moved >> (t % 64)) & 1 == 0;
                wide.touched.push(u32::from(e));
                wide.bucket[listed] = t;
                listed += 1;
            }
        }
        if first && self.can_twin(0) {
            self.split_first_codes(&wide.bucket[..listed], wide.touched, d);
        } else {
            self.split_listed_codes(lim, listed, d, wide)?;
        }
        Ok(Degrees::Split)
    }

    /// [`split_listed_codes`](Self::split_listed_codes) where every node is
    /// in the first class and no pair named it before, as the nodes of a
    /// wide parent node that is the only one to name them are, with no
    /// class read: the nodes of each code that are all of the first class
    /// stay in it, those of another code of two or more make a class of
    /// their own, and the rest leave alone, the classes made in code order
    /// as the general split makes them. The first class holds two nodes or
    /// more.
    #[inline(always)]
    fn split_first_codes(&mut self, nodes: &[u32], node_codes: &[u32], d: usize) {
        let mut count = [0u32; 1 << FEW_SIBLINGS];
        for &e in node_codes {
            count[e as usize] += 1;
        }
        let mut keep = nodes.len() == self.classes[0].size as usize;
        let mut dest = [ALONE; 1 << FEW_SIBLINGS];
        for e in 0..1usize << d {
            if count[e] >= 2 {
                dest[e] = if std::mem::take(&mut keep) {
                    0
                } else {
                    self.fresh(0, count[e])
                };
            }
        }
        for (&t, &e) in nodes.iter().zip(node_codes) {
            match dest[e as usize] {
                ALONE => self.alone(t, 0),
                id => self.put(t, id),
            }
        }
    }

    /// The first `listed` nodes in `wide.bucket` of a wide parent node of `d`
    /// siblings at most [`FEW_SIBLINGS`], each listed once with its code in
    /// `wide.touched`, split by their codes ([`Self::split_by_codes`]). A
    /// node read as the last of its class is marked alone, so that later
    /// parent nodes skip it by its bit, and named.
    #[inline(always)]
    fn split_listed_codes(
        &mut self,
        lim: &Limits,
        listed: usize,
        d: usize,
        wide: &mut Wide<'_>,
    ) -> Result<(), OperationError> {
        let Wide {
            bucket,
            touched,
            seen,
            met,
            rows,
            ..
        } = wide;
        let (nodes, node_codes) = (&bucket[..listed], &touched[..]);
        // A class that meets two codes has two members or more here.
        let stride = 1usize << d;
        rows.clear();
        lim.reserve_exact(rows, listed / 2 * stride)?;
        met.clear();
        for ((&t, &e), seen) in nodes.iter().zip(node_codes).zip(seen.iter_mut()) {
            let c = self.of(t);
            *seen = ALONE;
            if c == ALONE {
                continue;
            }
            if self.classes[c as usize].size < 2 {
                // The last node of its class, marked alone, and named, as
                // `split_sibling` marks it.
                let word = &mut self.bits[(t / 64) as usize];
                word[0] |= 1 << (t % 64);
                word[1] |= 1 << (t % 64);
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
            for ((&t, &c), &e) in nodes.iter().zip(seen.iter()).zip(node_codes) {
                if c == ALONE {
                    continue;
                }
                let k = self.classes[c as usize];
                let dest = if k.hits == 0 {
                    k.dest
                } else {
                    rows[(k.hits - 1) as usize + e as usize]
                };
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
        Ok(())
    }

    /// [`Self::split_sibling`] by the `nodes` one sibling is paired with,
    /// one or two of them without its lists: one node leaves its class
    /// alone, as a parent node of one pair leaves it, and two split as the
    /// two nodes of a parent node with one sibling ([`Self::split_nodes`]).
    #[inline(always)]
    fn split_bucket(&mut self, nodes: &[u32], met: &mut Vec<u32>, seen: &mut [u32]) {
        match *nodes {
            [t] => self.leave(t),
            [t0, t1] => self.split_nodes(t0, t1, true),
            _ => self.split_sibling(nodes.iter().copied(), met, seen),
        }
    }

    /// Split every class by the `nodes` one sibling is paired with: a class
    /// they all are is left whole; one member of a larger class leaves the
    /// partition alone; two or more members of a larger class become a class
    /// of their own, so the classes made still nest and hold two or more
    /// nodes each. Each class's `hits` is zero on entry and on return;
    /// `seen` keeps each node's class as the count read it, [`ALONE`] for a
    /// node that cannot have a twin, so the second pass reads no class. A
    /// node read as the last of its class is marked alone. Where every
    /// class met is named whole and none is the first, as when
    /// a node's second context confirms the class its first one made, no
    /// node changes class and the second pass is left out.
    #[inline]
    fn split_sibling(
        &mut self,
        nodes: impl Iterator<Item = u32> + Clone,
        met: &mut Vec<u32>,
        seen: &mut [u32],
    ) {
        met.clear();
        for (t, seen) in nodes.clone().zip(seen.iter_mut()) {
            let c = self.of(t);
            *seen = ALONE;
            if c == ALONE {
                continue;
            }
            let k = &mut self.classes[c as usize];
            if k.size < 2 {
                // The last node of its class, marked alone: a later split
                // reads that from its word, without its class.
                let word = &mut self.bits[(t / 64) as usize];
                word[0] |= 1 << (t % 64);
                word[1] |= 1 << (t % 64);
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
    fn apply_groups(
        &mut self,
        runs: &[(u64, u32, u32)],
        links: &[(u32, u32)],
        node: impl Fn((u64, u32, u32)) -> u32,
    ) {
        for (r, &run) in runs.iter().enumerate() {
            let (mut next, members) = links[r];
            if members == 0 {
                continue;
            }
            let c = (run.0 >> 32) as u32;
            if members == 1 {
                self.alone(node(run), c);
            } else if members < self.classes[c as usize].size || c == 0 {
                let id = if members < self.classes[c as usize].size {
                    self.fresh(c, members)
                } else {
                    c
                };
                self.put(node(run), id);
                while next != NO_RUN {
                    self.put(node(runs[next as usize]), id);
                    next = links[next as usize].0;
                }
            }
        }
    }
}

/// What a split of a wide parent node by its runs' codes or degrees did
/// ([`Partition::split_by_run_codes`], [`Partition::split_by_degree`]).
enum Degrees {
    /// The parent node is split.
    Split,
    /// Its pairs come in node order, but its nodes' sibling sets are not
    /// nested: the nodes of a degree no other node of their class has there
    /// are out of the partition, and the rest are to be split by siblings.
    Unsettled,
    /// Its pairs do not come in node order, and nothing is changed.
    Unordered,
    /// Its pairs name more than [`FEW_SIBLINGS`] siblings, or it holds too
    /// many pairs, for a split by codes, and nothing is changed.
    Many,
}

/// The most siblings a wide parent node's pairs name for its nodes to be
/// split by their codes ([`Partition::split_by_codes`]).
pub(super) const FEW_SIBLINGS: usize = 4;

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

/// The bit of sibling `code` among the `d` slots of `sib` in use, allocating a
/// new slot when `code` has not been seen yet and fewer than [`FEW_SIBLINGS`]
/// slots are in use.
#[inline(always)]
fn sibling_bit(sib: &mut [u32; FEW_SIBLINGS], d: &mut usize, code: u32) -> Option<u8> {
    let used = (1u8 << *d) - 1;
    let hit = u8::from(code == sib[0])
        | u8::from(code == sib[1]) << 1
        | u8::from(code == sib[2]) << 2
        | u8::from(code == sib[3]) << 3;
    let bit = hit & used;
    if bit != 0 {
        return Some(bit);
    }
    if *d == FEW_SIBLINGS {
        return None;
    }
    sib[*d] = code;
    let bit = 1 << *d;
    *d += 1;
    Some(bit)
}
