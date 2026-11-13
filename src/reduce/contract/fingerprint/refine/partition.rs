//! The partition a refinement splits, and its splits of parent nodes of
//! few pairs.

use crate::diagram::{ChildPair, ChildSide, EncodedNode, NodeKind, StoredPairs};
use crate::limits::{Limits, OperationError};
use crate::reduce::contract::bits;
use crate::reduce::contract::scratch::ClassInfo;

use super::super::{pack, split_pair};
use super::SMALL_PAIRS;

/// The class of a node known to have no twin.
pub(super) const ALONE: u32 = u32::MAX;

/// The partition of the child level's nodes. A node is in the first class
/// until its moved bit is set; then it has no twin if its alone bit is set,
/// and is in class `class[t]` if not. The bits stay in cache where the
/// classes would not, so a node that leaves the first class alone, as most
/// do on a level without twins, costs two bits; each word of moved bits
/// sits beside the word of alone bits for the same nodes, in one cache
/// line. A node a parent node names has its moved bit set even where it
/// stays in the first class, so the nodes whose bit is clear are the nodes
/// no pair names.
pub(super) struct Partition<'a> {
    pub(super) class: &'a mut [u32],
    /// Per 64 nodes, their moved bits and their alone bits.
    pub(super) bits: &'a mut [[u64; 2]],
    pub(super) classes: &'a mut Vec<ClassInfo>,
    /// The nodes in classes of two or more: those that can still have a
    /// twin.
    pub(super) twinable: usize,
}

impl Partition<'_> {
    /// The parent nodes of a stored level of [`SMALL_PAIRS`] pairs or
    /// fewer, split in node order, and the wider ones put in `wide`, each
    /// as the power of two of its width above its index, for the caller to
    /// split after. `LEFT` says whether the nodes split are the left
    /// children, so the side is a constant of the loop. Stops once no node
    /// can have a twin.
    pub(super) fn split_narrow<const LEFT: bool>(
        &mut self,
        stored: StoredPairs<'_>,
        lim: &Limits,
        wide: &mut Vec<u64>,
    ) -> Result<(), OperationError> {
        let side = if LEFT {
            ChildSide::Left
        } else {
            ChildSide::Right
        };
        let nodes = stored.nodes();
        let mut i = 0;
        while self.twinable != 0 {
            i = self.leave_run(nodes, stored.arena(), i, side);
            let Some(node) = nodes.get(i) else {
                break;
            };
            if self.twinable == 0 {
                break;
            }
            let pairs = stored.of(node);
            match pairs.len() {
                0 => {}
                1 => self.leave(split_pair(&pairs[0], side).0),
                2 => self.split_two(pairs, side),
                3 | 4 if self.split_few(pairs, side) => {}
                n if n <= SMALL_PAIRS => self.split_small(pairs, side),
                n => lim.try_push(
                    wide,
                    (u64::from(usize::BITS - n.leading_zeros()) << 32) | i as u64,
                )?,
            }
            i += 1;
        }
        Ok(())
    }

    /// The parent nodes from index `i` of `nodes` on that name one child
    /// each, an inline node or two pairs of one child on `side`, split as
    /// [`leave`](Self::leave) splits them; returns the index of the first
    /// other node, or the index past the node that left no node able to
    /// have a twin, or the end. The partition is held in locals and nothing
    /// is called, so that the loop keeps it in registers: most parent nodes
    /// of a level with twins name one child.
    #[inline(always)]
    fn leave_run(
        &mut self,
        nodes: &[EncodedNode],
        arena: &[ChildPair],
        i: usize,
        side: ChildSide,
    ) -> usize {
        let Partition {
            class,
            bits,
            classes,
            twinable,
        } = self;
        let (class, classes) = (&**class, &mut classes[..]);
        // The first class is always there, so taking a node out of it
        // checks no bound.
        assert!(!classes.is_empty(), "the first class");
        let mut left = *twinable;
        let mut at = i;
        while at < nodes.len() {
            let t = match nodes[at].kind() {
                NodeKind::Inline(pair) => split_pair(&pair, side).0,
                NodeKind::Multi { start, len: 2 } => {
                    let pairs = &arena[start as usize..start as usize + 2];
                    let t = split_pair(&pairs[0], side).0;
                    if split_pair(&pairs[1], side).0 != t {
                        break;
                    }
                    t
                }
                _ => break,
            };
            at += 1;
            let (w, b) = bits::at(t);
            let [moved, alone] = bits[w];
            if alone & b != 0 {
                continue;
            }
            bits[w] = [moved | b, alone | b];
            let k = if moved & b == 0 {
                &mut classes[0]
            } else {
                &mut classes[class[t as usize] as usize]
            };
            let members = k.size;
            k.size = members - 1;
            // A class that keeps two members or more keeps them able to
            // have a twin, so only a smaller one can leave none.
            if members > 2 {
                left -= 1;
            } else {
                left -= if members == 2 { 2 } else { 0 };
                if left == 0 {
                    break;
                }
            }
        }
        *twinable = left;
        at
    }

    /// Node `t`'s class, or [`ALONE`].
    #[inline(always)]
    pub(super) fn of(&self, t: u32) -> u32 {
        let (w, b) = bits::at(t);
        let [moved, alone] = self.bits[w];
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
    pub(super) fn can_twin(&self, c: u32) -> bool {
        c != ALONE && self.classes[c as usize].size >= 2
    }

    /// Take node `t` out of its class `c`: it has no twin, nor has the last
    /// node left in `c`.
    #[inline(always)]
    pub(super) fn alone(&mut self, t: u32, c: u32) {
        let (w, b) = bits::at(t);
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
    #[inline(always)]
    pub(super) fn leave(&mut self, t: u32) {
        let c = self.of(t);
        if c != ALONE {
            self.alone(t, c);
        }
    }

    /// Put node `t` in class `id`.
    #[inline(always)]
    pub(super) fn put(&mut self, t: u32, id: u32) {
        self.class[t as usize] = id;
        let (w, b) = bits::at(t);
        self.bits[w][0] |= b;
    }

    /// A new class for `members` nodes taken out of class `c`, which the
    /// caller puts there.
    #[inline(always)]
    pub(super) fn fresh(&mut self, c: u32, members: u32) -> u32 {
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
    #[inline(always)]
    pub(super) fn split_two(&mut self, pairs: &[ChildPair], side: ChildSide) {
        let ((t0, s0), (t1, s1)) = (split_pair(&pairs[0], side), split_pair(&pairs[1], side));
        if t0 == t1 {
            self.leave(t0);
        } else {
            self.split_nodes(t0, t1, s0 == s1);
        }
    }

    /// A parent node of three or four pairs naming one or two nodes, split
    /// as its pairs are read, with no sort: a node it names alone has no
    /// twin, nor have two nodes of different numbers of siblings there; two
    /// nodes of two siblings each are split as [`split_two`](Self::split_two)
    /// splits two of one. Returns false, having changed nothing, where the
    /// pairs name three nodes or more.
    #[inline(always)]
    pub(super) fn split_few(&mut self, pairs: &[ChildPair], side: ChildSide) -> bool {
        let (a, s) = split_pair(&pairs[0], side);
        let mut b = a;
        // The counts of the two nodes' pairs, and their first two siblings.
        let (mut na, mut nb) = (1usize, 0usize);
        let (mut sa, mut sb) = ([s, 0], [0u32; 2]);
        for pair in &pairs[1..] {
            let (t, s) = split_pair(pair, side);
            if t == a {
                if na < 2 {
                    sa[na] = s;
                }
                na += 1;
            } else if nb == 0 || t == b {
                b = t;
                if nb < 2 {
                    sb[nb] = s;
                }
                nb += 1;
            } else {
                return false;
            }
        }
        if nb == 0 {
            self.leave(a);
        } else {
            // Two multisets of two siblings are equal where their least and
            // greatest are.
            let same = na == 2
                && nb == 2
                && sa[0].min(sa[1]) == sb[0].min(sb[1])
                && sa[0].max(sa[1]) == sb[0].max(sb[1]);
            self.split_nodes(a, b, same);
        }
        true
    }

    /// Two distinct nodes of one parent node, `same` where their siblings
    /// there are the same: they are a group if they are of one class, and
    /// each is alone in its group otherwise.
    #[inline(always)]
    pub(super) fn split_nodes(&mut self, t0: u32, t1: u32, same: bool) {
        let (c0, c1) = (self.of(t0), self.of(t1));
        if !same || c0 != c1 || c0 == ALONE {
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
    /// Returns how many there were.
    pub(super) fn drop_unnamed(
        &mut self,
        width: usize,
        out: Option<&mut Vec<u32>>,
        lim: &Limits,
    ) -> Result<usize, OperationError> {
        let words = width.div_ceil(64);
        let mut unnamed = 0;
        for w in 0..words {
            let clear = !self.bits[w][0] & bits::word_mask(width, w);
            unnamed += clear.count_ones() as usize;
        }
        if unnamed < 2 {
            return Ok(unnamed);
        }
        debug_assert_eq!(
            self.classes[0].size as usize, unnamed,
            "the first class is the nodes no pair names"
        );
        if let Some(out) = out {
            lim.reserve_exact(out, unnamed)?;
            for (w, word) in self.bits.iter().enumerate() {
                let mut clear = !word[0] & bits::word_mask(width, w);
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
        Ok(unnamed)
    }

    /// A parent node of three to [`SMALL_PAIRS`] pairs that
    /// [`split_few`](Self::split_few) leaves, grouped on the stack: its
    /// pairs sorted, so that each node's siblings are a run in ascending
    /// order, and each run compared with the runs before it.
    #[inline]
    pub(super) fn split_small(&mut self, pairs: &[ChildPair], side: ChildSide) {
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
        let siblings = |r: usize| {
            key[from[r] as usize..to[r] as usize]
                .iter()
                .map(|&e| e as u32)
        };
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
                    if lead[q] as usize == q
                        && to[q] - from[q] == to[r] - from[r]
                        && siblings(q).eq(siblings(r))
                    {
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
}
