//! A pool member's size on its home: its own pairs and the pairs it would
//! hold at the home's nodes it has no level at, which is what the search on
//! projections minimizes.

use crate::diagram::{Tdd, TddLevel};
use crate::vtree::rotate::RotationInfo;
use crate::vtree::{RotationKind, Vtree, VtreeIdx};

use super::{Groups, NONE, children_table, pairs_and_owners};

impl Tdd {
    /// The pairs this diagram would hold with a level at every node of its
    /// home ([`Vtree::home`]): [`pair_count`](Self::pair_count) on a vtree
    /// that is not a projection or has no leaves. On a projection of `k` of
    /// the home's `V` leaves it adds one pair for each of the `V − k`
    /// internal home nodes that are not its own, and at each
    /// home node strictly between an own node `t` and the own node above
    /// it, the nodes of `t`'s level less one; a leaf counts the distinct
    /// children its parent's pairs name on its side, so an entailed literal
    /// counts none. For a canonical diagram this is the
    /// [`pair_count`](Self::pair_count) of the same function stored with a
    /// level at every node of the home's tree, which [`Engine::pool_search`](crate::Engine::pool_search)
    /// minimizes over a pool on projections.
    ///
    /// Not cached: it reads the pairs of every level with a leaf child.
    ///
    /// ```
    /// use tididi::Tdd;
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// // x1 ∧ x3 on the projection of a right-linear home of three leaves
    /// // pays for the home's node over x2 and x3, which it has no level at.
    /// let home = Home::new(Vtree::linear(3));
    /// let f = Tdd::clause(&home, [1])? & Tdd::clause(&home, [3])?;
    /// assert_eq!(f.vtree().leaf_bottomup().map(|(_, v)| v).collect::<Vec<_>>(), [VarId(1), VarId(3)]);
    /// assert!(f.embedded_pair_count() > f.pair_count());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn embedded_pair_count(&self) -> usize {
        let pairs = self.pair_count();
        let Some(home) = self.vtree.link_home() else {
            return pairs;
        };
        let hv: &Vtree = home;
        let pairs = pairs + not_own(self, hv.num_leaves());
        let kids = children_table(&self.vtree);
        let mut widths = vec![1u32; kids.len()];
        own_widths(self, &kids, &mut widths);
        let mut carried = 0usize;
        for (t, &(l, r)) in kids.iter().enumerate() {
            if l.idx() == t {
                continue;
            }
            let top = self.vtree.home_node(VtreeIdx(t as u32));
            for c in [l, r] {
                let below = self.vtree.home_node(c);
                debug_assert_eq!(home.depth(below) - home.depth(top), chain(hv, below, top).count() as u32);
                let between = (home.depth(below) - home.depth(top) - 1) as usize;
                carried += between * (widths[c.idx()] as usize - 1);
            }
        }
        pairs + carried
    }
}

/// The internal nodes of a home of `leaves` leaves that are none of `f`'s,
/// a projection of it; none for a constant, which holds no pair on the
/// home's root either.
fn not_own(f: &Tdd, leaves: u32) -> usize {
    match f.vtree.num_leaves() {
        0 => 0,
        k => leaves.saturating_sub(k) as usize,
    }
}

/// The nodes of `tree` from `u` up to `top`, `top` excluded; all of them if
/// `top` is not above `u`.
fn chain(tree: &Vtree, u: VtreeIdx, top: VtreeIdx) -> impl Iterator<Item = VtreeIdx> + '_ {
    std::iter::successors(Some(u), move |&x| tree.node(x).parent()).take_while(move |&x| x != top)
}

/// By own node of `m`, whose children are `kids` ([`children_table`]), its
/// width in `widths`: the nodes of an internal node's level, the distinct
/// children a leaf's parent's pairs name on its side; at least 1. The root
/// of a tree of one leaf is left as it is. Returns the nodes and pairs read.
pub(super) fn own_widths(m: &Tdd, kids: &[(VtreeIdx, VtreeIdx)], widths: &mut [u32]) -> u64 {
    let mut read = 0u64;
    for (t, &(l, r)) in kids.iter().enumerate() {
        if l.idx() == t {
            continue;
        }
        let level = &m.levels[t];
        widths[t] = level_width(level);
        read += level.node_count() as u64;
        let leaves = [kids[l.idx()].0 == l, kids[r.idx()].0 == r];
        if leaves == [false, false] {
            continue;
        }
        let (labels, pairs) = leaf_labels(&m.levels, t, [l, r], leaves);
        read += pairs;
        for (side, c) in [l, r].into_iter().enumerate() {
            if leaves[side] {
                widths[c.idx()] = labels[side].max(1);
            }
        }
    }
    read
}

fn level_width(level: &TddLevel) -> u32 {
    match level.is_marginal() {
        true => 1,
        false => pairs_and_owners(level).1.max(1) as u32,
    }
}

/// The distinct children the pairs of the level at `t` name on each side
/// marked in `leaves`, whose children `kids` are leaves, and the pairs read.
/// A leaf summed out is named by counts or values rather than labels, so a
/// level over one is read pair by pair rather than as bits
/// ([`TddLevel::leaf_label_bits`]).
fn leaf_labels(levels: &[TddLevel], t: usize, kids: [VtreeIdx; 2], leaves: [bool; 2]) -> ([u32; 2], u64) {
    let level = &levels[t];
    if level.is_marginal() {
        return ([0; 2], 0);
    }
    if !(0..2).any(|s| leaves[s] && levels[kids[s].idx()].is_marginal()) {
        let (bits, read) = level.leaf_label_bits();
        return (bits.map(u8::count_ones), read);
    }
    let (mut labels, mut read) = ([Vec::new(), Vec::new()], 0u64);
    level.for_each_node_pair(|_, pair| {
        read += 1;
        for (side, child) in [pair.left.0, pair.right.0].into_iter().enumerate() {
            if leaves[side] && !labels[side].contains(&child) {
                labels[side].push(child);
            }
        }
    });
    (labels.map(|l| l.len() as u32), read)
}

/// The O, M and L parts of a move made as `info`: the pivot's child that
/// stays, the promoted child's child that joins it, and the one lifted to the
/// pivot.
fn parts(info: &RotationInfo, kind: RotationKind) -> [VtreeIdx; 3] {
    match kind {
        RotationKind::Left => [info.a_idx, info.b_idx, info.c_idx],
        RotationKind::Right => [info.c_idx, info.b_idx, info.a_idx],
    }
}

/// Which of `parts` node `u` of `tree` is under, by index in `parts`; the
/// steps up added to `steps`.
fn part_of(tree: &Vtree, mut u: VtreeIdx, parts: &[VtreeIdx], steps: &mut u64) -> usize {
    loop {
        if let Some(p) = parts.iter().position(|&x| x == u) {
            return p;
        }
        u = tree.node(u).parent().expect("a node under one of a move's parts");
        *steps += 1;
    }
}

/// A move's change of the carried pairs ([`Carried::change`]).
pub(super) struct Change {
    /// The move's, and its crossing's.
    pub(super) by: [i64; 2],
    parts: [VtreeIdx; 3],
    /// The promoted node.
    w: VtreeIdx,
}

/// The search's table of the pairs its members on projections carry
/// through the home's nodes they have no level at ([`Tdd::embedded_pair_count`]
/// less the own pairs), on the search's tree, which has every leaf of the
/// home.
///
/// A member's `g(x)` at a node `x` of the tree is the width of its top own
/// node under `x` less one, where it has leaves under `x` and outside it,
/// else 0; it carries `g(c)` at each node with its leaves under one child `c`
/// only. A width is a function of the member and of the leaves under its own
/// node, so a kept move changes `g` only at the promoted node.
#[derive(Clone)]
pub(super) struct Carried {
    /// By member: the first entry of its own nodes in `widths`; unused for
    /// the members on the search's tree.
    pub(super) at: Vec<usize>,
    /// By member and own node: its width ([`own_widths`]).
    pub(super) widths: Vec<u32>,
    /// By node `x` of the tree: the sum of every member's `g(x)`.
    pub(super) passing: Vec<u64>,
    /// The pairs carried, over all members.
    pub(super) total: u64,
    /// The members' internal home nodes that are none of their own, one pair
    /// each.
    constant: u64,
}

impl Carried {
    /// The table of `members` on `tree` and `groups`; the nodes and pairs
    /// read.
    pub(super) fn new(members: &[&mut Tdd], tree: &Vtree, groups: &Groups) -> (Carried, u64) {
        let mut at = vec![usize::MAX; members.len()];
        let mut entries = 0;
        for group in &groups.groups {
            let Some(kids) = &group.kids else { continue };
            for &i in &group.members {
                at[i] = entries;
                entries += kids.len();
            }
        }
        let mut carried = Carried {
            at,
            widths: vec![1; entries],
            passing: vec![0; tree.num_nodes()],
            total: 0,
            constant: 0,
        };
        let mut read = 0u64;
        for group in &groups.groups {
            let Some(kids) = &group.kids else { continue };
            for &i in &group.members {
                carried.constant += not_own(members[i], tree.num_leaves()) as u64;
            }
            for &i in &group.members {
                let at = carried.at[i];
                read += own_widths(members[i], kids, &mut carried.widths[at..at + kids.len()]);
            }
            for t in (0..kids.len()).filter(|&t| group.up[t] != NONE) {
                let sum: u64 = group.members.iter().map(|&i| u64::from(carried.widths[carried.at[i] + t] - 1)).sum();
                let top = group.union_of[group.up[t] as usize];
                let mut len = 0u64;
                for x in chain(tree, group.union_of[t], top) {
                    carried.passing[x.idx()] += sum;
                    len += 1;
                }
                carried.total += sum * (len - 1);
                read += len;
            }
        }
        (carried, read)
    }

    /// The members' carried pairs and one pair for each home node none of
    /// their own.
    pub(super) fn pairs(&self) -> usize {
        (self.total + self.constant) as usize
    }

    fn width(&self, i: usize, t: VtreeIdx) -> i64 {
        i64::from(self.widths[self.at[i] + t.idx()])
    }

    /// The change of the carried pairs under the move made on `tree` as
    /// `info`, and under the same move with the promoted node's children the
    /// other way round, the members the move rebuilds counted as they stand;
    /// the entries and steps read. `tree` has the move's pointers, `groups`
    /// are as before it.
    ///
    /// By the parts of the move a member has leaves under (O, M, L:
    /// [`parts`]), with `out` whether it has leaves outside the pivot and
    /// `w(P)` the width of its top own node under P: O M changes by
    /// `[out]·(w(O∪M) − 1) − (w(M) − 1)`; M L by `(w(M) − 1) −
    /// [out]·(w(M∪L) − 1)`; O L by `w(O) − w(L)`; O alone by `g(O)`, L alone
    /// by `−g(L)`; M alone and all three by nothing. The members with two or
    /// three parts have an own node at the pivot or at the promoted node; the
    /// one-part sums are `passing` less the others' terms.
    pub(super) fn change(&self, tree: &Vtree, groups: &mut Groups, info: &RotationInfo, kind: RotationKind) -> (Change, u64) {
        let parts = parts(info, kind);
        let (v, w) = (info.v_idx, info.w_idx);
        let Groups { groups, lists, seen, .. } = groups;
        for (u, bit) in [(v, 1), (w, 2)] {
            for e in lists.of(u) {
                seen[e.group as usize] |= bit;
            }
        }
        let mut read = 0u64;
        // By part: the `g` of the members with leaves under it and another.
        let mut shared = [0i64; 3];
        let mut d = [0i64; 2];
        // A group with all three parts has its child at `w` there.
        let below_v = [parts[0], parts[1], parts[2], w];
        for e in lists.of(v) {
            let group = &groups[e.group as usize];
            let (a, b) = group.kids.as_ref().expect("a listed group")[e.own.idx()];
            let three = seen[e.group as usize] & 2 != 0;
            let pa = part_of(tree, group.union_of[a.idx()], &below_v, &mut read);
            let (o, x, px) = match pa {
                0 if !three => (a, b, part_of(tree, group.union_of[b.idx()], &parts, &mut read)),
                0 => (a, b, 3),
                _ => (b, a, pa),
            };
            let out = group.up[e.own.idx()] != NONE;
            for &i in &group.members {
                read += 1;
                let wo = self.width(i, o) - 1;
                shared[0] += wo;
                if three {
                    continue;
                }
                let wx = self.width(i, x) - 1;
                let wt = if out { self.width(i, e.own) - 1 } else { 0 };
                shared[px] += wx;
                let (with_m, with_l) = (wt - wx, wo - wx);
                d[0] += if px == 1 { with_m } else { with_l };
                d[1] += if px == 1 { with_l } else { with_m };
            }
        }
        for e in lists.of(w) {
            let group = &groups[e.group as usize];
            let (a, b) = group.kids.as_ref().expect("a listed group")[e.own.idx()];
            let (m, l) = match part_of(tree, group.union_of[a.idx()], &parts, &mut read) {
                1 => (a, b),
                _ => (b, a),
            };
            let three = seen[e.group as usize] & 1 != 0;
            let out = group.up[e.own.idx()] != NONE;
            for &i in &group.members {
                read += 1;
                let (wm, wl) = (self.width(i, m) - 1, self.width(i, l) - 1);
                shared[1] += wm;
                shared[2] += wl;
                if three {
                    continue;
                }
                let wt = if out { self.width(i, e.own) - 1 } else { 0 };
                d[0] += wm - wt;
                d[1] += wl - wt;
            }
        }
        for u in [v, w] {
            for e in lists.of(u) {
                seen[e.group as usize] = 0;
            }
        }
        let alone = |p: usize| self.passing[parts[p].idx()] as i64 - shared[p];
        d[0] += alone(0) - alone(2);
        d[1] += alone(0) - alone(1);
        (Change { by: d, parts, w }, read)
    }

    /// After the move whose `change` this is was kept on `tree`, `groups`
    /// moved with it and the members in `touched` rebuilt: their widths at
    /// the two rebuilt nodes and at the leaves below them taken again, and
    /// `passing` at the promoted node. Returns the entries and steps read.
    pub(super) fn kept(
        &mut self,
        tree: &Vtree,
        groups: &Groups,
        members: &[&mut Tdd],
        touched: &[(usize, RotationInfo)],
        change: &Change,
    ) -> u64 {
        self.total = self.total.checked_add_signed(change.by[0]).expect("carried pairs stay positive");
        let mut read = 0u64;
        for &(i, moved) in touched {
            let group = &groups.groups[groups.of[i]];
            let kids = group.kids.as_ref().expect("a listed group");
            for t in [moved.v_idx, moved.w_idx] {
                let level = &members[i].levels[t.idx()];
                let (l, r) = kids[t.idx()];
                let leaves = [kids[l.idx()].0 == l, kids[r.idx()].0 == r];
                let (labels, pairs) = leaf_labels(&members[i].levels, t.idx(), [l, r], leaves);
                read += pairs + level.node_count() as u64;
                let mut set = |c: VtreeIdx, width: u32| {
                    let entry = &mut self.widths[self.at[i] + c.idx()];
                    let delta = i64::from(width) - i64::from(*entry);
                    *entry = width;
                    // The promoted node's `passing` is taken again below.
                    if delta == 0 || c == moved.w_idx || group.up[c.idx()] == NONE {
                        return;
                    }
                    let top = group.union_of[group.up[c.idx()] as usize];
                    let mut len = 0u64;
                    for x in chain(tree, group.union_of[c.idx()], top) {
                        self.passing[x.idx()] = self.passing[x.idx()].checked_add_signed(delta).expect("passing pairs stay positive");
                        len += 1;
                    }
                    self.total = self.total.checked_add_signed(delta * (len as i64 - 1)).expect("carried pairs stay positive");
                    read += len;
                };
                set(t, level_width(level));
                for (side, c) in [l, r].into_iter().enumerate() {
                    if leaves[side] {
                        set(c, labels[side].max(1));
                    }
                }
            }
        }
        let ([o, m, _], w) = (change.parts, change.w);
        let mut at_w = self.passing[o.idx()] as i64 + self.passing[m.idx()] as i64;
        for e in groups.lists.of(w) {
            let group = &groups.groups[e.group as usize];
            let (a, b) = group.kids.as_ref().expect("a listed group")[e.own.idx()];
            let out = group.up[e.own.idx()] != NONE;
            for &i in &group.members {
                read += 1;
                let wt = if out { self.width(i, e.own) - 1 } else { 0 };
                at_w += wt - (self.width(i, a) - 1) - (self.width(i, b) - 1);
            }
        }
        self.passing[w.idx()] = u64::try_from(at_w).expect("passing pairs stay positive");
        read
    }
}
