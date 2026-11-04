//! Conjunction in skip form.
//!
//! With nothing summed out the conjunction is the overlay's walk without
//! residues ([`super::overlay::conjoin`]). Under sums, here, the products
//! are built from pair combinations, which needs neither disjoint classes
//! nor references read from one level only.
//!
//! The product of two references, read from the same child level, is built
//! where the first of them is a stored node: at that node's level when one
//! names a node above the other's, and where they part when neither does
//! (two leaves under one level). Above it both skip the same variables, so
//! the product skips them too. A reference that skips through the level
//! where the product is built acts as the one pair `true` on the skipped
//! variable makes.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{discard_nested, filled, ix, mk, normalize, normalize_sum, vt, Ref, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::limits::Limits;
use crate::OperationError;

/// Where a product's node is.
pub(crate) enum Land {
    /// Known without building: a reference into the result (a leaf label or
    /// false).
    Done(Ref),
    /// Built at this internal level.
    At(u32),
}

/// Where the product of `a` and `b` (references read from one child level)
/// is built: `Done` for a false or leaf result.
#[inline]
pub(crate) fn land_pair(sh: &Shape, a: Ref, b: Ref) -> Land {
    if a == ZREF || b == ZREF {
        return Land::Done(ZREF);
    }
    let (va, vb) = (vt(a), vt(b));
    if va == vb {
        if sh.is_leaf(va) {
            return Land::Done(leaf_and(va, ix(a), ix(b)));
        }
        return Land::At(va);
    }
    if sh.within(vb, va) {
        return Land::At(va);
    }
    if sh.within(va, vb) {
        return Land::At(vb);
    }
    // They part below a level both skip through: each skips the other's
    // single variable, so both are leaves under it.
    debug_assert!(sh.is_leaf(va) && sh.is_leaf(vb) && sh.parent[va as usize] == sh.parent[vb as usize]);
    Land::At(sh.parent[va as usize])
}

#[inline]
fn leaf_and(v: u32, a: u32, b: u32) -> Ref {
    match (a, b) {
        (ONE, x) | (x, ONE) => mk(v, x),
        (POS, POS) => mk(v, POS),
        (NEG, NEG) => mk(v, NEG),
        _ => ZREF,
    }
}

/// Whether `r` is `true` (`One` on a leaf, `true` wherever it is read).
#[inline]
pub(super) fn is_one(sh: &Shape, r: Ref) -> bool {
    r != ZREF && sh.is_leaf(vt(r)) && ix(r) == ONE
}

/// Per vtree node, whether it is in `g`'s cone: an ancestor (or itself) of
/// a level holding a node of `g` or of a leaf `g` reads a literal on. Below
/// the cone every reference of `g` is `true`; the cone holds every ancestor
/// of its levels.
pub(super) fn cone_of(sh: &Shape, g: &SkipForm, lim: &Limits) -> Result<Vec<bool>, OperationError> {
    let mut cone = filled(lim, sh.num_nodes(), false)?;
    let mut up = |mut t: u32| {
        while t != NONE && !cone[t as usize] {
            cone[t as usize] = true;
            t = sh.parent[t as usize];
        }
    };
    let literal = |x: Ref| x != ZREF && sh.is_leaf(vt(x)) && ix(x) != ONE;
    if literal(g.out) {
        up(vt(g.out));
    }
    for &t in &g.active {
        up(t);
        for &(a, b) in &g.levels[t as usize].pairs {
            for x in [a, b] {
                if literal(x) {
                    up(vt(x));
                }
            }
        }
    }
    Ok(cone)
}

/// `f`'s references outside the cone that a pair in the cone, or the
/// output, names, each met once.
struct Outside {
    /// The products with `true` of `g` that the conjunction's scans may
    /// meet, of those references: `(preorder position of the level the
    /// product would be built at, the reference, g's true)`, latest built
    /// first. The product is the reference itself. A product built at a
    /// level in the cone (two leaves under one, a literal of `g` on one of
    /// them) is built there instead.
    seeds: Vec<(u32, Ref, Ref)>,
    /// The internal ones, each marked 1 in `mark` (by `f`'s numbers; the
    /// leaves are marked 3).
    boundary: Vec<Ref>,
    mark: Vec<u8>,
}

/// [`Outside`], in one pass over the cone's pairs of `f`.
fn outside(sh: &Shape, f: &SkipForm, nf: &Numbered, ng: &Numbered, cone: &[bool], lim: &Limits) -> Result<Outside, OperationError> {
    let mut seeds: Vec<(u32, Ref, Ref)> = Vec::new();
    let mut boundary: Vec<Ref> = Vec::new();
    let mut tops: Vec<Ref> = Vec::new();
    for &c in &ng.leaves {
        let t = mk(c, ONE);
        if ng.top[ng.id(t)] != NONE {
            lim.try_push(&mut tops, t)?;
        }
    }
    let tt = if tops.is_empty() { None } else { Some(Tops::new(sh, ng, &tops, lim)?) };
    let mut mark = filled(lim, nf.len(), 0u8)?;
    let mut found: Vec<u32> = Vec::new();
    let mut note = |x: Ref, seeds: &mut Vec<(u32, Ref, Ref)>, boundary: &mut Vec<Ref>| -> Result<(), OperationError> {
        let v = vt(x);
        if cone[v as usize] {
            return Ok(());
        }
        let id = nf.id(x);
        if mark[id] != 0 {
            return Ok(());
        }
        if sh.is_leaf(v) {
            mark[id] = 3;
        } else {
            mark[id] = 1;
            lim.try_push(boundary, x)?;
        }
        let Some(tt) = &tt else {
            return Ok(());
        };
        let p = if sh.is_leaf(v) { sh.parent[v as usize] } else { v };
        if p == NONE || (sh.is_leaf(v) && cone[p as usize]) || !nf.read_at(sh, x, p) {
            return Ok(());
        }
        // `g`'s true on a leaf under `p` read at `p`: below an internal
        // reference's level, or beside a leaf reference.
        found.clear();
        if sh.is_leaf(v) {
            let (l, r) = sh.kids[p as usize];
            let sib = if l == v { r } else { l };
            let i = tt.at[sib as usize];
            if i != NONE {
                lim.try_push(&mut found, i)?;
            }
        } else {
            tt.under(sh, p, &mut found, lim)?;
            found.sort_unstable();
        }
        for &i in &found {
            let t = tops[i as usize];
            if ng.read_at(sh, t, p) {
                lim.try_push(seeds, (sh.pre[p as usize], x, t))?;
            }
        }
        Ok(())
    };
    note(f.out, &mut seeds, &mut boundary)?;
    for &u in &f.active {
        if cone[u as usize] {
            for &(a, b) in &f.levels[u as usize].pairs {
                note(a, &mut seeds, &mut boundary)?;
                note(b, &mut seeds, &mut boundary)?;
            }
        }
    }
    lim.discard(found);
    if let Some(tt) = tt {
        tt.discard(lim);
    }
    lim.discard(tops);
    seeds.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
    Ok(Outside { seeds, boundary, mark })
}

/// `g`'s true references on leaves ([`Outside::seeds`]), found by the
/// levels that read them: by leaf preorder, each with the depth of the
/// highest level it is read at, and a table of the least depth over runs
/// of a power of two, so that the ones read at a level are listed in time
/// for their number, not for every leaf under it.
struct Tops {
    /// Per leaf, the index of its true among the tops; `NONE` without one.
    at: Vec<u32>,
    /// (leaf preorder, depth of the highest level it is read at, index), by preorder.
    by_pre: Vec<(u32, u32, u32)>,
    /// `least[j][i]`: the position in `by_pre` of the least depth over
    /// `i..i + 2^j`.
    least: Vec<Vec<u32>>,
}

impl Tops {
    fn new(sh: &Shape, ng: &Numbered, tops: &[Ref], lim: &Limits) -> Result<Tops, OperationError> {
        let mut at = filled(lim, sh.num_nodes(), NONE)?;
        let mut by_pre: Vec<(u32, u32, u32)> = Vec::new();
        lim.reserve_exact(&mut by_pre, tops.len())?;
        for (i, &t) in tops.iter().enumerate() {
            at[vt(t) as usize] = i as u32;
            let top = ng.top[ng.id(t)];
            by_pre.push((sh.pre[vt(t) as usize], sh.depth[top as usize], i as u32));
        }
        by_pre.sort_unstable();
        let m = by_pre.len();
        let mut least: Vec<Vec<u32>> = Vec::new();
        let mut row: Vec<u32> = Vec::new();
        lim.reserve_exact(&mut row, m)?;
        row.extend(0..m as u32);
        let mut span = 1;
        loop {
            let next_span = span * 2;
            let mut next: Vec<u32> = Vec::new();
            if next_span <= m {
                lim.reserve_exact(&mut next, m + 1 - next_span)?;
                for i in 0..=m - next_span {
                    let (a, b) = (row[i], row[i + span]);
                    next.push(if by_pre[b as usize].1 < by_pre[a as usize].1 { b } else { a });
                }
            }
            lim.try_push(&mut least, row)?;
            if next.is_empty() {
                break;
            }
            row = next;
            span = next_span;
        }
        Ok(Tops { at, by_pre, least })
    }

    /// The position of the least depth over `a..b`, nonempty.
    fn argmin(&self, a: usize, b: usize) -> usize {
        let j = (usize::BITS - 1 - (b - a).leading_zeros()) as usize;
        let (x, y) = (self.least[j][a], self.least[j][b - (1 << j)]);
        (if self.by_pre[y as usize].1 < self.by_pre[x as usize].1 { y } else { x }) as usize
    }

    /// Append the indices of the tops on leaves under `p` read at `p` or
    /// above: those whose highest level is at most `p`'s depth.
    fn under(&self, sh: &Shape, p: u32, out: &mut Vec<u32>, lim: &Limits) -> Result<(), OperationError> {
        let (lo, hi) = (sh.pre[p as usize], sh.pre[p as usize] + sh.size[p as usize]);
        let a = self.by_pre.partition_point(|e| e.0 < lo);
        let b = self.by_pre.partition_point(|e| e.0 < hi);
        let d = sh.depth[p as usize];
        let mut stack: Vec<(usize, usize)> = vec![(a, b)];
        while let Some((a, b)) = stack.pop() {
            if a >= b {
                continue;
            }
            let i = self.argmin(a, b);
            if self.by_pre[i].1 > d {
                continue;
            }
            lim.try_push(out, self.by_pre[i].2)?;
            stack.push((a, i));
            stack.push((i + 1, b));
        }
        Ok(())
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.at);
        lim.discard(self.by_pre);
        discard_nested(lim, self.least);
    }
}

const NIL: u32 = u32::MAX;

/// A true product of a reference of one operand: the other operand's
/// reference, the result, and the preorder position of the highest level
/// the other reference is read at ([`Numbered::top_pre`]). A product is read
/// at the levels both its references are read at; on the path from where it
/// was built up, a level is one of them where its position is at least the
/// two references' tops'.
type End = (Ref, Ref, u32);

/// One listed product ([`End`]) and the next entry of its list.
#[derive(Clone, Copy)]
struct Ent {
    other: Ref,
    res: Ref,
    top: u32,
    next: u32,
}

/// Per reference of one operand, its true products that the scans read by
/// that reference: linked entries in one arena whose freed entries are
/// reused.
struct Lists {
    head: Vec<u32>,
    ent: Vec<Ent>,
    free: u32,
}

impl Lists {
    fn new(n: usize, lim: &Limits) -> Result<Lists, OperationError> {
        Ok(Lists { head: filled(lim, n, NIL)?, ent: Vec::new(), free: NIL })
    }

    #[inline]
    fn push(&mut self, i: usize, (other, res, top): End, lim: &Limits) -> Result<(), OperationError> {
        let e = Ent { other, res, top, next: self.head[i] };
        let at = if self.free != NIL {
            let at = self.free;
            self.free = self.ent[at as usize].next;
            self.ent[at as usize] = e;
            at
        } else {
            lim.try_push(&mut self.ent, e)?;
            (self.ent.len() - 1) as u32
        };
        self.head[i] = at;
        Ok(())
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.head);
        lim.discard(self.ent);
    }

    /// The entries of list `i` that `judge` takes, into `ends`: `judge`
    /// gives, for the other reference and its top, whether the entry stays
    /// listed and whether it is taken. Entries that do not stay are freed.
    #[inline]
    fn scan(&mut self, i: usize, ends: &mut Vec<End>, lim: &Limits, judge: impl Fn(Ref, u32) -> (bool, bool)) -> Result<(), OperationError> {
        let (mut prev, mut e) = (NIL, self.head[i]);
        while e != NIL {
            let Ent { other, res, top, next } = self.ent[e as usize];
            let (stay, take) = judge(other, top);
            if stay {
                if take {
                    lim.try_push(ends, (other, res, top))?;
                }
                prev = e;
            } else {
                if prev == NIL {
                    self.head[i] = next;
                } else {
                    self.ent[prev as usize].next = next;
                }
                self.ent[e as usize].next = self.free;
                self.free = e;
            }
            e = next;
        }
        Ok(())
    }

    /// The result of the product of reference `i` with `other`, if listed.
    fn find(&self, i: usize, other: Ref) -> Option<Ref> {
        let mut e = self.head[i];
        while e != NIL {
            let Ent { other: o, res, next, .. } = self.ent[e as usize];
            if o == other {
                return Some(res);
            }
            e = next;
        }
        None
    }
}

/// The products a level builds from its own nodes of `f` that only the
/// level's parent reads (one of the two references is read no higher than
/// the level), by node, in blocks held last built first: a level's block is
/// added once the level is built and read once its parent is, and a read
/// block goes once every block above it has gone. Every other product is
/// listed ([`Lists`]).
struct Fresh {
    ent: Vec<End>,
    /// Per block, the start of each node's products in `ent` and the end.
    off: Vec<u32>,
    /// Per block: its level, the start of its offsets in `off`, and whether
    /// it has been read.
    blocks: Vec<(u32, u32, bool)>,
}

impl Fresh {
    /// The unread block of level `c`, if any.
    #[inline]
    fn block(&self, c: u32) -> Option<usize> {
        self.blocks.iter().rposition(|&(t, _, read)| t == c && !read)
    }

    /// The products of node `i` of block `b`.
    #[inline]
    fn of(&self, b: usize, i: u32) -> &[End] {
        let o = (self.blocks[b].1 + i) as usize;
        &self.ent[self.off[o] as usize..self.off[o + 1] as usize]
    }

    /// Mark block `b` read and drop the read blocks on top.
    fn read(&mut self, b: usize) {
        self.blocks[b].2 = true;
        while let Some(&(_, o, true)) = self.blocks.last() {
            self.ent.truncate(self.off[o as usize] as usize);
            self.off.truncate(o as usize);
            self.blocks.pop();
        }
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.ent);
        lim.discard(self.off);
        lim.discard(self.blocks);
    }
}

/// The fewest pairs of `g` a walk by anchor meets for one pair of `f`
/// before that pair lists its other side's products: the walk by other
/// side is weighed against the walk by anchor, and either looks products
/// up in the list rather than in a map. Below it a map lookup per pair
/// costs less than the list.
const BY_OTHER_FROM: usize = 16;

/// One level's pairs of `g` by their other-side reference: linked entries
/// of the node and the anchor-side reference, with each list's length;
/// the other-side references that are `true`; and the pairs by both
/// sides' numbers, linked entries of the node.
struct ByOther {
    built: bool,
    mark: Vec<u32>,
    head: Vec<u32>,
    len: Vec<u32>,
    ent: Vec<(u32, Ref, u32)>,
    ones: Vec<Ref>,
    cell: FxHashMap<u64, u32>,
    cent: Vec<(u32, u32)>,
}

impl ByOther {
    fn new(n: usize, lim: &Limits) -> Result<ByOther, OperationError> {
        Ok(ByOther {
            built: false,
            mark: filled(lim, n, 0)?,
            head: filled(lim, n, NIL)?,
            len: filled(lim, n, 0)?,
            ent: Vec::new(),
            ones: Vec::new(),
            cell: FxHashMap::default(),
            cent: Vec::new(),
        })
    }

    /// Index the pairs of level `gl` (`split` puts the anchor side first),
    /// stamped `stamp`.
    fn build(&mut self, sh: &Shape, gl: &super::SLevel, ng: &Numbered, split: impl Fn((Ref, Ref)) -> (Ref, Ref), stamp: u32, lim: &Limits) -> Result<(), OperationError> {
        self.ent.clear();
        self.ones.clear();
        self.cell.clear();
        self.cent.clear();
        lim.reserve_map(&mut self.cell, gl.pairs.len())?;
        for j in 0..gl.width() as u32 {
            for &p in gl.node(j) {
                let (y, ob) = split(p);
                let id = ng.id(ob);
                let h = self.cell.entry(cell_key(ng.id(y), id)).or_insert(NIL);
                lim.try_push(&mut self.cent, (j, *h))?;
                *h = (self.cent.len() - 1) as u32;
                if self.mark[id] != stamp {
                    self.mark[id] = stamp;
                    self.head[id] = NIL;
                    self.len[id] = 0;
                    if is_one(sh, ob) {
                        lim.try_push(&mut self.ones, ob)?;
                    }
                }
                lim.try_push(&mut self.ent, (j, y, self.head[id]))?;
                self.head[id] = (self.ent.len() - 1) as u32;
                self.len[id] += 1;
            }
        }
        self.built = true;
        Ok(())
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.mark);
        lim.discard(self.head);
        lim.discard(self.len);
        lim.discard(self.ent);
        lim.discard(self.ones);
        lim.discard(self.cell);
        lim.discard(self.cent);
    }
}

/// The key of an (anchor, other side) cell by the two references' numbers.
#[inline]
fn cell_key(y: usize, ob: usize) -> u64 {
    ((y as u64) << 32) | ob as u64
}

/// A product to build: its two references and the last of its pair
/// combinations in the node's list of them ([`combo`]).
type Key = (Ref, Ref, u32);

/// A new key for the product of `a` and `b`, without pair combinations.
#[inline]
fn key_of(keys: &mut Vec<Key>, a: Ref, b: Ref, lim: &Limits) -> Result<u32, OperationError> {
    lim.try_push(keys, (a, b, NIL))?;
    Ok((keys.len() - 1) as u32)
}

/// Add the pair combination `(pl, pr)` to key `k`: `combos` links each
/// key's combinations, last first.
#[inline]
fn combo(keys: &mut [Key], combos: &mut Vec<(u32, Ref, Ref)>, k: u32, (pl, pr): (Ref, Ref), lim: &Limits) -> Result<(), OperationError> {
    let last = &mut keys[k as usize].2;
    lim.try_push(combos, (*last, pl, pr))?;
    *last = (combos.len() - 1) as u32;
    Ok(())
}

/// Build one node's products (`keys`, their pair combinations in `combos`)
/// at internal `u` into `lvl`, the result's level there, noting each with
/// its result in `made`; empties `keys` and `combos`. The work units they
/// took.
#[inline]
#[allow(clippy::too_many_arguments)]
fn emit(
    sh: &Shape,
    u: u32,
    lvl: &mut super::SLevel,
    keys: &mut Vec<Key>,
    combos: &mut Vec<(u32, Ref, Ref)>,
    buf: &mut Vec<(Ref, Ref)>,
    made: &mut Vec<(Ref, Ref, Ref)>,
    lim: &Limits,
) -> Result<u64, OperationError> {
    if keys.is_empty() {
        return Ok(0);
    }
    let work = (keys.len() + combos.len()) as u64;
    for &(a, b, last) in keys.iter() {
        buf.clear();
        let mut e = last;
        while e != NIL {
            let (next, pl, pr) = combos[e as usize];
            lim.try_push(buf, (pl, pr))?;
            e = next;
        }
        let made_here = if sh.weighted[u as usize] { normalize_sum(sh, u, buf) } else { normalize(sh, u, buf) };
        let res = match made_here {
            Err(t) => t,
            Ok(()) => mk(u, lvl.push(buf, lim)?),
        };
        lim.try_push(made, (a, b, res))?;
    }
    keys.clear();
    combos.clear();
    Ok(work)
}

/// An operand's references numbered densely: stored node `i` of level `t`
/// is `base[t] + i`, leaf label `k` of leaf `c` is `base[c] + k`; with, per
/// number, the highest level the reference is read at: the child level of
/// the highest level whose pairs name it, or the root for the output. A
/// reference is read at every level from there down to its own.
pub(super) struct Numbered {
    pub(super) base: Vec<u32>,
    pub(super) top: Vec<u32>,
    /// The leaves whose labels are numbered.
    pub(super) leaves: Vec<u32>,
}

impl Numbered {
    pub(super) fn of(s: &SkipForm, lim: &Limits) -> Result<Numbered, OperationError> {
        let all = vec![true; s.shape.num_nodes()];
        Self::of_cone(s, &all, lim)
    }

    /// Number only the references of `s` at levels in `cone`, and those
    /// named from `s.out` or `s`'s levels in `cone`: since `cone` holds every
    /// ancestor of its levels, the highest level where any such reference is
    /// read in `s` is attained at `s.out` or at a level in `cone`.
    pub(super) fn of_cone(s: &SkipForm, cone: &[bool], lim: &Limits) -> Result<Numbered, OperationError> {
        let sh = &*s.shape;
        let mut base = vec![NONE; sh.num_nodes()];
        let mut n = 0u32;
        for &t in &s.active {
            if cone[t as usize] {
                base[t as usize] = n;
                n += s.levels[t as usize].width() as u32;
            }
        }
        let mut top = Vec::new();
        lim.try_resize(&mut top, n as usize, NONE)?;
        let mut nb = Numbered { base, top, leaves: Vec::new() };
        nb.note(s, s.out, sh.root, lim)?;
        for &u in &s.active {
            if cone[u as usize] {
                let (l, r) = sh.kids[u as usize];
                for &(a, b) in &s.levels[u as usize].pairs {
                    nb.note(s, a, l, lim)?;
                    nb.note(s, b, r, lim)?;
                }
            }
        }
        Ok(nb)
    }

    pub(super) fn discard(self, lim: &Limits) {
        lim.discard(self.top);
    }

    #[inline]
    fn note(&mut self, s: &SkipForm, x: Ref, c: u32, lim: &Limits) -> Result<(), OperationError> {
        let sh = &*s.shape;
        let v = vt(x) as usize;
        if self.base[v] == NONE {
            self.base[v] = self.top.len() as u32;
            let w = if sh.is_leaf(v as u32) {
                self.leaves.push(v as u32);
                sh.labels[v] as usize
            } else {
                s.levels[v].width()
            };
            let new_len = self.top.len() + w;
            lim.try_resize_amortized(&mut self.top, new_len, NONE)?;
        }
        let t = &mut self.top[(self.base[v] + ix(x)) as usize];
        if *t == NONE || sh.pre[c as usize] < sh.pre[*t as usize] {
            *t = c;
        }
        Ok(())
    }

    pub(super) fn len(&self) -> usize {
        self.top.len()
    }

    /// The number of `x`, a reference of the operand.
    #[inline]
    pub(super) fn id(&self, x: Ref) -> usize {
        (self.base[vt(x) as usize] + ix(x)) as usize
    }

    /// The preorder position of the highest level number `id` is read at,
    /// `u32::MAX` where it is read nowhere. Every level a reference is read
    /// at is an ancestor of its own, so on the path from its level up it is
    /// read exactly where a level's position is at least this.
    #[inline]
    fn top_pre(&self, sh: &Shape, id: usize) -> u32 {
        match self.top[id] {
            NONE => u32::MAX,
            t => sh.pre[t as usize],
        }
    }

    /// Whether `x` is read at level `c`.
    #[inline]
    fn read_at(&self, sh: &Shape, x: Ref, c: u32) -> bool {
        let b = self.base[vt(x) as usize];
        if b == NONE || !sh.within(vt(x), c) {
            return false;
        }
        let t = self.top[(b + ix(x)) as usize];
        t != NONE && sh.within(c, t)
    }

    /// Whether `x`, read at a child of `u`, is read at `u` too: it skips
    /// through `u`.
    #[inline]
    fn above(&self, sh: &Shape, x: Ref, u: u32) -> bool {
        let t = self.top[self.id(x)];
        t != NONE && sh.within(u, t)
    }

    /// The leaf references on leaf `c` the operand reads at level `u` above it.
    fn leaf_refs<'a>(&'a self, sh: &'a Shape, c: u32, u: u32) -> impl Iterator<Item = Ref> + 'a {
        (0..sh.labels[c as usize]).map(move |lab| mk(c, lab)).filter(move |&x| self.read_at(sh, x, u))
    }
}

/// `f ∧ g`, pruned, in skip form; not contracted.
///
/// Built from the leaves up, as the stored conjunction is: a product exists
/// once one of its pair combinations has both child products true
/// somewhere, so false products are never expanded. The products of a
/// level are found from those of one inner child (the anchor): for a pair
/// of `f` at the level naming `x` there, each true product `x ∧ y` whose
/// `y` the level also reads gives the products of the level's nodes of `g`
/// naming `y` there, and of `y` itself where it skips through the level;
/// the products of skips of `f` with nodes of `g` come the same way from
/// `g`'s pairs. A product whose other reference is no longer read is
/// dropped from its reference's list when a scan meets it. Products no
/// output path reaches are dropped at the end.
pub(crate) fn and(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let (out, cone) = and_scoped(f, g, lim)?;
    lim.discard(cone);
    Ok(out)
}

/// [`and`] returning the upward-closed cone of levels that changed (those in
/// the smaller operand's cone and any level outside it that lost a node when
/// unreachable branches were dropped).
///
/// With nothing summed out this is the overlay's walk
/// ([`super::overlay::conjoin`]), which keeps the larger operand's nodes
/// where they are their own products. Under sums the classes of a level
/// overlap, a node being a multiset of products, and the products are built
/// from pair combinations as described above.
pub(crate) fn and_scoped(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<(SkipForm, Vec<bool>), OperationError> {
    // Shapes read from one vtree are equal; `f`'s serves both.
    debug_assert_eq!(f.shape.kids, g.shape.kids);
    let shape = Arc::clone(&f.shape);
    let sh = &*shape;
    let mut out = SkipForm::zero(Arc::clone(&shape));
    match land_pair(sh, f.out, g.out) {
        Land::Done(r) => {
            out.out = r;
            let cone = filled(lim, sh.num_nodes(), false)?;
            return Ok((out, cone));
        }
        Land::At(_) => {}
    }
    if !sh.is_weighted() {
        return super::overlay::conjoin(f, g, lim);
    }
    let ng = Numbered::of(g, lim)?;
    // Outside `g`'s cone `g` is `true`: there the result is `f`'s own levels,
    // and a product of a reference of `f` with `true` is that reference.
    let mut cone = cone_of(sh, g, lim)?;
    let nf = Numbered::of_cone(f, &cone, lim)?;
    let free = |x: Ref| !cone[vt(x) as usize];
    let Outside { seeds, boundary, mark: mut mark_out } = outside(sh, f, &nf, &ng, &cone, lim)?;
    // The levels in the cone products can land on, children first.
    let mut todo: Vec<u32> = Vec::new();
    lim.reserve(&mut todo, f.active.len() + g.active.len() + nf.leaves.len() + ng.leaves.len())?;
    todo.extend(f.active.iter().chain(&g.active).copied().filter(|&t| cone[t as usize]));
    for nb in [&nf, &ng] {
        for &c in &nb.leaves {
            let p = sh.parent[c as usize];
            if p != NONE && cone[p as usize] && nb.leaf_refs(sh, c, p).next().is_some() {
                todo.push(p);
            }
        }
    }
    todo.sort_unstable_by_key(|&t| std::cmp::Reverse(sh.pre[t as usize]));
    todo.dedup();
    let mut next_seed = 0;
    let mut st = AndSt::new(&nf, &ng, lim)?;
    let cx = AndCx { sh, f, g, nf: &nf, ng: &ng, cone: &cone, lim };
    for &u in &todo {
        // The products with `true` the levels below would have listed.
        while next_seed < seeds.len() && seeds[next_seed].0 > sh.pre[u as usize] {
            let (_, x, t) = seeds[next_seed];
            let (fx, gt) = (nf.id(x), ng.id(t));
            st.pf.push(fx, (t, x, ng.top_pre(sh, gt)), lim)?;
            st.pg.push(gt, (x, x, nf.top_pre(sh, fx)), lim)?;
            next_seed += 1;
        }
        and_level(&cx, &mut st, &mut out, u)?;
    }
    out.out = match land_pair(sh, f.out, g.out) {
        Land::Done(r) => r,
        Land::At(_) if is_one(sh, g.out) && free(f.out) => f.out,
        Land::At(_) => st.product(&nf, f.out, g.out).unwrap_or(ZREF),
    };
    lim.discard(seeds);
    lim.discard(todo);
    st.discard(lim);
    ng.discard(lim);
    out.collect_pushed(lim)?;
    if out.out == ZREF {
        lim.discard(boundary);
        lim.discard(mark_out);
        nf.discard(lim);
        return Ok((out, cone));
    }
    share_outside(sh, f, &mut out, &mut cone, &boundary, &mut mark_out, |x| nf.id(x), lim)?;
    lim.discard(boundary);
    lim.discard(mark_out);
    nf.discard(lim);
    Ok((out, cone))
}

/// The end of a conjunction whose levels in `cone` are built and collected:
/// `f`'s levels beyond the cone, shared where the result reaches one of
/// `f`'s references beyond it (`boundary`, the nodes a pair in the cone or
/// the output names, each marked 1 in `mark` by its number `id`), and
/// collected where it reaches only some, whose levels that lost a node join
/// `cone`.
#[allow(clippy::too_many_arguments)]
pub(super) fn share_outside(
    sh: &Shape,
    f: &SkipForm,
    out: &mut SkipForm,
    cone: &mut [bool],
    boundary: &[Ref],
    mark: &mut [u8],
    id: impl Fn(Ref) -> usize,
    lim: &Limits,
) -> Result<(), OperationError> {
    let f_outside = boundary.len();
    let mut out_outside = 0usize;
    let mut note_o = |x: Ref| {
        if x != ZREF && !cone[vt(x) as usize] && !sh.is_leaf(vt(x)) {
            let k = id(x);
            if mark[k] == 1 {
                mark[k] = 2;
                out_outside += 1;
            }
        }
    };
    note_o(out.out);
    for &u in &out.active {
        for &(a, b) in &out.levels[u as usize].pairs {
            note_o(a);
            note_o(b);
        }
    }
    if out_outside > 0 {
        for &t in &f.active {
            if !cone[t as usize] {
                out.levels.share(t as usize, &f.levels, lim)?;
                out.active.push(t);
            }
        }
        if out_outside == f_outside {
            out.active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
        } else {
            let mut dirty = filled(lim, sh.num_nodes(), false)?;
            dirty.copy_from_slice(cone);
            let mut dropped_roots = filled(lim, sh.num_nodes(), false)?;
            for &x in boundary {
                if mark[id(x)] == 1 {
                    let v = vt(x);
                    dropped_roots[v as usize] = true;
                    let mut u = v;
                    while u != NONE && !dirty[u as usize] {
                        dirty[u as usize] = true;
                        u = sh.parent[u as usize];
                    }
                }
            }
            for &t in &f.active {
                if !dirty[t as usize] {
                    let mut u = t;
                    while u != NONE && !cone[u as usize] {
                        if dropped_roots[u as usize] {
                            dirty[t as usize] = true;
                            break;
                        }
                        u = sh.parent[u as usize];
                    }
                }
            }
            lim.discard(dropped_roots);
            let cand: Vec<u32> = out.active.iter().copied().filter(|&t| dirty[t as usize]).collect();
            out.collect_in(cand, lim)?;
            for &t in &f.active {
                if !cone[t as usize] && dirty[t as usize] && out.levels[t as usize].width() < f.levels[t as usize].width() {
                    let mut u = t;
                    while u != NONE && !cone[u as usize] {
                        cone[u as usize] = true;
                        u = sh.parent[u as usize];
                    }
                }
            }
            lim.discard(dirty);
        }
    }
    Ok(())
}

/// The conjunction's immutable context at every level.
struct AndCx<'a> {
    sh: &'a Shape,
    f: &'a SkipForm,
    g: &'a SkipForm,
    nf: &'a Numbered,
    ng: &'a Numbered,
    cone: &'a [bool],
    lim: &'a Limits,
}

/// The conjunction's state carried from level to level.
struct AndSt {
    // True products: those only a level's parent reads, by the level's
    // nodes of `f`; the others per reference of `f`; and those whose
    // reference of `f` is read above where they were built, per reference
    // of `g`.
    fresh: Fresh,
    pf: Lists,
    pg: Lists,
    // Both references to the result, where a level looks products up so.
    live: FxHashMap<u64, Ref>,
    // `g`'s pairs of the current level by their anchor-side reference: one
    // run each in `gent`, ending at `ghead`, of `glen` pairs.
    gmark: Vec<u32>,
    ghead: Vec<u32>,
    glen: Vec<u32>,
    gent: Vec<(u32, Ref)>,
    // The same pairs by their other-side reference, built where a level
    // first needs them; the `true`s among those references; and the
    // anchor-side products of the current pair of `f`, and the other-side
    // ones, where the pair listed them, with the references met.
    by_other: ByOther,
    ymark: Vec<(u32, Ref)>,
    omark: Vec<(u32, Ref)>,
    pstamp: u32,
    qstamp: u32,
    oends: Vec<End>,
    qbuf: Vec<Ref>,
    // The current node's products, by the other side's reference.
    cg: Vec<(u32, u32)>,
    cf: Vec<(u32, u32)>,
    lstamp: u32,
    cstamp: u32,
    // One node's products against the other side and their pair
    // combinations, built as soon as the node is done; the level's
    // products, listed once the level is done.
    keys: Vec<Key>,
    combos: Vec<(u32, Ref, Ref)>,
    made: Vec<(Ref, Ref, Ref)>,
    ends: Vec<End>,
    buf: Vec<(Ref, Ref)>,
}

impl AndSt {
    fn new(nf: &Numbered, ng: &Numbered, lim: &Limits) -> Result<AndSt, OperationError> {
        Ok(AndSt {
            fresh: Fresh { ent: Vec::new(), off: Vec::new(), blocks: Vec::new() },
            pf: Lists::new(nf.len(), lim)?,
            pg: Lists::new(ng.len(), lim)?,
            live: FxHashMap::default(),
            gmark: filled(lim, ng.len(), 0u32)?,
            ghead: filled(lim, ng.len(), NIL)?,
            glen: filled(lim, ng.len(), 0u32)?,
            gent: Vec::new(),
            by_other: ByOther::new(ng.len(), lim)?,
            ymark: filled(lim, ng.len(), (0u32, ZREF))?,
            omark: filled(lim, ng.len(), (0u32, ZREF))?,
            pstamp: 0,
            qstamp: 0,
            oends: Vec::new(),
            qbuf: Vec::new(),
            cg: filled(lim, ng.len(), (0u32, 0u32))?,
            cf: filled(lim, nf.len(), (0u32, 0u32))?,
            lstamp: 0,
            cstamp: 0,
            keys: Vec::new(),
            combos: Vec::new(),
            made: Vec::new(),
            ends: Vec::new(),
            buf: Vec::new(),
        })
    }
}

impl AndSt {
    /// The product of `x`, a reference of `f`, with `y`, a reference of
    /// `g`, if it is true and was built.
    fn product(&self, nf: &Numbered, x: Ref, y: Ref) -> Option<Ref> {
        let fresh = self.fresh.block(vt(x)).and_then(|b| self.fresh.of(b, ix(x)).iter().find(|e| e.0 == y));
        fresh.map(|e| e.1).or_else(|| self.pf.find(nf.id(x), y))
    }

    /// Drop the state, handing the charge for its tables back to `lim`.
    fn discard(self, lim: &Limits) {
        let AndSt { fresh, pf, pg, live, gmark, ghead, glen, gent, by_other, ymark, omark, oends, qbuf, cg, cf, keys, combos, made, ends, buf, .. } = self;
        pf.discard(lim);
        pg.discard(lim);
        lim.discard(live);
        for v in [gmark, ghead, glen] {
            lim.discard(v);
        }
        lim.discard(ymark);
        lim.discard(omark);
        by_other.discard(lim);
        lim.discard(oends);
        lim.discard(qbuf);
        for v in [cg, cf] {
            lim.discard(v);
        }
        lim.discard(gent);
        lim.discard(keys);
        lim.discard(combos);
        lim.discard(made);
        lim.discard(ends);
        lim.discard(buf);
        fresh.discard(lim);
    }
}

/// Build the conjunction's level `u` into `out` and list its products, the
/// levels below it done.
fn and_level(cx: &AndCx<'_>, st: &mut AndSt, out: &mut SkipForm, u: u32) -> Result<(), OperationError> {
    let AndCx { sh, f, g, nf, ng, cone, lim } = *cx;
    let free = |x: Ref| !cone[vt(x) as usize];
    let key = |a: usize, b: usize| ((a as u64) << 32) | b as u64;
    let (mut pstamp, mut qstamp, mut lstamp, mut cstamp) = (st.pstamp, st.qstamp, st.lstamp, st.cstamp);
    let AndSt { fresh, pf, pg, live, gmark, ghead, glen, gent, by_other, ymark, omark, oends, qbuf, cg, cf, keys, combos, made, ends, buf, .. } = st;
    let mut fb = [(0, 0); 1];
    let mut gb = [(0, 0); 1];
    let (l, r) = sh.kids[u as usize];
    made.clear();
    let mut work = 0u64;
    let (fl, gl) = (&f.levels[u as usize], &g.levels[u as usize]);
    // The result's level, built here alone.
    let mut lvl = out.levels.take(u as usize);
    debug_assert_eq!(lvl.width(), 0, "a level is built once");
    let pu = sh.pre[u as usize];
    if sh.is_leaf(l) && sh.is_leaf(r) {
        // Every reference each side reads here, against every other.
        let refs = |s: &SkipForm, nb: &Numbered| -> Result<Vec<Ref>, OperationError> {
            let mut v: Vec<Ref> = Vec::new();
            lim.reserve_exact(&mut v, s.levels[u as usize].width() + (sh.labels[l as usize] + sh.labels[r as usize]) as usize)?;
            v.extend((0..s.levels[u as usize].width() as u32).map(|i| mk(u, i)));
            v.extend(nb.leaf_refs(sh, l, u));
            v.extend(nb.leaf_refs(sh, r, u));
            Ok(v)
        };
        let (fr, gr) = (refs(f, nf)?, refs(g, ng)?);
        for &a in &fr {
            for &b in &gr {
                if vt(a) != u && vt(a) == vt(b) {
                    continue;
                }
                let mut cell = NIL;
                for &(la, ra) in f.pairs_at(u, a, &mut fb) {
                    for &(lb, rb) in g.pairs_at(u, b, &mut gb) {
                        let (Land::Done(x), Land::Done(y)) = (land_pair(sh, la, lb), land_pair(sh, ra, rb)) else {
                            unreachable!("leaf products are known")
                        };
                        if x != ZREF && y != ZREF {
                            if cell == NIL {
                                cell = key_of(&mut *keys, a, b, lim)?;
                            }
                            combo(&mut *keys, &mut *combos, cell, (x, y), lim)?;
                        }
                    }
                }
                work += emit(sh, u, &mut lvl, &mut *keys, &mut *combos, &mut *buf, &mut *made, lim)?;
            }
        }
        lim.discard(fr);
        lim.discard(gr);
    } else {
        // The anchor: the right child unless it is a leaf.
        let right = !sh.is_leaf(r);
        let (cs, co) = if right { (r, l) } else { (l, r) };
        let split = |p: (Ref, Ref)| if right { (p.1, p.0) } else { (p.0, p.1) };
        let join = |anchor: Ref, other: Ref| if right { (other, anchor) } else { (anchor, other) };
        let skips = sh.is_leaf(co);
        let at = |r: Ref| vt(r) == cs;
        let (pcs, pco) = (sh.pre[cs as usize], sh.pre[co as usize]);
        // The children's products by node.
        let (bcs, bco) = (fresh.block(cs), if skips { None } else { fresh.block(co) });
        // The product of `oa` with `ob`: known, or looked up in `omark`
        // where the current pair listed `oa`'s products (stamped
        // `listed`), and in `live` otherwise.
        let other = |live: &FxHashMap<u64, Ref>, omark: &[(u32, Ref)], listed: Option<u32>, oa: Ref, ob: Ref| -> Ref {
            if skips {
                // Both are labels of the leaf `co`.
                return leaf_and(co, ix(oa), ix(ob));
            }
            if is_one(sh, ob) && free(oa) {
                return oa;
            }
            match (land_pair(sh, oa, ob), listed) {
                (Land::Done(x), _) => x,
                (Land::At(_), Some(st)) => match omark[ng.id(ob)] {
                    (s, x) if s == st => x,
                    _ => ZREF,
                },
                (Land::At(_), None) => live.get(&key(nf.id(oa), ng.id(ob))).copied().unwrap_or(ZREF),
            }
        };
        // `g`'s pairs here by anchor-side reference, one run each in
        // `gent`: counted (stamped `seen`), then laid out in the order the
        // references are first met and filled (stamped `lstamp`); a run
        // ends at `ghead`.
        lstamp += 2;
        let seen = lstamp - 1;
        by_other.built = false;
        for &p in &gl.pairs {
            let gy = ng.id(split(p).0);
            if gmark[gy] != seen {
                gmark[gy] = seen;
                glen[gy] = 0;
            }
            glen[gy] += 1;
        }
        gent.clear();
        lim.try_resize_amortized(&mut *gent, gl.pairs.len(), (0, ZREF))?;
        let mut runs = 0u32;
        for j in 0..gl.width() as u32 {
            for &p in gl.node(j) {
                let (y, ob) = split(p);
                let gy = ng.id(y);
                if gmark[gy] == seen {
                    gmark[gy] = lstamp;
                    ghead[gy] = runs;
                    runs += glen[gy];
                }
                gent[ghead[gy] as usize] = (j, ob);
                ghead[gy] += 1;
            }
        }
        let gbase = ng.base[u as usize];
        for i in 0..fl.width() as u32 {
            let a = mk(u, i);
            cstamp += 1;
            for &p in fl.node(i) {
                let (x, oa) = split(p);
                // The true products of `x` with references of `g` read
                // at the anchor: by node where only this level reads
                // them (`fs`), and those listed that are still read here,
                // every one built at or below the anchor.
                let fs: &[End] = match (bcs, at(x)) {
                    (Some(b), true) => fresh.of(b, ix(x)),
                    _ => &[],
                };
                ends.clear();
                pf.scan(nf.id(x), &mut *ends, lim, |y, t| {
                    let keep = t <= pcs;
                    debug_assert_eq!(keep, ng.read_at(sh, y, cs));
                    (keep, keep)
                })?;
                if sh.is_leaf(vt(x)) {
                    for y in ng.leaf_refs(sh, vt(x), cs) {
                        let res = leaf_and(vt(x), ix(x), ix(y));
                        if res != ZREF {
                            lim.try_push(&mut *ends, (y, res, ng.top_pre(sh, ng.id(y))))?;
                        }
                    }
                }
                // `g`'s pairs naming one of `ends` at the anchor, walked
                // by anchor; or, where that walk is long, by other side,
                // from the references `oa` has a product with, or by
                // cell, each of `ends` against each of those, whichever
                // is shortest.
                // A pair combination of the product of `a` with `b`
                // (numbered `cell` in `g`).
                let mut push = |cell: usize, b: Ref, pair: (Ref, Ref), keys: &mut Vec<Key>, combos: &mut Vec<(u32, Ref, Ref)>| -> Result<(), OperationError> {
                    let c = &mut cg[cell];
                    if c.0 != cstamp {
                        *c = (cstamp, key_of(keys, a, b, lim)?);
                    }
                    combo(keys, combos, c.1, pair, lim)
                };
                // Against a leaf the walk is by anchor.
                let by_anchor: usize = if skips {
                    0
                } else {
                    fs.iter()
                        .chain(ends.iter())
                        .map(|&(y, _, _)| {
                            let gy = ng.id(y);
                            if gmark[gy] == lstamp { glen[gy] as usize } else { 0 }
                        })
                        .sum()
                };
                let (mut walk_other, mut walk_cells) = (false, false);
                let listed = !skips && by_anchor >= BY_OTHER_FROM;
                if listed {
                    if !by_other.built {
                        by_other.build(sh, gl, ng, split, lstamp, lim)?;
                    }
                    // The other-side references `oa` may have a product
                    // with, and the product: those listed, `true` where
                    // `oa` is outside `g`'s cone, and the labels of `oa`'s
                    // own leaf.
                    let ofs: &[End] = match (bco, vt(oa) == co) {
                        (Some(b), true) => fresh.of(b, ix(oa)),
                        _ => &[],
                    };
                    oends.clear();
                    pf.scan(nf.id(oa), &mut *oends, lim, |ob, t| {
                        let keep = t <= pco;
                        debug_assert_eq!(keep, ng.read_at(sh, ob, co));
                        (keep, keep)
                    })?;
                    qbuf.clear();
                    qstamp += 1;
                    let mut cost = 0usize;
                    let mut meet = |ob: Ref, o: Ref, qbuf: &mut Vec<Ref>| -> Result<(), OperationError> {
                        if ng.base[vt(ob) as usize] == NONE {
                            return Ok(());
                        }
                        let id = ng.id(ob);
                        if omark[id].0 != qstamp {
                            omark[id] = (qstamp, o);
                            if by_other.mark[id] == lstamp {
                                cost += 1 + by_other.len[id] as usize;
                                lim.try_push(qbuf, ob)?;
                            }
                        }
                        Ok(())
                    };
                    for &(ob, res, _) in ofs.iter().chain(oends.iter()) {
                        meet(ob, res, &mut *qbuf)?;
                    }
                    // `other` knows these products before any lookup.
                    if free(oa) {
                        for &ob in &by_other.ones {
                            meet(ob, oa, &mut *qbuf)?;
                        }
                    }
                    if sh.is_leaf(vt(oa)) {
                        for k in 0..sh.labels[vt(oa) as usize] {
                            meet(mk(vt(oa), k), ZREF, &mut *qbuf)?;
                        }
                    }
                    // Or by cell, each anchor against each other side: a
                    // lookup in the level's small map costs about a step
                    // of a walk.
                    walk_cells = (fs.len() + ends.len()) * qbuf.len() < cost.min(by_anchor);
                    walk_other = !walk_cells && cost < by_anchor;
                }
                if walk_cells {
                    for &ob in qbuf.iter() {
                        let o = other(live, omark, Some(qstamp), oa, ob);
                        debug_assert_eq!(o, other(live, omark, None, oa, ob));
                        if o == ZREF {
                            continue;
                        }
                        let gob = ng.id(ob);
                        for &(y, res, _) in fs.iter().chain(ends.iter()) {
                            let Some(&h) = by_other.cell.get(&cell_key(ng.id(y), gob)) else {
                                continue;
                            };
                            let mut e = h;
                            while e != NIL {
                                let (j, next) = by_other.cent[e as usize];
                                e = next;
                                push((gbase + j) as usize, mk(u, j), join(res, o), &mut *keys, &mut *combos)?;
                            }
                        }
                    }
                }
                if walk_other {
                    pstamp += 1;
                    for &(y, res, _) in fs.iter().chain(ends.iter()) {
                        ymark[ng.id(y)] = (pstamp, res);
                    }
                    for &ob in qbuf.iter() {
                        let o = other(live, omark, Some(qstamp), oa, ob);
                        debug_assert_eq!(o, other(live, omark, None, oa, ob));
                        if o == ZREF {
                            continue;
                        }
                        let mut e = by_other.head[ng.id(ob)];
                        while e != NIL {
                            let (j, y, next) = by_other.ent[e as usize];
                            e = next;
                            let (st, res) = ymark[ng.id(y)];
                            if st == pstamp {
                                push((gbase + j) as usize, mk(u, j), join(res, o), &mut *keys, &mut *combos)?;
                            }
                        }
                    }
                }
                for &(y, res, ty) in fs.iter().chain(ends.iter()) {
                    let gy = ng.id(y);
                    if !walk_other && !walk_cells && gmark[gy] == lstamp {
                        let end = ghead[gy] as usize;
                        for &(j, ob) in gent[end - glen[gy] as usize..end].iter().rev() {
                            let o = other(live, omark, listed.then_some(qstamp), oa, ob);
                            debug_assert_eq!(o, other(live, omark, None, oa, ob));
                            if o != ZREF {
                                push((gbase + j) as usize, mk(u, j), join(res, o), &mut *keys, &mut *combos)?;
                            }
                        }
                    }
                    if skips && ty <= pu {
                        // `y` skips through this level, `true` on the
                        // other side.
                        debug_assert!(ng.above(sh, y, u));
                        push(gy, y, join(res, oa), &mut *keys, &mut *combos)?;
                    }
                }
            }
            work += emit(sh, u, &mut lvl, &mut *keys, &mut *combos, &mut *buf, &mut *made, lim)?;
        }
        if skips {
            // Skips of `f` through this level against `g`'s nodes.
            for j in 0..gl.width() as u32 {
                let b = mk(u, j);
                cstamp += 1;
                for &p in gl.node(j) {
                    let (y, ob) = split(p);
                    ends.clear();
                    pg.scan(ng.id(y), &mut *ends, lim, |x, t| {
                        let keep = t <= pcs;
                        debug_assert_eq!(keep, nf.read_at(sh, x, cs));
                        debug_assert_eq!(keep && t <= pu, keep && nf.above(sh, x, u));
                        (keep, keep && t <= pu)
                    })?;
                    if sh.is_leaf(vt(y)) {
                        for x in nf.leaf_refs(sh, vt(y), u) {
                            let res = leaf_and(vt(y), ix(x), ix(y));
                            if res != ZREF {
                                lim.try_push(&mut *ends, (x, res, nf.top_pre(sh, nf.id(x))))?;
                            }
                        }
                    }
                    for &(x, res, _) in ends.iter() {
                        let c = &mut cf[nf.id(x)];
                        if c.0 != cstamp {
                            *c = (cstamp, key_of(&mut *keys, x, b, lim)?);
                        }
                        combo(&mut *keys, &mut *combos, c.1, join(res, ob), lim)?;
                    }
                }
                work += emit(sh, u, &mut lvl, &mut *keys, &mut *combos, &mut *buf, &mut *made, lim)?;
            }
        }
        // The children's products by node are read.
        for b in [bcs, bco].into_iter().flatten() {
            fresh.read(b);
        }
    }
    if lvl.width() > 0 {
        out.active.push(u);
        out.levels.set(u as usize, lvl);
    }
    lim.charge_work(work);
    lim.check_stop()?;
    let lookup = sh.looked_up[u as usize];
    if lookup {
        lim.reserve_map(&mut *live, made.len())?;
    }
    // Each product goes by node where it is built from a node of `f` here
    // and one of its references is read no higher, so that only the parent
    // reads it; to the lists of `f`'s reference otherwise; and to those of
    // `g`'s reference too where `f`'s is read above this level, the one way
    // a level above reads it by `g`'s.
    let (w, o0) = (fl.width(), fresh.off.len());
    lim.try_resize_amortized(&mut fresh.off, o0 + w + 1, 0)?;
    let e0 = fresh.ent.len() as u32;
    let mut next = 0usize;
    for &(a, b, res) in made.iter() {
        let (fa, gb) = (nf.id(a), ng.id(b));
        let (ta, tb) = (nf.top_pre(sh, fa), ng.top_pre(sh, gb));
        debug_assert!(ta <= pu && tb <= pu, "a product is read where it is built");
        if vt(a) == u && (ta == pu || tb == pu) {
            let i = ix(a) as usize;
            debug_assert!(next <= i + 1, "a level's products are built node by node");
            while next <= i {
                fresh.off[o0 + next] = fresh.ent.len() as u32;
                next += 1;
            }
            lim.try_push(&mut fresh.ent, (b, res, tb))?;
        } else {
            pf.push(fa, (b, res, tb), lim)?;
        }
        if ta < pu {
            pg.push(gb, (a, res, ta), lim)?;
        }
        if lookup {
            live.insert(key(fa, gb), res);
        }
    }
    if fresh.ent.len() as u32 == e0 {
        fresh.off.truncate(o0);
    } else {
        while next <= w {
            fresh.off[o0 + next] = fresh.ent.len() as u32;
            next += 1;
        }
        lim.try_push(&mut fresh.blocks, (u, o0 as u32, false))?;
    }
    st.pstamp = pstamp;
    st.qstamp = qstamp;
    st.lstamp = lstamp;
    st.cstamp = cstamp;
    Ok(())
}

fn emit_clause_node(
    sh: &Shape,
    out: &mut SkipForm,
    u: u32,
    buf: &mut Vec<(Ref, Ref)>,
    lim: &Limits,
) -> Result<Ref, OperationError> {
    match normalize(sh, u, buf) {
        Err(target) => Ok(target),
        Ok(()) => Ok(mk(u, out.push_node(u, buf, lim)?)),
    }
}

/// The canonical skip form of `clause` over `vtree` and `shape`, built on the
/// ancestor paths of its literals, or `None` when `clause` names a variable in
/// both polarities (true).
pub(crate) fn clause_form<L: crate::LiteralInput>(
    vtree: &crate::vtree::Vtree,
    shape: &Arc<Shape>,
    clause: &[L],
    lim: &Limits,
) -> Result<Option<SkipForm>, OperationError> {
    let sh = &**shape;
    let mut lits: Vec<(u32, bool)> = Vec::new();
    lim.reserve_exact(&mut lits, clause.len())?;
    let mut gate = lim.gate();
    for raw in clause {
        gate.poll(1)?;
        let lit = raw.literal()?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        lits.push((leaf.0, lit.sign));
    }
    gate.flush()?;
    if lits.is_empty() {
        lim.discard(lits);
        return Ok(Some(SkipForm::zero(Arc::clone(shape))));
    }
    lits.sort_unstable_by_key(|&(leaf, sign)| (std::cmp::Reverse(sh.pre[leaf as usize]), sign));
    lits.dedup();
    if lits.windows(2).any(|w| w[0].0 == w[1].0) {
        lim.discard(lits);
        return Ok(None);
    }
    let mut out = SkipForm::zero(Arc::clone(shape));
    if sh.is_leaf(sh.root) {
        let (leaf, pos) = lits[0];
        lim.discard(lits);
        out.out = mk(leaf, if pos { POS } else { NEG });
        return Ok(Some(out));
    }
    let mut nodes: Vec<(u32, u32)> = Vec::new();
    for &(leaf, pos) in &lits {
        lim.try_push(&mut nodes, (leaf, if pos { POS } else { NEG }))?;
        let mut u = sh.parent[leaf as usize];
        while u != NONE {
            lim.try_push(&mut nodes, (u, ONE))?;
            u = sh.parent[sh.chain_up[u as usize] as usize];
        }
    }
    lim.discard(lits);
    nodes.sort_unstable_by_key(|&(u, _)| std::cmp::Reverse(sh.pre[u as usize]));
    nodes.dedup_by_key(|x| x.0);
    let m = nodes.len();
    let mut kids = filled(lim, m, (NONE, NONE))?;
    let mut stack: Vec<u32> = Vec::new();
    for (i, &(u, _)) in nodes.iter().enumerate() {
        if !sh.is_leaf(u) {
            let (l, r) = sh.kids[u as usize];
            let lc = if stack.last().is_some_and(|&top| sh.within(nodes[top as usize].0, l)) {
                stack.pop().unwrap()
            } else {
                NONE
            };
            let rc = if stack.last().is_some_and(|&top| sh.within(nodes[top as usize].0, r)) {
                stack.pop().unwrap()
            } else {
                NONE
            };
            kids[i] = (lc, rc);
        }
        lim.try_push(&mut stack, i as u32)?;
    }
    debug_assert_eq!(stack.len(), 1);
    lim.discard(stack);
    let mut need_dt = filled(lim, m, false)?;
    for i in (0..m).rev() {
        let (lc, rc) = kids[i];
        if need_dt[i] || (lc != NONE && rc != NONE) {
            if lc != NONE {
                need_dt[lc as usize] = true;
            }
            if rc != NONE {
                need_dt[rc as usize] = true;
            }
        }
    }
    let mut c_ref = filled(lim, m, ZREF)?;
    let mut d_ref = filled(lim, m, ZREF)?;
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    let mut gate = lim.gate();
    for i in 0..m {
        gate.poll(1)?;
        let (u, label) = nodes[i];
        if sh.is_leaf(u) {
            c_ref[i] = mk(u, label);
            if need_dt[i] {
                d_ref[i] = mk(u, if label == POS { NEG } else { POS });
            }
            continue;
        }
        let (l, r) = sh.kids[u as usize];
        let (lc, rc) = kids[i];
        match (lc != NONE, rc != NONE) {
            (true, true) => {
                let (cl, dl) = (c_ref[lc as usize], d_ref[lc as usize]);
                let (cr, dr) = (c_ref[rc as usize], d_ref[rc as usize]);
                buf.clear();
                lim.reserve(&mut buf, 3)?;
                buf.extend_from_slice(&[(cl, cr), (cl, dr), (dl, cr)]);
                c_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                if need_dt[i] {
                    buf.clear();
                    lim.try_push(&mut buf, (dl, dr))?;
                    d_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                }
            }
            (true, false) => {
                let top_r = super::build_top(sh, r, &mut |u, pair| Ok(mk(u, out.push_node(u, &[pair], lim)?)))?;
                let (cl, dl) = (c_ref[lc as usize], d_ref[lc as usize]);
                buf.clear();
                lim.try_push(&mut buf, (cl, top_r))?;
                c_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                if need_dt[i] {
                    buf.clear();
                    lim.try_push(&mut buf, (dl, top_r))?;
                    d_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                }
            }
            (false, true) => {
                let top_l = super::build_top(sh, l, &mut |u, pair| Ok(mk(u, out.push_node(u, &[pair], lim)?)))?;
                let (cr, dr) = (c_ref[rc as usize], d_ref[rc as usize]);
                buf.clear();
                lim.try_push(&mut buf, (top_l, cr))?;
                c_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                if need_dt[i] {
                    buf.clear();
                    lim.try_push(&mut buf, (top_l, dr))?;
                    d_ref[i] = emit_clause_node(sh, &mut out, u, &mut buf, lim)?;
                }
            }
            (false, false) => unreachable!("every internal spine node has a spine child"),
        }
    }
    gate.flush()?;
    out.out = c_ref[m - 1];
    out.active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
    lim.discard(nodes);
    lim.discard(kids);
    lim.discard(need_dt);
    lim.discard(c_ref);
    lim.discard(d_ref);
    lim.discard(buf);
    Ok(Some(out))
}
