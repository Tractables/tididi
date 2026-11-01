//! Conjunction in skip form.
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

use super::{filled, ix, mk, normalize, vt, Ref, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::limits::Limits;
use crate::OperationError;

/// The product of `a` and `b` (references read from one child level) when
/// known without building (a false or leaf result), or `None` when built at an
/// internal level.
#[inline]
pub(crate) fn land_pair(sh: &Shape, a: Ref, b: Ref) -> Option<Ref> {
    if a == ZREF || b == ZREF {
        return Some(ZREF);
    }
    let va = vt(a);
    if va == vt(b) && sh.is_leaf(va) {
        return Some(leaf_and(va, ix(a), ix(b)));
    }
    None
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
fn is_one(sh: &Shape, r: Ref) -> bool {
    r != ZREF && sh.is_leaf(vt(r)) && ix(r) == ONE
}

/// Per vtree node, whether it is in `g`'s cone: an ancestor (or itself) of a
/// level holding a node of `g` or of a leaf `g` reads a literal on. Below the
/// cone every reference of `g` is `true`; the cone holds every ancestor of its
/// levels.
fn cone_of(sh: &Shape, g: &SkipForm, ng: &Numbered, lim: &Limits) -> Result<Vec<bool>, OperationError> {
    let mut cone = filled(lim, sh.num_nodes(), false)?;
    let mut up = |mut t: u32| {
        while t != NONE && !cone[t as usize] {
            cone[t as usize] = true;
            t = sh.parent[t as usize];
        }
    };
    for &t in &g.active {
        up(t);
    }
    for &c in &ng.leaves {
        let b = ng.base[c as usize];
        if [POS, NEG].iter().any(|&k| ng.top[(b + k) as usize] != NONE) {
            up(c);
        }
    }
    Ok(cone)
}

/// The products with `true` of `g` that the conjunction's scans may meet,
/// of `f`'s references outside the cone that a pair in the cone, or the
/// output, names: `(preorder position of the level the product would be
/// built at, the reference, g's true)`, latest built first. The product is
/// the reference itself. A product built at a level in the cone (two leaves
/// under one, a literal of `g` on one of them) is built there instead.
fn identity_seeds(
    sh: &Shape,
    f: &SkipForm,
    nf: &Numbered,
    ng: &Numbered,
    cone: &[bool],
    f_cone: &[u32],
    lim: &Limits,
) -> Result<Vec<(u32, Ref, Ref)>, OperationError> {
    let mut seeds: Vec<(u32, Ref, Ref)> = Vec::new();
    let mut tops: Vec<Ref> = Vec::new();
    for &c in &ng.leaves {
        let t = mk(c, ONE);
        let top = ng.top[ng.id(t)];
        // Every product in `seeds` is built at an internal level `p`, where
        // `ng.read_at(sh, t, p)` holds only if `t` is read above leaf `c`.
        if top != NONE && top != c {
            lim.try_push(&mut tops, t)?;
        }
    }
    if tops.is_empty() {
        return Ok(seeds);
    }
    let mut seen = filled(lim, nf.len(), false)?;
    let mut note = |x: Ref, seeds: &mut Vec<(u32, Ref, Ref)>| -> Result<(), OperationError> {
        let vx = vt(x);
        if sh.is_leaf(vx) {
            let p = sh.parent[vx as usize];
            if p == NONE || cone[p as usize] {
                return Ok(());
            }
        } else if cone[vx as usize] {
            return Ok(());
        }
        if std::mem::replace(&mut seen[nf.id(x)], true) {
            return Ok(());
        }
        for &t in &tops {
            let p = if !sh.is_leaf(vx) {
                if !sh.within(vt(t), vx) {
                    continue;
                }
                vx
            } else {
                let p = sh.parent[vx as usize];
                if vt(t) == vx || sh.parent[vt(t) as usize] != p {
                    continue;
                }
                p
            };
            if nf.read_at(sh, x, p) && ng.read_at(sh, t, p) {
                lim.try_push(seeds, (sh.pre[p as usize], x, t))?;
            }
        }
        Ok(())
    };
    note(f.out, &mut seeds)?;
    for &u in f_cone {
        for &(a, b) in &f.levels[u as usize].pairs {
            note(a, &mut seeds)?;
            note(b, &mut seeds)?;
        }
    }
    lim.discard(seen);
    lim.discard(tops);
    seeds.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
    Ok(seeds)
}

const NIL: u32 = u32::MAX;

/// Per reference of one operand, its true products: linked entries of the
/// other operand's reference, the result and the next entry, in one arena
/// whose freed entries are reused.
struct Lists {
    head: Vec<u32>,
    ent: Vec<(Ref, Ref, u32)>,
    free: u32,
}

impl Lists {
    fn new(n: usize, lim: &Limits) -> Result<Lists, OperationError> {
        Ok(Lists { head: filled(lim, n, NIL)?, ent: Vec::new(), free: NIL })
    }

    #[inline]
    fn push(&mut self, i: usize, other: Ref, res: Ref, lim: &Limits) -> Result<(), OperationError> {
        let e = (other, res, self.head[i]);
        let at = if self.free != NIL {
            let at = self.free;
            self.free = self.ent[at as usize].2;
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
    /// gives, for the other reference, whether the entry stays listed and
    /// whether it is taken. Entries that do not stay are freed.
    #[inline]
    fn scan(&mut self, i: usize, ends: &mut Vec<(Ref, Ref)>, lim: &Limits, judge: impl Fn(Ref) -> (bool, bool)) -> Result<(), OperationError> {
        let (mut prev, mut e) = (NIL, self.head[i]);
        while e != NIL {
            let (other, res, next) = self.ent[e as usize];
            let (stay, take) = judge(other);
            if stay {
                if take {
                    lim.try_push(ends, (other, res))?;
                }
                prev = e;
            } else {
                if prev == NIL {
                    self.head[i] = next;
                } else {
                    self.ent[prev as usize].2 = next;
                }
                self.ent[e as usize].2 = self.free;
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
            let (o, res, next) = self.ent[e as usize];
            if o == other {
                return Some(res);
            }
            e = next;
        }
        None
    }
}

/// Scratch buffers for building one node's products (`keys` and their pair
/// combinations in `combos` by key) and recording the level's products in
/// `made`.
#[derive(Default)]
struct NodeBatch {
    keys: Vec<(Ref, Ref)>,
    combos: Vec<(u32, Ref, Ref)>,
    buf: Vec<(Ref, Ref)>,
    made: Vec<(Ref, Ref, Ref)>,
}

impl NodeBatch {
    #[inline]
    fn push_stamped(
        &mut self,
        slot: &mut (u32, u32),
        stamp: u32,
        key: (Ref, Ref),
        pair: (Ref, Ref),
        lim: &Limits,
    ) -> Result<(), OperationError> {
        if slot.0 != stamp {
            lim.try_push(&mut self.keys, key)?;
            *slot = (stamp, (self.keys.len() - 1) as u32);
        }
        lim.try_push(&mut self.combos, (slot.1, pair.0, pair.1))
    }

    /// Build the queued products at internal `u` into `out`, noting each with
    /// its result in `made`; empties `keys` and `combos`. The work units they
    /// took.
    #[inline]
    fn emit(&mut self, u: u32, out: &mut SkipForm, lim: &Limits) -> Result<u64, OperationError> {
        if self.keys.is_empty() {
            return Ok(0);
        }
        let work = (self.keys.len() + self.combos.len()) as u64;
        if self.keys.len() > 1 {
            self.combos.sort_unstable_by_key(|c| c.0);
        }
        let mut i = 0;
        for (k, &(a, b)) in self.keys.iter().enumerate() {
            self.buf.clear();
            while i < self.combos.len() && self.combos[i].0 == k as u32 {
                lim.try_push(&mut self.buf, (self.combos[i].1, self.combos[i].2))?;
                i += 1;
            }
            let res = match normalize(&out.shape, u, &mut self.buf) {
                Err(t) => t,
                Ok(()) => mk(u, out.push_node(u, &self.buf, lim)?),
            };
            lim.try_push(&mut self.made, (a, b, res))?;
        }
        self.keys.clear();
        self.combos.clear();
        Ok(work)
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.keys);
        lim.discard(self.combos);
        lim.discard(self.buf);
        lim.discard(self.made);
    }
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
        Self::of_levels(s, &s.active, lim)
    }

    /// Number only the references of `s` at levels in `f_cone`, and those named
    /// from `s.out` or `s`'s levels in `f_cone`: since the cone holds every
    /// ancestor of its levels, the highest level where any such reference is
    /// read in `s` is attained at `s.out` or at a level in `f_cone`.
    pub(super) fn of_cone(s: &SkipForm, f_cone: &[u32], lim: &Limits) -> Result<Numbered, OperationError> {
        Self::of_levels(s, f_cone, lim)
    }

    fn of_levels(s: &SkipForm, levels: &[u32], lim: &Limits) -> Result<Numbered, OperationError> {
        let sh = &*s.shape;
        let mut base = vec![NONE; sh.num_nodes()];
        let mut n = 0u32;
        for &t in levels {
            base[t as usize] = n;
            n += s.levels[t as usize].width() as u32;
        }
        let mut top = Vec::new();
        lim.try_resize(&mut top, n as usize, NONE)?;
        let mut nb = Numbered { base, top, leaves: Vec::new() };
        nb.note(s, s.out, sh.root, lim)?;
        for &u in levels {
            let (l, r) = sh.kids[u as usize];
            for &(a, b) in &s.levels[u as usize].pairs {
                nb.note(s, a, l, lim)?;
                nb.note(s, b, r, lim)?;
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
                3
            } else {
                s.levels[v].width()
            };
            lim.reserve(&mut self.top, w)?;
            self.top.resize(self.top.len() + w, NONE);
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
        [ONE, POS, NEG].into_iter().map(move |lab| mk(c, lab)).filter(move |&x| self.read_at(sh, x, u))
    }
}

/// `f ∧ g`, pruned, in skip form; not contracted, together with the
/// upward-closed cone of levels that changed (those in `g`'s cone and any
/// level outside it that lost a node when unreachable branches were dropped).
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
pub(crate) fn and_scoped(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<(SkipForm, Vec<bool>), OperationError> {
    // Shapes read from one vtree are equal; `f`'s serves both.
    debug_assert_eq!(f.shape.kids, g.shape.kids);
    let shape = Arc::clone(&f.shape);
    let sh = &*shape;
    let mut out = SkipForm::zero(Arc::clone(&shape));
    if let Some(r) = land_pair(sh, f.out, g.out) {
        out.out = r;
        let cone = filled(lim, sh.num_nodes(), false)?;
        return Ok((out, cone));
    }
    let ng = Numbered::of(g, lim)?;
    // Outside `g`'s cone `g` is `true`: there the result is `f`'s own levels,
    // and a product of a reference of `f` with `true` is that reference.
    let mut cone = cone_of(sh, g, &ng, lim)?;
    let mut f_cone: Vec<u32> = Vec::new();
    lim.reserve(&mut f_cone, f.active.len())?;
    f_cone.extend(f.active.iter().copied().filter(|&t| cone[t as usize]));
    let nf = Numbered::of_cone(f, &f_cone, lim)?;
    let free = |x: Ref| !cone[vt(x) as usize];
    let seeds = identity_seeds(sh, f, &nf, &ng, &cone, &f_cone, lim)?;
    // The levels in the cone products can land on, children first.
    let mut todo: Vec<u32> = Vec::new();
    lim.reserve(&mut todo, f_cone.len() + g.active.len() + nf.leaves.len() + ng.leaves.len())?;
    todo.extend(f_cone.iter().chain(&g.active).copied());
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
    let mut work = AnchorScratch::new(nf.len(), ng.len(), lim)?;
    for &u in &todo {
        // The products with `true` the levels below would have listed.
        while next_seed < seeds.len() && seeds[next_seed].0 > sh.pre[u as usize] {
            let (_, x, t) = seeds[next_seed];
            work.pf.push(nf.id(x), t, x, lim)?;
            work.pg.push(ng.id(t), x, x, lim)?;
            next_seed += 1;
        }
        let (l, r) = sh.kids[u as usize];
        work.batch.made.clear();
        let steps = if sh.is_leaf(l) && sh.is_leaf(r) {
            conjoin_leaf_pair(sh, u, (f, &nf), (g, &ng), &mut work.batch, &mut out, lim)?
        } else {
            work.conjoin_anchor_level(sh, u, (f, &nf), (g, &ng), &cone, &mut out, lim)?
        };
        lim.charge_work(steps);
        lim.check_stop()?;
        work.record_made(sh, u, &nf, &ng, lim)?;
    }
    out.out = match land_pair(sh, f.out, g.out) {
        Some(r) => r,
        None if is_one(sh, g.out) && free(f.out) => f.out,
        None => work.pf.find(nf.id(f.out), g.out).unwrap_or(ZREF),
    };
    lim.discard(seeds);
    work.discard(lim);
    ng.discard(lim);
    out.collect_pushed(lim)?;
    lim.discard(todo);
    splice_outside_cone(f, nf, f_cone, &mut out, &mut cone, lim)?;
    Ok((out, cone))
}

/// Reusable product lists and per-level scratch for [`and_scoped`].
struct AnchorScratch {
    /// True products per reference of `f`: `(g_ref, result)`.
    pf: Lists,
    /// True products per reference of `g`: `(f_ref, result)`.
    pg: Lists,
    /// Both references to the result, where a level looks products up so.
    live: FxHashMap<u64, Ref>,
    /// `g`'s pairs of the current level by their anchor-side reference.
    gmark: Vec<u32>,
    ghead: Vec<u32>,
    gent: Vec<(u32, Ref, u32)>,
    /// The current node's products, by the other side's reference.
    cg: Vec<(u32, u32)>,
    cf: Vec<(u32, u32)>,
    lstamp: u32,
    cstamp: u32,
    /// One node's products against the other side and their pair combinations.
    batch: NodeBatch,
    ends: Vec<(Ref, Ref)>,
}

impl AnchorScratch {
    fn new(nf_len: usize, ng_len: usize, lim: &Limits) -> Result<Self, OperationError> {
        Ok(Self {
            pf: Lists::new(nf_len, lim)?,
            pg: Lists::new(ng_len, lim)?,
            live: FxHashMap::default(),
            gmark: filled(lim, ng_len, 0u32)?,
            ghead: filled(lim, ng_len, NIL)?,
            gent: Vec::new(),
            cg: filled(lim, ng_len, (0u32, 0u32))?,
            cf: filled(lim, nf_len, (0u32, 0u32))?,
            lstamp: 0,
            cstamp: 0,
            batch: NodeBatch::default(),
            ends: Vec::new(),
        })
    }

    fn discard(self, lim: &Limits) {
        self.pf.discard(lim);
        self.pg.discard(lim);
        lim.discard(self.live);
        for v in [self.gmark, self.ghead] {
            lim.discard(v);
        }
        for v in [self.cg, self.cf] {
            lim.discard(v);
        }
        lim.discard(self.gent);
        self.batch.discard(lim);
        lim.discard(self.ends);
    }

    fn record_made(&mut self, sh: &Shape, u: u32, nf: &Numbered, ng: &Numbered, lim: &Limits) -> Result<(), OperationError> {
        let lookup = sh.looked_up[u as usize];
        if lookup {
            lim.reserve_map(&mut self.live, self.batch.made.len())?;
        }
        for &(a, b, res) in &self.batch.made {
            let (fa, gb) = (nf.id(a), ng.id(b));
            self.pf.push(fa, b, res, lim)?;
            self.pg.push(gb, a, res, lim)?;
            if lookup {
                self.live.insert(((fa as u64) << 32) | gb as u64, res);
            }
        }
        Ok(())
    }

    /// Conjoin a level `u` that has at least one internal child (the anchor).
    #[allow(clippy::too_many_arguments)]
    fn conjoin_anchor_level(
        &mut self,
        sh: &Shape,
        u: u32,
        (f, nf): (&SkipForm, &Numbered),
        (g, ng): (&SkipForm, &Numbered),
        cone: &[bool],
        out: &mut SkipForm,
        lim: &Limits,
    ) -> Result<u64, OperationError> {
        let (l, r) = sh.kids[u as usize];
        let (fl, gl) = (&f.levels[u as usize], &g.levels[u as usize]);
        // The anchor: the right child unless it is a leaf.
        let right = !sh.is_leaf(r);
        let (cs, co) = if right { (r, l) } else { (l, r) };
        let split = |p: (Ref, Ref)| if right { (p.1, p.0) } else { (p.0, p.1) };
        let join = |anchor: Ref, other: Ref| if right { (other, anchor) } else { (anchor, other) };
        let skips = sh.is_leaf(co);
        let other = |live: &FxHashMap<u64, Ref>, oa: Ref, ob: Ref| -> Ref {
            if is_one(sh, ob) && !cone[vt(oa) as usize] {
                return oa;
            }
            match land_pair(sh, oa, ob) {
                Some(x) => x,
                None => live.get(&(((nf.id(oa) as u64) << 32) | ng.id(ob) as u64)).copied().unwrap_or(ZREF),
            }
        };
        self.lstamp += 1;
        self.gent.clear();
        for j in 0..gl.width() as u32 {
            for &p in gl.node(j) {
                let (y, ob) = split(p);
                let gy = ng.id(y);
                if self.gmark[gy] != self.lstamp {
                    self.gmark[gy] = self.lstamp;
                    self.ghead[gy] = NIL;
                }
                lim.try_push(&mut self.gent, (j, ob, self.ghead[gy]))?;
                self.ghead[gy] = (self.gent.len() - 1) as u32;
            }
        }
        let mut work = 0u64;
        let gbase = ng.base[u as usize];
        for i in 0..fl.width() as u32 {
            let a = mk(u, i);
            self.cstamp += 1;
            for &p in fl.node(i) {
                let (x, oa) = split(p);
                // The true products of `x` with references of `g` read at the anchor.
                self.ends.clear();
                self.pf.scan(nf.id(x), &mut self.ends, lim, |y| {
                    let keep = ng.read_at(sh, y, cs);
                    (keep, keep)
                })?;
                if sh.is_leaf(vt(x)) {
                    for y in ng.leaf_refs(sh, vt(x), cs) {
                        let res = leaf_and(vt(x), ix(x), ix(y));
                        if res != ZREF {
                            lim.try_push(&mut self.ends, (y, res))?;
                        }
                    }
                }
                for &(y, res) in &self.ends {
                    let gy = ng.id(y);
                    if self.gmark[gy] == self.lstamp {
                        let mut e = self.ghead[gy];
                        while e != NIL {
                            let (j, ob, next) = self.gent[e as usize];
                            e = next;
                            let o = other(&self.live, oa, ob);
                            if o != ZREF {
                                self.batch.push_stamped(
                                    &mut self.cg[(gbase + j) as usize],
                                    self.cstamp,
                                    (a, mk(u, j)),
                                    join(res, o),
                                    lim,
                                )?;
                            }
                        }
                    }
                    if skips && ng.above(sh, y, u) {
                        // `y` is `true` on the other side here.
                        self.batch.push_stamped(&mut self.cg[gy], self.cstamp, (a, y), join(res, oa), lim)?;
                    }
                }
            }
            work += self.batch.emit(u, out, lim)?;
        }
        if skips {
            work += self.conjoin_f_skips(sh, u, (cs, right), (f, nf), (g, ng), out, lim)?;
        }
        Ok(work)
    }

    /// Skips of `f` through `u` against `g`'s nodes when the non-anchor child is a leaf.
    #[allow(clippy::too_many_arguments)]
    fn conjoin_f_skips(
        &mut self,
        sh: &Shape,
        u: u32,
        (cs, right): (u32, bool),
        (_f, nf): (&SkipForm, &Numbered),
        (g, ng): (&SkipForm, &Numbered),
        out: &mut SkipForm,
        lim: &Limits,
    ) -> Result<u64, OperationError> {
        let gl = &g.levels[u as usize];
        let split = |p: (Ref, Ref)| if right { (p.1, p.0) } else { (p.0, p.1) };
        let join = |anchor: Ref, other: Ref| if right { (other, anchor) } else { (anchor, other) };
        let mut work = 0u64;
        for j in 0..gl.width() as u32 {
            let b = mk(u, j);
            self.cstamp += 1;
            for &p in gl.node(j) {
                let (y, ob) = split(p);
                self.ends.clear();
                self.pg.scan(ng.id(y), &mut self.ends, lim, |x| {
                    let keep = nf.read_at(sh, x, cs);
                    (keep, keep && nf.above(sh, x, u))
                })?;
                if sh.is_leaf(vt(y)) {
                    for x in nf.leaf_refs(sh, vt(y), u) {
                        let res = leaf_and(vt(y), ix(x), ix(y));
                        if res != ZREF {
                            lim.try_push(&mut self.ends, (x, res))?;
                        }
                    }
                }
                for &(x, res) in &self.ends {
                    self.batch.push_stamped(&mut self.cf[nf.id(x)], self.cstamp, (x, b), join(res, ob), lim)?;
                }
            }
            work += self.batch.emit(u, out, lim)?;
        }
        Ok(work)
    }
}

/// Conjoin a level `u` whose children are both leaves: every reference each
/// side reads here, against every other.
fn conjoin_leaf_pair(
    sh: &Shape,
    u: u32,
    (f, nf): (&SkipForm, &Numbered),
    (g, ng): (&SkipForm, &Numbered),
    batch: &mut NodeBatch,
    out: &mut SkipForm,
    lim: &Limits,
) -> Result<u64, OperationError> {
    let (l, r) = sh.kids[u as usize];
    let refs = |s: &SkipForm, nb: &Numbered| -> Result<Vec<Ref>, OperationError> {
        let mut v: Vec<Ref> = Vec::new();
        lim.reserve_exact(&mut v, s.levels[u as usize].width() + 6)?;
        v.extend((0..s.levels[u as usize].width() as u32).map(|i| mk(u, i)));
        v.extend(nb.leaf_refs(sh, l, u));
        v.extend(nb.leaf_refs(sh, r, u));
        Ok(v)
    };
    let (fr, gr) = (refs(f, nf)?, refs(g, ng)?);
    let mut fb = [(0, 0); 1];
    let mut gb = [(0, 0); 1];
    let mut work = 0u64;
    for &a in &fr {
        for &b in &gr {
            if vt(a) != u && vt(a) == vt(b) {
                continue;
            }
            let mut cell = NIL;
            for &(la, ra) in f.pairs_at(u, a, &mut fb) {
                for &(lb, rb) in g.pairs_at(u, b, &mut gb) {
                    let (Some(x), Some(y)) = (land_pair(sh, la, lb), land_pair(sh, ra, rb)) else {
                        unreachable!("leaf products are known")
                    };
                    if x != ZREF && y != ZREF {
                        if cell == NIL {
                            lim.try_push(&mut batch.keys, (a, b))?;
                            cell = (batch.keys.len() - 1) as u32;
                        }
                        lim.try_push(&mut batch.combos, (cell, x, y))?;
                    }
                }
            }
            work += batch.emit(u, out, lim)?;
        }
    }
    lim.discard(fr);
    lim.discard(gr);
    Ok(work)
}

/// Copy `f`'s active levels outside `cone` that `out` still reaches into `out`,
/// pruning any outside-cone levels whose boundary references were dropped and
/// growing `cone` to include outside-cone levels whose width shrank.
fn splice_outside_cone(
    f: &SkipForm,
    nf: Numbered,
    f_cone: Vec<u32>,
    out: &mut SkipForm,
    cone: &mut [bool],
    lim: &Limits,
) -> Result<(), OperationError> {
    let sh = &*f.shape;
    if out.out == ZREF || f_cone.len() == f.active.len() {
        lim.discard(f_cone);
        nf.discard(lim);
        return Ok(());
    }
    let mut mark_out = filled(lim, nf.len(), 0u8)?;
    let mut boundary: Vec<Ref> = Vec::new();
    let mut note_f = |x: Ref, boundary: &mut Vec<Ref>| -> Result<(), OperationError> {
        if x != ZREF && !cone[vt(x) as usize] && !sh.is_leaf(vt(x)) {
            let id = nf.id(x);
            if mark_out[id] == 0 {
                mark_out[id] = 1;
                lim.try_push(boundary, x)?;
            }
        }
        Ok(())
    };
    note_f(f.out, &mut boundary)?;
    for &u in &f_cone {
        for &(a, b) in &f.levels[u as usize].pairs {
            note_f(a, &mut boundary)?;
            note_f(b, &mut boundary)?;
        }
    }
    lim.discard(f_cone);
    let f_outside = boundary.len();
    let mut out_outside = 0usize;
    if f_outside > 0 {
        let mut note_o = |x: Ref| {
            if x != ZREF && !cone[vt(x) as usize] && !sh.is_leaf(vt(x)) {
                let id = nf.id(x);
                if mark_out[id] == 1 {
                    mark_out[id] = 2;
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
    }
    if out_outside > 0 {
        for &t in &f.active {
            if !cone[t as usize] {
                out.levels[t as usize] = f.levels[t as usize].copy(lim)?;
                out.active.push(t);
            }
        }
        if out_outside == f_outside {
            out.active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
        } else {
            prune_outside_cone(f, &nf, &boundary, &mark_out, out, cone, lim)?;
        }
    }
    lim.discard(boundary);
    lim.discard(mark_out);
    nf.discard(lim);
    Ok(())
}

/// Prune outside-cone levels whose boundary references were dropped, and grow
/// `cone` to include any outside-cone level whose width shrank.
fn prune_outside_cone(
    f: &SkipForm,
    nf: &Numbered,
    boundary: &[Ref],
    mark_out: &[u8],
    out: &mut SkipForm,
    cone: &mut [bool],
    lim: &Limits,
) -> Result<(), OperationError> {
    let sh = &*f.shape;
    let mut dirty = filled(lim, sh.num_nodes(), false)?;
    dirty.copy_from_slice(cone);
    let mut dropped_roots = filled(lim, sh.num_nodes(), false)?;
    for &x in boundary {
        if mark_out[nf.id(x)] == 1 {
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
    Ok(())
}
