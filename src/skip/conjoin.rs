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

use super::build::Land;
use super::{ix, mk, normalize, vt, Ref, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::limits::Limits;
use crate::OperationError;

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
    fn new(n: usize) -> Lists {
        Lists { head: vec![NIL; n], ent: Vec::new(), free: NIL }
    }

    #[inline]
    fn push(&mut self, i: usize, other: Ref, res: Ref) {
        let e = (other, res, self.head[i]);
        let at = if self.free != NIL {
            let at = self.free;
            self.free = self.ent[at as usize].2;
            self.ent[at as usize] = e;
            at
        } else {
            self.ent.push(e);
            (self.ent.len() - 1) as u32
        };
        self.head[i] = at;
    }

    /// The entries of list `i` that `judge` takes, into `ends`: `judge`
    /// gives, for the other reference, whether the entry stays listed and
    /// whether it is taken. Entries that do not stay are freed.
    #[inline]
    fn scan(&mut self, i: usize, ends: &mut Vec<(Ref, Ref)>, judge: impl Fn(Ref) -> (bool, bool)) {
        let (mut prev, mut e) = (NIL, self.head[i]);
        while e != NIL {
            let (other, res, next) = self.ent[e as usize];
            let (stay, take) = judge(other);
            if stay {
                if take {
                    ends.push((other, res));
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

/// Build one node's products (`keys`, their pair combinations in `combos`
/// by key) at internal `u` into `out`, noting each with its result in
/// `made`; empties `keys` and `combos`. The work units they took.
#[inline]
fn emit(
    sh: &Shape,
    u: u32,
    out: &mut SkipForm,
    keys: &mut Vec<(Ref, Ref)>,
    combos: &mut Vec<(u32, Ref, Ref)>,
    buf: &mut Vec<(Ref, Ref)>,
    made: &mut Vec<(Ref, Ref, Ref)>,
) -> u64 {
    if keys.is_empty() {
        return 0;
    }
    let work = (keys.len() + combos.len()) as u64;
    if keys.len() > 1 {
        combos.sort_unstable_by_key(|c| c.0);
    }
    let mut i = 0;
    for (k, &(a, b)) in keys.iter().enumerate() {
        buf.clear();
        while i < combos.len() && combos[i].0 == k as u32 {
            buf.push((combos[i].1, combos[i].2));
            i += 1;
        }
        let res = match normalize(sh, u, buf) {
            Err(t) => t,
            Ok(()) => mk(u, out.push_node(u, buf)),
        };
        made.push((a, b, res));
    }
    keys.clear();
    combos.clear();
    work
}

/// An operand's references numbered densely: stored node `i` of level `t`
/// is `base[t] + i`, leaf label `k` of leaf `c` is `base[c] + k`; with, per
/// number, the highest level the reference is read at: the child level of
/// the highest level whose pairs name it, or the root for the output. A
/// reference is read at every level from there down to its own.
struct Numbered {
    base: Vec<u32>,
    top: Vec<u32>,
    /// The leaves whose labels are numbered.
    leaves: Vec<u32>,
}

impl Numbered {
    fn of(s: &SkipForm) -> Numbered {
        let sh = &*s.shape;
        let mut base = vec![NONE; sh.num_nodes()];
        let mut n = 0u32;
        for &t in &s.active {
            base[t as usize] = n;
            n += s.levels[t as usize].width() as u32;
        }
        let mut nb = Numbered { base, top: vec![NONE; n as usize], leaves: Vec::new() };
        nb.note(sh, s.out, sh.root);
        for &u in &s.active {
            let (l, r) = sh.kids[u as usize];
            for &(a, b) in &s.levels[u as usize].pairs {
                nb.note(sh, a, l);
                nb.note(sh, b, r);
            }
        }
        nb
    }

    #[inline]
    fn note(&mut self, sh: &Shape, x: Ref, c: u32) {
        let v = vt(x) as usize;
        if self.base[v] == NONE {
            debug_assert!(sh.is_leaf(v as u32), "a stored node's level is numbered");
            self.base[v] = self.top.len() as u32;
            self.top.extend([NONE; 3]);
            self.leaves.push(v as u32);
        }
        let t = &mut self.top[(self.base[v] + ix(x)) as usize];
        if *t == NONE || sh.pre[c as usize] < sh.pre[*t as usize] {
            *t = c;
        }
    }

    fn len(&self) -> usize {
        self.top.len()
    }

    /// The number of `x`, a reference of the operand.
    #[inline]
    fn id(&self, x: Ref) -> usize {
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
    debug_assert!(Arc::ptr_eq(&f.shape, &g.shape));
    let shape = Arc::clone(&f.shape);
    let sh = &*shape;
    let mut out = SkipForm::zero(Arc::clone(&shape));
    match land_pair(sh, f.out, g.out) {
        Land::Done(r) => {
            out.out = r;
            out.collect_pushed();
            return Ok(out);
        }
        Land::At(_) => {}
    }
    let (nf, ng) = (Numbered::of(f), Numbered::of(g));
    // The levels products can land on, children first.
    let mut todo: Vec<u32> = f.active.iter().chain(&g.active).copied().collect();
    for nb in [&nf, &ng] {
        for &c in &nb.leaves {
            let p = sh.parent[c as usize];
            if p != NONE && nb.leaf_refs(sh, c, p).next().is_some() {
                todo.push(p);
            }
        }
    }
    todo.sort_unstable_by_key(|&t| std::cmp::Reverse(sh.pre[t as usize]));
    todo.dedup();
    // True products, per reference of each side: the other side's
    // reference and the result.
    let (mut pf, mut pg) = (Lists::new(nf.len()), Lists::new(ng.len()));
    // Both references to the result, where a level looks products up so.
    let mut live: FxHashMap<u64, Ref> = FxHashMap::default();
    let key = |a: usize, b: usize| ((a as u64) << 32) | b as u64;
    // `g`'s pairs of the current level by their anchor-side reference.
    let (mut gmark, mut ghead) = (vec![0u32; ng.len()], vec![NIL; ng.len()]);
    let mut gent: Vec<(u32, Ref, u32)> = Vec::new();
    // The current node's products, by the other side's reference.
    let (mut cg, mut cf) = (vec![(0u32, 0u32); ng.len()], vec![(0u32, 0u32); nf.len()]);
    let (mut lstamp, mut cstamp) = (0u32, 0u32);
    // One node's products against the other side and their pair
    // combinations, built as soon as the node is done; the level's
    // products, listed once the level is done.
    let mut keys: Vec<(Ref, Ref)> = Vec::new();
    let mut combos: Vec<(u32, Ref, Ref)> = Vec::new();
    let mut made: Vec<(Ref, Ref, Ref)> = Vec::new();
    let mut ends: Vec<(Ref, Ref)> = Vec::new();
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    let mut fb = [(0, 0); 1];
    let mut gb = [(0, 0); 1];
    for &u in &todo {
        let (l, r) = sh.kids[u as usize];
        made.clear();
        let mut work = 0u64;
        let (fl, gl) = (&f.levels[u as usize], &g.levels[u as usize]);
        if sh.is_leaf(l) && sh.is_leaf(r) {
            // Every reference each side reads here, against every other.
            let refs = |s: &SkipForm, nb: &Numbered| -> Vec<Ref> {
                let mut v: Vec<Ref> = (0..s.levels[u as usize].width() as u32).map(|i| mk(u, i)).collect();
                v.extend(nb.leaf_refs(sh, l, u));
                v.extend(nb.leaf_refs(sh, r, u));
                v
            };
            let (fr, gr) = (refs(f, &nf), refs(g, &ng));
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
                                    keys.push((a, b));
                                    cell = (keys.len() - 1) as u32;
                                }
                                combos.push((cell, x, y));
                            }
                        }
                    }
                    work += emit(sh, u, &mut out, &mut keys, &mut combos, &mut buf, &mut made);
                }
            }
        } else {
            // The anchor: the right child unless it is a leaf.
            let right = !sh.is_leaf(r);
            let (cs, co) = if right { (r, l) } else { (l, r) };
            let split = |p: (Ref, Ref)| if right { (p.1, p.0) } else { (p.0, p.1) };
            let join = |anchor: Ref, other: Ref| if right { (other, anchor) } else { (anchor, other) };
            let skips = sh.is_leaf(co);
            let other = |live: &FxHashMap<u64, Ref>, oa: Ref, ob: Ref| -> Ref {
                match land_pair(sh, oa, ob) {
                    Land::Done(x) => x,
                    Land::At(_) => live.get(&key(nf.id(oa), ng.id(ob))).copied().unwrap_or(ZREF),
                }
            };
            lstamp += 1;
            gent.clear();
            for j in 0..gl.width() as u32 {
                for &p in gl.node(j) {
                    let (y, ob) = split(p);
                    let gy = ng.id(y);
                    if gmark[gy] != lstamp {
                        gmark[gy] = lstamp;
                        ghead[gy] = NIL;
                    }
                    gent.push((j, ob, ghead[gy]));
                    ghead[gy] = (gent.len() - 1) as u32;
                }
            }
            let gbase = ng.base[u as usize];
            for i in 0..fl.width() as u32 {
                let a = mk(u, i);
                cstamp += 1;
                for &p in fl.node(i) {
                    let (x, oa) = split(p);
                    // The true products of `x` with references of `g` read
                    // at the anchor.
                    ends.clear();
                    pf.scan(nf.id(x), &mut ends, |y| {
                        let keep = ng.read_at(sh, y, cs);
                        (keep, keep)
                    });
                    if sh.is_leaf(vt(x)) {
                        for y in ng.leaf_refs(sh, vt(x), cs) {
                            let res = leaf_and(vt(x), ix(x), ix(y));
                            if res != ZREF {
                                ends.push((y, res));
                            }
                        }
                    }
                    for &(y, res) in &ends {
                        let gy = ng.id(y);
                        if gmark[gy] == lstamp {
                            let mut e = ghead[gy];
                            while e != NIL {
                                let (j, ob, next) = gent[e as usize];
                                e = next;
                                let o = other(&live, oa, ob);
                                if o != ZREF {
                                    let c = &mut cg[(gbase + j) as usize];
                                    if c.0 != cstamp {
                                        keys.push((a, mk(u, j)));
                                        *c = (cstamp, (keys.len() - 1) as u32);
                                    }
                                    let (pl, pr) = join(res, o);
                                    combos.push((c.1, pl, pr));
                                }
                            }
                        }
                        if skips && ng.above(sh, y, u) {
                            // `y` is `true` on the other side here.
                            let c = &mut cg[gy];
                            if c.0 != cstamp {
                                keys.push((a, y));
                                *c = (cstamp, (keys.len() - 1) as u32);
                            }
                            let (pl, pr) = join(res, oa);
                            combos.push((c.1, pl, pr));
                        }
                    }
                }
                work += emit(sh, u, &mut out, &mut keys, &mut combos, &mut buf, &mut made);
            }
            if skips {
                // Skips of `f` through this level against `g`'s nodes.
                for j in 0..gl.width() as u32 {
                    let b = mk(u, j);
                    cstamp += 1;
                    for &p in gl.node(j) {
                        let (y, ob) = split(p);
                        ends.clear();
                        pg.scan(ng.id(y), &mut ends, |x| {
                            let keep = nf.read_at(sh, x, cs);
                            (keep, keep && nf.above(sh, x, u))
                        });
                        if sh.is_leaf(vt(y)) {
                            for x in nf.leaf_refs(sh, vt(y), u) {
                                let res = leaf_and(vt(y), ix(x), ix(y));
                                if res != ZREF {
                                    ends.push((x, res));
                                }
                            }
                        }
                        for &(x, res) in &ends {
                            let c = &mut cf[nf.id(x)];
                            if c.0 != cstamp {
                                keys.push((x, b));
                                *c = (cstamp, (keys.len() - 1) as u32);
                            }
                            let (pl, pr) = join(res, ob);
                            combos.push((c.1, pl, pr));
                        }
                    }
                    work += emit(sh, u, &mut out, &mut keys, &mut combos, &mut buf, &mut made);
                }
            }
        }
        lim.charge_work(work);
        lim.check_stop()?;
        let lookup = sh.looked_up[u as usize];
        for &(a, b, res) in &made {
            let (fa, gb) = (nf.id(a), ng.id(b));
            pf.push(fa, b, res);
            pg.push(gb, a, res);
            if lookup {
                live.insert(key(fa, gb), res);
            }
        }
    }
    out.out = match land_pair(sh, f.out, g.out) {
        Land::Done(r) => r,
        Land::At(_) => pf.find(nf.id(f.out), g.out).unwrap_or(ZREF),
    };
    out.collect_pushed();
    Ok(out)
}
