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
use super::{ix, mk, normalize, vt, Ref, Shape, SkipForm, NEG, ONE, POS, ZREF};
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

/// Per reference of an operand, the highest level it is read at: the child
/// level of the highest level whose pairs name it, or the root for the
/// output. A reference is read at every level from there down to its own.
fn reads(s: &SkipForm) -> FxHashMap<Ref, u32> {
    let sh = &*s.shape;
    let mut top: FxHashMap<Ref, u32> = FxHashMap::default();
    let mut note = |x: Ref, c: u32| {
        let t = top.entry(x).or_insert(c);
        if sh.pre[c as usize] < sh.pre[*t as usize] {
            *t = c;
        }
    };
    note(s.out, sh.root);
    for &u in &s.active {
        let (l, r) = sh.kids[u as usize];
        for &(a, b) in &s.levels[u as usize].pairs {
            note(a, l);
            note(b, r);
        }
    }
    top
}

/// Whether `x` (with read map `top`) is read at level `c`.
#[inline]
fn read_at(sh: &Shape, top: &FxHashMap<Ref, u32>, x: Ref, c: u32) -> bool {
    sh.within(vt(x), c) && top.get(&x).is_some_and(|&t| sh.within(c, t))
}

/// Whether `x` (with read map `top`) skips through level `u`.
#[inline]
fn crosses(sh: &Shape, top: &FxHashMap<Ref, u32>, x: Ref, u: u32) -> bool {
    vt(x) != u && read_at(sh, top, x, u)
}

/// The leaf references on leaf `c` an operand reads at level `u` above it.
fn leaf_refs<'a>(sh: &'a Shape, top: &'a FxHashMap<Ref, u32>, c: u32, u: u32) -> impl Iterator<Item = Ref> + 'a {
    [ONE, POS, NEG].into_iter().map(move |lab| mk(c, lab)).filter(move |&x| read_at(sh, top, x, u))
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
/// `g`'s pairs. Products no output path reaches are dropped at the end.
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
    let (tf, tg) = (reads(f), reads(g));
    // The levels products can land on, children first.
    let mut todo: Vec<u32> = f.active.iter().chain(&g.active).copied().collect();
    for top in [&tf, &tg] {
        for (&x, _) in top.iter() {
            let c = vt(x);
            if sh.is_leaf(c) && c != sh.root {
                let p = sh.parent[c as usize];
                if read_at(sh, top, x, p) {
                    todo.push(p);
                }
            }
        }
    }
    todo.sort_unstable_by_key(|&t| std::cmp::Reverse(sh.pre[t as usize]));
    todo.dedup();
    // True products: result, and per side the other side's references.
    let mut live: FxHashMap<(Ref, Ref), Ref> = FxHashMap::default();
    let mut by_f: FxHashMap<Ref, Vec<(Ref, Ref)>> = FxHashMap::default();
    let mut by_g: FxHashMap<Ref, Vec<(Ref, Ref)>> = FxHashMap::default();
    let product = |live: &FxHashMap<(Ref, Ref), Ref>, a: Ref, b: Ref| -> Ref {
        match land_pair(sh, a, b) {
            Land::Done(r) => r,
            Land::At(_) => live.get(&(a, b)).copied().unwrap_or(ZREF),
        }
    };
    let mut cells: FxHashMap<(Ref, Ref), u32> = FxHashMap::default();
    let mut keys: Vec<(Ref, Ref)> = Vec::new();
    let mut combos: Vec<(u32, Ref, Ref)> = Vec::new();
    let mut gpar: FxHashMap<Ref, smallvec::SmallVec<[(u32, Ref); 2]>> = FxHashMap::default();
    let mut ends: Vec<Ref> = Vec::new();
    let mut fb = [(0, 0); 1];
    let mut gb = [(0, 0); 1];
    for &u in &todo {
        let (l, r) = sh.kids[u as usize];
        cells.clear();
        keys.clear();
        combos.clear();
        let cell = |cells: &mut FxHashMap<(Ref, Ref), u32>, keys: &mut Vec<(Ref, Ref)>, k: (Ref, Ref)| -> u32 {
            *cells.entry(k).or_insert_with(|| {
                keys.push(k);
                (keys.len() - 1) as u32
            })
        };
        let (fl, gl) = (&f.levels[u as usize], &g.levels[u as usize]);
        if sh.is_leaf(l) && sh.is_leaf(r) {
            // Every reference each side reads here, against every other.
            let refs = |s: &SkipForm, top: &FxHashMap<Ref, u32>| -> Vec<Ref> {
                let mut v: Vec<Ref> = (0..s.levels[u as usize].width() as u32).map(|i| mk(u, i)).collect();
                v.extend(leaf_refs(sh, top, l, u));
                v.extend(leaf_refs(sh, top, r, u));
                v
            };
            let (fr, gr) = (refs(f, &tf), refs(g, &tg));
            for &a in &fr {
                for &b in &gr {
                    if vt(a) != u && vt(a) == vt(b) {
                        continue;
                    }
                    for &(la, ra) in f.pairs_at(u, a, &mut fb) {
                        for &(lb, rb) in g.pairs_at(u, b, &mut gb) {
                            let (x, y) = (product(&live, la, lb), product(&live, ra, rb));
                            if x != ZREF && y != ZREF {
                                let k = cell(&mut cells, &mut keys, (a, b));
                                combos.push((k, x, y));
                            }
                        }
                    }
                }
            }
        } else {
            // The anchor: the right child unless it is a leaf.
            let right = !sh.is_leaf(r);
            let (cs, co) = if right { (r, l) } else { (l, r) };
            let split = |p: (Ref, Ref)| if right { (p.1, p.0) } else { (p.0, p.1) };
            let join = |anchor: Ref, other: Ref| if right { (other, anchor) } else { (anchor, other) };
            let skips = sh.is_leaf(co);
            // `g`'s nodes here by the reference they name at the anchor.
            gpar.clear();
            for j in 0..gl.width() as u32 {
                for &p in gl.node(j) {
                    let (y, ob) = split(p);
                    gpar.entry(y).or_default().push((j, ob));
                }
            }
            // The true products of anchor-side references `x` (of f) and
            // `y` (of g, read at the anchor), with their results.
            let with_f = |x: Ref, ends: &mut Vec<Ref>| {
                ends.clear();
                if let Some(v) = by_f.get(&x) {
                    for &(y, res) in v {
                        if read_at(sh, &tg, y, cs) {
                            ends.push(y);
                            ends.push(res);
                        }
                    }
                }
                if sh.is_leaf(vt(x)) {
                    for y in leaf_refs(sh, &tg, vt(x), cs) {
                        let res = leaf_and(vt(x), ix(x), ix(y));
                        if res != ZREF {
                            ends.push(y);
                            ends.push(res);
                        }
                    }
                }
            };
            for i in 0..fl.width() as u32 {
                let a = mk(u, i);
                for &p in fl.node(i) {
                    let (x, oa) = split(p);
                    with_f(x, &mut ends);
                    for e in ends.chunks_exact(2) {
                        let (y, res) = (e[0], e[1]);
                        if let Some(ps) = gpar.get(&y) {
                            for &(j, ob) in ps {
                                let o = product(&live, oa, ob);
                                if o != ZREF {
                                    let k = cell(&mut cells, &mut keys, (a, mk(u, j)));
                                    let (pl, pr) = join(res, o);
                                    combos.push((k, pl, pr));
                                }
                            }
                        }
                        if skips && crosses(sh, &tg, y, u) {
                            // `y` is `true` on the other side here.
                            let k = cell(&mut cells, &mut keys, (a, y));
                            let (pl, pr) = join(res, oa);
                            combos.push((k, pl, pr));
                        }
                    }
                }
            }
            if skips {
                // Skips of `f` through this level against `g`'s nodes.
                for j in 0..gl.width() as u32 {
                    let b = mk(u, j);
                    for &p in gl.node(j) {
                        let (y, ob) = split(p);
                        ends.clear();
                        if let Some(v) = by_g.get(&y) {
                            for &(x, res) in v {
                                if sh.within(vt(x), cs) && crosses(sh, &tf, x, u) {
                                    ends.push(x);
                                    ends.push(res);
                                }
                            }
                        }
                        if sh.is_leaf(vt(y)) {
                            for x in leaf_refs(sh, &tf, vt(y), u) {
                                let res = leaf_and(vt(y), ix(x), ix(y));
                                if res != ZREF {
                                    ends.push(x);
                                    ends.push(res);
                                }
                            }
                        }
                        for e in ends.chunks_exact(2) {
                            let k = cell(&mut cells, &mut keys, (e[0], b));
                            let (pl, pr) = join(e[1], ob);
                            combos.push((k, pl, pr));
                        }
                    }
                }
            }
        }
        lim.charge_work(combos.len() as u64 + keys.len() as u64);
        lim.check_stop()?;
        if keys.is_empty() {
            continue;
        }
        // Each product's pairs, grouped by product.
        let mut start = vec![0u32; keys.len() + 1];
        for &(k, _, _) in &combos {
            start[k as usize + 1] += 1;
        }
        for k in 0..keys.len() {
            start[k + 1] += start[k];
        }
        let mut at = start.clone();
        let mut pairs = vec![(0, 0); combos.len()];
        for &(k, a, b) in &combos {
            pairs[at[k as usize] as usize] = (a, b);
            at[k as usize] += 1;
        }
        let mut buf = Vec::new();
        for (k, &(a, b)) in keys.iter().enumerate() {
            buf.clear();
            buf.extend_from_slice(&pairs[start[k] as usize..start[k + 1] as usize]);
            let res = match normalize(sh, u, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(u, out.push_node(u, &buf)),
            };
            live.insert((a, b), res);
            by_f.entry(a).or_default().push((b, res));
            by_g.entry(b).or_default().push((a, res));
        }
    }
    out.out = product(&live, f.out, g.out);
    out.collect_pushed();
    Ok(out)
}
