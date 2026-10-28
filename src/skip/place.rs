//! Placing a skip-form diagram on a larger vtree whose restriction to its
//! variables is its own vtree.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{ix, mk, normalize, vt, Ref, Shape, SkipForm, NEG, ONE, POS, ZREF};
use crate::vtree::{VarId, Vtree};

/// `s` (over `from`) on `into`, whose shape is `ish`, each variable renamed
/// by `map`: every level of `into` the image crosses holds `true` on its
/// other side. `None` where `into` does not hold `from`'s shape with the
/// children in the same order.
pub(crate) fn place(s: &SkipForm, from: &Vtree, map: impl Fn(VarId) -> VarId, into: &Vtree, ish: &Arc<Shape>) -> Option<SkipForm> {
    let fsh = &*s.shape;
    let sh = &**ish;
    let n = fsh.num_nodes();
    let mut img = vec![0u32; n];
    for i in 0..n {
        if fsh.is_leaf(i as u32) {
            img[i] = into.leaf_of(map(VarId(fsh.var[i])))?.0;
        }
    }
    for &t in &fsh.bottomup {
        let (l, r) = fsh.kids[t as usize];
        let (a, b) = (img[l as usize], img[r as usize]);
        let w = sh.lca(a, b);
        if w == a || w == b {
            return None;
        }
        let (wl, _) = sh.kids[w as usize];
        if !sh.within(a, wl) {
            return None;
        }
        img[t as usize] = w;
    }
    let _ = from;
    let mut out = SkipForm::zero(Arc::clone(ish));
    if s.out == ZREF {
        return Some(out);
    }
    // `true` over a subtree of `into`, as a reference read at its root.
    let mut tops: FxHashMap<u32, Ref> = FxHashMap::default();
    fn top(sh: &Shape, out: &mut SkipForm, tops: &mut FxHashMap<u32, Ref>, t: u32) -> Ref {
        if let Some(&r) = tops.get(&t) {
            return r;
        }
        let r = if sh.is_leaf(t) {
            mk(t, ONE)
        } else {
            let (l, rr) = sh.kids[t as usize];
            let (a, b) = (top(sh, out, tops, l), top(sh, out, tops, rr));
            let mut pair = vec![(a, b)];
            match normalize(sh, t, &mut pair) {
                Err(x) => x,
                Ok(()) => mk(t, out.levels[t as usize].push(&pair)),
            }
        };
        tops.insert(t, r);
        r
    }
    // Which level of `from` each level of `into` is the image of, and
    // whether a level of `from` names each leaf child by its literals.
    let mut of_img = vec![u32::MAX; sh.num_nodes()];
    let mut lits = vec![(false, false); n];
    for &t in &fsh.topdown {
        of_img[img[t as usize] as usize] = t;
        let (fl, fr) = fsh.kids[t as usize];
        let pairs = &s.levels[t as usize].pairs;
        lits[t as usize] = (
            fsh.is_leaf(fl) && pairs.iter().any(|p| ix(p.0) != ONE),
            fsh.is_leaf(fr) && pairs.iter().any(|p| ix(p.1) != ONE),
        );
    }
    let mut memo: FxHashMap<Vec<(Ref, Ref)>, Ref> = FxHashMap::default();
    // `true` on the side of `into`'s level `u` off the path a reference
    // crosses, as the classes there: where `u` is the image of a level of
    // `from` that names that side's leaf by its literals, both literals.
    let off = |out: &mut SkipForm, tops: &mut FxHashMap<u32, Ref>, memo: &mut FxHashMap<Vec<(Ref, Ref)>, Ref>, u: u32, o: u32, buf: &mut Vec<Ref>| {
        buf.clear();
        let t = of_img[u as usize];
        if t != u32::MAX && !sh.is_leaf(o) {
            let (fl, fr) = fsh.kids[t as usize];
            let (y, lit) = if sh.within(img[fl as usize], o) { (fl, lits[t as usize].0) } else { (fr, lits[t as usize].1) };
            if lit {
                // The leaf's literals, lifted to `o` through levels that
                // hold no variable of `from`.
                for label in [POS, NEG] {
                    let mut cur = mk(img[y as usize], label);
                    let mut at = img[y as usize];
                    while at != o {
                        let p = sh.parent[at as usize];
                        let (pl, pr) = sh.kids[p as usize];
                        let mut pair = vec![if pl == at { (cur, top(sh, out, tops, pr)) } else { (top(sh, out, tops, pl), cur) }];
                        cur = match normalize(sh, p, &mut pair) {
                            Err(x) => x,
                            Ok(()) => *memo.entry(pair.clone()).or_insert_with(|| mk(p, out.levels[p as usize].push(&pair))),
                        };
                        at = p;
                    }
                    buf.push(cur);
                }
                return;
            }
        }
        buf.push(top(sh, out, tops, o));
    };
    let mut sides = Vec::new();
    // The reference read at `into`'s level `c` to `x`, the resolved
    // reference of a node of `into`'s level `w` at or below `c`.
    let mut lift = |out: &mut SkipForm, tops: &mut FxHashMap<u32, Ref>, c: u32, w: u32, x: Ref| -> Ref {
        let mut cur = x;
        let mut at = w;
        while at != c {
            let u = sh.parent[at as usize];
            let (l, r) = sh.kids[u as usize];
            let left = l == at;
            off(out, tops, &mut memo, u, if left { r } else { l }, &mut sides);
            let mut p: Vec<(Ref, Ref)> = sides.iter().map(|&o| if left { (cur, o) } else { (o, cur) }).collect();
            cur = match normalize(sh, u, &mut p) {
                Err(t) => t,
                Ok(()) => *memo.entry(p.clone()).or_insert_with(|| mk(u, out.levels[u as usize].push(&p))),
            };
            at = u;
        }
        cur
    };
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let image = |res: &Vec<Vec<Ref>>, x: Ref| -> (u32, Ref) {
        let w = img[vt(x) as usize];
        if fsh.is_leaf(vt(x)) { (w, mk(w, ix(x))) } else { (w, res[vt(x) as usize][ix(x) as usize]) }
    };
    let mut buf = Vec::new();
    let mut cells = Vec::new();
    for &t in &fsh.bottomup {
        let u = img[t as usize];
        let (l, r) = sh.kids[u as usize];
        let (fl, fr) = fsh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        // A leaf child whose image is below an inner child of `u` names
        // classes there, which must not overlap: where the level names a
        // literal of it, `One` stands for both literals.
        let split = |c: u32, side: fn(&(Ref, Ref)) -> Ref| {
            fsh.is_leaf(c) && img[c as usize] != if side(&(0, 1)) == 0 { l } else { r } && lvl.pairs.iter().any(|p| ix(side(p)) != ONE)
        };
        let (sl, sr) = (split(fl, |p| p.0), split(fr, |p| p.1));
        let mut rt = Vec::with_capacity(lvl.width());
        for i in 0..lvl.width() {
            cells.clear();
            for &(a, b) in lvl.node(i as u32) {
                let las: &[Ref] = if sl && ix(a) == ONE { &[mk(fl, POS), mk(fl, NEG)] } else { &[a] };
                for &a in las {
                    if sr && ix(b) == ONE {
                        cells.push((a, mk(fr, POS)));
                        cells.push((a, mk(fr, NEG)));
                    } else {
                        cells.push((a, b));
                    }
                }
            }
            buf.clear();
            for &(a, b) in &cells {
                let (wa, xa) = image(&res, a);
                let (wb, xb) = image(&res, b);
                let a = lift(&mut out, &mut tops, l, wa, xa);
                let b = lift(&mut out, &mut tops, r, wb, xb);
                buf.push((a, b));
            }
            rt.push(match normalize(sh, u, &mut buf) {
                Err(x) => x,
                Ok(()) => mk(u, out.levels[u as usize].push(&buf)),
            });
        }
        res[t as usize] = rt;
    }
    let (w, x) = image(&res, s.out);
    out.out = lift(&mut out, &mut tops, sh.root, w, x);
    out.collect();
    Some(out)
}
