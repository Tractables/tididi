//! The tagged skip form: a reference that skips carries, in its top two
//! bits, the value every skipped one-variable side takes: `true`, `x` or
//! `¬x` (the leaf labels `One`, `Pos`, `Neg`). A node `lit_x ∧ D` at a
//! level whose other child is the leaf `x` is not stored; a reference to it
//! names `D` with the tag `lit`.
//!
//! One tag covers every level a reference crosses (the chain rule), so a
//! node `lit ∧ D` whose `D` is itself skipped with another tag stays
//! stored: it starts a run. A reference that names a node of its own child
//! level has tag zero, so a product of two references meets the levels
//! both cross with the conjunction of their tags, and the product of `x`
//! and `¬x` there is false.

use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::build::Land;
use super::{ix, mk, vt, Ref, Shape, SkipForm, SkipSize, ONE, POS, NEG, ZREF};
use crate::limits::Limits;
use crate::OperationError;

pub(crate) const TAG_SHIFT: u32 = 62;
/// The tag bits of a reference.
pub(crate) const TAG_MASK: Ref = 3 << TAG_SHIFT;

/// A reference's tag: a leaf label, `One` for `true`.
#[inline]
pub(crate) fn tag(r: Ref) -> u32 {
    (r >> TAG_SHIFT) as u32
}

#[inline]
pub(crate) fn with_tag(r: Ref, t: u32) -> Ref {
    (r & !TAG_MASK) | (t as u64) << TAG_SHIFT
}

#[inline]
pub(crate) fn strip(r: Ref) -> Ref {
    r & !TAG_MASK
}

/// The conjunction of two leaf labels, `None` for false.
#[inline]
fn label_and(a: u32, b: u32) -> Option<u32> {
    match (a, b) {
        (ONE, x) | (x, ONE) => Some(x),
        (POS, POS) => Some(POS),
        (NEG, NEG) => Some(NEG),
        _ => None,
    }
}

/// The reference read at `c` to the node `r` names when read at `w`, every
/// level from `c` down to `w`'s parent skipped with tag `t`; `None` where no
/// single reference does (the tag changes below `c`).
///
/// A node `(a, b)` at a level of two leaves is named through either leaf:
/// its right leaf's label `b` tagged `a`, or its left leaf's label `a`
/// tagged `b`. The right leaf is preferred; the left one is used where the
/// levels above carry `b` and not `a`.
#[inline]
pub(crate) fn extend(sh: &Shape, c: u32, t: u32, w: u32, r: Ref) -> Option<Ref> {
    if r == ZREF || w == c {
        return Some(r);
    }
    if sh.is_leaf(w) {
        let u = sh.parent[w as usize];
        let (l, rr) = sh.kids[u as usize];
        debug_assert!(sh.is_leaf(l) && sh.is_leaf(rr));
        let lab = ix(r);
        return Some(if w == l && (u == c || lab == t) { with_tag(mk(rr, t), lab) } else { with_tag(mk(w, lab), t) });
    }
    if vt(r) == w || tag(r) == t {
        return Some(with_tag(r, t));
    }
    // `r` names a node of `w` through one of `w`'s leaves.
    let (l, rr) = sh.kids[w as usize];
    if sh.is_leaf(l) && sh.is_leaf(rr) {
        let (a, b) = if vt(r) == rr { (tag(r), ix(r)) } else { (ix(r), tag(r)) };
        if a == t {
            return Some(with_tag(mk(rr, b), a));
        }
        if b == t {
            return Some(with_tag(mk(l, a), b));
        }
    }
    None
}

/// A reference read at `c` in the form [`extend`] gives at `c`: a node of
/// `c`'s two leaves named through its right leaf.
#[inline]
fn at_level(sh: &Shape, c: u32, e: Ref) -> Ref {
    let w = vt(e);
    if w != c && sh.is_leaf(w) && !sh.is_leaf(c) && sh.parent[w as usize] == c && sh.kids[c as usize].0 == w {
        with_tag(mk(sh.kids[c as usize].1, tag(e)), ix(e))
    } else {
        e
    }
}

/// [`super::normalize`] for the tagged form: a node of one pair `(lit, D)`
/// against a leaf resolves to `D` tagged `lit` where [`extend`] names it.
pub(crate) fn normalize(sh: &Shape, u: u32, pairs: &mut Vec<(Ref, Ref)>) -> Result<(), Ref> {
    super::merge_cells(sh, u, pairs);
    let (l, r) = sh.kids[u as usize];
    if pairs.len() == 1 {
        let (a, b) = pairs[0];
        if sh.is_leaf(l) {
            if let Some(x) = extend(sh, u, ix(a), r, b) {
                return Err(x);
            }
        }
        if sh.is_leaf(r) {
            if let Some(x) = extend(sh, u, ix(b), l, a) {
                return Err(x);
            }
        }
    }
    Ok(())
}

/// The pairs of the node `r` names, read at level `u` (tagged form).
#[inline]
pub(crate) fn pairs_at<'a>(s: &'a SkipForm, u: u32, r: Ref, buf: &'a mut [(Ref, Ref); 1]) -> &'a [(Ref, Ref)] {
    if vt(r) == u {
        return s.levels[u as usize].node(ix(r));
    }
    let sh = &*s.shape;
    let (l, rr) = sh.kids[u as usize];
    let t = tag(r);
    buf[0] = if sh.within(vt(r), l) {
        let cont = if vt(r) == l { strip(r) } else { r };
        (cont, mk(rr, t))
    } else {
        let cont = if vt(r) == rr { strip(r) } else { r };
        (mk(l, t), cont)
    };
    &buf[..]
}

/// Where the product of `a` and `b`, read at one child level, is built:
/// the result, or the level, the key there, and the tag of the levels both
/// cross above it.
#[inline]
fn land_pair(sh: &Shape, c: u32, a: Ref, b: Ref) -> (Land, (Ref, Ref), u32) {
    const NO: (Ref, Ref) = (ZREF, ZREF);
    if a == ZREF || b == ZREF {
        return (Land::Done(ZREF), NO, 0);
    }
    let (va, vb) = (vt(a), vt(b));
    let parted = va != vb && !sh.within(vb, va) && !sh.within(va, vb);
    if parted && sh.parent[va as usize] == c {
        // The two leaves of `c`: each skips the other, so no level is
        // crossed by both.
        return (Land::At(c), (a, b), ONE);
    }
    let Some(t) = label_and(tag(a), tag(b)) else {
        return (Land::Done(ZREF), NO, 0);
    };
    if va == vb {
        if sh.is_leaf(va) {
            return match label_and(ix(a), ix(b)) {
                Some(x) => (Land::Done(extend(sh, c, t, va, mk(va, x)).unwrap()), NO, 0),
                None => (Land::Done(ZREF), NO, 0),
            };
        }
        return (Land::At(va), (strip(a), strip(b)), t);
    }
    if sh.within(vb, va) {
        return (Land::At(va), (strip(a), b), t);
    }
    if sh.within(va, vb) {
        return (Land::At(vb), (a, strip(b)), t);
    }
    debug_assert!(sh.is_leaf(va) && sh.is_leaf(vb) && sh.parent[va as usize] == sh.parent[vb as usize]);
    (Land::At(sh.parent[va as usize]), (a, b), t)
}

/// Boundary nodes a reference needs where its tag changes, one per level
/// and pair.
#[derive(Default)]
struct Bounds {
    memo: FxHashMap<(u32, Ref, Ref), u32>,
}

impl Bounds {
    /// The reference read at `c` to the node `r` resolves `w`'s product
    /// to, every level from `c` down to `w`'s parent skipped with tag `t`.
    #[inline]
    fn cross(&mut self, sh: &Shape, form: &mut SkipForm, c: u32, t: u32, w: u32, r: Ref) -> Ref {
        if let Some(x) = extend(sh, c, t, w, r) {
            return x;
        }
        // The run changes below `w`'s parent: that node is stored.
        let u = sh.parent[w as usize];
        let (l, rr) = sh.kids[u as usize];
        let pair = if l == w { (r, mk(rr, t)) } else { (mk(l, t), r) };
        let idx = *self.memo.entry((u, pair.0, pair.1)).or_insert_with(|| form.levels[u as usize].push(&[pair]));
        let node = mk(u, idx);
        if u == c { node } else { with_tag(node, t) }
    }
}

/// A pending side's handle: tag bits 3 (no reference but [`ZREF`] has
/// them), then the tag of the levels above it, the level and the slot.
const PENDING: u64 = 3 << 62;

#[inline]
fn is_pending(h: u64) -> bool {
    h >> 62 == 3 && h != ZREF
}

/// [`super::build::construct`] for the tagged form: keys land with the tag
/// of the levels above them, and a side's reference is resolved through
/// [`Bounds::cross`].
fn construct<K: Copy + Eq + Hash>(
    shape: &Arc<Shape>,
    land: &mut dyn FnMut(u32, K) -> (Land, K, u32),
    expand: &mut dyn FnMut(u32, K, &mut Vec<(K, K)>),
    root: K,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let sh = &**shape;
    let n = sh.num_nodes();
    let mut form = SkipForm::zero(Arc::clone(shape));
    let mut maps: Vec<FxHashMap<K, u32>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut keys: Vec<Vec<K>> = vec![Vec::new(); n];
    let mut cstart: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut combos: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n];
    let insert = |maps: &mut Vec<FxHashMap<K, u32>>, keys: &mut Vec<Vec<K>>, u: u32, k: K, t: u32| -> u64 {
        let m = &mut maps[u as usize];
        let next = keys[u as usize].len() as u32;
        let slot = *m.entry(k).or_insert_with(|| {
            keys[u as usize].push(k);
            next
        });
        debug_assert!(u < 1 << 26);
        PENDING | (t as u64) << 58 | (u as u64) << 32 | slot as u64
    };
    let handle = |maps: &mut Vec<FxHashMap<K, u32>>, keys: &mut Vec<Vec<K>>, l: (Land, K, u32)| -> u64 {
        match l {
            (Land::Done(r), _, _) => r,
            (Land::At(w), k, t) => insert(maps, keys, w, k, t),
        }
    };
    let root_l = land(sh.root, root);
    if let (Land::Done(r), _, _) = root_l {
        form.out = r;
        return Ok(form);
    }
    let root_h = handle(&mut maps, &mut keys, root_l);
    let mut buf = Vec::new();
    for &u in &sh.topdown {
        if keys[u as usize].is_empty() {
            continue;
        }
        let (l, r) = sh.kids[u as usize];
        let mut cs = Vec::with_capacity(keys[u as usize].len() + 1);
        let mut out = Vec::new();
        cs.push(0u32);
        let mut slot = 0;
        while slot < keys[u as usize].len() {
            let key = keys[u as usize][slot];
            buf.clear();
            expand(u, key, &mut buf);
            for &(kl, kr) in &buf {
                let ll = land(l, kl);
                if matches!(ll.0, Land::Done(ZREF)) {
                    continue;
                }
                let lr = land(r, kr);
                if matches!(lr.0, Land::Done(ZREF)) {
                    continue;
                }
                let hl = handle(&mut maps, &mut keys, ll);
                let hr = handle(&mut maps, &mut keys, lr);
                out.push((hl, hr));
            }
            cs.push(out.len() as u32);
            slot += 1;
        }
        lim.charge_work(out.len() as u64 + keys[u as usize].len() as u64);
        lim.check_stop()?;
        maps[u as usize] = FxHashMap::default();
        cstart[u as usize] = cs;
        combos[u as usize] = out;
    }
    drop(maps);
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut bounds = Bounds::default();
    let get = |res: &Vec<Vec<Ref>>, bounds: &mut Bounds, form: &mut SkipForm, c: u32, h: u64| -> Ref {
        if is_pending(h) {
            let (t, w, slot) = (((h >> 58) & 3) as u32, ((h >> 32) & 0x03ff_ffff) as u32, h as u32);
            bounds.cross(sh, form, c, t, w, res[w as usize][slot as usize])
        } else {
            h
        }
    };
    let mut pairs = Vec::new();
    for &u in &sh.bottomup {
        let cs = std::mem::take(&mut cstart[u as usize]);
        if cs.is_empty() {
            continue;
        }
        let (l, r) = sh.kids[u as usize];
        let out = std::mem::take(&mut combos[u as usize]);
        let mut rv = Vec::with_capacity(cs.len() - 1);
        for s in 0..cs.len() - 1 {
            pairs.clear();
            for &(hl, hr) in &out[cs[s] as usize..cs[s + 1] as usize] {
                let a = get(&res, &mut bounds, &mut form, l, hl);
                if a == ZREF {
                    continue;
                }
                let b = get(&res, &mut bounds, &mut form, r, hr);
                if b != ZREF {
                    pairs.push((a, b));
                }
            }
            rv.push(if pairs.is_empty() {
                ZREF
            } else {
                match normalize(sh, u, &mut pairs) {
                    Err(target) => target,
                    Ok(()) => mk(u, form.levels[u as usize].push(&pairs)),
                }
            });
        }
        res[u as usize] = rv;
        keys[u as usize] = Vec::new();
    }
    form.out = get(&res, &mut bounds, &mut form, sh.root, root_h);
    form.collect();
    Ok(form)
}

/// `f ∧ g` in the tagged form, pruned, not contracted.
pub(crate) fn and(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    debug_assert!(Arc::ptr_eq(&f.shape, &g.shape));
    let sh = &*f.shape;
    let mut land = |c: u32, (a, b): (Ref, Ref)| land_pair(sh, c, a, b);
    let mut expand = |u: u32, (a, b): (Ref, Ref), out: &mut Vec<((Ref, Ref), (Ref, Ref))>| {
        let (mut ba, mut bb) = ([(0, 0); 1], [(0, 0); 1]);
        let pa = pairs_at(f, u, a, &mut ba);
        let pb = pairs_at(g, u, b, &mut bb);
        for &(la, ra) in pa {
            for &(lb, rb) in pb {
                out.push(((la, lb), (ra, rb)));
            }
        }
    };
    construct(&f.shape, &mut land, &mut expand, (f.out, g.out), lim)
}

const NONE: u32 = u32::MAX;

/// Twin contraction in the tagged form ([`super::contract::contract`]): a
/// skip that merges into a class contributes the pair its tag makes.
///
/// A skip owns one cell of each level it crosses, so the node it lands on
/// may also be named directly from the level above. A skip that no merge
/// takes in is a class of one pair at each level it crosses, and what it
/// continues to below is no twin of anything: it is kept apart.
pub(crate) fn contract(s: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        return Ok(SkipForm::zero(s.shape.clone()));
    }
    let mut classes: Vec<Vec<Vec<(Ref, Ref)>>> = vec![Vec::new(); n];
    let mut classmap: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut chainmap: Vec<FxHashMap<Ref, u32>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut occ: Vec<(Ref, u32, Ref)> = Vec::new();
    let mut buf1 = [(0, 0); 1];
    // Per level: the skips through it no merge took in.
    let mut lone: Vec<Vec<Ref>> = vec![Vec::new(); n];
    if vt(s.out) != sh.root {
        lone[sh.root as usize].push(s.out);
    }
    for &c in &sh.topdown {
        let lvl = &s.levels[c as usize];
        let mut cmap = vec![NONE; lvl.width()];
        let mut cls: Vec<Vec<(Ref, Ref)>> = Vec::new();
        let p = sh.parent[c as usize];
        if p != NONE {
            // What the parent level's lone skips continue to here.
            let mut pinned: rustc_hash::FxHashSet<Ref> = rustc_hash::FxHashSet::default();
            for &x in &lone[p as usize] {
                if sh.within(vt(x), c) {
                    pinned.insert(if vt(x) == c { strip(x) } else { at_level(sh, c, x) });
                }
            }
            let (pl, pr) = sh.kids[p as usize];
            let left = pl == c;
            let sib = if left { pr } else { pl };
            let sib_leaf = sh.is_leaf(sib);
            occ.clear();
            for (k, pairs) in classes[p as usize].iter().enumerate() {
                for &(a, b) in pairs {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    let e = at_level(sh, c, e);
                    if sib_leaf && ix(partner) == ONE {
                        occ.push((e, k as u32, mk(sib, POS)));
                        occ.push((e, k as u32, mk(sib, NEG)));
                    } else {
                        occ.push((e, k as u32, partner));
                    }
                }
            }
            occ.sort_unstable();
            let mut keyed: FxHashMap<Vec<(u32, Ref)>, u32> = FxHashMap::default();
            let mut members: Vec<Vec<Ref>> = Vec::new();
            let mut i = 0;
            while i < occ.len() {
                let mut j = i;
                while j < occ.len() && occ[j].0 == occ[i].0 {
                    j += 1;
                }
                let e = occ[i].0;
                if pinned.contains(&e) {
                    members.push(vec![e]);
                } else {
                    let key: Vec<(u32, Ref)> = occ[i..j].iter().map(|o| (o.1, o.2)).collect();
                    let g = *keyed.entry(key).or_insert_with(|| {
                        members.push(Vec::new());
                        (members.len() - 1) as u32
                    });
                    members[g as usize].push(e);
                }
                i = j;
            }
            for &e in &pinned {
                if vt(e) != c && occ.binary_search_by_key(&e, |o| o.0).is_err() {
                    lone[c as usize].push(e);
                }
            }
            for m in members {
                if m.len() == 1 && vt(m[0]) != c {
                    lone[c as usize].push(m[0]);
                    continue;
                }
                let k = cls.len() as u32;
                let mut pairs = Vec::new();
                for &e in &m {
                    if vt(e) == c {
                        cmap[ix(e) as usize] = k;
                        pairs.extend_from_slice(lvl.node(ix(e)));
                    } else {
                        chainmap[c as usize].insert(e, k);
                        pairs.extend_from_slice(pairs_at(s, c, e, &mut buf1));
                    }
                }
                cls.push(pairs);
            }
        }
        for i in 0..lvl.width() {
            if cmap[i] == NONE {
                cmap[i] = cls.len() as u32;
                cls.push(lvl.node(i as u32).to_vec());
            }
        }
        lim.charge_work(lvl.pairs.len() as u64);
        classes[c as usize] = cls;
        classmap[c as usize] = cmap;
    }
    lim.check_stop()?;
    let mut out = SkipForm::zero(s.shape.clone());
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut bounds = Bounds::default();
    let side = |res: &Vec<Vec<Ref>>, bounds: &mut Bounds, out: &mut SkipForm, d: u32, h: Ref| -> Ref {
        if sh.is_leaf(d) {
            return h;
        }
        if vt(h) == d {
            return res[d as usize][classmap[d as usize][ix(h) as usize] as usize];
        }
        if let Some(&k) = chainmap[d as usize].get(&at_level(sh, d, h)) {
            return res[d as usize][k as usize];
        }
        let w = vt(h);
        let target = if sh.is_leaf(w) { strip(h) } else { res[w as usize][classmap[w as usize][ix(h) as usize] as usize] };
        bounds.cross(sh, out, d, tag(h), w, target)
    };
    let mut buf = Vec::new();
    for &c in &sh.bottomup {
        let (l, r) = sh.kids[c as usize];
        let cls = std::mem::take(&mut classes[c as usize]);
        let mut rc = Vec::with_capacity(cls.len());
        for pairs in &cls {
            buf.clear();
            for &(a, b) in pairs {
                let a = side(&res, &mut bounds, &mut out, l, a);
                let b = side(&res, &mut bounds, &mut out, r, b);
                buf.push((a, b));
            }
            buf.sort_unstable();
            buf.dedup();
            rc.push(match normalize(sh, c, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(c, out.levels[c as usize].push(&buf)),
            });
        }
        res[c as usize] = rc;
    }
    out.out = side(&res, &mut bounds, &mut out, sh.root, s.out);
    out.collect();
    Ok(out)
}

/// The tagged form of `s`, tagged or untagged (an untagged reference that
/// skips skips with `true`): canonical when `s` is.
pub(crate) fn tag_form(s: &SkipForm) -> SkipForm {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    let mut out = SkipForm::zero(s.shape.clone());
    if s.out == ZREF {
        return out;
    }
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut bounds = Bounds::default();
    let side = |res: &Vec<Vec<Ref>>, bounds: &mut Bounds, out: &mut SkipForm, d: u32, h: Ref| -> Ref {
        let w = vt(h);
        let target = if sh.is_leaf(w) { strip(h) } else { res[w as usize][ix(h) as usize] };
        if w == d { target } else { bounds.cross(sh, out, d, tag(h), w, target) }
    };
    let mut buf = Vec::new();
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        let mut rt = Vec::with_capacity(lvl.width());
        for i in 0..lvl.width() {
            buf.clear();
            for &(a, b) in lvl.node(i as u32) {
                let a = side(&res, &mut bounds, &mut out, l, a);
                let b = side(&res, &mut bounds, &mut out, r, b);
                buf.push((a, b));
            }
            rt.push(match normalize(sh, t, &mut buf) {
                Err(x) => x,
                Ok(()) => mk(t, out.levels[t as usize].push(&buf)),
            });
        }
        res[t as usize] = rt;
    }
    out.out = side(&res, &mut bounds, &mut out, sh.root, s.out);
    out.collect();
    out
}

/// The untagged form of a tagged `s`: every level a reference skips with
/// `x` or `¬x` stores its node. Canonical when `s` is.
pub(crate) fn untag(s: &SkipForm) -> SkipForm {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    let mut out = SkipForm::zero(s.shape.clone());
    if s.out == ZREF {
        return out;
    }
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut memo: FxHashMap<(u32, Ref, Ref), Ref> = FxHashMap::default();
    let side = |res: &Vec<Vec<Ref>>, memo: &mut FxHashMap<(u32, Ref, Ref), Ref>, out: &mut SkipForm, d: u32, h: Ref| -> Ref {
        let w = vt(h);
        let mut cur = if sh.is_leaf(w) { strip(h) } else { res[w as usize][ix(h) as usize] };
        if w == d {
            return cur;
        }
        let t = tag(h);
        // From `w` up to `d`, each crossed level's node `t ∧ cur`.
        let mut c = w;
        loop {
            let u = sh.parent[c as usize];
            let (l, r) = sh.kids[u as usize];
            let mut pair = vec![if l == c { (cur, mk(r, t)) } else { (mk(l, t), cur) }];
            cur = match super::normalize(sh, u, &mut pair) {
                Err(x) => x,
                Ok(()) => *memo.entry((u, pair[0].0, pair[0].1)).or_insert_with(|| mk(u, out.levels[u as usize].push(&pair))),
            };
            if u == d {
                return cur;
            }
            c = u;
        }
    };
    let mut buf = Vec::new();
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        let mut rt = Vec::with_capacity(lvl.width());
        for i in 0..lvl.width() {
            buf.clear();
            for &(a, b) in lvl.node(i as u32) {
                let a = side(&res, &mut memo, &mut out, l, a);
                let b = side(&res, &mut memo, &mut out, r, b);
                buf.push((a, b));
            }
            rt.push(match super::normalize(sh, t, &mut buf) {
                Err(x) => x,
                Ok(()) => mk(t, out.levels[t as usize].push(&buf)),
            });
        }
        res[t as usize] = rt;
    }
    out.out = side(&res, &mut memo, &mut out, sh.root, s.out);
    out.collect();
    out
}

impl SkipForm {
    /// [`SkipForm::size`] for the tagged form: a forwarder is a distinct
    /// (child level, target); the tag is in the referencing side.
    pub(crate) fn size_tagged(&self) -> SkipSize {
        let sh = &self.shape;
        let mut s = SkipSize::default();
        let mut seen = rustc_hash::FxHashSet::default();
        for &t in &sh.topdown {
            let lvl = &self.levels[t as usize];
            s.nodes += lvl.width() as u64;
            s.pairs += lvl.pairs.len() as u64;
            let (l, r) = sh.kids[t as usize];
            for &(a, b) in &lvl.pairs {
                for (c, x) in [(l, a), (r, b)] {
                    if vt(x) != c {
                        s.skips += 1;
                        seen.insert((c, strip(x)));
                    }
                }
            }
        }
        if self.out != ZREF && vt(self.out) != sh.root {
            s.skips += 1;
            seen.insert((sh.root, strip(self.out)));
        }
        s.forwarders = seen.len() as u64;
        s
    }
}
