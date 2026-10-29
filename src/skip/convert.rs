//! Between the stored form, where every pair side names a node of its child
//! level, and the skip form.

use std::sync::Arc;

use super::conjoin::Numbered;
use super::{ix, mk, normalize, vt, Ref, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::diagram::{ChildPair, EncodedChildRef, NodeIdx, Tdd, TddNodeId, ZERO};
use crate::vtree::{Vtree, VtreeIdx};
use crate::{Engine, OperationError};

/// The skip form of `f`, `None` where a level is marginal or the diagram
/// carries weights. Any valid diagram converts; a canonical one converts to
/// the canonical skip form.
pub(crate) fn from_tdd(f: &Tdd, shape: Arc<Shape>) -> Option<SkipForm> {
    if f.weights.is_some() || f.levels.iter().any(|l| l.is_marginal()) {
        return None;
    }
    let sh = &*shape;
    let n = sh.num_nodes();
    let mut form = SkipForm::zero(Arc::clone(&shape));
    if f.is_zero() {
        return Some(form);
    }
    let root = sh.root;
    let levels = &f.levels;
    // A canonical diagram holds only nodes its output reaches, and its skip
    // form only nodes their referrers name.
    let canonical = levels.is_canonical(f.output);
    // The nodes the output reaches, top-down.
    let mut reach: Vec<Vec<bool>> = (0..n)
        .map(|i| if sh.is_leaf(i as u32) || canonical { Vec::new() } else { vec![false; levels[i].slot_count()] })
        .collect();
    if !sh.is_leaf(root) && !canonical {
        reach[root as usize][f.output().local.idx()] = true;
    }
    for &t in sh.topdown.iter().filter(|_| !canonical) {
        let (l, r) = sh.kids[t as usize];
        for i in 0..reach[t as usize].len() {
            if !reach[t as usize][i] {
                continue;
            }
            for pair in levels[t as usize].pairs_iter_of_idx(i) {
                if !sh.is_leaf(l) {
                    reach[l as usize][pair.left.raw() as usize] = true;
                }
                if !sh.is_leaf(r) {
                    reach[r as usize][pair.right.raw() as usize] = true;
                }
            }
        }
    }
    // Resolve bottom-up.
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut buf = Vec::new();
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let width = if sh.is_leaf(t) { 0 } else { levels[t as usize].slot_count() };
        let mut out = vec![ZREF; width];
        for i in 0..out.len() {
            if !canonical && !reach[t as usize][i] {
                continue;
            }
            buf.clear();
            for pair in levels[t as usize].pairs_iter_of_idx(i) {
                let side = |c: u32, raw: u32| if sh.is_leaf(c) { mk(c, raw) } else { res[c as usize][raw as usize] };
                buf.push((side(l, pair.left.raw()), side(r, pair.right.raw())));
            }
            out[i] = match normalize(sh, t, &mut buf) {
                Err(target) => target,
                Ok(()) => mk(t, form.levels[t as usize].push(&buf)),
            };
        }
        res[t as usize] = out;
    }
    let o = f.output().local.0;
    form.out = if sh.is_leaf(root) { mk(root, o) } else { res[root as usize][o as usize] };
    if canonical {
        form.active = sh.topdown.iter().copied().filter(|&t| form.levels[t as usize].width() > 0).collect();
    } else {
        form.collect();
    }
    Some(form)
}

/// The stored form of `s` over `vtree` (the vtree `s`'s shape was read
/// from): every level a skip crosses holds the node `true` on the skipped
/// variable makes, one per (level, target). The stored form of a canonical
/// skip form is canonical, but it is not certified as such here.
pub(crate) fn to_tdd(s: &SkipForm, eng: &Engine, vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        return Ok(Tdd::zero(vtree));
    }
    if sh.is_leaf(sh.root) {
        return Tdd::builder(eng, vtree)?
            .finish(TddNodeId { vtree: VtreeIdx(sh.root), local: NodeIdx(ix(s.out)) })
            .map_err(|_| OperationError::IndexOverflow);
    }
    let nb = Numbered::of(s);
    // Every reference, by number.
    let mut refs = vec![ZREF; nb.len()];
    for &t in &s.active {
        let b = nb.base[t as usize];
        for i in 0..s.levels[t as usize].width() as u32 {
            refs[(b + i) as usize] = mk(t, i);
        }
    }
    for &c in &nb.leaves {
        let b = nb.base[c as usize];
        for k in 0..3 {
            refs[(b + k) as usize] = mk(c, k);
        }
    }
    // A reference read at a level above its own has a pass-through node at
    // every level from the highest one it is read at down to its own:
    // per level, those references, numbered after the level's own nodes.
    let mut start = vec![0u32; n + 1];
    let chain = |x: Ref, top: u32, f: &mut dyn FnMut(u32)| {
        let mut c = top;
        while c != vt(x) {
            f(c);
            c = sh.toward(c, vt(x));
        }
    };
    for (id, &x) in refs.iter().enumerate() {
        let top = nb.top[id];
        if x != ZREF && top != NONE {
            chain(x, top, &mut |c| start[c as usize + 1] += 1);
        }
    }
    for t in 0..n {
        start[t + 1] += start[t];
    }
    let mut fill = start.clone();
    let mut pts = vec![0u32; start[n] as usize];
    for (id, &x) in refs.iter().enumerate() {
        let top = nb.top[id];
        if x != ZREF && top != NONE {
            chain(x, top, &mut |c| {
                pts[fill[c as usize] as usize] = id as u32;
                fill[c as usize] += 1;
            });
        }
    }
    // Per reference, its pass-through's index at the last level of its
    // chain written: the levels are written children first, and a
    // reference's next level up its chain is the parent of the last.
    let mut cur = vec![0u32; nb.len()];
    let width = |c: u32| if sh.is_leaf(c) { 0 } else { s.levels[c as usize].width() as u32 };
    let mut b = Tdd::builder(eng, vtree)?;
    let mut node: Vec<ChildPair> = Vec::new();
    for &t in &sh.bottomup {
        if sh.is_leaf(t) {
            continue;
        }
        let (l, r) = sh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        let here = &pts[start[t as usize] as usize..start[t as usize + 1] as usize];
        // A leaf child is named by literals in every pair of the level, or
        // by One in every pair; pass-through nodes name One there.
        let lit_l = sh.is_leaf(l) && lvl.pairs.iter().any(|p| ix(p.0) != ONE);
        let lit_r = sh.is_leaf(r) && lvl.pairs.iter().any(|p| ix(p.1) != ONE);
        let raw = |cur: &[u32], c: u32, x: Ref| -> u32 {
            if sh.is_leaf(c) || vt(x) == c { ix(x) } else { width(c) + cur[nb.id(x)] }
        };
        let expand = |node: &mut Vec<ChildPair>, left: u32, a_one: bool, right: u32, c_one: bool| {
            let lefts: &[u32] = if lit_l && a_one { &[POS, NEG] } else { std::slice::from_ref(&left) };
            for &la in lefts {
                if lit_r && c_one {
                    node.push(ChildPair::new(EncodedChildRef::from_raw(la), EncodedChildRef::from_raw(POS)));
                    node.push(ChildPair::new(EncodedChildRef::from_raw(la), EncodedChildRef::from_raw(NEG)));
                } else {
                    node.push(ChildPair::new(EncodedChildRef::from_raw(la), EncodedChildRef::from_raw(right)));
                }
            }
        };
        let grow = |one_l: bool, one_r: bool| (if lit_l && one_l { 2 } else { 1 }) * (if lit_r && one_r { 2 } else { 1 });
        let mut npairs: usize = lvl.pairs.iter().map(|p| grow(ix(p.0) == ONE && sh.is_leaf(l), ix(p.1) == ONE && sh.is_leaf(r))).sum();
        npairs += here.len() * grow(sh.is_leaf(l), sh.is_leaf(r));
        b.reserve(eng, VtreeIdx(t), lvl.width() + here.len(), npairs)?;
        for i in 0..lvl.width() {
            node.clear();
            for &(a, c) in lvl.node(i as u32) {
                let a_one = sh.is_leaf(l) && ix(a) == ONE;
                let c_one = sh.is_leaf(r) && ix(c) == ONE;
                expand(&mut node, raw(&cur, l, a), a_one, raw(&cur, r, c), c_one);
            }
            b.push(eng, VtreeIdx(t), &node)?;
        }
        for (k, &id) in here.iter().enumerate() {
            let x = refs[id as usize];
            node.clear();
            if sh.within(vt(x), l) {
                expand(&mut node, raw(&cur, l, x), false, ONE, sh.is_leaf(r));
            } else {
                expand(&mut node, ONE, sh.is_leaf(l), raw(&cur, r, x), false);
            }
            b.push(eng, VtreeIdx(t), &node)?;
            cur[id as usize] = k as u32;
        }
    }
    let o = if vt(s.out) == sh.root { ix(s.out) } else { width(sh.root) + cur[nb.id(s.out)] };
    // SAFETY: every pair side names a node written at its child level
    // (pass-through nodes before their parents), or a leaf label.
    let tdd = unsafe { b.finish_unchecked(TddNodeId { vtree: VtreeIdx(sh.root), local: NodeIdx(o) }) };
    debug_assert!(o != ZERO.0);
    Ok(tdd)
}
