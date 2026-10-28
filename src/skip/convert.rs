//! Between the stored form, where every pair side names a node of its child
//! level, and the skip form.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{ix, mk, normalize, vt, Ref, Shape, SkipForm, ONE, NEG, POS, ZREF};
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
    // The nodes the output reaches, top-down.
    let mut reach: Vec<Vec<bool>> =
        (0..n).map(|i| if sh.is_leaf(i as u32) { Vec::new() } else { vec![false; levels[i].slot_count()] }).collect();
    if !sh.is_leaf(root) {
        reach[root as usize][f.output().local.idx()] = true;
    }
    for &t in &sh.topdown {
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
        let mut out = vec![ZREF; reach[t as usize].len()];
        for i in 0..out.len() {
            if !reach[t as usize][i] {
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
    form.collect();
    Some(form)
}

/// The stored form of `s` over `vtree` (the vtree `s`'s shape was read
/// from): every level a skip crosses holds the node `true` on the skipped
/// variable makes. Not reduced: the caller minimizes where it needs the
/// canonical diagram.
pub(crate) fn to_tdd(s: &SkipForm, eng: &Engine, vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        return Ok(Tdd::zero(vtree));
    }
    // The pass-through nodes each level needs, by target, in first-need order.
    let mut pts: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut pt_index: Vec<FxHashMap<Ref, u32>> = (0..n).map(|_| FxHashMap::default()).collect();
    let need = |pts: &mut Vec<Vec<Ref>>, pt_index: &mut Vec<FxHashMap<Ref, u32>>, mut c: u32, r: Ref| {
        while vt(r) != c {
            let m = &mut pt_index[c as usize];
            if m.contains_key(&r) {
                return;
            }
            m.insert(r, pts[c as usize].len() as u32);
            pts[c as usize].push(r);
            c = sh.toward(c, vt(r));
        }
    };
    need(&mut pts, &mut pt_index, sh.root, s.out);
    for &t in &sh.topdown {
        let (l, r) = sh.kids[t as usize];
        for &(a, b) in &s.levels[t as usize].pairs {
            need(&mut pts, &mut pt_index, l, a);
            need(&mut pts, &mut pt_index, r, b);
        }
    }
    let raw = |c: u32, x: Ref| -> u32 {
        if sh.is_leaf(c) {
            ix(x)
        } else if vt(x) == c {
            ix(x)
        } else {
            s.levels[c as usize].width() as u32 + pt_index[c as usize][&x]
        }
    };
    let mut b = Tdd::builder(eng, vtree)?;
    let mut node = Vec::new();
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        // A leaf child is named by literals in every pair of the level, or
        // by One in every pair.
        let lit = |c: u32, pick: fn(&(Ref, Ref)) -> Ref| {
            sh.is_leaf(c)
                && (lvl.pairs.iter().any(|p| ix(pick(p)) != ONE)
                    || pts[t as usize].iter().any(|&x| vt(x) == c && ix(x) != ONE))
        };
        let (lit_l, lit_r) = (lit(l, |p| p.0), lit(r, |p| p.1));
        let mut emit = |b: &mut crate::diagram::TddBuilder, pairs: &mut dyn Iterator<Item = (Ref, Ref)>| -> Result<(), OperationError> {
            node.clear();
            for (a, c) in pairs {
                let lefts: &[u32] = if lit_l && ix(a) == ONE { &[POS, NEG] } else { &[u32::MAX] };
                for &la in lefts {
                    let left = if la == u32::MAX { raw(l, a) } else { la };
                    if lit_r && ix(c) == ONE {
                        node.push(ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(POS)));
                        node.push(ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(NEG)));
                    } else {
                        node.push(ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(raw(r, c))));
                    }
                }
            }
            b.push(eng, VtreeIdx(t), &node)?;
            Ok(())
        };
        for i in 0..lvl.width() {
            emit(&mut b, &mut lvl.node(i as u32).iter().copied())?;
        }
        for &x in &pts[t as usize] {
            let pair = if sh.within(vt(x), l) { (x, mk(r, ONE)) } else { (mk(l, ONE), x) };
            emit(&mut b, &mut std::iter::once(pair))?;
        }
    }
    let root = sh.root;
    let o = raw(root, s.out);
    let tdd = b.finish(TddNodeId { vtree: VtreeIdx(root), local: NodeIdx(o) }).map_err(|_| OperationError::IndexOverflow)?;
    debug_assert!(o != ZERO.0);
    Ok(tdd)
}
