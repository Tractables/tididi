//! Negation in skip form.
//!
//! The classes of `¬f` at a level are the classes of `f` there, less those
//! on which `f` is true whatever the variables outside the level are, plus
//! one new class: the assignments no class of `f` holds (`Z`). A kept class
//! keeps its pairs. `Z`'s pairs are the cells of the level's grid no class
//! owns: against a leaf child, the literal a pair leaves out of a column;
//! between two inner children, every pair of the classes the level names
//! that no node holds; and `Z` of a child against the other child's
//! classes. A skip from above owns whole columns, so it adds no cell to
//! `Z` except where it ends on a leaf label of a level whose children are
//! both leaves. The output of `¬f` is the root's `Z`.

use rustc_hash::{FxHashMap, FxHashSet};

use super::{build_top, filled, ix, mk, normalize, vt, Ref, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::limits::{Charged, Limits};
use crate::OperationError;

/// The literal cells a leaf label covers: bit 0 positive, bit 1 negative.
#[inline]
fn cells(label: u32) -> u8 {
    match label {
        ONE => 3,
        POS => 1,
        _ => 2,
    }
}

#[inline]
fn label_of(cells: u8) -> u32 {
    match cells {
        3 => ONE,
        1 => POS,
        _ => NEG,
    }
}

/// `¬f`, for `f` pruned (no false side, no node without pairs) and free of
/// twins; canonical when `f` is.
pub(crate) fn negate(f: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let shape = f.shape.clone();
    let sh = &*shape;
    let n = sh.num_nodes();
    let mut out = SkipForm::zero(shape.clone());
    if f.out == ZREF {
        // `true`, resolved from the root.
        let mut t = sh.root;
        while !sh.is_leaf(t) {
            let (l, r) = sh.kids[t as usize];
            if sh.is_leaf(l) {
                t = r;
            } else if sh.is_leaf(r) {
                t = l;
            } else {
                break;
            }
        }
        if sh.is_leaf(t) {
            out.out = mk(t, ONE);
            return Ok(out);
        }
        return top_at(f, t, lim);
    }
    // Skips that end on a leaf label, by the level they end below.
    let mut leaf_skips: FxHashMap<u32, Vec<Ref>> = FxHashMap::default();
    let mut note = |c: u32, x: Ref| -> Result<(), OperationError> {
        if vt(x) != c && sh.is_leaf(vt(x)) {
            lim.try_push(leaf_skips.entry(sh.parent[vt(x) as usize]).or_default(), x)?;
        }
        Ok(())
    };
    note(sh.root, f.out)?;
    for &t in &f.active {
        let (l, r) = sh.kids[t as usize];
        for &(a, b) in &f.levels[t as usize].pairs {
            note(l, a)?;
            note(r, b)?;
        }
    }
    // The levels with classes to complement, children first: those holding
    // nodes and those skips end on a leaf label below. A level between
    // them that a skip crosses has the complement of the level it lands
    // on, behind `true` on its single variable.
    let mut lv: Vec<u32> = f.active.iter().copied().chain(leaf_skips.keys().copied()).collect();
    lv.sort_unstable_by_key(|&t| std::cmp::Reverse(sh.pre[t as usize]));
    lv.dedup();
    let mut pos = vec![NONE; n];
    for (i, &t) in lv.iter().enumerate() {
        pos[t as usize] = i as u32;
    }
    // The level whose complement child level `c` reads, or `NONE` where
    // nothing is read at `c`.
    let zlev = |c: u32| -> u32 {
        let mut c = c;
        while !sh.is_leaf(c) && pos[c as usize] == NONE {
            let (l, r) = sh.kids[c as usize];
            c = if sh.is_leaf(l) {
                r
            } else if sh.is_leaf(r) {
                l
            } else {
                return NONE;
            };
        }
        if sh.is_leaf(c) { NONE } else { c }
    };
    // Pass 1, children first: each level's Z, as pairs over f's references
    // and the children's Z (a pending reference, resolved in pass 3), and
    // whether it is empty.
    let mut zpairs: Vec<Vec<(Ref, Ref)>> = vec![Vec::new(); lv.len()];
    // A child's Z as a side: `ZTAG | level`, the level `zlev` names.
    const ZTAG: Ref = 1 << 62;
    let zof = |zpairs: &Vec<Vec<(Ref, Ref)>>, c: u32| -> Option<Ref> {
        let w = zlev(c);
        (w != NONE && !zpairs[pos[w as usize] as usize].is_empty()).then_some(ZTAG | w as u64)
    };
    let mut dense: Vec<u8> = Vec::new();
    for (k, &t) in lv.iter().enumerate() {
        let (l, r) = sh.kids[t as usize];
        let lvl = &f.levels[t as usize];
        let mut z = Vec::new();
        match (sh.is_leaf(l), sh.is_leaf(r)) {
            (true, true) => {
                // The 2 x 2 grid.
                let mut owned = [[false; 2]; 2];
                let mut own = |a: u8, b: u8| {
                    for (i, row) in owned.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            if a >> i & 1 != 0 && b >> j & 1 != 0 {
                                *cell = true;
                            }
                        }
                    }
                };
                for &(a, b) in &lvl.pairs {
                    own(cells(ix(a)), cells(ix(b)));
                }
                for &x in leaf_skips.get(&t).map(Vec::as_slice).unwrap_or(&[]) {
                    if vt(x) == l { own(cells(ix(x)), 3) } else { own(3, cells(ix(x))) }
                }
                for i in 0..2u32 {
                    for j in 0..2u32 {
                        if !owned[i as usize][j as usize] {
                            lim.try_push(&mut z, (mk(l, POS + i), mk(r, POS + j)))?;
                        }
                    }
                }
            }
            (true, false) | (false, true) => {
                let left_leaf = sh.is_leaf(l);
                let (leaf, inner) = if left_leaf { (l, r) } else { (r, l) };
                // Per other side, the literal cells its pairs cover: by
                // index for the inner child's nodes (`dense`, zero again
                // once read), in `skipped` for a skip below it.
                lim.try_resize_amortized(&mut dense, f.levels[inner as usize].width(), 0u8)?;
                let mut skipped: FxHashMap<Ref, u8> = FxHashMap::default();
                for &(a, b) in &lvl.pairs {
                    let (lab, e) = if left_leaf { (a, b) } else { (b, a) };
                    if vt(e) == inner {
                        dense[ix(e) as usize] |= cells(ix(lab));
                    } else {
                        *skipped.entry(e).or_insert(0) |= cells(ix(lab));
                    }
                }
                let put = |z: &mut Vec<(Ref, Ref)>, e: Ref, c: u8| -> Result<(), OperationError> {
                    if c != 3 {
                        let lab = mk(leaf, label_of(3 & !c));
                        lim.try_push(z, if left_leaf { (lab, e) } else { (e, lab) })?;
                    }
                    Ok(())
                };
                for &(a, b) in &lvl.pairs {
                    let e = if left_leaf { b } else { a };
                    if vt(e) == inner {
                        let c = std::mem::take(&mut dense[ix(e) as usize]);
                        if c != 0 {
                            put(&mut z, e, c)?;
                        }
                    }
                }
                for (&e, &c) in &skipped {
                    put(&mut z, e, c)?;
                }
                if let Some(zi) = zof(&zpairs, inner) {
                    lim.try_push(&mut z, if left_leaf { (mk(leaf, ONE), zi) } else { (zi, mk(leaf, ONE)) })?;
                }
            }
            (false, false) => {
                let mut owned: FxHashSet<(Ref, Ref)> = FxHashSet::default();
                let (mut aa, mut bb) = (Vec::new(), Vec::new());
                lim.reserve_set(&mut owned, lvl.pairs.len())?;
                lim.reserve_exact(&mut aa, lvl.pairs.len())?;
                lim.reserve_exact(&mut bb, lvl.pairs.len())?;
                for &(a, b) in &lvl.pairs {
                    owned.insert((a, b));
                    aa.push(a);
                    bb.push(b);
                }
                aa.sort_unstable();
                aa.dedup();
                bb.sort_unstable();
                bb.dedup();
                let (zlh, zrh) = (zof(&zpairs, l), zof(&zpairs, r));
                for &a in &aa {
                    for &b in &bb {
                        if !owned.contains(&(a, b)) {
                            lim.try_push(&mut z, (a, b))?;
                        }
                    }
                    if let Some(zr) = zrh {
                        lim.try_push(&mut z, (a, zr))?;
                    }
                }
                if let Some(zl) = zlh {
                    for &b in &bb {
                        lim.try_push(&mut z, (zl, b))?;
                    }
                    if let Some(zr) = zrh {
                        lim.try_push(&mut z, (zl, zr))?;
                    }
                }
                lim.charge_work((aa.len() * bb.len()) as u64);
                lim.discard(owned);
                lim.discard(aa);
                lim.discard(bb);
            }
        }
        zpairs[k] = z;
    }
    lim.discard(dense);
    lim.check_stop()?;
    // Pass 2, parents first: the classes on which f is true whatever lies
    // outside them. At the root, the output.
    let mut top: Vec<Vec<bool>> = Vec::with_capacity(lv.len());
    for &t in &lv {
        top.push(filled(lim, f.levels[t as usize].width(), false)?);
    }
    let mark = |top: &mut Vec<Vec<bool>>, x: Ref| {
        let p = pos[vt(x) as usize];
        if !sh.is_leaf(vt(x)) && p != NONE {
            top[p as usize][ix(x) as usize] = true;
        }
    };
    mark(&mut top, f.out);
    for (k, &t) in lv.iter().enumerate().rev() {
        let (l, r) = sh.kids[t as usize];
        let lvl = &f.levels[t as usize];
        // Only a top class marks classes below it.
        if !top[k].contains(&true) {
            continue;
        }
        for (c, sib, left) in [(l, r, true), (r, l, false)] {
            if sh.is_leaf(c) || zof(&zpairs, sib).is_some() {
                continue;
            }
            // The sibling's classes each entity's true cells cover.
            let mut cov: FxHashMap<Ref, (u8, Vec<Ref>)> = FxHashMap::default();
            let mut all_sib: FxHashSet<Ref> = FxHashSet::default();
            lim.reserve_map(&mut cov, lvl.pairs.len())?;
            if !sh.is_leaf(sib) {
                lim.reserve_set(&mut all_sib, lvl.pairs.len())?;
            }
            for (i, &is_top) in top[k].iter().enumerate() {
                for &(a, b) in lvl.node(i as u32) {
                    let (e, s) = if left { (a, b) } else { (b, a) };
                    if !sh.is_leaf(sib) {
                        all_sib.insert(s);
                    }
                    if is_top {
                        let ent = cov.entry(e).or_default();
                        if sh.is_leaf(sib) {
                            ent.0 |= cells(ix(s));
                        } else {
                            lim.try_push(&mut ent.1, s)?;
                        }
                    }
                }
            }
            lim.release_bytes(cov.charged_bytes());
            for (e, (lits, mut sibs)) in cov {
                let full = if sh.is_leaf(sib) {
                    lits == 3
                } else {
                    sibs.sort_unstable();
                    sibs.dedup();
                    sibs.len() == all_sib.len()
                };
                if full {
                    mark(&mut top, e);
                }
                lim.discard(sibs);
            }
            lim.discard(all_sib);
        }
    }
    // Pass 3, children first: the kept classes and Z, renumbered and
    // resolved. Kept classes keep their pairs: they name kept classes only,
    // renumbered in order, so their stored form stands. A level with no top
    // class at or below it keeps its nodes and their numbers (`same`).
    let mut changed = vec![false; n];
    for (k, &t) in lv.iter().enumerate() {
        if top[k].contains(&true) {
            let mut u = t;
            while u != NONE && !changed[u as usize] {
                changed[u as usize] = true;
                u = sh.parent[u as usize];
            }
        }
    }
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); lv.len()];
    let mut same = vec![false; lv.len()];
    let mut zres: Vec<Ref> = vec![ZREF; lv.len()];
    let fix = |res: &Vec<Vec<Ref>>, same: &[bool], zres: &Vec<Ref>, x: Ref| -> Ref {
        if x & ZTAG != 0 && x != ZREF {
            zres[pos[(x & !ZTAG) as usize] as usize]
        } else if sh.is_leaf(vt(x)) {
            x
        } else {
            let p = pos[vt(x) as usize] as usize;
            if same[p] { x } else { res[p][ix(x) as usize] }
        }
    };
    let mut buf = Vec::new();
    for (k, &t) in lv.iter().enumerate() {
        let lvl = &f.levels[t as usize];
        let z = std::mem::take(&mut zpairs[k]);
        if !changed[t as usize] {
            same[k] = true;
            if lvl.width() > 0 {
                if out.levels[t as usize].width() == 0 {
                    out.active.push(t);
                }
                if z.is_empty() {
                    out.levels.share(t as usize, &f.levels, lim)?;
                } else {
                    out.levels[t as usize].copy_run(lvl, 0, lvl.width(), lim)?;
                }
            }
        } else {
            let mut rt = filled(lim, lvl.width(), ZREF)?;
            for (i, slot) in rt.iter_mut().enumerate() {
                if top[k][i] {
                    continue;
                }
                buf.clear();
                let node = lvl.node(i as u32);
                lim.reserve(&mut buf, node.len())?;
                for &(a, b) in node {
                    buf.push((fix(&res, &same, &zres, a), fix(&res, &same, &zres, b)));
                }
                debug_assert!(buf.iter().all(|&(a, b)| a != ZREF && b != ZREF), "a kept class names no true class");
                debug_assert!(
                    {
                        let mut v = buf.clone();
                        normalize(sh, t, &mut v).is_ok() && v == buf
                    },
                    "a kept class keeps its stored form"
                );
                *slot = mk(t, out.push_node(t, &buf, lim)?);
            }
            res[k] = rt;
        }
        if !z.is_empty() {
            buf.clear();
            lim.reserve(&mut buf, z.len())?;
            for &(a, b) in &z {
                let (a, b) = (fix(&res, &same, &zres, a), fix(&res, &same, &zres, b));
                if a != ZREF && b != ZREF {
                    buf.push((a, b));
                }
            }
            zres[k] = if buf.is_empty() {
                ZREF
            } else {
                match normalize(sh, t, &mut buf) {
                    Err(x) => x,
                    Ok(()) => mk(t, out.push_node(t, &buf, lim)?),
                }
            };
        }
        lim.discard(z);
    }
    // The root's Z, read from the root: where f skips below the root, the
    // Z of the level the skip lands on, behind `true` on the skipped
    // variables.
    out.out = if sh.is_leaf(sh.root) {
        let c = 3 & !cells(ix(f.out));
        if c == 0 { ZREF } else { mk(sh.root, label_of(c)) }
    } else {
        let w = zlev(sh.root);
        if w == NONE { ZREF } else { zres[pos[w as usize] as usize] }
    };
    for v in top {
        lim.discard(v);
    }
    for v in res {
        lim.discard(v);
    }
    lim.discard(buf);
    for (_, v) in leaf_skips {
        lim.discard(v);
    }
    out.collect_pushed(lim)?;
    Ok(out)
}

/// Whether `f` is `true`: its output is `true` over the root as
/// [`build_top`] writes it, the one form `true` has.
pub(crate) fn is_true(f: &SkipForm) -> bool {
    fn top(f: &SkipForm, mut t: u32, x: Ref) -> bool {
        let sh = &*f.shape;
        while !sh.is_leaf(t) {
            let (l, r) = sh.kids[t as usize];
            if sh.is_leaf(l) {
                t = r;
            } else if sh.is_leaf(r) {
                t = l;
            } else {
                return x != ZREF
                    && vt(x) == t
                    && matches!(*f.levels[t as usize].node(ix(x)), [(a, b)] if top(f, l, a) && top(f, r, b));
            }
        }
        x != ZREF && vt(x) == t && ix(x) == ONE
    }
    top(f, f.shape.root, f.out)
}

/// `true` over internal `t`'s variables where both children are inner
/// (a node `(true, true)`): built on `f`'s shape.
fn top_at(f: &SkipForm, t: u32, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*f.shape;
    let mut out = SkipForm::zero(f.shape.clone());
    let top = build_top(sh, t, &mut |u, pair| Ok(mk(u, out.push_node(u, &[pair], lim)?)))?;
    out.out = top;
    out.collect_pushed(lim)?;
    Ok(out)
}
