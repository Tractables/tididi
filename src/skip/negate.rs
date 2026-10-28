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

use super::{ix, mk, normalize, vt, Ref, SkipForm, NEG, ONE, POS, ZREF};
use crate::limits::Limits;
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
        return Ok(top_at(f, t));
    }
    // Skips that end on a leaf label, by the level they end below.
    let mut leaf_skips: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut note = |c: u32, x: Ref| {
        if vt(x) != c && sh.is_leaf(vt(x)) {
            leaf_skips[sh.parent[vt(x) as usize] as usize].push(x);
        }
    };
    note(sh.root, f.out);
    for &t in &sh.topdown {
        let (l, r) = sh.kids[t as usize];
        for &(a, b) in &f.levels[t as usize].pairs {
            note(l, a);
            note(r, b);
        }
    }
    // Pass 1, bottom-up: each level's Z, as pairs over f's references and
    // the children's Z (a pending reference, resolved in pass 3), and
    // whether it is empty.
    let mut zpairs: Vec<Vec<(Ref, Ref)>> = vec![Vec::new(); n];
    let zl = |zpairs: &Vec<Vec<(Ref, Ref)>>, c: u32| !sh.is_leaf(c) && !zpairs[c as usize].is_empty();
    // A child's Z as a side: `ZTAG | level`.
    const ZTAG: Ref = 1 << 62;
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let lvl = &f.levels[t as usize];
        let mut z = Vec::new();
        match (sh.is_leaf(l), sh.is_leaf(r)) {
            (true, true) => {
                // The 2 x 2 grid.
                let mut owned = [[false; 2]; 2];
                let mut own = |a: u8, b: u8| {
                    for i in 0..2 {
                        for j in 0..2 {
                            if a >> i & 1 != 0 && b >> j & 1 != 0 {
                                owned[i][j] = true;
                            }
                        }
                    }
                };
                for &(a, b) in &lvl.pairs {
                    own(cells(ix(a)), cells(ix(b)));
                }
                for &x in &leaf_skips[t as usize] {
                    if vt(x) == l { own(cells(ix(x)), 3) } else { own(3, cells(ix(x))) }
                }
                for i in 0..2u32 {
                    for j in 0..2u32 {
                        if !owned[i as usize][j as usize] {
                            z.push((mk(l, POS + i), mk(r, POS + j)));
                        }
                    }
                }
            }
            (true, false) | (false, true) => {
                let left_leaf = sh.is_leaf(l);
                let (leaf, inner) = if left_leaf { (l, r) } else { (r, l) };
                let mut cov: FxHashMap<Ref, u8> = FxHashMap::default();
                for &(a, b) in &lvl.pairs {
                    let (lab, e) = if left_leaf { (a, b) } else { (b, a) };
                    *cov.entry(e).or_insert(0) |= cells(ix(lab));
                }
                let mut v: Vec<(Ref, u8)> = cov.into_iter().filter(|&(_, c)| c != 3).collect();
                v.sort_unstable();
                for (e, c) in v {
                    let lab = mk(leaf, label_of(3 & !c));
                    z.push(if left_leaf { (lab, e) } else { (e, lab) });
                }
                if zl(&zpairs, inner) {
                    let zi = ZTAG | inner as u64;
                    z.push(if left_leaf { (mk(leaf, ONE), zi) } else { (zi, mk(leaf, ONE)) });
                }
            }
            (false, false) => {
                let mut owned: FxHashSet<(Ref, Ref)> = FxHashSet::default();
                let (mut aa, mut bb) = (Vec::new(), Vec::new());
                for &(a, b) in &lvl.pairs {
                    owned.insert((a, b));
                    aa.push(a);
                    bb.push(b);
                }
                aa.sort_unstable();
                aa.dedup();
                bb.sort_unstable();
                bb.dedup();
                let (zlh, zrh) = (zl(&zpairs, l), zl(&zpairs, r));
                for &a in &aa {
                    for &b in &bb {
                        if !owned.contains(&(a, b)) {
                            z.push((a, b));
                        }
                    }
                    if zrh {
                        z.push((a, ZTAG | r as u64));
                    }
                }
                if zlh {
                    for &b in &bb {
                        z.push((ZTAG | l as u64, b));
                    }
                    if zrh {
                        z.push((ZTAG | l as u64, ZTAG | r as u64));
                    }
                }
                lim.charge_work((aa.len() * bb.len()) as u64);
            }
        }
        zpairs[t as usize] = z;
    }
    lim.check_stop()?;
    // Pass 2, top-down: the classes on which f is true whatever lies
    // outside them. At the root, the output.
    let mut top: Vec<Vec<bool>> = f.levels.iter().map(|l| vec![false; l.width()]).collect();
    let mark = |top: &mut Vec<Vec<bool>>, x: Ref| {
        if !sh.is_leaf(vt(x)) {
            top[vt(x) as usize][ix(x) as usize] = true;
        }
    };
    mark(&mut top, f.out);
    for &t in &sh.topdown {
        let (l, r) = sh.kids[t as usize];
        let lvl = &f.levels[t as usize];
        for (c, sib, left) in [(l, r, true), (r, l, false)] {
            if sh.is_leaf(c) || zl(&zpairs, sib) {
                continue;
            }
            // The sibling's classes each entity's true cells cover.
            let mut cov: FxHashMap<Ref, (u8, Vec<Ref>)> = FxHashMap::default();
            let mut all_sib: FxHashSet<Ref> = FxHashSet::default();
            for i in 0..lvl.width() {
                let is_top = top[t as usize][i];
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
                            ent.1.push(s);
                        }
                    }
                }
            }
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
            }
        }
    }
    // Pass 3, bottom-up: the kept classes and Z, renumbered and resolved.
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let mut zres: Vec<Ref> = vec![ZREF; n];
    let fix = |res: &Vec<Vec<Ref>>, zres: &Vec<Ref>, x: Ref| -> Ref {
        if x & ZTAG != 0 && x != ZREF {
            zres[(x & !ZTAG) as usize]
        } else if sh.is_leaf(vt(x)) {
            x
        } else {
            res[vt(x) as usize][ix(x) as usize]
        }
    };
    let mut buf = Vec::new();
    for &t in &sh.bottomup {
        let lvl = &f.levels[t as usize];
        let mut rt = vec![ZREF; lvl.width()];
        for i in 0..lvl.width() {
            if top[t as usize][i] {
                continue;
            }
            buf.clear();
            for &(a, b) in lvl.node(i as u32) {
                buf.push((fix(&res, &zres, a), fix(&res, &zres, b)));
            }
            debug_assert!(buf.iter().all(|&(a, b)| a != ZREF && b != ZREF), "a kept class names no true class");
            rt[i] = match normalize(sh, t, &mut buf) {
                Err(x) => x,
                Ok(()) => mk(t, out.levels[t as usize].push(&buf)),
            };
        }
        res[t as usize] = rt;
        let z = std::mem::take(&mut zpairs[t as usize]);
        if !z.is_empty() {
            buf.clear();
            for &(a, b) in &z {
                let (a, b) = (fix(&res, &zres, a), fix(&res, &zres, b));
                if a != ZREF && b != ZREF {
                    buf.push((a, b));
                }
            }
            zres[t as usize] = if buf.is_empty() {
                ZREF
            } else {
                match normalize(sh, t, &mut buf) {
                    Err(x) => x,
                    Ok(()) => mk(t, out.levels[t as usize].push(&buf)),
                }
            };
        }
    }
    // The root's Z, read from the root; a root below which f skips has no
    // Z of its own: its Z is the skip's level's, behind `true` on the
    // skipped variables.
    out.out = if sh.is_leaf(sh.root) {
        let c = 3 & !cells(ix(f.out));
        if c == 0 { ZREF } else { mk(sh.root, label_of(c)) }
    } else {
        zres[sh.root as usize]
    };
    out.collect();
    Ok(out)
}

/// `true` over internal `t`'s variables where both children are inner
/// (a node `(true, true)`): built on `f`'s shape.
fn top_at(f: &SkipForm, t: u32) -> SkipForm {
    let sh = &*f.shape;
    let mut out = SkipForm::zero(f.shape.clone());
    fn build(sh: &super::Shape, out: &mut SkipForm, t: u32) -> Ref {
        let mut t = t;
        while !sh.is_leaf(t) {
            let (l, r) = sh.kids[t as usize];
            if sh.is_leaf(l) {
                t = r;
            } else if sh.is_leaf(r) {
                t = l;
            } else {
                let (a, b) = (build(sh, out, l), build(sh, out, r));
                return mk(t, out.levels[t as usize].push(&[(a, b)]));
            }
        }
        mk(t, ONE)
    }
    out.out = build(sh, &mut out, t);
    out.collect();
    out
}
