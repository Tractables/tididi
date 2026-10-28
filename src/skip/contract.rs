//! Twin contraction in skip form.
//!
//! The stored-form contraction merges, level by level from the root, the
//! classes of a level that every node of the parent level pairs with the
//! same partners. In skip form a level's classes are its stored nodes and
//! the references that skip through it. A reference that skips a level
//! from above has a single owner there (the class it skips through one
//! level up), so it is never a twin; only the stored nodes and the skips
//! the parent level's pairs start can be. Merging a skip into a class makes
//! the skipped level's node explicit: `true` on the skipped variable paired
//! with the rest of the skip, whose own continuation below is then a class
//! the next level compares. Skips that merge stay twins down to the first
//! level where one of them is stored, is named by another pair, or where
//! they part, so their merged class is made just above it; the levels
//! between resolve to it. After the merges the levels are rebuilt from
//! the leaves up, so that a class that became `true` on one side resolves
//! to the reference it skips to.

use std::cmp::Reverse;
use std::hash::{Hash, Hasher};
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use super::{ix, mk, normalize, vt, Ref, SkipForm, NEG, ONE, POS, ZREF};

use crate::limits::Limits;
use crate::OperationError;

const NONE: u32 = u32::MAX;

/// One level's classes while the contraction runs.
#[derive(Default)]
struct CLv {
    level: u32,
    /// Class `k`'s pairs are `pairs[start[k]..start[k + 1]]`, sides as the
    /// input's references.
    start: Vec<u32>,
    pairs: Vec<(Ref, Ref)>,
    /// The class of each stored node; and for each skip a merge took in,
    /// the entry (in the visited levels) and class it went to.
    classmap: Vec<u32>,
    chainmap: FxHashMap<Ref, (u32, u32)>,
    res: Vec<Ref>,
}

impl CLv {
    fn push_class(&mut self, pairs: &[(Ref, Ref)]) -> u32 {
        if self.start.is_empty() {
            self.start.push(0);
        }
        self.pairs.extend_from_slice(pairs);
        self.start.push(self.pairs.len() as u32);
        (self.start.len() - 2) as u32
    }
    fn width(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
    fn class(&self, k: usize) -> &[(Ref, Ref)] {
        &self.pairs[self.start[k] as usize..self.start[k + 1] as usize]
    }
}

/// `s` contracted to the canonical skip form of its function. `s` holds no
/// false side and no unreachable node.
pub(crate) fn contract(s: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        let mut z = SkipForm::zero(s.shape.clone());
        z.collect_in(Vec::new());
        return Ok(z);
    }
    // The levels visited, parents first: those holding nodes, and the
    // inner children of a level with classes, where skips it names may merge.
    let mut at = vec![NONE; n];
    let mut lvs: Vec<CLv> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut queued = vec![false; n];
    for &t in &s.active {
        queued[t as usize] = true;
        heap.push(Reverse((sh.pre[t as usize], t)));
    }
    let mut occ: Vec<(Ref, u32, Ref)> = Vec::new();
    let mut keyed: FxHashMap<u64, smallvec::SmallVec<[u32; 1]>> = FxHashMap::default();
    let mut members: Vec<(u32, Ref)> = Vec::new();
    let mut firsts: Vec<(usize, usize)> = Vec::new();
    // Merged skips sent down to the level where they part: per level, the
    // groups (entry they merged at, members).
    let mut pending: FxHashMap<u32, Vec<(u32, Vec<Ref>)>> = FxHashMap::default();
    let mut names: Option<Names> = None;
    while let Some(Reverse((_, c))) = heap.pop() {
        let lvl = &s.levels[c as usize];
        let mut cl = CLv { level: c, classmap: vec![NONE; lvl.width()], ..CLv::default() };
        let p = sh.parent[c as usize];
        let pe = if p == NONE { NONE } else { at[p as usize] };
        if pe != NONE {
            let pcl = &lvs[pe as usize];
            let (pl, pr) = sh.kids[p as usize];
            let left = pl == c;
            let sib = if left { pr } else { pl };
            let sib_leaf = sh.is_leaf(sib);
            // Each class of the parent that names an entity of this level,
            // with the partner it names it with (One as both literals).
            occ.clear();
            for k in 0..pcl.width() {
                for &(a, b) in pcl.class(k) {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    if sib_leaf && ix(partner) == ONE {
                        occ.push((e, k as u32, mk(sib, POS)));
                        occ.push((e, k as u32, mk(sib, NEG)));
                    } else {
                        occ.push((e, k as u32, partner));
                    }
                }
            }
            occ.sort_unstable();
            // Group entities by their sorted context list: a group is keyed
            // by the hash of its first member's list, compared in full.
            keyed.clear();
            members.clear();
            firsts.clear();
            let ctx = |a: (usize, usize), b: (usize, usize)| -> bool {
                a.1 - a.0 == b.1 - b.0 && occ[a.0..a.1].iter().zip(&occ[b.0..b.1]).all(|(x, y)| (x.1, x.2) == (y.1, y.2))
            };
            let mut i = 0;
            while i < occ.len() {
                let mut j = i;
                let mut h = rustc_hash::FxHasher::default();
                while j < occ.len() && occ[j].0 == occ[i].0 {
                    (occ[j].1, occ[j].2).hash(&mut h);
                    j += 1;
                }
                let bucket = keyed.entry(h.finish()).or_default();
                let g = match bucket.iter().find(|&&g| ctx(firsts[g as usize], (i, j))) {
                    Some(&g) => g,
                    None => {
                        firsts.push((i, j));
                        let g = (firsts.len() - 1) as u32;
                        bucket.push(g);
                        g
                    }
                };
                members.push((g, occ[i].0));
                i = j;
            }
            members.sort_unstable();
            let (l, r) = sh.kids[c as usize];
            let mut buf = Vec::new();
            for m in members.chunk_by(|x, y| x.0 == y.0) {
                if m.len() == 1 && vt(m[0].1) != c {
                    // A lone skip stays a skip.
                    continue;
                }
                if m.iter().all(|&(_, e)| vt(e) != c) {
                    let nm = names.get_or_insert_with(|| Names::of(s));
                    let v = parts(sh, nm, c, m.iter().map(|&(_, e)| e));
                    if v != c {
                        pending.entry(v).or_default().push((lvs.len() as u32, m.iter().map(|&(_, e)| e).collect()));
                        if !queued[v as usize] {
                            queued[v as usize] = true;
                            heap.push(Reverse((sh.pre[v as usize], v)));
                        }
                        continue;
                    }
                }
                buf.clear();
                let mut skips = Vec::new();
                for &(_, e) in m {
                    if vt(e) == c {
                        buf.extend_from_slice(lvl.node(ix(e)));
                    } else {
                        skips.push(e);
                        buf.push(if sh.within(vt(e), l) { (e, mk(r, ONE)) } else { (mk(l, ONE), e) });
                    }
                }
                let k = cl.push_class(&buf);
                for &(_, e) in m {
                    if vt(e) == c {
                        cl.classmap[ix(e) as usize] = k;
                    }
                }
                for e in skips {
                    cl.chainmap.insert(e, (lvs.len() as u32, k));
                }
            }
        }
        // Merged skips sent down here: each group is a class of its own,
        // since only the levels above name its members.
        if let Some(groups) = pending.remove(&c) {
            let (l, r) = sh.kids[c as usize];
            for (from, group) in groups {
                let pairs: Vec<(Ref, Ref)> =
                    group.iter().map(|&e| if sh.within(vt(e), l) { (e, mk(r, ONE)) } else { (mk(l, ONE), e) }).collect();
                let k = cl.push_class(&pairs);
                let here = lvs.len() as u32;
                let origin = &mut lvs[from as usize].chainmap;
                for e in group {
                    origin.insert(e, (here, k));
                }
            }
        }
        // Stored nodes no class of the parent names directly: the output,
        // and nodes skips reach. Each is its own class.
        for i in 0..lvl.width() {
            if cl.classmap[i] == NONE {
                cl.classmap[i] = cl.push_class(lvl.node(i as u32));
            }
        }
        lim.charge_work(lvl.pairs.len() as u64);
        if cl.width() > 0 {
            for ch in [sh.kids[c as usize].0, sh.kids[c as usize].1] {
                if !sh.is_leaf(ch) && !queued[ch as usize] {
                    queued[ch as usize] = true;
                    heap.push(Reverse((sh.pre[ch as usize], ch)));
                }
            }
        }
        at[c as usize] = lvs.len() as u32;
        lvs.push(cl);
    }
    lim.check_stop()?;
    // Rebuild from the leaves up.
    let mut out = SkipForm::zero(s.shape.clone());
    let target = |lvs: &Vec<CLv>, h: Ref| -> Ref {
        if sh.is_leaf(vt(h)) {
            h
        } else {
            let cl = &lvs[at[vt(h) as usize] as usize];
            cl.res[cl.classmap[ix(h) as usize] as usize]
        }
    };
    let side = |lvs: &Vec<CLv>, d: u32, h: Ref| -> Ref {
        if sh.is_leaf(d) || vt(h) == d {
            return target(lvs, h);
        }
        let e = at[d as usize];
        if e != NONE && let Some(&(to, k)) = lvs[e as usize].chainmap.get(&h) {
            return lvs[to as usize].res[k as usize];
        }
        target(lvs, h)
    };
    let mut buf = Vec::new();
    for e in (0..lvs.len()).rev() {
        let c = lvs[e].level;
        let (l, r) = sh.kids[c as usize];
        let mut rc = Vec::with_capacity(lvs[e].width());
        for k in 0..lvs[e].width() {
            buf.clear();
            for &(a, b) in lvs[e].class(k) {
                buf.push((side(&lvs, l, a), side(&lvs, r, b)));
            }
            // Merged children can repeat a pair.
            buf.sort_unstable();
            buf.dedup();
            rc.push(match normalize(sh, c, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(c, out.levels[c as usize].push(&buf)),
            });
        }
        lvs[e].res = rc;
    }
    out.out = side(&lvs, sh.root, s.out);
    let cand = lvs.iter().map(|l| l.level).collect();
    out.collect_in(cand);
    Ok(out)
}

/// Per reference of a diagram, the levels whose pairs name it, parents
/// first.
struct Names {
    /// Per vtree node, the number of its first stored node or leaf label.
    base: Vec<u32>,
    start: Vec<u32>,
    levels: Vec<u32>,
}

impl Names {
    fn of(s: &SkipForm) -> Names {
        let sh = &*s.shape;
        let mut base = vec![NONE; sh.num_nodes()];
        let mut n = 0u32;
        for &t in &s.active {
            base[t as usize] = n;
            n += s.levels[t as usize].width() as u32;
        }
        for (c, b) in base.iter_mut().enumerate() {
            if sh.is_leaf(c as u32) {
                *b = n;
                n += 3;
            }
        }
        let id = |x: Ref| (base[vt(x) as usize] + ix(x)) as usize;
        let mut start = vec![0u32; n as usize + 1];
        for &u in &s.active {
            for &(a, b) in &s.levels[u as usize].pairs {
                start[id(a) + 1] += 1;
                start[id(b) + 1] += 1;
            }
        }
        for i in 0..n as usize {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut levels = vec![0u32; start[n as usize] as usize];
        // `active` is in preorder, so each list is too.
        for &u in &s.active {
            for &(a, b) in &s.levels[u as usize].pairs {
                for x in [a, b] {
                    let i = id(x);
                    levels[fill[i] as usize] = u;
                    fill[i] += 1;
                }
            }
        }
        Names { base, start, levels }
    }

    fn of_ref(&self, x: Ref) -> &[u32] {
        let i = (self.base[vt(x) as usize] + ix(x)) as usize;
        &self.levels[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// The level whose class a group of skips that merge at `c` is: the parent
/// of the highest level below `c` where the members stop being twins (one
/// is stored there, a pair of the level above names one, or they part).
/// `c` itself when that level is `c`'s child.
fn parts(sh: &super::Shape, nm: &Names, c: u32, members: impl Iterator<Item = Ref> + Clone) -> u32 {
    // The highest member, and where the members part.
    let mut h = NONE;
    for e in members.clone() {
        if h == NONE || sh.pre[vt(e) as usize] < sh.pre[h as usize] {
            h = vt(e);
        }
    }
    let mut w = if members.clone().all(|e| sh.within(vt(e), h)) { h } else { sh.parent[h as usize] };
    for e in members {
        // The highest level at or below `c` naming `e`.
        let ls = nm.of_ref(e);
        let i = ls.partition_point(|&u| sh.pre[u as usize] < sh.pre[c as usize]);
        if let Some(&u) = ls.get(i) {
            let x = sh.toward(u, vt(e));
            if sh.pre[x as usize] < sh.pre[w as usize] {
                w = x;
            }
        }
    }
    // Members on both leaves under `c` part at `c` itself.
    if w == c {
        return c;
    }
    debug_assert!(sh.within(w, c));
    sh.parent[w as usize]
}
