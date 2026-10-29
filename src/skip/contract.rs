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
//! between resolve to it.
//!
//! Twins are found by a hash of each entity's partners, summed so that
//! order does not matter, and confirmed by comparing the partner lists of
//! entities whose hashes agree. The diagram is then patched from the
//! leaves up: a stored node keeps its number, a merged class is appended,
//! and only nodes whose sides changed are rewritten.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use super::{ix, mk, normalize, vt, Ref, SLevel, SkipForm, NEG, ONE, POS, ZREF};

use crate::limits::Limits;
use crate::OperationError;

const NONE: u32 = u32::MAX;
/// A node's result where it is unchanged.
const KEEP: Ref = ZREF;

/// A visited level while the contraction runs.
struct CLv {
    level: u32,
    /// The input level's width: class `k < width` is stored node `k`, class
    /// `width + g` is merged group `g`.
    width: u32,
    /// Per stored node, the group it merged into (empty where none did).
    group_of: Vec<u32>,
    /// Group `g`'s pairs, as the input's references: its stored members'
    /// pairs, and `true` beside each skip it took in.
    gstart: Vec<u32>,
    gpairs: Vec<(Ref, Ref)>,
    /// Per skip a group took in (here, or sent down from here), the entry
    /// and group it went to.
    chainmap: FxHashMap<Ref, (u32, u32)>,
    /// Per group, its result.
    gres: Vec<Ref>,
    /// Per stored node, its result where it no longer names itself (empty
    /// where none changed).
    rm: Vec<Ref>,
}

impl CLv {
    fn new(level: u32, width: u32) -> CLv {
        CLv {
            level,
            width,
            group_of: Vec::new(),
            gstart: vec![0],
            gpairs: Vec::new(),
            chainmap: FxHashMap::default(),
            gres: Vec::new(),
            rm: Vec::new(),
        }
    }
    fn groups(&self) -> usize {
        self.gstart.len() - 1
    }
    fn group(&self, g: usize) -> &[(Ref, Ref)] {
        &self.gpairs[self.gstart[g] as usize..self.gstart[g + 1] as usize]
    }
    /// Add a group of the stored nodes and skips `members` (`sh`'s level
    /// `c`, whose children are `l` and `r`), returning its number.
    fn push_group(&mut self, s: &SkipForm, members: &[Ref], l: u32, r: u32) -> u32 {
        let g = self.groups() as u32;
        for &e in members {
            if vt(e) == self.level {
                if self.group_of.is_empty() {
                    self.group_of = vec![NONE; self.width as usize];
                }
                self.group_of[ix(e) as usize] = g;
                self.gpairs.extend_from_slice(s.levels[self.level as usize].node(ix(e)));
            } else {
                self.gpairs.push(if s.shape.within(vt(e), l) { (e, mk(r, ONE)) } else { (mk(l, ONE), e) });
            }
        }
        self.gstart.push(self.gpairs.len() as u32);
        g
    }
}

/// An order-free hash of one (class, partner) occurrence.
#[inline]
fn mix(k: u32, p: Ref) -> u64 {
    let mut x = p.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 32)
}

/// `s` contracted to the canonical skip form of its function. `s` holds no
/// false side and no unreachable node.
pub(crate) fn contract(mut s: SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let shape = s.shape.clone();
    let sh = &*shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        let mut z = SkipForm::zero(s.shape.clone());
        z.collect_in(Vec::new());
        return Ok(z);
    }
    let ids = Ids::of(&s);
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
    // Per entity of the current level (a stored node or a skip the parent
    // names): its reference, partner hash and partner count.
    let (mut sstamp, mut sslot) = (vec![0u32; ids.len()], vec![0u32; ids.len()]);
    let mut epoch = 0u32;
    let mut ent: Vec<(Ref, u64, u32)> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    let mut seen: FxHashMap<(u64, u32), u32> = FxHashMap::default();
    let mut cand: Vec<bool> = Vec::new();
    let mut occ: Vec<(u32, u32, Ref)> = Vec::new();
    let mut ctx: Vec<(u32, u32)> = Vec::new();
    let mut firsts: Vec<u32> = Vec::new();
    let mut members: Vec<Ref> = Vec::new();
    // Merged skips sent down to the level where they part: per level, the
    // groups (entry they merged at, members).
    let mut pending: FxHashMap<u32, Vec<(u32, Vec<Ref>)>> = FxHashMap::default();
    let mut names: Option<Names> = None;
    let mut merged = false;
    while let Some(Reverse((_, c))) = heap.pop() {
        let width = s.levels[c as usize].width() as u32;
        let mut cl = CLv::new(c, width);
        let here = lvs.len() as u32;
        let (l, r) = sh.kids[c as usize];
        let p = sh.parent[c as usize];
        let pe = if p == NONE { NONE } else { at[p as usize] };
        if pe != NONE {
            let pcl = &lvs[pe as usize];
            let plvl = &s.levels[p as usize];
            let (pl, pr) = sh.kids[p as usize];
            let left = pl == c;
            let sib = if left { pr } else { pl };
            let sib_leaf = sh.is_leaf(sib);
            // Every class of the parent with its pairs.
            let classes = || {
                (0..pcl.width)
                    .filter(|&i| pcl.group_of.is_empty() || pcl.group_of[i as usize] == NONE)
                    .map(|i| (i, plvl.node(i)))
                    .chain((0..pcl.groups()).map(|g| (pcl.width + g as u32, pcl.group(g))))
            };
            // Partners (One as both literals against a leaf sibling).
            let partners = |b: Ref| -> [Ref; 2] {
                if sib_leaf && ix(b) == ONE { [mk(sib, POS), mk(sib, NEG)] } else { [b, ZREF] }
            };
            epoch += 1;
            ent.clear();
            for (k, pairs) in classes() {
                for &(a, b) in pairs {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    let id = ids.id(e);
                    if sstamp[id] != epoch {
                        sstamp[id] = epoch;
                        sslot[id] = ent.len() as u32;
                        ent.push((e, 0, 0));
                    }
                    let x = &mut ent[sslot[id] as usize];
                    for q in partners(partner) {
                        if q != ZREF {
                            x.1 = x.1.wrapping_add(mix(k, q));
                            x.2 += 1;
                        }
                    }
                }
            }
            // Entities whose hashes agree, confirmed by their partner lists.
            seen.clear();
            cand.clear();
            cand.resize(ent.len(), false);
            let mut any = false;
            for (i, x) in ent.iter().enumerate() {
                match seen.entry((x.1, x.2)) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        cand[*o.get() as usize] = true;
                        cand[i] = true;
                        any = true;
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(i as u32);
                    }
                }
            }
            if any {
                order.clear();
                order.extend((0..ent.len() as u32).filter(|&i| cand[i as usize]));
                order.sort_unstable_by_key(|&i| (ent[i as usize].1, ent[i as usize].2));
                occ.clear();
                for (k, pairs) in classes() {
                    for &(a, b) in pairs {
                        let (e, partner) = if left { (a, b) } else { (b, a) };
                        let i = sslot[ids.id(e)];
                        if cand[i as usize] {
                            for q in partners(partner) {
                                if q != ZREF {
                                    occ.push((i, k, q));
                                }
                            }
                        }
                    }
                }
                occ.sort_unstable();
                // Each candidate's partner list.
                ctx.clear();
                ctx.resize(ent.len(), (0, 0));
                for (j, w) in occ.chunk_by(|x, y| x.0 == y.0).scan(0u32, |o, w| {
                    let start = *o;
                    *o += w.len() as u32;
                    Some((start, w))
                }) {
                    ctx[w[0].0 as usize] = (j, j + w.len() as u32);
                }
                let list = |i: u32| {
                    let (a, b) = ctx[i as usize];
                    occ[a as usize..b as usize].iter().map(|x| (x.1, x.2))
                };
                for run in order.chunk_by(|&x, &y| (ent[x as usize].1, ent[x as usize].2) == (ent[y as usize].1, ent[y as usize].2)) {
                    if run.len() < 2 {
                        continue;
                    }
                    // Split the run into groups of equal lists.
                    firsts.clear();
                    let mut of: Vec<u32> = Vec::with_capacity(run.len());
                    for &i in run {
                        let g = firsts.iter().position(|&f| list(f).eq(list(i)));
                        of.push(match g {
                            Some(g) => g as u32,
                            None => {
                                firsts.push(i);
                                (firsts.len() - 1) as u32
                            }
                        });
                    }
                    for g in 0..firsts.len() as u32 {
                        members.clear();
                        members.extend(run.iter().zip(&of).filter(|x| *x.1 == g).map(|x| ent[*x.0 as usize].0));
                        if members.len() < 2 {
                            continue;
                        }
                        merged = true;
                        if members.iter().all(|&e| vt(e) != c) {
                            let nm = names.get_or_insert_with(|| Names::of(&s, &ids));
                            let v = parts(sh, nm, &ids, c, members.iter().copied());
                            if v != c {
                                pending.entry(v).or_default().push((here, members.clone()));
                                if !queued[v as usize] {
                                    queued[v as usize] = true;
                                    heap.push(Reverse((sh.pre[v as usize], v)));
                                }
                                continue;
                            }
                        }
                        let g = cl.push_group(&s, &members, l, r);
                        for &e in &members {
                            if vt(e) != c {
                                cl.chainmap.insert(e, (here, g));
                            }
                        }
                    }
                }
            }
            lim.charge_work(ent.len() as u64 + occ.len() as u64);
        }
        // Merged skips sent down here: each group is a class of its own,
        // since only the levels above name its members.
        if let Some(groups) = pending.remove(&c) {
            for (from, group) in groups {
                let g = cl.push_group(&s, &group, l, r);
                let origin = &mut lvs[from as usize].chainmap;
                for e in group {
                    origin.insert(e, (here, g));
                }
            }
        }
        lim.charge_work(s.levels[c as usize].pairs.len() as u64);
        if width > 0 || cl.groups() > 0 {
            for ch in [l, r] {
                if !sh.is_leaf(ch) && !queued[ch as usize] {
                    queued[ch as usize] = true;
                    heap.push(Reverse((sh.pre[ch as usize], ch)));
                }
            }
        }
        at[c as usize] = here;
        lvs.push(cl);
    }
    lim.check_stop()?;
    // Patch from the leaves up. A side read at `d` resolves through a
    // group a skip joined there, else through its node's result.
    let resolve = |lvs: &Vec<CLv>, d: u32, x: Ref| -> Ref {
        if !sh.is_leaf(d) && vt(x) != d {
            let e = at[d as usize];
            if e != NONE && !lvs[e as usize].chainmap.is_empty() && let Some(&(to, g)) = lvs[e as usize].chainmap.get(&x) {
                return lvs[to as usize].gres[g as usize];
            }
        }
        let v = vt(x);
        if !sh.is_leaf(v) {
            let e = at[v as usize];
            if e != NONE {
                let rm = &lvs[e as usize].rm;
                if !rm.is_empty() && rm[ix(x) as usize] != KEEP {
                    return rm[ix(x) as usize];
                }
            }
        }
        x
    };
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    for e in (0..lvs.len()).rev() {
        let c = lvs[e].level;
        let (l, r) = sh.kids[c as usize];
        let width = lvs[e].width as usize;
        let changes = |lvs: &Vec<CLv>, lvl: &SLevel, i: usize| {
            lvl.node(i as u32).iter().any(|&(a, b)| resolve(lvs, l, a) != a || resolve(lvs, r, b) != b)
        };
        let lvl = &s.levels[c as usize];
        let grouped = |i: usize| !lvs[e].group_of.is_empty() && lvs[e].group_of[i] != NONE;
        if lvs[e].groups() == 0 && !(0..width).any(|i| changes(&lvs, lvl, i)) {
            continue;
        }
        // Stored nodes keep their numbers; those merged into a group, or
        // that now resolve to another reference, stay behind unreachable.
        let mut out = SLevel::default();
        let mut rm = vec![KEEP; width];
        for (i, x) in rm.iter_mut().enumerate() {
            let node = lvl.node(i as u32);
            if !grouped(i) && changes(&lvs, lvl, i) {
                buf.clear();
                buf.extend(node.iter().map(|&(a, b)| (resolve(&lvs, l, a), resolve(&lvs, r, b))));
                match normalize(sh, c, &mut buf) {
                    Err(t) => {
                        *x = t;
                        merged = true;
                    }
                    Ok(()) => {
                        out.push(&buf);
                        continue;
                    }
                }
            }
            out.push(node);
        }
        let mut gres = Vec::with_capacity(lvs[e].groups());
        for g in 0..lvs[e].groups() {
            buf.clear();
            buf.extend(lvs[e].group(g).iter().map(|&(a, b)| (resolve(&lvs, l, a), resolve(&lvs, r, b))));
            // Merged members can repeat a pair.
            buf.sort_unstable();
            buf.dedup();
            gres.push(match normalize(sh, c, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(c, out.push(&buf)),
            });
        }
        for (x, &g) in rm.iter_mut().zip(&lvs[e].group_of) {
            if g != NONE {
                *x = gres[g as usize];
            }
        }
        lvs[e].rm = rm;
        lvs[e].gres = gres;
        s.levels[c as usize] = out;
    }
    s.out = resolve(&lvs, sh.root, s.out);
    if merged {
        let cand = s.active.iter().copied().chain(lvs.iter().map(|l| l.level)).collect();
        s.collect_in(cand);
    } else {
        s.active.retain(|&t| s.levels[t as usize].width() > 0);
    }
    Ok(s)
}

/// Dense numbers for a diagram's references: stored node `i` of level `t`
/// is `base[t] + i`, leaf label `k` of leaf `c` is `base[c] + k`.
struct Ids {
    base: Vec<u32>,
    n: u32,
}

impl Ids {
    fn of(s: &SkipForm) -> Ids {
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
        Ids { base, n }
    }
    fn len(&self) -> usize {
        self.n as usize
    }
    #[inline]
    fn id(&self, x: Ref) -> usize {
        (self.base[vt(x) as usize] + ix(x)) as usize
    }
}

/// Per reference of a diagram, the levels whose pairs name it, parents
/// first.
struct Names {
    start: Vec<u32>,
    levels: Vec<u32>,
}

impl Names {
    fn of(s: &SkipForm, ids: &Ids) -> Names {
        let n = ids.n;
        let id = |x: Ref| ids.id(x);
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
        Names { start, levels }
    }

    fn of_ref(&self, ids: &Ids, x: Ref) -> &[u32] {
        let i = ids.id(x);
        &self.levels[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// The level whose class a group of skips that merge at `c` is: the parent
/// of the highest level below `c` where the members stop being twins (one
/// is stored there, a pair of the level above names one, or they part).
/// `c` itself when that level is `c`'s child.
fn parts(sh: &super::Shape, nm: &Names, ids: &Ids, c: u32, members: impl Iterator<Item = Ref> + Clone) -> u32 {
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
        let ls = nm.of_ref(ids, e);
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
