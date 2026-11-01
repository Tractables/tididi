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

use super::{NEG, ONE, POS, Ref, SLevel, Shape, SkipForm, ZREF, filled, ix, mk, normalize, vt};

use crate::OperationError;
use crate::limits::Limits;

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
            gstart: Vec::new(),
            gpairs: Vec::new(),
            chainmap: FxHashMap::default(),
            gres: Vec::new(),
            rm: Vec::new(),
        }
    }
    fn groups(&self) -> usize {
        self.gstart.len().saturating_sub(1)
    }
    fn group(&self, g: usize) -> &[(Ref, Ref)] {
        &self.gpairs[self.gstart[g] as usize..self.gstart[g + 1] as usize]
    }
    /// Add a group of the stored nodes and skips `members` (`sh`'s level
    /// `c`, whose children are `l` and `r`), returning its number.
    fn push_group(
        &mut self,
        s: &SkipForm,
        members: &[Ref],
        l: u32,
        r: u32,
        lim: &Limits,
    ) -> Result<u32, OperationError> {
        if self.gstart.is_empty() {
            lim.try_push(&mut self.gstart, 0)?;
        }
        let g = self.groups() as u32;
        for &e in members {
            if vt(e) == self.level {
                if self.group_of.is_empty() {
                    self.group_of = filled(lim, self.width as usize, NONE)?;
                }
                self.group_of[ix(e) as usize] = g;
                let node = s.levels[self.level as usize].node(ix(e));
                lim.reserve(&mut self.gpairs, node.len())?;
                self.gpairs.extend_from_slice(node);
            } else {
                lim.try_push(
                    &mut self.gpairs,
                    if s.shape.within(vt(e), l) {
                        (e, mk(r, ONE))
                    } else {
                        (mk(l, ONE), e)
                    },
                )?;
            }
        }
        lim.try_push(&mut self.gstart, self.gpairs.len() as u32)?;
        Ok(g)
    }

    /// Hand the charge for the level's buffers back to `lim`.
    fn discard(self, lim: &Limits) {
        lim.discard(self.group_of);
        lim.discard(self.gstart);
        lim.discard(self.gpairs);
        lim.discard(self.chainmap);
        lim.discard(self.gres);
        lim.discard(self.rm);
    }
}

/// An order-free hash of one (class, partner) occurrence.
#[inline]
fn mix(k: u32, p: Ref) -> u64 {
    let mut x =
        p.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 32)
}

/// `s` contracted to the canonical skip form of its function. `s` holds no
/// false side and no unreachable node.
pub(crate) fn contract(s: SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    contract_in(s, None, lim)
}

/// Per-run scratch buffers for [`contract_in`].
struct ContractWork {
    ids: Ids,
    sstamp: Vec<u32>,
    sslot: Vec<u32>,
    epoch: u32,
    ent: Vec<(Ref, u64, u32)>,
    order: Vec<u32>,
    seen: FxHashMap<(u64, u32), u32>,
    cand: Vec<bool>,
    occ: Vec<(u32, u32, Ref)>,
    ctx: Vec<(u32, u32)>,
    firsts: Vec<u32>,
    members: Vec<Ref>,
    pending: FxHashMap<u32, Vec<(u32, Vec<Ref>)>>,
    names: Option<Names>,
    merged: bool,
}

impl ContractWork {
    fn new(n: usize) -> Self {
        Self {
            ids: Ids::new(n),
            sstamp: Vec::new(),
            sslot: Vec::new(),
            epoch: 0,
            ent: Vec::new(),
            order: Vec::new(),
            seen: FxHashMap::default(),
            cand: Vec::new(),
            occ: Vec::new(),
            ctx: Vec::new(),
            firsts: Vec::new(),
            members: Vec::new(),
            pending: FxHashMap::default(),
            names: None,
            merged: false,
        }
    }

    fn discard(self, lim: &Limits) {
        for v in [self.sstamp, self.sslot, self.order, self.firsts] {
            lim.discard(v);
        }
        lim.discard(self.cand);
        lim.discard(self.ent);
        lim.discard(self.seen);
        lim.discard(self.occ);
        lim.discard(self.ctx);
        lim.discard(self.members);
        if let Some(nm) = self.names {
            lim.discard(nm.start);
            lim.discard(nm.levels);
        }
    }

    /// Search entities at level `c` under parent `p` (at index `pe` in `lvs`)
    /// for twin groups, recording local groups in `cl` and deferring skip-only
    /// groups that part below `c` to `self.pending`.
    #[allow(clippy::too_many_arguments)]
    fn contract_under_parent(
        &mut self,
        s: &SkipForm,
        sh: &super::Shape,
        (c, l, r, here): (u32, u32, u32, u32),
        (p, pcl): (u32, &CLv),
        cl: &mut CLv,
        queued: &mut [bool],
        heap: &mut BinaryHeap<Reverse<(u32, u32)>>,
        lim: &Limits,
    ) -> Result<(), OperationError> {
        let plvl = &s.levels[p as usize];
        let (pl, pr) = sh.kids[p as usize];
        let left = pl == c;
        let sib = if left { pr } else { pl };
        let sib_leaf = sh.is_leaf(sib);
        let classes = || {
            (0..pcl.width)
                .filter(|&i| pcl.group_of.is_empty() || pcl.group_of[i as usize] == NONE)
                .map(|i| (i, plvl.node(i)))
                .chain((0..pcl.groups()).map(|g| (pcl.width + g as u32, pcl.group(g))))
        };
        let partners = |b: Ref| -> [Ref; 2] {
            if sib_leaf && ix(b) == ONE {
                [mk(sib, POS), mk(sib, NEG)]
            } else {
                [b, ZREF]
            }
        };
        self.epoch += 1;
        self.ent.clear();
        for (k, pairs) in classes() {
            for &(a, b) in pairs {
                let (e, partner) = if left { (a, b) } else { (b, a) };
                let id = self
                    .ids
                    .id_alloc(s, e, &mut self.sstamp, &mut self.sslot, lim)?;
                if self.sstamp[id] != self.epoch {
                    self.sstamp[id] = self.epoch;
                    self.sslot[id] = self.ent.len() as u32;
                    lim.try_push(&mut self.ent, (e, 0, 0))?;
                }
                let x = &mut self.ent[self.sslot[id] as usize];
                for q in partners(partner) {
                    if q != ZREF {
                        x.1 = x.1.wrapping_add(mix(k, q));
                        x.2 += 1;
                    }
                }
            }
        }
        let any = mark_hash_candidates(&self.ent, &mut self.cand, &mut self.seen, lim)?;
        if any {
            self.occ.clear();
            for (k, pairs) in classes() {
                for &(a, b) in pairs {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    let i = self.sslot[self.ids.id(e)];
                    if self.cand[i as usize] {
                        for q in partners(partner) {
                            if q != ZREF {
                                lim.try_push(&mut self.occ, (i, k, q))?;
                            }
                        }
                    }
                }
            }
            sort_candidate_contexts(
                &self.ent,
                &self.cand,
                &mut self.order,
                &mut self.occ,
                &mut self.ctx,
                lim,
            )?;
            self.emit_confirmed_groups(s, sh, (c, l, r, here), cl, queued, heap, lim)?;
        }
        lim.charge_work(self.ent.len() as u64 + self.occ.len() as u64);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_confirmed_groups(
        &mut self,
        s: &SkipForm,
        sh: &super::Shape,
        (c, l, r, here): (u32, u32, u32, u32),
        cl: &mut CLv,
        queued: &mut [bool],
        heap: &mut BinaryHeap<Reverse<(u32, u32)>>,
        lim: &Limits,
    ) -> Result<(), OperationError> {
        let ctx = &self.ctx;
        let occ = &self.occ;
        let list = |i: u32| {
            let (a, b) = ctx[i as usize];
            occ[a as usize..b as usize].iter().map(|x| (x.1, x.2))
        };
        let ent = &self.ent;
        for run in self.order.chunk_by(|&x, &y| {
            (ent[x as usize].1, ent[x as usize].2) == (ent[y as usize].1, ent[y as usize].2)
        }) {
            if run.len() < 2 {
                continue;
            }
            self.firsts.clear();
            let mut of: Vec<u32> = Vec::with_capacity(run.len());
            for &i in run {
                let g = self.firsts.iter().position(|&f| list(f).eq(list(i)));
                of.push(match g {
                    Some(g) => g as u32,
                    None => {
                        lim.try_push(&mut self.firsts, i)?;
                        (self.firsts.len() - 1) as u32
                    }
                });
            }
            for g in 0..self.firsts.len() as u32 {
                self.members.clear();
                lim.reserve(&mut self.members, run.len())?;
                self.members.extend(
                    run.iter()
                        .zip(&of)
                        .filter(|x| *x.1 == g)
                        .map(|x| ent[*x.0 as usize].0),
                );
                if self.members.len() < 2 {
                    continue;
                }
                self.merged = true;
                if self.members.iter().all(|&e| vt(e) != c) {
                    if self.names.is_none() {
                        self.ids.ensure_all(s);
                        self.names = Some(Names::of(s, &self.ids, lim)?);
                    }
                    let nm = self.names.as_ref().expect("names were just read");
                    let v = parts(sh, nm, &self.ids, c, self.members.iter().copied());
                    if v != c {
                        let mut group = Vec::new();
                        lim.reserve_exact(&mut group, self.members.len())?;
                        group.extend_from_slice(&self.members);
                        self.pending.entry(v).or_default().push((here, group));
                        let mut u = v;
                        while u != c && !queued[u as usize] {
                            queued[u as usize] = true;
                            heap.push(Reverse((sh.pre[u as usize], u)));
                            u = sh.parent[u as usize];
                        }
                        continue;
                    }
                }
                let g = cl.push_group(s, &self.members, l, r, lim)?;
                lim.reserve_map(&mut cl.chainmap, self.members.len())?;
                for &e in &self.members {
                    if vt(e) != c {
                        cl.chainmap.insert(e, (here, g));
                    }
                }
            }
        }
        Ok(())
    }
}

/// [`contract`] scoped to the upward-closed `cone` of changed levels when
/// `scope` is `Some(cone)`: levels outside `cone` whose parent had no merge
/// and lost no node are already canonical and are not walked.
pub(crate) fn contract_in(
    s: SkipForm,
    scope: Option<&[bool]>,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let shape = s.shape.clone();
    let sh = &*shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        let z = SkipForm::zero(s.shape.clone());
        s.discard(lim);
        return Ok(z);
    }
    // The levels visited, parents first: those holding nodes, and the
    // inner children of a level with classes, where skips it names may merge.
    let mut at = vec![NONE; n];
    let mut lvs: Vec<CLv> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut queued = vec![false; n];
    for &t in &s.active {
        if scope.is_none_or(|cone| cone[t as usize]) {
            queued[t as usize] = true;
            heap.push(Reverse((sh.pre[t as usize], t)));
        }
    }
    let mut work = ContractWork::new(n);
    while let Some(Reverse((_, c))) = heap.pop() {
        let width = s.levels[c as usize].width() as u32;
        let mut cl = CLv::new(c, width);
        let here = lvs.len() as u32;
        let (l, r) = sh.kids[c as usize];
        let p = sh.parent[c as usize];
        let pe = if p == NONE { NONE } else { at[p as usize] };
        if pe != NONE {
            let pcl = &lvs[pe as usize];
            work.contract_under_parent(
                &s,
                sh,
                (c, l, r, here),
                (p, pcl),
                &mut cl,
                &mut queued,
                &mut heap,
                lim,
            )?;
        }
        record_pending_skip_groups(&s, &mut work, &mut lvs, &mut cl, (c, l, r, here), lim)?;
        lim.charge_work(s.levels[c as usize].pairs.len() as u64);
        if (width > 0 || cl.groups() > 0)
            && scope.is_none_or(|cone| cone[c as usize] || cl.groups() > 0)
        {
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
    let merged = work.merged;
    work.discard(lim);
    if !merged {
        for cl in lvs {
            cl.discard(lim);
        }
        return Ok(s);
    }
    patch_merged(s, lvs, &at, lim)
}

/// Drain merged skip groups sent down to `c` and record their chainmap entries.
fn record_pending_skip_groups(
    s: &SkipForm,
    work: &mut ContractWork,
    lvs: &mut [CLv],
    cl: &mut CLv,
    (c, l, r, here): (u32, u32, u32, u32),
    lim: &Limits,
) -> Result<(), OperationError> {
    if let Some(groups) = work.pending.remove(&c) {
        for (from, group) in groups {
            let g = cl.push_group(s, &group, l, r, lim)?;
            let origin = &mut lvs[from as usize].chainmap;
            lim.reserve_map(origin, group.len())?;
            for &e in &group {
                origin.insert(e, (here, g));
            }
            lim.discard(group);
        }
    }
    Ok(())
}

/// Mark entities in `ent` whose `(hash, count)` collides with another entity.
fn mark_hash_candidates(
    ent: &[(Ref, u64, u32)],
    cand: &mut Vec<bool>,
    seen: &mut FxHashMap<(u64, u32), u32>,
    lim: &Limits,
) -> Result<bool, OperationError> {
    let mut any = false;
    if ent.len() > 1 && ent.len() <= 8 {
        for i in 1..ent.len() {
            for j in 0..i {
                if (ent[i].1, ent[i].2) == (ent[j].1, ent[j].2) {
                    if !any {
                        cand.clear();
                        lim.try_resize(cand, ent.len(), false)?;
                        any = true;
                    }
                    cand[j] = true;
                    cand[i] = true;
                }
            }
        }
    } else if ent.len() > 8 {
        seen.clear();
        cand.clear();
        lim.try_resize(cand, ent.len(), false)?;
        lim.reserve_map(seen, ent.len())?;
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
    }
    Ok(any)
}

/// Order hash-collision candidates by `(hash, count)` and index each
/// candidate's sorted partner-occurrence range in `ctx`.
fn sort_candidate_contexts(
    ent: &[(Ref, u64, u32)],
    cand: &[bool],
    order: &mut Vec<u32>,
    occ: &mut [(u32, u32, Ref)],
    ctx: &mut Vec<(u32, u32)>,
    lim: &Limits,
) -> Result<(), OperationError> {
    order.clear();
    lim.reserve(order, ent.len())?;
    order.extend((0..ent.len() as u32).filter(|&i| cand[i as usize]));
    order.sort_unstable_by_key(|&i| (ent[i as usize].1, ent[i as usize].2));
    occ.sort_unstable();
    ctx.clear();
    lim.try_resize(ctx, ent.len(), (0, 0))?;
    for (j, w) in occ.chunk_by(|x, y| x.0 == y.0).scan(0u32, |o, w| {
        let start = *o;
        *o += w.len() as u32;
        Some((start, w))
    }) {
        ctx[w[0].0 as usize] = (j, j + w.len() as u32);
    }
    Ok(())
}

/// Patch `s` from the leaves up with the merged groups and rewrites in `lvs`,
/// then collect unreachable nodes across the visited levels.
fn patch_merged(
    mut s: SkipForm,
    mut lvs: Vec<CLv>,
    at: &[u32],
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let shape = s.shape.clone();
    let sh = &*shape;
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    for e in (0..lvs.len()).rev() {
        let c = lvs[e].level;
        let (l, r) = sh.kids[c as usize];
        let width = lvs[e].width as usize;
        let changes = |lvs: &[CLv], lvl: &SLevel, i: usize| {
            lvl.node(i as u32).iter().any(|&(a, b)| {
                resolve_patched_ref(sh, lvs, at, l, a) != a
                    || resolve_patched_ref(sh, lvs, at, r, b) != b
            })
        };
        let lvl = &s.levels[c as usize];
        let grouped = |i: usize| !lvs[e].group_of.is_empty() && lvs[e].group_of[i] != NONE;
        if lvs[e].groups() == 0 && !(0..width).any(|i| changes(&lvs, lvl, i)) {
            continue;
        }
        // Stored nodes keep their numbers; those merged into a group, or
        // that now resolve to another reference, stay behind unreachable.
        let mut out = SLevel::default();
        let mut rm = filled(lim, width, KEEP)?;
        for (i, x) in rm.iter_mut().enumerate() {
            let node = lvl.node(i as u32);
            if !grouped(i) && changes(&lvs, lvl, i) {
                buf.clear();
                lim.reserve(&mut buf, node.len())?;
                buf.extend(node.iter().map(|&(a, b)| {
                    (
                        resolve_patched_ref(sh, &lvs, at, l, a),
                        resolve_patched_ref(sh, &lvs, at, r, b),
                    )
                }));
                match normalize(sh, c, &mut buf) {
                    Err(t) => *x = t,
                    Ok(()) => {
                        out.push(&buf, lim)?;
                        continue;
                    }
                }
            }
            out.push(node, lim)?;
        }
        let gres = build_patched_group_results((sh, c), (&lvs, at, e), &mut out, &mut buf, lim)?;
        for (x, &g) in rm.iter_mut().zip(&lvs[e].group_of) {
            if g != NONE {
                *x = gres[g as usize];
            }
        }
        lvs[e].rm = rm;
        lvs[e].gres = gres;
        lim.release_bytes(s.levels[c as usize].bytes());
        s.levels[c as usize] = out;
    }
    s.out = resolve_patched_ref(sh, &lvs, at, sh.root, s.out);
    lim.discard(buf);
    let visited: Vec<u32> = lvs.iter().map(|l| l.level).collect();
    for cl in lvs {
        cl.discard(lim);
    }
    s.collect_in(visited, lim)?;
    Ok(s)
}

/// Resolve a side reference `x` read at `d` through any skip-joined group at `d`,
/// or else through its node's patched result.
#[inline]
fn resolve_patched_ref(sh: &Shape, lvs: &[CLv], at: &[u32], d: u32, x: Ref) -> Ref {
    if !sh.is_leaf(d) && vt(x) != d {
        let e = at[d as usize];
        if e != NONE
            && !lvs[e as usize].chainmap.is_empty()
            && let Some(&(to, g)) = lvs[e as usize].chainmap.get(&x)
        {
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
}

/// Normalize and push each merged twin group of `lvs[e]` into `out`, returning
/// each group's resolved reference.
fn build_patched_group_results(
    (sh, c): (&Shape, u32),
    (lvs, at, e): (&[CLv], &[u32], usize),
    out: &mut SLevel,
    buf: &mut Vec<(Ref, Ref)>,
    lim: &Limits,
) -> Result<Vec<Ref>, OperationError> {
    let (l, r) = sh.kids[c as usize];
    let mut gres = Vec::new();
    lim.reserve_exact(&mut gres, lvs[e].groups())?;
    for g in 0..lvs[e].groups() {
        buf.clear();
        let group = lvs[e].group(g);
        lim.reserve(buf, 2 * group.len())?;
        buf.extend(group.iter().map(|&(a, b)| {
            (
                resolve_patched_ref(sh, lvs, at, l, a),
                resolve_patched_ref(sh, lvs, at, r, b),
            )
        }));
        // Merged members can repeat a pair.
        buf.sort_unstable();
        buf.dedup();
        gres.push(match normalize(sh, c, buf) {
            Err(t) => t,
            Ok(()) => mk(c, out.push(buf, lim)?),
        });
    }
    Ok(gres)
}

/// Dense numbers for a diagram's references: stored node `i` of level `t`
/// is `base[t] + i`, leaf label `k` of leaf `c` is `base[c] + k`.
struct Ids {
    base: Vec<u32>,
    n: u32,
}

impl Ids {
    fn new(num_nodes: usize) -> Ids {
        Ids {
            base: vec![NONE; num_nodes],
            n: 0,
        }
    }
    #[inline]
    fn id_alloc(
        &mut self,
        s: &SkipForm,
        x: Ref,
        sstamp: &mut Vec<u32>,
        sslot: &mut Vec<u32>,
        lim: &Limits,
    ) -> Result<usize, OperationError> {
        let v = vt(x) as usize;
        let mut b = self.base[v];
        if b == NONE {
            b = self.n;
            self.base[v] = b;
            let add = if s.shape.is_leaf(v as u32) {
                3
            } else {
                s.levels[v].width() as u32
            };
            self.n += add;
        }
        let id = (b + ix(x)) as usize;
        if id >= sstamp.len() {
            let need = self.n as usize - sstamp.len();
            lim.reserve(sstamp, need)?;
            sstamp.resize(self.n as usize, 0u32);
            lim.reserve(sslot, need)?;
            sslot.resize(self.n as usize, 0u32);
        }
        Ok(id)
    }
    fn ensure_all(&mut self, s: &SkipForm) {
        let sh = &*s.shape;
        for &t in &s.active {
            if self.base[t as usize] == NONE {
                self.base[t as usize] = self.n;
                self.n += s.levels[t as usize].width() as u32;
            }
        }
        for (c, b) in self.base.iter_mut().enumerate() {
            if *b == NONE && sh.is_leaf(c as u32) {
                *b = self.n;
                self.n += 3;
            }
        }
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
    fn of(s: &SkipForm, ids: &Ids, lim: &Limits) -> Result<Names, OperationError> {
        let n = ids.n;
        let id = |x: Ref| ids.id(x);
        let mut start = filled(lim, n as usize + 1, 0u32)?;
        for &u in &s.active {
            for &(a, b) in &s.levels[u as usize].pairs {
                start[id(a) + 1] += 1;
                start[id(b) + 1] += 1;
            }
        }
        for i in 0..n as usize {
            start[i + 1] += start[i];
        }
        let mut fill = filled(lim, start.len(), 0u32)?;
        fill.copy_from_slice(&start);
        let mut levels = filled(lim, start[n as usize] as usize, 0u32)?;
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
        lim.discard(fill);
        Ok(Names { start, levels })
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
fn parts(
    sh: &super::Shape,
    nm: &Names,
    ids: &Ids,
    c: u32,
    members: impl Iterator<Item = Ref> + Clone,
) -> u32 {
    // The highest member, and where the members part.
    let mut h = NONE;
    for e in members.clone() {
        if h == NONE || sh.pre[vt(e) as usize] < sh.pre[h as usize] {
            h = vt(e);
        }
    }
    let mut w = if members.clone().all(|e| sh.within(vt(e), h)) {
        h
    } else {
        sh.parent[h as usize]
    };
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
