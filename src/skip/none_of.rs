//! That none of a set of cubes holds, built in skip form from the cubes at
//! once.
//!
//! At each vtree node, the assignments to its variables that leave every
//! cube unsatisfied so far fall into classes by their residual: the cubes
//! whose literals under the node all hold and that have a literal outside
//! it. A class is a node of the level (canonical up to two residuals that
//! mean the same, which [`contract`] merges). Its pairs are the pairs of
//! child classes whose residuals make its own: a cube survives a pair
//! where it holds on each child it reads, and a cube that survives and
//! reads nothing outside the node kills the pair.
//!
//! At a level whose one child is a leaf, a class of the other child that
//! has none of the leaf's cubes keeps its residual whatever the leaf is, so
//! it is `true` on the leaf and passes up as the reference it was: only the
//! classes holding a cube of the leaf are rewritten. On a spine of switches
//! above a table's parents a level then costs its own row, where a clause
//! conjoined at a time costs every row below it, and the stored form holds
//! a node per row above the level.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHasher};

use super::contract::contract;
use super::{build_top, discard_nested, mk, normalize, vt, Ref, Shape, SkipForm, NEG, ONE, POS, ZREF};
use crate::limits::Limits;
use crate::vtree::Vtree;
use crate::{LiteralInput, OperationError};

const NONE: u32 = u32::MAX;

/// A class of one level: its residual, `res[start..start + len]`, ascending
/// cube numbers; its node; the next class with the same hash.
struct Class {
    start: u32,
    len: u32,
    hash: u64,
    next: u32,
    node: Ref,
    alive: bool,
    /// The level stamp at which its node was last set or its pairs listed.
    born: u32,
    /// The level stamp at which a leaf's cube found it.
    mark: u32,
}

/// The classes of one vtree node, carried up the levels whose other child
/// is a leaf and rebuilt elsewhere.
#[derive(Default)]
struct Classes {
    cls: Vec<Class>,
    res: Vec<u32>,
    map: FxHashMap<u64, u32>,
    /// Per cube, the classes holding it, once a leaf level needs it.
    index: Option<FxHashMap<u32, Vec<u32>>>,
    live: usize,
}

fn hash_of(r: &[u32]) -> u64 {
    let mut h = FxHasher::default();
    r.hash(&mut h);
    h.finish()
}

impl Classes {
    fn residual(&self, k: u32) -> &[u32] {
        let c = &self.cls[k as usize];
        &self.res[c.start as usize..(c.start + c.len) as usize]
    }

    fn find(&self, r: &[u32], h: u64) -> Option<u32> {
        let mut k = *self.map.get(&h)?;
        while k != NONE {
            let c = &self.cls[k as usize];
            if c.alive && c.hash == h && self.residual(k) == r {
                return Some(k);
            }
            k = c.next;
        }
        None
    }

    fn insert(&mut self, r: &[u32], h: u64, node: Ref, born: u32, lim: &Limits) -> Result<u32, OperationError> {
        let k = self.cls.len() as u32;
        let start = self.res.len() as u32;
        lim.reserve(&mut self.res, r.len())?;
        self.res.extend_from_slice(r);
        let next = self.map.insert(h, k).unwrap_or(NONE);
        lim.try_push(&mut self.cls, Class { start, len: r.len() as u32, hash: h, next, node, alive: true, born, mark: 0 })?;
        self.live += 1;
        if let Some(index) = self.index.as_mut() {
            for &c in &self.res[start as usize..] {
                index.entry(c).or_default().push(k);
            }
        }
        Ok(k)
    }

    fn kill(&mut self, k: u32) {
        self.cls[k as usize].alive = false;
        self.live -= 1;
    }

    fn ensure_index(&mut self) {
        if self.index.is_some() {
            return;
        }
        let mut index: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for (k, c) in self.cls.iter().enumerate() {
            if c.alive {
                for &cube in &self.res[c.start as usize..(c.start + c.len) as usize] {
                    index.entry(cube).or_default().push(k as u32);
                }
            }
        }
        self.index = Some(index);
    }

    /// Drop the classes, handing their charge back to `lim`.
    fn discard(self, lim: &Limits) {
        lim.discard(self.cls);
        lim.discard(self.res);
    }

    fn alive(&self) -> impl Iterator<Item = u32> + '_ {
        self.cls.iter().enumerate().filter(|(_, c)| c.alive).map(|(k, _)| k as u32)
    }
}

/// The union of three ascending lists of distinct cubes, none in two of
/// them, ascending, appended to `out`.
fn merge3(a: &[u32], b: &[u32], c: &[u32], out: &mut Vec<u32>) {
    let (mut i, mut j, mut k) = (0, 0, 0);
    loop {
        let x = a.get(i).copied().unwrap_or(u32::MAX);
        let y = b.get(j).copied().unwrap_or(u32::MAX);
        let z = c.get(k).copied().unwrap_or(u32::MAX);
        let m = x.min(y).min(z);
        if m == u32::MAX {
            return;
        }
        out.push(m);
        if m == x {
            i += 1;
        } else if m == y {
            j += 1;
        } else {
            k += 1;
        }
    }
}

/// The cubes as the build reads them: per cube its leaves' preorder
/// positions, ascending; per leaf its cubes and their signs.
struct Cubes {
    pres: Vec<u32>,
    pstart: Vec<u32>,
    at: Vec<Vec<(u32, bool)>>,
}

impl Cubes {
    fn leaves(&self, c: u32) -> &[u32] {
        &self.pres[self.pstart[c as usize] as usize..self.pstart[c as usize + 1] as usize]
    }

    /// Whether cube `c` reads a variable under `s`.
    fn touches(&self, sh: &Shape, c: u32, s: u32) -> bool {
        let (lo, hi) = (sh.pre[s as usize], sh.pre[s as usize] + sh.size[s as usize]);
        let p = self.leaves(c);
        let i = p.partition_point(|&x| x < lo);
        i < p.len() && p[i] < hi
    }

    /// Whether every variable cube `c` reads is under `t`.
    fn within(&self, sh: &Shape, c: u32, t: u32) -> bool {
        let (lo, hi) = (sh.pre[t as usize], sh.pre[t as usize] + sh.size[t as usize]);
        let p = self.leaves(c);
        p[0] >= lo && *p.last().expect("a cube reads a variable") < hi
    }
}

/// The skip form of `¬c₁ ∧ … ∧ ¬cₙ` over `vtree` and `shape` (the vtree's own
/// shape), canonical. A cube that names a variable in both signs never
/// holds and is left out; an empty one always holds, and the result is
/// false.
pub(crate) fn none_of<L: LiteralInput>(
    vtree: &Vtree,
    shape: &Arc<Shape>,
    cubes: &[Vec<L>],
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let sh = &**shape;
    let n = sh.num_nodes();
    let mut out = SkipForm::zero(Arc::clone(shape));
    // Each cube's leaves, by preorder, without repeats and contradictions.
    let mut pres: Vec<u32> = Vec::new();
    let mut pstart: Vec<u32> = vec![0];
    let mut at: Vec<Vec<(u32, bool)>> = (0..n).map(|_| Vec::new()).collect();
    let mut lits: Vec<(u32, u32, bool)> = Vec::new();
    let mut gate = lim.gate();
    for cube in cubes {
        lits.clear();
        for raw in cube {
            gate.poll(1)?;
            let lit = raw.literal()?;
            let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
            lim.try_push(&mut lits, (sh.pre[leaf.0 as usize], leaf.0, lit.sign))?;
        }
        if lits.is_empty() {
            // A cube that always holds: none of them never does.
            return Ok(out);
        }
        lits.sort_unstable();
        lits.dedup();
        if lits.windows(2).any(|w| w[0].1 == w[1].1) {
            continue;
        }
        let c = (pstart.len() - 1) as u32;
        for &(p, leaf, sign) in &lits {
            lim.try_push(&mut pres, p)?;
            lim.try_push(&mut at[leaf as usize], (c, sign))?;
        }
        lim.try_push(&mut pstart, pres.len() as u32)?;
    }
    gate.flush()?;
    lim.discard(lits);
    let cs = Cubes { pres, pstart, at };
    let ncubes = cs.pstart.len() - 1;
    // Nodes with a cube's variable under them.
    let mut touched = vec![false; n];
    for (v, a) in cs.at.iter().enumerate() {
        if !a.is_empty() {
            let mut u = v as u32;
            while u != NONE && !touched[u as usize] {
                touched[u as usize] = true;
                u = sh.parent[u as usize];
            }
        }
    }
    if ncubes == 0 {
        out.out = build_top(sh, sh.root, &mut |u, pair| Ok(mk(u, out.push_node(u, &[pair], lim)?)))?;
        return contract_out(out, lim);
    }
    let mut b = Build {
        sh,
        cs: &cs,
        stamp: 0,
        xsign: vec![(0, false); ncubes],
        mark: vec![0; ncubes],
        marks: 0,
        buf: Vec::new(),
        pairs: Vec::new(),
        scratch: Vec::new(),
        sides: Vec::new(),
    };
    if sh.is_leaf(sh.root) {
        let c = b.leaf_classes(sh.root, lim)?;
        out.out = c.alive().find(|&k| c.cls[k as usize].len == 0).map_or(ZREF, |k| c.cls[k as usize].node);
        return Ok(out);
    }
    let mut sets: Vec<Option<Classes>> = (0..n).map(|_| None).collect();
    for &t in &sh.bottomup {
        if !touched[t as usize] {
            continue;
        }
        let (l, r) = sh.kids[t as usize];
        let mut take = |c: u32, out: &mut SkipForm, b: &mut Build| -> Result<Classes, OperationError> {
            if !touched[c as usize] {
                let top = build_top(sh, c, &mut |u, pair| Ok(mk(u, out.push_node(u, &[pair], lim)?)))?;
                let mut k = Classes::default();
                k.insert(&[], hash_of(&[]), top, 0, lim)?;
                Ok(k)
            } else if sh.is_leaf(c) {
                b.leaf_classes(c, lim)
            } else {
                Ok(sets[c as usize].take().expect("a child is built before its parent"))
            }
        };
        let cl = take(l, &mut out, &mut b)?;
        let cr = take(r, &mut out, &mut b)?;
        if cl.live == 0 || cr.live == 0 {
            out.discard(lim);
            return Ok(SkipForm::zero(Arc::clone(shape)));
        }
        let ct = b.combine(t, cl, cr, &mut out, &mut gate, lim)?;
        if ct.live == 0 {
            out.discard(lim);
            return Ok(SkipForm::zero(Arc::clone(shape)));
        }
        sets[t as usize] = Some(ct);
    }
    gate.flush()?;
    let root = sets[sh.root as usize].take().expect("the root holds a cube's variable");
    // At the root every cube is read whole, so a live class has no residual.
    debug_assert!(root.alive().all(|k| root.cls[k as usize].len == 0));
    out.out = root.alive().next().map_or(ZREF, |k| root.cls[k as usize].node);
    root.discard(lim);
    b.discard(lim);
    lim.discard(cs.pres);
    lim.discard(cs.pstart);
    discard_nested(lim, cs.at);
    contract_out(out, lim)
}

fn contract_out(mut out: SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    if out.out == ZREF {
        let shape = Arc::clone(&out.shape);
        out.discard(lim);
        return Ok(SkipForm::zero(shape));
    }
    out.collect(lim)?;
    contract(out, lim)
}

/// The build's working state.
struct Build<'a> {
    sh: &'a Shape,
    cs: &'a Cubes,
    stamp: u32,
    /// Per cube, the sign it gives the current level's leaf, where the
    /// stamp is the level's.
    xsign: Vec<(u32, bool)>,
    /// Per cube, the stamp of the left class whose two-sided cubes
    /// [`Build::general`] marked, `marks` the latest.
    mark: Vec<u32>,
    marks: u32,
    buf: Vec<(Ref, Ref)>,
    /// Pairs listed for the current level's classes: class, left, right.
    pairs: Vec<(u32, Ref, Ref)>,
    scratch: Vec<u32>,
    /// A general level's residuals split by side: per class its cubes
    /// that read nothing under the other child, then those that do.
    sides: Vec<u32>,
}

impl Build<'_> {
    fn discard(self, lim: &Limits) {
        lim.discard(self.buf);
        lim.discard(self.pairs);
        lim.discard(self.scratch);
        lim.discard(self.sides);
    }

    /// The classes of leaf `x`: per value its cubes that read more than `x`,
    /// a value that satisfies a cube of `x` alone left out; one class `One`
    /// where both values leave no cube.
    fn leaf_classes(&mut self, x: u32, lim: &Limits) -> Result<Classes, OperationError> {
        let mut k = Classes::default();
        let mut res: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        let mut dead = [false, false];
        for &(c, sign) in &self.cs.at[x as usize] {
            let v = usize::from(!sign);
            if self.cs.leaves(c).len() == 1 {
                dead[v] = true;
            } else {
                lim.try_push(&mut res[v], c)?;
            }
        }
        for r in res.iter_mut() {
            r.sort_unstable();
        }
        if !dead[0] && !dead[1] && res[0].is_empty() && res[1].is_empty() {
            k.insert(&[], hash_of(&[]), mk(x, ONE), 0, lim)?;
        } else {
            for (v, label) in [(0, POS), (1, NEG)] {
                if !dead[v] {
                    k.insert(&res[v], hash_of(&res[v]), mk(x, label), 0, lim)?;
                }
            }
        }
        let [r0, r1] = res;
        lim.discard(r0);
        lim.discard(r1);
        Ok(k)
    }

    fn combine(
        &mut self,
        t: u32,
        cl: Classes,
        cr: Classes,
        out: &mut SkipForm,
        gate: &mut crate::limits::PollGate<'_>,
        lim: &Limits,
    ) -> Result<Classes, OperationError> {
        let sh = self.sh;
        let (l, r) = sh.kids[t as usize];
        self.stamp += 1;
        if sh.is_leaf(l) != sh.is_leaf(r) {
            if sh.is_leaf(l) && self.passes(l, r, &cl) {
                let ct = self.carry(t, l, true, &cl, cr, out, gate, lim);
                cl.discard(lim);
                return ct;
            }
            if sh.is_leaf(r) && self.passes(r, l, &cr) {
                let ct = self.carry(t, r, false, &cr, cl, out, gate, lim);
                cr.discard(lim);
                return ct;
            }
        }
        let ct = self.general(t, &cl, &cr, out, gate, lim);
        cl.discard(lim);
        cr.discard(lim);
        ct
    }

    /// Whether the classes of the other child that hold no cube of leaf `x`
    /// pass through the level: `x`'s classes cover both its values, and
    /// each cube of theirs reads under the other child `o`.
    fn passes(&self, x: u32, o: u32, cx: &Classes) -> bool {
        let both = cx.live == 2 || (cx.live == 1 && cx.alive().all(|k| cx.cls[k as usize].node == mk(x, ONE)));
        both && cx.alive().all(|k| cx.residual(k).iter().all(|&c| self.cs.touches(self.sh, c, o)))
    }

    /// The level at `t` over leaf `x` and the other child's classes `co`:
    /// only the classes holding a cube of `x` are rewritten.
    #[expect(clippy::too_many_arguments)]
    fn carry(
        &mut self,
        t: u32,
        x: u32,
        x_left: bool,
        cx: &Classes,
        mut co: Classes,
        out: &mut SkipForm,
        gate: &mut crate::limits::PollGate<'_>,
        lim: &Limits,
    ) -> Result<Classes, OperationError> {
        let sh = self.sh;
        let st = self.stamp;
        if self.cs.at[x as usize].is_empty() {
            return Ok(co);
        }
        co.ensure_index();
        for &(c, sign) in &self.cs.at[x as usize] {
            self.xsign[c as usize] = (st, sign);
        }
        let join = |a: Ref, b: Ref| if x_left { (a, b) } else { (b, a) };
        // The classes holding a cube of `x`.
        let mut hit: Vec<u32> = Vec::new();
        {
            let index = co.index.as_ref().expect("built above");
            for &(c, _) in &self.cs.at[x as usize] {
                if let Some(ks) = index.get(&c) {
                    for &k in ks {
                        let cl = &mut co.cls[k as usize];
                        if cl.alive && cl.mark != st {
                            cl.mark = st;
                            lim.try_push(&mut hit, k)?;
                        }
                    }
                }
            }
        }
        for &k in &hit {
            co.kill(k);
        }
        self.pairs.clear();
        let xs: Vec<(Ref, bool)> = cx.alive().map(|k| (cx.cls[k as usize].node, cx.cls[k as usize].node == mk(x, POS))).collect();
        for &k in &hit {
            let bnode = co.cls[k as usize].node;
            for &(anode, pos) in &xs {
                gate.poll(1 + co.cls[k as usize].len as u64)?;
                self.scratch.clear();
                let mut dead = false;
                let (s, e) = (co.cls[k as usize].start as usize, (co.cls[k as usize].start + co.cls[k as usize].len) as usize);
                for i in s..e {
                    let c = co.res[i];
                    let (cst, sign) = self.xsign[c as usize];
                    if cst == st {
                        if sign != pos {
                            continue;
                        }
                        if self.cs.within(sh, c, t) {
                            dead = true;
                            break;
                        }
                    }
                    lim.try_push(&mut self.scratch, c)?;
                }
                if dead {
                    continue;
                }
                let h = hash_of(&self.scratch);
                let cls = match co.find(&self.scratch, h) {
                    Some(e) if co.cls[e as usize].born != st => {
                        // A class passing through meets a new pair: it is
                        // `true` on `x` beside it.
                        co.cls[e as usize].born = st;
                        let enode = co.cls[e as usize].node;
                        let (a, b) = join(mk(x, ONE), enode);
                        lim.try_push(&mut self.pairs, (e, a, b))?;
                        e
                    }
                    Some(e) => e,
                    None => {
                        let scratch = std::mem::take(&mut self.scratch);
                        let e = co.insert(&scratch, h, ZREF, st, lim)?;
                        self.scratch = scratch;
                        e
                    }
                };
                let (a, b) = join(anode, bnode);
                lim.try_push(&mut self.pairs, (cls, a, b))?;
            }
        }
        lim.discard(hit);
        self.emit(t, &mut co, out, lim)?;
        Ok(co)
    }

    /// The level at `t` from every pair of its children's classes. A cube
    /// of a pair's residual reads outside `t` on one side only, and is kept,
    /// or reads under both children, and is kept where both classes hold it:
    /// each class's residual is split once into the two, and a pair
    /// intersects the second parts by reading the shorter one, so that a
    /// class with a long residual beside many short ones is not read whole
    /// at each pair. A kept cube read whole under `t` kills the pair.
    fn general(
        &mut self,
        t: u32,
        cl: &Classes,
        cr: &Classes,
        out: &mut SkipForm,
        gate: &mut crate::limits::PollGate<'_>,
        lim: &Limits,
    ) -> Result<Classes, OperationError> {
        let sh = self.sh;
        let (l, r) = sh.kids[t as usize];
        let st = self.stamp;
        let mut ct = Classes::default();
        self.pairs.clear();
        // Per class, (class, start, middle, end) in `sides`: the cubes that
        // read nothing under the other child, then those that do.
        self.sides.clear();
        let mut split = |k: &Classes, other: u32, sides: &mut Vec<u32>| -> Result<Vec<(u32, u32, u32, u32)>, OperationError> {
            let mut at = Vec::new();
            for x in k.alive() {
                let rx = k.residual(x);
                gate.poll(1 + rx.len() as u64)?;
                let start = sides.len() as u32;
                lim.reserve(sides, rx.len())?;
                sides.extend(rx.iter().copied().filter(|&c| !self.cs.touches(sh, c, other)));
                let mid = sides.len() as u32;
                sides.extend(rx.iter().copied().filter(|&c| self.cs.touches(sh, c, other)));
                lim.try_push(&mut at, (x, start, mid, sides.len() as u32))?;
            }
            Ok(at)
        };
        let mut sides = std::mem::take(&mut self.sides);
        let left = split(cl, r, &mut sides)?;
        let right = split(cr, l, &mut sides)?;
        for &(a, a0, a1, a2) in &left {
            self.marks += 1;
            let m = self.marks;
            for &c in &sides[a1 as usize..a2 as usize] {
                self.mark[c as usize] = m;
            }
            let (one_a, two_a) = (&sides[a0 as usize..a1 as usize], &sides[a1 as usize..a2 as usize]);
            for &(b, b0, b1, b2) in &right {
                let (one_b, two_b) = (&sides[b0 as usize..b1 as usize], &sides[b1 as usize..b2 as usize]);
                // Both sides' cubes, ascending, from the shorter list.
                self.scratch.clear();
                let mut dead = false;
                let read = two_a.len().min(two_b.len());
                gate.poll(1 + (one_a.len() + one_b.len() + read) as u64)?;
                let both = |c: &u32| if two_b.len() <= two_a.len() { self.mark[*c as usize] == m } else { two_b.binary_search(c).is_ok() };
                let shorter = if two_b.len() <= two_a.len() { two_b } else { two_a };
                for &c in shorter.iter().filter(|c| both(c)) {
                    if self.cs.within(sh, c, t) {
                        dead = true;
                        break;
                    }
                    lim.try_push(&mut self.scratch, c)?;
                }
                if dead {
                    continue;
                }
                if !one_a.is_empty() || !one_b.is_empty() {
                    let common = std::mem::take(&mut self.scratch);
                    lim.reserve(&mut self.scratch, common.len() + one_a.len() + one_b.len())?;
                    merge3(&common, one_a, one_b, &mut self.scratch);
                    lim.discard(common);
                }
                debug_assert!(self.scratch.iter().all(|&c| !self.cs.within(sh, c, t)));
                let h = hash_of(&self.scratch);
                let cls = match ct.find(&self.scratch, h) {
                    Some(e) => e,
                    None => ct.insert(&self.scratch, h, ZREF, st, lim)?,
                };
                lim.try_push(&mut self.pairs, (cls, cl.cls[a as usize].node, cr.cls[b as usize].node))?;
            }
        }
        lim.discard(left);
        lim.discard(right);
        self.sides = sides;
        self.emit(t, &mut ct, out, lim)?;
        Ok(ct)
    }

    /// The nodes of the classes the current level listed pairs for: stored
    /// at `t`, or the reference a node `true` on a leaf child names.
    fn emit(&mut self, t: u32, k: &mut Classes, out: &mut SkipForm, lim: &Limits) -> Result<(), OperationError> {
        self.pairs.sort_unstable_by_key(|p| p.0);
        let mut i = 0;
        while i < self.pairs.len() {
            let cls = self.pairs[i].0;
            self.buf.clear();
            while i < self.pairs.len() && self.pairs[i].0 == cls {
                lim.try_push(&mut self.buf, (self.pairs[i].1, self.pairs[i].2))?;
                i += 1;
            }
            let node = match normalize(self.sh, t, &mut self.buf) {
                Err(target) => target,
                Ok(()) => mk(t, out.push_node(t, &self.buf, lim)?),
            };
            debug_assert!(node != ZREF && (vt(node) == t || !self.sh.within(t, vt(node))));
            k.cls[cls as usize].node = node;
        }
        Ok(())
    }
}
