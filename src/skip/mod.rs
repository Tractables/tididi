//! Diagrams whose pairs skip levels.
//!
//! A pair side may name a node below the child level it belongs to, where
//! every level between skips a single variable: a node `true_x ∧ D` (pairs
//! `(One, D)`, or `(Pos, D), (Neg, D)`) at `t = (x, R)` is never stored, and a
//! reference to it names `D` instead; likewise `E ∧ true_y` at `t = (L, y)`.
//! The skip form of a canonical diagram is canonical: the expansion of a
//! reference is unique, because the path from the side's child level to the
//! named node fixes the skipped variables.
//!
//! References are resolved here: a [`Ref`] packs the vtree node that holds
//! the named node and its index there (a leaf label on a leaf level), so a
//! side whose node is not its child level's is a skip.

#![allow(dead_code)]

pub(crate) mod build;
pub(crate) mod conjoin;
pub(crate) mod contract;
pub(crate) mod convert;
pub(crate) mod negate;
pub(crate) mod place;
pub(crate) mod tag;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeNode};

/// A resolved reference: the vtree node in bits 32 to 61, the index in the
/// low word, and in the top two bits the tagged form's tag ([`tag`]),
/// zero elsewhere. [`ZREF`] is the constant false.
pub(crate) type Ref = u64;
/// The constant false.
pub(crate) const ZREF: Ref = u64::MAX;
pub(crate) const ONE: u32 = 0;
pub(crate) const POS: u32 = 1;
pub(crate) const NEG: u32 = 2;
const NONE: u32 = u32::MAX;

#[inline]
pub(crate) fn mk(v: u32, i: u32) -> Ref {
    ((v as u64) << 32) | i as u64
}
#[inline]
pub(crate) fn vt(r: Ref) -> u32 {
    ((r >> 32) & 0x3fff_ffff) as u32
}
#[inline]
pub(crate) fn ix(r: Ref) -> u32 {
    r as u32
}

/// The vtree as the skip operations read it.
#[derive(Debug)]
pub(crate) struct Shape {
    pub(crate) root: u32,
    /// Children of an internal node; `(NONE, NONE)` on a leaf.
    pub(crate) kids: Vec<(u32, u32)>,
    pub(crate) parent: Vec<u32>,
    /// Preorder position and subtree size: `a` is in `b`'s subtree when
    /// `pre[b] <= pre[a] < pre[b] + size[b]`.
    pub(crate) pre: Vec<u32>,
    pub(crate) size: Vec<u32>,
    /// Internal nodes, every parent before its children.
    pub(crate) topdown: Vec<u32>,
    /// Internal nodes, every child before its parent.
    pub(crate) bottomup: Vec<u32>,
    /// A leaf's variable id; 0 on an internal node.
    pub(crate) var: Vec<u32>,
    /// Edges from the root.
    pub(crate) depth: Vec<u32>,
}

impl Shape {
    pub(crate) fn of(vt: &Vtree) -> Shape {
        let n = vt.num_nodes();
        let mut kids = vec![(NONE, NONE); n];
        let mut parent = vec![NONE; n];
        let mut var = vec![0u32; n];
        for i in 0..n {
            match vt.node(crate::vtree::VtreeIdx(i as u32)) {
                VtreeNode::Leaf { var: v, parent: p } => {
                    var[i] = v.0;
                    parent[i] = p.map_or(NONE, |p| p.0);
                }
                VtreeNode::Internal { left, right, parent: p } => {
                    kids[i] = (left.0, right.0);
                    parent[i] = p.map_or(NONE, |p| p.0);
                }
            }
        }
        let root = vt.root().0;
        let (mut pre, mut size) = (vec![0u32; n], vec![1u32; n]);
        let mut topdown = Vec::new();
        let mut order = Vec::with_capacity(n);
        let mut stack = vec![root];
        while let Some(t) = stack.pop() {
            pre[t as usize] = order.len() as u32;
            order.push(t);
            let (l, r) = kids[t as usize];
            if l != NONE {
                topdown.push(t);
                stack.push(r);
                stack.push(l);
            }
        }
        for &t in order.iter().rev() {
            let (l, r) = kids[t as usize];
            if l != NONE {
                size[t as usize] = 1 + size[l as usize] + size[r as usize];
            }
        }
        let bottomup = topdown.iter().rev().copied().collect();
        let mut depth = vec![0u32; n];
        for &t in &order {
            if parent[t as usize] != NONE {
                depth[t as usize] = depth[parent[t as usize] as usize] + 1;
            }
        }
        Shape { root, kids, parent, pre, size, topdown, bottomup, var, depth }
    }

    #[inline]
    pub(crate) fn is_leaf(&self, v: u32) -> bool {
        self.kids[v as usize].0 == NONE
    }

    /// Whether `a` is in `b`'s subtree (`a == b` included).
    #[inline]
    pub(crate) fn within(&self, a: u32, b: u32) -> bool {
        let (pa, pb) = (self.pre[a as usize], self.pre[b as usize]);
        pa >= pb && pa < pb + self.size[b as usize]
    }

    /// The child of internal `t` on the path to `v`, a node strictly below `t`.
    #[inline]
    pub(crate) fn toward(&self, t: u32, v: u32) -> u32 {
        let (l, r) = self.kids[t as usize];
        if self.within(v, l) { l } else { r }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.kids.len()
    }

    /// The lowest common ancestor of `a` and `b`.
    pub(crate) fn lca(&self, mut a: u32, mut b: u32) -> u32 {
        while self.depth[a as usize] > self.depth[b as usize] {
            a = self.parent[a as usize];
        }
        while self.depth[b as usize] > self.depth[a as usize] {
            b = self.parent[b as usize];
        }
        while a != b {
            a = self.parent[a as usize];
            b = self.parent[b as usize];
        }
        a
    }
}

/// One level's real nodes: node `i`'s pairs are `pairs[start[i]..start[i + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SLevel {
    pub(crate) start: Vec<u32>,
    pub(crate) pairs: Vec<(Ref, Ref)>,
}

impl SLevel {
    #[inline]
    pub(crate) fn width(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
    #[inline]
    pub(crate) fn node(&self, i: u32) -> &[(Ref, Ref)] {
        &self.pairs[self.start[i as usize] as usize..self.start[i as usize + 1] as usize]
    }
    /// Append a node and return its index.
    pub(crate) fn push(&mut self, pairs: &[(Ref, Ref)]) -> u32 {
        if self.start.is_empty() {
            self.start.push(0);
        }
        self.pairs.extend_from_slice(pairs);
        self.start.push(self.pairs.len() as u32);
        (self.start.len() - 2) as u32
    }
}

/// A diagram in skip form over [`Shape`]: every reference resolved.
#[derive(Clone, Debug)]
pub(crate) struct SkipForm {
    pub(crate) shape: Arc<Shape>,
    pub(crate) levels: Vec<SLevel>,
    /// The function, as a reference from the root.
    pub(crate) out: Ref,
}

/// Sizes of a skip-form diagram.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SkipSize {
    pub(crate) nodes: u64,
    pub(crate) pairs: u64,
    /// Pair sides that skip, and their distinct (child level, target).
    pub(crate) skips: u64,
    pub(crate) forwarders: u64,
}

impl SkipForm {
    pub(crate) fn zero(shape: Arc<Shape>) -> SkipForm {
        let n = shape.num_nodes();
        SkipForm { shape, levels: vec![SLevel::default(); n], out: ZREF }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.out == ZREF
    }

    /// The pairs of the node `r` names, read at internal level `u` that
    /// `r`'s node is in or below: a node of a level `r` skips through `u`
    /// is the one pair `(r, One)` or `(One, r)` that `true` on the skipped
    /// leaf side makes.
    #[inline]
    pub(crate) fn pairs_at<'a>(&'a self, u: u32, r: Ref, buf: &'a mut [(Ref, Ref); 1]) -> &'a [(Ref, Ref)] {
        if vt(r) == u {
            return self.levels[u as usize].node(ix(r));
        }
        let (l, rr) = self.shape.kids[u as usize];
        buf[0] = if self.shape.within(vt(r), l) {
            debug_assert!(self.shape.is_leaf(rr), "a skip passes a single variable");
            (r, mk(rr, ONE))
        } else {
            debug_assert!(self.shape.is_leaf(l), "a skip passes a single variable");
            (mk(l, ONE), r)
        };
        &buf[..]
    }

    pub(crate) fn size(&self) -> SkipSize {
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
                        seen.insert((c, x));
                    }
                }
            }
        }
        if self.out != ZREF && vt(self.out) != sh.root {
            s.skips += 1;
            seen.insert((sh.root, self.out));
        }
        s.forwarders = seen.len() as u64;
        s
    }

    /// Drop nodes the output does not reach and number the rest in order.
    pub(crate) fn collect(&mut self) {
        let sh = Arc::clone(&self.shape);
        let n = sh.num_nodes();
        if self.out == ZREF {
            self.levels = vec![SLevel::default(); n];
            return;
        }
        let mut live: Vec<Vec<bool>> = self.levels.iter().map(|l| vec![false; l.width()]).collect();
        let mark = |live: &mut Vec<Vec<bool>>, r: Ref| {
            if !sh.is_leaf(vt(r)) {
                live[vt(r) as usize][ix(r) as usize] = true;
            }
        };
        mark(&mut live, self.out);
        for &t in &sh.topdown {
            let lvl = &self.levels[t as usize];
            for i in 0..lvl.width() {
                if live[t as usize][i] {
                    for &(a, b) in lvl.node(i as u32) {
                        mark(&mut live, a);
                        mark(&mut live, b);
                    }
                }
            }
        }
        let mut remap: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &t in &sh.bottomup {
            let mut next = 0u32;
            remap[t as usize] = live[t as usize].iter().map(|&k| if k { next += 1; next - 1 } else { NONE }).collect();
        }
        // A tagged reference keeps its tag.
        let fix = |remap: &Vec<Vec<u32>>, r: Ref| -> Ref {
            if sh.is_leaf(vt(r)) { r } else { (r & tag::TAG_MASK) | mk(vt(r), remap[vt(r) as usize][ix(r) as usize]) }
        };
        for &t in &sh.bottomup {
            let old = std::mem::take(&mut self.levels[t as usize]);
            let mut lvl = SLevel::default();
            for i in 0..old.width() {
                if live[t as usize][i] {
                    let start = lvl.pairs.len();
                    if lvl.start.is_empty() {
                        lvl.start.push(0);
                    }
                    for &(a, b) in old.node(i as u32) {
                        lvl.pairs.push((fix(&remap, a), fix(&remap, b)));
                    }
                    debug_assert!(lvl.pairs.len() > start);
                    lvl.start.push(lvl.pairs.len() as u32);
                }
            }
            self.levels[t as usize] = lvl;
        }
        self.out = fix(&remap, self.out);
    }

    /// The model count over the vtree's variables.
    pub(crate) fn model_count(&self) -> num_bigint::BigUint {
        use num_bigint::BigUint;
        let sh = &self.shape;
        if self.out == ZREF {
            return BigUint::from(0u32);
        }
        // Variables below each node, and depth, for the skipped count.
        let n = sh.num_nodes();
        let mut nv = vec![0u32; n];
        let mut depth = vec![0u32; n];
        for &t in &sh.topdown {
            let (l, r) = sh.kids[t as usize];
            depth[l as usize] = depth[t as usize] + 1;
            depth[r as usize] = depth[t as usize] + 1;
        }
        for t in 0..n {
            if sh.is_leaf(t as u32) {
                nv[t] = 1;
            }
        }
        for &t in &sh.bottomup {
            let (l, r) = sh.kids[t as usize];
            nv[t as usize] = nv[l as usize] + nv[r as usize];
        }
        let mut val: Vec<Vec<BigUint>> = vec![Vec::new(); n];
        let value = |val: &Vec<Vec<BigUint>>, c: u32, r: Ref| -> BigUint {
            let v = vt(r);
            let base = if sh.is_leaf(v) {
                BigUint::from(if ix(r) == ONE { 2u32 } else { 1 })
            } else {
                val[v as usize][ix(r) as usize].clone()
            };
            // Each level crossed from `c` to `v` skips one variable.
            base << (depth[v as usize] - depth[c as usize])
        };
        for &t in &sh.bottomup {
            let (l, r) = sh.kids[t as usize];
            let lvl = &self.levels[t as usize];
            let mut out = Vec::with_capacity(lvl.width());
            for i in 0..lvl.width() {
                let mut s = BigUint::from(0u32);
                for &(a, b) in lvl.node(i as u32) {
                    s += value(&val, l, a) * value(&val, r, b);
                }
                out.push(s);
            }
            val[t as usize] = out;
        }
        value(&val, sh.root, self.out)
    }
}

/// Normalize the pairs of a node at internal `u` (sorted, sides resolved,
/// no zero side) and resolve it: `Err(target)` when the node is `true` on
/// one single-variable side, `Ok(())` with `pairs` in their stored form
/// otherwise.
///
/// Against a leaf child the stored form names `One` where the node holds
/// both literals with the same other side; the left leaf is merged first,
/// then the right, so the form is a function of the node's assignments.
pub(crate) fn normalize(sh: &Shape, u: u32, pairs: &mut Vec<(Ref, Ref)>) -> Result<(), Ref> {
    merge_cells(sh, u, pairs);
    let (l, r) = sh.kids[u as usize];
    if pairs.len() == 1 {
        let (a, b) = pairs[0];
        if sh.is_leaf(l) && a == mk(l, ONE) {
            return Err(b);
        }
        if sh.is_leaf(r) && b == mk(r, ONE) {
            return Err(a);
        }
    }
    Ok(())
}

/// The stored form of a node's pairs at internal `u` ([`normalize`]'s
/// merges, without resolving the node).
pub(crate) fn merge_cells(sh: &Shape, u: u32, pairs: &mut Vec<(Ref, Ref)>) {
    let (l, r) = sh.kids[u as usize];
    let (ll, rl) = (sh.is_leaf(l), sh.is_leaf(r));
    // Cells: every One against a leaf child as both literals.
    if (ll && pairs.iter().any(|p| ix(p.0) == ONE)) || (rl && pairs.iter().any(|p| ix(p.1) == ONE)) {
        let mut cells = Vec::with_capacity(pairs.len() * 2);
        for &(a, b) in pairs.iter() {
            let lefts: &[Ref] = if ll && ix(a) == ONE { &[mk(l, POS), mk(l, NEG)] } else { &[a] };
            for &x in lefts {
                if rl && ix(b) == ONE {
                    cells.push((x, mk(r, POS)));
                    cells.push((x, mk(r, NEG)));
                } else {
                    cells.push((x, b));
                }
            }
        }
        *pairs = cells;
    }
    if ll {
        // Both literals with the same right side: One.
        pairs.sort_unstable_by_key(|p| (p.1, p.0));
        pairs.dedup();
        let (mut w, mut i) = (0, 0);
        while i < pairs.len() {
            if i + 1 < pairs.len() && pairs[i + 1].1 == pairs[i].1 {
                debug_assert!(ix(pairs[i].0) == POS && ix(pairs[i + 1].0) == NEG);
                pairs[w] = (mk(l, ONE), pairs[i].1);
                i += 2;
            } else {
                pairs[w] = pairs[i];
                i += 1;
            }
            w += 1;
        }
        pairs.truncate(w);
    }
    pairs.sort_unstable();
    pairs.dedup();
    if rl {
        let (mut w, mut i) = (0, 0);
        while i < pairs.len() {
            if i + 1 < pairs.len() && pairs[i + 1].0 == pairs[i].0 {
                debug_assert!(ix(pairs[i].1) == POS && ix(pairs[i + 1].1) == NEG);
                pairs[w] = (pairs[i].0, mk(r, ONE));
                i += 2;
            } else {
                pairs[w] = pairs[i];
                i += 1;
            }
            w += 1;
        }
        pairs.truncate(w);
    }
}
