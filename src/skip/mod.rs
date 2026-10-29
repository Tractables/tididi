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

pub(crate) mod api;
pub(crate) mod conjoin;
pub(crate) mod contract;
pub(crate) mod convert;
pub(crate) mod negate;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::limits::Limits;
use crate::vtree::{Vtree, VtreeNode};
use crate::OperationError;

/// A resolved reference: the vtree node in bits 32 to 61, the index in the
/// low word. [`ZREF`] is the constant false.
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
    /// Whether a node or one of its ancestors is the left child of a level
    /// whose children are both internal: a conjunction looks the products
    /// read there up by both references.
    pub(crate) looked_up: Vec<bool>,
    /// The highest ancestor a node reaches through parents whose other
    /// child is a leaf (the node itself where its sibling is internal or it
    /// is the root): a reference read there skips every level between.
    pub(crate) chain_up: Vec<u32>,
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
        let mut looked_up = vec![false; n];
        for &t in &topdown {
            let (l, r) = kids[t as usize];
            let both = kids[l as usize].0 != NONE && kids[r as usize].0 != NONE;
            looked_up[l as usize] = looked_up[t as usize] || both;
            looked_up[r as usize] = looked_up[t as usize];
        }
        let mut chain_up: Vec<u32> = (0..n as u32).collect();
        for &t in &topdown {
            let (l, r) = kids[t as usize];
            if kids[r as usize].0 == NONE {
                chain_up[l as usize] = chain_up[t as usize];
            }
            if kids[l as usize].0 == NONE {
                chain_up[r as usize] = chain_up[t as usize];
            }
        }
        Shape { root, kids, parent, pre, size, topdown, bottomup, var, depth, looked_up, chain_up }
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
    /// Append a node and return its index; the growth is charged to `lim`.
    pub(crate) fn push(&mut self, pairs: &[(Ref, Ref)], lim: &Limits) -> Result<u32, OperationError> {
        if self.start.is_empty() {
            lim.try_push(&mut self.start, 0)?;
        }
        lim.reserve(&mut self.pairs, pairs.len())?;
        self.pairs.extend_from_slice(pairs);
        lim.try_push(&mut self.start, self.pairs.len() as u32)?;
        Ok((self.start.len() - 2) as u32)
    }

    /// A copy of the level, charged to `lim`.
    pub(crate) fn copy(&self, lim: &Limits) -> Result<SLevel, OperationError> {
        let mut c = SLevel::default();
        lim.reserve_exact(&mut c.start, self.start.len())?;
        c.start.extend_from_slice(&self.start);
        lim.reserve_exact(&mut c.pairs, self.pairs.len())?;
        c.pairs.extend_from_slice(&self.pairs);
        Ok(c)
    }

    /// The bytes the level's arenas hold.
    pub(crate) fn bytes(&self) -> u64 {
        (self.start.capacity() * 4 + self.pairs.capacity() * 16) as u64
    }
}

/// Per vtree node its [`SLevel`], stored only for levels that were written:
/// reading an unwritten level gives the empty one.
#[derive(Clone, Debug)]
pub(crate) struct LevelMap {
    idx: Vec<u32>,
    lv: Vec<SLevel>,
}

static EMPTY: SLevel = SLevel { start: Vec::new(), pairs: Vec::new() };

impl LevelMap {
    fn new(n: usize) -> LevelMap {
        LevelMap { idx: vec![NONE; n], lv: Vec::new() }
    }
    /// Every vtree node's level, in vtree index order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &SLevel> + '_ {
        self.idx.iter().map(|&i| if i == NONE { &EMPTY } else { &self.lv[i as usize] })
    }
}

impl std::ops::Index<usize> for LevelMap {
    type Output = SLevel;
    #[inline]
    fn index(&self, u: usize) -> &SLevel {
        let i = self.idx[u];
        if i == NONE { &EMPTY } else { &self.lv[i as usize] }
    }
}

impl std::ops::IndexMut<usize> for LevelMap {
    #[inline]
    fn index_mut(&mut self, u: usize) -> &mut SLevel {
        let mut i = self.idx[u];
        if i == NONE {
            i = self.lv.len() as u32;
            self.idx[u] = i;
            self.lv.push(SLevel::default());
        }
        &mut self.lv[i as usize]
    }
}

/// A diagram in skip form over [`Shape`]: every reference resolved.
#[derive(Clone, Debug)]
pub(crate) struct SkipForm {
    pub(crate) shape: Arc<Shape>,
    pub(crate) levels: LevelMap,
    /// The levels that hold nodes, parents first (by preorder), once
    /// [`SkipForm::collect`] has run.
    pub(crate) active: Vec<u32>,
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
        SkipForm { shape, levels: LevelMap::new(n), active: Vec::new(), out: ZREF }
    }

    /// Append a node at level `u`, noting the level among those that hold
    /// nodes ([`SkipForm::collect_in`] orders them), and return its index.
    pub(crate) fn push_node(&mut self, u: u32, pairs: &[(Ref, Ref)], lim: &Limits) -> Result<u32, OperationError> {
        if self.levels[u as usize].width() == 0 {
            self.active.push(u);
        }
        self.levels[u as usize].push(pairs, lim)
    }

    /// [`SkipForm::collect_in`] over the levels [`SkipForm::push_node`]
    /// noted.
    pub(crate) fn collect_pushed(&mut self, lim: &Limits) -> Result<(), OperationError> {
        let cand = std::mem::take(&mut self.active);
        self.collect_in(cand, lim)
    }

    /// The bytes the levels' arenas hold.
    pub(crate) fn bytes(&self) -> u64 {
        self.active.iter().map(|&t| self.levels[t as usize].bytes()).sum()
    }

    /// Drop the diagram, handing the charge for its levels back to `lim`.
    pub(crate) fn discard(self, lim: &Limits) {
        lim.release_bytes(self.bytes());
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
        for &t in &self.active {
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
    pub(crate) fn collect(&mut self, lim: &Limits) -> Result<(), OperationError> {
        let all = self.shape.topdown.clone();
        self.collect_in(all, lim)
    }

    /// [`SkipForm::collect`] where only the upward-closed levels in `cand` may
    /// hold unreachable or renumbered nodes; active levels outside `cand` stay
    /// as they are.
    pub(crate) fn collect_in(&mut self, mut cand: Vec<u32>, lim: &Limits) -> Result<(), OperationError> {
        let sh = Arc::clone(&self.shape);
        cand.retain(|&t| self.levels[t as usize].width() > 0);
        cand.sort_unstable_by_key(|&t| sh.pre[t as usize]);
        cand.dedup();
        if self.out == ZREF {
            let mut drop_all = std::mem::take(&mut self.active);
            drop_all.extend(cand);
            drop_all.sort_unstable();
            drop_all.dedup();
            for &t in &drop_all {
                lim.release_bytes(self.levels[t as usize].bytes());
                self.levels[t as usize] = SLevel::default();
            }
            return Ok(());
        }
        // Per candidate level (by its position in `cand`): which nodes are
        // live, then their new numbers.
        let mut pos = vec![NONE; sh.num_nodes()];
        for (i, &t) in cand.iter().enumerate() {
            pos[t as usize] = i as u32;
        }
        let mut live: Vec<Vec<u32>> = Vec::with_capacity(cand.len());
        for &t in &cand {
            live.push(filled(lim, self.levels[t as usize].width(), 0)?);
        }
        let mark = |live: &mut Vec<Vec<u32>>, r: Ref| {
            let p = pos[vt(r) as usize];
            if p != NONE && !sh.is_leaf(vt(r)) {
                live[p as usize][ix(r) as usize] = 1;
            }
        };
        mark(&mut live, self.out);
        for (i, &t) in cand.iter().enumerate() {
            let lvl = &self.levels[t as usize];
            for k in 0..lvl.width() {
                if live[i][k] != 0 {
                    for &(a, b) in lvl.node(k as u32) {
                        mark(&mut live, a);
                        mark(&mut live, b);
                    }
                }
            }
        }
        for v in live.iter_mut() {
            let mut next = 0u32;
            for x in v.iter_mut() {
                *x = if *x != 0 {
                    next += 1;
                    next - 1
                } else {
                    NONE
                };
            }
        }
        let fix = |live: &Vec<Vec<u32>>, r: Ref| -> Ref {
            let p = pos[vt(r) as usize];
            if p == NONE || sh.is_leaf(vt(r)) { r } else { mk(vt(r), live[p as usize][ix(r) as usize]) }
        };
        let mut active = Vec::with_capacity(cand.len() + self.active.len());
        for &t in &self.active {
            if pos[t as usize] == NONE && self.levels[t as usize].width() > 0 {
                active.push(t);
            }
        }
        for (i, &t) in cand.iter().enumerate() {
            // Compacted in place: a kept node moves down to its new number.
            let lvl = &mut self.levels[t as usize];
            let (mut w, mut j, mut lo) = (0usize, 0usize, 0usize);
            for k in 0..lvl.width() {
                let hi = lvl.start[k + 1] as usize;
                if live[i][k] != NONE {
                    for q in lo..hi {
                        let (a, b) = lvl.pairs[q];
                        lvl.pairs[w] = (fix(&live, a), fix(&live, b));
                        w += 1;
                    }
                    j += 1;
                    lvl.start[j] = w as u32;
                }
                lo = hi;
            }
            if j == 0 {
                lim.release_bytes(lvl.bytes());
                *lvl = SLevel::default();
            } else {
                lvl.pairs.truncate(w);
                lvl.start.truncate(j + 1);
                active.push(t);
            }
        }
        active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
        active.dedup();
        self.active = active;
        self.out = fix(&live, self.out);
        for v in live {
            lim.discard(v);
        }
        Ok(())
    }

    /// The model count over the vtree's variables.
    pub(crate) fn model_count(&self, lim: &Limits) -> Result<num_bigint::BigUint, OperationError> {
        use num_bigint::BigUint;
        let sh = &self.shape;
        if self.out == ZREF {
            return Ok(BigUint::from(0u32));
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
        for (t, v) in nv.iter_mut().enumerate() {
            if sh.is_leaf(t as u32) {
                *v = 1;
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
            let mut out = Vec::new();
            lim.reserve_exact(&mut out, lvl.width())?;
            for i in 0..lvl.width() {
                let mut s = BigUint::from(0u32);
                for &(a, b) in lvl.node(i as u32) {
                    s += value(&val, l, a) * value(&val, r, b);
                }
                out.push(s);
            }
            val[t as usize] = out;
        }
        let count = value(&val, sh.root, self.out);
        for v in val {
            lim.discard(v);
        }
        Ok(count)
    }
}

/// `vec![x; n]` with its allocation charged to `lim`.
pub(crate) fn filled<T: Clone>(lim: &Limits, n: usize, x: T) -> Result<Vec<T>, OperationError> {
    let mut v = Vec::new();
    lim.try_resize(&mut v, n, x)?;
    Ok(v)
}

/// Normalize the pairs of a node at internal `u` (sorted, sides resolved,
/// no zero side) and resolve it: `Err(target)` when the node is `true` on
/// one single-variable side, `Ok(())` with `pairs` in their stored form
/// otherwise.
///
/// Against a leaf child the stored form names `One` where the node holds
/// both literals with the same other side; the left leaf is merged first,
/// then the right, so the form is a function of the node's assignments.
#[inline]
pub(crate) fn normalize(sh: &Shape, u: u32, pairs: &mut Vec<(Ref, Ref)>) -> Result<(), Ref> {
    // One pair is already in its stored form.
    if pairs.len() > 1 {
        merge_cells(sh, u, pairs);
    }
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
    if ll != rl {
        // One leaf side: per other side, the literals its pairs cover.
        let lit = |p: &(Ref, Ref)| if ll { p.0 } else { p.1 };
        let other = |p: &(Ref, Ref)| if ll { p.1 } else { p.0 };
        let leaf = if ll { l } else { r };
        if pairs.len() > 1 {
            pairs.sort_unstable_by_key(|p| (other(p), lit(p)));
        }
        let (mut w, mut i) = (0, 0);
        while i < pairs.len() {
            let o = other(&pairs[i]);
            let mut m = 0u8;
            while i < pairs.len() && other(&pairs[i]) == o {
                m |= literal_cells(ix(lit(&pairs[i])));
                i += 1;
            }
            let x = mk(leaf, cells_label(m));
            pairs[w] = if ll { (x, o) } else { (o, x) };
            w += 1;
        }
        pairs.truncate(w);
        if ll && w > 1 {
            pairs.sort_unstable();
        }
        return;
    }
    // Cells: every One against a leaf child as both literals.
    if (ll && pairs.iter().any(|p| ix(p.0) == ONE)) || (rl && pairs.iter().any(|p| ix(p.1) == ONE)) {
        // Expanded behind the input, which is then dropped from the front.
        let m = pairs.len();
        for i in 0..m {
            let (a, b) = pairs[i];
            let lefts: &[Ref] = if ll && ix(a) == ONE { &[mk(l, POS), mk(l, NEG)] } else { &[a] };
            for &x in lefts {
                if rl && ix(b) == ONE {
                    pairs.push((x, mk(r, POS)));
                    pairs.push((x, mk(r, NEG)));
                } else {
                    pairs.push((x, b));
                }
            }
        }
        pairs.drain(..m);
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

/// The literal cells a leaf label covers: bit 0 positive, bit 1 negative.
#[inline]
pub(crate) fn literal_cells(label: u32) -> u8 {
    match label {
        ONE => 3,
        POS => 1,
        _ => 2,
    }
}

/// The leaf label of a nonempty set of literal cells.
#[inline]
pub(crate) fn cells_label(cells: u8) -> u32 {
    match cells {
        3 => ONE,
        1 => POS,
        _ => NEG,
    }
}
