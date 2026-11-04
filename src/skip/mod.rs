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
pub(crate) mod embed;
pub(crate) mod eval;
pub(crate) mod negate;
pub(crate) mod none_of;
pub(crate) mod overlay;
pub(crate) mod pass;
pub(crate) mod project;
pub(crate) mod quantify;
pub(crate) mod sum;
#[cfg(test)]
mod tests;

/// Experiment: phase times of the skip conjunction (`TIDIDI_SKIP_TRACE`),
/// summed per phase until a call's trace line takes them. Phases 8 to 10
/// split the contraction: its top-down search, its patch and its collect.
pub(crate) mod trace {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    static PH: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("TIDIDI_SKIP_TRACE").is_some())
    }

    pub(crate) struct Clock(Option<Instant>);

    impl Clock {
        pub(crate) fn start() -> Clock {
            Clock(on().then(Instant::now))
        }

        /// Add the time since the last lap to phase `i`.
        pub(crate) fn lap(&mut self, i: usize) {
            if let Some(t) = &mut self.0 {
                let n = Instant::now();
                PH[i].fetch_add((n - *t).as_nanos() as u64, Relaxed);
                *t = n;
            }
        }
    }

    pub(crate) fn take() -> [u64; 11] {
        std::array::from_fn(|i| PH[i].swap(0, Relaxed))
    }

    static CN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

    /// Add `n` to count `i`, summed until a call's trace line takes them.
    pub(crate) fn count(i: usize, n: u64) {
        if on() {
            CN[i].fetch_add(n, Relaxed);
        }
    }

    pub(crate) fn take_counts() -> [u64; 4] {
        std::array::from_fn(|i| CN[i].swap(0, Relaxed))
    }
}

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
    /// Per node, `true` over its variables where [`build_top`] skips it to a
    /// leaf: that leaf's `One`; [`ZREF`] where it is a stored node.
    pub(crate) one: Vec<Ref>,
    /// Per internal node, whether a node there that is `true` over a leaf
    /// child of the shape is skipped ([`normalize`]): every node where the
    /// minimal run ([`skip_run`]) is 1, otherwise the nodes of the vtree
    /// with a leaf child in a run of at least that many, each one's internal
    /// child the next, read off the vtree before any subtree is summed out
    /// so that summing never changes it.
    pub(crate) skip: Vec<bool>,
    /// Per node, what it is: [`INNER`], [`VAR`], or a summed-out subtree,
    /// [`SUMMED_VAR`] or [`SUMMED`], which this shape reads as a leaf.
    pub(crate) kind: Vec<u8>,
    /// Per leaf, its labels: three on a variable, a summed-out subtree's
    /// value slots on one; 0 on an internal node.
    pub(crate) labels: Vec<u32>,
    /// Per node, whether a summed-out subtree is at or below it.
    pub(crate) weighted: Vec<bool>,
    /// Per node of the vtree, the root of the summed-out subtree it is in,
    /// or `NONE`.
    pub(crate) owner: Vec<u32>,
}

/// An internal node of the shape.
pub(crate) const INNER: u8 = 0;
/// A variable.
pub(crate) const VAR: u8 = 1;
/// A summed-out variable: its labels `One`, `Pos` and `Neg` weigh both
/// literals, the positive one and the negative one.
pub(crate) const SUMMED_VAR: u8 = 2;
/// A summed-out internal subtree: label `One` weighs every assignment to
/// its variables, and every other label is a value slot.
pub(crate) const SUMMED: u8 = 3;

impl Shape {
    pub(crate) fn of(vt: &Vtree) -> Shape {
        Self::summed(vt, &[])
    }

    /// The shape of `vt` where each subtree `(root, labels)` of `summed` is
    /// summed out: read as one leaf with `labels` labels, three on a
    /// variable. The roots are disjoint subtrees.
    pub(crate) fn summed(vt: &Vtree, summed: &[(u32, u32)]) -> Shape {
        let n = vt.num_nodes();
        let mut kids = vec![(NONE, NONE); n];
        let mut parent = vec![NONE; n];
        let mut var = vec![0u32; n];
        let mut kind = vec![INNER; n];
        let mut labels = vec![0u32; n];
        for i in 0..n {
            match vt.node(crate::vtree::VtreeIdx(i as u32)) {
                VtreeNode::Leaf { var: v, parent: p } => {
                    var[i] = v.0;
                    parent[i] = p.map_or(NONE, |p| p.0);
                    kind[i] = VAR;
                    labels[i] = 3;
                }
                VtreeNode::Internal { left, right, parent: p } => {
                    kids[i] = (left.0, right.0);
                    parent[i] = p.map_or(NONE, |p| p.0);
                }
            }
        }
        let skip = skip_flags(&kids, vt.root().0, skip_run());
        let mut owner = vec![NONE; n];
        for &(m, _) in summed {
            let mut stack = vec![m];
            while let Some(t) = stack.pop() {
                owner[t as usize] = m;
                let (l, r) = kids[t as usize];
                if l != NONE {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        for &(m, k) in summed {
            let m = m as usize;
            debug_assert!(kind[m] == INNER || k == 3, "a summed variable keeps its three labels");
            kind[m] = if kind[m] == VAR { SUMMED_VAR } else { SUMMED };
            kids[m] = (NONE, NONE);
            labels[m] = k;
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
            if !skip[t as usize] {
                continue;
            }
            if kids[r as usize].0 == NONE {
                chain_up[l as usize] = chain_up[t as usize];
            }
            if kids[l as usize].0 == NONE {
                chain_up[r as usize] = chain_up[t as usize];
            }
        }
        let mut weighted: Vec<bool> = kind.iter().map(|&k| k >= SUMMED_VAR).collect();
        let mut one = vec![ZREF; n];
        for &t in order.iter().rev() {
            let (l, r) = kids[t as usize];
            if l == NONE {
                one[t as usize] = mk(t, ONE);
                continue;
            }
            weighted[t as usize] = weighted[l as usize] || weighted[r as usize];
            // `build_top`'s descent.
            let leaf = |c: u32| kids[c as usize].0 == NONE;
            one[t as usize] = match skip[t as usize] {
                true if leaf(l) => one[r as usize],
                true if leaf(r) => one[l as usize],
                _ => ZREF,
            };
        }
        Shape { root, kids, parent, pre, size, topdown, bottomup, var, depth, looked_up, chain_up, one, skip, kind, labels, weighted, owner }
    }

    /// Whether a subtree is summed out somewhere.
    #[inline]
    pub(crate) fn is_weighted(&self) -> bool {
        self.weighted[self.root as usize]
    }

    /// Whether leaf `v`'s labels are literals, summed out or not.
    #[inline]
    pub(crate) fn literal_leaf(&self, v: u32) -> bool {
        matches!(self.kind[v as usize], VAR | SUMMED_VAR)
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
        if self.pairs.capacity() - self.pairs.len() < pairs.len() {
            lim.reserve(&mut self.pairs, pairs.len())?;
        }
        // One or two pairs, most nodes, without a call to copy them.
        match *pairs {
            [p] => self.pairs.push(p),
            [p, q] => {
                self.pairs.push(p);
                self.pairs.push(q);
            }
            _ => self.pairs.extend_from_slice(pairs),
        }
        lim.try_push(&mut self.start, self.pairs.len() as u32)?;
        Ok((self.start.len() - 2) as u32)
    }

    /// Append nodes `lo..hi` of `src` as they are, numbered on from this
    /// level's last.
    pub(crate) fn copy_run(&mut self, src: &SLevel, lo: usize, hi: usize, lim: &Limits) -> Result<(), OperationError> {
        if lo >= hi {
            return Ok(());
        }
        if self.start.is_empty() {
            lim.try_push(&mut self.start, 0)?;
        }
        let (a, b) = (src.start[lo], src.start[hi]);
        let shift = (self.pairs.len() as u32).wrapping_sub(a);
        lim.reserve(&mut self.pairs, (b - a) as usize)?;
        self.pairs.extend_from_slice(&src.pairs[a as usize..b as usize]);
        lim.reserve(&mut self.start, hi - lo)?;
        self.start.extend(src.start[lo + 1..=hi].iter().map(|&x| x.wrapping_add(shift)));
        Ok(())
    }

    /// The bytes the level's arenas hold.
    pub(crate) fn bytes(&self) -> u64 {
        (self.start.capacity() * 4 + self.pairs.capacity() * 16) as u64
    }
}

/// Per vtree node its [`SLevel`], stored only for levels that were written:
/// reading an unwritten level gives the empty one. A level may be shared
/// with another diagram's ([`LevelMap::share`]) and is copied when either
/// changes it.
#[derive(Clone, Debug)]
pub(crate) struct LevelMap {
    idx: Vec<u32>,
    lv: Vec<Arc<SLevel>>,
}

static EMPTY: SLevel = SLevel { start: Vec::new(), pairs: Vec::new() };

impl LevelMap {
    fn new(n: usize) -> LevelMap {
        LevelMap { idx: vec![NONE; n], lv: Vec::new() }
    }
    /// Every vtree node's level, in vtree index order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &SLevel> + '_ {
        self.idx.iter().map(|&i| if i == NONE { &EMPTY } else { &*self.lv[i as usize] })
    }

    /// Replace level `u` by `lvl`, without copying a shared level first.
    pub(crate) fn set(&mut self, u: usize, lvl: SLevel) {
        match self.idx[u] {
            NONE => {
                self.idx[u] = self.lv.len() as u32;
                self.lv.push(Arc::new(lvl));
            }
            i => self.lv[i as usize] = Arc::new(lvl),
        }
    }

    /// Level `u`, leaving it empty: moved out where this map held it alone.
    pub(crate) fn take(&mut self, u: usize) -> SLevel {
        match self.idx[u] {
            NONE => SLevel::default(),
            i => {
                let lvl = std::mem::take(&mut self.lv[i as usize]);
                Arc::try_unwrap(lvl).unwrap_or_else(|a| (*a).clone())
            }
        }
    }

    /// Level `u` of `src` as this map's level `u`, shared until either
    /// changes it, charged to `lim` as a copy is: each diagram releases its
    /// levels' bytes when it is discarded.
    pub(crate) fn share(&mut self, u: usize, src: &LevelMap, lim: &Limits) -> Result<(), OperationError> {
        let j = src.idx[u];
        let lvl = if j == NONE { Arc::new(SLevel::default()) } else { Arc::clone(&src.lv[j as usize]) };
        lim.charge_bytes(lvl.bytes())?;
        match self.idx[u] {
            NONE => {
                self.idx[u] = self.lv.len() as u32;
                self.lv.push(lvl);
            }
            i => self.lv[i as usize] = lvl,
        }
        Ok(())
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
            self.lv.push(Arc::new(SLevel::default()));
        }
        let lvl = &mut self.lv[i as usize];
        // No `Weak` to a level is ever made, so a count of one means this
        // map holds the level alone; the fence orders this access after a
        // drop of the last other holder on another thread.
        if Arc::strong_count(lvl) == 1 {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            // Safety: the level's only holder is `self`, borrowed mutably.
            return unsafe { &mut *Arc::as_ptr(lvl).cast_mut() };
        }
        Arc::make_mut(lvl)
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
                self.levels.set(t as usize, SLevel::default());
            }
            return Ok(());
        }
        // Per candidate level, the offset of its nodes in one array of all
        // of theirs (`NONE` on any other node of the vtree): which nodes are
        // live, then their new numbers. The per-node arrays are the thread's,
        // kept at `NONE` and `false` between calls ([`NodeArrays`]).
        let mut na = NodeArrays::take(sh.num_nodes());
        let res = self.collect_marked(&sh, &cand, &mut na, lim);
        na.put_back(&cand);
        res
    }

    fn collect_marked(&mut self, sh: &Shape, cand: &[u32], na: &mut NodeArrays, lim: &Limits) -> Result<(), OperationError> {
        let NodeArrays { off, moved, below, set_below } = na;
        let mut total = 0u32;
        for &t in cand {
            off[t as usize] = total;
            total += self.levels[t as usize].width() as u32;
        }
        let mut live = filled(lim, total as usize, 0u32)?;
        let slot = |off: &[u32], r: Ref| -> Option<usize> {
            let o = off[vt(r) as usize];
            (o != NONE).then(|| (o + ix(r)) as usize)
        };
        if let Some(k) = slot(off, self.out) {
            live[k] = 1;
        }
        let mut alive = 0u32;
        for &t in cand {
            let lvl = &self.levels[t as usize];
            let o = off[t as usize] as usize;
            let end = o + lvl.width();
            // A level's pairs name levels below it, later in `cand`, whose
            // slots follow its own.
            let (own, below) = live.split_at_mut(end);
            let own = &own[o..];
            let mut lo = 0usize;
            for (&flag, &hi) in own.iter().zip(&lvl.start[1..]) {
                let hi = hi as usize;
                if flag != 0 {
                    alive += 1;
                    for &(a, b) in &lvl.pairs[lo..hi] {
                        for x in [a, b] {
                            if let Some(q) = slot(off, x) {
                                below[q - end] = 1;
                            }
                        }
                    }
                }
                lo = hi;
            }
        }
        let mut active = std::mem::take(&mut self.active);
        if alive == total {
            // Every node is reached: the numbers stand.
            lim.discard(live);
            active.extend(cand.iter().copied());
            active.retain(|&t| self.levels[t as usize].width() > 0);
            active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
            active.dedup();
            self.active = active;
            return Ok(());
        }
        // The new numbers. `moved` keeps the offsets of the levels that lose
        // a node, the only ones whose references change, and `below` marks
        // the levels with one of them in their subtree, the only ones whose
        // pairs change.
        for &t in cand {
            let o = off[t as usize] as usize;
            let w = self.levels[t as usize].width();
            let mut next = 0u32;
            for x in &mut live[o..o + w] {
                *x = if *x != 0 {
                    next += 1;
                    next - 1
                } else {
                    NONE
                };
            }
            if next < w as u32 {
                moved[t as usize] = o as u32;
                let mut u = sh.parent[t as usize];
                while u != NONE && !below[u as usize] {
                    below[u as usize] = true;
                    set_below.push(u);
                    u = sh.parent[u as usize];
                }
            }
        }
        let fix = |live: &[u32], r: Ref| -> Ref {
            match slot(moved, r) {
                None => r,
                Some(q) => mk(vt(r), live[q]),
            }
        };
        active.retain(|&t| off[t as usize] == NONE && self.levels[t as usize].width() > 0);
        for &t in cand {
            if moved[t as usize] == NONE {
                // Every node kept: only the references below change. The
                // level is rewritten only from its first pair that names a
                // renumbered node, so a level shared with another diagram and
                // naming none stays shared.
                if below[t as usize] {
                    let pairs = &self.levels[t as usize].pairs;
                    if let Some(q0) = pairs.iter().position(|&(a, b)| fix(&live, a) != a || fix(&live, b) != b) {
                        for p in &mut self.levels[t as usize].pairs[q0..] {
                            *p = (fix(&live, p.0), fix(&live, p.1));
                        }
                    }
                }
                active.push(t);
                continue;
            }
            // Compacted in place: a kept node moves down to its new number.
            let o = off[t as usize] as usize;
            let lvl = &mut self.levels[t as usize];
            let (mut w, mut j, mut lo) = (0usize, 0usize, 0usize);
            for k in 0..lvl.width() {
                let hi = lvl.start[k + 1] as usize;
                if live[o + k] != NONE {
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
                // A level that lost most of its pairs (the conjunction copies
                // `P`'s level whole where it changes) hands the rest of its
                // arenas back, so that the result does not hold them.
                if lvl.pairs.capacity() > 2 * w {
                    let before = lvl.bytes();
                    lvl.pairs.shrink_to_fit();
                    lvl.start.shrink_to_fit();
                    lim.release_bytes(before.saturating_sub(lvl.bytes()));
                }
                active.push(t);
            }
        }
        active.sort_unstable_by_key(|&t| sh.pre[t as usize]);
        active.dedup();
        self.active = active;
        self.out = fix(&live, self.out);
        lim.discard(live);
        Ok(())
    }

    /// The model count over the vtree's variables.
    pub(crate) fn model_count(&self, lim: &Limits) -> Result<num_bigint::BigUint, OperationError> {
        self.evaluate(&eval::Count, lim)
    }
}

/// `true` over `t`'s variables as [`normalize`] writes it: skipped through
/// a leaf child, the left one first, and where both children are internal
/// the node `(true, true)`, which `node` makes at its level.
pub(crate) fn build_top(
    sh: &Shape,
    mut t: u32,
    node: &mut impl FnMut(u32, (Ref, Ref)) -> Result<Ref, OperationError>,
) -> Result<Ref, OperationError> {
    while !sh.is_leaf(t) {
        let (l, r) = sh.kids[t as usize];
        if sh.is_leaf(l) && sh.skip[t as usize] {
            t = r;
        } else if sh.is_leaf(r) && sh.skip[t as usize] {
            t = l;
        } else {
            let pair = (build_top(sh, l, node)?, build_top(sh, r, node)?);
            return node(t, pair);
        }
    }
    Ok(mk(t, ONE))
}

/// The least length of a run of nodes with a leaf child whose nodes skip
/// (`TIDIDI_SKIP_RUN`, default 1: every such node). An experiment's knob.
pub(crate) fn skip_run() -> u32 {
    static RUN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *RUN.get_or_init(|| std::env::var("TIDIDI_SKIP_RUN").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// [`Shape::skip`] over children `kids` (`(NONE, NONE)` on a leaf): a node
/// with a leaf child skips where its run, the chain of such nodes through
/// each one's internal child, has at least `run` nodes.
fn skip_flags(kids: &[(u32, u32)], root: u32, run: u32) -> Vec<bool> {
    let n = kids.len();
    let leaf = |v: u32| kids[v as usize].0 == NONE;
    let lop = |t: u32| !leaf(t) && (leaf(kids[t as usize].0) || leaf(kids[t as usize].1));
    // Every node, so that the shape's leaves decide: a summed-out subtree
    // read as a leaf is skipped through as the leaf it is read as.
    if run <= 1 {
        return vec![true; n];
    }
    // The run's next node: the internal child, where it has a leaf child too.
    let next = |t: u32| -> u32 {
        let (l, r) = kids[t as usize];
        let c = if !leaf(l) { l } else if !leaf(r) { r } else { return NONE };
        if lop(c) { c } else { NONE }
    };
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![root];
    while let Some(t) = stack.pop() {
        order.push(t);
        let (l, r) = kids[t as usize];
        if l != NONE {
            stack.push(l);
            stack.push(r);
        }
    }
    // Run lengths below each node, then each run's length on all its nodes.
    let mut down = vec![0u32; n];
    for &t in order.iter().rev() {
        if lop(t) {
            let c = next(t);
            down[t as usize] = 1 + if c == NONE { 0 } else { down[c as usize] };
        }
    }
    let mut len = vec![0u32; n];
    for &t in &order {
        if lop(t) {
            if len[t as usize] == 0 {
                len[t as usize] = down[t as usize];
            }
            let c = next(t);
            if c != NONE {
                len[c as usize] = len[t as usize];
            }
        }
    }
    (0..n).map(|i| len[i] >= run).collect()
}

/// Per vtree node arrays for [`SkipForm::collect_in`], kept on the thread
/// from one call to the next at `NONE` and `false`: a call resets only the
/// entries it set, so that its cost follows the levels it collects rather
/// than the vtree.
#[derive(Default)]
struct NodeArrays {
    off: Vec<u32>,
    moved: Vec<u32>,
    below: Vec<bool>,
    /// The entries of `below` set.
    set_below: Vec<u32>,
}

thread_local! {
    static NODE_ARRAYS: std::cell::Cell<NodeArrays> = std::cell::Cell::new(NodeArrays::default());
}

impl NodeArrays {
    fn take(n: usize) -> NodeArrays {
        let mut na = NODE_ARRAYS.with(|c| c.take());
        if na.off.len() < n {
            na.off.resize(n, NONE);
            na.moved.resize(n, NONE);
            na.below.resize(n, false);
        }
        na
    }

    /// Reset the entries a collection over `cand` set and keep the arrays.
    fn put_back(mut self, cand: &[u32]) {
        for &t in cand {
            self.off[t as usize] = NONE;
            self.moved[t as usize] = NONE;
        }
        for &u in &self.set_below {
            self.below[u as usize] = false;
        }
        self.set_below.clear();
        debug_assert!(self.off.iter().chain(&self.moved).all(|&x| x == NONE) && !self.below.contains(&true));
        NODE_ARRAYS.with(|c| c.set(self));
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
    if pairs.len() == 1 && sh.skip[u as usize] {
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
        if let [p, q] = pairs[..] {
            // Two pairs, the most a level against a leaf meets: one cell, or
            // two in their order.
            if other(&p) == other(&q) {
                let x = mk(leaf, cells_label(literal_cells(ix(lit(&p))) | literal_cells(ix(lit(&q)))));
                pairs[0] = if ll { (x, other(&p)) } else { (other(&p), x) };
                pairs.truncate(1);
            } else if if ll { q < p } else { other(&q) < other(&p) } {
                pairs.swap(0, 1);
            }
            return;
        }
        // Each pair packed in its first slot as one integer, the other side
        // above the literal's cells, and sorted by it unless the other sides
        // already come in order: then per other side the cells its pairs
        // cover.
        for p in pairs.iter_mut() {
            debug_assert!(other(p) >> 62 == 0, "a reference fits in 62 bits");
            *p = ((other(p) << 2) | u64::from(literal_cells(ix(lit(p)))), 0);
        }
        if !pairs.is_sorted_by_key(|p| p.0 >> 2) {
            pairs.sort_unstable_by_key(|p| p.0);
        }
        let (mut w, mut i) = (0, 0);
        while i < pairs.len() {
            let o = pairs[i].0 >> 2;
            let mut m = 0u8;
            while i < pairs.len() && pairs[i].0 >> 2 == o {
                m |= (pairs[i].0 & 3) as u8;
                i += 1;
            }
            let x = mk(leaf, cells_label(m));
            pairs[w] = if ll { (x, o) } else { (o, x) };
            w += 1;
        }
        pairs.truncate(w);
        if ll && w > 1 && !pairs.is_sorted_by_key(|p| ix(p.0)) {
            // By label, then other side: the pairs, in other side order,
            // copied label by label behind the input, which is then dropped.
            for l in [ONE, POS, NEG] {
                for i in 0..w {
                    if ix(pairs[i].0) == l {
                        pairs.push(pairs[i]);
                    }
                }
            }
            pairs.drain(..w);
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

/// Drop a charged vector of charged vectors, handing every charge back.
pub(crate) fn discard_nested<T>(lim: &Limits, mut v: Vec<Vec<T>>) {
    for x in v.iter_mut() {
        lim.discard(std::mem::take(x));
    }
    lim.discard(v);
}

/// [`normalize`] for a node at a level with a summed-out subtree below it,
/// whose pairs are a multiset of products to add: no pair goes as a repeat
/// of another. Against a literal leaf, a positive and a negative literal
/// with the same other side make one `One`; nothing else merges.
pub(crate) fn normalize_sum(sh: &Shape, u: u32, pairs: &mut Vec<(Ref, Ref)>) -> Result<(), Ref> {
    let (l, r) = sh.kids[u as usize];
    if pairs.len() > 1 {
        for (leaf, left) in [(l, true), (r, false)] {
            if sh.literal_leaf(leaf) {
                merge_literals(pairs, left);
            }
        }
        pairs.sort_unstable();
    }
    if pairs.len() == 1 && sh.skip[u as usize] {
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

/// Per other side, each positive literal of the leaf side (`left` or the
/// right) beside a negative one becomes `One` with it, in a multiset.
fn merge_literals(pairs: &mut Vec<(Ref, Ref)>, left: bool) {
    let lit = |p: &(Ref, Ref)| if left { p.0 } else { p.1 };
    let other = |p: &(Ref, Ref)| if left { p.1 } else { p.0 };
    pairs.sort_unstable_by_key(|p| (other(p), lit(p)));
    let (mut w, mut i) = (0, 0);
    while i < pairs.len() {
        let o = other(&pairs[i]);
        let leaf = vt(lit(&pairs[i]));
        let mut count = [0usize; 3];
        while i < pairs.len() && other(&pairs[i]) == o {
            count[ix(lit(&pairs[i])) as usize] += 1;
            i += 1;
        }
        let both = count[POS as usize].min(count[NEG as usize]);
        count[ONE as usize] += both;
        count[POS as usize] -= both;
        count[NEG as usize] -= both;
        for (label, &k) in count.iter().enumerate() {
            for _ in 0..k {
                let x = mk(leaf, label as u32);
                pairs[w] = if left { (x, o) } else { (o, x) };
                w += 1;
            }
        }
    }
    pairs.truncate(w);
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
