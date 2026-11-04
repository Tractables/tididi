//! A skip-form diagram on a larger vtree.
//!
//! The source levels go where the stored embed places them
//! ([`crate::restructure::EmbeddingPlan`]). Every other destination level a
//! reference crosses on its way down to its node has a side the function
//! ignores: where that side is a single leaf the reference skips through,
//! and elsewhere the level holds `true` on that side beside the reference.
//! `true` over a subtree of free variables is built as normalization writes
//! it ([`super::build_top`]). A source level whose image has its children
//! swapped is copied with every pair read the other way round.
//!
//! The nodes a level's pairs read on one side are pairwise disjoint. A leaf
//! side is the exception, where `One` sits beside the literals, so where a
//! source leaf `x` becomes a side of several variables (`x` and free ones)
//! `true` there is written as the two cells `x` and `¬x` wherever the level
//! reads a literal of `x`; where it reads none, as `true` (the two cells
//! are twins); a node whose cells meet a leaf side merges back as
//! normalization merges it. The images of distinct nodes stay distinct and
//! a node built twice is shared, so the copy of a canonical diagram is
//! canonical.
//!
//! A summed-out subtree goes to the image of its root, summed out there
//! ([`embed_summed`]): the variables the destination adds under it are
//! free, so its values are multiplied by `true` on each of them.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{build_top, discard_nested, filled, ix, mk, normalize, normalize_sum, vt, Ref, SLevel, Shape, SkipForm, NEG, NONE, ONE, POS, SUMMED, SUMMED_VAR, ZREF};
use crate::diagram::{LeafLabel, WeightStore, WeightValue};
use crate::limits::Limits;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::OperationError;

pub(super) struct Placer<'a> {
    d: &'a Shape,
    form: SkipForm,
    /// `true` over each destination node, once built.
    top: Vec<Ref>,
    /// Per destination node that holds one source leaf and free variables:
    /// that leaf where `true` is written as its two cells, else `NONE`.
    cells: Vec<u32>,
    /// The nodes of one or two pairs, by level and pairs.
    small: FxHashMap<(u32, [(Ref, Ref); 2]), Ref>,
    /// A reference from a destination child level to a node below it.
    lifted: FxHashMap<(u32, Ref), Ref>,
    /// A node's pairs while they are normalized.
    buf: Vec<(Ref, Ref)>,
    lim: &'a Limits,
}

impl<'a> Placer<'a> {
    /// A placer onto `d`, with `cells` as [`Placer::cells`] (`NONE` where no
    /// source leaf is widened).
    pub(super) fn new(d: &'a Arc<Shape>, cells: Vec<u32>, lim: &'a Limits) -> Result<Placer<'a>, OperationError> {
        Ok(Placer {
            d,
            form: SkipForm::zero(Arc::clone(d)),
            top: filled(lim, d.num_nodes(), ZREF)?,
            cells,
            small: FxHashMap::default(),
            lifted: FxHashMap::default(),
            buf: Vec::new(),
            lim,
        })
    }

    /// The nodes built, as a diagram with output `out`.
    pub(super) fn finish(self, out: Ref) -> Result<SkipForm, OperationError> {
        let lim = self.lim;
        let mut form = self.release();
        form.out = out;
        form.collect_pushed(lim)?;
        Ok(form)
    }

    /// The nodes built, uncollected, the placer's own tables released.
    pub(super) fn release(self) -> SkipForm {
        let Placer { form, top, cells, small, lifted, buf, lim, .. } = self;
        lim.discard(top);
        lim.discard(cells);
        lim.discard(small);
        lim.discard(lifted);
        lim.discard(buf);
        form
    }

    /// The node of `pairs` at internal `u` in its stored form
    /// ([`normalize`]): skipped where it is `true` on a leaf side, and a
    /// small one already built is reused.
    pub(super) fn node(&mut self, u: u32, pairs: &[(Ref, Ref)]) -> Result<Ref, OperationError> {
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        self.lim.reserve(&mut buf, pairs.len())?;
        buf.extend_from_slice(pairs);
        let made = match if self.d.weighted[u as usize] { normalize_sum(self.d, u, &mut buf) } else { normalize(self.d, u, &mut buf) } {
            Err(target) => Ok(target),
            Ok(()) => self.push(u, &buf),
        };
        self.buf = buf;
        made
    }

    fn push(&mut self, u: u32, pairs: &[(Ref, Ref)]) -> Result<Ref, OperationError> {
        if pairs.len() > 2 {
            return Ok(mk(u, self.form.push_node(u, pairs, self.lim)?));
        }
        let key = (u, [pairs[0], pairs.get(1).copied().unwrap_or((ZREF, ZREF))]);
        if let Some(&x) = self.small.get(&key) {
            return Ok(x);
        }
        let x = mk(u, self.form.push_node(u, pairs, self.lim)?);
        self.lim.reserve_map(&mut self.small, 1)?;
        self.small.insert(key, x);
        Ok(x)
    }

    /// `true` over the subtree of destination node `w`.
    pub(super) fn top(&mut self, w: u32) -> Result<Ref, OperationError> {
        if self.top[w as usize] == ZREF {
            let d = self.d;
            self.top[w as usize] = build_top(d, w, &mut |u, pair| self.node(u, &[pair]))?;
        }
        Ok(self.top[w as usize])
    }

    /// `true` as read from child level `c`: one reference, or the two
    /// cells of `c`'s source leaf.
    fn true_at(&mut self, c: u32) -> Result<([Ref; 2], usize), OperationError> {
        let x = self.cells[c as usize];
        if x == NONE {
            return Ok(([self.top(c)?, ZREF], 1));
        }
        Ok(([self.lift(c, mk(x, POS))?, self.lift(c, mk(x, NEG))?], 2))
    }

    /// [`Placer::lift`], without a lookup where `target` skips up to `c`:
    /// every level between has a single leaf on its other side and skips.
    #[inline]
    pub(super) fn reach(&mut self, c: u32, target: Ref) -> Result<Ref, OperationError> {
        let w = vt(target);
        if w == c || self.d.within(c, self.d.chain_up[w as usize]) {
            return Ok(target);
        }
        self.lift(c, target)
    }

    /// `target`, a destination reference to a node in `c`'s subtree, as a
    /// pair side at child level `c`: `true` on the other side of each level
    /// between.
    pub(super) fn lift(&mut self, c: u32, target: Ref) -> Result<Ref, OperationError> {
        if vt(target) == c {
            return Ok(target);
        }
        if let Some(&x) = self.lifted.get(&(c, target)) {
            return Ok(x);
        }
        let (mut below, mut w) = (target, vt(target));
        while w != c {
            // A run of levels that skip with a leaf child builds nothing.
            let up = self.d.chain_up[w as usize];
            if up != w {
                w = if self.d.within(c, up) { c } else { up };
                continue;
            }
            let p = self.d.parent[w as usize];
            let (l, r) = self.d.kids[p as usize];
            let other = if l == w { r } else { l };
            if !self.d.is_leaf(other) || !self.d.skip[p as usize] {
                let (t, k) = self.true_at(other)?;
                let mut pairs = [(ZREF, ZREF); 2];
                for (pair, &t) in pairs.iter_mut().zip(&t[..k]) {
                    *pair = if l == w { (below, t) } else { (t, below) };
                }
                below = self.node(p, &pairs[..k])?;
            }
            w = p;
        }
        self.lim.reserve_map(&mut self.lifted, 1)?;
        self.lifted.insert((c, target), below);
        Ok(below)
    }
}

/// `f` on the destination `d`, source node `s` placed at `img[s]` and, where
/// `mirrored` is not empty, read the other way round where `mirrored[s]`.
pub(crate) fn embed(
    f: &SkipForm,
    d: Arc<Shape>,
    img: &[VtreeIdx],
    mirrored: &[bool],
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    embed_copying(f, d, img, mirrored, true, lim)
}

/// [`embed`], with each level whose nodes keep their shapes copied where
/// `copy` is set: one whose leaf children stay leaves, not mirrored between
/// two leaves (the stored form merges the left leaf's literals first). Its
/// node `i` is then node `i` of its image, with the same pairs, each side
/// renamed and the pairs sorted where the renaming reorders them, so the
/// level is written in one pass without a node normalized or looked up, and
/// the result needs no collect. That takes every level with nodes skipping
/// where its image skips, and no subtree summed out. The other levels place
/// each node ([`Placer::node`]); a side is lifted, and `true` built, as
/// for a placed node.
pub(super) fn embed_copying(
    f: &SkipForm,
    d: Arc<Shape>,
    img: &[VtreeIdx],
    mirrored: &[bool],
    copy: bool,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let src = &*f.shape;
    let swapped = |s: u32| mirrored.get(s as usize).copied().unwrap_or(false);
    // The destination children of each source level, in its own order.
    let image_kids = |s: u32| {
        let (lc, rc) = d.kids[img[s as usize].idx()];
        if swapped(s) { (rc, lc) } else { (lc, rc) }
    };
    let mut cells = filled(lim, d.num_nodes(), NONE)?;
    for &s in &src.bottomup {
        let (sl, sr) = src.kids[s as usize];
        let (dl, dr) = image_kids(s);
        let pairs = &f.levels[s as usize].pairs;
        for (x, c, left) in [(sl, dl, true), (sr, dr, false)] {
            let literal = |&(a, b): &(Ref, Ref)| ix(if left { a } else { b }) != ONE;
            if src.literal_leaf(x) && !d.is_leaf(c) && pairs.iter().any(literal) {
                cells[c as usize] = img[x as usize].0;
            }
        }
    }
    let copy = copy
        && !src.is_weighted()
        && !d.is_weighted()
        && src.bottomup.iter().all(|&s| f.levels[s as usize].width() == 0 || src.skip[s as usize] == d.skip[img[s as usize].idx()]);
    let kept_shapes = |s: u32| {
        let ((sl, sr), (lc, rc)) = (src.kids[s as usize], image_kids(s));
        let leaf = |x: u32| src.is_leaf(x);
        (!leaf(sl) || d.is_leaf(lc)) && (!leaf(sr) || d.is_leaf(rc)) && !(swapped(s) && leaf(sl) && leaf(sr))
    };
    let mut cp = Placer::new(&d, cells, lim)?;
    // Whether every placed node is a new one, pushed last at its level, so
    // that no node is left unreached and the numbers stand.
    let mut fresh = true;
    let out = if f.out == ZREF {
        ZREF
    } else {
        // Where each source node went, as a destination reference: empty
        // where its level was copied, node `i` going to node `i` of the
        // image.
        let mut at: Vec<Vec<Ref>> = Vec::new();
        lim.try_resize(&mut at, src.num_nodes(), Vec::new())?;
        // A source reference as the pair sides it is at destination child
        // level `c`: one, or the two cells of `true`.
        let sides = |cp: &mut Placer<'_>, at: &Vec<Vec<Ref>>, c: u32, r: Ref| -> Result<([Ref; 2], usize), OperationError> {
            let v = vt(r);
            if src.is_leaf(v) && ix(r) == ONE {
                // `true` on the leaf, and so on everything under `c`.
                return cp.true_at(c);
            }
            let image = match &at[v as usize] {
                refs if !src.is_leaf(v) && !refs.is_empty() => refs[ix(r) as usize],
                _ => mk(img[v as usize].0, ix(r)),
            };
            Ok(([cp.reach(c, image)?, ZREF], 1))
        };
        let mut pairs = Vec::new();
        let mut gate = lim.gate();
        for &s in &src.bottomup {
            let lvl = &f.levels[s as usize];
            if lvl.width() == 0 {
                continue;
            }
            let u = img[s as usize].0;
            let (lc, rc) = image_kids(s);
            if copy && kept_shapes(s) {
                gate.poll(lvl.pairs.len() as u64)?;
                // Nothing else is placed at `u` before its own nodes: a lift
                // or a `true` lands under the level that reads it.
                debug_assert_eq!(cp.form.levels[u as usize].width(), 0);
                let mut start = Vec::new();
                lim.reserve_exact(&mut start, lvl.start.len())?;
                start.extend_from_slice(&lvl.start);
                let mut copied: Vec<(Ref, Ref)> = Vec::new();
                lim.reserve_exact(&mut copied, lvl.pairs.len())?;
                for &(a, b) in &lvl.pairs {
                    let ((xa, ka), (xb, kb)) = (sides(&mut cp, &at, lc, a)?, sides(&mut cp, &at, rc, b)?);
                    debug_assert!(ka == 1 && kb == 1, "no leaf child is widened");
                    copied.push(if swapped(s) { (xb[0], xa[0]) } else { (xa[0], xb[0]) });
                }
                let mut level = SLevel { start, pairs: copied };
                // A node's pairs ascend; a renaming that reorders the levels
                // a side names, or a mirror, reorders them.
                for i in 0..level.width() {
                    let (lo, hi) = (level.start[i] as usize, level.start[i + 1] as usize);
                    let node = &mut level.pairs[lo..hi];
                    if node.windows(2).any(|w| w[0] >= w[1]) {
                        node.sort_unstable();
                    }
                }
                cp.form.levels.set(u as usize, level);
                cp.form.active.push(u);
                continue;
            }
            let mut refs = Vec::new();
            lim.reserve_exact(&mut refs, lvl.width())?;
            for i in 0..lvl.width() {
                let node = lvl.node(i as u32);
                gate.poll(node.len() as u64)?;
                pairs.clear();
                for &(a, b) in node {
                    let ((xa, ka), (xb, kb)) = (sides(&mut cp, &at, lc, a)?, sides(&mut cp, &at, rc, b)?);
                    lim.reserve(&mut pairs, ka * kb)?;
                    for &x in &xa[..ka] {
                        for &y in &xb[..kb] {
                            pairs.push(if swapped(s) { (y, x) } else { (x, y) });
                        }
                    }
                }
                let next = cp.form.levels[u as usize].width() as u32;
                let made = cp.node(u, &pairs)?;
                fresh &= made == mk(u, next);
                refs.push(made);
            }
            at[s as usize] = refs;
        }
        gate.flush()?;
        let (out, k) = sides(&mut cp, &at, d.root, f.out)?;
        debug_assert_eq!(k, 1, "the root is no source leaf's side");
        lim.discard(pairs);
        discard_nested(lim, at);
        out[0]
    };
    if !(copy && fresh) {
        return cp.finish(out);
    }
    let mut form = cp.release();
    form.out = out;
    form.active.sort_unstable_by_key(|&t| d.pre[t as usize]);
    Ok(form)
}

/// [`embed`] for `f`, which sums subtrees out under `ws`, onto `dest` with
/// `store` as the result's weights: the image of each summed-out subtree is
/// summed out, with the values of its column times `true` on every variable
/// `dest` adds under it.
///
/// # Errors
///
/// [`OperationError::MarginalLevel`] at a summed-out subtree whose image
/// holds a variable of the rest of `f`.
pub(crate) fn embed_summed(
    f: &SkipForm,
    ws: &WeightStore,
    dest: &Vtree,
    img: &[VtreeIdx],
    mirrored: &[bool],
    mut store: WeightStore,
    lim: &Limits,
) -> Result<(SkipForm, WeightStore), OperationError> {
    let src = &*f.shape;
    let full = Shape::of(dest);
    // Per destination leaf, the source subtree its variable comes from: the
    // summed-out one it is in, or `NONE`.
    let mut from = filled(lim, full.num_nodes(), NONE)?;
    let mut placed = filled(lim, full.num_nodes(), false)?;
    for (x, d) in img.iter().enumerate() {
        if src.kids[x] == (NONE, NONE) && src.kind[x] != SUMMED {
            let i = d.idx();
            placed[i] = true;
            from[i] = src.owner[x];
        }
    }
    let mut leaves: Vec<(u32, u32)> = Vec::new();
    for m in 0..src.num_nodes() as u32 {
        let k = src.kind[m as usize];
        if k != SUMMED && k != SUMMED_VAR {
            continue;
        }
        let d = img[m as usize];
        let mut extra: Option<WeightValue> = None;
        for y in dest.subtree(d) {
            if !full.is_leaf(y.0) {
                continue;
            }
            if placed[y.idx()] && from[y.idx()] != m {
                return Err(OperationError::MarginalLevel(VtreeIdx(m)));
            }
            if !placed[y.idx()] {
                let w = store.leaf_val(VarId(full.var[y.idx()]), LeafLabel::One);
                extra = Some(match extra {
                    None => w,
                    Some(e) => e.mul(&w),
                });
            }
        }
        if k == SUMMED {
            let col = ws.level(m as usize).expect("a summed-out subtree has its column");
            let vals: Vec<WeightValue> = match &extra {
                None => col.to_vec(),
                Some(e) => col.iter().map(|v| v.mul(e)).collect(),
            };
            store.set_level(d.idx(), vals);
        }
        lim.try_push(&mut leaves, (d.0, src.labels[m as usize]))?;
    }
    lim.discard(from);
    lim.discard(placed);
    let shape = Arc::new(Shape::summed(dest, &leaves));
    lim.discard(leaves);
    let out = embed(f, shape, img, mirrored, lim)?;
    Ok((out, store))
}
