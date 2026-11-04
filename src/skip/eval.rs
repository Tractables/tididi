//! Evaluation in an algebra, a satisfying assignment, the support and
//! equality, in skip form.
//!
//! A reference from child level `c` to a node at `v` below it skips every
//! level between, and the levels crossed each leave the single variable of
//! their other child free: the reference's value is the node's times
//! `true` on each of those variables, as a stored diagram's
//! `true_x ∧ D` nodes would give it.

use rustc_hash::FxHashMap;

use super::{discard_nested, filled, ix, mk, vt, Ref, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::diagram::{EvalAlgebra, LeafLabel, Literal};
use crate::limits::Limits;
use crate::vtree::VarId;
use crate::OperationError;

/// The steps of a skip's product read at once ([`Valuer`]): a skip down a
/// chain of `n` levels multiplies about `BLOCK + n / BLOCK` values.
const BLOCK: u32 = 128;

pub(crate) fn label(i: u32) -> LeafLabel {
    match i {
        ONE => LeafLabel::One,
        POS => LeafLabel::Pos,
        NEG => LeafLabel::Neg,
        _ => unreachable!("a leaf reference is One, Pos or Neg"),
    }
}

/// The products of `true` over the levels a skip crosses, from the leaf
/// values: what a reference read above its node multiplies the node's value
/// by. They depend on the shape and the leaf values only, so that two
/// diagrams over one shape share them.
pub(crate) struct Crossings<V> {
    /// Per node, `true` on its sibling where the sibling is a leaf: what a
    /// skip through the parent leaves free.
    free: Vec<Option<V>>,
    /// Per node `w` with `free[w]`, the product of `free` over `w` and its
    /// ancestors below `bend[w]`: the steps up to the next depth that is a
    /// multiple of [`BLOCK`], or to the first ancestor without `free`.
    block: Vec<Option<V>>,
    bend: Vec<u32>,
    /// The product over the levels a skip from `c` to `v` crosses.
    crossed: FxHashMap<(u32, u32), V>,
}

impl<V: Clone> Crossings<V> {
    pub(crate) fn new<S: EvalAlgebra<Value = V>>(
        sh: &Shape,
        algebra: &S,
        leaf: &impl Fn(u32, u32) -> V,
        lim: &Limits,
    ) -> Result<Self, OperationError> {
        let n = sh.num_nodes();
        let mut free = Vec::new();
        lim.reserve_exact(&mut free, n)?;
        for u in 0..n {
            let p = sh.parent[u];
            free.push(if p == NONE || sh.kids[p as usize].0 == NONE {
                None
            } else {
                let (l, r) = sh.kids[p as usize];
                let sib = if l as usize == u { r } else { l };
                sh.is_leaf(sib).then(|| leaf(sib, ONE))
            });
        }
        let (mut block, mut bend): (Vec<Option<V>>, Vec<u32>) = (Vec::new(), Vec::new());
        lim.try_resize(&mut block, n, None)?;
        lim.try_resize(&mut bend, n, NONE)?;
        for &t in &sh.topdown {
            let (l, r) = sh.kids[t as usize];
            for ch in [l, r] {
                let Some(x) = &free[ch as usize] else { continue };
                let up = !sh.depth[t as usize].is_multiple_of(BLOCK) && block[t as usize].is_some();
                let (b, e) = if up {
                    (algebra.mul(x, block[t as usize].as_ref().expect("checked above")), bend[t as usize])
                } else {
                    (x.clone(), t)
                };
                block[ch as usize] = Some(b);
                bend[ch as usize] = e;
            }
        }
        Ok(Crossings { free, block, bend, crossed: FxHashMap::default() })
    }

    /// Crossings that hold nothing, in place of a table that is lent out.
    pub(crate) fn empty() -> Self {
        Crossings { free: Vec::new(), block: Vec::new(), bend: Vec::new(), crossed: FxHashMap::default() }
    }

    /// The product over the levels a skip from `c` to `v` crosses: known
    /// for the skip to `v`'s parent or from `c`'s child toward `v`, as the
    /// skips of one level's nodes down a chain mostly are, or else read in
    /// blocks of up to [`BLOCK`] steps.
    pub(crate) fn crossed<S: EvalAlgebra<Value = V>>(&mut self, sh: &Shape, algebra: &S, c: u32, v: u32) -> V {
        if let Some(x) = self.crossed.get(&(c, v)) {
            return x.clone();
        }
        let free = &self.free;
        let step = |w: u32| free[w as usize].as_ref().expect("a skip passes a single variable");
        let p = sh.parent[v as usize];
        let k = sh.toward(c, v);
        let x = if let Some(y) = (p != c).then(|| self.crossed.get(&(c, p))).flatten() {
            algebra.mul(step(v), y)
        } else if let Some(y) = (k != v).then(|| self.crossed.get(&(k, v))).flatten() {
            algebra.mul(y, step(k))
        } else {
            let mut w = v;
            let mut acc: Option<V> = None;
            while w != c {
                let e = self.bend[w as usize];
                let (x, next) = if e != NONE && sh.depth[e as usize] >= sh.depth[c as usize] {
                    (self.block[w as usize].as_ref().expect("a block where it ends"), e)
                } else {
                    (step(w), sh.parent[w as usize])
                };
                acc = Some(match acc {
                    None => x.clone(),
                    Some(a) => algebra.mul(&a, x),
                });
                w = next;
            }
            acc.expect("a skip crosses a level")
        };
        self.crossed.insert((c, v), x.clone());
        x
    }

    pub(crate) fn discard(self, lim: &Limits) {
        lim.discard(self.free);
        lim.discard(self.block);
        lim.discard(self.bend);
    }
}

/// The values of a diagram's references in an algebra, from leaf values
/// `leaf(leaf, label)`; a level's nodes once [`Valuer::fill`] has run on it.
pub(crate) struct Valuer<'a, S: EvalAlgebra, L> {
    sh: &'a Shape,
    algebra: &'a S,
    leaf: L,
    cross: Crossings<S::Value>,
    /// Per level, its nodes' values.
    val: Vec<Vec<S::Value>>,
}

impl<'a, S: EvalAlgebra, L: Fn(u32, u32) -> S::Value> Valuer<'a, S, L> {
    pub(crate) fn new(sh: &'a Shape, algebra: &'a S, leaf: L, lim: &Limits) -> Result<Self, OperationError> {
        let mut cross = Crossings::new(sh, algebra, &leaf, lim)?;
        let valuer = Self::with_crossings(sh, algebra, leaf, &mut cross, lim);
        if valuer.is_err() {
            cross.discard(lim);
        }
        valuer
    }

    /// A valuer that reads the skips' products off `cross`, crossings taken
    /// over `sh` with the same leaf values: another diagram's over the same
    /// shape.
    /// The crossings are taken out of `cross` only once the valuer is built,
    /// so a refusal leaves them where they were.
    pub(crate) fn with_crossings(
        sh: &'a Shape,
        algebra: &'a S,
        leaf: L,
        cross: &mut Crossings<S::Value>,
        lim: &Limits,
    ) -> Result<Self, OperationError> {
        let mut val = Vec::new();
        lim.try_resize(&mut val, sh.num_nodes(), Vec::new())?;
        let cross = std::mem::replace(cross, Crossings::empty());
        Ok(Valuer { sh, algebra, leaf, cross, val })
    }

    /// The value of `r`, read at child level `c` at or above its node.
    pub(crate) fn value(&mut self, c: u32, r: Ref) -> S::Value {
        let v = vt(r);
        let base = if self.sh.is_leaf(v) { (self.leaf)(v, ix(r)) } else { self.val[v as usize][ix(r) as usize].clone() };
        if v == c {
            return base;
        }
        let skip = self.cross.crossed(self.sh, self.algebra, c, v);
        self.algebra.mul(&base, &skip)
    }

    /// The values of `form`'s nodes at internal level `t`, from those below.
    pub(crate) fn fill(&mut self, form: &SkipForm, t: u32, gate: &mut crate::limits::PollGate<'_>, lim: &Limits) -> Result<(), OperationError> {
        let (l, r) = self.sh.kids[t as usize];
        let lvl = &form.levels[t as usize];
        let mut out = Vec::new();
        lim.reserve_exact(&mut out, lvl.width())?;
        for i in 0..lvl.width() {
            let pairs = lvl.node(i as u32);
            gate.poll(pairs.len() as u64)?;
            let mut s = self.algebra.zero();
            for &(a, b) in pairs {
                let (x, y) = (self.value(l, a), self.value(r, b));
                self.algebra.mul_add(&mut s, &x, &y);
            }
            out.push(s);
        }
        self.val[t as usize] = out;
        Ok(())
    }

    /// The crossings and every level's values, for a caller that keeps them.
    pub(crate) fn into_parts(self) -> (Crossings<S::Value>, Vec<Vec<S::Value>>) {
        (self.cross, self.val)
    }

    pub(crate) fn discard(self, lim: &Limits) {
        self.cross.discard(lim);
        discard_nested(lim, self.val);
    }
}

impl SkipForm {
    /// The value of the function in `algebra`, over the vtree's variables.
    pub(crate) fn evaluate<S: EvalAlgebra>(&self, algebra: &S, lim: &Limits) -> Result<S::Value, OperationError> {
        let sh = &*self.shape;
        self.evaluate_with(algebra, |v, k| algebra.leaf(VarId(sh.var[v as usize]), label(k)), lim)
    }

    /// [`SkipForm::evaluate`] with the value of label `k` of leaf `v` given
    /// by `leaf(v, k)`.
    pub(crate) fn evaluate_with<S: EvalAlgebra>(
        &self,
        algebra: &S,
        leaf: impl Fn(u32, u32) -> S::Value,
        lim: &Limits,
    ) -> Result<S::Value, OperationError> {
        let sh = &*self.shape;
        if self.out == ZREF {
            return Ok(algebra.zero());
        }
        let mut values = Valuer::new(sh, algebra, leaf, lim)?;
        let mut gate = lim.gate();
        for &t in &sh.bottomup {
            values.fill(self, t, &mut gate, lim)?;
        }
        gate.flush()?;
        let res = values.value(sh.root, self.out);
        values.discard(lim);
        Ok(res)
    }

    /// A model of the function, a literal for every variable of the vtree:
    /// a variable a path leaves free, `true` on a leaf or skipped, is false.
    /// `None` for the constant false.
    pub(crate) fn satisfying_assignment(&self, lim: &Limits) -> Result<Option<Vec<Literal>>, OperationError> {
        let sh = &*self.shape;
        if self.out == ZREF {
            return Ok(None);
        }
        let n = sh.num_nodes();
        let mut value = filled(lim, n, false)?;
        let mut pending: Vec<Ref> = Vec::new();
        lim.try_push(&mut pending, self.out)?;
        let mut gate = lim.gate();
        while let Some(r) = pending.pop() {
            gate.poll(1)?;
            let v = vt(r);
            if sh.is_leaf(v) {
                value[v as usize] = ix(r) == POS;
            } else {
                let (a, b) = self.levels[v as usize].node(ix(r))[0];
                lim.try_push(&mut pending, a)?;
                lim.try_push(&mut pending, b)?;
            }
        }
        gate.flush()?;
        lim.discard(pending);
        let mut model = Vec::new();
        for (u, &x) in value.iter().enumerate() {
            if sh.is_leaf(u as u32) {
                lim.try_push(&mut model, Literal::new(VarId(sh.var[u]), x))?;
            }
        }
        model.sort_unstable_by_key(|lit| lit.var);
        lim.discard(value);
        Ok(Some(model))
    }
}

impl SkipForm {
    /// The variables the function depends on, sorted: the leaves a pair
    /// names with a literal label. A canonical form names a literal only
    /// where the function depends on its variable, and holds only nodes
    /// reachable from `out`.
    pub(crate) fn support(&self, lim: &Limits) -> Result<Vec<VarId>, OperationError> {
        let sh = &*self.shape;
        let mut on = filled(lim, sh.num_nodes(), false)?;
        let mut note = |r: Ref| {
            if r != ZREF && sh.is_leaf(vt(r)) && ix(r) != ONE {
                on[vt(r) as usize] = true;
            }
        };
        note(self.out);
        let mut gate = lim.gate();
        for &t in &self.active {
            let pairs = &self.levels[t as usize].pairs;
            gate.poll(pairs.len() as u64)?;
            for &(a, b) in pairs {
                note(a);
                note(b);
            }
        }
        gate.flush()?;
        let mut vars = Vec::new();
        for (u, &x) in on.iter().enumerate() {
            if x {
                lim.try_push(&mut vars, VarId(sh.var[u]))?;
            }
        }
        vars.sort_unstable();
        lim.discard(on);
        Ok(vars)
    }

    /// Whether `self` and `other`, canonical over one shape, denote the
    /// same function: equal up to the numbering of each level's nodes and
    /// the order of a node's pairs. Bottom-up, each node of `self` is
    /// matched with the node of `other` whose pairs are its own once its
    /// children are renamed.
    pub(crate) fn same(&self, other: &SkipForm, lim: &Limits) -> Result<bool, OperationError> {
        use std::hash::{Hash, Hasher};
        let sh = &*self.shape;
        if self.active.len() != other.active.len() || (self.out == ZREF) != (other.out == ZREF) {
            return Ok(false);
        }
        // Where each level's nodes start in `to`, the match of every node.
        let mut base = filled(lim, sh.num_nodes(), 0u32)?;
        let mut total = 0usize;
        for &t in &self.active {
            base[t as usize] = total as u32;
            total += self.levels[t as usize].width();
        }
        let mut to = filled(lim, total, 0u32)?;
        let mut mine: Vec<(Ref, Ref)> = Vec::new();
        let mut theirs: Vec<(Ref, Ref)> = Vec::new();
        let mut keys: Vec<(u64, u32)> = Vec::new();
        let hash = |p: &[(Ref, Ref)]| {
            let mut h = rustc_hash::FxHasher::default();
            p.hash(&mut h);
            h.finish()
        };
        let mut gate = lim.gate();
        let mut same = true;
        'levels: for &u in &sh.bottomup {
            let (fl, gl) = (&self.levels[u as usize], &other.levels[u as usize]);
            if fl.width() != gl.width() {
                same = false;
                break;
            }
            if fl.width() == 0 {
                continue;
            }
            gate.poll((fl.pairs.len() + gl.pairs.len()) as u64)?;
            // `other`'s nodes with their pairs sorted, by hash.
            theirs.clear();
            lim.reserve_exact(&mut theirs, gl.pairs.len())?;
            theirs.extend_from_slice(&gl.pairs);
            keys.clear();
            lim.reserve_exact(&mut keys, gl.width())?;
            for j in 0..gl.width() as u32 {
                let r = gl.start[j as usize] as usize..gl.start[j as usize + 1] as usize;
                theirs[r.clone()].sort_unstable();
                keys.push((hash(&theirs[r]), j));
            }
            keys.sort_unstable();
            for i in 0..fl.width() as u32 {
                mine.clear();
                for &(a, b) in fl.node(i) {
                    let rename = |r: Ref| if r == ZREF || sh.is_leaf(vt(r)) { r } else { mk(vt(r), to[(base[vt(r) as usize] + ix(r)) as usize]) };
                    lim.try_push(&mut mine, (rename(a), rename(b)))?;
                }
                mine.sort_unstable();
                let h = hash(&mine);
                let lo = keys.partition_point(|&(k, _)| k < h);
                let hit = keys[lo..].iter().take_while(|&&(k, _)| k == h).map(|&(_, j)| j).find(|&j| {
                    theirs[gl.start[j as usize] as usize..gl.start[j as usize + 1] as usize] == mine[..]
                });
                match hit {
                    Some(j) => to[(base[u as usize] + i) as usize] = j,
                    None => {
                        same = false;
                        break 'levels;
                    }
                }
            }
        }
        gate.flush()?;
        if same && self.out != ZREF {
            let r = self.out;
            let mapped = if sh.is_leaf(vt(r)) { r } else { mk(vt(r), to[(base[vt(r) as usize] + ix(r)) as usize]) };
            same = mapped == other.out;
        }
        for v in [mine, theirs] {
            lim.discard(v);
        }
        lim.discard(keys);
        lim.discard(to);
        lim.discard(base);
        Ok(same)
    }
}

/// Model counts: `true` on a variable is its two assignments.
pub(crate) struct Count;

impl EvalAlgebra for Count {
    type Value = num_bigint::BigUint;
    fn zero(&self) -> Self::Value {
        Self::Value::default()
    }
    fn leaf(&self, _: VarId, label: LeafLabel) -> Self::Value {
        Self::Value::from(if label == LeafLabel::One { 2u32 } else { 1 })
    }
    fn add_assign(&self, acc: &mut Self::Value, other: &Self::Value) {
        *acc += other;
    }
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        a * b
    }
}
