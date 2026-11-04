//! A weighted count in skip form with both of its passes kept
//! ([`SkipCount`]). The pass up holds each node's value. The pass down holds
//! each node's context, the weight of the assignments to the variables
//! outside its level's subtree summed over the paths to it. From them the
//! count of the diagram with a literal, or with a second diagram over the
//! same vtree, is read without building their conjunction.
//!
//! A side read at child level `c` that names a node below `c` crosses every
//! level between, and each of those leaves its leaf child free. Besides the
//! nodes' contexts, the pass down keeps the context of each side that skips,
//! at the child level it is read at ([`Down::skips`]). A level the side
//! crosses sees it with that context times the free weights crossed above
//! the level. That is how the diagram's states at any level are read
//! ([`SkipCount::skip_states`]): its nodes there, and the skips that cross
//! it.

use std::collections::hash_map::Entry;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::api::SkipTdd;
use super::eval::{Crossings, Valuer, label};
use super::{NEG, ONE, POS, Ref, Shape, SkipForm, ZREF, discard_nested, ix, mk, vt};
use crate::OperationError;
use crate::diagram::{EvalAlgebra, LeafLabel, Literal};
use crate::limits::{Limits, PollGate};
use crate::vtree::VarId;

/// A state at a level and its context; `None` is a context of one, that of
/// the output read at the root.
type State<V> = (Ref, Option<V>);

/// The weighted count of a [`SkipTdd`] in an algebra, its pass up kept, and
/// its pass down once a joint count reads it: what
/// [`Engine::skip_count_literal`](crate::Engine::skip_count_literal) and
/// [`Engine::skip_count_and`](crate::Engine::skip_count_and) read.
///
/// The passes are those of [`Engine::skip_evaluate`](crate::Engine::skip_evaluate):
/// every value is a sum of products of the leaf values the algebra gives,
/// and the joint counts add and multiply only. An algebra that tracks
/// whether its values stay exact in floats sees every value the passes and
/// a joint count compute in the calls that compute them. Each call must take
/// the algebra the count was taken in, or one that gives the same leaf
/// values.
///
/// The values live outside the engine's byte budget once
/// [`Engine::skip_count`](crate::Engine::skip_count) returns, as the
/// diagram's levels do outside an operation: the count holds a copy of the
/// diagram, its levels shared.
pub struct SkipCount<V> {
    f: SkipTdd,
    cross: Crossings<V>,
    /// Per internal level, its nodes' values.
    up: Vec<Vec<V>>,
    total: V,
    down: Option<Down<V>>,
}

/// The pass down.
struct Down<V> {
    /// Per internal level, each node's context. The root's node, the
    /// output's, has context one, which is not stored.
    ctx: Vec<Vec<V>>,
    /// Per leaf, the context of each label a side names there, `One`, `Pos`
    /// and `Neg`, whether the side is read at the leaf or skips to it.
    leaf: Vec<Vec<V>>,
    /// Per child level, the sides read there that skip, by the preorder of
    /// the node they name, with their contexts summed per reference.
    skips: Vec<Vec<(Ref, V)>>,
    /// Whether the output, read at the root, skips.
    out_skips: bool,
    /// Per internal level, whether each node is `true` over the level's
    /// variables.
    top: Vec<Vec<bool>>,
}

impl<V> std::fmt::Debug for SkipCount<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkipCount")
            .field("pairs", &self.f.pair_count())
            .field("down", &self.down.is_some())
            .finish()
    }
}

impl<V: Clone> SkipCount<V> {
    /// The weighted count of the diagram.
    pub fn total(&self) -> &V {
        &self.total
    }

    /// The diagram counted.
    pub fn diagram(&self) -> &SkipTdd {
        &self.f
    }
}

/// `context × x`, a context of one where there is none.
fn scale<S: EvalAlgebra>(algebra: &S, context: &Option<S::Value>, x: S::Value) -> S::Value {
    match context {
        None => x,
        Some(c) => algebra.mul(c, &x),
    }
}

/// The value of leaf `v`'s label `k`.
fn leaf_value<S: EvalAlgebra>(sh: &Shape, algebra: &S, v: u32, k: u32) -> S::Value {
    algebra.leaf(VarId(sh.var[v as usize]), label(k))
}

/// The value of the conjunction of labels `kx` and `ky` at leaf `v`: `true`
/// and a label is the label, equal labels the label, and opposite literals
/// false.
fn labels<S: EvalAlgebra>(sh: &Shape, algebra: &S, v: u32, kx: u32, ky: u32) -> S::Value {
    match (kx, ky) {
        (ONE, k) | (k, ONE) => leaf_value(sh, algebra, v, k),
        (a, b) if a == b => leaf_value(sh, algebra, v, a),
        _ => algebra.zero(),
    }
}

/// The value of `r` read at child level `c`, from the values `up` of the
/// nodes of `r`'s diagram.
fn value_of<S: EvalAlgebra>(
    sh: &Shape,
    algebra: &S,
    cross: &mut Crossings<S::Value>,
    up: &[Vec<S::Value>],
    c: u32,
    r: Ref,
) -> S::Value {
    let v = vt(r);
    let base = if sh.is_leaf(v) {
        leaf_value(sh, algebra, v, ix(r))
    } else {
        up[v as usize][ix(r) as usize].clone()
    };
    if v == c {
        return base;
    }
    algebra.mul(&base, &cross.crossed(sh, algebra, c, v))
}

/// Per internal level of `form`, whether each node is `true` over the
/// level's variables: one pair, each side `true` over its child level's,
/// which a side is where it names the label `One` or a node that is `true`,
/// since the levels it crosses leave their variables free. A canonical
/// form has one way to write `true` over a subtree.
fn tops(form: &SkipForm, lim: &Limits) -> Result<Vec<Vec<bool>>, OperationError> {
    let sh = &*form.shape;
    let mut top: Vec<Vec<bool>> = Vec::new();
    lim.try_resize(&mut top, sh.num_nodes(), Vec::new())?;
    for &t in &sh.bottomup {
        let lvl = &form.levels[t as usize];
        let mut flags = Vec::new();
        lim.reserve_exact(&mut flags, lvl.width())?;
        for i in 0..lvl.width() as u32 {
            flags.push(match *lvl.node(i) {
                [(a, b)] => is_top(sh, &top, a) && is_top(sh, &top, b),
                _ => false,
            });
        }
        top[t as usize] = flags;
    }
    Ok(top)
}

/// Whether `r` is `true` over the subtree of any child level it is read at.
fn is_top(sh: &Shape, top: &[Vec<bool>], r: Ref) -> bool {
    r != ZREF
        && if sh.is_leaf(vt(r)) {
            ix(r) == ONE
        } else {
            top[vt(r) as usize][ix(r) as usize]
        }
}

impl<V: Clone> SkipCount<V> {
    /// The pass up over `f`.
    pub(crate) fn new<S: EvalAlgebra<Value = V>>(
        f: &SkipTdd,
        algebra: &S,
        lim: &Limits,
    ) -> Result<Self, OperationError> {
        let sh = &*f.form.shape;
        let mut values = Valuer::new(sh, algebra, |v, k| leaf_value(sh, algebra, v, k), lim)?;
        let mut gate = lim.gate();
        if !f.form.is_zero() {
            for &t in &sh.bottomup {
                values.fill(&f.form, t, &mut gate, lim)?;
            }
        }
        gate.flush()?;
        let total = if f.form.is_zero() {
            algebra.zero()
        } else {
            values.value(sh.root, f.form.out)
        };
        let (cross, up) = values.into_parts();
        Ok(SkipCount {
            f: f.clone(),
            cross,
            up,
            total,
            down: None,
        })
    }

    /// The pass down, once.
    fn ensure_down<S: EvalAlgebra<Value = V>>(
        &mut self,
        algebra: &S,
        lim: &Limits,
    ) -> Result<(), OperationError> {
        if self.down.is_some() {
            return Ok(());
        }
        let form = &self.f.form;
        let sh = &*form.shape;
        let n = sh.num_nodes();
        let (cross, up) = (&mut self.cross, &self.up);
        let mut ctx: Vec<Vec<V>> = Vec::new();
        lim.try_resize(&mut ctx, n, Vec::new())?;
        let mut leaf: Vec<Vec<V>> = Vec::new();
        lim.try_resize(&mut leaf, n, Vec::new())?;
        for v in 0..n {
            let width = if sh.is_leaf(v as u32) {
                3
            } else {
                form.levels[v].width()
            };
            lim.try_resize(
                if sh.is_leaf(v as u32) {
                    &mut leaf[v]
                } else {
                    &mut ctx[v]
                },
                width,
                algebra.zero(),
            )?;
        }
        // The skipping sides' contexts, summed per child level and reference.
        let mut at: FxHashMap<(u32, Ref), usize> = FxHashMap::default();
        let mut skipping: Vec<(u32, Ref, V)> = Vec::new();
        let out = form.out;
        let out_skips = out != ZREF && vt(out) != sh.root;
        // Adds `m`, the context of a side read at `c` that names `x`.
        let mut side = |cross: &mut Crossings<V>,
                        ctx: &mut Vec<Vec<V>>,
                        leaf: &mut Vec<Vec<V>>,
                        c: u32,
                        x: Ref,
                        m: V|
         -> Result<(), OperationError> {
            let v = vt(x);
            let m = if v == c {
                m
            } else {
                let y = algebra.mul(&m, &cross.crossed(sh, algebra, c, v));
                match at.entry((c, x)) {
                    Entry::Occupied(e) => algebra.add_assign(&mut skipping[*e.get()].2, &m),
                    Entry::Vacant(e) => {
                        e.insert(skipping.len());
                        lim.try_push(&mut skipping, (c, x, m))?;
                    }
                }
                y
            };
            let slot = if sh.is_leaf(v) {
                &mut leaf[v as usize][ix(x) as usize]
            } else {
                &mut ctx[v as usize][ix(x) as usize]
            };
            algebra.add_assign(slot, &m);
            Ok(())
        };
        if out_skips {
            // The output's node: its context is the free weights the output
            // crosses from the root.
            let v = vt(out);
            let m = cross.crossed(sh, algebra, sh.root, v);
            let slot = if sh.is_leaf(v) {
                &mut leaf[v as usize][ix(out) as usize]
            } else {
                &mut ctx[v as usize][ix(out) as usize]
            };
            algebra.add_assign(slot, &m);
        }
        let mut gate = lim.gate();
        if out != ZREF {
            for &t in &sh.topdown {
                let lvl = &form.levels[t as usize];
                let (l, r) = sh.kids[t as usize];
                for i in 0..lvl.width() {
                    let c = match t == sh.root {
                        false => Some(ctx[t as usize][i].clone()),
                        true if out == mk(t, i as u32) => None,
                        true => continue,
                    };
                    let pairs = lvl.node(i as u32);
                    gate.poll(pairs.len() as u64)?;
                    for &(a, b) in pairs {
                        if a == ZREF || b == ZREF {
                            continue;
                        }
                        let (va, vb) = (
                            value_of(sh, algebra, cross, up, l, a),
                            value_of(sh, algebra, cross, up, r, b),
                        );
                        side(cross, &mut ctx, &mut leaf, l, a, scale(algebra, &c, vb))?;
                        side(cross, &mut ctx, &mut leaf, r, b, scale(algebra, &c, va))?;
                    }
                }
            }
        }
        gate.flush()?;
        let mut skips: Vec<Vec<(Ref, V)>> = Vec::new();
        lim.try_resize(&mut skips, n, Vec::new())?;
        skipping.sort_unstable_by_key(|&(c, x, _)| (c, sh.pre[vt(x) as usize], x));
        for (c, x, m) in skipping {
            lim.try_push(&mut skips[c as usize], (x, m))?;
        }
        let top = tops(form, lim)?;
        self.down = Some(Down {
            ctx,
            leaf,
            skips,
            out_skips,
            top,
        });
        Ok(())
    }

    /// Appends to `out` the states at internal level `s` that skip and name
    /// a node in `under`'s subtree (`s`'s or a child's), with their
    /// contexts: the sides read at `s` that skip, and those read above `s`
    /// that cross it, their contexts times the free weights crossed above
    /// `s`. A side that names `s`'s own node is in that node's context.
    fn skip_states<S: EvalAlgebra<Value = V>>(
        &mut self,
        algebra: &S,
        s: u32,
        under: u32,
        out: &mut Vec<State<V>>,
        lim: &Limits,
    ) -> Result<(), OperationError> {
        let sh = &*self.f.form.shape;
        let down = self.down.as_ref().expect("the pass down is taken");
        // The preorders of the nodes a state may name.
        let lo = sh.pre[under as usize] + u32::from(under == s);
        let hi = sh.pre[under as usize] + sh.size[under as usize];
        let named = |list: &[(Ref, V)]| {
            let a = list.partition_point(|&(x, _)| sh.pre[vt(x) as usize] < lo);
            let b = list.partition_point(|&(x, _)| sh.pre[vt(x) as usize] < hi);
            a..b
        };
        let here = &down.skips[s as usize];
        for k in named(here) {
            let (x, ref m) = here[k];
            lim.try_push(out, (x, Some(m.clone())))?;
        }
        // Up the chain of levels whose other child is a leaf, that a side
        // read above `s` crosses on its way to `s`.
        let mut w = s;
        loop {
            let p = sh.parent[w as usize];
            if p == super::NONE {
                break;
            }
            let (l, r) = sh.kids[p as usize];
            let sib = if l == w { r } else { l };
            if !sh.is_leaf(sib) || !sh.skip[p as usize] {
                break;
            }
            let list = &down.skips[p as usize];
            let range = named(list);
            if !range.is_empty() {
                let x = self.cross.crossed(sh, algebra, p, s);
                for k in range {
                    let (r, ref m) = list[k];
                    lim.try_push(out, (r, Some(algebra.mul(m, &x))))?;
                }
            }
            w = p;
        }
        let o = self.f.form.out;
        if down.out_skips && w == sh.root && (lo..hi).contains(&sh.pre[vt(o) as usize]) {
            let m = (s != sh.root).then(|| self.cross.crossed(sh, algebra, sh.root, s));
            lim.try_push(out, (o, m))?;
        }
        Ok(())
    }

    /// The weighted count of the diagram and `literal`.
    pub(crate) fn literal<S: EvalAlgebra<Value = V>>(
        &mut self,
        algebra: &S,
        literal: Literal,
        lim: &Limits,
    ) -> Result<V, OperationError> {
        let x = self
            .f
            .vtree()
            .leaf_of(literal.var)
            .ok_or(OperationError::VariableNotInVtree(literal.var))?
            .0;
        if self.f.form.is_zero() {
            return Ok(algebra.zero());
        }
        let sh = Arc::clone(&self.f.form.shape);
        let w = algebra.leaf(
            literal.var,
            if literal.sign {
                LeafLabel::Pos
            } else {
                LeafLabel::Neg
            },
        );
        let asserts = if literal.sign { POS } else { NEG };
        // The weight of the literal on a side that names label `k`.
        let on = |k: u32| (k == ONE || k == asserts).then(|| w.clone());
        if sh.is_leaf(sh.root) {
            return Ok(on(ix(self.f.form.out)).unwrap_or_else(|| algebra.zero()));
        }
        self.ensure_down(algebra, lim)?;
        let mut acc = algebra.zero();
        let down = self.down.as_ref().expect("taken above");
        for k in [ONE, POS, NEG] {
            if let Some(y) = on(k) {
                algebra.mul_add(&mut acc, &down.leaf[x as usize][k as usize], &y);
            }
        }
        // The states at the parent that skip into the sibling's subtree
        // leave the variable free: its literal takes its own weight there.
        let p = sh.parent[x as usize];
        let (l, r) = sh.kids[p as usize];
        let sib = if l == x { r } else { l };
        let mut states = Vec::new();
        self.skip_states(algebra, p, sib, &mut states, lim)?;
        let mut gate = lim.gate();
        for (r, m) in &states {
            gate.poll(1)?;
            let v = value_of(&sh, algebra, &mut self.cross, &self.up, sib, *r);
            algebra.add_assign(&mut acc, &scale(algebra, m, algebra.mul(&w, &v)));
        }
        gate.flush()?;
        lim.discard(states);
        Ok(acc)
    }

    /// The weighted count of the diagram and `g`, a diagram over its vtree
    /// without summed-out subtrees.
    pub(crate) fn and<S: EvalAlgebra<Value = V>>(
        &mut self,
        algebra: &S,
        g: &SkipTdd,
        lim: &Limits,
    ) -> Result<V, OperationError> {
        if !Arc::ptr_eq(self.f.vtree(), g.vtree()) {
            return Err(OperationError::VtreeMismatch);
        }
        if self.f.form.is_zero() || g.form.is_zero() {
            return Ok(algebra.zero());
        }
        let sh = Arc::clone(&self.f.form.shape);
        let (e_out, g_out) = (self.f.form.out, g.form.out);
        if sh.is_leaf(sh.root) {
            return Ok(labels(&sh, algebra, sh.root, ix(e_out), ix(g_out)));
        }
        // `g`'s values and `true` nodes, its skips valued by the crossings
        // the count holds.
        let g_top = tops(&g.form, lim)?;
        if is_top(&sh, &g_top, g_out) {
            return Ok(self.total.clone());
        }
        let s = vt(g_out);
        if sh.is_leaf(s) {
            return self.literal(
                algebra,
                Literal::new(VarId(sh.var[s as usize]), ix(g_out) == POS),
                lim,
            );
        }
        let mut values = Valuer::with_crossings(
            &sh,
            algebra,
            |v, k| leaf_value(&sh, algebra, v, k),
            &mut self.cross,
            lim,
        )?;
        let mut gate = lim.gate();
        let filled = sh
            .bottomup
            .iter()
            .try_for_each(|&t| values.fill(&g.form, t, &mut gate, lim));
        let (cross, g_up) = values.into_parts();
        self.cross = cross;
        filled?;
        self.ensure_down(algebra, lim)?;
        // The diagram's states at `s`: its nodes there and the skips that
        // cross `s`, against `g`'s node there, true on every variable above.
        let mut states: Vec<State<V>> = Vec::new();
        let down = self.down.as_ref().expect("taken above");
        for i in 0..self.f.form.levels[s as usize].width() {
            let x = mk(s, i as u32);
            match s == sh.root {
                false => lim.try_push(&mut states, (x, Some(down.ctx[s as usize][i].clone())))?,
                true if x == e_out => lim.try_push(&mut states, (x, None))?,
                true => {}
            }
        }
        self.skip_states(algebra, s, s, &mut states, lim)?;
        let down = self.down.as_ref().expect("taken above");
        let mut walk = Walk {
            sh: &sh,
            algebra,
            e: &self.f.form,
            g: &g.form,
            e_up: &self.up,
            g_up: &g_up,
            e_top: &down.top,
            g_top: &g_top,
            cross: &mut self.cross,
            lim,
            index: Vec::new(),
            keys: Vec::new(),
            vals: Vec::new(),
        };
        let res = walk.run(s, &states, g_out, &mut gate);
        walk.release();
        gate.flush()?;
        let mut acc = algebra.zero();
        for ((_, m), j) in states.iter().zip(res?) {
            if let Some(j) = j {
                algebra.add_assign(&mut acc, &scale(algebra, m, j));
            }
        }
        lim.discard(states);
        Ok(acc)
    }
}

/// What a pair of sides read at one child level is worth.
enum Child {
    Zero,
    /// `g` is `true` there: the first diagram's side.
    First(u32, Ref),
    /// The first diagram is `true` there: `g`'s side.
    Second(u32, Ref),
    /// Two labels at a leaf.
    Labels(u32, u32, u32),
    /// A product of nodes the walk values, at the level of its key.
    Key(u32, Ref, Ref),
}

/// The count of the first diagram and `g` from a level down: the products
/// of their nodes the states at the level reach, each read at the highest
/// level where either names a node of its own, valued bottom-up.
struct Walk<'a, S: EvalAlgebra> {
    sh: &'a Shape,
    algebra: &'a S,
    e: &'a SkipForm,
    g: &'a SkipForm,
    e_up: &'a [Vec<S::Value>],
    g_up: &'a [Vec<S::Value>],
    e_top: &'a [Vec<bool>],
    g_top: &'a [Vec<bool>],
    cross: &'a mut Crossings<S::Value>,
    lim: &'a Limits,
    /// Per level, its products' keys, their places and their values, each
    /// charged to `lim` as it grows and handed back by [`Walk::release`].
    index: Vec<FxHashMap<(Ref, Ref), u32>>,
    keys: Vec<Vec<(Ref, Ref)>>,
    vals: Vec<Vec<S::Value>>,
}

impl<S: EvalAlgebra> Walk<'_, S> {
    /// Sides `x` of the first diagram and `y` of `g`, read at child level
    /// `c`, and the crossing `(c, d)` they share above `d` where both skip
    /// past `c` toward one node: `true` on both, its free weights multiply
    /// the product read at `d`.
    fn norm(&self, c: u32, x: Ref, y: Ref) -> (Option<(u32, u32)>, Child) {
        let sh = self.sh;
        if x == ZREF || y == ZREF {
            return (None, Child::Zero);
        }
        if is_top(sh, self.g_top, y) {
            return (None, Child::First(c, x));
        }
        if is_top(sh, self.e_top, x) {
            return (None, Child::Second(c, y));
        }
        let (vx, vy) = (vt(x), vt(y));
        let (mut d, mut crossing) = (c, None);
        if vx != c && vy != c {
            let (hi, lo) = if sh.depth[vx as usize] <= sh.depth[vy as usize] {
                (vx, vy)
            } else {
                (vy, vx)
            };
            if sh.within(lo, hi) {
                (d, crossing) = (hi, Some((c, hi)));
            }
        }
        if sh.is_leaf(d) {
            return (crossing, Child::Labels(d, ix(x), ix(y)));
        }
        (crossing, Child::Key(d, x, y))
    }

    fn register(&mut self, c: u32, x: Ref, y: Ref) -> Result<(), OperationError> {
        if let (_, Child::Key(d, x, y)) = self.norm(c, x, y) {
            let (index, keys) = (&mut self.index[d as usize], &mut self.keys[d as usize]);
            if !index.contains_key(&(x, y)) {
                self.lim.reserve_map(index, 1)?;
                self.lim.try_push(keys, (x, y))?;
                index.insert((x, y), (keys.len() - 1) as u32);
            }
        }
        Ok(())
    }

    /// Hands the tables' charge back to the limits.
    fn release(&mut self) {
        for map in std::mem::take(&mut self.index) {
            self.lim.discard(map);
        }
        discard_nested(self.lim, std::mem::take(&mut self.keys));
        discard_nested(self.lim, std::mem::take(&mut self.vals));
    }

    /// The value of sides `x` and `y` read at `c`; `None` for zero.
    fn value(&mut self, c: u32, x: Ref, y: Ref) -> Option<S::Value> {
        let (crossing, child) = self.norm(c, x, y);
        let (sh, algebra) = (self.sh, self.algebra);
        let v = match child {
            Child::Zero => return None,
            Child::First(c, x) => value_of(sh, algebra, self.cross, self.e_up, c, x),
            Child::Second(c, y) => value_of(sh, algebra, self.cross, self.g_up, c, y),
            Child::Labels(v, kx, ky) => labels(sh, algebra, v, kx, ky),
            Child::Key(d, x, y) => {
                self.vals[d as usize][self.index[d as usize][&(x, y)] as usize].clone()
            }
        };
        Some(match crossing {
            None => v,
            Some((c, d)) => algebra.mul(&v, &self.cross.crossed(sh, algebra, c, d)),
        })
    }

    /// The value of each state at `s` against `g`'s node `g_out` there.
    fn run(
        &mut self,
        s: u32,
        states: &[State<S::Value>],
        g_out: Ref,
        gate: &mut PollGate<'_>,
    ) -> Result<Vec<Option<S::Value>>, OperationError> {
        let (sh, e, g) = (self.sh, self.e, self.g);
        let (n, lim) = (sh.num_nodes(), self.lim);
        lim.try_resize(&mut self.index, n, FxHashMap::default())?;
        lim.try_resize(&mut self.keys, n, Vec::new())?;
        lim.try_resize(&mut self.vals, n, Vec::new())?;
        for &(x, _) in states {
            self.register(s, x, g_out)?;
        }
        let (mut bx, mut by) = ([(ZREF, ZREF)], [(ZREF, ZREF)]);
        // The products the states reach, parents first.
        for &t in sh.topdown.iter().filter(|&&t| sh.within(t, s)) {
            let (l, r) = sh.kids[t as usize];
            for k in 0..self.keys[t as usize].len() {
                let (x, y) = self.keys[t as usize][k];
                let (px, py) = (e.pairs_at(t, x, &mut bx), g.pairs_at(t, y, &mut by));
                gate.poll((px.len() * py.len()) as u64)?;
                for &(a, b) in px {
                    for &(a2, b2) in py {
                        self.register(l, a, a2)?;
                        self.register(r, b, b2)?;
                    }
                }
            }
        }
        // Their values, children first.
        for &t in sh.bottomup.iter().filter(|&&t| sh.within(t, s)) {
            let (l, r) = sh.kids[t as usize];
            let keys = std::mem::take(&mut self.keys[t as usize]);
            let mut out = Vec::new();
            lim.reserve_exact(&mut out, keys.len())?;
            for &(x, y) in &keys {
                let (px, py) = (e.pairs_at(t, x, &mut bx), g.pairs_at(t, y, &mut by));
                gate.poll((px.len() * py.len()) as u64)?;
                let mut acc = self.algebra.zero();
                for &(a, b) in px {
                    for &(a2, b2) in py {
                        let Some(u) = self.value(l, a, a2) else {
                            continue;
                        };
                        let Some(v) = self.value(r, b, b2) else {
                            continue;
                        };
                        self.algebra.mul_add(&mut acc, &u, &v);
                    }
                }
                out.push(acc);
            }
            self.keys[t as usize] = keys;
            self.vals[t as usize] = out;
        }
        Ok(states
            .iter()
            .map(|&(x, _)| self.value(s, x, g_out))
            .collect())
    }
}
