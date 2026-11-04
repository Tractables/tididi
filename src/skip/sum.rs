//! Summed-out subtrees in skip form.
//!
//! A diagram with weights may sum a subtree out. The shape then reads the
//! subtree as one leaf: a summed-out variable keeps its labels `One`, `Pos`
//! and `Neg`, worth both literals' weights, the positive one's and the
//! negative one's; a summed-out internal subtree has a column of values in
//! the weight store, whose slot 0, `One`, is `true` over the subtree, and a
//! reference into it names the slot of its value. `One` stays `true`: a
//! pair `true` on such a leaf is skipped as on a variable, and its weight is
//! what the skip leaves free.
//!
//! Above a summed-out subtree a node's pairs are a multiset of products to
//! add, as in a stored diagram with marginal levels: a conjunction keeps a
//! repeated pair, and only a positive and a negative literal beside the
//! same side merge ([`super::normalize_sum`]). Summing out builds each new
//! column from the values of the references that enter the subtree, then
//! rebuilds the levels above from the leaves up: a node's pairs that name
//! one side beside slots of a column are fused into one slot holding their
//! sum, and equal nodes are shared. Slots no reference names are dropped.
//! Where both children of a node are summed out, so is the node.
//!
//! A conjunction runs on a shape that sums out both operands' subtrees. An
//! operand must be `true` over a subtree the other sums out, and only one
//! of them may read slots other than `One` of a column.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::conjoin;
use super::contract::contract_in;
use super::eval::{label, Valuer};
use super::{discard_nested, filled, ix, mk, normalize, normalize_sum, vt, Ref, Shape, SkipForm, NONE, ONE, SUMMED, SUMMED_VAR, VAR, ZREF};
use crate::diagram::{weight_key, WeightKey};
use crate::diagram::{EvalAlgebra, LeafLabel, WeightStore, WeightValue};
use crate::limits::Limits;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::OperationError;

/// A store's weights as an algebra.
pub(crate) struct Weights<'a>(pub(crate) &'a WeightStore);

impl EvalAlgebra for Weights<'_> {
    type Value = WeightValue;
    fn zero(&self) -> WeightValue {
        self.0.wzero()
    }
    fn leaf(&self, var: VarId, label: LeafLabel) -> WeightValue {
        self.0.leaf_val(var, label)
    }
    fn add_assign(&self, acc: &mut WeightValue, other: &WeightValue) {
        acc.add_assign(other);
    }
    fn mul(&self, a: &WeightValue, b: &WeightValue) -> WeightValue {
        a.mul(b)
    }
}

/// The value of label `k` of leaf `v` of `sh`: a summed-out subtree's from
/// its column in `ws`, a variable's from its weights.
fn leaf_value(sh: &Shape, ws: &WeightStore, v: u32, k: u32) -> WeightValue {
    if sh.kind[v as usize] == SUMMED {
        ws.level(v as usize).expect("a summed-out subtree has its column")[k as usize].clone()
    } else {
        ws.leaf_val(VarId(sh.var[v as usize]), label(k))
    }
}

/// The value of `f` under `ws`, its summed-out subtrees' columns included.
pub(crate) fn weighted_value(f: &SkipForm, ws: &WeightStore, lim: &Limits) -> Result<WeightValue, OperationError> {
    let sh = &*f.shape;
    f.evaluate_with(&Weights(ws), |v, k| leaf_value(sh, ws, v, k), lim)
}

/// The weight of `true` over the subtree of `tree` at `m`: the product of
/// its variables' weights of `true`.
pub(crate) fn top_weight(tree: &Vtree, ws: &WeightStore, m: VtreeIdx) -> WeightValue {
    let mut acc: Option<WeightValue> = None;
    let mut stack = vec![m];
    while let Some(t) = stack.pop() {
        match *tree.node(t) {
            crate::vtree::VtreeNode::Leaf { var, .. } => {
                let w = ws.leaf_val(var, LeafLabel::One);
                acc = Some(match acc {
                    Some(a) => a.mul(&w),
                    None => w,
                });
            }
            crate::vtree::VtreeNode::Internal { left, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
        }
    }
    acc.expect("a subtree has a leaf")
}

/// The summed-out subtrees of `sh`, as `(root, labels)`.
fn roots(sh: &Shape) -> Vec<(u32, u32)> {
    (0..sh.num_nodes() as u32).filter(|&v| sh.kind[v as usize] >= SUMMED_VAR).map(|v| (v, sh.labels[v as usize])).collect()
}

/// A column while it is built: a value already held is not added again.
/// Slot 0, `One`, is `true` alone: a function whose value equals `true`'s
/// takes a slot of its own, since a conjunction reads `One` as `true`
/// (under signed weights a function other than `true` can weigh as much).
struct Column {
    vals: Vec<WeightValue>,
    keys: FxHashMap<WeightKey, u32>,
}

impl Column {
    fn new(vals: Vec<WeightValue>, lim: &Limits) -> Result<Column, OperationError> {
        let mut keys = FxHashMap::default();
        lim.reserve_map(&mut keys, vals.len())?;
        for (i, v) in vals.iter().enumerate().skip(1) {
            keys.entry(weight_key(v)).or_insert(i as u32);
        }
        Ok(Column { vals, keys })
    }

    fn slot(&mut self, v: WeightValue, lim: &Limits) -> Result<u32, OperationError> {
        let key = weight_key(&v);
        if let Some(&i) = self.keys.get(&key) {
            return Ok(i);
        }
        let i = self.vals.len() as u32;
        lim.try_push(&mut self.vals, v)?;
        lim.reserve_map(&mut self.keys, 1)?;
        self.keys.insert(key, i);
        Ok(i)
    }
}

/// Per node of `vt`, whether it is summed out once `targets` and the
/// subtrees `sh` sums out are: below a summed-out node, and where both
/// children are.
fn summed_after(tree: &Vtree, sh: &Shape, targets: &[u32], lim: &Limits) -> Result<(Shape, Vec<bool>), OperationError> {
    let full = Shape::of(tree);
    let mut inside = filled(lim, full.num_nodes(), false)?;
    for &t in targets {
        inside[t as usize] = true;
    }
    for (v, &k) in sh.kind.iter().enumerate() {
        if k >= SUMMED_VAR {
            inside[v] = true;
        }
    }
    for &t in &full.topdown {
        if inside[t as usize] {
            let (l, r) = full.kids[t as usize];
            inside[l as usize] = true;
            inside[r as usize] = true;
        }
    }
    for &t in &full.bottomup {
        let (l, r) = full.kids[t as usize];
        if inside[l as usize] && inside[r as usize] {
            inside[t as usize] = true;
        }
    }
    Ok((full, inside))
}

/// `f` with the subtrees of `targets` summed out under `ws`, whose columns
/// it updates, and every node both of whose children are summed out, its
/// equal nodes shared, twins merged and unread slots dropped; `None` where nothing
/// more is summed out, unless `rebuild` asks for that sharing anyway.
///
/// `changed`, where given, is the upward-closed set of levels that may
/// differ from a canonical diagram: the others, outside every new summed
/// subtree's ancestors, are copied as they are and not contracted again.
pub(crate) fn sum_out(
    f: &SkipForm,
    tree: &Vtree,
    ws: &mut WeightStore,
    targets: &[u32],
    rebuild: bool,
    changed: Option<&[bool]>,
    lim: &Limits,
) -> Result<Option<SkipForm>, OperationError> {
    let sh = &*f.shape;
    let (full, inside) = summed_after(tree, sh, targets, lim)?;
    let is_root = |v: usize| inside[v] && (full.parent[v] == NONE || !inside[full.parent[v] as usize]);
    let mut new_roots: Vec<(u32, u32)> = Vec::new();
    for v in 0..full.num_nodes() {
        if is_root(v) {
            let labels = if full.kind[v] == VAR { 3 } else { sh.labels[v] };
            lim.try_push(&mut new_roots, (v as u32, labels))?;
        }
    }
    lim.discard(inside);
    if !rebuild && new_roots.iter().all(|&(m, _)| sh.kind[m as usize] >= SUMMED_VAR) {
        lim.discard(new_roots);
        return Ok(None);
    }
    // The levels rebuilt: `changed` and every new summed subtree's
    // ancestors, or all of them without `changed`.
    let mut rebuilt = match changed {
        None => None,
        Some(cone) => {
            let mut r = filled(lim, sh.num_nodes(), false)?;
            r.copy_from_slice(cone);
            for &(m, _) in &new_roots {
                if sh.kind[m as usize] < SUMMED_VAR {
                    let mut u = full.parent[m as usize];
                    while u != NONE && !r[u as usize] {
                        r[u as usize] = true;
                        u = full.parent[u as usize];
                    }
                }
            }
            Some(r)
        }
    };
    let shaped = Shape::summed(tree, &new_roots);
    // The values below the new subtrees, in `f`'s shape, and `true` over
    // each new internal one.
    let mut cols: FxHashMap<u32, Column> = FxHashMap::default();
    let mut gate = lim.gate();
    let wsr: &WeightStore = ws;
    let algebra = Weights(wsr);
    let mut values = Valuer::new(sh, &algebra, |v, k| leaf_value(sh, wsr, v, k), lim)?;
    let mut top: Vec<Option<WeightValue>> = Vec::new();
    lim.try_resize(&mut top, sh.num_nodes(), None)?;
    for &t in &sh.bottomup {
        if shaped.owner[t as usize] != NONE {
            values.fill(f, t, &mut gate, lim)?;
            let (l, r) = sh.kids[t as usize];
            let side = |c: u32, top: &Vec<Option<WeightValue>>| {
                if sh.is_leaf(c) { leaf_value(sh, wsr, c, ONE) } else { top[c as usize].clone().expect("children first") }
            };
            let both = side(l, &top).mul(&side(r, &top));
            top[t as usize] = Some(both);
        }
    }
    for &(m, _) in &new_roots {
        if shaped.kind[m as usize] == SUMMED_VAR {
            continue;
        }
        let vals = if sh.kind[m as usize] == SUMMED {
            wsr.level(m as usize).expect("a summed-out subtree has its column").to_vec()
        } else {
            vec![top[m as usize].take().expect("a new subtree's top")]
        };
        lim.reserve_map(&mut cols, 1)?;
        cols.insert(m, Column::new(vals, lim)?);
    }
    lim.discard(top);
    let shaped = Arc::new(shaped);
    let mut out = SkipForm::zero(Arc::clone(&shaped));
    let mut remap: Vec<Vec<Ref>> = Vec::new();
    lim.try_resize(&mut remap, sh.num_nodes(), Vec::new())?;
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    let mut seen: FxHashMap<Vec<(Ref, Ref)>, Ref> = FxHashMap::default();
    let mut tops: FxHashMap<Ref, bool> = FxHashMap::default();
    for &t in &shaped.bottomup {
        let lvl = &f.levels[t as usize];
        if rebuilt.as_ref().is_some_and(|r| !r[t as usize]) {
            // Its references stay, as `remap` left empty says.
            if lvl.width() > 0 {
                out.levels.share(t as usize, &f.levels, lim)?;
                out.active.push(t);
            }
            continue;
        }
        let mut refs = Vec::new();
        lim.reserve_exact(&mut refs, lvl.width())?;
        seen.clear();
        for i in 0..lvl.width() as u32 {
            let node = lvl.node(i);
            gate.poll(node.len() as u64)?;
            buf.clear();
            lim.reserve(&mut buf, node.len())?;
            for &(a, b) in node {
                let (x, y) = (place(f, &mut tops, sh, &shaped, &mut values, &mut cols, &remap, a, lim)?, place(f, &mut tops, sh, &shaped, &mut values, &mut cols, &remap, b, lim)?);
                buf.push((x, y));
            }
            fuse(&shaped, t, &mut buf, &mut cols, lim)?;
            let made = if shaped.weighted[t as usize] { normalize_sum(&shaped, t, &mut buf) } else { normalize(&shaped, t, &mut buf) };
            let r = match made {
                Err(target) => target,
                Ok(()) => match seen.get(&buf) {
                    Some(&r) => r,
                    None => {
                        let r = mk(t, out.push_node(t, &buf, lim)?);
                        lim.reserve_map(&mut seen, 1)?;
                        seen.insert(buf.clone(), r);
                        r
                    }
                },
            };
            refs.push(r);
        }
        remap[t as usize] = refs;
    }
    gate.flush()?;
    out.out = place(f, &mut tops, sh, &shaped, &mut values, &mut cols, &remap, f.out, lim)?;
    lim.discard(tops);
    values.discard(lim);
    lim.discard(buf);
    lim.discard(seen);
    discard_nested(lim, remap);
    out.collect_pushed(lim)?;
    // A copied level that lost a node to the collection is walked again,
    // as the conjunction's cone takes such levels in.
    if let Some(r) = rebuilt.as_mut() {
        for &t in &shaped.bottomup {
            if !r[t as usize] && out.levels[t as usize].width() < f.levels[t as usize].width() {
                let mut u = t;
                while u != NONE && !r[u as usize] {
                    r[u as usize] = true;
                    u = full.parent[u as usize];
                }
            }
        }
    }
    // Drop the slots no reference names, `One` kept, and write the columns:
    // one pass over the pairs marks the slots named, per column, and one
    // renumbers them.
    let mut keep: Vec<Vec<u32>> = Vec::new();
    lim.try_resize(&mut keep, sh.num_nodes(), Vec::new())?;
    for (&m, col) in &cols {
        keep[m as usize] = filled(lim, col.vals.len(), NONE)?;
        keep[m as usize][0] = 0;
    }
    let mut mark = |x: Ref| {
        if x != ZREF && let Some(k) = keep[vt(x) as usize].get_mut(ix(x) as usize) {
            *k = 0;
        }
    };
    mark(out.out);
    for &u in &out.active {
        for &(a, b) in &out.levels[u as usize].pairs {
            mark(a);
            mark(b);
        }
    }
    let mut dropped = false;
    for (&m, col) in cols.iter_mut() {
        let k = &mut keep[m as usize];
        let mut next = 0u32;
        for s in k.iter_mut() {
            if *s == 0 {
                *s = next;
                next += 1;
            }
        }
        if (next as usize) < col.vals.len() {
            dropped = true;
            let mut slots = k.iter();
            col.vals.retain(|_| slots.next().is_some_and(|&s| s != NONE));
        }
    }
    if dropped {
        let fix = |x: Ref| {
            if x == ZREF {
                return x;
            }
            keep[vt(x) as usize].get(ix(x) as usize).map_or(x, |&s| mk(vt(x), s))
        };
        out.out = fix(out.out);
        for &u in &out.active {
            for p in out.levels[u as usize].pairs.iter_mut() {
                *p = (fix(p.0), fix(p.1));
            }
        }
    }
    discard_nested(lim, keep);
    for (v, &k) in sh.kind.iter().enumerate() {
        if k == SUMMED && !cols.contains_key(&(v as u32)) {
            ws.take_level(v);
        }
    }
    let mut labels = new_roots;
    for (m, k) in labels.iter_mut() {
        if let Some(col) = cols.get(m) {
            *k = col.vals.len() as u32;
        }
    }
    for (m, col) in cols {
        ws.set_level(m as usize, col.vals);
    }
    out.shape = Arc::new(Shape::summed(tree, &labels));
    // Twins merge, as a stored diagram's contraction merges the nodes that
    // summing out made twins.
    let res = contract_in(out, rebuilt.as_deref(), lim);
    if let Some(r) = rebuilt {
        lim.discard(r);
    }
    Ok(Some(res?))
}

/// A reference `x` of `f`, on `sh`, as one of the
/// result on `shaped`: a node rebuilt below an unchanged level as `remap`
/// has it, one of a level copied as it is (no `remap`) as it is, a summed-out variable's label or an old column's slot as it is,
/// and a reference into a new summed-out subtree as `One` where it is
/// `true` (`tops` remembers which are) and otherwise the slot of its value.
#[allow(clippy::too_many_arguments)]
fn place<L: Fn(u32, u32) -> WeightValue>(
    f: &SkipForm,
    tops: &mut FxHashMap<Ref, bool>,
    sh: &Shape,
    shaped: &Shape,
    values: &mut Valuer<'_, Weights<'_>, L>,
    cols: &mut FxHashMap<u32, Column>,
    remap: &[Vec<Ref>],
    x: Ref,
    lim: &Limits,
) -> Result<Ref, OperationError> {
    if x == ZREF {
        return Ok(ZREF);
    }
    let v = vt(x);
    let m = shaped.owner[v as usize];
    if m == NONE {
        let r = &remap[v as usize];
        return Ok(if shaped.is_leaf(v) || r.is_empty() { x } else { r[ix(x) as usize] });
    }
    let Some(col) = cols.get_mut(&m) else {
        // A summed-out variable keeps its labels.
        return Ok(x);
    };
    if sh.kind[m as usize] == SUMMED {
        // An old column keeps its slots.
        return Ok(x);
    }
    if is_top(f, x, tops, lim)? {
        return Ok(mk(m, ONE));
    }
    let value = values.value(m, x);
    Ok(mk(m, col.slot(value, lim)?))
}

/// Fuse the pairs of a node at `t` that name one side beside slots of a
/// summed-out internal child into one pair, whose slot holds their sum.
fn fuse(sh: &Shape, t: u32, pairs: &mut Vec<(Ref, Ref)>, cols: &mut FxHashMap<u32, Column>, lim: &Limits) -> Result<(), OperationError> {
    if pairs.len() < 2 {
        return Ok(());
    }
    let (l, r) = sh.kids[t as usize];
    let left = if sh.kind[l as usize] == SUMMED {
        true
    } else if sh.kind[r as usize] == SUMMED {
        false
    } else {
        return Ok(());
    };
    let m = if left { l } else { r };
    let slot = |p: &(Ref, Ref)| if left { p.0 } else { p.1 };
    let other = |p: &(Ref, Ref)| if left { p.1 } else { p.0 };
    pairs.sort_unstable_by_key(|p| (other(p), slot(p)));
    let col = cols.get_mut(&m).expect("a column per summed-out subtree");
    let (mut w, mut i) = (0, 0);
    while i < pairs.len() {
        let o = other(&pairs[i]);
        let mut j = i + 1;
        while j < pairs.len() && other(&pairs[j]) == o {
            j += 1;
        }
        let x = if j == i + 1 {
            slot(&pairs[i])
        } else {
            let mut sum = col.vals[ix(slot(&pairs[i])) as usize].clone();
            for p in &pairs[i + 1..j] {
                let v = col.vals[ix(slot(p)) as usize].clone();
                sum.add_assign(&v);
            }
            mk(m, col.slot(sum, lim)?)
        };
        pairs[w] = if left { (x, o) } else { (o, x) };
        w += 1;
        i = j;
    }
    pairs.truncate(w);
    Ok(())
}

/// Whether `r`, a reference of `x`, is `true`: every node it reaches holds
/// one pair, and every leaf it reaches reads `One`.
fn is_top(x: &SkipForm, r: Ref, memo: &mut FxHashMap<Ref, bool>, lim: &Limits) -> Result<bool, OperationError> {
    let sh = &*x.shape;
    let known = |z: Ref, memo: &FxHashMap<Ref, bool>| if sh.is_leaf(vt(z)) { Some(ix(z) == ONE) } else { memo.get(&z).copied() };
    let mut stack: Vec<Ref> = Vec::new();
    lim.try_push(&mut stack, r)?;
    let mut top = true;
    while let Some(&y) = stack.last() {
        if known(y, memo).is_some() {
            stack.pop();
            continue;
        }
        let node = x.levels[vt(y) as usize].node(ix(y));
        let ok = match node {
            &[(a, b)] => match (known(a, memo), known(b, memo)) {
                (Some(p), Some(q)) => Some(p && q),
                (Some(false), _) | (_, Some(false)) => Some(false),
                (pa, pb) => {
                    for (z, k) in [(a, pa), (b, pb)] {
                        if k.is_none() {
                            lim.try_push(&mut stack, z)?;
                        }
                    }
                    None
                }
            },
            _ => Some(false),
        };
        if let Some(ok) = ok {
            stack.pop();
            lim.reserve_map(memo, 1)?;
            memo.insert(y, ok);
            if !ok {
                top = false;
                break;
            }
        }
    }
    lim.discard(stack);
    Ok(top && known(r, memo) != Some(false))
}

/// `x` on `to`, which sums out every subtree `x` does and maybe more: a
/// reference into a subtree only `to` sums out must be `true`, and reads
/// `One` there.
///
/// # Errors
///
/// [`OperationError::MarginalLevel`] at such a subtree where `x` constrains
/// its variables.
fn reshape(x: &SkipForm, to: &Arc<Shape>, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*x.shape;
    let mut out = SkipForm::zero(Arc::clone(to));
    let mut memo: FxHashMap<Ref, bool> = FxHashMap::default();
    let mut remap: Vec<Vec<Ref>> = Vec::new();
    lim.try_resize(&mut remap, sh.num_nodes(), Vec::new())?;
    let mut map = |remap: &Vec<Vec<Ref>>, r: Ref| -> Result<Ref, OperationError> {
        if r == ZREF {
            return Ok(ZREF);
        }
        let v = vt(r);
        let m = to.owner[v as usize];
        if m != NONE && sh.kind[m as usize] < SUMMED_VAR {
            if !is_top(x, r, &mut memo, lim)? {
                return Err(OperationError::MarginalLevel(VtreeIdx(m)));
            }
            return Ok(mk(m, ONE));
        }
        Ok(if sh.is_leaf(v) { r } else { remap[v as usize][ix(r) as usize] })
    };
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    let mut gate = lim.gate();
    for &t in &to.bottomup {
        let lvl = &x.levels[t as usize];
        let mut refs = Vec::new();
        lim.reserve_exact(&mut refs, lvl.width())?;
        for i in 0..lvl.width() as u32 {
            let node = lvl.node(i);
            gate.poll(node.len() as u64)?;
            buf.clear();
            lim.reserve(&mut buf, node.len())?;
            for &(a, b) in node {
                buf.push((map(&remap, a)?, map(&remap, b)?));
            }
            let made = if to.weighted[t as usize] { normalize_sum(to, t, &mut buf) } else { normalize(to, t, &mut buf) };
            refs.push(match made {
                Err(target) => target,
                Ok(()) => mk(t, out.push_node(t, &buf, lim)?),
            });
        }
        remap[t as usize] = refs;
    }
    gate.flush()?;
    out.out = map(&remap, x.out)?;
    lim.discard(buf);
    lim.discard(memo);
    discard_nested(lim, remap);
    out.collect_pushed(lim)?;
    Ok(out)
}

/// Per node of `x`'s shape, whether `x` names a label of it other than
/// `One`.
fn slot_readers(x: &SkipForm, lim: &Limits) -> Result<Vec<bool>, OperationError> {
    let sh = &*x.shape;
    let mut reads = filled(lim, sh.num_nodes(), false)?;
    let mut note = |r: Ref| {
        if r != ZREF && ix(r) != ONE && sh.is_leaf(vt(r)) {
            reads[vt(r) as usize] = true;
        }
    };
    note(x.out);
    for &u in &x.active {
        for &(a, b) in &x.levels[u as usize].pairs {
            note(a);
            note(b);
        }
    }
    Ok(reads)
}

/// `f ∧ g` where either sums a subtree out, under `ws`, `f`'s weights, to
/// which the columns `g` brings from `gw` are added; then the subtrees of
/// `targets` summed out, and every node both of whose children are.
///
/// # Errors
///
/// [`OperationError::MarginalLevel`] at a subtree one operand sums out
/// where the other constrains its variables or both read its slots.
pub(crate) fn and(
    f: &SkipForm,
    g: &SkipForm,
    tree: &Vtree,
    ws: &mut WeightStore,
    gw: &WeightStore,
    targets: &[u32],
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let (fs, gs) = (&*f.shape, &*g.shape);
    let full = Shape::of(tree);
    let (f_roots, g_roots) = (roots(fs), roots(gs));
    let mut cand: Vec<u32> = f_roots.iter().chain(&g_roots).map(|&(m, _)| m).collect();
    cand.sort_unstable();
    cand.dedup();
    // The conjunction's leaves: the largest of both operands' subtrees,
    // each with the column of the operand that reads its slots. Which
    // labels an operand reads is found only where both sum a subtree out.
    let (mut f_reads, mut g_reads) = (None, None);
    let mut leaves: Vec<(u32, u32)> = Vec::new();
    for &m in &cand {
        if cand.iter().any(|&o| o != m && full.within(m, o)) {
            continue;
        }
        let v = m as usize;
        let (fo, go) = (fs.kind[v] >= SUMMED_VAR, gs.kind[v] >= SUMMED_VAR);
        let gr = go && {
            if fo && g_reads.is_none() {
                g_reads = Some(slot_readers(g, lim)?);
            }
            // A subtree only `g` sums out keeps `g`'s labels whatever it reads.
            !fo || g_reads.as_ref().is_some_and(|r| r[v])
        };
        if gr && fo {
            if f_reads.is_none() {
                f_reads = Some(slot_readers(f, lim)?);
            }
            if f_reads.as_ref().is_some_and(|r| r[v]) {
                return Err(OperationError::MarginalLevel(VtreeIdx(m)));
            }
        }
        let labels = if fo && !gr {
            fs.labels[v]
        } else {
            if gs.kind[v] == SUMMED {
                ws.set_level(v, gw.level(v).expect("a summed-out subtree has its column").to_vec());
            }
            gs.labels[v]
        };
        lim.try_push(&mut leaves, (m, labels))?;
    }
    lim.discard(cand);
    for v in [f_reads, g_reads].into_iter().flatten() {
        lim.discard(v);
    }
    for &(m, _) in &f_roots {
        if fs.kind[m as usize] == SUMMED && !leaves.iter().any(|&(o, _)| o == m) {
            ws.take_level(m as usize);
        }
    }
    // An operand already on the conjunction's shape is read as it is.
    let on = |own: &[(u32, u32)]| own.len() == leaves.len() && own.iter().zip(&leaves).all(|(a, b)| a == b);
    let (f_on, g_on) = (on(&f_roots), on(&g_roots));
    let shape = if f_on {
        Arc::clone(&f.shape)
    } else if g_on {
        Arc::clone(&g.shape)
    } else {
        Arc::new(Shape::summed(tree, &leaves))
    };
    lim.discard(leaves);
    let placed = |x: &SkipForm, on: bool| -> Result<Option<SkipForm>, OperationError> {
        if on { Ok(None) } else { reshape(x, &shape, lim).map(Some) }
    };
    let (f2, g2) = (placed(f, f_on)?, placed(g, g_on)?);
    // The operand on more levels first: outside the other's cone the
    // conjunction is its levels as they are, canonical where it was read
    // as it is.
    let (fa, ga) = (f2.as_ref().unwrap_or(f), g2.as_ref().unwrap_or(g));
    let ((a, a_on), b) = if fa.active.len() < ga.active.len() { ((ga, g_on), fa) } else { ((fa, f_on), ga) };
    let (out, cone) = conjoin::and_scoped(a, b, lim)?;
    for x in [f2, g2].into_iter().flatten() {
        x.discard(lim);
    }
    // The conjunction keeps every product, so equal nodes are shared here.
    let closed = sum_out(&out, tree, ws, targets, true, a_on.then_some(&cone[..]), lim);
    lim.discard(cone);
    Ok(match closed? {
        Some(closed) => {
            out.discard(lim);
            closed
        }
        None => out,
    })
}
