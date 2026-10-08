//! The inner product of two count-marginal diagrams, counted without building
//! their product.
//!
//! A diagram whose levels have been summed out ([`Tdd::marginalize_levels`],
//! [`Engine::and_marginalizing`]) denotes a count over the variables it kept.
//! [`Engine::marginal_product_count`] multiplies two such counts pointwise and
//! sums the product, with the two diagrams' summed-out variables independent:
//! where the conjunction of two diagrams that summed out the same scope keeps
//! that scope's count once, this product multiplies the two counts.
//!
//! The walk is the product construction of a conjunction, reduced to a value
//! per pair of nodes: for vtree node `t` it holds `M_t(u, v)`, the inner
//! product of node `u` of one operand with node `v` of the other over the
//! variables both kept below `t`. Three shapes of `M_t` are stored without
//! enumerating its entries:
//!
//! - `Diag`: both operands are one diagram, structural below `t`. Distinct
//!   nodes of a level denote disjoint functions, so `M_t(u, v)` is `u`'s
//!   count when `u = v` and zero otherwise.
//! - `Factor`: `M_t(u, v) = a(u) · b(v)`. A level both operands summed out
//!   holds values only, and a level one summed out while the other is
//!   constant there is its values times the other's constants.
//! - `Sparse`: every nonzero entry, keyed by the node pair.
//!
//! A structural level combines its children's shapes through its pairs; a
//! `Factor` child is contracted over its side before the other side's entries
//! are joined, so a level with a summed-out child costs its pairs, not the
//! product of its children's widths.

use num_bigint::BigUint;
use rustc_hash::FxHashMap;

use crate::diagram::{ChildDecoder, ChildRef, EncodedChildRef, LeafLabel, Tdd, TddLevel, ValueRef, LEAF_WIDTH};
use crate::limits::{Limits, OperationError, PollGate, Transient};
use crate::value::CountRead;
use crate::vtree::{VtreeIdx, VtreeNode};
use crate::Engine;

impl Engine {
    /// Sum, over the assignments that neither operand has summed out, of the
    /// product of the two operands' counts, with the variables each operand
    /// summed out counted independently.
    ///
    /// A diagram `T` whose marginal levels cover the variables `Y_T` denotes
    /// the count `m_T(x)` of each assignment `x` to the other variables `X_T`:
    /// the number of assignments to `Y_T` that extend `x` to a model of the
    /// function `T` held before those levels were summed out. A structural
    /// diagram's count is its indicator. The result is
    ///
    /// `Σ_{x ∈ {0,1}^(X_l ∩ X_r)} m_l(x) · m_r(x)`,
    ///
    /// where `m_l` is read at any value of the variables only `r` summed out,
    /// and `m_r` at any value of those only `l` summed out; each operand must
    /// not depend on them. That is checked where the two differ: at a vtree
    /// node where one operand is marginal and the other is not, every node
    /// of the other that a pair reaches there must be built from the
    /// constant-true leaf and summed-out values alone.
    ///
    /// - Over structural operands this is the model count of `l ∧ r`.
    /// - Where the two operands summed out disjoint sets of variables it is
    ///   [`Engine::and_model_count`]`(l, r, &[])`: the product of `m_l` and
    ///   `m_r` is the count of the conjunction.
    /// - Where they summed out the same variables it is `Σ_x m_l(x) · m_r(x)`,
    ///   which the conjunction does not compute: it keeps a scope both summed
    ///   out once. With `l` and `r` one diagram it is the sum of squares of its
    ///   counts, `Σ_x m(x)²`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// // x1 ∨ x2 on the left of the root, x3 ∨ x4 on the right: summing the
    /// // right out leaves a count of 3 on each model of x1 ∨ x2.
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (_, right) = vtree.children(vtree.root());
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// let counted = engine.and_marginalizing(f, g, &[right])?;
    /// assert_eq!(engine.model_count(&counted)?, 9u32.into());
    /// // Three models of x1 ∨ x2, each counted 3 · 3 times.
    /// assert_eq!(engine.marginal_product_count(&counted, &counted)?, 27u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Exactness
    ///
    /// For a vtree node `t`, node `u` of `l` and node `v` of `r` at `t`, let
    /// `M_t(u, v)` be the sum above restricted to `t`'s variables. A node's
    /// count factors over its pairs, `m_u(x_L, x_R) = Σ_{(p, q) ∈ u} m_p(x_L)
    /// · m_q(x_R)`, since its pairs denote disjoint products over the two
    /// children's disjoint variables, and the assignments both operands kept
    /// below `t` are the product of those kept below each child. Multiplying
    /// two such sums and summing over that product gives
    /// `M_t(u, v) = Σ_{(p, q) ∈ u, (p', q') ∈ v} M_L(p, p') · M_R(q, q')`.
    /// At a node both operands summed out no variable is left, so
    /// `M_t(u, v) = c_u · c_v`, the two stored counts. At a node one summed
    /// out, the other's count does not depend on the variables it kept there,
    /// so `M_t(u, v) = c_u · κ_v`, `κ_v` its constant value: the product of
    /// the summed-out values below it. At a leaf the entries are the counts of
    /// `u ∧ v` over one variable. The result is `M` at the root for the two
    /// output nodes, and each shape the walk stores is one of these identities
    /// written without its zero entries.
    ///
    /// # Cost
    ///
    /// Passing one diagram twice is recognized, and its structural levels are
    /// then read once, as a model count reads them. The level above a summed-out
    /// child costs its pairs. Elsewhere the walk is a conjunction's product
    /// construction without its nodes: its work and memory grow with the
    /// pairs of nodes whose product is satisfiable.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] for different vtree allocations,
    /// [`OperationError::IncompatibleWeights`] when either operand has weights
    /// attached, [`OperationError::MarginalLevel`] at a level one operand
    /// summed out where the other still depends on the variables below it,
    /// [`OperationError::OverBudget`] when a buffer is refused, and
    /// [`OperationError::Stopped`] on cancellation. The operands are borrowed
    /// and unchanged.
    pub fn marginal_product_count(&self, l: &Tdd, r: &Tdd) -> Result<BigUint, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        crate::apply::check_vtree(l, r)?;
        if l.weights.is_some() || r.weights.is_some()
            || l.levels.iter().chain(r.levels.iter()).any(TddLevel::is_weight_marginal)
        {
            return Err(OperationError::IncompatibleWeights);
        }
        let same = std::ptr::eq(l, r) || same_diagram(l, r);
        match product::<u128>(lim, l, r, same) {
            Ok(v) => Ok(BigUint::from(v)),
            Err(Halt::Overflow) => match product::<BigUint>(lim, l, r, same) {
                Ok(v) => Ok(v),
                Err(Halt::Overflow) => unreachable!("an exact count does not overflow"),
                Err(Halt::Refused(e)) => Err(e),
            },
            Err(Halt::Refused(e)) => Err(e),
        }
    }
}

/// Why a pass under one arithmetic ended early.
enum Halt {
    /// A value outgrew the arithmetic; the pass is repeated exactly.
    Overflow,
    /// The operation was refused.
    Refused(OperationError),
}

impl From<OperationError> for Halt {
    fn from(e: OperationError) -> Self {
        Halt::Refused(e)
    }
}

/// The counts the walk multiplies: `u128`, checked, then exact.
trait Arith: Clone + Default {
    fn small(c: u128) -> Self;
    fn stored(read: CountRead<'_>) -> Result<Self, Halt>;
    fn plus(&mut self, other: &Self) -> Result<(), Halt>;
    fn times(&self, other: &Self) -> Result<Self, Halt>;
    fn is_zero(&self) -> bool;
}

impl Arith for u128 {
    fn small(c: u128) -> Self {
        c
    }
    fn stored(read: CountRead<'_>) -> Result<Self, Halt> {
        match read {
            CountRead::Fast(c) => Ok(c),
            CountRead::Big(_) => Err(Halt::Overflow),
        }
    }
    #[inline]
    fn plus(&mut self, other: &Self) -> Result<(), Halt> {
        *self = self.checked_add(*other).ok_or(Halt::Overflow)?;
        Ok(())
    }
    #[inline]
    fn times(&self, other: &Self) -> Result<Self, Halt> {
        self.checked_mul(*other).ok_or(Halt::Overflow)
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl Arith for BigUint {
    fn small(c: u128) -> Self {
        BigUint::from(c)
    }
    fn stored(read: CountRead<'_>) -> Result<Self, Halt> {
        Ok(match read {
            CountRead::Fast(c) => BigUint::from(c),
            CountRead::Big(b) => b.clone(),
        })
    }
    fn plus(&mut self, other: &Self) -> Result<(), Halt> {
        *self += other;
        Ok(())
    }
    fn times(&self, other: &Self) -> Result<Self, Halt> {
        Ok(self * other)
    }
    fn is_zero(&self) -> bool {
        *self == BigUint::ZERO
    }
}

/// `M_t` for one vtree node, in the shape the walk stores it.
enum Form<'a, V: Arith> {
    /// `M_t(u, v) = [u = v] · d[u]`: one diagram, structural below `t`.
    Diag(Transient<'a, Vec<V>>),
    /// `M_t(u, v) = a[u] · b[v]`, each indexed by the node or value slot a
    /// side names; a side carrying its value inline is read as that value.
    Factor(Transient<'a, Vec<V>>, Transient<'a, Vec<V>>),
    /// Every nonzero `M_t(u, v)`, keyed `u << 32 | v`, sorted by key.
    Sparse(Transient<'a, Vec<(u64, V)>>),
}

/// What a pair side names in its child level: a node or value slot, or a
/// value carried in the side itself.
#[derive(Clone, Copy)]
enum Side {
    At(u32),
    Inline(u32),
}

#[inline]
fn side(view: ChildDecoder, s: EncodedChildRef) -> Side {
    match view.child(s) {
        ChildRef::Node(n) => Side::At(n.0),
        ChildRef::Value(ValueRef::Slot(j)) => Side::At(j),
        ChildRef::Value(ValueRef::Inline(k)) => Side::Inline(k),
    }
}

/// The index a side into a structural child names.
#[inline]
fn node(view: ChildDecoder, s: EncodedChildRef) -> u32 {
    match side(view, s) {
        Side::At(i) => i,
        Side::Inline(_) => unreachable!("a structural child's side names a node"),
    }
}

/// `vals` read at a side: its slot, or its inline value.
#[inline]
fn value_at<V: Arith>(vals: &[V], view: ChildDecoder, s: EncodedChildRef) -> V {
    match side(view, s) {
        Side::At(i) => vals[i as usize].clone(),
        Side::Inline(k) => V::small(u128::from(k)),
    }
}

#[inline]
fn key(u: u32, v: u32) -> u64 {
    (u64::from(u) << 32) | u64::from(v)
}

/// The entries of a sorted `Sparse` form whose row is `u`.
fn row<V>(entries: &[(u64, V)], u: u32) -> &[(u64, V)] {
    let lo = entries.partition_point(|e| e.0 < key(u, 0));
    let hi = lo + entries[lo..].partition_point(|e| (e.0 >> 32) as u32 == u);
    &entries[lo..hi]
}

/// Whether two diagrams hold the same levels and output, so that the walk
/// may read them as one.
fn same_diagram(l: &Tdd, r: &Tdd) -> bool {
    l.output == r.output
        && l.levels.iter().zip(r.levels.iter()).all(|(a, b)| {
            a.is_marginal() == b.is_marginal()
                && a.nodes == b.nodes
                && a.pairs == b.pairs
                && a.ranges == b.ranges
                && a.marginal_counts() == b.marginal_counts()
                && match (a.count_column(), b.count_column()) {
                    (Some(x), Some(y)) => (0..a.slot_count()).all(|i| x.get(i).to_count() == y.get(i).to_count()),
                    (x, y) => x.is_none() && y.is_none(),
                }
        })
}

/// A marginal level's values as the walk's counts.
fn column<'a, V: Arith>(lim: &'a Limits, level: &TddLevel, gate: &mut PollGate) -> Result<Transient<'a, Vec<V>>, Halt> {
    let col = level.count_column().expect("a count-marginal level");
    let n = level.slot_count();
    let mut out = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut out, n)?;
    for i in 0..n {
        out.push(V::stored(col.get(i))?);
    }
    gate.poll(n as u64)?;
    Ok(out)
}

/// The constant value of each node of `tdd` at `t`, or `None` for a node
/// whose count depends on a variable `tdd` kept below `t`.
///
/// A node is constant when every pair's two sides are: the constant-true
/// leaf (value 1), a summed-out value, or a constant node. Its value is the
/// sum over its pairs of the two sides' products, which is then its count at
/// every assignment of the variables it kept.
fn constants<'a, V: Arith>(
    lim: &'a Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    gate: &mut PollGate,
) -> Result<Transient<'a, Vec<Option<V>>>, Halt> {
    let vtree = tdd.vtree();
    // Children before parents, stopping at marginal levels.
    let mut order = Transient::new(lim, Vec::new());
    let mut stack = Transient::new(lim, Vec::new());
    lim.try_push(&mut stack, t)?;
    while let Some(s) = stack.pop() {
        lim.try_push(&mut order, s)?;
        if let VtreeNode::Internal { left, right, .. } = vtree.node(s)
            && !tdd.levels[s.idx()].is_marginal()
        {
            lim.try_push(&mut stack, *left)?;
            lim.try_push(&mut stack, *right)?;
        }
        gate.poll(1)?;
    }
    let mut vals: FxHashMap<u32, Transient<'a, Vec<Option<V>>>> = FxHashMap::default();
    for &s in order.iter().rev() {
        let level = &tdd.levels[s.idx()];
        let mut out = Transient::new(lim, Vec::new());
        if level.is_marginal() {
            let col = column::<V>(lim, level, gate)?;
            lim.reserve_exact(&mut out, col.len())?;
            out.extend(col.iter().cloned().map(Some));
        } else if vtree.node(s).is_leaf() {
            lim.reserve_exact(&mut out, LEAF_WIDTH)?;
            for i in 0..LEAF_WIDTH {
                out.push((LeafLabel::from_idx(i) == LeafLabel::One).then(|| V::small(1)));
            }
        } else {
            let (left, right) = vtree.children(s);
            let lv = vals.remove(&(left.0)).expect("a child is folded before its parent");
            let rv = vals.remove(&(right.0)).expect("a child is folded before its parent");
            let (ld, rd) = (tdd.levels[left.idx()].child_decoder(), tdd.levels[right.idx()].child_decoder());
            let at = |vals: &[Option<V>], view: ChildDecoder, e: EncodedChildRef| match side(view, e) {
                Side::At(i) => vals[i as usize].clone(),
                Side::Inline(k) => Some(V::small(u128::from(k))),
            };
            lim.reserve_exact(&mut out, level.nodes().len())?;
            for (_, pairs) in level.internal_inputs_iter() {
                gate.poll(pairs.len() as u64 + 1)?;
                let mut acc = V::default();
                let mut constant = true;
                for p in pairs {
                    let (Some(a), Some(b)) = (at(&lv, ld, p.left), at(&rv, rd, p.right)) else {
                        constant = false;
                        break;
                    };
                    acc.plus(&a.times(&b)?)?;
                }
                out.push(constant.then_some(acc));
            }
        }
        vals.insert(s.0, out);
    }
    Ok(vals.remove(&t.0).expect("the subtree's root is folded"))
}

/// Per-operand facts the walk reads: which levels are marginal, and which
/// subtrees hold no marginal level.
struct Shape<'a> {
    marginal: Transient<'a, Vec<bool>>,
    pure: Transient<'a, Vec<bool>>,
}

fn shape<'a>(lim: &'a Limits, tdd: &Tdd) -> Result<Shape<'a>, OperationError> {
    let vtree = tdd.vtree();
    let n = vtree.num_nodes();
    let mut marginal = Transient::new(lim, Vec::new());
    lim.try_resize(&mut marginal, n, false)?;
    let mut pure = Transient::new(lim, Vec::new());
    lim.try_resize(&mut pure, n, false)?;
    for t in vtree.bottomup() {
        marginal[t.idx()] = tdd.levels[t.idx()].is_marginal();
        pure[t.idx()] = !marginal[t.idx()]
            && match vtree.node(t) {
                VtreeNode::Leaf { .. } => true,
                VtreeNode::Internal { left, right, .. } => pure[left.idx()] && pure[right.idx()],
            };
    }
    Ok(Shape { marginal, pure })
}

/// One operand's level at a structural vtree node, with its children's
/// decoders and the nodes the walk reads (all, or only the output).
struct Level<'t> {
    level: &'t TddLevel,
    left: ChildDecoder,
    right: ChildDecoder,
    only: Option<u32>,
}

impl<'t> Level<'t> {
    fn new(tdd: &'t Tdd, t: VtreeIdx, only: Option<u32>) -> Self {
        let (l, r) = tdd.vtree().children(t);
        Level {
            level: &tdd.levels[t.idx()],
            left: tdd.levels[l.idx()].child_decoder(),
            right: tdd.levels[r.idx()].child_decoder(),
            only,
        }
    }

    /// Every `(node, pairs)` the walk reads.
    fn nodes(&self) -> impl Iterator<Item = (u32, crate::diagram::PairsIter<'t>)> + '_ {
        let level = self.level;
        let range = match self.only {
            Some(u) => u as usize..u as usize + 1,
            None => 0..level.nodes().len(),
        };
        range.map(move |i| (i as u32, level.pairs_iter_of_idx(i)))
    }
}

/// `Σ_{x ∈ {0,1}^(X_l ∩ X_r)} m_l(x) · m_r(x)` under arithmetic `V`.
fn product<V: Arith>(lim: &Limits, l: &Tdd, r: &Tdd, same: bool) -> Result<V, Halt> {
    if l.is_zero() || r.is_zero() {
        return Ok(V::default());
    }
    let vtree = l.vtree();
    let n = vtree.num_nodes();
    let root = vtree.root();
    let mut gate = lim.gate();
    let (sl, sr) = (shape(lim, l)?, shape(lim, r)?);
    // A node below a level either operand summed out is never read: the
    // summed-out level's values stand for its whole subtree.
    let mut covered = Transient::new(lim, Vec::new());
    lim.try_resize(&mut covered, n, false)?;
    for t in vtree.bottomup().rev() {
        if let Some(p) = vtree.node(t).parent() {
            covered[t.idx()] = covered[p.idx()] || sl.marginal[p.idx()] || sr.marginal[p.idx()];
        }
    }
    let (out_l, out_r) = (l.output.local.0, r.output.local.0);
    let mut forms: Transient<'_, Vec<Option<Form<'_, V>>>> = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut forms, n)?;
    forms.resize_with(n, || None);
    for t in vtree.bottomup() {
        let ti = t.idx();
        if covered[ti] {
            continue;
        }
        let (ml, mr) = (sl.marginal[ti], sr.marginal[ti]);
        let form = if ml || mr {
            cut(lim, l, r, t, ml, mr, &mut gate)?
        } else if vtree.node(t).is_leaf() {
            leaf(lim, same)?
        } else if same && sl.pure[ti] {
            let (left, right) = vtree.children(t);
            let (Some(Form::Diag(dl)), Some(Form::Diag(dr))) = (forms[left.idx()].take(), forms[right.idx()].take()) else {
                unreachable!("a structural subtree's children are diagonal")
            };
            Form::Diag(diag(lim, &Level::new(l, t, None), &dl, &dr, &mut gate)?)
        } else {
            let (left, right) = vtree.children(t);
            let fl = forms[left.idx()].take().expect("a child's form precedes its parent's");
            let fr = forms[right.idx()].take().expect("a child's form precedes its parent's");
            let only = t == root;
            let ll = Level::new(l, t, only.then_some(out_l));
            let lr = Level::new(r, t, only.then_some(out_r));
            if only && (matches!(fl, Form::Factor(..)) ^ matches!(fr, Form::Factor(..))) {
                let (lw, rw) = (l.reference_slot_count(left), r.reference_slot_count(left));
                let (lw, rw) = if matches!(fr, Form::Factor(..)) { (lw, rw) } else {
                    (l.reference_slot_count(right), r.reference_slot_count(right))
                };
                let count = root_contract(lim, &ll, &lr, &fl, &fr, (lw, rw), &mut gate)?;
                gate.finish()?;
                return Ok(count);
            }
            let entries = combine(lim, &ll, &lr, &fl, &fr, &mut gate)?;
            if only {
                gate.finish()?;
                return Ok(match entries {
                    Combined::Factor(a, b) => a[out_l as usize].times(&b[out_r as usize])?,
                    Combined::Sparse(e) => e.first().map_or_else(V::default, |(_, v)| v.clone()),
                });
            }
            match entries {
                Combined::Factor(a, b) => Form::Factor(a, b),
                Combined::Sparse(e) => Form::Sparse(e),
            }
        };
        forms[ti] = Some(form);
    }
    gate.finish()?;
    // The root is a leaf, a summed-out level or diagonal.
    Ok(match forms[root.idx()].take().expect("the root is read") {
        Form::Diag(d) => if out_l == out_r { d[out_l as usize].clone() } else { V::default() },
        Form::Factor(a, b) => a[out_l as usize].times(&b[out_r as usize])?,
        Form::Sparse(e) => row(&e, out_l).iter().find(|x| x.0 == key(out_l, out_r)).map_or_else(V::default, |x| x.1.clone()),
    })
}

/// `M_t` at a leaf both operands keep: the count of `u ∧ v` over its
/// variable.
fn leaf<'a, V: Arith>(lim: &'a Limits, same: bool) -> Result<Form<'a, V>, OperationError> {
    let (one, pos, neg) = (LeafLabel::One as u32, LeafLabel::Pos as u32, LeafLabel::Neg as u32);
    if same {
        // One diagram's leaf references are all the constant or all
        // literals, so its referenced nodes are pairwise disjoint.
        let mut d = Transient::new(lim, Vec::new());
        lim.try_resize(&mut d, LEAF_WIDTH, V::default())?;
        d[one as usize] = V::small(2);
        d[pos as usize] = V::small(1);
        d[neg as usize] = V::small(1);
        return Ok(Form::Diag(d));
    }
    let mut e = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut e, 7)?;
    for (u, v, c) in [(one, one, 2), (one, pos, 1), (one, neg, 1), (pos, one, 1), (pos, pos, 1), (neg, one, 1), (neg, neg, 1)] {
        e.push((key(u, v), V::small(c)));
    }
    e.sort_unstable_by_key(|x| x.0);
    Ok(Form::Sparse(e))
}

/// `M_t` at a level one or both operands summed out.
fn cut<'a, V: Arith>(
    lim: &'a Limits,
    l: &Tdd,
    r: &Tdd,
    t: VtreeIdx,
    ml: bool,
    mr: bool,
    gate: &mut PollGate,
) -> Result<Form<'a, V>, Halt> {
    let values = |tdd: &Tdd, gate: &mut PollGate| column::<V>(lim, &tdd.levels[t.idx()], gate);
    let constant = |tdd: &Tdd, gate: &mut PollGate| -> Result<Transient<'a, Vec<V>>, Halt> {
        let k = constants::<V>(lim, tdd, t, gate)?;
        refuse_unless_constant(tdd, t, &k)?;
        let mut out = Transient::new(lim, Vec::new());
        lim.reserve_exact(&mut out, k.len())?;
        out.extend(k.iter().map(|c| c.clone().unwrap_or_default()));
        Ok(out)
    };
    Ok(match (ml, mr) {
        (true, true) => Form::Factor(values(l, gate)?, values(r, gate)?),
        (true, false) => Form::Factor(values(l, gate)?, constant(r, gate)?),
        (false, true) => Form::Factor(constant(l, gate)?, values(r, gate)?),
        (false, false) => unreachable!("a cut is marginal in one operand"),
    })
}

/// Refuse `tdd` at `t` when a pair of its parent level, or its output,
/// reaches a node of `t` that is not constant.
fn refuse_unless_constant<V>(tdd: &Tdd, t: VtreeIdx, k: &[Option<V>]) -> Result<(), OperationError> {
    let vtree = tdd.vtree();
    let reached_constant = |i: u32| k.get(i as usize).is_some_and(Option::is_some);
    let ok = match vtree.node(t).parent() {
        None => reached_constant(tdd.output.local.0),
        Some(p) => {
            let view = tdd.levels[t.idx()].child_decoder();
            let on_left = vtree.children(p).0 == t;
            tdd.levels[p.idx()].internal_inputs_iter().all(|(_, pairs)| {
                pairs.into_iter().all(|pair| reached_constant(node(view, if on_left { pair.left } else { pair.right })))
            })
        }
    };
    if ok { Ok(()) } else { Err(OperationError::MarginalLevel(t)) }
}

/// `M_t` of one structural diagram from its children's diagonals: each
/// node's count.
fn diag<'a, V: Arith>(
    lim: &'a Limits,
    level: &Level<'_>,
    dl: &[V],
    dr: &[V],
    gate: &mut PollGate,
) -> Result<Transient<'a, Vec<V>>, Halt> {
    let mut d = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut d, level.level.nodes().len())?;
    for (_, pairs) in level.nodes() {
        gate.poll(pairs.len() as u64 + 1)?;
        let mut acc = V::default();
        for p in pairs {
            let a = &dl[node(level.left, p.left) as usize];
            let b = &dr[node(level.right, p.right) as usize];
            acc.plus(&a.times(b)?)?;
        }
        d.push(acc);
    }
    Ok(d)
}

/// What [`combine`] produced: a factored `M_t`, or its nonzero entries.
enum Combined<'a, V: Arith> {
    Factor(Transient<'a, Vec<V>>, Transient<'a, Vec<V>>),
    Sparse(Transient<'a, Vec<(u64, V)>>),
}

/// The nonzero entries of one row of a child's `M`, as `(column, value)`.
enum Row<'f, V: Arith> {
    One(u32, &'f V),
    Many(&'f [(u64, V)]),
}

impl<'f, V: Arith> Row<'f, V> {
    fn of(form: &'f Form<'_, V>, u: u32) -> Self {
        match form {
            Form::Diag(d) => Row::One(u, &d[u as usize]),
            Form::Sparse(e) => Row::Many(row(e, u)),
            Form::Factor(..) => unreachable!("a factored child is contracted, not read by rows"),
        }
    }

    fn for_each(&self, mut f: impl FnMut(u32, &V) -> Result<(), Halt>) -> Result<(), Halt> {
        match self {
            Row::One(v, w) => f(*v, w),
            Row::Many(e) => e.iter().try_for_each(|(k, w)| f(*k as u32, w)),
        }
    }
}

/// Group one operand's pairs at a level by the node one side names, summing
/// the other side's factored values: `(side node, level node) → Σ value`,
/// sorted by side node.
fn contract<'a, V: Arith>(
    lim: &'a Limits,
    level: &Level<'_>,
    vals: &[V],
    keep_left: bool,
    gate: &mut PollGate,
) -> Result<Transient<'a, Vec<(u64, V)>>, Halt> {
    let mut acc: Transient<'a, FxHashMap<u64, V>> = Transient::new(lim, FxHashMap::default());
    for (u, pairs) in level.nodes() {
        gate.poll(pairs.len() as u64 + 1)?;
        lim.reserve_map(&mut acc, pairs.len())?;
        for p in pairs {
            let (kept, summed, view) = if keep_left {
                (node(level.left, p.left), p.right, level.right)
            } else {
                (node(level.right, p.right), p.left, level.left)
            };
            acc.entry(key(kept, u)).or_default().plus(&value_at(vals, view, summed))?;
        }
    }
    let mut out = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut out, acc.len())?;
    out.extend(acc.drain());
    out.sort_unstable_by_key(|e| e.0);
    Ok(out)
}

/// `M_t(out_l, out_r)` at the root when exactly one child is factored: each
/// output's pairs contracted over the factored side into a dense column over
/// the other child's nodes, `A(p) = Σ_{(p, q) ∈ out_l} a(q)` and likewise
/// `B` for `out_r`, and then `Σ_{p, p'} A(p) · M(p, p') · B(p')` over the
/// other child's entries. `widths` are the two operands' slot counts at the
/// kept child.
fn root_contract<V: Arith>(
    lim: &Limits,
    ll: &Level<'_>,
    lr: &Level<'_>,
    fl: &Form<'_, V>,
    fr: &Form<'_, V>,
    widths: (usize, usize),
    gate: &mut PollGate,
) -> Result<V, Halt> {
    let right_factored = matches!(fr, Form::Factor(..));
    let (Form::Factor(a, b), other) = (if right_factored { (fr, fl) } else { (fl, fr) }) else {
        unreachable!("one child is factored")
    };
    let dense = |level: &Level<'_>, vals: &[V], width: usize, gate: &mut PollGate| -> Result<Transient<'_, Vec<V>>, Halt> {
        let mut col = Transient::new(lim, Vec::new());
        lim.try_resize(&mut col, width, V::default())?;
        for (_, pairs) in level.nodes() {
            gate.poll(pairs.len() as u64 + 1)?;
            for p in pairs {
                let (kept, summed, view) = if right_factored {
                    (node(level.left, p.left), p.right, level.right)
                } else {
                    (node(level.right, p.right), p.left, level.left)
                };
                col[kept as usize].plus(&value_at(vals, view, summed))?;
            }
        }
        Ok(col)
    };
    let ca = dense(ll, a, widths.0, gate)?;
    let cb = dense(lr, b, widths.1, gate)?;
    let mut sum = V::default();
    gate.poll(ca.len() as u64)?;
    for (p, x) in ca.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        let p = p as u32;
        Row::of(other, p).for_each(|p2, w| {
            let y = &cb[p2 as usize];
            if !y.is_zero() {
                sum.plus(&x.times(w)?.times(y)?)?;
            }
            Ok(())
        })?;
    }
    Ok(sum)
}

/// `M_t` at a level both operands keep, from its children's forms.
fn combine<'a, V: Arith>(
    lim: &'a Limits,
    ll: &Level<'_>,
    lr: &Level<'_>,
    fl: &Form<'_, V>,
    fr: &Form<'_, V>,
    gate: &mut PollGate,
) -> Result<Combined<'a, V>, Halt> {
    // Both children factored: so is the level, one sum per node.
    if let (Form::Factor(al, bl), Form::Factor(ar, br)) = (fl, fr) {
        let side_sums = |level: &Level<'_>, a: &[V], b: &[V], gate: &mut PollGate| -> Result<Transient<'a, Vec<V>>, Halt> {
            let mut out = Transient::new(lim, Vec::new());
            lim.try_resize(&mut out, level.level.nodes().len(), V::default())?;
            for (u, pairs) in level.nodes() {
                gate.poll(pairs.len() as u64 + 1)?;
                let mut acc = V::default();
                for p in pairs {
                    acc.plus(&value_at(a, level.left, p.left).times(&value_at(b, level.right, p.right))?)?;
                }
                out[u as usize] = acc;
            }
            Ok(out)
        };
        return Ok(Combined::Factor(side_sums(ll, al, ar, gate)?, side_sums(lr, bl, br, gate)?));
    }
    let mut acc: Transient<'a, FxHashMap<u64, V>> = Transient::new(lim, FxHashMap::default());
    let add = |acc: &mut Transient<'a, FxHashMap<u64, V>>, u: u32, v: u32, w: V| -> Result<(), Halt> {
        if acc.len() == acc.capacity() {
            let grow = acc.len().max(16);
            lim.reserve_map(acc, grow)?;
        }
        acc.entry(key(u, v)).or_default().plus(&w)?;
        Ok(())
    };
    match (fl, fr) {
        // One child factored: contract it over its side, then join the
        // other side's entries through the nodes they name.
        (_, Form::Factor(a, b)) | (Form::Factor(a, b), _) => {
            let right_factored = matches!(fr, Form::Factor(..));
            let other = if right_factored { fl } else { fr };
            let gl = contract(lim, ll, a, right_factored, gate)?;
            let gr = contract(lim, lr, b, right_factored, gate)?;
            for &(k, ref x) in gl.iter() {
                let (p, u) = ((k >> 32) as u32, k as u32);
                gate.poll(1)?;
                Row::of(other, p).for_each(|p2, w| {
                    let xw = x.times(w)?;
                    for &(k2, ref y) in row(&gr, p2) {
                        add(&mut acc, u, k2 as u32, xw.times(y)?)?;
                    }
                    Ok(())
                })?;
            }
        }
        // Neither: every pair of `l` meets the pairs of `r` whose children
        // pair with its own in the children's entries.
        _ => {
            // `r`'s pairs by their two children. Over summed-out storage a
            // node may repeat a pair, and each copy counts.
            let mut pairs_r: Transient<'a, Vec<(u64, u32)>> = Transient::new(lim, Vec::new());
            for (v, pairs) in lr.nodes() {
                gate.poll(pairs.len() as u64 + 1)?;
                for p in pairs {
                    lim.try_push(&mut pairs_r, (key(node(lr.left, p.left), node(lr.right, p.right)), v))?;
                }
            }
            pairs_r.sort_unstable();
            let mut index: Transient<'a, FxHashMap<u64, (u32, u32)>> = Transient::new(lim, FxHashMap::default());
            let mut start = 0;
            while start < pairs_r.len() {
                let k = pairs_r[start].0;
                let end = start + pairs_r[start..].partition_point(|e| e.0 == k);
                if index.len() == index.capacity() {
                    let grow = index.len().max(16);
                    lim.reserve_map(&mut index, grow)?;
                }
                index.insert(k, (start as u32, (end - start) as u32));
                start = end;
            }
            for (u, pairs) in ll.nodes() {
                gate.poll(pairs.len() as u64 + 1)?;
                for p in pairs {
                    let (rows_l, rows_r) = (Row::of(fl, node(ll.left, p.left)), Row::of(fr, node(ll.right, p.right)));
                    rows_l.for_each(|p2, wl| {
                        rows_r.for_each(|q2, wr| {
                            if let Some(&(at, len)) = index.get(&key(p2, q2)) {
                                let w = wl.times(wr)?;
                                for &(_, v) in &pairs_r[at as usize..(at + len) as usize] {
                                    add(&mut acc, u, v, w.clone())?;
                                }
                            }
                            Ok(())
                        })
                    })?;
                }
            }
        }
    }
    let mut out = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut out, acc.len())?;
    out.extend(acc.drain().filter(|e| !e.1.is_zero()));
    out.sort_unstable_by_key(|e| e.0);
    Ok(Combined::Sparse(out))
}

#[cfg(test)]
#[path = "tests/product.rs"]
mod tests;
