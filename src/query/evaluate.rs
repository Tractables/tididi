//! Numeric evaluation with a supplied algebra or the diagram's attached weights.

use num_bigint::BigUint;
use num_traits::Zero;

use crate::value::{walk_bottom_up, CountRead, Retention, FoldInput, ValueDomain, WeightFold};
use crate::diagram::{ChildDecoder, ChildRef, ColumnAlgebra, EncodedChildRef, EvalAlgebra, InlineSlots, LeafLabel, NodeIdx, PairsIter, SlotPairs, Tdd, TddLevel, ValueRef, WeightStore, WeightValue, LEAF_WIDTH};
use crate::Engine;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

use super::fold::{fold_bottom_up_from, fold_held, LevelFold, Side};
use crate::limits::{OperationError, PollGate};

impl Engine {
    /// Run [`Tdd::evaluate`](crate::Tdd::evaluate) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Column buffers are charged to the byte budget and released after their
    /// parent consumes them; allocations inside algebra values are not charged.
    /// Stops are checked at entry, at amortized node boundaries and before return.
    /// An individual algebra callback or node fold cannot be interrupted.
    /// Caller algebra panics propagate as described on the diagram method.
    pub fn evaluate<S: EvalAlgebra>(&self, tdd: &Tdd, algebra: &S) -> Result<S::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        require_counts(tdd)?;
        let mut gate = lim.gate();
        let result = if tdd.is_zero() {
            algebra.zero()
        } else {
            let root = tdd.vtree.root();
            let fold = Evaluate::counting(algebra, tdd, root)?;
            let held = beneath_marginal(tdd, root, tdd.output.vtree);
            let mut cols = Vec::new();
            lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
            cols.resize_with(tdd.vtree.num_nodes(), Vec::new);
            fold_bottom_up_from(&fold, self, tdd, &mut cols, &held, Retention::Frontier, &mut gate)?;
            cols[tdd.output.vtree.idx()].swap_remove(tdd.output.local.idx())
        };
        gate.finish()?;
        Ok(result)
    }

    /// Run [`Tdd::evaluate_columns`](crate::Tdd::evaluate_columns) using this batch's resource limits.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure the
    /// algebra cannot value, [`OperationError::Stopped`] on cancellation, or
    /// [`OperationError::OverBudget`] if the table of columns is refused.
    ///
    /// The columns themselves are the algebra's, and are not charged to the
    /// byte budget. Stops are checked at entry, at node boundaries and before
    /// return; an individual algebra callback cannot be interrupted. Caller
    /// algebra panics propagate as described on the diagram method.
    pub fn evaluate_columns<A: ColumnAlgebra>(&self, tdd: &Tdd, algebra: &A) -> Result<A::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        require_counts(tdd)?;
        let mut gate = lim.gate();
        let result = if tdd.is_zero() {
            algebra.zero()
        } else {
            let keep = tdd.output.vtree;
            let mut cols = fold_columns_under(self, algebra, tdd, tdd.vtree.root(), keep, &mut gate)?;
            algebra.read(keep, std::mem::take(&mut cols[keep.idx()]), tdd.output.local.idx())
        };
        gate.finish()?;
        Ok(result)
    }

    /// Run [`Tdd::evaluate_at`](crate::Tdd::evaluate_at) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`evaluate`](Self::evaluate), and
    /// [`OperationError::LevelNotInVtree`] for an `at` outside the diagram's
    /// vtree.
    pub fn evaluate_at<S: EvalAlgebra>(&self, tdd: &Tdd, at: VtreeIdx, algebra: &S) -> Result<Vec<S::Value>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if at.idx() >= tdd.vtree.num_nodes() {
            return Err(OperationError::LevelNotInVtree(at));
        }
        require_counts(tdd)?;
        let mut gate = lim.gate();
        let fold = Evaluate::counting(algebra, tdd, at)?;
        let held = beneath_marginal(tdd, at, at);
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
        cols.resize_with(tdd.vtree.num_nodes(), Vec::new);
        fold_held::<_, false>(&fold, self, tdd, at, at, &mut cols, &held, Retention::Frontier, &mut gate)?;
        let result = std::mem::take(&mut cols[at.idx()]);
        gate.finish()?;
        Ok(result)
    }

    /// Run [`Tdd::evaluate_columns_at`](crate::Tdd::evaluate_columns_at) using this batch's resource limits.
    ///
    /// # Errors
    ///
    /// As [`evaluate_columns`](Self::evaluate_columns), and
    /// [`OperationError::LevelNotInVtree`] for an `at` outside the diagram's
    /// vtree.
    pub fn evaluate_columns_at<A: ColumnAlgebra>(&self, tdd: &Tdd, at: VtreeIdx, algebra: &A) -> Result<A::Column, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if at.idx() >= tdd.vtree.num_nodes() {
            return Err(OperationError::LevelNotInVtree(at));
        }
        require_counts(tdd)?;
        let mut gate = lim.gate();
        let mut cols = fold_columns_under(self, algebra, tdd, at, at, &mut gate)?;
        let result = std::mem::take(&mut cols[at.idx()]);
        gate.finish()?;
        Ok(result)
    }

    /// Run [`Tdd::weighted_value`](crate::Tdd::weighted_value) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Stops are checked at entry, at amortized node boundaries and before return.
    /// Numeric payload allocations are outside the best-effort byte budget.
    pub fn weighted_value(&self, tdd: &Tdd) -> Result<Option<WeightValue>, OperationError> {
        let _op = self.limits().enter()?;
        let Some(ws) = tdd.weights.as_ref() else { return Ok(None); };
        let mut gate = self.limits().gate();
        let value = weighted_output_value(self, tdd, ws, &mut gate)?;
        gate.finish()?;
        Ok(Some(value))
    }
}

/// [`evaluate`] as an instance of the shared bottom-up walk.
pub(crate) struct Evaluate<'a, S: EvalAlgebra> {
    algebra: &'a S,
    pins: &'a [super::cache::PinState],
    /// The distinct counts the walked levels carry inline in their
    /// references to a marginal child, ascending, and the value of each.
    inline_counts: Vec<u32>,
    inline_values: Vec<S::Value>,
}

impl<'a, S: EvalAlgebra> Evaluate<'a, S> {
    /// The fold of a diagram without marginal levels, under `pins`.
    pub(crate) fn new(algebra: &'a S, pins: &'a [super::cache::PinState]) -> Self {
        Self { algebra, pins, inline_counts: Vec::new(), inline_values: Vec::new() }
    }

    /// The fold of the levels under `root`, marginal ones read through the
    /// algebra's [`count`](EvalAlgebra::count): every count the walked
    /// levels carry inline valued up front, so a pair borrows its value as
    /// it borrows a column's.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] when a count is carried inline and
    /// the algebra values none.
    fn counting(algebra: &'a S, tdd: &Tdd, root: VtreeIdx) -> Result<Self, OperationError> {
        let mut fold = Self::new(algebra, &[]);
        if !tdd.levels.iter().any(TddLevel::is_marginal) {
            return Ok(fold);
        }
        let mut counts = Vec::new();
        let mut first = None;
        for t in subtree(&tdd.vtree, root) {
            let inline = inline_counts(tdd, t, root);
            if !inline.is_empty() {
                first.get_or_insert(t);
                counts.extend(inline);
            }
        }
        counts.sort_unstable();
        counts.dedup();
        let mut n = BigUint::ZERO;
        let mut values = Vec::with_capacity(counts.len());
        for &c in &counts {
            set_count(&mut n, CountRead::Fast(u128::from(c)));
            let level = first.expect("a level carries the counts");
            values.push(algebra.count(&n).ok_or(OperationError::MarginalLevel(level))?);
        }
        fold.inline_counts = counts;
        fold.inline_values = values;
        Ok(fold)
    }

    /// One side of one pair, borrowed from the child's column, or from the
    /// values of the inline counts.
    fn child<'c>(&'c self, side: Side<'c, Vec<S::Value>>, r: EncodedChildRef) -> &'c S::Value {
        match side.view.child(r) {
            ChildRef::Node(NodeIdx(i)) | ChildRef::Value(ValueRef::Slot(i)) => &side.col[i as usize],
            ChildRef::Value(ValueRef::Inline(c)) => {
                let k = self.inline_counts.binary_search(&c).expect("every inline count is valued up front");
                &self.inline_values[k]
            }
        }
    }
}

impl<S: EvalAlgebra> LevelFold for Evaluate<'_, S> {
    type Value = S::Value;
    type Col = Vec<S::Value>;

    fn alloc(&self, eng: &Engine, width: usize) -> Result<Vec<S::Value>, OperationError> {
        let mut col = Vec::new();
        eng.limits().reserve_exact(&mut col, width)?;
        col.resize(width, self.algebra.zero());
        Ok(col)
    }

    fn release(&self, eng: &Engine, col: &mut Self::Col) {
        eng.limits().discard(std::mem::take(col));
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<S::Value>, i: usize, v: S::Value) -> Result<(), OperationError> {
        col[i] = v;
        Ok(())
    }

    fn leaf(&self, leaf: VtreeIdx, var: VarId, mut label: LeafLabel) -> S::Value {
        if let Some(pin) = self.pins.get(leaf.idx()).and_then(|pin| pin.value) {
            let observed = if pin { LeafLabel::Pos } else { LeafLabel::Neg };
            if label == LeafLabel::One { label = observed; }
            else if label != observed { return self.algebra.zero(); }
        }
        match label {
            LeafLabel::Zero => self.algebra.zero(),
            _ => self.algebra.leaf(var, label),
        }
    }

    /// Each stored count `n` as the algebra's `count(n)`. A marginal level
    /// keeps how many assignments to its subtree each node holds and nothing
    /// of which, so an algebra values it only as `n` identities; see
    /// *Marginal levels* on [`EvalAlgebra`]. The incremental evaluator never
    /// reaches here: it refuses a marginal level when it is built.
    fn marginal_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<S::Value>) -> Result<(), OperationError> {
        let counts = tdd.levels[t.idx()].count_column().ok_or(OperationError::MarginalLevel(t))?;
        let mut n = BigUint::ZERO;
        for (i, slot) in col.iter_mut().enumerate().take(counts.len()) {
            set_count(&mut n, counts.get(i));
            *slot = self.algebra.count(&n).ok_or(OperationError::MarginalLevel(t))?;
        }
        Ok(())
    }

    /// `Σ over pairs (left × right)`, reading both children's values in
    /// place: the algebra's `sum_of_products`, by default a `mul_add` per
    /// pair.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<S::Value>>,
        right: Side<'_, Vec<S::Value>>,
    ) -> S::Value {
        self.algebra.sum_of_products(pairs.map(|pair| (self.child(left, pair.left), self.child(right, pair.right))))
    }
}

/// Refuse a level marginalized under weights, whose values live in a store
/// no algebra reads; a level of counts is read through the algebra.
fn require_counts(tdd: &Tdd) -> Result<(), OperationError> {
    match tdd.levels.iter().position(TddLevel::is_weight_marginal) {
        Some(level) => Err(OperationError::MarginalLevel(VtreeIdx(level as u32))),
        None => Ok(()),
    }
}

/// Set `n` to the count `read` holds, reusing its buffer.
fn set_count(n: &mut BigUint, read: CountRead<'_>) {
    match read {
        CountRead::Fast(c) => {
            n.set_zero();
            *n += c;
        }
        CountRead::Big(b) => n.clone_from(b),
    }
}

/// The levels under `root`, `root` included.
fn subtree(vtree: &Vtree, root: VtreeIdx) -> Vec<VtreeIdx> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(t) = stack.pop() {
        out.push(t);
        if !vtree.node(t).is_leaf() {
            let (l, r) = vtree.children(t);
            stack.extend([l, r]);
        }
    }
    out
}

/// The levels a walk from `root` need not visit: those under a marginal
/// level, which reads its own stored counts and never its children's. Empty
/// when no level is marginal, or when `keep` — the level read afterwards —
/// itself lies under one.
fn beneath_marginal(tdd: &Tdd, root: VtreeIdx, keep: VtreeIdx) -> Vec<bool> {
    if !tdd.levels.iter().any(TddLevel::is_marginal) {
        return Vec::new();
    }
    let parent_marginal = |t: VtreeIdx| t != root && tdd.vtree.node(t).parent().is_some_and(|p| tdd.levels[p.idx()].is_marginal());
    let mut up = Some(keep);
    while let Some(t) = up {
        if parent_marginal(t) {
            return Vec::new();
        }
        up = if t == root { None } else { tdd.vtree.node(t).parent() };
    }
    let mut held = vec![false; tdd.vtree.num_nodes()];
    for t in subtree(&tdd.vtree, root) {
        held[t.idx()] = parent_marginal(t);
    }
    held
}

/// The distinct counts that level `t`'s parent carries inline in its
/// references to `t`, ascending: none unless `t` is a marginal level below
/// `root` whose parent is structural.
fn inline_counts(tdd: &Tdd, t: VtreeIdx, root: VtreeIdx) -> Vec<u32> {
    let mut counts = Vec::new();
    if t == root || !tdd.levels[t.idx()].is_marginal() {
        return counts;
    }
    let Some(parent) = tdd.vtree.node(t).parent() else { return counts };
    let level = &tdd.levels[parent.idx()];
    if level.is_marginal() {
        return counts;
    }
    let on_left = tdd.vtree.children(parent).0 == t;
    let view = ChildDecoder::marginal();
    for (_, pairs) in level.internal_inputs_iter() {
        for pair in pairs {
            let side = if on_left { pair.left } else { pair.right };
            if let ChildRef::Value(ValueRef::Inline(c)) = view.child(side) {
                counts.push(c);
            }
        }
    }
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// [`Engine::evaluate_columns`]'s walk over the levels under `root`, `keep`
/// exempt from release: every column, the ones released left at their
/// default.
fn fold_columns_under<A: ColumnAlgebra>(
    eng: &Engine,
    algebra: &A,
    tdd: &Tdd,
    root: VtreeIdx,
    keep: VtreeIdx,
    gate: &mut PollGate,
) -> Result<Vec<A::Column>, OperationError> {
    let lim = eng.limits();
    let mut cols: Vec<A::Column> = Vec::new();
    lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
    cols.resize_with(tdd.vtree.num_nodes(), A::Column::default);
    let held = beneath_marginal(tdd, root, keep);
    // Each marginal level's inline counts, kept until its parent is folded.
    let mut tables: Vec<Vec<u32>> = Vec::new();
    if tdd.levels.iter().any(TddLevel::is_marginal) {
        lim.reserve_exact(&mut tables, tdd.vtree.num_nodes())?;
        tables.resize_with(tdd.vtree.num_nodes(), Vec::new);
    }
    walk_bottom_up(
        &tdd.vtree,
        root,
        &mut cols,
        |_, i| held.get(i).copied().unwrap_or(false),
        |cols, t| fold_columns(algebra, tdd, cols, &mut tables, t, root, gate),
        |cols, i| cols[i] = A::Column::default(),
        Retention::Frontier.frontier(keep),
    )?;
    Ok(cols)
}

/// Write level `t`'s column for [`Engine::evaluate_columns`]: a leaf's three
/// slots, a marginal level's counts, or each node's fold over its children's
/// columns; then, where `t`'s parent carries counts inline (`tables` is
/// empty when no level is marginal), a slot for each.
fn fold_columns<A: ColumnAlgebra>(
    algebra: &A,
    tdd: &Tdd,
    cols: &mut [A::Column],
    tables: &mut [Vec<u32>],
    t: VtreeIdx,
    root: VtreeIdx,
    gate: &mut PollGate,
) -> Result<(), OperationError> {
    gate.poll(1)?;
    let base = tdd.reference_slot_count(t);
    let inline = if tables.is_empty() { Vec::new() } else { inline_counts(tdd, t, root) };
    let mut out = algebra.column(t, base + inline.len());
    let level = &tdd.levels[t.idx()];
    let mut n = BigUint::ZERO;
    if tdd.vtree.node(t).is_leaf() {
        let var = tdd.vtree.leaf_var(t);
        for i in 0..LEAF_WIDTH {
            algebra.leaf(t, var, LeafLabel::from_idx(i), &mut out);
        }
    } else if level.is_marginal() {
        let counts = level.count_column().ok_or(OperationError::MarginalLevel(t))?;
        gate.poll(counts.len() as u64)?;
        for i in 0..counts.len() {
            set_count(&mut n, counts.get(i));
            if !algebra.count(t, i, &n, &mut out) {
                return Err(OperationError::MarginalLevel(t));
            }
        }
    } else {
        let (left, right) = tdd.vtree.children(t);
        let slots = |c: VtreeIdx| InlineSlots {
            base: tdd.reference_slot_count(c),
            counts: tables.get(c.idx()).map_or(&[][..], Vec::as_slice),
        };
        let (on_left, on_right) = (slots(left), slots(right));
        // An implicit level's nodes generated a chunk at a time.
        level.try_for_each_node(0..level.nodes().len(), |i, pairs| {
            gate.poll(pairs.len() as u64 + 1)?;
            let pairs = SlotPairs::new(PairsIter::slice(pairs), on_left, on_right);
            algebra.fold(t, i, pairs, &cols[left.idx()], &cols[right.idx()], &mut out);
            Ok::<(), OperationError>(())
        })?;
        if !tables.is_empty() {
            tables[left.idx()] = Vec::new();
            tables[right.idx()] = Vec::new();
        }
    }
    for (k, &c) in inline.iter().enumerate() {
        set_count(&mut n, CountRead::Fast(u128::from(c)));
        if !algebra.count(t, base + k, &n, &mut out) {
            return Err(OperationError::MarginalLevel(t));
        }
    }
    cols[t.idx()] = out;
    if !tables.is_empty() {
        tables[t.idx()] = inline;
    }
    Ok(())
}

/// The weighted value of `tdd`'s output node under `ws`; `tdd` must have been
/// weighted with `ws`.
fn weighted_output_value(eng: &Engine, tdd: &Tdd, ws: &WeightStore, gate: &mut PollGate) -> Result<WeightValue, OperationError> {
    let vtree = &tdd.vtree;
    // UNSAT / constant-false output: the `ZERO` sentinel carries no level slot
    // (`output.local` is the `ZERO` idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return Ok(ws.wzero());
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return Ok(ws.level(out_t).expect("output level weight-marginalized")[out_i].clone());
    }
    // Leaf output level: the fold below stores nothing for leaves (their values
    // come from the semiring on demand), so read the leaf value directly.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return Ok(ws.leaf_val(var, LeafLabel::from_idx(out_i)));
    }
    let mut computed: Vec<Option<Vec<WeightValue>>> = Vec::new();
    eng.limits().try_resize(&mut computed, vtree.num_nodes(), None)?;
    // Only the root value is read, so child columns are released as their
    // parent completes (`Retention::Frontier`). The "already stored" test
    // is this diagram's own marginality rather than `WeightStore::is_set`: the
    // store is shared, so a column at this index may belong to another live
    // `Tdd` while this diagram's level is still structural.
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    WeightFold::ensure(
        eng,
        VtreeIdx(out_t as u32),
        FoldInput { vtree, levels: &tdd.levels, store: ws },
        &mut computed,
        &marginal,
        Retention::Frontier,
        |work| gate.poll(work),
    )?;
    Ok(computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone())
}
