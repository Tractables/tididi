//! Boolean queries over structural diagrams.

use crate::diagram::{ChildPair, EncodedChildRef, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX, TddNodeId};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};
use crate::{Engine, Literal, OperationError, Tdd};
use rustc_hash::FxHashMap;

impl Engine {
    /// Run [`Tdd::equivalent`](crate::Tdd::equivalent) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    pub fn equivalent(&self, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(f, g)?;
        f.require_structure()?;
        g.require_structure()?;
        if f.is_zero() || g.is_zero() {
            return Ok(f.is_zero() == g.is_zero());
        }
        if std::ptr::eq(f, g) {
            return Ok(true);
        }
        let f = self.canonical(f)?;
        let g = self.canonical(g)?;
        same_minimized(self, &f, &g)
    }

    /// Run [`Tdd::implies`](crate::Tdd::implies) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn implies(&self, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(f, g)?;
        f.require_structure()?;
        g.require_structure()?;
        if f.is_zero() || std::ptr::eq(f, g) {
            return Ok(true);
        }
        let mut f = f.try_clone_on(self)?;
        let mut g = g.try_clone_on(self)?;
        f.weights = None;
        g.weights = None;
        Ok(self.and(f, self.negate(g)?)?.is_zero())
    }

    /// Run [`Tdd::support`](crate::Tdd::support) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    pub fn support(&self, f: &Tdd) -> Result<Vec<VarId>, OperationError> {
        self.collect_leaf_labels(f, |var, labels| labels.depends().then_some(var), |var| *var)
    }

    /// Run [`Tdd::implied_literals`](crate::Tdd::implied_literals) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    pub fn implied_literals(&self, f: &Tdd) -> Result<Vec<Literal>, OperationError> {
        if !f.is_zero() {
            return self.collect_leaf_labels(f, |var, labels| labels.implied(var), |literal| literal.var);
        }
        let lim = self.limits();
        let _op = lim.enter()?;
        f.require_structure()?;
        let mut result = Vec::new();
        let mut gate = lim.gate();
        // False implies both signs, including variables absent from its support.
        for t in f.vtree().bottomup() {
            gate.poll(1)?;
            if let VtreeNode::Leaf { var, .. } = *f.vtree().node(t) {
                lim.try_push(&mut result, Literal::neg(var))?;
                lim.try_push(&mut result, Literal::pos(var))?;
            }
        }
        result.sort_unstable_by_key(|literal| (literal.var, literal.sign));
        gate.flush()?;
        Ok(result)
    }

    /// Collect one optional answer per structural leaf from a checked canonical diagram.
    fn collect_leaf_labels<T>(
        &self,
        f: &Tdd,
        select: impl Fn(VarId, LeafLabels) -> Option<T>,
        key: impl Fn(&T) -> VarId,
    ) -> Result<Vec<T>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.require_structure()?;
        if f.is_zero() { return Ok(Vec::new()); }
        let f = self.canonical(f)?;
        let mut result = Vec::new();
        let mut gate = lim.gate();
        visit_leaf_labels(&f, |work| gate.poll(work), |var, labels| {
            if let Some(value) = select(var, labels) {
                lim.try_push(&mut result, value)?;
            }
            Ok(())
        })?;
        result.sort_unstable_by_key(key);
        gate.flush()?;
        Ok(result)
    }

    /// Borrow an established canonical representation; otherwise minimize a private copy.
    fn canonical<'a>(&self, f: &'a Tdd) -> Result<std::borrow::Cow<'a, Tdd>, OperationError> {
        if f.levels.is_canonical(f.output()) {
            return Ok(std::borrow::Cow::Borrowed(f));
        }
        let mut copy = f.try_clone_on(self)?;
        self.minimize(&mut copy)?;
        Ok(std::borrow::Cow::Owned(copy))
    }

    /// Run [`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    pub fn satisfying_assignment(&self, f: &Tdd) -> Result<Option<Vec<Literal>>, OperationError> {
        Ok(satisfying_assignments_on(self, f, 1)?.pop())
    }

    /// Run [`Tdd::satisfying_assignments`](crate::Tdd::satisfying_assignments) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::satisfying_assignments`](crate::Tdd::satisfying_assignments).
    pub fn satisfying_assignments(&self, f: &Tdd, limit: usize) -> Result<Vec<Vec<Literal>>, OperationError> {
        satisfying_assignments_on(self, f, limit)
    }
}

/// One node of the assignment walk: a leaf polarity cursor, or an internal pair cursor.
struct AssignmentFrame {
    id: TddNodeId,
    parent: Option<usize>,
    start: usize,
    left_end: usize,
    next: usize,
    /// `0` starts the next pair or polarity; `1` runs the left child; `3` runs the right child.
    phase: u8,
    pairs: Vec<ChildPair>,
}

fn leaf_signs(local: NodeIdx) -> &'static [bool] {
    if local == POS_LEAF_IDX {
        &[true]
    } else if local == NEG_LEAF_IDX {
        &[false]
    } else if local == ONE_LEAF_IDX {
        &[false, true]
    } else {
        &[]
    }
}

fn push_assignment_frame(
    stack: &mut Vec<AssignmentFrame>,
    lim: &crate::limits::Limits,
    f: &Tdd,
    id: TddNodeId,
    parent: Option<usize>,
    start: usize,
    gate: &mut crate::limits::PollGate<'_>,
) -> Result<(), OperationError> {
    let mut pairs = Vec::new();
    if let VtreeNode::Internal { .. } = *f.vtree().node(id.vtree) {
        for pair in f.level(id.vtree).pairs_iter_of_idx(id.local.idx()) {
            gate.poll(1)?;
            lim.try_push(&mut pairs, pair)?;
        }
    }
    lim.try_push(stack, AssignmentFrame {
        id,
        parent,
        start,
        left_end: start,
        next: 0,
        phase: 0,
        pairs,
    })
}

fn finish_exhausted(stack: &mut Vec<AssignmentFrame>, current: &mut Vec<Literal>) {
    let parent = stack.last().expect("an exhausted frame is on the stack").parent;
    let start = stack.last().expect("an exhausted frame is on the stack").start;
    stack.pop();
    current.truncate(start);
    if let Some(p) = parent {
        if stack[p].phase == 1 {
            stack[p].next += 1;
            stack[p].phase = 0;
            current.truncate(stack[p].start);
        } else if stack[p].phase == 3 {
            stack[p].phase = 1;
            current.truncate(stack[p].left_end);
        }
    }
}

fn record_assignment(
    lim: &crate::limits::Limits,
    current: &[Literal],
    results: &mut Vec<Vec<Literal>>,
) -> Result<(), OperationError> {
    let mut model = Vec::new();
    for &lit in current {
        lim.try_push(&mut model, lit)?;
    }
    model.sort_unstable_by_key(|lit| lit.var);
    lim.try_push(results, model)
}

/// Depth-first models of `f`, at most `limit` of them, in the same order as
/// [`Engine::satisfying_assignment`].
fn satisfying_assignments_on(
    eng: &Engine,
    f: &Tdd,
    limit: usize,
) -> Result<Vec<Vec<Literal>>, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    f.require_structure()?;
    if f.is_zero() || limit == 0 {
        return Ok(Vec::new());
    }
    let mut results = Vec::new();
    let mut current = Vec::new();
    let mut stack = Vec::new();
    let mut gate = lim.gate();
    push_assignment_frame(&mut stack, lim, f, f.output(), None, 0, &mut gate)?;
    while !stack.is_empty() && results.len() < limit {
        gate.poll(1)?;
        let top = stack.len() - 1;
        match *f.vtree().node(stack[top].id.vtree) {
            VtreeNode::Leaf { var, .. } => {
                let signs = leaf_signs(stack[top].id.local);
                if stack[top].next >= signs.len() {
                    finish_exhausted(&mut stack, &mut current);
                    continue;
                }
                current.truncate(stack[top].start);
                lim.try_push(&mut current, Literal::new(var, signs[stack[top].next]))?;
                stack[top].next += 1;
                let mut i = top;
                loop {
                    if i == 0 {
                        record_assignment(lim, &current, &mut results)?;
                        break;
                    }
                    let p = stack[i].parent.expect("a non-root frame has a parent");
                    if stack[p].phase == 1 {
                        stack[p].left_end = current.len();
                        stack[p].phase = 3;
                        let pair = stack[p].pairs[stack[p].next];
                        let VtreeNode::Internal { right, .. } = *f.vtree().node(stack[p].id.vtree) else {
                            unreachable!("an internal frame holds an internal vtree node");
                        };
                        push_assignment_frame(
                            &mut stack,
                            lim,
                            f,
                            TddNodeId { vtree: right, local: NodeIdx(pair.right.raw()) },
                            Some(p),
                            current.len(),
                            &mut gate,
                        )?;
                        break;
                    }
                    i = p;
                }
            }
            VtreeNode::Internal { left, .. } => {
                if stack[top].phase != 0 {
                    unreachable!("an internal frame is on top only when starting a pair");
                }
                if stack[top].next >= stack[top].pairs.len() {
                    finish_exhausted(&mut stack, &mut current);
                    continue;
                }
                let pair = stack[top].pairs[stack[top].next];
                stack[top].phase = 1;
                push_assignment_frame(
                    &mut stack,
                    lim,
                    f,
                    TddNodeId { vtree: left, local: NodeIdx(pair.left.raw()) },
                    Some(top),
                    current.len(),
                    &mut gate,
                )?;
            }
        }
    }
    gate.flush()?;
    Ok(results)
}

/// Exact interning of bottom-up signatures; hash collisions still compare full keys.
fn same_minimized(eng: &Engine, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    let n = f.vtree().num_nodes();
    let mut keys = [Vec::<Vec<u32>>::new(), Vec::<Vec<u32>>::new()];
    for side in &mut keys {
        lim.try_resize(side, n, Vec::new())?;
    }
    for t in f.vtree().bottomup() {
        gate.poll(1)?;
        match *f.vtree().node(t) {
            VtreeNode::Leaf { .. } => {
                for side in &mut keys {
                    lim.reserve_exact(&mut side[t.idx()], 3)?;
                    side[t.idx()].extend([0, 1, 2]);
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                let mut intern = FxHashMap::<Vec<ChildPair>, u32>::default();
                for (diagram, side) in [f, g].into_iter().zip(&mut keys) {
                    let level = diagram.level(t);
                    lim.reserve_exact(&mut side[t.idx()], level.nodes().len())?;
                    for node in level.nodes() {
                        let mut signature = Vec::new();
                        for pair in level.pairs_iter_of(&node) {
                            gate.poll(1)?;
                            let a = side[left.idx()][pair.left.raw() as usize];
                            let b = side[right.idx()][pair.right.raw() as usize];
                            lim.try_push(&mut signature, ChildPair::new(NodeIdx(a), NodeIdx(b)))?;
                        }
                        signature.sort_unstable();
                        let id = if let Some(&id) = intern.get(&signature) {
                            id
                        } else {
                            let id = u32::try_from(intern.len())
                                .map_err(|_| OperationError::OverBudget)?;
                            lim.reserve_map(&mut intern, 1)?;
                            intern.insert(signature, id);
                            id
                        };
                        side[t.idx()].push(id);
                    }
                    side[left.idx()] = Vec::new();
                    side[right.idx()] = Vec::new();
                }
            }
        }
    }
    gate.flush()?;
    Ok(keys[0][f.output().vtree.idx()][f.output().local.idx()]
        == keys[1][g.output().vtree.idx()][g.output().local.idx()])
}

/// Referenced positive, negative and free labels of one non-marginal leaf.
#[derive(Clone, Copy, Default)]
pub(super) struct LeafLabels(u8);

impl LeafLabels {
    /// Include a referenced label; zero sentinels contribute nothing.
    fn insert(&mut self, child: EncodedChildRef) {
        self.0 |= if child == POS_LEAF_IDX.into() { 1 }
            else if child == NEG_LEAF_IDX.into() { 2 }
            else if child == ONE_LEAF_IDX.into() { 4 }
            else { 0 };
    }

    /// Whether a positive or negative label makes this variable part of the support.
    pub(super) fn depends(self) -> bool { self.0 & 3 != 0 }

    /// The forced literal, if every reference uses the same non-free label.
    pub(super) fn implied(self, var: VarId) -> Option<Literal> {
        match self.0 {
            1 => Some(Literal::pos(var)),
            2 => Some(Literal::neg(var)),
            _ => None,
        }
    }
}

/// Visit each referenced structural leaf's label summary on a minimized diagram.
///
/// Each leaf has one parent, so scanning that parent's pairs finishes its summary.
/// `poll` receives one work unit per leaf reference, within the pair loop.
pub(super) fn visit_leaf_labels(
    f: &Tdd,
    mut poll: impl FnMut(u64) -> Result<(), OperationError>,
    mut visit: impl FnMut(VarId, LeafLabels) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    if f.is_zero() { return Ok(()); }
    if let VtreeNode::Leaf { var, .. } = *f.vtree.node(f.output.vtree) {
        if !f.levels[f.output.vtree.idx()].is_marginal() {
            poll(1)?;
            let mut labels = LeafLabels::default();
            labels.insert(f.output.local.into());
            visit(var, labels)?;
        }
        return Ok(());
    }
    let leaf_var = |child: VtreeIdx| match *f.vtree.node(child) {
        VtreeNode::Leaf { var, .. } if !f.levels[child.idx()].is_marginal() => Some(var),
        _ => None,
    };
    for (t, left, right) in f.vtree.internal_bottomup() {
        let vars = [leaf_var(left), leaf_var(right)];
        let work = vars.iter().filter(|var| var.is_some()).count() as u64;
        if work == 0 { continue; }
        let mut labels = [LeafLabels::default(); 2];
        let level = &f.levels[t.idx()];
        for node in level.nodes().iter() {
            for pair in level.pairs_iter_of(&node) {
                poll(work)?;
                if vars[0].is_some() { labels[0].insert(pair.left); }
                if vars[1].is_some() { labels[1].insert(pair.right); }
            }
        }
        for (var, labels) in vars.into_iter().zip(labels) {
            if let Some(var) = var && labels.0 != 0 {
                visit(var, labels)?;
            }
        }
    }
    Ok(())
}
