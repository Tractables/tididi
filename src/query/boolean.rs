//! Boolean queries over structural diagrams.

use crate::diagram::{ChildPair, ChildSide, EncodedChildRef, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX, TddNodeId};
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
        self.collect_leaf_labels(f, |var, labels| labels.depends().then_some(var), LeafLabels::depends, |var| *var)
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
            return self.collect_leaf_labels(f, |var, labels| labels.implied(var), LeafLabels::mixed, |literal| literal.var);
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
    ///
    /// `settled` says of a summary that further labels cannot change what
    /// `select` makes of it.
    fn collect_leaf_labels<T>(
        &self,
        f: &Tdd,
        select: impl Fn(VarId, LeafLabels) -> Option<T>,
        settled: impl Fn(LeafLabels) -> bool,
        key: impl Fn(&T) -> VarId,
    ) -> Result<Vec<T>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.require_structure()?;
        if f.is_zero() { return Ok(Vec::new()); }
        let f = self.canonical(f)?;
        let mut result = Vec::new();
        let mut gate = lim.gate();
        visit_leaf_labels(&f, settled, |work, pairs| gate.poll_times(work, pairs), |var, labels| {
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
        let lim = self.limits();
        let _op = lim.enter()?;
        f.require_structure()?;
        if f.is_zero() {
            return Ok(None);
        }
        let mut pending = Vec::new();
        let mut result = Vec::new();
        lim.try_push(&mut pending, f.output())?;
        let mut gate = lim.gate();
        while let Some(id) = pending.pop() {
            gate.poll(1)?;
            match *f.vtree().node(id.vtree) {
                VtreeNode::Leaf { var, .. } => {
                    lim.try_push(&mut result, Literal::new(var, id.local == POS_LEAF_IDX))?
                }
                VtreeNode::Internal { left, right, .. } => {
                    let level = f.level(id.vtree);
                    let pair = level
                        .pairs_iter_of_idx(id.local.idx())
                        .next()
                        .expect("structural node is satisfiable");
                    lim.try_push(
                        &mut pending,
                        TddNodeId {
                            vtree: right,
                            local: NodeIdx(pair.right.raw()),
                        },
                    )?;
                    lim.try_push(
                        &mut pending,
                        TddNodeId {
                            vtree: left,
                            local: NodeIdx(pair.left.raw()),
                        },
                    )?;
                }
            }
        }
        result.sort_unstable_by_key(|lit| lit.var);
        gate.flush()?;
        Ok(Some(result))
    }
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
                    // The output's level, which can sit below the root,
                    // keeps its keys for the comparison.
                    for child in [left, right] {
                        if child != diagram.output().vtree {
                            side[child.idx()] = Vec::new();
                        }
                    }
                }
            }
        }
    }
    gate.flush()?;
    Ok(keys[0][f.output().vtree.idx()][f.output().local.idx()]
        == keys[1][g.output().vtree.idx()][g.output().local.idx()])
}

/// Referenced positive, negative and free labels of one non-marginal leaf.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

    /// Whether two labels are referenced, or the free one: no literal is
    /// forced whatever else is.
    pub(super) fn mixed(self) -> bool { !matches!(self.0, 0..=2) }

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
/// Each leaf has one parent, so that parent's pairs finish its summary. They
/// are read until each leaf side's summary is `settled`, past which further
/// labels change nothing the caller reads, or to the end. An implicit parent
/// is read off its digits instead ([`ImplicitLevel::each_slot_on`]): the
/// slots they reach on a leaf side are the labels its pairs name there.
/// `poll(units, pairs)` is charged a parent's pairs, each one unit per leaf
/// side, as a read of every pair charging a poll of those units would be,
/// before they are read.
///
/// [`ImplicitLevel::each_slot_on`]: crate::diagram::ImplicitLevel::each_slot_on
pub(super) fn visit_leaf_labels(
    f: &Tdd,
    settled: impl Fn(LeafLabels) -> bool,
    mut poll: impl FnMut(u64, u64) -> Result<(), OperationError>,
    mut visit: impl FnMut(VarId, LeafLabels) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    if f.is_zero() { return Ok(()); }
    if let VtreeNode::Leaf { var, .. } = *f.vtree.node(f.output.vtree) {
        if !f.levels[f.output.vtree.idx()].is_marginal() {
            poll(1, 1)?;
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
        poll(work, f.levels.live_pairs_at(t.idx()) as u64)?;
        let mut labels = [LeafLabels::default(); 2];
        let mut open = vars.map(|var| var.is_some());
        let level = &f.levels[t.idx()];
        if let Some(d) = level.implicit() {
            for (k, side) in [ChildSide::Left, ChildSide::Right].into_iter().enumerate() {
                if open[k] {
                    d.each_slot_on(side, |s, _| labels[k].insert(EncodedChildRef::from_raw(s as u32)));
                }
            }
        } else {
            'read: for node in level.nodes().iter() {
                for pair in level.pairs_iter_of(&node) {
                    for (k, side) in [pair.left, pair.right].into_iter().enumerate() {
                        if open[k] {
                            labels[k].insert(side);
                            open[k] = !settled(labels[k]);
                        }
                    }
                    if open == [false; 2] {
                        break 'read;
                    }
                }
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
