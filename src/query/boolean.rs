//! Boolean queries over structural diagrams.

use crate::diagram::{ChildPair, NEG_LEAF_IDX, NodeIdx, POS_LEAF_IDX, TddNodeId};
use crate::limits::PollGate;
use crate::vtree::{VarId, VtreeNode};
use crate::{Engine, Literal, OperationError, Tdd};
use rustc_hash::FxHashMap;

impl Engine {
    /// Whether two structural diagrams compute the same Boolean function.
    ///
    /// Borrows both operands, checks their shared vtree, then minimizes checked
    /// copies and compares their structure independently of local node numbering
    /// and pair order. Literal weights do not affect Boolean equality. This
    /// avoids constructing a Boolean product; temporary storage is proportional
    /// to the diagrams, with sorting of each node's pairs. Copies are minimized
    /// even if the caller has already minimized the originals.
    ///
    /// # Errors
    ///
    /// A vtree/root mismatch, a marginal level, or an allocation or stop refusal.
    /// Both inputs remain unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let x = engine.literal(&tree, 1)?;
    /// let y = engine.literal(&tree, 2)?;
    /// let xy = engine.and(x.clone(), y.clone())?;
    /// let absorbed = engine.or(x.clone(), xy)?;
    /// assert!(engine.equivalent(&x, &absorbed)?); // x OR (x AND y) = x
    /// assert!(!engine.equivalent(&x, &y)?); // equal counts do not imply equality
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn equivalent(&self, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
        crate::apply::check_conjunction_operands(f, g)?;
        f.require_structure()?;
        g.require_structure()?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        if f.is_zero() || g.is_zero() {
            return Ok(f.is_zero() == g.is_zero());
        }
        if std::ptr::eq(f, g) {
            return Ok(true);
        }
        let mut f = f.try_clone_on(self)?;
        let mut g = g.try_clone_on(self)?;
        crate::reduce::try_minimize(self, &mut f)?;
        crate::reduce::try_minimize(self, &mut g)?;
        same_minimized(self, &f, &g)
    }

    /// Whether every model of `f` is a model of `g`.
    ///
    /// Borrows structural operands over the same vtree and ignores literal
    /// weights. Checks whether `f AND NOT g` is false using checked copies;
    /// its intermediate diagram can be larger than either operand.
    ///
    /// # Errors
    ///
    /// The compatibility, structure and resource errors of [`Engine::equivalent`],
    /// plus the component operations' output-node cap. Inputs remain unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let both = engine.cube(&tree, [1, 2])?;
    /// let x = engine.literal(&tree, 1)?;
    /// assert!(engine.implies(&both, &x)?);
    /// assert!(!engine.implies(&x, &both)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn implies(&self, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
        crate::apply::check_conjunction_operands(f, g)?;
        f.require_structure()?;
        g.require_structure()?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        if f.is_zero() || std::ptr::eq(f, g) {
            return Ok(true);
        }
        let mut f = f.try_clone_on(self)?;
        let mut g = g.try_clone_on(self)?;
        f.weights = None;
        g.weights = None;
        Ok(self.and(f, self.negate(g)?)?.is_zero())
    }

    /// Variables whose values can change the Boolean function, sorted by ID.
    ///
    /// Minimizes a checked copy, then reads its referenced leaf labels. Constants
    /// have empty support. Free variables carried by the vtree are excluded;
    /// [`implied_literals`](crate::query::implied_literals) instead asks which
    /// literals hold in every model. Weights do not affect support.
    ///
    /// # Errors
    ///
    /// A marginal level or an allocation or cancellation refusal. The borrowed
    /// input remains unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.clause(&tree, [1, 2])?;
    /// assert_eq!(engine.support(&f)?, vec![VarId(0), VarId(1)]); // z is free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn support(&self, f: &Tdd) -> Result<Vec<VarId>, OperationError> {
        f.require_structure()?;
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() {
            return Err(OperationError::Stopped);
        }
        if f.is_zero() {
            return Ok(Vec::new());
        }
        let mut f = f.try_clone_on(self)?;
        crate::reduce::try_minimize(self, &mut f)?;
        let mut seen = Vec::new();
        lim.try_resize(&mut seen, f.vtree().num_nodes(), false)?;
        let mut result = Vec::new();
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        for (var, label) in super::support::leaf_references(&f) {
            lim.poll(&mut gate, 1)?;
            if label == POS_LEAF_IDX.into() || label == NEG_LEAF_IDX.into() {
                let slot = f.vtree().leaf_of(var).expect("referenced leaf").idx();
                if !seen[slot] {
                    seen[slot] = true;
                    lim.try_push(&mut result, var)?;
                }
            }
        }
        result.sort_unstable();
        lim.flush_poll(&mut gate)?;
        Ok(result)
    }

    /// One total satisfying assignment, or `None` if the function is false.
    ///
    /// Returns one literal per vtree variable, sorted by ID, including free
    /// variables. Traverses one pair at each visited internal node and assigns
    /// false at free leaves; the chosen model can change after minimization or
    /// restructuring. The structural input need not be minimized, and literal
    /// weights are ignored. The walk and temporary space are linear in the vtree;
    /// sorting the returned literals takes O(n log n) for n variables.
    ///
    /// # Errors
    ///
    /// A marginal level or an allocation or cancellation refusal; input is unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.cube(&tree, [1, -2])?;
    /// let model = engine.satisfying_assignment(&f)?.unwrap();
    /// assert_eq!(model, vec![1.into(), (-2).into(), (-3).into()]);
    /// assert!(engine.implies(&engine.cube(&tree, model)?, &f)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn satisfying_assignment(&self, f: &Tdd) -> Result<Option<Vec<Literal>>, OperationError> {
        f.require_structure()?;
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() {
            return Err(OperationError::Stopped);
        }
        if f.is_zero() {
            return Ok(None);
        }
        let mut pending = Vec::new();
        let mut result = Vec::new();
        lim.try_push(&mut pending, f.output())?;
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        while let Some(id) = pending.pop() {
            lim.poll(&mut gate, 1)?;
            match *f.vtree().node(id.vtree) {
                VtreeNode::Leaf { var, .. } => {
                    lim.try_push(&mut result, Literal::new(var, id.local == POS_LEAF_IDX))?
                }
                VtreeNode::Internal { left, right, .. } => {
                    let level = f.level(id.vtree);
                    let pair = level
                        .pairs_of(&level.nodes[id.local.idx()])
                        .first()
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
        lim.flush_poll(&mut gate)?;
        Ok(Some(result))
    }
}

/// Exact interning of bottom-up signatures; hash collisions still compare full keys.
fn same_minimized(eng: &Engine, f: &Tdd, g: &Tdd) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let mut gate = PollGate::new(lim.reduce_poll_stride());
    let n = f.vtree().num_nodes();
    let mut keys = [Vec::<Vec<u32>>::new(), Vec::<Vec<u32>>::new()];
    for side in &mut keys {
        lim.try_resize(side, n, Vec::new())?;
    }
    for t in f.vtree().bottomup() {
        lim.poll(&mut gate, 1)?;
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
                    lim.reserve_exact(&mut side[t.idx()], level.nodes.len())?;
                    for node in &level.nodes {
                        let mut signature = Vec::new();
                        for pair in level.pairs_of(node) {
                            lim.poll(&mut gate, 1)?;
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
    lim.flush_poll(&mut gate)?;
    Ok(keys[0][f.output().vtree.idx()][f.output().local.idx()]
        == keys[1][g.output().vtree.idx()][g.output().local.idx()])
}
