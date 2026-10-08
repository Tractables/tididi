//! Expand each variable of a diagram into a class of new variables.
//!
//! A leaf of the diagram's vtree becomes a balanced subtree over its class.
//! The levels above the classes are copied, with each reference into a class
//! leaf redirected to a node of the class subtree that stands for the leaf's
//! value there: the class's variables all agreeing with the value, positive
//! or negative, or for the constant-true leaf, with each other. Only the
//! nodes some reference needs are built, in one pass over each class subtree.
//! The constants are then added as one cube on a balanced tree of their own,
//! and the free variables by grafting both parts with them.

use std::sync::Arc;

use super::{ExpandError, GraftError};

use crate::Engine;
use crate::diagram::{
    ChildPair, ChildRef, EncodedChildRef, LevelView, Literal, NodeIdx, Tdd, TddLevel, TddNodeId,
    NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO,
};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// How each variable of a diagram expands into a larger variable space.
///
/// [`Tdd::expand_variables`] states what the expanded function is and how its
/// vtree is laid out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VariableExpansion {
    /// The class of each variable of the diagram, indexed by
    /// [`VarId::idx`]: a positive literal's variable equals the diagram's
    /// variable, and a negative literal's variable equals its negation.
    pub classes: Vec<Vec<Literal>>,
    /// Variables fixed to a value: each literal holds.
    pub constants: Vec<Literal>,
    /// Variables the expanded function leaves unconstrained.
    pub free: Vec<VarId>,
    /// The id space of the result: every expanded variable is in `1..=num_vars`.
    pub num_vars: u32,
}

impl VariableExpansion {
    /// Refuse an expansion that does not fit `vtree` or names a variable twice
    /// or out of range.
    fn check(&self, eng: &Engine, vtree: &Vtree) -> Result<(), ExpandError> {
        for (_, variable) in vtree.leaf_bottomup() {
            if self.classes.get(variable.idx()).is_none_or(Vec::is_empty) {
                return Err(ExpandError::MissingClass { variable });
            }
        }
        if let Some(r) = (0..self.classes.len()).find(|&r| !self.classes[r].is_empty() && vtree.leaf_of(VarId(r as u32 + 1)).is_none()) {
            return Err(ExpandError::ClassWithoutLeaf { variable: VarId(r as u32 + 1) });
        }
        let lim = eng.limits();
        let mut named = Vec::new();
        let count = self.classes.iter().map(Vec::len).sum::<usize>() + self.constants.len() + self.free.len();
        lim.reserve_exact(&mut named, count)?;
        named.extend(self.classes.iter().flatten().chain(&self.constants).map(|literal| literal.var).chain(self.free.iter().copied()));
        if let Some(&variable) = named.iter().find(|var| var.0 == 0 || var.0 > self.num_vars) {
            return Err(ExpandError::VariableOutOfRange { variable, num_vars: self.num_vars });
        }
        named.sort_unstable();
        if let Some(pair) = named.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(crate::vtree::VtreeError::OverlappingVariable(pair[0]).into());
        }
        Ok(())
    }
}

impl Tdd {
    /// This function over a larger variable space, in which each variable of
    /// this diagram stands for a class of new variables.
    ///
    /// Variable `v` of this diagram's vtree expands into
    /// `expansion.classes[v.idx()]`: the variable of each positive literal
    /// there equals `v`, and that of each negative literal equals `¬v`. The
    /// constants hold their literals' values and the free variables are
    /// unconstrained. So the result is this function with every `v` replaced
    /// by its class's first literal, conjoined with the agreement of each
    /// class and with the constants; its model count is this diagram's
    /// doubled once per free variable. A variable's class must be nonempty,
    /// and every class must belong to a variable of this vtree; each expanded
    /// variable is named once.
    ///
    /// The result's vtree is [`Vtree::expand_leaves`] on this vtree, with each
    /// class's variables in the order given, joined as by [`Vtree::graft`]
    /// with a balanced tree over the constants and then with the free
    /// variables. It has the id space `1..=expansion.num_vars` and this
    /// vtree's execution context. No apply runs: the levels above the classes
    /// are copied, and each level of a class subtree gets at most three nodes,
    /// one for each value of the class's leaf that the levels above reference.
    /// The result is canonical when this diagram is. Weights are not carried,
    /// and a diagram with summed-out levels is refused. Runs on this diagram's
    /// context; use [`Engine::expand_variables`] inside a bounded batch.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Literal, Tdd, Vtree};
    /// use tididi::restructure::VariableExpansion;
    /// use tididi::vtree::VarId;
    ///
    /// // x1 ∨ x2 over a simplified formula.
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// // Original 1 is x1, original 3 is ¬x1, original 2 is x2, original 4 is
    /// // forced true and original 5 is free.
    /// let expansion = VariableExpansion {
    ///     classes: vec![vec![Literal::pos(VarId(1)), Literal::neg(VarId(3))], vec![Literal::pos(VarId(2))]],
    ///     constants: vec![Literal::pos(VarId(4))],
    ///     free: vec![VarId(5)],
    ///     num_vars: 5,
    /// };
    /// let g = f.expand_variables(&expansion)?;
    /// assert_eq!(g.vtree().num_leaves(), 5);
    /// assert_eq!(g.model_count()?, 6u32.into()); // 3 · 2 (x5 is free)
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`ExpandError::MissingClass`] for a variable of this vtree without a
    /// class, [`ExpandError::ClassWithoutLeaf`] for a class of a variable the
    /// vtree does not carry, [`ExpandError::VariableOutOfRange`] for an
    /// expanded variable outside `1..=num_vars`, and [`ExpandError::Vtree`]
    /// with [`VtreeError::OverlappingVariable`](crate::vtree::VtreeError::OverlappingVariable)
    /// for one named twice, or with another [`VtreeError`](crate::vtree::VtreeError)
    /// for an id space too wide for the result. [`ExpandError::Operation`]
    /// carries [`OperationError::MarginalLevel`] for a diagram with a
    /// summed-out level, and a refused allocation or an armed stop. The
    /// expansion is checked before any level is built.
    pub fn expand_variables(&self, expansion: &VariableExpansion) -> Result<Tdd, ExpandError> {
        self.context().run(|eng| eng.expand_variables(self, expansion))
    }
}

impl Engine {
    /// [`Tdd::expand_variables`] using this batch's scratch and resource limits.
    ///
    /// Checks cancellation at entry and before the constants and the graft
    /// are added; each of those steps, and the rebuild of the classes before
    /// them, is charged to the byte budget as an operation of its own.
    ///
    /// # Errors
    ///
    /// As [`Tdd::expand_variables`], with refused work reported as
    /// [`ExpandError::Operation`].
    pub fn expand_variables(&self, tdd: &Tdd, expansion: &VariableExpansion) -> Result<Tdd, ExpandError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        expansion.check(self, tdd.vtree())?;
        let classes = &expansion.classes;
        let class_vtree = Arc::new(tdd.vtree().expand_leaves(
            move |var| classes[var.idx()].iter().map(|literal| literal.var),
            expansion.num_vars,
        )?);
        let expanded = expand_classes(self, tdd, &class_vtree, classes)?;
        if expansion.constants.is_empty() && expansion.free.is_empty() {
            return Ok(expanded);
        }
        let context = Arc::clone(expanded.context());
        let mut parts = vec![expanded];
        // One balanced tree for every constant: a tree per constant would size
        // each one's variable table by its id, quadratic in their number.
        if !expansion.constants.is_empty() {
            lim.next_phase()?;
            let order: Vec<VarId> = expansion.constants.iter().map(|literal| literal.var).collect();
            let vtree = context.bind(Vtree::balanced_over(&order)?);
            parts.push(self.cube(&vtree, expansion.constants.iter().copied())?);
        }
        lim.next_phase()?;
        self.graft(parts, &expansion.free).map_err(|error| match error {
            GraftError::Operation(error) => ExpandError::Operation(error),
            GraftError::Vtree(error) => ExpandError::Vtree(error),
            error => unreachable!("the checked expansion grafts disjoint structural parts: {error}"),
        })
    }
}

/// The node a structural reference into a leaf level names.
fn leaf_node(level: &TddLevel, side: EncodedChildRef) -> usize {
    match level.child_decoder().child(side) {
        ChildRef::Node(node) => node.idx(),
        ChildRef::Value(_) => unreachable!("a structural diagram's pairs name nodes"),
    }
}

/// `tdd` on `class_vtree`, its vtree with each leaf replaced by the leaf's
/// class, before the constants and free variables are added.
fn expand_classes(eng: &Engine, tdd: &Tdd, class_vtree: &Arc<Vtree>, classes: &[Vec<Literal>]) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let reduced = tdd.vtree();
    let idx_map = lockstep_map(reduced, class_vtree);
    let mut out = Tdd::builder(eng, class_vtree)?;

    // Which of a class leaf's three nodes the levels above reference. Only
    // those are built, which keeps the result free of unreachable nodes, and
    // a diagram whose output is itself a leaf builds just the output's node.
    // A class of one positive literal renames its leaf and needs nothing.
    let mut used: Vec<Option<[bool; 3]>> = Vec::new();
    lim.try_resize(&mut used, reduced.num_nodes(), None)?;
    for (leaf, var) in reduced.leaf_bottomup() {
        if !matches!(classes[var.idx()][..], [Literal { sign: true, .. }]) {
            used[leaf.idx()] = Some([false; 3]);
        }
    }
    let output = tdd.output();
    if let Some(used) = &mut used[output.vtree.idx()]
        && output.local.idx() < 3
    {
        used[output.local.idx()] = true;
    }
    for (t, left, right) in reduced.internal_bottomup() {
        let level = tdd.level(t);
        for node in level.nodes() {
            for pair in level.pairs_iter_of(&node) {
                for (child, side) in [(left, pair.left), (right, pair.right)] {
                    if let Some(used) = &mut used[child.idx()]
                        && side != ZERO.into()
                    {
                        used[leaf_node(tdd.level(child), side)] = true;
                    }
                }
            }
        }
    }

    // The node each referenced leaf node became, in its class subtree.
    let mut remap: Vec<Option<[Option<NodeIdx>; 3]>> = Vec::new();
    lim.try_resize(&mut remap, reduced.num_nodes(), None)?;
    for (leaf, var) in reduced.leaf_bottomup() {
        if let Some(needed) = used[leaf.idx()] {
            remap[leaf.idx()] = Some(build_class(eng, class_vtree, idx_map[leaf.idx()], &classes[var.idx()], needed, &mut out)?);
        }
    }

    for (t, left, right) in reduced.internal_bottomup() {
        let expanded = idx_map[t.idx()];
        let (left_map, right_map) = (remap[left.idx()].as_ref(), remap[right.idx()].as_ref());
        if left_map.is_none() && right_map.is_none() {
            let level = LevelView::unweighted(tdd.level(t)).expect("a structural level has no weighted column");
            out.replace_level(eng, expanded, level)?;
            continue;
        }
        // A reference into a class leaf goes to the node built for it;
        // false and references into other levels are kept.
        let redirect = |child: VtreeIdx, map: Option<&[Option<NodeIdx>; 3]>, side: EncodedChildRef| match map {
            Some(map) if side != ZERO.into() => {
                map[leaf_node(tdd.level(child), side)].expect("a referenced class node is built").into()
            }
            _ => side,
        };
        let level = tdd.level(t);
        let mut pairs = Vec::new();
        for node in level.nodes() {
            pairs.clear();
            for pair in level.pairs_iter_of(&node) {
                pairs.push(ChildPair::new(redirect(left, left_map, pair.left), redirect(right, right_map, pair.right)));
            }
            out.push(eng, expanded, &pairs)?;
        }
    }

    let local = match &remap[output.vtree.idx()] {
        Some(map) if output.local != ZERO => map[output.local.idx()].expect("the output's class node is built"),
        _ => output.local,
    };
    Ok(out.finish(TddNodeId { vtree: idx_map[reduced.root().idx()], local })?)
}

/// Map each node of `reduced` to the node at the same position of
/// `expanded`, walking both from their roots: a leaf of `reduced` maps to the
/// root of its class subtree.
fn lockstep_map(reduced: &Vtree, expanded: &Vtree) -> Vec<VtreeIdx> {
    let mut map = vec![VtreeIdx(0); reduced.num_nodes()];
    let mut stack = vec![(reduced.root(), expanded.root())];
    while let Some((r, e)) = stack.pop() {
        map[r.idx()] = e;
        match (reduced.node(r), expanded.node(e)) {
            (VtreeNode::Leaf { .. }, _) => {}
            (VtreeNode::Internal { left: rl, right: rr, .. }, VtreeNode::Internal { left: el, right: er, .. }) => {
                stack.push((*rl, *el));
                stack.push((*rr, *er));
            }
            _ => unreachable!("the expanded vtree keeps the reduced shape above its classes"),
        }
    }
    map
}

/// Build the nodes of the class subtree rooted at `root` that `needed` asks
/// for, and return where each landed.
///
/// `needed` and the result are indexed by a leaf's node positions, true at
/// `ONE_LEAF_IDX`, the positive literal at `POS_LEAF_IDX` and the negative at
/// `NEG_LEAF_IDX`. The positive node of a subtree joins the positive nodes of
/// its children, and the negative node the negative ones; the true node,
/// needed at the class root only, holds both pairs, so the children's needs
/// are their positive and negative nodes. At a leaf, a negative literal of
/// the class exchanges the positive and the negative node. The subtree's
/// leaves are the class's literals in order, left to right, as
/// [`Vtree::expand_leaves`] places them.
fn build_class(
    eng: &Engine,
    vtree: &Vtree,
    root: VtreeIdx,
    class: &[Literal],
    needed: [bool; 3],
    out: &mut crate::diagram::TddBuilder,
) -> Result<[Option<NodeIdx>; 3], OperationError> {
    // An explicit stack: a recursion one frame per level would overflow on
    // deep subtrees, and both stacks stay within the subtree's size.
    enum Frame {
        Begin(VtreeIdx, [bool; 3]),
        Combine(VtreeIdx, [bool; 3]),
    }
    const ONE: usize = ONE_LEAF_IDX.0 as usize;
    const POS: usize = POS_LEAF_IDX.0 as usize;
    const NEG: usize = NEG_LEAF_IDX.0 as usize;

    let mut work = vec![Frame::Begin(root, needed)];
    let mut results: Vec<[Option<NodeIdx>; 3]> = Vec::new();
    let mut next_leaf = 0;
    while let Some(frame) = work.pop() {
        match frame {
            Frame::Begin(t, needed) => {
                if let VtreeNode::Leaf { var, .. } = vtree.node(t) {
                    let literal = class[next_leaf];
                    debug_assert_eq!(literal.var, *var, "class leaves are visited in class order");
                    next_leaf += 1;
                    let (pos, neg) = if literal.sign { (POS_LEAF_IDX, NEG_LEAF_IDX) } else { (NEG_LEAF_IDX, POS_LEAF_IDX) };
                    results.push([Some(ONE_LEAF_IDX), Some(pos), Some(neg)]);
                    continue;
                }
                let child_needed = [false, needed[POS] || needed[ONE], needed[NEG] || needed[ONE]];
                let (left, right) = vtree.children(t);
                // The combination runs after both children; the left child
                // pops first, so its result lands below the right one's.
                work.push(Frame::Combine(t, needed));
                work.push(Frame::Begin(right, child_needed));
                work.push(Frame::Begin(left, child_needed));
            }
            Frame::Combine(t, needed) => {
                let right = results.pop().expect("the right child's nodes");
                let left = results.pop().expect("the left child's nodes");
                let join = |side: usize| ChildPair::new(
                    left[side].expect("a needed node's children are built"),
                    right[side].expect("a needed node's children are built"),
                );
                let mut built = [None; 3];
                if needed[POS] {
                    built[POS] = Some(out.push(eng, t, &[join(POS)])?);
                }
                if needed[NEG] {
                    built[NEG] = Some(out.push(eng, t, &[join(NEG)])?);
                }
                if needed[ONE] {
                    built[ONE] = Some(out.push(eng, t, &[join(POS), join(NEG)])?);
                }
                results.push(built);
            }
        }
    }
    debug_assert_eq!(results.len(), 1, "the stack ends with the root's nodes");
    Ok(results[0])
}

#[cfg(test)]
#[path = "tests/expand.rs"]
mod tests;
