//! Simultaneous substitution through the Boolean apply kernels.

use crate::diagram::{LeafLabel, Literal};
use crate::limits::PollGate;
use crate::vtree::{VarId, VtreeNode};
use crate::{Engine, OperationError, Tdd};

impl Engine {
    /// Simultaneously replace variables with Boolean functions over the same vtree.
    ///
    /// In `replacements`, each source variable appears once; variables absent
    /// from the map keep their meaning. Replacement functions are used as given:
    /// substitutions are not recursively applied inside them. All diagrams must
    /// be structural and share the same vtree allocation. The result keeps `f`'s
    /// literal weights; replacement weights are ignored because they describe
    /// evaluation, not the substituted Boolean functions.
    ///
    /// Consumes `f`, borrows replacements, and returns a minimized diagram.
    /// Rebuilds bottom-up with conjunction/disjunction over the destination
    /// universe, retaining temporary diagrams for a frontier of source levels.
    /// Each source pair can require an apply and checked operand copies, so
    /// intermediate storage can greatly exceed the input and final result.
    /// An empty map returns `f` unchanged after validation.
    ///
    /// # Errors
    ///
    /// An absent or duplicate source variable, a replacement vtree/root mismatch,
    /// or a marginal level is rejected before rebuilding. Allocation, stop and
    /// output-cap errors propagate from construction and apply. Replacement
    /// diagrams remain unchanged on every outcome.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.cube(&tree, [1, -2])?; // x AND NOT y
    /// let replacement = engine.clause(&tree, [2, 3])?; // y OR z
    /// let g = engine.substitute(f, &[(VarId(0), &replacement)])?;
    /// let expected = engine.cube(&tree, [-2, 3])?;
    /// assert!(engine.equivalent(&g, &expected)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn substitute(
        &self,
        mut f: Tdd,
        replacements: &[(VarId, &Tdd)],
    ) -> Result<Tdd, OperationError> {
        f.require_structure()?;
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() {
            return Err(OperationError::Stopped);
        }
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        let mut by_leaf = Vec::new();
        lim.try_resize(&mut by_leaf, f.vtree().num_nodes(), None)?;
        for (i, &(var, replacement)) in replacements.iter().enumerate() {
            lim.poll(&mut gate, 1)?;
            let leaf = f
                .vtree()
                .leaf_of(var)
                .ok_or(OperationError::VariableNotInVtree(var))?;
            if by_leaf[leaf.idx()].replace(i).is_some() {
                return Err(OperationError::DuplicateVariable(var));
            }
            super::check_conjunction_operands(&f, replacement)?;
            replacement.require_structure()?;
        }
        lim.flush_poll(&mut gate)?;
        if replacements.is_empty() || f.is_zero() {
            return Ok(f);
        }
        crate::reduce::try_minimize(self, &mut f)?;
        let tree = f.vtree().clone();
        let mut columns = Vec::<Vec<Tdd>>::new();
        lim.try_resize(&mut columns, tree.num_nodes(), Vec::new())?;
        for t in tree.bottomup() {
            lim.poll(&mut gate, 1)?;
            match *tree.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    let mut positive = if let Some(i) = by_leaf[t.idx()] {
                        replacements[i].1.try_clone_on(self)?
                    } else {
                        self.literal(&tree, Literal::pos(var))?
                    };
                    positive.weights = None;
                    let negative = self.negate(positive.try_clone_on(self)?)?;
                    let one = self.cube(&tree, std::iter::empty::<Literal>())?;
                    lim.reserve_exact(&mut columns[t.idx()], 3)?;
                    // LeafLabel's stable slot order is One, Pos, Neg.
                    debug_assert_eq!(LeafLabel::Pos as usize, 1);
                    columns[t.idx()].extend([one, positive, negative]);
                }
                VtreeNode::Internal { left, right, .. } => {
                    let level = f.level(t);
                    lim.reserve_exact(&mut columns[t.idx()], level.nodes.len())?;
                    for node in &level.nodes {
                        let mut sum = None;
                        for pair in level.pairs_of(node) {
                            lim.poll(&mut gate, 1)?;
                            let a =
                                columns[left.idx()][pair.left.raw() as usize].try_clone_on(self)?;
                            let b = columns[right.idx()][pair.right.raw() as usize]
                                .try_clone_on(self)?;
                            let term = self.and(a, b)?;
                            sum = Some(match sum {
                                None => term,
                                Some(sum) => self.or(sum, term)?,
                            });
                        }
                        let mut result = sum.expect("structural node has a pair");
                        crate::reduce::try_minimize(self, &mut result)?;
                        columns[t.idx()].push(result);
                    }
                    columns[left.idx()] = Vec::new();
                    columns[right.idx()] = Vec::new();
                }
            }
        }
        lim.flush_poll(&mut gate)?;
        let mut result = columns[f.output().vtree.idx()].swap_remove(f.output().local.idx());
        // The destination universe is unchanged; weights stay bound to its variables.
        result.weights = f.weights.take().map(|weights| weights.empty_like());
        crate::reduce::try_minimize(self, &mut result)?;
        Ok(result)
    }

    /// Simultaneously rename variables within the existing vtree universe.
    ///
    /// Each `(source, target)` replaces every occurrence of `source` by `target`;
    /// omitted variables stay unchanged. Distinct sources may share a target,
    /// identifying variables. Swaps and cycles are simultaneous. A bijective
    /// map permutes variables; the vtree shape and its variable IDs stay fixed.
    /// Delegates to [`Engine::substitute`], with the same costs and ownership.
    ///
    /// # Errors
    ///
    /// Unknown source/target variables, duplicate sources, a marginal level, or
    /// a resource refusal. All map entries are validated before construction.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let f = engine.cube(&tree, [1, -2])?;
    /// let swapped = engine.rename_vars(f, &[(VarId(0), VarId(1)), (VarId(1), VarId(0))])?;
    /// let expected = engine.cube(&tree, [-1, 2])?;
    /// assert!(engine.equivalent(&swapped, &expected)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn rename_vars(&self, f: Tdd, renames: &[(VarId, VarId)]) -> Result<Tdd, OperationError> {
        f.require_structure()?;
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() {
            return Err(OperationError::Stopped);
        }
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        let mut seen = Vec::new();
        lim.try_resize(&mut seen, f.vtree().num_nodes(), false)?;
        for &(source, target) in renames {
            lim.poll(&mut gate, 1)?;
            let leaf = f
                .vtree()
                .leaf_of(source)
                .ok_or(OperationError::VariableNotInVtree(source))?;
            if std::mem::replace(&mut seen[leaf.idx()], true) {
                return Err(OperationError::DuplicateVariable(source));
            }
            if f.vtree().leaf_of(target).is_none() {
                return Err(OperationError::VariableNotInVtree(target));
            }
        }
        lim.flush_poll(&mut gate)?;
        if renames.is_empty() || f.is_zero() {
            return Ok(f);
        }
        let mut values = Vec::new();
        lim.reserve_exact(&mut values, renames.len())?;
        for &(_, target) in renames {
            values.push(self.literal(f.vtree(), Literal::pos(target))?);
        }
        let mut replacements = Vec::new();
        lim.reserve_exact(&mut replacements, renames.len())?;
        replacements.extend(
            renames
                .iter()
                .zip(&values)
                .map(|(&(source, _), value)| (source, value)),
        );
        self.substitute(f, &replacements)
    }
}
