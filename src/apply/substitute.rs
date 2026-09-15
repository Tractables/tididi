//! Simultaneous substitution through the Boolean apply kernels.

use crate::diagram::{LeafLabel, Literal};
use crate::limits::PollGate;
use crate::vtree::{VarId, VtreeNode};
use crate::{Engine, OperationError, Tdd};

impl Engine {
    /// Run [`Tdd::substitute`](crate::Tdd::substitute) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn substitute(
        &self,
        f: Tdd,
        replacements: &[(VarId, &Tdd)],
    ) -> Result<Tdd, OperationError> {
        self.substitute_with(f, replacements.iter().map(|&(var, diagram)| (var, Replacement::Diagram(diagram))))
    }

    /// Run [`Tdd::rename_vars`](crate::Tdd::rename_vars) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn rename_vars(&self, f: Tdd, renames: &[(VarId, VarId)]) -> Result<Tdd, OperationError> {
        self.substitute_with(f, renames.iter().map(|&(source, target)| (source, Replacement::Literal(Literal::pos(target)))))
    }

    /// Validate replacements into one leaf-indexed table and rebuild through the Boolean kernels.
    fn substitute_with<'a>(
        &self,
        mut f: Tdd,
        replacements: impl ExactSizeIterator<Item = (VarId, Replacement<'a>)>,
    ) -> Result<Tdd, OperationError> {
        f.require_structure()?;
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() {
            return Err(OperationError::Stopped);
        }
        if replacements.len() == 0 { return Ok(f); }
        let mut gate = PollGate::new(lim.reduce_poll_stride());
        let mut by_leaf = Vec::new();
        lim.try_resize(&mut by_leaf, f.vtree().num_nodes(), None)?;
        for (var, replacement) in replacements {
            lim.poll(&mut gate, 1)?;
            let leaf = f
                .vtree()
                .leaf_of(var)
                .ok_or(OperationError::VariableNotInVtree(var))?;
            if by_leaf[leaf.idx()].replace(replacement).is_some() {
                return Err(OperationError::DuplicateVariable(var));
            }
            match replacement {
                Replacement::Diagram(diagram) => {
                    super::check_conjunction_operands(&f, diagram)?;
                    diagram.require_structure()?;
                }
                Replacement::Literal(literal) => {
                    if f.vtree().leaf_of(literal.var).is_none() {
                        return Err(OperationError::VariableNotInVtree(literal.var));
                    }
                }
            }
        }
        lim.flush_poll(&mut gate)?;
        if f.is_zero() {
            self.minimize(&mut f)?;
            return Ok(f);
        }
        self.substitute_prepared(f, &by_leaf, gate)
    }

    /// Rebuild from validated simultaneous replacements while retaining the operation's poll state.
    fn substitute_prepared(
        &self,
        mut f: Tdd,
        by_leaf: &[Option<Replacement<'_>>],
        mut gate: PollGate,
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        self.minimize(&mut f)?;
        let tree = f.vtree().clone();
        let mut columns = Vec::<Vec<Tdd>>::new();
        lim.try_resize(&mut columns, tree.num_nodes(), Vec::new())?;
        for t in tree.bottomup() {
            lim.poll(&mut gate, 1)?;
            match *tree.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    let replacement = by_leaf[t.idx()].unwrap_or(Replacement::Literal(Literal::pos(var)));
                    let mut positive = match replacement {
                        Replacement::Diagram(diagram) => diagram.try_clone_on(self)?,
                        Replacement::Literal(literal) => self.literal(&tree, literal)?,
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
                        self.minimize(&mut result)?;
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
        // Internal columns are minimized when built; a leaf can return a cloned replacement.
        if tree.node(f.output().vtree).is_leaf() {
            self.minimize(&mut result)?;
        }
        Ok(result)
    }
}

/// A borrowed function or a literal to materialize when its source leaf is visited.
#[derive(Clone, Copy)]
enum Replacement<'a> {
    Diagram(&'a Tdd),
    Literal(Literal),
}
