//! Simultaneous substitution through the Boolean apply kernels.

use crate::diagram::{LeafLabel, Literal};
use super::compose::SharedCircuit;
use crate::limits::PollGate;
use crate::vtree::{VarId, VtreeNode};
use crate::{Engine, OperationError, Tdd};

impl Engine {
    /// Run [`Tdd::substitute`](crate::Tdd::substitute) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn substitute(
        &self,
        f: Tdd,
        replacements: &[(VarId, &Tdd)],
    ) -> Result<Tdd, OperationError> {
        self.substitute_with(f, replacements.iter().map(|&(var, diagram)| (var, Replacement::Diagram(diagram))))
    }

    /// Run [`Tdd::rename_vars`](crate::Tdd::rename_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
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
        lim.check_stop()?;
        if replacements.len() == 0 { return Ok(f); }
        let mut gate = lim.gate();
        let mut by_leaf = Vec::new();
        lim.try_resize(&mut by_leaf, f.vtree().num_nodes(), None)?;
        for (var, replacement) in replacements {
            gate.poll(1)?;
            let leaf = f
                .vtree()
                .leaf_of(var)
                .ok_or(OperationError::VariableNotInVtree(var))?;
            if by_leaf[leaf.idx()].replace(replacement).is_some() {
                return Err(OperationError::DuplicateVariable(var));
            }
            match replacement {
                Replacement::Diagram(diagram) => {
                    super::check_vtree(&f, diagram)?;
                    diagram.require_structure()?;
                }
                Replacement::Literal(literal) => {
                    if f.vtree().leaf_of(literal.var).is_none() {
                        return Err(OperationError::VariableNotInVtree(literal.var));
                    }
                }
            }
        }
        gate.flush()?;
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
        let vtree = f.vtree().clone();
        let mut uses = Vec::<Vec<usize>>::new();
        lim.try_resize(&mut uses, vtree.num_nodes(), Vec::new())?;
        for t in vtree.bottomup() {
            gate.poll(1)?;
            let width = if vtree.node(t).is_leaf() { 3 } else { f.level(t).nodes.len() };
            lim.try_resize(&mut uses[t.idx()], width, 0)?;
            if let VtreeNode::Internal { left, right, .. } = *vtree.node(t) {
                for node in &f.level(t).nodes {
                    for pair in f.level(t).pairs_of(node) {
                        gate.poll(1)?;
                        uses[left.idx()][pair.left.raw() as usize] += 1;
                        uses[right.idx()][pair.right.raw() as usize] += 1;
                    }
                }
            }
        }
        uses[f.output().vtree.idx()][f.output().local.idx()] += 1;
        let mut columns = Vec::<Vec<SharedCircuit>>::new();
        lim.reserve_exact(&mut columns, vtree.num_nodes())?;
        columns.resize_with(vtree.num_nodes(), Vec::new);
        for t in vtree.bottomup() {
            gate.poll(1)?;
            match *vtree.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    let replacement = by_leaf[t.idx()].unwrap_or(Replacement::Literal(Literal::pos(var)));
                    lim.reserve_exact(&mut columns[t.idx()], 3)?;
                    // LeafLabel's stable slot order is One, Pos, Neg.
                    debug_assert_eq!(LeafLabel::Pos as usize, 1);
                    for (slot, &count) in uses[t.idx()].iter().enumerate() {
                        let value = if count == 0 {
                            SharedCircuit::default()
                        } else {
                            let value = if slot == 0 {
                                self.cube(&vtree, std::iter::empty::<Literal>())?
                            } else {
                                match replacement {
                                    Replacement::Diagram(diagram) => {
                                        let mut value = diagram.try_clone_on(self)?;
                                        value.weights = None;
                                        if slot == 2 { self.negate(value)? } else { value }
                                    }
                                    Replacement::Literal(literal) => {
                                        self.literal(&vtree, if slot == 2 { literal.negated() } else { literal })?
                                    }
                                }
                            };
                            SharedCircuit::new(value, count)
                        };
                        columns[t.idx()].push(value);
                    }
                }
                VtreeNode::Internal { left, right, .. } => {
                    let level = f.level(t);
                    lim.reserve_exact(&mut columns[t.idx()], level.nodes.len())?;
                    for (i, node) in level.nodes.iter().enumerate() {
                        let mut sum = None;
                        for pair in level.pairs_of(node) {
                            gate.poll(1)?;
                            let a =
                                columns[left.idx()][pair.left.raw() as usize].take(self)?;
                            let b = columns[right.idx()][pair.right.raw() as usize]
                                .take(self)?;
                            let term = self.and(a, b)?;
                            sum = Some(match sum {
                                None => term,
                                Some(sum) => self.or(sum, term)?,
                            });
                        }
                        let mut result = sum.expect("structural node has a pair");
                        self.minimize(&mut result)?;
                        columns[t.idx()].push(SharedCircuit::new(result, uses[t.idx()][i]));
                    }
                    columns[left.idx()] = Vec::new();
                    columns[right.idx()] = Vec::new();
                }
            }
        }
        gate.flush()?;
        let mut result = columns[f.output().vtree.idx()][f.output().local.idx()].take(self)?;
        // The destination universe is unchanged; weights stay bound to its variables.
        result.weights = f.weights.take().map(|weights| weights.empty_like());
        // Internal columns are minimized when built; a leaf can return a cloned replacement.
        if vtree.node(f.output().vtree).is_leaf() {
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
