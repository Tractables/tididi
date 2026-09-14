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
    /// Consumes `f`, including on error, and borrows replacements. A nonempty map
    /// returns a minimized diagram; an empty map returns `f` unchanged.
    /// Rebuilds bottom-up with conjunction/disjunction over the destination
    /// universe, retaining temporary diagrams for a frontier of source levels.
    /// Each source pair can require an apply and checked operand copies, so
    /// intermediate storage can greatly exceed the input and final result.
    /// Validation also runs for an empty map.
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
    /// let expected = engine.cube(&tree, [-2, 3])?; // (y OR z) AND NOT y = z AND NOT y
    /// assert!(engine.equivalent(&g, &expected)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn substitute(
        &self,
        f: Tdd,
        replacements: &[(VarId, &Tdd)],
    ) -> Result<Tdd, OperationError> {
        self.substitute_with(f, replacements.iter().map(|&(var, diagram)| (var, Replacement::Diagram(diagram))))
    }

    /// Replace variable occurrences simultaneously within the existing vtree.
    ///
    /// Each `(source, target)` uses zero-based [`VarId`]s. Variables omitted as
    /// sources keep their meaning, and a target must already be a variable of the
    /// vtree. Swaps and cycles happen simultaneously: `(x, y), (y, x)` exchanges
    /// both variables instead of applying two sequential renames.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let f = engine.cube(&tree, [1, -2])?; // x AND NOT y
    /// let swapped = engine.rename_vars(f.clone(), &[(VarId(0), VarId(1)), (VarId(1), VarId(0))])?;
    /// assert!(engine.equivalent(&swapped, &engine.cube(&tree, [-1, 2])?)?);
    ///
    /// // Mapping x to y while leaving y unchanged identifies the two variables.
    /// let identified = engine.rename_vars(f, &[(VarId(0), VarId(1))])?;
    /// assert!(identified.is_zero()); // y AND NOT y
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// The vtree's allocation, shape, variable ids, and literal weights stay fixed;
    /// the function is rebuilt with different variable occurrences. Thus a rename
    /// can change a weighted value even when it is a permutation. Distinct sources
    /// may share a target, but a source may appear only once.
    ///
    /// The operand is consumed, including on error. A nonempty map produces a
    /// minimized diagram; an empty map returns the operand unchanged after validation.
    /// Uses [`Engine::substitute`]'s rebuilding algorithm and can require large
    /// intermediate diagrams. The complete
    /// [symbolic reachability example](https://github.com/Tractables/tididi/blob/main/examples/symbolic_reachability.rs)
    /// uses renaming to turn next-state variables into current-state variables.
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for an unknown source or target,
    /// [`OperationError::DuplicateVariable`] for a repeated source,
    /// [`OperationError::MarginalLevel`] for discarded structure, or a resource
    /// refusal. All map entries are validated before rebuilding.
    pub fn rename_vars(&self, f: Tdd, renames: &[(VarId, VarId)]) -> Result<Tdd, OperationError> {
        self.substitute_with(f, renames.iter().map(|&(source, target)| (source, Replacement::Literal(Literal::pos(target)))))
    }

    /// Validate replacements into one leaf-indexed table and rebuild through the Boolean kernels.
    fn substitute_with<'a>(
        &self,
        f: Tdd,
        replacements: impl ExactSizeIterator<Item = (VarId, Replacement<'a>)>,
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
        let empty = replacements.len() == 0;
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
        if empty || f.is_zero() {
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
        crate::reduce::try_minimize(self, &mut f)?;
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
}

/// A borrowed function or a literal to materialize when its source leaf is visited.
#[derive(Clone, Copy)]
enum Replacement<'a> {
    Diagram(&'a Tdd),
    Literal(Literal),
}
