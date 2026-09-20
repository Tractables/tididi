//! The two updates: adding and dropping one assignment.
//!
//! Both are the same walk. A probe resolves the assignment's node at every
//! level bottom-up; where every block it meets holds that one assignment the
//! update is an edit of the output node's pair list, with a chain of new
//! singleton nodes under it for an insertion whose value is new. Anything else
//! — a block with more than the one assignment, a free variable, a leaf named
//! through `One`, an output below the root — is the rebuild's to answer.

use super::*;

impl Maintenance<'_> {
    /// Add one assignment to the diagram's models.
    ///
    /// Accepts arrays, slices and vectors of signed, one-based integers or
    /// typed [`Literal`] values, as [`Tdd::or_cube`] does. An assignment the
    /// diagram already has leaves it alone; a variable in both polarities
    /// names no assignment and does too.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for a level that has discarded its
    /// structure, or [`OperationError::OverBudget`] for a refused allocation.
    pub fn insert_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.update(model.as_ref(), Edit::Insert)
    }

    /// Drop one assignment from the diagram's models.
    ///
    /// An assignment the diagram does not have leaves it alone, as does a
    /// variable named in both polarities. Inputs and errors follow
    /// [`insert_model`](Self::insert_model).
    ///
    /// # Errors
    ///
    /// As [`insert_model`](Self::insert_model).
    pub fn remove_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.update(model.as_ref(), Edit::Remove)
    }

    /// Convert the input once, then run the update inside the batch's context.
    fn update<L: crate::LiteralInput>(&mut self, model: &[L], edit: Edit) -> Result<(), OperationError> {
        let context = Arc::clone(&self.context);
        context.run(|eng| {
            let mut input = std::mem::take(&mut self.literals);
            input.clear();
            let outcome = L::collect(eng, &self.vtree, model, &mut input)
                .and_then(|()| match edit {
                    Edit::Insert => self.insert_literals(eng, &input),
                    Edit::Remove => self.remove_literals(eng, &input),
                });
            self.literals = input;
            outcome
        })
    }

    /// [`insert_model`](Self::insert_model) on typed literals inside a context.
    fn insert_literals(&mut self, eng: &Engine, model: &[Literal]) -> Result<(), OperationError> {
        match self.read_model(model)? {
            // The false cube adds nothing.
            ModelShape::Inconsistent => return Ok(()),
            ModelShape::Partial => return self.rebuild(eng, model, Edit::Insert),
            ModelShape::Complete => {}
        }
        if !self.editable(eng)? { return self.rebuild(eng, model, Edit::Insert); }
        let root = self.vtree.root();
        match self.probe() {
            Probe::Splits => return self.rebuild(eng, model, Edit::Insert),
            // The assignment is a model already, or sits in a root-level node
            // the output does not name, whose pair an edit cannot take over.
            Probe::Found(owner) => {
                if NodeIdx(owner) != self.tdd.output.local {
                    return self.rebuild(eng, model, Edit::Insert);
                }
                self.misses = 0;
                return Ok(());
            }
            Probe::Absent => self.misses = 0,
        }

        // Bottom-up: every level the assignment's value is new to gains a
        // node for it, and the output node gains the one pair naming it.
        let output = self.tdd.output.local;
        let vtree = Arc::clone(&self.vtree);
        for (t, _, _) in vtree.internal_bottomup() {
            if self.path[t.idx()] != FRESH { continue; }
            let pair = self.path_pair(t);
            let index = self.index.as_mut().expect("`editable` refreshed the index");
            if t == root {
                self.tdd.levels[t.idx()].push_pair_onto_node(eng, output.idx(), pair)?;
                index.note_appended_pair(eng, t, pair, output)?;
            } else {
                let idx = self.tdd.levels[t.idx()].push_node_on(eng, &[pair])?;
                self.path[t.idx()] = idx.0;
                index.note_appended_node(eng, t, pair, idx)?;
            }
            self.tdd.try_invalidate(eng, t)?;
        }
        Ok(())
    }

    /// [`remove_model`](Self::remove_model) on typed literals inside a context.
    fn remove_literals(&mut self, eng: &Engine, model: &[Literal]) -> Result<(), OperationError> {
        match self.read_model(model)? {
            // The false cube takes nothing away.
            ModelShape::Inconsistent => return Ok(()),
            ModelShape::Partial => return self.rebuild(eng, model, Edit::Remove),
            ModelShape::Complete => {}
        }
        if self.tdd.is_zero() { return Ok(()); }
        if !self.editable(eng)? { return self.rebuild(eng, model, Edit::Remove); }
        let root = self.vtree.root();
        let owner = match self.probe() {
            Probe::Splits => return self.rebuild(eng, model, Edit::Remove),
            // No node names the assignment: it is not a model.
            Probe::Absent => { self.misses = 0; return Ok(()); }
            Probe::Found(owner) => owner,
        };
        self.misses = 0;
        // A root-level node the output does not name is not part of the
        // function, so the assignment is not a model of it.
        if NodeIdx(owner) != self.tdd.output.local { return Ok(()); }

        let pair = self.path_pair(root);
        if self.tdd.levels[root.idx()].remove_pair_from_node(eng, owner as usize, pair)? {
            self.index.as_mut().expect("`editable` refreshed the index").note_removed_pair(root, pair);
            self.tdd.try_invalidate(eng, root)?;
        } else {
            // The output node named this assignment and nothing else, so the
            // diagram is now false. A node with no pairs is not a
            // representation the invariants allow, so the levels go instead.
            let weights = self.tdd.weights.take();
            let mut zero = crate::build::constant_zero(eng, &self.vtree);
            zero.weights = weights;
            *self.tdd = zero;
            self.index = None;
        }
        Ok(())
    }

    /// Whether the edit route is open at all, refreshing the index first.
    ///
    /// A false diagram has no output node to edit, a one-variable vtree has no
    /// internal level to hang the chain on, an output below the root leaves
    /// levels the walk does not reach, and a leaf named through `One` has no
    /// node for a single value. A batch that has rebuilt [`GIVE_UP`] times in
    /// a row stops asking and stops indexing.
    fn editable(&mut self, eng: &Engine) -> Result<bool, OperationError> {
        if self.misses >= GIVE_UP { return Ok(false); }
        if self.index.is_none() {
            self.index = Some(Index::build(eng, self.tdd)?);
        }
        let root = self.vtree.root();
        Ok(!self.tdd.is_zero()
            && !self.vtree.node(root).is_leaf()
            && self.tdd.output.vtree == root
            && !self.index.as_ref().expect("just built").any_one_mode)
    }

    /// The rebuild route: the whole-diagram operation the edit stands in for.
    fn rebuild(&mut self, eng: &Engine, model: &[Literal], edit: Edit) -> Result<(), OperationError> {
        let taken = self.take_diagram(eng);
        let out = match edit {
            Edit::Insert => crate::apply::conjoin_clause::disjoin_cube_owned(eng, taken, model)?,
            Edit::Remove => {
                let mut clause = Vec::new();
                eng.limits().reserve_exact(&mut clause, model.len())?;
                clause.extend(model.iter().map(|lit| lit.negated()));
                crate::apply::conjoin_clause::conjoin_clause_owned(eng, taken, &clause)?
            }
        };
        *self.tdd = out;
        self.index = None;
        self.rebuilds += 1;
        self.misses += 1;
        Ok(())
    }
}

/// Which way an update moves the assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edit {
    /// Add it to the models.
    Insert,
    /// Drop it from them.
    Remove,
}
