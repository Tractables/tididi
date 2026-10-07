//! Restrict a diagram to some of its variables, on the restricted vtree.
//!
//! The inverse of an embedding. Restricting the vtree to the kept variables
//! ([`Vtree::project_to_vars`]) keeps every node with kept variables on both
//! sides, splices out a node with kept variables on one side only, and drops
//! a node with none. On a canonical diagram that does not depend on the
//! dropped variables the levels follow the same pattern: a level with no kept
//! variable holds the single true node, and a level with kept variables on
//! one side pairs each node of that side with the true node once, so it is a
//! bijection the splice composes away. The kept levels are copied with their
//! references read through those bijections. A diagram that depends on a
//! dropped variable is quantified and minimized first.

use std::borrow::Cow;
use std::sync::Arc;

use super::EmbedError;
use super::embed::Plan;

use crate::Engine;
use crate::diagram::{for_each_side_ref_mut, Assembly, ChildSide, LevelView, NodeIdx, Tdd, TddNodeId, ONE_LEAF_IDX};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree, VtreeError};

impl Tdd {
    /// This diagram restricted to some of its variables: every other variable
    /// existentially quantified, and the vtree restricted as
    /// [`Vtree::project_to_vars`] restricts it.
    ///
    /// `local_of(v)` gives a kept variable its id in the result's
    /// `1..=num_local` space and returns `None` for a variable to drop. The
    /// result is canonical, its vtree a new allocation on this diagram's
    /// execution context, and it is the inverse of [`embed`](Self::embed):
    /// embedding the result back under the inverse renaming gives
    /// `∃ dropped. self`.
    ///
    /// A diagram that does not depend on the dropped variables is moved in
    /// `O(size)`: the levels with kept variables on both sides are copied,
    /// and the ones with kept variables on one side only are composed away.
    /// One that depends on them is first quantified with
    /// [`exists_vars`](Self::exists_vars) and minimized. Structural diagrams
    /// only; any weights are dropped. Runs on this diagram's execution
    /// context; use [`Engine::project_to_vars`] inside a batch with resource
    /// limits.
    ///
    /// # Errors
    ///
    /// [`EmbedError::VariableOutOfRange`] when `local_of` yields an id outside
    /// `1..=num_local`; [`EmbedError::Vtree`] with
    /// [`VtreeError::OverlappingVariable`] when it yields one id twice, with
    /// [`VtreeError::Invalid`] when it keeps no variable, and with
    /// [`VtreeError::VariableSpaceTooLarge`] when `num_local` is wider than
    /// the result's leaves can index; [`EmbedError::Operation`] with
    /// [`OperationError::MarginalLevel`] for a diagram that has discarded the
    /// structure at a level, and for a refused allocation or an armed stop.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // (x1 ∨ x2) ∧ (x3 ∨ x4), restricted to x1 and x3.
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&vtree, [1, 2])? & Tdd::clause(&vtree, [3, 4])?;
    /// let keep = |v: VarId| match v.0 {
    ///     1 => Some(VarId(1)),
    ///     3 => Some(VarId(2)),
    ///     _ => None,
    /// };
    /// let g = f.project_to_vars(keep, 2)?;
    /// assert_eq!(g.vtree().num_leaves(), 2);
    /// assert_eq!(g.model_count()?, 4u32.into());     // x2 and x4 cover every case
    ///
    /// // Embedded back, it is f with x2 and x4 quantified away.
    /// let (back, _) = g.embed(&vtree, |v| VarId(2 * v.0 - 1))?;
    /// assert!(back.equivalent(&f.exists_vars(&[VarId(2), VarId(4)])?)?);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn project_to_vars(
        &self,
        local_of: impl Fn(VarId) -> Option<VarId>,
        num_local: u32,
    ) -> Result<Tdd, EmbedError> {
        self.vtree.context().run(|eng| eng.project_to_vars(self, local_of, num_local))
    }
}

impl Engine {
    /// [`Tdd::project_to_vars`] under this batch's scratch and resource
    /// limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::project_to_vars`].
    pub fn project_to_vars(
        &self,
        tdd: &Tdd,
        local_of: impl Fn(VarId) -> Option<VarId>,
        num_local: u32,
    ) -> Result<Tdd, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = tdd.vtree();
        // The kept variables' originals by local id, and the dropped ones.
        let mut original = Vec::new();
        lim.try_resize(&mut original, num_local as usize, None)?;
        let mut dropped = Vec::new();
        let mut kept = 0usize;
        for (_, var) in vtree.leaf_bottomup() {
            match local_of(var) {
                Some(local) => {
                    if local.0 == 0 || local.0 > num_local {
                        return Err(EmbedError::VariableOutOfRange { variable: local, num_vars: num_local });
                    }
                    if original[local.idx()].replace(var).is_some() {
                        return Err(VtreeError::OverlappingVariable(local).into());
                    }
                    kept += 1;
                }
                None => lim.try_push(&mut dropped, var)?,
            }
        }
        if kept == 0 {
            return Err(VtreeError::Invalid("a projection keeps at least one variable".to_string()).into());
        }
        crate::vtree::check_var_space(num_local, 2 * kept - 1)?;
        let into = Arc::new(
            vtree
                .project_to_vars(&local_of, num_local)
                .expect("a checked renaming that keeps a variable restricts the vtree"),
        );
        let plan = Plan::build(self, &into, vtree, |local| original[local.idx()].expect("a leaf of the restriction"), false)?;
        let result = self.project_along(tdd, &into, &plan, &dropped);
        plan.recycle(self);
        result
    }

    /// [`project_to_vars`](Self::project_to_vars) once `into` is matched in
    /// the diagram's vtree as `plan`, `dropped` the variables it leaves out.
    fn project_along(&self, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan, dropped: &[VarId]) -> Result<Tdd, EmbedError> {
        if tdd.is_zero() {
            return Ok(crate::build::constant_zero(self, into));
        }
        let f = self.canonical_copy(tdd)?;
        if let Some(result) = splice(self, &f, into, plan)? {
            return Ok(result);
        }
        // The diagram reads a dropped variable: quantify it first.
        let mut g = self.exists_vars(f.into_owned(), dropped)?;
        self.minimize(&mut g)?;
        match splice(self, &g, into, plan)? {
            Some(result) => Ok(result),
            None => unreachable!("a canonical diagram over the kept variables alone splices"),
        }
    }

    /// `f` where it is certified canonical, else a minimized copy.
    fn canonical_copy<'a>(&self, f: &'a Tdd) -> Result<Cow<'a, Tdd>, OperationError> {
        if f.levels.is_canonical(f.output()) {
            return Ok(Cow::Borrowed(f));
        }
        let mut copy = f.try_clone_on(self)?;
        self.minimize(&mut copy)?;
        Ok(Cow::Owned(copy))
    }
}

/// The reference to the true node of a level with no kept variable.
fn true_ref(vtree: &Vtree, t: crate::vtree::VtreeIdx) -> u32 {
    if vtree.node(t).is_leaf() { ONE_LEAF_IDX.0 } else { 0 }
}

/// `f`, canonical, restricted to `into` along `plan` (`into`'s match in
/// `f`'s vtree); `None` when `f` depends on a dropped variable, which shows
/// as a level of dropped variables other than the true node, or a level with
/// kept variables on one side whose nodes are not each one pair with it.
fn splice(eng: &Engine, f: &Tdd, into: &Arc<Vtree>, plan: &Plan) -> Result<Option<Tdd>, OperationError> {
    let lim = eng.limits();
    let vtree = f.vtree();
    let mut gate = lim.gate();
    // Per node of `f`'s vtree with no kept variable: whether its level is
    // the true node alone.
    let mut free = Vec::new();
    lim.try_resize(&mut free, vtree.num_nodes(), false)?;
    // Per node with kept variables on one side only: each of its nodes'
    // image at the first kept level below, through the levels spliced out.
    let mut chain: Vec<Option<Vec<u32>>> = Vec::new();
    lim.try_resize(&mut chain, vtree.num_nodes(), None)?;
    for t in vtree.bottomup() {
        gate.poll(1)?;
        let leaf = vtree.node(t).is_leaf();
        if plan.free[t.idx()] {
            free[t.idx()] = leaf || {
                let (left, right) = vtree.children(t);
                let level = f.level(t);
                let mut nodes = level.internal_inputs_iter();
                let true_pair = |(_, mut pairs): (usize, crate::diagram::PairsIter<'_>)| {
                    matches!(
                        (pairs.next(), pairs.next()),
                        (Some(p), None) if p.left.raw() == true_ref(vtree, left) && p.right.raw() == true_ref(vtree, right)
                    )
                };
                free[left.idx()] && free[right.idx()] && nodes.next().is_some_and(true_pair) && nodes.next().is_none()
            };
            continue;
        }
        if leaf || plan.covered_by[t.idx()].is_some() {
            continue;
        }
        let (left, right) = vtree.children(t);
        let (carried, open, carried_left) = match !plan.free[left.idx()] {
            true => (left, right, true),
            false => (right, left, false),
        };
        if !free[open.idx()] {
            return Ok(None);
        }
        let level = f.level(t);
        let mut map = Vec::new();
        lim.try_resize(&mut map, level.nodes().len(), u32::MAX)?;
        for (i, mut pairs) in level.internal_inputs_iter() {
            gate.poll(1)?;
            let (Some(p), None) = (pairs.next(), pairs.next()) else { return Ok(None) };
            let (kept, other) = match carried_left {
                true => (p.left.raw(), p.right.raw()),
                false => (p.right.raw(), p.left.raw()),
            };
            if other != true_ref(vtree, open) {
                return Ok(None);
            }
            map[i] = match &chain[carried.idx()] {
                Some(below) => below[kept as usize],
                None => kept,
            };
        }
        chain[t.idx()] = Some(map);
    }

    let mut assembly = Assembly::new(eng, into)?;
    for (s, _, _) in into.internal_bottomup() {
        gate.poll(1)?;
        let d = plan.embedding.levels[s.idx()];
        let view = LevelView::unweighted(f.level(d)).expect("a structural diagram");
        assembly.replace_level(eng, s, view)?;
        let (dl, dr) = vtree.children(d);
        let (levels, _) = assembly.parts_mut();
        for (child, side) in [(dl, ChildSide::Left), (dr, ChildSide::Right)] {
            if let Some(map) = &chain[child.idx()] {
                for_each_side_ref_mut(&mut levels[s.idx()], side, |r| *r = map[*r as usize]);
            }
        }
    }
    gate.flush()?;
    let output = f.output();
    debug_assert_eq!(output.vtree, vtree.root(), "a diagram's output sits at its root level");
    let local = match &chain[vtree.root().idx()] {
        Some(map) => NodeIdx(map[output.local.idx()]),
        None => output.local,
    };
    Ok(Some(assembly.finish_asserted(TddNodeId { vtree: into.root(), local }, None)?))
}

#[cfg(test)]
#[path = "tests/project.rs"]
mod tests;
