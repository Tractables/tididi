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
use crate::diagram::{
    for_each_side_ref_mut, Assembly, ChildSide, LevelView, NodeIdx, Tdd, TddBuildError, TddNodeId, WeightStore, ONE_LEAF_IDX,
};
use crate::limits::{Limits, OperationError};
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
        let (original, dropped) = renaming(lim, vtree, &local_of, num_local)?;
        let kept = original.iter().filter(|v| v.is_some()).count();
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

    /// [`project_to_vars`](Self::project_to_vars) onto `into`, for a
    /// diagram whose levels may hold weighted marginal values, with
    /// `weights` as the result's store: the inverse of
    /// [`embed_over`](Self::embed_over).
    ///
    /// `into` is this diagram's vtree restricted to the variables `local_of`
    /// keeps, as [`Vtree::project_to_vars`] restricts it, and `local_of`
    /// gives each of its leaves' variables. A structural diagram is
    /// projected as by [`project_to_vars`](Self::project_to_vars) and takes
    /// `weights` as its store when one is given. A diagram with a marginal
    /// level keeps its values, which sum over the variables under the
    /// level: each variable it drops must lie under a marginal level that
    /// `into` keeps, and the levels under that level that `into` does not
    /// keep go with the variables. Its weighted value is the source's. It
    /// needs `weights`, whose table must give each kept variable the
    /// weights the source's gives it.
    ///
    /// A walk that sums out finished subtrees of its conjunctions can so
    /// carry one level and two leaves of each such subtree into the
    /// conjunctions after it, instead of every level of the subtree.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_rational::BigRational;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    /// use tididi::vtree::VarId;
    ///
    /// let third = || BigRational::new(1.into(), 3.into());
    /// let table = |n| RationalWeights::from_literals(&vec![
    ///     LiteralWeights { negative: third(), positive: third() }; n
    /// ]);
    /// let engine = Engine::new();
    /// // (x1 ∨ x2 ∨ x3 ∨ x4) ∧ (x4 ∨ x5) on a right-linear vtree, the level
    /// // over x2 to x5 summed out.
    /// let vtree = Arc::new(Vtree::linear(5));
    /// let mut f = engine.and(engine.clause(&vtree, [1, 2, 3, 4])?, engine.clause(&vtree, [4, 5])?)?;
    /// f.set_weights(WeightStore::new(table(5), Arithmetic::ExactRational))?;
    /// let summed = vtree.children(vtree.root()).1;
    /// engine.marginalize_levels(&mut f, &[summed])?;
    /// let value = f.weighted_value()?.expect("weighted").into_rational();
    ///
    /// // Kept: x1, and x2 and x5, one on each side of the summed level.
    /// let local_of = |v: VarId| match v.0 {
    ///     1 => Some(VarId(1)),
    ///     2 => Some(VarId(2)),
    ///     5 => Some(VarId(3)),
    ///     _ => None,
    /// };
    /// let into = Arc::new(vtree.project_to_vars(local_of, 3).expect("it keeps a variable"));
    /// let store = WeightStore::new(table(3), Arithmetic::ExactRational);
    /// let g = engine.project_over(&f, &into, local_of, Some(store))?;
    /// assert_eq!(g.vtree().num_leaves(), 3);
    /// assert_eq!(g.weighted_value()?.expect("weighted").into_rational(), value);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`EmbedError::VariableOutOfRange`] when `local_of` yields an id
    /// outside `into`'s variables; [`EmbedError::Vtree`] with
    /// [`VtreeError::OverlappingVariable`] when it yields one id twice;
    /// [`EmbedError::NotIsomorphic`] when `into` is not the restriction,
    /// naming the node of `into` where it stops matching;
    /// [`EmbedError::Unsummed`] naming a variable that a diagram with a
    /// marginal level drops outside its marginal levels;
    /// [`OperationError::MarginalLevel`] naming a level that holds integer
    /// counts; [`EmbedError::SourceWeights`] for a marginal level without
    /// `weights`, or a table that disagrees with the source's on a kept
    /// variable; [`EmbedError::DestinationWeights`] for a table that does
    /// not cover `into`'s variables; [`OperationError::OverBudget`] where
    /// the copy is refused.
    pub fn project_over(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        local_of: impl Fn(VarId) -> Option<VarId>,
        weights: Option<WeightStore>,
    ) -> Result<Tdd, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let vtree = tdd.vtree();
        let (original, dropped) = renaming(lim, vtree, &local_of, into.num_vars())?;
        if let Some((leaf, _)) = into.leaf_bottomup().find(|&(_, var)| original[var.idx()].is_none()) {
            return Err(EmbedError::NotIsomorphic { source: leaf });
        }
        if let Some(t) = vtree.bottomup().find(|&t| tdd.level(t).is_marginal() && !tdd.level(t).is_weight_marginal()) {
            return Err(OperationError::MarginalLevel(t).into());
        }
        if let Some(store) = &weights {
            store
                .check_variables(into.leaf_bottomup().map(|(_, var)| var))
                .map_err(EmbedError::DestinationWeights)?;
        }
        let summed = tdd.has_marginal_level();
        if summed {
            check_kept_weights(tdd, weights.as_ref(), &local_of).map_err(EmbedError::SourceWeights)?;
        }
        let plan = Plan::build(self, into, vtree, |local| original[local.idx()].expect("a leaf of the restriction"), false)?;
        let result = match weights {
            Some(store) if summed => carry_marginal(self, tdd, into, &plan, store),
            weights => self.project_along(tdd, into, &plan, &dropped).and_then(|mut result| {
                if let Some(store) = weights {
                    result.set_weights(store).map_err(OperationError::from)?;
                }
                Ok(result)
            }),
        };
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
        // The diagram reads a dropped variable: quantify it first, on a copy
        // the limits are charged for where `f` is the caller's.
        let f = match f {
            Cow::Borrowed(f) => f.try_clone_on(self)?,
            Cow::Owned(f) => f,
        };
        let mut g = self.exists_vars(f, dropped)?;
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

/// The original of each local variable `local_of` keeps, by local id, and
/// the variables of `vtree` it drops, in leaf order.
fn renaming(
    lim: &Limits,
    vtree: &Vtree,
    local_of: impl Fn(VarId) -> Option<VarId>,
    num_local: u32,
) -> Result<(Vec<Option<VarId>>, Vec<VarId>), EmbedError> {
    let mut original = Vec::new();
    lim.try_resize(&mut original, num_local as usize, None)?;
    let mut dropped = Vec::new();
    for (_, var) in vtree.leaf_bottomup() {
        match local_of(var) {
            Some(local) => {
                if local.0 == 0 || local.0 > num_local {
                    return Err(EmbedError::VariableOutOfRange { variable: local, num_vars: num_local });
                }
                if original[local.idx()].replace(var).is_some() {
                    return Err(VtreeError::OverlappingVariable(local).into());
                }
            }
            None => lim.try_push(&mut dropped, var)?,
        }
    }
    Ok((original, dropped))
}

/// Check that a diagram with a weight-marginal level has a destination
/// store, and that the store gives each kept variable the source's weights.
fn check_kept_weights(
    tdd: &Tdd,
    destination: Option<&WeightStore>,
    local_of: impl Fn(VarId) -> Option<VarId>,
) -> Result<(), TddBuildError> {
    let Some(destination) = destination else {
        let level = tdd.vtree().bottomup().find(|&t| tdd.level(t).is_weight_marginal()).expect("a marginal level");
        return Err(TddBuildError::WeightedLevelWithoutStore { level });
    };
    let kept = tdd.vtree().leaf_bottomup().filter_map(|(_, var)| local_of(var).map(|local| (var, local)));
    if tdd.weights().is_some_and(|source| !source.compatible_after_rename(destination, kept)) {
        return Err(TddBuildError::IncompatibleWeights);
    }
    Ok(())
}

/// `tdd`, which has a marginal level, on `into` along `plan` (`into`'s
/// match in `tdd`'s vtree), with `weights` as its store. Every node `into`
/// does not keep lies under a marginal level, whose values sum over the
/// variables under it and which reads no level under it: each level `into`
/// keeps is copied with its values, references and all, and the others
/// are left behind.
fn carry_marginal(eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan, weights: WeightStore) -> Result<Tdd, EmbedError> {
    if tdd.is_zero() {
        let mut result = crate::build::constant_zero(eng, into);
        result.weights = Some(weights);
        return Ok(result);
    }
    let lim = eng.limits();
    let vtree = tdd.vtree();
    let mut gate = lim.gate();
    for t in vtree.bottomup() {
        gate.poll(1)?;
        if plan.covered_by[t.idx()].is_some() || vtree.node(t).parent().is_some_and(|p| tdd.level(p).is_marginal()) {
            continue;
        }
        // A node `into` does not keep has a dropped variable under it, on
        // its side with none kept.
        let mut n = t;
        while !vtree.node(n).is_leaf() {
            let (left, right) = vtree.children(n);
            n = if plan.free[right.idx()] { right } else { left };
        }
        return Err(EmbedError::Unsummed { variable: vtree.leaf_var(n) });
    }
    let source = tdd.weights();
    let mut assembly = Assembly::new(eng, into)?;
    let (levels, store) = assembly.parts_mut();
    let store = store.insert(weights);
    for s in into.bottomup() {
        gate.poll(1)?;
        let d = plan.embedding.levels[s.idx()];
        let level = tdd.level(d);
        level.try_copy_over(lim, &mut levels[s.idx()])?;
        if level.is_weight_marginal() {
            let column = source.and_then(|w| w.level(d.idx())).unwrap_or(&[]);
            let mut copy = Vec::new();
            lim.reserve_exact(&mut copy, column.len())?;
            copy.extend_from_slice(column);
            store.set_level(s.idx(), copy);
        }
    }
    gate.flush()?;
    store.check_levels(into, levels).map_err(EmbedError::DestinationWeights)?;
    let output = TddNodeId { vtree: into.root(), local: tdd.output().local };
    Ok(assembly.finish_asserted(output, None)?)
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
                for_each_side_ref_mut(eng.limits(), &mut levels[s.idx()], side, |r| *r = map[*r as usize])?;
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
