//! Literal conditioning (cofactor): fix variables to constants, removing them.
//!
//! Conditioning only rewrites the target leaf's parent level (drops the
//! opposite-polarity pairs, fixes the kept side to One) and never disjoins, so
//! it is sound when sibling levels are marginal. Both the conditioned leaf's
//! own level and its parent's must be structural; `check_conditionable`
//! checks that.

use crate::Engine;
use std::sync::Arc;

use crate::build::constant_like;
use crate::diagram::ChildSide;
use crate::limits::OperationError;
use crate::reduce::ReductionPlan;
use crate::diagram::{ChildPair, Tdd, ZERO};
use super::falsity::{propagate_false_nodes, rewrite_level_pairs};
use crate::vtree::{VarId, VtreeIdx};
use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};

/// The implementation behind [`Engine::condition_vars`](crate::Engine::condition_vars).
pub(crate) fn condition_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
    condition_on(eng, f, vars.iter().map(|&var| crate::diagram::Literal::new(var, value)))
}

/// Validate a mixed assignment and condition all its leaves in one reduction.
pub(crate) fn condition_on(eng: &Engine, f: Tdd, assignment: impl IntoIterator<Item = impl TryInto<crate::diagram::Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.begin_operation();
    lim.check_stop()?;
    let mut gate = lim.gate();
    let mut targets = Vec::new();
    for literal in assignment {
        let literal = literal.try_into().map_err(Into::into)?;
        let leaf = f.vtree.leaf_of(literal.var).ok_or(OperationError::VariableNotInVtree(literal.var))?;
        lim.try_push(&mut targets, (leaf, literal.sign))?;
        gate.poll(1)?;
    }
    gate.flush()?;
    targets.sort_unstable_by_key(|&(leaf, _)| leaf);
    let contradictory = targets.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
    if contradictory {
        return constant_like(eng, &f, false);
    }
    targets.dedup_by_key(|entry| entry.0);
    if f.is_zero() || targets.is_empty() { return Ok(f); }
    condition_targets(eng, f, &mut targets)
}

/// Validate distinct target leaves, rewrite their parents, propagate falsity,
/// and reduce once. Each target is a leaf and whether its positive branch is
/// the one kept.
fn condition_targets(
    eng: &Engine,
    mut tdd: Tdd,
    targets: &mut [(VtreeIdx, bool)],
) -> Result<Tdd, OperationError> {
    for &(leaf, _) in targets.iter() { check_conditionable(&tdd, leaf)?; }
    if let Some(&(_, keep_positive)) = targets.iter().find(|&&(leaf, _)| leaf == tdd.output.vtree) {
        return condition_leaf_output(eng, &tdd, keep_positive);
    }
    let vtree = Arc::clone(&tdd.vtree);
    // Parent index, then left before right, fixes the rewrite and invalidation order.
    let route = |leaf| {
        let parent = vtree.node(leaf).parent().expect("a non-output leaf has a parent");
        (parent, vtree.children(parent).1 == leaf)
    };
    targets.sort_unstable_by_key(|&(leaf, _)| route(leaf));
    let mut emptied = false;
    for &(leaf, keep_positive) in targets.iter() {
        let (parent, right) = route(leaf);
        let side = if right { ChildSide::Right } else { ChildSide::Left };
        emptied |= rewrite_for_restrict(&mut tdd, parent, side, keep_positive);
    }
    if emptied { propagate_false_nodes(&mut tdd); }

    // Set the false sentinel before pruning, so its empty nodes are unreachable.
    if !eng.is_sat(&tdd)? { tdd.output.local = ZERO; }
    eng.reduce(&mut tdd, ReductionPlan::default())?;
    Ok(tdd)
}

/// Precondition of the leaf rewrites: neither the conditioned leaf's level nor
/// its parent's may be marginal. `rewrite_for_restrict` matches the
/// target-side label against `POS_LEAF_IDX`/`NEG_LEAF_IDX`/`ONE_LEAF_IDX`, and
/// a bare marginal-slot ref occupies the same numeric space
/// (`diagram/level/marginal.rs`), so a marginal leaf level would be silently
/// mis-conditioned; a marginal parent has no pairs, so the rewrite is a no-op.
fn check_conditionable(t: &Tdd, leaf_idx: VtreeIdx) -> Result<(), OperationError> {
    t.require_structure_at(leaf_idx)?;
    if let Some(parent) = t.vtree.node(leaf_idx).parent() {
        t.require_structure_at(parent)?;
    }
    Ok(())
}

/// Handle conditioning when the diagram output sits directly at the conditioned leaf.
fn condition_leaf_output(eng: &Engine, t: &Tdd, keep_positive: bool) -> Result<Tdd, OperationError> {
    let output_label = t.output.local;
    let satisfied = if output_label == ZERO {
        false
    } else if output_label == ONE_LEAF_IDX {
        true
    } else if output_label == POS_LEAF_IDX {
        keep_positive
    } else if output_label == NEG_LEAF_IDX {
        !keep_positive
    } else {
        unreachable!("unexpected output local index {:?} at leaf", output_label)
    };

    constant_like(eng, t, satisfied)
}

/// Rewrite parent level `parent_vi` so that references to the target leaf side
/// are constrained: kept labels become One (value fixed), dropped labels are removed.
/// Answers whether any node lost its last pair.
///
/// Compacted in place: restriction never adds a pair, so each node's survivors
/// fit in the prefix of the arena range it already owns and node indices are
/// preserved. Abandoned range tails are reported through `note_dead_pairs`
/// and reclaimed by the arena's own amortized sweep.
fn rewrite_for_restrict(tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide, keep_positive: bool) -> bool {
    // Restriction of one pair: `None` = dropped (the pair belongs to the
    // opposite cofactor), `Some` = kept, with the target side fixed to One when
    // it named the conditioned leaf. `One`, and any reference to an internal
    // child, is carried through as-is.
    if tdd.levels[parent_vi.idx()].nodes.is_empty() { return false; }
    tdd.rewrite_level(parent_vi, |level| rewrite_level_pairs(level, |_, _, p: ChildPair| {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        if label != POS_LEAF_IDX.into() && label != NEG_LEAF_IDX.into() {
            return Some(p);
        }
        // x=⊤ pairs are excluded from the x=⊥ cofactor, and vice versa.
        if (label == POS_LEAF_IDX.into()) != keep_positive {
            return None;
        }
        Some(if side == ChildSide::Left {
            ChildPair::new(ONE_LEAF_IDX, p.right)
        } else {
            ChildPair::new(p.left, ONE_LEAF_IDX)
        })
    }))
}

#[cfg(test)]
#[path = "tests/condition/mod.rs"]
mod tests;

impl crate::Engine {
    /// Run [`Tdd::condition`](crate::Tdd::condition) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors, [`OperationError::Stopped`] on
    /// cancellation, or [`OperationError::OverBudget`] on allocation refusal.
    pub fn condition(&self, f: Tdd, assignment: impl IntoIterator<Item = impl TryInto<crate::diagram::Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        condition_on(self, f, assignment)
    }

    /// Run [`Tdd::condition_var`](crate::Tdd::condition_var) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors, [`OperationError::Stopped`] on
    /// cancellation, or [`OperationError::OverBudget`] on allocation refusal.
    pub fn condition_var(&self, f: Tdd, x: VarId, value: bool) -> Result<Tdd, OperationError> {
        crate::apply::condition::condition_vars_on(self, f, &[x], value)
    }

    /// Run [`Tdd::condition_vars`](crate::Tdd::condition_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors, [`OperationError::Stopped`] on
    /// cancellation, or [`OperationError::OverBudget`] on allocation refusal.
    pub fn condition_vars(&self, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
        crate::apply::condition::condition_vars_on(self, f, vars, value)
    }
}
