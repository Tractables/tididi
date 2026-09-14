//! Literal conditioning (cofactor): fix variables to constants, removing them.
//!
//! Conditioning only rewrites the target leaf's parent level (drops the
//! opposite-polarity pairs, fixes the kept side to One) and never disjoins, so
//! it is sound when sibling levels are marginal. Both the conditioned leaf's
//! own level and its parent's must be structural; `check_conditionable`
//! checks that.

use crate::diagram::Changed;
use crate::engine::Engine;
use std::sync::Arc;

use crate::build::constant_like;
use crate::diagram::ChildSide;
use crate::limits::OperationError;
use crate::reduce::{try_reduce, ReductionPlan};
use crate::diagram::sort_pairs;
use crate::diagram::{EncodedChildRef, ChildDecoder, ChildPair, Tdd, TddLevel, EncodedNode, ZERO};
use crate::vtree::{VarId, VtreeIdx};
use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};

/// Polarity of a leaf restriction: keep the positive (Pos) or negative (Neg) branch.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Polarity {
    /// Keep the positive (x=⊤) branch.
    Positive,
    /// Keep the negative (x=⊥) branch.
    Negative,
}

/// The implementation behind [`Engine::condition_var`](crate::Engine::condition_var).
pub(crate) fn condition_var_on(eng: &Engine, f: Tdd, x: VarId, value: bool) -> Result<Tdd, OperationError> {
    condition_vars_on(eng, f, &[x], value)
}

/// The implementation behind [`Engine::condition_vars`](crate::Engine::condition_vars).
pub(crate) fn condition_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
    condition_on(eng, f, vars.iter().map(|&var| crate::diagram::Literal::new(var, value)))
}

/// Validate a mixed assignment and condition all its leaves in one reduction.
pub(crate) fn condition_on(eng: &Engine, f: Tdd, assignment: impl IntoIterator<Item = impl Into<crate::diagram::Literal>>) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    let mut targets = Vec::new();
    for literal in assignment {
        let literal = literal.into();
        let leaf = f.vtree.leaf_of(literal.var).ok_or(OperationError::VariableNotInVtree(literal.var))?;
        let pol = if literal.positive { Polarity::Positive } else { Polarity::Negative };
        eng.limits().try_push(&mut targets, (leaf, pol))?;
    }
    targets.sort_unstable_by_key(|&(leaf, _)| leaf);
    let contradictory = targets.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
    if contradictory {
        return Ok(constant_like(eng, &f, false));
    }
    targets.dedup_by_key(|entry| entry.0);
    if f.is_zero() || targets.is_empty() { return Ok(f); }
    condition_targets(eng, f, &mut targets)
}

/// Propagate falsity upward after a restriction rewrite, so that no node left
/// in the diagram computes ⊥.
///
/// Restriction empties a node whenever every one of its pairs belonged to the
/// opposite cofactor. That node computes ⊥, which invariant 2
/// (`docs/architecture.md`) forbids, and every reduction rule that follows
/// assumes such a node is already gone. Conditioning rewrites in place, so it
/// restores the invariant here: one bottom-up pass dropping every pair whose
/// structural child is empty, which empties further nodes above and cascades.
/// What is left unreferenced is removed by `prune_unreachable` in the
/// reduction that follows; an emptied output is collapsed to the sentinel by
/// `canonicalize_false_output`.
///
/// Marginal levels are passed over: their structure is summed out, so they hold
/// no node that could have been emptied by a leaf restriction.
fn propagate_false_nodes(tdd: &mut Tdd) {
    let vtree = Arc::clone(&tdd.vtree);
    for (vi, left, right) in vtree.internal_bottomup() {
        if tdd.levels[vi.idx()].is_marginal() { continue; }
        let [parent, left_level, right_level] = tdd.levels
            .get_disjoint_mut([vi.idx(), left.idx(), right.idx()])
            .expect("a parent and its children are distinct levels");
        // Leaf labels and marginal values do not name structural nodes.
        let left_structural = !vtree.node(left).is_leaf() && !left_level.is_marginal();
        let right_structural = !vtree.node(right).is_leaf() && !right_level.is_marginal();
        let has_empty = |structural: bool, level: &TddLevel| {
            structural && (0..level.nodes.len()).any(|i| empty_node(level, i))
        };
        if parent.nodes.is_empty()
            || !(has_empty(left_structural, left_level) || has_empty(right_structural, right_level))
        { continue; }
        let dead = |structural: bool, level: &TddLevel, child: EncodedChildRef| {
            child == ZERO.into()
                || (structural && empty_node(level, ChildDecoder::structural().node(child).idx()))
        };
        rewrite_level_pairs(parent, |pair| {
            if dead(left_structural, left_level, pair.left) || dead(right_structural, right_level, pair.right) {
                None
            } else {
                Some(pair)
            }
        });
        tdd.invalidate(vi, Changed::PAIRS);
    }
    let output = tdd.output;
    if empty_node(&tdd.levels[output.vtree.idx()], output.local.idx()) {
        tdd.output.local = ZERO;
    }
}

/// Whether an existing structural node owns no pairs; tombstones are not nodes.
fn empty_node(level: &TddLevel, i: usize) -> bool {
    level.nodes[i].is_internal() && level.pair_count_at(i) == 0
}

/// Condition one already-resolved leaf for the cofactor-OR quantifier.
pub(crate) fn condition_leaf(eng: &Engine, t: Tdd, leaf: VtreeIdx, polarity: Polarity) -> Result<Tdd, OperationError> {
    condition_targets(eng, t, &mut [(leaf, polarity)])
}

/// Validate distinct target leaves, rewrite their parents, propagate falsity, and reduce once.
fn condition_targets(
    eng: &Engine,
    mut tdd: Tdd,
    targets: &mut [(VtreeIdx, Polarity)],
) -> Result<Tdd, OperationError> {
    for &(leaf, _) in targets.iter() { check_conditionable(&tdd, leaf)?; }
    if let Some(&(_, pol)) = targets.iter().find(|&&(leaf, _)| leaf == tdd.output.vtree) {
        return Ok(condition_leaf_output(eng, &tdd, pol));
    }
    let vtree = Arc::clone(&tdd.vtree);
    // Parent index, then left before right, fixes the rewrite and invalidation order.
    let route = |leaf| {
        let parent = vtree.node(leaf).parent().expect("a non-output leaf has a parent");
        (parent, vtree.children(parent).1 == leaf)
    };
    targets.sort_unstable_by_key(|&(leaf, _)| route(leaf));
    let mut emptied = false;
    for &(leaf, pol) in targets.iter() {
        let (parent, right) = route(leaf);
        let side = if right { ChildSide::Right } else { ChildSide::Left };
        emptied |= rewrite_for_restrict(&mut tdd, parent, side, pol);
    }
    if emptied { propagate_false_nodes(&mut tdd); }

    // Set the false sentinel before pruning, so its empty nodes are unreachable.
    canonicalize_false_output(&mut tdd);
    try_reduce(eng, &mut tdd, ReductionPlan::default())?;
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
fn condition_leaf_output(eng: &Engine, t: &Tdd, polarity: Polarity) -> Tdd {
    let output_label = t.output.local;
    let satisfied = if output_label == ZERO {
        false
    } else if output_label == ONE_LEAF_IDX {
        true
    } else if output_label == POS_LEAF_IDX {
        polarity == Polarity::Positive
    } else if output_label == NEG_LEAF_IDX {
        polarity == Polarity::Negative
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
fn rewrite_for_restrict(tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide, polarity: Polarity) -> bool {
    // Restriction of one pair: `None` = dropped (the pair belongs to the
    // opposite cofactor), `Some` = kept, with the target side fixed to One when
    // it named the conditioned leaf. `One`, and any reference to an internal
    // child, is carried through as-is.
    if tdd.levels[parent_vi.idx()].nodes.is_empty() { return false; }
    let emptied = rewrite_level_pairs(&mut tdd.levels[parent_vi.idx()], |p: ChildPair| {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        if label != POS_LEAF_IDX.into() && label != NEG_LEAF_IDX.into() {
            return Some(p);
        }
        // x=⊤ pairs are excluded from the x=⊥ cofactor, and vice versa.
        if (label == POS_LEAF_IDX.into()) != (polarity == Polarity::Positive) {
            return None;
        }
        Some(if side == ChildSide::Left {
            ChildPair::new(ONE_LEAF_IDX, p.right)
        } else {
            ChildPair::new(p.left, ONE_LEAF_IDX)
        })
    });
    tdd.invalidate(parent_vi, Changed::PAIRS);
    emptied
}

/// Rewrite a level's pair lists in place through `rewrite_pair`,
/// dropping every pair it answers `None` for. Answers whether any node was left
/// with no pairs at all.
///
/// Both of conditioning's rewrites are this pass under a different predicate:
/// the leaf restriction above, and the falsity sweep below.
fn rewrite_level_pairs(
    level: &mut TddLevel,
    rewrite_pair: impl Fn(ChildPair) -> Option<ChildPair>,
) -> bool {
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return false;
    }

    let mut emptied = false;
    let mut dead = 0usize;
    for i in 0..n_nodes {
        if !level.nodes[i].is_internal() {
            // Leaves and tombstones own no pair list — leave the slot as it is.
            continue;
        }

        if level.nodes[i].is_inline() {
            // The single pair lives in the node's own two words, not the arena.
            let p = level.nodes[i].inline_pair();
            match rewrite_pair(p) {
                Some(np) => {
                    // Still inlinable: the untouched side keeps whatever bit it
                    // had, and the rewritten side becomes One (index 0), which
                    // sets none — so `can_inline` cannot go from true to false.
                    debug_assert!(np.can_inline(), "restricting an inline node cannot un-inline it");
                    level.nodes[i] = EncodedNode::inline(np);
                }
                None => {
                    // Emptied: the slot holds no pair, which `propagate_false_nodes`
                    // reads as the node computing false and drops every reference to.
                    let empty = level.encode_multi(0, 0);
                    level.nodes[i] = empty;
                    emptied = true;
                }
            }
            continue;
        }

        let start = level.multi_start_at(i);
        let old_len = level.multi_len_at(i);
        let pairs = level.pairs_mut(i);
        let mut w = 0usize;
        for r in 0..old_len {
            if let Some(np) = rewrite_pair(pairs[r]) {
                // `w <= r`, so this write is at or below a slot already read.
                pairs[w] = np;
                w += 1;
            }
        }
        sort_pairs(&mut pairs[..w]);

        match w {
            0 => {
                // As in the inline arm: an empty slot is the node computing
                // false, and the falsity sweep drops what still names it.
                let empty = level.encode_multi(0, 0);
                level.nodes[i] = empty;
                emptied = true;
            }
            1 => {
                // `pair_len == 1` aliases the extended encoding, so a lone
                // survivor is either inlined or pointed at through `multi_pairs`
                // at the slot it already occupies.
                let survivor = level.pairs[start];
                level.nodes[i] = level.encode_single(start, survivor);
            }
            _ => level.set_pair_len(i, w as u32),
        }
        // What the re-encoded node still owns is `arena_pairs_at` — 0 once it
        // went inline or empty; the rest of its old range is now garbage.
        dead += old_len - level.arena_pairs_at(i);
    }

    level.note_dead_pairs(dead);
    // The sweep's precondition holds: every node kept a prefix of its own
    // range, so live ranges stay pairwise disjoint.
    level.compact_pairs_if_stale();
    emptied
}

/// Collapse a structurally false diagram (output node has pairs,
/// `model_count == 0`, `is_zero() == false`) to the `ZERO` sentinel; conditioning
/// plus `minimize` produces that state whenever it kills every model without
/// emptying the output node. `is_sat_structural` agrees with `model_count > 0`,
/// so the count is unchanged.
///
/// Declines on a weighted diagram: a weight-marginal level keeps its values in
/// the `WeightStore`, which the satisfiability pass cannot evaluate.
fn canonicalize_false_output(tdd: &mut crate::diagram::Tdd) {
    if tdd.is_zero() {
        return;
    }
    if tdd.levels.iter().any(|l| l.is_weight_marginal()) {
        return;
    }
    let sat = crate::query::sat::is_sat_structural(tdd);
    debug_assert_eq!(
        sat,
        crate::query::model_count(tdd) != num_bigint::BigUint::ZERO,
        "is_sat_structural disagrees with model_count > 0"
    );
    if !sat {
        tdd.output.local = crate::diagram::ZERO;
    }
}

#[cfg(test)]
mod tests;

/// Fix `x` to `value`, on a transient engine with no limits armed.
///
/// `f` is borrowed and cloned; the result is what [`Engine::condition_var`]
/// returns, so its contract — `x` stays a free variable of the vtree, the
/// result is canonical — is stated there. That method is this operation on a
/// caller's engine: it keeps the per-level buffers warm between calls, takes
/// the operand by value, and hands a refused allocation or a variable outside
/// the vtree back instead of panicking.
///
/// # Panics
///
/// Panics if `x` is not a variable of `f`'s vtree, if `x`'s leaf level or its
/// parent level is marginal, and if an allocation is refused.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::apply::condition_var;
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);  // x1 ∧ x2
/// assert_eq!(f.model_count(), 2u32.into());
///
/// // x1 := true leaves x2, and x1 stays a variable of the vtree, now free.
/// let g = condition_var(&f, VarId(0), true);
/// assert_eq!(g.model_count(), 4u32.into());
/// // x1 := false leaves the constant-false function.
/// assert!(condition_var(&f, VarId(0), false).is_zero());
/// ```
#[must_use]
pub fn condition_var(f: &Tdd, x: VarId, value: bool) -> Tdd {
    condition_var_on(&Engine::new(), f.clone(), x, value)
        .expect("condition_var: use Engine::condition_var to handle a refusal or a variable outside the vtree")
}

/// Fix every variable in `vars` to `value`, on a transient engine with no
/// limits armed.
///
/// `f` is borrowed and cloned. [`Engine::condition_vars`] is this operation on
/// a caller's engine.
///
/// # Panics
///
/// Panics if any of `vars` is not a variable of `f`'s vtree, if a named
/// variable's leaf level or its parent level is marginal, and if an
/// allocation is refused.
#[must_use]
pub fn condition_vars(f: &Tdd, vars: &[VarId], value: bool) -> Tdd {
    condition_vars_on(&Engine::new(), f.clone(), vars, value)
        .expect("condition_vars: use Engine::condition_vars to handle a refusal or a variable outside the vtree")
}

/// The conditioning entry points on a caller's engine.
impl crate::engine::Engine {
    /// Substitute a mixed assignment, returning a minimized cofactor.
    ///
    /// Repeated equal literals are ignored; opposite literals for one variable
    /// produce the constant-false diagram. All variables are checked before a
    /// contradictory assignment or false input is returned. Fixed variables
    /// remain free leaves of the vtree, as with [`Self::condition_var`].
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable and
    /// propagates allocation or stop refusals from the caller's engine.
    ///
    /// [`OperationError::MarginalLevel`] when a consistent assignment needs
    /// a leaf or parent level whose structure was summed out.
    ///
    /// # Panics
    ///
    /// If a literal conversion panics, including integer zero.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd};
    /// use tididi::vtree::Vtree;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&tree, [1, 2, 3]);
    /// let result = engine.condition(f, [-1, -2]).unwrap();
    /// assert_eq!(result.model_count(), 4u32.into());
    /// ```
    pub fn condition(&self, f: Tdd, assignment: impl IntoIterator<Item = impl Into<crate::diagram::Literal>>) -> Result<Tdd, OperationError> {
        condition_on(self, f, assignment)
    }

    /// Substitute `value` for `x`, returning a minimized cofactor.
    ///
    /// The vtree is unchanged: `x` becomes free, so the result's model count includes
    /// a factor of two for it. To count observations without this free-variable
    /// factor, use [`ModelCounter`](crate::query::ModelCounter) with
    /// [`PinSemantics::Evidence`](crate::query::PinSemantics::Evidence).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.clause(&tree, [1, 2])?; // x1 or x2, with x3 free
    /// let g = engine.condition_var(f, VarId(0), false)?; // x2, with x1 and x3 free
    /// assert_eq!(engine.model_count(&g)?, 4u32.into());
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// The operand is consumed on success or error. Only the target's leaf and
    /// parent need their structure; other levels may be marginal. Conditioning
    /// preserves attached weights and does not grow the diagram.
    /// For structural and count-marginal inputs, a false result uses the
    /// [`Tdd::is_zero`] sentinel. With weighted marginal values, zero evaluation
    /// can instead reflect zero weights or cancellation; it does not prove Boolean
    /// unsatisfiability.
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] when a nonfalse input needs a target leaf
    /// or parent already summed out, or an allocation or stop refusal.
    /// The variable is validated even for a false input.
    pub fn condition_var(&self, f: Tdd, x: VarId, value: bool) -> Result<Tdd, OperationError> {
        crate::apply::condition::condition_var_on(self, f, x, value)
    }

    /// Substitute the same Boolean `value` for every variable in `vars`.
    ///
    /// Uses one propagation and reduction pass for the whole assignment; repeated
    /// variables are ignored. Each conditioned variable remains free in the vtree,
    /// as described by [`Engine::condition_var`]. An empty list returns `f`
    /// unchanged. Use [`Engine::condition`] for mixed polarities.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.clause(&tree, [1, 2, 3])?;
    /// let g = engine.condition_vars(f, &[VarId(0), VarId(1)], false)?;
    /// assert_eq!(engine.model_count(&g)?, 4u32.into()); // x3, with x1 and x2 free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Engine::condition_var`]; all variables are validated before rewriting.
    /// The operand is consumed on error as well as success.
    pub fn condition_vars(&self, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
        crate::apply::condition::condition_vars_on(self, f, vars, value)
    }
}
