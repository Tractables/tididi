//! Literal conditioning (cofactor): fix variables to constants, removing them.
//!
//! Conditioning only rewrites the target leaf's parent level (drops the
//! opposite-polarity pairs, fixes the kept side to One) and never disjoins, so
//! it is sound when sibling levels are marginal. Both the conditioned leaf's
//! own level and its parent's must be structural; `assert_conditionable`
//! checks that.

use crate::diagram::Changed;
use crate::engine::Engine;
use std::sync::Arc;

use crate::build::{constant_one, constant_zero};
use crate::diagram::ChildSide;
use crate::limits::ApplyError;
use crate::reduce::{try_minimize, MinimizeOptions};
use crate::diagram::sort_pairs;
use crate::diagram::{InputPair, Tdd, TddNodeData, ZERO};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};
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
pub(crate) fn condition_var_on(eng: &Engine, f: Tdd, x: VarId, value: bool) -> Result<Tdd, ApplyError> {
    condition_vars_on(eng, f, &[x], value)
}

/// The implementation behind [`Engine::condition_vars`](crate::Engine::condition_vars).
pub(crate) fn condition_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, ApplyError> {
    // Caller input, so the whole set is answered before any work and before the
    // shortcuts: the same request is refused whatever the operand happens to be.
    let mut targets: Vec<VtreeIdx> = vars
        .iter()
        .map(|&x| f.vtree.leaf_of(x).ok_or(ApplyError::VariableNotInVtree(x)))
        .collect::<Result<_, _>>()?;
    if f.is_zero() || vars.is_empty() {
        return Ok(f);
    }
    targets.sort_unstable();
    let pol = if value { Polarity::Positive } else { Polarity::Negative };
    condition_leaves(eng, f, &targets, pol)
}

/// Restrict every reference to a target leaf, on whichever side of its parent
/// it appears, to the given polarity.
///
/// The two conditioning entry points differ only in which leaves are targets —
/// one leaf, or a set of them.
///
/// Answers whether the rewrite left any node with no pairs, which is what
/// decides whether the caller runs `propagate_false_nodes`: a restriction can
/// only make a node compute ⊥ by taking away its last pair, and the only pairs
/// it takes away are those on the parent levels rewritten here, so a rewrite
/// that emptied nothing has left nothing for the sweep to propagate.
fn rewrite_parents_of(tdd: &mut Tdd, is_target: impl Fn(VtreeIdx) -> bool, pol: Polarity) -> bool {
    let vtree = Arc::clone(&tdd.vtree);
    let mut emptied = false;
    for vi in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        if is_target(left) {
            emptied |= rewrite_for_restrict(tdd, VtreeIdx(vi as u32), ChildSide::Left, pol);
        }
        if is_target(right) {
            emptied |= rewrite_for_restrict(tdd, VtreeIdx(vi as u32), ChildSide::Right, pol);
        }
    }
    emptied
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
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a level's flag table cannot be
/// reserved; nothing has been rewritten by then.
fn propagate_false_nodes(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    let vtree = Arc::clone(&tdd.vtree);
    let lim = eng.limits();
    // `is_false[v][i]`: node `i` of level `v` has no pairs left. Filled in
    // bottom-up, so a level's children are decided before the level is.
    let mut is_false: Vec<Vec<bool>> = Vec::with_capacity(tdd.levels.len());
    for l in &tdd.levels {
        let mut flags = Vec::new();
        lim.try_resize(&mut flags, l.nodes.len(), false)?;
        is_false.push(flags);
    }

    for vi in vtree.bottomup() {
        let (left, right) = match *vtree.node(vi) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        if tdd.levels[vi.idx()].is_marginal() {
            continue;
        }
        // A child side is "opaque" when its references are not node indices into
        // a structural level: leaf labels, or marginal slots.
        let opaque = |c: VtreeIdx| {
            matches!(*vtree.node(c), VtreeNode::Leaf { .. }) || tdd.levels[c.idx()].is_marginal()
        };
        let (l_opaque, r_opaque) = (opaque(left), opaque(right));
        let l_false = std::mem::take(&mut is_false[left.idx()]);
        let r_false = std::mem::take(&mut is_false[right.idx()]);
        let dead = |opaque: bool, table: &[bool], c: crate::diagram::NodeIdx| {
            // `ZERO` is ⊥ on any side; otherwise only a structural side can carry
            // a node this pass has decided.
            c == ZERO || (!opaque && table[c.idx()])
        };
        if l_false.iter().any(|&b| b) || r_false.iter().any(|&b| b) {
            rewrite_level_pairs(tdd, vi, |p: InputPair| {
                if dead(l_opaque, &l_false, p.left) || dead(r_opaque, &r_false, p.right) {
                    None
                } else {
                    Some(p)
                }
            });
        }
        is_false[left.idx()] = l_false;
        is_false[right.idx()] = r_false;

        let level = &tdd.levels[vi.idx()];
        let flags = &mut is_false[vi.idx()];
        for (i, node) in level.nodes.iter().enumerate() {
            flags[i] = node.is_internal() && level.pair_count_at(i) == 0;
        }
    }
    Ok(())
}

/// Condition `t` at every leaf of `targets` (sorted) at once, fixing each
/// variable to ⊤ (polarity=Pos) or ⊥ (polarity=Neg). Returns a fully minimized
/// diagram. The leaf-space primitive behind [`condition_vars_on`] and the
/// cofactor-OR in [`project_var`](crate::apply::project_var).
///
/// Consumes `t`: the rewrite runs in the level arenas the caller hands over,
/// and the reduction that follows may refuse. Nothing comes back on `Err`.
///
/// After conditioning every reference to a target leaf from its parent level
/// becomes `ONE_LEAF_IDX`, so the leaf contributes a free (×2) factor in
/// `model_count`. The vtree is **unchanged** — the leaf remains in place.
pub(crate) fn condition_leaves(eng: &Engine, t: Tdd, targets: &[VtreeIdx], polarity: Polarity) -> Result<Tdd, ApplyError> {
    for &leaf in targets {
        assert_conditionable(&t, leaf);
    }

    // When the diagram output is a target leaf itself, the diagram is that one
    // literal and its label alone decides the result.
    if targets.binary_search(&t.output.vtree).is_ok() {
        return Ok(condition_leaf_output(eng, &t, polarity));
    }

    let mut tdd = t;
    if rewrite_parents_of(&mut tdd, |t| targets.binary_search(&t).is_ok(), polarity) {
        propagate_false_nodes(eng, &mut tdd)?;
    }

    try_minimize(eng, &mut tdd, MinimizeOptions::default())?;
    // Conditioning + minimize can leave a semantically-false diagram non-canonical
    // (output node still has pairs, `model_count == 0`, `is_zero() == false`).
    // Counting it is correct, but re-conjoining it revives models the
    // restriction killed.
    canonicalize_false_output(&mut tdd);
    Ok(tdd)
}

/// Precondition of the leaf rewrites: neither the conditioned leaf's level nor
/// its parent's may be marginal. `rewrite_for_restrict` matches the
/// target-side label against `POS_LEAF_IDX`/`NEG_LEAF_IDX`/`ONE_LEAF_IDX`, and
/// a bare marginal-slot ref occupies the same numeric space
/// (`diagram/level/marginal.rs`), so a marginal leaf level would be silently
/// mis-conditioned; a marginal parent has no pairs, so the rewrite is a no-op.
fn assert_conditionable(t: &Tdd, leaf_idx: VtreeIdx) {
    assert!(
        !t.levels[leaf_idx.idx()].is_marginal(),
        "condition: leaf level {leaf_idx:?} is marginal — the variable was already summed out"
    );
    if let Some(parent) = t.vtree.node(leaf_idx).parent() {
        assert!(
            !t.levels[parent.idx()].is_marginal(),
            "condition: parent level {parent:?} of leaf {leaf_idx:?} is marginal — \
             the leaf's references are marginal slots, not leaf labels"
        );
    }
}

/// Handle conditioning when the diagram output sits directly at the conditioned leaf.
fn condition_leaf_output(eng: &Engine, t: &Tdd, polarity: Polarity) -> Tdd {
    let vtree = &t.vtree;
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

    if satisfied {
        constant_one(eng, &Arc::clone(vtree))
    } else {
        constant_zero(eng, &Arc::clone(vtree))
    }
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
    rewrite_level_pairs(tdd, parent_vi, |p: InputPair| {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        if label != POS_LEAF_IDX && label != NEG_LEAF_IDX {
            return Some(p);
        }
        // x=⊤ pairs are excluded from the x=⊥ cofactor, and vice versa.
        if (label == POS_LEAF_IDX) != (polarity == Polarity::Positive) {
            return None;
        }
        Some(if side == ChildSide::Left {
            InputPair { left: ONE_LEAF_IDX, right: p.right }
        } else {
            InputPair { left: p.left, right: ONE_LEAF_IDX }
        })
    })
}

/// Rewrite level `parent_vi`'s pair lists in place through `rewrite_pair`,
/// dropping every pair it answers `None` for. Answers whether any node was left
/// with no pairs at all.
///
/// Both of conditioning's rewrites are this pass under a different predicate:
/// the leaf restriction above, and the falsity sweep below.
fn rewrite_level_pairs(
    tdd: &mut Tdd,
    parent_vi: VtreeIdx,
    rewrite_pair: impl Fn(InputPair) -> Option<InputPair>,
) -> bool {
    let level = &mut tdd.levels[parent_vi.idx()];
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
                    level.nodes[i] = TddNodeData::inline(np);
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
    tdd.invalidate(parent_vi, Changed::PAIRS);
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
    /// Condition `x` to a constant `value` (cofactor). `x` stays a variable of
    /// the vtree, now free, so the count keeps its factor of two for `x`.
    /// Only `x`'s leaf-parent level is rewritten (the opposite-polarity pairs
    /// are dropped, the kept side fixed to One) and nothing is disjoined, so
    /// this is sound when other levels are marginal and never grows the
    /// diagram. The result is reduced: canonical, and ⊥ (`is_zero()`) when no
    /// model is left, except on a weighted diagram, where a result with no
    /// model may keep its nodes and only its value says so. A ⊥ operand
    /// comes back unchanged.
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, the rule
    /// [`Engine::and`] states: the rewrite runs in `f`'s own level arenas.
    /// Clone it first if you need to keep it.
    ///
    /// # Errors
    ///
    /// [`ApplyError::VariableNotInVtree`] when `x` is not a variable of `f`'s
    /// vtree, reported before any work is done,
    /// [`ApplyError::OverBudget`] when the reduction's reservation is refused,
    /// [`ApplyError::Deadline`] on the armed deadline or a stop decision.
    ///
    /// # Panics
    ///
    /// Panics if `x`'s leaf level or its parent level is marginal: a
    /// summed-out variable cannot be conditioned.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::limits::LimitSet;
    /// # use tididi::vtree::{VarId, Vtree};
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    /// let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    /// let g = engine.condition_var(f, VarId(0), true).unwrap();
    /// assert!(!g.is_zero());
    ///
    /// // A byte budget of zero refuses the rewrite's reservations.
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let h = Tdd::clause(&wide, [1, -2]) & Tdd::clause(&wide, [2, 3]);
    /// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
    /// match engine.condition_var(h, VarId(0), true) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
    /// }
    /// ```
    pub fn condition_var(&self, f: Tdd, x: VarId, value: bool) -> Result<Tdd, ApplyError> {
        crate::apply::condition::condition_var_on(self, f, x, value)
    }

    /// Condition every variable in `vars` to the same constant `value`, with
    /// one reduction at the end rather than one per variable as in
    /// [`Engine::condition_var`]. As there, each conditioned variable stays in
    /// the vtree as a free variable, so the count keeps a factor of two per
    /// variable.
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::condition_var`]. An empty `vars` returns `f` unchanged.
    ///
    /// # Errors
    ///
    /// As [`Engine::condition_var`]; every variable is checked against the
    /// vtree before any work is done.
    ///
    /// # Panics
    ///
    /// As [`Engine::condition_var`], for any variable in `vars`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::limits::LimitSet;
    /// # use tididi::vtree::{VarId, Vtree};
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    /// // A byte budget of zero refuses the rewrite's first reservation.
    /// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
    /// let h = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    /// match engine.condition_vars(h, &[VarId(0), VarId(1)], true) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
    /// }
    /// ```
    pub fn condition_vars(&self, f: Tdd, vars: &[VarId], value: bool) -> Result<Tdd, ApplyError> {
        crate::apply::condition::condition_vars_on(self, f, vars, value)
    }
}
