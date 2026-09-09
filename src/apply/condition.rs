//! Literal conditioning (cofactor): fix variables to constants, removing them.
//!
//! Unlike `project`, conditioning only rewrites the target leaf's parent level
//! (drops the opposite-polarity pairs, fixes the kept side to One) and never
//! calls `apply_or`, so it is sound when sibling levels are marginal (mc mode).
//! The conditioned leaf's OWN level and its parent's must be explicit — see
//! `assert_conditionable`, which fails fast instead of mis-conditioning.

use crate::engine::Engine;
use std::sync::Arc;

use crate::build::{constant_one, constant_zero};
use crate::marg_slots::ChildSide;
use crate::reduce::minimize;
use crate::diagram::sort_pairs;
use crate::diagram::{ExtMulti, InputPair, Tdd, TddNodeData, ZERO};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};
use crate::apply::project::{POS, NEG, ONE};

/// Polarity of a leaf restriction: keep the positive (Pos) or negative (Neg) branch.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Polarity {
    /// Keep the positive (x=⊤) branch.
    Pos,
    /// Keep the negative (x=⊥) branch.
    Neg,
}

/// The implementation behind [`Engine::condition_var`](crate::Engine::condition_var).
pub(crate) fn condition_var_on(eng: &Engine, f: &Tdd, x: VarId, value: bool) -> Tdd {
    if f.is_zero() {
        return f.clone();
    }
    let vtree = &f.vtree;
    let leaf_idx = vtree.leaf_of(x).expect("the vtree carries this variable");
    let pol = if value { Polarity::Pos } else { Polarity::Neg };
    condition_leaf(eng, f, leaf_idx, pol)
}

/// The implementation behind [`Engine::condition_vars`](crate::Engine::condition_vars).
pub(crate) fn condition_vars_on(eng: &Engine, f: &Tdd, vars: &[VarId], value: bool) -> Tdd {
    if f.is_zero() || vars.is_empty() {
        return f.clone();
    }
    let pol = if value { Polarity::Pos } else { Polarity::Neg };
    let vtree = Arc::clone(&f.vtree);
    let targets: std::collections::HashSet<VtreeIdx> =
        vars.iter().map(|&x| vtree.leaf_of(x).expect("the vtree carries this variable")).collect();
    for &leaf in &targets {
        assert_conditionable(f, leaf);
    }
    // Output sits at one of the target leaves: that var alone determines the result;
    // fall back to the per-var path for correctness (rare; copies are interior).
    if targets.contains(&f.output.vtree) {
        let mut result = f.clone();
        for &x in vars {
            result = condition_var_on(eng, &result, x, value);
        }
        return result;
    }
    let mut tdd = f.clone();
    for vi in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        if targets.contains(&left) {
            rewrite_for_restrict(&mut tdd, VtreeIdx(vi as u32), ChildSide::Left, pol);
        }
        if targets.contains(&right) {
            rewrite_for_restrict(&mut tdd, VtreeIdx(vi as u32), ChildSide::Right, pol);
        }
    }
    minimize(&mut tdd);
    canonicalize_false_output(eng, &mut tdd);
    tdd
}

/// Condition TDD `t` by fixing the variable at `leaf_idx` to ⊤ (polarity=Pos)
/// or ⊥ (polarity=Neg). Returns a fully minimized TDD. The leaf-space primitive
/// behind [`condition_var`] (by variable) and the cofactor-OR in [`project_var`].
///
/// After conditioning every reference to `leaf_idx` from its parent level becomes
/// `ONE`, so the leaf contributes a free (×2) factor in `model_count`. The vtree
/// is **unchanged** — the leaf remains in place.
pub(crate) fn condition_leaf(eng: &Engine, t: &Tdd, leaf_idx: VtreeIdx, polarity: Polarity) -> Tdd {
    assert_conditionable(t, leaf_idx);

    // When the TDD output is the leaf itself (single-variable vtree), the
    // conditioning is determined solely by the output label.
    if t.output.vtree == leaf_idx {
        return condition_leaf_output(eng, t, polarity);
    }

    let mut tdd = t.clone();
    let vtree = Arc::clone(&tdd.vtree);

    for vi in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        let left_is_target = left == leaf_idx;
        let right_is_target = right == leaf_idx;
        if !left_is_target && !right_is_target {
            continue;
        }
        if left_is_target {
            rewrite_for_restrict(&mut tdd, VtreeIdx(vi as u32), ChildSide::Left, polarity);
        }
        if right_is_target {
            rewrite_for_restrict(&mut tdd, VtreeIdx(vi as u32), ChildSide::Right, polarity);
        }
    }

    minimize(&mut tdd);
    // Conditioning + minimize can leave a semantically-false diagram non-canonical
    // (output node still has pairs, `model_count == 0`, `is_zero() == false`).
    // Counting it is correct; RE-CONJOINING it revives models the restriction
    // killed.
    canonicalize_false_output(eng, &mut tdd);
    tdd
}

/// Fail-fast precondition of the leaf rewrites: neither the conditioned leaf's level
/// nor its parent's may be marginal. `rewrite_for_restrict` matches the target-side
/// label against `POS`/`NEG`/`ONE` (LeafLabel indices 1/2/0) and a bare marg-slot ref
/// occupies the SAME numeric space (`types/marg.rs`) — slot 1 reads as `POS`, slot 5
/// falls into the keep-as-is arm — so a marginal level silently mis-conditions
/// instead of failing, and a marginal parent has no `nodes` at all (the rewrite is a
/// no-op). Soundness contract, not perf: a variable whose clauses are not all
/// compiled cannot have been marginalized, so a firing assert means the caller's
/// marginalize schedule is wrong.
fn assert_conditionable(t: &Tdd, leaf_idx: VtreeIdx) {
    assert!(
        !t.levels[leaf_idx.idx()].is_marginal(),
        "condition: leaf level {leaf_idx:?} is marginal — the variable was already summed out"
    );
    if let Some(parent) = t.vtree.node(leaf_idx).parent() {
        assert!(
            !t.levels[parent.idx()].is_marginal(),
            "condition: parent level {parent:?} of leaf {leaf_idx:?} is marginal — \
             the leaf's references are marg slots, not leaf labels"
        );
    }
}

/// Handle conditioning when the TDD output sits directly at the conditioned leaf.
fn condition_leaf_output(eng: &Engine, t: &Tdd, polarity: Polarity) -> Tdd {
    let vtree = &t.vtree;
    let output_label = t.output.local;
    let satisfied = if output_label == ZERO {
        false
    } else if output_label == ONE {
        true
    } else if output_label == POS {
        polarity == Polarity::Pos
    } else if output_label == NEG {
        polarity == Polarity::Neg
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
///
/// Compacted IN PLACE — no second arena beside the live one. Restriction is
/// NON-GROWING per node: every input pair either survives (its target-side label
/// rewritten to One) or is dropped, and none is ever added. So each node's
/// survivors fit in the PREFIX of the arena range that node already owns — the
/// write cursor `w` stays at or behind the read cursor `r`, every write lands on
/// a slot already read, and node indices are preserved exactly as the former
/// clear-and-repush rebuild preserved them. Same shrink-in-place shape as
/// `minimize/contract/dup_resolve.rs`'s duplicate resolution.
///
/// Abandoned range tails are reported through `note_dead_pairs` and reclaimed by
/// the arena's own amortized sweep. Unlike the rebuild — which went through
/// `TddLevel::clear` — this leaves `marg_flags`, the marginal width fields and the
/// tombstone count alone: restriction retires no marginal slot and changes no
/// side's inline-count encoding, so zeroing those was an artifact of reusing
/// `clear` for the arena reset, not a property of the rewrite.
fn rewrite_for_restrict(tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide, polarity: Polarity) {
    // Restriction of ONE pair: `None` = dropped (the pair belongs to the
    // opposite cofactor), `Some` = kept, with the target side fixed to One when
    // it named the conditioned leaf. `One`, and any reference to an internal
    // child, is carried through as-is.
    let restrict_pair = |p: InputPair| -> Option<InputPair> {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        if label != POS && label != NEG {
            return Some(p);
        }
        // x=⊤ pairs are excluded from the x=⊥ cofactor, and vice versa.
        if (label == POS) != (polarity == Polarity::Pos) {
            return None;
        }
        Some(if side == ChildSide::Left {
            InputPair { left: ONE, right: p.right }
        } else {
            InputPair { left: p.left, right: ONE }
        })
    };

    let level = &mut tdd.levels[parent_vi.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return;
    }

    let mut dead = 0usize;
    for i in 0..n_nodes {
        debug_assert!(level.nodes[i].is_internal(), "restrict rewrites internal nodes only");
        if !level.nodes[i].is_internal() {
            // Leaves and tombstones own no pair list (unreachable per the
            // assert) — leave the slot exactly as it is.
            continue;
        }

        if level.nodes[i].is_inline() {
            // The single pair lives in the node's own two words, not the arena.
            let p = level.nodes[i].inline_pair();
            match restrict_pair(p) {
                Some(np) => {
                    // Still inlinable: the untouched side keeps whatever bit it
                    // had, and the rewritten side becomes One (index 0), which
                    // sets none — so `can_inline` cannot go from true to false.
                    debug_assert!(np.can_inline(), "restricting an inline node cannot un-inline it");
                    level.nodes[i] = TddNodeData::inline(np);
                }
                None => {
                    // Emptied: the zero-pair placeholder the rebuild also
                    // produced here (an unsatisfiable node minimize prunes).
                    let empty = level.encode_multi(0, 0);
                    level.nodes[i] = empty;
                }
            }
            continue;
        }

        let start = level.multi_start_at(i);
        let old_len = level.multi_len_at(i);
        let pairs = level.pairs_mut(i);
        let mut w = 0usize;
        for r in 0..old_len {
            if let Some(np) = restrict_pair(pairs[r]) {
                // `w <= r`, so this write is at or below a slot already read.
                pairs[w] = np;
                w += 1;
            }
        }
        sort_pairs(&mut pairs[..w]);

        match w {
            0 => {
                let empty = level.encode_multi(0, 0);
                level.nodes[i] = empty;
            }
            1 => {
                // `pair_len == 1` aliases the extended encoding, so a lone
                // survivor is either inlined or pointed at through `ext` — at
                // the slot it ALREADY occupies, so even this shrink adds no
                // arena (unlike dup_resolve, whose survivor is a rewritten pair
                // that has to be pushed at the tail).
                let survivor = level.pairs[start];
                let data = if survivor.can_inline() {
                    TddNodeData::inline(survivor)
                } else {
                    let ext_idx = level.ext.len();
                    level.ext.push(ExtMulti { start: start as u64, len: 1 });
                    TddNodeData::multi_extended(ext_idx as u32)
                };
                level.nodes[i] = data;
            }
            _ => level.set_pair_len(i, w as u32),
        }
        // What the re-encoded node still owns is `arena_pairs_at` — 0 once it
        // went inline or empty; the rest of its old range is now garbage.
        dead += old_len - level.arena_pairs_at(i);
    }

    level.note_dead_pairs(dead);
    // Self-gated amortized sweep: it fires only past its dead-slot floor AND
    // when over half the arena is garbage, so a small restriction pays nothing.
    // Its precondition holds by construction — every node kept a PREFIX of its
    // own range, so live ranges stay pairwise disjoint.
    level.compact_pairs_if_stale();
    tdd.dirty.contract.push(parent_vi.0);
}

/// Restore the `is_zero`/`is_sat_minimized` invariant on `tdd`: collapse a structurally-false
/// diagram (output node has pairs, `model_count == 0`, `is_zero() == false`) to the
/// `ZERO` sentinel. `condition_leaf`/`condition_vars` produce that state whenever
/// conditioning plus `minimize` kills every model without emptying the output node.
/// `is_sat_structural` agrees with `model_count > 0` by construction, so this
/// never changes a model count — only the false case's structural form.
///
/// Declines on a WEIGHTED diagram: a weight-marginal level keeps its per-node values
/// in the external `WeightStore`, not in `marginal_counts`, so the satisfiability pass
/// cannot evaluate it. No weighted path re-conjoins a conditioned diagram today; one
/// that does needs a `WeightStore`-aware satisfiability pass first.
fn canonicalize_false_output(_eng: &Engine, tdd: &mut crate::diagram::Tdd) {
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
#[path = "condition_tests.rs"]
mod restrict_in_place_tests;

/// Fix `x` to `value`, on a transient engine.
///
/// [`Engine::condition_var`] is this operation on a caller's engine, where the
/// per-level buffers stay warm between calls.
#[must_use]
pub fn condition_var(f: &Tdd, x: VarId, value: bool) -> Tdd {
    condition_var_on(&Engine::new(), f, x, value)
}

/// Fix every variable in `vars` to `value`, on a transient engine.
///
/// [`Engine::condition_vars`] is this operation on a caller's engine.
#[must_use]
pub fn condition_vars(f: &Tdd, vars: &[VarId], value: bool) -> Tdd {
    condition_vars_on(&Engine::new(), f, vars, value)
}

/// The conditioning entry points on a caller's engine.
impl crate::engine::Engine {
    /// Condition `x` to a constant `value`, removing it from the result (cofactor).
    /// Marginal-safe: unlike `project_var`, this only rewrites x's leaf-parent level
    /// (drops the opposite-polarity pairs, fixes the kept side to One) and never calls
    /// `apply_or`, so it is sound when sibling levels are marginal (mc mode). Restriction
    /// is monotone non-increasing in size — it can never blow up like a general apply.
    #[must_use]
    pub fn condition_var(&self, f: &Tdd, x: VarId, value: bool) -> Tdd {
        crate::apply::condition::condition_var_on(self, f, x, value)
    }

    /// Condition a SET of variables to the same constant `value`, removing them all,
    /// with a SINGLE `minimize` at the end (vs one per var in `condition_var`). Much
    /// cheaper when conditioning many copies of one hub on a large diagram. Marginal-safe
    /// for the same reason as `condition_var`. Like `condition_var`, the kept side is set
    /// to One (free) — the caller must divide the final count by 2^(#vars conditioned).
    #[must_use]
    pub fn condition_vars(&self, f: &Tdd, vars: &[VarId], value: bool) -> Tdd {
        crate::apply::condition::condition_vars_on(self, f, vars, value)
    }
}
