//! Negation: make-full transformation and complement operation.
//!
//! A diagram is **t-full** at vtree level `t` if the disjunction of all t-nodes
//! equals the constant-true function. A diagram is **full** if t-full at every level.
//! `expand_full` materializes the fill nodes explicitly; `negate` complements
//! the full diagram at its root.

use std::collections::HashMap;
use crate::engine::Engine;
use crate::limits::{ApplyError, PollGate};
use std::sync::Arc;

use crate::diagram::*;

/// Negate a diagram: make it full, then complement at the root, then minimize.
///
/// Consumes `f`; the result is canonical, ⊥ for ⊤ and ⊤ for ⊥. `f` must have
/// no marginal level: a summed-out level has no structure to complement, and
/// the result over one is not defined. [`Engine::negate`] runs this operation
/// under the caller's limits.
///
/// Exact, but it can grow the diagram sharply — a diagram stores only the pair
/// structure of its satisfying assignments, so the fill that has to precede the
/// complement typically dominates. When only the count of `¬f` is wanted,
/// `2^n - count(f)` avoids building it at all.
///
/// # Panics
///
/// Panics if `f` has a marginal level or an allocation is refused.
#[must_use]
pub fn negate(f: Tdd) -> Tdd {
    Engine::new().negate(f).expect("negate: use Engine::negate to handle a refusal")
}

impl Engine {
    /// Complement a structural diagram and reduce it under this engine's limits.
    ///
    /// # Errors
    ///
    /// Returns allocation and stop refusals from the fill, complement, or reduction.
    ///
    /// # Panics
    ///
    /// Panics if the diagram has a marginal level, whose structure was summed out.
    pub fn negate(&self, f: Tdd) -> Result<Tdd, ApplyError> {
        let _op = self.limits().begin_operation();
        let mut result = negate_tdd_owned(self, f)?;
        crate::reduce::try_minimize(self, &mut result, crate::reduce::ReductionPlan::default())?;
        Ok(result)
    }
}

/// Make `tdd` full and complement it at the root, consuming the operand;
/// [`negate()`] is this plus `minimize`.
pub(crate) fn negate_tdd_owned(eng: &Engine, mut tdd: Tdd) -> Result<Tdd, ApplyError> {
    assert!(!tdd.has_marginal_level(), "negate requires a structural diagram");
    if eng.limits().should_stop() { return Err(ApplyError::Deadline); }
    let weights = tdd.weights.take();
    let mut result = if tdd.is_zero() {
        crate::build::constant_one(eng, &tdd.vtree)
    } else {
        let vtree = Arc::clone(&tdd.vtree);
        expand_full(eng, &mut tdd)?;
        complement_full_at_root(eng, tdd, &vtree)?
    };
    result.weights = weights;
    Ok(result)
}

/// Complement a full diagram at its root: collect the root-level pairs not in
/// the output node and drop the dead ones. `orig_vtree` is the operand's
/// vtree, used for the constant fallbacks.
fn complement_full_at_root(eng: &Engine, full_tdd: Tdd, orig_vtree: &Arc<crate::vtree::Vtree>) -> Result<Tdd, ApplyError> {
    let vtree = &full_tdd.vtree;
    let root = vtree.root();
    let root_idx = root.idx();
    let out_local = full_tdd.output.local;

    let mut levels = full_tdd.levels;

    if vtree.node(root).is_leaf() {
        // Implicit leaf: the output index and the label are the same number. It stays in
        // {Pos,Neg,One} unless the output is One (complement = Zero, returned as the zero
        // constant diagram).
        let Some(neg_local) = complement_leaf_root(out_local) else {
            return Ok(crate::build::constant_zero(eng, orig_vtree));
        };
        Ok(Tdd::from_levels_unchecked(
            Arc::clone(orig_vtree),
            levels,
            TddNodeId { vtree: root, local: neg_local },
        ))
    } else {
        // After expand_full (which expands One → Pos+Neg), child widths:
        // - Leaf children: 2 (the disjoint set {Pos, Neg})
        // - Internal children: stored width (includes any fill nodes)
        let (left, right) = vtree.children(root);
        let lefts = ChildBasis::of(vtree, &levels, left);
        let rights = ChildBasis::of(vtree, &levels, right);

        let mut neg_pairs = collect_complement_pairs(eng, &levels[root_idx], out_local, lefts, rights)?;

        // Drop the pairs whose child computes the Zero function (an internal
        // node with no pairs), which a fill node of `expand_full` can be. The
        // root's children must not be marginal: the structural complement is
        // undefined on a partially aggregated diagram.
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();
        debug_assert!(
            left_is_leaf || !levels[left.idx()].is_marginal(),
            "negate_tdd: root's left child VtreeIdx({}) is marginal — \
             complement is undefined on a partially-aggregated TDD",
            left.idx()
        );
        debug_assert!(
            right_is_leaf || !levels[right.idx()].is_marginal(),
            "negate_tdd: root's right child VtreeIdx({}) is marginal",
            right.idx()
        );
        neg_pairs.retain(|pair| {
            let left_alive = left_is_leaf || {
                let node = &levels[left.idx()].nodes[pair.left.idx()];
                node.is_internal() && !levels[left.idx()].pairs_of(node).is_empty()
            };
            let right_alive = right_is_leaf || {
                let node = &levels[right.idx()].nodes[pair.right.idx()];
                node.is_internal() && !levels[right.idx()].pairs_of(node).is_empty()
            };
            left_alive && right_alive
        });

        if neg_pairs.is_empty() {
            return Ok(crate::build::constant_zero(eng, orig_vtree));
        }

        let neg_idx = levels[root_idx].push_node_on(eng, &neg_pairs)?;

        Ok(Tdd::from_levels_unchecked(
            Arc::clone(orig_vtree),
            levels,
            TddNodeId { vtree: root, local: neg_idx },
        ))
    }
}

// ── expand_full: explicit fill-node materialization ────────────────────────────

/// Make a diagram full by materializing fill nodes explicitly at every
/// structural level; marginal levels are skipped.
pub(crate) fn expand_full(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    let vtree = tdd.vtree.clone();

    // Expand One → {Pos, Neg} at levels with leaf children so all leaf
    // references are disjoint. After this, the leaf basis is {Pos, Neg}.
    expand_ones_at_leaf_parents(eng, tdd)?;

    for (t, left, right) in vtree.internal_bottomup() {
        // A marginal level has no node list to make full.
        if tdd.levels[t.idx()].is_marginal() {
            continue;
        }
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        expand_internal_explicit(eng, &mut tdd.levels[t.idx()], lefts, rights)?;
        if eng.limits().should_stop() { return Err(ApplyError::Deadline); }
    }
    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Complement a leaf label: Pos↔Neg, One↔Zero.
fn complement_label(label: LeafLabel) -> LeafLabel {
    match label {
        LeafLabel::Pos => LeafLabel::Neg,
        LeafLabel::Neg => LeafLabel::Pos,
        LeafLabel::One => LeafLabel::Zero,
        LeafLabel::Zero => LeafLabel::One,
    }
}

/// Complement of an implicit-leaf root, where the output index and the label
/// are the same number.
///
/// Returns `Some(complement_index)` when the complement is expressible in the
/// implicit leaf set {Pos, Neg, One}, or `None` if the complement would be Zero
/// (callers should substitute `Tdd::zero` in that case).
#[inline]
fn complement_leaf_root(out_local: NodeIdx) -> Option<NodeIdx> {
    let neg_label = complement_label(LeafLabel::from_idx(out_local.idx()));
    if neg_label == LeafLabel::Zero {
        None
    } else {
        Some(NodeIdx(neg_label as u32))
    }
}

/// Expand One-references at levels with leaf children to Pos+Neg pairs.
///
/// With implicit leaves, One (index 0) overlaps Pos (1) and Neg (2), and the
/// cross-product of `expand_full` needs disjoint leaf references; after
/// expansion the leaf basis is {Pos, Neg}.
fn expand_ones_at_leaf_parents(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    let vtree = tdd.vtree.clone();
    for (t, left, right) in vtree.internal_bottomup() {
        let left_leaf = vtree.node(left).is_leaf();
        let right_leaf = vtree.node(right).is_leaf();
        if left_leaf || right_leaf {
            expand_ones_in_level(eng, &mut tdd.levels[t.idx()], left_leaf, right_leaf)?;
        }
    }
    Ok(())
}

/// Expand One-references in a single level's pairs to Pos+Neg.
///
/// For each input pair, if a leaf child index is 0 (One), replace it with two
/// pairs: one for Pos (1) and one for Neg (2). Rebuilds the pairs arena and
/// nodes in-place with sorted, deduplicated pairs per node.
fn expand_ones_in_level(eng: &Engine, level: &mut TddLevel, left_leaf: bool, right_leaf: bool) -> Result<(), ApplyError> {
    let mut rebuilt = TddLevel::new();
    let mut pairs = Vec::new();
    let mut poll = PollGate::new(eng.limits().reduce_poll_stride());
    for node in &level.nodes {
        if !node.is_internal() {
            eng.limits().try_push(&mut rebuilt.nodes, *node)?;
            continue;
        }
        pairs.clear();
        for pair in level.pairs_of(node) {
            let lefts: &[NodeIdx] = if left_leaf && pair.left == ONE_LEAF_IDX { &[POS_LEAF_IDX, NEG_LEAF_IDX] } else { std::slice::from_ref(&pair.left) };
            let rights: &[NodeIdx] = if right_leaf && pair.right == ONE_LEAF_IDX { &[POS_LEAF_IDX, NEG_LEAF_IDX] } else { std::slice::from_ref(&pair.right) };
            for &left in lefts {
                for &right in rights { eng.limits().try_push(&mut pairs, InputPair { left, right })?; }
            }
            eng.limits().poll(&mut poll, 1)?;
        }
        pairs.sort_unstable();
        pairs.dedup();
        rebuilt.push_node_on(eng, &pairs)?;
    }
    rebuilt.n_tombstones = level.n_tombstones;
    *level = rebuilt;
    eng.limits().flush_poll(&mut poll)
}

// The leaf basis is a contiguous range only because Pos and Neg are adjacent.
const _: () = assert!(NEG_LEAF_IDX.0 == POS_LEAF_IDX.0 + 1);

/// Expanded child basis: the actual local indices the cross-product spans.
///
/// For a leaf child after One-expansion the basis is `{Pos, Neg}`, indices
/// `{1, 2}`; for an internal child every stored node is a basis element, so
/// the range is `0..width`. Both are contiguous, so membership is a bounds
/// test and enumeration a range walk.
#[derive(Copy, Clone)]
struct ChildBasis {
    /// First index in the basis.
    start: u32,
    /// One past the last index in the basis.
    end: u32,
}

impl ChildBasis {
    /// The basis `child` contributes, read off the vtree shape and level width.
    fn of(
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        child: crate::vtree::VtreeIdx,
    ) -> ChildBasis {
        if vtree.node(child).is_leaf() {
            ChildBasis { start: POS_LEAF_IDX.0, end: NEG_LEAF_IDX.0 + 1 }
        } else {
            ChildBasis { start: 0, end: levels[child.idx()].width() as u32 }
        }
    }

    /// Number of basis elements — one factor of the cross-product cell count.
    #[inline]
    fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    /// The one membership check: in-basis iff inside the contiguous range.
    #[inline]
    fn contains(self, idx: u32) -> bool {
        idx >= self.start && idx < self.end
    }

    /// Enumerate the basis in ascending index order.
    #[inline]
    fn iter(self) -> std::ops::Range<u32> {
        self.start..self.end
    }
}

/// Make an internal level t-full by materializing fill pairs explicitly.
fn expand_internal_explicit(
    eng: &Engine,
    level: &mut TddLevel,
    lefts: ChildBasis,
    rights: ChildBasis,
) -> Result<(), ApplyError> {
    // A level is full iff its nodes cover every cell of the `lefts × rights`
    // basis. Covered cells are collected into a set, since an un-minimized
    // level can list the same cell under two nodes, and only in-basis cells
    // count, so an out-of-range pair cannot mask a gap.
    let basis = lefts.len().checked_mul(rights.len()).ok_or(ApplyError::OverBudget)?;
    // `used` gains at most one entry per pair iterated, so reserve against the
    // level's pair mass rather than the basis, which is quadratic in the child
    // widths.
    let cap = basis.min(level.pair_count() + level.nodes.len());
    let mut used = HashMap::new();
    eng.limits().reserve_map(&mut used, cap)?;
    let mut poll = PollGate::new(eng.limits().reduce_poll_stride());
    for node in &level.nodes {
        if node.is_internal() {
            for pair in level.pairs_of(node) {
                if lefts.contains(pair.left.0) && rights.contains(pair.right.0) {
                    used.insert((pair.left.0, pair.right.0), ());
                    eng.limits().poll(&mut poll, 1)?;
                }
            }
        }
    }
    // Covered every basis cell ⇒ already full; skip the O(|L|·|R|) enumeration.
    if used.len() == basis {
        return eng.limits().flush_poll(&mut poll);
    }

    let fill_pairs = missing_cells(eng, &used, lefts, rights)?;
    if fill_pairs.is_empty() {
        return eng.limits().flush_poll(&mut poll);
    }

    level.push_node_on(eng, &fill_pairs)?;
    eng.limits().flush_poll(&mut poll)
}

/// The cells of `lefts × rights` that are not pairs of `exclude_node`.
fn collect_complement_pairs(
    eng: &Engine,
    level: &TddLevel,
    exclude_node: NodeIdx,
    lefts: ChildBasis,
    rights: ChildBasis,
) -> Result<Vec<InputPair>, ApplyError> {
    let exclude_pairs = level.pairs_of_idx(exclude_node.idx());
    let mut excluded = HashMap::new();
    eng.limits().reserve_map(&mut excluded, exclude_pairs.len())?;
    for pair in exclude_pairs.iter() {
        excluded.insert((pair.left.0, pair.right.0), ());
    }
    missing_cells(eng, &excluded, lefts, rights)
}

/// The cells of `lefts × rights` that `used` does not cover.
fn missing_cells(
    eng: &Engine,
    used: &HashMap<(u32, u32), ()>,
    lefts: ChildBasis,
    rights: ChildBasis,
) -> Result<Vec<InputPair>, ApplyError> {
    let mut out = Vec::new();
    let mut poll = PollGate::new(eng.limits().reduce_poll_stride());
    for l in lefts.iter() {
        for r in rights.iter() {
            if !used.contains_key(&(l, r)) {
                eng.limits().try_push(&mut out, InputPair { left: NodeIdx(l), right: NodeIdx(r) })?;
            }
            eng.limits().poll(&mut poll, 1)?;
        }
    }
    eng.limits().flush_poll(&mut poll)?;
    Ok(out)
}

#[cfg(test)]
mod tests;
