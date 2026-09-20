//! Negation: make-full transformation and complement operation.
//!
//! A diagram is **t-full** at vtree level `t` if the disjunction of all t-nodes
//! equals the constant-true function. A diagram is **full** if t-full at every level.
//! `expand_full` materializes the fill nodes explicitly; `negate` complements
//! the full diagram at its root.

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd, TddLevel, TddNodeId};

use crate::Engine;
use crate::limits::OperationError;
use crate::limits::pool::Pool;
use std::sync::Arc;

/// The buffers negation reuses between calls.
///
/// Take-and-return, as everywhere else: the capacity is what is being reused,
/// so a warmed engine allocates nothing per level, and a negation over a deep
/// vtree does not pay a `Vec` for each of its levels.
#[derive(Default)]
pub(crate) struct NegateScratch {
    /// One level's `lefts x rights` cover, one bit per cell.
    cover: Pool<Vec<u64>>,
    /// The cells of a basis that no node covers: a level's fill pairs, then
    /// the root's complement pairs.
    cells: Pool<Vec<ChildPair>>,
    /// One node's pair list while its `One` references are being split.
    node_pairs: Pool<Vec<ChildPair>>,
    /// The level a `One` split is rebuilt into, swapped with the original.
    rebuild: Pool<TddLevel>,
}

impl NegateScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.cover.drain();
        self.cells.drain();
        self.node_pairs.drain();
        self.rebuild.drain();
    }
}

impl Engine {
    /// Run [`Tdd::negate`](crate::Tdd::negate) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn negate(&self, f: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().begin_operation();
        let mut result = negate_tdd_owned(self, f)?;
        self.reduce(&mut result, crate::reduce::ReductionPlan::default())?;
        Ok(result)
    }
}

/// Make `tdd` full and complement it at the root, consuming the operand;
/// [`Engine::negate`] also minimizes the result.
pub(crate) fn negate_tdd_owned(eng: &Engine, mut tdd: Tdd) -> Result<Tdd, OperationError> {
    tdd.require_structure()?;
    eng.limits().check_stop()?;
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
fn complement_full_at_root(eng: &Engine, full_tdd: Tdd, orig_vtree: &Arc<crate::vtree::Vtree>) -> Result<Tdd, OperationError> {
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
        // After `expand_full` (which expands One → Pos+Neg), child widths:
        // - Leaf children: 2 (the disjoint set {Pos, Neg})
        // - Internal children: stored width (includes any fill nodes)
        let (left, right) = vtree.children(root);
        let lefts = ChildBasis::of(vtree, &levels, left);
        let rights = ChildBasis::of(vtree, &levels, right);

        let scratch = eng.negate_scratch();
        let mut bits = scratch.cover.take();
        let mut neg_pairs = scratch.cells.take();
        let collected = collect_complement_pairs(
            eng, &levels[root_idx], out_local, lefts, rights, &mut bits, &mut neg_pairs,
        );
        scratch.cover.put_bounded(eng.limits(), bits);
        if let Err(e) = collected {
            scratch.cells.put_bounded(eng.limits(), neg_pairs);
            return Err(e);
        }

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
                let node = &levels[left.idx()].nodes[ChildDecoder::structural().node(pair.left).idx()];
                node.is_internal() && !levels[left.idx()].pairs_of(node).is_empty()
            };
            let right_alive = right_is_leaf || {
                let node = &levels[right.idx()].nodes[ChildDecoder::structural().node(pair.right).idx()];
                node.is_internal() && !levels[right.idx()].pairs_of(node).is_empty()
            };
            left_alive && right_alive
        });

        if neg_pairs.is_empty() {
            scratch.cells.put_bounded(eng.limits(), neg_pairs);
            return Ok(crate::build::constant_zero(eng, orig_vtree));
        }

        let pushed = levels[root_idx].push_node_on(eng, &neg_pairs);
        scratch.cells.put_bounded(eng.limits(), neg_pairs);
        let neg_idx = pushed?;

        Ok(Tdd::from_levels_unchecked(
            Arc::clone(orig_vtree),
            levels,
            TddNodeId { vtree: root, local: neg_idx },
        ))
    }
}

// ── `expand_full`: explicit fill-node materialization ────────────────────────────

/// Make a diagram full by materializing fill nodes explicitly at every
/// structural level; marginal levels are skipped.
///
/// One bottom-up pass does both halves of the job at each level: splitting the
/// `One` references a leaf child cannot express in the `{Pos, Neg}` basis, and
/// adding the node that covers whatever cells of `lefts x rights` are left
/// over. The same walk decides both, so the split runs only where a `One` is
/// there to split, and the fill is read off a bitmap of the cells the walk
/// marked rather than a second enumeration of the basis.
pub(crate) fn expand_full(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let scratch = eng.negate_scratch();
    let mut bits = scratch.cover.take();
    let mut cells = scratch.cells.take();
    let mut node_pairs = scratch.node_pairs.take();
    let mut rebuild = scratch.rebuild.take();

    let result = expand_full_with(eng, tdd, &mut bits, &mut cells, &mut node_pairs, &mut rebuild);

    let lim = eng.limits();
    scratch.cover.put_bounded(lim, bits);
    scratch.cells.put_bounded(lim, cells);
    scratch.node_pairs.put_bounded(lim, node_pairs);
    crate::diagram::reset_level(&mut rebuild);
    scratch.rebuild.put(rebuild);
    result
}

/// [`expand_full`] with the scratch checked out.
fn expand_full_with(
    eng: &Engine,
    tdd: &mut Tdd,
    bits: &mut Vec<u64>,
    cells: &mut Vec<ChildPair>,
    node_pairs: &mut Vec<ChildPair>,
    rebuild: &mut TddLevel,
) -> Result<(), OperationError> {
    let vtree = tdd.vtree.clone();
    for (t, left, right) in vtree.internal_bottomup() {
        // A marginal level has no node list to make full.
        if tdd.levels[t.idx()].is_marginal() {
            continue;
        }
        // Read bottom-up: a child's basis includes the fill node its own visit
        // added.
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        let left_leaf = vtree.node(left).is_leaf();
        let right_leaf = vtree.node(right).is_leaf();
        let level = &mut tdd.levels[t.idx()];

        let mut cover = Cover::reset(eng, bits, lefts, rights)?;
        let mut split = false;
        let mut poll = eng.limits().gate();
        for node in &level.nodes {
            if !node.is_internal() {
                continue;
            }
            for pair in level.pairs_of(node) {
                let (l, r) = (pair.left.0, pair.right.0);
                // With implicit leaves, One (index 0) overlaps Pos (1) and
                // Neg (2), so a leaf side's One stands for both basis cells.
                let l_one = left_leaf && l == ONE_LEAF_IDX.0;
                let r_one = right_leaf && r == ONE_LEAF_IDX.0;
                split |= l_one | r_one;
                let (la, lb) = if l_one { (POS_LEAF_IDX.0, NEG_LEAF_IDX.0) } else { (l, l) };
                let (ra, rb) = if r_one { (POS_LEAF_IDX.0, NEG_LEAF_IDX.0) } else { (r, r) };
                cover.mark(la, ra);
                if r_one {
                    cover.mark(la, rb);
                }
                if l_one {
                    cover.mark(lb, ra);
                    if r_one {
                        cover.mark(lb, rb);
                    }
                }
                poll.poll(1)?;
            }
        }
        poll.flush()?;

        // The split rewrites pair lists; the cover already accounts for it.
        if split {
            split_ones_in_level(eng, level, rebuild, node_pairs, left_leaf, right_leaf)?;
        }
        // Covered every basis cell => already full; nothing to add.
        if !cover.is_full() {
            cells.clear();
            cover.missing_into(eng, cells)?;
            if !cells.is_empty() {
                level.push_node_on(eng, cells)?;
            }
        }
        eng.limits().check_stop()?;
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

/// Split the `One` references of one level's pairs into `Pos` and `Neg`.
///
/// After the split the level's leaf references are all in the `{Pos, Neg}`
/// basis, which is disjoint, as the cover of `expand_full` and the complement
/// at the root both need. A node's pair list grows, so the level is rebuilt
/// into `rebuild` and swapped in; `rebuild` keeps its arenas for the next
/// level.
fn split_ones_in_level(
    eng: &Engine,
    level: &mut TddLevel,
    rebuild: &mut TddLevel,
    node_pairs: &mut Vec<ChildPair>,
    left_leaf: bool,
    right_leaf: bool,
) -> Result<(), OperationError> {
    rebuild.clear();
    let lim = eng.limits();
    let mut poll = lim.gate();
    for node in &level.nodes {
        if !node.is_internal() {
            lim.try_push(&mut rebuild.nodes, *node)?;
            continue;
        }
        node_pairs.clear();
        for pair in level.pairs_of(node) {
            let lefts: &[EncodedChildRef] = if left_leaf && pair.left == ONE_LEAF_IDX.into() {
                &[POS_LEAF_IDX.into(), NEG_LEAF_IDX.into()]
            } else {
                std::slice::from_ref(&pair.left)
            };
            let rights: &[EncodedChildRef] = if right_leaf && pair.right == ONE_LEAF_IDX.into() {
                &[POS_LEAF_IDX.into(), NEG_LEAF_IDX.into()]
            } else {
                std::slice::from_ref(&pair.right)
            };
            for &left in lefts {
                for &right in rights {
                    lim.try_push(node_pairs, ChildPair { left, right })?;
                }
            }
            poll.poll(1)?;
        }
        node_pairs.sort_unstable();
        node_pairs.dedup();
        rebuild.push_node_on(eng, node_pairs)?;
    }
    rebuild.n_tombstones = level.n_tombstones;
    std::mem::swap(level, rebuild);
    poll.flush()
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
            ChildBasis { start: 0, end: levels[child.idx()].slot_count() as u32 }
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
}

/// A bitmap over one level's `lefts x rights` basis: cell `(l, r)` is bit
/// `(l - lefts.start) * rights.len() + (r - rights.start)`.
///
/// The basis is what a fill node or a complement enumerates, and it is
/// quadratic in the child widths, so the cover is kept at a bit per cell
/// rather than an entry per cell. It also answers "already full" without a
/// second pass: `covered` counts distinct cells as they are marked.
struct Cover<'a> {
    bits: &'a mut Vec<u64>,
    lefts: ChildBasis,
    rights: ChildBasis,
    /// `rights.len()`, the row stride.
    stride: usize,
    /// `lefts.len() * rights.len()`, the cell count.
    len: usize,
    /// Distinct cells marked so far.
    covered: usize,
}

impl<'a> Cover<'a> {
    /// Clear `bits` and size it for the `lefts x rights` basis.
    fn reset(
        eng: &Engine,
        bits: &'a mut Vec<u64>,
        lefts: ChildBasis,
        rights: ChildBasis,
    ) -> Result<Cover<'a>, OperationError> {
        let len = lefts.len().checked_mul(rights.len()).ok_or(OperationError::OverBudget)?;
        bits.clear();
        eng.limits().try_resize(bits, len.div_ceil(64), 0u64)?;
        Ok(Cover { bits, lefts, rights, stride: rights.len(), len, covered: 0 })
    }

    /// Mark cell `(l, r)`; a cell outside the basis is not one to cover.
    #[inline]
    fn mark(&mut self, l: u32, r: u32) {
        if !self.lefts.contains(l) || !self.rights.contains(r) {
            return;
        }
        let cell = (l - self.lefts.start) as usize * self.stride + (r - self.rights.start) as usize;
        let word = &mut self.bits[cell >> 6];
        let bit = 1u64 << (cell & 63);
        if *word & bit == 0 {
            *word |= bit;
            self.covered += 1;
        }
    }

    /// Whether every basis cell is covered, so the level is already full.
    #[inline]
    fn is_full(&self) -> bool {
        self.covered == self.len
    }

    /// Append the unmarked cells to `out`, in ascending `(l, r)` order.
    fn missing_into(&self, eng: &Engine, out: &mut Vec<ChildPair>) -> Result<(), OperationError> {
        let lim = eng.limits();
        let mut poll = lim.gate();
        for (w, &word) in self.bits.iter().enumerate() {
            let base = w * 64;
            let mut missing = !word;
            // The last word runs past the basis; those bits are not cells.
            let width = (self.len - base).min(64);
            if width < 64 {
                missing &= (1u64 << width) - 1;
            }
            while missing != 0 {
                let cell = base + missing.trailing_zeros() as usize;
                missing &= missing - 1;
                let l = self.lefts.start + (cell / self.stride) as u32;
                let r = self.rights.start + (cell % self.stride) as u32;
                lim.try_push(
                    out,
                    ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)),
                )?;
            }
            poll.poll(64)?;
        }
        poll.flush()
    }
}

/// Collect into `out` the cells of `lefts x rights` that are not pairs of
/// `exclude_node`.
fn collect_complement_pairs(
    eng: &Engine,
    level: &TddLevel,
    exclude_node: NodeIdx,
    lefts: ChildBasis,
    rights: ChildBasis,
    bits: &mut Vec<u64>,
    out: &mut Vec<ChildPair>,
) -> Result<(), OperationError> {
    let mut cover = Cover::reset(eng, bits, lefts, rights)?;
    // The root level has been through `expand_full`, so its leaf references
    // are already in the `{Pos, Neg}` basis and a pair is one cell.
    for pair in level.pairs_of_idx(exclude_node.idx()) {
        cover.mark(pair.left.0, pair.right.0);
    }
    out.clear();
    cover.missing_into(eng, out)
}

#[cfg(test)]
#[path = "tests/negate/mod.rs"]
mod tests;
