//! Negation: make-full transformation and complement operation.
//!
//! A diagram is **t-full** at vtree level `t` if the disjunction of all t-nodes
//! equals the constant-true function. A diagram is **full** if t-full at every level.
//! `expand_full` materializes the fill nodes explicitly; `negate` complements
//! the full diagram at its root.

use crate::diagram::{Assembly, ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd, TddLevel, TddNodeId};

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
}

impl NegateScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.cover.drain();
        self.cells.drain();
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
        self.negate_with(f, crate::reduce::ReductionPlan::default())
    }

    /// [`negate`](Self::negate) with the reduction it ends with chosen
    /// explicitly. See [`Tdd::negate_with`] for what the plans cost and what
    /// they leave behind.
    ///
    /// # Errors
    ///
    /// As [`negate`](Self::negate).
    pub fn negate_with(
        &self,
        f: Tdd,
        plan: crate::reduce::ReductionPlan<'_>,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().begin_operation();
        let mut result = negate_tdd_owned(self, f)?;
        // `expand_full` left every level covering its children's whole basis,
        // so every node is named from the level above and the only nodes the
        // complement can have orphaned are below the root cells it dropped.
        self.reduce_scoped(&mut result, plan, crate::reduce::prune::PruneScope::BelowRoot)?;
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
        let root_form = expand_full(eng, &mut tdd)?;
        complement_full_at_root(eng, tdd, &vtree, root_form)?
    };
    result.weights = weights;
    Ok(result)
}

/// Complement a full diagram at its root: collect the root-level pairs not in
/// the output node and drop the dead ones. `orig_vtree` is the operand's
/// vtree, used for the constant fallbacks.
fn complement_full_at_root(
    eng: &Engine,
    full_tdd: Tdd,
    orig_vtree: &Arc<crate::vtree::Vtree>,
    root_form: LeafForm,
) -> Result<Tdd, OperationError> {
    let vtree = &full_tdd.vtree;
    let root = vtree.root();
    let root_idx = root.idx();
    let out_local = full_tdd.output.local;

    let mut assembly = Assembly::from_levels(eng, Arc::clone(orig_vtree), full_tdd.levels.into_vec(), None);
    let (levels, _) = assembly.parts_mut();

    if vtree.node(root).is_leaf() {
        // Implicit leaf: the output index and the label are the same number. It stays in
        // {Pos,Neg,One} unless the output is One (complement = Zero, returned as the zero
        // constant diagram).
        let Some(neg_local) = complement_leaf_root(out_local) else {
            return Ok(crate::build::constant_zero(eng, orig_vtree));
        };
        Ok(assembly.finish_untracked(TddNodeId { vtree: root, local: neg_local }))
    } else {
        // Child widths the complement's basis spans:
        // - Leaf children: 2 (the disjoint set {Pos, Neg})
        // - Internal children: stored width (includes any fill node)
        let (left, right) = vtree.children(root);
        let basis = Basis {
            lefts: ChildBasis::of(vtree, levels, left),
            rights: ChildBasis::of(vtree, levels, right),
            form: root_form,
        };

        let scratch = eng.negate_scratch();
        let mut bits = scratch.cover.checkout_preserving(eng.limits());
        let mut neg_pairs = scratch.cells.checkout_preserving(eng.limits());
        collect_complement_pairs(
            eng, &levels[root_idx], out_local, basis, &mut bits, &mut neg_pairs,
        )?;

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
            return Ok(crate::build::constant_zero(eng, orig_vtree));
        }

        let neg_idx = levels[root_idx].push_node_on(eng, &neg_pairs)?;

        Ok(assembly.finish_untracked(TddNodeId { vtree: root, local: neg_idx }))
    }
}

// ── `expand_full`: explicit fill-node materialization ────────────────────────────

/// Make a diagram full by materializing fill nodes explicitly at every
/// structural level; marginal levels are skipped. Returns the leaf form of the
/// root's level, which the complement above it emits in.
///
/// One bottom-up pass per level marks the cells of `lefts x rights` that the
/// level's nodes cover, in a bitmap held in engine scratch, and the fill node
/// is read off the zero bits. The walk also records, per leaf child, whether
/// the level refers to it as `One` or as `Pos`/`Neg`, and the fill is emitted
/// in that same form.
pub(crate) fn expand_full(eng: &Engine, tdd: &mut Tdd) -> Result<LeafForm, OperationError> {
    let scratch = eng.negate_scratch();
    let mut bits = scratch.cover.checkout_preserving(eng.limits());
    let mut cells = scratch.cells.checkout_preserving(eng.limits());

    expand_full_with(eng, tdd, &mut bits, &mut cells)
}

/// [`expand_full`] with the scratch checked out.
fn expand_full_with(
    eng: &Engine,
    tdd: &mut Tdd,
    bits: &mut Vec<u64>,
    cells: &mut Vec<ChildPair>,
) -> Result<LeafForm, OperationError> {
    let vtree = tdd.vtree.clone();
    let root_idx = vtree.root().idx();
    let mut root_form = LeafForm::LITERAL;
    for (t, left, right) in vtree.internal_bottomup() {
        // A marginal level has no node list to make full.
        if tdd.levels[t.idx()].is_marginal() {
            continue;
        }
        // Read bottom-up: a child's basis includes the fill node its own visit
        // added.
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        let leaf = LeafForm {
            left_one: vtree.node(left).is_leaf(),
            right_one: vtree.node(right).is_leaf(),
        };
        let level = &mut tdd.levels[t.idx()];

        let mut cover = Cover::reset(eng, bits, lefts, rights, leaf)?;
        let mut poll = eng.limits().gate();
        for node in &level.nodes {
            if !node.is_internal() {
                continue;
            }
            for pair in level.pairs_of(node) {
                cover.mark_pair(*pair);
                poll.poll(1)?;
            }
        }
        poll.flush()?;
        // The walk has seen every reference, so this is the level's form.
        let form = cover.one_seen;
        if t.idx() == root_idx {
            root_form = form;
        }

        // Covered every basis cell => already full; nothing to add.
        if !cover.is_full() {
            cells.clear();
            cover.missing_into(eng, form, cells)?;
            if !cells.is_empty() {
                level.push_node_on(eng, cells)?;
            }
        }
        eng.limits().check_stop()?;
    }
    Ok(root_form)
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

// The leaf basis is a contiguous range only because Pos and Neg are adjacent.
const _: () = assert!(NEG_LEAF_IDX.0 == POS_LEAF_IDX.0 + 1);

/// The other half of a leaf's `{Pos, Neg}` couple.
#[inline]
fn other_polarity(idx: u32) -> u32 {
    POS_LEAF_IDX.0 + NEG_LEAF_IDX.0 - idx
}

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

/// How one level refers to a leaf child: as `One`, or as `Pos`/`Neg`.
///
/// Determinism keeps the two apart — the labels a level uses at one leaf are a
/// subset of `{Pos, Neg}` or of `{One}`, never both
/// (`test_helpers::check::check_determinism`) — so this is a property of the
/// level, read off the first reference the cover walk sees. A fill node and the
/// complement at the root are emitted in the same form, which is what keeps the
/// level's labels on one side of that line.
#[derive(Copy, Clone, Default)]
pub(crate) struct LeafForm {
    /// The left child is a leaf the level refers to as `One`.
    left_one: bool,
    /// The right child is a leaf the level refers to as `One`.
    right_one: bool,
}

impl LeafForm {
    /// Neither side is in `One` form: what a level with no leaf-side `One`
    /// reference uses, and what an empty level defaults to.
    const LITERAL: LeafForm = LeafForm { left_one: false, right_one: false };
}

/// One level's cross-product basis together with the leaf form it names its
/// leaf children in: what the complement at the root needs to read and write.
#[derive(Copy, Clone)]
struct Basis {
    lefts: ChildBasis,
    rights: ChildBasis,
    form: LeafForm,
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
    /// Which sides are leaf children, so index 0 there reads as `One`.
    leaf: LeafForm,
    /// Which of those the walk has actually seen named as `One` — the level's
    /// leaf form, which the fill is emitted in.
    one_seen: LeafForm,
    /// `rights.len()`, the row stride.
    stride: usize,
    /// `lefts.len() * rights.len()`, the cell count.
    len: usize,
    /// Distinct cells marked so far.
    covered: usize,
}

impl<'a> Cover<'a> {
    /// Clear `bits` and size it for the `lefts x rights` basis. `leaf` says
    /// which sides are leaf children.
    fn reset(
        eng: &Engine,
        bits: &'a mut Vec<u64>,
        lefts: ChildBasis,
        rights: ChildBasis,
        leaf: LeafForm,
    ) -> Result<Cover<'a>, OperationError> {
        let len = lefts.len().checked_mul(rights.len()).ok_or(OperationError::OverBudget)?;
        bits.clear();
        eng.limits().try_resize(bits, len.div_ceil(64), 0u64)?;
        Ok(Cover {
            bits,
            lefts,
            rights,
            leaf,
            one_seen: LeafForm::LITERAL,
            stride: rights.len(),
            len,
            covered: 0,
        })
    }

    /// Mark the cells `pair` covers, a leaf side's `One` standing for both
    /// cells of that leaf's `{Pos, Neg}` couple.
    #[inline]
    fn mark_pair(&mut self, pair: ChildPair) {
        let (l, r) = (pair.left.0, pair.right.0);
        let l_one = self.leaf.left_one && l == ONE_LEAF_IDX.0;
        let r_one = self.leaf.right_one && r == ONE_LEAF_IDX.0;
        self.one_seen.left_one |= l_one;
        self.one_seen.right_one |= r_one;
        let (la, lb) = if l_one { (POS_LEAF_IDX.0, NEG_LEAF_IDX.0) } else { (l, l) };
        let (ra, rb) = if r_one { (POS_LEAF_IDX.0, NEG_LEAF_IDX.0) } else { (r, r) };
        self.mark(la, ra);
        if r_one {
            self.mark(la, rb);
        }
        if l_one {
            self.mark(lb, ra);
            if r_one {
                self.mark(lb, rb);
            }
        }
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

    /// Whether cell `(l, r)` is unmarked. Out-of-basis indices are not cells
    /// and answer false.
    #[inline]
    fn is_missing(&self, l: u32, r: u32) -> bool {
        if !self.lefts.contains(l) || !self.rights.contains(r) {
            return false;
        }
        let cell = (l - self.lefts.start) as usize * self.stride + (r - self.rights.start) as usize;
        self.bits[cell >> 6] & (1u64 << (cell & 63)) == 0
    }

    /// Append the unmarked cells to `out`, in ascending `(l, r)` order, writing
    /// a side the level refers to as `One` back in that form.
    ///
    /// A `One` side's covered cells come in `{Pos, Neg}` couples — the walk
    /// marks both for each reference — so the unmarked cells do too, and the
    /// couple is emitted once, on its `Pos` row, as a `One`. Keeping the form
    /// is what lets the fill join a level whose other labels are `One` without
    /// splitting them: mixing the two at one leaf would break determinism's
    /// label rule and, downstream, block leaf-twin contraction, which is
    /// all-or-nothing per level.
    fn missing_into(
        &self,
        eng: &Engine,
        form: LeafForm,
        out: &mut Vec<ChildPair>,
    ) -> Result<(), OperationError> {
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
                let couple_l = form.left_one && self.is_missing(other_polarity(l), r);
                let couple_r = form.right_one && self.is_missing(l, other_polarity(r));
                debug_assert!(
                    !form.left_one || couple_l,
                    "negate: a One-form leaf's unmarked cells are not couples"
                );
                debug_assert!(
                    !form.right_one || couple_r,
                    "negate: a One-form leaf's unmarked cells are not couples"
                );
                // Emit a couple once, from its Pos row.
                if (couple_l && l == NEG_LEAF_IDX.0) || (couple_r && r == NEG_LEAF_IDX.0) {
                    continue;
                }
                let l = if couple_l { ONE_LEAF_IDX.0 } else { l };
                let r = if couple_r { ONE_LEAF_IDX.0 } else { r };
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
/// `exclude_node`, in the root level's leaf form.
fn collect_complement_pairs(
    eng: &Engine,
    level: &TddLevel,
    exclude_node: NodeIdx,
    basis: Basis,
    bits: &mut Vec<u64>,
    out: &mut Vec<ChildPair>,
) -> Result<(), OperationError> {
    let mut cover = Cover::reset(eng, bits, basis.lefts, basis.rights, basis.form)?;
    for pair in level.pairs_of_idx(exclude_node.idx()) {
        cover.mark_pair(*pair);
    }
    out.clear();
    cover.missing_into(eng, basis.form, out)
}

#[cfg(test)]
#[path = "tests/negate/mod.rs"]
mod tests;
