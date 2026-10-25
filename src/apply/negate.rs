//! Negation: make-full transformation and complement operation.
//!
//! A diagram is **t-full** at vtree level `t` if the disjunction of all t-nodes
//! equals the constant-true function. `expand_full` materializes the fill nodes
//! that make every level below the root t-full; `negate` then complements the
//! output at the root, whose cells the root's children now span.

use crate::diagram::{Assembly, ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd, TddLevel, TddNodeId};

use crate::Engine;
use crate::limits::OperationError;
use crate::execution::pool::{Drain, Pool, Pools};
use crate::apply::falsity::empty_node;
use crate::reduce::prune::Completion;
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

impl Pools for NegateScratch {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        visit(&self.cover);
        visit(&self.cells);
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
        let _op = self.limits().enter()?;
        let canonical = f.levels.is_canonical(f.output);
        let (mut result, done) = complement_on(self, f)?;
        // `expand_full` left every level below the root covering its
        // children's whole basis, and the old output and its complement cover
        // the root's, so every node is named from the level above and the only
        // nodes the complement can have orphaned are below the root cells it
        // dropped.
        // The complement of a canonical operand has no twins to contract.
        let full = matches!(plan, crate::reduce::ReductionPlan::Full(_));
        if canonical && (full || matches!(plan, crate::reduce::ReductionPlan::Prune)) {
            self.certify_complement(&mut result, full, done.as_ref())?;
        } else {
            self.reduce_scoped(&mut result, plan, crate::reduce::prune::PruneScope::BelowRoot)?;
        }
        Ok(result)
    }
}

/// Make `tdd` full and complement it at the root, consuming the operand;
/// [`Engine::negate`] also minimizes the result.
pub(crate) fn negate_on(eng: &Engine, tdd: Tdd) -> Result<Tdd, OperationError> {
    complement_on(eng, tdd).map(|(result, _)| result)
}

/// [`negate_on`], with what the completion added where the complement was
/// built beside the old output.
pub(crate) fn complement_on(eng: &Engine, mut tdd: Tdd) -> Result<(Tdd, Option<Completion>), OperationError> {
    tdd.require_structure()?;
    eng.limits().check_stop()?;
    let weights = tdd.weights.take();
    let (mut result, done) = if tdd.is_zero() {
        (crate::build::constant_on(eng, &tdd.vtree, true)?, None)
    } else {
        let mut filled = Vec::new();
        eng.limits().try_resize(&mut filled, tdd.vtree.num_nodes(), false)?;
        expand_full(eng, &mut tdd, &mut filled)?;
        let (result, old) = complement_full_at_root(eng, tdd, &filled)?;
        (result, old.map(|old| Completion { old, filled }))
    };
    result.weights = weights;
    Ok((result, done))
}

/// Complement at the root a diagram full below it: collect the root-level
/// pairs not in the output node and drop the dead ones. With the old
/// output's slot when the complement is a node beside it.
fn complement_full_at_root(
    eng: &Engine,
    full_tdd: Tdd,
    filled: &[bool],
) -> Result<(Tdd, Option<NodeIdx>), OperationError> {
    let vtree = Arc::clone(&full_tdd.vtree);
    let root = vtree.root();
    let root_idx = root.idx();
    let out_local = full_tdd.output.local;

    let mut assembly = Assembly::from_levels(eng, Arc::clone(&vtree), full_tdd.levels.into_vec(), None);
    let (levels, _) = assembly.parts_mut();

    if vtree.node(root).is_leaf() {
        // Implicit leaf: the output index and the label are the same number. It stays in
        // {Pos,Neg,One} unless the output is One (complement = Zero, returned as the zero
        // constant diagram).
        let Some(neg_local) = complement_leaf_root(out_local) else {
            return Ok((crate::build::constant_on(eng, &vtree, false)?, None));
        };
        Ok((assembly.finish(TddNodeId { vtree: root, local: neg_local })?, None))
    } else {
        // Child widths the complement's basis spans:
        // - Leaf children: 2 (the disjoint set {Pos, Neg})
        // - Internal children: stored width (includes any fill node)
        let (left, right) = vtree.children(root);
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();

        let scratch = &eng.scratch.negate;
        let mut bits = scratch.cover.checkout_preserving(eng);
        let mut neg_pairs = scratch.cells.checkout_preserving(eng);
        let lefts = ChildBasis::of(&vtree, levels, left);
        let rights = ChildBasis::of(&vtree, levels, right);
        let leaf = LeafForm { left_one: left_is_leaf, right_one: right_is_leaf };
        let fill = [filled[left.idx()].then(|| lefts.end - 1), filled[right.idx()].then(|| rights.end - 1)];
        // The other root nodes are dropped with the old output, so the
        // complement takes the leaf form the output uses.
        let mut form = LeafForm::LITERAL;
        let covered = areas(levels[root_idx].pairs_iter_of_idx(out_local.idx()), leaf, &mut form);
        neg_pairs.clear();
        if covered == old_cells(lefts, rights, fill) {
            // The output covers every cell of the children's old nodes: its
            // complement is the lines of their added nodes.
            fill_lines_into(eng, lefts, rights, fill, form, &mut neg_pairs)?;
            #[cfg(debug_assertions)]
            {
                let mut cover = Cover::reset(eng, &mut bits, lefts, rights, leaf)?;
                cover.mark_pairs(levels[root_idx].pairs_iter_of_idx(out_local.idx()));
                let mut by_cover = Vec::new();
                cover.missing_into(eng, form, &mut by_cover)?;
                debug_assert_eq!(*neg_pairs, by_cover, "negate: the lines of the added nodes are not the root cells the output misses");
            }
        } else {
            let mut cover = Cover::reset(eng, &mut bits, lefts, rights, leaf)?;
            cover.mark_pairs(levels[root_idx].pairs_iter_of_idx(out_local.idx()));
            cover.missing_into(eng, form, &mut neg_pairs)?;
        }

        // Drop the pairs whose child computes the Zero function (an internal
        // node with no pairs), which a fill node of `expand_full` can be.
        neg_pairs.retain(|pair| {
            let left_alive = left_is_leaf
                || !empty_node(&levels[left.idx()], ChildDecoder::structural().node(pair.left).idx());
            let right_alive = right_is_leaf
                || !empty_node(&levels[right.idx()], ChildDecoder::structural().node(pair.right).idx());
            left_alive && right_alive
        });

        if neg_pairs.is_empty() {
            return Ok((crate::build::constant_on(eng, &vtree, false)?, None));
        }

        let neg_idx = levels[root_idx].push_node(eng.limits(), &neg_pairs)?;

        Ok((assembly.finish(TddNodeId { vtree: root, local: neg_idx })?, Some(out_local)))
    }
}

// ── `expand_full`: explicit fill-node materialization ────────────────────────────

/// Make every level of a structural diagram below the root full by
/// materializing fill nodes explicitly. The root is left alone: a fill there
/// would be the complement of its nodes, which the negation computes from the
/// output node alone.
///
/// One bottom-up pass per level marks the cells of `lefts x rights` that the
/// level's nodes cover, in a bitmap held in engine scratch, and the fill node
/// is read off the zero bits. The walk also records, per leaf child, whether
/// the level refers to it as `One` or as `Pos`/`Neg`, and the fill is emitted
/// in that same form. `filled[t]` is set for each level `t` it adds a node
/// to, as the level's last.
pub(crate) fn expand_full(eng: &Engine, tdd: &mut Tdd, filled: &mut [bool]) -> Result<(), OperationError> {
    let scratch = &eng.scratch.negate;
    let mut bits = scratch.cover.checkout_preserving(eng);
    let mut cells = scratch.cells.checkout_preserving(eng);

    expand_full_with(eng, tdd, &mut bits, &mut cells, filled)
}

/// [`expand_full`] with the scratch checked out.
fn expand_full_with(
    eng: &Engine,
    tdd: &mut Tdd,
    bits: &mut Vec<u64>,
    cells: &mut Vec<ChildPair>,
    filled: &mut [bool],
) -> Result<(), OperationError> {
    let vtree = tdd.vtree.clone();
    let root = vtree.root();
    // Read before the first change to a level, which drops both.
    let (canonical, closed) = (tdd.levels.is_canonical(tdd.output), tdd.levels.is_closed());
    for (t, left, right) in vtree.internal_bottomup().filter(|&(t, _, _)| t != root) {
        // Read bottom-up: a child's basis includes the fill node its own visit
        // added.
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        let leaf = LeafForm {
            left_one: vtree.node(left).is_leaf(),
            right_one: vtree.node(right).is_leaf(),
        };
        // A child the completion added a node to has it last.
        let fill = [filled[left.idx()].then(|| lefts.end - 1), filled[right.idx()].then(|| rights.end - 1)];
        let level = &mut tdd.levels[t.idx()];

        // On a canonical diagram every stored node is reachable and
        // deterministic, so the pairs of a level cover disjoint cells, and
        // their count says whether they cover every cell, the level being
        // full already, or every cell of the children's old nodes, the
        // missing cells then being the lines of the children's added nodes,
        // read off without a cover. An operand with unreachable nodes may
        // hold two that cover one cell, which only the cover sees.
        if canonical {
            let (covered, form) = covered_cells(level, leaf, closed);
            #[cfg(debug_assertions)]
            {
                let (by_pairs, form_by_pairs) = covered_cells_by_pairs(level, leaf);
                debug_assert_eq!(covered, by_pairs, "negate: the covered cells are not the pairs' areas");
                debug_assert!(
                    (form.left_one, form.right_one) == (form_by_pairs.left_one, form_by_pairs.right_one),
                    "negate: the level's first pair does not name its leaves in the level's form",
                );
            }
            if covered == (lefts.len() * rights.len()) as u64 {
                #[cfg(debug_assertions)]
                debug_assert!(missing_by_cover(eng, bits, level, lefts, rights, leaf)?.is_empty(), "negate: a level whose pairs fill its basis misses a cell");
                eng.limits().check_stop()?;
                continue;
            }
            if covered == old_cells(lefts, rights, fill) {
                if fill != [None, None] {
                    cells.clear();
                    fill_lines_into(eng, lefts, rights, fill, form, cells)?;
                    #[cfg(debug_assertions)]
                    debug_assert_eq!(*cells, missing_by_cover(eng, bits, level, lefts, rights, leaf)?, "negate: the lines of the added nodes are not the cells no pair covers");
                    level.push_node(eng.limits(), cells)?;
                    filled[t.idx()] = true;
                }
                eng.limits().check_stop()?;
                continue;
            }
        }

        // Mark the cells the pairs cover and read the fill off the rest, the
        // walk compiled for the level's leaf sides.
        let mut cover = Cover::reset(eng, bits, lefts, rights, leaf)?;
        match (leaf.left_one, leaf.right_one) {
            (false, false) => cover.mark_level::<false, false>(eng, level)?,
            (false, true) => cover.mark_level::<false, true>(eng, level)?,
            (true, false) => cover.mark_level::<true, false>(eng, level)?,
            (true, true) => cover.mark_level::<true, true>(eng, level)?,
        }
        // The walk has seen every reference, so this is the level's form.
        let form = cover.one_seen;
        cells.clear();
        cover.missing_into(eng, form, cells)?;
        debug_assert!(!canonical || !cells.is_empty(), "negate: a level short of its cells has none missing");
        if !cells.is_empty() {
            level.push_node(eng.limits(), cells)?;
            filled[t.idx()] = true;
        }
        eng.limits().check_stop()?;
    }
    Ok(())
}

/// The cells of `lefts x rights` the pairs of `level` cover, with the leaf
/// form the level uses, without reading the pairs. The pairs of a level
/// cover disjoint cells, determinism's rule, one cell a pair and two for
/// each side that is a leaf named `One`; and a level names a leaf child as
/// `One` everywhere or nowhere ([`LeafForm`]). So the count is the level's
/// pairs, doubled for each side its first pair names `One`. The pairs are
/// counted off the level's nodes, or read off the count a level of a
/// `closed` diagram keeps.
fn covered_cells(level: &TddLevel, leaf: LeafForm, closed: bool) -> (u64, LeafForm) {
    let pairs = if closed { level.live_pairs_closed() } else { level.live_pairs() } as u64;
    let mut form = LeafForm::LITERAL;
    if (leaf.left_one || leaf.right_one)
        && let Some(pair) = (0..level.node_count()).find_map(|i| level.pairs_iter_of_idx(i).next())
    {
        areas(std::iter::once(pair), leaf, &mut form);
    }
    (pairs << (u32::from(form.left_one) + u32::from(form.right_one)), form)
}

/// [`covered_cells`] off every pair's area, for a debug build to check it
/// against.
#[cfg(debug_assertions)]
fn covered_cells_by_pairs(level: &TddLevel, leaf: LeafForm) -> (u64, LeafForm) {
    let mut form = LeafForm::LITERAL;
    let covered = (0..level.node_count()).map(|i| areas(level.pairs_iter_of_idx(i), leaf, &mut form)).sum();
    (covered, form)
}

/// The cells `pairs` cover, one a pair and two for each side that is a
/// leaf named `One`, with the sides so named added to `form`.
#[inline(always)]
fn areas(pairs: impl Iterator<Item = ChildPair>, leaf: LeafForm, form: &mut LeafForm) -> u64 {
    let one = ONE_LEAF_IDX.0;
    let mut covered = 0u64;
    for pair in pairs {
        let l_one = leaf.left_one && pair.left.0 == one;
        let r_one = leaf.right_one && pair.right.0 == one;
        form.left_one |= l_one;
        form.right_one |= r_one;
        covered += 1u64 << (u32::from(l_one) + u32::from(r_one));
    }
    covered
}

/// The cells of `lefts x rights` whose sides are both old nodes: not the
/// node `fill` names on either side.
fn old_cells(lefts: ChildBasis, rights: ChildBasis, fill: [Option<u32>; 2]) -> u64 {
    let old = |b: ChildBasis, f: Option<u32>| (b.len() - usize::from(f.is_some())) as u64;
    old(lefts, fill[0]) * old(rights, fill[1])
}

/// The cells of `lefts x rights` no pair of `level` covers, by a cover, as
/// [`expand_full`] reads them where some cell of two old nodes is missing.
#[cfg(debug_assertions)]
fn missing_by_cover(
    eng: &Engine,
    bits: &mut Vec<u64>,
    level: &TddLevel,
    lefts: ChildBasis,
    rights: ChildBasis,
    leaf: LeafForm,
) -> Result<Vec<ChildPair>, OperationError> {
    let mut cover = Cover::reset(eng, bits, lefts, rights, leaf)?;
    cover.mark_pairs((0..level.node_count()).flat_map(|i| level.pairs_iter_of_idx(i)));
    let mut out = Vec::new();
    cover.missing_into(eng, cover.one_seen, &mut out)?;
    Ok(out)
}

/// Append to `out` the cells of `lefts x rights` that name a child's added
/// node, `fill_l` on the left and `fill_r` on the right, in ascending
/// `(l, r)` order and in the leaf form `form`, as
/// [`Cover::missing_into`] writes them: the missing cells of a level whose
/// pairs cover every other cell.
fn fill_lines_into(
    eng: &Engine,
    lefts: ChildBasis,
    rights: ChildBasis,
    fill: [Option<u32>; 2],
    form: LeafForm,
    out: &mut Vec<ChildPair>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let missing = |l: u32, r: u32| Some(l) == fill[0] || Some(r) == fill[1];
    let mut emit = |l: u32, r: u32| -> Result<(), OperationError> {
        let couple_l = form.left_one && missing(other_polarity(l), r);
        let couple_r = form.right_one && missing(l, other_polarity(r));
        if (couple_l && l == NEG_LEAF_IDX.0) || (couple_r && r == NEG_LEAF_IDX.0) {
            return Ok(());
        }
        let l = if couple_l { ONE_LEAF_IDX.0 } else { l };
        let r = if couple_r { ONE_LEAF_IDX.0 } else { r };
        lim.try_push(out, ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)))
    };
    for l in lefts.start..lefts.end {
        if Some(l) == fill[0] {
            for r in rights.start..rights.end {
                emit(l, r)?;
            }
        } else if let Some(r) = fill[1] {
            emit(l, r)?;
        }
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
/// level, read off its first pair ([`covered_cells`]) or the cover's walk. A
/// fill node and the complement at the root are emitted in the same form,
/// which is what keeps the level's labels on one side of that line.
#[derive(Copy, Clone, Default)]
struct LeafForm {
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

/// A bitmap over one level's `lefts x rights` basis: cell `(l, r)` is bit
/// `(l - lefts.start) * rights.len() + (r - rights.start)`.
///
/// The basis is what a fill node or a complement enumerates, and it is
/// quadratic in the child widths, so the cover is kept at a bit per cell
/// rather than an entry per cell.
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
        let len = lefts.len().checked_mul(rights.len()).ok_or(OperationError::IndexOverflow)?;
        bits.clear();
        // And the cell past the basis, which a walk marks for a node it
        // reads no pair of ([`mark_level`](Self::mark_level)).
        eng.limits().try_resize(bits, (len + 1).div_ceil(64), 0u64)?;
        Ok(Cover {
            bits,
            lefts,
            rights,
            leaf,
            one_seen: LeafForm::LITERAL,
            stride: rights.len(),
            len,
        })
    }

    /// Mark the cells of every pair of `level`'s nodes, charging the work
    /// clock a unit a pair.
    ///
    /// A stored level whose nodes' ranges cover its arena is read node by
    /// node and then the arena whole, with no branch on a node's kind
    /// ([`StoredPairs::for_each_pair_whole`]): a node whose pairs are in the
    /// arena marks the cell past the basis, which the cover keeps for it and
    /// [`missing_into`](Self::missing_into) never reads, and the pairs are
    /// charged after the walk as a poll each would charge them. Otherwise a
    /// stored level's nodes are read as slices of its arena, and an implicit
    /// level's generated into a buffer a node at a time.
    ///
    /// [`StoredPairs::for_each_pair_whole`]: crate::diagram::StoredPairs::for_each_pair_whole
    fn mark_level<const LEFT_LEAF: bool, const RIGHT_LEAF: bool>(
        &mut self,
        eng: &Engine,
        level: &TddLevel,
    ) -> Result<(), OperationError> {
        let mut poll = eng.limits().gate();
        if let Some(stored) = level.stored() {
            let past = ChildPair::new(EncodedChildRef::from_raw(self.lefts.end), EncodedChildRef::from_raw(self.rights.start));
            let mut pairs = 0u64;
            let whole = stored.for_each_pair_whole(past, |pair| {
                self.mark_pair::<LEFT_LEAF, RIGHT_LEAF>(pair);
                pairs += u64::from(pair.left != past.left);
            });
            if whole {
                poll.poll_each(pairs)?;
                return poll.flush();
            }
            for node in stored.nodes() {
                for &pair in stored.of(node) {
                    self.mark_pair::<LEFT_LEAF, RIGHT_LEAF>(pair);
                    poll.poll(1)?;
                }
            }
            return poll.flush();
        }
        let (mut buf, mut cursor) = (Vec::new(), None);
        for i in 0..level.node_count() {
            for &pair in level.described_read_next(&mut cursor, i, &mut buf) {
                self.mark_pair::<LEFT_LEAF, RIGHT_LEAF>(pair);
                poll.poll(1)?;
            }
        }
        poll.flush()
    }

    /// Mark the cells of `pairs`, a walk outside the levels' loop.
    fn mark_pairs(&mut self, pairs: impl Iterator<Item = ChildPair>) {
        match (self.leaf.left_one, self.leaf.right_one) {
            (false, false) => pairs.for_each(|pair| self.mark_pair::<false, false>(pair)),
            (false, true) => pairs.for_each(|pair| self.mark_pair::<false, true>(pair)),
            (true, false) => pairs.for_each(|pair| self.mark_pair::<true, false>(pair)),
            (true, true) => pairs.for_each(|pair| self.mark_pair::<true, true>(pair)),
        }
    }

    /// Mark the cells `pair` covers, a leaf side's `One` standing for both
    /// cells of that leaf's `{Pos, Neg}` couple; `LEFT_LEAF` and
    /// `RIGHT_LEAF` say which sides are leaves, as `leaf` does. Fixed per
    /// walk, they leave a pair of two internal children a single cell's
    /// mark. Inlined into the walk over a level's pairs, which calls it once
    /// a pair: out of line, the call cost as much as the marking.
    #[inline(always)]
    fn mark_pair<const LEFT_LEAF: bool, const RIGHT_LEAF: bool>(&mut self, pair: ChildPair) {
        debug_assert!(LEFT_LEAF == self.leaf.left_one && RIGHT_LEAF == self.leaf.right_one);
        let (l, r) = (pair.left.0, pair.right.0);
        let l_one = LEFT_LEAF && l == ONE_LEAF_IDX.0;
        let r_one = RIGHT_LEAF && r == ONE_LEAF_IDX.0;
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

    /// Mark cell `(l, r)`, a cell of the basis, an internal child's pairs
    /// naming its slots and a leaf's `Pos` or `Neg` once its `One` is split,
    /// or the cell past it, `(lefts.end, rights.start)`.
    #[inline(always)]
    fn mark(&mut self, l: u32, r: u32) {
        debug_assert!(
            (self.lefts.contains(l) && self.rights.contains(r)) || (l == self.lefts.end && r == self.rights.start),
            "negate: a pair names a cell outside its basis"
        );
        let cell = (l - self.lefts.start) as usize * self.stride + (r - self.rights.start) as usize;
        self.bits[cell >> 6] |= 1u64 << (cell & 63);
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
        for (w, &word) in self.bits[..self.len.div_ceil(64)].iter().enumerate() {
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

#[cfg(test)]
#[path = "tests/negate/mod.rs"]
mod tests;
