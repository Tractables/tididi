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
use std::borrow::Cow;
use std::sync::Arc;

/// The buffers negation reuses between calls.
///
/// Take-and-return, as everywhere else: the capacity is what is being reused,
/// so a warmed engine allocates nothing per level, and a negation over a deep
/// vtree does not pay a `Vec` for each of its levels.
#[derive(Default)]
pub(crate) struct NegateScratch {
    /// One level's `lefts x rights` cover, one byte per cell.
    cover: Pool<Vec<u8>>,
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
    /// Run [`Tdd::negate`](crate::Tdd::negate) using this batch's scratch and
    /// resource limits, on an owned or borrowed operand as
    /// [`negate_with`](Self::negate_with) says.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn negate<'f>(&self, f: impl Into<Cow<'f, Tdd>>) -> Result<Tdd, OperationError> {
        self.negate_with(f, crate::reduce::ReductionPlan::default())
    }

    /// [`negate`](Self::negate) with the reduction it ends with chosen
    /// explicitly. See [`Tdd::negate_with`] for what the plans cost and what
    /// they leave behind.
    ///
    /// The operand is a [`Tdd`], consumed on success and on error, or a
    /// `&Tdd`, read in place and left as it is. The result holds every
    /// level of the operand with the node negation adds to it, so an owned
    /// operand's levels grow in place and a borrowed one's are copied, each
    /// once and with room for that node, where cloning first copies each
    /// level and then grows it. The copies are charged to the memory
    /// budget, not to the work clock; owned or borrowed, the result is the
    /// same diagram for the same work.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let either = Tdd::clause(&vtree, [1, 2])?;
    /// let neither = Engine::new().negate(&either)?;
    /// assert!(neither.equivalent(&Tdd::cube(&vtree, [-1, -2])?)?);
    /// assert_eq!(either.model_count()?, 6u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`negate`](Self::negate).
    pub fn negate_with<'f>(
        &self,
        f: impl Into<Cow<'f, Tdd>>,
        plan: crate::reduce::ReductionPlan<'_>,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        let f = f.into();
        let canonical = f.levels.is_canonical(f.output);
        let (mut result, old) = complement_on(self, f)?;
        // `expand_full` left every level below the root covering its
        // children's whole basis, and the old output and its complement cover
        // the root's, so every node is named from the level above and the only
        // nodes the complement can have orphaned are below the root cells it
        // dropped.
        // The complement of a canonical operand has no twins to contract.
        let full = matches!(plan, crate::reduce::ReductionPlan::Full(_));
        if canonical && (full || matches!(plan, crate::reduce::ReductionPlan::Prune)) {
            self.certify_complement(&mut result, full, old)?;
        } else {
            self.reduce_scoped(&mut result, plan, crate::reduce::prune::PruneScope::BelowRoot)?;
        }
        Ok(result)
    }
}

/// Make `tdd` full and complement it at the root, consuming the operand;
/// [`Engine::negate`] also minimizes the result.
pub(crate) fn negate_on(eng: &Engine, tdd: Tdd) -> Result<Tdd, OperationError> {
    complement_on(eng, Cow::Owned(tdd)).map(|(result, _)| result)
}

/// [`negate_on`] of an owned or borrowed operand, with the old output's slot
/// where the complement was built beside it.
pub(crate) fn complement_on(eng: &Engine, tdd: Cow<'_, Tdd>) -> Result<(Tdd, Option<NodeIdx>), OperationError> {
    tdd.require_structure()?;
    eng.limits().check_stop()?;
    let vtree = Arc::clone(&tdd.vtree);
    // Read before the first change to a level, which drops both.
    let (canonical, closed) = (tdd.levels.is_canonical(tdd.output), tdd.levels.is_closed());
    let (zero, output) = (tdd.is_zero(), tdd.output.local);
    let at_root = tdd.output.vtree == vtree.root();
    let (mut levels, source, weights, owed) = match tdd {
        Cow::Owned(mut tdd) => {
            let weights = tdd.weights.take();
            let owed = tdd.take_worklists();
            (tdd.levels.into_vec(), None, weights, owed)
        }
        Cow::Borrowed(tdd) => {
            let weights = tdd.weights.as_ref().map(|w| w.try_clone_on(eng.limits())).transpose()?;
            let owed = tdd.dirty.clone_on(eng)?;
            (Target::empty_levels(eng, &tdd.levels)?, Some(&tdd.levels[..]), weights, owed)
        }
    };
    let (mut result, old) = if zero {
        (crate::build::constant_on(eng, &vtree, true)?, None)
    } else {
        let mut filled = Vec::new();
        eng.limits().try_resize(&mut filled, vtree.num_nodes(), false)?;
        expand_full(eng, &vtree, &mut Target { levels: &mut levels, source }, canonical, closed, &mut filled)?;
        let owed = at_root.then(|| Owed::of(&vtree, owed, canonical, &filled, &levels));
        complement_full_at_root(eng, &vtree, levels, source, output, &filled, closed, owed)?
    };
    #[cfg(debug_assertions)]
    if at_root && !zero && result.output.vtree == vtree.root() {
        let parents: Vec<_> = vtree.internal_bottomup().map(|(t, _, _)| t).collect();
        crate::reduce::contract::debug_assert_no_twins_under(eng, &result, &parents);
    }
    result.weights = weights;
    Ok((result, old))
}

/// What a complement owes the reduction passes, where its operand's output
/// is the root's: the operand's own worklists, the root, and each level
/// `expand_full` filled that may have given two nodes of a child equal
/// contexts. Every other level is the operand's, with at most a fill node
/// added, and the fill node gives a child's nodes contexts in itself only:
/// two nodes with equal contexts after it had equal contexts before it,
/// which the operand, not owing the level, has only for nodes no pair of
/// the level names. Those are on a child the operand lists loose (any
/// child, where it does not know them), and the fill names each with every
/// node of the other child. A level the operand had no node at is owed
/// too: its fill alone decides its leaf form.
struct Owed {
    carried: crate::diagram::Dirty,
    rebuilt: Vec<crate::vtree::VtreeIdx>,
}

impl Owed {
    fn of(
        vtree: &crate::vtree::Vtree,
        mut carried: crate::diagram::Dirty,
        canonical: bool,
        filled: &[bool],
        levels: &[TddLevel],
    ) -> Self {
        // A canonical operand has every node named; otherwise the operand's
        // loose levels, read before the seeding forgets them.
        let loose = if canonical { Some(Vec::new()) } else { carried.take_loose() };
        let may_be_loose = |c: crate::vtree::VtreeIdx| {
            !vtree.node(c).is_leaf() && loose.as_ref().is_none_or(|l| l.contains(&c.0))
        };
        let root = vtree.root();
        let mut rebuilt = vec![root];
        for (t, left, right) in vtree.internal_bottomup() {
            if t != root && filled[t.idx()] && (may_be_loose(left) || may_be_loose(right) || levels[t.idx()].node_count() == 1) {
                rebuilt.push(t);
            }
        }
        Self { carried, rebuilt }
    }
}

/// The levels a negation writes: the operand's own, or, for a borrowed
/// operand (`source`), empty levels each internal level of the operand is
/// copied into when the negation reaches it ([`write`](Self::write)). A
/// leaf level of a structural diagram is empty, so it is its own copy.
pub(crate) struct Target<'l, 's> {
    pub(crate) levels: &'l mut Vec<TddLevel>,
    pub(crate) source: Option<&'s [TddLevel]>,
}

impl Target<'_, '_> {
    /// Empty levels for the copy of `source`'s.
    fn empty_levels(eng: &Engine, source: &[TddLevel]) -> Result<Vec<TddLevel>, OperationError> {
        debug_assert!(
            source.iter().all(|level| level.node_count() > 0 || level.arena_len() == 0),
            "a structural level without nodes holds no pairs",
        );
        let mut levels = Vec::new();
        eng.limits().reserve_exact(&mut levels, source.len())?;
        levels.resize_with(source.len(), TddLevel::new);
        Ok(levels)
    }

    /// Level `t` as the operand has it, to read before it is written.
    fn read(&self, t: usize) -> &TddLevel {
        self.source.map_or(&self.levels[t], |source| &source[t])
    }

    /// Level `t` to write, once a level: with room for a node of `fill`
    /// pairs where one is about to be added, an owned level grown in place
    /// and a borrowed operand's copied with that room. An owned level with
    /// nothing to add, which negation meets at nearly every level it
    /// visits, is reached without a call.
    #[inline(always)]
    fn write(&mut self, eng: &Engine, t: usize, fill: Option<usize>) -> Result<&mut TddLevel, OperationError> {
        if self.source.is_none() && fill.is_none() {
            return Ok(&mut self.levels[t]);
        }
        self.copy_or_grow(eng, t, fill)
    }

    /// [`write`](Self::write) of a borrowed operand's level or of a level
    /// about to take a fill node.
    #[inline(never)]
    fn copy_or_grow(&mut self, eng: &Engine, t: usize, fill: Option<usize>) -> Result<&mut TddLevel, OperationError> {
        if let Some(source) = self.source {
            let from = &source[t];
            let (nodes, pairs) = match fill {
                Some(len) if from.stored().is_some() => (1, fill_pairs(len)),
                _ => (0, 0),
            };
            self.levels[t] = from.try_clone_with_room_on(eng.limits(), nodes, pairs)?;
        }
        let level = &mut self.levels[t];
        if let Some(len) = fill {
            reserve_fill(eng, level, len)?;
        }
        Ok(level)
    }
}

/// Complement at the root a diagram full below it: collect the root-level
/// pairs not in the output node and drop the dead ones. With the old
/// output's slot when the complement is a node beside it. `levels` and
/// `source` are an [`expand_full`] target's. `closed` says the levels were
/// closed before [`expand_full`], which left the root as it was.
#[expect(clippy::too_many_arguments)]
fn complement_full_at_root(
    eng: &Engine,
    vtree: &Arc<crate::vtree::Vtree>,
    levels: Vec<TddLevel>,
    source: Option<&[TddLevel]>,
    out_local: NodeIdx,
    filled: &[bool],
    closed: bool,
    owed: Option<Owed>,
) -> Result<(Tdd, Option<NodeIdx>), OperationError> {
    let root = vtree.root();
    let root_idx = root.idx();

    let mut assembly = Assembly::from_levels(eng, Arc::clone(vtree), levels, None);
    let (levels, _) = assembly.parts_mut();

    if vtree.node(root).is_leaf() {
        // Implicit leaf: the output index and the label are the same number. It stays in
        // {Pos,Neg,One} unless the output is One (complement = Zero, returned as the zero
        // constant diagram).
        let Some(neg_local) = complement_leaf_root(out_local) else {
            return Ok((crate::build::constant_on(eng, vtree, false)?, None));
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
        let mut marked = scratch.cover.checkout_preserving(eng);
        let mut neg_pairs = scratch.cells.checkout_preserving(eng);
        let mut target = Target { levels, source };
        let lefts = ChildBasis::of(vtree, &target.levels[..], left);
        let rights = ChildBasis::of(vtree, &target.levels[..], right);
        let leaf = LeafForm { left_one: left_is_leaf, right_one: right_is_leaf };
        let fill = [filled[left.idx()].then(|| lefts.end - 1), filled[right.idx()].then(|| rights.end - 1)];
        // The other root nodes are dropped with the old output, so the
        // complement takes the leaf form the output uses.
        let mut form = LeafForm::LITERAL;
        let root_level = target.read(root_idx);
        let covered = areas(root_level.pairs_iter_of_idx(out_local.idx()), leaf, &mut form);
        neg_pairs.clear();
        // The work clock is charged a scan of the root's basis, by either
        // route.
        charge_scan(eng, lefts, rights)?;
        if covered == old_cells(lefts, rights, fill) {
            // The output covers every cell of the children's old nodes: its
            // complement is the lines of their added nodes.
            fill_lines_into(eng, lefts, rights, fill, form, &mut neg_pairs)?;
            #[cfg(debug_assertions)]
            {
                let mut cover = Cover::reset(eng, &mut marked, lefts, rights, form)?;
                cover.mark_pairs(root_level.pairs_iter_of_idx(out_local.idx()));
                let mut by_cover = Vec::new();
                cover.missing_into(eng, &mut by_cover)?;
                debug_assert_eq!(*neg_pairs, by_cover, "negate: the lines of the added nodes are not the root cells the output misses");
            }
        } else {
            let mut cover = Cover::reset(eng, &mut marked, lefts, rights, form)?;
            cover.mark_pairs(root_level.pairs_iter_of_idx(out_local.idx()));
            cover.missing_into(eng, &mut neg_pairs)?;
        }

        // Drop the pairs whose child computes the Zero function (an internal
        // node with no pairs), which a fill node of `expand_full` can be.
        neg_pairs.retain(|pair| {
            let left_alive = left_is_leaf
                || !empty_node(&target.levels[left.idx()], ChildDecoder::structural().node(pair.left).idx());
            let right_alive = right_is_leaf
                || !empty_node(&target.levels[right.idx()], ChildDecoder::structural().node(pair.right).idx());
            left_alive && right_alive
        });

        if neg_pairs.is_empty() {
            return Ok((crate::build::constant_on(eng, vtree, false)?, None));
        }

        let pairs = level_pairs(target.read(root_idx), closed) as usize;
        let root_level = target.write(eng, root_idx, Some(neg_pairs.len()))?;
        let neg_idx = root_level.push_node(eng.limits(), &neg_pairs)?;
        root_level.leave_live_pairs(pairs + neg_pairs.len());

        let output = TddNodeId { vtree: root, local: neg_idx };
        let result = match owed {
            Some(owed) => {
                let mut result = assembly.finish_with(output, owed.carried, &owed.rebuilt, None)?;
                // Each level under the root's children covers its children's
                // whole basis, every cell a pair, so names each of their
                // nodes. The complement drops the cells of an empty node, so
                // the root's children may hold one no pair names.
                let loose = [(left, left_is_leaf), (right, right_is_leaf)].into_iter().filter(|&(_, leaf)| !leaf);
                result.dirty.set_loose(Some(loose.map(|(c, _)| c.0).collect()));
                result
            }
            None => assembly.finish(output)?,
        };
        Ok((result, Some(out_local)))
    }
}

// ── `expand_full`: explicit fill-node materialization ────────────────────────────

/// Make every level of a structural diagram below the root full by
/// materializing fill nodes explicitly. The root is left alone: a fill there
/// would be the complement of its nodes, which the negation computes from the
/// output node alone.
///
/// One bottom-up pass per level marks the cells of `lefts x rights` that the
/// level's nodes cover, a byte per cell held in engine scratch, and the fill
/// node is read off the cells left unmarked. The walk also records, per leaf
/// child, whether the level refers to it as `One` or as `Pos`/`Neg`, and the
/// fill is emitted in that same form. `filled[t]` is set for each level `t`
/// it adds a node to, as the level's last. Each level it visits is left its
/// live pairs for the close that ends the negation
/// ([`TddLevel::leave_live_pairs`]), off the pairs the cover counts. A
/// borrowed operand's levels are read in `target`'s source and each copied
/// once into its levels.
///
/// `canonical` and `closed` are the operand's, read before any change.
pub(crate) fn expand_full(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    target: &mut Target<'_, '_>,
    canonical: bool,
    closed: bool,
    filled: &mut [bool],
) -> Result<(), OperationError> {
    let scratch = &eng.scratch.negate;
    let mut marked = scratch.cover.checkout_preserving(eng);
    let mut cells = scratch.cells.checkout_preserving(eng);

    expand_full_with(eng, vtree, target, (canonical, closed), &mut marked, &mut cells, filled)
}

/// [`expand_full`] with the scratch checked out.
fn expand_full_with(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    target: &mut Target<'_, '_>,
    (canonical, closed): (bool, bool),
    marked: &mut Vec<u8>,
    cells: &mut Vec<ChildPair>,
    filled: &mut [bool],
) -> Result<(), OperationError> {
    let root = vtree.root();
    for (t, left, right) in vtree.internal_bottomup().filter(|&(t, _, _)| t != root) {
        // Read bottom-up: a child's basis includes the fill node its own visit
        // added.
        let lefts = ChildBasis::of(vtree, &target.levels[..], left);
        let rights = ChildBasis::of(vtree, &target.levels[..], right);
        let leaf = LeafForm {
            left_one: vtree.node(left).is_leaf(),
            right_one: vtree.node(right).is_leaf(),
        };
        // A child the completion added a node to has it last.
        let fill = [filled[left.idx()].then(|| lefts.end - 1), filled[right.idx()].then(|| rights.end - 1)];
        let level = target.read(t.idx());

        // On a canonical diagram every stored node is reachable and
        // deterministic, so the pairs of a level cover disjoint cells, and
        // their count says whether they cover every cell, the level being
        // full already, or every cell of the children's old nodes, the
        // missing cells then being the lines of the children's added nodes,
        // read off without a cover. An operand with unreachable nodes may
        // hold two that cover one cell, which only the cover sees.
        let form = level_form(level, leaf);
        let pairs = level_pairs(level, closed);
        if canonical {
            let covered = pairs << (u32::from(form.left_one) + u32::from(form.right_one));
            #[cfg(debug_assertions)]
            {
                let (by_pairs, form_by_pairs) = covered_cells_by_pairs(level, leaf);
                debug_assert_eq!(covered, by_pairs, "negate: the covered cells are not the pairs' areas");
                debug_assert!(
                    (form.left_one, form.right_one) == (form_by_pairs.left_one, form_by_pairs.right_one),
                    "negate: the level's first pair does not name its leaves in the level's form",
                );
            }
            // Both routes charge the work clock what the cover below does:
            // a unit a pair, and a scan of the basis where a cell is
            // missing.
            if covered == (lefts.len() * rights.len()) as u64 {
                #[cfg(debug_assertions)]
                debug_assert!(missing_by_cover(eng, marked, level, lefts, rights, form)?.is_empty(), "negate: a level whose pairs fill its basis misses a cell");
                charge_pairs(eng, pairs)?;
                target.write(eng, t.idx(), None)?.leave_live_pairs(pairs as usize);
                eng.limits().check_stop()?;
                continue;
            }
            if covered == old_cells(lefts, rights, fill) {
                charge_pairs(eng, pairs)?;
                charge_scan(eng, lefts, rights)?;
                if fill != [None, None] {
                    cells.clear();
                    fill_lines_into(eng, lefts, rights, fill, form, cells)?;
                    #[cfg(debug_assertions)]
                    debug_assert_eq!(*cells, missing_by_cover(eng, marked, level, lefts, rights, form)?, "negate: the lines of the added nodes are not the cells no pair covers");
                    let level = target.write(eng, t.idx(), Some(cells.len()))?;
                    level.push_node(eng.limits(), cells)?;
                    level.leave_live_pairs(pairs as usize + cells.len());
                    filled[t.idx()] = true;
                } else {
                    target.write(eng, t.idx(), None)?.leave_live_pairs(pairs as usize);
                }
                eng.limits().check_stop()?;
                continue;
            }
        }

        // Mark the cells the pairs cover and read the fill off the rest, in
        // the level's form.
        let mut cover = Cover::reset(eng, marked, lefts, rights, form)?;
        // On a canonical diagram the fill holds the cells the pairs leave,
        // so its room is known before the cover is read. Written first, a
        // level whose arena moves or is copied for it is read from its new
        // place, which the write has just made.
        let level = if canonical {
            let missing = cover.len.saturating_sub(pairs as usize);
            &*target.write(eng, t.idx(), (missing > 0).then_some(missing))?
        } else {
            level
        };
        cover.mark_level(eng, level, pairs)?;
        cells.clear();
        cover.missing_into(eng, cells)?;
        debug_assert!(!canonical || !cells.is_empty(), "negate: a level short of its cells has none missing");
        debug_assert!(!canonical || cells.len() + pairs as usize == cover.len, "negate: the fill is not the cells the pairs leave");
        if !cells.is_empty() {
            charge_scan(eng, lefts, rights)?;
            // Written with its room before the cover where the diagram is
            // canonical.
            let level = if canonical { &mut target.levels[t.idx()] } else { target.write(eng, t.idx(), Some(cells.len()))? };
            level.push_node(eng.limits(), cells)?;
            filled[t.idx()] = true;
        } else if !canonical {
            target.write(eng, t.idx(), None)?;
        }
        target.levels[t.idx()].leave_live_pairs(pairs as usize + cells.len());
        eng.limits().check_stop()?;
    }
    Ok(())
}

/// The arena pairs a fill node of `len` pairs takes: none for one pair,
/// which its node word holds.
fn fill_pairs(len: usize) -> usize {
    if len > 1 { len } else { 0 }
}

/// Room on `level` for a fill node of `len` pairs, exactly: the arena grows
/// by the fill rather than doubling, which a negated diagram would carry
/// until it is dropped. A described level is stored first, as the push
/// would store it, so that it ends with the capacities a stored one does.
fn reserve_fill(eng: &Engine, level: &mut TddLevel, len: usize) -> Result<(), OperationError> {
    level.store_if_implicit(eng.limits())?;
    level.reserve_on(eng.limits(), 1, fill_pairs(len))
}

/// The leaf form `level` uses, read off its first pair: a level names a
/// leaf child as `One` everywhere or nowhere ([`LeafForm`]). `leaf` says
/// which sides are leaf children.
fn level_form(level: &TddLevel, leaf: LeafForm) -> LeafForm {
    let mut form = LeafForm::LITERAL;
    if (leaf.left_one || leaf.right_one)
        && let Some(pair) = (0..level.node_count()).find_map(|i| level.pairs_iter_of_idx(i).next())
    {
        areas(std::iter::once(pair), leaf, &mut form);
    }
    form
}

/// The pairs of `level`, counted off its nodes, or read off the count a
/// level of a `closed` diagram keeps. On a canonical operand they cover
/// disjoint cells, determinism's rule, one cell a pair and two for each
/// side that is a leaf named `One`, which the level's form says, so the
/// cells they cover are the pairs doubled for each such side, without
/// reading the pairs.
fn level_pairs(level: &TddLevel, closed: bool) -> u64 {
    (if closed { level.live_pairs_closed() } else { level.live_pairs() }) as u64
}

/// Charge the work clock `pairs` units, a unit a pair, as a cover's walk
/// over them would.
fn charge_pairs(eng: &Engine, pairs: u64) -> Result<(), OperationError> {
    let mut poll = eng.limits().gate();
    poll.poll_each(pairs)?;
    poll.flush()
}

/// Charge the work clock a scan of the `lefts x rights` basis's cover: 64
/// units a word of 64 cells, each leaf side's `{Pos, Neg}` couple apart.
fn charge_scan(eng: &Engine, lefts: ChildBasis, rights: ChildBasis) -> Result<(), OperationError> {
    let words = (lefts.len() as u64 * rights.len() as u64).div_ceil(64);
    charge_pairs(eng, 64 * words)
}

/// The cells `level`'s pairs cover off every pair's area, with the level's
/// form, for a debug build to check the pair count against.
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
    marked: &mut Vec<u8>,
    level: &TddLevel,
    lefts: ChildBasis,
    rights: ChildBasis,
    form: LeafForm,
) -> Result<Vec<ChildPair>, OperationError> {
    let mut cover = Cover::reset(eng, marked, lefts, rights, form)?;
    cover.mark_pairs((0..level.node_count()).flat_map(|i| level.pairs_iter_of_idx(i)));
    let mut out = Vec::new();
    cover.missing_into(eng, &mut out)?;
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
/// level, read off its first pair ([`level_form`]), and a cover's basis
/// ([`Cover::reset`]). A fill node and the complement at the root are
/// emitted in the same form, which is what keeps the level's labels on one
/// side of that line.
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

/// A walk's marking of a [`Cover`] ([`Cover::marks`]).
struct Marks<'b> {
    marked: &'b mut [u8],
    lefts: ChildBasis,
    rights: ChildBasis,
    stride: usize,
    /// `lefts.start * stride + rights.start`: cell `(l, r)` is byte
    /// `l * stride + r - origin`.
    origin: usize,
}

impl Marks<'_> {
    /// Mark the cell `pair` covers. Inlined into the walk over a level's
    /// pairs, which calls it once a pair: out of line, the call cost as
    /// much as the marking.
    #[inline(always)]
    fn pair(&mut self, pair: ChildPair) {
        self.cell(pair.left.0, pair.right.0);
    }

    /// Mark cell `(l, r)`, a cell of the cover's basis, or the cell past it,
    /// `(lefts.end, rights.start)`.
    #[inline(always)]
    fn cell(&mut self, l: u32, r: u32) {
        debug_assert!(
            (self.lefts.contains(l) && self.rights.contains(r)) || (l == self.lefts.end && r == self.rights.start),
            "negate: a pair names a cell outside its basis"
        );
        let cell = (l as usize * self.stride + r as usize).wrapping_sub(self.origin);
        self.marked[cell] = 1;
    }
}

/// A byte per cell of one level's `lefts x rights` basis in the level's
/// leaf form, 1 where a pair covers the cell: cell `(l, r)` is byte `(l -
/// lefts.start) * rights.len() + (r - rights.start)`, where a leaf side
/// named `One` has the one index `One`, standing for the `{Pos, Neg}`
/// couple its every pair covers.
///
/// A cell is marked by one store, where a bit would be read, set and
/// written back, and the levels a negation covers are nearly full, their
/// pairs most of their cells. The completion of the level holds a pair of
/// eight bytes for every cell, so the cover is at most an eighth of it.
struct Cover<'a> {
    marked: &'a mut Vec<u8>,
    lefts: ChildBasis,
    rights: ChildBasis,
    /// `rights.len()`, the row stride.
    stride: usize,
    /// `lefts.len() * rights.len()`, the cell count.
    len: usize,
}

impl<'a> Cover<'a> {
    /// Clear `marked` and size it for the `lefts x rights` basis, a leaf side
    /// the level names `One` (`form`) as that one index.
    fn reset(
        eng: &Engine,
        marked: &'a mut Vec<u8>,
        lefts: ChildBasis,
        rights: ChildBasis,
        form: LeafForm,
    ) -> Result<Cover<'a>, OperationError> {
        lefts.len().checked_mul(rights.len()).ok_or(OperationError::IndexOverflow)?;
        let one = ChildBasis { start: ONE_LEAF_IDX.0, end: ONE_LEAF_IDX.0 + 1 };
        let (lefts, rights) = (if form.left_one { one } else { lefts }, if form.right_one { one } else { rights });
        let len = lefts.len() * rights.len();
        marked.clear();
        // And the cell past the basis, which a walk marks for a node it
        // reads no pair of ([`mark_level`](Self::mark_level)).
        eng.limits().try_resize(marked, len + 1, 0u8)?;
        Ok(Cover { marked, lefts, rights, stride: rights.len(), len })
    }

    /// Mark the cells of every pair of `level`'s nodes, `pairs` of them,
    /// and charge the work clock a unit a pair.
    ///
    /// A stored level whose nodes' ranges cover its arena is read node by
    /// node and then the arena whole, with no branch on a node's kind
    /// ([`StoredPairs::for_each_pair_whole`]): a node whose pairs are in the
    /// arena marks the cell past the basis, which the cover keeps for it and
    /// [`missing_into`](Self::missing_into) never reads. On a level whose
    /// arena holds dead pairs, that read has marked the pairs held in node
    /// words, and the other nodes' pairs are read as slices of the arena. An
    /// implicit level's nodes are generated into a buffer a node at a time.
    ///
    /// [`StoredPairs::for_each_pair_whole`]: crate::diagram::StoredPairs::for_each_pair_whole
    fn mark_level(&mut self, eng: &Engine, level: &TddLevel, pairs: u64) -> Result<(), OperationError> {
        let mut marks = self.marks();
        if let Some(stored) = level.stored() {
            let past = ChildPair::new(EncodedChildRef::from_raw(marks.lefts.end), EncodedChildRef::from_raw(marks.rights.start));
            if !stored.for_each_pair_whole(past, |pair| marks.pair(pair)) {
                for node in stored.nodes().iter().filter(|node| node.kind().pairs_in_arena()) {
                    stored.of(node).iter().for_each(|&pair| marks.pair(pair));
                }
            }
        } else {
            let (mut buf, mut cursor) = (Vec::new(), None);
            for i in 0..level.node_count() {
                level.described_read_next(&mut cursor, i, &mut buf).iter().for_each(|&pair| marks.pair(pair));
            }
        }
        charge_pairs(eng, pairs)
    }

    /// Mark the cells of `pairs`, a walk outside the levels' loop.
    fn mark_pairs(&mut self, pairs: impl Iterator<Item = ChildPair>) {
        let mut marks = self.marks();
        pairs.for_each(|pair| marks.pair(pair));
    }

    /// The cover's bytes and basis for a walk to mark, held apart from it
    /// so that the walk keeps them in registers.
    #[inline(always)]
    fn marks(&mut self) -> Marks<'_> {
        debug_assert!(self.marked.len() > self.len, "the cover has the cell past its basis");
        let origin = self.lefts.start as usize * self.stride + self.rights.start as usize;
        Marks { marked: &mut self.marked[..], lefts: self.lefts, rights: self.rights, stride: self.stride, origin }
    }

    /// Append the unmarked cells to `out`, in ascending `(l, r)` order, a
    /// side the level names `One` in that form: its every pair covers both
    /// cells of a `{Pos, Neg}` couple, so a fill names the couple as `One`,
    /// which is what lets the fill join a level whose other labels are
    /// `One` without splitting them. Mixing the two at one leaf would break
    /// determinism's label rule and, downstream, block leaf-twin
    /// contraction, which is all-or-nothing per level. The caller charges
    /// the scan ([`charge_scan`]).
    fn missing_into(&self, eng: &Engine, out: &mut Vec<ChildPair>) -> Result<(), OperationError> {
        let lim = eng.limits();
        let mut push = |cell: usize| {
            let l = self.lefts.start + (cell / self.stride) as u32;
            let r = self.rights.start + (cell % self.stride) as u32;
            lim.try_push(out, ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)))
        };
        // Eight cells a word: a marked cell's byte is 1, so the low bit of
        // each byte of the word's complement is set where a cell is missing.
        let (body, tail) = self.marked[..self.len].split_at(self.len & !7);
        for (w, chunk) in body.chunks_exact(8).enumerate() {
            let word = u64::from_le_bytes(chunk.try_into().expect("a chunk of eight"));
            let mut missing = !word & 0x0101_0101_0101_0101;
            while missing != 0 {
                push(w * 8 + (missing.trailing_zeros() >> 3) as usize)?;
                missing &= missing - 1;
            }
        }
        for (i, &byte) in tail.iter().enumerate() {
            if byte == 0 {
                push(body.len() + i)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/negate/mod.rs"]
mod tests;
