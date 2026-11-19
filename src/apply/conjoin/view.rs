//! A diagram on a wider vtree, read where it is.
//!
//! The wider vtree `U` holds the diagram's vtree up to pass-through nodes,
//! which have the diagram's variables under one child only, and free nodes,
//! which have none ([`Plan`]). The view gives each node of `U` the level the
//! diagram has there once placed: its own level at the image of one of its
//! nodes, a level built in the view, or the empty level a free node leaves.
//! The diagram itself is not changed. A placement
//! ([`crate::restructure::embed::place`]) is a view whose levels the result
//! takes; a conjunction onto `U` reads its plain operands through views and
//! moves or copies a level only when the result takes it
//! ([`super::held::Held::carry`]).
//!
//! A pass-through chain over a leaf that its reader names as `Pos` or `Neg`
//! holds those two labels as its nodes 0 and 1, so the own level that reads
//! the chain is renumbered: the view holds a copy of it, renumbered, in
//! place of the diagram's. So does an image whose inline counts a
//! pass-through over a region takes as nodes.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{
    return_levels,
    finish_sides_remap, plan_sides_remap, ChildPair, ChildSide, Dirty, EncodedChildRef, NodeIdx, Sides, Tdd, TddLevel,
    TddNodeId, ValueRef,
};
use crate::limits::{Limits, OperationError};
use crate::restructure::embed::{image_above, literal_chain, moved_loose, Chain, Plan};
use crate::restructure::placement::{first_ref, push_free_level, PassThrough};
use crate::vtree::{Vtree, VtreeIdx};

/// Where the view finds the operand's level at a node of `U`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    /// The operand's own level of this index.
    Own(u32),
    /// The view's level of this index.
    Built(u32),
    /// No variable of the operand is under the node.
    Free,
}

/// The operand's levels as the conjunction on `U` reads them; see the
/// module documentation.
pub(crate) struct View {
    vtree: Arc<Vtree>,
    at: Vec<Slot>,
    /// The levels the view built, at the index their slot names: the node's
    /// own index where `by_node`, the result's level array of a placement.
    built: Vec<TddLevel>,
    by_node: bool,
    empty: TddLevel,
    output: TddNodeId,
    dirty: Dirty,
    /// The tops of the chains over a leaf read as `Pos` and `Neg`, sorted.
    tops: Vec<VtreeIdx>,
}

impl View {
    /// The view of `tdd`, a diagram that is not constant, on `into`, as
    /// `plan` embeds it. `tdd` is structural, or summed out at leaves of a
    /// projection only. Where `plan` mirrors an image, the view's level
    /// there is `tdd`'s own with the source node's sides, which
    /// [`View::level`] does not exchange, or a copy of it with the image's.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when a buffer, a built level or the
    /// copy of the worklists is refused, and
    /// [`OperationError::MarginalLevel`] where a pass-through would carry a
    /// summed variable.
    pub(crate) fn build(eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan) -> Result<View, OperationError> {
        Self::build_in(eng, tdd, into, plan, Vec::new()).map_err(|(error, _)| error)
    }

    /// [`View::build`] in `at`, a table every entry of which is
    /// [`Slot::Free`], grown to `into` where shorter, and of which only the
    /// entries of the nodes `plan` gives the operand's variables are written;
    /// [`View::release`] clears them. Refused, `at` comes back cleared.
    pub(crate) fn build_in(
        eng: &Engine,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        plan: &Plan,
        at: Vec<Slot>,
    ) -> Result<View, (OperationError, Vec<Slot>)> {
        Self::build_as(eng, tdd, into, plan, at, Vec::new(), true)
    }

    /// [`View::build_in`] for a placement, which moves the worklists itself
    /// ([`reserve_moved`], [`moved_dirty`]): the view's are left empty. The
    /// view builds its levels in place in `levels`, the result's level array,
    /// of `into`'s length; refused, the array goes back to the pool.
    pub(crate) fn placing(
        eng: &Engine,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        plan: &Plan,
        at: Vec<Slot>,
        levels: Vec<TddLevel>,
    ) -> Result<View, (OperationError, Vec<Slot>)> {
        debug_assert_eq!(levels.len(), into.num_nodes());
        Self::build_as(eng, tdd, into, plan, at, levels, false)
    }

    fn build_as(
        eng: &Engine,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        plan: &Plan,
        mut at: Vec<Slot>,
        built: Vec<TddLevel>,
        worklists: bool,
    ) -> Result<View, (OperationError, Vec<Slot>)> {
        let by_node = !built.is_empty();
        if at.len() < into.num_nodes()
            && let Err(error) = eng.limits().try_resize(&mut at, into.num_nodes(), Slot::Free)
        {
            return_levels(eng, built);
            return Err((error, at));
        }
        let mut view = View {
            vtree: Arc::clone(into),
            at,
            built,
            by_node,
            empty: TddLevel::new(),
            output: TddNodeId { vtree: into.root(), local: tdd.output.local },
            dirty: Dirty::default(),
            tops: Vec::new(),
        };
        match view.fill(eng, tdd, into, plan, worklists) {
            Ok(()) => Ok(view),
            Err(error) => {
                let built = std::mem::take(&mut view.built);
                if view.by_node {
                    return_levels(eng, built);
                }
                Err((error, view.release(plan)))
            }
        }
    }

    /// Where the view finds each level, the levels it built, the output and
    /// the sorted tops of the chains over a leaf read as `Pos` and `Neg`.
    pub(crate) fn into_parts(self) -> (Vec<Slot>, Vec<TddLevel>, TddNodeId, Vec<VtreeIdx>) {
        (self.at, self.built, self.output, self.tops)
    }

    /// The table of [`View::build_in`], every entry free again.
    pub(crate) fn release(mut self, plan: &Plan) -> Vec<Slot> {
        for &d in &plan.embedding.levels {
            self.at[d.idx()] = Slot::Free;
        }
        for &t in plan.cone_nodes() {
            self.at[t.idx()] = Slot::Free;
        }
        if plan.cone_nodes().is_empty() {
            self.at.clear();
        }
        self.at
    }

    fn fill(&mut self, eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan, worklists: bool) -> Result<(), OperationError> {
        let lim = eng.limits();
        let view = self;
        for (s, &d) in plan.embedding.levels.iter().enumerate() {
            view.at[d.idx()] = Slot::Own(s as u32);
        }
        if !view.by_node {
            lim.reserve_exact(&mut view.built, plan.pass_throughs)?;
        }
        let mut literal_tops = Vec::new();
        let mut gate = lim.gate();
        for &t in plan.pass_throughs(into) {
            gate.poll(1)?;
            let (left, _) = into.children(t);
            let free_side = if plan.is_free(left) { ChildSide::Left } else { ChildSide::Right };
            let through = PassThrough::new(into, t, free_side, |c| view.level(tdd, c.idx()).slot_count());
            let carries = into.node(through.carries);
            if carries.is_region() {
                view.pass_over_region(lim, tdd, plan, t, &through)?;
                continue;
            }
            let chain = match carries.is_leaf() {
                false => None,
                true if view.level(tdd, through.carries.idx()).is_marginal() => {
                    return Err(marginal_leaf(plan, through.carries));
                }
                true => {
                    let read = |p: VtreeIdx, side: ChildSide| first_ref(view.level(tdd, p.idx()), view.oriented(plan, p, side));
                    let chain = literal_chain(into, plan, read, tdd.output.local, t);
                    match chain {
                        Chain::Literals(top) => lim.try_push(&mut literal_tops, top)?,
                        Chain::Output(_) => view.output.local = NodeIdx(0),
                        Chain::One => {}
                    }
                    Some(chain)
                }
            };
            let level = view.slot_mut(lim, t)?;
            match chain {
                None if through.width > 0 => through.build(lim, level)?,
                None => {}
                Some(chain) => {
                    for &label in chain.labels() {
                        let (l, r) = free_side.pick((through.one, label), (label, through.one));
                        level.push_node(lim, &[ChildPair::new(l, r)])?;
                    }
                }
            }
            level.close();
            level.close_held_pairs();
        }
        gate.flush()?;
        view.renumber(lim, tdd, plan, &literal_tops)?;
        literal_tops.sort_unstable();
        if worklists {
            let mut carried = tdd.dirty.clone_on(eng)?;
            let listed = reserve_moved(eng, &mut carried, tdd.vtree(), !literal_tops.is_empty())?;
            view.dirty = moved_dirty(carried, listed, tdd.vtree(), into, plan, &literal_tops);
        }
        view.tops = literal_tops;
        Ok(())
    }

    /// The side of the level the view gives at `t` that reads `t`'s child
    /// on `side`: the other one where `t` mirrors its source node and the
    /// view reads the operand's own level, whose sides are the source's.
    #[inline]
    fn oriented(&self, plan: &Plan, t: VtreeIdx, side: ChildSide) -> ChildSide {
        match self.at[t.idx()] {
            Slot::Own(_) if plan.is_mirrored(t) => side.pick(ChildSide::Right, ChildSide::Left),
            _ => side,
        }
    }

    /// The view's level at `t`, where it builds it: the one it built there
    /// before, if any, else an empty level, the result's in a placement.
    #[inline]
    fn slot_mut(&mut self, lim: &Limits, t: VtreeIdx) -> Result<&mut TddLevel, OperationError> {
        let b = match self.at[t.idx()] {
            Slot::Built(b) => b as usize,
            _ if self.by_node => t.idx(),
            _ => {
                lim.try_push(&mut self.built, TddLevel::new())?;
                self.built.len() - 1
            }
        };
        self.at[t.idx()] = Slot::Built(b as u32);
        Ok(&mut self.built[b])
    }

    /// The view's level at the image `t`, with `t`'s sides, for the view to
    /// change: the one it built there, else a copy of the operand's.
    fn reader_mut(&mut self, lim: &Limits, tdd: &Tdd, plan: &Plan, t: VtreeIdx) -> Result<&mut TddLevel, OperationError> {
        let own = match self.at[t.idx()] {
            Slot::Own(s) => Some(s as usize),
            _ => None,
        };
        let level = self.slot_mut(lim, t)?;
        if let Some(s) = own {
            tdd.levels[s].try_clone_over(lim, 0, 0, level)?;
            if plan.is_mirrored(t) {
                level.swap_sides();
            }
        }
        Ok(level)
    }

    /// Build the pass-through at `t` over the region `through` carries: a
    /// node per slot of the region at the same index. Where the region holds
    /// counts and an image reads it through the chain, one more node per
    /// inline value that image holds on that side, which the image's copy in
    /// the view names instead.
    #[cold]
    #[inline(never)]
    fn pass_over_region(&mut self, lim: &Limits, tdd: &Tdd, plan: &Plan, t: VtreeIdx, through: &PassThrough) -> Result<(), OperationError> {
        let region = self.level(tdd, through.carries.idx());
        let counts = region.is_marginal() && !region.is_weight_marginal();
        let reader = image_above(&self.vtree, plan, t).filter(|_| counts);
        let decoder = crate::diagram::ChildDecoder::marginal();
        let inline_value = |r: EncodedChildRef| !r.is_reserved() && matches!(decoder.value(r), ValueRef::Inline(_));
        let mut inline = Vec::new();
        let mut value_refs = None;
        if let Some((parent, side)) = reader {
            let read = self.level(tdd, parent.idx());
            let read_side = self.oriented(plan, parent, side);
            value_refs = Some(read.has_value_refs(read_side));
            for (_, pairs) in read.internal_inputs_iter() {
                for pair in pairs {
                    let r = read_side.pick(pair.left, pair.right);
                    if inline_value(r) {
                        lim.try_push(&mut inline, r)?;
                    }
                }
            }
        }
        let free_side = through.free_side;
        let level = self.slot_mut(lim, t)?;
        for (left, right) in through.pairs() {
            level.push_node(lim, &[ChildPair::new(left, right)])?;
        }
        for &value in &inline {
            let (left, right) = free_side.pick((through.one, NodeIdx(value.0)), (NodeIdx(value.0), through.one));
            level.push_node(lim, &[ChildPair::new(left, right)])?;
        }
        if let Some(value_refs) = value_refs {
            level.set_has_value_refs(free_side.pick(ChildSide::Right, ChildSide::Left), value_refs);
        }
        level.close();
        level.close_held_pairs();
        let (Some((parent, side)), false) = (reader, inline.is_empty()) else { return Ok(()) };
        let copy = self.reader_mut(lim, tdd, plan, parent)?;
        let mut next = through.width as u32;
        crate::diagram::for_each_side_ref_mut(lim, copy, side, |r| {
            if inline_value(EncodedChildRef(*r)) {
                *r = next;
                next += 1;
            }
        })?;
        copy.set_has_value_refs(side, false);
        copy.close();
        copy.close_held_pairs();
        Ok(())
    }

    /// Point the readers of the chain tops `tops` at the chains' nodes 0
    /// and 1 for `Pos` and `Neg`: each reader a renumbered copy of its own
    /// level.
    fn renumber(&mut self, lim: &Limits, tdd: &Tdd, plan: &Plan, tops: &[VtreeIdx]) -> Result<(), OperationError> {
        const REMAP: [u32; 3] = [u32::MAX, 0, 1];
        let vtree = Arc::clone(&self.vtree);
        // Each reader once, with the sides on which it reads a chain.
        let mut parents: Vec<(VtreeIdx, Sides<bool>)> = Vec::new();
        lim.reserve_exact(&mut parents, tops.len())?;
        for &top in tops {
            let parent = vtree.node(top).parent().expect("a chain that reaches the root reads no level");
            let left = vtree.children(parent).0 == top;
            parents.push((parent, Sides { left, right: !left }));
        }
        parents.sort_unstable_by_key(|&(parent, _)| parent.0);
        parents.dedup_by(|later, kept| {
            let same = later.0 == kept.0;
            if same {
                kept.1 = Sides { left: kept.1.left || later.1.left, right: kept.1.right || later.1.right };
            }
            same
        });
        for (parent, moving) in parents {
            let (left, right) = vtree.children(parent);
            let decoders = Sides {
                left: self.level(tdd, left.idx()).child_decoder(),
                right: self.level(tdd, right.idx()).child_decoder(),
            };
            let copy = self.reader_mut(lim, tdd, plan, parent)?;
            let planned = plan_sides_remap(lim, copy, moving, decoders, &REMAP)?;
            finish_sides_remap(copy, planned, moving, decoders, &REMAP);
            copy.forget_held_pairs();
            copy.close();
            copy.settle_held_pairs();
        }
        Ok(())
    }

    /// The vtree the view is on.
    #[inline]
    pub(crate) fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// The operand's output on the view's vtree.
    #[inline]
    pub(crate) fn output(&self) -> TddNodeId {
        self.output
    }

    /// The operand's worklists on the view's vtree.
    #[inline]
    pub(crate) fn dirty(&self) -> &Dirty {
        &self.dirty
    }

    /// The widest level the view built.
    pub(crate) fn built_width(&self) -> usize {
        debug_assert!(!self.by_node, "a placement's view is not conjoined");
        self.built.iter().map(TddLevel::slot_count).max().unwrap_or(0)
    }

    /// The level of `tdd`, the operand, at node `t` of the view's vtree.
    #[inline]
    pub(crate) fn level<'a>(&'a self, tdd: &'a Tdd, t: usize) -> &'a TddLevel {
        match self.at[t] {
            Slot::Own(s) => &tdd.levels[s as usize],
            Slot::Built(b) => &self.built[b as usize],
            Slot::Free => &self.empty,
        }
    }

    /// The level at `t` where the view holds it, `Err` with the operand's
    /// own index where the operand does, `Ok(None)` where `t` is free.
    #[inline]
    pub(crate) fn held_mut(&mut self, t: usize) -> Result<Option<&mut TddLevel>, usize> {
        match self.at[t] {
            Slot::Own(s) => Err(s as usize),
            Slot::Built(b) => Ok(Some(&mut self.built[b as usize])),
            Slot::Free => Ok(None),
        }
    }

    /// Put the level at the free node `t` into `out`, which is empty.
    #[inline]
    pub(crate) fn build_free(&self, t: usize, out: &mut TddLevel) {
        push_free_level(out, &self.vtree, VtreeIdx(t as u32));
    }
}

/// The refusal of a pass-through over the leaf `carried` that holds a summed
/// variable, whose values live in its parent's references, which a new
/// parent between them does not take.
#[cold]
fn marginal_leaf(plan: &Plan, carried: VtreeIdx) -> OperationError {
    OperationError::MarginalLevel(plan.covered_by(carried).unwrap_or(carried))
}

/// Ready `carried`, the worklists of a diagram on `source`, for
/// [`moved_dirty`]: room for the inner pass's share of the leaf pass's, and
/// the room for the listed loose levels, returned, where some are listed or
/// `chains` (a chain over a leaf read as `Pos` and `Neg`) may list them.
pub(crate) fn reserve_moved(eng: &Engine, carried: &mut Dirty, source: &Vtree, chains: bool) -> Result<Vec<u32>, OperationError> {
    carried.reserve_leaf_owes_inner(eng)?;
    let mut listed = Vec::new();
    if carried.loose().is_some_and(|loose| !loose.is_empty() || chains) {
        // `moved_loose` lists at most the two children of each image of an
        // internal level.
        eng.limits().reserve_exact(&mut listed, 2 * source.internal_bottomup_slice().len())?;
    }
    Ok(listed)
}

/// `carried`, readied by [`reserve_moved`] with the room `listed`, moved
/// from `source` onto `into` as [`crate::restructure::embed::place`] moves
/// worklists; `tops` are the sorted tops of the chains over a leaf read as
/// `Pos` and `Neg`.
pub(crate) fn moved_dirty(
    mut carried: Dirty,
    mut listed: Vec<u32>,
    source: &Vtree,
    into: &Vtree,
    plan: &Plan,
    tops: &[VtreeIdx],
) -> Dirty {
    let loose = carried.take_loose();
    carried.remap(&plan.embedding.levels);
    if !tops.is_empty() {
        carried.leaf_owes_inner(|t| {
            let (left, right) = into.children(VtreeIdx(t));
            tops.binary_search(&left).is_ok() || tops.binary_search(&right).is_ok()
        });
    }
    carried.dedup_above(into.num_nodes());
    if let Some(mut loose) = loose {
        loose.sort_unstable();
        moved_loose(source, into, plan, &loose, tops, &mut listed);
        carried.set_loose(Some(listed));
    }
    carried
}
