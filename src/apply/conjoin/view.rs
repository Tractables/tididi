//! An operand of a conjunction onto a wider vtree, read where it is.
//!
//! The conjunction's vtree `U` holds the operand's vtree up to pass-through
//! nodes, which have the operand's variables under one child only, and free
//! nodes, which have none ([`Plan`]). The view gives each node of `U` the
//! level the operand has there once placed ([`crate::restructure::embed::place`]):
//! its own level at the image of one of its nodes, a pass-through built in
//! the view, or the empty level a free node leaves. The operand itself is
//! not changed, and a level of it is moved or copied only when the result
//! takes it ([`super::held::Held::carry`]).
//!
//! A pass-through chain over a leaf that its reader names as `Pos` or `Neg`
//! holds those two labels as its nodes 0 and 1, as placement builds it, so
//! the own level that reads the chain is renumbered: the view holds a copy
//! of it, renumbered, in place of the operand's.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{
    finish_sides_remap, plan_sides_remap, ChildPair, ChildSide, Dirty, NodeIdx, Sides, Tdd, TddLevel, TddNodeId,
};
use crate::limits::{Limits, OperationError};
use crate::restructure::embed::{literal_chain, moved_loose, Chain, Plan};
use crate::restructure::placement::{first_ref, push_free_level, PassThrough};
use crate::vtree::{Vtree, VtreeIdx};

/// The arena capacity the levels a view built keep for the next view.
const KEPT_ARENA_BYTES: u64 = 1 << 20;

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

/// What a view is built in and gives back ([`View::build_in`]): its slot
/// table and its levels.
pub(crate) type ViewTables = (Vec<Slot>, Vec<TddLevel>);

/// The operand's levels as the conjunction on `U` reads them; see the
/// module documentation.
pub(crate) struct View {
    vtree: Arc<Vtree>,
    at: Vec<Slot>,
    /// The levels the view built, then empty levels kept for reuse.
    built: Vec<TddLevel>,
    used: usize,
    empty: TddLevel,
    output: TddNodeId,
    dirty: Dirty,
}

impl View {
    /// The view of `tdd`, a structural diagram that is not false, on `into`,
    /// as `plan` (unmirrored) embeds it.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when a buffer, a built level or the
    /// copy of the worklists is refused.
    pub(crate) fn build(eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan) -> Result<View, OperationError> {
        Self::build_in(eng, tdd, into, plan, (Vec::new(), Vec::new())).map_err(|(error, _)| error)
    }

    /// [`View::build`] in `at`, a table every entry of which is
    /// [`Slot::Free`], grown to `into` where shorter, and of which only the
    /// entries of the nodes `plan` gives the operand's variables are written,
    /// building its levels in those of `built`, every one empty;
    /// [`View::release`] clears both. Refused, both come back cleared.
    pub(crate) fn build_in(
        eng: &Engine,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        plan: &Plan,
        (mut at, built): ViewTables,
    ) -> Result<View, (OperationError, ViewTables)> {
        if at.len() < into.num_nodes()
            && let Err(error) = eng.limits().try_resize(&mut at, into.num_nodes(), Slot::Free)
        {
            return Err((error, (at, built)));
        }
        let mut view = View {
            vtree: Arc::clone(into),
            at,
            built,
            used: 0,
            empty: TddLevel::new(),
            output: TddNodeId { vtree: into.root(), local: tdd.output.local },
            dirty: Dirty::default(),
        };
        match view.fill(eng, tdd, into, plan) {
            Ok(()) => Ok(view),
            Err(error) => Err((error, view.release(plan))),
        }
    }

    /// The tables of [`View::build_in`], every entry free and every level
    /// empty again.
    pub(crate) fn release(mut self, plan: &Plan) -> ViewTables {
        let mut room = KEPT_ARENA_BYTES;
        for level in &mut self.built[..self.used] {
            crate::diagram::reset_level(level, &mut room);
        }
        for &d in &plan.embedding.levels {
            self.at[d.idx()] = Slot::Free;
        }
        for &t in plan.cone_nodes() {
            self.at[t.idx()] = Slot::Free;
        }
        if plan.cone_nodes().is_empty() {
            self.at.clear();
        }
        (self.at, self.built)
    }

    /// The next empty level of `built`, and its index.
    fn next_level(&mut self, lim: &Limits) -> Result<u32, OperationError> {
        if self.used == self.built.len() {
            lim.try_push(&mut self.built, TddLevel::new())?;
        }
        self.used += 1;
        Ok(self.used as u32 - 1)
    }

    fn fill(&mut self, eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan) -> Result<(), OperationError> {
        let lim = eng.limits();
        let view = self;
        for (s, &d) in plan.embedding.levels.iter().enumerate() {
            view.at[d.idx()] = Slot::Own(s as u32);
        }
        if let Some(more) = plan.pass_throughs.checked_sub(view.built.len()) {
            lim.reserve_exact(&mut view.built, more)?;
        }
        let own = |p: VtreeIdx| &tdd.levels[plan.covered_by(p).expect("a chain's reader is an image").idx()];
        let mut literal_tops = Vec::new();
        let mut gate = lim.gate();
        for &t in plan.pass_throughs(into) {
            gate.poll(1)?;
            let (left, _) = into.children(t);
            let free_side = if plan.is_free(left) { ChildSide::Left } else { ChildSide::Right };
            let through = PassThrough::new(into, t, free_side, |c| view.level(tdd, c.idx()).slot_count());
            let chain = through.carries_leaf.then(|| {
                let read = |p: VtreeIdx, side: ChildSide| first_ref(own(p), side);
                literal_chain(into, plan, read, tdd.output.local, t)
            });
            let b = view.next_level(lim)?;
            let level = &mut view.built[b as usize];
            if let Some(chain) = chain {
                match chain {
                    Chain::Literals(top) => lim.try_push(&mut literal_tops, top)?,
                    Chain::Output(_) => view.output.local = NodeIdx(0),
                    Chain::One => {}
                }
                for &label in chain.labels() {
                    let (l, r) = free_side.pick((through.one, label), (label, through.one));
                    level.push_node(lim, &[ChildPair::new(l, r)])?;
                }
            } else {
                if through.width > 0 {
                    through.build(lim, level)?;
                }
            }
            level.close();
            level.close_held_pairs();
            view.at[t.idx()] = Slot::Built(b);
        }
        gate.flush()?;
        view.renumber(lim, tdd, &literal_tops)?;
        view.dirty = moved_dirty(eng, tdd, into, plan, &mut literal_tops)?;
        Ok(())
    }

    /// Point the readers of the chain tops `tops` at the chains' nodes 0
    /// and 1 for `Pos` and `Neg`: each reader a renumbered copy of its own
    /// level.
    fn renumber(&mut self, lim: &Limits, tdd: &Tdd, tops: &[VtreeIdx]) -> Result<(), OperationError> {
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
            let Slot::Own(s) = self.at[parent.idx()] else { unreachable!("a chain's reader is an own level") };
            let b = self.next_level(lim)?;
            let copy = &mut self.built[b as usize];
            tdd.levels[s as usize].try_clone_over(lim, 0, 0, copy)?;
            let planned = plan_sides_remap(lim, copy, moving, decoders, &REMAP)?;
            finish_sides_remap(copy, planned, moving, decoders, &REMAP);
            copy.forget_held_pairs();
            copy.close();
            copy.settle_held_pairs();
            self.at[parent.idx()] = Slot::Built(b);
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
        self.built[..self.used].iter().map(TddLevel::slot_count).max().unwrap_or(0)
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

/// `tdd`'s worklists on `into`, as [`crate::restructure::embed::place`]
/// moves them; sorts `literal_tops`.
fn moved_dirty(
    eng: &Engine,
    tdd: &Tdd,
    into: &Vtree,
    plan: &Plan,
    literal_tops: &mut [VtreeIdx],
) -> Result<Dirty, OperationError> {
    let mut carried = tdd.dirty.clone_on(eng)?;
    carried.reserve_leaf_owes_inner(eng)?;
    let loose = carried.take_loose();
    carried.remap(&plan.embedding.levels);
    literal_tops.sort_unstable();
    if !literal_tops.is_empty() {
        carried.leaf_owes_inner(|t| {
            let (left, right) = into.children(VtreeIdx(t));
            literal_tops.binary_search(&left).is_ok() || literal_tops.binary_search(&right).is_ok()
        });
    }
    carried.dedup_above(into.num_nodes());
    if let Some(mut loose) = loose {
        let mut listed = Vec::new();
        if !loose.is_empty() || !literal_tops.is_empty() {
            // `moved_loose` lists at most the two children of each image of
            // an internal level.
            eng.limits().reserve_exact(&mut listed, 2 * tdd.vtree().internal_bottomup_slice().len())?;
        }
        loose.sort_unstable();
        moved_loose(tdd.vtree(), into, plan, &loose, literal_tops, &mut listed);
        carried.set_loose(Some(listed));
    }
    Ok(carried)
}
