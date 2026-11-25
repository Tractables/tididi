//! Copy a diagram onto a larger vtree under a renaming of its variables.
//!
//! The destination must contain the source vtree's shape under the renaming.
//! A destination node with renamed variables on both sides is the image of a
//! source node and receives that source level unchanged, node indices and all.
//! A node with renamed variables on one side only passes that side's nodes
//! through, one node per node at the same index, so the copied level above
//! still finds each child where it left it. A subtree the renaming does not
//! reach becomes constant true. Nothing is applied.

use std::sync::Arc;

use super::{EmbedError, Embedding, placement::push_free_level};

use crate::Engine;
use crate::apply::conjoin::view::{copied_dirty, moved_dirty, reserve_moved, Slot, View};
use crate::diagram::{
    ChildSide, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd, TddLevel,
    return_levels, try_take_levels,
};
use crate::limits::OperationError;
use crate::vtree::{Home, VarId, Vtree, VtreeError, VtreeIdx};

/// What each destination node does in the copy, and where each source level went.
#[derive(Debug)]
pub(crate) struct Plan {
    /// Per destination node, what its level is.
    targets: Vec<Target>,
    /// The destination node each source node maps to.
    pub(crate) embedding: Embedding,
    /// Whether any image is mirrored.
    mirrors: bool,
    /// The number of pass-throughs.
    pub(crate) pass_throughs: usize,
    /// For a plan [`Plan::cone`] built, every internal destination node with
    /// a source variable under it, children before parents; empty otherwise.
    cone: Vec<VtreeIdx>,
}

/// What a destination node's level is under a [`Plan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// No renamed source variable is under the node: a free level, constant
    /// true.
    Free,
    /// The image of `source`; `mirrored` when its children are the source
    /// node's swapped, so the level is copied with every pair read the other
    /// way round.
    Image { source: VtreeIdx, mirrored: bool },
    /// An internal node with renamed variables on one side only, carrying
    /// that side's source node.
    PassThrough(VtreeIdx),
}

impl Plan {
    /// Whether no renamed source variable is under destination node `t`.
    #[inline]
    pub(crate) fn is_free(&self, t: VtreeIdx) -> bool {
        self.targets[t.idx()] == Target::Free
    }

    /// The source node destination node `t` is the image of, if any.
    #[inline]
    pub(crate) fn covered_by(&self, t: VtreeIdx) -> Option<VtreeIdx> {
        match self.targets[t.idx()] {
            Target::Image { source, .. } => Some(source),
            _ => None,
        }
    }

    /// Whether destination node `t` is the image of a source node with its
    /// children swapped.
    #[inline]
    pub(crate) fn is_mirrored(&self, t: VtreeIdx) -> bool {
        matches!(self.targets[t.idx()], Target::Image { mirrored: true, .. })
    }

    /// The destination nodes that are mirrored images.
    pub(crate) fn mirrored_images(&self) -> impl Iterator<Item = VtreeIdx> + '_ {
        let images: &[VtreeIdx] = if self.mirrors { &self.embedding.levels } else { &[] };
        images.iter().copied().filter(|&t| self.is_mirrored(t))
    }

    /// The pass-throughs of the plan, `into`'s internal nodes with renamed
    /// variables on one side only, bottom-up.
    pub(crate) fn pass_throughs<'a>(&'a self, into: &'a Vtree) -> impl Iterator<Item = &'a VtreeIdx> {
        let all = match self.cone.is_empty() {
            true => into.internal_bottomup_slice(),
            false => &self.cone[..],
        };
        all.iter()
            .filter(|&&t| matches!(self.targets[t.idx()], Target::PassThrough(_)))
            .take(self.pass_throughs)
    }

    /// For a plan [`Plan::cone`] built, the internal destination nodes with
    /// a source variable under them, children before parents.
    pub(crate) fn cone_nodes(&self) -> &[VtreeIdx] {
        &self.cone
    }

    /// The plan that embeds `source` in `into`, a projection of the same home
    /// on a superset of its variables, with `leaves` the image of each leaf
    /// of `source`: the nodes of `into` with a variable of `source` under
    /// them are found from the leaves upward, and only their entries of
    /// `targets`, every one [`Target::Free`] and grown to `into` where
    /// shorter, are written. [`Plan::release`] clears them. The work clock is
    /// charged one unit a node found.
    ///
    /// The chain from the image of `source`'s root to `into`'s root is left
    /// free, but for its first level where that root is a leaf: a
    /// conjunction reads `source` there as one node passing through, and
    /// builds the levels of the chain it reaches one at a time
    /// ([`View::pass_through`](crate::apply::conjoin::view::View::pass_through)).
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when a table cannot grow, or the stop;
    /// `targets` comes back with every entry free.
    pub(crate) fn cone(
        lim: &crate::limits::Limits,
        source: &Vtree,
        into: &Vtree,
        leaves: &[(VtreeIdx, VtreeIdx)],
        mut targets: Vec<Target>,
    ) -> Result<Plan, (OperationError, Vec<Target>)> {
        let mut plan = Plan { targets: Vec::new(), embedding: Embedding { levels: Vec::new() }, mirrors: false, pass_throughs: 0, cone: Vec::new() };
        if let Err(error) = lim.try_resize(&mut plan.embedding.levels, source.num_nodes(), into.root()) {
            return Err((error, targets));
        }
        if targets.len() < into.num_nodes()
            && let Err(error) = lim.try_resize(&mut targets, into.num_nodes(), Target::Free)
        {
            return Err((error, targets));
        }
        plan.targets = targets;
        match plan.find_cone(lim, source, into, leaves) {
            Ok(()) => Ok(plan),
            Err(error) => Err((error, plan.release())),
        }
    }

    fn find_cone(&mut self, lim: &crate::limits::Limits, source: &Vtree, into: &Vtree, leaves: &[(VtreeIdx, VtreeIdx)]) -> Result<(), OperationError> {
        let mut gate = lim.gate();
        for &(s, d) in leaves {
            self.embedding.levels[s.idx()] = d;
            self.targets[d.idx()] = Target::Image { source: s, mirrored: false };
        }
        let mut found = |plan: &mut Plan, t: VtreeIdx, target: Target| -> Result<(), OperationError> {
            gate.poll(1)?;
            lim.try_push(&mut plan.cone, t)?;
            plan.targets[t.idx()] = target;
            plan.pass_throughs += usize::from(matches!(target, Target::PassThrough(_)));
            Ok(())
        };
        for &s in source.internal_bottomup_slice() {
            let home = source.home_node(s);
            let mut image = into.root();
            for c in <[VtreeIdx; 2]>::from(source.children(s)) {
                let mut u = self.embedding.levels[c.idx()];
                loop {
                    u = into.node(u).parent().expect("a source node's image is under its parent's");
                    if into.home_node(u) == home {
                        break;
                    }
                    found(self, u, Target::PassThrough(c))?;
                }
                image = u;
            }
            self.embedding.levels[s.idx()] = image;
            found(self, image, Target::Image { source: s, mirrored: false })?;
        }
        let top = source.root();
        if source.node(top).is_leaf()
            && let Some(parent) = into.node(self.embedding.levels[top.idx()]).parent()
        {
            found(self, parent, Target::PassThrough(top))?;
        }
        gate.flush()
    }

    /// The plan of a source whose levels are read through `targets`, by node
    /// of the destination, with `embedding` the image of each source node and
    /// `cone`, children before parents, every internal destination node with
    /// a source variable under it, `pass_throughs` of them pass-throughs.
    pub(crate) fn from_parts(targets: Vec<Target>, embedding: Vec<VtreeIdx>, cone: Vec<VtreeIdx>, pass_throughs: usize) -> Plan {
        Plan { targets, embedding: Embedding { levels: embedding }, mirrors: false, pass_throughs, cone }
    }

    /// The tables of [`Plan::from_parts`], as they are.
    pub(crate) fn into_parts(self) -> (Vec<Target>, Vec<VtreeIdx>, Vec<VtreeIdx>) {
        (self.targets, self.embedding.levels, self.cone)
    }

    /// The targets table of a plan [`Plan::cone`] built, every entry free.
    pub(crate) fn release(mut self) -> Vec<Target> {
        for &d in &self.embedding.levels {
            self.targets[d.idx()] = Target::Free;
        }
        for &t in &self.cone {
            self.targets[t.idx()] = Target::Free;
        }
        self.targets
    }

    /// Check the renaming against both trees and record the correspondence.
    ///
    /// Carries to each node of `into` the source node whose leaves are the
    /// renamed variables under it. A node with renamed variables on one side
    /// only is a pass-through carrying that side's source node, which is
    /// exactly the node [`Vtree::project_to_vars`] splices out. A node with
    /// renamed variables on both sides is the image of the parent of the two
    /// source nodes its children carry, which must be siblings, and is
    /// mirrored when the left child carries the right sibling, which an exact
    /// match refuses. Linear in the nodes of `into` with a renamed variable
    /// under them, plus the two tables.
    ///
    /// The work clock is charged one unit for each source leaf, one for each
    /// internal node of `into`, and one for each internal node of `into` with
    /// a renamed variable under it, in bottom-up order up to the one the
    /// match fails at, that one included.
    ///
    /// `mirror` is unset where the assembly cannot mirror a level: a
    /// marginal one, or a projection's. With [`Free::Build`], a region of
    /// `into` that is not the image of one of `source`'s is refused
    /// ([`EmbedError::UnsummedRegion`]).
    pub(crate) fn build(
        eng: &Engine,
        source: &Vtree,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
        free: Free,
    ) -> Result<Plan, EmbedError> {
        Self::build_as(eng, source, into, map, mirror, free, true)
    }

    /// [`build`](Self::build), or without `climb` by a [`Vtree::leaf_of`] per
    /// leaf and one pass over every internal node of `into`, which the tests
    /// compare it with.
    fn build_as(
        eng: &Engine,
        source: &Vtree,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
        free: Free,
        climb: bool,
    ) -> Result<Plan, EmbedError> {
        let lim = eng.limits();
        let (mut targets, mut embedding) = (Vec::new(), Vec::new());
        lim.try_resize(&mut targets, into.num_nodes(), Target::Free)?;
        lim.try_resize(&mut embedding, source.num_nodes(), into.root())?;
        let mut gate = lim.gate();
        if climb
            && let Some((home, ranks)) = into.leaf_ranks()
            && Self::match_by_rank(source, home, ranks, &map, &mut targets, &mut embedding)
        {
            gate.poll_each(source.num_leaves() as u64)?;
        } else {
            for (leaf, var) in source.leaf_bottomup() {
                gate.poll(1)?;
                let image = map(var);
                let target = into.leaf_of(image).ok_or(EmbedError::VariableOutOfRange {
                    variable: image,
                    num_vars: into.num_vars(),
                })?;
                if targets[target.idx()] != Target::Free {
                    return Err(VtreeError::OverlappingVariable(image).into());
                }
                embedding[leaf.idx()] = target;
                targets[target.idx()] = Target::Image { source: leaf, mirrored: false };
            }
        }
        if source.has_regions() {
            map_regions(source, into, &mut targets, &mut embedding)?;
        }
        if free == Free::Build && into.has_regions() {
            refuse_free_region(into, &targets)?;
        }
        gate.poll_each(into.internal_bottomup_slice().len() as u64)?;
        let climbed = match climb {
            true => climb_from_leaves(source, into, mirror, &mut targets, &mut embedding),
            false => None,
        };
        let Matched { mirrors, pass_throughs, visited, refused } = match climbed {
            Some(matched) => matched,
            None => match_by_pass(source, into, mirror, &mut targets, &mut embedding),
        };
        gate.poll_each(visited)?;
        if let Some(s) = refused {
            return Err(EmbedError::NotIsomorphic { source: s });
        }
        debug_assert_eq!(
            targets
                .iter()
                .filter(|target| matches!(target, Target::Image { .. }))
                .count(),
            source.num_nodes(),
            "a completed match gives every source level an image",
        );
        gate.flush()?;
        Ok(Plan {
            targets,
            embedding: Embedding { levels: embedding },
            mirrors,
            pass_throughs,
            cone: Vec::new(),
        })
    }

    /// [`build`](Self::build)'s leaf match when `into` is a projection of
    /// `home` without regions, by home rank in `ranks`, `into`'s, instead of
    /// a [`Vtree::leaf_of`] per leaf. A projection of the same home is read
    /// in rank order, so under the identity the match is one merge; a rank
    /// below the last one matched is found by binary search. Returns `false`,
    /// with `targets` all free again, at the first leaf it cannot place, for
    /// `build`'s own loop to report; it charges no work.
    #[inline(never)]
    fn match_by_rank(
        source: &Vtree,
        home: &Home,
        ranks: &[u32],
        map: &impl Fn(VarId) -> VarId,
        targets: &mut [Target],
        embedding: &mut [VtreeIdx],
    ) -> bool {
        let mut cursor = RankCursor { ranks, next: 0 };
        let placed = match source.leaf_ranks() {
            Some((own, own_ranks)) if Arc::ptr_eq(own, home) => {
                own_ranks.iter().enumerate().all(|(i, &rank)| {
                    let leaf = VtreeIdx(i as u32);
                    let var = source.leaf_var(leaf);
                    let image = map(var);
                    let rank = if image == var { Some(rank) } else { home.rank(image) };
                    cursor.place(leaf, rank, targets, embedding)
                })
            }
            _ => source
                .leaf_bottomup()
                .all(|(leaf, var)| cursor.place(leaf, home.rank(map(var)), targets, embedding)),
        };
        if !placed {
            targets.fill(Target::Free);
        }
        placed
    }
}

/// The internal match of [`Plan::build`]: whether any image is mirrored, the
/// number of pass-throughs, the work units, and the source node refused.
struct Matched {
    mirrors: bool,
    pass_throughs: usize,
    visited: u64,
    refused: Option<VtreeIdx>,
}

/// [`Plan::build`]'s internal match from the images of `source`'s childless
/// nodes upward, in `targets` and `embedding`. The image of each internal
/// source node is where the climbs from its children's images meet,
/// stepping each side in turn; a node a climb passes is a pass-through, and
/// the steps the longer side took past the meeting are undone. `None` where
/// a climb reaches a node marked for another source node, or a mirrored
/// image `mirror` refuses: [`match_by_pass`] then decides, overwriting what
/// was written.
fn climb_from_leaves(
    source: &Vtree,
    into: &Vtree,
    mirror: bool,
    targets: &mut [Target],
    embedding: &mut [VtreeIdx],
) -> Option<Matched> {
    let carries = |target: Target, s: VtreeIdx| matches!(target, Target::Image { source, .. } | Target::PassThrough(source) if source == s);
    let (mut mirrors, mut pass_throughs) = (false, 0usize);
    for &s in source.internal_bottomup_slice() {
        let (a, b) = source.children(s);
        let sides = [a, b];
        let mut ends = [embedding[a.idx()], embedding[b.idx()]];
        let mut side = 0;
        let meet = loop {
            let up = match into.node(ends[side]).parent() {
                Some(up) => up,
                None => {
                    side ^= 1;
                    into.node(ends[side]).parent()?
                }
            };
            match targets[up.idx()] {
                Target::Free => {
                    targets[up.idx()] = Target::PassThrough(sides[side]);
                    pass_throughs += 1;
                    ends[side] = up;
                    side ^= 1;
                }
                Target::PassThrough(other) if other == sides[side ^ 1] => break up,
                _ => return None,
            }
        };
        let mut past = meet;
        while past != ends[side ^ 1] {
            past = into.node(past).parent().expect("a climb's end is above where it met the other");
            targets[past.idx()] = Target::Free;
            pass_throughs -= 1;
        }
        pass_throughs -= 1;
        let swapped = carries(targets[into.children(meet).0.idx()], b);
        if swapped && !mirror {
            return None;
        }
        mirrors |= swapped;
        targets[meet.idx()] = Target::Image { source: s, mirrored: swapped };
        embedding[s.idx()] = meet;
    }
    if source.num_nodes() > 0 {
        let top = source.root();
        let mut u = embedding[top.idx()];
        while let Some(up) = into.node(u).parent() {
            if targets[up.idx()] != Target::Free {
                return None;
            }
            targets[up.idx()] = Target::PassThrough(top);
            pass_throughs += 1;
            u = up;
        }
    }
    let visited = (source.num_leaves() as usize + source.internal_bottomup_slice().len() + pass_throughs) as u64;
    Some(Matched { mirrors, pass_throughs, visited, refused: None })
}

/// [`Plan::build`]'s internal match in one bottom-up pass over every internal
/// node of `into`, whatever `targets` holds there.
#[cold]
#[inline(never)]
fn match_by_pass(source: &Vtree, into: &Vtree, mirror: bool, targets: &mut [Target], embedding: &mut [VtreeIdx]) -> Matched {
    let carried = |target: Target| match target {
        Target::Free => None,
        Target::Image { source, .. } | Target::PassThrough(source) => Some(source),
    };
    let mut visited = source.num_leaves() as u64;
    let (mut mirrors, mut pass_throughs) = (false, 0);
    for &t in into.internal_bottomup_slice() {
        let (left, right) = into.children(t);
        targets[t.idx()] = match (carried(targets[left.idx()]), carried(targets[right.idx()])) {
            (None, None) => Target::Free,
            (Some(s), None) | (None, Some(s)) => {
                visited += 1;
                pass_throughs += 1;
                Target::PassThrough(s)
            }
            (Some(a), Some(b)) => {
                visited += 1;
                let p = source.node(a).parent().expect("a source node beside another is not the root");
                let swapped = source.children(p).1 == a;
                if source.node(b).parent() != Some(p) || (swapped && !mirror) {
                    return Matched { mirrors, pass_throughs, visited, refused: Some(p) };
                }
                embedding[p.idx()] = t;
                mirrors |= swapped;
                Target::Image { source: p, mirrored: swapped }
            }
        };
    }
    Matched { mirrors, pass_throughs, visited, refused: None }
}

/// `into`'s leaves by home rank, `ranks` being `into`'s by node index, and
/// the index after the last one matched.
struct RankCursor<'a> {
    ranks: &'a [u32],
    next: usize,
}

impl RankCursor<'_> {
    /// Make `into`'s leaf of home rank `rank` the image of source leaf
    /// `leaf`; `false` when `into` has no such leaf or it is an image already.
    #[inline(always)]
    fn place(&mut self, leaf: VtreeIdx, rank: Option<u32>, targets: &mut [Target], embedding: &mut [VtreeIdx]) -> bool {
        let Some(rank) = rank else { return false };
        if self.ranks.get(self.next) != Some(&rank) {
            match seek(self.ranks, self.next, rank) {
                Some(i) => self.next = i,
                None => return false,
            }
        }
        let target = VtreeIdx(self.next as u32);
        self.next += 1;
        if targets[target.idx()] != Target::Free {
            return false;
        }
        embedding[leaf.idx()] = target;
        targets[target.idx()] = Target::Image { source: leaf, mirrored: false };
        true
    }
}

/// The index of `rank` in `ranks`, ascending, where `ranks[from]` is not
/// it: by steps doubling from `from` when it is larger, else by binary
/// search below `from`.
#[inline(never)]
fn seek(ranks: &[u32], from: usize, rank: u32) -> Option<usize> {
    let i = match ranks.get(from) {
        Some(&r) if r < rank => {
            let (mut lo, mut step) = (from, 1);
            while lo + step < ranks.len() && ranks[lo + step] < rank {
                lo += step;
                step *= 2;
            }
            let end = (lo + step + 1).min(ranks.len());
            lo + 1 + ranks[lo + 1..end].partition_point(|&r| r < rank)
        }
        _ => ranks[..from].partition_point(|&r| r < rank),
    };
    (ranks.get(i) == Some(&rank)).then_some(i)
}

/// Map each region of `source` to its image in [`Plan::fill`]: the region of
/// `into` at its anchor, its ancestors not free.
#[cold]
fn map_regions(source: &Vtree, into: &Vtree, targets: &mut [Target], embedding: &mut [VtreeIdx]) -> Result<(), EmbedError> {
    for region in source.regions() {
        let target = into
            .node_at_home(source.home_node(region))
            .filter(|&t| into.node(t).is_region() && into.region_at(t) == source.region_at(region))
            .ok_or(EmbedError::NotIsomorphic { source: region })?;
        targets[target.idx()] = Target::Image { source: region, mirrored: false };
        embedding[region.idx()] = target;
    }
    Ok(())
}

/// Refuse a region of `into` no source region maps to: the copy would have
/// to give it a value, and the diagram has none for variables it has not
/// summed out.
#[cold]
fn refuse_free_region(into: &Vtree, targets: &[Target]) -> Result<(), EmbedError> {
    match into.regions().find(|t| targets[t.idx()] == Target::Free) {
        Some(region) => Err(EmbedError::UnsummedRegion { region }),
        None => Ok(()),
    }
}

/// What [`place_moving`] does with the free levels, those of the destination
/// nodes no renamed variable is under.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Free {
    /// Build each: one node, true on both sides. A free region is refused.
    Build,
    /// Build each internal one; a free region keeps an empty level that is
    /// not summed out and is named `One`, as a free variable leaf is, for an
    /// operand that another operand's sum meets there.
    Operand,
    /// Leave each empty, for a conjunction that reads it as the level it
    /// stands for and takes the other operand's level there; a projection,
    /// whose free nodes are those it drops, builds none either.
    Leave,
}

/// `tdd` moved onto `into`, which holds its shape under `map`, with the plan
/// it placed `tdd` by; with [`Free::Leave`], a diagram whose free levels are
/// empty. A
/// false diagram is the false diagram on `into`, every level empty, whatever
/// `free` says.
pub(crate) fn place_moving(
    eng: &Engine,
    tdd: Tdd,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    free: Free,
) -> Result<(Tdd, Plan), EmbedError> {
    place(eng, std::borrow::Cow::Owned(tdd), into, map, free).map_err(|(error, _)| error)
}

/// [`place_moving`] of an owned or borrowed diagram: a borrowed one has its
/// levels copied into the destination's, weight values included. On a
/// projection, a summed-out level may sit at a leaf: a region, or a summed
/// variable whose parent stays its parent. Refused, the diagram comes back
/// as it was given.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
pub(crate) fn place<'a>(
    eng: &Engine,
    tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    free: Free,
) -> Result<(Tdd, Plan), (EmbedError, std::borrow::Cow<'a, Tdd>)> {
    if let Err(e) = require_placeable(&tdd) {
        return Err((e.into(), tdd));
    }
    let plan = match Plan::build(eng, tdd.vtree(), into, map, true, free) {
        Ok(plan) => plan,
        Err(error) => return Err((error, tdd)),
    };
    place_by(eng, tdd, into, &plan, free).map(|result| (result, plan))
}

/// Refuse a summed-out level [`place`] does not move: any on a root vtree,
/// an internal one on a home.
fn require_placeable(tdd: &Tdd) -> Result<(), OperationError> {
    match tdd.vtree().on_home() {
        false => tdd.require_structure(),
        true if !tdd.has_marginal_level() => Ok(()),
        true => match tdd.vtree().bottomup().find(|&t| !tdd.vtree().node(t).is_leaf() && tdd.level(t).is_marginal()) {
            Some(t) => Err(OperationError::MarginalLevel(t)),
            None => Ok(()),
        },
    }
}

/// [`place`] by a `plan` built for `tdd`, a structural diagram or one whose
/// summed-out levels are leaves of a projection.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
pub(crate) fn place_by<'a>(
    eng: &Engine,
    tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
) -> Result<Tdd, (EmbedError, std::borrow::Cow<'a, Tdd>)> {
    if tdd.is_zero() || tdd.is_one() {
        return match crate::build::constant_padded_on(eng, into, tdd.is_one()) {
            Ok(constant) => Ok(constant),
            Err(e) => Err((e.into(), tdd)),
        };
    }
    assemble_view(eng, tdd, into, plan, free).map_err(|(e, tdd)| (e.into(), tdd))
}

/// Make the diagram `tdd`'s [`View`] on `into` describes: a moved diagram's
/// levels moved to their images, a borrowed one's copied, the view's built
/// levels taken, and free levels built or left empty as `free` says. The
/// result takes the view's worklists. Refused, `tdd` comes back as it was
/// given.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
fn assemble_view<'a>(
    eng: &Engine,
    tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
) -> Result<Tdd, (OperationError, std::borrow::Cow<'a, Tdd>)> {
    if !is_region_alone(&tdd, into) {
        return seat_view(eng, tdd, into, plan, free);
    }
    // The inlining can be refused after the levels moved: it works on a copy.
    let placed = seat_view(eng, std::borrow::Cow::Borrowed(&*tdd), into, plan, free)
        .map_err(|(e, _)| e)
        .and_then(|mut result| inline_root_region(eng, &mut result).map(|()| result));
    placed.map_err(|e| (e, tdd))
}

/// Whether `tdd` is an unweighted region alone placed under a parent, whose
/// small counts are inline references there instead of slots.
fn is_region_alone(tdd: &Tdd, into: &Vtree) -> bool {
    tdd.vtree().num_nodes() == 1
        && tdd.weight_table().is_none()
        && tdd.vtree().node(tdd.vtree().root()).is_region()
        && into.num_nodes() > 1
}

/// Inline the small counts of `result`, a region alone placed under a
/// parent, and prune their slots from the region.
#[cold]
fn inline_root_region(eng: &Engine, result: &mut Tdd) -> Result<(), OperationError> {
    crate::diagram::inline_small_marginal_refs(eng.limits(), result, None)
        .and_then(|()| crate::reduce::slot_prune::prune_value_slots(eng, result).map(drop))
}

/// [`assemble_view`] without the inlining of a region alone.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
fn seat_view<'a>(
    eng: &Engine,
    mut tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
) -> Result<Tdd, (OperationError, std::borrow::Cow<'a, Tdd>)> {
    use std::borrow::Cow;
    let levels = match try_take_levels(eng, into.num_nodes()) {
        Ok(levels) => levels,
        Err(e) => return Err((e, tdd)),
    };
    let mut slots = eng.scratch.apply.cone_slots.checkout_preserving(eng);
    let view = match View::placing(eng, &tdd, into, plan, std::mem::take(&mut *slots), levels) {
        Ok(view) => view,
        Err((e, back)) => {
            *slots = back;
            return Err((e, tdd));
        }
    };
    let (mut at, mut levels, output, tops) = view.into_parts();
    let closed = tdd.levels.is_closed();
    // A moved diagram's worklists are taken, with room made in place.
    let carried = match &mut tdd {
        Cow::Owned(moved) => reserve_moved(eng, &mut moved.dirty, &moved.vtree, !tops.is_empty()).map(|listed| (None, listed)),
        Cow::Borrowed(source) => {
            copied_dirty(eng, &source.dirty, source.vtree(), into, plan, &tops).map(|copied| (Some(copied), Vec::new()))
        }
    };
    let borrowed = match &tdd {
        Cow::Borrowed(source) => Some(*source),
        Cow::Owned(_) => None,
    };
    let ready = carried.and_then(|carried| {
        take_placement(eng, borrowed, into, plan, &at, &mut levels, free).map(|()| carried)
    });
    let (copied, listed) = match ready {
        Ok(ready) => ready,
        Err(e) => {
            return_levels(eng, levels);
            at.clear();
            *slots = at;
            return Err((e, tdd));
        }
    };
    let own = |d: VtreeIdx| matches!(at[d.idx()], Slot::Own(_));
    let dirty = match &mut tdd {
        Cow::Owned(moved) => {
            let source = &mut moved.levels[..];
            for (s, &d) in plan.embedding.levels.iter().enumerate() {
                if own(d) {
                    levels[d.idx()].swap_with(&mut source[s]);
                }
            }
            let carried = moved.take_worklists();
            moved_dirty(carried, listed, (&moved.vtree, into), (&plan.embedding.levels, |t| plan.is_mirrored(t)), &tops)
        }
        Cow::Borrowed(_) => copied.expect("a borrowed diagram's worklists are copied"),
    };
    if free != Free::Leave {
        for &t in into.internal_bottomup_slice() {
            if plan.is_free(t) {
                push_free_level(&mut levels[t.idx()], into, t);
            }
        }
    }
    for t in plan.mirrored_images() {
        if own(t) {
            levels[t.idx()].swap_sides();
            levels[t.idx()].close();
            levels[t.idx()].close_held_pairs();
        }
    }
    let mut result = Tdd { vtree: Arc::clone(into), levels: levels.into(), output, dirty };
    // The view closed what it built, and the levels taken unchanged were
    // closed where `tdd`'s were.
    match closed {
        true => result.close_changed_levels(&[]),
        false => result.close_levels(),
    }
    // Every level of a moved `tdd` the result took is now one taken from
    // the pool, empty: they go back for the takes that follow.
    if let Cow::Owned(moved) = &mut tdd {
        return_levels(eng, std::mem::take(&mut moved.levels).into_vec());
    }
    at.clear();
    *slots = at;
    Ok(result)
}

/// What [`seat_view`] can be refused once the view is built, taken before
/// anything moves: the free levels' work units where they are built, a
/// borrowed diagram's own levels copied into `levels`, the result's level
/// array. `at` is the view's table, and `borrowed` the diagram when it is
/// copied.
fn take_placement(
    eng: &Engine,
    borrowed: Option<&Tdd>,
    into: &Vtree,
    plan: &Plan,
    at: &[Slot],
    levels: &mut [TddLevel],
    free: Free,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    if free != Free::Leave {
        let mut gate = lim.gate();
        for &t in into.internal_bottomup_slice() {
            if plan.is_free(t) {
                gate.poll(1)?;
            }
        }
        gate.flush()?;
    }
    if let Some(source) = borrowed {
        for (s, &d) in plan.embedding.levels.iter().enumerate() {
            if matches!(at[d.idx()], Slot::Own(_)) {
                source.levels[s].try_copy_over(lim, &mut levels[d.idx()])?;
            }
        }
    }
    Ok(())
}

/// The nearest ancestor of `t` that is the image of a source level, and the
/// side of it `t` is under.
pub(crate) fn image_above(into: &Vtree, plan: &Plan, mut t: VtreeIdx) -> Option<(VtreeIdx, ChildSide)> {
    while let Some(parent) = into.node(t).parent() {
        if plan.covered_by(parent).is_some() {
            return Some((parent, ChildSide::of(into, parent, t)));
        }
        t = parent;
    }
    None
}

/// The levels of a placement's result that may hold a node no pair
/// of their parent level names, pushed to `out` bottom-up, given `loose`,
/// the source's such levels, and `literal_tops`, the tops of the
/// pass-through chains that read a leaf as `Pos` and `Neg`, both sorted.
/// `source` is the vtree the levels were moved from, `images` the image of
/// each of its nodes and `mirrored` the images whose children are their
/// source's swapped; the walk visits the images of its internal nodes and
/// the chains under them, not the levels built with no source variable
/// under them.
///
/// Only a level whose parent is the image of a source level can hold one. A
/// level built under a built level is named whole: a pass-through has a
/// node for each slot of the level it carries, and the single node of a
/// level with no source variable is the true node every node of its parent
/// names, of which there is one at least, as on every level of a diagram
/// that is not false. Under an image, an image is named as its source level
/// was, and the top of a pass-through chain as the level at the chain's
/// foot, whose slots the chain carries one for one; a chain over a leaf
/// holds `One`, which the image reads, or `Pos` and `Neg`, of which it may
/// read one.
pub(crate) fn moved_loose(
    source: &Vtree,
    into: &Vtree,
    (images, mirrored): (&[VtreeIdx], impl Fn(VtreeIdx) -> bool),
    loose: &crate::diagram::LooseLevels,
    literal_tops: &[VtreeIdx],
    out: &mut Vec<u32>,
) {
    // With neither, every level is named whole.
    if loose.is_empty() && literal_tops.is_empty() {
        return;
    }
    for &s in source.internal_bottomup_slice() {
        let p = images[s.idx()];
        let (left, right) = into.children(p);
        let (sl, sr) = source.children(s);
        let feet = if mirrored(p) { [sr, sl] } else { [sl, sr] };
        // The chain under a child of `p` ends at the image of the source
        // child on that side.
        for (top, from) in [left, right].into_iter().zip(feet) {
            if into.node(top).is_leaf() {
                continue;
            }
            let named_whole = match source.node(from).is_leaf() {
                true => literal_tops.binary_search(&top).is_err(),
                false => !loose.contains(from.0),
            };
            if !named_whole {
                out.push(top.0);
            }
        }
    }
}

/// How the chain of pass-throughs over a leaf that `t` is on reads the leaf.
pub(crate) enum Chain {
    /// As `One`: the chain has one node a level.
    One,
    /// As `Pos` and `Neg`, read by the level above the chain, the top given:
    /// the chain has the two as its nodes 0 and 1, and its reader is
    /// renumbered.
    Literals(VtreeIdx),
    /// As the output's literal, the chain reaching the root: the chain has
    /// that literal alone, as its node 0, which the output names.
    Output(NodeIdx),
}

impl Chain {
    /// The leaf labels the chain holds, in node order.
    pub(crate) fn labels(&self) -> &[NodeIdx] {
        match self {
            Chain::One => &[ONE_LEAF_IDX],
            Chain::Literals(_) => &[POS_LEAF_IDX, NEG_LEAF_IDX],
            Chain::Output(label) => std::slice::from_ref(label),
        }
    }
}

/// How the chain of pass-throughs a pass-through over a leaf at `t` starts
/// reads the leaf: as the level above the chain names it, the first node
/// above `t` that `is_image`, or as the output does when the chain reaches
/// the root or a node that `is_free`. Determinism makes the first reference
/// the level's form.
pub(crate) fn literal_chain(
    into: &Vtree,
    (is_image, is_free): (impl Fn(VtreeIdx) -> bool, impl Fn(VtreeIdx) -> bool),
    first_ref: impl Fn(VtreeIdx, ChildSide) -> Option<NodeIdx>,
    output: NodeIdx,
    t: VtreeIdx,
) -> Chain {
    let mut top = t;
    loop {
        // A free parent is above the chain a cone plan leaves free, over
        // which no source level is.
        match into.node(top).parent().filter(|&p| !is_free(p)) {
            Some(p) if !is_image(p) => top = p,
            Some(p) => {
                return match first_ref(p, ChildSide::of(into, p, top)) {
                    Some(named) if named != ONE_LEAF_IDX => Chain::Literals(top),
                    _ => Chain::One,
                };
            }
            None if output == POS_LEAF_IDX || output == NEG_LEAF_IDX => return Chain::Output(output),
            None => return Chain::One,
        }
    }
}

#[cfg(test)]
#[path = "tests/embed/mod.rs"]
mod tests;
