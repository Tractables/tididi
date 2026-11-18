//! Trim a diagram to the projection of the leaves it names, in place.
//!
//! A leaf is dropped when its parent's pairs name it as `One`, which they do
//! all or none (`test_helpers::check::check_determinism`). A vtree node with
//! no kept leaf under one child is dropped too: every pair of its level names
//! the true node on that side, so each of its nodes is the union of the nodes
//! of the kept side it names, and the unions are disjoint. Down a chain of
//! dropped nodes from its top `c` to the first node `e` with a kept leaf
//! under each child, each node of `e` belongs to at most one node of `c`.
//! `e`'s level takes `c`'s numbering, so that the references above `c` stand
//! as they are: its node words permuted when each node of `c` holds one node
//! of `e`, its pair lists concatenated by owner otherwise. A chain ending at
//! a kept leaf maps each node of its top to the labels it holds, and the
//! level above rewrites its references on that side, the only pass over
//! pairs a kept level takes; a chain ending at a marginal level maps each
//! node of its top to the value reference its one pair holds, and the level
//! above names those values. A top whose nodes hold both labels is true, and
//! its parent is dropped in the same pass. The levels then move to their
//! nodes' indices in the projection.

use std::sync::Arc;

use num_traits::ToPrimitive;

use crate::Engine;
use crate::build::constant_without_leaf;
use crate::diagram::{
    rewrite_stored_side, ChildDecoder, ChildPair, ChildSide, EncodedChildRef, EncodedNode, LevelCounts, NodeIdx, StoreRoom, Tdd, TddLevel, TddNodeId,
    NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ValueRef,
};
use crate::limits::{OperationError, PollGate};
use crate::value::{Count, CountRead, CountVec};
use crate::vtree::{Home, Vtree, VtreeIdx, VtreeNode};

/// A node of `e` that no node of its chain's top holds.
const UNNAMED: u32 = u32::MAX;

/// The labels a node of a chain ending at a leaf holds.
const POS: u8 = 1;
const NEG: u8 = 2;
const BOTH: u8 = POS | NEG;

impl Engine {
    /// Run [`Tdd::trim`](crate::Tdd::trim) using this batch's scratch and
    /// resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::trim`](crate::Tdd::trim).
    pub fn trim(&self, f: &mut Tdd, home: &Home) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        f.require_structure()?;
        if !f.vtree.is_on(home) {
            return Err(OperationError::VtreeMismatch);
        }
        trim(self, f, home, &[])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// No kept leaf under the node.
    True,
    Leaf,
    /// Dropped, its kept side ending at a leaf.
    LeafChain(ChildSide),
    /// Dropped, its kept side ending at a `Both` node.
    LevelChain(ChildSide),
    /// A kept leaf under each child.
    Both,
    /// A region, or a summed-out level that becomes one.
    Region,
}

/// What a bottom-up pass over the vtree finds, reading only the first pair of
/// each kept level and the levels of chains that end at a leaf.
struct Shape {
    kind: Vec<Kind>,
    /// For a `LeafChain` node, the labels each of its nodes holds.
    labels: Vec<Vec<u8>>,
    /// For a `LeafChain` node ending at a marginal level, the reference into
    /// that level each of its nodes holds, in place of `labels`.
    values: Vec<Vec<u32>>,
    /// The leaves under each node, and the kept ones.
    leaves: Vec<u32>,
    kept: Vec<u32>,
    dropped: bool,
    /// A node of a chain ending at a leaf held more than one pair.
    merged: bool,
}

fn side_ref(p: ChildPair, side: ChildSide) -> u32 {
    side.pick(p.left, p.right).raw()
}

fn child(vtree: &Vtree, t: VtreeIdx, side: ChildSide) -> VtreeIdx {
    let (left, right) = vtree.children(t);
    side.pick(left, right)
}

fn label(bits: u8) -> NodeIdx {
    match bits {
        POS => POS_LEAF_IDX,
        NEG => NEG_LEAF_IDX,
        _ => ONE_LEAF_IDX,
    }
}

impl Shape {
    fn bits(&self, vtree: &Vtree, c: VtreeIdx, r: u32) -> u8 {
        if !vtree.node(c).is_leaf() {
            self.labels[c.idx()][r as usize]
        } else if r == POS_LEAF_IDX.0 {
            POS
        } else if r == NEG_LEAF_IDX.0 {
            NEG
        } else {
            BOTH
        }
    }

    /// `summed`: `f` has marginal levels, and its internal ones become
    /// regions.
    fn read(eng: &Engine, f: &Tdd, summed: bool) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let vtree = &*f.vtree;
        let n = vtree.num_nodes();
        let mut shape = Shape {
            kind: Vec::new(),
            labels: Vec::new(),
            values: Vec::new(),
            leaves: Vec::new(),
            kept: Vec::new(),
            dropped: false,
            merged: false,
        };
        lim.try_resize(&mut shape.kind, n, Kind::True)?;
        lim.reserve_exact(&mut shape.labels, n)?;
        shape.labels.resize_with(n, Vec::new);
        if summed {
            lim.reserve_exact(&mut shape.values, n)?;
            shape.values.resize_with(n, Vec::new);
        }
        lim.try_resize(&mut shape.leaves, n, 0)?;
        lim.try_resize(&mut shape.kept, n, 0)?;
        let mut gate = lim.gate();
        for t in vtree.bottomup() {
            let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
                let kind = if summed && vtree.node(t).is_region() { Kind::Region } else { Kind::Leaf };
                (shape.kind[t.idx()], shape.leaves[t.idx()], shape.kept[t.idx()]) = (kind, 1, 1);
                continue;
            };
            gate.poll(1)?;
            let level = &f.levels[t.idx()];
            if summed && level.is_marginal() {
                (shape.kind[t.idx()], shape.kept[t.idx()], shape.dropped) = (Kind::Region, 1, true);
                continue;
            }
            let first = level.pairs_iter_of_idx(0).next().expect("a node holds a pair");
            let marginal = |c: VtreeIdx| summed && f.levels[c.idx()].is_marginal();
            let over_values = |c: VtreeIdx| marginal(c) || shape.values.get(c.idx()).is_some_and(|v| !v.is_empty());
            let keeps = |c: VtreeIdx, side| match shape.kind[c.idx()] {
                Kind::True => false,
                Kind::Leaf | Kind::LeafChain(_) if over_values(c) => true,
                Kind::Leaf | Kind::LeafChain(_) => shape.bits(vtree, c, side_ref(first, side)) != BOTH,
                Kind::LevelChain(_) | Kind::Both | Kind::Region => true,
            };
            let (kl, kr) = (keeps(left, ChildSide::Left), keeps(right, ChildSide::Right));
            let kept_under = |c: VtreeIdx, k: bool| if k { shape.kept[c.idx()] } else { 0 };
            let kept = kept_under(left, kl) + kept_under(right, kr);
            let leaves = shape.leaves[left.idx()] + shape.leaves[right.idx()];
            (shape.kept[t.idx()], shape.leaves[t.idx()]) = (kept, leaves);
            shape.dropped |= !(kl && kr);
            shape.kind[t.idx()] = match (kl, kr) {
                (false, false) => Kind::True,
                (true, true) => Kind::Both,
                _ => {
                    let (c, side) = if kl { (left, ChildSide::Left) } else { (right, ChildSide::Right) };
                    if matches!(shape.kind[c.idx()], Kind::LevelChain(_) | Kind::Both) {
                        Kind::LevelChain(side)
                    } else if over_values(c) {
                        let held = shape.chain_values(eng, f, t, c, side, &mut gate)?;
                        shape.values[t.idx()] = held;
                        Kind::LeafChain(side)
                    } else {
                        let mut held = Vec::new();
                        lim.try_resize(&mut held, level.slot_count(), 0u8)?;
                        for (i, h) in held.iter_mut().enumerate() {
                            let pairs = level.pairs_iter_of_idx(i);
                            gate.poll(pairs.len() as u64)?;
                            shape.merged |= pairs.len() > 1;
                            for p in pairs {
                                *h |= shape.bits(vtree, c, side_ref(p, side));
                            }
                        }
                        shape.labels[t.idx()] = held;
                        Kind::LeafChain(side)
                    }
                }
            };
        }
        gate.flush()?;
        Ok(shape)
    }

    /// For `t` dropped over a kept side `c` that is a marginal level or a
    /// chain ending at one, the reference into that level each node of `t`
    /// stands for. A node of `t` with more than one pair would stand for a
    /// sum of values, which no reference names: refused.
    fn chain_values(
        &self,
        eng: &Engine,
        f: &Tdd,
        t: VtreeIdx,
        c: VtreeIdx,
        side: ChildSide,
        gate: &mut PollGate<'_>,
    ) -> Result<Vec<u32>, OperationError> {
        let level = &f.levels[t.idx()];
        let below = &self.values[c.idx()];
        let mut held = Vec::new();
        eng.limits().reserve_exact(&mut held, level.slot_count())?;
        for i in 0..level.slot_count() {
            let mut pairs = level.pairs_iter_of_idx(i);
            gate.poll(pairs.len() as u64)?;
            let (Some(p), None) = (pairs.next(), pairs.next()) else {
                return Err(OperationError::MarginalLevel(c));
            };
            let r = side_ref(p, side);
            held.push(if below.is_empty() { r } else { below[r as usize] });
        }
        Ok(held)
    }

    /// The first node down `t`'s chain that is not dropped.
    fn end(&self, vtree: &Vtree, mut t: VtreeIdx) -> VtreeIdx {
        while let Kind::LeafChain(side) | Kind::LevelChain(side) = self.kind[t.idx()] {
            t = child(vtree, t, side);
        }
        t
    }
}

/// How the level of an own internal node is had from the levels of `f`.
enum Form {
    /// `e`'s level as it is.
    Moved,
    /// `e`'s node words in the top's numbering: `owner[j]` is the node of the
    /// top that holds node `j` of `e`, each node of the top holding one;
    /// `words` the room for them, one per node of the top.
    Permuted { owner: Vec<u32>, words: Vec<EncodedNode> },
    /// Built by concatenating the pair lists of the nodes of `e` each node
    /// of the top holds.
    Built(TddLevel),
}

struct Plan {
    top: VtreeIdx,
    end: VtreeIdx,
    form: Form,
    /// For each side of `e` toward a chain ending at a leaf, the label each
    /// node of the chain's top stands for.
    sides: [Option<Vec<u32>>; 2],
    /// Room to store an implicit level of `e` that is edited in place.
    room: Option<StoreRoom>,
}

const SIDES: [ChildSide; 2] = [ChildSide::Left, ChildSide::Right];

/// The trim, all of its allocations made before `f` changes, so that a
/// refusal leaves `f` as it was.
///
/// `collapse` lists, ascending, internal marginal levels of `f` not under
/// another; each becomes the region [`Vtree::folded_region`] gives, and its
/// subtree goes. Another internal marginal level is refused. A region stays,
/// a marginal leaf stays.
pub(crate) fn trim(eng: &Engine, f: &mut Tdd, home: &Home, collapse: &[VtreeIdx]) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let vtree = Arc::clone(&f.vtree);
    if vtree.num_nodes() == 0 {
        return Ok(false);
    }
    if f.is_zero() {
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), false));
        return Ok(true);
    }
    lim.check_stop()?;
    let root = vtree.root();
    if vtree.node(root).is_leaf() {
        if f.output.local != ONE_LEAF_IDX || vtree.node(root).is_region() {
            return Ok(false);
        }
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), true));
        return Ok(true);
    }
    let summed = f.has_marginal_level();
    let shape = Shape::read(eng, f, summed)?;
    if !shape.dropped {
        return Ok(false);
    }
    let local = match shape.kind[root.idx()] {
        Kind::True => None,
        Kind::LeafChain(_) if summed && !shape.values[root.idx()].is_empty() => {
            // The output names a slot of a marginal root; an inline value has none.
            let side = EncodedChildRef::from_raw(shape.values[root.idx()][f.output.local.idx()]);
            match ChildDecoder::marginal().value(side) {
                ValueRef::Slot(slot) => Some(NodeIdx(slot)),
                ValueRef::Inline(_) => return Err(OperationError::MarginalLevel(shape.end(&vtree, root))),
            }
        }
        Kind::LeafChain(_) => match shape.labels[root.idx()][f.output.local.idx()] {
            BOTH => None,
            bits => Some(label(bits)),
        },
        _ => Some(f.output.local),
    };
    let Some(local) = local else {
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), true));
        return Ok(true);
    };

    let mut ranks = Vec::new();
    lim.reserve_exact(&mut ranks, shape.kept[root.idx()] as usize)?;
    let mut regions = Vec::new();
    let mut stack = Vec::new();
    lim.try_push(&mut stack, root)?;
    while let Some(t) = stack.pop() {
        match shape.kind[t.idx()] {
            Kind::Leaf => ranks.push(home.rank(vtree.leaf_var(t)).expect("a variable of the home")),
            Kind::Region if vtree.node(t).is_region() => lim.try_push(&mut regions, vtree.region_at(t))?,
            Kind::Region => {
                collapse.binary_search(&t).map_err(|_| OperationError::MarginalLevel(t))?;
                lim.try_push(&mut regions, vtree.folded_region(t))?;
            }
            Kind::LeafChain(side) | Kind::LevelChain(side) => lim.try_push(&mut stack, child(&vtree, t, side))?,
            Kind::Both => {
                let (left, right) = vtree.children(t);
                lim.try_push(&mut stack, left)?;
                lim.try_push(&mut stack, right)?;
            }
            Kind::True => unreachable!("a dropped subtree is not walked"),
        }
    }
    let own = Vtree::project_regions(home, ranks, regions);
    let m = own.num_nodes();

    // Each own node's chain, top and end. `map` takes each node of `f`'s
    // vtree to the own node its worklist entries go to: a node on a chain to
    // the node the chain ends at, a node under a dropped side to the root.
    let mut map = Vec::new();
    lim.try_resize(&mut map, vtree.num_nodes(), own.root())?;
    let mut plans: Vec<Option<Plan>> = Vec::new();
    lim.reserve_exact(&mut plans, m)?;
    plans.resize_with(m, || None);
    let mut placed = Vec::new();
    let mut walk = Vec::new();
    lim.try_push(&mut walk, (root, own.root()))?;
    while let Some((top, u)) = walk.pop() {
        let end = shape.end(&vtree, top);
        // A dropped level above a leaf owes its work to the level whose
        // references it became.
        let above = match shape.kind[end.idx()] {
            Kind::Leaf => own.node(u).parent().unwrap_or(u),
            _ => u,
        };
        let mut t = top;
        loop {
            map[t.idx()] = if t == end { u } else { above };
            if t == end {
                break;
            }
            t = match shape.kind[t.idx()] {
                Kind::LeafChain(side) | Kind::LevelChain(side) => child(&vtree, t, side),
                _ => unreachable!("a chain ends at its end"),
            };
        }
        match shape.kind[end.idx()] {
            Kind::Leaf => {
                debug_assert_eq!(own.home_node(u), vtree.home_node(end), "own node {} stands for node {}", u.0, end.0);
                continue;
            }
            Kind::Region => {
                lim.try_push(&mut placed, (u, end))?;
                continue;
            }
            _ => debug_assert_eq!(own.home_node(u), vtree.home_node(end), "own node {} stands for node {}", u.0, end.0),
        }
        let (el, er) = vtree.children(end);
        let (ul, ur) = own.children(u);
        lim.try_push(&mut walk, (el, ul))?;
        lim.try_push(&mut walk, (er, ur))?;
        plans[u.idx()] = Some(plan(eng, f, &shape, top, end)?);
    }
    lim.discard(walk);

    let counts = match f.levels.counts() {
        Some(kept) if !summed => Some(own_counts(eng, kept, &shape, &plans, m)?),
        _ => None,
    };
    let mut moved = Vec::new();
    if summed && f.weights.is_some() {
        lim.try_resize(&mut moved, vtree.num_nodes(), UNNAMED)?;
    }
    let mut levels = Vec::new();
    lim.try_resize(&mut levels, m, TddLevel::default())?;
    let mut changed = Vec::new();
    lim.reserve_exact(&mut changed, m)?;
    let mut owes = Vec::new();
    lim.reserve_exact(&mut owes, m)?;
    f.dirty.reserve_all(m);

    // Nothing below is refused.
    let canonical = f.levels.is_canonical(f.output);
    let closed = f.levels.is_closed();
    let mut built = false;
    let mut old = std::mem::take(&mut f.levels).into_vec();
    for (u, slot) in plans.iter_mut().enumerate() {
        let Some(Plan { end, form, sides, room, .. }) = slot.take() else {
            continue;
        };
        let (mut level, form) = match form {
            Form::Built(level) => {
                built = true;
                changed.push(VtreeIdx(u as u32));
                owes.push(VtreeIdx(u as u32));
                levels[u] = level;
                continue;
            }
            form => (std::mem::take(&mut old[end.idx()]), form),
        };
        if let Some(to) = moved.get_mut(end.idx()) {
            *to = u as u32;
        }
        let (mut edited, mut owed) = (false, false);
        if let Some(room) = room {
            level.store_moved(room, |_| true, |l| l, |r| r);
        }
        for (side, map) in SIDES.into_iter().zip(&sides) {
            if let Some(map) = map {
                rewrite_stored_side(&mut level, side, |r| *r = map[*r as usize]);
                (edited, owed) = (true, true);
            }
        }
        if let Form::Permuted { owner, mut words } = form {
            let mut dead = 0;
            for (j, &o) in owner.iter().enumerate() {
                match o {
                    UNNAMED => {
                        dead += level.arena_pairs_at(j);
                        owed = true;
                    }
                    _ => words[o as usize] = level.node(j),
                }
            }
            *level.nodes.stored_mut() = words;
            level.note_dead_pairs(dead);
            edited = true;
        }
        levels[u] = level;
        if edited {
            changed.push(VtreeIdx(u as u32));
        }
        if owed {
            owes.push(VtreeIdx(u as u32));
        }
    }
    for u in (0..m).map(|u| VtreeIdx(u as u32)) {
        if own.node(u).is_leaf() && !(summed && own.node(u).is_region()) {
            let t = vtree.leaf_of(own.leaf_var(u)).expect("an own leaf");
            levels[u.idx()] = std::mem::take(&mut old[t.idx()]);
            if let Some(to) = moved.get_mut(t.idx()) {
                *to = u.0;
            }
        }
    }
    for &(u, t) in &placed {
        levels[u.idx()] = std::mem::take(&mut old[t.idx()]);
        if let Some(to) = moved.get_mut(t.idx()) {
            *to = u.0;
        }
    }
    crate::diagram::return_levels(eng, old);
    if let Some(ws) = f.weights.as_mut().filter(|_| !moved.is_empty()) {
        ws.rekey_levels(|t| Some(moved[t]).filter(|&u| u != UNNAMED).map(|u| u as usize));
    }

    if m == 1 {
        f.dirty = Default::default();
    } else {
        f.dirty.remap_keeping(&map, |t| shape.kind[t] == Kind::Both);
        f.dirty.dedup_above(m);
    }
    f.vtree = own;
    f.levels = levels.into();
    f.output = TddNodeId { vtree: f.vtree.root(), local };
    for &u in &changed {
        f.levels.mark_changed(u);
    }
    for &u in &owes {
        f.invalidate(u);
    }
    f.close_marked_levels(closed);
    if canonical && !built && !shape.merged {
        f.levels.certify(f.output);
    }
    if let Some(counts) = counts {
        f.levels.keep_counts(counts);
    }
    Ok(true)
}

/// The plan of the own internal node whose chain runs from `top` to `end`.
fn plan(eng: &Engine, f: &Tdd, shape: &Shape, top: VtreeIdx, end: VtreeIdx) -> Result<Plan, OperationError> {
    let lim = eng.limits();
    let vtree = &*f.vtree;
    let mut sides = [None, None];
    for (side, slot) in SIDES.into_iter().zip(&mut sides) {
        let c = child(vtree, end, side);
        if let Kind::LeafChain(_) = shape.kind[c.idx()] {
            let mut map = Vec::new();
            match shape.values.get(c.idx()).filter(|v| !v.is_empty()) {
                Some(values) => {
                    lim.reserve_exact(&mut map, values.len())?;
                    map.extend_from_slice(values);
                }
                None => {
                    lim.reserve_exact(&mut map, shape.labels[c.idx()].len())?;
                    map.extend(shape.labels[c.idx()].iter().map(|&bits| label(bits).0));
                }
            }
            *slot = Some(map);
        }
    }
    let level = &f.levels[end.idx()];
    let form = if top == end { Form::Moved } else { merge(eng, f, shape, top, end, &sides)? };
    let room = match (&form, level.implicit()) {
        (Form::Built(_), _) | (_, None) => None,
        (Form::Moved, Some(_)) if sides.iter().all(Option::is_none) => None,
        _ => Some(level.store_room(lim)?),
    };
    Ok(Plan { top, end, form, sides, room })
}

/// How `end`'s level takes `top`'s numbering.
fn merge(
    eng: &Engine,
    f: &Tdd,
    shape: &Shape,
    top: VtreeIdx,
    end: VtreeIdx,
    sides: &[Option<Vec<u32>>; 2],
) -> Result<Form, OperationError> {
    let lim = eng.limits();
    let vtree = &*f.vtree;
    let width = f.levels[top.idx()].slot_count();
    let mut gate = lim.gate();
    let mut owner = Vec::new();
    lim.reserve_exact(&mut owner, width)?;
    owner.extend(0..width as u32);
    let mut t = top;
    while t != end {
        let Kind::LevelChain(side) = shape.kind[t.idx()] else { unreachable!("a level chain") };
        let c = child(vtree, t, side);
        let mut below = Vec::new();
        lim.try_resize(&mut below, f.levels[c.idx()].slot_count(), UNNAMED)?;
        let level = &f.levels[t.idx()];
        for (i, &o) in owner.iter().enumerate() {
            if o != UNNAMED {
                let pairs = level.pairs_iter_of_idx(i);
                gate.poll(pairs.len() as u64)?;
                for p in pairs {
                    below[side_ref(p, side) as usize] = o;
                }
            }
        }
        lim.discard(std::mem::replace(&mut owner, below));
        t = c;
    }
    gate.flush()?;
    let mut held = Vec::new();
    lim.try_resize(&mut held, width + 1, 0u32)?;
    for &o in &owner {
        if o != UNNAMED {
            held[o as usize + 1] += 1;
        }
    }
    if held[1..].iter().all(|&h| h == 1) {
        if width == owner.len() && owner.iter().enumerate().all(|(j, &o)| o == j as u32) {
            return Ok(Form::Moved);
        }
        let mut words = Vec::new();
        lim.try_resize(&mut words, width, EncodedNode::multi_pair(0, 0))?;
        return Ok(Form::Permuted { owner, words });
    }
    // `held` becomes the start of each owner's members.
    for i in 0..width {
        held[i + 1] += held[i];
    }
    let mut members = Vec::new();
    lim.try_resize(&mut members, held[width] as usize, 0u32)?;
    let mut next = held.clone();
    for (j, &o) in owner.iter().enumerate() {
        if o != UNNAMED {
            members[next[o as usize] as usize] = j as u32;
            next[o as usize] += 1;
        }
    }
    let source = &f.levels[end.idx()];
    let side = |map: &Option<Vec<u32>>, r: crate::diagram::EncodedChildRef| match map {
        Some(map) => NodeIdx(map[r.raw() as usize]).into(),
        None => r,
    };
    let mut level = TddLevel::new();
    let mut pairs = Vec::new();
    for i in 0..width {
        pairs.clear();
        for &j in &members[held[i] as usize..held[i + 1] as usize] {
            for p in source.pairs_iter_of_idx(j as usize) {
                lim.try_push(&mut pairs, ChildPair::new(side(&sides[0], p.left), side(&sides[1], p.right)))?;
            }
        }
        gate.poll(pairs.len() as u64)?;
        level.push_node(lim, &pairs)?;
    }
    gate.flush()?;
    Ok(Form::Built(level))
}

/// The level counts `kept` holds, moved to the own nodes and divided by two
/// for each dropped leaf under a level, which its nodes do not depend on.
fn own_counts(
    eng: &Engine,
    kept: &LevelCounts,
    shape: &Shape,
    plans: &[Option<Plan>],
    m: usize,
) -> Result<LevelCounts, OperationError> {
    let mut out = LevelCounts::none(eng, m)?;
    for (u, plan) in plans.iter().enumerate() {
        let Some(Plan { top, .. }) = plan else { continue };
        let Some(column) = kept.column(*top) else { continue };
        let dropped = (shape.leaves[top.idx()] - shape.kept[top.idx()]) as usize;
        let column = match dropped {
            0 => Arc::clone(column),
            _ => {
                let mut halved = CountVec::try_with_capacity(eng, column.len())?;
                for i in 0..column.len() {
                    let count = match column.get(i) {
                        CountRead::Fast(v) => Count::from_u128(v.checked_shr(dropped as u32).unwrap_or(0)),
                        CountRead::Big(v) => {
                            let v = v >> dropped;
                            v.to_u128().map_or(Count::Big(v), Count::from_u128)
                        }
                    };
                    halved.push(eng, count)?;
                }
                Arc::new(halved)
            }
        };
        out.set(VtreeIdx(u as u32), column);
    }
    Ok(out)
}

fn replace(eng: &Engine, f: &mut Tdd, mut by: Tdd) {
    if by.weights.is_none() {
        by.weights = f.weights.take();
    }
    let old = std::mem::replace(f, by);
    crate::diagram::return_levels(eng, old.levels.into_vec());
}
