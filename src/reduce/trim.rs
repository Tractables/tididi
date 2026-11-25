//! Trim a diagram to the projection of the leaves it names, in place.
//!
//! A leaf is dropped when its parent's pairs name it as `One`, which they do
//! all or none (`test_helpers::check::check_determinism`). A vtree node with
//! no kept leaf under one child is dropped too: every pair of its level names
//! the true node on that side, so each of its nodes is the sum of the nodes
//! of the kept side it names, a node named twice counted twice where a
//! summed-out level is under it and once elsewhere. Down a chain of dropped
//! nodes from its top `c` to the first node `e` with a kept leaf under each
//! child, `e`'s level takes `c`'s numbering, so that the references above
//! `c` stand as they are: its node words permuted when each node of `c`
//! holds one node of `e` and no two hold the same one, otherwise its pair
//! lists concatenated by owner, once per time the owner holds them. A
//! chain ending at
//! a kept leaf maps each node of its top to the labels it holds, and the
//! level above rewrites its references on that side, the only pass over
//! pairs a kept level takes; a chain ending at a marginal level maps each
//! node of its top to the value reference its one pair holds, and the level
//! above names those values. Where a node of the chain holds more than one
//! pair, it stands for the sum of their values: the marginal level is
//! rebuilt with one value per node of the chain's level, added in the
//! diagram's algebra. A top whose nodes hold both labels is true, and
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
use rustc_hash::FxHashMap;

use crate::diagram::{weight_key, MarginalStorage, WeightKey};
use crate::limits::{OperationError, PollGate};
use crate::value::slots::{push_count_slot, sum_marginal_counts};
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
        if !f.vtree.is_on(home) {
            return Err(OperationError::VtreeMismatch);
        }
        trim(self, f, home, None, &[])
    }
}

/// The closing step of a reduction on a diagram on a home: false, or a leaf
/// with output `One`, becomes the constant on the empty projection, and the
/// leaves of `dropped` leave as [`Tdd::trim`] drops leaves named true. A leaf
/// of `dropped` is named true in every pair of its parent, or by a literal in
/// every pair, each node holding `(Pos, S)` exactly when it holds `(Neg, S)`.
/// Another leaf goes only with a sibling whose chain ends at it holding both
/// labels. With a summed-out level `dropped` holds at most the caller's seed
/// and a diagram whose summed levels are regions or leaves is trimmed by
/// detection instead; the smooth form, with an internal summed level, stays.
/// Elsewhere a false `f` has its levels emptied. `marginal` is whether `f`
/// has a marginal level, where the caller knows it: the close before a
/// settle forgets the cached answer, and reading it scans every level.
///
/// # Errors
///
/// As [`Tdd::trim`]; `f` is then as it was.
#[inline]
pub(crate) fn settle(
    eng: &Engine,
    f: &mut Tdd,
    dropped: &mut Vec<VtreeIdx>,
    marginal: Option<bool>,
) -> Result<(), OperationError> {
    if !f.vtree.on_home() || dropped.is_empty() && !marginal.unwrap_or_else(|| f.has_marginal_level()) {
        close_constant(eng, f);
        return Ok(());
    }
    settle_cold(eng, f, dropped)
}

/// [`settle`] with no leaf dropped, which is never refused. A false `f` it
/// does not replace has its levels emptied.
#[inline]
pub(crate) fn close_constant(eng: &Engine, f: &mut Tdd) {
    if f.vtree.num_nodes() != 0 && (f.is_zero() || f.is_one_leaf_one()) {
        close_constant_cold(eng, f);
    }
}

#[cold]
#[inline(never)]
fn close_constant_cold(eng: &Engine, f: &mut Tdd) {
    let one = !f.is_zero();
    let kept = one && f.vtree.node(f.vtree.root()).is_region();
    match f.vtree.home() {
        Some(home) if !kept => {
            replace(eng, f, constant_without_leaf(&Vtree::empty_projection(&home), one));
        }
        _ if !one => super::prune::clear_false(f),
        _ => {}
    }
}

#[cold]
#[inline(never)]
fn settle_cold(eng: &Engine, f: &mut Tdd, dropped: &mut Vec<VtreeIdx>) -> Result<(), OperationError> {
    let Some(home) = f.vtree.home() else {
        close_constant(eng, f);
        return Ok(());
    };
    if f.has_marginal_level() {
        // A summed internal level is the smooth form `marginalize_levels`
        // leaves on the home, which stays as it is.
        if f.vtree.bottomup().any(|t| !f.vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal()) {
            return Ok(());
        }
        return trim(eng, f, &home, None, &[]).map(|_| ());
    }
    dropped.sort_unstable();
    dropped.dedup();
    trim(eng, f, &home, Some(dropped), &[]).map(|_| ())
}

/// [`trim`] by detection of a diagram on `home` with a summed-out level,
/// each internal summed level not under another becoming a region.
#[cold]
#[inline(never)]
pub(crate) fn trim_summed(eng: &Engine, f: &mut Tdd, home: &Home) -> Result<bool, OperationError> {
    let vtree = Arc::clone(&f.vtree);
    let summed = |t: VtreeIdx| !vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal();
    let mut tops = Vec::new();
    for t in vtree.bottomup() {
        if summed(t) && vtree.node(t).parent().is_none_or(|p| !summed(p)) {
            eng.limits().try_push(&mut tops, t)?;
        }
    }
    tops.sort_unstable();
    trim(eng, f, home, None, &tops)
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
    /// Marginal levels rebuilt with the sums a chain above them holds, by the
    /// node of `f` whose level each replaces; `values` refer into these.
    regrouped: Vec<(VtreeIdx, TddLevel)>,
    /// The leaves under each node, and the kept ones.
    leaves: Vec<u32>,
    kept: Vec<u32>,
    /// With a summed-out level, whether a kept one is under each node: a
    /// node there may hold a pair twice.
    adds: Vec<bool>,
    dropped: bool,
    /// A node of a chain ending at a leaf held more than one pair.
    merged: bool,
    /// A dropped node stands for a multiple of a label or of true, which
    /// only a region dropped as full carried.
    multiplied: bool,
}

fn side_ref(p: ChildPair, side: ChildSide) -> u32 {
    side.pick(p.left, p.right).raw()
}

fn child(vtree: &Vtree, t: VtreeIdx, side: ChildSide) -> VtreeIdx {
    let (left, right) = vtree.children(t);
    side.pick(left, right)
}

fn other(side: ChildSide) -> ChildSide {
    side.pick(ChildSide::Right, ChildSide::Left)
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

    /// `dropped`, sorted, names the leaves to drop; without it a leaf is
    /// dropped when the first pair of its parent names it `One`.
    /// `summed`: `f` has marginal levels, and its internal ones become
    /// regions. `keep_full`: no region is dropped as full.
    fn read(eng: &Engine, f: &Tdd, dropped: Option<&[VtreeIdx]>, summed: bool, keep_full: bool) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let vtree = &*f.vtree;
        let n = vtree.num_nodes();
        let mut shape = Shape {
            kind: Vec::new(),
            labels: Vec::new(),
            values: Vec::new(),
            regrouped: Vec::new(),
            leaves: Vec::new(),
            kept: Vec::new(),
            adds: Vec::new(),
            dropped: false,
            merged: false,
            multiplied: false,
        };
        lim.try_resize(&mut shape.kind, n, Kind::True)?;
        lim.reserve_exact(&mut shape.labels, n)?;
        shape.labels.resize_with(n, Vec::new);
        if summed {
            lim.reserve_exact(&mut shape.values, n)?;
            shape.values.resize_with(n, Vec::new);
            lim.try_resize(&mut shape.adds, n, false)?;
        }
        lim.try_resize(&mut shape.leaves, n, 0)?;
        lim.try_resize(&mut shape.kept, n, 0)?;
        // With a summed-out level, whether a region dropped as full is under
        // each node.
        let mut full_under = Vec::new();
        if summed {
            lim.try_resize(&mut full_under, n, false)?;
        }
        let mut gate = lim.gate();
        for t in vtree.bottomup() {
            let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
                // A region whose level is not summed out is one an operand
                // was placed beside (`Free::Operand`), named true by every
                // pair: a leaf the trim drops.
                let marginal = summed && f.levels[t.idx()].is_marginal();
                let kind = if marginal { Kind::Region } else { Kind::Leaf };
                (shape.kind[t.idx()], shape.leaves[t.idx()], shape.kept[t.idx()]) = (kind, 1, 1);
                if summed {
                    shape.adds[t.idx()] = marginal;
                    shape.dropped |= marginal && vtree.node(t).is_var_leaf();
                }
                continue;
            };
            gate.poll(1)?;
            let level = &f.levels[t.idx()];
            if summed && level.is_marginal() {
                (shape.kind[t.idx()], shape.kept[t.idx()], shape.dropped) = (Kind::Region, 1, true);
                shape.adds[t.idx()] = true;
                continue;
            }
            let first = level.pairs_iter_of_idx(0).next().expect("a node holds a pair");
            let marginal = |c: VtreeIdx| summed && f.levels[c.idx()].is_marginal();
            let over_values = |c: VtreeIdx| marginal(c) || shape.values.get(c.idx()).is_some_and(|v| !v.is_empty());
            // A region whose sibling stands for values is kept: dropping it
            // would leave nodes that stand for sums of values.
            let mut full = [false; 2];
            if summed && !keep_full {
                for (slot, (c, other)) in full.iter_mut().zip([(left, right), (right, left)]) {
                    *slot = vtree.node(c).is_region() && !over_values(other) && names_only_full_counts(f, c, &mut gate)?;
                }
            }
            let keeps = |c: VtreeIdx, side| match shape.kind[c.idx()] {
                Kind::True => false,
                Kind::Leaf | Kind::LeafChain(_) if over_values(c) => true,
                Kind::Leaf => match dropped {
                    Some(dropped) => dropped.binary_search(&c).is_err(),
                    None => shape.bits(vtree, c, side_ref(first, side)) != BOTH,
                },
                Kind::LeafChain(_) => shape.bits(vtree, c, side_ref(first, side)) != BOTH,
                Kind::LevelChain(_) | Kind::Both => true,
                Kind::Region => !full[usize::from(side == ChildSide::Right)],
            };
            let (kl, kr) = (keeps(left, ChildSide::Left), keeps(right, ChildSide::Right));
            let kept_under = |c: VtreeIdx, k: bool| if k { shape.kept[c.idx()] } else { 0 };
            let kept = kept_under(left, kl) + kept_under(right, kr);
            let leaves = shape.leaves[left.idx()] + shape.leaves[right.idx()];
            (shape.kept[t.idx()], shape.leaves[t.idx()]) = (kept, leaves);
            if summed {
                shape.adds[t.idx()] = (kl && shape.adds[left.idx()]) || (kr && shape.adds[right.idx()]);
                full_under[t.idx()] = full[0] || full[1] || full_under[left.idx()] || full_under[right.idx()];
            }
            shape.dropped |= !(kl && kr);
            shape.kind[t.idx()] = match (kl, kr) {
                (false, false) => {
                    // A node of a level whose subtree goes is true only with
                    // one pair; more stand for a multiple a full region held.
                    if summed && full_under[t.idx()] && (0..level.slot_count()).any(|i| level.pairs_iter_of_idx(i).len() > 1) {
                        shape.multiplied = true;
                    }
                    Kind::True
                }
                (true, true) => Kind::Both,
                _ => {
                    let (c, side) = if kl { (left, ChildSide::Left) } else { (right, ChildSide::Right) };
                    let values = over_values(c);
                    if matches!(shape.kind[c.idx()], Kind::LevelChain(_) | Kind::Both) {
                        Kind::LevelChain(side)
                    } else if values {
                        let held = shape.chain_values(eng, f, t, c, side, &mut gate)?;
                        shape.values[t.idx()] = held;
                        Kind::LeafChain(side)
                    } else {
                        let twin = negative_half(vtree, t, side);
                        let mut held = Vec::new();
                        lim.try_resize(&mut held, level.slot_count(), 0u8)?;
                        for (i, h) in held.iter_mut().enumerate() {
                            let pairs = level.pairs_iter_of_idx(i);
                            gate.poll(pairs.len() as u64)?;
                            let mut read = 0;
                            for p in pairs.filter(|&p| Some(side_ref(p, other(side))) != twin) {
                                read += 1;
                                let bits = shape.bits(vtree, c, side_ref(p, side));
                                // With every region kept, a level over a chain
                                // to a leaf has no summed-out level under it
                                // and holds no label twice.
                                shape.multiplied |= summed && !keep_full && *h & bits != 0;
                                *h |= bits;
                            }
                            shape.merged |= read > 1;
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
    /// stands for. A node of `t` with other than one pair stands for the sum
    /// of their values, which [`Shape::regroup`] mints.
    fn chain_values(
        &mut self,
        eng: &Engine,
        f: &Tdd,
        t: VtreeIdx,
        c: VtreeIdx,
        side: ChildSide,
        gate: &mut PollGate<'_>,
    ) -> Result<Vec<u32>, OperationError> {
        let level = &f.levels[t.idx()];
        let twin = negative_half(&f.vtree, t, side);
        let below = &self.values[c.idx()];
        let mut held = Vec::new();
        eng.limits().reserve_exact(&mut held, level.slot_count())?;
        for i in 0..level.slot_count() {
            let pairs = level.pairs_iter_of_idx(i);
            gate.poll(pairs.len() as u64)?;
            let mut pairs = pairs.filter(|&p| Some(side_ref(p, other(side))) != twin);
            let (Some(p), None) = (pairs.next(), pairs.next()) else {
                held.clear();
                self.regroup(eng, f, t, c, side, &mut held, gate)?;
                return Ok(held);
            };
            let r = side_ref(p, side);
            held.push(if below.is_empty() { r } else { below[r as usize] });
        }
        Ok(held)
    }

    /// The level at the end of `c`'s chain, rebuilt with one value per node
    /// of `t`: the sum over the node's pairs of the values they reach, in the
    /// diagram's algebra. Equal values share a slot, and a small count rides
    /// inline. `held` receives the references.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn regroup(
        &mut self,
        eng: &Engine,
        f: &Tdd,
        t: VtreeIdx,
        c: VtreeIdx,
        side: ChildSide,
        held: &mut Vec<u32>,
        gate: &mut PollGate<'_>,
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        let vtree = &*f.vtree;
        let end = self.end(vtree, c);
        let at = self.regrouped.iter().position(|&(e, _)| e == end);
        let source = at.map_or(&f.levels[end.idx()], |k| &self.regrouped[k].1);
        let (level, below) = (&f.levels[t.idx()], &self.values[c.idx()]);
        let twin = negative_half(vtree, t, side);
        let mut refs = Vec::new();
        let mut node_refs = |i: usize, refs: &mut Vec<u32>| -> Result<(), OperationError> {
            refs.clear();
            for p in level.pairs_iter_of_idx(i).filter(|&p| Some(side_ref(p, other(side))) != twin) {
                let r = side_ref(p, side);
                lim.try_push(refs, if below.is_empty() { r } else { below[r as usize] })?;
            }
            gate.poll(refs.len() as u64)
        };
        let mut out = TddLevel::new();
        if let Some(counts) = source.count_column() {
            out.become_marginal(Vec::new(), None);
            let mut slots: FxHashMap<Count, u32> = FxHashMap::default();
            for i in 0..level.slot_count() {
                node_refs(i, &mut refs)?;
                let sum = sum_marginal_counts(counts, &refs);
                let inline = match sum {
                    Count::Fast(v) => ValueRef::inline_raw(v),
                    Count::Big(_) => None,
                };
                let r = match inline {
                    Some(r) => r,
                    None => match slots.get(&sum) {
                        Some(&slot) => ValueRef::slot_raw(slot),
                        None => {
                            lim.reserve_map(&mut slots, 1)?;
                            let slot = push_count_slot(eng, &mut out, sum.clone())?;
                            slots.insert(sum, slot);
                            ValueRef::slot_raw(slot)
                        }
                    },
                };
                lim.try_push(held, r)?;
            }
        } else {
            let values = source.weight_values().expect("a summed level holds counts or weights");
            out.become_marginal_weighted(Vec::new());
            let ws = f.weight_store();
            let mut slots: FxHashMap<WeightKey, u32> = FxHashMap::default();
            for i in 0..level.slot_count() {
                node_refs(i, &mut refs)?;
                let mut sum = ws.wzero();
                for &raw in &refs {
                    let r = EncodedChildRef::from_raw(raw);
                    if !r.is_reserved() {
                        sum.add_assign(&values[ChildDecoder::marginal().index(r)]);
                    }
                }
                let key = weight_key(&sum);
                let slot = match slots.get(&key) {
                    Some(&slot) => slot,
                    None => {
                        lim.reserve_map(&mut slots, 1)?;
                        let slot = MarginalStorage::new(&mut out).push_weight(eng, sum)?;
                        slots.insert(key, slot);
                        slot
                    }
                };
                lim.try_push(held, ValueRef::slot_raw(slot))?;
            }
        }
        match at {
            Some(k) => self.regrouped[k].1 = out,
            None => lim.try_push(&mut self.regrouped, (end, out))?,
        }
        Ok(())
    }

    /// The first node down `t`'s chain that is not dropped.
    fn end(&self, vtree: &Vtree, mut t: VtreeIdx) -> VtreeIdx {
        while let Kind::LeafChain(side) | Kind::LevelChain(side) = self.kind[t.idx()] {
            t = child(vtree, t, side);
        }
        t
    }
}

/// Whether every reference into integer region `t` from its parent's pairs
/// names 2^|extent|: the region is then true over its extent and is dropped,
/// its variables free again. A weighted region is never dropped: with signed
/// weights its value says nothing of the function.
fn names_only_full_counts(f: &Tdd, t: VtreeIdx, gate: &mut PollGate<'_>) -> Result<bool, OperationError> {
    let (level, vtree) = (&f.levels[t.idx()], &*f.vtree);
    let (Some(counts), Some(parent)) = (level.count_column(), vtree.node(t).parent()) else { return Ok(false) };
    let width: u32 = vtree.region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum();
    let full_slot = |slot: u32| match counts.get(slot as usize) {
        CountRead::Fast(c) => width < 128 && c == 1u128 << width,
        CountRead::Big(b) => *b == num_bigint::BigUint::from(1u32) << width as usize,
    };
    let side = if vtree.children(parent).0 == t { ChildSide::Left } else { ChildSide::Right };
    let (decoder, above) = (level.child_decoder(), &f.levels[parent.idx()]);
    for i in 0..above.slot_count() {
        let pairs = above.pairs_iter_of_idx(i);
        gate.poll(pairs.len() as u64)?;
        for p in pairs {
            let full = match decoder.value(EncodedChildRef::from_raw(side_ref(p, side))) {
                ValueRef::Inline(v) => width < 32 && v == 1 << width,
                ValueRef::Slot(slot) => full_slot(slot),
            };
            if !full {
                return Ok(false);
            }
        }
    }
    Ok(true)
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

/// The reference on the dropped side of `t`, kept on `side`, that a pair of
/// a level recorded by leaf contraction holds besides its positive twin:
/// `Neg` when that side is a leaf. Such a level is read on its positive half.
fn negative_half(vtree: &Vtree, t: VtreeIdx, side: ChildSide) -> Option<u32> {
    vtree.node(child(vtree, t, other(side))).is_leaf().then_some(NEG_LEAF_IDX.0)
}

/// The trim, all of its allocations made before `f` changes, so that a
/// refusal leaves `f` as it was.
///
/// `collapse` lists, ascending, internal marginal levels of `f` not under
/// another; each becomes the region [`Vtree::folded_region`] gives, and its
/// subtree goes. Another internal marginal level is refused. A region stays;
/// a marginal variable leaf becomes the region of its variable.
/// `dropped` as [`Shape::read`].
pub(crate) fn trim(eng: &Engine, f: &mut Tdd, home: &Home, dropped: Option<&[VtreeIdx]>, collapse: &[VtreeIdx]) -> Result<bool, OperationError> {
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
        if f.output.local != ONE_LEAF_IDX || vtree.node(root).is_region() || f.levels[root.idx()].is_marginal() {
            return Ok(false);
        }
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), true));
        return Ok(true);
    }
    let summed = f.has_marginal_level();
    // A dropped node that holds what it keeps more than once stands for a
    // multiple of it. Above a summed-out level the chain's end holds the
    // multiple as a repeated pair; elsewhere only a full region the drop
    // would take carried it, and the trim is planned again with every
    // region kept.
    let mut keep_full = false;
    let (mut shape, local, own, map, mut plans, placed) = 'read: loop {
        let mut shape = Shape::read(eng, f, dropped, summed, keep_full)?;
        if shape.multiplied {
            keep_full = true;
            continue;
        }
        if !shape.dropped {
            return Ok(false);
        }
        let local = match shape.kind[root.idx()] {
            Kind::True => None,
            Kind::LeafChain(_) if summed && !shape.values[root.idx()].is_empty() => {
                // The output names a slot of a marginal root; an inline count
                // gets a level of one slot.
                let side = EncodedChildRef::from_raw(shape.values[root.idx()][f.output.local.idx()]);
                match ChildDecoder::marginal().value(side) {
                    ValueRef::Slot(slot) => Some(NodeIdx(slot)),
                    ValueRef::Inline(count) => {
                        let end = shape.end(&vtree, root);
                        let mut level = TddLevel::new();
                        level.become_marginal(Vec::new(), None);
                        let slot = push_count_slot(eng, &mut level, Count::from_u128(u128::from(count)))?;
                        shape.regrouped.retain(|&(e, _)| e != end);
                        lim.try_push(&mut shape.regrouped, (end, level))?;
                        Some(NodeIdx(slot))
                    }
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
                Kind::Region if vtree.node(t).is_leaf() => lim.try_push(&mut regions, vtree.folded_region(t))?,
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
            let Some(plan) = plan(eng, f, &shape, top, end, keep_full)? else {
                keep_full = true;
                continue 'read;
            };
            plans[u.idx()] = Some(plan);
        }
        lim.discard(walk);
        break (shape, local, own, map, plans, placed);
    };
    let m = own.num_nodes();

    let counts = match f.levels.counts() {
        Some(kept) if !summed => Some(own_counts(eng, kept, &shape, &plans, m)?),
        _ => None,
    };
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
        }
    }
    for &(u, t) in &placed {
        levels[u.idx()] = match shape.regrouped.iter().position(|&(e, _)| e == t) {
            Some(k) => {
                changed.push(u);
                std::mem::take(&mut shape.regrouped[k].1)
            }
            None => std::mem::take(&mut old[t.idx()]),
        };
    }
    crate::diagram::return_levels(eng, old);

    if m == 1 {
        f.dirty = Default::default();
    } else {
        f.dirty.remap_keeping(&map, |t| shape.kind[t] == Kind::Both);
        if summed {
            f.dirty.drop_leaves(&own);
        }
        f.dirty.dedup_above(m);
    }
    f.vtree = own;
    f.levels = levels.into();
    f.levels.hold_counts(closed);
    f.output = TddNodeId { vtree: f.vtree.root(), local };
    for &u in &changed {
        f.levels.mark_changed(u);
    }
    for &u in &owes {
        f.invalidate(u);
    }
    if m == 1 {
        f.dirty.set_loose(Some(Vec::new()));
    }
    f.close_marked_levels();
    if canonical && !built && !shape.merged && shape.regrouped.is_empty() {
        f.levels.certify(f.output);
        f.dirty.set_loose(Some(Vec::new()));
        f.dirty.clear_lists();
        #[cfg(debug_assertions)]
        {
            super::contract::contract_leaf::assert_no_full_twin_side(f);
            super::prune::debug_assert_all_reached(f);
        }
    }
    if let Some(counts) = counts {
        f.levels.keep_counts(counts);
    }
    Ok(true)
}

/// The plan of the own internal node whose chain runs from `top` to `end`;
/// `None` when a node of `top` stands for a multiple of a node of `end` that
/// `end`'s level cannot hold and `keep_full` is not set.
fn plan(eng: &Engine, f: &Tdd, shape: &Shape, top: VtreeIdx, end: VtreeIdx, keep_full: bool) -> Result<Option<Plan>, OperationError> {
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
    let form = match top == end {
        true => Form::Moved,
        false => match merge(eng, f, shape, top, end, &sides, keep_full)? {
            Some(form) => form,
            None => return Ok(None),
        },
    };
    let room = match (&form, level.implicit()) {
        (Form::Built(_), _) | (_, None) => None,
        (Form::Moved, Some(_)) if sides.iter().all(Option::is_none) => None,
        _ => Some(level.store_room(lim)?),
    };
    Ok(Some(Plan { top, end, form, sides, room }))
}

/// How `end`'s level takes `top`'s numbering, as [`plan`].
fn merge(
    eng: &Engine,
    f: &Tdd,
    shape: &Shape,
    top: VtreeIdx,
    end: VtreeIdx,
    sides: &[Option<Vec<u32>>; 2],
    keep_full: bool,
) -> Result<Option<Form>, OperationError> {
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
        let twin = negative_half(vtree, t, side);
        let mut below = Vec::new();
        lim.try_resize(&mut below, f.levels[c.idx()].slot_count(), UNNAMED)?;
        let level = &f.levels[t.idx()];
        for (i, &o) in owner.iter().enumerate() {
            if o != UNNAMED {
                let pairs = level.pairs_iter_of_idx(i);
                gate.poll(pairs.len() as u64)?;
                for p in pairs.filter(|&p| Some(side_ref(p, other(side))) != twin) {
                    let named = &mut below[side_ref(p, side) as usize];
                    if *named != UNNAMED {
                        gate.flush()?;
                        return merge_counted(eng, f, shape, top, end, sides, keep_full);
                    }
                    *named = o;
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
            return Ok(Some(Form::Moved));
        }
        let mut words = Vec::new();
        lim.try_resize(&mut words, width, EncodedNode::multi_pair(0, 0))?;
        return Ok(Some(Form::Permuted { owner, words }));
    }
    // `held` becomes the start of each owner's members.
    for i in 0..width {
        held[i + 1] += held[i];
    }
    let mut members = Vec::new();
    lim.try_resize(&mut members, held[width] as usize, (0u32, 1u64))?;
    let mut next = held.clone();
    for (j, &o) in owner.iter().enumerate() {
        if o != UNNAMED {
            members[next[o as usize] as usize].0 = j as u32;
            next[o as usize] += 1;
        }
    }
    built(eng, f, end, sides, &held, &members).map(Some)
}

/// [`merge`] where a node of the chain is named twice, by one node of `top`
/// or by two: each node of `top` reads as a multiset of nodes of `end`. A
/// multiple is held as that many copies of the node's pairs where a
/// summed-out level is under `end`; elsewhere it is `None` before every
/// region is kept, and taken once after.
#[cold]
#[inline(never)]
fn merge_counted(
    eng: &Engine,
    f: &Tdd,
    shape: &Shape,
    top: VtreeIdx,
    end: VtreeIdx,
    sides: &[Option<Vec<u32>>; 2],
    keep_full: bool,
) -> Result<Option<Form>, OperationError> {
    let lim = eng.limits();
    let vtree = &*f.vtree;
    let summed = !shape.adds.is_empty();
    let width = f.levels[top.idx()].slot_count();
    let mut gate = lim.gate();
    // Node `o` of `top` adds up `members[starts[o]..starts[o + 1]]`, nodes of
    // the current level with their multiplicities, ascending.
    let mut starts = Vec::new();
    lim.reserve_exact(&mut starts, width + 1)?;
    starts.extend(0..=width as u32);
    let mut members = Vec::new();
    lim.reserve_exact(&mut members, width)?;
    members.extend((0..width as u32).map(|o| (o, 1u64)));
    let mut t = top;
    while t != end {
        let Kind::LevelChain(side) = shape.kind[t.idx()] else { unreachable!("a level chain") };
        let twin = negative_half(vtree, t, side);
        let level = &f.levels[t.idx()];
        let mut next_starts = Vec::new();
        lim.reserve_exact(&mut next_starts, width + 1)?;
        next_starts.push(0u32);
        let mut next: Vec<(u32, u64)> = Vec::new();
        for o in 0..width {
            let from = next.len();
            for &(i, k) in &members[starts[o] as usize..starts[o + 1] as usize] {
                let pairs = level.pairs_iter_of_idx(i as usize);
                gate.poll(pairs.len() as u64)?;
                for p in pairs.filter(|&p| Some(side_ref(p, other(side))) != twin) {
                    lim.try_push(&mut next, (side_ref(p, side), k))?;
                }
            }
            next[from..].sort_unstable_by_key(|&(j, _)| j);
            let mut to = from;
            for at in from..next.len() {
                let (j, k) = next[at];
                if to > from && next[to - 1].0 == j {
                    let held = &mut next[to - 1].1;
                    *held = if summed { held.checked_add(k).ok_or(OperationError::OverBudget)? } else { 1 };
                } else {
                    next[to] = (j, k);
                    to += 1;
                }
            }
            next.truncate(to);
            next_starts.push(u32::try_from(to).map_err(|_| OperationError::OverBudget)?);
        }
        lim.discard(std::mem::replace(&mut members, next));
        lim.discard(std::mem::replace(&mut starts, next_starts));
        t = child(vtree, t, side);
    }
    gate.flush()?;
    if summed && !shape.adds[end.idx()] && members.iter().any(|&(_, k)| k > 1) {
        if !keep_full {
            return Ok(None);
        }
        members.iter_mut().for_each(|m| m.1 = 1);
    }
    built(eng, f, end, sides, &starts, &members).map(Some)
}

/// `end`'s level in `top`'s numbering: node `o` holds the pairs of each node
/// of `members[starts[o]..starts[o + 1]]` as many times as it is listed with,
/// their sides read through `sides`.
fn built(
    eng: &Engine,
    f: &Tdd,
    end: VtreeIdx,
    sides: &[Option<Vec<u32>>; 2],
    starts: &[u32],
    members: &[(u32, u64)],
) -> Result<Form, OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    let source = &f.levels[end.idx()];
    let side = |map: &Option<Vec<u32>>, r: EncodedChildRef| match map {
        Some(map) => NodeIdx(map[r.raw() as usize]).into(),
        None => r,
    };
    let mut level = TddLevel::new();
    let mut pairs = Vec::new();
    for range in starts.windows(2) {
        pairs.clear();
        for &(j, k) in &members[range[0] as usize..range[1] as usize] {
            for _ in 0..k {
                for p in source.pairs_iter_of_idx(j as usize) {
                    lim.try_push(&mut pairs, ChildPair::new(side(&sides[0], p.left), side(&sides[1], p.right)))?;
                }
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

fn replace(eng: &Engine, f: &mut Tdd, by: Tdd) {
    let old = std::mem::replace(f, by);
    crate::diagram::return_levels(eng, old.levels.into_vec());
}
