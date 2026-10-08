//! Store planned atoms as diagram nodes in the order of their least pairs.

use std::sync::Arc;

use crate::diagram::{Assembly, ChildPair, NodeIdx, TddNodeId, ONE_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{Vtree, VtreeIdx};
use crate::Engine;
use super::layout::Layout;
use super::split::{Decomposition, Plan};

/// A vtree node the pass has finished with, waiting for its parent.
struct Finished {
    /// The node each atom was stored at, in atom order.
    locals: Vec<NodeIdx>,
}

impl Charged for Finished {
    fn charged_bytes(&self) -> u64 {
        self.locals.charged_bytes()
    }
}

/// Store every level bottom-up from the plans and return the diagram's
/// output node.
pub(super) fn fill(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    layout: &Layout,
    plans: &mut [Option<Plan>],
) -> Result<TddNodeId, OperationError> {
    let lim = eng.limits();
    let mut state: Vec<Option<Finished>> = Vec::new();
    lim.reserve_exact(&mut state, vtree.num_nodes())?;
    state.resize_with(vtree.num_nodes(), || None);
    let mut pair_list = Vec::new();
    let mut emitted = 0u64;

    for t in vtree.bottomup() {
        lim.check_stop()?;
        let leaf = vtree.node(t).is_leaf();
        let finished = match (layout.count[t.idx()], one_sided_child(vtree, layout, t)) {
            (0, _) => free_subtree(eng, assembly, vtree, &state, t, leaf)?,
            (_, Some(constrained)) => {
                carry_child(eng, assembly, vtree, &state, t, constrained)?
            }
            _ => match plans[t.idx()].take().expect("every split node is planned") {
                Plan::Leaf(locals) => Finished { locals },
                Plan::Branch(split) => {
                    store_level(eng, assembly, vtree, &state, &mut pair_list, t, split)?
                }
            },
        };
        if !leaf {
            // A leaf level stores no node, so it emits none either.
            emitted += finished.locals.len() as u64;
            lim.check_output_cap(emitted)?;
            let (left, right) = vtree.children(t);
            for child in [left, right] {
                if let Some(done) = state[child.idx()].take() {
                    lim.discard(done);
                }
            }
        }
        state[t.idx()] = Some(finished);
    }

    let root = vtree.root();
    let done = state[root.idx()].as_ref().expect("the root was just finished");
    debug_assert_eq!(done.locals.len(), 1, "the root level holds one atom");
    Ok(TddNodeId { vtree: root, local: done.locals[0] })
}

/// The constrained child of an internal node whose other child constrains
/// nothing. `None` for a leaf, and for an internal node with constrained
/// variables on both sides or on neither.
fn one_sided_child(vtree: &Vtree, layout: &Layout, t: VtreeIdx) -> Option<VtreeIdx> {
    if vtree.node(t).is_leaf() {
        return None;
    }
    let (left, right) = vtree.children(t);
    match (layout.count[left.idx()], layout.count[right.idx()]) {
        (0, 0) => None,
        (0, _) => Some(right),
        (_, 0) => Some(left),
        _ => None,
    }
}

/// A node whose subtree holds no constrained variable: one atom, true over
/// every assignment to the subtree's leaves.
fn free_subtree(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    t: VtreeIdx,
    leaf: bool,
) -> Result<Finished, OperationError> {
    let local = if leaf {
        ONE_LEAF_IDX
    } else {
        let (left, right) = vtree.children(t);
        let pair = ChildPair::new(true_node(state, left), true_node(state, right));
        assembly.push(eng, t, &[pair])?
    };
    let mut locals = Vec::new();
    eng.limits().reserve_exact(&mut locals, 1)?;
    locals.push(local);
    Ok(Finished { locals })
}

/// The node a finished free subtree stored its one atom at.
fn true_node(state: &[Option<Finished>], t: VtreeIdx) -> NodeIdx {
    below(state, t).locals[0]
}

/// The finished record of a child.
fn below(state: &[Option<Finished>], t: VtreeIdx) -> &Finished {
    state[t.idx()].as_ref().expect("a child is finished before its parent")
}

/// An internal node whose constrained variables all sit under one child: its
/// atoms are that child's, each paired with the free side's true node, and
/// stored in the order of the child's nodes.
fn carry_child(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    t: VtreeIdx,
    constrained: VtreeIdx,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let (left, _) = vtree.children(t);
    let free_local = true_node(state, vtree.sibling(constrained));
    let from = below(state, constrained);
    let mut locals = Vec::new();
    lim.try_resize(&mut locals, from.locals.len(), NodeIdx(0))?;
    assembly.reserve(eng, t, from.locals.len(), 0)?;
    let order = stored_order(lim, from.locals.iter().map(|&child| u64::from(child.0)))?;
    for atom in in_order(&order, from.locals.len()) {
        let child = from.locals[atom];
        let pair = if constrained == left {
            ChildPair::new(child, free_local)
        } else {
            ChildPair::new(free_local, child)
        };
        locals[atom] = assembly.push(eng, t, &[pair])?;
    }
    lim.discard(order);
    Ok(Finished { locals })
}

/// The key a level's nodes are stored by: a pair's right node above its
/// left. The nodes ascend by their least pair so keyed, which is the order
/// in which a build level by level that meets each right node with each
/// left node in turn first meets them, so that the values of a variable
/// are numbered alike here and in a diagram so built.
#[inline]
fn pair_key(pair: ChildPair) -> u64 {
    u64::from(pair.right.0) << 32 | u64::from(pair.left.0)
}

/// The atoms in the order their nodes are stored: ascending by `keys`,
/// one distinct key an atom. Empty where the keys ascend already, so that
/// the atoms are stored in their own order.
fn stored_order(lim: &Limits, keys: impl ExactSizeIterator<Item = u64> + Clone) -> Result<Vec<u32>, OperationError> {
    let mut previous = None;
    if keys.clone().all(|key| previous.replace(key).is_none_or(|last| last < key)) {
        return Ok(Vec::new());
    }
    let mut keyed: Vec<u128> = Vec::new();
    lim.reserve_exact(&mut keyed, keys.len())?;
    keyed.extend(keys.enumerate().map(|(atom, key)| u128::from(key) << 32 | atom as u128));
    keyed.sort_unstable();
    let mut order = Vec::new();
    let reserved = lim.reserve_exact(&mut order, keyed.len());
    if reserved.is_ok() {
        order.extend(keyed.iter().map(|&k| k as u32));
    }
    lim.discard(keyed);
    reserved.map(|()| order)
}

/// The atoms below `atoms` in the order `order` from [`stored_order`]
/// gives: their own where it is empty.
fn in_order(order: &[u32], atoms: usize) -> impl Iterator<Item = usize> + '_ {
    let own = if order.is_empty() { 0..atoms } else { 0..0 };
    order.iter().map(|&atom| atom as usize).chain(own)
}

/// Store one node per atom of a node constrained on both sides, in the
/// order of their least pairs (see [`pair_key`]), and record where each
/// landed.
#[allow(clippy::too_many_arguments)]
fn store_level(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    pair_list: &mut Vec<ChildPair>,
    t: VtreeIdx,
    split: Decomposition,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let (left, right) = vtree.children(t);
    let (l, r) = (&below(state, left).locals, &below(state, right).locals);
    // A child whose nodes are not in atom order, as a leaf's literals and
    // a level stored in the order of its pairs need not be, reorders the
    // pairs.
    let ascending = |nodes: &[NodeIdx]| nodes.windows(2).all(|n| n[0] < n[1]);
    let one = |local: NodeIdx| -> Result<Finished, OperationError> {
        let mut locals = Vec::new();
        lim.reserve_exact(&mut locals, 1)?;
        locals.push(local);
        Ok(Finished { locals })
    };
    let (atoms, triples) = match split {
        Decomposition::Triples { atoms, triples } => (atoms, triples),
        Decomposition::Pairs { mut pairs } => {
            // Neither child is a leaf, and the pairs name their atoms: where
            // each child stored its atoms in atom order, their nodes.
            lim.gate().poll(pairs.len() as u64)?;
            let in_atom_order = |nodes: &[NodeIdx]| nodes.iter().enumerate().all(|(i, node)| node.idx() == i);
            if !(in_atom_order(l) && in_atom_order(r)) {
                for pair in pairs.iter_mut() {
                    *pair = ChildPair::new(l[pair.left.0 as usize], r[pair.right.0 as usize]);
                }
                pairs.sort_unstable();
            }
            let local = match pairs.len() {
                1 => {
                    let local = assembly.push(eng, t, &pairs);
                    lim.discard(pairs);
                    local?
                }
                _ => assembly.push_owned(eng, t, pairs)?,
            };
            return one(local);
        }
        Decomposition::ByAtom { ends, pairs } => {
            // The atoms' pairs run in child-atom order.
            let ordered = ascending(l) && ascending(r);
            let mut locals = Vec::new();
            lim.try_resize(&mut locals, ends.len(), NodeIdx(0))?;
            assembly.reserve(eng, t, ends.len(), pairs.len())?;
            let widest = ends.iter().scan(0, |start, &end| Some(end - std::mem::replace(start, end))).max().unwrap_or(0);
            let group = |atom: usize| &pairs[if atom == 0 { 0 } else { ends[atom - 1] as usize }..ends[atom] as usize];
            let node_pair = |pair: u64| ChildPair::new(l[(pair >> 32) as usize], r[pair as u32 as usize]);
            let least = (0..ends.len()).map(|atom| group(atom).iter().map(|&pair| pair_key(node_pair(pair))).min().unwrap_or(0));
            let order = stored_order(lim, least)?;
            pair_list.clear();
            lim.reserve_exact(pair_list, widest as usize)?;
            let mut gate = lim.gate();
            for atom in in_order(&order, ends.len()) {
                let group = group(atom);
                debug_assert!(!group.is_empty(), "every atom is realized by a row");
                gate.poll(group.len() as u64)?;
                pair_list.clear();
                pair_list.extend(group.iter().map(|&pair| node_pair(pair)));
                if !ordered {
                    pair_list.sort_unstable();
                }
                locals[atom] = assembly.push(eng, t, pair_list)?;
            }
            gate.flush()?;
            lim.discard(order);
            lim.discard(ends);
            lim.discard(pairs);
            return Ok(Finished { locals });
        }
        Decomposition::Grouped { ends, lows } => {
            // The one atom's pairs, which run in child-atom order: each high
            // atom's in turn, in the order of the high atoms' nodes, and
            // each of those in the order of the low atoms' nodes.
            assembly.reserve(eng, t, 1, lows.len())?;
            lim.gate().poll(lows.len() as u64)?;
            let group = |high: usize| &lows[if high == 0 { 0 } else { ends[high - 1] as usize }..ends[high] as usize];
            if ascending(l) && ascending(r) {
                // Already in order: written into the level as they are read.
                let highs = l.iter().enumerate().flat_map(|(high, &node)| group(high).iter().map(move |&low| ChildPair::new(node, r[low as usize])));
                let local = assembly.push_from(eng, t, lows.len(), highs)?;
                lim.discard(ends);
                lim.discard(lows);
                return one(local);
            }
            let order = stored_order(lim, l.iter().map(|&node| u64::from(node.0)))?;
            pair_list.clear();
            lim.reserve_exact(pair_list, lows.len())?;
            let low_ordered = ascending(r);
            for high in in_order(&order, l.len()) {
                let start = pair_list.len();
                pair_list.extend(group(high).iter().map(|&low| ChildPair::new(l[high], r[low as usize])));
                if !low_ordered {
                    pair_list[start..].sort_unstable();
                }
            }
            let local = assembly.push(eng, t, pair_list)?;
            lim.discard(order);
            lim.discard(ends);
            lim.discard(lows);
            return one(local);
        }
    };
    let mut locals = Vec::new();
    lim.try_resize(&mut locals, atoms, NodeIdx(0))?;
    assembly.reserve(eng, t, atoms, triples.len())?;
    let mut gate = lim.gate();
    if atoms == triples.len() {
        // Each atom has one pair, which needs neither sorting nor merging.
        let node_pair = |&[_, a, b]: &[u32; 3]| ChildPair::new(l[a as usize], r[b as usize]);
        let order = stored_order(lim, triples.iter().map(|triple| pair_key(node_pair(triple))))?;
        for atom in in_order(&order, atoms) {
            gate.poll(1)?;
            debug_assert_eq!(triples[atom][0] as usize, atom);
            locals[atom] = assembly.push(eng, t, &[node_pair(&triples[atom])])?;
        }
        lim.discard(order);
    } else {
        // The triples run in child-atom order, atom by atom.
        let ordered = ascending(l) && ascending(r);
        pair_list.clear();
        lim.reserve_exact(pair_list, triples.len())?;
        pair_list.extend(triples.iter().map(|&[_, a, b]| ChildPair::new(l[a as usize], r[b as usize])));
        let mut starts = Vec::new();
        lim.reserve_exact(&mut starts, atoms + 1)?;
        starts.push(0u32);
        for (at, w) in triples.windows(2).enumerate() {
            if w[0][0] != w[1][0] {
                starts.push(at as u32 + 1);
            }
        }
        starts.push(triples.len() as u32);
        debug_assert_eq!(starts.len(), atoms + 1, "every atom is realized by a row");
        let range = |atom: usize| starts[atom] as usize..starts[atom + 1] as usize;
        let least = (0..atoms).map(|atom| pair_list[range(atom)].iter().map(|&pair| pair_key(pair)).min().unwrap_or(0));
        let order = stored_order(lim, least)?;
        for atom in in_order(&order, atoms) {
            let range = range(atom);
            gate.poll(range.len() as u64)?;
            let pairs = &mut pair_list[range];
            if !ordered {
                pairs.sort_unstable();
            }
            locals[atom] = assembly.push(eng, t, pairs)?;
        }
        lim.discard(order);
        lim.discard(starts);
    }
    gate.flush()?;
    lim.discard(triples);
    Ok(Finished { locals })
}
