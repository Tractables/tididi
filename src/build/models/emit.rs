//! Store planned atoms as diagram nodes in the order of their least pairs.

use std::sync::Arc;

use super::filled;
use super::layout::Layout;
use super::plan::{Decomposition, Plan};
use crate::Engine;
use crate::diagram::{Assembly, ChildPair, NodeIdx, ONE_LEAF_IDX, Tdd, TddNodeId};
use crate::limits::{Limits, OperationError};
use crate::sort::Radix;
use crate::vtree::{Vtree, VtreeIdx};

/// The per-level emission pass over a planned [`Layout`].
struct Emit<'a> {
    eng: &'a Engine,
    assembly: Assembly<'a>,
    vtree: &'a Arc<Vtree>,
    /// For each vtree node finished and waiting for its parent, the node
    /// each atom was stored at, in atom order.
    state: Vec<Option<Vec<NodeIdx>>>,
    pair_list: Vec<ChildPair>,
    sort: OrderScratch,
}

/// Store every level bottom-up from the plans and seat the resulting canonical
/// diagram.
pub(super) fn fill(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    layout: &Layout,
    plans: &mut [Option<Plan>],
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let assembly = Assembly::new(eng, vtree)?;
    let mut state: Vec<Option<Vec<NodeIdx>>> = Vec::new();
    lim.reserve_exact(&mut state, vtree.num_nodes())?;
    state.resize_with(vtree.num_nodes(), || None);
    let mut emit = Emit {
        eng,
        assembly,
        vtree,
        state,
        pair_list: Vec::new(),
        sort: OrderScratch::default(),
    };
    let mut emitted = 0u64;

    for t in vtree.bottomup() {
        lim.check_stop()?;
        let leaf = vtree.node(t).is_leaf();
        let finished = match (layout.count[t.idx()], one_sided_child(vtree, layout, t)) {
            (0, _) => emit.free_subtree(t, leaf)?,
            (_, Some(constrained)) => emit.carry_child(t, constrained)?,
            _ => match plans[t.idx()].take().expect("every split node is planned") {
                Plan::Leaf(locals) => locals,
                Plan::Branch(split) => emit.store_level(t, split)?,
            },
        };
        if !leaf {
            // A leaf level stores no node, so it emits none either.
            emitted += finished.len() as u64;
            lim.check_output_cap(emitted)?;
            let (left, right) = vtree.children(t);
            for child in [left, right] {
                if let Some(done) = emit.state[child.idx()].take() {
                    lim.discard(done);
                }
            }
        }
        emit.state[t.idx()] = Some(finished);
    }
    let OrderScratch { keys, radix, wide, wide_radix } = std::mem::take(&mut emit.sort);
    lim.discard(keys);
    radix.discard(lim);
    lim.discard(wide);
    wide_radix.discard(lim);

    let root = vtree.root();
    let done = emit.state[root.idx()]
        .as_ref()
        .expect("the root was just finished");
    debug_assert_eq!(done.len(), 1, "the root level holds one atom");
    let output = TddNodeId {
        vtree: root,
        local: done[0],
    };
    // The levels are canonical as built: seat them with nothing to reduce.
    let (levels, _) = emit.assembly.parts_mut();
    Ok(crate::build::seat_canonical(
        eng,
        vtree,
        std::mem::take(levels),
        output,
    ))
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

/// The node a finished free subtree stored its one atom at.
fn true_node(state: &[Option<Vec<NodeIdx>>], t: VtreeIdx) -> NodeIdx {
    below(state, t)[0]
}

/// The nodes a finished child stored its atoms at.
fn below(state: &[Option<Vec<NodeIdx>>], t: VtreeIdx) -> &[NodeIdx] {
    state[t.idx()]
        .as_deref()
        .expect("a child is finished before its parent")
}

impl Emit<'_> {
    /// A node whose subtree holds no constrained variable: one atom, true over
    /// every assignment to the subtree's leaves.
    fn free_subtree(&mut self, t: VtreeIdx, leaf: bool) -> Result<Vec<NodeIdx>, OperationError> {
        let local = if leaf {
            ONE_LEAF_IDX
        } else {
            let (left, right) = self.vtree.children(t);
            let pair = ChildPair::new(true_node(&self.state, left), true_node(&self.state, right));
            self.assembly.push(self.eng, t, &[pair])?
        };
        filled(self.eng.limits(), 1, local)
    }

    /// An internal node whose constrained variables all sit under one child:
    /// its atoms are that child's, each paired with the free side's true node,
    /// and stored in the order of the child's nodes.
    fn carry_child(
        &mut self,
        t: VtreeIdx,
        constrained: VtreeIdx,
    ) -> Result<Vec<NodeIdx>, OperationError> {
        let lim = self.eng.limits();
        let (left, _) = self.vtree.children(t);
        let free_local = true_node(&self.state, self.vtree.sibling(constrained));
        let from = below(&self.state, constrained);
        let mut locals = filled(lim, from.len(), NodeIdx(0))?;
        self.assembly.reserve(self.eng, t, from.len(), 0)?;
        let order = stored_order(lim, &mut self.sort, from.iter().map(|&child| u64::from(child.0)))?;
        for atom in in_order(&order, from.len()) {
            let child = from[atom];
            let pair = if constrained == left {
                ChildPair::new(child, free_local)
            } else {
                ChildPair::new(free_local, child)
            };
            locals[atom] = self.assembly.push(self.eng, t, &[pair])?;
        }
        lim.discard(order);
        Ok(locals)
    }
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

/// The buffers [`stored_order`] sorts through, kept from level to level.
#[derive(Default)]
struct OrderScratch {
    keys: Vec<u64>,
    radix: Radix<u64>,
    wide: Vec<u128>,
    wide_radix: Radix<u128>,
}

/// The atoms in the order their nodes are stored: ascending by `keys`,
/// one distinct key an atom. Empty where the keys ascend already, so that
/// the atoms are stored in their own order.
fn stored_order(
    lim: &Limits,
    s: &mut OrderScratch,
    keys: impl ExactSizeIterator<Item = u64>,
) -> Result<Vec<u32>, OperationError> {
    let n = keys.len();
    s.keys.clear();
    lim.reserve_exact(&mut s.keys, n)?;
    let (mut high, mut low, mut ascending, mut last) = (0u64, 0u64, true, None);
    for key in keys {
        ascending &= last.replace(key).is_none_or(|last| last < key);
        high |= key >> 32;
        low |= key & u64::from(u32::MAX);
        s.keys.push(key);
    }
    if ascending {
        return Ok(Vec::new());
    }
    // The halves of a pair key are node indices, mostly far narrower than
    // 32 bits: the radix sort places only the bits they use, and where
    // those and the atom's fit in a word, sorts words.
    let low_bits = u64::BITS - low.leading_zeros();
    let bits = (low_bits + u64::BITS - high.leading_zeros()) as usize;
    let squeezed = |key: u64| (key >> 32) << low_bits | key & u64::from(u32::MAX);
    let atom_bits = (usize::BITS - (n - 1).leading_zeros()) as usize;
    let mut gate = lim.gate();
    let mut order = Vec::new();
    if bits + atom_bits <= 64 {
        for (atom, key) in s.keys.iter_mut().enumerate() {
            *key = squeezed(*key) << atom_bits | atom as u64;
        }
        s.radix.sort_polling(lim, &mut s.keys, atom_bits, bits, &mut |work| gate.poll(work))?;
        gate.flush()?;
        lim.reserve_exact(&mut order, n)?;
        let atom = (1u64 << atom_bits) - 1;
        order.extend(s.keys.iter().map(|&record| (record & atom) as u32));
    } else {
        s.wide.clear();
        lim.reserve_exact(&mut s.wide, n)?;
        s.wide.extend(s.keys.iter().enumerate().map(|(atom, &key)| u128::from(squeezed(key)) << 32 | atom as u128));
        s.wide_radix.sort_by_key_polling(lim, &mut s.wide, |&record| (record >> 32) as u64, 0, bits, &mut |work| {
            gate.poll(work)
        })?;
        gate.flush()?;
        lim.reserve_exact(&mut order, n)?;
        order.extend(s.wide.iter().map(|&record| record as u32));
    }
    Ok(order)
}

/// The atoms below `atoms` in the order `order` from [`stored_order`]
/// gives: their own where it is empty.
fn in_order(order: &[u32], atoms: usize) -> impl Iterator<Item = usize> + '_ {
    let own = if order.is_empty() { 0..atoms } else { 0..0 };
    order.iter().map(|&atom| atom as usize).chain(own)
}

impl Emit<'_> {
    /// Store one node per atom of a node constrained on both sides, in the
    /// order of their least pairs (see [`pair_key`]), and record where each
    /// landed.
    fn store_level(
        &mut self,
        t: VtreeIdx,
        split: Decomposition,
    ) -> Result<Vec<NodeIdx>, OperationError> {
        let Self {
            eng,
            assembly,
            vtree,
            state,
            pair_list,
            sort,
        } = self;
        let lim = eng.limits();
        let (left, right) = vtree.children(t);
        let (l, r) = (below(state, left), below(state, right));
        // A child whose nodes are not in atom order, as a leaf's literals and
        // a level stored in the order of its pairs need not be, reorders the
        // pairs.
        let ascending = |nodes: &[NodeIdx]| nodes.windows(2).all(|n| n[0] < n[1]);
        let (atoms, triples) = match split {
            Decomposition::Triples { atoms, triples } => (atoms, triples),
            Decomposition::ByAtom { ends, pairs } => {
                // The atoms' pairs run in child-atom order.
                let ordered = ascending(l) && ascending(r);
                let mut locals = filled(lim, ends.len(), NodeIdx(0))?;
                assembly.reserve(eng, t, ends.len(), pairs.len())?;
                let widest = ends
                    .iter()
                    .scan(0, |start, &end| Some(end - std::mem::replace(start, end)))
                    .max()
                    .unwrap_or(0);
                let group = |atom: usize| {
                    &pairs[if atom == 0 {
                        0
                    } else {
                        ends[atom - 1] as usize
                    }..ends[atom] as usize]
                };
                let node_pair =
                    |pair: u64| ChildPair::new(l[(pair >> 32) as usize], r[pair as u32 as usize]);
                let least = (0..ends.len()).map(|atom| {
                    group(atom)
                        .iter()
                        .map(|&pair| pair_key(node_pair(pair)))
                        .min()
                        .unwrap_or(0)
                });
                let order = stored_order(lim, sort, least)?;
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
                return Ok(locals);
            }
            Decomposition::Grouped { ends, lows } => {
                // The one atom's pairs, which run in child-atom order: each high
                // atom's in turn, in the order of the high atoms' nodes, and
                // each of those in the order of the low atoms' nodes.
                assembly.reserve(eng, t, 1, lows.len())?;
                lim.gate().poll(lows.len() as u64)?;
                let group = |high: usize| {
                    &lows[if high == 0 {
                        0
                    } else {
                        ends[high - 1] as usize
                    }..ends[high] as usize]
                };
                let order = stored_order(lim, sort, l.iter().map(|&node| u64::from(node.0)))?;
                pair_list.clear();
                lim.reserve_exact(pair_list, lows.len())?;
                let low_ordered = ascending(r);
                for high in in_order(&order, l.len()) {
                    let start = pair_list.len();
                    pair_list.extend(
                        group(high)
                            .iter()
                            .map(|&low| ChildPair::new(l[high], r[low as usize])),
                    );
                    if !low_ordered {
                        pair_list[start..].sort_unstable();
                    }
                }
                let local = assembly.push(eng, t, pair_list)?;
                lim.discard(order);
                lim.discard(ends);
                lim.discard(lows);
                return filled(lim, 1, local);
            }
        };
        let mut locals = filled(lim, atoms, NodeIdx(0))?;
        assembly.reserve(eng, t, atoms, triples.len())?;
        let mut gate = lim.gate();
        if atoms == triples.len() {
            // Each atom has one pair, which needs neither sorting nor merging.
            let node_pair = |&[_, a, b]: &[u32; 3]| ChildPair::new(l[a as usize], r[b as usize]);
            let order = stored_order(
                lim,
                sort,
                triples.iter().map(|triple| pair_key(node_pair(triple))),
            )?;
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
            pair_list.extend(
                triples
                    .iter()
                    .map(|&[_, a, b]| ChildPair::new(l[a as usize], r[b as usize])),
            );
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
            let least = (0..atoms).map(|atom| {
                pair_list[range(atom)]
                    .iter()
                    .map(|&pair| pair_key(pair))
                    .min()
                    .unwrap_or(0)
            });
            let order = stored_order(lim, sort, least)?;
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
        Ok(locals)
    }
}

#[cfg(test)]
#[path = "tests/emit.rs"]
mod tests;
