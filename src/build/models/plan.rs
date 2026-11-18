//! What the splitters hand to the emission pass: the plan of every split
//! vtree node, the root-down walk that makes it, and the sort of the
//! triples a node decomposes into.

use crate::diagram::NodeIdx;
use crate::limits::{Charged, Limits, OperationError};
use crate::sort::Radix;
use crate::vtree::Vtree;

use super::layout::Layout;
use super::rows::index_bits;

/// What the build needs at one vtree node that is split.
pub(super) enum Plan {
    /// A constrained leaf's atoms, as the leaf nodes that stand for them.
    Leaf(Vec<NodeIdx>),
    /// A node with constrained variables under both children.
    Branch(Decomposition),
}

/// The atoms of a node constrained on both sides, as child atoms.
pub(super) enum Decomposition {
    /// Atoms listed through their triples.
    Triples {
        /// How many atoms the node has.
        atoms: usize,
        /// Every realized `[atom, high atom, low atom]`, once each, in
        /// ascending order: the child atoms in the vtree's order, left then
        /// right. The two child atoms determine the parent's, so the
        /// triples of one atom are the pairs of its node.
        triples: Vec<[u32; 3]>,
    },
    /// One atom, whose pairs are listed by high atom: high atom `a` pairs
    /// with the low atoms `lows[ends[a - 1]..ends[a]]`, ascending, taking
    /// `ends[-1]` as zero. The node over every constrained variable has one
    /// atom and as many pairs as there are rows, and this list takes a
    /// third of the triples' space.
    Grouped {
        /// Where each high atom's low atoms end.
        ends: Vec<u32>,
        /// The low atoms of every pair.
        lows: Vec<u32>,
    },
    /// Atoms whose pairs are listed atom by atom: atom `a`'s pairs are
    /// `pairs[ends[a - 1]..ends[a]]`, taking `ends[-1]` as zero, each a
    /// high atom above a low atom, 32 bits apiece, ascending. The triples
    /// without their parent atom, in two thirds of their space.
    ByAtom {
        /// Where each atom's pairs end.
        ends: Vec<u32>,
        /// Every pair, its high atom in the upper half.
        pairs: Vec<u64>,
    },
}

impl Decomposition {
    /// How many atoms the node has.
    pub(super) fn atoms(&self) -> usize {
        match self {
            Decomposition::Triples { atoms, .. } => *atoms,
            Decomposition::Grouped { .. } => 1,
            Decomposition::ByAtom { ends, .. } => ends.len(),
        }
    }
}

/// The plan of every split vtree node, indexed by vtree node, from `root`,
/// the values of the node over every constrained variable. `leaf` gives a
/// constrained leaf's nodes from its values, and `split` a node's
/// decomposition and its low and high children's values from its own, the
/// widths given low first.
pub(super) fn plan<V: Charged>(
    lim: &Limits,
    vtree: &Vtree,
    layout: &Layout,
    root: V,
    mut leaf: impl FnMut(&V) -> Result<Vec<NodeIdx>, OperationError>,
    mut split: impl FnMut(&mut V, (usize, usize)) -> Result<(Decomposition, V, V), OperationError>,
) -> Result<Vec<Option<Plan>>, OperationError> {
    let mut plans: Vec<Option<Plan>> = Vec::new();
    lim.reserve_exact(&mut plans, vtree.num_nodes())?;
    plans.resize_with(vtree.num_nodes(), || None);
    let mut pending = Vec::new();
    lim.try_push(&mut pending, (layout.split_node(vtree, vtree.root()), root))?;
    let mut planned = 0u64;
    while let Some((t, mut values)) = pending.pop() {
        lim.check_stop()?;
        if vtree.node(t).is_leaf() {
            plans[t.idx()] = Some(Plan::Leaf(leaf(&values)?));
            lim.discard(values);
            continue;
        }
        let (high, low) = vtree.children(t);
        let widths = (
            layout.count[low.idx()] as usize,
            layout.count[high.idx()] as usize,
        );
        let (split, low_values, high_values) = split(&mut values, widths)?;
        // Each atom is a node of the result, so the output cap can refuse
        // before the build begins.
        planned += split.atoms() as u64;
        lim.check_output_cap(planned)?;
        lim.discard(values);
        plans[t.idx()] = Some(Plan::Branch(split));
        lim.try_push(&mut pending, (layout.split_node(vtree, low), low_values))?;
        lim.try_push(&mut pending, (layout.split_node(vtree, high), high_values))?;
    }
    Ok(plans)
}

/// How the triples handed to [`sorted_triples`] are grouped.
#[derive(Clone, Copy)]
pub(super) enum Emitted {
    /// In no order.
    Unordered,
    /// By their last atom, ascending.
    ByLast,
    /// By their middle atom, ascending.
    ByMid,
}

/// The triples `each` emits, sorted and each once: values of one atom can
/// decompose into the same pair of child atoms. `atoms` bounds each field,
/// and `emitted` says how `each` groups them. `packed` and `wide` are
/// buffers for the triples packed into one word or two.
pub(super) fn sorted_triples(
    lim: &Limits,
    radix: &mut Radix,
    packed: &mut Vec<u64>,
    wide: &mut Vec<u128>,
    atoms: (u32, u32, u32),
    emitted: Emitted,
    each: impl Fn(&mut dyn FnMut([u32; 3])),
) -> Result<Vec<[u32; 3]>, OperationError> {
    let mut out = Vec::new();
    let bits = (
        index_bits(atoms.0 as usize),
        index_bits(atoms.1 as usize),
        index_bits(atoms.2 as usize),
    );
    if bits.0 + bits.1 + bits.2 <= 64 {
        let (low, mid) = (bits.2, bits.1 + bits.2);
        let mut keys = std::mem::take(packed);
        keys.clear();
        let mut refused = Ok(());
        each(&mut |[a, l, r]| {
            if refused.is_ok() {
                refused = lim.try_push(&mut keys, (a as u64) << mid | (l as u64) << low | r as u64);
            }
        });
        // Keys grouped by one child atom, ascending, are ordered by the child
        // atoms by a sort of each middle atom's small group by its last
        // atoms, or by nothing at all, and only the parent atom's bits are
        // left to the radix sort.
        let ordered = match emitted {
            Emitted::Unordered => 0,
            Emitted::ByLast => low,
            Emitted::ByMid => {
                let (low_mask, group_mask) = ((1u64 << low) - 1, (1u64 << (mid - low)) - 1);
                let mut start = 0;
                while start < keys.len() {
                    let group = (keys[start] >> low) & group_mask;
                    let mut end = start + 1;
                    while end < keys.len() && (keys[end] >> low) & group_mask == group {
                        end += 1;
                    }
                    keys[start..end].sort_unstable_by_key(|&key| key & low_mask);
                    start = end;
                }
                mid
            }
        };
        let sorted = refused.and_then(|()| {
            radix.sort(
                lim,
                &mut keys,
                ordered as usize,
                (bits.0 + mid - ordered) as usize,
            )
        });
        keys.dedup();
        *packed = keys;
        sorted?;
        lim.reserve_exact(&mut out, packed.len())?;
        let (low_mask, mid_mask) = ((1u64 << low) - 1, (1u64 << (mid - low)) - 1);
        out.extend(packed.iter().map(|&key| {
            [
                (key >> mid) as u32,
                ((key >> low) & mid_mask) as u32,
                (key & low_mask) as u32,
            ]
        }));
    } else {
        let mut keys = std::mem::take(wide);
        keys.clear();
        let mut refused = Ok(());
        each(&mut |[a, l, r]| {
            if refused.is_ok() {
                refused =
                    lim.try_push(&mut keys, (a as u128) << 64 | (l as u128) << 32 | r as u128);
            }
        });
        keys.sort_unstable();
        keys.dedup();
        *wide = keys;
        refused?;
        lim.reserve_exact(&mut out, wide.len())?;
        out.extend(
            wide.iter()
                .map(|&key| [(key >> 64) as u32, (key >> 32) as u32, key as u32]),
        );
    }
    lim.check_stop()?;
    Ok(out)
}
