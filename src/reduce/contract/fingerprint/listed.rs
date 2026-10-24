//! Twin searches that read only the nodes a contraction could have given a
//! twin.
//!
//! A parent level off the contraction worklist is at its fixpoint: no two
//! nodes of either child level share their contexts under it. Two nodes
//! become twins only when a context moves, and merging a twin group at a
//! level `t` moves few. The group's pair lists are unioned into its
//! survivor, so under `t` only the nodes the survivors' pairs name gain or
//! lose an entry ([`named_by`]). (At `t`'s sibling contexts move too, but
//! two of them are equal after the merge exactly where they were before:
//! see the sweep's `joint_contract_fixpoint`.)
//!
//! A group that forms afterwards holds such a node, every member shares that
//! node's context, and that context has an entry naming a survivor, so every
//! member is such a node: [`find_listed_twin_groups`] reads the listed nodes'
//! entries and groups them alone. The sweep lists nodes this way only in a
//! diagram without a marginal level, where every group merges whole and the
//! survivor holds every member's pairs.

use crate::diagram::{ChildSide, Tdd};
use crate::Engine;
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

use super::super::scratch::ContractScratch;
use super::{for_each_target_sibling, group_twins_by_entries, pack, split_pair, TwinEntries};

/// Entries listed outright, `(node, entry)` in the order they were read.
struct ListedEntries<'a>(&'a [(u32, u64)]);

impl TwinEntries for ListedEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for &(node, entry) in self.0 {
            f(node, entry);
        }
    }
}

/// A set of a level's nodes, one bit each, cleared over `width` nodes.
fn node_set<'a>(lim: &Limits, bits: &'a mut Vec<u64>, width: usize) -> Result<&'a mut [u64], OperationError> {
    let words = width.div_ceil(64);
    lim.try_resize(bits, words, 0u64)?;
    let bits = &mut bits[..words];
    bits.fill(0);
    Ok(bits)
}

#[inline]
fn insert(bits: &mut [u64], n: u32) {
    bits[(n / 64) as usize] |= 1 << (n % 64);
}


/// Replace `out` with the members of `bits`, ascending.
fn members_into(lim: &Limits, bits: &[u64], out: &mut Vec<u32>) -> Result<(), OperationError> {
    out.clear();
    let n: usize = bits.iter().map(|w| w.count_ones() as usize).sum();
    lim.reserve_exact(out, n)?;
    for (w, &word) in bits.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            out.push(w as u32 * 64 + word.trailing_zeros());
            word &= word - 1;
        }
    }
    Ok(())
}

/// The pairs the parent nodes `nodes` hold together.
pub(in crate::reduce::contract) fn pair_mass(tdd: &Tdd, parent: VtreeIdx, nodes: &[u32]) -> usize {
    let level = &tdd.levels[parent.idx()];
    nodes.iter().map(|&n| level.pair_count_at(n as usize)).sum()
}

/// Set `out` to the nodes of the child level `t1` (on side `t1_side` of
/// `parent`) that the pairs of the parent nodes `nodes` name, ascending.
/// `bits` is scratch, one bit per node of `t1`.
#[allow(clippy::too_many_arguments)]
pub(in crate::reduce::contract) fn named_by(
    lim: &Limits,
    tdd: &Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    nodes: &[u32],
    bits: &mut Vec<u64>,
    out: &mut Vec<u32>,
) -> Result<(), OperationError> {
    let parent_level = &tdd.levels[parent.idx()];
    let level = &tdd.levels[t1.idx()];
    let named = node_set(lim, bits, level.slot_count())?;
    for &n in nodes {
        for pair in parent_level.pairs_iter_of_idx(n as usize) {
            insert(named, split_pair(&pair, t1_side).0);
        }
    }
    members_into(lim, named, out)
}

/// Find the twin groups among `listed`, nodes of the explicit child level on
/// side `t1_side` of `parent` ascending, by their whole contexts. Every node
/// with a twin must be listed with it, which the module doc says when that
/// holds.
/// Leaves the groups in `scratch.flat_groups` and `scratch.group_starts` as
/// [`find_twin_groups`](super::find_twin_groups) does, in the level's own
/// numbering, and returns whether there is one.
pub(in crate::reduce::contract) fn find_listed_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    listed: &[u32],
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    let Some(&last) = listed.last() else {
        return Ok(false);
    };
    if listed.len() < 2 {
        return Ok(false);
    }
    // A bit per node, and per word of bits the listed nodes before it: a
    // listed node's rank in `listed` is two reads.
    let words = (last as usize) / 64 + 1;
    lim.try_resize(&mut scratch.listed_bits, words, 0u64)?;
    lim.try_resize(&mut scratch.listed_ranks, words, 0u32)?;
    let bits = &mut scratch.listed_bits[..words];
    bits.fill(0);
    for &n in listed {
        bits[(n / 64) as usize] |= 1 << (n % 64);
    }
    let ranks = &mut scratch.listed_ranks[..words];
    let mut before = 0u32;
    for (rank, word) in ranks.iter_mut().zip(bits.iter()) {
        *rank = before;
        before += word.count_ones();
    }
    // Each listed node's entries under its rank, in the order the whole
    // search reads them.
    let mut entries = std::mem::take(&mut scratch.listed_entries);
    entries.clear();
    let parent_level = &tdd.levels[parent.idx()];
    let mut grow = Ok(());
    for_each_target_sibling(parent_level, t1_side, |pi, t, sibling| {
        let (word, bit) = ((t / 64) as usize, t % 64);
        if let Some(&w) = bits.get(word)
            && w & (1 << bit) != 0
            && grow.is_ok()
        {
            let rank = ranks[word] + (w & ((1u64 << bit) - 1)).count_ones();
            grow = lim.try_push(&mut entries, (rank, pack(pi, sibling)));
        }
    });
    let found = grow.and_then(|()| group_twins_by_entries(eng, &ListedEntries(&entries), listed.len(), scratch));
    scratch.listed_entries = entries;
    let found = found?;
    // Ranks back to nodes: `listed` ascends, so the groups keep their order
    // and each its members'.
    for member in &mut scratch.flat_groups {
        *member = listed[*member as usize];
    }
    Ok(found)
}
