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
//! member is such a node: [`find_listed_twin_groups`] refines the listed
//! nodes alone, every other node out of the partition from the start. The
//! sweep lists nodes this way only in a diagram without a marginal level,
//! where every group merges whole and the survivor holds every member's
//! pairs.

use crate::diagram::{ChildSide, Tdd};
use crate::Engine;
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

use super::super::scratch::ContractScratch;
use super::refine::refine_context_groups;
use super::split_pair;

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

/// Find the twin groups among `listed`, nodes of the explicit child level
/// `t1` on side `t1_side` of `parent` ascending, by their whole contexts.
/// Every node with a twin must be listed with it, which the module doc says
/// when that holds: the refinement starts with every other node out of the
/// partition ([`refine_context_groups`]). Leaves the groups in
/// `scratch.flat_groups` and `scratch.group_starts` as
/// [`find_twin_groups`](super::find_twin_groups) does, and returns whether
/// there is one.
pub(in crate::reduce::contract) fn find_listed_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    listed: &[u32],
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    if listed.len() < 2 {
        scratch.flat_groups.clear();
        scratch.group_starts.clear();
        return Ok(false);
    }
    let simple = !super::super::sweep::diagram_marginal(tdd, scratch);
    let width = tdd.levels[t1.idx()].slot_count();
    refine_context_groups(&tdd.levels[parent.idx()], t1_side, eng.limits(), scratch, width, simple, Some(listed))
}
