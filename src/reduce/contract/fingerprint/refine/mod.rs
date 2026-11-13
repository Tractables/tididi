//! Twin groups by refining one partition of the child level, parent node by
//! parent node.

use crate::diagram::{ChildPair, ChildSide, EncodedNode, NodeIdx, StoredPairs, TddLevel};
use crate::limits::{Limits, OperationError};

use super::super::bits;
use super::super::scratch::{ClassInfo, ContractScratch, Marks, MergeRemap};
use super::split_pair;

mod list;
mod partition;
mod wide;

use list::Listing;
use partition::Partition;
use wide::{Wide, FEW_SIBLINGS};

pub(in crate::reduce::contract) use list::list_members;


/// The most pairs a parent node holds where its nodes are grouped on the
/// stack, by comparing every two; a wider one sorts them in scratch.
pub(super) const SMALL_PAIRS: usize = 8;

/// The fewest pairs of a scattered parent node, one whose pairs come in no
/// order of their siblings and name more than [`FEW_SIBLINGS`] of them, at
/// which a refinement allowed to give up leaves the level to the
/// fingerprints ([`refine_context_groups`]). The sibling split of such a
/// node costs several times the fingerprints' work a pair.
pub(super) const HAND_OFF_PAIRS: usize = 1 << 18;

/// The nodes a refinement starts with in its first class
/// ([`refine_context_groups`]).
pub(super) enum Start<'a> {
    /// Every node of the child level, with whether a large scattered parent
    /// node may hand the level back to fingerprinting.
    Whole { hand_off: bool },
    /// These nodes, ascending, which hold every node with a twin
    /// ([`super::listed`]).
    Listed(&'a [u32]),
}

/// The twin groups of the `width` nodes the pairs of `level` name on side
/// `side`, written to `scratch.flat_groups` and `scratch.group_starts` as
/// [`super::group_twins_by_entries`] writes them; returns whether there is
/// one.
///
/// Two nodes are twins when every parent node pairs them with the same
/// siblings. The nodes start in one class, and each parent node in turn
/// splits every class by the siblings it pairs each member with: after
/// parent node `p`, two nodes share a class exactly when parent nodes
/// `0..=p` pair them alike, so the classes left after the last are the twin
/// groups. A node alone in its group at some parent node has no twin (math
/// §2.8 is the case of a sibling no other node has there) and leaves the
/// partition, so that later parent nodes skip it. A class whose members a
/// parent node all names, in one group, stays whole; else each of its
/// groups of two or more there becomes a class of its own, one of which
/// may keep the class's id where every member is named there. The nodes no
/// pair names stay in the first class, and leave it at the end: a node of
/// no context is a node nothing reaches and has no twin. A refinement from
/// [`Start::Whole`] lists them in `scratch.unnamed` where there are two or
/// more, for the merge to drop.
///
/// The work is one read of the class of each node a parent node names, and
/// one write for each node that changes class, with each parent node's pairs
/// sorted in cache; a node in the first class, or without a twin, is read
/// off two bits. A wide parent node is split without a sort: sibling by
/// sibling ([`Partition::split_by_siblings`]), or by degree where its pairs
/// come in node order and its nodes' sibling sets are nested
/// ([`Partition::split_by_degree`]). Both count sets: no parent node may
/// repeat a pair, as none does in a diagram without a marginal level. No
/// fingerprint is formed and nothing is verified after: the classes are
/// exact.
///
/// Where `start` lists nodes that hold every node with a twin, every other
/// node is out of the partition from the start, as a node found to have no
/// twin is, and the groups are those among the listed nodes.
///
/// Where `child` holds the child level's node words, the pass that lists
/// the groups also marks their members as the merge's marking pass does
/// (`merge::contract_twins`), every group merging whole, as it does in a
/// diagram without a marginal level, the only kind refined: each member but
/// the lowest absorbed into the lowest. It counts each group's pairs at
/// its class, which the merge reads to rebuild the level in node order.
/// The pass reads the nodes in order, where the marking pass would read the
/// groups' members scattered over the level; `scratch.marked` holds what it
/// counted. It writes only each group's lowest member to `flat_groups`, at
/// the group's start, which is all a merge in node order reads of the
/// groups; [`list_members`] writes the rest.
///
/// Under [`Start::Whole`] with `hand_off`, a refinement of a stored level
/// that would split a scattered parent node of [`HAND_OFF_PAIRS`] pairs or
/// more gives up before its wide parent nodes, which it splits last, and
/// returns `None` with no groups listed; else it returns whether some group
/// holds two nodes or more.
pub(super) fn refine_context_groups(
    level: &TddLevel,
    side: ChildSide,
    lim: &Limits,
    scratch: &mut ContractScratch,
    width: usize,
    start: Start<'_>,
    child: Option<&[EncodedNode]>,
) -> Result<Option<bool>, OperationError> {
    scratch.marked = None;
    let ContractScratch {
        twin_class,
        twin_bits,
        classes,
        refine_keys,
        refine_runs,
        refine_links,
        refine_table,
        refine_wide,
        refine_radix,
        refine_column,
        refine_reach,
        refine_touched,
        refine_bucket,
        refine_met,
        refine_seen,
        refine_code,
        refine_rows,
        flat_groups,
        group_starts,
        unnamed,
        unnamed_exact,
        remap,
        marked,
        member_groups,
        ..
    } = scratch;
    // Zero between parent nodes, so it only grows.
    lim.try_resize(refine_code, width, 0u8)?;
    let mut wide = Wide {
        keys: refine_keys,
        runs: refine_runs,
        links: refine_links,
        table: refine_table,
        radix: refine_radix,
        column: refine_column,
        reach: refine_reach,
        touched: refine_touched,
        bucket: refine_bucket,
        met: refine_met,
        seen: refine_seen,
        code: &mut refine_code[..width],
        rows: refine_rows,
    };
    flat_groups.clear();
    group_starts.clear();
    let words = width.div_ceil(64);
    lim.try_resize(twin_class, width, 0u32)?;
    lim.try_resize(twin_bits, words, [0u64; 2])?;
    let bits = &mut twin_bits[..words];
    // The first class: every node, or the listed ones.
    let (whole, hand_off, first) = match start {
        Start::Whole { hand_off } => {
            bits.fill([0; 2]);
            (true, hand_off, width)
        }
        Start::Listed(listed) => {
            bits.fill([u64::MAX; 2]);
            for &t in listed {
                let (w, b) = bits::at(t);
                bits[w][0] &= !b;
                bits[w][1] &= !b;
            }
            (false, false, listed.len())
        }
    };
    // The classes made are nested sets of two or more nodes each, so there
    // are fewer than `width` besides the first.
    classes.clear();
    lim.reserve_exact(classes, width + 1)?;
    classes.push(ClassInfo::new(first as u32));
    let twinable = if first >= 2 { first } else { 0 };
    let mut part = Partition {
        class: &mut twin_class[..width],
        bits,
        classes,
        twinable,
    };
    // A stored level's wide parent nodes are refined after the narrow ones,
    // narrowest first by the power of two of their widths, and in arena
    // order within one, so that their pairs are read forward: by then most
    // nodes have left the partition, and a wide node sorts only the pairs
    // of the nodes still in it. The refinement stops once no node can have
    // a twin.
    let stored = level.stored();
    refine_wide.clear();
    match (stored, side) {
        (Some(stored), ChildSide::Left) => part.split_narrow::<true>(stored, lim, refine_wide)?,
        (Some(stored), ChildSide::Right) => part.split_narrow::<false>(stored, lim, refine_wide)?,
        (None, _) => {
            let (mut buf, mut cursor) = (Vec::new(), None);
            for i in 0..level.node_count() {
                if part.twinable == 0 {
                    break;
                }
                let pairs = level.described_read_next(&mut cursor, i, &mut buf);
                match pairs.len() {
                    0 => {}
                    1 => part.leave(split_pair(&pairs[0], side).0),
                    2 => part.split_two(pairs, side),
                    3 | 4 if part.split_few(pairs, side) => {}
                    n if n <= SMALL_PAIRS => part.split_small(pairs, side),
                    _ => part.split_wide(lim, pairs, side, &mut wide)?,
                }
            }
        }
    }
    if let Some(stored) = stored {
        refine_wide.sort_unstable();
        // The widest come last, past every node of fewer than
        // `HAND_OFF_PAIRS` pairs.
        if hand_off
            && refine_wide.last().is_some_and(|&w| w >= LARGE_WIDE)
            && part.twinable != 0
            && any_scattered(stored, refine_wide, side)
        {
            return Ok(None);
        }
        for &w in refine_wide.iter() {
            if part.twinable == 0 {
                break;
            }
            part.split_wide(lim, stored.of_idx(w as u32 as usize), side, &mut wide)?;
        }
    }
    // A node leaves the first class once a parent node names it, so a
    // refinement from the whole level that ends with no node there, or with
    // the two or more it lists, has every other node named.
    let left_unnamed = part.drop_unnamed(width, whole.then_some(unnamed), lim)?;
    *unnamed_exact = whole && left_unnamed != 1;
    debug_assert_eq!(
        part.twinable,
        part.classes
            .iter()
            .filter(|k| k.size >= 2)
            .map(|k| k.size as usize)
            .sum::<usize>(),
        "the count of nodes that can have a twin"
    );
    if part.twinable == 0 {
        return Ok(Some(false));
    }
    let Partition {
        class,
        bits,
        classes,
        twinable,
    } = part;
    lim.try_resize(flat_groups, twinable, 0u32)?;
    lim.reserve_exact(group_starts, twinable / 2)?;
    let mut list = Listing {
        class,
        bits,
        classes,
        flat_groups,
        group_starts,
        width,
    };
    match child {
        Some(nodes) => {
            debug_assert_eq!(nodes.len(), width, "the child level's every node word");
            let MergeRemap {
                final_remap,
                absorbed,
                ..
            } = remap;
            lim.try_resize(final_remap, width, NodeIdx(0))?;
            lim.try_resize(absorbed, width.div_ceil(64), 0u64)?;
            member_groups.clear();
            lim.reserve_exact(member_groups, twinable)?;
            *marked = Some(Marks {
                width,
                ..list.list::<true>(final_remap, absorbed, member_groups, nodes)
            });
        }
        None => {
            list.list::<false>(&mut [], &mut [], &mut Vec::new(), &[]);
        }
    }
    Ok(Some(true))
}

/// The least entry in the list of wide parent nodes
/// ([`Partition::split_narrow`]) of a node of [`HAND_OFF_PAIRS`] pairs or
/// more. An entry holds the bit length of the node's pair count above its
/// index.
const LARGE_WIDE: u64 = ((usize::BITS - HAND_OFF_PAIRS.leading_zeros()) as u64) << 32;

/// Whether a node of [`HAND_OFF_PAIRS`] pairs or more in the sorted list of
/// wide parent nodes `wide` is [`scattered`]. Out of line: a refinement
/// calls it only on a level that holds such a node.
#[inline(never)]
fn any_scattered(stored: StoredPairs<'_>, wide: &[u64], side: ChildSide) -> bool {
    wide.iter()
        .rev()
        .take_while(|&&w| w >= LARGE_WIDE)
        .any(|&w| scattered(stored.of_idx(w as u32 as usize), side))
}

/// Whether a parent node's pairs come in no order of their siblings and
/// name more than [`FEW_SIBLINGS`] siblings: the pairs the sibling split
/// sorts ([`Partition::split_by_siblings`]). Reads the pairs until it has
/// seen both.
fn scattered(pairs: &[ChildPair], side: ChildSide) -> bool {
    let Some(first) = pairs.first() else {
        return false;
    };
    let mut last = split_pair(first, side).1;
    let mut met = [last; FEW_SIBLINGS];
    let (mut ordered, mut many) = (true, false);
    let mut n = 1;
    for pair in &pairs[1..] {
        let s = split_pair(pair, side).1;
        ordered &= s >= last;
        last = s;
        if !many && !met[..n].contains(&s) {
            if n == FEW_SIBLINGS {
                many = true;
            } else {
                met[n] = s;
                n += 1;
            }
        }
        if !ordered && many {
            return true;
        }
    }
    false
}
