//! The groups of the partition a refinement leaves.

use crate::diagram::{EncodedNode, NodeIdx, NodeKind};
use crate::limits::{Limits, OperationError};
use crate::reduce::contract::bits;
use crate::reduce::contract::merge::DROPPED;
use crate::reduce::contract::scratch::{ClassInfo, ContractScratch, Marks};

/// The partition a refinement leaves, and the lists of its groups.
pub(super) struct Listing<'a> {
    pub(super) class: &'a [u32],
    pub(super) bits: &'a [[u64; 2]],
    pub(super) classes: &'a mut Vec<ClassInfo>,
    pub(super) flat_groups: &'a mut [u32],
    pub(super) group_starts: &'a mut Vec<u32>,
    pub(super) width: usize,
}

impl Listing<'_> {
    /// List the classes of two or more, in the order of their lowest
    /// members, each ascending: one pass over the nodes not alone, a word of
    /// bits at a time, so the nodes without a twin are skipped 64 at once.
    /// `MARK` marks the members as the merge's marking pass does: each but
    /// a group's lowest absorbed, its `final_remap` entry the lowest's index
    /// under [`DROPPED`]; every word of `absorbed` is written. It counts
    /// each group's pairs at its class's `hits`, off the members' words in
    /// `nodes`, and returns what [`Marks`] holds, but the width. It writes
    /// only each group's lowest member to `flat_groups`, each class's
    /// group to its `pos`, and each other member's group to
    /// `member_groups`, within its reserve.
    #[inline(always)]
    pub(super) fn list<const MARK: bool>(
        &mut self,
        final_remap: &mut [NodeIdx],
        absorbed: &mut [u64],
        member_groups: &mut Vec<u32>,
        nodes: &[EncodedNode],
    ) -> Marks {
        let Listing {
            class,
            bits,
            classes,
            flat_groups,
            group_starts,
            width,
        } = self;
        let width = *width;
        let mut next = 0u32;
        let mut marks = Marks {
            inline: 0,
            in_order: true,
            width: 0,
        };
        for (w, &[moved, alone]) in bits.iter().enumerate() {
            let mut live = !alone & bits::word_mask(width, w);
            let mut merged = 0u64;
            while live != 0 {
                let b = live.trailing_zeros() as usize;
                live &= live - 1;
                let t = w * 64 + b;
                let c = if (moved >> b) & 1 == 0 {
                    0
                } else {
                    class[t] as usize
                };
                let k = &mut classes[c];
                if k.size < 2 {
                    continue;
                }
                let pairs = if MARK {
                    match nodes[t].kind() {
                        NodeKind::Inline(_) => {
                            marks.inline += 1;
                            1
                        }
                        NodeKind::Multi { len, .. } => {
                            marks.in_order &= len > 0;
                            len
                        }
                        NodeKind::MultiRanged(_) => {
                            marks.in_order = false;
                            0
                        }
                    }
                } else {
                    0
                };
                if k.pos == u32::MAX {
                    flat_groups[next as usize] = t as u32;
                    k.pos = if MARK {
                        group_starts.len() as u32
                    } else {
                        next + 1
                    };
                    group_starts.push(next);
                    next += k.size;
                    k.dest = t as u32;
                    k.hits = pairs;
                } else if MARK {
                    merged |= 1 << b;
                    final_remap[t] = NodeIdx(k.dest | DROPPED);
                    k.hits = k.hits.saturating_add(pairs);
                    debug_assert!(
                        member_groups.len() < member_groups.capacity(),
                        "within the reserve"
                    );
                    member_groups.push(k.pos);
                } else {
                    flat_groups[k.pos as usize] = t as u32;
                    k.pos += 1;
                }
            }
            if MARK {
                absorbed[w] = merged;
            }
        }
        debug_assert_eq!(
            next as usize,
            flat_groups.len(),
            "every node that can have a twin is listed"
        );
        marks
    }
}

/// Where the last search marked the members ([`ContractScratch::marked`]),
/// write each group's members after its lowest to `flat_groups`, which
/// then lists the groups whole, as a search that does not mark lists them:
/// one pass over the absorbed bits, the members in node order beside their
/// groups in `member_groups`, with each group's next place in `rooms`.
/// Writes nothing otherwise; writes the same where it ran already.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` where the room for the places is
/// refused.
pub(in crate::reduce::contract) fn list_members(
    lim: &Limits,
    scratch: &mut ContractScratch,
) -> Result<(), OperationError> {
    let Some(Marks { width, .. }) = scratch.marked else {
        return Ok(());
    };
    let ContractScratch {
        remap,
        flat_groups,
        group_starts,
        member_groups,
        rooms,
        ..
    } = scratch;
    rooms.clear();
    lim.reserve_exact(rooms, group_starts.len())?;
    rooms.extend(group_starts.iter().map(|&s| s + 1));
    let mut groups = member_groups.iter();
    for (w, &word) in remap.absorbed[..width.div_ceil(64)].iter().enumerate() {
        let mut members = word;
        while members != 0 {
            let t = w * 64 + members.trailing_zeros() as usize;
            members &= members - 1;
            let g = *groups.next().expect("a member's group") as usize;
            flat_groups[rooms[g] as usize] = t as u32;
            rooms[g] += 1;
        }
    }
    debug_assert!(groups.next().is_none(), "every member is absorbed");
    Ok(())
}
