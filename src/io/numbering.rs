//! Local node indices shared by the text and binary writers.

use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

pub(super) const OMITTED: u32 = u32::MAX;

/// One structural level's file indices, in its stored node order.
/// Leaves retain all three labels, including those the output does not reach.
pub(super) struct LevelNumbering {
    /// The file index of each node, or [`OMITTED`] for an unreachable node.
    pub(super) local: Vec<u32>,
    /// The number of nodes the reader reconstructs at this level.
    pub(super) width: usize,
}

/// Number reachable internal nodes consecutively, keeping leaf labels fixed.
/// The caller has already rejected marginal levels.
pub(super) fn number_reachable(tdd: &Tdd) -> Vec<LevelNumbering> {
    tdd.reachable_nodes().into_iter().enumerate().map(|(i, reach)| {
        let leaf = tdd.vtree().node(VtreeIdx(i as u32)).is_leaf();
        let mut width = 0;
        let local = reach.into_iter().map(|reachable| {
            if leaf || reachable {
                let index = width as u32;
                width += 1;
                index
            } else {
                OMITTED
            }
        }).collect();
        LevelNumbering { local, width }
    }).collect()
}
