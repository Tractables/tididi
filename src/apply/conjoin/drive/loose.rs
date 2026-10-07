//! Which levels of a plain conjunction's result may hold a node that no pair
//! of its parent level names: the loose levels a prune after the conjunction
//! has to walk down to ([`Dirty::loose`](crate::diagram::Dirty::loose)).

use super::super::setup::{ApplyRun, Operands};
use crate::vtree::{Vtree, VtreeIdx};

/// The internal levels below the root of a plain conjunction's result that
/// may hold a node no pair of their parent level names, given the operands'
/// own such levels, as [`settle_loose`](crate::reduce::prune::settle_loose)
/// leaves them, and `carrier`, which operand each carried level came from (1
/// for `f`, 2 for `g`, 0 for a level the conjunction built and for the
/// leaves).
///
/// A carried level brings its subtree along, so a level whose parent was
/// carried is loose where its carrier has it loose. A level `c` whose parent
/// `p` the conjunction built is tight, every node of it named by a pair of
/// `p`, when
///
/// 1. each operand's level `c` is tight in that operand, or has one node
///    while `p` has a node; and
/// 2. no product of `p` loses a pair to `c`'s sibling `s`: every cell of
///    `s`'s product grid is a node (its node count is the product of the
///    operands' widths there), or `s` is a leaf where an operand's pairs name
///    only the constant true node, whose product with any leaf node is that
///    node.
///
/// Proof. A node of `c` is the product of a node `a` of `f`'s level `c` and
/// a node `b` of `g`'s (one of them the single node of the operand that is
/// the identity there, when `c` was carried). By (1), `f`'s level `p` has a
/// pair `(a, x)`: `a` is named there, or it is the level's only node and the
/// node `p` has came from a pair naming it. Likewise `g`'s level `p` has a
/// pair `(b, y)`. The products of `p`'s nodes `i ∋ (a, x)` and `j ∋ (b, y)`
/// join these two pairs into the pair `((a, b), (x, y))`, which every route
/// writes unless one side is no node: `(a, b)` is one, and `(x, y)` is one
/// by (2). So `p`'s product `(i, j)` is a node, and it names `(a, b)`.
///
/// Every other level under a built one is listed, for the prune to read
/// what it needs of it.
///
/// A leaf level is never compacted, and the root has no parent level, so
/// neither is listed.
pub(super) fn loose_levels(vtree: &Vtree, run: &ApplyRun<'_, '_>, carrier: &[u8], operands: Operands<&[u32]>) -> Vec<u32> {
    // Which operand has each level loose: 1 for `f`, 2 for `g`, as `carrier`.
    let mut loose_in = vec![0u8; vtree.num_nodes()];
    for &t in operands.f {
        loose_in[t as usize] |= 1;
    }
    for &t in operands.g {
        loose_in[t as usize] |= 2;
    }
    let internal = |t: VtreeIdx| !vtree.node(t).is_leaf();
    let mut loose = Vec::new();
    for (p, left, right) in vtree.internal_bottomup() {
        if carrier[p.idx()] != 0 {
            let by = carrier[p.idx()];
            loose.extend([left, right].into_iter().filter(|&c| internal(c) && loose_in[c.idx()] & by != 0).map(|c| c.0));
            continue;
        }
        let built = !run.levels[p.idx()].nodes.is_empty();
        let tight_in = |c: VtreeIdx, by: u8, widths: &[usize]| {
            loose_in[c.idx()] & by == 0 || (widths[c.idx()] == 1 && built)
        };
        let kills_none = |s: VtreeIdx| match internal(s) {
            true => run.levels[s.idx()].nodes.len() == run.f_widths[s.idx()] * run.g_widths[s.idx()],
            false => run.f_identity[s.idx()] || run.g_identity[s.idx()],
        };
        for (c, s) in [(left, right), (right, left)] {
            if internal(c) && !(tight_in(c, 1, run.f_widths) && tight_in(c, 2, run.g_widths) && kills_none(s)) {
                loose.push(c.0);
            }
        }
    }
    loose
}
