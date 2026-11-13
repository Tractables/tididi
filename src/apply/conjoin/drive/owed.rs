//! What a plain conjunction's result owes the reductions after it, as lists
//! of levels each reduction reads: the loose levels a prune has to walk down
//! to ([`loose_levels`], [`Dirty::loose`](crate::diagram::Dirty::loose)),
//! the built levels owed the twin contraction ([`split_built`]), and the
//! built levels pair fusion would find nothing at ([`fused_built_levels`]).

use super::super::setup::{ApplyRun, Operands, VtreeMask};
use super::region::carried_from_f;
use crate::diagram::{Tdd, TddLevel};
use crate::vtree::{Vtree, VtreeIdx};

/// The internal levels below the root of a plain conjunction's result that
/// may hold a node no pair of their parent level names, given the operands'
/// own such levels, as [`settle_loose`](crate::reduce::prune::settle_loose)
/// leaves them, and which operand each level the sweep carried as an
/// identity level came from
/// ([`SweepLists::identity_carrier`](super::super::setup::SweepLists::identity_carrier));
/// the result took each free level from the operand [`carried_from_f`]
/// names.
///
/// A carried level, and a free one, brings its subtree along, so a level
/// whose parent was carried or free is loose where the operand the result
/// took the parent from has it loose: read off the operands' lists, for
/// the levels under a free one, with no walk of the regions. A level `c` whose parent
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
/// A level `c` whose parent `p` the conjunction built as an operand holds
/// its level `p`
/// ([`SweepLists::as_operand`](super::super::setup::SweepLists::as_operand))
/// is also tight when that operand's level `c` is tight in it and the
/// result's level `c` has as many nodes as the operand's. Proof. `p`'s pairs
/// are the operand's, index for index, so the nodes of `c` they name are those the operand's level `p`
/// names of its level `c`: all of its nodes, which are as many as `c`'s.
///
/// A level `c` whose sibling `s` is a vtree leaf is also tight when `f`'s
/// level `c` is tight in `f`, as in (1), and the sparse leaf arm built `p`
/// with `g`'s level `p` holding each live product of `c` for every label
/// ([`SweepLists::outer_held`](super::super::setup::SweepLists::outer_held)).
/// Proof. A node of `c` is a live product `(a, b)`, and `f`'s level `p` has
/// a pair `(a, x)` in a node `i`, as above. `g`'s level `p` holds `b` with the label One, or with both
/// literals; one of those pairs, `(b, y)` in a node `j`, has a label that
/// meets `x`: One meets every label, and a literal meets itself. The leaf
/// arm joins `(a, x)` and `(b, y)` under the product `(a, b)`, and `x ∧ y`
/// is a leaf node, so `p`'s product `(i, j)` is a node, and it names
/// `(a, b)`.
///
/// Every other level under a built one is listed, for the prune to read
/// what it needs of it.
///
/// A leaf level is never compacted, and the root has no parent level, so
/// neither is listed.
///
/// The list is written into the allocation of the larger operand list, grown
/// once to the most it can hold. It lists each level once.
pub(super) fn loose_levels(
    vtree: &Vtree,
    run: &ApplyRun<'_, '_>,
    operands: Operands<Vec<u32>>,
) -> Vec<u32> {
    // Which operand has each level loose: 1 for `f`, 2 for `g`, as
    // `identity_carrier` names them; `LISTED` once a level under a free one
    // is listed.
    const LISTED: u8 = 4;
    let mut loose_in = vec![0u8; vtree.num_nodes()];
    for &t in &operands.f {
        loose_in[t as usize] |= 1;
    }
    for &t in &operands.g {
        loose_in[t as usize] |= 2;
    }
    let internal = |t: VtreeIdx| !vtree.node(t).is_leaf();
    let cone = run.cone;
    // Whether the result lists the level `c` of the list of the operand
    // `by` names because its parent is free and was taken from that operand.
    let under_free = |c: u32, by: u8, loose_in: &mut [u8]| {
        let c = VtreeIdx(c);
        let from = |p: VtreeIdx| match carried_from_f(cone, p.idx()) {
            true => 1,
            false => 2,
        };
        let listed = internal(c)
            && loose_in[c.idx()] & LISTED == 0
            && vtree
                .node(c)
                .parent()
                .is_some_and(|p| cone.free_at(p.idx()) && from(p) == by);
        if listed {
            loose_in[c.idx()] |= LISTED;
        }
        listed
    };
    let Operands {
        f: f_list,
        g: g_list,
    } = operands;
    let ((mut loose, by), (other, other_by)) = match f_list.capacity() >= g_list.capacity() {
        true => ((f_list, 1), (g_list, 2)),
        false => ((g_list, 2), (f_list, 1)),
    };
    loose.retain(|&c| under_free(c, by, &mut loose_in));
    // Each level below the root is listed once at most, from its parent.
    loose.reserve(
        vtree
            .internal_bottomup_slice()
            .len()
            .saturating_sub(1)
            .saturating_sub(loose.len()),
    );
    for &c in &other {
        if under_free(c, other_by, &mut loose_in) {
            loose.push(c);
        }
    }
    let under_free_levels = loose.len();
    for &p in cone.built {
        let (left, right) = vtree.children(p);
        let by = run.lists.identity_carrier(p.idx());
        if by != 0 {
            loose.extend(
                [left, right]
                    .into_iter()
                    .filter(|&c| internal(c) && loose_in[c.idx()] & by != 0)
                    .map(|c| c.0),
            );
            continue;
        }
        let built = !run.levels[p.idx()].nodes().is_empty();
        let tight_in = |c: VtreeIdx, by: u8, widths: &[usize]| {
            loose_in[c.idx()] & by == 0 || (widths[c.idx()] == 1 && built)
        };
        let kills_none = |s: VtreeIdx| match internal(s) {
            true => {
                run.f_widths[s.idx()].checked_mul(run.g_widths[s.idx()])
                    == Some(run.levels[s.idx()].nodes().len())
            }
            false => run.f_identity[s.idx()] || run.g_identity[s.idx()],
        };
        // The operand the conjunction built `p` as, if any.
        let as_operand = run
            .lists
            .as_operand
            .binary_search_by_key(&p.idx(), |e| e.0)
            .ok()
            .map(|i| run.lists.as_operand[i].1);
        let held = run.lists.outer_held.binary_search(&p.idx()).is_ok();
        let named_as_operand = |c: VtreeIdx| {
            as_operand.is_some_and(|by| {
                let widths = if by == 1 {
                    &run.f_widths[..]
                } else {
                    &run.g_widths[..]
                };
                loose_in[c.idx()] & by == 0 && run.levels[c.idx()].node_count() == widths[c.idx()]
            })
        };
        for (c, s) in [(left, right), (right, left)] {
            if !internal(c) || named_as_operand(c) {
                continue;
            }
            let tight_f = tight_in(c, 1, run.f_widths);
            if tight_f && tight_in(c, 2, run.g_widths) && kills_none(s) {
                continue;
            }
            if tight_f && held && !internal(s) {
                super::super::note_held_tight();
                continue;
            }
            loose.push(c.0);
        }
    }
    // In the order of the levels' parents bottom-up, the left child first,
    // as a walk of every level lists them: the levels under the free ones
    // are sorted into those under the levels of the cone, which are in that
    // order already.
    if under_free_levels > 0 {
        loose.sort_by_key(|&c| {
            let c = VtreeIdx(c);
            let p = vtree.node(c).parent().expect("a listed level has a parent");
            (vtree.topo_pos(p), vtree.children(p).1 == c)
        });
    }
    loose
}

/// Note in [`SweepLists::as_operand`](super::super::setup::SweepLists::as_operand)
/// the level `t` just built when it holds an operand's level at `t`, node
/// for node and pair for pair: each node of its children then has there the
/// contexts the operand's node of its index has. Not compared where the
/// level adds no twins anyway or the leaf arm noted it already. The
/// operands' levels at `t` are dropped only when its parent is built.
pub(super) fn note_operand_level(
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
) {
    let ti = t.idx();
    if run.lists.as_operand.last().is_some_and(|e| e.0 == ti)
        || adds_no_twins(vtree, &run.f_identity[..], &run.g_identity[..], t)
    {
        return;
    }
    let level = &run.levels[ti];
    let of_f = same_pairs(level, f.level(t));
    if of_f || same_pairs(level, g.level(t)) {
        run.lists.as_operand.push((ti, if of_f { 1 } else { 2 }));
        super::super::note_as_operand(ti, of_f);
    }
}

/// Whether `a` holds `b`'s nodes and pairs, node for node and each node's
/// pairs in order: stored as `b` stores them (the same node words, ranges
/// and pair arena, or the same description), laid out otherwise in its
/// arena, or one of them stored and the other described. Never on a
/// marginal level or one without nodes. The lengths and the last node and
/// pair are compared first, then the rest as memory, so that a level that
/// differs costs a read of its first difference; where the memory differs,
/// or one level is described, the first and last nodes' pairs, then each
/// node's. A level that holds a node's pairs in another order is not found.
fn same_pairs(a: &TddLevel, b: &TddLevel) -> bool {
    let n = a.node_count();
    if n == 0 || n != b.node_count() || a.is_marginal() || b.is_marginal() {
        return false;
    }
    match (a.stored(), b.stored()) {
        (Some(x), Some(y)) => {
            let (xa, ya) = (x.arena(), y.arena());
            let (xn, yn) = (x.nodes(), y.nodes());
            if xa.len() == ya.len()
                && xa.last() == ya.last()
                && xn.last() == yn.last()
                && a.ranges == b.ranges
                && xn == yn
                && xa == ya
            {
                return true;
            }
            let same = |k: usize| x.of(&xn[k]) == y.of(&yn[k]);
            same(0) && same(n - 1) && (1..n - 1).all(same)
        }
        (None, None) => a.pairs == b.pairs && a.nodes == b.nodes,
        _ => {
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            let (mut xc, mut yc) = (None, None);
            let mut same = |k: usize| {
                a.pairs_read_next(&mut xc, k, &mut xs) == b.pairs_read_next(&mut yc, k, &mut ys)
            };
            same(0) && same(n - 1) && (1..n - 1).all(same)
        }
    }
}

/// The levels the sweep built that hold no node with two pairs sharing a
/// ref into their structural child, since their carrier's level holds none
/// (`Dirty::is_fused`): a pair fusion there would find nothing.
///
/// Such a level's one marginal child is a pass-through side: the carrier
/// operand was marginal there, and the other operand is the identity over
/// it, so a node of the other holds one pair per structural ref. A node of
/// the level is the product of a carrier node, no two of whose pairs share
/// a structural ref, and a node of the other operand; its pairs are the
/// live products of their pairs, on the structural side the product nodes
/// of their refs. On an internal structural child, whose products are one
/// node per pair of operand nodes, distinct pairs of pairs meet in distinct
/// nodes. Not on a leaf child, where two label products can be one label;
/// nor on a child quantified, summed or marginal in an operand.
pub(super) fn fused_built_levels(
    vtree: &Vtree,
    run: &ApplyRun,
    f: &Tdd,
    g: &Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
) -> Vec<VtreeIdx> {
    use super::super::marginal_plan::Carrier;
    if !run.entry_marginality.any() {
        return Vec::new();
    }
    let marginal = |x: VtreeIdx| run.levels[x.idx()].is_marginal();
    let entry = &run.entry_marginality;
    run.cone
        .built
        .iter()
        .copied()
        .filter(|&t| {
            if marginal(t) || quantified.contains(t.idx()) {
                return false;
            }
            let (left, right) = vtree.children(t);
            let (m, c) = match (marginal(left), marginal(right)) {
                (true, false) => (left, right),
                (false, true) => (right, left),
                _ => return false,
            };
            let (carrier, identity) = match entry.only(m.idx()) {
                Some(Carrier::F) => (f, &run.g_identity[..]),
                Some(Carrier::G) => (g, &run.f_identity[..]),
                None => return false,
            };
            identity[m.idx()]
                && carrier.dirty.is_fused(t.0)
                && !vtree.node(c).is_leaf()
                && !quantified.contains(c.idx())
                && !targets.contains(c.idx())
                && !entry.either(c.idx())
        })
        .collect()
}

/// Whether a plain conjunction gave no two nodes of an inner child of the
/// level `t` it built equal contexts, where its operands' nodes there had
/// none. Where one operand is constant true over a child, every pair of
/// the other operand's nodes meets it there in a live product, so a node of
/// the product on the other side, the product of an `f` node and a `g` node,
/// has for its contexts every pair of a context of its `f` node with a
/// context of its `g` node, each of the product node of their parents and
/// the product of their siblings. Two product nodes with equal contexts are
/// then products of operand nodes with equal contexts, which an operand
/// that does not owe the contraction the level has only for one node. A
/// leaf child has no inner twins.
fn adds_no_twins(
    vtree: &Vtree,
    f_identity: &[bool],
    g_identity: &[bool],
    t: VtreeIdx,
) -> bool {
    let (left, right) = vtree.children(t);
    let side = |child: VtreeIdx, sibling: VtreeIdx| {
        vtree.node(child).is_leaf() || f_identity[sibling.idx()] || g_identity[sibling.idx()]
    };
    side(left, right) && side(right, left)
}

/// List the levels a plain conjunction built ([`SweepLists::collect_built`])
/// and split them into those it owes the twin contraction and those it does
/// not. It is owed the built levels under which the conjunction can have
/// given two nodes equal contexts ([`adds_no_twins`]); the leaf contraction
/// every built level. A level built as an operand holds it gives the nodes
/// of its children the operand's contexts there, which the operand, not
/// owing the contraction the level, holds once each; where it owes it, so
/// does the result.
///
/// [`SweepLists::collect_built`]: super::super::setup::SweepLists::collect_built
pub(super) fn split_built(vtree: &Vtree, run: &mut ApplyRun) {
    run.lists.collect_built(run.cone.built);
    let lists = &mut *run.lists;
    for &t in &lists.built {
        let twin_free = lists
            .as_operand
            .binary_search_by_key(&t.idx(), |e| e.0)
            .is_ok()
            || adds_no_twins(vtree, &run.f_identity[..], &run.g_identity[..], t);
        if twin_free {
            lists.twin_free.push(t);
        } else {
            lists.owed_twins.push(t);
        }
    }
}
