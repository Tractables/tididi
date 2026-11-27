//! What a plain conjunction's result owes the reductions after it, as lists
//! of levels each reduction reads: the loose levels a prune has to walk down
//! to ([`loose_levels`], [`Dirty::loose`](crate::diagram::Dirty::loose)),
//! the built levels owed the twin contraction ([`split_built`]), and the
//! built levels pair fusion would find nothing at ([`fused_built_levels`]).

use crate::apply::conjoin::held::Held;
use super::super::setup::{ApplyRun, Operands, VtreeMask};
use super::region::carried_from_f;
use crate::diagram::{LooseLevels, TddLevel};
use crate::vtree::{Vtree, VtreeIdx};

/// The internal levels below the root of a plain conjunction's result that
/// may hold a node no pair of their parent level names, given the operands'
/// own such levels ([`Dirty::loose`](crate::diagram::Dirty::loose)), and
/// which operand each level the sweep carried as an
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
/// A level `c` whose parent `p` is a pass level over an internal passed
/// child `c'`
/// ([`SweepLists::pass_levels`](super::super::setup::SweepLists::pass_levels))
/// is tight under (1) alone where `c` is `c'`, and where `c` is the other
/// child and each carrier pair met every passing node in a live product.
/// Proof. The passing operand's level `p` holds the identity over its
/// nodes at `c'`, and it is constant true under the other child. A node of
/// `c'` is a live product of carrier node `a` and passing node `b`; the
/// carrier's level `p` has a pair `(a, y)` in a node `j`, and the passing
/// operand's level `p` a pair `(b, ·)` in a node `i`, so the cell `(i, j)`
/// holds that product with `y`. A node of the other child is a carrier
/// node `y` there, with a pair `(a, y)` of the carrier's level `p` in a
/// node `j`, and the cell `(i, j)` holds `y` with `a`'s product with `i`'s
/// node, which is live where none was lost.
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
/// A level of `chain`, the chain above the cone the sweep built
/// ([`Cone::chain`]), is a built level, though the plan leaves it free in
/// `g`; a level the sweep stopped short of is free in `g`, and taken from
/// `f`, with what it holds.
///
/// A level keeps what an operand's entry counted of it
/// ([`Engine::unnamed_nodes`](crate::Engine::unnamed_nodes)) where its
/// nodes, their pairs and its parent's pairs are that operand's: a level
/// free, or moved untouched, under a free one taken from that operand, and
/// a level moved untouched or taken free from the operand its parent was
/// moved untouched from. A level the sweep built under a free one is
/// uncounted: its parent names the nodes it named, but their pairs are the
/// sweep's. Every other level is uncounted.
///
/// [`Cone::chain`]: super::super::setup::Cone::chain
pub(super) fn loose_levels(
    vtree: &Vtree,
    run: &ApplyRun<'_, '_>,
    chain: &[VtreeIdx],
    operands: Operands<&LooseLevels>,
) -> LooseLevels {
    let loose_in = |c: VtreeIdx, by: u8| match by {
        1 => operands.f.contains(c.0),
        _ => operands.g.contains(c.0),
    };
    let internal = |t: VtreeIdx| !vtree.node(t).is_leaf();
    let cone = run.cone;
    let taken_free = |t: VtreeIdx| cone.free_at(t.idx()) && !chain.contains(&t);
    let mut loose = LooseLevels::over(vtree.num_nodes());
    for (by, list) in [(1, operands.f), (2, operands.g)] {
        for (c, unnamed) in list.entries() {
            let under_free = vtree
                .node(VtreeIdx(c))
                .parent()
                .is_some_and(|p| taken_free(p) && (carried_from_f(cone, p.idx()) == (by == 1)));
            if under_free && internal(VtreeIdx(c)) {
                let untouched = taken_free(VtreeIdx(c)) || run.lists.identity_carrier(c as usize) == by;
                loose.insert_counted(c, unnamed.filter(|_| untouched));
            }
        }
    }
    for levels in [cone.built, chain] {
        for &p in levels {
            let (left, right) = vtree.children(p);
            let by = run.lists.identity_carrier(p.idx());
            if by != 0 {
                let carrier = if by == 1 { operands.f } else { operands.g };
                for c in [left, right] {
                    if internal(c) && loose_in(c, by) {
                        let moved_too = run.lists.identity_carrier(c.idx()) == by
                            || (taken_free(c) && carried_from_f(cone, c.idx()) == (by == 1));
                        loose.insert_counted(c.0, carrier.unnamed(c.0).filter(|_| moved_too));
                    }
                }
                continue;
            }
            let built = !run.levels[p.idx()].nodes().is_empty();
            let tight_in = |c: VtreeIdx, by: u8, widths: &[usize]| {
                !loose_in(c, by) || (widths[c.idx()] == 1 && built)
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
            let pass = run
                .lists
                .pass_levels
                .binary_search_by_key(&p.idx(), |e| e.0)
                .ok()
                .map(|i| run.lists.pass_levels[i]);
            let named_as_operand = |c: VtreeIdx| {
                as_operand.is_some_and(|by| {
                    let widths = if by == 1 {
                        &run.f_widths[..]
                    } else {
                        &run.g_widths[..]
                    };
                    !loose_in(c, by) && run.levels[c.idx()].node_count() == widths[c.idx()]
                })
            };
            for (c, s) in [(left, right), (right, left)] {
                if !internal(c) || named_as_operand(c) {
                    continue;
                }
                let tight_f = tight_in(c, 1, run.f_widths);
                let named = match pass {
                    Some((_, passed, dropped)) => c == passed || !dropped,
                    None => kills_none(s),
                };
                if tight_f && tight_in(c, 2, run.g_widths) && named {
                    continue;
                }
                if tight_f && held && !internal(s) {
                    super::super::note_held_tight();
                    continue;
                }
                loose.insert(c.0);
            }
        }
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
    f: &Held<'_>,
    g: &Held<'_>,
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
/// nor on a child summed or marginal in an operand.
pub(super) fn fused_built_levels(
    vtree: &Vtree,
    run: &ApplyRun,
    f: &Held<'_>,
    g: &Held<'_>,
    targets: VtreeMask<'_>,
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
            if marginal(t) {
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
                && carrier.dirty().is_fused(t.0)
                && !vtree.node(c).is_leaf()
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
/// A pass level over an internal passed child `c'`
/// ([`SweepLists::pass_levels`](super::super::setup::SweepLists::pass_levels))
/// adds no twins either where its other child is a vtree leaf or each
/// carrier pair met every passing node in a live product. Each live cell
/// `(i, j)`, passing node `i` with carrier node `j`, is its own node, as is
/// each live product at `c'` of a carrier node `a` with a passing node
/// `b`. A node `(a, b)` of `c'` has the contexts `((i, j), y)` for the
/// carrier's pairs `(a, y)` of `j` and the passing nodes `i` over `b`, lost
/// products or not. Where none was lost, a node `y` of the other child has
/// the contexts `((i, j), (a, b))` for the carrier's pairs `(a, y)` of `j`
/// and every `i`, `b` the node `i` is over. Two nodes with equal contexts
/// are then products of carrier nodes with equal contexts, and of one
/// passing node. On a leaf `c'` two labels can conjoin to one label, and
/// two carrier pairs then to one product. A node no pair names is no twin
/// here: the products of one with two nodes of the other operand share the
/// empty context, and a loose `c'` leaves them to the prune.
///
/// [`SweepLists::collect_built`]: super::super::setup::SweepLists::collect_built
pub(super) fn split_built(vtree: &Vtree, run: &mut ApplyRun, chain: &[VtreeIdx]) {
    run.lists.collect_built(run.cone.built, chain);
    let lists = &mut *run.lists;
    for &t in &lists.built {
        let twin_free = lists
            .as_operand
            .binary_search_by_key(&t.idx(), |e| e.0)
            .is_ok()
            || lists.pass_levels.binary_search_by_key(&t.idx(), |e| e.0).is_ok_and(|i| {
                let (_, passed, dropped) = lists.pass_levels[i];
                let (left, right) = vtree.children(t);
                !dropped || vtree.node(if passed == left { right } else { left }).is_leaf()
            })
            || adds_no_twins(vtree, &run.f_identity[..], &run.g_identity[..], t);
        if twin_free {
            lists.twin_free.push(t);
        } else {
            lists.owed_twins.push(t);
        }
    }
}
