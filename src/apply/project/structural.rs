//! The structural existential forget: rewrite x's leaf-to-root path in place,
//! never calling apply or negate.

use crate::diagram::Changed;
use crate::reduce::minimize;
use crate::diagram::{InputPair, NodeIdx, Tdd};
use crate::diagram::sort_pairs;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};

// `project_var_structural` computes ∃x.T by rewriting only the leaf-to-root path of
// x, in place, never calling apply/negate. It is therefore safe on marginal
// sibling levels (mc mode), where the cofactor-OR `project_var` crashes.
//
// It exploits the diagram **global partition property**:
// distinct nodes at any vtree level are pairwise mutually exclusive. So distinct
// sibling-child references in a pair list are mutex, and ∃x is a pure structural
// regrouping — no Boolean apply is ever needed.
//
// Path levels are processed leaf→root. At each level we:
//   • substitute the path-side child reference (toward x) by its forgotten image
//     — at the leaf-parent this turns Pos/Neg/One into One and groups by owner;
//     at higher levels it replaces a child index `c` by `child_remap[c]`;
//   • re-establish the partition by merging any nodes that now share an identical
//     atom (same path-image and same sibling ref), deduping atoms inside a node.
// The resulting per-level node remap feeds the next level up. Sibling refs are
// copied verbatim and never dereferenced, so marginal sibling levels are safe.

use std::collections::HashMap;

/// Per-level fan-out map: `remap[old_node_idx]` lists every new node index that
/// the old node contributes to after the ∃x regroup. Multi-valued because
/// forgetting x can split one old node's sibling refs across several new
/// partition cells (owner classes); the level above re-expands a reference to
/// `old` over all listed new cells.
type Remap = Vec<Vec<u32>>;

/// Existentially quantify variable `x` from diagram `t` by an in-place leaf-to-root
/// rewrite. Tolerates marginal sibling levels, which the cofactor rewrite does
/// not. Returns a fully minimized diagram.
///
/// Precondition: `x` is a leaf in `t.vtree`, and no ancestor of x's leaf is a
/// marginal level (an already-counted-out ancestor would make ∃x ill-defined).
/// Marginal levels in disjoint sub-vtrees (siblings along the path, or unrelated
/// subtrees) are permitted and left byte-identical.
///
/// # Panics
///
/// Panics if `x` is not a variable present in `t.vtree`.
pub(super) fn project_var_structural(t: &Tdd, x: VarId) -> Tdd {
    if t.is_zero() {
        return t.clone();
    }
    let vtree = &t.vtree;
    assert!(
        x.idx() < vtree.num_vars() as usize,
        "project_var_structural: variable {:?} is not in the vtree (var_to_leaf len={})",
        x,
        vtree.num_vars()
    );
    let leaf_idx = vtree.leaf_of(x).expect("the vtree carries this variable");
    assert!(
        vtree.node(leaf_idx).is_leaf(),
        "project_var_structural: var_to_leaf[{:?}] = {:?} is not a leaf node",
        x,
        leaf_idx
    );

    // Single-var vtree / output at the leaf: ∃x.F = constant_one.
    if t.output.vtree == leaf_idx {
        return Tdd::one(&t.vtree);
    }
    assert_path_is_rewritable(t, x, leaf_idx);

    let mut tdd = t.clone();
    let path = ancestor_path(vtree, leaf_idx);
    let child_remap = rewrite_path(&mut tdd, &path, leaf_idx);

    let root_vi = *path.last().expect("path is non-empty (output not at leaf)");
    let out_pairs = union_of_root_cells(&tdd, root_vi, &child_remap[tdd.output.local.idx()]);
    // Append the union node and point the output at it (prune drops the rest).
    let new_out = tdd.levels[root_vi.idx()].push_internal_node(&out_pairs);
    tdd.output.local = new_out;
    tdd.invalidate(root_vi, Changed::PAIRS);

    minimize(&mut tdd);
    tdd
}

/// The two preconditions on the leaf→root path this rewrite touches.
///
/// (1) No ancestor of x's leaf may be marginal (it would mean x was already
///     counted out). A marginal level hanging off the path as a sibling is
///     fine — we copy sibling refs verbatim and never dereference them
///     (`scoped_marginal_sibling_succeeds`).
///
/// (2) No ancestor may be the grandparent of a marginal level. A marginal
///     level's parent is a "boundary parent", and the boundary content-twin
///     merge (`reduce::contract::content_twin`) merges content-equal nodes
///     there and repoints the grandparent's refs at the survivor — which can
///     leave the same (left,right) pair twice in a grandparent node. Duplicate
///     pairs are legal, count-carrying multiset entries, but the owner-class
///     regroup below indexes sibling refs into owner sets (`OwnerKey` here, the
///     `owners` Vec in `regroup_internal`) which cannot represent multiplicity,
///     so a duplicate landing on a rewritten level would be silently folded to
///     one — a miscount, not a crash. Depth ≥3 marginals are harmless: their
///     duplicates land inside a sibling subtree we only copy refs into.
///
/// Production cannot build the (2) shape, so this is a contract check, not a
/// live guard: the downstream driver's projected-sibling shield skip-set
/// shields every un-forgotten projected var's whole ancestor path and every
/// path-sibling subtree from streaming marginalization; path + path-siblings
/// cover the entire vtree, so nothing marginalizes at all while any projected
/// var is still un-forgotten, and the forget fires before the leaf's own
/// marginalize step.
///
/// # Panics
///
/// Panics if either precondition is violated.
fn assert_path_is_rewritable(t: &Tdd, x: VarId, leaf_idx: VtreeIdx) {
    let vtree = &t.vtree;
    let mut anc = vtree.node(leaf_idx).parent();
    while let Some(ai) = anc {
        assert!(
            !t.levels[ai.idx()].is_marginal(),
            "project_var_structural: variable {:?} has a marginal ancestor at {:?}",
            x,
            ai
        );
        let (al, ar) = vtree.children(ai);
        for c in [al, ar] {
            if vtree.node(c).is_leaf() {
                continue;
            }
            let (gl, gr) = vtree.children(c);
            for g in [gl, gr] {
                assert!(
                    !t.levels[g.idx()].is_marginal(),
                    "project_var_structural: variable {:?} — rewritten ancestor {:?} is the \
                     grandparent of marginal level {:?}; the boundary content-twin merge \
                     can mint duplicate pairs there and the owner-class regroup folds \
                     them (silent miscount)",
                    x,
                    ai,
                    g
                );
            }
        }
        anc = vtree.node(ai).parent();
    }
}

/// The leaf→root ancestor path: `[leaf_parent, grandparent, …, root]`.
fn ancestor_path(vtree: &crate::vtree::Vtree, leaf_idx: VtreeIdx) -> Vec<VtreeIdx> {
    let mut path = Vec::new();
    let mut cur = vtree.node(leaf_idx).parent();
    while let Some(p) = cur {
        path.push(p);
        cur = vtree.node(p).parent();
    }
    path
}

/// Regroup every level on `path`, leaf-parent first, and return the root
/// level's fan-out map.
///
/// Each step consumes the level below's map — which new cells the child's old
/// nodes became — and produces its own for the level above. The leaf-parent
/// step is the special one: its "child" is x's own leaf, which has no map.
fn rewrite_path(tdd: &mut Tdd, path: &[VtreeIdx], leaf_idx: VtreeIdx) -> Remap {
    let vtree = tdd.vtree.clone();
    let mut child_remap: Remap = Vec::new();
    let mut child_vi = leaf_idx;
    for (step, &parent) in path.iter().enumerate() {
        let (left_child, right_child) = match *vtree.node(parent) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            _ => unreachable!("path node must be internal"),
        };
        let path_is_left = left_child == child_vi;
        debug_assert!(path_is_left || right_child == child_vi);

        child_remap = if step == 0 {
            regroup_leaf_parent(tdd, parent, path_is_left)
        } else {
            regroup_internal(tdd, parent, path_is_left, &child_remap)
        };
        child_vi = parent;
    }
    child_remap
}

/// The pairs of ∃x.f's single output node: the deduped union of the root cells
/// the old output fanned out into.
///
/// The cells are mutex among themselves and each is a valid
/// deterministic/decomposable pair list, so their union is a sound single root
/// node. The other (unreferenced) root cells are dropped by `minimize`'s prune.
fn union_of_root_cells(tdd: &Tdd, root_vi: VtreeIdx, out_cells: &[u32]) -> Vec<InputPair> {
    let level = &tdd.levels[root_vi.idx()];
    let mut out_pairs: Vec<InputPair> = Vec::new();
    for &k in out_cells {
        // `k` indexes the freshly-written root level; copy its pairs. Pushing
        // unconditionally and deduping once is what keeps this linear: a
        // `contains` guard never matched (the cells are mutex) yet re-scanned
        // the growing union per pair, which dominated ∃-forget self-time on
        // wide-fanout outputs.
        out_pairs.extend(level.pairs_iter_of_idx(k as usize));
    }
    sort_pairs(&mut out_pairs);
    out_pairs.dedup();
    out_pairs
}

/// The (pos-owner, neg-owner) old-node indices for a single sibling ref at the
/// leaf parent. `u32::MAX` means "no owner on that polarity".
///
/// One owner per polarity, so this cannot carry a pair's multiplicity: two
/// copies of the same `(x_label, sib)` pair collapse to one owner entry. Sound
/// only under `project_var_structural`'s precondition (2) — no rewritten level is
/// the grandparent of a marginal level — which excludes the boundary
/// content-twin merge's duplicate pairs from every level this rewrites.
#[derive(Copy, Clone)]
struct OwnerKey {
    pos: u32,
    neg: u32,
}

/// Forget the leaf on `path_is_left`'s side at the leaf-parent level `parent`.
///
/// Each pair `(x_label, sib)` has `x_label ∈ {Pos, Neg, One}` on the leaf side
/// and `sib` the other-side ref. For each distinct `sib` we record its
/// pos-owner (the node whose pair is `(Pos, sib)`) and neg-owner (`(Neg, sib)`);
/// a `(One, sib)` owns both. We then group sibling refs by their unordered owner
/// pair into new partition cells, each holding pairs `{(One, sib)}`. Returns the
/// fan-out `Remap`: each old node → the new cells it contributed a sibling ref
/// to.
fn regroup_leaf_parent(tdd: &mut Tdd, parent: VtreeIdx, path_is_left: bool) -> Remap {
    let level = &tdd.levels[parent.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return Vec::new();
    }

    let read_pair = |p: &InputPair| -> (NodeIdx, NodeIdx) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // Per-sibling-ref owner pair, in first-seen order.
    let mut owners: HashMap<u32, OwnerKey> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();

    for i in 0..n_nodes {
        let mut handle = |x_label: NodeIdx, sib: NodeIdx| {
            let e = owners.entry(sib.0).or_insert_with(|| {
                order.push(sib.0);
                OwnerKey { pos: u32::MAX, neg: u32::MAX }
            });
            if x_label == POS_LEAF_IDX {
                e.pos = i as u32;
            } else if x_label == NEG_LEAF_IDX {
                e.neg = i as u32;
            } else {
                // One: x already irrelevant for this sib — owned on both sides.
                e.pos = i as u32;
                e.neg = i as u32;
            }
        };
        for p in level.pairs_of_idx(i) {
            let (xl, sib) = read_pair(p);
            handle(xl, sib);
        }
    }

    // Group sibling refs by their unordered owner key → one new cell each.
    //
    // The key is the cell's owner set (its two entries, minus the `u32::MAX`
    // "no owner" slot), so the fan-out is recorded once at cell creation rather
    // than into a per-cell member set that is inverted afterwards. Cells are
    // created in increasing index order, so each old node's fan-out list still
    // comes out ascending — the same `Remap` the inversion produced.
    let mut key_to_new: HashMap<(u32, u32), usize> = HashMap::new();
    let mut new_nodes: Vec<Vec<InputPair>> = Vec::new();
    let mut remap: Remap = vec![Vec::new(); n_nodes];

    for &sib in &order {
        let ok = owners[&sib];
        let (a, b) = (ok.pos, ok.neg);
        let key = if a <= b { (a, b) } else { (b, a) };
        let idx = *key_to_new.entry(key).or_insert_with(|| {
            let k = new_nodes.len();
            new_nodes.push(Vec::new());
            // `u32::MAX` sorts last, so key.0 is a real owner whenever the cell
            // has one; the `!= key.0` guard is the set's dedup for a both-sides
            // owner (a `(One, sib)` pair, where pos == neg).
            if key.0 != u32::MAX {
                remap[key.0 as usize].push(k as u32);
            }
            if key.1 != u32::MAX && key.1 != key.0 {
                remap[key.1 as usize].push(k as u32);
            }
            k
        });
        let pair = if path_is_left {
            InputPair { left: ONE_LEAF_IDX, right: NodeIdx(sib) }
        } else {
            InputPair { left: NodeIdx(sib), right: ONE_LEAF_IDX }
        };
        // `order` holds distinct sibs and the pair is injective in `sib`, so
        // within a cell every pushed pair is already distinct — no dedup is
        // needed here (`write_level` dedups defensively). A linear `!contains`
        // guard would scan the whole growing cell for a match that cannot
        // happen, which is quadratic in cell width.
        new_nodes[idx].push(pair);
    }

    write_level(tdd, parent, &mut new_nodes);
    remap
}

/// Regroup an internal path level `parent` after the level below it was forgotten.
///
/// Each old pair `(c, sib)` has its path-side child `c` expanded via
/// `child_remap[c]` into new child cells. The expanded atom `(Pc, sib)` (Pc a
/// new child cell) is then re-partitioned by the **owner-set** rule — the exact
/// generalization of the leaf-parent owner-pair grouping:
///
///   For each distinct atom `(Pc, sib)`, its owner set is `{old L-node g : g
///   contributes atom (Pc, sib)}`. A new cell ↔ a distinct owner set; its pairs
///   are all atoms sharing that owner set. `remap[g]` = the new cells whose
///   owner set contains `g` (so a reference to `g` from above ∃x-expands to the
///   disjunction of exactly those cells, = ∃x.g).
///
/// This keeps the new level a valid partition: two atoms with different owner
/// sets land in different cells (mutex by construction of the owner set), and
/// `∃x.g` is reconstructed exactly as the OR over `g`'s cells.
fn regroup_internal(
    tdd: &mut Tdd,
    parent: VtreeIdx,
    path_is_left: bool,
    child_remap: &Remap,
) -> Remap {
    let level = &tdd.levels[parent.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return Vec::new();
    }

    let read_pair = |p: &InputPair| -> (NodeIdx, NodeIdx) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // For each expanded atom (Pc, sib), accumulate its owner set (the old L-nodes
    // contributing it). First-seen order kept for determinism. The owner set is a
    // sorted `Vec<u32>` rather than a `BTreeSet`: the outer loop pushes `i` in
    // strictly non-decreasing order (within one `i`, repeated pushes of the same
    // value are dropped by the `last()` guard), so the Vec is sorted+unique by
    // construction — identical content to a BTreeSet but with O(1) amortized push
    // and one allocation per set instead of a tree node per element.
    let mut atom_owners: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut atom_order: Vec<(u32, u32)> = Vec::new();

    for i in 0..n_nodes {
        let mut handle = |path_child: NodeIdx, sib: NodeIdx| {
            for &cell in &child_remap[path_child.idx()] {
                let key = (cell, sib.0);
                let owners = atom_owners.entry(key).or_insert_with(|| {
                    atom_order.push(key);
                    Vec::new()
                });
                // Owner sets, not multisets: if node `i` reaches the same
                // `(cell, sibling)` atom twice — which a marginalized diagram's
                // multiset pair list permits — the second arrival is dropped. That is the intended ∃-forget
                // semantics (projection is an OR; a projection with two witnesses
                // is still one projection), but it does mean *any* multiplicity a
                // duplicate pair carried in the count dimension is not preserved
                // across this rewrite. `OwnerKey` in `regroup_leaf_parent` cannot
                // represent multiplicity at all. That is sound only because no
                // duplicate pair can reach a rewritten level: `project_var_structural`
                // asserts precondition (2) — no rewritten ancestor is the
                // grandparent of a marginal level — which is exactly where the
                // boundary content-twin merge mints duplicates. See the
                // precondition block there.
                if owners.last() != Some(&(i as u32)) {
                    owners.push(i as u32);
                }
            }
        };
        for p in level.pairs_of_idx(i) {
            let (pc, sib) = read_pair(p);
            handle(pc, sib);
        }
    }

    // Group atoms by owner set → one new cell per distinct owner set. The owner
    // Vec is already sorted+unique, so it is the canonical hashmap key directly.
    let mut key_to_new: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut new_nodes: Vec<Vec<InputPair>> = Vec::new();
    let mut cell_owners: Vec<Vec<u32>> = Vec::new();

    for atom in &atom_order {
        let owners = &atom_owners[atom];
        let key: Vec<u32> = owners.clone();
        let idx = *key_to_new.entry(key).or_insert_with(|| {
            new_nodes.push(Vec::new());
            cell_owners.push(owners.clone());
            new_nodes.len() - 1
        });
        let (cell, sib) = *atom;
        let pair = if path_is_left {
            InputPair { left: NodeIdx(cell), right: NodeIdx(sib) }
        } else {
            InputPair { left: NodeIdx(sib), right: NodeIdx(cell) }
        };
        // `atom_order` holds distinct (cell, sib) atoms and the pair is
        // injective in the atom, so within a cell every pushed pair is already
        // distinct — no dedup needed (`write_level` dedups defensively). A
        // linear `!contains` guard would scan the whole growing cell for a
        // match that cannot happen, which is quadratic in cell width.
        new_nodes[idx].push(pair);
    }

    // Invert: remap[g] = new cells whose owner set contains g.
    let mut remap: Remap = vec![Vec::new(); n_nodes];
    for (k, owners) in cell_owners.iter().enumerate() {
        for &g in owners {
            remap[g as usize].push(k as u32);
        }
    }

    write_level(tdd, parent, &mut new_nodes);
    remap
}

/// Replace level `parent`'s nodes with `new_nodes` (each a pair list), sorting each
/// pair list canonically, and mark the level dirty for contraction.
fn write_level(tdd: &mut Tdd, parent: VtreeIdx, new_nodes: &mut [Vec<InputPair>]) {
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    for pairs in new_nodes.iter_mut() {
        level.push_internal_node_canonical(pairs);
    }
    tdd.invalidate(parent, Changed::PAIRS);
}
