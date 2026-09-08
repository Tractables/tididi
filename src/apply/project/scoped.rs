//! Scoped in-place existential forget (no apply/negate) plus the caller-supplied
//! projected-leaf scope state read by the compile orchestrator. Cut verbatim
//! from the former `project.rs`.

use crate::scoped::Scoped;
use crate::build::constant_one;
use crate::reduce::minimize;
use crate::diagram::{InputPair, LocalNodeIdx, Tdd};
use crate::utils::sort_pairs;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

use super::{POS, NEG, ONE};

// ── Caller-supplied projected-leaf scope (installed by compile, read here) ────
//
// Projection-at-marginalization: set of LOCAL VarIds (within the current
// component vtree) to project before their leaf-parents are marginalized.
// Outer scope stores GLOBAL VarIds; per-component scoping translates to local.
// Set via `ScopedProjectLeaves::new` (RAII guard); read by the leaf-marginalize
// shortcuts (`caller_projection_active`) and by the empty-formula count fast
// paths. The per-forget bookkeeping
// counters `PROJECT_APPLIED_COUNT` / `PROJECT_FORGOTTEN_SCOPED` live here too
// (P2-cfg: the guard reads/resets them, so the state is owned where it is
// read); the compile orchestrator (in the downstream driver crate) writes
// them during the forget schedule via direct imports.
thread_local! {
    /// The global projected/show variable set (`Some` while a
    /// [`ScopedProjectLeaves`] guard is live), consulted by the marginalize
    /// schedule to ∃-forget these vars during compile.
    #[doc(hidden)]
    pub static PROJECT_LEAF_IDXS_SCOPED: std::cell::RefCell<Option<std::collections::HashSet<VarId>>>
        = const { std::cell::RefCell::new(None) };
    /// Number of projected vars actually ∃-forgotten so far in the innermost
    /// active projection scope. Reset by `ScopedProjectLeaves::new`;
    /// incremented by the forget schedule (in the downstream driver crate);
    /// read back via `ScopedProjectLeaves::applied_count`.
    ///
    /// Re-based per component by the downstream driver's per-component guard
    /// (zeroed on install, folded back into the enclosing total on drop), so a
    /// reader under that guard sees ONE component's count and a reader outside
    /// every guard sees the run total. The streaming counter's per-component
    /// `2^k` divisor depends on that: without it, component *n* was divided by
    /// the forgets of components 1..*n*.
    #[doc(hidden)]
    pub static PROJECT_APPLIED_COUNT: std::cell::Cell<usize>
        = const { std::cell::Cell::new(0) };
    /// LOCAL VarIds already ∃-forgotten during the current component compile.
    /// Lets the driver's per-step marginalize forget each projected var exactly once: at its
    /// clause-scope internal step if that is internal, else (unit/free vars
    /// whose scope is a leaf-step that the main loop never visits) as a
    /// backstop when its leaf-parent internal step is processed. Reset
    /// whenever the projected set is (re)installed.
    #[doc(hidden)]
    pub static PROJECT_FORGOTTEN_SCOPED: std::cell::RefCell<std::collections::HashSet<VarId>>
        = std::cell::RefCell::new(std::collections::HashSet::new());
}

/// RAII guard: installs the global projected/show variable set into
/// `PROJECT_LEAF_IDXS_SCOPED` (consulted by the marginalize schedule to ∃-forget
/// these vars during compile) and resets the per-run `PROJECT_APPLIED_COUNT` and
/// `PROJECT_FORGOTTEN_SCOPED` trackers. Restores the previous slot on drop.
#[doc(hidden)]
pub struct ScopedProjectLeaves(
    #[allow(dead_code)] Scoped<std::cell::RefCell<Option<std::collections::HashSet<VarId>>>>,
);

impl ScopedProjectLeaves {
    /// Install `vars` as the projected set and reset the per-run trackers; the
    /// previous set is restored when the returned guard drops.
    pub fn new(vars: std::collections::HashSet<VarId>) -> Self {
        PROJECT_APPLIED_COUNT.with(|c| c.set(0));
        PROJECT_FORGOTTEN_SCOPED.with(|c| c.borrow_mut().clear());
        Self(Scoped::install(&PROJECT_LEAF_IDXS_SCOPED, Some(vars)))
    }

    /// Number of ∃-projections applied since the current scope was installed.
    pub fn applied_count() -> usize {
        PROJECT_APPLIED_COUNT.with(|c| c.get())
    }
}

/// True when a caller has installed a GLOBAL projected set (`ScopedProjectLeaves`)
/// — i.e. the streaming engine is being driven for PROJECTED counting. Used by
/// the dedup variant to bail out of signature-grouping: signature-identical local
/// components can map to DIFFERENT global show/projected partitions, so a single
/// representative's `^group_size` would be unsound under caller-supplied
/// projection. When this is true, dedup delegates to the per-component
/// (non-grouped) streaming path.
#[doc(hidden)]
pub fn caller_projection_active() -> bool {
    PROJECT_LEAF_IDXS_SCOPED.with(|c| c.borrow().is_some())
}

/// For an empty (no-clause) formula every variable is free and contributes ×2.
/// Under a caller-supplied projected set, projected vars ∃-away to factor 1
/// (∃x.true = true), so only NON-projected vars contribute ×2. Returns the count
/// of free non-projected vars in `0..num_vars` — equal to `num_vars` when no
/// projection is installed (plain MC / per-component gate paths), so the
/// empty-formula shortcut stays `2^num_vars` there.
#[doc(hidden)]
pub fn free_nonprojected_count(num_vars: u32) -> usize {
    PROJECT_LEAF_IDXS_SCOPED.with(|c| {
        let guard = c.borrow();
        match guard.as_ref() {
            Some(proj) => (0..num_vars).filter(|i| !proj.contains(&VarId(*i))).count(),
            None => num_vars as usize,
        }
    })
}

// ── Scoped in-place existential forget (no apply/negate) ───────────────────────
//
// `project_var_scoped` computes ∃x.T by rewriting only the leaf-to-root path of
// x, in place, never calling apply/negate. It is therefore safe on marginal
// SIBLING levels (mc mode), where the cofactor-OR `project_var` crashes.
//
// It exploits the TDD **global partition property**:
// distinct nodes at any vtree level are pairwise mutually exclusive. So distinct
// sibling-child references in a pair list are mutex, and ∃x is a pure structural
// regrouping — no Boolean apply is ever needed.
//
// Path levels are processed leaf→root. At each level we:
//   • substitute the PATH-side child reference (toward x) by its forgotten image
//     — at the leaf-parent this turns Pos/Neg/One into One and groups by owner;
//     at higher levels it replaces a child index `c` by `child_remap[c]`;
//   • re-establish the partition by merging any nodes that now share an identical
//     atom (same path-image AND same sibling ref), deduping atoms inside a node.
// The resulting per-level node remap feeds the next level up. Sibling refs are
// copied verbatim and NEVER dereferenced, so marginal sibling levels are safe.

use std::collections::HashMap;

/// Per-level fan-out map: `remap[old_node_idx]` lists every NEW node index that
/// the old node contributes to after the ∃x regroup. Multi-valued because
/// forgetting x can split one old node's sibling refs across several new
/// partition cells (owner classes); the level above re-expands a reference to
/// `old` over ALL listed new cells.
type Remap = Vec<Vec<u32>>;

/// Existentially quantify variable `x` from TDD `t` by an in-place leaf-to-root
/// rewrite (no apply/negate). Sound drop-in for [`project_var`](super::project_var) that additionally
/// tolerates marginal sibling levels (mc mode). Returns a fully minimized TDD.
///
/// Precondition: `x` is a leaf in `t.vtree`, and no ANCESTOR of x's leaf is a
/// marginal level (an already-counted-out ancestor would make ∃x ill-defined).
/// Marginal levels in DISJOINT sub-vtrees (siblings along the path, or unrelated
/// subtrees) are permitted and left byte-identical.
///
/// # Panics
///
/// Panics if `x` is not a variable present in `t.vtree`.
#[doc(hidden)]
pub fn project_var_scoped(t: &Tdd, x: VarId) -> Tdd {
    if t.is_zero() {
        return t.clone();
    }
    let vtree = &t.vtree;
    assert!(
        x.idx() < vtree.num_vars() as usize,
        "project_var_scoped: variable {:?} is not in the vtree (var_to_leaf len={})",
        x,
        vtree.num_vars()
    );
    let leaf_idx = vtree.leaf_of(x);
    assert!(
        vtree.node(leaf_idx).is_leaf(),
        "project_var_scoped: var_to_leaf[{:?}] = {:?} is not a leaf node",
        x,
        leaf_idx
    );

    // Single-var vtree / output at the leaf: ∃x.F = constant_one.
    if t.output.vtree == leaf_idx {
        return constant_one(&t.vtree);
    }

    // Two preconditions on the leaf→root path this rewrite touches.
    //
    // (1) No ancestor of x's leaf may be marginal (it would mean x was already
    //     counted out). A marginal level hanging off the path as a SIBLING is
    //     fine — we copy sibling refs verbatim and never dereference them
    //     (`scoped_marginal_sibling_succeeds`).
    //
    // (2) No ancestor may be the GRANDPARENT of a marginal level. A marginal
    //     level's parent is a "boundary parent", and the boundary content-twin
    //     merge (`minimize::contract::content_twin`) merges content-equal nodes
    //     there and repoints the grandparent's refs at the survivor — which can
    //     leave the SAME (left,right) pair twice in a grandparent node. Duplicate
    //     pairs are legal, count-carrying multiset entries,
    //     but the owner-class regroup below indexes sibling refs into owner SETS
    //     (`OwnerKey` here, the `owners` Vec in `regroup_internal`) which cannot
    //     represent multiplicity, so a duplicate landing on a REWRITTEN level
    //     would be silently folded to one — a MISCOUNT, not a crash. Depth ≥3
    //     marginals are harmless: their duplicates land inside a sibling subtree
    //     we only copy refs into.
    //
    // Production cannot build the (2) shape, so this is a contract check, not a
    // live guard: the downstream driver's projected-sibling shield skip-set
    // shields every un-forgotten projected var's whole ancestor path AND every
    // path-sibling subtree from streaming marginalization; path + path-siblings
    // cover the entire vtree, so nothing marginalizes at all while any projected
    // var is still un-forgotten, and the forget fires before the leaf's own
    // marginalize step.
    let mut anc = vtree.node(leaf_idx).parent();
    while let Some(ai) = anc {
        assert!(
            !t.levels[ai.idx()].is_marginal(),
            "project_var_scoped: variable {:?} has a marginal ancestor at {:?}",
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
                    "project_var_scoped: variable {:?} — rewritten ancestor {:?} is the \
                     GRANDPARENT of marginal level {:?}; the boundary content-twin merge \
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

    let mut tdd = t.clone();

    // Build the leaf→root ancestor path: [leaf_parent, grandparent, …, root].
    let mut path: Vec<VtreeIdx> = Vec::new();
    let mut cur = vtree.node(leaf_idx).parent();
    while let Some(p) = cur {
        path.push(p);
        cur = vtree.node(p).parent();
    }

    // `child_remap` maps each old node index at the level BELOW the current one
    // to the list of new node indices it expanded into. For the leaf-parent step
    // the "child" is x's leaf, handled specially (no remap consumed there).
    let mut child_remap: Remap = Vec::new();
    let mut child_vi = leaf_idx;

    for (step, &pvi) in path.iter().enumerate() {
        let (left_child, right_child) = match *vtree.node(pvi) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            _ => unreachable!("path node must be internal"),
        };
        let path_is_left = left_child == child_vi;
        debug_assert!(path_is_left || right_child == child_vi);

        let remap = if step == 0 {
            regroup_leaf_parent(&mut tdd, pvi, path_is_left)
        } else {
            regroup_internal(&mut tdd, pvi, path_is_left, &child_remap)
        };

        child_remap = remap;
        child_vi = pvi;
    }

    // The output node fanned out into a set of new root nodes; ∃x.f is their
    // disjunction. Build ONE output node whose pairs are the union of those
    // cells' pairs (deduped). The cells are mutex among themselves and each is a
    // valid deterministic/decomposable pair list, so their union is a sound
    // single root node. The other (unreferenced) root cells are dropped by
    // `minimize`'s prune.
    let root_vi = *path.last().expect("path is non-empty (output not at leaf)");
    let out_cells = &child_remap[tdd.output.local.idx()];
    let mut out_pairs: Vec<InputPair> = Vec::new();
    {
        let level = &tdd.levels[root_vi.idx()];
        for &k in out_cells {
            // `k` indexes the freshly-written root level; copy its pairs.
            for p in level.pairs_iter_of_idx(k as usize) {
                // The out_cells are mutex among themselves (disjoint pair sets),
                // so the old `!out_pairs.contains(&p)` guard never matched — yet it
                // re-scanned the whole growing union per pair, an O(total_pairs²)
                // cost that dominated ∃-forget self-time on wide-fanout outputs
                // (~95% of project_vars_scoped on the 029-class gap). Push
                // unconditionally; the `sort_pairs` + `dedup` below produces the
                // identical deduped set in O(n log n), independent of mutex.
                out_pairs.push(p);
            }
        }
    }
    sort_pairs(&mut out_pairs);
    out_pairs.dedup();
    // Append the union node and point the output at it (prune drops the rest).
    let new_out = tdd.levels[root_vi.idx()].push_internal_node(&out_pairs);
    tdd.output.local = new_out;
    tdd.scratch.dirty_contract.push(root_vi.0);

    minimize(&mut tdd);
    tdd
}

/// Existentially quantify all variables in `vars` via [`project_var_scoped`].
#[doc(hidden)]
pub fn project_vars_scoped(t: &Tdd, vars: &[VarId]) -> Tdd {
    let mut result = t.clone();
    for &x in vars {
        result = project_var_scoped(&result, x);
    }
    result
}

/// The (pos-owner, neg-owner) old-node indices for a single sibling ref at the
/// leaf parent. `u32::MAX` means "no owner on that polarity".
///
/// One owner per polarity, so this cannot carry a pair's MULTIPLICITY: two
/// copies of the same `(x_label, sib)` pair collapse to one owner entry. Sound
/// only under `project_var_scoped`'s precondition (2) — no rewritten level is
/// the grandparent of a marginal level — which excludes the boundary
/// content-twin merge's duplicate pairs from every level this rewrites.
#[derive(Copy, Clone)]
struct OwnerKey {
    pos: u32,
    neg: u32,
}

/// Forget the leaf on `path_is_left`'s side at the leaf-parent level `pvi`.
///
/// Each pair `(x_label, sib)` has `x_label ∈ {Pos, Neg, One}` on the leaf side
/// and `sib` the other-side ref. For each distinct `sib` we record its
/// pos-owner (the node whose pair is `(Pos, sib)`) and neg-owner (`(Neg, sib)`);
/// a `(One, sib)` owns both. We then group sibling refs by their UNORDERED owner
/// pair into new partition cells, each holding pairs `{(One, sib)}`. Returns the
/// fan-out `Remap`: each old node → the new cells it contributed a sibling ref
/// to.
fn regroup_leaf_parent(tdd: &mut Tdd, pvi: VtreeIdx, path_is_left: bool) -> Remap {
    let level = &tdd.levels[pvi.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return Vec::new();
    }

    let read_pair = |p: &InputPair| -> (LocalNodeIdx, LocalNodeIdx) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // Per-sibling-ref owner pair, in first-seen order.
    let mut owners: HashMap<u32, OwnerKey> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();

    for i in 0..n_nodes {
        let mut handle = |x_label: LocalNodeIdx, sib: LocalNodeIdx| {
            let e = owners.entry(sib.0).or_insert_with(|| {
                order.push(sib.0);
                OwnerKey { pos: u32::MAX, neg: u32::MAX }
            });
            if x_label == POS {
                e.pos = i as u32;
            } else if x_label == NEG {
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

    // Group sibling refs by their UNORDERED owner key → one new cell each.
    //
    // The key IS the cell's owner set (its two entries, minus the `u32::MAX`
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
            InputPair { left: ONE, right: LocalNodeIdx(sib) }
        } else {
            InputPair { left: LocalNodeIdx(sib), right: ONE }
        };
        // `order` holds distinct sibs and the pair is injective in `sib`, so
        // within a cell every pushed pair is already distinct — no dedup is
        // needed here (`write_level` dedups defensively). The old linear
        // `!contains` guard never matched yet scanned the whole growing cell,
        // making it ~90% of ∃-forget self-time (quadratic in cell width).
        new_nodes[idx].push(pair);
    }

    write_level(tdd, pvi, &mut new_nodes);
    remap
}

/// Regroup an internal path level `pvi` after the level below it was forgotten.
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
    pvi: VtreeIdx,
    path_is_left: bool,
    child_remap: &Remap,
) -> Remap {
    let level = &tdd.levels[pvi.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return Vec::new();
    }

    let read_pair = |p: &InputPair| -> (LocalNodeIdx, LocalNodeIdx) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // For each expanded atom (Pc, sib), accumulate its owner set (the old L-nodes
    // contributing it). First-seen order kept for determinism. The owner set is a
    // sorted `Vec<u32>` rather than a `BTreeSet`: the outer loop pushes `i` in
    // strictly non-decreasing order (within one `i`, repeated pushes of the same
    // value are dropped by the `last()` guard), so the Vec is sorted+unique by
    // construction — identical content to a BTreeSet but with O(1) amortized push
    // and one allocation per set instead of a tree node per element. On the
    // structural ∃-forget path this BTreeSet insert/drop churn was ~18% of
    // self-time on owner-set-heavy levels.
    let mut atom_owners: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut atom_order: Vec<(u32, u32)> = Vec::new();

    for i in 0..n_nodes {
        let mut handle = |path_child: LocalNodeIdx, sib: LocalNodeIdx| {
            for &cell in &child_remap[path_child.idx()] {
                let key = (cell, sib.0);
                let owners = atom_owners.entry(key).or_insert_with(|| {
                    atom_order.push(key);
                    Vec::new()
                });
                // Owner SETS, not multisets: if node `i` reaches the same
                // `(cell, sibling)` atom twice — which a marginalized diagram's
                // multiset pair list permits — the second arrival is dropped. That is the intended ∃-forget
                // semantics (projection is an OR; a projection with two witnesses
                // is still one projection), but it does mean *any* multiplicity a
                // duplicate pair carried in the count dimension is not preserved
                // across this rewrite. `OwnerKey` in `regroup_leaf_parent` cannot
                // represent multiplicity at all. That is SAFE only because no
                // duplicate pair can reach a rewritten level: `project_var_scoped`
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
            InputPair { left: LocalNodeIdx(cell), right: LocalNodeIdx(sib) }
        } else {
            InputPair { left: LocalNodeIdx(sib), right: LocalNodeIdx(cell) }
        };
        // `atom_order` holds distinct (cell, sib) atoms and the pair is
        // injective in the atom, so within a cell every pushed pair is already
        // distinct — no dedup needed (`write_level` dedups defensively). The old
        // linear `!contains` guard never matched yet scanned the whole growing
        // cell: ~90% of ∃-forget self-time, quadratic in cell width.
        new_nodes[idx].push(pair);
    }

    // Invert: remap[g] = new cells whose owner set contains g.
    let mut remap: Remap = vec![Vec::new(); n_nodes];
    for (k, owners) in cell_owners.iter().enumerate() {
        for &g in owners {
            remap[g as usize].push(k as u32);
        }
    }

    write_level(tdd, pvi, &mut new_nodes);
    remap
}

/// Replace level `pvi`'s nodes with `new_nodes` (each a pair list), sorting each
/// pair list canonically, and mark the level dirty for contraction.
fn write_level(tdd: &mut Tdd, pvi: VtreeIdx, new_nodes: &mut [Vec<InputPair>]) {
    let level = &mut tdd.levels[pvi.idx()];
    level.clear();
    for pairs in new_nodes.iter_mut() {
        sort_pairs(pairs);
        // Defensive: regroup callers already push distinct pairs per cell, but
        // dedup after the canonical sort guarantees the no-duplicate-pairs node
        // invariant independently of that reasoning (O(n) on a sorted slice).
        pairs.dedup();
        level.push_internal_node(pairs);
    }
    tdd.scratch.dirty_contract.push(pvi.0);
}
