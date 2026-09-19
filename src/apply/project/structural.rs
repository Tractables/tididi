//! The structural existential forget: rewrite x's leaf-to-root path in place,
//! never calling apply or negate, so marginal levels off the path are safe.
//!
//! Distinct nodes at one level compute disjoint functions (determinism,
//! decided by `test_helpers::check::check_determinism`), so distinct
//! sibling-child references in a pair list are mutually exclusive and ∃x is a
//! structural regrouping. Path levels are processed leaf to root: at each, the
//! path-side child reference is replaced by its forgotten image (Pos/Neg/One
//! become One at the leaf parent; `c` becomes `child_remap.get(c)` above), then
//! nodes that now share an atom (same path image, same sibling ref) are
//! merged to restore the partition. The per-level node remap feeds the next
//! level up; sibling refs are copied verbatim and never dereferenced.

use crate::Engine;
use crate::limits::{OperationError, PollGate};
use crate::reduce::{ReductionPlan};
use crate::diagram::{EncodedChildRef, ChildDecoder, ChildPair, Tdd};
use crate::diagram::sort_pairs;
use crate::vtree::{VtreeIdx, VtreeNode};

use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};

use rustc_hash::FxHashMap;

/// Items grouped by a `u32` key into contiguous runs: key `k` owns
/// `items[starts[k]..starts[k + 1]]`.
///
/// Both regroups emit their `(key, item)` entries in scan order rather than
/// grouped by key — a new cell is opened part way through a level and later
/// entries join it — so the grouping is a counting sort over the keys. The
/// scatter is stable, which is what keeps a node's fan-out ascending: cells
/// are opened in increasing order, so the entries naming one node arrive in
/// that order too.
struct Runs<T> {
    starts: Vec<u32>,
    items: Vec<T>,
}

impl<T: Copy> Runs<T> {
    /// The runs of a level that held no nodes: every key maps to nothing.
    fn empty() -> Runs<T> {
        Runs { starts: Vec::new(), items: Vec::new() }
    }

    /// Group `entries` under `keys` keys, `zero` seeding the scatter buffer.
    fn pack(
        lim: &crate::limits::Limits,
        keys: usize,
        entries: &[(u32, T)],
        zero: T,
    ) -> Result<Runs<T>, OperationError> {
        u32::try_from(entries.len()).map_err(|_| OperationError::OverBudget)?;
        let mut starts = Vec::new();
        lim.try_resize(&mut starts, keys + 1, 0u32)?;
        for &(key, _) in entries {
            starts[key as usize + 1] += 1;
        }
        for k in 0..keys {
            starts[k + 1] += starts[k];
        }
        let mut items = Vec::new();
        lim.try_resize(&mut items, entries.len(), zero)?;
        let mut cursor = Vec::new();
        lim.reserve_exact(&mut cursor, keys)?;
        cursor.extend_from_slice(&starts[..keys]);
        for &(key, item) in entries {
            let slot = &mut cursor[key as usize];
            items[*slot as usize] = item;
            *slot += 1;
        }
        lim.discard(cursor);
        Ok(Runs { starts, items })
    }

    /// How many keys the runs cover.
    fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    fn get(&self, key: usize) -> &[T] {
        match self.starts.get(key + 1) {
            Some(&end) => &self.items[self.starts[key] as usize..end as usize],
            None => &[],
        }
    }

    fn get_mut(&mut self, key: usize) -> &mut [T] {
        let (lo, hi) = (self.starts[key] as usize, self.starts[key + 1] as usize);
        &mut self.items[lo..hi]
    }
}

/// Per-level fan-out map: `remap.get(old_node_idx)` lists every new node index
/// that the old node contributes to after the ∃x regroup. Multi-valued because
/// forgetting x can split one old node's sibling refs across several new
/// partition cells (owner classes); the level above re-expands a reference to
/// `old` over all listed new cells.
type Remap = Runs<u32>;

/// Existentially quantify `x` from `tdd` by an in-place leaf-to-root rewrite,
/// then reduce on `eng`, checking its limits throughout the rewrite. `leaf_idx` is
/// `x`'s leaf, looked up by the caller. Marginal levels off the path are left
/// byte-identical; the preconditions on the path are those of
/// `check_path_is_rewritable`.
///
/// # Errors
///
/// The [`OperationError`] the rewrite or reduction stopped on; the operand is consumed.
pub(super) fn exists_var_structural(
    eng: &Engine,
    mut tdd: Tdd,
    leaf_idx: VtreeIdx,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    lim.check_stop()?;
    let mut work = Rewrite { eng, gate: lim.gate(), emitted: 0 };
    if tdd.is_zero() {
        return Ok(tdd);
    }
    tdd.require_structure_at(leaf_idx)?;

    // Single-var vtree / output at the leaf: ∃x.F = `constant_one`.
    if tdd.output.vtree == leaf_idx {
        let mut result = eng.cube(&tdd.vtree, std::iter::empty::<crate::Literal>())?;
        result.weights = tdd.weights.as_ref().map(crate::diagram::WeightStore::empty_like);
        return Ok(result);
    }
    check_path_is_rewritable(&mut work, &tdd, leaf_idx)?;

    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let path = ancestor_path(&mut work, &vtree, leaf_idx)?;
    let child_remap = rewrite_path(&mut work, &mut tdd, &path, leaf_idx)?;

    let root_vi = *path.last().expect("path is non-empty (output not at leaf)");
    let out_pairs = union_of_root_cells(&mut work, &tdd, root_vi, child_remap.get(tdd.output.local.idx()))?;
    // Append the union node and point the output at it (prune drops the rest).
    let new_out = tdd.levels[root_vi.idx()].push_node_on(eng, &out_pairs)?;
    tdd.output.local = new_out;
    tdd.try_invalidate(eng, root_vi)?;

    work.emitted += 1;
    lim.level_done(work.emitted)?;
    work.gate.flush()?;
    eng.reduce(&mut tdd, ReductionPlan::default())?;
    Ok(tdd)
}

/// Check the ancestors of a structural target leaf before rewriting its path.
///
/// (1) No ancestor of x's leaf is marginal: x would already be summed out.
///
/// (2) No ancestor is the grandparent of a marginal level. The boundary
///     content-twin merge (`reduce::contract::content_twin`) can leave the
///     same pair twice in such a grandparent, a count-carrying duplicate, and
///     the owner-set regroup here cannot represent multiplicity, so it would
///     fold the two into one and miscount. A marginal level three or more
///     levels below the path is harmless: its duplicates land in a sibling
///     subtree whose refs are only copied.
fn check_path_is_rewritable(work: &mut Rewrite<'_>, t: &Tdd, leaf_idx: VtreeIdx) -> Result<(), OperationError> {
    let vtree = &t.vtree;
    let mut anc = vtree.node(leaf_idx).parent();
    while let Some(ai) = anc {
        work.poll()?;
        t.require_structure_at(ai)?;
        let (al, ar) = vtree.children(ai);
        for child in [al, ar] {
            if vtree.node(child).is_leaf() { continue; }
            let (left, right) = vtree.children(child);
            t.require_structure_at(left)?;
            t.require_structure_at(right)?;
        }
        anc = vtree.node(ai).parent();
    }
    Ok(())
}

/// The leaf→root ancestor path: `[leaf_parent, grandparent, …, root]`.
fn ancestor_path(work: &mut Rewrite<'_>, vtree: &crate::vtree::Vtree, leaf_idx: VtreeIdx) -> Result<Vec<VtreeIdx>, OperationError> {
    let mut path = Vec::new();
    let mut cur = vtree.node(leaf_idx).parent();
    while let Some(p) = cur {
        work.poll()?;
        work.eng.limits().try_push(&mut path, p)?;
        cur = vtree.node(p).parent();
    }
    Ok(path)
}

/// Regroup every level on `path`, leaf-parent first, and return the root
/// level's fan-out map.
///
/// Each step consumes the level below's map — which new cells the child's old
/// nodes became — and produces its own for the level above. The leaf-parent
/// step is the special one: its "child" is x's own leaf, which has no map.
fn rewrite_path(work: &mut Rewrite<'_>, tdd: &mut Tdd, path: &[VtreeIdx], leaf_idx: VtreeIdx) -> Result<Remap, OperationError> {
    let vtree = tdd.vtree.clone();
    let mut child_remap: Remap = Runs::empty();
    let mut child_vi = leaf_idx;
    for (step, &parent) in path.iter().enumerate() {
        let (left_child, right_child) = match *vtree.node(parent) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            _ => unreachable!("path node must be internal"),
        };
        let path_is_left = left_child == child_vi;
        debug_assert!(path_is_left || right_child == child_vi);

        child_remap = if step == 0 {
            regroup_leaf_parent(work, tdd, parent, path_is_left)?
        } else {
            regroup_internal(work, tdd, parent, path_is_left, &child_remap)?
        };
        child_vi = parent;
    }
    Ok(child_remap)
}

/// The pairs of ∃x.f's single output node: the deduped union of the root cells
/// the old output fanned out into.
///
/// The cells are mutex among themselves and each is a valid
/// deterministic/decomposable pair list, so their union is a sound single root
/// node. The other (unreferenced) root cells are dropped by `minimize`'s prune.
fn union_of_root_cells(work: &mut Rewrite<'_>, tdd: &Tdd, root_vi: VtreeIdx, out_cells: &[u32]) -> Result<Vec<ChildPair>, OperationError> {
    let level = &tdd.levels[root_vi.idx()];
    let mut out_pairs: Vec<ChildPair> = Vec::new();
    for &k in out_cells {
        // Copy the cell's pairs; the cells are mutually exclusive, so the one
        // dedup below is all the union needs.
        for pair in level.pairs_iter_of_idx(k as usize) {
            work.poll()?;
            work.eng.limits().try_push(&mut out_pairs, pair)?;
        }
    }
    sort_pairs(&mut out_pairs);
    out_pairs.dedup();
    Ok(out_pairs)
}

/// The (pos-owner, neg-owner) old-node indices for a single sibling ref at the
/// leaf parent. `u32::MAX` means "no owner on that polarity".
///
/// One owner per polarity, so this cannot carry a pair's multiplicity; sound
/// under precondition (2) of `check_path_is_rewritable`.
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
fn regroup_leaf_parent(work: &mut Rewrite<'_>, tdd: &mut Tdd, parent: VtreeIdx, path_is_left: bool) -> Result<Remap, OperationError> {
    let level = &tdd.levels[parent.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 { return Ok(Runs::empty()); }
    let lim = work.eng.limits();

    let read_pair = |p: &ChildPair| -> (EncodedChildRef, EncodedChildRef) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // Per-sibling-ref owner pair, in first-seen order.
    let mut owners: FxHashMap<u32, OwnerKey> = FxHashMap::default();
    let mut order: Vec<u32> = Vec::new();

    for i in 0..n_nodes {
        work.poll()?;
        for p in level.pairs_of_idx(i) {
            work.poll()?;
            let (x_label, sib) = read_pair(p);
            if !owners.contains_key(&sib.0) {
                lim.reserve_map(&mut owners, 1)?;
                lim.try_push(&mut order, sib.0)?;
            }
            let e = owners.entry(sib.0).or_insert(OwnerKey { pos: u32::MAX, neg: u32::MAX });
            if x_label == POS_LEAF_IDX.into() {
                e.pos = i as u32;
            } else if x_label == NEG_LEAF_IDX.into() {
                e.neg = i as u32;
            } else {
                e.pos = i as u32;
                e.neg = i as u32;
            }
        }
    }

    // Group sibling refs by their unordered owner key, one new cell each. The
    // fan-out is recorded at cell creation; cells are created in increasing
    // index order, so each old node's fan-out list comes out ascending.
    let mut key_to_new: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut cell_pairs: Vec<(u32, ChildPair)> = Vec::new();
    lim.reserve_exact(&mut cell_pairs, order.len())?;
    let mut fanout: Vec<(u32, u32)> = Vec::new();
    let mut n_cells = 0u32;

    for &sib in &order {
        work.poll()?;
        let ok = owners[&sib];
        let (a, b) = (ok.pos, ok.neg);
        let key = if a <= b { (a, b) } else { (b, a) };
        let cell = if let Some(&cell) = key_to_new.get(&key) {
            cell
        } else {
            let cell = n_cells;
            if cell == u32::MAX { return Err(OperationError::OverBudget); }
            n_cells += 1;
            if key.0 != u32::MAX { lim.try_push(&mut fanout, (key.0, cell))?; }
            if key.1 != u32::MAX && key.1 != key.0 { lim.try_push(&mut fanout, (key.1, cell))?; }
            lim.reserve_map(&mut key_to_new, 1)?;
            key_to_new.insert(key, cell);
            cell
        };
        let pair = if path_is_left {
            ChildPair::new(ONE_LEAF_IDX, EncodedChildRef::from_raw(sib))
        } else {
            ChildPair::new(EncodedChildRef::from_raw(sib), ONE_LEAF_IDX)
        };
        // `order` holds distinct sibs and the pair is injective in `sib`, so
        // every pair within a cell is already distinct.
        lim.try_push(&mut cell_pairs, (cell, pair))?;
    }

    let mut new_nodes =
        Runs::pack(lim, n_cells as usize, &cell_pairs, ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX))?;
    lim.discard(cell_pairs);
    write_level(work, tdd, parent, &mut new_nodes)?;
    let remap = Runs::pack(lim, n_nodes, &fanout, 0u32)?;
    lim.discard(fanout);
    Ok(remap)
}

/// Regroup an internal path level `parent` after the level below it was forgotten.
///
/// Each old pair `(c, sib)` has its path-side child `c` expanded via
/// `child_remap.get(c)` into new child cells. The expanded atom `(Pc, sib)` (Pc a
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
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    path_is_left: bool,
    child_remap: &Remap,
) -> Result<Remap, OperationError> {
    let level = &tdd.levels[parent.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 { return Ok(Runs::empty()); }
    let lim = work.eng.limits();

    let read_pair = |p: &ChildPair| -> (EncodedChildRef, EncodedChildRef) {
        if path_is_left { (p.left, p.right) } else { (p.right, p.left) }
    };

    // Each expanded atom (Pc, sib) gets an index, and its owner set — the old
    // nodes contributing it — is accumulated as a run of (atom, owner) entries.
    // The outer loop visits `i` in increasing order and `last_owner` drops
    // repeats, so each atom's owners come out sorted and unique.
    //
    // The level's pairs are a lower bound on the distinct atoms — a pair whose
    // path child fans out contributes several — so reserving that many never
    // over-allocates, and it takes most of the doublings off a map that is
    // otherwise grown one insert at a time.
    let mut atom_index: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    lim.reserve_map(&mut atom_index, level.live_pairs())?;
    let mut atoms: Vec<(u32, u32)> = Vec::new();
    let mut owner_count: Vec<u32> = Vec::new();
    let mut last_owner: Vec<u32> = Vec::new();
    let mut entries: Vec<(u32, u32)> = Vec::new();

    for i in 0..n_nodes {
        work.poll()?;
        let owner = u32::try_from(i).map_err(|_| OperationError::OverBudget)?;
        for p in level.pairs_of_idx(i) {
            work.poll()?;
            let (path_child, sib) = read_pair(p);
            for &cell in child_remap.get(ChildDecoder::structural().node(path_child).idx()) {
                work.poll()?;
                let key = (cell, sib.0);
                let atom = match atom_index.get(&key) {
                    Some(&atom) => atom,
                    None => {
                        let atom = u32::try_from(atoms.len()).map_err(|_| OperationError::OverBudget)?;
                        lim.reserve_map(&mut atom_index, 1)?;
                        atom_index.insert(key, atom);
                        lim.try_push(&mut atoms, key)?;
                        lim.try_push(&mut owner_count, 0)?;
                        lim.try_push(&mut last_owner, u32::MAX)?;
                        atom
                    }
                };
                if last_owner[atom as usize] != owner {
                    last_owner[atom as usize] = owner;
                    owner_count[atom as usize] += 1;
                    lim.try_push(&mut entries, (atom, owner))?;
                }
            }
        }
    }

    // Owner sets, packed one run per atom: the entries were produced in
    // increasing owner order, so scattering them by atom keeps each run sorted.
    let mut starts = Vec::new();
    lim.reserve_exact(&mut starts, atoms.len() + 1)?;
    let mut total = 0u32;
    for &count in &owner_count {
        starts.push(total);
        total += count;
    }
    starts.push(total);
    let mut cursor = Vec::new();
    lim.reserve_exact(&mut cursor, atoms.len())?;
    cursor.extend_from_slice(&starts[..atoms.len()]);
    let mut owners = Vec::new();
    lim.try_resize(&mut owners, entries.len(), 0u32)?;
    for &(atom, owner) in &entries {
        work.poll()?;
        let slot = &mut cursor[atom as usize];
        owners[*slot as usize] = owner;
        *slot += 1;
    }

    // Group atoms by owner set → one new cell per distinct owner set, found by
    // hashing the run and comparing it against the cells that hash alike.
    let mut by_hash: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
    let mut cell_atom: Vec<u32> = Vec::new();
    let mut cell_pairs: Vec<(u32, ChildPair)> = Vec::new();
    lim.reserve_exact(&mut cell_pairs, atoms.len())?;
    let mut fanout: Vec<(u32, u32)> = Vec::new();
    let mut n_cells = 0u32;

    for (atom, &(cell, sib)) in atoms.iter().enumerate() {
        work.poll()?;
        let run = |atom: usize| starts[atom] as usize..starts[atom + 1] as usize;
        let mine = run(atom);
        let digest = owner_set_hash(&owners[mine.clone()]);
        lim.reserve_map(&mut by_hash, 1)?;
        let candidates = by_hash.entry(digest).or_default();
        let found = candidates.iter().copied()
            .find(|&idx| owners[run(cell_atom[idx as usize] as usize)] == owners[mine.clone()]);
        let new_cell = match found {
            Some(idx) => idx,
            None => {
                let new_cell = n_cells;
                if new_cell == u32::MAX { return Err(OperationError::OverBudget); }
                n_cells += 1;
                lim.try_push(candidates, new_cell)?;
                lim.try_push(&mut cell_atom, u32::try_from(atom).map_err(|_| OperationError::OverBudget)?)?;
                for &owner in &owners[mine] {
                    work.poll()?;
                    lim.try_push(&mut fanout, (owner, new_cell))?;
                }
                new_cell
            }
        };
        let pair = if path_is_left {
            ChildPair::new(EncodedChildRef::from_raw(cell), EncodedChildRef::from_raw(sib))
        } else {
            ChildPair::new(EncodedChildRef::from_raw(sib), EncodedChildRef::from_raw(cell))
        };
        // `atoms` holds distinct atoms and the pair is injective in the
        // atom, so every pair within a cell is already distinct.
        lim.try_push(&mut cell_pairs, (new_cell, pair))?;
    }

    let mut new_nodes =
        Runs::pack(lim, n_cells as usize, &cell_pairs, ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX))?;
    lim.discard(cell_pairs);
    write_level(work, tdd, parent, &mut new_nodes)?;
    let remap = Runs::pack(lim, n_nodes, &fanout, 0u32)?;
    lim.discard(fanout);
    Ok(remap)
}

/// Replace level `parent`'s nodes with `new_nodes` (one run of pairs per new
/// cell), sorting each pair list canonically, and mark the level dirty for
/// contraction.
fn write_level(work: &mut Rewrite<'_>, tdd: &mut Tdd, parent: VtreeIdx, new_nodes: &mut Runs<ChildPair>) -> Result<(), OperationError> {
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    for cell in 0..new_nodes.len() {
        work.poll()?;
        let pairs = new_nodes.get_mut(cell);
        sort_pairs(pairs);
        let kept = dedup_sorted(pairs);
        level.push_node_on(work.eng, &pairs[..kept])?;
        work.emitted += 1;
        work.eng.limits().level_done(work.emitted)?;
    }
    tdd.try_invalidate(work.eng, parent)?;
    Ok(())
}

/// Drop the repeats from a sorted slice in place, returning how many entries
/// are kept — `Vec::dedup` for a slice. Both regroups build pair lists that
/// are already distinct by construction, so this is a guard, not a pass that
/// normally removes anything.
fn dedup_sorted(pairs: &mut [ChildPair]) -> usize {
    let mut kept = 0usize;
    for i in 0..pairs.len() {
        if kept == 0 || pairs[i] != pairs[kept - 1] {
            pairs[kept] = pairs[i];
            kept += 1;
        }
    }
    kept
}

/// Hash one atom's owner run, so runs are compared only against those that
/// agree on it.
fn owner_set_hash(owners: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    owners.hash(&mut hasher);
    hasher.finish()
}

/// The rewrite's cancellation clock and number of emitted intermediate nodes.
struct Rewrite<'a> {
    eng: &'a Engine,
    gate: PollGate<'a>,
    emitted: u64,
}

impl Rewrite<'_> {
    /// Account for one visited node, pair, or owner association.
    fn poll(&mut self) -> Result<(), OperationError> {
        self.gate.poll(1)
    }
}

#[cfg(test)]
#[path = "tests/structural.rs"]
mod tests;
