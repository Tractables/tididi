//! The structural existential forget: rewrite the quantified leaves and their
//! ancestors in place, never calling apply or negate, so marginal levels
//! elsewhere are safe.
//!
//! Distinct nodes at one level compute disjoint functions (determinism,
//! decided by `test_helpers::check::check_determinism`), so distinct
//! child references in a pair list are mutually exclusive and ∃ is a
//! structural regrouping. The whole request is one bottom-up sweep that gives
//! each vtree node one of three roles:
//!
//! * **whole** — every leaf below it is quantified. The level becomes the
//!   single ⊤ node and its descendants do too, so a block of any width is
//!   freed by one write per level of its subtree rather than by one
//!   leaf-to-root rewrite per variable.
//! * **split** — some but not all of its leaves are quantified. The level is
//!   regrouped once: each pair's rewritten sides are expanded over the cells
//!   the child became, and the atoms that now share an owner set merge.
//! * **free** — no quantified leaf below it. The level is left untouched and
//!   its references are copied verbatim, never dereferenced, so it may be
//!   marginal.
//!
//! The per-level node remap feeds the level above. A regroup that merges
//! nothing returns no remap: its cells are its old nodes in order, so every
//! level above it would rewrite itself to what it already holds and the sweep
//! stops climbing.
//!
//! The sweep reads every node of a level it regroups, so an operand that has
//! not been pruned since it was built makes it read nodes nothing references;
//! it discharges that debt before it starts.

use crate::Engine;
use crate::limits::{Charged, OperationError, PollGate, Transient};
use crate::reduce::ReductionPlan;
use crate::diagram::{EncodedChildRef, ChildDecoder, ChildPair, Tdd};
use crate::diagram::sort_pairs;
use crate::vtree::{Vtree, VtreeIdx};

use crate::diagram::LEAF_WIDTH;
use crate::apply::TRUE_PAIR;

use rustc_hash::FxHashMap;

/// Items grouped by a `u32` key into contiguous runs: key `k` owns
/// `items[starts[k]..starts[k + 1]]`.
///
/// The regroup emits its `(key, item)` entries in scan order rather than
/// grouped by key — a new cell is opened part way through a level and later
/// entries join it — so the grouping is a counting sort over the keys. The
/// scatter is stable, which is what keeps a node's fan-out ascending: cells
/// are opened in increasing order, so the entries naming one node arrive in
/// that order too.
struct Runs<T> {
    starts: Vec<u32>,
    items: Vec<T>,
}

impl<T> Charged for Runs<T> {
    fn charged_bytes(&self) -> u64 { self.starts.charged_bytes() + self.items.charged_bytes() }
}

impl<T: Copy> Runs<T> {
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

    /// Every key mapped to run `0`, the map of a level replaced by one node.
    fn all_to_first(lim: &crate::limits::Limits, keys: usize, zero: T) -> Result<Runs<T>, OperationError> {
        let mut starts = Vec::new();
        lim.reserve_exact(&mut starts, keys + 1)?;
        starts.extend(0..=u32::try_from(keys).map_err(|_| OperationError::OverBudget)?);
        let mut items = Vec::new();
        lim.try_resize(&mut items, keys, zero)?;
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
/// that the old node contributes to after the regroup. Multi-valued because
/// forgetting below a level can split one old node's references across several
/// new partition cells (owner classes); the level above re-expands a reference
/// to `old` over all listed new cells.
///
/// `None` in place of one stands for the identity: the level above keeps the
/// reference it stores, which is also how a level with no quantified leaf below
/// it is treated.
type Remap = Runs<u32>;

/// How a vtree node relates to the quantified leaves.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Role {
    /// No quantified leaf below it.
    Free,
    /// Every leaf below it is quantified.
    Whole,
    /// Some, but not all, of its leaves are quantified.
    Split,
}

/// Existentially quantify the variables at `targets` out of `tdd` by one
/// in-place bottom-up sweep, then reduce on `eng`, checking its limits
/// throughout. `targets` are distinct leaves of the diagram's vtree, looked up
/// by the caller. An operand that still owes the reduction passes is pruned
/// first. The sweep then leaves levels with no quantified leaf below them
/// byte-identical; the preconditions on the rest are those of
/// `check_levels_are_rewritable`.
///
/// Freeing a whole subtree at once denotes the same function as freeing its
/// leaves one at a time, in the same representation. Every node of the subtree
/// is satisfiable (invariant 2), so each denotes ⊤ once all of its variables
/// are free, and the level's partition collapses to that single node — which is
/// where a run of per-variable rewrites over the same leaves ends too. Above the
/// subtree the two agree level by level: the pairs are expanded over the cells
/// the children became and regrouped by owner set either way, and the same
/// `reduce` finishes both, so the canonical form the caller sees is the same.
///
/// # Errors
///
/// The [`OperationError`] the rewrite or reduction stopped on; the operand is consumed.
pub(super) fn exists_leaves_structural(
    eng: &Engine,
    mut tdd: Tdd,
    targets: &[VtreeIdx],
    collapsed: bool,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    lim.check_stop()?;
    let mut work = Rewrite { eng, gate: lim.gate(), emitted: 0 };
    if tdd.is_zero() {
        return Ok(tdd);
    }
    if !tdd.dirty.is_empty() {
        // An operand straight out of an apply still owes the reduction passes,
        // and the debt it owes the prune is one the sweep would pay for twice:
        // a node nothing reaches costs a share of its level's regroup, and the
        // atoms it owns refine the owner-set partition the level above then
        // has to fan out over. A diagram already at the fixpoint owes nothing
        // and skips the pass.
        eng.reduce(&mut tdd, ReductionPlan::Prune)?;
    }
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let role = roles(&mut work, &vtree, targets)?;
    check_levels_are_rewritable(&mut work, &tdd, &vtree, &role)?;

    let root_vi = vtree.root();
    if role[root_vi.idx()] == Role::Whole {
        // Every variable is quantified and the operand is satisfiable: ∃.F = ⊤.
        let mut result = eng.cube(&tdd.vtree, std::iter::empty::<crate::Literal>())?;
        result.weights = tdd.weights.as_ref().map(crate::diagram::WeightStore::empty_like);
        return Ok(result);
    }

    let mut remap: Vec<Option<Remap>> = Vec::new();
    lim.reserve_exact(&mut remap, vtree.num_nodes())?;
    remap.resize_with(vtree.num_nodes(), || None);
    for &level in vtree.bottomup_slice() {
        work.poll()?;
        remap[level.idx()] = match role[level.idx()] {
            Role::Free => None,
            Role::Whole => {
                let above = vtree.node(level).parent();
                let wanted = above.is_some_and(|p| role[p.idx()] == Role::Split);
                free_subtree_level(&mut work, &mut tdd, &vtree, level, wanted, collapsed)?
            }
            Role::Split => {
                let (left, right) = vtree.children(level);
                let below_left = remap[left.idx()].take().map(|v| Transient::new(lim, v));
                let below_right = remap[right.idx()].take().map(|v| Transient::new(lim, v));
                if below_left.is_none() && below_right.is_none() {
                    None
                } else {
                    regroup(&mut work, &mut tdd, level, below_left.as_deref(), below_right.as_deref())?
                }
            }
        };
    }

    if let Some(root_cells) = remap[root_vi.idx()].take() {
        let root_cells = Transient::new(lim, root_cells);
        let out_pairs = Transient::new(lim,
            union_of_root_cells(&mut work, &tdd, root_vi, root_cells.get(tdd.output.local.idx()))?);
        // Append the union node and point the output at it (prune drops the rest).
        let new_out = tdd.levels[root_vi.idx()].push_node(eng.limits(), &out_pairs)?;
        tdd.output.local = new_out;
        tdd.try_invalidate(eng, root_vi)?;
        work.emitted += 1;
        lim.level_done(work.emitted)?;
    }
    work.gate.flush()?;
    eng.reduce(&mut tdd, ReductionPlan::default())?;
    Ok(tdd)
}

/// Label every vtree node by how many of its leaves `targets` names.
fn roles(work: &mut Rewrite<'_>, vtree: &Vtree, targets: &[VtreeIdx]) -> Result<Vec<Role>, OperationError> {
    let mut role = Vec::new();
    work.eng.limits().try_resize(&mut role, vtree.num_nodes(), Role::Free)?;
    for &leaf in targets {
        work.poll()?;
        role[leaf.idx()] = Role::Whole;
    }
    for &level in vtree.bottomup_slice() {
        work.poll()?;
        if vtree.node(level).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(level);
        role[level.idx()] = match (role[left.idx()], role[right.idx()]) {
            (Role::Free, Role::Free) => Role::Free,
            (Role::Whole, Role::Whole) => Role::Whole,
            _ => Role::Split,
        };
    }
    Ok(role)
}

/// Check the levels the sweep reads or rewrites.
///
/// (1) No quantified leaf and no regrouped level is marginal: a marginal
///     level's variables are already summed out, and its values carry a
///     multiplicity the owner-set regroup cannot represent. Marginality is
///     downward-closed (invariant 5), so checking a freed subtree's leaves
///     settles the whole subtree.
///
/// (2) No regrouped level is the grandparent of a marginal level. The boundary
///     content-twin merge (`reduce::contract::content_twin`) can leave the
///     same pair twice in such a grandparent, a count-carrying duplicate, and
///     the owner-set regroup cannot represent multiplicity, so it would fold
///     the two into one and miscount. A marginal level three or more levels
///     below is harmless: its duplicates land in a subtree whose references
///     are only copied.
fn check_levels_are_rewritable(
    work: &mut Rewrite<'_>,
    t: &Tdd,
    vtree: &Vtree,
    role: &[Role],
) -> Result<(), OperationError> {
    for &level in vtree.bottomup_slice() {
        work.poll()?;
        match role[level.idx()] {
            Role::Free => {}
            Role::Whole => {
                if vtree.node(level).is_leaf() {
                    t.require_structure_at(level)?;
                }
            }
            Role::Split => {
                t.require_structure_at(level)?;
                let (al, ar) = vtree.children(level);
                for child in [al, ar] {
                    if vtree.node(child).is_leaf() { continue; }
                    let (left, right) = vtree.children(child);
                    t.require_structure_at(left)?;
                    t.require_structure_at(right)?;
                }
            }
        }
    }
    Ok(())
}

/// Replace a level all of whose variables are quantified by the single ⊤ node,
/// and report the map the level above needs.
///
/// A leaf level stores nothing, so freeing it is only a statement about the
/// references into it: all three labels become ⊤. An internal level's nodes are
/// each satisfiable, so each denotes ⊤ once its variables are free, and the
/// whole partition becomes one node whose pair names ⊤ on both sides.
///
/// `wanted` is whether the level above reads the map. `None` comes back when it
/// does not, and when the map is the identity because the level held one node —
/// the level above then keeps its own references and is left alone.
///
/// `collapsed` suspends that second shortcut. A level a fused conjunction
/// already reduced to `⊤` holds one node *now* but stood for many when the
/// level above was built, so its references no longer separate what they
/// separated: the level above still owes the regroup, and only a map it is
/// handed makes it run.
fn free_subtree_level(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    vtree: &Vtree,
    level: VtreeIdx,
    wanted: bool,
    collapsed: bool,
) -> Result<Option<Remap>, OperationError> {
    let lim = work.eng.limits();
    if vtree.node(level).is_leaf() {
        return if wanted { Ok(Some(Runs::all_to_first(lim, LEAF_WIDTH, 0u32)?)) } else { Ok(None) };
    }
    let n_nodes = tdd.levels[level.idx()].nodes.len();
    let store = &mut tdd.levels[level.idx()];
    store.clear();
    store.push_node(work.eng.limits(), &[TRUE_PAIR])?;
    work.emitted += 1;
    lim.level_done(work.emitted)?;
    tdd.try_invalidate(work.eng, level)?;
    if !wanted || (n_nodes == 1 && !collapsed) {
        return Ok(None);
    }
    Runs::all_to_first(lim, n_nodes, 0u32).map(Some)
}

/// The pairs of the result's single output node: the deduped union of the root
/// cells the old output fanned out into.
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

/// One side of a pair, as the cells its child level became. A side whose child
/// the sweep did not touch stands for itself and is never decoded — that child
/// may be marginal.
#[inline]
fn expand<'a>(remap: Option<&'a Remap>, kept: &'a mut u32, side: EncodedChildRef) -> &'a [u32] {
    match remap {
        Some(map) => map.get(ChildDecoder::structural().node(side).idx()),
        None => {
            *kept = side.0;
            std::slice::from_ref(kept)
        }
    }
}

/// Regroup a level with quantified leaves under one or both of its children.
///
/// Each old pair is expanded into the atoms `(Pl, Pr)`, `Pl` a cell the left
/// child became and `Pr` one of the right child's, a side whose child was not
/// rewritten standing for itself. Distinct cells of one level are mutually
/// exclusive, so the atoms of one pair are too, and the atoms are then
/// re-partitioned by the **owner-set** rule:
///
///   For each distinct atom, its owner set is `{old node g : g contributes it}`.
///   A new cell ↔ a distinct owner set; its pairs are all atoms sharing that
///   owner set. `remap[g]` = the new cells whose owner set contains `g` (so a
///   reference to `g` from above expands to the disjunction of exactly those
///   cells, = `∃.g`).
///
/// This keeps the new level a valid partition: two atoms with different owner
/// sets land in different cells (mutex by construction of the owner set), and
/// `∃.g` is reconstructed exactly as the OR over `g`'s cells.
///
/// `None` comes back when the partition did not change — cell `i` holds
/// exactly what node `i` expanded to — because the level above would then
/// rewrite itself to what it already holds.
fn regroup(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left_remap: Option<&Remap>,
    right_remap: Option<&Remap>,
) -> Result<Option<Remap>, OperationError> {
    let level = &tdd.levels[parent.idx()];
    let n_nodes = level.nodes.len();
    if n_nodes == 0 { return Ok(None); }
    let lim = work.eng.limits();

    // Each expanded atom gets an index, and its owner set — the old nodes
    // contributing it — is accumulated as a run of (atom, owner) entries.
    // The outer loop visits `i` in increasing order and `last_owner` drops
    // repeats, so each atom's owners come out sorted and unique.
    //
    // The level's pairs are a lower bound on the distinct atoms — a pair whose
    // rewritten side fans out contributes several — so reserving that many
    // never over-allocates, and it takes most of the doublings off a map that
    // is otherwise grown one insert at a time.
    let mut atom_index: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    lim.reserve_map(&mut atom_index, level.live_pairs())?;
    let mut atoms: Vec<(u32, u32)> = Vec::new();
    let mut owner_count: Vec<u32> = Vec::new();
    let mut last_owner: Vec<u32> = Vec::new();
    let mut entries: Vec<(u32, u32)> = Vec::new();
    let (mut kept_left, mut kept_right) = (0u32, 0u32);

    for i in 0..n_nodes {
        work.poll()?;
        let owner = u32::try_from(i).map_err(|_| OperationError::OverBudget)?;
        for p in level.pairs_of_idx(i) {
            work.poll()?;
            let lefts = expand(left_remap, &mut kept_left, p.left);
            let rights = expand(right_remap, &mut kept_right, p.right);
            for &left in lefts {
                for &right in rights {
                    work.poll()?;
                    let key = (left, right);
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
    }

    lim.discard(atom_index);
    lim.discard(last_owner);

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

    lim.discard(entries);
    lim.discard(cursor);
    lim.discard(owner_count);

    // Group atoms by owner set → one new cell per distinct owner set, found by
    // hashing the run and comparing it against the cells that hash alike.
    let mut by_hash: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
    let mut cell_atom: Vec<u32> = Vec::new();
    let mut cell_pairs: Vec<(u32, ChildPair)> = Vec::new();
    lim.reserve_exact(&mut cell_pairs, atoms.len())?;
    let mut fanout: Vec<(u32, u32)> = Vec::new();
    let mut n_cells = 0u32;

    for (atom, &(left, right)) in atoms.iter().enumerate() {
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
        let pair = ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(right));
        // `atoms` holds distinct atoms and the pair is injective in the
        // atom, so every pair within a cell is already distinct.
        lim.try_push(&mut cell_pairs, (new_cell, pair))?;
    }

    // One cell per node and one owner per cell: the owner sets are singletons
    // and pairwise distinct, and cells open in node order, so cell `i` is
    // node `i` expanded. Every reference from above still names the node it
    // named, and the level above would hand back its own pairs.
    let unchanged = n_cells as usize == n_nodes && fanout.len() == n_nodes;

    lim.discard(atoms);
    lim.discard(starts);
    lim.discard(owners);
    lim.discard(cell_atom);
    for candidates in by_hash.values_mut() { lim.discard(std::mem::take(candidates)); }
    lim.discard(by_hash);
    let mut new_nodes = Transient::new(lim, Runs::pack(lim, n_cells as usize, &cell_pairs, TRUE_PAIR)?);
    lim.discard(cell_pairs);
    write_level(work, tdd, parent, &mut new_nodes)?;
    if unchanged {
        lim.discard(fanout);
        return Ok(None);
    }
    let remap = Runs::pack(lim, n_nodes, &fanout, 0u32)?;
    lim.discard(fanout);
    Ok(Some(remap))
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
        level.push_node(work.eng.limits(), &pairs[..kept])?;
        work.emitted += 1;
        work.eng.limits().level_done(work.emitted)?;
    }
    tdd.try_invalidate(work.eng, parent)?;
    Ok(())
}

/// Drop the repeats from a sorted slice in place, returning how many entries
/// are kept — `Vec::dedup` for a slice. The regroup builds pair lists that are
/// already distinct by construction, so this is a guard, not a pass that
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
