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
use crate::diagram::{EncodedChildRef, ChildDecoder, ChildPair, Tdd, TddLevel};
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
        Runs::pack_by(lim, keys, entries, |&entry| entry, zero)
    }

    /// [`pack`](Self::pack) over records that `entry` splits into a key below
    /// `keys` and an item.
    fn pack_by<S>(
        lim: &crate::limits::Limits,
        keys: usize,
        records: &[S],
        entry: impl Fn(&S) -> (u32, T),
        zero: T,
    ) -> Result<Runs<T>, OperationError> {
        u32::try_from(records.len()).map_err(|_| OperationError::IndexOverflow)?;
        let mut starts = Vec::new();
        lim.try_resize(&mut starts, keys + 1, 0u32)?;
        for record in records {
            starts[entry(record).0 as usize + 1] += 1;
        }
        for k in 0..keys {
            starts[k + 1] += starts[k];
        }
        let mut items = Vec::new();
        lim.try_resize(&mut items, records.len(), zero)?;
        let mut cursor = Vec::new();
        lim.reserve_exact(&mut cursor, keys)?;
        cursor.extend_from_slice(&starts[..keys]);
        for record in records {
            let (key, item) = entry(record);
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
        starts.extend(0..=u32::try_from(keys).map_err(|_| OperationError::IndexOverflow)?);
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

impl Runs<u32> {
    /// The inverse map over `keys` keys, every item below `keys`: run `k`
    /// lists, ascending, the keys whose runs hold `k`.
    fn transpose(&self, lim: &crate::limits::Limits, keys: usize) -> Result<Runs<u32>, OperationError> {
        let mut starts = Vec::new();
        lim.try_resize(&mut starts, keys + 1, 0u32)?;
        for &item in &self.items {
            starts[item as usize + 1] += 1;
        }
        for k in 0..keys {
            starts[k + 1] += starts[k];
        }
        let mut items = Vec::new();
        lim.try_resize(&mut items, self.items.len(), 0u32)?;
        let mut cursor = Vec::new();
        lim.reserve_exact(&mut cursor, keys)?;
        cursor.extend_from_slice(&starts[..keys]);
        for key in 0..self.len() {
            // A run's key fits the `u32` its items are stored as.
            let key32 = key as u32;
            for &item in self.get(key) {
                let slot = &mut cursor[item as usize];
                items[*slot as usize] = key32;
                *slot += 1;
            }
        }
        lim.discard(cursor);
        Ok(Runs { starts, items })
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
/// in-place bottom-up sweep, then reduce on `eng` by `reduction`, checking its
/// limits throughout. The full default plan leaves the result canonical; a
/// prune leaves it the same function, with every node reachable but twins the
/// sweep made still apart. `targets` are distinct leaves of the diagram's
/// vtree, looked up by the caller. An operand that still owes the reduction
/// passes is pruned first. The sweep then leaves levels with no quantified
/// leaf below them byte-identical; the preconditions on the rest are those of
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
    reduction: ReductionPlan<'_>,
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
        return crate::build::constant_like(eng, &tdd, true);
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
    eng.reduce(&mut tdd, reduction)?;
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

/// The pairs of the result's single output node: the union of the root cells
/// the old output fanned out into.
///
/// The cells are mutex among themselves and each is a valid
/// deterministic/decomposable pair list, so their union is a sound single root
/// node with no repeated pair. The other (unreferenced) root cells are dropped
/// by `minimize`'s prune.
fn union_of_root_cells(work: &mut Rewrite<'_>, tdd: &Tdd, root_vi: VtreeIdx, out_cells: &[u32]) -> Result<Vec<ChildPair>, OperationError> {
    let level = &tdd.levels[root_vi.idx()];
    let mut out_pairs: Vec<ChildPair> = Vec::new();
    for &k in out_cells {
        for pair in level.pairs_iter_of_idx(k as usize) {
            work.poll()?;
            work.eng.limits().try_push(&mut out_pairs, pair)?;
        }
    }
    sort_pairs(&mut out_pairs);
    debug_assert!(out_pairs.windows(2).all(|w| w[0] != w[1]), "root cells share a pair");
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

/// The side of a level's pairs that [`regroup`] files them under; the atoms
/// filed under one reference differ only on the other side.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Side {
    Left,
    Right,
}

impl Side {
    /// This side's reference in `pair`.
    #[inline]
    fn of(self, pair: ChildPair) -> EncodedChildRef {
        match self {
            Side::Left => pair.left,
            Side::Right => pair.right,
        }
    }

    #[inline]
    fn other(self) -> Side {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }

    /// The pair naming `here` on this side and `there` on the other.
    #[inline]
    fn pair(self, here: u32, there: u32) -> ChildPair {
        let (left, right) = match self {
            Side::Left => (here, there),
            Side::Right => (there, here),
        };
        ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(right))
    }
}

/// A pair of the level being regrouped, with the node that holds it.
#[derive(Copy, Clone)]
struct Owned {
    pair: ChildPair,
    owner: u32,
}

/// An empty slot of [`regroup`]'s dense lookups.
const NONE: u32 = u32::MAX;

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
/// `∃.g` is reconstructed exactly as the OR over `g`'s cells. The cells are
/// numbered in the order the level's scan — node by node, pair by pair, and
/// each pair's atoms by left and then right side — first reaches one of their
/// atoms.
///
/// No atom is found by hashing it. The pairs are filed under the references
/// one side expands to ([`bucket_pairs`]), and the atoms filed under one
/// reference are told apart by their other side alone: a cell of a rewritten
/// child, which indexes an array stamped with the bucket it was last seen in.
/// A level of one node has one owner and so one cell, which
/// [`regroup_single_rows`] or, when its rows would cost more than the atoms,
/// [`regroup_single`] writes without the owner bookkeeping.
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
    let n_nodes = tdd.levels[parent.idx()].nodes.len();
    if n_nodes == 0 { return Ok(None); }
    // The pairs are filed by a side left as it is, when there is one: only a
    // rewritten side names the dense cells the stamp is indexed by.
    let (mut by, mut bucket_remap, mut stamped) = match (left_remap, right_remap) {
        (None, None) => return Ok(None),
        (Some(left), None) => (Side::Right, None, left),
        (None, Some(right)) => (Side::Left, None, right),
        (Some(left), Some(right)) => (Side::Left, Some(left), right),
    };
    if n_nodes == 1 {
        let plan = RowPlan::of(work, tdd.levels[parent.idx()].pairs_of_idx(0), left_remap, right_remap)?;
        if plan.pays {
            regroup_single_rows(work, tdd, parent, left_remap, right_remap, &plan)?;
            return Ok(None);
        }
    }
    let lim = work.eng.limits();
    let level = &tdd.levels[parent.idx()];
    let owned = scan_level(work, level)?;
    if let Some(bucket) = bucket_remap
        && fan_out(stamped, &owned, Side::Right) < fan_out(bucket, &owned, Side::Left)
    {
        // Both sides were rewritten: file by the one that expands into fewer
        // references.
        (by, bucket_remap, stamped) = (Side::Right, Some(stamped), bucket);
    }
    let buckets = bucket_pairs(work, &owned, by, bucket_remap)?;
    if n_nodes == 1 && !owned.is_empty() {
        regroup_single(work, tdd, parent, &owned, &buckets, by, stamped)?;
        lim.discard(owned);
        lim.discard(buckets);
        return Ok(None);
    }

    // Each atom gets an index, and its owner set — the old nodes contributing
    // it — is accumulated as a run of (atom, owner) entries. A bucket lists
    // its pairs in scan order and holds every pair an atom filed under it
    // comes from, so `last_owner` drops repeats, each atom's owners come out
    // sorted and unique, and `first` is the pair the scan expands it from
    // first.
    let n_stamp = cell_count(stamped);
    let mut stamp = Vec::new();
    lim.try_resize(&mut stamp, n_stamp, 0u32)?;
    let mut slot = Vec::new();
    lim.try_resize(&mut slot, n_stamp, 0u32)?;
    let mut atoms: Vec<ChildPair> = Vec::new();
    let mut first: Vec<u32> = Vec::new();
    let mut last_owner: Vec<u32> = Vec::new();
    let mut entries: Vec<(u32, u32)> = Vec::new();
    let mut run = 0u32;

    for bucket in buckets.chunk_by(|a, b| a.0 == b.0) {
        run = run.checked_add(1).ok_or(OperationError::IndexOverflow)?;
        let here = bucket[0].0;
        for &(_, s) in bucket {
            let Owned { pair, owner } = owned[s as usize];
            let other = ChildDecoder::structural().node(by.other().of(pair));
            let list = stamped.get(other.idx());
            work.gate.poll(list.len() as u64 + 1)?;
            for &there in list {
                let t = there as usize;
                let atom = if stamp[t] == run {
                    slot[t]
                } else {
                    let atom = u32::try_from(atoms.len()).map_err(|_| OperationError::IndexOverflow)?;
                    stamp[t] = run;
                    slot[t] = atom;
                    lim.try_push(&mut atoms, by.pair(here, there))?;
                    lim.try_push(&mut first, s)?;
                    lim.try_push(&mut last_owner, NONE)?;
                    atom
                };
                if last_owner[atom as usize] != owner {
                    last_owner[atom as usize] = owner;
                    lim.try_push(&mut entries, (atom, owner))?;
                }
            }
        }
    }

    lim.discard(stamp);
    lim.discard(slot);
    lim.discard(last_owner);
    lim.discard(buckets);
    lim.discard(owned);

    // Owner sets, one run per atom: each atom's entries were produced in
    // increasing owner order, and the stable scatter keeps each run sorted.
    let owners = Runs::pack(lim, atoms.len(), &entries, 0u32)?;
    lim.discard(entries);

    // Group atoms by owner set → one cell per distinct owner set. A single
    // owner finds its cell through `by_owner`; a larger set by hashing its
    // run and comparing it against the cells that hash alike, chained
    // through `next_alike`. Each cell keeps the atom the scan reaches first.
    let scan_key = |atom: u32| (first[atom as usize], atoms[atom as usize]);
    let mut by_owner = Vec::new();
    lim.try_resize(&mut by_owner, n_nodes, NONE)?;
    let mut by_hash: FxHashMap<u64, u32> = FxHashMap::default();
    let mut next_alike: Vec<u32> = Vec::new();
    let mut cell_first: Vec<u32> = Vec::new();
    let mut cell_of = Vec::new();
    lim.reserve_exact(&mut cell_of, atoms.len())?;

    for atom in 0..atoms.len() {
        work.poll()?;
        let mine = owners.get(atom);
        let atom = u32::try_from(atom).map_err(|_| OperationError::IndexOverflow)?;
        let digest = match *mine {
            [_] => None,
            _ => Some(owner_set_hash(mine)),
        };
        let found = match digest {
            None => by_owner[mine[0] as usize],
            Some(digest) => {
                let mut cell = by_hash.get(&digest).copied().unwrap_or(NONE);
                while cell != NONE && owners.get(cell_first[cell as usize] as usize) != mine {
                    cell = next_alike[cell as usize];
                }
                cell
            }
        };
        let cell = if found != NONE {
            let kept = &mut cell_first[found as usize];
            if scan_key(atom) < scan_key(*kept) { *kept = atom; }
            found
        } else {
            let cell = u32::try_from(cell_first.len()).ok()
                .filter(|&cell| cell != NONE)
                .ok_or(OperationError::IndexOverflow)?;
            let alike = match digest {
                None => {
                    by_owner[mine[0] as usize] = cell;
                    NONE
                }
                Some(digest) => {
                    lim.reserve_map(&mut by_hash, 1)?;
                    by_hash.insert(digest, cell).unwrap_or(NONE)
                }
            };
            lim.try_push(&mut next_alike, alike)?;
            lim.try_push(&mut cell_first, atom)?;
            cell
        };
        cell_of.push(cell);
    }

    lim.discard(by_owner);
    lim.discard(by_hash);
    lim.discard(next_alike);

    // Number the cells in the order the scan reaches them, which is the order
    // of their first atoms, and list each cell's owners in that order: the
    // fan-out of a node then comes out ascending.
    let n_cells = cell_first.len();
    let mut numbered = Vec::new();
    lim.reserve_exact(&mut numbered, n_cells)?;
    for (cell, &atom) in cell_first.iter().enumerate() {
        let (first_pair, pair) = scan_key(atom);
        numbered.push((first_pair, pair, u32::try_from(cell).map_err(|_| OperationError::IndexOverflow)?));
    }
    if !numbered.is_sorted() {
        numbered.sort_unstable();
    }
    let mut number = Vec::new();
    lim.try_resize(&mut number, n_cells, 0u32)?;
    // One entry per owner of each cell.
    let mut fanout: Vec<(u32, u32)> = Vec::new();
    lim.reserve_exact(&mut fanout, cell_first.iter().map(|&atom| owners.get(atom as usize).len()).sum())?;
    for (new_cell, &(_, _, cell)) in numbered.iter().enumerate() {
        let new_cell = u32::try_from(new_cell).map_err(|_| OperationError::IndexOverflow)?;
        number[cell as usize] = new_cell;
        for &owner in owners.get(cell_first[cell as usize] as usize) {
            work.poll()?;
            fanout.push((owner, new_cell));
        }
    }
    let mut cell_pairs: Vec<(u32, ChildPair)> = Vec::new();
    lim.reserve_exact(&mut cell_pairs, atoms.len())?;
    for (&cell, &pair) in cell_of.iter().zip(&atoms) {
        // `atoms` holds distinct atoms and the pair is injective in the
        // atom, so every pair within a cell is already distinct.
        cell_pairs.push((number[cell as usize], pair));
    }

    // One cell per node and one owner per cell: the owner sets are singletons
    // and pairwise distinct, and cells are numbered in node order, so cell `i`
    // is node `i` expanded. Every reference from above still names the node
    // it named, and the level above would hand back its own pairs.
    let unchanged = n_cells == n_nodes && fanout.len() == n_nodes;

    lim.discard(atoms);
    lim.discard(first);
    lim.discard(owners);
    lim.discard(cell_first);
    lim.discard(cell_of);
    lim.discard(numbered);
    lim.discard(number);
    let mut new_nodes = Transient::new(lim, Runs::pack(lim, n_cells, &cell_pairs, TRUE_PAIR)?);
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

/// [`regroup`] for a level of one node, which owns every atom: the level
/// becomes the one cell holding them all, and its partition is unchanged.
///
/// The cell comes out in canonical order without a comparison sort over it.
/// Filed by right reference, the buckets ascend by their right side and a
/// stable sort by left cell finishes the order; filed by left reference, the
/// buckets ascend by their left side and each sorts only its own atoms.
fn regroup_single(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    owned: &[Owned],
    buckets: &[(u32, u32)],
    by: Side,
    stamped: &Remap,
) -> Result<(), OperationError> {
    let lim = work.eng.limits();
    let n_stamp = cell_count(stamped);
    let mut stamp = Vec::new();
    lim.try_resize(&mut stamp, n_stamp, 0u32)?;
    let mut pairs: Vec<ChildPair> = Vec::new();
    let mut run = 0u32;
    for bucket in buckets.chunk_by(|a, b| a.0 == b.0) {
        run = run.checked_add(1).ok_or(OperationError::IndexOverflow)?;
        let here = bucket[0].0;
        let start = pairs.len();
        for &(_, s) in bucket {
            let other = ChildDecoder::structural().node(by.other().of(owned[s as usize].pair));
            let list = stamped.get(other.idx());
            work.gate.poll(list.len() as u64 + 1)?;
            for &there in list {
                let seen = &mut stamp[there as usize];
                if *seen != run {
                    *seen = run;
                    lim.try_push(&mut pairs, by.pair(here, there))?;
                }
            }
        }
        if by == Side::Left {
            sort_pairs(&mut pairs[start..]);
        }
    }
    lim.discard(stamp);
    if by == Side::Right {
        let width = usize::BITS - n_stamp.saturating_sub(1).leading_zeros();
        sort_by_key_stable(work, &mut pairs, width, |pair| pair.left.0)?;
    }
    debug_assert!(pairs.is_sorted() && pairs.windows(2).all(|w| w[0] != w[1]),
        "a single cell's atoms are distinct and in canonical order");
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    level.push_node(lim, &pairs)?;
    lim.discard(pairs);
    work.emitted += 1;
    lim.level_done(work.emitted)?;
    tdd.try_invalidate(work.eng, parent)?;
    Ok(())
}

/// What one read of a single node's pairs says about writing its cell row by
/// row ([`regroup_single_rows`]).
///
/// A row is a left reference of the cell: a cell of the left child when it
/// was rewritten, else a stored left reference. A column is a right reference
/// the same way. Counts are `u64` so products of them cannot wrap.
#[derive(Debug)]
struct RowPlan {
    /// Whether the rows cost no more to read than the atoms the pairs expand
    /// to: every row's words are read and cleared whether the pairs set any,
    /// and a node of several rows first files its pairs and inverts the left
    /// child's map.
    pays: bool,
    /// How many rows there are: the left child's cells, or one past the
    /// largest stored left reference.
    rows: u64,
    /// The words of one row's bit array.
    words: u64,
    /// One past the largest left reference the pairs store: the keys they are
    /// filed under.
    left_keys: u64,
    /// The row every pair lands in, when there is only one.
    single: Option<u32>,
    /// The most pairs the cell can come to hold: its atoms when they were
    /// counted, and never more than its rows hold columns.
    most: u64,
}

impl RowPlan {
    fn of(
        work: &mut Rewrite<'_>,
        pairs: &[ChildPair],
        left_remap: Option<&Remap>,
        right_remap: Option<&Remap>,
    ) -> Result<RowPlan, OperationError> {
        let n_pairs = pairs.len() as u64;
        let left_cells = left_remap.map(|map| cell_count(map) as u64);
        if let (Some(1), Some(right)) = (left_cells, right_remap) {
            // Both children rewritten, the left one into a single cell: there
            // is one row, and when it has no more words than the node has
            // pairs it pays however many atoms there are, so they are not
            // counted.
            let columns = cell_count(right) as u64;
            let words = columns.div_ceil(64);
            if words <= n_pairs {
                return Ok(RowPlan { pays: true, rows: 1, words, left_keys: 0, single: Some(0), most: columns });
            }
        }
        work.gate.poll(n_pairs)?;
        let width = |remap: Option<&Remap>, side: EncodedChildRef| match remap {
            Some(map) => map.get(ChildDecoder::structural().node(side).idx()).len() as u64,
            None => 1,
        };
        let (mut atoms, mut left_min, mut left_max, mut right_max) = (0u64, u32::MAX, 0u32, 0u32);
        for pair in pairs {
            atoms = atoms.saturating_add(width(left_remap, pair.left) * width(right_remap, pair.right));
            left_min = left_min.min(pair.left.0);
            left_max = left_max.max(pair.left.0);
            right_max = right_max.max(pair.right.0);
        }
        let rows = left_cells.unwrap_or(u64::from(left_max) + 1);
        let columns = match right_remap {
            Some(map) => cell_count(map) as u64,
            None => u64::from(right_max) + 1,
        };
        let words = columns.div_ceil(64);
        let left_keys = u64::from(left_max) + 1;
        let single = match left_remap {
            Some(_) => (rows == 1).then_some(0),
            None => (left_min == left_max).then_some(left_min),
        };
        let reads = match single {
            Some(_) => words,
            None => {
                let inverted = left_remap.map_or(0, |map| map.items.len() as u64 + rows);
                (rows * words).saturating_add(left_keys).saturating_add(inverted)
            }
        };
        Ok(RowPlan {
            pays: reads <= atoms.saturating_add(n_pairs),
            rows,
            words,
            left_keys,
            single,
            most: atoms.min(rows.saturating_mul(columns)),
        })
    }
}

/// [`regroup`] for a level of one node, written row by row: its one cell
/// comes out in canonical order with no sort and no per-atom test for a
/// repeat.
///
/// A row gathers its right references as bits of one word array — each pair
/// whose left side expands to the row sets the bits its right side expands
/// to, and a repeat sets a bit already set — then reads them back in
/// ascending order, which is the canonical order within the row, and appends
/// them to the level's arena. The rows are visited in ascending order: a
/// rewritten left child's through the inverse of its map, the nodes each of
/// its cells came from, with the pairs filed by left reference. The pairs of
/// a node that several rows visit are marked once into words of their own
/// when they fill enough of a row ([`dense_groups`]), and each of those rows
/// then ORs the words in.
///
/// `plan` is [`RowPlan::of`] for the level's node.
fn regroup_single_rows(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left_remap: Option<&Remap>,
    right_remap: Option<&Remap>,
    plan: &RowPlan,
) -> Result<(), OperationError> {
    let lim = work.eng.limits();
    let pairs = tdd.levels[parent.idx()].pairs_of_idx(0);
    let mut row = Vec::new();
    // A row's words were counted from a `u32` column bound, so they fit.
    lim.try_resize(&mut row, plan.words as usize, 0u64)?;
    if let Some(only) = plan.single {
        // A pair whose left side expands to no cell has no atom. When every
        // left reference expands to one, the left sides are not read.
        let every_left_expands = left_remap.is_none_or(|map| map.starts.windows(2).all(|w| w[0] < w[1]));
        if every_left_expands {
            for pair in pairs {
                let marked = mark(&mut row, std::slice::from_ref(&pair.right.0), right_remap);
                work.gate.poll(marked + 1)?;
            }
        } else {
            for pair in pairs {
                let expands = left_remap
                    .is_none_or(|map| !map.get(ChildDecoder::structural().node(pair.left).idx()).is_empty());
                let marked = if expands { mark(&mut row, std::slice::from_ref(&pair.right.0), right_remap) } else { 0 };
                work.gate.poll(marked + 1)?;
            }
        }
        work.gate.poll(plan.words)?;
        let level = &mut tdd.levels[parent.idx()];
        level.clear();
        drain_row(lim, &mut row, only, &mut level.pairs, 0, 1, plan.most)?;
    } else {
        let filed = |pair: &ChildPair| match left_remap {
            Some(_) => (ChildDecoder::structural().node(pair.left).0, pair.right.0),
            None => (pair.left.0, pair.right.0),
        };
        let groups = Transient::new(lim, Runs::pack_by(lim, plan.left_keys as usize, pairs, filed, 0u32)?);
        let (inverse, dense) = match left_remap {
            Some(map) => (
                Some(Transient::new(lim, map.transpose(lim, plan.rows as usize)?)),
                Transient::new(lim, dense_groups(work, &groups, map, right_remap, plan.words)?),
            ),
            None => (None, Transient::new(lim, Runs { starts: Vec::new(), items: Vec::new() })),
        };
        let level = &mut tdd.levels[parent.idx()];
        level.clear();
        for left in 0..plan.rows {
            // Rows are left references, which are `u32`.
            let left = left as u32;
            let own = [left];
            let nodes = match inverse.as_deref() {
                Some(inverse) => inverse.get(left as usize),
                None => &own[..],
            };
            for &node in nodes {
                let bits = dense.get(node as usize);
                if bits.is_empty() {
                    let marked = mark(&mut row, groups.get(node as usize), right_remap);
                    work.gate.poll(marked + 1)?;
                } else {
                    for (word, &bit) in row.iter_mut().zip(bits) {
                        *word |= bit;
                    }
                    work.gate.poll(plan.words)?;
                }
            }
            work.gate.poll(plan.words)?;
            drain_row(lim, &mut row, left, &mut level.pairs, u64::from(left), plan.rows, plan.most)?;
        }
    }
    lim.discard(row);
    let level = &mut tdd.levels[parent.idx()];
    debug_assert!(level.pairs.is_sorted() && level.pairs.windows(2).all(|w| w[0] != w[1]),
        "a single cell's atoms are distinct and in canonical order");
    let len = level.pairs.len();
    if len < 2 {
        // A cell of at most one pair is pushed the way any node is, which
        // stores a lone pair in the node itself when it fits.
        let lone = level.pairs.pop();
        level.push_node(lim, lone.as_slice())?;
    } else {
        let before = level.arena_capacity_bytes();
        level.try_push_multi_by_range(0, len).map_err(|()| OperationError::OverBudget)?;
        lim.charge_bytes(level.arena_capacity_bytes().saturating_sub(before))?;
    }
    work.emitted += 1;
    lim.level_done(work.emitted)?;
    tdd.try_invalidate(work.eng, parent)?;
    Ok(())
}

/// The groups of [`regroup_single_rows`] worth marking once: run `g` holds a
/// row's worth of words with the columns of group `g` set when at least two
/// rows visit that group — `left_remap` maps its node to several cells — and
/// its columns number at least a quarter of the `words`, so that ORing the
/// words in costs no more than marking the columns again; every other run is
/// empty.
fn dense_groups(
    work: &mut Rewrite<'_>,
    groups: &Runs<u32>,
    left_remap: &Remap,
    right_remap: Option<&Remap>,
    words: u64,
) -> Result<Runs<u64>, OperationError> {
    let lim = work.eng.limits();
    let keys = groups.len();
    let mut starts = Vec::new();
    lim.reserve_exact(&mut starts, keys + 1)?;
    starts.push(0u32);
    let mut total = 0u64;
    for g in 0..keys {
        let rights = groups.get(g);
        work.gate.poll(rights.len() as u64 + 1)?;
        let columns: u64 = match right_remap {
            Some(map) => rights.iter().map(|&right| right_cells(map, right).len() as u64).sum(),
            None => rights.len() as u64,
        };
        if left_remap.get(g).len() >= 2 && columns > 0 && columns.saturating_mul(4) >= words {
            total += words;
        }
        starts.push(u32::try_from(total).map_err(|_| OperationError::IndexOverflow)?);
    }
    let mut dense = Runs { starts, items: Vec::new() };
    // The total was checked against `u32` above.
    lim.try_resize(&mut dense.items, total as usize, 0u64)?;
    for g in 0..keys {
        let bits = dense.get_mut(g);
        if !bits.is_empty() {
            let marked = mark(bits, groups.get(g), right_remap);
            work.gate.poll(marked + 1)?;
        }
    }
    Ok(dense)
}

/// The cells `remap` lists for the stored right reference `right`.
#[inline]
fn right_cells(remap: &Remap, right: u32) -> &[u32] {
    remap.get(ChildDecoder::structural().node(EncodedChildRef::from_raw(right)).idx())
}

/// Set in `row` the columns that the stored right references `rights` stand
/// for: the cells `remap` lists for them, or the references themselves.
/// Returns how many columns were set, counting a repeat.
#[inline]
fn mark(row: &mut [u64], rights: &[u32], remap: Option<&Remap>) -> u64 {
    match remap {
        None => {
            for &right in rights {
                row[(right >> 6) as usize] |= 1u64 << (right & 63);
            }
            rights.len() as u64
        }
        Some(map) => {
            let mut marked = 0u64;
            for &right in rights {
                let cells = right_cells(map, right);
                marked += cells.len() as u64;
                for &cell in cells {
                    row[(cell >> 6) as usize] |= 1u64 << (cell & 63);
                }
            }
            marked
        }
    }
}

/// Append `(left, c)` to `out` for every column `c` set in `row`, in
/// ascending order, clearing the row for the next. `out` makes room through
/// [`grow_rows`], row `done` of `rows` being written into a cell of at most
/// `most` pairs.
#[inline]
fn drain_row(
    lim: &crate::limits::Limits,
    row: &mut [u64],
    left: u32,
    out: &mut Vec<ChildPair>,
    done: u64,
    rows: u64,
    most: u64,
) -> Result<(), OperationError> {
    let left = EncodedChildRef::from_raw(left);
    for (at, word) in row.iter_mut().enumerate() {
        let mut bits = std::mem::take(word);
        // A row has at most 2^26 words, so its columns fit a `u32`.
        let base = (at as u32) << 6;
        while bits != 0 {
            if out.len() == out.capacity() {
                grow_rows(lim, out, done, rows, most)?;
            }
            out.push(ChildPair { left, right: EncodedChildRef::from_raw(base | bits.trailing_zeros()) });
            bits &= bits - 1;
        }
    }
    Ok(())
}

/// Room for more of the pairs [`drain_row`] appends: at least half again
/// what `out` holds, and once enough rows are done, the rows still to come at
/// the average so far, but never past `most` pairs in all.
#[cold]
#[inline(never)]
fn grow_rows(
    lim: &crate::limits::Limits,
    out: &mut Vec<ChildPair>,
    done: u64,
    rows: u64,
    most: u64,
) -> Result<(), OperationError> {
    let len = out.len() as u64;
    let rest = if done >= 64 { len.div_ceil(done) * (rows - done) } else { len };
    let additional = rest.max(len / 2).min(most.saturating_sub(len)).max(1);
    lim.reserve_exact(out, usize::try_from(additional).map_err(|_| OperationError::OverBudget)?)
}

/// The level's pairs in scan order — node by node, each node's pairs as
/// stored — with the node holding each.
fn scan_level(work: &mut Rewrite<'_>, level: &TddLevel) -> Result<Vec<Owned>, OperationError> {
    let lim = work.eng.limits();
    let mut owned = Vec::new();
    lim.reserve_exact(&mut owned, level.live_pairs())?;
    for i in 0..level.nodes.len() {
        work.poll()?;
        let owner = u32::try_from(i).map_err(|_| OperationError::IndexOverflow)?;
        for &pair in level.pairs_of_idx(i) {
            work.poll()?;
            lim.try_push(&mut owned, Owned { pair, owner })?;
        }
    }
    // Positions in the scan are stored as `u32`.
    u32::try_from(owned.len()).map_err(|_| OperationError::IndexOverflow)?;
    Ok(owned)
}

/// How many references `side` of the scanned pairs expands to through `remap`.
fn fan_out(remap: &Remap, owned: &[Owned], side: Side) -> usize {
    owned.iter()
        .map(|o| remap.get(ChildDecoder::structural().node(side.of(o.pair)).idx()).len())
        .sum()
}

/// One past the largest cell `remap` maps to: the length of an array indexed
/// by those cells.
fn cell_count(remap: &Remap) -> usize {
    remap.items.iter().max().map_or(0, |&cell| cell as usize + 1)
}

/// File the scanned pairs under the references their `by` side expands to,
/// through `remap` when that side was rewritten: one `(reference, position)`
/// record per reference, `position` indexing `owned`, sorted by reference and
/// then position — scan order within one reference.
fn bucket_pairs(
    work: &mut Rewrite<'_>,
    owned: &[Owned],
    by: Side,
    remap: Option<&Remap>,
) -> Result<Vec<(u32, u32)>, OperationError> {
    let lim = work.eng.limits();
    let mut records = Vec::new();
    lim.reserve_exact(&mut records, remap.map_or(owned.len(), |map| fan_out(map, owned, by)))?;
    let mut kept = 0u32;
    let mut span = 0u32;
    for (position, o) in owned.iter().enumerate() {
        work.poll()?;
        // `scan_level` bounds the positions.
        let position = position as u32;
        for &bucket in expand(remap, &mut kept, by.of(o.pair)) {
            span |= bucket;
            records.push((bucket, position));
        }
    }
    sort_by_key_stable(work, &mut records, u32::BITS - span.leading_zeros(), |&(bucket, _)| bucket)?;
    Ok(records)
}

/// Radix digits at most this wide: a histogram of 2048 counters.
const RADIX_BITS: u32 = 11;

/// Below this many items an insertion sort replaces the counting passes.
const RADIX_MIN: usize = 64;

/// Sort `items` stably by `key`, whose values are below `2^bits`: a least
/// significant digit radix sort, one counting pass per digit, which skips a
/// digit every item shares and a list already in order.
fn sort_by_key_stable<T: Copy>(
    work: &mut Rewrite<'_>,
    items: &mut Vec<T>,
    bits: u32,
    key: impl Fn(&T) -> u32,
) -> Result<(), OperationError> {
    if items.is_sorted_by_key(&key) { return Ok(()); }
    let n = items.len();
    if n < RADIX_MIN {
        for i in 1..n {
            let item = items[i];
            let k = key(&item);
            let mut j = i;
            while j > 0 && key(&items[j - 1]) > k {
                items[j] = items[j - 1];
                j -= 1;
            }
            items[j] = item;
        }
        return Ok(());
    }
    let lim = work.eng.limits();
    let everyone = u32::try_from(n).map_err(|_| OperationError::IndexOverflow)?;
    let passes = bits.div_ceil(RADIX_BITS);
    let digit = bits.div_ceil(passes);
    let mask = (1u32 << digit) - 1;
    let mut counts = Vec::new();
    lim.try_resize(&mut counts, mask as usize + 1, 0u32)?;
    let mut other = Vec::new();
    lim.try_resize(&mut other, n, items[0])?;
    for pass in 0..passes {
        work.gate.poll(n as u64)?;
        let shift = pass * digit;
        counts.fill(0);
        for item in items.iter() {
            counts[((key(item) >> shift) & mask) as usize] += 1;
        }
        if counts.contains(&everyone) { continue; }
        let mut sum = 0u32;
        for count in &mut counts {
            let here = *count;
            *count = sum;
            sum += here;
        }
        for item in items.iter() {
            let slot = &mut counts[((key(item) >> shift) & mask) as usize];
            other[*slot as usize] = *item;
            *slot += 1;
        }
        std::mem::swap(items, &mut other);
    }
    lim.discard(other);
    lim.discard(counts);
    Ok(())
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
        debug_assert!(pairs.windows(2).all(|w| w[0] != w[1]), "a regrouped cell repeats a pair");
        level.push_node(work.eng.limits(), pairs)?;
        work.emitted += 1;
        work.eng.limits().level_done(work.emitted)?;
    }
    tdd.try_invalidate(work.eng, parent)?;
    Ok(())
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
