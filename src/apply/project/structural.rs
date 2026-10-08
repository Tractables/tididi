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
    /// Whether every key holds exactly one item, which is then `items[key]`.
    fn one_each(&self) -> bool {
        self.items.len() == self.len() && self.starts.windows(2).all(|w| w[0] + 1 == w[1])
    }

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
    // The levels that may hold a node their parent level does not name
    // (`Dirty::loose`), where that is known: none after the prune. The sweep
    // adds none, so the prune after it walks only as far as these and the
    // levels that lose a node.
    let loose = tdd.dirty.loose().map(<[u32]>::to_vec);
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
    // No level holds a node its parent level does not name that did not
    // before. A level the sweep left alone is named through every pair that
    // named it: such a pair is expanded over the cells its other side became,
    // at least one, as every node has a pair and every pair a cell. A
    // regrouped level's cells are each in the map of an old node, and the
    // level above expands every reference to that node over them. A
    // quantified level's one node is in the map of every old node, or is the
    // one node the level held, or, under another quantified level, is the
    // node that level's `⊤` pair names (index 0 of an internal child). The
    // root's other nodes are dropped by the walk from the output.
    tdd.dirty.set_loose(loose);
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
    let n_nodes = tdd.levels[level.idx()].nodes().len();
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
/// [`regroup_single_by_left`] writes run by run of its left side (an
/// untouched one first, a rewritten one when the rows would cost more than the
/// atoms), [`regroup_single_rows`] row by row, or [`regroup_single`] without
/// the owner bookkeeping when neither applies; and a
/// level each of whose pairs expands to one atom reads its owner sets off the
/// atoms alone ([`regroup_atom_per_pair`]).
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
    let n_nodes = tdd.levels[parent.idx()].nodes().len();
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
        if left_remap.is_none() && regroup_single_by_left(work, tdd, parent, None, right_remap)? {
            return Ok(None);
        }
        // An implicit level's one node's pairs are generated into `buf`.
        let mut buf = Vec::new();
        let plan = RowPlan::of(work, tdd.levels[parent.idx()].pairs_read(0, &mut buf), left_remap, right_remap)?;
        if plan.pays {
            regroup_single_rows(work, tdd, parent, left_remap, right_remap, &plan)?;
            return Ok(None);
        }
        if left_remap.is_some() && regroup_single_by_left(work, tdd, parent, left_remap, right_remap)? {
            return Ok(None);
        }
    }
    if let OneAtom::Done(remap) = regroup_atom_per_pair(work, tdd, parent, left_remap, right_remap)? {
        return Ok(remap);
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

/// What [`regroup_atom_per_pair`] did with a level.
enum OneAtom {
    /// The level is not of its shape and was left as it was.
    Declined,
    /// The level was regrouped; the map for the level above, `None` when the
    /// partition did not change.
    Done(Option<Remap>),
}

/// The most values a stamped side of [`regroup_atom_per_pair`] may span per
/// pair of the level, beyond a fixed allowance: the stamp is an array over
/// them.
const ONE_ATOM_STAMP_PER_PAIR: usize = 4;

/// [`regroup`] for a level each of whose pairs expands to one atom, every
/// rewritten side mapping the pair's reference to one cell: the shape of
/// the levels above a block of variables quantified out of a diagram that
/// holds one node per value of the block, where the maps below merge few
/// cells.
///
/// The owner-set rule then reads off the atoms alone. An atom only one node
/// has goes to that node's own cell, which holds every such atom of the
/// node; an atom of several nodes — two cells below were merged — goes to
/// the cell of its owner set, found by hashing the set as [`regroup`] finds
/// it. Cells are numbered by the first pair the scan expands into them, which
/// is [`regroup`]'s order since each pair has one atom, and a level none of
/// whose atoms has two owners keeps its partition.
///
/// Every pass but one reads the level in scan order. The atoms are told
/// apart without hashing: the pairs are put in order of one side by a stable
/// radix sort, and within a run of one value of that side the other side's
/// values are told apart by an array stamped with the run. That pass finds
/// the atoms of several owners and lists their pairs; the rest of the level
/// is written node by node from the atoms as the scan met them. The side
/// stamped is the one spanning fewer values, and a level whose sides both
/// span more than a few per pair is declined, as is any level of another
/// shape, before anything is written.
fn regroup_atom_per_pair(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left_remap: Option<&Remap>,
    right_remap: Option<&Remap>,
) -> Result<OneAtom, OperationError> {
    /// A pair's atom, its sides ordered as the sort and the stamp read them,
    /// with the pair's position in the scan and the node holding it.
    #[derive(Copy, Clone)]
    struct Keyed {
        sorted: u32,
        stamped: u32,
        pair: u32,
        owner: u32,
    }
    /// An atom while its run is read: the node of its first pair, and its
    /// number among the shared atoms once a second node is found to have it.
    #[derive(Copy, Clone)]
    struct Seen {
        owner: u32,
        shared: u32,
    }
    let lim = work.eng.limits();
    let level = &tdd.levels[parent.idx()];
    let n = level.nodes().len();
    let n_pairs = level.live_pairs();
    if n < 2 || u32::try_from(n_pairs).is_err() || u32::try_from(n).is_err() {
        return Ok(OneAtom::Declined);
    }
    // A map of one cell per node is read at its item alone.
    let (left_each, right_each) = (left_remap.is_some_and(Remap::one_each), right_remap.is_some_and(Remap::one_each));
    let one_cell = |remap: Option<&Remap>, each: bool, side: EncodedChildRef| {
        let idx = ChildDecoder::structural().node(side).idx();
        match remap {
            Some(map) if each => map.items.get(idx).copied(),
            Some(map) => match map.get(idx) {
                [cell] => Some(*cell),
                _ => None,
            },
            None => Some(side.0),
        }
    };
    // Each pair's atom in scan order, where each node's run starts, and the
    // atoms keyed for the sort.
    let mut atoms: Transient<'_, Vec<ChildPair>> = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut atoms, n_pairs)?;
    let mut starts: Transient<'_, Vec<u32>> = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut starts, n + 1)?;
    let (mut left_max, mut right_max) = (0u32, 0u32);
    {
        let (atoms, starts): (&mut Vec<ChildPair>, &mut Vec<u32>) = (&mut atoms, &mut starts);
        // An implicit level's pairs are generated into `buf` a node at a time.
        let mut buf = Vec::new();
        for i in 0..n {
            let pairs = level.pairs_read(i, &mut buf);
            if pairs.is_empty() {
                return Ok(OneAtom::Declined);
            }
            work.gate.poll(pairs.len() as u64)?;
            // The pairs were counted in a `u32`.
            starts.push(atoms.len() as u32);
            for pair in pairs {
                let (Some(left), Some(right)) =
                    (one_cell(left_remap, left_each, pair.left), one_cell(right_remap, right_each, pair.right))
                else {
                    return Ok(OneAtom::Declined);
                };
                left_max = left_max.max(left);
                right_max = right_max.max(right);
                atoms.push(ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(right)));
            }
        }
        starts.push(atoms.len() as u32);
    }
    let stamp_left = left_max <= right_max;
    let span = left_max.min(right_max) as usize + 1;
    if span > n_pairs.saturating_mul(ONE_ATOM_STAMP_PER_PAIR).saturating_add(1 << 16) {
        return Ok(OneAtom::Declined);
    }
    let mut keyed = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut keyed, n_pairs)?;
    for (owner, node) in starts.windows(2).enumerate() {
        for p in node[0]..node[1] {
            let atom = atoms[p as usize];
            let (sorted, stamped) = if stamp_left { (atom.right.0, atom.left.0) } else { (atom.left.0, atom.right.0) };
            // `owner` fits: the nodes were counted in a `u32`.
            keyed.push(Keyed { sorted, stamped, pair: p, owner: owner as u32 });
        }
    }
    work.gate.poll(n_pairs as u64)?;
    let bits = u32::BITS - left_max.max(right_max).leading_zeros();
    sort_by_key_stable(work, &mut keyed, bits, |k| k.sorted)?;

    // The atoms of several owners. A run holds every pair of each of its
    // atoms, in scan order, so an atom's later owners differ from its first
    // exactly when it is shared; a second read of a run that found one lists
    // every pair of its shared atoms, numbering those atoms as it goes.
    let mut stamp = Transient::new(lim, Vec::new());
    lim.try_resize(&mut stamp, span, 0u32)?;
    let mut slot = Transient::new(lim, Vec::new());
    lim.try_resize(&mut slot, span, 0u32)?;
    let mut seen: Transient<'_, Vec<Seen>> = Transient::new(lim, Vec::new());
    // Each pair's shared atom, `NONE` for a pair whose atom has one owner.
    let mut shared_of = Transient::new(lim, Vec::new());
    lim.try_resize(&mut shared_of, n_pairs, NONE)?;
    let mut shared_atoms: Transient<'_, Vec<ChildPair>> = Transient::new(lim, Vec::new());
    {
        let (stamp, slot, seen, shared_of): (&mut Vec<u32>, &mut Vec<u32>, &mut Vec<Seen>, &mut Vec<u32>) =
            (&mut stamp, &mut slot, &mut seen, &mut shared_of);
        let mut run = 0u32;
        for same in keyed.chunk_by(|a, b| a.sorted == b.sorted) {
            run += 1;
            work.gate.poll(same.len() as u64)?;
            seen.clear();
            let mut found = false;
            for k in same {
                let s = k.stamped as usize;
                if stamp[s] != run {
                    stamp[s] = run;
                    // A run's atoms number no more than its pairs.
                    slot[s] = seen.len() as u32;
                    lim.try_push(seen, Seen { owner: k.owner, shared: NONE })?;
                } else if seen[slot[s] as usize].owner != k.owner {
                    seen[slot[s] as usize].owner = NONE;
                    found = true;
                }
            }
            if !found {
                continue;
            }
            for k in same {
                let atom = &mut seen[slot[k.stamped as usize] as usize];
                if atom.owner != NONE {
                    continue;
                }
                if atom.shared == NONE {
                    atom.shared = u32::try_from(shared_atoms.len()).map_err(|_| OperationError::IndexOverflow)?;
                    lim.try_push(&mut shared_atoms, atoms[k.pair as usize])?;
                }
                shared_of[k.pair as usize] = atom.shared;
            }
        }
    }
    drop(keyed);
    drop(stamp);
    drop(slot);
    drop(seen);

    if shared_atoms.is_empty() {
        // Each node keeps one cell, its own atoms: the partition is kept and
        // the level is written over in node order.
        let level = &mut tdd.levels[parent.idx()];
        level.clear();
        lim.reserve_exact(level.nodes.stored_mut(), n)?;
        for node in starts.windows(2) {
            write_cell(work, level, &mut atoms[node[0] as usize..node[1] as usize])?;
        }
        lim.level_done(work.emitted)?;
        tdd.try_invalidate(work.eng, parent)?;
        return Ok(OneAtom::Done(None));
    }

    // The shared atoms grouped by owner set as [`regroup`] groups them. When
    // every node has one pair, distinct atoms have disjoint owner sets, and
    // each shared atom is a group of its own.
    let shared: &[u32] = &shared_of;
    let n_shared = shared_atoms.len();
    let one_pair_each = n_pairs == n;
    let mut group_of: Transient<'_, Vec<u32>> = Transient::new(lim, Vec::new());
    let n_groups = if one_pair_each {
        n_shared
    } else {
        // Each shared atom's owners, ascending: the scan meets them in order.
        let mut last = Transient::new(lim, Vec::new());
        lim.try_resize(&mut last, n_shared, NONE)?;
        let mut entries: Transient<'_, Vec<(u32, u32)>> = Transient::new(lim, Vec::new());
        for (owner, node) in starts.windows(2).enumerate() {
            // `owner` fits: the nodes were counted in a `u32`.
            let owner = owner as u32;
            work.gate.poll(u64::from(node[1] - node[0]))?;
            for &atom in &shared[node[0] as usize..node[1] as usize] {
                if atom != NONE && last[atom as usize] != owner {
                    last[atom as usize] = owner;
                    lim.try_push(&mut entries, (atom, owner))?;
                }
            }
        }
        drop(last);
        let owners = Transient::new(lim, Runs::pack(lim, n_shared, &entries, 0u32)?);
        drop(entries);
        lim.reserve_exact(&mut group_of, n_shared)?;
        let mut group_atom: Transient<'_, Vec<u32>> = Transient::new(lim, Vec::new());
        let mut next_alike: Transient<'_, Vec<u32>> = Transient::new(lim, Vec::new());
        let mut by_hash: FxHashMap<u64, u32> = FxHashMap::default();
        for atom in 0..n_shared {
            work.poll()?;
            let mine = owners.get(atom);
            let digest = owner_set_hash(mine);
            let mut group = by_hash.get(&digest).copied().unwrap_or(NONE);
            while group != NONE && owners.get(group_atom[group as usize] as usize) != mine {
                group = next_alike[group as usize];
            }
            if group == NONE {
                group = u32::try_from(group_atom.len()).map_err(|_| OperationError::IndexOverflow)?;
                lim.reserve_map(&mut by_hash, 1)?;
                let alike = by_hash.insert(digest, group).unwrap_or(NONE);
                lim.try_push(&mut next_alike, alike)?;
                // The shared atoms were numbered in a `u32`.
                lim.try_push(&mut group_atom, atom as u32)?;
            }
            group_of.push(group);
        }
        lim.discard(by_hash);
        group_atom.len()
    };
    let group_ix: &[u32] = &group_of;
    let group = |atom: u32| if one_pair_each { atom } else { group_ix[atom as usize] };
    u32::try_from(n + n_groups).map_err(|_| OperationError::IndexOverflow)?;

    // Number the cells in the order the scan reaches them — a node's own
    // cell at its first unshared pair, a group's at its first pair — and map
    // each node to its cells, ascending. A cell before numbering is a node
    // `g` or `n` plus a group.
    let mut number = Transient::new(lim, Vec::new());
    lim.try_resize(&mut number, n + n_groups, NONE)?;
    let mut order: Transient<'_, Vec<u32>> = Transient::new(lim, Vec::new());
    let remap = {
        let (number, order): (&mut Vec<u32>, &mut Vec<u32>) = (&mut number, &mut order);
        let mut reach = |cell: usize| -> Result<u32, OperationError> {
            let numbered = &mut number[cell];
            if *numbered == NONE {
                *numbered = u32::try_from(order.len()).map_err(|_| OperationError::IndexOverflow)?;
                // `cell` is below `n` plus the groups, which fit a `u32`.
                lim.try_push(order, cell as u32)?;
            }
            Ok(*numbered)
        };
        let cell_of = |g: usize, p: u32| match shared[p as usize] {
            NONE => g,
            atom => n + group(atom) as usize,
        };
        if one_pair_each {
            // One cell per node, the pair's.
            let mut items = Transient::new(lim, Vec::new());
            lim.reserve_exact(&mut items, n)?;
            {
                let items: &mut Vec<u32> = &mut items;
                for g in 0..n {
                    if g % 4096 == 0 {
                        work.gate.poll(4096)?;
                    }
                    // Node `g`'s one pair is pair `g`, which fits a `u32`.
                    items.push(reach(cell_of(g, g as u32))?);
                }
            }
            let mut starts_out = Transient::new(lim, Vec::new());
            lim.reserve_exact(&mut starts_out, n + 1)?;
            // The nodes were counted in a `u32`.
            starts_out.extend(0..=n as u32);
            Runs { starts: starts_out.keep(), items: items.keep() }
        } else {
            let mut fanout: Transient<'_, Vec<(u32, u32)>> = Transient::new(lim, Vec::new());
            lim.reserve_exact(&mut fanout, n)?;
            let mut mine: Vec<u32> = Vec::new();
            for (g, node) in starts.windows(2).enumerate() {
                work.gate.poll(u64::from(node[1] - node[0]))?;
                mine.clear();
                for p in node[0]..node[1] {
                    let numbered = reach(cell_of(g, p))?;
                    if mine.last() != Some(&numbered) {
                        lim.try_push(&mut mine, numbered)?;
                    }
                }
                mine.sort_unstable();
                mine.dedup();
                for &cell in &mine {
                    // `g` fits: the nodes were counted in a `u32`.
                    lim.try_push(&mut fanout, (g as u32, cell))?;
                }
            }
            lim.discard(mine);
            Runs::pack(lim, n, &fanout, 0u32)?
        }
    };
    let n_cells = order.len();

    // Each group's atoms, and then every cell in its number's order: a
    // node's own cell is its unshared atoms, read in scan order again.
    let mut grouped: Transient<'_, Vec<(u32, ChildPair)>> = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut grouped, n_shared)?;
    grouped.extend(shared_atoms.iter().enumerate().map(|(atom, &pair)| (group(atom as u32), pair)));
    drop(shared_atoms);
    let mut groups = Transient::new(lim, Runs::pack(lim, n_groups, &grouped, TRUE_PAIR)?);
    drop(grouped);
    let mut own: Vec<ChildPair> = Vec::new();
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    lim.reserve_exact(level.nodes.stored_mut(), n_cells)?;
    for &cell in order.iter() {
        match (cell as usize).checked_sub(n) {
            Some(group) => write_cell(work, level, groups.get_mut(group))?,
            None => {
                own.clear();
                let (from, to) = (starts[cell as usize] as usize, starts[cell as usize + 1] as usize);
                for (p, &atom) in shared[from..to].iter().enumerate() {
                    if atom == NONE {
                        lim.try_push(&mut own, atoms[from + p])?;
                    }
                }
                write_cell(work, level, &mut own)?;
            }
        }
    }
    lim.discard(own);
    lim.level_done(work.emitted)?;
    tdd.try_invalidate(work.eng, parent)?;
    drop(groups);
    drop(order);
    drop(number);
    drop(group_of);
    drop(shared_of);
    drop(atoms);
    drop(starts);
    Ok(OneAtom::Done(Some(remap)))
}

/// Write one cell into `level` from its atoms, sorted and without repeats,
/// held to the output cap as [`write_level`] holds each cell.
fn write_cell(work: &mut Rewrite<'_>, level: &mut TddLevel, pairs: &mut [ChildPair]) -> Result<(), OperationError> {
    let lim = work.eng.limits();
    work.poll()?;
    sort_pairs(pairs);
    let kept = dedup_sorted(pairs);
    level.push_node(lim, &pairs[..kept])?;
    work.emitted += 1;
    lim.check_output_cap(work.emitted)
}

/// The most pairs a run of [`regroup_single_by_left`] may average when its
/// right side was rewritten: a run's atoms are then sorted on their own, and a
/// node of longer runs is written as rows.
const BY_LEFT_PAIRS_PER_RUN: usize = 64;

/// [`regroup`] for a level of one node whose pairs are stored in order of
/// their left side, which canonical order implies: each run of one left
/// reference expands, cell by cell of that reference, to the cell beside the
/// cells its right sides map to, sorted and without repeats, in one pass over
/// the pairs. With the left side untouched the segments come out in canonical
/// order one after another; a rewritten left side can merge or reorder its
/// cells, and a stable sort by left cell then puts each cell's segments
/// together, in left order, a cell made of several sorted and deduplicated on
/// its own. `false` when the level is not of that shape, its runs are long
/// while the right side was rewritten, or it expands to no atom, before
/// anything is written.
fn regroup_single_by_left(
    work: &mut Rewrite<'_>,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left_remap: Option<&Remap>,
    right_remap: Option<&Remap>,
) -> Result<bool, OperationError> {
    let lim = work.eng.limits();
    let level = &tdd.levels[parent.idx()];
    // An implicit level's one node's pairs are generated into `buf`.
    let mut buf = Vec::new();
    let pairs = level.pairs_read(0, &mut buf);
    work.gate.poll(pairs.len() as u64)?;
    if !pairs.is_sorted_by_key(|pair| pair.left) {
        return Ok(false);
    }
    let runs = match right_remap {
        Some(_) => {
            let runs = pairs.chunk_by(|a, b| a.left == b.left).count();
            if pairs.len() > runs.saturating_mul(BY_LEFT_PAIRS_PER_RUN) {
                return Ok(false);
            }
            runs
        }
        None => 0,
    };
    let mut out: Transient<'_, Vec<ChildPair>> = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut out, pairs.len())?;
    // Whether a left cell came before one already written, or again after
    // another left reference's: the cells then need putting in order.
    let (mut reordered, mut merged) = (false, false);
    match (left_remap, right_remap) {
        (None, Some(right)) if right.one_each() && runs == pairs.len() => {
            // Every run one pair and every right side one cell: the level
            // maps pair by pair, already in order.
            out.extend(pairs.iter().map(|pair| {
                let cell = right.items[ChildDecoder::structural().node(pair.right).idx()];
                ChildPair::new(pair.left, EncodedChildRef::from_raw(cell))
            }));
            work.gate.poll(pairs.len() as u64)?;
        }
        (None, Some(right)) => expand_by_untouched_left(work, &mut out, pairs, right)?,
        (Some(left), right) => {
            (reordered, merged) = expand_by_rewritten_left(work, &mut out, pairs, left, right)?;
        }
        (None, None) => return Ok(false),
    }
    if out.is_empty() {
        return Ok(false);
    }
    if reordered {
        let widest = out.iter().map(|pair| pair.left.0).max().unwrap_or(0);
        sort_by_key_stable(work, &mut out, u32::BITS - widest.leading_zeros(), |pair| pair.left.0)?;
    }
    if reordered || merged {
        let out: &mut Vec<ChildPair> = &mut out;
        let (mut kept, mut from) = (0, 0);
        while from < out.len() {
            let left = out[from].left;
            let to = from + out[from..].iter().take_while(|pair| pair.left == left).count();
            work.gate.poll((to - from) as u64)?;
            let run = &mut out[from..to];
            if !run.is_sorted() {
                run.sort_unstable();
            }
            let n = dedup_sorted(run);
            out.copy_within(from..from + n, kept);
            kept += n;
            from = to;
        }
        out.truncate(kept);
    }
    debug_assert!(out.is_sorted() && out.windows(2).all(|w| w[0] != w[1]),
        "a single cell's atoms are distinct and in canonical order");
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    level.push_node(lim, &out)?;
    drop(out);
    work.emitted += 1;
    lim.level_done(work.emitted)?;
    tdd.try_invalidate(work.eng, parent)?;
    Ok(true)
}

/// [`regroup_single_by_left`]'s expansion of a level whose left side was not
/// rewritten: each run of one left reference beside the cells its right sides
/// map to, sorted and without repeats, the runs in canonical order one after
/// another.
fn expand_by_untouched_left(
    work: &mut Rewrite<'_>,
    out: &mut Vec<ChildPair>,
    pairs: &[ChildPair],
    right: &Remap,
) -> Result<(), OperationError> {
    let lim = work.eng.limits();
    // A map of one cell per node is read at its item alone.
    let one_each = right.one_each();
    for same in pairs.chunk_by(|a, b| a.left == b.left) {
        let start = out.len();
        if one_each {
            out.extend(same.iter().map(|pair| {
                let cell = right.items[ChildDecoder::structural().node(pair.right).idx()];
                ChildPair::new(pair.left, EncodedChildRef::from_raw(cell))
            }));
        } else {
            for pair in same {
                for &cell in right.get(ChildDecoder::structural().node(pair.right).idx()) {
                    lim.try_push(out, ChildPair::new(pair.left, EncodedChildRef::from_raw(cell)))?;
                }
            }
        }
        let added = out.len() - start;
        work.gate.poll(added as u64 + 1)?;
        if added > 1 {
            let run = &mut out[start..];
            if !run.is_sorted() {
                run.sort_unstable();
            }
            let kept = dedup_sorted(run);
            out.truncate(start + kept);
        }
    }
    Ok(())
}

/// [`regroup_single_by_left`]'s expansion of a level whose left side was
/// rewritten: each run of one left reference, cell by cell of that reference,
/// beside the cells its right sides map to (or the right sides themselves),
/// each segment sorted and without repeats. Returns whether a cell came
/// before one already written, and whether one came again after another
/// reference's: the segments then need putting together in order.
fn expand_by_rewritten_left(
    work: &mut Rewrite<'_>,
    out: &mut Vec<ChildPair>,
    pairs: &[ChildPair],
    left_remap: &Remap,
    right_remap: Option<&Remap>,
) -> Result<(bool, bool), OperationError> {
    let lim = work.eng.limits();
    // A map of one cell per node is read at its item alone.
    let (left_each, right_each) = (left_remap.one_each(), right_remap.is_some_and(Remap::one_each));
    let (mut reordered, mut merged) = (false, false);
    let mut last: Option<u32> = None;
    for same in pairs.chunk_by(|a, b| a.left == b.left) {
        let idx = ChildDecoder::structural().node(same[0].left).idx();
        let lefts: &[u32] = if left_each {
            left_remap.items.get(idx..idx + 1).unwrap_or(&[])
        } else {
            match left_remap.starts.get(idx + 1) {
                Some(&end) => &left_remap.items[left_remap.starts[idx] as usize..end as usize],
                None => &[],
            }
        };
        for &cell in lefts {
            let left = EncodedChildRef::from_raw(cell);
            let start = out.len();
            // A left reference of several cells writes past the first
            // reservation.
            if (right_remap.is_none() || right_each) && out.capacity() - out.len() < same.len() {
                lim.reserve(out, same.len())?;
            }
            match right_remap {
                None => out.extend(same.iter().map(|pair| ChildPair::new(left, pair.right))),
                Some(map) if right_each => out.extend(same.iter().map(|pair| {
                    let right = map.items[ChildDecoder::structural().node(pair.right).idx()];
                    ChildPair::new(left, EncodedChildRef::from_raw(right))
                })),
                Some(map) => {
                    for pair in same {
                        for &right in map.get(ChildDecoder::structural().node(pair.right).idx()) {
                            lim.try_push(out, ChildPair::new(left, EncodedChildRef::from_raw(right)))?;
                        }
                    }
                }
            }
            let added = out.len() - start;
            work.gate.poll(added as u64 + 1)?;
            if added == 0 {
                continue;
            }
            if right_remap.is_some() && added > 1 {
                let run = &mut out[start..];
                if !run.is_sorted() {
                    run.sort_unstable();
                }
                let kept = dedup_sorted(run);
                out.truncate(start + kept);
            }
            match last {
                Some(before) if cell < before => reordered = true,
                Some(before) if cell == before => merged = true,
                _ => {}
            }
            last = Some(cell);
        }
    }
    Ok((reordered, merged))
}

/// Keep the first of each run of equal items in the sorted `items`, moved to
/// the front; how many there are.
fn dedup_sorted(items: &mut [ChildPair]) -> usize {
    let mut kept = 0;
    for i in 0..items.len() {
        if kept == 0 || items[kept - 1] != items[i] {
            items[kept] = items[i];
            kept += 1;
        }
    }
    kept
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
    /// to: a row is read back through its summary ([`Row`]), whose top words
    /// are read whether the pairs set any, and a node of several rows first
    /// files its pairs and inverts the left child's map.
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
        // The row's words are cleared once, and every row reads its top
        // words; the words it set are read back at no more than one per atom.
        let reads = match single {
            Some(_) => words,
            None => {
                let inverted = left_remap.map_or(0, |map| map.items.len() as u64 + rows);
                let tops = Row::tops_for(words);
                words.saturating_add(rows.saturating_mul(tops)).saturating_add(left_keys).saturating_add(inverted)
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
/// them to the level's arena. A summary of the words the row set ([`Row`])
/// takes the read-back to those words alone, so a row costs what it holds
/// rather than the width of the right child. The rows are visited in ascending order: a
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
    let mut buf = Vec::new();
    let pairs = tdd.levels[parent.idx()].pairs_read(0, &mut buf);
    let mut row = Row::new(lim, plan.words)?;
    if let Some(only) = plan.single {
        // A pair whose left side expands to no cell has no atom. When every
        // left reference expands to one, the left sides are not read.
        let every_left_expands = left_remap.is_none_or(|map| map.starts.windows(2).all(|w| w[0] < w[1]));
        if every_left_expands {
            for pair in pairs {
                let marked = row.mark(std::slice::from_ref(&pair.right.0), right_remap);
                work.gate.poll(marked + 1)?;
            }
        } else {
            for pair in pairs {
                let expands = left_remap
                    .is_none_or(|map| !map.get(ChildDecoder::structural().node(pair.left).idx()).is_empty());
                let marked = if expands { row.mark(std::slice::from_ref(&pair.right.0), right_remap) } else { 0 };
                work.gate.poll(marked + 1)?;
            }
        }
        let level = &mut tdd.levels[parent.idx()];
        level.clear();
        let read = row.drain(lim, only, level.pairs.stored_mut(), 0, 1, plan.most)?;
        work.gate.poll(read)?;
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
                    let marked = row.mark(groups.get(node as usize), right_remap);
                    work.gate.poll(marked + 1)?;
                } else {
                    row.or_words(bits);
                    work.gate.poll(plan.words)?;
                }
            }
            let read = row.drain(lim, left, level.pairs.stored_mut(), u64::from(left), plan.rows, plan.most)?;
            work.gate.poll(read)?;
        }
    }
    lim.discard(row.words);
    lim.discard(row.marks);
    lim.discard(row.tops);
    let level = &mut tdd.levels[parent.idx()];
    let arena = level.pairs.stored_mut();
    debug_assert!(arena.is_sorted() && arena.windows(2).all(|w| w[0] != w[1]),
        "a single cell's atoms are distinct and in canonical order");
    let len = arena.len();
    if len < 2 {
        // A cell of at most one pair is pushed the way any node is, which
        // stores a lone pair in the node itself when it fits.
        let lone = arena.pop();
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

/// One row of [`regroup_single_rows`]: its columns as bits, with a bit per
/// word that a column was set in since the row was last read back, and a bit
/// per word of those, so the read-back visits the words the row set and
/// skips the rest.
struct Row {
    /// Bit `c` for column `c`.
    words: Vec<u64>,
    /// Bit `w` when word `w` may hold a column.
    marks: Vec<u64>,
    /// Bit `m` when word `m` of `marks` may hold a bit.
    tops: Vec<u64>,
    /// Whole words were OR-ed in ([`Row::or_words`]) without marking them,
    /// so the row is read back word by word.
    whole: bool,
}

impl Row {
    /// A row of `words` words, all clear. A row's words were counted from a
    /// `u32` column bound, so they fit.
    fn new(lim: &crate::limits::Limits, words: u64) -> Result<Row, OperationError> {
        let words = words as usize;
        let mut row = Row { words: Vec::new(), marks: Vec::new(), tops: Vec::new(), whole: false };
        lim.try_resize(&mut row.words, words, 0u64)?;
        lim.try_resize(&mut row.marks, words.div_ceil(64), 0u64)?;
        lim.try_resize(&mut row.tops, words.div_ceil(64 * 64), 0u64)?;
        Ok(row)
    }

    /// The top words of a row of `words` words: what every read-back reads.
    fn tops_for(words: u64) -> u64 {
        words.div_ceil(64 * 64)
    }

    /// Set column `column`.
    #[inline(always)]
    fn set(&mut self, column: u32) {
        let c = column as usize;
        self.words[c >> 6] |= 1u64 << (c & 63);
        self.marks[c >> 12] |= 1u64 << ((c >> 6) & 63);
        self.tops[c >> 18] |= 1u64 << ((c >> 12) & 63);
    }

    /// Set the columns that the stored right references `rights` stand for:
    /// the cells `remap` lists for them, or the references themselves.
    /// Returns how many columns were set, counting a repeat.
    #[inline]
    fn mark(&mut self, rights: &[u32], remap: Option<&Remap>) -> u64 {
        match remap {
            None => {
                for &right in rights {
                    self.set(right);
                }
                rights.len() as u64
            }
            Some(map) => {
                let mut marked = 0u64;
                for &right in rights {
                    let cells = right_cells(map, right);
                    marked += cells.len() as u64;
                    for &cell in cells {
                        self.set(cell);
                    }
                }
                marked
            }
        }
    }

    /// OR a row's worth of words in.
    fn or_words(&mut self, bits: &[u64]) {
        for (word, &bit) in self.words.iter_mut().zip(bits) {
            *word |= bit;
        }
        self.whole = true;
    }

    /// Append `(left, c)` to `out` for every column `c` set, in ascending
    /// order, clearing the row for the next. `out` makes room through
    /// [`grow_rows`], row `done` of `rows` being written into a cell of at
    /// most `most` pairs. Returns how many words were read.
    #[inline]
    fn drain(
        &mut self,
        lim: &crate::limits::Limits,
        left: u32,
        out: &mut Vec<ChildPair>,
        done: u64,
        rows: u64,
        most: u64,
    ) -> Result<u64, OperationError> {
        let left = EncodedChildRef::from_raw(left);
        if self.whole {
            self.whole = false;
            self.marks.fill(0);
            self.tops.fill(0);
            for at in 0..self.words.len() {
                drain_word(lim, &mut self.words[at], at, left, out, done, rows, most)?;
            }
            return Ok(self.words.len() as u64);
        }
        let mut read = self.tops.len() as u64;
        for t in 0..self.tops.len() {
            let mut top = std::mem::take(&mut self.tops[t]);
            while top != 0 {
                let m = (t << 6) | top.trailing_zeros() as usize;
                top &= top - 1;
                let mut marks = std::mem::take(&mut self.marks[m]);
                read += 1;
                while marks != 0 {
                    let at = (m << 6) | marks.trailing_zeros() as usize;
                    marks &= marks - 1;
                    read += 1;
                    drain_word(lim, &mut self.words[at], at, left, out, done, rows, most)?;
                }
            }
        }
        Ok(read)
    }
}

/// Append `(left, c)` to `out` for every column `c` set in word `at`,
/// ascending, and clear the word.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn drain_word(
    lim: &crate::limits::Limits,
    word: &mut u64,
    at: usize,
    left: EncodedChildRef,
    out: &mut Vec<ChildPair>,
    done: u64,
    rows: u64,
    most: u64,
) -> Result<(), OperationError> {
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
    Ok(())
}

/// Room for more of the pairs [`Row::drain`] appends: at least half again
/// what `out` holds, and once enough rows are done, the rows still to come at
/// the average so far, but never past `most` pairs in all.
///
/// Under a soft budget the new block must fit beside the old one: growing
/// `out` allocates a block of its capacity plus the growth and copies into
/// it before the old block is freed, and the meter charges only the growth.
/// So the growth is cut to what the budget has left besides everything in
/// flight and the new block's copy of the old one, and refused when not even
/// [`GROW_ROWS_MIN_PAIRS`] more pairs (or the rest of the cell, if fewer)
/// would fit.
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
    let mut additional = rest.max(len / 2).min(most.saturating_sub(len)).max(1);
    if let Some(room) = lim.budget_headroom() {
        let fits = (room / crate::limits::PAIR_ELEM_BYTES).saturating_sub(out.capacity() as u64);
        if fits < additional {
            if fits < GROW_ROWS_MIN_PAIRS.min(additional) {
                return Err(OperationError::OverBudget);
            }
            additional = fits;
        }
    }
    lim.reserve_exact(out, usize::try_from(additional).map_err(|_| OperationError::OverBudget)?)
}

/// The least growth [`grow_rows`] cuts a growth to before it refuses: 8 MiB
/// of pairs, so a cell near the budget does not grow by a few pairs at a
/// time, copying itself each time.
const GROW_ROWS_MIN_PAIRS: u64 = 1 << 20;

/// The level's pairs in scan order — node by node, each node's pairs as
/// stored — with the node holding each.
fn scan_level(work: &mut Rewrite<'_>, level: &TddLevel) -> Result<Vec<Owned>, OperationError> {
    let lim = work.eng.limits();
    let mut owned = Vec::new();
    lim.reserve_exact(&mut owned, level.live_pairs())?;
    // The nodes in order, an implicit level's off its description.
    for (i, pairs) in level.internal_inputs_iter() {
        work.poll()?;
        let owner = u32::try_from(i).map_err(|_| OperationError::IndexOverflow)?;
        for pair in pairs {
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

/// Radix digits at most this wide below the length that takes wider ones: a
/// histogram of 2048 counters.
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
    // Many items take wider digits, as the crate's own radix sort does: a
    // pass costs about the same per item up to that width, and fewer passes
    // move the items fewer times.
    let widest = if n >= 1 << crate::sort::RADIX_LARGE_BITS { crate::sort::RADIX_LARGE_BITS as u32 } else { RADIX_BITS };
    let passes = bits.div_ceil(widest);
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
///
/// Each cell is held to the output cap as it is written, and cancellation is
/// tested through the gate: a test per cell reads the clock under a deadline,
/// which on a level of millions of cells costs more than writing them.
fn write_level(work: &mut Rewrite<'_>, tdd: &mut Tdd, parent: VtreeIdx, new_nodes: &mut Runs<ChildPair>) -> Result<(), OperationError> {
    let lim = work.eng.limits();
    let level = &mut tdd.levels[parent.idx()];
    level.clear();
    for cell in 0..new_nodes.len() {
        work.poll()?;
        let pairs = new_nodes.get_mut(cell);
        sort_pairs(pairs);
        debug_assert!(pairs.windows(2).all(|w| w[0] != w[1]), "a regrouped cell repeats a pair");
        level.push_node(lim, pairs)?;
        work.emitted += 1;
        lim.check_output_cap(work.emitted)?;
    }
    lim.level_done(work.emitted)?;
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
