//! Prune phase: remove diagram nodes not reachable from the output.
//!
//! Marks reachability top-down from the output node, then compacts the levels
//! it walked bottom-up while remapping child references. Levels that lost a
//! node go onto the contract worklists; `reduce` runs the contraction.
//!
//! [`PruneScope`] says how many levels that is: every one of them, or only the
//! ones a change at the root can have reached.

use crate::diagram::{EncodedChildRef, NodeIdx, NodeKind, Tdd};

use crate::Engine;

use crate::vtree::VtreeIdx;
use crate::limits::{OperationError, Transient};

/// `remap` entry for a slot the pass-1 walk never reached — the whole
/// reachability bitmap, folded into the remap array (they are indexed
/// identically and pass 2 never needs the mark after it has written the
/// compacted index into the same slot). `u32::MAX` can never collide with a
/// real remap value: both a compacted index and an identity index are `< width`,
/// and a level of `u32::MAX` nodes cannot be allocated.
const UNREACHED: u32 = u32::MAX;

/// Placeholder pass 1 stamps on a reached slot; pass 2 overwrites it with the
/// slot's compacted index. Only its inequality with `UNREACHED` is meaningful.
const REACHED: u32 = 0;

/// How much of the diagram a prune has to walk.
pub(crate) enum PruneScope {
    /// Every level, assuming nothing about how the nodes became unreachable.
    Whole,
    /// Only what a change at the root can have made unreachable.
    ///
    /// Valid when every node of every non-root level is referenced by a node of
    /// its parent level — reachable or not — and the output is the root. A
    /// level that then loses no node has a subtree that loses no node: each of
    /// its children's slots is named by one of its own surviving nodes, and so
    /// on down. The walk stops at such a level and never looks below it.
    ///
    /// `expand_full` establishes the condition for the diagram a negation
    /// complements: it leaves every structural level covering the whole
    /// `lefts x rights` basis of its children, so every child slot is named by
    /// some pair of the level above.
    BelowRoot,
}

/// Remove nodes not reachable from the output.
///
/// Marks reachability top-down, then compacts each level bottom-up, remapping
/// child references in the same pass. Every level that lost a node is pushed
/// onto the contract worklists (`seed_dirty_levels`), so the caller needs no
/// reseed.
///
/// `scope` says how much of the diagram has to be walked; see [`PruneScope`].
/// A `BelowRoot` scope on a diagram that is not the shape that walk starts
/// from — an output below the root, a leaf or marginal root level — walks the
/// whole diagram instead.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if the reservation of `remap`, the
/// one buffer proportional to the walked level widths, is refused. It is taken
/// before any mutation of `tdd`, so the diagram is then untouched.
pub(crate) fn prune_unreachable(
    eng: &Engine,
    tdd: &mut Tdd,
    scope: PruneScope,
) -> Result<(), OperationError> {
    // `ZERO` sentinel: the entire diagram computes ⊥ (UNSAT). No nodes are reachable.
    if tdd.is_zero() {
        for level in &mut tdd.levels {
            level.nodes.clear();
            level.pairs.clear();
            level.ranges.clear();
            level.dead_pairs = 0;
        }
        return Ok(());
    }

    if matches!(scope, PruneScope::BelowRoot) && below_root_walk_applies(tdd) {
        prune_below_root(eng, tdd)
    } else {
        prune_whole(eng, tdd)
    }
}

/// Whether the seeded walk can start on `tdd` at all: it starts at the output,
/// which has to be the root, and the root level has to be one it can compact.
/// An output below the root leaves every level above it unreachable, which is
/// not a change below the root.
pub(crate) fn below_root_walk_applies(tdd: &Tdd) -> bool {
    let root = tdd.vtree.root();
    tdd.output.vtree == root
        && !tdd.vtree.node(root).is_leaf()
        && !tdd.levels[root.idx()].is_marginal()
}

/// [`prune_unreachable`] over every level: mark from the output, then compact
/// each level bottom-up.
fn prune_whole(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let num_nodes = tdd.vtree.num_nodes();

    let pool = &eng.scratch.reduce;
    let mut level_base = pool.prune_level_base.checkout_preserving(eng);
    let mut remap = pool.prune_remap.checkout_preserving(eng);

    // Flat offset table: level t occupies remap[level_base[t]..level_base[t+1]].
    // Use `reference_slot_count()` so leaf levels get `LEAF_WIDTH` slots for marginal nodes.
    eng.limits().try_resize(&mut level_base, num_nodes + 1, 0usize)?;
    level_base[0] = 0;
    for i in 0..num_nodes {
        level_base[i + 1] = level_base[i] + tdd.reference_slot_count(VtreeIdx(i as u32));
    }
    let total = level_base[num_nodes];

    // The exact form leaves the Vec untouched on failure, so the pooled buffer
    // can be returned; through the engine's limits so the reservation is
    // charged against the byte budget.
    let need_remap = total.saturating_sub(remap.len());
    eng.limits().reserve_exact(&mut remap, need_remap)?;

    // Reset the marks. Split so the grown tail is initialized once, by `resize`,
    // rather than written by `resize` and again by the fill.
    let warm = remap.len().min(total);
    remap[..warm].fill(UNREACHED);
    if remap.len() < total {
        remap.resize(total, UNREACHED);
    }
    // ── Pass 1 (top-down): mark reachable nodes ──────────────────────────
    classic_mark(tdd, &level_base, &mut remap[..total]);

    // `level_dirty[t]` records whether level `t` lost a node; charged for
    // this prune and handed back with it.
    let mut level_dirty: Transient<'_, Vec<bool>> = Transient::new(eng.limits(), Vec::new());
    eng.limits().try_resize(&mut level_dirty, num_nodes, false)?;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    compact_levels(tdd, &vtree, &level_base, &mut remap[..total], &mut level_dirty);
    seed_dirty_levels(tdd, &level_dirty);

    tdd.output.local = NodeIdx(
        remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()],
    );

    // Both passes cross every reference slot, and neither can stop partway:
    // `compact_levels` rewrites levels in place, so a diagram abandoned mid-pass
    // has some levels compacted and others still naming their old indices.
    // Charge the walk so a work budget sees it; the cancellation test belongs to
    // the callers, between prunes.
    eng.limits().charge_work(2 * total as u64);

    Ok(())
}

/// Pass 2 (bottom-up): compact unreachable nodes, overwriting the pass-1 marks
/// in `remap` with each surviving node's compacted index and rewriting child
/// references as it goes. Sets `level_dirty[t]` where level `t` lost a node.
fn compact_levels(
    tdd: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    level_base: &[usize],
    remap: &mut [u32],
    level_dirty: &mut [bool],
) {
    // Overwrite the pass-1 marks with the remap (old index → new index) in
    // place, rewrite child references, and drop unreachable nodes in one
    // pass. A slot's mark is only read before its own remap value is written,
    // and `UNREACHED` survives on every slot that is never remapped, so
    // `remap[s] != UNREACHED` stays the reachability predicate throughout.
    //
    // A level's child-ref rewrite is needed only when a child level shrank
    // (otherwise that child's remap is the identity), and its retain only when
    // the level itself shrank, so all-reachable levels skip both walks.

    // Bottom-up topological order, so a level's child levels are remapped
    // before it rewrites its child references; raw indices do not encode
    // parent/child order on a rotated vtree.
    for v in vtree.bottomup_slice() {
        let t = *v;
        let t_idx = t.idx();
        let base = level_base[t_idx];
        let eff_width = tdd.reference_slot_count(t);

        if eff_width == 0 {
            continue;
        }

        // Leaf and marginal levels keep their indices. A leaf level's nodes
        // are implicit, so there is nothing to compact.
        //
        // A marginal level keeps its content in a value store that parents
        // reference by store-relative slot index, and such refs may be minted
        // after this prune (contract's inline-to-slot redirect). The walk marks
        // only the slots referenced right now, so compacting the store here
        // could drop a slot a later ref reads. Marginal stores keep their full
        // length; `prune_value_slots` collects their orphans.
        if !tdd.is_structural_internal(t) {
            for i in 0..eff_width {
                remap[base + i] = i as u32;
            }
            continue;
        }

        let (left, right) = vtree.children(t);
        level_dirty[t_idx] = compact_one_level(
            tdd,
            t,
            base,
            remap,
            &[],
            Child::at(level_base[left.idx()], level_dirty[left.idx()]),
            Child::at(level_base[right.idx()], level_dirty[right.idx()]),
        );
    }
}

/// Where a child level's remap is, as the parent rewriting its references
/// needs to read it.
#[derive(Clone, Copy)]
struct Child {
    /// Offset of the child's block in `remap`; unused when `identity`.
    base: usize,
    /// The child kept every slot, so its remap is the identity and it has no
    /// block of its own — the shared identity run stands in for it.
    identity: bool,
    /// The child lost a node, so the parent has to rewrite its references.
    dirty: bool,
}

impl Child {
    /// A child with its own block in `remap`.
    const fn at(base: usize, dirty: bool) -> Child {
        Child { base, identity: false, dirty }
    }

    /// A child that kept every slot.
    const IDENTITY: Child = Child { base: 0, identity: true, dirty: false };
}

/// Compact one structural level: overwrite each reached slot's mark with the
/// slot's compacted index, rewrite the level's child references through the
/// children's remaps, and drop the unreachable nodes. Returns whether the
/// level lost a node.
///
/// `identity` is the ascending run a child marked [`Child::IDENTITY`] reads its
/// remap from; it must be at least as long as that child's width.
fn compact_one_level(
    tdd: &mut Tdd,
    t: VtreeIdx,
    base: usize,
    remap: &mut [u32],
    identity: &[u32],
    left: Child,
    right: Child,
) -> bool {
    let t_idx = t.idx();
    let width = tdd.levels[t_idx].slot_count();
    let mut new_idx = 0u32;
    // The arena slots the dropped nodes own, summed here where each slot's
    // mark is read anyway and noted once after the retain.
    let mut dead = 0usize;
    for i in 0..width {
        if remap[base + i] != UNREACHED {
            remap[base + i] = new_idx;
            new_idx += 1;
        } else {
            dead += tdd.levels[t_idx].arena_pairs_at(i);
        }
    }
    // Did this level lose any node? (remap stays valid identity either way.)
    let this_dirty = (new_idx as usize) != width;

    if left.dirty || right.dirty {
        let marks: &[u32] = remap;
        let left_remap = if left.identity { identity } else { &marks[left.base..] };
        let right_remap = if right.identity { identity } else { &marks[right.base..] };
        rewrite_child_refs(tdd, t, width, &marks[base..], left_remap, right_remap);
    }

    // Compact the node Vec in place, O(width) with no allocation. Dropping a
    // node abandons its pair range; the arena is swept once enough of it is
    // dead, which is legal here because no pair offset is held across the
    // call.
    if this_dirty {
        let mut i = 0;
        tdd.levels[t_idx].nodes.retain(|_| {
            let keep = remap[base + i] != UNREACHED;
            i += 1;
            keep
        });
        tdd.levels[t_idx].note_dead_pairs(dead);
        tdd.levels[t_idx].compact_pairs_if_stale();
    }
    this_dirty
}

/// Rewrite level `t`'s child references through its child levels' remaps.
/// `own` is `t`'s own block of marks; `left_remap` and `right_remap` are the
/// children's remaps, each starting at its own level's first slot.
///
/// Called only when a child level actually shrank: otherwise both child remaps
/// are the identity and every write would store a value back onto itself.
///
/// The three tables are parameters rather than one struct: a slice loaded out
/// of a struct loses the aliasing facts a slice parameter carries, and the
/// rewrite loop then re-reads the table pointers on every node.
fn rewrite_child_refs(
    tdd: &mut Tdd,
    t: VtreeIdx,
    width: usize,
    own: &[u32],
    left_remap: &[u32],
    right_remap: &[u32],
) {
    let t_idx = t.idx();
    let (left, right) = tdd.vtree.children(t);
    let left_view = tdd.levels[left.idx()].child_decoder();
    let right_view = tdd.levels[right.idx()].child_decoder();
    for (i, &mark) in own.iter().take(width).enumerate() {
        if mark == UNREACHED {
            continue;
        }
        match tdd.levels[t_idx].nodes[i].kind() {
            NodeKind::Inline(_) => {
                let node = &mut tdd.levels[t_idx].nodes[i];
                node.a = left_view.remap(EncodedChildRef::from_raw(node.a), left_remap).0;
                node.b = right_view.remap(EncodedChildRef::from_raw(node.b), right_remap).0;
            }
            k if k.pairs_in_arena() => tdd.levels[t_idx]
                .pairs_remap_indexed(i, left_remap, right_remap, left_view, right_view),
            _ => {}
        }
    }
}

/// Push every level prune shrank onto the contract worklists.
///
/// Removing a node simplifies the parent context of that level's children,
/// which can make two of them twins; contraction processes the parent, so the
/// shrunk level itself is what is pushed. Its own nodes cannot become twins
/// (a removal never equates two survivors), and its parents see only a
/// bijective child remap, so neither needs seeding. `level_dirty` is set on
/// non-leaf levels only, so every index pushed is a parent level.
fn seed_dirty_levels(tdd: &mut Tdd, level_dirty: &[bool]) {
    for (t_idx, dirty) in level_dirty.iter().enumerate() {
        if *dirty {
            tdd.invalidate(VtreeIdx(t_idx as u32));
        }
    }
}

/// Pass 1: mark every slot reachable from the output node, walking the vtree
/// root first (raw indices do not follow topological order on a rotated
/// vtree). `remap` must be `UNREACHED`-filled and `level_base`-indexed; each
/// slot reached is stamped `REACHED`, which pass 2 replaces with the slot's
/// compacted index.
fn classic_mark(tdd: &Tdd, level_base: &[usize], remap: &mut [u32]) {
    let vtree = &tdd.vtree;
    remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()] = REACHED;
    let topo = vtree.bottomup_slice();
    for v in topo.iter().rev() {
        let t = *v;
        // A marginal level has no pairs, and by invariant 5 every level
        // beneath it is marginal too, so there is nothing to mark below.
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let (left, right) = vtree.children(t);
        mark_children_of_level(
            tdd,
            t,
            level_base[t.idx()],
            level_base[left.idx()],
            level_base[right.idx()],
            remap,
        );
    }
}

/// Mark every child slot the reached nodes of level `t` name. `base`,
/// `left_base` and `right_base` are the three levels' blocks in `remap`.
///
/// A side of a marginal child may be an inline count rather than a slot; such a
/// side names no child slot, so the decoder answers `None` for it and only real
/// slots are marked. A structural side is its own slot.
fn mark_children_of_level(
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    left_base: usize,
    right_base: usize,
    remap: &mut [u32],
) {
    let (left, right) = tdd.vtree.children(t);
    let left_view = tdd.levels[left.idx()].child_decoder();
    let right_view = tdd.levels[right.idx()].child_decoder();
    let level = &tdd.levels[t.idx()];
    for i in 0..level.slot_count() {
        if remap[base + i] == UNREACHED {
            continue;
        }
        for pair in level.pairs_of_idx(i) {
            if let Some(s) = left_view.child(pair.left).index() {
                remap[left_base + s] = REACHED;
            }
            if let Some(s) = right_view.child(pair.right).index() {
                remap[right_base + s] = REACHED;
            }
        }
    }
}

// ── The seeded walk ──────────────────────────────────────────────────────────

/// One level the seeded walk marked: where its marks live, and what it reads
/// its children's remaps from.
#[derive(Clone, Copy)]
pub(crate) struct Visit {
    level: VtreeIdx,
    /// This level's block in `remap`.
    base: usize,
    left: Child,
    right: Child,
}

impl Visit {
    /// A level whose block is marked but whose children are not settled yet.
    const fn new(level: VtreeIdx, base: usize) -> Visit {
        Visit { level, base, left: Child::IDENTITY, right: Child::IDENTITY }
    }
}

/// [`prune_unreachable`] under [`PruneScope::BelowRoot`]: walk down from the
/// output, stopping at every level that loses no node.
fn prune_below_root(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let pool = &eng.scratch.reduce;
    let mut remap = pool.prune_remap.checkout_preserving(eng);
    let mut identity = pool.prune_identity.checkout_preserving(eng);
    let mut visits = pool.prune_visits.checkout_preserving(eng);

    prune_below_root_with(eng, tdd, &mut remap, &mut identity, &mut visits)
}

/// [`prune_below_root`] with the scratch checked out.
fn prune_below_root_with(
    eng: &Engine,
    tdd: &mut Tdd,
    remap: &mut Vec<u32>,
    identity: &mut Vec<u32>,
    visits: &mut Vec<Visit>,
) -> Result<(), OperationError> {
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let root = tdd.output.vtree;
    visits.clear();
    let mut used = 0usize;

    // The root's block, with the output node the only slot reached: whatever
    // else the root level holds is what the change at the root dropped.
    let base = alloc_block(eng, remap, &mut used, tdd.levels[root.idx()].slot_count())?;
    remap[base + tdd.output.local.idx()] = REACHED;
    eng.limits().try_push(visits, Visit::new(root, base))?;

    // Where the marks of a child the walk does not descend into go: a leaf
    // level, whose remap is the identity, or a marginal one, which is never
    // compacted. Sized to the widest such child, shared by every level, and
    // never read. `(base, width)`.
    let mut sink = (0usize, 0usize);

    // Marking, top-down. A level has one parent, so a level's marks are
    // complete as soon as that parent has been walked, and a queue is a valid
    // order.
    let mut i = 0;
    while i < visits.len() {
        let t = visits[i].level;
        let (left, right) = vtree.children(t);
        let (lw, rw) = (tdd.reference_slot_count(left), tdd.reference_slot_count(right));
        let (l_own, r_own) = (tdd.is_structural_internal(left), tdd.is_structural_internal(right));
        // The identity run stands in for either child the level keeps whole.
        identity_upto(eng, identity, lw.max(rw))?;
        let lb = if l_own {
            alloc_block(eng, remap, &mut used, lw)?
        } else {
            sink_block(eng, remap, &mut used, &mut sink, lw)?
        };
        let rb = if r_own {
            alloc_block(eng, remap, &mut used, rw)?
        } else {
            sink_block(eng, remap, &mut used, &mut sink, rw)?
        };

        mark_children_of_level(tdd, t, visits[i].base, lb, rb, remap);

        visits[i].left = settle_child(remap, lb, lw, l_own);
        visits[i].right = settle_child(remap, rb, rw, r_own);
        if visits[i].left.dirty {
            eng.limits().try_push(visits, Visit::new(left, lb))?;
        }
        if visits[i].right.dirty {
            eng.limits().try_push(visits, Visit::new(right, rb))?;
        }
        i += 1;
    }

    // Compaction, bottom-up: a level was pushed after its parent, so the walk
    // order reversed puts every level after its own children.
    for k in (0..visits.len()).rev() {
        let v = visits[k];
        if compact_one_level(tdd, v.level, v.base, remap, identity, v.left, v.right) {
            tdd.invalidate(v.level);
        }
    }

    tdd.output.local = NodeIdx(remap[visits[0].base + tdd.output.local.idx()]);

    // As in `prune_whole`: both walks cross every slot they reserved, and
    // neither can stop partway.
    eng.limits().charge_work(2 * used as u64);
    Ok(())
}

/// What a walked level's parent reads for it: a child that kept every slot
/// needs no block, and one that lost a slot is walked in turn.
fn settle_child(remap: &[u32], base: usize, width: usize, walked: bool) -> Child {
    if !walked {
        return Child::IDENTITY;
    }
    if remap[base..base + width].iter().all(|&m| m != UNREACHED) {
        Child::IDENTITY
    } else {
        Child::at(base, true)
    }
}

/// Reserve `width` fresh slots at the end of `remap`, `UNREACHED` throughout.
///
/// The reservation goes through the engine's limits. Nothing in the diagram has
/// been written while the walk is taking blocks, so a refusal leaves it as it
/// was.
fn alloc_block(
    eng: &Engine,
    remap: &mut Vec<u32>,
    used: &mut usize,
    width: usize,
) -> Result<usize, OperationError> {
    let base = *used;
    let end = base + width;
    if remap.len() < end {
        eng.limits().try_resize(remap, end, UNREACHED)?;
    }
    remap[base..end].fill(UNREACHED);
    *used = end;
    Ok(base)
}

/// The shared block the marks of a child the walk ignores go into. It is never
/// read, so it needs no reset — only room for the widest such child.
fn sink_block(
    eng: &Engine,
    remap: &mut Vec<u32>,
    used: &mut usize,
    sink: &mut (usize, usize),
    width: usize,
) -> Result<usize, OperationError> {
    if sink.1 < width {
        let base = *used;
        let end = base + width;
        if remap.len() < end {
            eng.limits().try_resize(remap, end, UNREACHED)?;
        }
        *used = end;
        *sink = (base, width);
    }
    Ok(sink.0)
}

/// Grow the shared identity run to `n` entries, `identity[i] == i`.
///
/// Every child level the walk keeps whole remaps through it, so one ascending
/// run serves all of them, and it only ever grows.
fn identity_upto(eng: &Engine, identity: &mut Vec<u32>, n: usize) -> Result<(), OperationError> {
    if identity.len() < n {
        let start = identity.len();
        eng.limits().try_resize(identity, n, 0u32)?;
        for (i, slot) in identity.iter_mut().enumerate().skip(start) {
            *slot = i as u32;
        }
    }
    Ok(())
}
