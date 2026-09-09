//! Content-twin merge (a twin-contraction mechanism).
//!
//! Two nodes at the same level can be raw-identical (same pair multisets) yet
//! sit in different parent contexts, so the context-based
//! `strategies::contract_all_twins_topdown` — which groups by the multiset of
//! `(parent_node, sibling)` contexts — cannot see them.
//! `merge_content_equal_nodes` detects them by pair-multiset content and
//! rewrites parent/output refs onto the canonical node.
//!
//! This is only the merge MECHANISM; the prune→merge→contract fixpoint that drives
//! it lives in `minimize::canonicalize_content_twins` (orchestration). The two
//! communicate through the `Tdd` dirty-contract worklists and `c2_rescan`, the
//! same by-design shared state the prune and contract phases use.

use crate::engine::Engine;

use rustc_hash::FxHashMap;

use crate::diagram::{ChildSide, remap_side_refs};
use crate::diagram::Tdd;
use crate::error::ApplyError;
use crate::vtree::VtreeIdx;

/// Per-pass working buffers of [`merge_content_equal_nodes`], bundled so one
/// take/put covers the whole set.
///
/// They are already hoisted out of the per-level loop; pooling them lifts the
/// same four allocations out of the PASS as well — the
/// `canonicalize_content_twins` fixpoint runs one pass per round, and the
/// per-merge minimize runs that fixpoint over and over across a compile.
#[derive(Default)]
pub(crate) struct C2Scratch {
    /// Per-node content fingerprint at the level being scanned.
    pub(super) node_fp: Vec<u64>,
    /// Fingerprint → number of nodes carrying it (the collision pre-filter).
    /// Probed by key only, never iterated.
    pub(super) fp_counts: FxHashMap<u64, u32>,
    /// Sorted pair-multiset key → canonical node index. Probed by key only.
    /// Its KEYS own `Vec`s, but the take-side `clear()` drops every one of them
    /// — only the table itself is carried across passes, so this pool retains
    /// one allocation, not a fan-out.
    pub(super) key_to_canonical: FxHashMap<Vec<(u32, u32)>, u32>,
    /// Node index → canonical node index at the level being scanned.
    pub(super) remap: Vec<u32>,
}

impl C2Scratch {
    /// Empty every buffer, retaining capacity (and dropping the map keys' own
    /// allocations).
    fn clear(&mut self) {
        self.node_fp.clear();
        self.fp_counts.clear();
        self.key_to_canonical.clear();
        self.remap.clear();
    }
}

/// Take the engine's content-twin scratch, cleared and ready to use. Returns a fresh one
/// when the pool is empty (first use, after a capacity-capped
/// return, or when an outer pass already holds it).
pub(super) fn take_scratch(eng: &Engine) -> C2Scratch {
    let mut s = eng.reduce().content_twin.take().unwrap_or_default();
    s.clear();
    s
}

/// Return the scratch for the next pass, each buffer released
/// independently if its retained capacity exceeds the byte cap (same policy as
/// `contract::scratch::return_scratch`). Not returning it — the `?` bails on the
/// budget-gated reserves — is safe: the pool simply stays empty.
pub(super) fn return_scratch(eng: &Engine, mut s: C2Scratch) {
    let cap = crate::diagram::MAX_LEVEL_ARENA_BYTES;
    crate::engine::pool::release_if_oversized(&mut s.node_fp, cap);
    crate::engine::pool::release_if_oversized(&mut s.remap, cap);
    // The maps have no `Vec` shape for `release_if_oversized`; bound them by the
    // same element-count estimate the contract scratch uses.
    if s.fp_counts.capacity().saturating_mul(std::mem::size_of::<(u64, u32)>()) > cap {
        s.fp_counts = FxHashMap::default();
    }
    if s
        .key_to_canonical
        .capacity()
        .saturating_mul(std::mem::size_of::<(Vec<(u32, u32)>, u32)>())
        > cap
    {
        s.key_to_canonical = FxHashMap::default();
    }
    eng.reduce().content_twin.put(Some(s));
}

/// The levels this merge canonicalizes, in `internal_topo_slice` (children-before-parents)
/// order — the single source of truth for "where content twins are merged", shared
/// by the merge itself and by the twin-canonicality checker in `check::marg`.
///
/// Empty on a diagram with no marginal level — see "Scope" on
/// `merge_content_equal_nodes`: there content equality IS function equality, which
/// Invariant 2 forbids between two nodes of one level, so the merge has nothing to find
/// and its redirect would in any case mint an illegal duplicate pair.
/// Otherwise: every internal vtree node whose own level is explicit and whose
/// parent's level is explicit. A marginal level has counts, not pair structure; a
/// level under a marginal ancestor is dead (the ancestor replaced its whole
/// subtree with counts, so nothing references it and there is no parent pair list
/// to rewrite).
pub(crate) fn c2_scan_levels(tdd: &Tdd) -> Vec<VtreeIdx> {
    if !tdd.has_marginal_level() {
        return Vec::new();
    }
    tdd.vtree
        .internal_topo_slice()
        .iter()
        .copied()
        .filter(|&v| {
            !tdd.levels[v.idx()].is_marginal()
                && tdd
                    .vtree
                    .node(v)
                    .parent()
                    .is_none_or(|p| !tdd.levels[p.idx()].is_marginal())
        })
        .collect()
}

/// Content-based twin merge over every explicit level of a marginalized diagram.
///
/// Two or more nodes at one level can become raw-identical (same pair
/// multisets): `prune_marg_slots` value-merges equal-valued slots, the tagger
/// and p-fusion emit small counts INLINE without touching a slot, and this
/// function's own ref rewrites (below) collapse two of a parent's refs onto one
/// child. Context-based `contract_all_twins_topdown` cannot detect the result
/// when the twins have different parent-context signatures (different parent
/// nodes, or the same parent node with different siblings). This function
/// detects them by pair-multiset content and rewrites the PARENT's refs (and the
/// TDD output ref, which can sit at any level after mc-projection) from each dup
/// index to the first (canonical) index.
///
/// The dup nodes are left in place as valid-but-unreferenced internal nodes —
/// NOT tombstoned: streaming applies assert tombstone-free levels, and
/// node-prune's index-stable branch can preserve an interior tombstone all the
/// way to a later apply. The caller MUST follow up with `instrumented_prune`,
/// whose reachability GC removes the unreferenced dups through the established
/// machinery. The parent level is also marked dirty for the subsequent contract
/// pass so any context-equal twins the ref rewrite minted are handled.
///
/// Returns `merged`: the number of dup nodes redirected onto their canonical
/// twin. 0 means the TDD already satisfies twin canonicality everywhere the filter reached and
/// the caller can skip the follow-up prune+contract round.
///
/// ## Level set: every explicit level, children before parents
///
/// The scan covers every explicit level, not only the parents of marginal
/// levels. Restricting it to those leaks in both directions: plain levels get no
/// content-addressed merge at all, and the boundary merge itself MINTS plain
/// content twins whenever its ref rewrites make two parents identical
/// (measured at ~23k raw-identical plain nodes out of ~24.8k reachable on
/// `mc2020_track1_052`).
///
/// The scan now covers every explicit internal level — marginal levels have no
/// pair structure to compare, and a level under a marginal ancestor is dead
/// (nothing references it) — in `internal_topo_slice` order, which is
/// children-before-parents. Bottom-up matters: a merge at level `L` rewrites
/// `parent(L)`'s refs and can make two of ITS nodes content-equal, and
/// `parent(L)` is visited later in the SAME pass, so one pass chases the cascade
/// all the way to the root instead of needing one fixpoint round per level of
/// vtree depth. The cascade is also fed to `c2_rescan` for the next round, since
/// a merge can equally enable one through prune or contract.
///
/// ## Scope: marginalized diagrams only
///
/// The whole function stands down on a diagram with no marginal level, and that
/// is expected to be a pure no-op rather than a missed opportunity. In a purely
/// Boolean diagram content equality IS function equality, which Invariant 2
/// (determinism, `f_i ∧ f_j ≡ 0` for distinct nodes of one level) plus Invariant 5
/// (no ⊥ nodes) forbid — the same argument that lets apply skip a dedup pass
/// outright (the no-compress proof). Every Boolean-mode
/// rewrite either emits nodes with pairwise-disjoint pair sets (apply cells,
/// rotation's inner regroup, ∃-forget's owner classes) or is a bijective/deleting
/// ref remap (prune, context-based twin contraction), so none can mint a content
/// twin; `check::check_canonicity` (equal semiring signature at a level)
/// is the standing detector there and strictly subsumes a twin-canonicality check.
///
/// The redirect would also be WRONG here: the duplicate pair it can leave at a
/// parent is a legal count-carrying multiset entry only once some level is
/// marginal. So a content twin observed in Boolean mode is an
/// upstream determinism violation to fix at its source, not work for this pass.
/// The old level set made the stand-down implicit (`boundary_marginal_levels` is
/// empty without a marginal level); with the wider set the guard is explicit
/// (`Tdd::has_marginal_level`), and byte-identical Boolean behaviour is preserved.
///
/// ## Duplicate pairs at the parent are legal
///
/// Redirecting `dup → canonical` can leave a parent node holding the same
/// `(left, right)` pair twice — when it referenced both twins with the same
/// sibling. This used to be CANCELLED ("deferred") at plain, non-marg-flagged
/// parents on the grounds that duplicate pairs are unrepresentable there.
/// They are not: a pair list is a MULTISET feeding a sum. Nothing dedups it, and
/// every count consumer folds `Σ_pairs c(left)·c(right)` over the stored pairs
/// (`marginalize::compute_marginal_node_int` / `..._weight`, `query::count`), so
/// pre-merge `c(x)·c(B₁) + c(x)·c(B₂)` with `B₁`, `B₂` content-identical (hence
/// count-identical, by induction on the level order) equals post-merge
/// `2·c(x)·c(B₁)`. Per-level mutex is in any case not an invariant of a
/// marginalized diagram: after slot value-merges two raw-identical nodes denote
/// two DISTINCT assignment families that happen to share a count, which is
/// exactly why both terms must survive.
///
/// The old cancellation therefore protected code assumptions, not semantics, and
/// left "twins that are neither redirect-safe nor context-symmetric deferred
/// forever" — measured at ~99% of one dominant level on `mc2020_track1_052`
/// during the contraction-leak campaign. The redirect now always proceeds; the
/// parent is still pushed onto `dirty_contract` / `c2_rescan` below so p-fusion
/// folds the minted duplicates into one summed count wherever the parent level
/// IS marg-flagged, and they simply stay as multiset terms where it is not.
///
/// ## Termination
///
/// Each productive merge strictly decreases the number of REFERENCED nodes: the
/// dup loses its last reference (every ref to it — parent pairs and the output —
/// is rewritten onto the canonical node) and no node is ever created. That
/// argument is level-set-independent, so it carries over unchanged to the wider
/// set. Within one pass each level is visited exactly once (the cascade only
/// ever adds levels that come LATER in the topological order), and the outer
/// prune→merge→contract→prune loop in `canonicalize_content_twins` breaks on
/// `merged == 0`, which the finite node count forces.
///
/// `filter`: when `Some(set)`, only scan level `P` if
/// `P ∈ set || marg_child(P) ∈ set` — restricts the scan to levels that could
/// have gained new content-twins since the last round. A marginal child is
/// checked because slot-prune reports its value-merges under the MARGINAL
/// level's index while the twins they mint appear at the parent; an explicit
/// child needs no such check, since whatever changed it also pushed `P` itself.
/// When `None`, every explicit level is scanned (first round / worklist-off).
///
/// SOUNDNESS: a filtered-out level can at most be left with redundant unmerged
/// twins (size suboptimality, not correctness).  Every merge that does execute
/// is still certified by the exact sorted-pair-key check.
pub(crate) fn merge_content_equal_nodes(
    eng: &Engine,
    tdd: &mut Tdd,
    filter: Option<&rustc_hash::FxHashSet<u32>>,
) -> Result<usize, ApplyError> {
    use rustc_hash::FxHashSet;

    // Marginalized diagrams only — see "Scope" above. Also the cheap early-out
    // that keeps a Boolean compile from paying for the level walk at all.
    if !tdd.has_marginal_level() {
        return Ok(0);
    }

    let mut dups_merged = 0usize;

    // Children-before-parents order, collected upfront to avoid borrow issues
    // during the mut walk. `internal_topo_slice` is the bottom-up topological
    // order, so a level's parent is always visited strictly later in this pass —
    // which is what lets a single pass chase the merge cascade upward.
    let order = c2_scan_levels(tdd);

    // In-pass copy of the worklist filter. A merge at level L rewrites
    // parent(L)'s refs, so parent(L) must be scanned even if last round's
    // worklist did not name it; it is later in `order`, so inserting it here
    // takes effect within this same pass.
    let mut live: Option<FxHashSet<u32>> = filter.cloned();

    // Per-level scratch, hoisted: the scan now visits every explicit level, so
    // allocating these collections per level would dominate the pass on a deep
    // vtree. `clear()` keeps the capacity. Checked out of the thread-local pool
    // (cleared on take) and destructured into the same locals, so the body below
    // is unchanged and the capacity also carries ACROSS passes; parked back at
    // the productive exit.
    let C2Scratch { mut node_fp, mut fp_counts, mut key_to_canonical, mut remap } =
        take_scratch(eng);

    for parent_v in order {
        let parent_idx = parent_v.idx();

        // Worklist filter: skip this level if neither it nor a marginal child
        // was touched in the previous round (or earlier in this pass).
        if let Some(set) = live.as_ref() {
            let (cl, cr) = tdd.vtree.children(parent_v);
            let touched = set.contains(&parent_v.0)
                || (tdd.levels[cl.idx()].is_marginal() && set.contains(&cl.0))
                || (tdd.levels[cr.idx()].is_marginal() && set.contains(&cr.0));
            if !touched {
                continue;
            }
        }

        let width = tdd.levels[parent_idx].width();
        if width <= 1 {
            continue;
        }

        if !fingerprint_level_nodes(&tdd.levels[parent_idx], width, &mut node_fp, &mut fp_counts)? {
            // No two nodes share a fingerprint ⇒ no content-equal pair can exist.
            continue;
        }
        if !group_content_equal(
            &tdd.levels[parent_idx], width, &node_fp, &fp_counts,
            &mut key_to_canonical, &mut remap,
        )? {
            continue;
        }
        dups_merged += remap.iter().enumerate().filter(|&(n, &r)| r != n as u32).count();
        redirect_parent_refs(tdd, parent_v, &remap, &mut live);
    }

    return_scratch(eng, C2Scratch { node_fp, fp_counts, key_to_canonical, remap });
    Ok(dups_merged)
}

/// Fingerprint every non-leaf node at a level with an order-independent u64 over
/// its pair multiset. Returns whether two nodes share a fingerprint — `false`
/// means no content-equal pair can exist, so the exact key pass can be skipped.
fn fingerprint_level_nodes(
    level: &crate::diagram::TddLevel,
    width: usize,
    node_fp: &mut Vec<u64>,
    fp_counts: &mut rustc_hash::FxHashMap<u64, u32>,
) -> Result<bool, ApplyError> {
    // --- Fingerprint pre-filter -------------------------------------------
    // Compute a cheap order-independent u64 fingerprint per node (no alloc,
    // no sort). Equal pair multisets → equal fingerprints (necessary but not
    // sufficient). Nodes with a unique fingerprint cannot have a content-equal
    // twin; skip them in the exact sorted-key pass below.
    //
    // pair_fingerprint mixes one (left,right) pair into a u64 via the shared
    // splitmix64 finalizer (`fingerprint::mix64`). Node fingerprint =
    // wrapping_add over all its pairs' pair_fingerprints XOR'd with the pair
    // count (commutative across pairs, so order-independent).
    //
    // The golden-ratio increment is this rule's own prelude — it is what
    // makes the content-twin pair distribution distinct from `context_hash`'s; keep it
    // here, out of the shared finalizer.
    #[inline(always)]
    fn pair_fingerprint(l: u32, r: u32) -> u64 {
        let x = ((l as u64) << 32) | (r as u64);
        super::fingerprint::mix64(x.wrapping_add(0x9E3779B97F4A7C15))
    }

    node_fp.clear();
    node_fp.try_reserve(width).map_err(|_| ApplyError::OverBudget)?;
    node_fp.resize(width, 0u64);
    fp_counts.clear();
    let mut any_fp_collision = false;
    {
        // Indexes `level.nodes`, the level's pair arena and `node_fp` at the same position.
        #[allow(clippy::needless_range_loop)]
        for n in 0..width {
            if level.nodes[n].is_leaf() {
                continue;
            }
            let pairs_slice = level.pairs_of_idx(n);
            let len = pairs_slice.len() as u64;
            let fp_sum: u64 = pairs_slice
                .iter()
                .fold(0u64, |acc, p| acc.wrapping_add(pair_fingerprint(p.left.0, p.right.0)));
            let fp = fp_sum ^ len.wrapping_mul(0x9E3779B97F4A7C15);
            node_fp[n] = fp;
            let c = fp_counts.entry(fp).or_insert(0);
            *c += 1;
            any_fp_collision |= *c > 1;
        }
    }
    // The caller skips the exact sorted-key pass (and its allocations) outright
    // on the overwhelmingly common clean level.
    Ok(any_fp_collision)
}

/// Group the fingerprint-colliding nodes by their sorted pair multiset, filling
/// `remap[n]` with each node's canonical index. Returns whether any duplicate
/// was found.
fn group_content_equal(
    level: &crate::diagram::TddLevel,
    width: usize,
    node_fp: &[u64],
    fp_counts: &rustc_hash::FxHashMap<u64, u32>,
    key_to_canonical: &mut rustc_hash::FxHashMap<Vec<(u32, u32)>, u32>,
    remap: &mut Vec<u32>,
) -> Result<bool, ApplyError> {
    // Group non-leaf (non-tombstone) nodes at this level by their sorted pair
    // multiset. Two nodes with the same sorted key compute the same function
    // (and, in a marginalized diagram, carry the same count) and must be merged.
    key_to_canonical.clear();
    // remap[n] = canonical node index for node n (identity if n is canonical).
    //
    // u32-wide because it IS a table of node indices, and it is consumed as
    // one: the parent ref rewrite at the bottom writes its entries straight
    // into `NodeIdx(u32)` ref fields. (`width` fits u32 for the same
    // reason — an index that doesn't fit cannot be stored in a ref.)
    remap.clear();
    remap.try_reserve(width).map_err(|_| ApplyError::OverBudget)?;
    debug_assert!(
        width <= u32::MAX as usize,
        "level width {width} exceeds the u32 node-index range",
    );
    remap.extend(0..width as u32);
    let mut any_dup = false;

    {
        for n in 0..width {
            if level.nodes[n].is_leaf() {
                // is_leaf() is true for both real leaves AND tombstones; skip both.
                continue;
            }
            // Fast-path: unique fingerprint → no twin possible, skip alloc+sort.
            if fp_counts.get(&node_fp[n]).copied().unwrap_or(0) <= 1 {
                continue;
            }
            // Build a sorted pair-multiset key for content comparison. The
            // key is owned by the map on a first occurrence, so it cannot be
            // a reused buffer — only fingerprint-colliding nodes reach here,
            // so the allocation is paid on candidates, not on every node.
            let pairs_slice = level.pairs_of_idx(n);
            let mut key: Vec<(u32, u32)> = Vec::new();
            key.try_reserve(pairs_slice.len()).map_err(|_| ApplyError::OverBudget)?;
            key.extend(pairs_slice.iter().map(|p| (p.left.0, p.right.0)));
            key.sort_unstable();

            use std::collections::hash_map::Entry;
            match key_to_canonical.entry(key) {
                Entry::Vacant(e) => {
                    e.insert(n as u32); // n is the first (canonical) occurrence
                }
                Entry::Occupied(e) => {
                    remap[n] = *e.get(); // n is a dup; map to the canonical
                    any_dup = true;
                }
            }
        }
    }
    Ok(any_dup)
}

/// Point the output ref and the grandparent's refs at each dup's canonical node,
/// then mark the grandparent for the follow-up contract and content scans.
fn redirect_parent_refs(
    tdd: &mut Tdd,
    parent_v: VtreeIdx,
    remap: &[u32],
    live: &mut Option<rustc_hash::FxHashSet<u32>>,
) {
    // NOTE: the dup nodes are NOT tombstoned here. The per-clause
    // streaming applies assert tombstone-free levels (`expected internal
    // node` panic, see apply_clause.rs), and node-prune's index-stable
    // branch preserves interior tombstones — so a tombstone minted here
    // can survive to a later apply. Instead the dups are left in place as
    // valid (now unreferenced) internal nodes after the ref rewrite below;
    // the caller MUST follow up with `instrumented_prune`, whose
    // reachability GC removes unreferenced nodes through the established
    // machinery.

    // The TDD output can reference a node at ANY level (e.g. after
    // mc-projection it need not sit at the vtree root). If it points at a
    // tombstoned dup here, the next apply walks straight into the
    // tombstone ("expected internal node" panic — m139_count regression).
    // The bounds check skips leaf-label outputs, which don't index nodes.
    if tdd.output.vtree == parent_v && (tdd.output.local.0 as usize) < remap.len() {
        tdd.output.local = crate::diagram::NodeIdx(remap[tdd.output.local.idx()]);
    }

    // Rewrite the parent's refs into parent_v's node array from dup
    // indices to canonical indices.
    let Some(grandparent) = tdd.vtree.node(parent_v).parent() else {
        // parent_v is the vtree root — no parent refs to rewrite;
        // the output remap above already covered the only external ref.
        return;
    };

    let (gp_left, _gp_right) = tdd.vtree.children(grandparent);
    let parent_is_left = gp_left == parent_v;

    let side = if parent_is_left { ChildSide::Left } else { ChildSide::Right };
    // Pair-fusion dirty tracking: this remap can collapse two of a parent
    // node's refs onto the same child, minting a duplicate `(Q,c),(Q,c)` pair.
    // At a marg-flagged parent the dirty push below hands it to p-fusion, which
    // folds the two into one summed count; at a plain parent the two entries
    // simply stay as multiset terms (see the ruling in this function's doc
    // comment).
    let view = tdd.levels[parent_v.idx()].side_view();
    remap_side_refs(&mut tdd.levels[grandparent.idx()], side, view, remap);

    // Mark the parent dirty so the subsequent context-based contract
    // pass re-scans it for any context-equal twins the ref rewrite created.
    // Also invalidate any cached leaf-contract verdict: the ref rewrite may
    // have changed which leaf labels appear in the parent's pairs.
    tdd.dirty.contract.push(grandparent.0);
    tdd.dirty.leaf_contract.push(grandparent.idx() as u32);
    tdd.dirty.c2_rescan.push(grandparent.0);
    // In-pass cascade: the rewrite may have made two of the parent's nodes
    // content-equal. The parent is later in `order`, so admitting it to the
    // live worklist now makes THIS pass catch the new twins.
    if let Some(set) = live.as_mut() {
        set.insert(grandparent.0);
    }
}

