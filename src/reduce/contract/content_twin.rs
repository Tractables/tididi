//! Content-twin merge (a twin-contraction mechanism).
//!
//! Two nodes at the same level can be raw-identical (same pair multisets) yet
//! sit in different parent contexts, so the context-based
//! `strategies::contract_all_twins` — which groups by the multiset of
//! `(parent_node, sibling)` contexts — cannot see them.
//! `merge_content_equal_nodes` detects them by pair-multiset content and
//! rewrites parent/output refs onto the canonical node.
//!
//! This is only the merge mechanism; the prune→merge→contract fixpoint that drives
//! it lives in `reduce::canonicalize_content_twins` (orchestration). The two
//! communicate through the `Tdd` dirty-contract worklists and `right_rescan`, the
//! same by-design shared state the prune and contract phases use.

use crate::diagram::Changed;
use crate::engine::Engine;

use rustc_hash::FxHashMap;

use crate::diagram::remap_refs_into;
use crate::diagram::Tdd;
use crate::limits::{ApplyError, Limits};
use crate::vtree::VtreeIdx;

/// Per-pass working buffers of [`merge_content_equal_nodes`], bundled so one
/// take/put covers the whole set.
///
/// They are already hoisted out of the per-level loop; pooling them lifts the
/// same four allocations out of the pass as well — the
/// `canonicalize_content_twins` fixpoint runs one pass per round, and the
/// per-merge minimize runs that fixpoint over and over across a compile.
#[derive(Default)]
pub(crate) struct ContentTwinScratch {
    /// Per-node content fingerprint at the level being scanned.
    pub(super) node_fp: Vec<u64>,
    /// Fingerprint → number of nodes carrying it (the collision pre-filter).
    /// Probed by key only, never iterated.
    pub(super) fp_counts: FxHashMap<u64, u32>,
    /// Sorted pair-multiset key → canonical node index. Probed by key only.
    /// Its keys own `Vec`s, but the take-side `clear()` drops every one of them
    /// — only the table itself is carried across passes, so this pool retains
    /// one allocation, not a fan-out.
    pub(super) key_to_canonical: FxHashMap<Vec<(u32, u32)>, u32>,
    /// Node index → canonical node index at the level being scanned.
    pub(super) remap: Vec<u32>,
}

impl ContentTwinScratch {
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
pub(super) fn take_scratch(eng: &Engine) -> ContentTwinScratch {
    let mut s = eng.reduce().content_twin.take().unwrap_or_default();
    s.clear();
    s
}

/// Return the scratch for the next pass, each buffer released
/// independently if its retained capacity exceeds the scratch-retention cap
/// (same policy as `contract::scratch::return_scratch`). Not returning it — the `?` bails on the
/// budget-gated reserves — is safe: the pool simply stays empty.
pub(super) fn return_scratch(eng: &Engine, mut s: ContentTwinScratch) {
    crate::limits::pool::release_if_oversized(&mut s.node_fp);
    crate::limits::pool::release_if_oversized(&mut s.remap);
    // The maps have no `Vec` shape for `release_if_oversized`; bound them by the
    // same element-count estimate the contract scratch uses.
    if s.fp_counts.capacity().saturating_mul(std::mem::size_of::<(u64, u32)>()) > crate::limits::pool::SCRATCH_RETAIN_BYTES {
        s.fp_counts = FxHashMap::default();
    }
    if s
        .key_to_canonical
        .capacity()
        .saturating_mul(std::mem::size_of::<(Vec<(u32, u32)>, u32)>())
        > crate::limits::pool::SCRATCH_RETAIN_BYTES
    {
        s.key_to_canonical = FxHashMap::default();
    }
    eng.reduce().content_twin.put(Some(s));
}

/// The levels this merge canonicalizes, in `internal_bottomup_slice` (children-before-parents)
/// order — the single source of truth for "where content twins are merged", shared
/// by the merge itself and by the twin-canonicality checker in `test_helpers::check::marginal`.
///
/// Empty on a diagram with no marginal level — see "Scope" on
/// `merge_content_equal_nodes`: there content equality and function equality coincide,
/// which Invariant 1 forbids between two nodes of one level, so the merge has nothing to find
/// and its redirect would in any case mint an illegal duplicate pair.
/// Otherwise: every internal vtree node whose own level is explicit and whose
/// parent's level is explicit. A marginal level has counts, not pair structure; a
/// level under a marginal ancestor is dead (the ancestor replaced its whole
/// subtree with counts, so nothing references it and there is no parent pair list
/// to rewrite).
pub(crate) fn content_twin_scan_levels(tdd: &Tdd) -> Vec<VtreeIdx> {
    if !tdd.has_marginal_level() {
        return Vec::new();
    }
    tdd.vtree
        .internal_bottomup_slice()
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

/// Content-based twin merge over every explicit level of a marginalized
/// diagram, returning how many duplicate nodes were redirected onto their
/// canonical twin.
///
/// Two nodes at one level can become raw-identical — the slot prune merges
/// equal-valued slots, the tagger and pair fusion emit small counts inline, and
/// this function's own rewrites collapse two of a parent's references onto one
/// child. Context-based contraction cannot see that when the twins sit in
/// different parent contexts. This scan finds them by pair multiset and
/// rewrites the parent's references, and the output reference, onto the first
/// index.
///
/// The duplicates are left in place as valid-but-unreferenced nodes, not
/// tombstoned: a streaming apply asserts tombstone-free levels. The caller must
/// follow with a prune, whose reachability collection removes them. The parent
/// level is marked dirty so the next contract pass handles any context-equal
/// twin the rewrite minted. A return of 0 means invariant 9 already holds
/// everywhere the filter reached.
///
/// Levels are scanned children-before-parents. A merge at level `L` rewrites
/// `parent(L)`'s references and can make two of its nodes content-equal, and
/// `parent(L)` comes later in the same pass, so one pass chases the cascade to
/// the root.
///
/// # Soundness
///
/// The pass stands down on a diagram with no marginal level, where it would be
/// both useless and wrong. Content equality is function equality there, which
/// invariants 1 and 2 forbid between two nodes of one level; and the duplicate
/// pair a redirect can leave at a parent is legal only once some level is
/// marginal. A content twin seen in Boolean mode is an upstream determinism
/// violation, not work for this pass.
///
/// A duplicate pair at the parent is legal because a pair list is a multiset
/// feeding a sum: every consumer folds `Σ c(left)·c(right)` over the stored
/// pairs, so `c(x)·c(B₁) + c(x)·c(B₂)` with `B₁` and `B₂` content-identical
/// equals `2·c(x)·c(B₁)`. Two raw-identical nodes of a marginalized diagram
/// denote two distinct assignment families that share a value, which is why
/// both terms must survive.
///
/// Each productive merge strictly decreases the number of referenced nodes and
/// creates none, each level is visited once per pass, and the caller's loop
/// stops at zero merges, so the pass terminates.
///
/// `filter`: with `Some(set)`, a level `P` is scanned only when `P` or its
/// marginal child is in `set` — the levels that could have gained a twin since
/// the last round. The marginal child is checked because the slot prune reports
/// its value merges under the marginal level's index while the twins they mint
/// appear at the parent. With `None`, every explicit level is scanned. A
/// filtered-out level can only be left with unmerged twins, which costs size,
/// not correctness; every merge that runs is certified by the exact sorted-pair
/// key.
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

    let lim = eng.limits();
    let mut dups_merged = 0usize;

    // Children-before-parents order, collected upfront to avoid borrow issues
    // during the mut walk. `internal_bottomup_slice` is the bottom-up topological
    // order, so a level's parent is always visited strictly later in this pass —
    // which is what lets a single pass chase the merge cascade upward.
    let order = content_twin_scan_levels(tdd);

    // In-pass copy of the worklist filter. A merge at level L rewrites
    // parent(L)'s refs, so parent(L) must be scanned even if last round's
    // worklist did not name it; it is later in `order`, so inserting it here
    // takes effect within this same pass.
    let mut live: Option<FxHashSet<u32>> = filter.cloned();

    // Per-level scratch, hoisted out of the walk: the pass visits every
    // explicit level, so allocating these collections per level would dominate
    // it on a deep vtree. The pool keeps the capacity across passes too.
    let ContentTwinScratch { mut node_fp, mut fp_counts, mut key_to_canonical, mut remap } =
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

        if !fingerprint_level_nodes(lim, &tdd.levels[parent_idx], width, &mut node_fp, &mut fp_counts)? {
            // No two nodes share a fingerprint ⇒ no content-equal pair can exist.
            continue;
        }
        if !group_content_equal(
            lim, &tdd.levels[parent_idx], width, &node_fp, &fp_counts,
            &mut key_to_canonical, &mut remap,
        )? {
            continue;
        }
        dups_merged += remap.iter().enumerate().filter(|&(n, &r)| r != n as u32).count();
        redirect_parent_refs(tdd, parent_v, &remap, &mut live);
    }

    return_scratch(eng, ContentTwinScratch { node_fp, fp_counts, key_to_canonical, remap });
    Ok(dups_merged)
}

/// Fingerprint every non-leaf node at a level with an order-independent u64 over
/// its pair multiset. Returns whether two nodes share a fingerprint — `false`
/// means no content-equal pair can exist, so the exact key pass can be skipped.
fn fingerprint_level_nodes(
    lim: &Limits,
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
    // wrapping_add over all its pairs' pair_fingerprints, combined by exclusive-or
    // with the mixed pair count (commutative across pairs, so order-independent).
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
    lim.reserve(node_fp, width)?;
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
    lim: &Limits,
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
    // u32-wide because it is nothing but a table of node indices, and it is consumed as
    // one: the parent ref rewrite at the bottom writes its entries straight
    // into `NodeIdx(u32)` ref fields. (`width` fits u32 for the same
    // reason — an index that doesn't fit cannot be stored in a ref.)
    remap.clear();
    lim.reserve(remap, width)?;
    debug_assert!(
        width <= u32::MAX as usize,
        "level width {width} exceeds the u32 node-index range",
    );
    remap.extend(0..width as u32);
    let mut any_dup = false;

    {
        for n in 0..width {
            if level.nodes[n].is_leaf() {
                // is_leaf() is true for real leaves as well as tombstones; skip both.
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
                    remap[n] = *e.get(); // n is a duplicate; map to the canonical
                    any_dup = true;
                }
            }
        }
    }
    Ok(any_dup)
}

/// Point the output ref and the grandparent's refs at each duplicate's canonical node,
/// then mark the grandparent for the follow-up contract and content scans.
fn redirect_parent_refs(
    tdd: &mut Tdd,
    parent_v: VtreeIdx,
    remap: &[u32],
    live: &mut Option<rustc_hash::FxHashSet<u32>>,
) {
    // Note: the duplicate nodes are not tombstoned here. The per-clause
    // streaming applies assert tombstone-free levels (`expected internal
    // node` panic, see `apply::conjoin_clause`), and node-prune's index-stable
    // branch preserves interior tombstones — so a tombstone minted here
    // can survive to a later apply. Instead the dups are left in place as
    // valid (now unreferenced) internal nodes after the ref rewrite below;
    // the caller must follow up with `prune_unreachable`, whose
    // reachability GC removes unreferenced nodes through the established
    // machinery.

    // The diagram output can reference a node at any level (after a
    // projection it need not sit at the vtree root); pointing it at a
    // duplicate would send the next apply into a node the prune removes.
    //
    // Pair-fusion dirty tracking: the remap can collapse two of a grandparent
    // node's refs onto the same child, minting a duplicate `(Q,c),(Q,c)` pair.
    // At a marginal-flagged parent the dirty push below hands it to pair fusion, which
    // folds the two into one summed count; at a plain parent the two entries
    // simply stay as multiset terms (see the ruling in this function's doc
    // comment).
    remap_refs_into(tdd, parent_v, remap);

    let Some(grandparent) = tdd.vtree.node(parent_v).parent() else {
        // parent_v is the vtree root: the output was the only external ref.
        return;
    };

    // The ref rewrite may have created context-equal twins at the grandparent,
    // and may have changed which leaf labels appear in its pairs.
    tdd.invalidate(grandparent, Changed::VALUES);
    // In-pass cascade: the rewrite may have made two of the parent's nodes
    // content-equal. The parent is later in `order`, so admitting it to the
    // live worklist means the current pass catches the new twins.
    if let Some(set) = live.as_mut() {
        set.insert(grandparent.0);
    }
}

