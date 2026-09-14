//! Content-twin merge (a twin-contraction mechanism).
//!
//! Two nodes at the same level can be raw-identical (same pair multisets) yet
//! sit in different parent contexts, so the context-based
//! `strategies::contract_all_twins` — which groups by the multiset of
//! `(parent_node, sibling)` contexts — cannot see them.
//! `merge_content_equal_nodes` detects them by pair-multiset content and
//! rewrites parent/output refs onto the canonical node.
//!
//! The prune, merge, contract fixpoint that drives the merge is
//! `reduce::canonicalize_content_twins`.

use crate::diagram::Changed;
use crate::engine::Engine;
use crate::limits::pool::PooledScratch;

use rustc_hash::FxHashMap;

use crate::diagram::remap_refs_into;
use crate::diagram::Tdd;
use crate::limits::{OperationError, Limits};
use crate::vtree::VtreeIdx;

/// Per-pass working buffers of [`merge_content_equal_nodes`], bundled so one
/// take/put covers the whole set.
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

impl PooledScratch for ContentTwinScratch {
    fn prepare(&mut self) {
        self.node_fp.clear();
        self.fp_counts.clear();
        self.key_to_canonical.clear();
        self.remap.clear();
    }

    fn retain(&mut self) {
        crate::limits::pool::release_if_oversized(&mut self.node_fp);
        crate::limits::pool::release_if_oversized(&mut self.remap);
        // The maps have no `Vec` shape for `release_if_oversized`; bound them by the
        // same element-count estimate the contract scratch uses.
        if self.fp_counts.capacity().saturating_mul(std::mem::size_of::<(u64, u32)>()) > crate::limits::pool::SCRATCH_RETAIN_BYTES {
            self.fp_counts = FxHashMap::default();
        }
        if self
            .key_to_canonical
            .capacity()
            .saturating_mul(std::mem::size_of::<(Vec<(u32, u32)>, u32)>())
            > crate::limits::pool::SCRATCH_RETAIN_BYTES
        {
            self.key_to_canonical = FxHashMap::default();
        }
    }
}

/// The levels this merge canonicalizes, children before parents; the checker
/// `test_helpers::check::marginal::check_twin_canonicality` walks the same set.
///
/// Empty on a diagram with no marginal level (the `# Soundness` block on
/// `merge_content_equal_nodes`). Otherwise every internal vtree node whose own
/// level and whose parent's level are explicit: a marginal level has counts,
/// not pair structure, and a level under a marginal ancestor is unreferenced.
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
/// canonical twin. A return of 0 means invariant 9 holds everywhere the filter
/// reached.
///
/// Two nodes at one level with equal pair multisets are found by sorted key,
/// and the parent's references and the output reference are rewritten onto
/// the first index. The duplicates are left in place as unreferenced nodes,
/// not tombstoned (a streaming apply asserts tombstone-free levels); the
/// caller must follow with a prune. The parent level is marked dirty for the
/// next contract pass. Levels are scanned children before parents, so a merge
/// at `L` that makes two nodes of `parent(L)` content-equal is caught later in
/// the same pass.
///
/// # Soundness
///
/// The pass stands down on a diagram with no marginal level. Content equality
/// is function equality there, which invariant 1 forbids between two nodes of
/// one level, and the duplicate pair a redirect leaves at a parent is legal
/// only in a marginalized diagram: a pair list is a multiset feeding a sum,
/// so `c(x)·c(B₁) + c(x)·c(B₂)` with `B₁`, `B₂` content-identical is
/// `2·c(x)·c(B₁)`, two assignment families sharing a value. A content twin in
/// Boolean mode is an upstream determinism violation, not work for this pass.
///
/// `filter`: with `Some(set)`, a level `P` is scanned only when `P` or its
/// marginal child is in `set` (the slot prune reports value merges under the
/// marginal level's index while the twins they mint appear at the parent).
/// With `None`, every explicit level is scanned. A filtered-out level can only
/// be left with unmerged twins, which costs size, not correctness.
pub(crate) fn merge_content_equal_nodes(
    eng: &Engine,
    tdd: &mut Tdd,
    filter: Option<&rustc_hash::FxHashSet<u32>>,
) -> Result<usize, OperationError> {
    use rustc_hash::FxHashSet;

    // Marginalized diagrams only (`# Soundness` above).
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
    let mut scratch = eng.reduce().content_twin.checkout();
    let ContentTwinScratch { node_fp, fp_counts, key_to_canonical, remap } = &mut *scratch;

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

        let width = tdd.levels[parent_idx].slot_count();
        if width <= 1 {
            continue;
        }

        if !fingerprint_level_nodes(lim, &tdd.levels[parent_idx], width, node_fp, fp_counts)? {
            // No two nodes share a fingerprint ⇒ no content-equal pair can exist.
            continue;
        }
        if !group_content_equal(
            lim, &tdd.levels[parent_idx], width, node_fp, fp_counts,
            key_to_canonical, remap,
        )? {
            continue;
        }
        dups_merged += remap.iter().enumerate().filter(|&(n, &r)| r != n as u32).count();
        redirect_parent_refs(tdd, parent_v, remap, &mut live);
    }

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
) -> Result<bool, OperationError> {
    // Equal pair multisets give equal fingerprints (necessary, not
    // sufficient), so a node with a unique fingerprint has no content-equal
    // twin and skips the exact sorted-key pass. A pair is mixed into a u64
    // through the shared finalizer `fingerprint::mix64`; the node fingerprint
    // is the wrapping sum over its pairs, xor the mixed pair count, so it is
    // order-independent. The golden-ratio increment keeps this distribution
    // distinct from `context_hash`'s.
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
) -> Result<bool, OperationError> {
    // Group non-leaf (non-tombstone) nodes at this level by their sorted pair
    // multiset. Two nodes with the same sorted key compute the same function
    // (and, in a marginalized diagram, carry the same count) and must be merged.
    key_to_canonical.clear();
    // remap[n] = canonical node index for node n (identity if n is canonical),
    // u32-wide because the entries are written straight into `NodeIdx` refs.
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
            key.try_reserve(pairs_slice.len()).map_err(|_| OperationError::OverBudget)?;
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
    // The output can reference a node at any level, so it is remapped too:
    // pointing it at a duplicate would send the next apply into a node the
    // prune removes. The remap can collapse two of a grandparent node's refs
    // onto one child, minting a duplicate pair; the dirty push below hands a
    // marginal-flagged grandparent to pair fusion, and at a plain one the two
    // entries stay as multiset terms.
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

