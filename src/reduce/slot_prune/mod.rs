//! Slot-prune of orphaned marginal-count slots.
//!
//! Slots become garbage at a boundary marginal level (a marginal child of a
//! structural parent) when the tagger, pair fusion or a twin merge redirects
//! the refs that named them; no other pass shrinks such a store. The
//! postcondition is `test_helpers::check::marginal::check_no_orphan_slots`.
//!
//! `prune_value_slots` compacts each boundary store to exactly the slots
//! referenced from its parent's marginal-side refs and rewrites those refs
//! through the same remap. The output level's store is exempt: it holds the
//! result. Equal-valued surviving slots are merged, which is where slot-count
//! uniqueness (invariant 10) is established for stores born at an apply emit
//! site. A large bounded-precision weighted store, which invariant 10 does
//! not ask to merge, merges only where a sample of its values repeats one.
//!
//! **Precondition:** parent levels must be in post-tagger form (marginal-side refs
//! decodable with `ChildDecoder`) — never mid-apply.
//!
//! Runs at the end of every prune and every full reduction, and after each
//! `fuse_pairs_at_parents` sweep; the contract-only plan kills no pairs and
//! skips it.
//!
//! Every buffer a level's compaction grows is reserved through the engine's
//! limits before the level is touched, so a refusal is an error at a level
//! boundary, never an allocation failure partway through a store.

use crate::Engine;
use crate::limits::OperationError;

use crate::diagram::{bounded_key, Tdd, MarginalValues, MarginalStorage, WeightValue};

use crate::value::slots::referenced_marginal_slots;
use crate::diagram::boundary_marginal_levels;

/// Compact every boundary store to the slots its parent references, merging
/// equal-valued survivors, and rewrite the parent's refs through the composed
/// remap. Returns the marginal levels where a value merge fired: such a merge
/// can make two parent nodes content-equal, so the content-twin scan restricts
/// itself to their boundary parents.
///
/// Precondition: the diagram is in post-tagger form (module doc).
///
/// # Errors
///
/// `OverBudget` where the limits refuse a level's buffers: the levels before
/// it are compacted, it and the levels after it are as they were, and the
/// diagram denotes what it did.
pub(crate) fn prune_value_slots(eng: &Engine, tdd: &mut Tdd) -> Result<Vec<u32>, OperationError> {
    // Invariant 11 (`check_leaf_columns_pinned`), checked here because this
    // pass runs after every pass that could break it.
    #[cfg(debug_assertions)]
    if let Err(e) = crate::test_helpers::check::marginal::check_leaf_columns_pinned(tdd) {
        panic!("leaf column pin: {e}");
    }
    let lim = eng.limits();
    let mut merged_levels = Vec::new();
    // Both buffers are refilled per level, so a pooled pair differs from a
    // fresh one only in capacity.
    let mut slots = eng.scratch.reduce.slot_prune_slots.checkout(eng);
    let mut remap = eng.scratch.reduce.slot_prune_remap.checkout(eng);
    // The output level's store is the result; never touch it.
    let out_v = tdd.output.vtree;
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        // Invariant 11: a weight-marginal leaf's column is the label-ordered
        // cache of `WeightStore::leaf_val`, shared by every diagram over this
        // vtree, so compacting it would corrupt every other holder.
        if tdd.vtree.node(v).is_leaf() && tdd.levels[v.idx()].is_weight_marginal() {
            continue;
        }
        // Empty stores need no parent walks. Weighted live width still needs
        // synchronization, even when no store column is installed.
        let store_len = MarginalValues::read(&tdd.levels[v.idx()], tdd.weights.as_ref(), v.idx()).map_or(0, |values| values.len());
        if store_len == 0 {
            tdd.reindex_level(v, &[], &mut [], |level, weights, referenced, remap| {
                MarginalStorage::new(level, weights, v.idx()).compact(lim, referenced, remap, true)
            })?;
            continue;
        }

        slots.reserve(lim, store_len)?;
        remap.clear();
        lim.reserve(&mut *remap, store_len)?;
        let referenced =
            referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut slots);
        if referenced.last().is_some_and(|&s| (s as usize) >= store_len) {
            continue; // OOB ref: broken upstream (the marginal-canonicality checker's domain)
        }

        // On a large bounded-precision level the merge map is most of the
        // prune's cost, and floats from different sums seldom agree to the
        // bit: the level merges where a sample of its referenced values
        // repeats one, and otherwise keeps its values apart, left as it is
        // when every slot is referenced.
        let merge = match tdd.weights.as_ref().filter(|w| w.is_bounded() && tdd.levels[v.idx()].is_weight_marginal()) {
            Some(weights) if referenced.len() >= SAMPLE_FROM => {
                sample_repeats(weights.level(v.idx()).expect("a nonempty weighted store has a column"), referenced)
            }
            _ => true,
        };
        if !merge && referenced.len() == store_len {
            continue;
        }

        // The composed remap, old slot to final slot: unreferenced slots are
        // dropped, and equal-valued referenced slots map to one output slot
        // (first occurrence wins).
        remap.resize(store_len, u32::MAX);
        let values_merged = tdd.reindex_level(v, referenced, &mut remap, |level, weights, referenced, remap| {
            MarginalStorage::new(level, weights, v.idx()).compact(lim, referenced, remap, merge)
        })?;
        if values_merged > 0 {
            merged_levels.push(v.0);
        }
    }

    // It rewrites stores in place and so cannot stop partway, for the reason
    // `prune_unreachable` gives; charging keeps the walk on the work clock.
    lim.charge_work(tdd.vtree.num_nodes() as u64);

    Ok(merged_levels)
}

/// The referenced values a bounded level's merge test reads.
const SAMPLE: usize = 1024;

/// The fewest referenced slots on which a bounded level samples before it
/// merges; a smaller level merges outright.
const SAMPLE_FROM: usize = 4 * SAMPLE;

/// Whether [`SAMPLE`] of the `referenced` slots, at an even stride, hold two
/// bit-identical values of the bounded column `values`.
fn sample_repeats(values: &[WeightValue], referenced: &[u32]) -> bool {
    let step = referenced.len() / SAMPLE;
    let mut keys = [(0u64, 0i64); SAMPLE];
    for (k, key) in keys.iter_mut().enumerate() {
        *key = bounded_key(&values[referenced[k * step] as usize]).expect("a bounded store holds no exact value");
    }
    keys.sort_unstable();
    keys.windows(2).any(|w| w[0] == w[1])
}

#[cfg(test)]
mod tests;
