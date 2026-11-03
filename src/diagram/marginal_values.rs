//! Coordinated access to marginal payloads and their level metadata.
//!
//! Counts remain inline in the level; weighted columns remain in the side
//! store to keep structural levels compact. Passes use this boundary to move,
//! install or compact values without separately maintaining their slot count.

use super::{TddLevel, WeightStore, WeightValue};
use crate::value::{CountRead, CountRef};
use crate::limits::{Limits, OperationError};
use crate::value::slots::{key_map, move_count_slots, move_slots, plan_count_slots, plan_slots, plan_slots_apart, release_key_map, truncate_with_slack};

/// What [`MarginalStorage::compact`] does to a level's store, decided
/// before any slot moves ([`MarginalStorage::plan_compact`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Compaction {
    /// Nothing: the level has no slot to compact.
    Unchanged,
    /// The count store keeps `len` slots, `merged` of the referenced ones
    /// merged onto an earlier equal one.
    Counts { len: usize, merged: usize },
    /// A weighted level none of whose slots is referenced: its width goes
    /// to zero, with or without a column in the store.
    NoSlots,
    /// The weighted column keeps `len` slots, `merged` merged.
    Weights { len: usize, merged: usize },
}

/// A level's stored values, with their numeric representation already selected.
pub(crate) enum MarginalValues<'a> {
    Counts(CountRef<'a>),
    Weights(&'a [WeightValue]),
}

impl<'a> MarginalValues<'a> {
    pub(crate) fn read(level: &'a TddLevel, weights: Option<&'a WeightStore>, index: usize) -> Option<Self> {
        if let Some(counts) = level.count_column() {
            Some(Self::Counts(counts))
        } else if level.is_weight_marginal() {
            Some(Self::Weights(weights.expect("weighted level requires its store").level(index).unwrap_or(&[])))
        } else {
            None
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self { Self::Counts(values) => values.len(), Self::Weights(values) => values.len() }
    }

    pub(crate) fn count(&self, slot: usize) -> CountRead<'a> {
        match self {
            Self::Counts(values) => values.get(slot),
            Self::Weights(_) => panic!("integer read of weighted values"),
        }
    }
}

/// Mutable payload access. Structural references and reduction worklists are
/// updated by the enclosing diagram edit, after this storage edit completes.
pub(crate) struct MarginalStorage<'a> {
    level: &'a mut TddLevel,
    weights: Option<&'a mut WeightStore>,
    index: usize,
}

impl<'a> MarginalStorage<'a> {
    pub(crate) fn new(level: &'a mut TddLevel, weights: Option<&'a mut WeightStore>, index: usize) -> Self {
        Self { level, weights, index }
    }

    pub(crate) fn install_counts(&mut self, counts: Vec<u128>, big: Option<super::CountOverflow>) {
        self.level.become_marginal(counts, big);
    }

    pub(crate) fn install_weights(&mut self, column: Vec<WeightValue>) {
        let width = column.len() as u32;
        self.weights.as_mut().expect("weighted installation requires its store").set_level(self.index, column);
        self.level.become_marginal_weighted(width);
    }

    /// Make the level weight-marginal at `width` slots whose values the
    /// caller holds: the store keeps no column for it until the caller
    /// installs one with [`Self::install_weights`].
    pub(crate) fn hold_weights(&mut self, width: u32) {
        self.weights.as_mut().expect("weighted installation requires its store").take_level(self.index);
        self.level.become_marginal_weighted(width);
    }

    /// Decide the compaction to the referenced slots, merging equal values,
    /// without moving any: `remap[old]` is written for every referenced slot,
    /// and [`compact`](Self::compact) then moves them. With `merge` off, a
    /// bounded-precision weighted store keeps equal values in their own
    /// slots, which invariant 10 allows it; every other store merges
    /// regardless.
    /// Weighted leaf columns must never be passed here: they are pinned by label.
    ///
    /// # Errors
    ///
    /// `OverBudget` where `lim` refuses the map that finds equal values.
    pub(crate) fn plan_compact(&self, lim: &Limits, referenced: &[u32], remap: &mut [u32], merge: bool) -> Result<Compaction, OperationError> {
        let kept = referenced.iter().map(|&i| i as usize);
        if let Some(counts) = self.level.marginal_counts() {
            if counts.is_empty() { return Ok(Compaction::Unchanged); }
            let mut by_key = key_map(lim, referenced.len())?;
            let (len, merged) = plan_count_slots(counts, self.level.marginal_counts_big(), kept, &mut by_key, remap);
            release_key_map(lim, by_key);
            return Ok(Compaction::Counts { len, merged });
        }
        if !self.level.is_weight_marginal() { return Ok(Compaction::Unchanged); }
        if remap.is_empty() { return Ok(Compaction::NoSlots); }
        let weights = self.weights.as_deref().expect("weighted compaction requires its store");
        let values = weights.level(self.index).expect("a nonempty weighted store has a column");
        let (len, merged) = match weights.is_bounded() {
            true if !merge => (plan_slots_apart(kept, remap), 0),
            // Two words a key, where a `WeightKey` can hold a big rational.
            true => {
                let mut by_key = key_map(lim, referenced.len())?;
                let planned = plan_slots(
                    values, kept, &mut by_key,
                    |values, old| super::bounded_key(&values[old]).expect("a bounded store holds no exact value"),
                    remap,
                );
                release_key_map(lim, by_key);
                planned
            }
            false => {
                let mut by_key = key_map(lim, referenced.len())?;
                let planned = plan_slots(values, kept, &mut by_key, |values, old| super::semiring::weight_key(&values[old]), remap);
                release_key_map(lim, by_key);
                planned
            }
        };
        Ok(Compaction::Weights { len, merged })
    }

    /// Keep only the referenced slots as [`plan_compact`](Self::plan_compact)
    /// decided, with `referenced` and `remap` as it had and filled them, and
    /// return the number merged.
    pub(crate) fn compact(&mut self, referenced: &[u32], remap: &[u32], plan: Compaction) -> usize {
        let kept = referenced.iter().map(|&i| i as usize);
        match plan {
            Compaction::Unchanged => 0,
            Compaction::Counts { len, merged } => {
                let (counts, big) = self.level.marginal_store_mut().expect("a count compaction of a count store");
                let old_len = counts.len();
                move_count_slots(counts, big, kept, remap, len);
                truncate_with_slack(counts, len);
                self.level.retire_marginal_slots((old_len - len) as u32);
                merged
            }
            Compaction::NoSlots => {
                self.level.set_weight_width(0);
                0
            }
            Compaction::Weights { len, merged } => {
                let weights = self.weights.as_mut().expect("weighted compaction requires its store");
                let values = weights.level_vals_mut(self.index).expect("a nonempty weighted store has a column");
                move_slots(values.as_mut_slice(), kept, |values, dst, src| values.swap(dst, src), remap);
                truncate_with_slack(values, len);
                self.level.set_weight_width(len as u32);
                merged
            }
        }
    }

    /// Release a subsumed column without making the level structural again.
    pub(crate) fn clear(&mut self, leaf: bool) {
        if let Some((counts, big)) = self.level.marginal_store_mut() {
            if !counts.is_empty() { *counts = Vec::new(); }
            *big = None;
        }
        if self.level.is_weight_marginal() && self.level.slot_count() != 0 && !leaf {
            self.level.set_weight_width(0);
            if let Some(weights) = self.weights.as_mut() { weights.set_level(self.index, Vec::new()); }
        }
    }

    /// Move the source's structure and optional weighted payload to this
    /// level. The source is left with what this level held, an empty level
    /// with its arenas where the destination was taken from the pool, so
    /// that they go back to it with the source's levels.
    pub(crate) fn move_from(&mut self, source: &mut TddLevel, weights: Option<&mut WeightStore>, index: usize) {
        if source.is_weight_marginal()
            && let (Some(destination), Some(weights)) = (self.weights.as_mut(), weights)
            && let Some(values) = weights.take_level(index)
        {
            destination.set_level(self.index, values);
        }
        std::mem::swap(self.level, source);
    }

    pub(crate) fn push_weight(&mut self, eng: &crate::Engine, value: WeightValue) -> Result<u32, crate::OperationError> {
        let values = self.weights.as_mut().expect("weighted append requires its store")
            .level_vals_mut(self.index).expect("weighted level has a column");
        let slot = crate::value::slots::next_slot_index(values.len())?;
        eng.limits().try_push(values, value)?;
        self.level.set_weight_width(slot + 1);
        Ok(slot)
    }
}
