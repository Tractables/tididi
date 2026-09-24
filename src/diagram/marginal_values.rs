//! Coordinated access to marginal payloads and their level metadata.
//!
//! Counts remain inline in the level; weighted columns remain in the side
//! store to keep structural levels compact. Passes use this boundary to move,
//! install or compact values without separately maintaining their slot count.

use super::{TddLevel, WeightStore, WeightValue};
use crate::value::CountRead;
use crate::value::slots::{compact_count_slots, compact_slots, truncate_with_slack};

/// A level's stored values, with their numeric representation already selected.
pub(crate) enum MarginalValues<'a> {
    Counts(&'a [u128], Option<&'a super::CountOverflow>),
    Weights(&'a [WeightValue]),
}

impl<'a> MarginalValues<'a> {
    pub(crate) fn read(level: &'a TddLevel, weights: Option<&'a WeightStore>, index: usize) -> Option<Self> {
        if let Some(counts) = level.marginal_counts() {
            Some(Self::Counts(counts, level.marginal_counts_big()))
        } else if level.is_weight_marginal() {
            Some(Self::Weights(weights.expect("weighted level requires its store").level(index).unwrap_or(&[])))
        } else {
            None
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self { Self::Counts(values, _) => values.len(), Self::Weights(values) => values.len() }
    }

    pub(crate) fn count(&self, slot: usize) -> CountRead<'a> {
        match self {
            Self::Counts(values, big) => CountRead::from_slot(values, *big, slot),
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

    /// Keep only referenced slots, merging equal values, and return the number merged.
    /// Weighted leaf columns must never be passed here: they are pinned by label.
    pub(crate) fn compact(&mut self, referenced: &[u32], remap: &mut [u32]) -> usize {
        if let Some((counts, big)) = self.level.marginal_store_mut() {
            let old_len = counts.len();
            if old_len == 0 { return 0; }
            let (new_len, merged) = compact_count_slots(counts, big, referenced.iter().map(|&i| i as usize), remap);
            truncate_with_slack(counts, new_len);
            self.level.retire_marginal_slots((old_len - new_len) as u32);
            return merged;
        }
        if !self.level.is_weight_marginal() { return 0; }
        if remap.is_empty() {
            self.level.set_weight_width(0);
            return 0;
        }
        let values = self.weights.as_mut().expect("weighted compaction requires its store")
            .level_vals_mut(self.index).expect("a nonempty weighted store has a column");
        let (new_len, merged) = compact_slots(
            values, referenced.iter().map(|&i| i as usize),
            |values, old| super::semiring::weight_key(&values[old]),
            |values, dst, src| values.swap(dst, src), remap,
        );
        truncate_with_slack(values, new_len);
        self.level.set_weight_width(new_len as u32);
        merged
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

    /// Move the source's structure and optional weighted payload to this level.
    pub(crate) fn move_from(&mut self, source: &mut TddLevel, weights: Option<&mut WeightStore>, index: usize) {
        if source.is_weight_marginal()
            && let (Some(destination), Some(weights)) = (self.weights.as_mut(), weights)
            && let Some(values) = weights.take_level(index)
        {
            destination.set_level(self.index, values);
        }
        *self.level = std::mem::take(source);
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
