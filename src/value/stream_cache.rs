//! The per-apply cache of already-computed child columns for streaming levels.

use crate::diagram::WeightValue;
use crate::limits::pool::{Buffers, PooledScratch, Scratch};
use super::CountVec;

/// Lazily computed child columns for streaming-target levels whose children
/// are still explicit, indexed by vtree node.
///
/// The value kind is fixed for the whole apply by whether a
/// [`WeightStore`](crate::diagram::WeightStore) is in play, so this is one enum rather than two parallel
/// caches: an apply is integer, weighted, or does no streaming at all.
#[derive(Default)]
pub(crate) enum StreamCache {
    /// Nothing is being marginalized, so no streaming route is reachable.
    #[default]
    None,
    Int(Vec<Option<CountVec>>),
    Weighted(Vec<Option<Vec<WeightValue>>>),
}

impl StreamCache {
    /// Select the column kind and size; nonstreaming operations leave it parked.
    pub(crate) fn reset(&mut self, num_nodes: usize, weighted: Option<bool>) {
        let Some(weighted) = weighted else { return };
        match (self, weighted) {
            (StreamCache::Weighted(cols), true) => cols.resize_with(num_nodes, || None),
            (StreamCache::Int(cols), false) => cols.resize_with(num_nodes, || None),
            (cache, true) => *cache = StreamCache::Weighted(vec_of_none(num_nodes)),
            (cache, false) => *cache = StreamCache::Int(vec_of_none(num_nodes)),
        }
    }

    /// The integer columns. Only ever asked for on the integer route.
    pub(crate) fn int(&self) -> &[Option<CountVec>] {
        match self {
            StreamCache::Int(cols) => cols,
            _ => unreachable!("integer streaming without an integer cache"),
        }
    }

    /// The weighted columns. Only ever asked for on the weighted route.
    pub(crate) fn weighted(&self) -> &[Option<Vec<WeightValue>>] {
        match self {
            StreamCache::Weighted(cols) => cols,
            _ => unreachable!("weighted streaming without a weighted cache"),
        }
    }

    /// The integer columns, for the walk that fills them in.
    pub(crate) fn int_mut(&mut self) -> &mut [Option<CountVec>] {
        match self {
            StreamCache::Int(cols) => cols,
            _ => unreachable!("integer streaming without an integer cache"),
        }
    }

    /// The weighted columns, for the walk that fills them in.
    pub(crate) fn weighted_mut(&mut self) -> &mut [Option<Vec<WeightValue>>] {
        match self {
            StreamCache::Weighted(cols) => cols,
            _ => unreachable!("weighted streaming without a weighted cache"),
        }
    }
}

/// Allocate an empty slot for each vtree node.
fn vec_of_none<T>(num_nodes: usize) -> Vec<Option<T>> {
    let mut cols = Vec::new();
    cols.resize_with(num_nodes, || None);
    cols
}

impl StreamCache {
    /// Drop the computed columns, keeping the table they sat in.
    pub(crate) fn discard_columns(&mut self) {
        match self {
            Self::None => {},
            Self::Int(cols) => cols.clear(),
            Self::Weighted(cols) => cols.clear(),
        }
    }
}

/// The column table alone: its columns are dropped before it is parked.
impl Buffers for StreamCache {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        match self {
            Self::None => {},
            Self::Int(cols) => visit(cols),
            Self::Weighted(cols) => visit(cols),
        }
    }
}

impl PooledScratch for StreamCache {
    fn prepare(&mut self) {}
    fn retain(&mut self, lim: &crate::limits::Limits) {
        self.discard_columns();
        self.release_oversized(lim);
    }
}
