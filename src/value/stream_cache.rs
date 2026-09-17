//! The per-apply cache of already-computed child columns for streaming levels.

use crate::diagram::WeightValue;
use crate::limits::pool::{Pool, release_if_oversized};
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
    /// Take an empty cache with `num_nodes` slots, or return
    /// [`StreamCache::None`] when this apply marginalizes nothing.
    ///
    /// `weighted` is `None` when nothing is being marginalized. A pooled cache
    /// of the other kind is dropped: an engine runs one kind, so this costs a
    /// reallocation only on the first apply after a kind switch.
    pub(crate) fn take(pool: &Pool<StreamCache>, num_nodes: usize, weighted: Option<bool>) -> Self {
        let Some(weighted) = weighted else { return StreamCache::None };
        match (pool.take(), weighted) {
            (StreamCache::Weighted(mut cols), true) => {
                cols.resize_with(num_nodes, || None);
                StreamCache::Weighted(cols)
            }
            (StreamCache::Int(mut cols), false) => {
                cols.resize_with(num_nodes, || None);
                StreamCache::Int(cols)
            }
            (_, true) => StreamCache::Weighted(vec_of_none(num_nodes)),
            (_, false) => StreamCache::Int(vec_of_none(num_nodes)),
        }
    }

    /// Discard computed columns and return bounded table capacity to the pool.
    pub(crate) fn put(mut self, lim: &crate::limits::Limits, pool: &Pool<StreamCache>) {
        match &mut self {
            StreamCache::None => return,
            StreamCache::Int(cols) => retire(lim, cols),
            StreamCache::Weighted(cols) => retire(lim, cols),
        }
        pool.put(self);
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

/// Discard column values and retain the table allocation within the scratch cap.
fn retire<T>(lim: &crate::limits::Limits, cols: &mut Vec<Option<T>>) {
    cols.clear();
    release_if_oversized(lim, cols);
}
