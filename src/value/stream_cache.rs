//! The per-apply cache of already-computed child columns for streaming levels.

use crate::diagram::WeightVal;
use crate::limits::pool::Pool;
use crate::limits::ApplyBudget;
use super::CountVec;

/// Lazily computed child columns for streaming-target levels whose children
/// are still explicit, indexed by vtree node.
///
/// The value kind is fixed for the whole apply by whether a
/// [`WeightStore`] is in play, so this is one enum rather than two parallel
/// caches: an apply is integer, weighted, or does no streaming at all.
#[derive(Default)]
pub(crate) enum StreamCache {
    /// Nothing is being marginalized, so no streaming route is reachable.
    #[default]
    None,
    Int(Vec<Option<CountVec<ApplyBudget>>>),
    Weighted(Vec<Option<Vec<WeightVal>>>),
}

impl StreamCache {
    /// Take the pooled cache and clear its first `num_nodes` slots, or return
    /// [`StreamCache::None`] when this apply marginalizes nothing.
    ///
    /// `weighted` is `None` when nothing is being marginalized. A pooled cache
    /// of the other kind is dropped: an engine runs one kind, so this costs a
    /// reallocation only on the first apply after a kind switch.
    pub(crate) fn take(pool: &Pool<StreamCache>, num_nodes: usize, weighted: Option<bool>) -> Self {
        let Some(weighted) = weighted else { return StreamCache::None };
        match (pool.take(), weighted) {
            (StreamCache::Weighted(mut cols), true) => {
                reset(&mut cols, num_nodes);
                StreamCache::Weighted(cols)
            }
            (StreamCache::Int(mut cols), false) => {
                reset(&mut cols, num_nodes);
                StreamCache::Int(cols)
            }
            (_, true) => StreamCache::Weighted(vec_of_none(num_nodes)),
            (_, false) => StreamCache::Int(vec_of_none(num_nodes)),
        }
    }

    /// Return the cache to the pool, so the next apply reuses its columns.
    pub(crate) fn put(self, pool: &Pool<StreamCache>) {
        if !matches!(self, StreamCache::None) {
            pool.put(self);
        }
    }

    /// The integer columns. Only ever asked for on the integer route.
    pub(crate) fn int(&self) -> &[Option<CountVec<ApplyBudget>>] {
        match self {
            StreamCache::Int(cols) => cols,
            _ => unreachable!("integer streaming without an integer cache"),
        }
    }

    /// The weighted columns. Only ever asked for on the weighted route.
    pub(crate) fn weighted(&self) -> &[Option<Vec<WeightVal>>] {
        match self {
            StreamCache::Weighted(cols) => cols,
            _ => unreachable!("weighted streaming without a weighted cache"),
        }
    }

    /// The integer columns, for the walk that fills them in.
    pub(crate) fn int_mut(&mut self) -> &mut [Option<CountVec<ApplyBudget>>] {
        match self {
            StreamCache::Int(cols) => cols,
            _ => unreachable!("integer streaming without an integer cache"),
        }
    }

    /// The weighted columns, for the walk that fills them in.
    pub(crate) fn weighted_mut(&mut self) -> &mut [Option<Vec<WeightVal>>] {
        match self {
            StreamCache::Weighted(cols) => cols,
            _ => unreachable!("weighted streaming without a weighted cache"),
        }
    }
}

fn vec_of_none<T>(num_nodes: usize) -> Vec<Option<T>> {
    let mut cols = Vec::new();
    cols.resize_with(num_nodes, || None);
    cols
}

fn reset<T>(cols: &mut Vec<Option<T>>, num_nodes: usize) {
    if cols.len() < num_nodes {
        cols.resize_with(num_nodes, || None);
    }
    for slot in cols[..num_nodes].iter_mut() {
        *slot = None;
    }
}
