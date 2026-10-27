//! The per-apply cache of already-computed child columns for streaming levels.

use crate::diagram::WeightValue;
use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use super::{CountVec, FloatCol};

/// Lazily computed child columns for streaming-target levels whose children
/// are still explicit, indexed by vtree node.
///
/// The value kind is fixed for the whole apply by the
/// [`WeightStore`](crate::diagram::WeightStore) in play, if any, so this is
/// one enum rather than parallel caches: an apply is integer, weighted,
/// scaled-float, or does no streaming at all.
///
/// A scaled-float apply also keeps here the column of each level it summed
/// out until that level's parent has read it; the columns of levels whose
/// parent it does not sum out go to the store when the sweep ends.
#[derive(Default)]
pub(crate) enum StreamCache {
    /// Nothing is being marginalized, so no streaming route is reachable.
    #[default]
    None,
    Int(Vec<Option<CountVec>>),
    Weighted(Vec<Option<Vec<WeightValue>>>),
    Float(Vec<Option<FloatCol>>),
}

/// The value kind of an apply's streamed columns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StreamKind {
    Int,
    Weighted,
    Float,
}

impl StreamCache {
    /// Select the column kind and size; nonstreaming operations leave it parked.
    pub(crate) fn reset(&mut self, num_nodes: usize, kind: Option<StreamKind>) {
        let Some(kind) = kind else { return };
        match (self, kind) {
            (StreamCache::Weighted(cols), StreamKind::Weighted) => cols.resize_with(num_nodes, || None),
            (StreamCache::Int(cols), StreamKind::Int) => cols.resize_with(num_nodes, || None),
            (StreamCache::Float(cols), StreamKind::Float) => cols.resize_with(num_nodes, || None),
            (cache, StreamKind::Weighted) => *cache = StreamCache::Weighted(vec_of_none(num_nodes)),
            (cache, StreamKind::Int) => *cache = StreamCache::Int(vec_of_none(num_nodes)),
            (cache, StreamKind::Float) => *cache = StreamCache::Float(vec_of_none(num_nodes)),
        }
    }

    /// The scaled-float columns. Only ever asked for on the scaled-float route.
    pub(crate) fn float(&self) -> &[Option<FloatCol>] {
        match self {
            StreamCache::Float(cols) => cols,
            _ => unreachable!("scaled-float streaming without a scaled-float cache"),
        }
    }

    /// The scaled-float columns, for the walk that fills them in.
    pub(crate) fn float_mut(&mut self) -> &mut [Option<FloatCol>] {
        match self {
            StreamCache::Float(cols) => cols,
            _ => unreachable!("scaled-float streaming without a scaled-float cache"),
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
            Self::Float(cols) => cols.clear(),
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
            Self::Float(cols) => visit(cols),
        }
    }
}

impl PooledScratch for StreamCache {
    fn prepare(&mut self) {}
    fn retain(&mut self, lim: &crate::limits::Limits) -> usize {
        self.discard_columns();
        self.release_oversized(lim)
    }
}
