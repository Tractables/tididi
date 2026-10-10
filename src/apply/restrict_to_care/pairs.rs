//! Compact liveness marks with an exact spill bitmap for wide nodes.

use rustc_hash::FxHashMap;
use crate::{Engine, Tdd};
use crate::diagram::NodeIdx;
use crate::limits::OperationError;
use crate::vtree::VtreeIdx;

/// Narrow nodes keep one word; wide nodes allocate only when first marked.
/// Marks are unioned across every context in which a shared node is visited.
pub(super) struct PairMarks {
    small: Vec<Vec<u64>>,
    wide: FxHashMap<(VtreeIdx, NodeIdx), Vec<u64>>,
    all: bool,
}

impl PairMarks {
    /// Allocate narrow marks through the caller's limits.
    pub(super) fn new(eng: &Engine, f: &Tdd) -> Result<Self, OperationError> {
        let mut small = Vec::new();
        eng.limits().reserve_exact(&mut small, f.levels.len())?;
        for level in &f.levels {
            let mut row = Vec::new();
            eng.limits().try_resize(&mut row, level.nodes().len(), 0)?;
            small.push(row);
        }
        Ok(Self { small, wide: FxHashMap::default(), all: false })
    }

    /// Identity marking for walks with no constrained pairs.
    pub(super) fn all() -> Self {
        Self { small: Vec::new(), wide: FxHashMap::default(), all: true }
    }

    /// Mark pairs `pairs` of one node live in at least one care context,
    /// with one look-up of a wide node's row for all of them.
    pub(super) fn mark_many(&mut self, eng: &Engine, v: VtreeIdx, node: NodeIdx,
                           count: usize, pairs: impl IntoIterator<Item = u32>) -> Result<(), OperationError> {
        if self.all { return Ok(()); }
        if count <= 64 {
            let word = &mut self.small[v.idx()][node.idx()];
            for k in pairs {
                *word |= 1 << k;
            }
            return Ok(());
        }
        let key = (v, node);
        if !self.wide.contains_key(&key) {
            let mut words = Vec::new();
            eng.limits().try_resize(&mut words, count.div_ceil(64), 0)?;
            if self.wide.len() == self.wide.capacity() {
                eng.limits().reserve_map(&mut self.wide, 1)?;
            }
            self.wide.insert(key, words);
        }
        let words = self.wide.get_mut(&key).expect("the mark row exists");
        for k in pairs {
            words[k as usize / 64] |= 1 << (k % 64);
        }
        Ok(())
    }

    /// One node's marks, looked up once for a pass over its pairs.
    pub(super) fn row(&self, v: VtreeIdx, node: NodeIdx, count: usize) -> PairRow<'_> {
        if self.all {
            PairRow::All
        } else if count <= 64 {
            PairRow::Word(self.small[v.idx()][node.idx()])
        } else {
            self.wide.get(&(v, node)).map_or(PairRow::Word(0), |words| PairRow::Words(words))
        }
    }

    /// Whether all source pairs survive, including a partial last word.
    pub(super) fn complete(&self, v: VtreeIdx, node: NodeIdx, count: usize) -> bool {
        if self.all { return true; }
        if count <= 64 {
            self.small[v.idx()][node.idx()].count_ones() as usize == count
        } else {
            self.wide.get(&(v, node)).is_some_and(|words|
                words.iter().map(|w| w.count_ones() as usize).sum::<usize>() == count)
        }
    }
}

/// One node's pair marks ([`PairMarks::row`]).
#[derive(Clone, Copy)]
pub(super) enum PairRow<'a> {
    All,
    Word(u64),
    Words(&'a [u64]),
}

impl PairRow<'_> {
    /// Whether pair `k` of the node is marked.
    #[inline]
    pub(super) fn contains(self, k: usize) -> bool {
        match self {
            PairRow::All => true,
            PairRow::Word(w) => k < 64 && w & (1 << k) != 0,
            PairRow::Words(words) => words[k / 64] & (1 << (k % 64)) != 0,
        }
    }
}

#[cfg(test)]
#[path = "tests/pairs.rs"]
mod tests;
