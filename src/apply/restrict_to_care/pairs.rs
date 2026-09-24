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
            eng.limits().try_resize(&mut row, level.nodes.len(), 0)?;
            small.push(row);
        }
        Ok(Self { small, wide: FxHashMap::default(), all: false })
    }

    /// Identity marking for walks with no constrained pairs.
    pub(super) fn all() -> Self {
        Self { small: Vec::new(), wide: FxHashMap::default(), all: true }
    }

    /// Mark one pair as live in at least one care context.
    pub(super) fn mark(&mut self, eng: &Engine, v: VtreeIdx, node: NodeIdx,
                      pair: usize, count: usize) -> Result<(), OperationError> {
        if self.all { return Ok(()); }
        if count <= 64 {
            self.small[v.idx()][node.idx()] |= 1 << pair;
        } else {
            let key = (v, node);
            if !self.wide.contains_key(&key) {
                let mut words = Vec::new();
                eng.limits().try_resize(&mut words, count.div_ceil(64), 0)?;
                if self.wide.len() == self.wide.capacity() {
                    eng.limits().reserve_map(&mut self.wide, 1)?;
                }
                self.wide.insert(key, words);
            }
            self.wide.get_mut(&key).expect("the mark row exists")[pair / 64] |= 1 << (pair % 64);
        }
        Ok(())
    }

    /// Whether a source pair survived any care context.
    pub(super) fn contains(&self, v: VtreeIdx, node: NodeIdx, pair: usize, count: usize) -> bool {
        if self.all { return true; }
        if count <= 64 {
            self.small[v.idx()][node.idx()] & (1 << pair) != 0
        } else {
            self.wide.get(&(v, node)).is_some_and(|words| words[pair / 64] & (1 << (pair % 64)) != 0)
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

#[cfg(test)]
#[path = "tests/pairs.rs"]
mod tests;
