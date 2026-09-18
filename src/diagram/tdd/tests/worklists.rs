//! Worklist access for the reduction tests: reading what a rewrite seeded, and
//! starting a sweep from an exact set of levels.

use super::{Dirty, Pass, Tdd};

impl Dirty {
    /// What `pass` still owes, without taking it.
    pub(crate) fn levels(&self, pass: Pass) -> &[u32] {
        &self.lists[pass as usize]
    }
}

impl Tdd {
    /// The twin-contraction worklist, for a test that asserts on what a rewrite
    /// seeded.
    pub(crate) fn contract_worklist(&self) -> &[u32] {
        self.dirty.levels(Pass::Contract)
    }

    /// Drive the twin-contraction worklist directly, for a test that wants a
    /// sweep to start from exactly `levels`.
    pub(crate) fn seed_contract_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.clear(Pass::Contract);
        self.dirty.requeue(Pass::Contract, levels);
    }

    /// The same for the leaf-contraction worklist.
    pub(crate) fn seed_leaf_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.clear(Pass::LeafContract);
        self.dirty.requeue(Pass::LeafContract, levels);
    }
}
