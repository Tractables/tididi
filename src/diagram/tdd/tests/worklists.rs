//! Worklist access for the reduction tests: reading what a rewrite seeded, and
//! starting a sweep from an exact set of levels.

use super::{Dirty, Pass, Tdd};

impl Dirty {
    /// What `pass` still owes, without taking it.
    pub(crate) fn levels(&self, pass: Pass) -> &[u32] {
        &self.lists[pass as usize]
    }

    /// Drop every fusion record, for a test that runs each pair fusion a
    /// record would skip.
    pub(crate) fn forget_all_fused(&mut self) {
        self.fused.clear();
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

#[test]
fn stacked_lists_have_room_for_the_seeds_and_the_prune() {
    let mut under = Dirty::default();
    under.requeue(Pass::Contract, [1, 2]);
    let mut over = Dirty::default();
    over.requeue(Pass::ContentTwin, [3]);
    let out = Dirty::stacked(&under, &over, 3);
    assert_eq!(out.levels(Pass::Contract), [1, 2]);
    assert!(out.levels(Pass::LeafContract).is_empty());
    assert_eq!(out.levels(Pass::ContentTwin), [3]);
    for (pass, room) in [(Pass::Contract, 6), (Pass::LeafContract, 6), (Pass::ContentTwin, 3)] {
        let list = &out.lists[pass as usize];
        assert!(list.capacity() - list.len() >= room, "{pass:?} has room for a seed and an invalidation of each rebuilt level");
    }
}

#[test]
fn seeding_leaves_room_for_the_prune_after_it() {
    let vtree = crate::Vtree::balanced(4);
    let eng = crate::Engine::new();
    let rebuilt: Vec<_> = vtree.internal_bottomup().map(|(t, _, _)| t).collect();
    let dirty = Dirty::default().seeded(&vtree, rebuilt.iter().copied(), Some(&eng)).unwrap();
    for pass in [Pass::Contract, Pass::LeafContract] {
        let list = &dirty.lists[pass as usize];
        assert_eq!(list.len(), rebuilt.len());
        assert!(list.capacity() - list.len() >= rebuilt.len(), "{pass:?} has room for an invalidation of each seeded level");
    }
}

#[test]
fn reserving_for_invalidations_makes_room_in_every_list() {
    let mut dirty = Dirty::default();
    dirty.requeue(Pass::LeafContract, [4]);
    dirty.reserve_all(5);
    for list in &dirty.lists {
        assert!(list.capacity() - list.len() >= 5);
    }
    assert_eq!(dirty.levels(Pass::LeafContract), [4]);
}

#[test]
fn a_push_drops_its_levels_fusion_record_and_a_merge_move_or_seed_drops_them_all() {
    let eng = crate::Engine::new();
    let mut dirty = Dirty::default();
    for level in [3, 70] {
        dirty.set_fused(&eng, level);
    }
    assert!(dirty.is_fused(3) && dirty.is_fused(70));
    assert!(!dirty.is_fused(4) && !dirty.is_fused(1000), "a level never fused has no record");
    dirty.push_all(3, None).unwrap();
    assert!(!dirty.is_fused(3) && dirty.is_fused(70), "a push drops its own level's record only");
    dirty.forget_fused(70);
    assert!(!dirty.is_fused(70));
    dirty.set_fused(&eng, 70);

    let copy = dirty.clone_on(&eng).unwrap();
    assert!(copy.is_fused(70), "a copy of the diagram keeps the record");
    let mut merged = copy.clone();
    merged.merge_under(Dirty::default());
    assert!(!merged.is_fused(70));
    let mut moved = copy.clone();
    let identity: Vec<crate::vtree::VtreeIdx> = (0..80).map(crate::vtree::VtreeIdx).collect();
    moved.remap(&identity);
    assert!(!moved.is_fused(70));
    let vtree = crate::Vtree::balanced(4);
    let seeded = copy.seeded(&vtree, std::iter::empty(), Some(&eng)).unwrap();
    assert!(!seeded.is_fused(70));
}
