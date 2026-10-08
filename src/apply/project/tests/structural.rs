use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, compile_clauses, or_of_cubes, rand_cnf, same_as_stored, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};

thread_local! {
    /// The cells on this thread [`regroup_single_by_left`] wrote with a
    /// rewritten left side and the right side as the level stored it, whose
    /// order only the sort of a run's right sides made canonical
    /// ([`unsorted_right_runs`]).
    static UNSORTED_RIGHT_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn note_unsorted_right_run() {
    UNSORTED_RIGHT_RUNS.with(|n| n.set(n.get() + 1));
}

/// How many cells of one node on this thread a quantification wrote in
/// canonical order from a run of right sides the level stored out of order,
/// with no other step that would have sorted them.
fn unsorted_right_runs() -> u64 {
    UNSORTED_RIGHT_RUNS.with(std::cell::Cell::get)
}

/// The map a quantified leaf hands its parent: every label becomes ⊤.
fn freed_leaf(eng: &Engine) -> Remap {
    Runs::all_to_first(eng.limits(), crate::diagram::LEAF_WIDTH, 0u32).unwrap()
}

#[test]
fn regrouping_checks_its_allocations_before_reduction() {
    let tree = Arc::new(Vtree::balanced(4));
    let original = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    assert_canonical(&original);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    {
        let freed = freed_leaf(&eng);
        let (left, _) = tree.children(parent);
        let (below_left, below_right) =
            if left == leaf { (Some(&freed), None) } else { (None, Some(&freed)) };
        let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        let mut work = Rewrite { eng: &eng, gate: eng.limits().gate_with(1), emitted: 0 };
        assert_eq!(regroup(&mut work, &mut original.clone(), parent, below_left, below_right).err(), Some(OperationError::OverBudget));
    }
    let result = exists_leaves_structural(&eng, original, &[leaf], false, ReductionPlan::default()).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 16u32.into());
}

#[test]
fn regrouping_polls_inside_a_level() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    let freed = freed_leaf(&eng);
    let (left, _) = tree.children(parent);
    let (below_left, below_right) =
        if left == leaf { (Some(&freed), None) } else { (None, Some(&freed)) };
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(2)), ..StopRules::default()
    }));
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate_with(1), emitted: 0 };
    assert_eq!(regroup(&mut work, &mut f, parent, below_left, below_right).err(), Some(OperationError::Stopped));
    assert_eq!(work.emitted, 0);
}

#[test]
fn freeing_a_whole_subtree_matches_one_variable_at_a_time() {
    let tree = Arc::new(Vtree::balanced(6));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap() & Tdd::clause(&tree, [-4, 5, 6]).unwrap();
    let mut f = f;
    f.minimize().unwrap();
    assert_canonical(&f);
    // The balanced vtree over six variables groups 1..=3 under one subtree.
    let block: Vec<VtreeIdx> = [1, 2, 3].iter().map(|&v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let at_once = exists_leaves_structural(&eng, f.clone(), &block, false, ReductionPlan::default()).unwrap();
    assert_canonical(&at_once);
    let mut one_at_a_time = f;
    for &leaf in &block {
        one_at_a_time = exists_leaves_structural(&eng, one_at_a_time, &[leaf], false, ReductionPlan::default()).unwrap();
    }
    assert_canonical(&one_at_a_time);
    assert!(at_once.equivalent(&one_at_a_time).unwrap());
    assert_eq!(at_once.node_count(), one_at_a_time.node_count());
    assert_eq!(at_once.pair_count(), one_at_a_time.pair_count());
    assert_eq!(at_once.model_count().unwrap(), one_at_a_time.model_count().unwrap());
}

#[test]
fn quantifying_every_variable_leaves_the_constant() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    let all: Vec<VtreeIdx> = (1..=4).map(|v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let result = exists_leaves_structural(&eng, f, &all, false, ReductionPlan::default()).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 16u32.into());
}

#[test]
fn structural_projection_recovers_after_each_refused_reservation() {
    let tree = Arc::new(Vtree::balanced(5));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap() & Tdd::clause(&tree, [-1, 4, 5]).unwrap();
    let mut f = f;
    f.minimize().unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let mut reached_success = false;
    for nth in 0..512 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        match exists_leaves_structural(&eng, f.clone(), &[leaf], false, ReductionPlan::default()) {
            Ok(result) => { assert_canonical(&result); reached_success = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        eng.limits().grant_every_reserve();
        let result = exists_leaves_structural(&eng, f.clone(), &[leaf], false, ReductionPlan::default()).unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), 30u32.into());
    }
    assert!(reached_success, "the sweep must cover every reservation");
}

#[test]
fn quantifying_an_unreduced_product_matches_quantifying_a_reduced_one() {
    // `and` leaves its result for the caller to reduce, so the product holds
    // nodes nothing reaches. The sweep prunes them first; what comes out is
    // what comes out of the same quantification over the reduced product.
    let tree = Arc::new(Vtree::balanced(6));
    let product = Tdd::clause(&tree, [1, -2, 3]).unwrap()
        & Tdd::clause(&tree, [-1, 4, 5]).unwrap()
        & Tdd::clause(&tree, [2, -4, 6]).unwrap();
    assert!(!product.dirty.is_empty(), "the product still owes the passes");
    let mut reduced = product.clone();
    reduced.minimize().unwrap();
    assert!(reduced.dirty.is_empty(), "minimize discharges them");

    let leaves: Vec<VtreeIdx> = [1, 4].iter().map(|&v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let from_product = exists_leaves_structural(&eng, product, &leaves, false, ReductionPlan::default()).unwrap();
    let from_reduced = exists_leaves_structural(&eng, reduced, &leaves, false, ReductionPlan::default()).unwrap();
    assert_canonical(&from_product);
    assert_eq!(from_product.node_count(), from_reduced.node_count());
    assert_eq!(from_product.pair_count(), from_reduced.pair_count());
    assert_eq!(from_product.model_count().unwrap(), from_reduced.model_count().unwrap());
}

/// The sweep adds no level holding a node its parent level does not name, so
/// the prune after it walks down from the output only where a level lost a
/// node. What it leaves has as many nodes as a prune of the whole diagram
/// leaves, and minimizes to what the full reduction gives.
#[test]
fn the_prune_after_the_sweep_drops_what_a_whole_prune_drops() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x0001_005e_5ee9);
    for nvars in [6u32, 9] {
        for (name, tree) in vtree_shapes(nvars) {
            for _ in 0..6 {
                let clauses = rand_cnf(&mut rng, nvars, CnfShape { clauses: 2 * nvars as usize, width: 3 });
                let f = compile_clauses(&tree, &clauses);
                let leaves: Vec<VtreeIdx> = (1..=nvars)
                    .filter(|_| rng.below(3) == 0)
                    .map(|v| tree.leaf_of(VarId(v)).unwrap())
                    .collect();
                if leaves.is_empty() || leaves.len() == nvars as usize {
                    continue;
                }
                let canonical = exists_leaves_structural(&eng, f.clone(), &leaves, false, ReductionPlan::default()).unwrap();
                let pruned = exists_leaves_structural(&eng, f, &leaves, false, ReductionPlan::Prune).unwrap();
                let mut whole = pruned.clone();
                whole.dirty.set_loose(None);
                eng.reduce(&mut whole, ReductionPlan::Prune).unwrap();
                assert_eq!(pruned.node_count(), whole.node_count(), "{name}: {clauses:?} without {leaves:?}");
                let mut minimized = pruned;
                minimized.minimize().unwrap();
                assert_canonical(&minimized);
                assert_eq!(minimized.node_count(), canonical.node_count(), "{name}: {clauses:?} without {leaves:?}");
                assert_eq!(minimized.pair_count(), canonical.pair_count(), "{name}: {clauses:?} without {leaves:?}");
            }
        }
    }
}

/// A fan-out for each of `keys` child nodes: a nonempty ascending set of the
/// cells below `cells`, drawn at random.
fn random_remap(eng: &Engine, rng: &mut Lcg, keys: usize, cells: u32) -> Remap {
    let mut entries = Vec::new();
    for key in 0..keys as u32 {
        let before = entries.len();
        for cell in 0..cells {
            if rng.coin() { entries.push((key, cell)); }
        }
        if entries.len() == before { entries.push((key, rng.below(u64::from(cells)) as u32)); }
    }
    Runs::pack(eng.limits(), keys, &entries, 0u32).unwrap()
}

/// The owner-set rule written out: every atom with the nodes whose pairs
/// expand to it, one cell per distinct owner set, numbered by the first atom
/// of it the scan reaches, and each node mapped to the cells of its atoms.
fn regroup_by_definition(
    level: &TddLevel,
    left: Option<&Remap>,
    right: Option<&Remap>,
) -> (Vec<Vec<ChildPair>>, Vec<Vec<u32>>) {
    let expand = |remap: Option<&Remap>, side: EncodedChildRef| match remap {
        Some(map) => map.get(ChildDecoder::structural().node(side).idx()).to_vec(),
        None => vec![side.0],
    };
    let mut atoms: Vec<ChildPair> = Vec::new();
    let mut owners: HashMap<ChildPair, Vec<u32>> = HashMap::new();
    for node in 0..level.nodes().len() as u32 {
        for pair in level.pairs_vec(node as usize) {
            for &left in &expand(left, pair.left) {
                for &right in &expand(right, pair.right) {
                    let atom = ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(right));
                    let set = owners.entry(atom).or_insert_with(|| {
                        atoms.push(atom);
                        Vec::new()
                    });
                    if set.last() != Some(&node) { set.push(node); }
                }
            }
        }
    }
    let mut sets: Vec<&Vec<u32>> = Vec::new();
    let mut cells: Vec<Vec<ChildPair>> = Vec::new();
    for atom in &atoms {
        let set = &owners[atom];
        let cell = match sets.iter().position(|&other| other == set) {
            Some(cell) => cell,
            None => {
                sets.push(set);
                cells.push(Vec::new());
                cells.len() - 1
            }
        };
        cells[cell].push(*atom);
    }
    for cell in &mut cells { cell.sort(); }
    let remap = (0..level.nodes().len() as u32)
        .map(|node| (0..sets.len() as u32).filter(|&cell| sets[cell as usize].contains(&node)).collect())
        .collect();
    (cells, remap)
}

/// The regroup against the owner-set rule written out, on every internal level
/// of random diagrams, with either child or both fanned out at random: the same
/// cells in the same order, the same fan-out, and no map exactly when nothing
/// changed. Levels of one node take the single-cell path, the others the
/// owner-set path.
#[test]
fn regrouping_matches_the_owner_set_rule() {
    let mut rng = Lcg::new(20260925);
    let eng = Engine::new();
    let (mut single, mut several) = (0usize, 0usize);
    for round in 0..24u32 {
        let num_vars = 6 + round % 9;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4 + round as usize, width: 4 });
        for (shape, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            if f.is_zero() { continue; }
            let keys = |child: VtreeIdx| {
                if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes().len() }
            };
            for &parent in vtree.bottomup_slice() {
                let level = &f.levels[parent.idx()];
                if vtree.node(parent).is_leaf() || level.nodes().is_empty() { continue; }
                let (left_child, right_child) = vtree.children(parent);
                for fanned in [(true, false), (false, true), (true, true)] {
                    // Few cells make owner sets collide; many make a level of
                    // one node hold more atoms than an insertion sort takes.
                    let mut cells = || {
                        let most = if rng.coin() { 5 } else { 80 };
                        1 + rng.below(most) as u32
                    };
                    let (left_cells, right_cells) = (cells(), cells());
                    let left = fanned.0.then(|| random_remap(&eng, &mut rng, keys(left_child), left_cells));
                    let right = fanned.1.then(|| random_remap(&eng, &mut rng, keys(right_child), right_cells));
                    let (cells, fan_out) = regroup_by_definition(level, left.as_ref(), right.as_ref());

                    let mut rewritten = f.clone();
                    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
                    let remap = regroup(&mut work, &mut rewritten, parent, left.as_ref(), right.as_ref()).unwrap();
                    let written = &rewritten.levels[parent.idx()];
                    let got: Vec<Vec<ChildPair>> =
                        (0..written.nodes().len()).map(|cell| written.pairs_vec(cell).to_vec()).collect();
                    assert_eq!(got, cells, "{shape}, level {parent:?}, fanned {fanned:?}: cells");
                    let identity: Vec<Vec<u32>> = (0..level.nodes().len() as u32).map(|node| vec![node]).collect();
                    match remap {
                        Some(remap) => {
                            let got: Vec<Vec<u32>> = (0..level.nodes().len()).map(|node| remap.get(node).to_vec()).collect();
                            assert_eq!(got, fan_out, "{shape}, level {parent:?}, fanned {fanned:?}: fan-out");
                            assert_ne!(fan_out, identity, "{shape}, level {parent:?}: an unchanged level returns no map");
                        }
                        None => assert_eq!(fan_out, identity, "{shape}, level {parent:?}, fanned {fanned:?}: no map"),
                    }
                    if level.nodes().len() == 1 { single += 1 } else { several += 1 }
                }
            }
        }
    }
    assert!(single > 100 && several > 1000, "levels of one node {single}, of several {several}");
}

/// A fan-out of one cell for each of `keys` child nodes, drawn below `cells`.
fn one_cell_remap(eng: &Engine, rng: &mut Lcg, keys: usize, cells: u32) -> Remap {
    let entries: Vec<(u32, u32)> = (0..keys as u32).map(|key| (key, rng.below(u64::from(cells)) as u32)).collect();
    Runs::pack(eng.limits(), keys, &entries, 0u32).unwrap()
}

/// What a regroup wrote at `parent`, cell by cell, and its map node by node,
/// the identity when there is none.
fn written(f: &Tdd, before: &TddLevel, parent: VtreeIdx, remap: Option<Remap>) -> (Vec<Vec<ChildPair>>, Vec<Vec<u32>>) {
    let level = &f.levels[parent.idx()];
    let cells = (0..level.nodes().len()).map(|cell| level.pairs_vec(cell)).collect();
    let fan_out = (0..before.nodes().len())
        .map(|node| match &remap {
            Some(remap) => remap.get(node).to_vec(),
            None => vec![node as u32],
        })
        .collect();
    (cells, fan_out)
}

/// A level each of whose pairs expands to one atom, against the owner-set
/// rule written out: maps of one cell per node into a few cells, so that
/// atoms of several owners are common, or into many, so that the partition
/// is often kept; on every internal level of several nodes of random
/// diagrams, either child or both rewritten. The diagrams are random CNFs
/// and sparse sets of models, whose levels often hold one pair per node.
#[test]
fn a_level_of_one_atom_per_pair_follows_the_owner_set_rule() {
    let mut rng = Lcg::new(0x0a70_b1e5);
    let eng = Engine::new();
    let (mut done, mut shared, mut kept, mut one_pair_shared) = (0usize, 0usize, 0usize, 0usize);
    for round in 0..24u32 {
        let num_vars = 6 + round % 9;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4 + round as usize, width: 4 });
        // About three models a variable, drawn once for every vtree.
        let rows: Vec<bool> = (0..1u64 << num_vars).map(|_| rng.below(1 << num_vars) < 3 * u64::from(num_vars)).collect();
        let vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
        for (shape, vtree) in vtree_shapes(num_vars) {
            for f in [compile_clauses(&vtree, &clauses), or_of_cubes(&vtree, &vars, |row| rows[row])] {
                if f.is_zero() { continue; }
                let keys = |child: VtreeIdx| {
                    if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes().len() }
                };
                for &parent in vtree.bottomup_slice() {
                    let level = &f.levels[parent.idx()];
                    if vtree.node(parent).is_leaf() || level.nodes().len() < 2 { continue; }
                    let (left_child, right_child) = vtree.children(parent);
                    for fanned in [(true, false), (false, true), (true, true)] {
                        let mut cells = |keys: usize| match rng.below(3) {
                            0 => 1 + rng.below(3) as u32,
                            1 => 1 + keys as u32 / 2,
                            _ => 4 * keys as u32,
                        };
                        let (left_cells, right_cells) = (cells(keys(left_child)), cells(keys(right_child)));
                        let left = fanned.0.then(|| one_cell_remap(&eng, &mut rng, keys(left_child), left_cells));
                        let right = fanned.1.then(|| one_cell_remap(&eng, &mut rng, keys(right_child), right_cells));
                        let expected = regroup_by_definition(level, left.as_ref(), right.as_ref());

                        let mut rewritten = f.clone();
                        let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
                        let OneAtom::Done(remap) =
                            regroup_atom_per_pair(&mut work, &mut rewritten, parent, left.as_ref(), right.as_ref()).unwrap()
                        else {
                            panic!("{shape}, level {parent:?}, fanned {fanned:?}: declined");
                        };
                        let unchanged = remap.is_none();
                        let got = written(&rewritten, level, parent, remap);
                        assert_eq!(got.0, expected.0, "{shape}, level {parent:?}, fanned {fanned:?}: cells");
                        assert_eq!(got.1, expected.1, "{shape}, level {parent:?}, fanned {fanned:?}: fan-out");
                        let identity: Vec<Vec<u32>> = (0..level.nodes().len() as u32).map(|node| vec![node]).collect();
                        assert_eq!(unchanged, expected.1 == identity, "{shape}, level {parent:?}: a map exactly when the partition changed");
                        done += 1;
                        let merged = expected.1.iter().any(|cells| cells.len() > 1)
                            || expected.0.len() < level.nodes().len();
                        shared += usize::from(merged);
                        kept += usize::from(unchanged);
                        let one_pair = (0..level.nodes().len()).all(|node| level.pair_count_at(node) == 1);
                        one_pair_shared += usize::from(one_pair && merged);
                    }
                }
            }
        }
    }
    assert!(done > 1000 && shared > 100 && kept > 100 && one_pair_shared > 50,
        "levels {done}, with a shared atom {shared}, kept {kept}, of one pair a node and merged {one_pair_shared}");
}

/// A level of one node written run by run of its left side, against the
/// owner-set rule written out: the same cell, whether the left side was left
/// as it is (the right one rewritten) or rewritten into cells that merge and
/// reorder its references (the right one either way); and a node whose runs
/// are long while its right side was rewritten is left to the rows.
#[test]
fn a_level_of_one_node_is_written_alike_run_by_run() {
    let mut rng = Lcg::new(0x51_6e1e);
    let eng = Engine::new();
    let (mut by_left, mut by_rewritten_left, mut declined) = (0usize, 0usize, 0usize);
    for round in 0..24u32 {
        let num_vars = 6 + round % 9;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4 + round as usize, width: 4 });
        for (shape, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            if f.is_zero() { continue; }
            for &parent in vtree.bottomup_slice() {
                let level = &f.levels[parent.idx()];
                if vtree.node(parent).is_leaf() || level.nodes().len() != 1 { continue; }
                let keys = |child: VtreeIdx| {
                    if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes().len() }
                };
                let (left_child, right_child) = vtree.children(parent);
                let remap = |rng: &mut Lcg, keys: usize, cells: u32| {
                    if rng.coin() { one_cell_remap(&eng, rng, keys, cells) } else { random_remap(&eng, rng, keys, cells) }
                };
                for cells in [1, 3, 80, 400] {
                    let right = remap(&mut rng, keys(right_child), cells);
                    let left = remap(&mut rng, keys(left_child), cells);
                    let right_too = rng.coin().then(|| remap(&mut rng, keys(right_child), cells));
                    for (left, right) in [(None, Some(&right)), (Some(&left), right_too.as_ref())] {
                        let expected = regroup_by_definition(level, left, right);
                        let mut rewritten = f.clone();
                        let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
                        if regroup_single_by_left(&mut work, &mut rewritten, parent, left, right).unwrap() {
                            let got = written(&rewritten, level, parent, None);
                            assert_eq!(got, expected, "{shape}, level {parent:?}, {cells} cells, left rewritten {}", left.is_some());
                            if left.is_some() { by_rewritten_left += 1 } else { by_left += 1 }
                        } else {
                            let pairs = level.pairs_vec(0);
                            let runs = pairs.chunk_by(|a, b| a.left == b.left).count();
                            assert!(
                                (right.is_some() && pairs.len() > runs * BY_LEFT_PAIRS_PER_RUN)
                                    || !pairs.is_sorted_by_key(|pair| pair.left)
                                    || expected.0.is_empty(),
                                "{shape}, level {parent:?}: declined {} pairs in {runs} runs", pairs.len());
                            declined += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(by_left > 100 && by_rewritten_left > 100 && declined > 0,
        "written run by run {by_left}, with a rewritten left side {by_rewritten_left}, declined {declined}");
}

/// A fan-out of one cell for each of `keys` child nodes, the cells rising
/// in the order the runs of `pairs` name the keys by their left sides, the
/// keys no run names after them, and some cells skipped: a map that neither
/// merges nor reorders the runs' cells. A leaf's labels are not stored in
/// the order of their keys.
fn rising_remap(eng: &Engine, rng: &mut Lcg, keys: usize, pairs: &[ChildPair]) -> Remap {
    let named: Vec<usize> =
        pairs.chunk_by(|a, b| a.left == b.left).map(|run| ChildDecoder::structural().node(run[0].left).idx()).collect();
    let mut cells = vec![None; keys];
    let mut cell = 0u32;
    for key in named.iter().copied().chain(0..keys) {
        if cells[key].is_none() {
            cell += rng.below(2) as u32;
            cells[key] = Some(cell);
            cell += 1;
        }
    }
    let entries: Vec<(u32, u32)> = cells.iter().enumerate().map(|(key, cell)| (key as u32, cell.unwrap())).collect();
    Runs::pack(eng.limits(), keys, &entries, 0u32).unwrap()
}

/// A level of one node whose runs of one left reference store their right
/// sides out of order, as a conjunction can leave them (the relabelling
/// route where its map of the right side is not monotone, and the compiler's
/// own levels at times), written run by run with the left side rewritten and
/// the right one as it is: the owner-set rule's one cell, in canonical
/// order. Under a map that merges or reorders the left cells the pass sorts
/// them anyway; under one whose cells rise along the runs only the sort of
/// each run's right sides puts the cell in order, which
/// `unsorted_right_runs` counts.
#[test]
fn a_level_of_one_node_with_unsorted_runs_is_written_in_order() {
    let mut rng = Lcg::new(0x0f2_5011);
    let eng = Engine::new();
    let (mut levels, mut in_order_by_runs) = (0usize, 0u64);
    for round in 0..48u32 {
        let num_vars = 6 + round % 9;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4 + round as usize, width: 4 });
        for (shape, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            if f.is_zero() { continue; }
            for &parent in vtree.bottomup_slice() {
                if vtree.node(parent).is_leaf() || f.levels[parent.idx()].nodes().len() != 1 { continue; }
                let mut pairs = f.levels[parent.idx()].pairs_vec(0);
                // A node not stored in order of its left side is the other
                // regroups' to write.
                if !pairs.is_sorted_by_key(|pair| pair.left)
                    || !pairs.chunk_by(|a, b| a.left == b.left).any(|run| run.len() > 1)
                {
                    continue;
                }
                // Descending, which a stored run can be already: no invariant
                // sorts a node's pairs.
                for run in pairs.chunk_by_mut(|a, b| a.left == b.left) {
                    run.sort_unstable_by(|a, b| b.cmp(a));
                }
                let mut stored = f.clone();
                {
                    let level = &mut stored.levels[parent.idx()];
                    level.clear();
                    level.push_node(eng.limits(), &pairs).unwrap();
                }
                let level = &stored.levels[parent.idx()];
                let (left_child, _) = vtree.children(parent);
                let keys = if vtree.node(left_child).is_leaf() { LEAF_WIDTH } else { f.levels[left_child.idx()].nodes().len() };
                let maps = [
                    ("rising", rising_remap(&eng, &mut rng, keys, &pairs)),
                    ("one cell", one_cell_remap(&eng, &mut rng, keys, 3)),
                    ("fanned", random_remap(&eng, &mut rng, keys, 80)),
                ];
                for (what, left) in &maps {
                    let expected = regroup_by_definition(level, Some(left), None);
                    let mut rewritten = stored.clone();
                    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
                    let before = unsorted_right_runs();
                    assert!(regroup_single_by_left(&mut work, &mut rewritten, parent, Some(left), None).unwrap(),
                        "{shape}, level {parent:?}, {what}: declined");
                    let got = written(&rewritten, level, parent, None);
                    assert_eq!(got, expected, "{shape}, level {parent:?}, {what} map");
                    let by_runs = unsorted_right_runs() - before;
                    if *what == "rising" {
                        assert_eq!(by_runs, 1, "{shape}, level {parent:?}: a rising map leaves the order to the runs");
                    }
                    in_order_by_runs += by_runs;
                }
                levels += 1;
            }
        }
    }
    assert!(levels > 80 && in_order_by_runs >= levels as u64,
        "levels {levels}, put in order by the runs alone {in_order_by_runs}");
}

/// [`random_remap`] with about one key in six mapped to no cell.
fn remap_with_gaps(eng: &Engine, rng: &mut Lcg, keys: usize, cells: u32) -> Remap {
    let mut entries = Vec::new();
    for key in 0..keys as u32 {
        if rng.below(6) == 0 { continue; }
        let before = entries.len();
        for cell in 0..cells {
            if rng.coin() { entries.push((key, cell)); }
        }
        if entries.len() == before { entries.push((key, rng.below(u64::from(cells)) as u32)); }
    }
    Runs::pack(eng.limits(), keys, &entries, 0u32).unwrap()
}

/// Both writers of a level of one node against the owner-set rule written
/// out, on every such level of random diagrams: the rows whatever they cost,
/// and the sorted atoms under each side the pairs can be filed by. They write
/// the same node, and every shape of the rows is reached: one row with a left
/// side stored, rewritten, rewritten with a reference that expands to no
/// cell, or rewritten into one cell with the right side rewritten too, so the
/// atoms go uncounted; several rows with a left side stored, or rewritten
/// with some pairs marked through bits of their own and some not.
///
/// On implicit levels and on stored ones alike ([`same_as_stored`]): the
/// rows write the level an implicit one-node level leaves as they write the
/// one its stored copy leaves, once closed.
#[test]
fn a_level_of_one_node_is_written_alike_by_rows_and_by_sorting() {
    same_as_stored(written_alike_by_rows_and_by_sorting);
}

/// The cases of [`a_level_of_one_node_is_written_alike_by_rows_and_by_sorting`]:
/// the diagrams the rows wrote, their levels closed.
fn written_alike_by_rows_and_by_sorting() -> Vec<Tdd> {
    let mut out = Vec::new();
    let mut rng = Lcg::new(20260926);
    let eng = Engine::new();
    let mut reached: HashMap<&str, usize> = HashMap::new();
    for round in 0..24u32 {
        let num_vars = 6 + round % 9;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4 + round as usize, width: 4 });
        for (shape, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            if f.is_zero() { continue; }
            let keys = |child: VtreeIdx| {
                if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes().len() }
            };
            for &parent in vtree.bottomup_slice() {
                let level = &f.levels[parent.idx()];
                if vtree.node(parent).is_leaf() || level.nodes().len() != 1 { continue; }
                let pairs = &level.pairs_vec(0)[..];
                let (left_child, right_child) = vtree.children(parent);
                for fanned in [(true, false), (false, true), (true, true)] {
                    // One cell makes a rewritten left side a single row.
                    let mut cells = || match rng.below(3) {
                        0 => 1,
                        1 => 1 + rng.below(5) as u32,
                        _ => 1 + rng.below(200) as u32,
                    };
                    let (left_cells, right_cells) = (cells(), cells());
                    let gaps = rng.below(4) == 0;
                    let left = fanned.0.then(|| match gaps {
                        true => remap_with_gaps(&eng, &mut rng, keys(left_child), left_cells),
                        false => random_remap(&eng, &mut rng, keys(left_child), left_cells),
                    });
                    let right = fanned.1.then(|| random_remap(&eng, &mut rng, keys(right_child), right_cells));
                    let (cells, _) = regroup_by_definition(level, left.as_ref(), right.as_ref());
                    assert!(cells.len() <= 1, "{shape}, level {parent:?}: one node owns at most one cell");
                    let want = cells.first().map_or(&[][..], |cell| &cell[..]);
                    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };

                    let mut by_rows = f.clone();
                    let plan = RowPlan::of(&mut work, pairs, left.as_ref(), right.as_ref()).unwrap();
                    regroup_single_rows(&mut work, &mut by_rows, parent, left.as_ref(), right.as_ref(), &plan).unwrap();
                    let written = &by_rows.levels[parent.idx()];
                    assert_eq!(written.nodes().len(), 1, "{shape}, level {parent:?}, fanned {fanned:?}: rows");
                    assert_eq!(written.pairs_vec(0), want, "{shape}, level {parent:?}, fanned {fanned:?}: rows");

                    let expands_nowhere = |map: &Remap| {
                        pairs.iter().any(|pair| map.get(ChildDecoder::structural().node(pair.left).idx()).is_empty())
                    };
                    let shape_reached = match (plan.single, &left, &right) {
                        (Some(_), None, _) => "one row, left stored",
                        (Some(_), Some(map), _) if expands_nowhere(map) => "one row, a left reference expanding nowhere",
                        (Some(_), Some(map), Some(other))
                            if cell_count(map) == 1 && (cell_count(other) as u64).div_ceil(64) <= pairs.len() as u64 =>
                            "one row, atoms uncounted",
                        (Some(_), Some(_), _) => "one row, left rewritten",
                        (None, None, _) => "several rows, left stored",
                        (None, Some(map), _) => {
                            let filed = |pair: &ChildPair| (ChildDecoder::structural().node(pair.left).0, pair.right.0);
                            let groups = Runs::pack_by(eng.limits(), plan.left_keys as usize, pairs, filed, 0u32).unwrap();
                            let dense = dense_groups(&mut work, &groups, map, right.as_ref(), plan.words).unwrap();
                            let marked_once = (0..groups.len()).filter(|&g| !dense.get(g).is_empty()).count();
                            let visited = (0..groups.len()).filter(|&g| !groups.get(g).is_empty()).count();
                            match marked_once {
                                0 => "several rows, left rewritten, no group marked once",
                                n if n == visited => "several rows, left rewritten, every group marked once",
                                _ => "several rows, left rewritten, some groups marked once",
                            }
                        }
                    };
                    *reached.entry(shape_reached).or_default() += 1;

                    for by in [Side::Left, Side::Right] {
                        let (filed, stamped) = match by {
                            Side::Left => (left.as_ref(), right.as_ref()),
                            Side::Right => (right.as_ref(), left.as_ref()),
                        };
                        // The atoms filed under one reference are told apart
                        // by a rewritten other side.
                        let Some(stamped) = stamped else { continue };
                        let owned = scan_level(&mut work, level).unwrap();
                        let buckets = bucket_pairs(&mut work, &owned, by, filed).unwrap();
                        let mut by_sorting = f.clone();
                        regroup_single(&mut work, &mut by_sorting, parent, &owned, &buckets, by, stamped).unwrap();
                        let sorted = &by_sorting.levels[parent.idx()];
                        assert!(sorted.nodes == written.nodes, "{shape}, level {parent:?}, filed by {by:?}: node");
                        assert_eq!(sorted.pairs_vec(0), written.pairs_vec(0), "{shape}, level {parent:?}, filed by {by:?}");
                    }
                    for level in &mut by_rows.levels {
                        level.close();
                    }
                    out.push(by_rows);
                }
            }
        }
    }
    for shape in [
        "one row, left stored",
        "one row, a left reference expanding nowhere",
        "one row, atoms uncounted",
        "one row, left rewritten",
        "several rows, left stored",
        "several rows, left rewritten, some groups marked once",
    ] {
        let count = reached.get(shape).copied().unwrap_or(0);
        assert!(count > 10, "{shape}: {count} levels, of {reached:?}");
    }
    out
}

/// A row reads back exactly the columns set since it was last read, in
/// ascending order, and is clear afterwards: across widths on both sides of
/// one and several summary words, with sparse rows spread over the width,
/// dense ones, repeats, and whole words OR-ed in.
#[test]
fn a_row_reads_back_its_columns_in_order_however_wide() {
    let mut rng = Lcg::new(20260927);
    let eng = Engine::new();
    let lim = eng.limits();
    for words in [1u64, 2, 63, 64, 65, 4095, 4096, 4097, 9000, 70_000] {
        let width = words * 64;
        let mut row = Row::new(lim, words).unwrap();
        let mut out = Vec::new();
        for round in 0..12u32 {
            let many = match round % 4 {
                0 => 0,
                1 => 1 + rng.below(8),
                2 => 1 + rng.below(width.min(3000)),
                _ => width.min(20_000),
            };
            let mut want: Vec<u32> = (0..many).map(|_| rng.below(width) as u32).collect();
            // Repeats set a bit already set.
            let again: Vec<u32> = want.iter().copied().filter(|_| rng.coin()).collect();
            row.mark(&want, None);
            row.mark(&again, None);
            if round % 3 == 2 {
                let mut bits = vec![0u64; words as usize];
                for _ in 0..1 + rng.below(40) {
                    let column = rng.below(width) as u32;
                    bits[(column >> 6) as usize] |= 1u64 << (column & 63);
                    want.push(column);
                }
                row.or_words(&bits);
            }
            want.sort_unstable();
            want.dedup();
            let left = round * 7;
            out.clear();
            row.drain(lim, left, &mut out, 0, 1, u64::MAX).unwrap();
            let got: Vec<u32> = out.iter().map(|pair| pair.right.0).collect();
            assert_eq!(got, want, "{words} words, round {round}");
            assert!(out.iter().all(|pair| pair.left.0 == left));
            assert!(row.words.iter().chain(&row.marks).chain(&row.tops).all(|&word| word == 0),
                "{words} words, round {round}: the row is clear after it is read");
        }
    }
}

/// A level of one node whose right side stores references wide enough that
/// its rows span several summary words is written alike by the rows and by
/// sorting its atoms, and like the owner-set rule written out.
#[test]
fn a_wide_level_of_one_node_is_written_alike_by_rows_and_by_sorting() {
    let mut rng = Lcg::new(20260928);
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let host = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    let parent = tree.root();
    for (keys, cells, width, count) in [(3usize, 2u32, 1u32 << 13, 40usize), (50, 30, 1 << 19, 2000), (400, 5, 1 << 20, 6000)] {
        let mut pairs: Vec<ChildPair> = (0..count)
            .map(|_| ChildPair::new(
                EncodedChildRef::from_raw(rng.below(keys as u64) as u32),
                EncodedChildRef::from_raw(rng.below(u64::from(width)) as u32),
            ))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let mut f = host.clone();
        {
            let level = &mut f.levels[parent.idx()];
            level.clear();
            level.push_node(eng.limits(), &pairs).unwrap();
        }
        let left = random_remap(&eng, &mut rng, keys, cells);
        let level = &f.levels[parent.idx()];
        let (want, _) = regroup_by_definition(level, Some(&left), None);
        let want = want.first().map_or(&[][..], |cell| &cell[..]);
        let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };

        let mut by_rows = f.clone();
        let plan = RowPlan::of(&mut work, &pairs, Some(&left), None).unwrap();
        assert!(Row::tops_for(plan.words) >= 2 || width < 1 << 18, "{keys} keys over {width} columns: several top words");
        regroup_single_rows(&mut work, &mut by_rows, parent, Some(&left), None, &plan).unwrap();
        let written = &by_rows.levels[parent.idx()];
        assert_eq!(written.nodes().len(), 1);
        assert_eq!(written.pairs_vec(0), want, "{keys} keys over {width} columns: rows");

        let owned = scan_level(&mut work, level).unwrap();
        let buckets = bucket_pairs(&mut work, &owned, Side::Right, None).unwrap();
        let mut by_sorting = f.clone();
        regroup_single(&mut work, &mut by_sorting, parent, &owned, &buckets, Side::Right, &left).unwrap();
        assert_eq!(by_sorting.levels[parent.idx()].pairs_vec(0), want, "{keys} keys over {width} columns: sorting");
    }
}

/// The rows are taken when they cost no more than the atoms: always for one
/// row of a few words, never for many empty rows over few atoms; and one row
/// of rewritten sides is taken without counting the atoms.
#[test]
fn rows_are_written_when_they_cost_no_more_than_the_atoms() {
    let eng = Engine::new();
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
    let pair = |left: u32, right: u32| ChildPair::new(EncodedChildRef::from_raw(left), EncodedChildRef::from_raw(right));
    // Stored on both sides: the rows and columns are the references.
    let dense: Vec<ChildPair> = (0..64).map(|i| pair(i % 8, i / 8)).collect();
    let plan = RowPlan::of(&mut work, &dense, None, None).unwrap();
    assert_eq!((plan.rows, plan.words, plan.single, plan.most), (8, 1, None, 64));
    assert!(plan.pays);
    let sparse = [pair(0, 0), pair(1 << 20, 1 << 20)];
    let plan = RowPlan::of(&mut work, &sparse, None, None).unwrap();
    assert_eq!((plan.rows, plan.single, plan.most), ((1 << 20) + 1, None, 2));
    assert!(!plan.pays);
    let one_row = [pair(7, 0), pair(7, 1 << 12)];
    let plan = RowPlan::of(&mut work, &one_row, None, None).unwrap();
    assert_eq!(plan.single, Some(7));
    assert!(!plan.pays, "65 words cost more than 2 atoms");
    // A left side rewritten into one cell is one row, whatever it stored.
    let one_cell = Runs::all_to_first(eng.limits(), 4, 0u32).unwrap();
    let spread = [pair(0, 3), pair(3, 1), pair(2, 3)];
    let plan = RowPlan::of(&mut work, &spread, Some(&one_cell), None).unwrap();
    assert_eq!((plan.single, plan.most), (Some(0), 3));
    assert!(plan.pays);
    // With the right side rewritten too, the row's columns bound the cell.
    let wide = Runs::pack(eng.limits(), 4, &[(0, 0), (1, 99), (3, 7)], 0u32).unwrap();
    let plan = RowPlan::of(&mut work, &spread, Some(&one_cell), Some(&wide)).unwrap();
    assert_eq!((plan.single, plan.words, plan.most), (Some(0), 2, 100));
    assert!(plan.pays);
    // A row wider than the pairs are many is costed from the atoms.
    let wider = Runs::pack(eng.limits(), 4, &[(0, 0), (1, 999), (3, 7)], 0u32).unwrap();
    let plan = RowPlan::of(&mut work, &spread, Some(&one_cell), Some(&wider)).unwrap();
    assert_eq!((plan.single, plan.words, plan.most), (Some(0), 16, 3));
    assert!(!plan.pays, "16 words cost more than 3 atoms and 3 pairs");
}

/// A group is marked once when at least two rows visit it and its columns
/// fill a quarter of a row; its words hold exactly those columns.
#[test]
fn a_group_several_rows_visit_is_marked_once_when_it_fills_a_quarter_of_a_row() {
    let eng = Engine::new();
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
    // Node 0 visited by rows 0 and 1, node 1 by row 1 only, node 2 by rows 0
    // and 2.
    let left = Runs::pack(eng.limits(), 3, &[(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)], 0u32).unwrap();
    // Four words a row: node 0 has one column, node 1 and node 2 have two.
    let groups = Runs::pack(eng.limits(), 3, &[(0, 5), (1, 64), (1, 200), (2, 3), (2, 130)], 0u32).unwrap();
    let dense = dense_groups(&mut work, &groups, &left, None, 4).unwrap();
    assert_eq!(dense.get(0), &[1 << 5, 0, 0, 0], "one column fills a quarter of four words");
    assert!(dense.get(1).is_empty(), "one row visits node 1");
    assert_eq!(dense.get(2), &[1 << 3, 0, 1 << 2, 0]);
    // Through a rewritten right side, the columns are its cells.
    let right = Runs::pack(eng.limits(), 201, &[(5, 9), (5, 70), (3, 1), (130, 1)], 0u32).unwrap();
    let dense = dense_groups(&mut work, &groups, &left, Some(&right), 2).unwrap();
    assert_eq!(dense.get(0), &[1 << 9, 1 << 6]);
    assert!(dense.get(1).is_empty(), "one row visits node 1");
    assert_eq!(dense.get(2), &[1 << 1, 0], "a repeated cell is set once");
    let dense = dense_groups(&mut work, &groups, &left, Some(&right), 9).unwrap();
    assert!(dense.get(0).is_empty(), "two columns fill less than a quarter of nine words");
    assert!(dense.get(2).is_empty());
}

/// Inverting a map lists, for every item, the keys that hold it, ascending.
#[test]
fn a_map_inverted_lists_the_keys_holding_each_item() {
    let eng = Engine::new();
    let mut rng = Lcg::new(11);
    for (keys, cells) in [(1usize, 1u32), (5, 3), (40, 70), (300, 9)] {
        let map = random_remap(&eng, &mut rng, keys, cells);
        let inverse = map.transpose(eng.limits(), cells as usize).unwrap();
        assert_eq!(inverse.len(), cells as usize);
        for cell in 0..cells {
            let want: Vec<u32> = (0..keys as u32).filter(|&key| map.get(key as usize).contains(&cell)).collect();
            assert_eq!(inverse.get(cell as usize), &want[..], "{keys} keys, cell {cell}");
        }
    }
}

/// The radix sort against the standard stable sort, across lengths on both
/// sides of the insertion-sort cutoff and of the length that takes wider
/// digits, and key widths from none to all 32 bits.
#[test]
fn radix_sort_is_a_stable_sort_by_key() {
    let mut rng = Lcg::new(7);
    let eng = Engine::new();
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
    let wide = 1usize << crate::sort::RADIX_LARGE_BITS;
    for len in [0usize, 1, 2, RADIX_MIN - 1, RADIX_MIN, 1000, 5000, wide - 1, wide + 3] {
        for bits in [0u32, 1, 5, 11, 12, 23, 32] {
            let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
            let mut items: Vec<(u32, u32)> = (0..len as u32)
                .map(|i| (rng.next_u64() as u32 & mask & if rng.coin() { u32::MAX } else { 7 }, i))
                .collect();
            let mut want = items.clone();
            want.sort_by_key(|&(key, _)| key);
            sort_by_key_stable(&mut work, &mut items, bits, |&(key, _)| key).unwrap();
            assert_eq!(items, want, "{len} items, {bits}-bit keys");
        }
    }
}

/// A cell written row by row grows as it always did where its new block fits
/// the budget beside the old one, is cut to what fits where the old rule
/// would have grown it further, and refuses, unchanged, where not even the
/// least growth fits: growing copies the old block into a new one before the
/// old is freed, and the meter charges only the growth.
#[test]
fn a_row_cell_grows_only_where_its_new_block_fits_beside_the_old() {
    let pair = ChildPair { left: EncodedChildRef::from_raw(0), right: EncodedChildRef::from_raw(1) };
    let mib = 1u64 << 20;
    // A full cell of 4M pairs (32 MiB) charged to an operation, grown as row
    // 64 of 128: the rule asks for as many again, capped at `most` pairs.
    let len = 4usize << 20;
    let grow = |budget: Option<u64>, most: u64| -> (Result<(), OperationError>, usize, u64) {
        let eng = Engine::new();
        let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(budget));
        let lim = eng.limits();
        let _op = lim.begin_operation();
        let mut out = Vec::new();
        lim.reserve_exact(&mut out, len).unwrap();
        out.resize(len, pair);
        assert_eq!(out.capacity(), len);
        let result = grow_rows(lim, &mut out, 64, 128, most);
        (result, out.capacity(), lim.meters().in_flight_bytes)
    };
    let unbounded = u64::MAX;
    // No budget, and a budget with room for both blocks: doubled.
    assert_eq!(grow(None, unbounded).1, 2 * len);
    assert_eq!(grow(Some(1024 * mib), unbounded).1, 2 * len);
    // Room for the new block only if it holds 6M pairs: cut to that, where
    // the old rule, charging the 32 MiB of growth against 48 MiB of room,
    // doubled it.
    let (cut, cap, in_flight) = grow(Some(32 * mib + 48 * mib), unbounded);
    assert_eq!(cut, Ok(()));
    assert_eq!(cap, 6 << 20);
    assert_eq!(in_flight, 48 * mib);
    // Room for 4.5M pairs: the new block would hold only 0.5M more, below
    // the least growth, so the cell refuses and keeps its block and charge.
    let (refused, cap, in_flight) = grow(Some(32 * mib + 36 * mib), unbounded);
    assert_eq!(refused, Err(OperationError::OverBudget));
    assert_eq!(cap, len);
    assert_eq!(in_flight, 32 * mib);
    // A cell 100 pairs from its most grows by those 100 where 120 more fit,
    // though that is below the least growth.
    let (rest, cap, _) = grow(Some(32 * mib + (len as u64 + 120) * 8), len as u64 + 100);
    assert_eq!(rest, Ok(()));
    assert_eq!(cap, len + 100);
}
