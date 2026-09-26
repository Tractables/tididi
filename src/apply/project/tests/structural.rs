use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, compile_clauses, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};

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
    for node in 0..level.nodes.len() as u32 {
        for pair in level.pairs_of_idx(node as usize) {
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
    let remap = (0..level.nodes.len() as u32)
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
                if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes.len() }
            };
            for &parent in vtree.bottomup_slice() {
                let level = &f.levels[parent.idx()];
                if vtree.node(parent).is_leaf() || level.nodes.is_empty() { continue; }
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
                        (0..written.nodes.len()).map(|cell| written.pairs_of_idx(cell).to_vec()).collect();
                    assert_eq!(got, cells, "{shape}, level {parent:?}, fanned {fanned:?}: cells");
                    let identity: Vec<Vec<u32>> = (0..level.nodes.len() as u32).map(|node| vec![node]).collect();
                    match remap {
                        Some(remap) => {
                            let got: Vec<Vec<u32>> = (0..level.nodes.len()).map(|node| remap.get(node).to_vec()).collect();
                            assert_eq!(got, fan_out, "{shape}, level {parent:?}, fanned {fanned:?}: fan-out");
                            assert_ne!(fan_out, identity, "{shape}, level {parent:?}: an unchanged level returns no map");
                        }
                        None => assert_eq!(fan_out, identity, "{shape}, level {parent:?}, fanned {fanned:?}: no map"),
                    }
                    if level.nodes.len() == 1 { single += 1 } else { several += 1 }
                }
            }
        }
    }
    assert!(single > 100 && several > 1000, "levels of one node {single}, of several {several}");
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
#[test]
fn a_level_of_one_node_is_written_alike_by_rows_and_by_sorting() {
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
                if vtree.node(child).is_leaf() { LEAF_WIDTH } else { f.levels[child.idx()].nodes.len() }
            };
            for &parent in vtree.bottomup_slice() {
                let level = &f.levels[parent.idx()];
                if vtree.node(parent).is_leaf() || level.nodes.len() != 1 { continue; }
                let pairs = level.pairs_of_idx(0);
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
                    assert_eq!(written.nodes.len(), 1, "{shape}, level {parent:?}, fanned {fanned:?}: rows");
                    assert_eq!(written.pairs_of_idx(0), want, "{shape}, level {parent:?}, fanned {fanned:?}: rows");

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
                        assert_eq!(sorted.pairs_of_idx(0), written.pairs_of_idx(0), "{shape}, level {parent:?}, filed by {by:?}");
                    }
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
/// sides of the insertion-sort cutoff and key widths from none to all 32 bits.
#[test]
fn radix_sort_is_a_stable_sort_by_key() {
    let mut rng = Lcg::new(7);
    let eng = Engine::new();
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate(), emitted: 0 };
    for len in [0usize, 1, 2, RADIX_MIN - 1, RADIX_MIN, 1000, 5000] {
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
