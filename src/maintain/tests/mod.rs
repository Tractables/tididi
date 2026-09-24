use super::*;

use rustc_hash::FxHashMap;

use crate::test_helpers::check::check_determinism;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses, test_cases, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::VarId;

/// The assignment whose `v`-th bit is `bits >> (v - 1)`, as signed literals.
fn assignment(num_vars: u32, bits: u64) -> Vec<i32> {
    (1..=num_vars as i32)
        .map(|v| if bits >> (v - 1) & 1 == 1 { v } else { -v })
        .collect()
}

/// The three vtree shapes the sweeps take. The oracle compiles a second
/// diagram per cell, so the full list is more than the coverage is worth.
fn sweep_shapes(num_vars: u32) -> Vec<(&'static str, Arc<Vtree>)> {
    vtree_shapes(num_vars)
        .into_iter()
        .filter(|(shape, _)| matches!(*shape, "balanced" | "linear" | "random(42)"))
        .collect()
}

/// What the update stands in for, built by the general operations: `f ∨ M`
/// through a compiled cube diagram, and `f ∧ ¬M` through the clause walk.
fn rebuilt(vtree: &Arc<Vtree>, f: &Tdd, model: &[i32], insert: bool) -> Tdd {
    let mut out = if insert {
        let cube = Tdd::cube(vtree, model).expect("the cube is over the vtree");
        crate::or(f.clone(), cube).expect("the disjunction fits")
    } else {
        let clause: Vec<i32> = model.iter().map(|lit| -lit).collect();
        f.clone().and_clause(&clause).expect("the conjunction fits")
    };
    out.minimize().expect("minimizing fits");
    out
}

#[test]
fn an_edit_and_a_rebuild_reach_the_same_diagram() {
    // Both routes over every assignment of every small case. `check_determinism`
    // is quadratic in the level widths, so the sweep stops at five variables.
    let mut edits = [0u32; 2];
    for (num_vars, clauses) in test_cases() {
        if num_vars > 5 { continue; }
        for (shape, vtree) in sweep_shapes(num_vars) {
            let mut f = compile_clauses(&vtree, &clauses);
            f.minimize().expect("minimizing fits");
            for bits in 0..(1u64 << num_vars) {
                let model = assignment(num_vars, bits);
                for insert in [true, false] {
                    let mut got = f.clone();
                    let mut batch = got.maintain().expect("the index fits");
                    if insert { batch.insert_model(&model) } else { batch.remove_model(&model) }
                        .expect("the update fits");
                    let took_the_edit = batch.rebuilds() == 0;
                    drop(batch);
                    if took_the_edit {
                        edits[usize::from(insert)] += 1;
                        check_determinism(&got).unwrap_or_else(|e| {
                            panic!("{shape}: {model:?} left a level that is not a partition: {e}")
                        });
                    }
                    got.minimize().expect("minimizing fits");
                    assert_canonical(&got);
                    let what = if insert { "insert" } else { "remove" };
                    assert_same_shape(&got, &rebuilt(&vtree, &f, &model, insert),
                        &format!("{shape}: {what} {model:?}"));
                }
            }
        }
    }
    assert!(edits[0] > 0 && edits[1] > 0,
        "the edit route was not exercised in both directions: {edits:?}");
}

#[test]
fn a_batch_of_row_updates_matches_rebuilding_the_relation() {
    // Two two-bit attributes: the shape a dictionary-coded relation compiles to.
    let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    let row = |a: u64, b: u64| (a >> 1) | ((a & 1) << 1) | ((b >> 1) << 2) | ((b & 1) << 3);
    let mut rng = Lcg::new(11);
    let mut edits = 0u64;
    for (shape, vtree) in vtree_shapes(4) {
        for round in 0..16u32 {
            let start = 1 + rng.below(6) as usize;
            let mut rows: Vec<u64> = (0..start).map(|_| row(rng.below(4), rng.below(4))).collect();
            rows.sort_unstable();
            rows.dedup();
            let mut f = Tdd::from_models(&vtree, &vars, &rows).expect("the rows fit");

            let mut updates = 0u64;
            {
                let mut batch = f.maintain().expect("the index fits");
                for _ in 0..8 {
                    let r = row(rng.below(4), rng.below(4));
                    if rng.coin() {
                        batch.insert_model(assignment(4, r)).expect("the insert fits");
                        if let Err(at) = rows.binary_search(&r) { rows.insert(at, r); }
                    } else {
                        batch.remove_model(assignment(4, r)).expect("the remove fits");
                        if let Ok(at) = rows.binary_search(&r) { rows.remove(at); }
                    }
                    updates += 1;
                }
                edits += updates - batch.rebuilds();
            }

            check_determinism(&f)
                .unwrap_or_else(|e| panic!("{shape} round {round}: a level is not a partition: {e}"));
            f.minimize().expect("minimizing fits");
            assert_canonical(&f);
            let want = Tdd::from_models(&vtree, &vars, &rows).expect("the rows fit");
            assert_same_shape(&f, &want, &format!("{shape} round {round}: {} rows", rows.len()));
        }
    }
    assert!(edits > 0, "no update of the batch took the edit route");
}

#[test]
fn removing_the_last_model_leaves_the_false_diagram() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::cube(&vtree, [1, -2, 3]).expect("the cube is over the vtree");
    f.remove_model([1, -2, 3]).expect("the remove fits");
    assert!(f.is_zero(), "the diagram kept a model");
    assert_canonical(&f);

    // And the false diagram takes the assignment back.
    f.insert_model([1, -2, 3]).expect("the insert fits");
    f.minimize().expect("minimizing fits");
    assert_same_shape(&f, &Tdd::cube(&vtree, [1, -2, 3]).expect("the cube is over the vtree"),
        "false ∨ the assignment");
}

#[test]
fn an_update_that_changes_nothing_leaves_the_diagram_alone() {
    let vtree = Arc::new(Vtree::balanced(4));
    let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    let mut f = Tdd::from_models(&vtree, &vars, &[0b0000, 0b0101, 0b1111]).expect("the rows fit");
    let before = f.clone();
    {
        let mut batch = f.maintain().expect("the index fits");
        // A row the relation already has, and one it does not have.
        batch.insert_model(assignment(4, 0b0101)).expect("the insert fits");
        batch.remove_model(assignment(4, 0b0011)).expect("the remove fits");
        assert_eq!(batch.rebuilds(), 0, "both updates are decided by the probe");
    }
    assert_same_shape(&f, &before, "an update that changes nothing");
}

#[test]
fn a_free_variable_in_the_diagram_closes_the_edit_route() {
    // `One` is `Pos` and `Neg` together, so a leaf named through it has no
    // node for a single value and the probe cannot resolve the path.
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2]).expect("the clause is over the vtree");
    f.minimize().expect("minimizing fits");
    let before = f.clone();
    {
        let mut batch = f.maintain().expect("the index fits");
        batch.insert_model([-1, -2, 3]).expect("the insert fits");
        assert_eq!(batch.rebuilds(), 1, "variable 3 is free, so the rebuild answers");
    }
    f.minimize().expect("minimizing fits");
    assert_canonical(&f);
    assert_same_shape(&f, &rebuilt(&vtree, &before, &[-1, -2, 3], true), "a free-variable diagram");
}

#[test]
fn a_partial_or_contradictory_assignment_is_answered_by_the_rebuild() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2]).expect("the clause is over the vtree");
    f.minimize().expect("minimizing fits");
    let before = f.clone();

    {
        let mut batch = f.maintain().expect("the index fits");
        // Both polarities name no assignment, so neither update changes anything.
        batch.insert_model([1, -1, 2, 3]).expect("the insert fits");
        batch.remove_model([1, -1, 2, 3]).expect("the remove fits");
        assert_eq!(batch.rebuilds(), 0, "a false cube is decided without a rebuild");
        // A free variable makes a cube, not an assignment: the rebuild answers.
        batch.insert_model([-1, -2]).expect("the insert fits");
        assert_eq!(batch.rebuilds(), 1);
    }

    f.minimize().expect("minimizing fits");
    assert_canonical(&f);
    assert_same_shape(&f, &rebuilt(&vtree, &before, &[-1, -2], true), "a partial cube");
}

#[test]
fn a_run_of_rebuilds_latches_the_batch_onto_the_rebuild() {
    // An index a batch cannot use still costs a pass over the diagram to
    // build, so a run of rebuilds stops the batch trying.
    let vtree = Arc::new(Vtree::balanced(4));
    let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    let mut f = Tdd::from_models(&vtree, &vars, &[0b0000, 0b0101, 0b1111]).expect("the rows fit");
    let mut want = f.clone();
    {
        let mut batch = f.maintain().expect("the index fits");
        // A complete assignment: the edit route is open.
        batch.insert_model(assignment(4, 0b0001)).expect("the insert fits");
        assert_eq!(batch.rebuilds(), 0);
        want = want.or_cube(assignment(4, 0b0001)).expect("the disjunction fits");

        // Partial cubes, which only the rebuild can answer.
        for _ in 0..GIVE_UP {
            batch.insert_model([1, 2]).expect("the insert fits");
            want = want.or_cube([1, 2]).expect("the disjunction fits");
        }
        assert_eq!(batch.rebuilds(), u64::from(GIVE_UP));

        // The batch has stopped indexing, so a complete assignment the edit
        // route could have taken goes to the rebuild too.
        batch.insert_model(assignment(4, 0b0010)).expect("the insert fits");
        want = want.or_cube(assignment(4, 0b0010)).expect("the disjunction fits");
        assert_eq!(batch.rebuilds(), u64::from(GIVE_UP) + 1);
        assert!(batch.index.is_none(), "the batch kept an index it stopped using");
    }
    f.minimize().expect("minimizing fits");
    want.minimize().expect("minimizing fits");
    assert_canonical(&f);
    assert_same_shape(&f, &want, "a latched batch");
}

#[test]
fn an_absent_variable_and_a_zero_literal_are_refused() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::one(&vtree);
    assert_eq!(f.insert_model([1, 9]).err(), Some(OperationError::VariableNotInVtree(VarId(9))));
    assert_eq!(f.remove_model([1, 0]).err(), Some(OperationError::InvalidLiteral(0)));
}

#[test]
fn a_one_variable_vtree_goes_through_the_rebuild() {
    let vtree = Arc::new(Vtree::balanced(1));
    let mut f = Tdd::zero(&vtree);
    f.insert_model([1]).expect("the insert fits");
    assert_eq!(f.model_count().expect("counting fits"), 1u32.into());
    f.insert_model([-1]).expect("the insert fits");
    f.minimize().expect("minimizing fits");
    assert_eq!(f.model_count().expect("counting fits"), 2u32.into());
    assert_canonical(&f);
}

#[test]
fn an_edited_node_outgrows_its_arena_range() {
    // Every assignment over five variables, one at a time into the false
    // diagram: the output node's pair list outgrows the range it was encoded
    // in and has to move, which is the one place an edit is not constant work.
    let vtree = Arc::new(Vtree::balanced(5));
    let mut f = Tdd::zero(&vtree);
    {
        let mut batch = f.maintain().expect("the index fits");
        for bits in 0..32u64 {
            batch.insert_model(assignment(5, bits)).expect("the insert fits");
        }
        // Only the first update, into the false diagram, takes the rebuild.
        assert_eq!(batch.rebuilds(), 1);
    }
    check_determinism(&f).unwrap_or_else(|e| panic!("a level is not a partition: {e}"));
    assert_eq!(f.model_count().expect("counting fits"), 32u32.into());

    let mut every = f.clone();
    every.minimize().expect("minimizing fits");
    assert_canonical(&every);
    assert_same_shape(&every, &Tdd::one(&vtree), "every assignment");

    // And back out again, one at a time, off the diagram the edits built.
    {
        let mut batch = f.maintain().expect("the index fits");
        for bits in 0..31u64 {
            batch.remove_model(assignment(5, bits)).expect("the remove fits");
        }
        assert_eq!(batch.rebuilds(), 0);
    }
    check_determinism(&f).unwrap_or_else(|e| panic!("a level is not a partition: {e}"));
    f.minimize().expect("minimizing fits");
    assert_canonical(&f);
    assert_same_shape(&f, &Tdd::cube(&vtree, assignment(5, 31)).expect("the cube is over the vtree"),
        "the one assignment left");
}

#[test]
fn every_refused_update_preserves_the_function_and_can_be_retried() {
    let vtree = Arc::new(Vtree::balanced(4));
    let vars: Vec<_> = (1..=4).map(VarId).collect();
    for (model, insert) in [
        (vec![1, -2, -3, -4], true),
        (vec![1], true),
        (vec![-1, -2, -3, -4], false),
        (vec![1], false),
    ] {
        let original = Tdd::from_models(&vtree, &vars, &[0, 5, 15]).unwrap();
        let want = rebuilt(&vtree, &original, &model, insert);
        let mut refusals = 0;
        let mut finished = false;
        for cut in 0..600 {
            let eng = Engine::new();
            let mut got = original.clone();
            {
                let mut batch = eng.maintain(&mut got).unwrap();
                eng.limits().refuse_nth_reserve(cut);
                let result = if insert { batch.insert_model(&model) } else { batch.remove_model(&model) };
                eng.limits().grant_every_reserve();
                match result {
                    Err(error) => {
                        assert_eq!(error, OperationError::OverBudget, "cut {cut}");
                        assert!(batch.diagram().equivalent(&original).unwrap(), "cut {cut} changed the function");
                        refusals += 1;
                    }
                    Ok(()) => finished = true,
                }
                // The same batch must be usable after refusal, including when
                // a new unreachable node was appended before the refusal.
                if insert { batch.insert_model(&model) } else { batch.remove_model(&model) }.unwrap();
            }
            got.minimize().unwrap();
            assert_canonical(&got);
            assert_same_shape(&got, &want, &format!("retry after cut {cut}"));
            if finished { break; }
        }
        assert!(finished && refusals > 0, "the sweep must reach success and a refusal");
    }
}

#[test]
fn sparse_ids_still_take_the_single_assignment_edit() {
    let vtree = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(5), VarId(9)]).unwrap());
    let vars = [VarId(2), VarId(5), VarId(9)];
    let mut f = Tdd::from_models(&vtree, &vars, &[0, 7]).unwrap();
    {
        let mut batch = f.maintain().unwrap();
        batch.insert_model([2, -5, -9]).unwrap();
        batch.remove_model([-2, -5, -9]).unwrap();
        assert_eq!(batch.rebuilds(), 0);
    }
    f.minimize().unwrap();
    assert_canonical(&f);
    let want = Tdd::from_models(&vtree, &vars, &[1, 7]).unwrap();
    assert_same_shape(&f, &want, "sparse variable IDs");
}

#[test]
fn empty_cubes_add_or_remove_every_assignment() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::cube(&vtree, [1, 2, 3]).unwrap();
    f.insert_model([] as [i32; 0]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 8u32.into());
    f.remove_model([] as [i32; 0]).unwrap();
    assert_canonical(&f);
    assert!(f.is_zero());
}

#[test]
fn maintenance_uses_the_explicit_engines_limits() {
    use crate::limits::LimitConfig;
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::cube(&vtree, [1, 2, 3]).unwrap();
    let eng = Engine::new();
    let zero_budget = LimitConfig::none().with_memory_budget_bytes(Some(0));
    {
        let _scope = eng.limits().scope(zero_budget.clone());
        assert_eq!(eng.maintain(&mut f).unwrap_err(), OperationError::OverBudget);
    }
    {
        let mut batch = eng.maintain(&mut f).unwrap();
        let scope = eng.limits().scope(zero_budget);
        assert_eq!(batch.insert_model([-1, 2, 3]).unwrap_err(), OperationError::OverBudget);
        drop(scope);
        batch.insert_model([-1, 2, 3]).unwrap();
    }
    f.minimize().unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 2u32.into());
}

#[test]
fn weighted_updates_preserve_weights_on_refusal_and_on_the_last_removal() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use crate::test_helpers::rat;
    let vtree = Arc::new(Vtree::balanced(3));
    for (model, insert) in [(vec![-1], true), (vec![1, 2, 3], false)] {
        let mut original = Tdd::cube(&vtree, [1, 2, 3]).unwrap();
        let table = vec![LiteralWeights { negative: rat(2, 1), positive: rat(2, 1) }; 3];
        original.set_weights(WeightStore::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)).unwrap();
        let mut succeeded = false;
        for cut in 0..400 {
            let eng = Engine::new();
            let mut f = original.clone();
            {
                let mut batch = eng.maintain(&mut f).unwrap();
                eng.limits().refuse_nth_reserve(cut);
                let outcome = if insert { batch.insert_model(&model) } else { batch.remove_model(&model) };
                eng.limits().grant_every_reserve();
                match outcome {
                    Err(e) => {
                        assert_eq!(e, OperationError::OverBudget);
                        assert!(batch.diagram().equivalent(&original).unwrap());
                        assert_eq!(batch.diagram().weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(8, 1));
                    }
                    Ok(()) => succeeded = true,
                }
                if insert { batch.insert_model(&model) } else { batch.remove_model(&model) }.unwrap();
            }
            f.minimize().unwrap();
            assert_canonical(&f);
            assert_eq!(f.model_count().unwrap(), if insert { 5u32 } else { 0 }.into());
            assert_eq!(f.weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(if insert { 40 } else { 0 }, 1));
            if succeeded { break; }
        }
        assert!(succeeded);
    }
}

#[test]
fn the_pair_index_charges_one_entry_per_pair_of_every_node() {
    // A cube's internal levels each hold one node with one pair, stored inline
    // and not in the pair arena, so the owner maps' charge must follow the
    // pair count of the nodes, not the arena length.
    let vtree = Arc::new(Vtree::balanced(8));
    let tdd = Tdd::cube(&vtree, assignment(8, 0b1010_0110)).unwrap();
    let live_pairs: usize = vtree.internal_bottomup().map(|(t, _, _)| tdd.level(t).live_pairs()).sum();
    assert_eq!(live_pairs, 7);
    let eng = Engine::new();
    let index = Index::build(&eng, &tdd).unwrap();
    assert_eq!(index.owners.iter().map(FxHashMap::len).sum::<usize>(), live_pairs);
    let n = vtree.num_nodes();
    let fixed = n * (std::mem::size_of::<FxHashMap<(u32, u32), u32>>() + std::mem::size_of::<Vec<bool>>());
    let per_entry = std::mem::size_of::<((u32, u32), u32)>() + 1;
    assert!(
        eng.limits().meters().in_flight_bytes as usize >= fixed + live_pairs * per_entry,
        "the owner maps grew past their charged reservation"
    );
}

#[test]
fn a_long_run_of_edits_keeps_the_reduction_worklists_bounded() {
    use crate::diagram::Pass;
    let vtree = Arc::new(Vtree::balanced(6));
    let vars: Vec<VarId> = (1..=6).map(VarId).collect();
    let mut f = Tdd::from_models(&vtree, &vars, &[0b000000]).expect("the rows fit");
    let mut rng = Lcg::new(5);
    {
        let mut batch = f.maintain().expect("the index fits");
        for _ in 0..1000 {
            let model = assignment(6, rng.below(64));
            if rng.coin() {
                batch.insert_model(&model).expect("the insert fits");
            } else {
                batch.remove_model(&model).expect("the remove fits");
            }
        }
    }
    let bound = vtree.num_nodes();
    for pass in [Pass::Contract, Pass::LeafContract, Pass::ContentTwin] {
        let len = f.dirty.levels(pass).len();
        assert!(len <= bound, "{pass:?} holds {len} entries over {bound} levels");
    }
    f.minimize().expect("minimizing fits");
    assert_canonical(&f);
}
