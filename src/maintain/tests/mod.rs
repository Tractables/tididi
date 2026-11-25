use crate::vtree::Home;
use crate::test_helpers::bare_roots;
use super::*;

use rustc_hash::FxHashMap;

use crate::test_helpers::check::check_determinism;
use crate::test_helpers::{assert_canonical, assert_same_shape, assignment, compile_clauses, same_as_stored, sweep_shapes, test_cases, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::VarId;

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

fn apply_update(batch: &mut Maintenance<'_>, model: &[i32], insert: bool) -> Result<(), OperationError> {
    if insert { batch.insert_model(model) } else { batch.remove_model(model) }
}

/// On implicit levels and on stored ones alike ([`same_as_stored`]).
#[test]
fn an_edit_and_a_rebuild_reach_the_same_diagram() {
    same_as_stored(|| an_edit_and_a_rebuild(false));
}

#[test]
fn an_edit_and_a_rebuild_reach_the_same_diagram_on_a_home() {
    an_edit_and_a_rebuild(true);
}

#[test]
#[ignore = "on a home an implicit level records another arena capacity than the stored route builds"]
fn an_edit_and_a_rebuild_on_a_home_hold_what_the_stored_route_holds() {
    same_as_stored(|| an_edit_and_a_rebuild(true));
}

/// The cases of [`an_edit_and_a_rebuild_reach_the_same_diagram`], on homes
/// or on roots that are not homes: the edited diagrams, minimized.
fn an_edit_and_a_rebuild(homes: bool) -> Vec<Tdd> {
    let mut out = Vec::new();
    // Both routes over every assignment of every small case. `check_determinism`
    // is quadratic in the level widths, so the sweep stops at five variables.
    let mut edits = [0u32; 2];
    for (num_vars, clauses) in test_cases() {
        if num_vars > 5 { continue; }
        let shapes = match homes {
            true => sweep_shapes(num_vars).into_iter().map(|(shape, home)| (shape, Arc::clone(&home))).collect(),
            false => bare_roots(sweep_shapes(num_vars)),
        };
        for (shape, vtree) in shapes {
            let mut f = compile_clauses(&vtree, &clauses);
            f.minimize().expect("minimizing fits");
            for bits in 0..(1u64 << num_vars) {
                let model = assignment(num_vars, bits);
                for insert in [true, false] {
                    let mut got = f.clone();
                    let mut batch = got.maintain().expect("the index fits");
                    apply_update(&mut batch, &model, insert).expect("the update fits");
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
                    out.push(got);
                }
            }
        }
    }
    assert!(edits[0] > 0 && edits[1] > 0,
        "the edit route was not exercised in both directions: {edits:?}");
    out
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
    let vtree = Home::new(Vtree::balanced(3));
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
    let vtree = Home::new(Vtree::balanced(4));
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
    let vtree = Home::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2, 3]).expect("the clause is over the vtree");
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
    let vtree = Home::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2, 3]).expect("the clause is over the vtree");
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
    let vtree = Home::new(Vtree::balanced(4));
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
    let vtree = Home::new(Vtree::balanced(3));
    let mut f = Tdd::one(&vtree);
    assert_eq!(f.insert_model([1, 9]).err(), Some(OperationError::VariableNotInVtree(VarId(9))));
    assert_eq!(f.remove_model([1, 0]).err(), Some(OperationError::InvalidLiteral(0)));
}

#[test]
fn a_one_variable_vtree_goes_through_the_rebuild() {
    let vtree = Home::new(Vtree::balanced(1));
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
    let vtree = Home::new(Vtree::balanced(5));
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
    let vtree = Home::new(Vtree::balanced(4));
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
                eng.refuse_nth_reserve(cut);
                let result = apply_update(&mut batch, &model, insert);
                let pending = eng.limits().refusal_pending();
                eng.limits().grant_every_reserve();
                match result {
                    Err(error) => {
                        assert_eq!(error, OperationError::OverBudget, "cut {cut}");
                        assert!(batch.diagram().equivalent(&original).unwrap(), "cut {cut} changed the function");
                        refusals += 1;
                    }
                    Ok(()) => finished = pending,
                }
                // The same batch must be usable after refusal, including when
                // a new unreachable node was appended before the refusal.
                apply_update(&mut batch, &model, insert).unwrap();
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
    let vtree = Home::new(Vtree::balanced_over(&[VarId(2), VarId(5), VarId(9)]).unwrap());
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
    let vtree = Home::new(Vtree::balanced(3));
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
    let vtree = Home::new(Vtree::balanced(3));
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
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
    use crate::test_helpers::rat;
    let table = vec![LiteralWeights { negative: rat(2, 1), positive: rat(2, 1) }; 3];
    let vtree = Home::new(Vtree::balanced(3)
        .with_weights(Weights::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)).unwrap());
    for (model, insert) in [(vec![-1], true), (vec![1, 2, 3], false)] {
        let original = Tdd::cube(&vtree, [1, 2, 3]).unwrap();
        let mut succeeded = false;
        for cut in 0..400 {
            let eng = Engine::new();
            let mut f = original.clone();
            {
                let mut batch = eng.maintain(&mut f).unwrap();
                eng.refuse_nth_reserve(cut);
                let outcome = apply_update(&mut batch, &model, insert);
                let pending = eng.limits().refusal_pending();
                eng.limits().grant_every_reserve();
                match outcome {
                    Err(e) => {
                        assert_eq!(e, OperationError::OverBudget);
                        assert!(batch.diagram().equivalent(&original).unwrap());
                        assert_eq!(batch.diagram().weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(8, 1));
                    }
                    Ok(()) => succeeded = pending,
                }
                apply_update(&mut batch, &model, insert).unwrap();
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
    let vtree = Home::new(Vtree::balanced(8));
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
    let vtree = Home::new(Vtree::balanced(6));
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
    // A list is deduplicated once it holds more than twice the levels.
    let bound = 2 * vtree.num_nodes();
    for pass in [Pass::Contract, Pass::LeafContract, Pass::ContentTwin] {
        let len = f.dirty.levels(pass).len();
        assert!(len <= bound, "{pass:?} holds {len} entries over {bound} levels");
    }
    f.minimize().expect("minimizing fits");
    assert_canonical(&f);
}

/// On a projection of a home, an update may name a home variable the
/// diagram has no leaf for, and a rebuild or the false exit moves the diagram
/// to another projection: the batch reads the moved diagram afterwards.
#[test]
fn a_batch_on_a_projection_follows_the_diagram_across_projections() {
    use crate::vtree::Home;
    let eng = Engine::new();
    let home = Home::new(Vtree::join(&Vtree::join(&Vtree::leaf(VarId(1)), &Vtree::leaf(VarId(2))).unwrap(), &Vtree::leaf(VarId(3))).unwrap());
    let all = Arc::clone(&home);
    let on_home = |f: &Tdd| crate::test_helpers::embedded(f, &all, |v| v).unwrap().0;
    let mut f = eng.literal(&all, 2).unwrap();
    f.trim(&home).unwrap();
    let mut expected = on_home(&f);
    let steps: [(&[i32], bool); 6] = [(&[1, 3], true), (&[1, -2, 3], true), (&[2], false), (&[1, 2, 3], true), (&[-1], true), (&[3], false)];
    let mut batch = eng.maintain(&mut f).unwrap();
    for (model, insert) in steps {
        apply_update(&mut batch, model, insert).unwrap();
        expected = rebuilt(&all, &expected, model, insert);
        let got = batch.diagram();
        assert!(got.vtree().is_on(&home) || got.vtree().num_nodes() == 0, "{model:?}");
        assert!(eng.equivalent(&on_home(got), &expected).unwrap(), "{model:?} {insert}");
    }
}

/// The same update by the operation a rebuild stands in for, minimized.
fn by_operations(f: &Tdd, model: &[i32], insert: bool) -> Tdd {
    let mut out = if insert {
        f.clone().or_cube(model).unwrap()
    } else {
        let clause: Vec<i32> = model.iter().map(|lit| -lit).collect();
        f.clone().and_clause(&clause).unwrap()
    };
    out.minimize().unwrap();
    out
}

/// The batch's diagram, minimized, sits on the projection of its support,
/// names no leaf true, and equals the same updates made by `or_cube` and
/// `and_clause` node for node.
fn assert_matches_operations(got: &Tdd, want: &Tdd, what: &str) {
    use crate::test_helpers::check::check_trimmed;
    let mut got = got.clone();
    got.minimize().unwrap();
    check_trimmed(&got, true).unwrap_or_else(|e| panic!("{what}: {e}"));
    let leaves = |t: &Tdd| -> Vec<VarId> {
        let v = t.vtree();
        v.bottomup().filter(|&n| v.node(n).is_leaf()).map(|n| v.leaf_var(n)).collect()
    };
    assert_eq!(leaves(&got), leaves(want), "{what}: leaves differ");
    assert_eq!(got.is_zero(), want.is_zero(), "{what}");
    if got.vtree().num_nodes() > 0 {
        assert_same_shape(&got, want, what);
    } else {
        assert_eq!(got.is_one(), want.is_one(), "{what}");
    }
}

/// Both constants on a home, each on its empty projection, take both
/// updates: a complete model, a partial one, an empty one and a
/// contradictory one, each through a fresh batch.
#[test]
fn constants_on_a_home_take_both_updates() {
    use crate::vtree::Home;
    let home = Home::new(Vtree::balanced(4));
    let models: [&[i32]; 4] = [&[1, -2, 3, 4], &[-2, 4], &[], &[1, -1, 3]];
    for constant in [Tdd::zero(&home), Tdd::one(&home)] {
        assert_eq!(constant.vtree().num_nodes(), 0);
        for model in models {
            for insert in [true, false] {
                let mut f = constant.clone();
                apply_update(&mut f.maintain().unwrap(), model, insert).unwrap();
                let what = format!("{} {model:?} insert {insert}", constant.is_one());
                assert_matches_operations(&f, &by_operations(&constant, model, insert), &what);
            }
        }
    }
}

/// Random batches on homes of at most eight variables, starting from a
/// diagram on the projection of its support, against the same updates made
/// one by one with `or_cube` and `and_clause`.
#[test]
fn random_batches_on_a_home_match_the_clause_and_cube_entries() {
    use crate::vtree::Home;
    let mut rng = Lcg::new(0x3a1f);
    let mut rebuilds = 0u64;
    let mut updates = 0u64;
    let mut moved = 0u64;
    for seed in 0..500u64 {
        let n = 2 + rng.below(7) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let mut f = crate::test_helpers::rand_conj(&home, n, 3, 3, false, &mut rng);
        f.minimize().unwrap();
        let mut want = f.clone();
        let start = Arc::clone(f.vtree());
        {
            let mut batch = f.maintain().unwrap();
            for step in 0..1 + rng.below(6) {
                let mut model = Vec::new();
                for v in 1..=n as i32 {
                    if rng.below(4) != 0 {
                        model.push(if rng.coin() { v } else { -v });
                    }
                }
                if rng.below(8) == 0 {
                    let v = 1 + rng.below(u64::from(n)) as i32;
                    model.extend([v, -v]);
                }
                let insert = rng.coin();
                apply_update(&mut batch, &model, insert).unwrap();
                want = by_operations(&want, &model, insert);
                assert_matches_operations(batch.diagram(), &want, &format!("seed {seed} step {step}: {model:?} insert {insert}"));
                updates += 1;
            }
            rebuilds += batch.rebuilds();
            moved += u64::from(!Arc::ptr_eq(batch.diagram().vtree(), &start));
        }
    }
    assert!(rebuilds > 0 && rebuilds < updates, "{rebuilds} rebuilds of {updates} updates");
    assert!(moved > 0, "no batch left its first projection");
}

/// After a rebuild that moved the diagram to a wider projection, an edit on
/// the new vtree takes the edit route.
#[test]
fn an_edit_after_a_rebuild_that_moved_the_vtree_is_an_edit() {
    use crate::vtree::Home;
    let home = Home::new(Vtree::balanced(3));
    let mut f = Tdd::cube(&home, [1, 2]).unwrap();
    assert_eq!(f.vtree().num_leaves(), 2);
    let first = Arc::clone(f.vtree());
    let mut batch = f.maintain().unwrap();
    batch.remove_model([1, 2, 3]).unwrap();
    assert_eq!(batch.rebuilds(), 1);
    assert!(!Arc::ptr_eq(batch.diagram().vtree(), &first));
    assert_eq!(batch.diagram().vtree().num_leaves(), 3);
    // The rebuild leaves unreachable nodes at (x2 x3), one of them with two
    // pairs; this assignment's path meets only singletons.
    batch.insert_model([-1, 2, 3]).unwrap();
    assert_eq!(batch.rebuilds(), 1, "the second update is an edit");
    let want = by_operations(&by_operations(&Tdd::cube(&home, [1, 2]).unwrap(), &[1, 2, 3], false), &[-1, 2, 3], true);
    assert_matches_operations(batch.diagram(), &want, "remove then insert");
}

/// Removing the last model on a weighted home leaves false on the empty
/// projection, which reads the home's weights, and the batch goes on from
/// there.
#[test]
fn the_false_exit_on_a_weighted_home_keeps_its_weights() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
    use crate::test_helpers::rat;
    use crate::vtree::Home;
    let table = vec![LiteralWeights { negative: rat(1, 1), positive: rat(3, 1) }; 3];
    let home = Home::weighted(Vtree::balanced(3), Weights::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)).unwrap();
    let mut f = Tdd::cube(&home, [1, -2, 3]).unwrap();
    let mut batch = f.maintain().unwrap();
    batch.remove_model([1, -2, 3]).unwrap();
    assert_eq!(batch.rebuilds(), 0, "the removal is an edit");
    let zero = batch.diagram();
    assert!(zero.is_zero());
    assert_eq!(zero.vtree().num_nodes(), 0);
    assert!(zero.weights().is_some());
    batch.insert_model([1, 2, 3]).unwrap();
    f.minimize().unwrap();
    assert_eq!(f.weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(27, 1));
}
