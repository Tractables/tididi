use crate::engine::Engine;
use std::sync::Arc;

use crate::reduce::minimize;
use crate::query::model_count;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeError};
use crate::test_helpers::assert_canonical;

fn count(t: &Tdd) -> u64 {
    model_count(t).try_into().expect("small count")
}

#[test]
fn graft_counts_the_product_times_two_per_spine_var() {
    let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
    let b = Arc::new(Vtree::linear_from_order(&[VarId(3), VarId(2)]));
    let f = Tdd::clause(&a, [1, 2]); // 3 models over {x1, x2}
    let g = Tdd::clause(&b, [3, -4]) & Tdd::clause(&b, [4]); // x3 ∧ x4: 1 model
    assert_eq!((count(&f), count(&g)), (3, 1));

    let fg = Tdd::graft(vec![f.clone(), g.clone()], &[VarId(4), VarId(5)]).unwrap();
    assert_eq!(fg.vtree.num_vars(), 6);
    assert_canonical(&fg);
    assert_eq!(count(&fg), 3 * 4);

    // Already canonical: minimize changes nothing.
    let mut m = fg.clone();
    minimize(&mut m);
    assert_canonical(&m);
    assert_eq!(m.size(), fg.size());

    // The same conjunction on the same vtree, built through apply, agrees.
    let f_on = Tdd::clause(&fg.vtree, [1, 2]);
    let g_on = Tdd::clause(&fg.vtree, [3, -4]) & Tdd::clause(&fg.vtree, [4]);
    assert_eq!(count(&(f_on & g_on)), count(&fg));
}

#[test]
fn graft_of_one_part_keeps_its_count_and_a_lone_spine_is_true() {
    let a = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&a, [1, -2, 3]);
    let same = Tdd::graft(vec![f.clone()], &[]).unwrap();
    assert!(same.vtree.same_tree(&a));
    assert_canonical(&same);
    assert_eq!(count(&same), count(&f));

    let free = Tdd::graft(vec![], &[VarId(2)]).unwrap();
    assert_canonical(&free);
    assert_eq!(free.vtree.num_leaves(), 1);
    assert_eq!(count(&free), 2); // unconstrained in x3
}

#[test]
fn graft_rejects_overlap_and_nothing() {
    let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
    let f = Tdd::clause(&a, [1]);
    assert_eq!(
        Tdd::graft(vec![f.clone(), f.clone()], &[]).err(),
        Some(VtreeError::OverlappingVariable(VarId(0)))
    );
    assert_eq!(
        Tdd::graft(vec![f], &[VarId(1)]).err(),
        Some(VtreeError::OverlappingVariable(VarId(1)))
    );
    assert!(matches!(Tdd::graft(vec![], &[]), Err(VtreeError::Invalid(_))));
}

#[test]
fn graft_with_layout_renames_local_parts_and_maps_their_levels() {
    let eng = Engine::new();
    // Two parts compiled in local spaces {0,1}, placed at globals {2,3} and {0,1}.
    let local = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&local, [1, 2]);
    let g = Tdd::clause(&local, [-1]);
    let (t, layout) = Tdd::graft_over(
        &eng,
        vec![(f, vec![VarId(2), VarId(3)]), (g, vec![VarId(0), VarId(1)])],
        &[VarId(4)],
        5,
        None,
    )
    .expect("the parts carry disjoint variables");
    assert_canonical(&t);
    assert_eq!(count(&t), 3 * 2 * 2);
    assert_eq!(layout.comp_to_full.len(), 2);
    assert_eq!(layout.chain_internals.len(), 2);
    // Part 0's root maps to the left child of the first chain join.
    let (left, _) = t.vtree.children(layout.chain_internals[0]);
    assert_eq!(layout.comp_to_full[0][local.root().idx()], left);
}

/// A weighted part keeps its values across the graft: each part's store rows
/// move into the merged store under their grafted level index, so the merged
/// diagram evaluates to the product the parts stand for — and still does after
/// the reduction that follows a graft.
#[test]
fn graft_over_carries_each_part_weight_store_into_the_merged_diagram() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    use crate::marginal::marginalize;
    use crate::query::weighted_value;
    use crate::query::evaluate;
    use crate::test_helpers::{compile_clauses, rat};

    let weight_of = |v: usize| (rat(v as i64 + 1, 7), rat(2, v as i64 + 3));
    let global = RationalWeights::from_weights(&(0..7).map(weight_of).collect::<Vec<_>>());

    let eng = Engine::new();
    let local = Arc::new(Vtree::balanced(3));
    // `balanced(3)`'s root has an internal right child: marginalizing it leaves
    // a store column no reader can recompute from the weight table alone, and a
    // structural root above it to reference that column.
    let (_, inner) = local.children(local.root());
    let clauses = [vec![vec![1, 2], vec![2, -3]], vec![vec![-1, 3]]];
    let placements = [
        vec![VarId(0), VarId(1), VarId(2)],
        vec![VarId(3), VarId(4), VarId(5)],
    ];

    let mut parts = Vec::new();
    for (cs, l2g) in clauses.iter().zip(placements.iter()) {
        // A part's semiring is in its OWN variable space, which is why the
        // merged store cannot be derived from the parts.
        let localized = RationalWeights::from_weights(
            &l2g.iter().map(|g| weight_of(g.idx())).collect::<Vec<_>>(),
        );
        let mut part = compile_clauses(&local, cs);
        part.set_weights(WeightStore::new(localized, Arithmetic::ExactRational));
        marginalize(&eng, &mut part, &[inner]).expect("no wall is installed in a test");
        assert!(part.levels[inner.idx()].is_weight_marginal(), "setup: a part must carry values");
        parts.push((part, l2g.clone()));
    }

    let (mut grafted, _) = Tdd::graft_over(
        &eng,
        parts,
        &[VarId(6)],
        7,
        Some(WeightStore::new(global.clone(), Arithmetic::ExactRational)),
    )
    .expect("the parts carry disjoint variables");

    // The oracle: the same conjunction built structurally over the global space.
    let global_parts: Vec<Tdd> = clauses
        .iter()
        .zip(placements.iter())
        .map(|(cs, l2g)| {
            let vtree = Arc::new(Vtree::balanced_over(l2g));
            let renamed: Vec<Vec<i32>> = cs
                .iter()
                .map(|c| c.iter().map(|l| l.signum() * (l2g[l.unsigned_abs() as usize - 1].0 as i32 + 1)).collect())
                .collect();
            compile_clauses(&vtree, &renamed)
        })
        .collect();
    let want = evaluate(
        &Tdd::graft(global_parts, &[VarId(6)]).expect("disjoint parts"),
        &global,
    );

    let got = |t: &Tdd| {
        weighted_value(t)
            .expect("the merged diagram is weighted")
            .as_rational()
            .into_owned()
    };
    assert_eq!(got(&grafted), want, "the graft lost or misplaced a part's weights");
    minimize(&mut grafted);
    assert_canonical(&grafted);
    assert_eq!(got(&grafted), want, "reduction after the graft moved the store off its levels");
}

/// [`Tdd::take_weights`] is the one public way to separate a store from its
/// diagram, and a diagram whose levels still read their values out of it keeps
/// it.
///
/// The shape that needs the detach is the component graft: a part is compiled
/// weighted, its store is held aside while the part itself is rebuilt, and the
/// graft re-attaches it under the merged level numbering. A part with no
/// weight-marginal level hands its store over and takes it back with its value
/// intact. A part that has one is refused instead, because handing the store
/// away would leave that level reading values nothing holds any more.
#[test]
fn a_part_whose_levels_still_read_the_store_keeps_it() {
    use crate::diagram::{Arithmetic, RationalWeights, TddBuildError, WeightStore};
    use crate::marginal::marginalize;
    use crate::query::weighted_value;
    use crate::test_helpers::{compile_clauses, rat};

    let eng = Engine::new();
    let local = Arc::new(Vtree::balanced(3));
    let (_, inner) = local.children(local.root());
    let table = RationalWeights::from_weights(
        &(0..3).map(|v| (rat(v + 1, 7), rat(2, v + 3))).collect::<Vec<_>>(),
    );
    let store = || WeightStore::new(table.clone(), Arithmetic::ExactRational);
    let clauses = [vec![1, 2], vec![2, -3]];
    let value = |t: &Tdd| {
        weighted_value(t).expect("a weighted diagram has a value").as_rational().into_owned()
    };

    // Held aside and given back, which is what the component graft does.
    let mut part = compile_clauses(&local, &clauses);
    part.set_weights(store());
    let want = value(&part);
    let held = part.take_weights().expect("no level of this part reads the store");
    assert!(held.is_some(), "the part was put in weighted mode, so it had a store to give");
    assert!(part.weights().is_none(), "a granted detach leaves the diagram in integer mode");
    part.set_weights(held.expect("the detach handed the store over"));
    let (grafted, _) =
        Tdd::graft_over(&eng, vec![(part, vec![VarId(0), VarId(1), VarId(2)])], &[], 3, Some(store()))
            .expect("one part covers every variable exactly once");
    assert_eq!(value(&grafted), want, "the store did not survive the round trip through the graft");

    // A marginalized level's values live in the store, so the detach is refused.
    let mut part = compile_clauses(&local, &clauses);
    part.set_weights(store());
    marginalize(&eng, &mut part, &[inner]).expect("no wall is installed in a test");
    assert!(part.levels[inner.idx()].is_weight_marginal(), "setup: the part must carry values");
    let want = value(&part);
    match part.take_weights() {
        Err(TddBuildError::WeightedLevelWithoutStore { level }) => {
            assert_eq!(level, inner, "the refusal names the level that would be stranded");
        }
        other => panic!("a level reading the store must refuse the detach, got {other:?}"),
    }
    assert!(part.weights().is_some(), "a refused detach leaves the store where it was");
    assert_eq!(value(&part), want, "a refused detach left the diagram readable");
}
