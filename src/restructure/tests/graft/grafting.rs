use crate::Engine;
use crate::restructure::GraftError;
use std::sync::Arc;



use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Tdd, Weights};
use crate::vtree::{VarId, Vtree, VtreeError};
use crate::test_helpers::{assert_canonical, compile_clauses, rat};

fn count(t: &Tdd) -> u64 {
    t.model_count().unwrap().try_into().expect("small count")
}

fn test_literal_weight(v: usize) -> LiteralWeights<num_rational::BigRational> {
    LiteralWeights { negative: rat(v as i64 + 1, 7), positive: rat(2, v as i64 + 3) }
}

fn rational_value(t: &Tdd) -> num_rational::BigRational {
    t.weighted_value().unwrap().expect("a weighted diagram has a value").as_rational().into_owned()
}

#[test]
fn engine_graft_refuses_an_expired_batch_and_retries_unbounded() {
    use crate::{Context, OperationError};
    use std::time::{Duration, Instant};

    let context = Arc::new(Context::new());
    let vtree = context.bind(Vtree::leaf(VarId(1)));
    let part = crate::literal(&vtree, 1).unwrap();
    assert_canonical(&part);
    let eng = Engine::new();
    {
        let deadline = Instant::now() - Duration::from_secs(1);
        let _scope = eng.limits().edit(|s| s.with_deadline(Some(deadline)));
        for parts in [vec![part.clone()], vec![]] {
            assert!(matches!(eng.graft(parts, &[VarId(2)]), Err(GraftError::Operation(OperationError::Stopped))));
        }
    }
    let result = eng.graft(vec![part], &[VarId(2)]).unwrap();
    assert_canonical(&result);
    assert_eq!(count(&result), 2);
    assert!(Arc::ptr_eq(result.context(), &context));
    let free = eng.graft(vec![], &[VarId(7)]).unwrap();
    assert_canonical(&free);
    assert_eq!(count(&free), 2);
}

#[test]
fn graft_counts_the_product_times_two_per_spine_var() {
    let a = Arc::new(Vtree::balanced_over(&[VarId(1), VarId(2)]).unwrap());
    let b = Arc::new(Vtree::linear_from_order(&[VarId(4), VarId(3)]).unwrap());
    let f = Tdd::clause(&a, [1, 2]).unwrap(); // 3 models over {x1, x2}
    let g = Tdd::clause(&b, [3, -4]).unwrap() & Tdd::clause(&b, [4]).unwrap(); // x3 ∧ x4: 1 model
    assert_eq!((count(&f), count(&g)), (3, 1));

    let fg = Engine::new().graft(vec![f.clone(), g.clone()], &[VarId(5), VarId(6)]).unwrap();
    assert_eq!(fg.vtree.num_vars(), 6);
    assert_canonical(&fg);
    assert_eq!(count(&fg), 3 * 4);

    // Already canonical: minimize changes nothing.
    let mut m = fg.clone();
    m.minimize().unwrap();
    assert_canonical(&m);
    assert_eq!(m.pair_count(), fg.pair_count());

    // The same conjunction on the same vtree, built through apply, agrees.
    let f_on = Tdd::clause(&fg.vtree, [1, 2]).unwrap();
    let g_on = Tdd::clause(&fg.vtree, [3, -4]).unwrap() & Tdd::clause(&fg.vtree, [4]).unwrap();
    assert_eq!(count(&(f_on & g_on)), count(&fg));
}

#[test]
fn graft_of_one_part_keeps_its_count_and_a_lone_spine_is_true() {
    let a = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&a, [1, -2, 3]).unwrap();
    let same = Engine::new().graft(vec![f.clone()], &[]).unwrap();
    assert!(same.vtree.same_tree(&a));
    assert_canonical(&same);
    assert_eq!(count(&same), count(&f));

    let free = Engine::new().graft(vec![], &[VarId(3)]).unwrap();
    assert_canonical(&free);
    assert_eq!(free.vtree.num_leaves(), 1);
    assert_eq!(count(&free), 2); // unconstrained in x3
}

#[test]
fn graft_rejects_overlap_and_nothing() {
    let a = Arc::new(Vtree::balanced_over(&[VarId(1), VarId(2)]).unwrap());
    let f = Tdd::clause(&a, [1]).unwrap();
    assert_eq!(
        Engine::new().graft(vec![f.clone(), f.clone()], &[]).err(),
        Some(GraftError::Vtree(VtreeError::OverlappingVariable(VarId(1))))
    );
    assert_eq!(
        Engine::new().graft(vec![f], &[VarId(2)]).err(),
        Some(GraftError::Vtree(VtreeError::OverlappingVariable(VarId(2))))
    );
    assert!(matches!(Engine::new().graft(vec![], &[]), Err(GraftError::Vtree(VtreeError::Invalid(_)))));
}

#[test]
fn graft_with_layout_renames_local_parts_and_maps_their_levels() {
    let eng = Engine::new();
    // Two parts compiled in local spaces {0,1}, placed at globals {2,3} and {0,1}.
    let local = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&local, [1, 2]).unwrap();
    let g = Tdd::clause(&local, [-1]).unwrap();
    let (t, layout) = Tdd::graft_over(
        &eng,
        vec![(f, vec![VarId(3), VarId(4)]), (g, vec![VarId(1), VarId(2)])],
        &[VarId(5)],
        5,
        None,
    )
    .expect("the parts carry disjoint variables");
    assert_canonical(&t);
    assert_eq!(count(&t), 3 * 2 * 2);
    assert_eq!(layout.parts().len(), 2);
    assert_eq!(layout.chain_internals.len(), 2);
    // Part 0's root maps to the left child of the first chain join.
    let (left, _) = t.vtree.children(layout.chain_internals[0]);
    assert_eq!(layout.part(0).level_of(local.root()), left);
}

/// A weighted part keeps its values across the graft: the merged diagram
/// evaluates to the product the parts stand for, and still does after the
/// reduction that follows a graft.
#[test]
fn graft_over_carries_each_part_weighted_values_into_the_merged_diagram() {
    let global = RationalWeights::from_literals(&(0..7).map(test_literal_weight).collect::<Vec<_>>());

    let eng = Engine::new();
    let local = Arc::new(Vtree::balanced(3));
    // `balanced(3)`'s root has an internal right child: marginalizing it leaves
    // a column no reader can recompute from the weight table alone, and a
    // structural root above it to reference that column.
    let (_, inner) = local.children(local.root());
    let clauses = [vec![vec![1, 2], vec![2, -3]], vec![vec![-1, 3]]];
    let placements = [
        vec![VarId(1), VarId(2), VarId(3)],
        vec![VarId(4), VarId(5), VarId(6)],
    ];

    let mut parts = Vec::new();
    for (cs, l2g) in clauses.iter().zip(placements.iter()) {
        // A part's semiring is in its OWN variable space, which is why the
        // merged weights cannot be derived from the parts.
        let localized = RationalWeights::from_literals(
            &l2g.iter().map(|g| test_literal_weight(g.idx())).collect::<Vec<_>>(),
        );
        let weighted = Vtree::balanced(3).with_weights(Weights::new(localized, Arithmetic::ExactRational)).unwrap();
        let mut part = compile_clauses(&Arc::new(weighted), cs);
        eng.marginalize_levels(&mut part, &[inner]).expect("no wall is installed in a test");
        assert!(part.levels[inner.idx()].is_weight_marginal(), "setup: a part must carry values");
        parts.push((part, l2g.clone()));
    }

    let (mut grafted, _) = Tdd::graft_over(
        &eng,
        parts,
        &[VarId(7)],
        7,
        Some(Weights::new(global.clone(), Arithmetic::ExactRational)),
    )
    .expect("the parts carry disjoint variables");

    // The oracle: the same conjunction built structurally over the global space.
    let global_parts: Vec<Tdd> = clauses
        .iter()
        .zip(placements.iter())
        .map(|(cs, l2g)| {
            let vtree = Arc::new(Vtree::balanced_over(l2g).unwrap());
            let renamed: Vec<Vec<i32>> = cs
                .iter()
                .map(|c| c.iter().map(|l| l.signum() * (l2g[l.unsigned_abs() as usize - 1].0 as i32)).collect())
                .collect();
            compile_clauses(&vtree, &renamed)
        })
        .collect();
    let want = (Engine::new().graft(global_parts, &[VarId(7)]).expect("disjoint parts")).evaluate(&global).unwrap();

    assert_eq!(rational_value(&grafted), want, "the graft lost or misplaced a part's weights");
    grafted.minimize().unwrap();
    assert_canonical(&grafted);
    assert_eq!(rational_value(&grafted), want, "reduction after the graft moved values off their levels");
}

#[test]
fn a_weighted_part_grafted_over_its_own_weights_keeps_its_value() {
    let eng = Engine::new();
    let table = RationalWeights::from_literals(&(0..3).map(test_literal_weight).collect::<Vec<_>>());
    let store = || Weights::new(table.clone(), Arithmetic::ExactRational);
    let local = Arc::new(Vtree::balanced(3).with_weights(store()).unwrap());
    let part = compile_clauses(&local, &[vec![1, 2], vec![2, -3]]);
    let want = rational_value(&part);
    let (grafted, _) =
        Tdd::graft_over(&eng, vec![(part, vec![VarId(1), VarId(2), VarId(3)])], &[], 3, Some(store()))
            .expect("one part covers every variable exactly once");
    assert_eq!(rational_value(&grafted), want);
}

#[test]
fn a_false_part_makes_the_graft_false() {
    let a = Arc::new(Vtree::balanced_over(&[VarId(1), VarId(2)]).unwrap());
    let b = Arc::new(Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap());
    let f = Tdd::clause(&a, [1, 2]).unwrap();
    let g = Tdd::clause(&b, [3]).unwrap() & Tdd::clause(&b, [-3]).unwrap();
    assert!(g.is_zero());

    let fg = Engine::new().graft(vec![f, g], &[VarId(5)]).unwrap();
    assert!(fg.is_zero());
    assert_eq!(count(&fg), 0);
    assert_eq!(fg.vtree.num_vars(), 5);
}
