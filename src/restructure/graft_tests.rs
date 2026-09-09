use crate::engine::Engine;
use std::sync::Arc;

use super::graft_over;
use crate::reduce::minimize;
use crate::query::model_count;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeError};

fn count(_eng: &Engine, t: &Tdd) -> u64 {
    model_count(t).try_into().expect("small count")
}

#[test]
fn graft_counts_the_product_times_two_per_spine_var() {
    let eng = Engine::new();
    let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
    let b = Arc::new(Vtree::linear_over(&[VarId(3), VarId(2)]));
    let f = Tdd::clause(&a, [1, 2]); // 3 models over {x1, x2}
    let g = Tdd::clause(&b, [3, -4]) & Tdd::clause(&b, [4]); // x3 ∧ x4: 1 model
    assert_eq!((count(&eng, &f), count(&eng, &g)), (3, 1));

    let fg = Tdd::graft(vec![f.clone(), g.clone()], &[VarId(4), VarId(5)]).unwrap();
    assert_eq!(fg.vtree.num_vars(), 6);
    assert_eq!(count(&eng, &fg), 3 * 4);

    // Already canonical: minimize changes nothing.
    let mut m = fg.clone();
    minimize(&mut m);
    assert_eq!(m.size(), fg.size());

    // The same conjunction on the same vtree, built through apply, agrees.
    let f_on = Tdd::clause(&fg.vtree, [1, 2]);
    let g_on = Tdd::clause(&fg.vtree, [3, -4]) & Tdd::clause(&fg.vtree, [4]);
    assert_eq!(count(&eng, &(f_on & g_on)), count(&eng, &fg));
}

#[test]
fn graft_of_one_part_keeps_its_count_and_a_lone_spine_is_true() {
    let eng = Engine::new();
    let a = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&a, [1, -2, 3]);
    let same = Tdd::graft(vec![f.clone()], &[]).unwrap();
    assert!(same.vtree.same_tree(&a));
    assert_eq!(count(&eng, &same), count(&eng, &f));

    let free = Tdd::graft(vec![], &[VarId(2)]).unwrap();
    assert_eq!(free.vtree.num_leaves(), 1);
    assert_eq!(count(&eng, &free), 2); // unconstrained in x3
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
    let (t, layout) = graft_over(
        &eng,
        vec![(f, vec![VarId(2), VarId(3)]), (g, vec![VarId(0), VarId(1)])],
        &[VarId(4)],
        5,
    );
    assert_eq!(count(&eng, &t), 3 * 2 * 2);
    assert_eq!(layout.comp_to_full.len(), 2);
    assert_eq!(layout.chain_internals.len(), 2);
    // Part 0's root maps to the left child of the first chain join.
    let (left, _) = t.vtree.children(layout.chain_internals[0]);
    assert_eq!(layout.comp_to_full[0][local.root().idx()], left);
}
