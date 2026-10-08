use super::*;

/// Within a compound operation the dense preflight compares the predicted
/// cells with the budget left after the earlier steps' charges, not with the
/// whole budget, so it refuses before the conjunction starts growing.
#[test]
fn dense_preflight_reads_the_budget_left_in_the_operation() {
    use crate::limits::LimitConfig;
    let engine = Engine::new();
    let lim = engine.limits();
    let _limit = lim.scope(LimitConfig::none().with_memory_budget_bytes(Some(10 * APPLY_BYTES_PER_CELL)));
    let _op = lim.begin_operation();
    assert_eq!(preflight_dense_budget(lim, 10), Ok(()));
    lim.charge_in_flight(APPLY_BYTES_PER_CELL);
    assert_eq!(preflight_dense_budget(lim, 10), Err(OperationError::OverBudget));
    assert_eq!(preflight_dense_budget(lim, 9), Ok(()));
}

/// With free levels, the survey lists what the masks say: the levels no
/// operand is free at, bottom-up; the tops of the regions; every free level,
/// bottom-up; the leaves under no free level; and the node counts of the
/// free levels before each built level and in all. With none, the cone is
/// every level.
#[test]
fn the_cone_is_the_levels_no_free_level_is_over() {
    use std::sync::Arc;
    use crate::restructure::embed::{place_moving, Free};
    use crate::test_helpers::{compile_clauses, vtree_shapes};
    let eng = Engine::new();
    // A diagram on `into` restricted to `vars`, which embeds in it.
    let over = |into: &Vtree, vars: &[u32], clauses: &[Vec<i32>]| {
        let source = into.project_to_vars(|v| vars.contains(&v.0).then_some(v), into.num_vars()).unwrap();
        let mut tdd = compile_clauses(&Arc::new(source), clauses);
        tdd.minimize().unwrap();
        tdd
    };
    let sorted = |list: &[VtreeIdx]| {
        let mut list = list.to_vec();
        list.sort();
        list
    };
    let mut lists = ConeLists::default();
    let mut regions = 0;
    for (shape, into) in vtree_shapes(8) {
        let place = |d: &Tdd| {
            place_moving(&eng, d.clone(), &into, |v| v, Free::Leave, false)
                .unwrap_or_else(|_| panic!("{shape}: not placed"))
        };
        let a = over(&into, &[1, 2, 3], &[vec![1, -2], vec![2, 3]]);
        let b = over(&into, &[3, 6, 7, 8], &[vec![-3, 6], vec![7, -8], vec![6, 8]]);
        let c = over(&into, &[5, 6], &[vec![5, 6]]);
        let n = into.num_nodes();
        let internal = into.internal_bottomup_slice();
        for (f, g) in [(&a, &b), (&b, &a), (&a, &c), (&a, &a)] {
            let ((f, f_plan), (g, g_plan)) = (place(f), place(g));
            let (mut f_widths, mut g_widths) = (vec![0; n], vec![0; n]);

            let (_, cone) = survey(&eng, &into, &f, &g, Operands::default(), 0, &mut f_widths, &mut g_widths, &mut lists).unwrap();
            assert_eq!(cone.built, internal, "{shape}: no free level");
            assert_eq!(cone.leaves, into.leaf_bottomup_slice(), "{shape}: no free level");
            assert!(cone.tops.is_empty() && cone.free.is_empty() && cone.free_nodes == 0, "{shape}: no free level");

            let masks = Operands { f: VtreeMask::new(Some(&f_plan.free)), g: VtreeMask::new(Some(&g_plan.free)) };
            let (_, cone) = survey(&eng, &into, &f, &g, masks, 0, &mut f_widths, &mut g_widths, &mut lists).unwrap();
            let free_at = |t: VtreeIdx| f_plan.free[t.idx()] || g_plan.free[t.idx()];
            let under_free = |t: VtreeIdx| {
                let mut up = into.node(t).parent();
                while let Some(p) = up {
                    if free_at(p) {
                        return true;
                    }
                    up = into.node(p).parent();
                }
                false
            };
            let width = |d: &Tdd, free: &[bool], t: VtreeIdx| match free[t.idx()] {
                true => 1,
                false => d.levels[t.idx()].slot_count() as u64,
            };
            let nodes = |t: VtreeIdx| width(&f, &f_plan.free, t) * width(&g, &g_plan.free, t);
            let built: Vec<_> = internal.iter().copied().filter(|&t| !free_at(t)).collect();
            let free: Vec<_> = internal.iter().copied().filter(|&t| free_at(t)).collect();
            let tops: Vec<_> = free.iter().copied().filter(|&t| !under_free(t)).collect();
            let leaves: Vec<_> = into.leaf_bottomup_slice().iter().copied().filter(|&t| !under_free(t)).collect();
            assert_eq!(cone.built, &built[..], "{shape}: built levels");
            assert_eq!(cone.free, &free[..], "{shape}: free levels");
            assert_eq!(sorted(cone.tops), sorted(&tops), "{shape}: tops");
            assert_eq!(sorted(cone.leaves), sorted(&leaves), "{shape}: leaves");
            assert_eq!(cone.free_nodes, free.iter().map(|&t| nodes(t)).sum::<u64>(), "{shape}: free nodes");
            for (k, &t) in built.iter().enumerate() {
                let before = free.iter().filter(|&&s| into.topo_pos(s) < into.topo_pos(t)).map(|&s| nodes(s)).sum::<u64>();
                assert_eq!(cone.free_before(k), before, "{shape}: free nodes before level {}", t.0);
                assert_eq!(f_widths[t.idx()], f.levels[t.idx()].slot_count(), "{shape}: f's width at {}", t.0);
                assert_eq!(g_widths[t.idx()], g.levels[t.idx()].slot_count(), "{shape}: g's width at {}", t.0);
            }
            for &t in &leaves {
                assert_eq!((f_widths[t.idx()], g_widths[t.idx()]), (LEAF_WIDTH, LEAF_WIDTH), "{shape}: leaf {}", t.0);
            }
            regions += tops.len();
        }
    }
    assert!(regions > 0, "no free level in any case");
}
