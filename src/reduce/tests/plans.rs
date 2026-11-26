//! What each public reduction plan and content-twin policy leaves behind.

use crate::vtree::Home;

use crate::diagram::{NEG_LEAF_IDX, POS_LEAF_IDX, Tdd};
use crate::reduce::{ContentTwinPolicy, ContentTwinSchedule, ReductionPlan};
use crate::query::count::PinSemantics;
use crate::test_helpers::check::marginal::check_no_orphan_slots;
use crate::test_helpers::pinned_counts;
use crate::test_helpers::{
    CnfShape, Lcg, assert_canonical, assert_same_shape, compile_clauses, compile_clauses_on, rand_cnf,
    same_as_stored, toy, vtree_shapes, with_floor,
};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

use super::edited_marginal_diagram;

/// A boundary store of three slots, of which the root node names two.
fn with_an_orphaned_slot() -> Tdd {
    let f = toy(vec![5, 7, 9], &[&[(POS_LEAF_IDX.0, 0), (NEG_LEAF_IDX.0, 2)]]);
    assert!(check_no_orphan_slots(&f).is_err(), "the fixture must start with an orphan");
    f
}

#[test]
fn a_full_reduce_without_a_content_scan_drops_orphaned_slots() {
    let mut f = with_an_orphaned_slot();
    let count = f.model_count().unwrap();
    f.reduce(ReductionPlan::Full(ContentTwinPolicy::Skip)).unwrap();
    check_no_orphan_slots(&f).unwrap();
    assert_eq!(f.model_count().unwrap(), count);
}

#[test]
fn a_prune_drops_orphaned_slots() {
    let mut f = with_an_orphaned_slot();
    let count = f.model_count().unwrap();
    f.reduce(ReductionPlan::Prune).unwrap();
    check_no_orphan_slots(&f).unwrap();
    assert_eq!(f.model_count().unwrap(), count);
}

#[test]
fn the_adaptive_policy_reaches_the_fresh_policy_s_diagram() {
    let eng = Engine::new();
    let f = edited_marginal_diagram(&eng);
    let mut fresh = f.clone();
    eng.reduce(&mut fresh, ReductionPlan::Full(ContentTwinPolicy::Fresh)).unwrap();
    assert_canonical(&fresh);
    let mut schedule = ContentTwinSchedule::default();
    let mut adaptive = f;
    eng.reduce(&mut adaptive, ReductionPlan::Full(ContentTwinPolicy::Adaptive(&mut schedule))).unwrap();
    assert_canonical(&adaptive);
    assert_eq!(schedule.next_scan_at_nodes, 0, "a small diagram is scanned on every call");
    assert_same_shape(&fresh, &adaptive, "fresh against adaptive");
}

/// A sampled scan whose bar any sample clears is the adaptive scan; one whose
/// bar none clears stops after its sample with the count kept; one on fewer
/// pairs than its `min_pairs` is the adaptive scan, unsampled.
#[test]
fn a_sampled_scan_goes_on_or_stops_at_its_bar() {
    let eng = Engine::new();
    let f = edited_marginal_diagram(&eng);
    let count = f.model_count().unwrap();
    let mut adaptive = f.clone();
    eng.reduce(&mut adaptive, ReductionPlan::Full(ContentTwinPolicy::Adaptive(&mut ContentTwinSchedule::default()))).unwrap();
    let run = |min_yield: f64, min_pairs: usize| {
        let (mut g, mut schedule) = (f.clone(), ContentTwinSchedule::default());
        eng.reduce(&mut g, ReductionPlan::Full(ContentTwinPolicy::Sampled { schedule: &mut schedule, min_yield, min_pairs })).unwrap();
        assert_eq!(g.model_count().unwrap(), count);
        check_no_orphan_slots(&g).unwrap();
        (g, schedule.last_sample_yield())
    };
    let sampled = |min_yield: f64| {
        let (g, found) = run(min_yield, 0);
        (g, found.expect("a sampled scan reports its yield"))
    };
    let (unsampled, none) = run(2.0, usize::MAX);
    assert_eq!(none, None);
    assert_same_shape(&adaptive, &unsampled, "adaptive against a scan under min_pairs");
    let (gone_on, found) = sampled(0.0);
    assert!((0.0..=1.0).contains(&found));
    assert_same_shape(&adaptive, &gone_on, "adaptive against a sample with no bar");
    let (stopped, again) = sampled(2.0);
    assert_eq!(found, again);
    assert!(stopped.node_count() >= adaptive.node_count());
}

/// Home ((x1 x2) (x3 x4)) with x2 summed out, onto a region. The structure
/// of a diagram with a summed-out region is a function of the operations
/// that built it, not of its function (`Tdd::minimize` certifies no such
/// diagram): summing before or after the conjunction with the clauses on
/// x1, x3 and x4 gives 12 or 10 pairs, with or without content scans, and
/// the same count under every assignment.
#[test]
fn a_sum_out_before_or_after_a_conjunction_moves_a_count_diagram_s_form() {
    let home = Home::new(Vtree::balanced(4));
    let (f1, f2) = (vec![vec![-2, -3], vec![2, 4]], vec![vec![-1, 4], vec![1, -3]]);
    let compile = |clauses: &[Vec<i32>]| {
        let mut f = clauses.iter().map(|c| Tdd::clause(&home, c.as_slice()).unwrap()).reduce(|f, g| crate::and(f, g).unwrap()).unwrap();
        f.minimize().unwrap();
        f
    };
    let full = |f: &mut Tdd, scan: bool| {
        let twins = if scan { ContentTwinPolicy::Fresh } else { ContentTwinPolicy::Skip };
        f.reduce(ReductionPlan::Full(twins)).unwrap();
    };
    let summed = |clauses: &[Vec<i32>], scan: bool| {
        let mut f = compile(clauses);
        f.marginalize(|v| v == VarId(2)).unwrap();
        full(&mut f, scan);
        f
    };
    for scan in [false, true] {
        let after = summed(&[f1.clone(), f2.clone()].concat(), scan);
        let mut before = crate::and(summed(&f1, scan), compile(&f2)).unwrap();
        full(&mut before, scan);
        for bits in 0..8u32 {
            let pins = [Some(bits & 1 != 0), None, Some(bits & 2 != 0), Some(bits & 4 != 0)];
            assert_eq!(
                pinned_counts(&after, &pins, PinSemantics::Evidence),
                pinned_counts(&before, &pins, PinSemantics::Evidence),
                "scan {scan}, pins {pins:?}",
            );
        }
        assert_eq!((after.pair_count(), before.pair_count()), (12, 10), "scan {scan}");
    }
}

#[test]
fn the_contract_plan_drains_its_worklist_and_a_full_plan_finishes_the_job() {
    let vtree = Home::new(Vtree::balanced(5));
    let clauses = [vec![1, 2, 3], vec![-2, 4], vec![3, -5], vec![-1, 5]];
    let mut stepwise = compile_clauses(&vtree, &clauses);
    stepwise.reduce(ReductionPlan::Contract).unwrap();
    assert!(stepwise.contract_worklist().is_empty(), "the contract plan owes contraction nothing");
    stepwise.minimize().unwrap();
    assert_canonical(&stepwise);
    let mut direct = compile_clauses(&vtree, &clauses);
    direct.minimize().unwrap();
    assert_same_shape(&stepwise, &direct, "contract then minimize against minimize");
}

/// The close at a plan's end changes how levels hold their pairs, not the
/// diagram: a canonical diagram stays certified through the prune and
/// contract plans, at the floor and at a floor of two pairs.
#[test]
fn the_prune_and_contract_plans_keep_a_canonical_diagram_certified() {
    for floor in [crate::diagram::FLOOR, 2] {
        with_floor(floor, || {
            let vtree = Home::new(Vtree::balanced(5));
            let mut f = compile_clauses(&vtree, &[vec![1, 2, 3], vec![-2, 4], vec![3, -5], vec![-1, 5]]);
            f.minimize().unwrap();
            assert!(f.levels.is_canonical(f.output), "a minimized diagram is certified");
            for plan in [ReductionPlan::Prune, ReductionPlan::Contract] {
                f.reduce(plan).unwrap();
                assert!(f.levels.is_canonical(f.output), "floor {floor}: a plan's close forgot the certificate");
            }
        });
    }
}

/// The contract plan on unminimized conjunctions, on implicit levels and on
/// stored ones alike ([`same_as_stored`]).
#[test]
fn the_contract_plan_leaves_the_diagram_the_stored_route_leaves() {
    same_as_stored(|| {
        let eng = Engine::new();
        let mut rng = Lcg::new(0x6c0e_77a1);
        let mut out = Vec::new();
        for num_vars in [4u32, 6, 8] {
            for (_name, vtree) in vtree_shapes(num_vars) {
                for _ in 0..6 {
                    let mut operand = || {
                        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 5, width: 3 });
                        compile_clauses_on(&eng, &vtree, &clauses)
                    };
                    let (f, g) = (operand(), operand());
                    let mut h = eng.and(f, g).unwrap();
                    eng.reduce(&mut h, ReductionPlan::Contract).unwrap();
                    out.push(h);
                }
            }
        }
        out
    });
}

