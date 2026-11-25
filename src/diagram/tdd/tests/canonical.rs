use super::*;
use crate::{Engine, Vtree};
use crate::vtree::Home;

#[test]
fn canonical_guarantee_follows_storage_and_output() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [1, 2]).unwrap();
    f.minimize().unwrap();
    assert!(f.levels.is_canonical(f.output));
    assert!(f.clone().levels.is_canonical(f.output));
    assert!(f.try_clone_on(&Engine::new()).unwrap().levels.is_canonical(f.output));
    let output = f.output;
    f.output.local = ZERO;
    assert!(!f.levels.is_canonical(f.output));
    f.output = output;
    assert!(f.levels.is_canonical(f.output));
    let _ = &mut f.levels[output.vtree.idx()];
    assert!(!f.levels.is_canonical(f.output));
    f.minimize().unwrap();
    assert!(f.levels.is_canonical(f.output));
    let same = vtree.clone();
    unsafe { f.reseat_vtree_unchecked(&same); }
    assert!(!f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn builder_and_marginal_storage_are_not_certified() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [1, 2]).unwrap();
    f.minimize().unwrap();
    let mut rebuilt = Tdd::from_levels_unchecked(vtree.clone(), f.levels.clone().into_vec(), f.output);
    assert!(!rebuilt.levels.is_canonical(rebuilt.output));
    assert_eq!(rebuilt.support().unwrap(), f.support().unwrap());
    rebuilt.marginalize(crate::test_helpers::summing(rebuilt.vtree(), &[vtree.root()])).unwrap();
    rebuilt.minimize().unwrap();
    assert!(!rebuilt.levels.is_canonical(rebuilt.output));
}

#[test]
fn certified_leaf_queries_borrow_and_still_honor_cancellation() {
    use crate::OperationError;
    use crate::limits::LimitConfig;
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&crate::vtree::Home::unchecked(&vtree));
    f.minimize().unwrap();
    let unproven = Tdd::from_levels_unchecked(vtree, f.levels.clone().into_vec(), f.output);
    let eng = Engine::new();
    {
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        // A free-variable scan needs no result allocation; an unproven input needs a copy.
        assert!(eng.support(&f).unwrap().is_empty());
        assert!(eng.implied_literals(&f).unwrap().is_empty());
        assert_eq!(eng.support(&unproven), Err(OperationError::OverBudget));
    }
    let stopped = crate::test_helpers::stopping_engine();
    assert_eq!(stopped.support(&f), Err(OperationError::Stopped));
    assert_eq!(stopped.implied_literals(&f), Err(OperationError::Stopped));
    assert_eq!(stopped.equivalent(&f, &f.clone()), Err(OperationError::Stopped));
    assert!(f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn refused_minimization_does_not_certify_partial_work() {
    let vtree = Arc::new(Vtree::balanced(4));
    let source = Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [1, 2, 3]).unwrap();
    let mut f = Tdd::from_levels_unchecked(vtree, source.levels.clone().into_vec(), source.output);
    let eng = Engine::new();
    eng.refuse_nth_reserve(0);
    assert_eq!(eng.minimize(&mut f), Err(crate::OperationError::OverBudget));
    assert!(!f.levels.is_canonical(f.output));
    eng.limits().grant_every_reserve();
    eng.minimize(&mut f).unwrap();
    assert!(f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

/// A diagram on a projection reseated on a second allocation of that
/// projection stays on the leaves it uses and names none true, and conjoins
/// with a diagram on the first allocation.
#[test]
fn a_diagram_reseated_on_another_allocation_of_its_projection_stays_closed() {
    use crate::test_helpers::check::check_trimmed;
    use crate::vtree::{Home, VarId};
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(5));
    let mut f = eng.xor(eng.literal(&home, 1).unwrap(), eng.clause(&home, [3, -4]).unwrap()).unwrap();
    let first = Arc::clone(f.vtree());
    assert_eq!(first.num_leaves(), 3);
    let second = Vtree::projection(&home, [1, 3, 4].map(VarId)).unwrap();
    assert!(!Arc::ptr_eq(&first, &second), "the test needs two allocations");
    check_trimmed(&f, true).unwrap();
    let before = f.clone();
    // Safety: `second` is `first` built again: same nodes, indices,
    // variables and home.
    unsafe { f.reseat_vtree_unchecked(&second) };
    assert!(Arc::ptr_eq(f.vtree(), &second));
    check_trimmed(&f, true).unwrap();
    let same = eng.and(f.clone(), before.clone()).unwrap();
    check_trimmed(&same, true).unwrap();
    assert!(eng.equivalent(&same, &before).unwrap());
    let g = eng.negate(before).unwrap();
    assert!(Arc::ptr_eq(g.vtree(), &first));
    assert!(eng.and(f, g).unwrap().is_zero());
}

/// In a debug build, reseating a diagram on a home onto a vtree off every
/// home is refused.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "off every home")]
fn reseating_a_diagram_on_a_home_off_the_home_panics_in_debug() {
    use crate::vtree::Home;
    let home = Home::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&home, [1, 2, 3]).unwrap();
    let bare = Arc::new(Vtree::balanced(3));
    // Safety: never reached past the debug check.
    unsafe { f.reseat_vtree_unchecked(&bare) };
}

mod producers {
    use super::*;
    use crate::reduce::ReductionPlan;
    use crate::test_helpers::check::check_certified;
    use crate::test_helpers::{rand_conj, rand_conj_over};
    use crate::vtree::rng::Lcg;
    use crate::vtree::VarId;

    fn certified(t: &Tdd, what: &str) {
        assert!(t.is_canonical(), "{what}: not certified");
        check_certified(t).unwrap_or_else(|e| panic!("{what}: {e}"));
    }

    /// A minimized diagram on `vtree` over `vars`: a literal one time in
    /// four, else a random CNF.
    fn source(vtree: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
        let mut f = match rng.below(4) {
            0 => {
                let v = vars[rng.below(vars.len() as u64) as usize] as i32;
                Tdd::clause(&crate::vtree::Home::unchecked(vtree), [if rng.coin() { v } else { -v }]).unwrap()
            }
            _ => rand_conj_over(vtree, vars, 4, 3, false, rng),
        };
        Engine::new().minimize(&mut f).unwrap();
        f
    }

    #[test]
    fn grafts_of_certified_parts_are_certified() {
        for seed in 0..200u64 {
            let mut rng = Lcg::new(seed ^ 0x6a7);
            let mut next = 1u32;
            let mut parts = Vec::new();
            for _ in 0..1 + rng.below(3) {
                let vars: Vec<u32> = (next..next + 1 + rng.below(3) as u32).collect();
                next += vars.len() as u32;
                let order: Vec<VarId> = vars.iter().map(|&v| VarId(v)).collect();
                let vtree = Arc::new(Vtree::balanced_over(&order).unwrap());
                parts.push(source(&vtree, &vars, &mut rng));
            }
            let spine: Vec<VarId> = (0..rng.below(3) as u32).map(|i| VarId(next + i)).collect();
            let count = parts.iter().map(|p| p.model_count().unwrap()).product::<num_bigint::BigUint>() << spine.len();
            let Ok(g) = crate::restructure::graft::graft(&Engine::new(), parts, &spine) else { continue };
            certified(&g, &format!("seed {seed}: graft"));
            assert_eq!(g.model_count().unwrap(), count, "seed {seed}");
        }
    }

    #[test]
    fn projections_to_a_subtree_are_certified() {
        let eng = Engine::new();
        for seed in 0..100u64 {
            let mut rng = Lcg::new(seed ^ 0x51b);
            let n = 2 + rng.below(6) as u32;
            let vtree = Arc::new(Vtree::random(n, seed));
            let f = rand_conj(&vtree, n, 4, 3, false, &mut rng);
            for key in vtree.bottomup().filter(|&t| t != vtree.root()) {
                let g = eng.project_to_subtree(&f, key).unwrap();
                certified(&g, &format!("seed {seed}: project_to_subtree at {}", key.0));
            }
        }
    }

    /// The same on homes, including a key whose subtree holds every leaf of
    /// the operand's projection (seed 6, key 2: a literal).
    #[test]
    fn projections_to_a_subtree_on_a_home_are_certified() {
        let eng = Engine::new();
        for seed in 0..100u64 {
            let mut rng = Lcg::new(seed ^ 0x51b);
            let n = 2 + rng.below(6) as u32;
            let vtree = Home::new(Vtree::random(n, seed));
            let f = rand_conj(&vtree, n, 4, 3, false, &mut rng);
            for key in vtree.bottomup().filter(|&t| t != vtree.root()) {
                let g = eng.project_to_subtree(&f, key).unwrap();
                certified(&g, &format!("seed {seed}: project_to_subtree at {}", key.0));
            }
        }
    }

    #[test]
    fn constants_without_a_node_are_certified() {
        let home = crate::vtree::Home::new(Vtree::random(3, 1));
        let empty = Vtree::empty_projection(&home);
        assert!(Tdd::one(&crate::vtree::Home::unchecked(&empty)).is_canonical());
        assert!(Tdd::zero(&crate::vtree::Home::unchecked(&empty)).is_canonical());
    }

    /// Prune and contraction keep a certificate with nothing owed; a
    /// negation certifies its result under every plan from a certified
    /// operand, and from any operand once minimized.
    #[test]
    fn idle_reductions_and_negations_certify() {
        let eng = Engine::new();
        for seed in 0..300u64 {
            let mut rng = Lcg::new(seed ^ 0x4e9);
            let n = 2 + rng.below(7) as u32;
            let vtree = Home::new(Vtree::random(n, seed));
            let f = rand_conj(&vtree, n, 5, 3, false, &mut rng);
            let mut c = f.clone();
            eng.minimize(&mut c).unwrap();
            for plan in [ReductionPlan::Prune, ReductionPlan::Contract] {
                let (mut d, label) = (c.clone(), format!("seed {seed}: reduce({plan:?})"));
                eng.reduce(&mut d, plan).unwrap();
                certified(&d, &label);
            }
            let expected = eng.negate(&c).unwrap();
            certified(&expected, &format!("seed {seed}: negate"));
            for plan in [ReductionPlan::Prune, ReductionPlan::Contract] {
                let label = format!("seed {seed}: negate_with({plan:?})");
                let g = crate::apply::with_plan::negate_with(&eng, &c, plan).unwrap();
                certified(&g, &label);
                assert!(eng.equivalent(&g, &expected).unwrap(), "seed {seed}");
            }
            let g = crate::test_helpers::minimized(&eng, eng.negate(&f));
            certified(&g, &format!("seed {seed}: negate of an uncertified operand"));
            assert_eq!(g.node_count(), expected.node_count(), "seed {seed}");
        }
    }
}

/// The checker rejects a certificate on a diagram with twins.
#[test]
fn a_false_certificate_fails_the_check() {
    let vtree = Home::new(Vtree::balanced(2));
    let f = Tdd::clause(&vtree, [-1, 2]).unwrap();
    let mut y = crate::and(f, Tdd::clause(&vtree, [1, 2]).unwrap()).unwrap();
    y.levels.certify(y.output);
    assert!(crate::test_helpers::check::check_certified(&y).is_err());
}

