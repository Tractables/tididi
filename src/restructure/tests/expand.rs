//! `Tdd::expand_variables`: each variable becomes a signed class, with
//! constants and free variables grafted on.

use std::sync::atomic::{AtomicUsize, Ordering};

use num_bigint::BigUint;

use super::*;
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::test_helpers::{assert_canonical, compile_clauses_on, marginal_diagrams, rand_cnf, truth_table, vtree_shapes, CnfShape, Lcg};
use crate::vtree::VtreeError;

fn pos(v: u32) -> Literal {
    Literal::pos(VarId(v))
}

fn neg(v: u32) -> Literal {
    Literal::neg(VarId(v))
}

/// A diagram over one variable whose output is that leaf's node `local`.
fn leaf_diagram(eng: &Engine, local: NodeIdx) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(1));
    Tdd::builder(eng, &vtree).unwrap().finish(TddNodeId { vtree: vtree.root(), local }).unwrap()
}

/// Every variable of `expansion` in a class of its own, positive.
fn identity(vtree: &Vtree) -> VariableExpansion {
    VariableExpansion {
        classes: (1..=vtree.num_vars()).map(|v| if vtree.leaf_of(VarId(v)).is_some() { vec![pos(v)] } else { vec![] }).collect(),
        num_vars: vtree.num_vars(),
        ..VariableExpansion::default()
    }
}

/// Which assignments of variables `1..=num_vars`, read as bit masks with
/// variable `v` in bit `v - 1`, `g` holds on.
fn models(g: &Tdd, num_vars: u32) -> Vec<bool> {
    let mut counter = g.counter().unwrap();
    let mut pins: Vec<(VarId, Option<bool>)> = (1..=num_vars).map(|v| (VarId(v), None)).collect();
    (0..1u64 << num_vars)
        .map(|mask| {
            for (bit, pin) in pins.iter_mut().enumerate() {
                pin.1 = Some((mask >> bit) & 1 == 1);
            }
            counter.set_pins(&pins).unwrap();
            counter.model_count().unwrap() == BigUint::from(1u32)
        })
        .collect()
}

/// What the expansion of a function with truth table `truth` over
/// `classes.len()` variables holds on, by the same masks as [`models`].
fn expected_models(truth: &[bool], expansion: &VariableExpansion) -> Vec<bool> {
    let value = |mask: u64, literal: &Literal| ((mask >> literal.var.idx()) & 1 == 1) == literal.sign;
    (0..1u64 << expansion.num_vars)
        .map(|mask| {
            let agree = expansion.classes.iter().all(|class| class.iter().all(|l| value(mask, l) == value(mask, &class[0])));
            let constants = expansion.constants.iter().all(|l| value(mask, l));
            let reduced = expansion.classes.iter().enumerate().map(|(r, class)| u64::from(value(mask, &class[0])) << r).sum::<u64>();
            agree && constants && truth[reduced as usize]
        })
        .collect()
}

/// A random expansion of `n` variables: classes of one to three signed
/// variables, up to two constants and up to two free variables, on a shuffled
/// numbering.
fn random_expansion(rng: &mut Lcg, n: u32) -> VariableExpansion {
    let sizes: Vec<u32> = (0..n).map(|_| 1 + rng.below(3) as u32).collect();
    let (constants, free) = (rng.below(3) as u32, rng.below(3) as u32);
    let num_vars = sizes.iter().sum::<u32>() + constants + free;
    let mut ids: Vec<u32> = (1..=num_vars).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut next = ids.into_iter();
    let mut take = |rng: &mut Lcg| Literal::new(VarId(next.next().unwrap()), rng.coin());
    let classes: Vec<Vec<Literal>> = sizes.iter().map(|&size| (0..size).map(|_| take(rng)).collect()).collect();
    let constants: Vec<Literal> = (0..constants).map(|_| take(rng)).collect();
    let free: Vec<VarId> = (0..free).map(|_| take(rng).var).collect();
    VariableExpansion { classes, constants, free, num_vars }
}

#[test]
fn a_negative_singleton_exchanges_the_literals_and_keeps_true() {
    let eng = Engine::new();
    let expansion = VariableExpansion { classes: vec![vec![neg(1)]], num_vars: 1, ..VariableExpansion::default() };
    for (local, expected) in [(POS_LEAF_IDX, NEG_LEAF_IDX), (NEG_LEAF_IDX, POS_LEAF_IDX), (ONE_LEAF_IDX, ONE_LEAF_IDX)] {
        let f = leaf_diagram(&eng, local);
        assert_canonical(&f);
        let g = eng.expand_variables(&f, &expansion).unwrap();
        assert_canonical(&g);
        assert_eq!(g.output().local, expected);
    }
}

#[test]
fn a_class_with_constants_and_a_free_variable_reconstructs_the_count() {
    let eng = Engine::new();
    let expansion = VariableExpansion {
        classes: vec![vec![pos(1), neg(2), pos(3)]],
        constants: vec![pos(4), neg(5)],
        free: vec![VarId(6)],
        num_vars: 6,
    };
    for (local, count) in [(POS_LEAF_IDX, 2u32), (NEG_LEAF_IDX, 2), (ONE_LEAF_IDX, 4)] {
        let f = leaf_diagram(&eng, local);
        assert_canonical(&f);
        let g = eng.expand_variables(&f, &expansion).unwrap();
        assert_canonical(&g);
        let mut vars: Vec<u32> = g.vtree().leaf_bottomup().map(|(_, var)| var.0).collect();
        vars.sort_unstable();
        assert_eq!(vars, [1, 2, 3, 4, 5, 6]);
        assert_eq!(g.vtree().num_vars(), 6);
        assert_eq!(g.model_count().unwrap(), count.into());
    }
}

/// On every vtree shape, random functions and random expansions agree with
/// the expansion's definition on every assignment.
#[test]
fn random_expansions_hold_exactly_on_the_expanded_models() {
    let mut rng = Lcg::new(0x5eed_e4a9);
    for n in 1..=4u32 {
        for (shape, vtree) in vtree_shapes(n) {
            for _ in 0..3 {
                let eng = Engine::new();
                let clauses = rand_cnf(&mut rng, n, CnfShape { clauses: n as usize + 1, width: 3 });
                let f = compile_clauses_on(&eng, &vtree, &clauses);
                assert_canonical(&f);
                let expansion = random_expansion(&mut rng, n);
                let g = eng.expand_variables(&f, &expansion).unwrap();
                assert_canonical(&g);
                assert_eq!(g.vtree().num_vars(), expansion.num_vars);
                assert!(Arc::ptr_eq(g.context(), f.context()), "{shape}");
                assert_eq!(models(&g, expansion.num_vars), expected_models(&truth_table(n, &clauses), &expansion), "{shape}: {clauses:?} under {expansion:?}");
            }
        }
    }
}

#[test]
fn the_constants_false_and_true_diagrams_expand() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let expansion = VariableExpansion {
        classes: vec![vec![pos(2), pos(5)], vec![neg(1)]],
        constants: vec![neg(3)],
        free: vec![VarId(4)],
        num_vars: 5,
    };
    let zero = eng.expand_variables(&Tdd::zero(&vtree), &expansion).unwrap();
    assert_canonical(&zero);
    assert!(zero.is_zero());
    assert_eq!(zero.vtree().num_leaves(), 5);
    // True over the reduced variables: each class agrees, the constant holds
    // and variable 4 is free.
    let one = eng.expand_variables(&Tdd::one(&vtree), &expansion).unwrap();
    assert_canonical(&one);
    assert_eq!(one.model_count().unwrap(), 8u32.into());
}

/// The expansion follows the diagram's own vtree, rotated or not, and keeps
/// its shape above the classes.
#[test]
fn the_result_vtree_is_the_expanded_leaves_grafted_with_constants_and_free_variables() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::random(3, 7));
    let f = compile_clauses_on(&eng, &vtree, &[vec![1, -2], vec![2, 3]]);
    assert_canonical(&f);
    let expansion = VariableExpansion {
        classes: vec![vec![pos(1)], vec![pos(2), neg(3)], vec![neg(4)]],
        constants: vec![pos(5), pos(6)],
        free: vec![VarId(7)],
        num_vars: 7,
    };
    let g = eng.expand_variables(&f, &expansion).unwrap();
    assert_canonical(&g);
    let classes = vtree.expand_leaves(|v| expansion.classes[v.idx()].iter().map(|l| l.var), 7).unwrap();
    let constants = Vtree::balanced_over(&[VarId(5), VarId(6)]).unwrap();
    let expected = Vtree::graft(&[classes, constants], &[VarId(7)]).unwrap();
    assert_eq!(g.vtree().to_text(), expected.to_text());
}

#[test]
fn an_identity_expansion_keeps_the_function() {
    let eng = Engine::new();
    for (shape, vtree) in vtree_shapes(4) {
        let clauses = [vec![1, -3], vec![2, 4], vec![-1, -4]];
        let f = compile_clauses_on(&eng, &vtree, &clauses);
        assert_canonical(&f);
        let g = eng.expand_variables(&f, &identity(&vtree)).unwrap();
        assert_canonical(&g);
        assert_eq!(g.vtree().to_text(), vtree.to_text(), "{shape}");
        assert_eq!(models(&g, 4), truth_table(4, &clauses), "{shape}");
    }
}

#[test]
fn a_weighted_diagram_expands_without_its_weights() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    f.set_weights(WeightStore::new(RationalWeights::unit(2), Arithmetic::ExactRational)).unwrap();
    let expansion = VariableExpansion { classes: vec![vec![pos(3)], vec![pos(1), pos(2)]], num_vars: 3, ..VariableExpansion::default() };
    let g = eng.expand_variables(&f, &expansion).unwrap();
    assert_canonical(&g);
    assert!(g.weights().is_none());
    assert_eq!(g.model_count().unwrap(), 3u32.into());
}

#[test]
fn a_diagram_with_summed_out_levels_is_refused() {
    for (f, _) in marginal_diagrams(3, 4, 3..6) {
        let eng = Engine::new();
        let error = eng.expand_variables(&f, &identity(f.vtree())).unwrap_err();
        assert!(matches!(error, ExpandError::Operation(OperationError::MarginalLevel(_))), "{error:?}");
    }
}

#[test]
fn expansions_that_do_not_fit_the_diagram_are_refused() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&vtree, [1, -2]).unwrap();
    assert_canonical(&f);
    let refused = |expansion: VariableExpansion| eng.expand_variables(&f, &expansion).unwrap_err();
    let base = || VariableExpansion { classes: vec![vec![pos(1)], vec![neg(2), pos(3)]], num_vars: 5, ..VariableExpansion::default() };

    assert_eq!(refused(VariableExpansion { classes: vec![vec![pos(1)]], ..base() }), ExpandError::MissingClass { variable: VarId(2) });
    assert_eq!(refused(VariableExpansion { classes: vec![vec![pos(1)], vec![]], ..base() }), ExpandError::MissingClass { variable: VarId(2) });
    let mut extra = base();
    extra.classes.push(vec![pos(4)]);
    assert_eq!(refused(extra), ExpandError::ClassWithoutLeaf { variable: VarId(3) });
    let mut empty_extra = base();
    empty_extra.classes.push(vec![]);
    assert!(eng.expand_variables(&f, &empty_extra).is_ok(), "an empty class needs no leaf");

    assert_eq!(refused(VariableExpansion { free: vec![VarId(6)], ..base() }), ExpandError::VariableOutOfRange { variable: VarId(6), num_vars: 5 });
    assert_eq!(refused(VariableExpansion { constants: vec![pos(0)], ..base() }), ExpandError::VariableOutOfRange { variable: VarId(0), num_vars: 5 });
    for clash in [
        VariableExpansion { classes: vec![vec![pos(1)], vec![neg(2), pos(2)]], ..base() },
        VariableExpansion { constants: vec![neg(3)], ..base() },
        VariableExpansion { constants: vec![pos(4)], free: vec![VarId(4)], ..base() },
        VariableExpansion { free: vec![VarId(5), VarId(5)], ..base() },
    ] {
        let variable = refused(clash);
        assert!(matches!(variable, ExpandError::Vtree(VtreeError::OverlappingVariable(_))), "{variable:?}");
    }
}

#[test]
fn an_armed_stop_comes_before_the_expansion_is_checked() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Stop))));
    let missing = VariableExpansion::default();
    assert_eq!(eng.expand_variables(&f, &missing).unwrap_err(), ExpandError::Operation(OperationError::Stopped));
}

/// A stop at a poll of the expansion, in the class rebuild, the constants'
/// cube or the graft, is reported as refused work, and the same engine then
/// expands the diagram in full.
#[test]
fn a_stop_at_a_poll_is_refused_work_and_a_retry_expands() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = compile_clauses_on(&eng, &vtree, &[vec![1, 2], vec![-2, 3]]);
    assert_canonical(&f);
    let expansion = VariableExpansion {
        classes: vec![vec![pos(1), neg(4)], vec![pos(2)], vec![neg(3), pos(5), pos(6)]],
        constants: vec![pos(7), neg(8)],
        free: vec![VarId(9)],
        num_vars: 9,
    };
    let expected = f.model_count().unwrap() * 2u32;
    let polls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&polls);
    let counting = StopCallback::new(move |_, _| {
        seen.fetch_add(1, Ordering::Relaxed);
        StopDecision::Continue
    });
    {
        let _full = eng.limits().edit(|config| config.with_stop_callback(Some(counting)));
        let g = eng.expand_variables(&f, &expansion).unwrap();
        assert_canonical(&g);
        assert_eq!(g.model_count().unwrap(), expected);
    }
    let total = polls.load(Ordering::Relaxed);
    let mut stopped = 0;
    for allowed in 0..total {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let stop_after = StopCallback::new(move |_, _| {
            if seen.fetch_add(1, Ordering::Relaxed) >= allowed { StopDecision::Stop } else { StopDecision::Continue }
        });
        let _scope = eng.limits().edit(|config| config.with_stop_callback(Some(stop_after)));
        match eng.expand_variables(&f, &expansion) {
            Err(ExpandError::Operation(OperationError::Stopped)) => stopped += 1,
            Err(error) => panic!("stopped at poll {allowed}: {error}"),
            Ok(g) => {
                assert_canonical(&g);
                assert_eq!(g.model_count().unwrap(), expected, "stopped at poll {allowed}");
            }
        }
    }
    // The entry, the step to the constants and the step to the graft each
    // test the stop.
    assert!(stopped >= 3, "{stopped} of {total} polls stopped the expansion");
    let retry = eng.expand_variables(&f, &expansion).unwrap();
    assert_canonical(&retry);
    assert_eq!(retry.model_count().unwrap(), expected);
}
