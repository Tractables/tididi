//! The probe join against the scatter it replaces: the same diagram, the
//! same count and the same quantified diagram, on operands that share a
//! block of variables, laid out every way the shared block and the blocks
//! each operand pairs with it can sit in a vtree, and on random operands at
//! every vtree shape; each probe taken where it prices, and wherever it is
//! admissible.

use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, assert_same_shape, rand_conj_over, same_storage, vtree_shapes, Lcg};
use crate::test_helpers::check::check_no_false_nodes_in_levels;
use crate::vtree::{VarId, Vtree};

/// Bits of the shared block, of the narrow operand's own block and of the
/// wide operand's.
const SHARED_BITS: u32 = 6;
const NARROW_BITS: u32 = 3;
const WIDE_BITS: u32 = 4;

/// The variables of a block of `bits` bits starting after `first - 1`.
fn block(first: u32, bits: u32) -> Vec<VarId> {
    (first..first + bits).map(VarId).collect()
}

/// The shared, narrow and wide blocks, in that variable order.
fn blocks() -> [Vec<VarId>; 3] {
    [block(1, SHARED_BITS), block(1 + SHARED_BITS, NARROW_BITS), block(1 + SHARED_BITS + NARROW_BITS, WIDE_BITS)]
}

/// Vtrees over the three blocks: the narrow block beside the shared one on
/// either side, the shared block above both, and the narrow block across the
/// root from the shared one; each with balanced and linear blocks.
fn shared_block_vtrees() -> Vec<(String, Arc<Vtree>)> {
    let [k, a, x] = blocks();
    let mut out = Vec::new();
    for (blocks_name, shape) in [("balanced", true), ("linear", false)] {
        let sub = |vars: &[VarId]| {
            if shape { Vtree::balanced_over(vars).unwrap() } else { Vtree::linear_from_order(vars).unwrap() }
        };
        let (k, a, x) = (sub(&k), sub(&a), sub(&x));
        let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
        for (name, vtree) in [
            ("((k a) x)", join(&join(&k, &a), &x)),
            ("((a k) x)", join(&join(&a, &k), &x)),
            ("(x (k a))", join(&x, &join(&k, &a))),
            ("(k (x a))", join(&k, &join(&x, &a))),
            ("((k x) a)", join(&join(&k, &x), &a)),
        ] {
            out.push((format!("{name}, {blocks_name} blocks"), Arc::new(vtree)));
        }
    }
    out
}

/// The narrow operand's models, at most one per value of the shared block
/// and a value of its own block each, and the wide operand's, several per
/// value of the shared block, some values the narrow one lacks; the shared
/// block in the low bits of each model, then the operand's own block.
fn shared_block_models(rng: &mut Lcg) -> (Vec<u64>, Vec<u64>) {
    let values = 1u64 << SHARED_BITS;
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    for k in 0..values {
        if rng.below(4) != 0 {
            narrow.push(k | (rng.below(1 << NARROW_BITS) << SHARED_BITS));
        }
        for _ in 0..rng.below(4) {
            wide.push(k | (rng.below(1 << WIDE_BITS) << SHARED_BITS));
        }
    }
    (narrow, wide)
}

/// The wide and the narrow operand on `vtree`, minimized.
fn shared_block_operands(vtree: &Arc<Vtree>, rng: &mut Lcg) -> (Tdd, Tdd) {
    let [k, a, x] = blocks();
    let (narrow, wide) = shared_block_models(rng);
    let vars = |attr: &[VarId]| k.iter().chain(attr).copied().collect::<Vec<_>>();
    let mut d = Tdd::from_models(vtree, &vars(&a), &narrow).unwrap();
    let mut f = Tdd::from_models(vtree, &vars(&x), &wide).unwrap();
    d.minimize().unwrap();
    f.minimize().unwrap();
    (f, d)
}

/// The sparse gate at its floor, so every level the grid would take is
/// joined sparsely and reaches the probe.
const EVERY_LEVEL_SPARSE: SparseRoute = SparseRoute { sparsity: 1, min_grid: 0 };

/// Conjoin both ways round with the probe open and closed, and require the
/// same diagram, with no false node, canonical once minimized. One engine
/// serves a whole test, so each level finds the workspace a wider or a
/// narrower one left.
fn same_as_scatter(eng: &Engine, what: &str, f: &Tdd, g: &Tdd, route: Option<SparseRoute>, always: bool) {
    for (a, b) in [(f, g), (g, f)] {
        let _scope = route.map(|r| eng.limits().edit(|c| c.with_sparse_route(r)));
        let conjoin = || eng.and(a.clone(), b.clone()).unwrap();
        let mut out = if always { always_probe(conjoin) } else { conjoin() };
        let oracle = no_probe(conjoin);
        check_no_false_nodes_in_levels(&out).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_same_shape(&out, &oracle, what);
        eng.minimize(&mut out).unwrap();
        assert_canonical(&out);
    }
}

#[test]
fn shared_block_conjunctions_are_the_scatters_own() {
    let before = probe_census();
    let mut rng = Lcg::new(0x6e_7a01);
    let eng = Engine::new();
    for (name, vtree) in shared_block_vtrees() {
        for round in 0..3 {
            let (f, d) = shared_block_operands(&vtree, &mut rng);
            let what = format!("{name}, round {round}");
            same_as_scatter(&eng, &what, &f, &d, None, false);
            same_as_scatter(&eng, &what, &f, &d, Some(EVERY_LEVEL_SPARSE), false);
            same_as_scatter(&eng, &what, &f, &d, Some(EVERY_LEVEL_SPARSE), true);
            hashed_pairs(|| same_as_scatter(&eng, &what, &f, &d, Some(EVERY_LEVEL_SPARSE), true));
        }
    }
    let census = probe_census();
    for (mode, name) in ["by left", "by right", "by pair"].iter().enumerate() {
        assert!(census[mode] > before[mode], "no level was probed {name}");
    }
}

#[test]
fn random_operands_are_the_scatters() {
    let before = probe_census();
    let mut rng = Lcg::new(0x9_0be);
    let eng = Engine::new();
    for (shape, vtree) in vtree_shapes(10) {
        let vars: Vec<u32> = (1..=10).collect();
        for round in 0..4 {
            let mut f = rand_conj_over(&vtree, &vars, 10, 4, false, &mut rng);
            let mut g = rand_conj_over(&vtree, &vars, 10, 4, false, &mut rng);
            f.minimize().unwrap();
            g.minimize().unwrap();
            let what = format!("{shape}, round {round}");
            same_as_scatter(&eng, &what, &f, &g, Some(EVERY_LEVEL_SPARSE), true);
            hashed_pairs(|| same_as_scatter(&eng, &what, &f, &g, Some(EVERY_LEVEL_SPARSE), true));
        }
    }
    let census = probe_census();
    assert!(census.iter().zip(before).all(|(now, was)| *now > was), "a probe never ran: {census:?}");
}

/// The count of a conjunction the probe built levels of, and the
/// conjunction that quantifies the shared block on the way, are the
/// scatter's own.
#[test]
fn counts_and_quantified_conjunctions_agree() {
    let mut rng = Lcg::new(0xc0_1d);
    let shared = blocks()[0].clone();
    for (name, vtree) in shared_block_vtrees() {
        let (f, d) = shared_block_operands(&vtree, &mut rng);
        for always in [false, true] {
            let eng = Engine::new();
            let _scope = eng.limits().scope(LimitConfig::none().with_sparse_route(EVERY_LEVEL_SPARSE));
            let probed = |op: &dyn Fn() -> Tdd| if always { always_probe(op) } else { op() };
            let count = if always {
                always_probe(|| eng.and_model_count(f.clone(), d.clone(), &[]).unwrap())
            } else {
                eng.and_model_count(f.clone(), d.clone(), &[]).unwrap()
            };
            let oracle = no_probe(|| eng.and_model_count(f.clone(), d.clone(), &[]).unwrap());
            assert_eq!(count, oracle, "{name}: count");
            let out = probed(&|| eng.and_exists(f.clone(), d.clone(), &shared).unwrap());
            let oracle = no_probe(|| eng.and_exists(f.clone(), d.clone(), &shared).unwrap());
            assert!(eng.equivalent(&out, &oracle).unwrap(), "{name}: and_exists");
        }
    }
}

/// A conjunction refused at any work point while the probe builds its levels
/// gives both operands back as they were, and leaves the engine's sparse
/// workspace fit for the next conjunction.
#[test]
fn a_refused_probe_leaves_operands_and_workspace_intact() {
    let mut rng = Lcg::new(0x5_70b);
    let (name, vtree) = shared_block_vtrees().swap_remove(0);
    let (f, d) = shared_block_operands(&vtree, &mut rng);
    let eng = Engine::new();
    let _route = eng.limits().scope(LimitConfig::none().with_sparse_route(EVERY_LEVEL_SPARSE));
    let expected = no_probe(|| eng.and(f.clone(), d.clone()).unwrap());
    let before = probe_census();
    let mut refusals = 0;
    for n in 0.. {
        let start = eng.limits().work_units();
        let rules = StopRules { unconditional: Some(StopAt::WorkUnits(start + n)), after_pairs: None };
        let scope = eng.limits().edit(|config| config.with_stop_rules(rules));
        // Boxed: the refusal carries both operands back.
        let outcome = always_probe(|| eng.and_restoring(f.clone(), d.clone()).map_err(Box::new));
        drop(scope);
        match outcome {
            Ok(out) => {
                assert!(eng.equivalent(&out, &expected).unwrap(), "{name}: granted at {n}: a different function");
                break;
            }
            Err(refused) => {
                assert!(same_storage(&refused.f, &f) && same_storage(&refused.g, &d), "{name}: refused at {n}: an operand changed");
                refusals += 1;
            }
        }
        assert!(n < 100_000, "never granted");
    }
    assert!(refusals > 0, "never refused");
    assert!(probe_census().iter().zip(before).any(|(now, was)| *now > was), "the probe never ran");
    // The workspace the refusals left behind joins the next level right.
    let again = always_probe(|| eng.and(f.clone(), d.clone()).unwrap());
    assert_same_shape(&again, &expected, &name);
}
