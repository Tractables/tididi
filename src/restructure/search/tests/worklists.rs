//! What a rotation trial does with the reduction obligations it inherits.

use crate::test_helpers::bare_roots;
use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Dirty, Pass, Tdd};
use crate::restructure::search::{MinimizePairs, MultistartConfig, PoolMove, RotationMove, RotationSearchConfig};
use crate::restructure::search::probe::rotate_if_on;
use crate::test_helpers::check::marginal::check_pair_fusion_saturation;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses_on, rand_cnf, vtree_shapes, CnfShape};
use crate::vtree::rng::Lcg;
use crate::vtree::{RotationKind, Vtree, VtreeIdx};

/// The levels the diagram still owes each reduction pass, read without
/// consuming them.
fn owed(tdd: &Tdd) -> [Vec<u32>; 3] {
    let mut dirty = tdd.dirty.clone();
    [dirty.take(Pass::Contract), dirty.take(Pass::LeafContract), dirty.take(Pass::ContentTwin)]
}

#[test]
fn an_accepted_rotation_keeps_what_the_diagram_already_owed() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    // A conjunction puts the levels it rebuilt on the worklists, and nothing
    // drains them before the probe, so the diagram enters the trial owing work
    // the way a mid-compile caller's does.
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    let count = f.model_count().unwrap();
    let before = owed(&f);
    assert!(before.iter().any(|list| !list.is_empty()), "the fixture owes no pass anything");

    let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    let kept = rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| true);
    assert_eq!(kept, Ok(true));

    // A level absent from a pass's list is taken to be at that pass's fixpoint,
    // so a level dropped here is one the next reduce walks past for good.
    let after = owed(&f);
    for (pass, (was, now)) in before.iter().zip(after.iter()).enumerate() {
        for level in was {
            assert!(now.contains(level), "pass {pass} lost level {level}");
        }
    }
    // Underneath, not instead of: the rebuilt level is on the list as well.
    assert!(after[0].contains(&vtree.root().0), "the rotation recorded no level of its own");

    eng.minimize(&mut f).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), count);
}

/// The levels whose pair fusion record is set.
fn recorded(tdd: &Tdd) -> Vec<VtreeIdx> {
    (0..tdd.vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| tdd.dirty.is_fused(t.0)).collect()
}

/// Random formulas on every vtree shape with a level marginalized and the
/// result minimized, then rotated at random pivots: an accepted rotation
/// keeps the records of levels it neither rebuilt nor pushed, each of those
/// levels has no fusion left to find, and minimizing gives the diagram
/// minimizing with every record forgotten gives.
#[test]
fn an_accepted_rotation_keeps_the_fusion_records_it_left_alone() {
    let mut rng = Lcg::new(41);
    let mut kept_records = 0;
    for num_vars in 5..9u32 {
        for (shape, vtree) in bare_roots(vtree_shapes(num_vars)) {
            let eng = Engine::new();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 2 * num_vars as usize, width: 3 });
            let mut f = compile_clauses_on(&eng, &vtree, &clauses);
            if f.is_zero() {
                continue;
            }
            let inner: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32)
                .map(VtreeIdx)
                .filter(|&t| t != vtree.root() && !vtree.node(t).is_leaf())
                .collect();
            let target = inner[rng.below(inner.len() as u64) as usize];
            eng.marginalize_levels(&mut f, &[target]).unwrap();
            eng.minimize(&mut f).unwrap();
            let count = f.model_count().unwrap();
            for step in 0..6 {
                let label = format!("{shape} over {num_vars} variables, step {step}");
                let before = recorded(&f);
                let pivots: Vec<VtreeIdx> = (0..f.vtree.num_nodes() as u32)
                    .map(VtreeIdx)
                    .filter(|&t| !f.vtree.node(t).is_leaf())
                    .collect();
                let pivot = pivots[rng.below(pivots.len() as u64) as usize];
                let kind = if rng.below(2) == 0 { RotationKind::Left } else { RotationKind::Right };
                if rotate_if_on(&eng, &mut f, &[RotationMove { pivot, kind }], usize::MAX, |_| true) != Ok(true) {
                    continue;
                }
                let after = recorded(&f);
                kept_records += before.iter().filter(|t| after.contains(t)).count();
                if let Err(e) = check_pair_fusion_saturation(&f, Some(&after)) {
                    panic!("{label}: {e}");
                }
                let mut forced = f.clone();
                forced.dirty.forget_all_fused();
                eng.minimize(&mut f).unwrap();
                eng.minimize(&mut forced).unwrap();
                forced.dirty.forget_all_fused();
                assert_same_shape(&f, &forced, &label);
                assert_eq!(f.model_count().unwrap(), count, "{label}");
            }
        }
    }
    assert!(kept_records > 0, "no rotation kept a record");
}

#[test]
fn a_kept_rotation_of_an_uncertified_diagram_owes_its_outer_level_every_pass() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    eng.minimize(&mut f).unwrap();
    assert!(f.dirty.is_empty());
    f.levels.forget();
    let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    assert_eq!(rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| true), Ok(true));
    // What a release build leaves: the test build's locality check must not
    // drain the lists it checks.
    assert_eq!(owed(&f), [vec![vtree.root().0], vec![vtree.root().0], vec![vtree.root().0]]);
}

/// Whether a minimize of `f` that owed every level and forgot the
/// certificate would drop a node or a pair.
fn minimize_changes(eng: &Engine, f: &Tdd) -> bool {
    let mut g = f.clone();
    g.levels.forget();
    let internal = g.vtree.internal_bottomup_slice().to_vec();
    g.dirty = Dirty::default().seeded(&g.vtree, internal.into_iter(), None).unwrap();
    eng.minimize(&mut g).unwrap();
    let slots = |t: &Tdd| t.levels.iter().map(|l| l.slot_count()).collect::<Vec<_>>();
    g.pair_count() != f.pair_count() || slots(&g) != slots(f)
}

/// What the results of one operation showed.
#[derive(Default, Debug)]
struct Tally {
    results: usize,
    certified: usize,
    /// Canonical, yet uncertified: the next minimize runs passes that find
    /// nothing.
    spurious: usize,
    /// Results a minimize runs its passes on.
    passes: usize,
}

impl Tally {
    fn record(&mut self, eng: &Engine, f: &Tdd, label: &str) {
        let changes = minimize_changes(eng, f);
        self.results += 1;
        self.certified += usize::from(f.is_canonical());
        self.spurious += usize::from(!f.is_canonical() && !changes);
        self.passes += usize::from(!(f.is_canonical() && f.dirty.is_empty()));
        assert!(!(f.is_canonical() && changes), "{label}: certified, and a minimize changes it");
        assert_canonical(f);
    }
}

/// Random formulas on every vtree shape, minimized, through each rotation
/// operation: a result is certified exactly when a minimize would change
/// nothing.
#[test]
fn a_rotation_of_a_canonical_diagram_keeps_the_certificate() {
    let mut rng = Lcg::new(7);
    let mut tallies: [Tally; 4] = Default::default();
    for num_vars in 4..10u32 {
        for ((shape, vtree), round) in bare_roots(vtree_shapes(num_vars)).into_iter().flat_map(|s| [(s.clone(), 0), (s.clone(), 1), (s, 2)]) {
            let eng = Engine::new();
            let cnf = |rng: &mut Lcg| rand_cnf(rng, num_vars, CnfShape { clauses: 2 * num_vars as usize, width: 3 });
            let mut f = compile_clauses_on(&eng, &vtree, &cnf(&mut rng));
            let mut g = compile_clauses_on(&eng, &vtree, &cnf(&mut rng));
            eng.minimize(&mut f).unwrap();
            eng.minimize(&mut g).unwrap();
            if f.is_zero() || f.is_one() {
                continue;
            }
            let label = format!("{shape} over {num_vars} variables, round {round}");
            let pivots: Vec<VtreeIdx> = vtree.internal_bottomup_slice().to_vec();
            let kind = |rng: &mut Lcg| if rng.below(2) == 0 { RotationKind::Left } else { RotationKind::Right };

            for _ in 0..6 {
                let mut r = f.clone();
                let moves: Vec<RotationMove> = (0..1 + rng.below(2))
                    .map(|_| RotationMove { pivot: pivots[rng.below(pivots.len() as u64) as usize], kind: kind(&mut rng) })
                    .collect();
                if rotate_if_on(&eng, &mut r, &moves, usize::MAX, |_| true) == Ok(true) {
                    tallies[0].record(&eng, &r, &format!("rotate_if, {label}"));
                }
            }

            let mut r = f.clone();
            eng.rotation_search(&mut r, &mut MinimizePairs, &RotationSearchConfig::default()).unwrap();
            tallies[1].record(&eng, &r, &format!("rotation_search, {label}"));

            let mut r = f.clone();
            let mut config = MultistartConfig { restarts: 2, kick: 3, seed: rng.below(1000), ..MultistartConfig::default() };
            config.search.max_inner_pairs = 1 << 20;
            eng.rotation_multistart(&mut r, &mut MinimizePairs, &config).unwrap();
            tallies[2].record(&eng, &r, &format!("rotation_multistart, {label}"));

            if Arc::ptr_eq(f.vtree(), g.vtree()) {
                for _ in 0..4 {
                    let mut pool = [f.clone(), g.clone()];
                    let rotation = RotationMove { pivot: pivots[rng.below(pivots.len() as u64) as usize], kind: kind(&mut rng) };
                    let mv = PoolMove { rotation, crossed: rng.below(2) == 0 };
                    let [a, b] = &mut pool;
                    if eng.rotate_pool_if(&mut [a, b], mv, usize::MAX, |_| true) == Ok(true) {
                        for m in &pool {
                            tallies[3].record(&eng, m, &format!("rotate_pool_if {mv:?}, {label}"));
                        }
                    }
                }
            }
        }
    }
    let names = ["rotate_if", "rotation_search", "rotation_multistart", "rotate_pool_if"];
    for (name, tally) in names.iter().zip(&tallies) {
        eprintln!("{name}: {tally:?}");
    }
    for (name, tally) in names.iter().zip(&tallies) {
        assert!(tally.results > 0, "{name}: no result");
        assert_eq!(tally.spurious, 0, "{name}: canonical results left uncertified");
    }
}
