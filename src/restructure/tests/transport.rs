//! `Engine::transport` against `Engine::embed` followed by a trim.

use std::borrow::Cow;
use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::diagram::TddBuildError;
use crate::restructure::{EmbedError, TransportError};
use crate::test_helpers::assert_canonical;
use crate::test_helpers::check::check_trimmed;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx, VtreeNode};
use crate::{Engine, OperationError, Tdd};

/// `vtree` with every variable renamed through `rename` and the children of
/// each node swapped with probability one half.
fn renamed_mirrored(vtree: &Vtree, rename: &[VarId], num_vars: u32, rng: &mut Lcg) -> Vtree {
    let mut nodes = Vec::with_capacity(vtree.num_nodes());
    let mut copy_of = vec![VtreeIdx(0); vtree.num_nodes()];
    for t in vtree.bottomup() {
        copy_of[t.idx()] = VtreeIdx(nodes.len() as u32);
        nodes.push(match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var: rename[var.idx()], parent: None },
            VtreeNode::Internal { left, right, .. } => {
                let (left, right) = if rng.coin() { (right, left) } else { (left, right) };
                VtreeNode::Internal { left: copy_of[left.idx()], right: copy_of[right.idx()], parent: None }
            }
        });
    }
    Vtree::from_nodes(nodes, copy_of[vtree.root().idx()], num_vars).unwrap()
}

/// A random permutation of the variables `1..=n`, indexed by `VarId::idx`.
fn permutation(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    let mut vars: Vec<VarId> = (1..=n).map(VarId).collect();
    for i in (1..vars.len()).rev() {
        vars.swap(i, rng.below(i as u64 + 1) as usize);
    }
    vars
}

fn inverse(perm: &[VarId]) -> Vec<VarId> {
    let mut inv = vec![VarId(0); perm.len()];
    for (i, &v) in perm.iter().enumerate() {
        inv[v.idx()] = VarId(i as u32 + 1);
    }
    inv
}

/// A random CNF of one to five clauses over `vtree`'s leaves, trimmed when
/// `trim` is set.
fn diagram(eng: &Engine, vtree: &Arc<Vtree>, trim: bool, rng: &mut Lcg) -> Tdd {
    let vars: Vec<VarId> = vtree.leaf_bottomup().map(|(_, var)| var).collect();
    let mut f = Tdd::one(vtree);
    for _ in 0..1 + rng.below(5) {
        let clause: Vec<i32> = (0..1 + rng.below(3))
            .map(|_| {
                let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    if trim {
        eng.trim(&mut f, &Home::new(Arc::clone(vtree))).unwrap();
    }
    f
}

fn leaf_vars(vtree: &Vtree) -> Vec<VarId> {
    let mut vars: Vec<VarId> = vtree.leaf_bottomup().map(|(_, var)| var).collect();
    vars.sort_unstable();
    vars
}

/// A table of `n` variables, `weights[i % weights.len()]` for variable
/// `i + 1`.
fn table(n: u32, weights: &[(i64, i64)]) -> WeightStore {
    let sixths = |k: i64| BigRational::new(k.into(), 6.into());
    let table: Vec<_> = (0..n as usize)
        .map(|i| weights[i % weights.len()])
        .map(|(negative, positive)| LiteralWeights { negative: sixths(negative), positive: sixths(positive) })
        .collect();
    WeightStore::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)
}

/// One random pair of weights for every variable, so that any renaming
/// keeps them.
fn store(n: u32, rng: &mut Lcg) -> WeightStore {
    table(n, &[(1 + rng.below(5) as i64, 1 + rng.below(5) as i64)])
}

/// `f` is `g` as it was: the same vtree, levels, function and knowledge.
fn same_as(eng: &Engine, f: &Tdd, g: &Tdd) {
    assert!(Arc::ptr_eq(f.vtree(), g.vtree()));
    assert_eq!((f.node_count(), f.pair_count()), (g.node_count(), g.pair_count()));
    assert_eq!(f.levels.is_canonical(f.output), g.levels.is_canonical(g.output));
    assert_eq!(f.weights().is_some(), g.weights().is_some());
    assert!(eng.equivalent(f, g).unwrap());
}

/// One diagram `f` on a projection of a home that is `dest`'s home renamed
/// and mirrored at random, or on a root vtree of that shape; `back` takes
/// its variables to `dest`'s.
struct Case {
    dest: Home,
    f: Tdd,
    back: Vec<VarId>,
}

fn case(eng: &Engine, seed: u64, rng: &mut Lcg) -> Case {
    let n = 2 + rng.below(11) as u32;
    let dest = Home::new(Vtree::random(n, seed));
    let perm = permutation(n, rng);
    let back = inverse(&perm);
    let source_home = Home::new(renamed_mirrored(dest.vtree(), &perm, n, rng));
    let vars: Vec<VarId> = (1..=n).map(VarId).filter(|_| rng.below(3) != 0).collect();
    let vars = if vars.is_empty() { vec![VarId(1 + rng.below(n as u64) as u32)] } else { vars };
    let vtree = match rng.below(3) {
        0 => {
            let own = Vtree::projection(&dest, vars.iter().map(|v| back[v.idx()])).unwrap();
            Arc::clone(Home::new(renamed_mirrored(&own, &perm, n, rng)).vtree())
        }
        _ => Vtree::projection(&source_home, vars).unwrap(),
    };
    let f = diagram(eng, &vtree, rng.below(4) != 0, rng);
    Case { dest, f, back }
}

/// The transport of `f`, owned or borrowed, checked against the embed and
/// trim; a borrowed `f` must come through intact.
fn check_against_embed(eng: &Engine, c: &Case, owned: bool, weighted: Option<&WeightStore>) {
    let map = |v: VarId| c.back[v.idx()];
    let (f, dest) = (&c.f, &c.dest);
    let pairs = f.pair_count();
    let (got, stats) = match owned {
        true => eng.transport(f.clone(), dest, map, 0),
        false => eng.transport(f, dest, map, 0),
    }
    .map_err(|r| r.error)
    .unwrap();
    assert_eq!(f.pair_count(), pairs);
    assert_eq!((stats.rotations, stats.peak_pairs), (0, pairs));
    if f.vtree().num_nodes() == 0 {
        assert_eq!(got.vtree().num_nodes(), 0);
        assert!(Arc::ptr_eq(got.home().unwrap().vtree(), dest.vtree()));
        assert_eq!((got.is_one(), got.is_zero()), (f.is_one(), f.is_zero()));
        assert_eq!(got.weights().is_some(), weighted.is_some());
        return;
    }

    let (mut expected, _) = eng.embed(f, dest.vtree(), map).unwrap();
    if let Some(ws) = weighted {
        expected.set_weights(ws.clone()).unwrap();
    }
    eng.trim(&mut expected, dest).unwrap();
    let unweighted = |t: &Tdd| {
        let mut t = t.clone();
        t.take_weights().unwrap();
        t
    };
    assert!(eng.equivalent(&unweighted(&got), &unweighted(&expected)).unwrap());
    assert_eq!(got.vtree().num_leaves(), f.vtree().num_leaves());
    let images: Vec<VarId> = {
        let mut v: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, var)| map(var)).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(leaf_vars(got.vtree()), images);
    assert!(Arc::ptr_eq(got.home().unwrap().vtree(), dest.vtree()));
    let trimmed = check_trimmed(f, false).is_ok();
    if trimmed {
        check_trimmed(&got, false).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(leaf_vars(got.vtree()), leaf_vars(expected.vtree()));
    }
    assert_eq!(eng.model_count(&got).unwrap(), eng.model_count(&expected).unwrap());
    if weighted.is_some() {
        assert!(got.weights().is_some());
        assert_eq!(
            eng.weighted_value(&got).unwrap().map(|w| w.into_rational()),
            eng.weighted_value(&expected).unwrap().map(|w| w.into_rational()),
        );
    }
    if trimmed {
        let (mut got, mut expected) = (got, expected);
        eng.minimize(&mut got).unwrap();
        eng.minimize(&mut expected).unwrap();
        assert_eq!((got.node_count(), got.pair_count()), (expected.node_count(), expected.pair_count()));
    }
}

#[test]
fn transport_is_embed_then_trim() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x7a5_0b0e);
    for seed in 0..300 {
        let c = case(&eng, seed, &mut rng);
        check_against_embed(&eng, &c, rng.coin(), None);
        let mut weighted = Case { dest: c.dest.clone(), f: c.f.clone(), back: c.back.clone() };
        let ws = store(c.dest.vtree().num_vars(), &mut rng);
        weighted.f.set_weights(ws.clone()).unwrap();
        check_against_embed(&eng, &weighted, rng.coin(), Some(&ws));
    }
}

/// Where the embed finds no level-for-level copy, `bound` 0 refuses with `f`
/// as given, and without a bound the move equals `restructure_to` onto the
/// projection's shape in `f`'s names followed by the renaming.
#[test]
fn a_map_that_breaks_the_shape_rotates_as_restructure_to_then_renames() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5ba9e);
    let (mut accepted, mut refused) = (0, 0);
    for seed in 0..300 {
        let n = 3 + rng.below(8) as u32;
        let dest = Home::new(Vtree::random(n, seed));
        let source = Arc::new(Vtree::random(n, seed ^ 0xffff));
        let f = diagram(&eng, &source, false, &mut rng);
        let perm = permutation(n, &mut rng);
        let map = |v: VarId| perm[v.idx()];
        let embedded = eng.embed(&f, dest.vtree(), map);
        let (pairs, vtree) = (f.pair_count(), Arc::clone(f.vtree()));
        let owned = eng.transport(f.clone(), &dest, map, 0);
        let borrowed = eng.transport(&f, &dest, map, 0);
        match embedded {
            Ok(_) => {
                accepted += 1;
                assert!(owned.is_ok() && borrowed.is_ok());
            }
            Err(EmbedError::NotIsomorphic { .. }) => {
                refused += 1;
                for r in [owned.unwrap_err(), borrowed.unwrap_err()] {
                    assert!(matches!(r.error, TransportError::Bound { .. }), "{:?}", r.error);
                    assert!(Arc::ptr_eq(r.tdd.vtree(), &vtree));
                    assert_eq!(r.tdd.pair_count(), pairs);
                    assert!(eng.equivalent(&r.tdd, &f).unwrap());
                }
                let (got, stats) = eng.transport(&f, &dest, map, usize::MAX).map_err(|r| r.error).unwrap();
                assert!(stats.rotations > 0 || f.is_zero(), "{stats:?}");
                assert_canonical(&got);
                let into = Vtree::projection(&dest, f.vtree().leaf_bottomup().map(|(_, v)| map(v))).unwrap();
                let shape = Arc::new(renamed_mirrored(&into, &inverse(&perm), n, &mut rng));
                let mut expected = f.clone();
                expected.restructure_to(&shape, usize::MAX).unwrap();
                let (mut expected, _) = eng.transport(expected, &dest, map, 0).map_err(|r| r.error).unwrap();
                expected.trim(&dest).unwrap();
                assert!(Arc::ptr_eq(got.home().unwrap().vtree(), dest.vtree()));
                assert_eq!(leaf_vars(got.vtree()), leaf_vars(expected.vtree()));
                assert_eq!((got.node_count(), got.pair_count()), (expected.node_count(), expected.pair_count()));
                assert!(eng.equivalent(&got, &expected).unwrap());
            }
            Err(e) => panic!("{e:?}"),
        }
    }
    assert!(accepted > 10 && refused > 10, "{accepted} accepted, {refused} refused");
}

/// Every refusal on the rotation path, at a bound or at any allocation (so
/// also after kept rotations of the copy), hands back an owned `f` as it was.
#[test]
fn a_refused_rotation_hands_back_f_as_it_was() {
    let mut rng = Lcg::new(11);
    let home = Home::new(Vtree::linear(10));
    let kept = [1u32, 2, 3, 5, 6, 8, 9];
    let own = Vtree::projection(&home, kept.map(VarId)).unwrap();
    let eng = Engine::new();
    let f = crate::test_helpers::rand_conj_over(&own, &kept, 8, 3, false, &mut rng);
    let f = eng.embed(&f, &own, |v| v).unwrap().0;
    let order = [1, 10, 2, 9, 3, 8, 4, 7, 5, 6].map(VarId);
    let other = Home::new(Vtree::balanced_over(&order).unwrap());
    let (_, stats) = eng.transport(&f, &other, |v| v, usize::MAX).map_err(|r| r.error).unwrap();
    assert!(stats.rotations > 1, "{stats:?}");
    let mut bound = 1;
    loop {
        let before = f.clone();
        match eng.transport(f.clone(), &other, |v| v, bound) {
            Ok(_) => break,
            Err(r) => {
                assert!(matches!(r.error, TransportError::Bound { .. }), "{:?}", r.error);
                assert!(matches!(r.tdd, Cow::Owned(_)));
                same_as(&eng, &r.tdd, &before);
            }
        }
        bound += 1;
    }
    assert!(bound > 1);
    let mut refused = 0;
    for cut in 0.. {
        let eng = Engine::new();
        eng.refuse_nth_reserve(cut);
        let before = f.clone();
        let result = eng.transport(f.clone(), &other, |v| v, usize::MAX);
        let pending = eng.limits().refusal_pending();
        match result {
            Ok(_) if pending => break,
            Ok(_) => {}
            Err(r) => {
                refused += 1;
                assert!(matches!(r.error, TransportError::Operation(OperationError::OverBudget)), "{:?}", r.error);
                same_as(&eng, &r.tdd, &before);
            }
        }
    }
    assert!(refused > 3, "{refused} refusals");
}

/// Under a renaming, a weighted `f` moves only where the table weighs each
/// image as the variable it renames, and then weighs the same.
#[test]
fn a_renaming_keeps_the_weights_or_is_refused() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3), VarId(4)]).unwrap();
    let f = eng.and_clause(eng.clause(&own, [1, -2, 3]).unwrap(), &[2, 4]).unwrap();
    let shift = |v: VarId| VarId(v.0 + 4);
    for (weights, kept) in [(&[(1, 5), (2, 4), (3, 3), (5, 1)][..], true), (&[(1, 5), (2, 4), (3, 3)][..], false)] {
        let mut weighted = f.clone();
        weighted.set_weights(table(8, weights)).unwrap();
        for bound in [0, usize::MAX] {
            match eng.transport(&weighted, &home, shift, bound) {
                Ok((g, _)) => {
                    assert!(kept);
                    assert_eq!(leaf_vars(g.vtree()), [5, 6, 7, 8].map(VarId));
                    let value = |t: &Tdd| eng.weighted_value(t).unwrap().map(|w| w.into_rational());
                    assert_eq!(value(&g), value(&weighted));
                }
                Err(r) => {
                    assert!(!kept);
                    assert_eq!(r.error, TransportError::Weights(TddBuildError::IncompatibleWeights));
                }
            }
        }
    }
    let mut weighted = f.clone();
    weighted.set_weights(table(8, &[(1, 5), (2, 4), (3, 3)])).unwrap();
    assert!(eng.transport(&weighted, &home, |v| v, 0).is_ok());
}

#[test]
fn bad_images_marginal_levels_and_constants() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let f = eng.clause(&own, [1, -2, 3]).unwrap();

    let r = eng.transport(&f, &home, |v| VarId(v.0 + 6), 0).unwrap_err();
    assert!(matches!(r.error, TransportError::Variable { variable: VarId(7..=9) }), "{:?}", r.error);
    let r = eng.transport(f.clone(), &home, |v| VarId(v.0.min(2)), 0).unwrap_err();
    assert_eq!(r.error, TransportError::Overlap { variable: VarId(2) });
    assert!(matches!(r.tdd, Cow::Owned(_)));

    let mut weighted = f.clone();
    weighted.set_weights(WeightStore::new(RationalWeights::unit(3), Arithmetic::ExactRational)).unwrap();
    let r = eng.transport(&weighted, &home, |v| VarId(v.0 + 3), 0).unwrap_err();
    assert!(matches!(r.error, TransportError::Weights(TddBuildError::MissingVariableWeight(_))), "{:?}", r.error);

    let mut summed = f.clone();
    crate::marginal::marginalize_levels(&eng, &mut summed, &[own.root()]).unwrap();
    let r = eng.transport(summed, &home, |v| v, usize::MAX).unwrap_err();
    assert!(matches!(r.error, TransportError::Operation(OperationError::MarginalLevel(_))), "{:?}", r.error);

    let other = Home::new(Vtree::linear(4));
    for value in [false, true] {
        let empty = Vtree::empty_projection(&home);
        let c = if value { Tdd::one(&empty) } else { Tdd::zero(&empty) };
        let (moved, _) = eng.transport(c, &other, |v| v, 0).map_err(|r| r.error).unwrap();
        assert_eq!(moved.vtree().num_nodes(), 0);
        assert!(Arc::ptr_eq(moved.home().unwrap().vtree(), other.vtree()));
        assert_eq!(moved.is_one(), value);
        assert_eq!(moved.is_zero(), !value);
    }
}

/// A clause over variables 1 to 3 of a balanced home of 8 with variables 1
/// and 2 summed into a region, and the region's level.
fn with_region(eng: &Engine, home: &Home) -> (Tdd, VtreeIdx) {
    let own = Vtree::projection(home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = eng.and_clause(eng.clause(&own, [1, -2, 3]).unwrap(), &[2, -3]).unwrap();
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    let region = f.vtree().regions().next().expect("variables 1 and 2 are summed into a region");
    assert!(f.level(region).is_marginal());
    (f, region)
}

/// Onto another home, only the weights of a summed-out region's variables
/// are compared: counts onto a weighted home, weight values onto an
/// unweighted one, and weights that differ at a variable of the region are
/// refused, with `f` given back.
#[test]
fn a_summed_region_moves_only_onto_the_weights_it_was_summed_under() {
    let eng = Engine::new();
    let weighted = |weights: &[(i64, i64)]| Home::weighted(Vtree::balanced(8), table(8, weights)).unwrap();
    let value = |t: &Tdd| eng.weighted_value(t).unwrap().map(|w| w.into_rational());

    let (counts, region) = with_region(&eng, &Home::new(Vtree::balanced(8)));
    let r = eng.transport(&counts, &weighted(&[(1, 5)]), |v| v, 0).unwrap_err();
    assert_eq!(r.error, TransportError::Weights(TddBuildError::CountLevelWithWeights { level: region }));
    assert!(matches!(r.tdd, Cow::Borrowed(_)));

    let source = weighted(&[(1, 5), (2, 4), (3, 3)]);
    let (f, region) = with_region(&eng, &source);
    assert!(f.weights.is_none() && f.weights().is_some());
    let r = eng.transport(f.clone(), &Home::new(Vtree::balanced(8)), |v| v, 0).unwrap_err();
    assert_eq!(r.error, TransportError::Weights(TddBuildError::WeightedLevelWithoutStore { level: region }));
    assert!(matches!(r.tdd, Cow::Owned(_)));

    // Variable 1 weighed otherwise: refused. Variable 3, a leaf and not in
    // the region, and variables 4 to 8, unused: not compared.
    for (weights, kept) in [
        (&[(2, 5), (2, 4), (3, 3)][..], false),
        (&[(1, 5), (2, 4), (1, 1), (6, 6)][..], true),
        (&[(1, 5), (2, 4), (3, 3)][..], true),
    ] {
        match eng.transport(&f, &weighted(weights), |v| v, 0) {
            Ok((g, _)) => {
                assert!(kept, "{weights:?}");
                assert!(g.weights.is_none());
                assert_eq!(g.vtree().regions().count(), 1);
                assert_canonical(&g);
                let mut again = eng.transport(&g, &source, |v| v, 0).unwrap().0;
                assert_eq!(value(&again), value(&f));
                eng.trim(&mut again, &source).unwrap();
                assert_eq!(value(&again), value(&f));
            }
            Err(r) => {
                assert!(!kept, "{weights:?}");
                assert_eq!(r.error, TransportError::Weights(TddBuildError::IncompatibleWeights));
            }
        }
    }

    let scaled = WeightStore::new(table(8, &[(1, 5), (2, 4), (3, 3)]).algebra().clone(), Arithmetic::ScaledFloat);
    let r = eng.transport(&f, &Home::weighted(Vtree::balanced(8), scaled).unwrap(), |v| v, 0).unwrap_err();
    assert_eq!(r.error, TransportError::Weights(TddBuildError::IncompatibleWeights));

    // Under a renaming the image is compared with the variable it renames.
    let shift = |v: VarId| VarId(v.0 + 4);
    let shifted = weighted(&[(1, 5), (2, 4), (3, 3), (3, 3), (1, 5), (2, 4), (3, 3), (3, 3)]);
    let g = eng.transport(&f, &shifted, shift, 0).unwrap().0;
    assert_eq!(value(&g), value(&f));
    let wrong = weighted(&[(1, 5), (2, 4), (3, 3), (3, 3), (1, 5), (2, 5)]);
    let r = eng.transport(&f, &wrong, shift, 0).unwrap_err();
    assert_eq!(r.error, TransportError::Weights(TddBuildError::IncompatibleWeights));

    // Weights a diagram carries are compared as a home's, and left behind.
    let home = Home::new(Vtree::balanced(8));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut carried = eng.and_clause(eng.clause(&own, [1, -2, 3]).unwrap(), &[2, -3]).unwrap();
    carried.set_weights(table(8, &[(1, 5), (2, 4), (3, 3)])).unwrap();
    eng.marginalize(&mut carried, |v| v.0 <= 2).unwrap();
    let g = eng.transport(&carried, &source, |v| v, 0).unwrap().0;
    assert!(g.weights.is_none());
    assert_eq!(value(&g), value(&carried));
    assert_eq!(value(&g), value(&f));
    let r = eng.transport(&carried, &weighted(&[(1, 4)]), |v| v, 0).unwrap_err();
    assert_eq!(r.error, TransportError::Weights(TddBuildError::IncompatibleWeights));
}

/// A diagram with no summed-out region stores no value of any weights:
/// it moves onto any home, and reads that home's.
#[test]
fn a_structural_diagram_moves_onto_any_weights() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = eng.clause(&own, [1, -2, 3]).unwrap();
    f.set_weights(table(8, &[(1, 5)])).unwrap();
    let onto = Home::weighted(Vtree::balanced(8), table(8, &[(2, 3), (1, 1)])).unwrap();
    let g = eng.transport(&f, &onto, |v| VarId(v.0 + 1), usize::MAX).unwrap().0;
    assert!(g.weights.is_none());
    assert!(Arc::ptr_eq(&g.vtree().home().unwrap().vtree().clone(), onto.vtree()));
    let structural = eng.clause(&Vtree::projection(&onto, [VarId(2), VarId(3), VarId(4)]).unwrap(), [2, -3, 4]).unwrap();
    let value = |t: &Tdd| eng.weighted_value(t).unwrap().map(|w| w.into_rational());
    assert_eq!(value(&g), value(&structural));
}
