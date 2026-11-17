//! `Engine::transport` against `Engine::embed_mirrored` followed by a trim.

use std::borrow::Cow;
use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::restructure::EmbedError;
use crate::test_helpers::check::check_trimmed;
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
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
        eng.trim(&mut f).unwrap();
    }
    f
}

fn leaf_vars(vtree: &Vtree) -> Vec<VarId> {
    let mut vars: Vec<VarId> = vtree.leaf_bottomup().map(|(_, var)| var).collect();
    vars.sort_unstable();
    vars
}

fn store(n: u32, rng: &mut Lcg) -> WeightStore {
    let weight = |rng: &mut Lcg| BigRational::new((1 + rng.below(5) as i64).into(), 6.into());
    let table: Vec<_> =
        (0..n).map(|_| LiteralWeights { negative: weight(rng), positive: weight(rng) }).collect();
    WeightStore::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)
}

/// One diagram `f` on a projection of a home that is `dest`'s home renamed
/// and mirrored at random, or on a root vtree of that shape; `back` takes
/// its variables to `dest`'s.
struct Case {
    dest: Arc<Vtree>,
    f: Tdd,
    back: Vec<VarId>,
}

fn case(eng: &Engine, seed: u64, rng: &mut Lcg) -> Case {
    let n = 2 + rng.below(11) as u32;
    let dest = Arc::new(Vtree::random(n, seed));
    let perm = permutation(n, rng);
    let back = inverse(&perm);
    let source_home = Arc::new(renamed_mirrored(&dest, &perm, n, rng));
    let vars: Vec<VarId> = (1..=n).map(VarId).filter(|_| rng.below(3) != 0).collect();
    let vars = if vars.is_empty() { vec![VarId(1 + rng.below(n as u64) as u32)] } else { vars };
    let vtree = match rng.below(3) {
        0 => {
            let own = Vtree::projection(&dest, vars.iter().map(|v| back[v.idx()])).unwrap();
            Arc::new(renamed_mirrored(&own, &perm, n, rng))
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
    let got = match owned {
        true => eng.transport(f.clone(), dest, map),
        false => eng.transport(f, dest, map),
    }
    .map_err(|r| r.error)
    .unwrap();
    assert_eq!(f.pair_count(), pairs);
    if f.vtree().num_nodes() == 0 {
        assert_eq!(got.vtree().num_nodes(), 0);
        assert!(Arc::ptr_eq(got.vtree().home().unwrap(), dest));
        assert_eq!((got.is_one(), got.is_zero()), (f.is_one(), f.is_zero()));
        assert_eq!(got.weights().is_some(), weighted.is_some());
        return;
    }

    let (mut expected, _) = eng.embed_mirrored(f, dest, map).unwrap();
    if let Some(ws) = weighted {
        expected.set_weights(ws.clone()).unwrap();
    }
    eng.trim(&mut expected).unwrap();
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
    assert!(Arc::ptr_eq(got.vtree().home().unwrap(), dest));
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
        let mut weighted = Case { dest: Arc::clone(&c.dest), f: c.f.clone(), back: c.back.clone() };
        let ws = store(c.dest.num_vars(), &mut rng);
        weighted.f.set_weights(ws.clone()).unwrap();
        check_against_embed(&eng, &weighted, rng.coin(), Some(&ws));
    }
}

#[test]
fn a_map_that_breaks_the_shape_is_refused_as_embed_refuses_it() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5ba9e);
    let (mut accepted, mut refused) = (0, 0);
    for seed in 0..300 {
        let n = 3 + rng.below(8) as u32;
        let dest = Arc::new(Vtree::random(n, seed));
        let source = Arc::new(Vtree::random(n, seed ^ 0xffff));
        let f = diagram(&eng, &source, false, &mut rng);
        let perm = permutation(n, &mut rng);
        let map = |v: VarId| perm[v.idx()];
        let embedded = eng.embed_mirrored(&f, &dest, map);
        let (pairs, vtree) = (f.pair_count(), Arc::clone(f.vtree()));
        let owned = eng.transport(f.clone(), &dest, map);
        let borrowed = eng.transport(&f, &dest, map);
        match embedded {
            Ok(_) => {
                accepted += 1;
                assert!(owned.is_ok() && borrowed.is_ok());
            }
            Err(EmbedError::NotIsomorphic { .. }) => {
                refused += 1;
                for r in [owned.unwrap_err(), borrowed.unwrap_err()] {
                    assert!(matches!(r.error, EmbedError::NotIsomorphic { .. }), "{:?}", r.error);
                    assert!(Arc::ptr_eq(r.tdd.vtree(), &vtree));
                    assert_eq!(r.tdd.pair_count(), pairs);
                    assert!(eng.equivalent(&r.tdd, &f).unwrap());
                }
            }
            Err(e) => panic!("{e:?}"),
        }
    }
    assert!(accepted > 10 && refused > 10, "{accepted} accepted, {refused} refused");
}

#[test]
fn bad_images_marginal_levels_and_constants() {
    let eng = Engine::new();
    let home = Arc::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let f = eng.clause(&own, [1, -2, 3]).unwrap();

    let r = eng.transport(&f, &home, |v| VarId(v.0 + 6)).unwrap_err();
    assert!(matches!(r.error, EmbedError::VariableOutOfRange { variable: VarId(7..=9), .. }), "{:?}", r.error);
    let r = eng.transport(f.clone(), &home, |v| VarId(v.0.min(2))).unwrap_err();
    assert_eq!(r.error, EmbedError::Vtree(VtreeError::OverlappingVariable(VarId(2))));
    assert!(matches!(r.tdd, Cow::Owned(_)));

    let mut weighted = f.clone();
    weighted.set_weights(WeightStore::new(RationalWeights::unit(3), Arithmetic::ExactRational)).unwrap();
    let r = eng.transport(&weighted, &home, |v| VarId(v.0 + 3)).unwrap_err();
    assert!(matches!(r.error, EmbedError::DestinationWeights(_)), "{:?}", r.error);

    let mut summed = f.clone();
    eng.marginalize_levels(&mut summed, &[own.root()]).unwrap();
    let r = eng.transport(summed, &home, |v| v).unwrap_err();
    assert!(matches!(r.error, EmbedError::Operation(OperationError::MarginalLevel(_))), "{:?}", r.error);

    let other = Arc::new(Vtree::linear(4));
    for value in [false, true] {
        let empty = Vtree::empty_projection(&home);
        let c = if value { Tdd::one(&empty) } else { Tdd::zero(&empty) };
        let moved = eng.transport(c, &other, |v| v).map_err(|r| r.error).unwrap();
        assert_eq!(moved.vtree().num_nodes(), 0);
        assert!(Arc::ptr_eq(moved.vtree().home().unwrap(), &other));
        assert_eq!(moved.is_one(), value);
        assert_eq!(moved.is_zero(), !value);
    }
}
