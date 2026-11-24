//! A search on a projection ends on the projection of a lifted home, with
//! its counts over the home's variables; `Engine::transport` brings other
//! diagrams of the old home there.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::restructure::search::{MinimizePairs, RotationMove, RotationSearchConfig};
use crate::restructure::TransportError;
use crate::test_helpers::check::check_all_fast;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, RotationKind, VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};

fn subset(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    let keep: Vec<VarId> = (1..=n).filter(|_| rng.below(3) != 0).map(VarId).collect();
    if keep.is_empty() { vec![VarId(1 + rng.below(n as u64) as u32)] } else { keep }
}

/// A random CNF over `vtree`'s leaves, minimized: one to six clauses, then
/// one clause of every leaf none of them names, so that the diagram is on
/// `vtree` but for the leaves the clauses make irrelevant.
fn diagram(eng: &Engine, vtree: &Arc<Vtree>, rng: &mut Lcg) -> Tdd {
    let vars: Vec<VarId> = vtree.leaf_bottomup().map(|(_, var)| var).collect();
    let literal = |var: VarId, rng: &mut Lcg| if rng.coin() { var.0 as i32 } else { -(var.0 as i32) };
    let mut clauses: Vec<Vec<i32>> = (0..1 + rng.below(6))
        .map(|_| (0..1 + rng.below(3)).map(|_| literal(vars[rng.below(vars.len() as u64) as usize], rng)).collect())
        .collect();
    let rest: Vec<i32> = vars
        .iter()
        .filter(|var| !clauses.iter().flatten().any(|&l| l.unsigned_abs() == var.0))
        .map(|&var| literal(var, rng))
        .collect();
    if !rest.is_empty() {
        clauses.push(rest);
    }
    let mut f = Tdd::one(vtree);
    for c in &clauses {
        f = eng.and_clause(f, c).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

fn store(n: u32, rng: &mut Lcg) -> Weights {
    let weight = |rng: &mut Lcg| BigRational::new((1 + rng.below(5) as i64).into(), 6.into());
    let table: Vec<_> =
        (0..n).map(|_| LiteralWeights { negative: weight(rng), positive: weight(rng) }).collect();
    Weights::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)
}

fn weighted(eng: &Engine, f: &Tdd) -> Option<BigRational> {
    eng.weighted_value(f).unwrap().map(|w| w.into_rational())
}

/// `after` is `before` moved by a search that kept a move iff `kept`.
fn check_end(eng: &Engine, before: &Tdd, after: &Tdd, kept: bool) {
    let home = before.home().unwrap();
    check_all_fast(before, "before the search");
    check_all_fast(after, "after the search");
    assert_eq!(after.vtree().validate(), Ok(()));
    assert_eq!(eng.model_count(after).unwrap(), eng.model_count(before).unwrap());
    assert_eq!(weighted(eng, after), weighted(eng, before));
    let lifted = after.home().expect("a projection stays a projection");
    if !kept {
        assert!(Arc::ptr_eq(after.vtree(), before.vtree()));
        return;
    }
    assert!(!Arc::ptr_eq(lifted.vtree(), home.vtree()));
    assert_eq!(lifted.vtree().num_leaves(), home.vtree().num_leaves());
    let vars = |v: &Vtree| {
        let mut vars: Vec<VarId> = v.leaf_bottomup().map(|(_, var)| var).collect();
        vars.sort_unstable();
        vars
    };
    assert_eq!(vars(after.vtree()), vars(before.vtree()));
    assert_eq!(vars(lifted.vtree()), vars(home.vtree()));
    let again = Vtree::projection(&lifted, vars(after.vtree())).unwrap();
    assert_eq!(again.num_nodes(), after.vtree().num_nodes());
    for t in (0..again.num_nodes() as u32).map(VtreeIdx) {
        assert_eq!(again.node(t), after.vtree().node(t));
        assert_eq!(again.home_node(t), after.vtree().home_node(t));
    }
    let mut expected = before.clone();
    eng.restructure_to(&mut expected, after.vtree(), usize::MAX).unwrap();
    assert!(eng.equivalent(&expected, after).unwrap());
}

/// Random homes up to 64 leaves, half of them weighted, random leaf sets and
/// diagrams: a rotation search and random `rotate_if` calls, kept and
/// declined, keep counts, weighted values and the function, and end on a
/// projection.
#[test]
fn a_search_on_a_projection_keeps_its_counts_on_a_lifted_home() {
    let eng = Engine::new();
    let (mut kept_searches, mut kept_turns, mut declined) = (0, 0, 0);
    for seed in 0..128u64 {
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let n = 2 + rng.below(63) as u32;
        // A second generator for the weights, so that `rng` draws the leaf
        // sets and diagrams it drew when weights came after them: seed 7
        // otherwise draws a constant on an empty projection, which
        // `check_all_fast` cannot read.
        let mut weights_rng = Lcg::new(seed ^ 0x77e1_9475);
        let home = match weights_rng.coin() {
            true => Home::weighted(Vtree::random(n, seed), store(n, &mut weights_rng)).unwrap(),
            false => Home::new(Vtree::random(n, seed)),
        };
        let own = Vtree::projection(&home, subset(n, &mut rng)).unwrap();
        let f = diagram(&eng, &own, &mut rng);

        let mut searched = f.clone();
        let stats = eng.rotation_search(&mut searched, &mut MinimizePairs, &RotationSearchConfig::default()).unwrap();
        let kept = !Arc::ptr_eq(searched.vtree(), f.vtree());
        assert!(stats.accepts > 0 || !kept);
        kept_searches += usize::from(kept);
        check_end(&eng, &f, &searched, kept);

        let internal: Vec<VtreeIdx> = f.vtree().internal_bottomup().map(|(v, _, _)| v).collect();
        for _ in 0..4 {
            if internal.is_empty() {
                break;
            }
            let mut turned = f.clone();
            let moves: Vec<RotationMove> = (0..1 + rng.below(3))
                .map(|_| RotationMove {
                    pivot: internal[rng.below(internal.len() as u64) as usize],
                    kind: if rng.coin() { RotationKind::Left } else { RotationKind::Right },
                })
                .collect();
            let accept = rng.coin();
            let kept = turned.rotate_if(&moves, usize::MAX, |_| accept).unwrap();
            assert!(!kept || accept);
            kept_turns += usize::from(kept);
            declined += usize::from(!kept);
            check_end(&eng, &f, &turned, kept);
        }
    }
    assert!(kept_searches > 20 && kept_turns > 50 && declined > 50, "{kept_searches} {kept_turns} {declined}");
}

/// The own leaves under `t`.
fn leaves_under(v: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    v.leaf_bottomup().filter(|&(leaf, _)| v.in_subtree(leaf, t)).map(|(_, var)| var).collect()
}

/// After one kept left rotation `(x, (z, y))` to `((x, z), y)` of a
/// projection, a diagram of the old home on the leaves of two of the three
/// parts transports with bound 0 and conjoins with the moved one; one on all
/// three is refused at bound 0 and moves without a bound.
#[test]
fn a_diagram_in_two_parts_transports_without_rotating() {
    let eng = Engine::new();
    let mut cases = [0usize; 2];
    for seed in 0..120u64 {
        let mut rng = Lcg::new(seed ^ 0x9e37_79b9);
        let n = 4 + rng.below(61) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let own = Vtree::projection(&home, subset(n, &mut rng)).unwrap();
        let pivots: Vec<VtreeIdx> = own
            .internal_bottomup()
            .map(|(v, _, _)| v)
            .filter(|&v| !own.node(own.children(v).1).is_leaf())
            .collect();
        if pivots.is_empty() {
            continue;
        }
        let v = pivots[rng.below(pivots.len() as u64) as usize];
        let (x, w) = own.children(v);
        let (z, y) = own.children(w);
        let parts = [leaves_under(&own, x), leaves_under(&own, z), leaves_under(&own, y)];

        let f = diagram(&eng, &own, &mut rng);
        // The parts are read on `own`: a diagram that does not depend on
        // every leaf of it is on a narrower projection. One on the same
        // leaves has the same node indices.
        if f.vtree().num_leaves() != own.num_leaves() {
            continue;
        }
        let mut moved = f.clone();
        assert!(moved.rotate_if(&[RotationMove { pivot: v, kind: RotationKind::Left }], usize::MAX, |_| true).unwrap());
        check_end(&eng, &f, &moved, true);
        let lifted = moved.home().unwrap();

        let three = rng.coin();
        let mut vars: Vec<VarId> = match three {
            true => parts.iter().map(|p| p[rng.below(p.len() as u64) as usize]).collect(),
            false => {
                let skip = rng.below(3) as usize;
                parts.iter().enumerate().filter(|&(i, _)| i != skip).flat_map(|(_, p)| p.iter().copied()).collect()
            }
        };
        // A leaf of the home outside `own` sits in no part when it is
        // outside the subtree of `v`'s home node.
        let outside: Vec<VarId> = (1..=n)
            .map(VarId)
            .filter(|&var| own.leaf_of(var).is_none())
            .filter(|&var| {
                let leaf = home.vtree().leaf_of(var).unwrap();
                !home.vtree().in_subtree(leaf, own.home_node(v))
            })
            .collect();
        if !outside.is_empty() && rng.coin() {
            vars.push(outside[rng.below(outside.len() as u64) as usize]);
        }
        let g_own = Vtree::projection(&home, vars).unwrap();
        let g = diagram(&eng, &g_own, &mut rng);
        if g.vtree().num_leaves() != g_own.num_leaves() {
            continue;
        }
        cases[usize::from(three)] += 1;
        let rehomed = match three {
            false => eng.transport(&g, &lifted, |v| v, 0).unwrap().0,
            true => {
                let refused = eng.transport(g.clone(), &lifted, |v| v, 0).unwrap_err();
                assert!(matches!(refused.error, TransportError::Bound { .. }));
                assert!(matches!(refused.tdd, std::borrow::Cow::Owned(_)));
                assert!(Arc::ptr_eq(refused.tdd.vtree(), g.vtree()));
                let (rotated, stats) = eng.transport(&g, &lifted, |v| v, usize::MAX).unwrap();
                assert!(stats.rotations > 0 || g.is_zero(), "{stats:?}");
                rotated
            }
        };
        check_all_fast(&rehomed, "rehomed");
        assert_eq!(rehomed.vtree().validate(), Ok(()));
        assert!(Arc::ptr_eq(rehomed.home().unwrap().vtree(), lifted.vtree()));
        assert_eq!(eng.model_count(&rehomed).unwrap(), eng.model_count(&g).unwrap());
        let both = eng.and(&moved, &rehomed).unwrap();
        let before = eng.and(&f, &g).unwrap();
        assert_eq!(eng.model_count(&both).unwrap(), eng.model_count(&before).unwrap());
    }
    assert!(cases[0] > 15 && cases[1] > 15, "{cases:?}");
}

/// A search that keeps nothing leaves the shared vtree and makes no home.
#[test]
fn a_declined_search_keeps_the_vtree() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(16));
    let own = Vtree::projection(&home, [1, 2, 5, 6, 9, 13, 14].map(VarId)).unwrap();
    let f = eng.clause(&own, [1, -2, -5, 6, 9, -13, 14]).unwrap();
    let mut g = f.clone();
    let turn = RotationMove { pivot: own.root(), kind: RotationKind::Right };
    assert!(!g.rotate_if(&[turn], usize::MAX, |_| false).unwrap());
    assert!(Arc::ptr_eq(g.vtree(), f.vtree()));
    assert!(Arc::ptr_eq(g.home().unwrap().vtree(), home.vtree()));
}

/// A home's root held by one diagram alone moves in place: a clone the
/// probe's callback makes is on no home, a declined move gives the root its
/// index back, and a kept one ends on a home of the new shape.
#[test]
fn a_home_root_held_alone_is_no_home_while_it_moves() {
    for accept in [false, true] {
        let mut f = {
            let root = Arc::clone(Home::new(Vtree::balanced(8)).vtree());
            Tdd::clause(&root, [1, 2, -3, -4, 5, 6, -7, 8]).unwrap()
        };
        let before = Arc::as_ptr(f.vtree());
        let pivot = f.vtree().root();
        let mut copy_on_home = None;
        let kept = f
            .rotate_if(&[RotationMove { pivot, kind: RotationKind::Right }], usize::MAX, |probe| {
                copy_on_home = Some(probe.diagram().clone().home().is_some());
                accept
            })
            .unwrap();
        assert_eq!((kept, copy_on_home), (accept, Some(false)));
        assert_eq!(Arc::as_ptr(f.vtree()), before);
        assert!(Arc::ptr_eq(f.home().expect("on a home").vtree(), f.vtree()));
        assert_eq!(f.vtree().validate(), Ok(()));
        check_all_fast(&f, "after the move");
    }
}
