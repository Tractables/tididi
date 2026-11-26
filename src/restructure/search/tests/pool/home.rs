//! Pools on projections of one home: the search runs on the home's leaves,
//! scores each member as if it had a level at every home node, and ends with
//! every member on a projection of one new home, with the functions, counts
//! and moves of the search over the members on the home's root.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::restructure::search::{PoolMove, PoolRestarts, PoolSearchConfig, RotationMove};
use crate::test_helpers::check::check_all_fast;
use crate::test_helpers::same_storage;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, LeafSets, RotationKind, VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};

fn subset(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    let keep: Vec<VarId> = (1..=n).filter(|_| rng.below(3) != 0).map(VarId).collect();
    if keep.is_empty() { vec![VarId(1 + rng.below(n as u64) as u32)] } else { keep }
}

/// One to six clauses of one to three literals over `vars`, then one clause
/// of every variable none of them names.
fn clauses(vars: &[VarId], rng: &mut Lcg) -> Vec<Vec<i32>> {
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
    clauses
}

/// The minimized CNF, on `vtree` itself when it depends on every leaf of
/// `vtree`, else on the projection of its support.
fn compile(eng: &Engine, vtree: &crate::vtree::Home, clauses: &[Vec<i32>]) -> Tdd {
    let mut f = Tdd::one(vtree);
    for clause in clauses {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    match f.vtree().num_leaves() == vtree.num_leaves() {
        true => crate::test_helpers::padded(f, vtree),
        false => f,
    }
}

fn store(n: u32, rng: &mut Lcg) -> Weights {
    let weight = |rng: &mut Lcg| BigRational::new((1 + rng.below(5) as i64).into(), 6.into());
    let table: Vec<_> = (0..n).map(|_| LiteralWeights { negative: weight(rng), positive: weight(rng) }).collect();
    Weights::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)
}

/// A random home of `n` leaves, weighted half the time.
fn random_home(n: u32, seed: u64, rng: &mut Lcg) -> Home {
    match rng.coin() {
        true => Home::weighted(Vtree::random(n, seed), store(n, rng)).unwrap(),
        false => Home::new(Vtree::random(n, seed)),
    }
}

fn weighted(eng: &Engine, f: &Tdd) -> Option<BigRational> {
    eng.weighted_value(f).unwrap().map(|w| w.into_rational())
}

fn vars(v: &Vtree) -> Vec<VarId> {
    let mut vars: Vec<VarId> = v.leaf_bottomup().map(|(_, var)| var).collect();
    vars.sort_unstable();
    vars
}

/// Two to six members on projections of `home`, the home's root among them
/// now and then, members on one leaf set now and then sharing their vtree;
/// and the same functions on the home's root. With `full`,
/// every member is on all the home's leaves, all on one vtree if `share`.
fn pools(eng: &Engine, home: &Home, full: bool, share: bool, rng: &mut Lcg) -> (Vec<Tdd>, Vec<Tdd>) {
    let n = home.num_leaves();
    let root = Arc::clone(home);
    let (mut own, mut embedded): (Vec<Tdd>, Vec<Tdd>) = (Vec::new(), Vec::new());
    let all: Vec<VarId> = (1..=n).map(VarId).collect();
    let full_vtree = Vtree::projection(home, all.iter().copied()).unwrap();
    for i in 0..2 + rng.below(5) as usize {
        let twin = full || (i > 0 && rng.below(3) == 0 && own[i - 1].vtree().num_leaves() > 0);
        let leaves = match (full, twin) {
            (true, _) => all.clone(),
            (false, true) => vars(own[i - 1].vtree()),
            (false, false) => subset(n, rng),
        };
        let vtree = match full || rng.below(6) != 0 {
            true if full => Arc::clone(&full_vtree),
            true => Vtree::projection(home, leaves.iter().copied()).unwrap(),
            false => Arc::clone(&root),
        };
        let cnf = clauses(&leaves, rng);
        let mut f = compile(eng, &crate::vtree::Home::new(std::sync::Arc::clone(&vtree)), &cnf);
        if twin && (share || !full) && i > 0 && let Ok(LeafSets::Equal(same)) = Vtree::leaf_sets(f.vtree(), own[i - 1].vtree()) {
            f.reseat(&same);
        }
        own.push(f);
        embedded.push(compile(eng, &crate::vtree::Home::new(std::sync::Arc::clone(&root)), &cnf));
    }
    (own, embedded)
}

/// The subtree of `t` as nested pairs of variables.
fn shape(v: &Vtree, t: VtreeIdx) -> String {
    match v.node(t) {
        crate::vtree::VtreeNode::Leaf { var, .. } => var.0.to_string(),
        crate::vtree::VtreeNode::Internal { left, right, .. } => format!("({} {})", shape(v, *left), shape(v, *right)),
    }
}

fn with_members<R>(members: &mut [Tdd], f: impl FnOnce(&mut [&mut Tdd]) -> R) -> R {
    let mut refs: Vec<&mut Tdd> = members.iter_mut().collect();
    f(&mut refs)
}

/// `after` is the pool `before` after a search that kept a move iff `kept`,
/// `embedded` the same functions after the search on the home's root.
fn check_pool(eng: &Engine, before: &[Tdd], after: &[Tdd], embedded: &[Tdd], kept: bool) {
    let home = after[0].home();
    for ((b, a), e) in before.iter().zip(after).zip(embedded) {
        check_all_fast(a, "a member after the search");
        assert_eq!(a.vtree().validate(), Ok(()));
        let count = eng.model_count(a).unwrap();
        assert_eq!(count, eng.model_count(b).unwrap());
        assert_eq!(count, eng.model_count(e).unwrap());
        assert_eq!(weighted(eng, a), weighted(eng, b));
        assert_eq!(weighted(eng, a), weighted(eng, e));
        assert_eq!(vars(a.vtree()), vars(b.vtree()));
        if !kept {
            assert!(Arc::ptr_eq(a.vtree(), b.vtree()));
            continue;
        }
        if a.vtree().num_nodes() == 0 {
            continue;
        }
        let h = a.home();
        assert!(Arc::ptr_eq(&h, &home));
        let again = Vtree::projection(&home, vars(a.vtree())).unwrap();
        assert_eq!(again.num_nodes(), a.vtree().num_nodes());
        for t in (0..again.num_nodes() as u32).map(VtreeIdx) {
            assert_eq!(again.node(t), a.vtree().node(t));
            assert_eq!(again.home_node(t), a.vtree().home_node(t));
        }
        let mut expected = b.clone();
        eng.restructure_to(&mut expected, a.vtree(), usize::MAX).unwrap();
        assert!(eng.equivalent(&expected, a).unwrap());
    }
}

fn config(rng: &mut Lcg) -> PoolSearchConfig {
    let config = PoolSearchConfig::default().with_crossed(rng.below(4) != 0);
    match rng.below(3) {
        0 => config.with_restarts(PoolRestarts::default().with_count(2).with_kick(3).with_seed(rng.below(1000))),
        _ => config,
    }
}

fn accepted() -> Vec<PoolMove> {
    super::ACCEPTED.with(|a| std::mem::take(&mut *a.borrow_mut()))
}

/// Random homes up to 64 leaves and pools of two to six members on random
/// leaf sets: the search ends on projections of one new home with every
/// function and count, those of the search on the home's root, and a search
/// that keeps nothing leaves every member as it was.
#[test]
fn a_pool_on_projections_searches_as_on_the_root() {
    let eng = Engine::new();
    let (mut kept_searches, mut declined) = (0, 0);
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed ^ 0x51ed_270b);
        let n = 2 + rng.below(63) as u32;
        let home = random_home(n, seed, &mut rng);
        let (own, embedded) = pools(&eng, &home, false, false, &mut rng);
        let config = config(&mut rng);
        let mut searched = own.clone();
        let stats = with_members(&mut searched, |m| eng.pool_search(m, &config)).unwrap();
        let mut on_root = embedded.clone();
        with_members(&mut on_root, |m| eng.pool_search(m, &config)).unwrap();
        let kept = stats.accepts > 0;
        kept_searches += usize::from(kept);
        declined += usize::from(!kept);
        check_pool(&eng, &own, &searched, &on_root, kept);
    }
    assert!(kept_searches > 20 && declined > 0, "{kept_searches} {declined}");
}

/// Members on all the home's leaves, on one vtree or on several: the search
/// keeps the moves the search on the root keeps, and ends on its shape.
#[test]
fn members_with_every_leaf_make_the_moves_of_the_search_on_the_root() {
    let eng = Engine::new();
    let mut moves = 0;
    for seed in 0..80u64 {
        let mut rng = Lcg::new(seed ^ 0x0dd_ba11);
        let n = 2 + rng.below(40) as u32;
        let home = random_home(n, seed, &mut rng);
        let share = rng.coin();
        let (own, embedded) = pools(&eng, &home, true, share, &mut rng);
        let config = config(&mut rng);
        if own.iter().any(|f| f.vtree().num_leaves() != n) {
            continue;
        }
        let full = Arc::clone(own[0].vtree());
        accepted();
        let mut searched = own.clone();
        let stats = with_members(&mut searched, |m| eng.pool_search(m, &config)).unwrap();
        let on_union: Vec<PoolMove> = accepted()
            .into_iter()
            .map(|mv| PoolMove { rotation: RotationMove { pivot: full.home_node(mv.rotation.pivot), ..mv.rotation }, ..mv })
            .collect();
        let mut on_root = embedded.clone();
        with_members(&mut on_root, |m| eng.pool_search(m, &config)).unwrap();
        assert_eq!(on_union, accepted(), "seed {seed}");
        moves += on_union.len();
        let kept = stats.accepts > 0;
        check_pool(&eng, &own, &searched, &on_root, kept);
        if kept {
            let lifted = Arc::clone(&searched[0].home());
            assert_eq!(shape(&lifted, lifted.root()), shape(on_root[0].vtree(), on_root[0].vtree().root()), "seed {seed}");
        }
    }
    assert!(moves > 40, "{moves}");
}

/// Random moves at nodes of the home, kept and declined: as the search's
/// end; with every leaf, kept as on the root with the same change.
#[test]
fn a_pool_move_on_projections_ends_on_one_home() {
    let eng = Engine::new();
    let (mut kept_moves, mut declined) = (0, 0);
    for seed in 0..80u64 {
        let mut rng = Lcg::new(seed ^ 0x007a_11ed);
        let n = 2 + rng.below(63) as u32;
        let home = random_home(n, seed, &mut rng);
        let full = rng.below(4) == 0;
        let (own, embedded) = pools(&eng, &home, full, rng.coin(), &mut rng);
        let internal: Vec<VtreeIdx> = home.internal_bottomup().map(|(v, _, _)| v).collect();
        if internal.is_empty() {
            continue;
        }
        for _ in 0..4 {
            let pivot = internal[rng.below(internal.len() as u64) as usize];
            let kind = if rng.coin() { RotationKind::Left } else { RotationKind::Right };
            let mv = PoolMove { rotation: RotationMove { pivot, kind }, crossed: rng.coin() };
            let accept = rng.coin();
            let mut moved = own.clone();
            let mut delta = None;
            let kept = with_members(&mut moved, |m| {
                eng.rotate_pool_if(m, mv, usize::MAX, |p| {
                    delta = Some(p.live_pairs_delta());
                    accept
                })
            })
            .unwrap();
            let mut on_root = embedded.clone();
            let mut root_delta = None;
            let root_kept = with_members(&mut on_root, |m| {
                eng.rotate_pool_if(m, mv, usize::MAX, |p| {
                    root_delta = Some(p.live_pairs_delta());
                    accept
                })
            })
            .unwrap();
            if full && own.iter().all(|f| f.vtree().num_leaves() == n) {
                assert_eq!((kept, delta), (root_kept, root_delta), "seed {seed} {mv:?}");
            }
            kept_moves += usize::from(kept);
            declined += usize::from(!kept);
            check_pool(&eng, &own, &moved, &on_root, kept);
            if !kept {
                assert!(own.iter().zip(&moved).all(|(b, a)| same_storage(a, b)));
            }
        }
    }
    assert!(kept_moves > 40 && declined > 40, "{kept_moves} {declined}");
}

/// Members on random leaf sets and the same functions padded onto the home's
/// root: each member's embedded count is its padded copy's pairs, and the
/// search on the projections makes the moves of the search on the root, with
/// the same totals.
#[test]
fn a_pool_on_projections_makes_the_moves_of_the_search_on_the_root() {
    let eng = Engine::new();
    let (mut moves, mut carried) = (0, 0);
    for seed in 0..80u64 {
        let mut rng = Lcg::new(seed ^ 0x3b1d_c0de);
        let n = 2 + rng.below(40) as u32;
        let home = random_home(n, seed, &mut rng);
        let (own, _) = pools(&eng, &home, false, false, &mut rng);
        let padded: Vec<Tdd> = own.iter().map(|f| on_root(f, &home)).collect();
        for (f, p) in own.iter().zip(&padded) {
            assert_eq!(f.embedded_pair_count(), p.pair_count(), "seed {seed}");
            carried += f.embedded_pair_count() - f.pair_count();
        }
        let config = config(&mut rng);
        let full = Vtree::projection(&home, (1..=n).map(VarId)).unwrap();
        accepted();
        let mut searched = own.clone();
        let stats = with_members(&mut searched, |m| eng.pool_search(m, &config)).unwrap();
        let on_home: Vec<PoolMove> = accepted()
            .into_iter()
            .map(|mv| PoolMove { rotation: RotationMove { pivot: full.home_node(mv.rotation.pivot), ..mv.rotation }, ..mv })
            .collect();
        let mut on_root = padded.clone();
        let root_stats = with_members(&mut on_root, |m| eng.pool_search(m, &config)).unwrap();
        assert_eq!(on_home, accepted(), "seed {seed}");
        assert_eq!(
            (stats.pairs_before, stats.pairs_after),
            (root_stats.pairs_before, root_stats.pairs_after),
            "seed {seed}"
        );
        moves += on_home.len();
        check_pool(&eng, &own, &searched, &on_root, stats.accepts > 0);
    }
    assert!(moves > 40 && carried > 0, "{moves} {carried}");
}

/// Hand-counted: x6 ∧ (x1 xor x4) and x1 xor x6 on projections of the
/// right-linear home over six variables.
#[test]
fn the_embedded_count_adds_the_home_nodes_a_diagram_has_no_level_at() {
    let eng = Engine::new();
    let home = Home::new(Vtree::linear(6));
    let on = |vars: &[u32], clauses: &[Vec<i32>]| {
        let f = compile(&eng, &home, clauses);
        assert_eq!(f.vtree().leaf_bottomup().map(|(_, v)| v.0).collect::<std::collections::BTreeSet<_>>(), vars.iter().copied().collect());
        let padded = crate::test_helpers::padded(f.clone(), &home);
        assert_eq!(f.embedded_pair_count(), padded.pair_count());
        f
    };
    // Three home nodes none of its own; its level over x4 and x6 has two
    // nodes and two home nodes above it below the root; x6 is entailed.
    let f = on(&[1, 4, 6], &[vec![6], vec![1, 4], vec![-1, -4]]);
    assert_eq!(f.embedded_pair_count(), f.pair_count() + 3 + 2);
    // Four home nodes none of its own; x6 takes two labels at the root, four
    // home nodes below it.
    let g = on(&[1, 6], &[vec![1, 6], vec![-1, -6]]);
    assert_eq!(g.embedded_pair_count(), g.pair_count() + 4 + 4);
    for constant in [Tdd::one(&home), Tdd::zero(&home)] {
        assert_eq!(constant.vtree().num_leaves(), 0);
        assert_eq!(constant.embedded_pair_count(), 0);
    }
}

/// `f` padded onto the home's root; a constant stays on the empty
/// projection, where padding would give it a level at every node.
fn on_root(f: &Tdd, home: &Home) -> Tdd {
    match f.vtree().num_leaves() {
        0 => f.clone(),
        _ => crate::test_helpers::padded(f.clone(), home),
    }
}

/// x1 xor x2 and x1 ∨ x2 on the projection of (x1, (x2, x3)) on x1 and x2:
/// the one move that gains has two parts for both members and changes
/// none of their levels, only the home node x2 passes through. A search
/// that scores the members' own pairs on their union keeps nothing.
#[test]
fn a_move_that_only_shortens_what_members_carry_is_kept() {
    let eng = Engine::new();
    let home = Home::new(Vtree::linear(3));
    let vtree = Vtree::projection(&home, [VarId(1), VarId(2)]).unwrap();
    let mut own = vec![compile(&eng, &home, &[vec![1, 2], vec![-1, -2]]), compile(&eng, &home, &[vec![1, 2]])];
    for f in &mut own {
        let Ok(LeafSets::Equal(on)) = Vtree::leaf_sets(f.vtree(), &vtree) else { panic!("{}", f.vtree().to_text()) };
        f.reseat(&on);
        f.minimize().unwrap();
    }
    let same = own[0].vtree().clone();
    for f in &mut own {
        assert!(Arc::ptr_eq(f.vtree(), &vtree));
        assert!(f.is_canonical());
    }
    let before = own.clone();
    let padded: Vec<Tdd> = own.iter().map(|f| on_root(f, &home)).collect();
    let stats = with_members(&mut own, |m| eng.pool_search(m, &PoolSearchConfig::default())).unwrap();
    assert_eq!(stats.accepts, 1);
    assert_eq!(stats.pairs_before, padded.iter().map(Tdd::pair_count).sum::<usize>());
    assert_eq!(stats.pairs_before - stats.pairs_after, 2);
    let lifted = own[0].home();
    assert_eq!(shape(&lifted, lifted.root()), "((1 2) 3)");
    for f in &own {
        assert!(f.is_canonical());
        assert!(!Arc::ptr_eq(f.vtree(), &same));
    }
    check_pool(&eng, &before, &own, &padded, true);
}

/// Members on one projection share its allocation: a move kept through
/// `rotate_pool_if` and a search that keeps moves end on one home with
/// every function.
#[test]
fn a_pool_on_one_projection_keeps_moves() {
    let eng = Engine::new();
    let (mut rotated, mut searched) = (0, 0);
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x0e7e_5eed);
        let n = 4 + rng.below(20) as u32;
        let home = random_home(n, seed, &mut rng);
        let leaves = subset(n, &mut rng);
        let vtree = Vtree::projection(&home, leaves.iter().copied()).unwrap();
        let mut own: Vec<Tdd> = Vec::new();
        for _ in 0..2 + rng.below(3) {
            let f = compile(&eng, &home, &clauses(&leaves, &mut rng));
            if let Ok(LeafSets::Equal(same)) = Vtree::leaf_sets(f.vtree(), &vtree) {
                let mut f = f;
                f.reseat(&same);
                own.push(f);
            }
        }
        if own.len() < 2 || vtree.num_leaves() < 3 {
            continue;
        }
        assert!(own.iter().all(|f| Arc::ptr_eq(f.vtree(), &vtree)));
        let padded: Vec<Tdd> = own.iter().map(|f| on_root(f, &home)).collect();
        let config = config(&mut rng);
        let mut after = own.clone();
        let stats = with_members(&mut after, |m| eng.pool_search(m, &config)).unwrap();
        searched += usize::from(stats.accepts > 0);
        check_pool(&eng, &own, &after, &padded, stats.accepts > 0);
        for &t in vtree.internal_bottomup_slice() {
            let pivot = vtree.home_node(t);
            let mv = PoolMove { rotation: RotationMove { pivot, kind: RotationKind::Left }, crossed: rng.coin() };
            let mut moved = own.clone();
            let kept = with_members(&mut moved, |m| eng.rotate_pool_if(m, mv, usize::MAX, |_| true)).unwrap();
            rotated += usize::from(kept);
            check_pool(&eng, &own, &moved, &padded, kept);
        }
    }
    assert!(rotated > 10 && searched > 5, "{rotated} {searched}");
}
