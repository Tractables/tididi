//! Pools on projections of one home: the search runs on the union of the
//! members' projections and ends with every member on a projection of one
//! new home, with the functions and counts of the search over the members
//! on the home's root.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
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

/// One to six clauses of one to three literals over `vars`.
fn clauses(vars: &[VarId], rng: &mut Lcg) -> Vec<Vec<i32>> {
    (0..1 + rng.below(6))
        .map(|_| {
            (0..1 + rng.below(3))
                .map(|_| {
                    let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                    if rng.coin() { var } else { -var }
                })
                .collect()
        })
        .collect()
}

fn compile(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>], weights: Option<&WeightStore>) -> Tdd {
    let mut f = Tdd::one(vtree);
    for clause in clauses {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.minimize_plain(&mut f).unwrap();
    if let Some(store) = weights {
        f.set_weights(store.clone()).unwrap();
    }
    f
}

fn store(n: u32, rng: &mut Lcg) -> WeightStore {
    let weight = |rng: &mut Lcg| BigRational::new((1 + rng.below(5) as i64).into(), 6.into());
    let table: Vec<_> = (0..n).map(|_| LiteralWeights { negative: weight(rng), positive: weight(rng) }).collect();
    WeightStore::new(RationalWeights::from_literals(&table), Arithmetic::ExactRational)
}

fn unweighted(f: &Tdd) -> Tdd {
    let mut f = f.clone();
    f.take_weights().unwrap();
    f
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
/// now and then, members on one leaf set now and then sharing their vtree,
/// some weighted; and the same functions on the home's root. With `full`,
/// every member is on all the home's leaves, all on one vtree if `share`.
fn pools(eng: &Engine, home: &Home, full: bool, share: bool, rng: &mut Lcg) -> (Vec<Tdd>, Vec<Tdd>) {
    let n = home.vtree().num_leaves();
    let root = Arc::clone(home.vtree());
    let (mut own, mut embedded): (Vec<Tdd>, Vec<Tdd>) = (Vec::new(), Vec::new());
    let all: Vec<VarId> = (1..=n).map(VarId).collect();
    let full_vtree = Vtree::projection(home, all.iter().copied()).unwrap();
    for i in 0..2 + rng.below(5) as usize {
        let twin = full || (i > 0 && rng.below(3) == 0);
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
        let weights = rng.coin().then(|| store(n, rng));
        let mut f = compile(eng, &vtree, &cnf, weights.as_ref());
        if twin && (share || !full) && i > 0 && let Ok(LeafSets::Equal(same)) = Vtree::leaf_sets(f.vtree(), own[i - 1].vtree()) {
            f.reseat(&same);
        }
        own.push(f);
        embedded.push(compile(eng, &root, &cnf, weights.as_ref()));
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
    let home = after[0].home().expect("a member after a search is on a home");
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
        match a.vtree().home() {
            None => assert!(Arc::ptr_eq(a.vtree(), home.vtree())),
            Some(h) => {
                assert!(Arc::ptr_eq(h.vtree(), home.vtree()));
                let again = Vtree::projection(&home, vars(a.vtree())).unwrap();
                assert_eq!(again.num_nodes(), a.vtree().num_nodes());
                for t in (0..again.num_nodes() as u32).map(VtreeIdx) {
                    assert_eq!(again.node(t), a.vtree().node(t));
                    assert_eq!(again.home_node(t), a.vtree().home_node(t));
                }
            }
        }
        let mut expected = unweighted(b);
        eng.restructure_to(&mut expected, a.vtree(), usize::MAX).unwrap();
        assert!(eng.equivalent(&expected, &unweighted(a)).unwrap());
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
        let home = Home::new(Vtree::random(n, seed));
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
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x0dd_ba11);
        let n = 2 + rng.below(40) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let share = rng.coin();
        let (own, embedded) = pools(&eng, &home, true, share, &mut rng);
        let config = config(&mut rng);
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
            let lifted = Arc::clone(searched[0].home().unwrap().vtree());
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
        let home = Home::new(Vtree::random(n, seed));
        let full = rng.below(4) == 0;
        let (own, embedded) = pools(&eng, &home, full, rng.coin(), &mut rng);
        let internal: Vec<VtreeIdx> = home.vtree().internal_bottomup().map(|(v, _, _)| v).collect();
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
            if full {
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
