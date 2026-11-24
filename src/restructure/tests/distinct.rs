use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::restructure::DistinctError;
use crate::test_helpers::{assert_canonical, compile_clauses, eval, rand_conj, random_diagrams, test_cases, under, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// Every vtree node a count may be keyed on: all but the root.
fn keys(vtree: &Vtree) -> Vec<VtreeIdx> {
    vtree.bottomup().filter(|&t| t != vtree.root()).collect()
}

fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    vtree.leaf_bottomup().filter(|&(leaf, _)| under(vtree, leaf, t)).map(|(_, v)| v).collect()
}

/// Every count of `f` keyed on each node [`keys`] gives, for `m` up to
/// four, against enumerating its truth table.
fn check(eng: &Engine, f: &Tdd) {
    let vtree = f.vtree();
    let n = vtree.num_vars() as usize;
    let truth: Vec<Vec<bool>> = (0..1u64 << n)
        .map(|bits| (0..n).map(|i| bits >> i & 1 == 1).collect::<Vec<bool>>())
        .filter(|asn| eval(f, asn))
        .collect();
    for key in keys(vtree) {
        let counted = vtree.sibling(key);
        let (kv, sv) = (vars_under(vtree, key), vars_under(vtree, counted));
        let mut distinct: HashMap<Vec<bool>, HashSet<Vec<bool>>> = HashMap::new();
        for asn in &truth {
            let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
            let s: Vec<bool> = sv.iter().map(|v| asn[v.idx()]).collect();
            distinct.entry(k).or_default().insert(s);
        }
        for m in 0..=4u64 {
            let g = eng.at_least_distinct(f, key, m).unwrap();
            assert_canonical(&g);
            assert_eq!(g.vtree().num_leaves() as usize, kv.len());
            for bits in 0..1u64 << kv.len() {
                let mut asn = vec![false; n];
                for (i, v) in kv.iter().enumerate() {
                    asn[v.idx()] = bits >> i & 1 == 1;
                }
                let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
                let count = distinct.get(&k).map_or(0, HashSet::len) as u64;
                assert_eq!(eval(&g, &asn), count >= m, "key {key:?} m {m} at {k:?}: {count} distinct");
            }
        }
    }
}

#[test]
fn the_threshold_counts_the_distinct_values_of_the_sibling() {
    for (num_vars, clauses) in test_cases() {
        for (_, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            check(&Engine::new(), &f);
        }
    }
}

#[test]
fn unreduced_diagrams_count_as_their_functions_do() {
    for f in random_diagrams(11, 24, 3..9) {
        check(&Engine::new(), &f);
    }
}

#[test]
fn a_projection_onto_a_subtree_is_the_quantified_diagram() {
    let eng = Engine::new();
    for f in random_diagrams(5, 16, 3..9) {
        let vtree = f.vtree();
        for key in keys(vtree) {
            let kept = vars_under(vtree, key);
            let dropped: Vec<VarId> = vtree.leaf_bottomup().map(|(_, v)| v).filter(|v| !kept.contains(v)).collect();
            let g = eng.project_to_subtree(&f, key).unwrap();
            assert_canonical(&g);
            let (back, _) = crate::test_helpers::embedded(&g, vtree, |v| v).unwrap();
            let expected = eng.exists_vars(f.clone(), &dropped).unwrap();
            assert!(eng.equivalent(&back, &expected).unwrap(), "key {key:?}");
        }
    }
}

#[test]
fn false_counts_nothing_and_zero_is_true() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1], vec![-1]]);
    let left = vtree.children(vtree.root()).0;
    assert!(eng.at_least_distinct(&f, left, 1).unwrap().is_zero());
    let all = eng.at_least_distinct(&f, left, 0).unwrap();
    assert_eq!(eng.model_count(&all).unwrap(), 4u32.into());
}

#[test]
fn the_root_is_refused_as_a_key() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::linear(4));
    let f = compile_clauses(&vtree, &[vec![1, 2, 3, 4]]);
    assert!(matches!(eng.at_least_distinct(&f, vtree.root(), 1), Err(DistinctError::Root)));
    // x4 sits three levels down a linear vtree of four variables: a key
    // there reads the levels on its path only.
    let deep = vtree.leaf_of(VarId(4)).unwrap();
    let g = eng.project_to_subtree(&f, deep).unwrap();
    assert_eq!(eng.model_count(&g).unwrap(), 2u32.into());
}

/// A distinct count and a subtree projection refused at each reserve in
/// turn: each refusal answers `OverBudget` with the operand's model count
/// as given, the engine runs after it, and every granted attempt is the
/// unrefused result.
#[test]
fn every_refused_reserve_of_a_distinct_count_leaves_the_operand() {
    let refusal = DistinctError::Operation(OperationError::OverBudget);
    let mut refused = 0;
    for (num_vars, clauses) in test_cases().into_iter().filter(|(n, _)| (3..=6).contains(n)).take(4) {
        let vtree = vtree_shapes(num_vars).into_iter().next().unwrap().1;
        let f = compile_clauses(&vtree, &clauses);
        let count = f.model_count().unwrap();
        for key in keys(&vtree) {
            for m in [None, Some(1u64), Some(2)] {
                let run = |eng: &Engine| match m {
                    Some(m) => eng.at_least_distinct(&f, key, m),
                    None => eng.project_to_subtree(&f, key),
                };
                let want = run(&Engine::new()).unwrap();
                let eng = Engine::new();
                refused += eng
                    .sweep_reserves(
                        &refusal,
                        || {
                            let out = run(&eng);
                            assert_eq!(f.model_count().unwrap(), count, "key {key:?} m {m:?}");
                            out
                        },
                        |g, at| {
                            assert_canonical(&g);
                            assert_eq!(format!("{:?}", g.levels), format!("{:?}", want.levels), "key {key:?} m {m:?}, {at}");
                        },
                    )
                    .refused;
            }
        }
    }
    assert!(refused > 0, "no distinct count refused");
}

/// A diagram on a projection of `home`: random clauses over the home,
/// conditioned and quantified at random, then trimmed to the leaves it
/// names, so that a sibling may hold a home variable it has no leaf for.
fn on_a_projection(home: &Home, n: u32, rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj(home.vtree(), n, 4, 3, false, rng);
    for _ in 0..rng.below(4) {
        let var = VarId(1 + rng.below(n as u64) as u32);
        f = match rng.coin() {
            true => f.condition_var(var, rng.coin()).unwrap(),
            false => f.exists_vars(&[var]).unwrap(),
        };
    }
    f.trim(home).unwrap();
    f
}

/// `g`'s value at `asn`, over the home's variables: a constant has no leaf.
fn value(g: &Tdd, asn: &[bool]) -> bool {
    match g.vtree().num_nodes() {
        0 => g.is_one(),
        _ => eval(g, asn),
    }
}

#[test]
fn counts_on_a_projection_range_over_the_home() {
    let eng = Engine::new();
    // Both sides used, the sibling unused, the key unused; a sibling with
    // a home variable the diagram lacks; an index of another own node.
    let mut seen = [0usize; 5];
    for seed in 0..120u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(11) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let at_home = home.vtree();
        let f = on_a_projection(&home, n, &mut rng);
        let own = f.vtree();
        let truth: Vec<Vec<bool>> = (0..1u64 << n)
            .map(|bits| (0..n as usize).map(|i| bits >> i & 1 == 1).collect::<Vec<bool>>())
            .filter(|asn| value(&f, asn))
            .collect();
        for key in keys(at_home) {
            let sibling = at_home.sibling(key);
            let (kv, sv) = (vars_under(at_home, key), vars_under(at_home, sibling));
            let has = |v: &VarId| own.leaf_of(*v).is_some();
            let (k_used, s_used) = (kv.iter().filter(|v| has(v)).count(), sv.iter().filter(|v| has(v)).count());
            seen[match (k_used, s_used) {
                (0, _) => 2,
                (_, 0) => 1,
                _ => 0,
            }] += 1;
            seen[3] += usize::from(s_used > 0 && s_used < sv.len());
            seen[4] += usize::from(key.idx() < own.num_nodes() && own.home_node(key) != key);
            let mut distinct: HashMap<Vec<bool>, HashSet<Vec<bool>>> = HashMap::new();
            for asn in &truth {
                let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
                let s: Vec<bool> = sv.iter().map(|v| asn[v.idx()]).collect();
                distinct.entry(k).or_default().insert(s);
            }
            let dropped: Vec<VarId> = (1..=n).map(VarId).filter(|v| !kv.contains(v)).collect();
            let exists = f.clone().exists_vars(&dropped).unwrap();
            let all = 1u64 << sv.len();
            for m in [0, 1, 2, 3, all] {
                let label = format!("seed {seed}, key {key:?}, m {m}");
                let g = eng.at_least_distinct(&f, key, m).unwrap();
                assert!(g.vtree().is_on(&home), "{label}: another home");
                assert!(g.vtree().leaf_bottomup().all(|(_, v)| kv.contains(&v) && has(&v)), "{label}: a leaf outside K");
                if g.vtree().num_nodes() > 0 {
                    assert_canonical(&g);
                }
                let p = (m == 1).then(|| eng.project_to_subtree(&f, key).unwrap());
                for bits in 0..1u64 << kv.len() {
                    let mut asn = vec![false; n as usize];
                    for (i, v) in kv.iter().enumerate() {
                        asn[v.idx()] = bits >> i & 1 == 1;
                    }
                    let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
                    let count = distinct.get(&k).map_or(0, HashSet::len) as u64;
                    assert_eq!(value(&g, &asn), count >= m, "{label} at {k:?}: {count} distinct");
                    if let Some(p) = &p {
                        assert_eq!(value(p, &asn), value(&exists, &asn), "{label}: the projection at {k:?}");
                    }
                }
            }
        }
        assert!(matches!(eng.at_least_distinct(&f, at_home.root(), 1), Err(DistinctError::Root)), "seed {seed}");
        let past = VtreeIdx(at_home.num_nodes() as u32);
        assert!(
            matches!(eng.at_least_distinct(&f, past, 1), Err(DistinctError::Operation(OperationError::LevelNotInVtree(_)))),
            "seed {seed}"
        );
    }
    assert!(seen.iter().all(|&c| c > 20), "cases seen {seen:?}");
}

/// Whether `a` and `b` agree on every assignment to `n` variables.
fn same_function(a: &Tdd, b: &Tdd, n: u32) -> bool {
    (0..1u64 << n).all(|bits| {
        let asn: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
        value(a, &asn) == value(b, &asn)
    })
}

#[test]
fn a_key_by_variables_names_the_home_node_where_an_own_index_does_not() {
    let eng = Engine::new();
    let mut trapped = 0;
    for seed in 0..120u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(9) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let at_home = home.vtree();
        let f = on_a_projection(&home, n, &mut rng);
        let root = eng.and(rand_conj(at_home, n, 4, 3, false, &mut rng), rand_conj(at_home, n, 4, 3, false, &mut rng)).unwrap();
        for g in [&f, &root] {
            for key in keys(at_home) {
                let kv = vars_under(at_home, key);
                for m in [1, 2] {
                    let by_vars = eng.at_least_distinct(g, kv.as_slice(), m).unwrap();
                    let by_node = eng.at_least_distinct(g, key, m).unwrap();
                    assert!(same_function(&by_vars, &by_node, n), "seed {seed}, key {key:?}, m {m}");
                }
            }
        }
        // An own node over all of a home node's variables, numbered
        // differently in the two vtrees.
        let own = f.vtree();
        for t in keys(own) {
            let (h, kv) = (own.home_node(t), vars_under(own, t));
            if h == t || h == at_home.root() || vars_under(at_home, h).len() != kv.len() {
                continue;
            }
            trapped += 1;
            let sv = vars_under(at_home, at_home.sibling(h));
            let mut distinct: HashMap<Vec<bool>, HashSet<Vec<bool>>> = HashMap::new();
            for bits in 0..1u64 << n {
                let asn: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
                if value(&f, &asn) {
                    let k = kv.iter().map(|v| asn[v.idx()]).collect();
                    distinct.entry(k).or_default().insert(sv.iter().map(|v| asn[v.idx()]).collect::<Vec<bool>>());
                }
            }
            let g = eng.at_least_distinct(&f, kv.as_slice(), 2).unwrap();
            for bits in 0..1u64 << n {
                let asn: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
                let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
                let count = distinct.get(&k).map_or(0, HashSet::len);
                assert_eq!(value(&g, &asn), count >= 2, "seed {seed}, own {t:?}, home {h:?}");
            }
            // The own index is below the home's node count, so it is read
            // as home node `t`, over other variables, without an error.
            let other = vars_under(at_home, t);
            assert_ne!(other, kv);
            match t == at_home.root() {
                true => assert!(matches!(eng.at_least_distinct(&f, t, 2), Err(DistinctError::Root))),
                false => {
                    let wrong = eng.at_least_distinct(&f, t, 2).unwrap();
                    let at_other = eng.at_least_distinct(&f, other.as_slice(), 2).unwrap();
                    assert!(same_function(&wrong, &at_other, n), "seed {seed}, own {t:?}");
                }
            }
        }
        let mut refused = vec![vec![], vec![VarId(1), VarId(1)], vec![VarId(n + 1)]];
        let leaf = |v: u32| at_home.leaf_of(VarId(v)).unwrap();
        if let Some(far) = (1..=n).find(|&v| at_home.lca(leaf(1), leaf(v)) == at_home.root()).filter(|_| n > 2) {
            refused.push(vec![VarId(1), VarId(far)]);
        }
        for vars in refused {
            for g in [&f, &root] {
                let error = eng.at_least_distinct(g, vars.as_slice(), 1).unwrap_err();
                assert_eq!(error, DistinctError::Key(vars.clone()), "seed {seed}");
            }
        }
    }
    assert!(trapped > 20, "{trapped} own nodes numbered apart from their home node");
}

/// A diagram on a projection whose leaves all sit under the home key has
/// every value of the key's sibling, so the count is the diagram itself.
/// That copy is reserved: under a budget of half the copy's bytes the count
/// is refused rather than copied past the budget.
#[test]
fn a_count_that_returns_its_operand_copies_it_within_the_budget() {
    use crate::limits::LimitConfig;
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(16));
    let at_home = home.vtree();
    let key = at_home.children(at_home.root()).0;
    let kv = vars_under(at_home, key);
    let mut rng = Lcg::new(7);
    let clauses: Vec<Vec<i32>> = (0..12)
        .map(|_| {
            (0..3)
                .map(|_| {
                    let v = kv[rng.below(kv.len() as u64) as usize].0 as i32;
                    if rng.coin() { v } else { -v }
                })
                .collect()
        })
        .collect();
    let mut f = compile_clauses(at_home, &clauses);
    f.trim(&home).unwrap();
    assert_eq!(f.vtree().own_at(key), Some(f.vtree().root()));
    assert!(f.vtree().own_at(at_home.sibling(key)).is_none());
    let copy = eng.try_clone(&f).unwrap();
    let bytes = eng.limits().meters().in_flight_bytes;
    assert!(bytes > 0);
    let g = eng.at_least_distinct(&f, key, 2).unwrap();
    assert_eq!(format!("{:?}", g.levels), format!("{:?}", copy.levels));
    let _budget = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(bytes / 2)));
    assert_eq!(eng.at_least_distinct(&f, key, 2).unwrap_err(), DistinctError::Operation(OperationError::OverBudget));
}
