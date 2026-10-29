use std::sync::Arc;

use super::convert::{from_tdd, to_tdd};
use super::{conjoin, contract::{contract, contract_in}, ix, vt, Ref, Shape, SkipForm, ZREF};
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::{Engine, Tdd};

fn skip(f: &Tdd) -> SkipForm {
    from_tdd(f, Arc::new(Shape::of(f.vtree())), Engine::new().limits()).expect("structural")
}

/// Random canonical diagrams over every vtree shape, with their vtree.
fn cases(seed: u64, count: usize) -> Vec<(Engine, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = 2 + rng.below(9) as u32;
        let shape = CnfShape { clauses: n as usize, width: 3 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            out.push((eng, f, g));
        }
    }
    out
}

#[test]
fn skip_form_expands_to_the_same_function() {
    for (eng, f, _) in cases(1, 200) {
        let s = skip(&f);
        assert_eq!(s.model_count(eng.limits()).unwrap(), eng.model_count(&f).unwrap());
        let mut back = to_tdd(&s, &eng, f.vtree()).unwrap();
        // The stored form of a canonical skip form is canonical.
        let (pairs, nodes) = (back.pair_count(), back.node_count());
        eng.minimize(&mut back).unwrap();
        assert_eq!((back.pair_count(), back.node_count()), (pairs, nodes));
        assert!(eng.equivalent(&back, &f).unwrap());
        // Canonical: the round trip's skip form has the same size.
        assert_eq!(skip(&back).size(), s.size());
    }
}

#[test]
fn conjunction_matches_the_stored_form() {
    for (eng, f, g) in cases(2, 200) {
        let (sf, sg) = (Arc::new(skip(&f)), skip(&g));
        let sg = SkipForm { shape: Arc::clone(&sf.shape), ..sg };
        let h = conjoin::and(&sf, &sg, eng.limits()).unwrap();
        let want = eng.and(&f, &g).unwrap();
        assert_eq!(h.model_count(eng.limits()).unwrap(), eng.model_count(&want).unwrap());
        let mut back = to_tdd(&h, &eng, f.vtree()).unwrap();
        eng.minimize(&mut back).unwrap();
        assert!(eng.equivalent(&back, &want).unwrap());
    }
}

/// A structural fingerprint: per level the sorted node hashes, and the
/// output's. Two skip forms of one function are canonical when equal.
fn fingerprint(s: &SkipForm) -> (u64, Vec<Vec<u64>>) {
    use std::hash::{Hash, Hasher};
    let sh = &s.shape;
    let n = sh.num_nodes();
    let mut h: Vec<Vec<u64>> = vec![Vec::new(); n];
    let href = |h: &Vec<Vec<u64>>, r: Ref| -> u64 {
        if r == ZREF {
            return 0;
        }
        let mut x = rustc_hash::FxHasher::default();
        vt(r).hash(&mut x);
        if sh.is_leaf(vt(r)) { ix(r).hash(&mut x) } else { h[vt(r) as usize][ix(r) as usize].hash(&mut x) }
        x.finish()
    };
    for &t in &sh.bottomup {
        let lvl = &s.levels[t as usize];
        let mut v = Vec::new();
        for i in 0..lvl.width() {
            let mut ps: Vec<(u64, u64)> = lvl.node(i as u32).iter().map(|&(a, b)| (href(&h, a), href(&h, b))).collect();
            ps.sort_unstable();
            let mut x = rustc_hash::FxHasher::default();
            ps.hash(&mut x);
            v.push(x.finish());
        }
        h[t as usize] = v;
    }
    let out = href(&h, s.out);
    let mut levels = h;
    levels.iter_mut().for_each(|v| v.sort_unstable());
    (out, levels)
}

#[test]
fn contracted_conjunction_is_the_canonical_skip_form() {
    for (eng, f, g) in cases(3, 300) {
        let (sf, sg) = (skip(&f), skip(&g));
        let sg = SkipForm { shape: Arc::clone(&sf.shape), ..sg };
        let (raw, cone) = conjoin::and_scoped(&sf, &sg, eng.limits()).unwrap();
        let h = contract_in(raw, Some(&cone), eng.limits()).unwrap();
        eng.limits().discard(cone);
        let mut fun = eng.and(&f, &g).unwrap();
        eng.minimize(&mut fun).unwrap();
        let want = skip(&fun);
        let mut back = super::convert::to_tdd(&h, &eng, f.vtree()).unwrap();
        eng.minimize(&mut back).unwrap();
        if fingerprint(&h) != fingerprint(&want) {
            eprintln!("vtree {:?}", sf.shape.kids);
            eprintln!("equivalent {}", eng.equivalent(&back, &fun).unwrap());
            eprintln!("mine {:?}", h.levels.iter().enumerate().filter(|l| l.1.width() > 0).map(|(i, l)| (i, (0..l.width()).map(|k| l.node(k as u32).iter().map(|&(a, b)| ((vt(a), ix(a)), (vt(b), ix(b)))).collect::<Vec<_>>()).collect::<Vec<_>>())).collect::<Vec<_>>());
            eprintln!("mine out {:?}", (vt(h.out), ix(h.out)));
            eprintln!("want {:?}", want.levels.iter().enumerate().filter(|l| l.1.width() > 0).map(|(i, l)| (i, (0..l.width()).map(|k| l.node(k as u32).iter().map(|&(a, b)| ((vt(a), ix(a)), (vt(b), ix(b)))).collect::<Vec<_>>()).collect::<Vec<_>>())).collect::<Vec<_>>());
            eprintln!("want out {:?}", (vt(want.out), ix(want.out)));
        }
        assert_eq!(fingerprint(&h), fingerprint(&want));
        // Contracting a canonical form changes nothing.
        assert_eq!(fingerprint(&contract(want.clone(), eng.limits()).unwrap()), fingerprint(&want));
    }
}

#[test]
fn negation_is_the_canonical_skip_form() {
    for (eng, f, g) in cases(4, 300) {
        for x in [&f, &g] {
            let s = skip(x);
            let h = super::negate::negate(&s, eng.limits()).unwrap();
            let mut fun = eng.negate(x).unwrap();
            eng.minimize(&mut fun).unwrap();
            let want = skip(&fun);
            assert_eq!(h.model_count(eng.limits()).unwrap(), eng.model_count(&fun).unwrap());
            if fingerprint(&h) != fingerprint(&want) {
                eprintln!("vtree {:?}", s.shape.kids);
                eprintln!("f {:?} out {:?}", dump(&s), (vt(s.out), ix(s.out)));
                eprintln!("mine {:?} out {:?}", dump(&h), (vt(h.out), ix(h.out)));
                eprintln!("want {:?} out {:?}", dump(&want), (vt(want.out), ix(want.out)));
            }
            assert_eq!(fingerprint(&h), fingerprint(&want));
            // Negating twice gives the input back.
            assert_eq!(fingerprint(&super::negate::negate(&h, eng.limits()).unwrap()), fingerprint(&s));
        }
    }
    // The constants.
    let (eng, f, _) = cases(5, 1).pop().unwrap();
    let s = skip(&f);
    let zero = SkipForm::zero(Arc::clone(&s.shape));
    let one = super::negate::negate(&zero, eng.limits()).unwrap();
    assert_eq!(one.model_count(eng.limits()).unwrap(), num_bigint::BigUint::from(1u32) << f.vtree().num_vars());
    assert!(super::negate::negate(&one, eng.limits()).unwrap().is_zero());
}

type Dump = Vec<(usize, Vec<Vec<((u32, u32, u32), (u32, u32, u32))>>)>;

fn dump(s: &SkipForm) -> Dump {
    s.levels
        .iter()
        .enumerate()
        .filter(|l| l.1.width() > 0)
        .map(|(i, l)| (i, (0..l.width()).map(|k| l.node(k as u32).iter().map(|&(a, b)| ((vt(a), ix(a), 0), (vt(b), ix(b), 0))).collect()).collect()))
        .collect()
}

#[test]
fn conjunction_of_projections_is_canonical() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(10);
    let mut done = 0;
    while done < 300 {
        let n = 3 + rng.below(8) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let ish = Arc::new(Shape::of(&vtree));
            let mut placed: Vec<SkipForm> = Vec::new();
            let mut embedded = Vec::new();
            for _ in 0..2 {
                // A random subset of the variables, and a function over it.
                let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
                if keep.is_empty() {
                    break;
                }
                let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
                let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { break };
                let small = eng.bind_vtree(small);
                let m = keep.len() as u32;
                let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
                let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
                let (mut e, _) = eng.embed(&f, &vtree, map).unwrap();
                eng.minimize(&mut e).unwrap();
                placed.push(from_tdd(&e, Arc::clone(&ish), eng.limits()).unwrap());
                embedded.push(e);
            }
            if placed.len() < 2 {
                continue;
            }
            let (raw, cone) = conjoin::and_scoped(&placed[0], &placed[1], eng.limits()).unwrap();
            let h = contract_in(raw, Some(&cone), eng.limits()).unwrap();
            eng.limits().discard(cone);
            let mut want = eng.and(&embedded[0], &embedded[1]).unwrap();
            eng.minimize(&mut want).unwrap();
            assert_eq!(fingerprint(&h), fingerprint(&skip(&want)));
            done += 1;
        }
    }
}

/// Wide sparse functions, whose stored conjunctions hold long chains of
/// twins: the skip form of the raw stored result, contracted, is the
/// canonical skip form; and so is the contracted skip conjunction.
#[test]
fn contraction_of_raw_stored_conjunctions_is_canonical() {
    let mut rng = Lcg::new(14);
    let mut done = 0;
    while done < 400 {
        let n = 8 + rng.below(11) as u32;
        let shape = CnfShape { clauses: 1 + rng.below(6) as usize, width: 3 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let mut raw = eng.and(&f, &g).unwrap();
            eng.reduce(&mut raw, crate::reduce::ReductionPlan::Prune).unwrap();
            let mut fun = raw.clone();
            eng.minimize(&mut fun).unwrap();
            let want = skip(&fun);
            assert_eq!(fingerprint(&contract(skip(&raw), eng.limits()).unwrap()), fingerprint(&want));
            let (sf, sg) = (skip(&f), skip(&g));
            let sg = SkipForm { shape: Arc::clone(&sf.shape), ..sg };
            let (raw, cone) = conjoin::and_scoped(&sf, &sg, eng.limits()).unwrap();
            let h = contract_in(raw, Some(&cone), eng.limits()).unwrap();
            eng.limits().discard(cone);
            assert_eq!(fingerprint(&h), fingerprint(&want));
            done += 1;
        }
    }
}

/// The public operations against the stored ones: a fold of clauses with
/// `skip_and_clause`, conjunctions, disjunctions and negations of the
/// results, each the canonical skip form of the stored result, on every
/// vtree shape (a forwarder at the root included).
#[test]
fn public_operations_match_the_stored_ones() {
    let mut rng = Lcg::new(11);
    for round in 0..120 {
        let n = 2 + rng.below(8) as u32;
        let shape = CnfShape { clauses: 2 + n as usize, width: 3 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let canon = |s: &crate::SkipTdd, t: &Tdd| {
                let want = eng.to_skip(t).unwrap();
                assert_eq!(fingerprint(&s.form), fingerprint(&want.form), "round {round}");
                assert_eq!(eng.skip_model_count(s).unwrap(), eng.model_count(t).unwrap());
                let back = eng.to_stored(s).unwrap();
                assert!(eng.equivalent(&back, t).unwrap());
                assert_eq!(s.pair_count(), want.pair_count());
            };
            let fold = |cnf: &[Vec<i32>]| {
                let mut stored = Tdd::one(&vtree);
                let mut s = eng.to_skip(&stored).unwrap();
                for c in cnf {
                    stored = eng.and_clause(stored, c).unwrap();
                    s = eng.skip_and_clause(&s, c).unwrap();
                    canon(&s, &stored);
                }
                (s, stored)
            };
            let (f, ft) = fold(&rand_cnf(&mut rng, n, shape));
            let (g, gt) = fold(&rand_cnf(&mut rng, n, shape));
            canon(&eng.skip_and(&f, &g).unwrap(), &eng.and(&ft, &gt).unwrap());
            canon(&eng.skip_or(&f, &g).unwrap(), &eng.or(ft.clone(), gt.clone()).unwrap());
            canon(&eng.skip_negate(&f).unwrap(), &eng.negate(&ft).unwrap());
            // An operand converted on its own is accepted.
            let h = eng.to_skip(&gt).unwrap();
            canon(&eng.skip_and(&f, &h).unwrap(), &eng.and(&ft, &gt).unwrap());
            canon(&eng.skip_and_clause::<i32>(&f, &[]).unwrap(), &Tdd::zero(&vtree));
        }
    }
}

#[test]
fn skip_operands_must_share_a_vtree() {
    let eng = Engine::new();
    let a = Arc::new(crate::Vtree::linear(3));
    let b = Arc::new(crate::Vtree::linear(3));
    let f = eng.to_skip(&Tdd::one(&a)).unwrap();
    let g = eng.to_skip(&Tdd::one(&b)).unwrap();
    assert_eq!(eng.skip_and(&f, &g).unwrap_err(), crate::OperationError::VtreeMismatch);
}

/// A fold of small batches as a compile merges them, each a few clauses
/// over a window of variables, so that the accumulator is `true`-free over
/// most of the levels a batch is `true` at: every step is the canonical
/// skip form of the stored fold, from either side.
#[test]
fn fold_of_small_batches_is_canonical() {
    let mut rng = Lcg::new(15);
    for round in 0..60 {
        let n = 4 + rng.below(10) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut stored = Tdd::one(&vtree);
            let mut acc = eng.to_skip(&stored).unwrap();
            for _ in 0..5 {
                let lo = 1 + rng.below(n as u64) as i32;
                let hi = (lo + rng.below(3) as i32).min(n as i32);
                let clauses: Vec<Vec<i32>> = (0..1 + rng.below(3))
                    .map(|_| {
                        (0..1 + rng.below(3))
                            .map(|_| {
                                let v = lo + rng.below((hi - lo + 1) as u64) as i32;
                                if rng.below(2) == 0 { v } else { -v }
                            })
                            .collect()
                    })
                    .collect();
                let batch = compile_clauses_on(&eng, &vtree, &clauses);
                let sb = eng.to_skip(&batch).unwrap();
                stored = eng.and(&stored, &batch).unwrap();
                eng.minimize(&mut stored).unwrap();
                let want = eng.to_skip(&stored).unwrap();
                let (ab, ba) = (eng.skip_and(&acc, &sb).unwrap(), eng.skip_and(&sb, &acc).unwrap());
                assert_eq!(fingerprint(&ab.form), fingerprint(&want.form), "round {round}");
                assert_eq!(fingerprint(&ba.form), fingerprint(&want.form), "round {round}");
                acc = ab;
            }
        }
    }
}

#[test]
fn skip_operations_stop_at_the_memory_budget() {
    use crate::limits::LimitConfig;
    use crate::OperationError;
    let (eng, f, g) = cases(11, 40).into_iter().max_by_key(|c| c.1.pair_count() + c.2.pair_count()).unwrap();
    let (sf, sg) = (eng.to_skip(&f).unwrap(), eng.to_skip(&g).unwrap());
    let _guard = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(64)));
    assert_eq!(eng.skip_and(&sf, &sg).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.skip_negate(&sf).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.to_skip(&f).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.to_stored(&sf).err(), Some(OperationError::OverBudget));
}
