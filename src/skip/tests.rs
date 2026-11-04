use std::sync::Arc;

use super::convert::{from_tdd, to_tdd};
use super::{conjoin, contract::{contract, contract_in}, ix, mk, vt, Ref, Shape, SkipForm, ONE, ZREF};
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::{Engine, Tdd};

fn skip(f: &Tdd) -> SkipForm {
    from_tdd(f, Arc::new(Shape::of(f.vtree())), Engine::new().limits()).expect("structural")
}

/// The stored diagram of `s`'s function: its negation built where `s`
/// marks one.
fn built(eng: &Engine, s: &crate::SkipTdd) -> SkipForm {
    s.built(eng.limits()).unwrap().into_owned().form
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
/// The node types hold: a side read from child `c` names a node in `c`'s
/// subtree, and no level stores a node that is `true` over a leaf child.
fn assert_node_types(s: &SkipForm) {
    let sh = &s.shape;
    let crosses = |c: u32, x: Ref| {
        assert!(x == ZREF || sh.within(vt(x), c), "a side read from {c} names a node outside it");
    };
    crosses(sh.root, s.out);
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        let lvl = &s.levels[t as usize];
        for i in 0..lvl.width() {
            let node = lvl.node(i as u32);
            for &(a, b) in node {
                crosses(l, a);
                crosses(r, b);
            }
            if node.len() == 1 {
                let (a, b) = node[0];
                let skippable = (sh.is_leaf(l) && a == mk(l, ONE)) || (sh.is_leaf(r) && b == mk(r, ONE));
                assert!(!skippable, "level {t} stores a skippable node");
            }
        }
    }
}

fn fingerprint(s: &SkipForm) -> (u64, Vec<Vec<u64>>) {
    use std::hash::{Hash, Hasher};
    assert_node_types(s);
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

/// Twins in groups of three: in a canonical form, every node of a level
/// with three pairs or more that only its parent level names is split, a
/// level at a time, into three nodes, one per pair in turn, each paired with every partner of the
/// original; the contraction merges each group back into the canonical
/// form, on every vtree shape.
#[test]
fn contraction_merges_twins_three_at_a_time() {
    let mut rng = Lcg::new(31);
    let mut groups = 0;
    for _ in 0..100 {
        let n = 10 + rng.below(8) as u32;
        let shape = CnfShape { clauses: n as usize, width: 3 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut fun = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            eng.minimize(&mut fun).unwrap();
            let want = skip(&fun);
            let sh = Arc::clone(&want.shape);
            // Per level, how many references name each of its nodes from
            // anywhere but its parent level.
            let mut named: Vec<Vec<u32>> = (0..sh.num_nodes()).map(|t| vec![0; want.levels[t].width()]).collect();
            for &u in &sh.topdown {
                for &(a, b) in &want.levels[u as usize].pairs {
                    for r in [a, b] {
                        if r != ZREF && !sh.is_leaf(vt(r)) && sh.parent[vt(r) as usize] != u {
                            named[vt(r) as usize][ix(r) as usize] += 1;
                        }
                    }
                }
            }
            if want.out != ZREF && !sh.is_leaf(vt(want.out)) {
                named[vt(want.out) as usize][ix(want.out) as usize] += 1;
            }
            for &t in &sh.topdown {
                let lvl = &want.levels[t as usize];
                if t == sh.root || !(0..lvl.width()).any(|i| lvl.node(i as u32).len() >= 3 && named[t as usize][i] == 0) {
                    continue;
                }
                let p = sh.parent[t as usize];
                let mut s = want.clone();
                let lvl = s.levels.take(t as usize);
                let width = lvl.width() as u32;
                let (mut start, mut pairs, mut extra) = (vec![0u32], Vec::new(), Vec::new());
                let mut copies: Vec<Option<[Ref; 2]>> = vec![None; width as usize];
                for i in 0..width {
                    let node = lvl.node(i);
                    let mut parts: [Vec<(Ref, Ref)>; 3] = Default::default();
                    for (k, &q) in node.iter().enumerate() {
                        parts[k % 3].push(q);
                    }
                    let ok = node.len() >= 3 && named[t as usize][i as usize] == 0 && parts.iter().all(|part| {
                        let mut x = part.clone();
                        super::normalize(&sh, t, &mut x).is_ok() && x == *part
                    });
                    if ok {
                        let base = width + extra.len() as u32;
                        copies[i as usize] = Some([mk(t, base), mk(t, base + 1)]);
                        extra.push(parts[1].clone());
                        extra.push(parts[2].clone());
                        pairs.extend_from_slice(&parts[0]);
                        groups += 1;
                    } else {
                        pairs.extend_from_slice(node);
                    }
                    start.push(pairs.len() as u32);
                }
                for part in &extra {
                    pairs.extend_from_slice(part);
                    start.push(pairs.len() as u32);
                }
                s.levels.set(t as usize, super::SLevel { start, pairs });
                // Every pair of the parent level that names a split node, once per
                // copy.
                let up = s.levels.take(p as usize);
                let (mut start, mut pairs) = (vec![0u32], Vec::new());
                for i in 0..up.width() as u32 {
                    for &(a, b) in up.node(i) {
                        pairs.push((a, b));
                        for (r, left) in [(a, true), (b, false)] {
                            let cs = if r != ZREF && vt(r) == t { copies[ix(r) as usize] } else { None };
                            for c in cs.into_iter().flatten() {
                                pairs.push(if left { (c, b) } else { (a, c) });
                            }
                        }
                    }
                    start.push(pairs.len() as u32);
                }
                s.levels.set(p as usize, super::SLevel { start, pairs });
                assert_eq!(fingerprint(&contract(s, eng.limits()).unwrap()), fingerprint(&want), "vtree {:?}, level {t}", sh.kids);
            }
        }
    }
    assert!(groups >= 500, "{groups} groups split");
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
                let held = built(&eng, s);
                assert_eq!(fingerprint(&held), fingerprint(&want.form), "round {round}");
                assert_eq!(eng.skip_model_count(s).unwrap(), eng.model_count(t).unwrap());
                let back = eng.to_stored(s).unwrap();
                assert!(eng.equivalent(&back, t).unwrap());
                assert_eq!(held.size().pairs as usize, want.pair_count());
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
    // A negation is marked without memory, and built when read.
    let nf = eng.skip_negate(&sf).unwrap();
    assert_eq!(eng.to_stored(&nf).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.skip_and(&nf, &sg).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.skip_or(&sf, &sg).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.skip_and_not(&sf, &sg).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.to_skip(&f).err(), Some(OperationError::OverBudget));
    assert_eq!(eng.to_stored(&sf).err(), Some(OperationError::OverBudget));
}

#[test]
fn skip_and_clause_handles_edge_cases_and_large_chains() {
    use crate::vtree::VarId;
    use crate::OperationError;
    let eng = Engine::new();
    let vtree = Arc::new(crate::Vtree::linear(64));
    let one = eng.to_skip(&Tdd::one(&vtree)).unwrap();

    // Out-of-vtree variable returns `VariableNotInVtree`.
    assert_eq!(
        eng.skip_and_clause(&one, &[65i32]).unwrap_err(),
        OperationError::VariableNotInVtree(VarId(65))
    );

    // Tautological clause leaves the operand unchanged.
    let c1 = eng.skip_and_clause(&one, &[-3i32, 40, -60]).unwrap();
    let taut = eng.skip_and_clause(&c1, &[12i32, -40, -12]).unwrap();
    assert_eq!(fingerprint(&taut.form), fingerprint(&c1.form));

    // Duplicate literals and far-apart literals on a 64-var linear chain match stored `and_clause`.
    let mut stored = Tdd::one(&vtree);
    let mut skip = one;
    for clause in [
        vec![-1i32, 32, -64],
        vec![2i32, 2, -32, 63],
        vec![-2i32, 15, -48, 64],
        vec![1i32, -15, 32],
    ] {
        stored = eng.and_clause(stored, &clause).unwrap();
        eng.minimize(&mut stored).unwrap();
        skip = eng.skip_and_clause(&skip, &clause).unwrap();
        let want = eng.to_skip(&stored).unwrap();
        assert_eq!(fingerprint(&skip.form), fingerprint(&want.form));
    }
}

/// Weights per variable, not normalized and some negative, so that a
/// skipped variable's `true` is neither 1 nor 2.
struct Weighted;

impl crate::diagram::EvalAlgebra for Weighted {
    type Value = f64;
    fn zero(&self) -> f64 {
        0.0
    }
    fn leaf(&self, var: crate::vtree::VarId, label: crate::diagram::LeafLabel) -> f64 {
        let (p, q) = (0.3 + 0.17 * var.0 as f64, 0.9 - 0.41 * var.0 as f64);
        match label {
            crate::diagram::LeafLabel::Pos => p,
            crate::diagram::LeafLabel::Neg => q,
            _ => p + q,
        }
    }
    fn add_assign(&self, acc: &mut f64, other: &f64) {
        *acc += other;
    }
    fn mul(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// The canonical skip form of a stored diagram's function.
fn canonical(eng: &Engine, f: Tdd) -> SkipForm {
    let mut f = f;
    eng.minimize(&mut f).unwrap();
    skip(&f)
}

#[test]
fn evaluation_matches_the_stored_form() {
    for (eng, f, g) in cases(11, 200) {
        for h in [f.clone(), eng.and(&f, &g).unwrap(), eng.negate(&f).unwrap()] {
            let s = eng.to_skip(&h).unwrap();
            let (want, got) = (eng.evaluate(&h, &Weighted).unwrap(), eng.skip_evaluate(&s, &Weighted).unwrap());
            assert!(close(want, got), "{want} against {got}");
            assert_eq!(eng.skip_model_count(&s).unwrap(), eng.model_count(&h).unwrap());
        }
    }
}

/// Up to three literals over the vtree's variables, opposite ones included.
fn assignment(rng: &mut Lcg, f: &Tdd) -> Vec<i32> {
    let n = f.vtree().num_leaves();
    (0..1 + rng.below(3)).map(|_| {
        let v = 1 + rng.below(n as u64) as i32;
        if rng.below(2) == 0 { v } else { -v }
    }).collect()
}

#[test]
fn conditioning_is_the_stored_cofactor_in_canonical_skip_form() {
    let mut rng = Lcg::new(12);
    for (eng, f, g) in cases(12, 200) {
        let a = assignment(&mut rng, &f);
        let s = eng.skip_condition(&eng.to_skip(&f).unwrap(), &a).unwrap();
        let want = canonical(&eng, eng.condition(f.clone(), a.iter().copied()).unwrap());
        assert_eq!(fingerprint(&s.form), fingerprint(&want), "assignment {a:?}");
        assert_stored_order(&s.form, &format!("assignment {a:?}"));
        // The result is a working operand: conjoined further, it is still
        // the canonical form.
        let sg = eng.to_skip(&g).unwrap();
        let both = eng.skip_and(&s, &sg).unwrap();
        let stored = eng.and(eng.condition(f, a.iter().copied()).unwrap(), &g).unwrap();
        assert_eq!(fingerprint(&both.form), fingerprint(&canonical(&eng, stored)));
    }
}

#[test]
fn quantifiers_match_the_stored_form() {
    let mut rng = Lcg::new(13);
    for (eng, f, g) in cases(13, 150) {
        let vars: Vec<crate::vtree::VarId> = assignment(&mut rng, &f).iter().map(|l| crate::vtree::VarId(l.unsigned_abs())).collect();
        let s = eng.to_skip(&f).unwrap();
        let e = eng.skip_exists(&s, &vars).unwrap();
        assert_eq!(fingerprint(&e.form), fingerprint(&canonical(&eng, eng.exists_vars(f.clone(), &vars).unwrap())));
        let a = eng.skip_forall(&s, &vars).unwrap();
        assert_eq!(fingerprint(&built(&eng, &a)), fingerprint(&canonical(&eng, eng.forall_vars(f.clone(), &vars).unwrap())));
        let sg = eng.to_skip(&g).unwrap();
        let ae = eng.skip_and_exists(&s, &sg, &vars).unwrap();
        assert_eq!(fingerprint(&ae.form), fingerprint(&canonical(&eng, eng.and_exists(f.clone(), g.clone(), &vars).unwrap())));
    }
}

/// The variables of the leaves under vtree node `t`.
fn vars_under(sh: &Shape, t: u32) -> Vec<crate::vtree::VarId> {
    let mut out = Vec::new();
    let mut stack = vec![t];
    while let Some(u) = stack.pop() {
        let (l, r) = sh.kids[u as usize];
        if sh.is_leaf(u) {
            out.push(crate::vtree::VarId(sh.var[u as usize]));
        } else {
            stack.extend([l, r]);
        }
    }
    out
}

/// `k` distinct variables of `1..=n`, drawn at random.
fn draw_vars(rng: &mut Lcg, n: u32, k: u32) -> Vec<crate::vtree::VarId> {
    let mut pool: Vec<u32> = (1..=n).collect();
    (0..k).map(|_| crate::vtree::VarId(pool.swap_remove(rng.below(pool.len() as u64) as usize))).collect()
}

/// The skip quantifiers of `f` (and of `f ∧ g`) over `vars` against the
/// stored ones, in canonical skip form.
fn assert_quantifiers(eng: &Engine, f: &Tdd, g: &Tdd, vars: &[crate::vtree::VarId], what: &str) {
    let s = eng.to_skip(f).unwrap();
    let e = eng.skip_exists(&s, vars).unwrap();
    assert_eq!(fingerprint(&e.form), fingerprint(&canonical(eng, eng.exists_vars(f.clone(), vars).unwrap())), "{what}: exists {vars:?}");
    let a = eng.skip_forall(&s, vars).unwrap();
    assert_eq!(fingerprint(&built(eng, &a)), fingerprint(&canonical(eng, eng.forall_vars(f.clone(), vars).unwrap())), "{what}: forall {vars:?}");
    let sg = eng.to_skip(g).unwrap();
    let ae = eng.skip_and_exists(&s, &sg, vars).unwrap();
    assert_eq!(fingerprint(&ae.form), fingerprint(&canonical(eng, eng.and_exists(f.clone(), g.clone(), vars).unwrap())), "{what}: and_exists {vars:?}");
}

/// One pass for the whole set: sets of every size from one variable to
/// all, and every subtree of the vtree whole, with or without a variable
/// outside it, on every vtree shape.
#[test]
fn quantifying_sets_of_every_size_matches_the_stored_form() {
    let mut rng = Lcg::new(31);
    for (eng, f, g) in cases(31, 150) {
        let n = f.vtree().num_leaves();
        for k in 1..=n {
            let vars = draw_vars(&mut rng, n, k);
            assert_quantifiers(&eng, &f, &g, &vars, "random");
        }
        let sh = Shape::of(f.vtree());
        for &t in &sh.topdown {
            let mut vars = vars_under(&sh, t);
            if rng.below(2) == 0 {
                vars.push(crate::vtree::VarId(1 + rng.below(n as u64) as u32));
            }
            assert_quantifiers(&eng, &f, &g, &vars, "subtree");
        }
    }
}

/// Larger diagrams, a third to all but a few of their variables
/// quantified: random formulas and tables ruled out a row at a time, whose
/// spines carry skips past many levels.
#[test]
fn quantifying_most_variables_of_larger_diagrams_matches_the_stored_form() {
    let mut rng = Lcg::new(32);
    for round in 0..10 {
        let n = 12 + rng.below(7) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = if round % 2 == 0 {
                compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: 2 * n as usize, width: 3 }))
            } else {
                let cubes: Vec<Vec<i32>> = (0..2 * n)
                    .map(|_| {
                        (0..2 + rng.below(5))
                            .map(|_| {
                                let v = 1 + rng.below(n as u64) as i32;
                                if rng.below(2) == 0 { v } else { -v }
                            })
                            .collect()
                    })
                    .collect();
                none_of_stored(&eng, &vtree, &cubes)
            };
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            for k in [n / 3, n / 2, 2 * n / 3, n - 2] {
                let vars = draw_vars(&mut rng, n, k);
                assert_quantifiers(&eng, &f, &g, &vars, shape);
            }
        }
    }
}

#[test]
fn quantifier_edge_cases_match_the_stored_form() {
    use crate::vtree::VarId;
    for n in 1..=4u32 {
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let all: Vec<VarId> = (1..=n).map(VarId).collect();
            let fs = [
                Tdd::zero(&vtree),
                Tdd::one(&vtree),
                compile_clauses_on(&eng, &vtree, &[vec![1]]),
                compile_clauses_on(&eng, &vtree, &[vec![-(n as i32)], vec![1, n as i32]]),
            ];
            for f in &fs {
                for vars in [&[][..], &all[..], &all[..1], &all[n as usize - 1..], &[VarId(1), VarId(1)][..]] {
                    assert_quantifiers(&eng, f, &fs[3], vars, shape);
                }
            }
            let s = eng.to_skip(&fs[2]).unwrap();
            assert!(matches!(eng.skip_exists(&s, &[VarId(n + 1)]), Err(crate::OperationError::VariableNotInVtree(_))));
        }
    }
}

/// Sets whose `∃` is `true` at the root or only below it. `d` is a
/// disjunction of cubes with distinct variables, one of them over quantified
/// variables alone, and `¬d` is quantified too, so that `∀` meets the same
/// case. The rounds: that cube beside random ones (`true` at the root); a
/// guard literal outside the set in every cube (`∃` is the guard, `true`
/// below it); the guard's two literals, each with a cube over the set (`true`
/// at the root through both); and a CNF each of whose clauses a quantified
/// variable of its own satisfies (`true` at the root).
#[test]
fn quantifying_to_true_at_the_root_and_below_matches_the_stored_form() {
    let mut rng = Lcg::new(33);
    for round in 0..16 {
        let n = 6 + rng.below(7) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let vars = draw_vars(&mut rng, n, n / 2);
            let inside: Vec<u32> = vars.iter().map(|v| v.0).collect();
            let outside: Vec<u32> = (1..=n).filter(|v| !inside.contains(v)).collect();
            let signed = |rng: &mut Lcg, v: u32| if rng.below(2) == 0 { v as i32 } else { -(v as i32) };
            // A cube of `k` distinct variables of `pool`, beside `with`.
            let cube = |rng: &mut Lcg, pool: &[u32], k: usize, with: &[i32]| -> Vec<i32> {
                let mut pool: Vec<u32> = pool.iter().copied().filter(|&v| with.iter().all(|l| l.unsigned_abs() != v)).collect();
                let mut c = with.to_vec();
                for _ in 0..k.min(pool.len()) {
                    let v = pool.swap_remove(rng.below(pool.len() as u64) as usize);
                    c.push(signed(rng, v));
                }
                c
            };
            let all: Vec<u32> = (1..=n).collect();
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            let guard = signed(&mut rng, outside[0]);
            let f = match round % 4 {
                3 => {
                    let mut clauses: Vec<Vec<i32>> = Vec::new();
                    for &v in &inside {
                        let (k, own) = (1 + rng.below(2) as usize, signed(&mut rng, v));
                        clauses.push(cube(&mut rng, &all, k, &[own]));
                    }
                    compile_clauses_on(&eng, &vtree, &clauses)
                }
                r => {
                    let mut cubes: Vec<Vec<i32>> = Vec::new();
                    for _ in 0..n {
                        let with: Vec<i32> = match r {
                            0 => Vec::new(),
                            1 => vec![guard],
                            _ => vec![if rng.below(2) == 0 { guard } else { -guard }],
                        };
                        let k = 2 + rng.below(3) as usize;
                        cubes.push(cube(&mut rng, &all, k, &with));
                    }
                    let k = 1 + rng.below(3) as usize;
                    match r {
                        0 => cubes.push(cube(&mut rng, &inside, k, &[])),
                        1 => cubes.push(cube(&mut rng, &inside, k, &[guard])),
                        _ => {
                            cubes.push(cube(&mut rng, &inside, k, &[guard]));
                            cubes.push(cube(&mut rng, &inside, k, &[-guard]));
                        }
                    }
                    eng.negate(none_of_stored(&eng, &vtree, &cubes)).unwrap()
                }
            };
            let expect = if round % 4 == 1 { compile_clauses_on(&eng, &vtree, &[vec![guard]]) } else { Tdd::one(&vtree) };
            assert!(eng.equivalent(&eng.exists_vars(f.clone(), &vars).unwrap(), &expect).unwrap(), "{shape} round {round}: ∃ is not the case's");
            let not = eng.negate(&f).unwrap();
            for h in [&f, &not] {
                assert_quantifiers(&eng, h, &g, &vars, shape);
            }
        }
    }
}

#[test]
fn a_satisfying_assignment_is_a_model_and_equivalence_is_the_stored_one() {
    for (eng, f, g) in cases(14, 200) {
        let s = eng.to_skip(&f).unwrap();
        match eng.skip_satisfying_assignment(&s).unwrap() {
            None => assert!(f.is_zero()),
            Some(model) => {
                assert_eq!(model.len(), f.vtree().num_leaves() as usize);
                assert!(!eng.condition(f.clone(), model).unwrap().is_zero());
            }
        }
        let sg = eng.to_skip(&g).unwrap();
        assert_eq!(eng.skip_equivalent(&s, &sg).unwrap(), eng.equivalent(&f, &g).unwrap());
        assert!(eng.skip_equivalent(&s, &eng.to_skip(&eng.to_stored(&s).unwrap()).unwrap()).unwrap());
        // The same function built two ways, numbered differently.
        let (a, b) = (eng.skip_or(&s, &sg).unwrap(), eng.skip_negate(&eng.skip_and(&eng.skip_negate(&sg).unwrap(), &eng.skip_negate(&s).unwrap()).unwrap()).unwrap());
        assert!(eng.skip_equivalent(&a, &b).unwrap());
        let fg = eng.skip_and(&s, &sg).unwrap();
        assert_eq!(eng.skip_equivalent(&fg, &s).unwrap(), eng.equivalent(&eng.to_stored(&fg).unwrap(), &f).unwrap());
    }
}

#[test]
fn the_support_is_the_stored_one() {
    for (eng, f, g) in cases(17, 200) {
        let (s, sg) = (eng.to_skip(&f).unwrap(), eng.to_skip(&g).unwrap());
        let and = eng.skip_and(&s, &sg).unwrap();
        let or = eng.skip_or(&s, &sg).unwrap();
        let not = eng.skip_negate(&s).unwrap();
        for x in [&s, &sg, &and, &or, &not] {
            assert_eq!(eng.skip_support(x).unwrap(), eng.support(&eng.to_stored(x).unwrap()).unwrap());
        }
    }
}

#[test]
fn embedding_is_the_stored_embed_in_canonical_skip_form() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(16);
    let mut done = 0;
    while done < 300 {
        let n = 3 + rng.below(9) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.is_empty() {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { continue };
            let small = eng.bind_vtree(small);
            let m = keep.len() as u32;
            let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: 2, width: 3 }));
            let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
            let s = eng.to_skip(&f).unwrap();
            let (want, _) = eng.embed(&f, &vtree, map).unwrap();
            let got = eng.skip_embed(&s, &vtree, map).unwrap();
            assert_eq!(fingerprint(&got.form), fingerprint(&canonical(&eng, want.clone())));
            // The copy is a working operand.
            let both = eng.skip_and(&got, &eng.to_skip(&g).unwrap()).unwrap();
            assert_eq!(fingerprint(&both.form), fingerprint(&canonical(&eng, eng.and(&want, &g).unwrap())));
            // Up to mirrors: some children of the destination swapped.
            let mut mirror = (*vtree).clone();
            for &t in vtree.internal_bottomup_slice() {
                if rng.below(2) == 0 {
                    mirror.swap_children(t);
                }
            }
            let mirror = eng.bind_vtree(mirror);
            let (want, _) = eng.embed_mirrored(&f, &mirror, map).unwrap();
            let got = eng.skip_embed_mirrored(&s, &mirror, map).unwrap();
            assert_eq!(fingerprint(&got.form), fingerprint(&canonical(&eng, want)));
            done += 1;
        }
    }
}

/// An embedding that copies the levels whose nodes keep their shapes
/// writes what placing every node writes, level for level and node for
/// node, mirrored or not, on every vtree shape: the copied levels, the
/// placed ones beside them (a widened leaf child), and the nodes lifted
/// and `true` built where an added free side holds several variables.
#[test]
fn copied_levels_are_the_placed_ones() {
    use super::embed::embed_copying;
    use crate::vtree::VarId;
    let mut rng = Lcg::new(31);
    let (mut done, mut copied, mut placed_beside, mut added) = (0, 0, 0, 0);
    while done < 300 {
        let n = 3 + rng.below(9) as u32;
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.is_empty() {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { continue };
            let small = eng.bind_vtree(small);
            let m = keep.len() as u32;
            let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
            let s = eng.to_skip(&f).unwrap();
            let mut mirror = (*vtree).clone();
            for &t in vtree.internal_bottomup_slice() {
                if rng.below(2) == 0 {
                    mirror.swap_children(t);
                }
            }
            let mirror = eng.bind_vtree(mirror);
            let plans = [eng.embedding_plan(&small, &vtree, map).unwrap(), eng.embedding_plan_mirrored(&small, &mirror, map).unwrap()];
            for plan in &plans {
                let layout = plan.layout();
                let (img, mirrored) = (&layout.embedding.levels, &layout.mirrored);
                let d = Arc::new(Shape::of(plan.destination()));
                let want = embed_copying(&s.form, Arc::clone(&d), img, mirrored, false, eng.limits()).unwrap();
                let got = embed_copying(&s.form, Arc::clone(&d), img, mirrored, true, eng.limits()).unwrap();
                assert_eq!(got.out, want.out, "{name}");
                assert_eq!(got.active, want.active, "{name}");
                for t in 0..d.num_nodes() {
                    assert_eq!(got.levels[t], want.levels[t], "{name}, level {t}");
                }
                // What the copy met: levels copied, levels placed beside
                // them, and nodes the destination adds.
                let src = &s.form.shape;
                let widened = |t: u32| {
                    let (l, r) = src.kids[t as usize];
                    [l, r].iter().any(|&x| src.is_leaf(x) && d.parent[img[x as usize].idx()] != img[t as usize].0)
                };
                copied += s.form.active.iter().filter(|&&t| !widened(t)).count();
                placed_beside += s.form.active.iter().filter(|&&t| widened(t)).count();
                added += usize::from(got.size().nodes > s.form.size().nodes);
            }
            done += 1;
        }
    }
    assert!(copied > 500 && placed_beside > 100 && added > 100, "copied {copied}, placed {placed_beside}, added {added}");
}

#[test]
fn projection_is_the_stored_projection_in_canonical_skip_form() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(17);
    let mut done = 0;
    while done < 300 {
        let n = 3 + rng.below(9) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.is_empty() {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let m = keep.len() as u32;
            // A diagram that reads dropped variables, and one that does not.
            let wide = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            let small = eng.bind_vtree(vtree.project_to_vars(local, m).unwrap());
            let narrow = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let (narrow, _) = eng.embed(&narrow, &vtree, |v| VarId(keep[v.0 as usize - 1])).unwrap();
            for (f, reads) in [(wide, true), (narrow, false)] {
                let want = eng.project_to_vars(&f, local, m).unwrap();
                let s = eng.to_skip(&f).unwrap();
                let got = eng.skip_project_to_vars(&s, local, m).unwrap();
                assert_eq!(fingerprint(&got.form), fingerprint(&canonical(&eng, want.clone())));
                assert_eq!(eng.skip_model_count(&got).unwrap(), eng.model_count(&want).unwrap());
                if !reads {
                    // Moved in one pass, without quantifying.
                    let plan = crate::restructure::embed::Plan::build(&eng, got.vtree(), &vtree, |v| VarId(keep[v.0 as usize - 1]), false).unwrap();
                    let shape = Arc::new(Shape::of(got.vtree()));
                    assert!(super::project::project(&s.form, &shape, &plan, eng.limits()).unwrap().is_some());
                }
            }
            done += 1;
        }
    }
}

/// Random literal weights over variables 1 to `n`, some negative with
/// `signed`.
fn draw_store(rng: &mut Lcg, n: u32, signed: bool, arithmetic: crate::diagram::Arithmetic) -> crate::diagram::WeightStore {
    use crate::diagram::{LiteralWeights, RationalWeights, WeightStore};
    let mut draw = || {
        let w = crate::test_helpers::rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64);
        if signed && rng.below(4) == 0 { -w } else { w }
    };
    let lits: Vec<_> = (0..=n).map(|_| LiteralWeights { negative: draw(), positive: draw() }).collect();
    WeightStore::new(RationalWeights::from_literals(&lits), arithmetic)
}

/// The weighted count of `f` under `ws`, from the stored form.
fn wmc(eng: &Engine, f: &Tdd, ws: &crate::diagram::WeightStore) -> crate::diagram::WeightValue {
    let mut f = f.clone();
    f.set_weights(ws.empty_like()).unwrap();
    eng.weighted_value(&f).unwrap().expect("weights are attached")
}

fn weighed(eng: &Engine, f: &Tdd, ws: &crate::diagram::WeightStore) -> crate::SkipTdd {
    let mut s = eng.to_skip(f).unwrap();
    s.set_weights(ws.empty_like()).unwrap();
    s
}

/// A value agrees with an exact one: equal, or for a bounded one to `tol`
/// relative.
fn agree(got: &crate::diagram::WeightValue, want: &crate::diagram::WeightValue, tol: f64) -> bool {
    let float = |v: &crate::diagram::WeightValue| {
        if let Some(g) = v.as_scaled() {
            Some(g.to_f64())
        } else {
            v.as_log().map(|g| f64::from(g.sign) * g.ln_abs.exp())
        }
    };
    let (a, b) = match (float(got), float(want)) {
        (None, None) => return got.as_rational() == want.as_rational(),
        (a, b) => {
            let exact = |v: &crate::diagram::WeightValue| crate::diagram::ScaledFloat::from_rational(&v.as_rational()).to_f64();
            (a.unwrap_or_else(|| exact(got)), b.unwrap_or_else(|| exact(want)))
        }
    };
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-300)
}

/// The internal nodes and, with `leaves`, the leaves of `tree` whose
/// variables `live` does not hold.
fn untouched(tree: &crate::Vtree, live: &std::collections::HashSet<u32>, leaves: bool) -> Vec<crate::vtree::VtreeIdx> {
    use crate::vtree::VtreeNode;
    tree.bottomup()
        .filter(|&t| leaves || !tree.node(t).is_leaf())
        .filter(|&t| tree.subtree(t).all(|x| match *tree.node(x) {
            VtreeNode::Leaf { var, .. } => !live.contains(&var.0),
            _ => true,
        }))
        .collect()
}

fn vars_of(cnf: &[Vec<i32>]) -> std::collections::HashSet<u32> {
    cnf.iter().flatten().map(|l| l.unsigned_abs()).collect()
}

/// No level holds two nodes with the same pairs.
fn shared(s: &crate::SkipTdd) -> bool {
    s.form.levels.iter().all(|l| {
        let mut nodes: Vec<Vec<(Ref, Ref)>> = (0..l.width() as u32).map(|i| {
            let mut p = l.node(i).to_vec();
            p.sort_unstable();
            p
        }).collect();
        let n = nodes.len();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len() == n
    })
}

/// Summing out the subtrees no test function reads keeps the weight of
/// every conjunction with such a function: `f ∧ g` with random targets
/// summed out, conjoined with a CNF over the other variables, weighs what
/// the whole conjunction does.
#[test]
fn summing_out_keeps_the_weight_of_every_extension() {
    use crate::diagram::Arithmetic;
    let mut rng = Lcg::new(21);
    for (k, (eng, f, g)) in cases(21, 200).into_iter().enumerate() {
        let tree = Arc::clone(f.vtree());
        let n = tree.num_leaves();
        let ws = draw_store(&mut rng, n, k % 2 == 1, Arithmetic::ExactRational);
        // A random set of variables stays live; the rest is summed out
        // where whole subtrees of it are.
        let live: std::collections::HashSet<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
        let targets: Vec<_> = untouched(&tree, &live, k % 3 == 0).into_iter().filter(|_| rng.below(2) == 0).collect();
        let (sf, sg) = (weighed(&eng, &f, &ws), weighed(&eng, &g, &ws));
        // A weighted Boolean diagram converts with its weights, both ways.
        let mut fw = f.clone();
        fw.set_weights(ws.empty_like()).unwrap();
        let fs = eng.to_skip(&fw).unwrap();
        assert!(agree(&eng.skip_weighted_value(&fs).unwrap().unwrap(), &wmc(&eng, &f, &ws), 0.0));
        assert!(agree(&eng.weighted_value(&eng.to_stored(&fs).unwrap()).unwrap().unwrap(), &wmc(&eng, &f, &ws), 0.0));
        let m = eng.skip_and_marginalizing(&sf, &sg, &targets).unwrap();
        let both = eng.and(&f, &g).unwrap();
        assert!(agree(&eng.skip_weighted_value(&m).unwrap().unwrap(), &wmc(&eng, &both, &ws), 0.0), "case {k}");
        // The other variables' functions: a CNF over them, conjoined.
        let inside: std::collections::HashSet<u32> = targets.iter().flat_map(|&t| tree.subtree(t)).filter_map(|x| match *tree.node(x) {
            crate::vtree::VtreeNode::Leaf { var, .. } => Some(var.0),
            _ => None,
        }).collect();
        for round in 0..3 {
            let cnf: Vec<Vec<i32>> = rand_cnf(&mut rng, n, CnfShape { clauses: 4, width: 3 })
                .into_iter()
                .filter(|c| c.iter().all(|l| !inside.contains(&l.unsigned_abs())))
                .collect();
            let h = compile_clauses_on(&eng, &tree, &cnf);
            let mh = if round == 0 {
                let mut s = m.clone();
                for c in &cnf {
                    s = eng.skip_and_clause(&s, c).unwrap();
                }
                s
            } else {
                eng.skip_and(&m, &eng.to_skip(&h).unwrap()).unwrap()
            };
            let want = wmc(&eng, &eng.and(&both, &h).unwrap(), &ws);
            assert!(agree(&eng.skip_weighted_value(&mh).unwrap().unwrap(), &want, 0.0), "case {k} round {round}");
            assert!(!m.has_marginal_level() || shared(&mh), "case {k} round {round}: equal nodes");
        }
    }
}

/// A walk that conjoins one clause at a time and sums out each subtree
/// the later clauses do not read, leaves included, weighs the CNF: exactly,
/// and under scaled floats and logs to within rounding.
#[test]
fn a_walk_that_sums_out_finished_subtrees_weighs_the_formula() {
    use crate::diagram::Arithmetic;
    let mut rng = Lcg::new(22);
    for num_vars in [3u32, 6, 9, 12] {
        for (shape, tree) in vtree_shapes(num_vars) {
            for round in 0..4 {
                let eng = Engine::new();
                let cnf = rand_cnf(&mut rng, num_vars, CnfShape { clauses: num_vars as usize + 3, width: 3 });
                let signed = round % 2 == 1;
                let seed = rng.below(1 << 30);
                let stored = compile_clauses_on(&eng, &tree, &cnf);
                let want = wmc(&eng, &stored, &draw_store(&mut Lcg::new(seed), num_vars, signed, Arithmetic::ExactRational));
                for arithmetic in [Arithmetic::ExactRational, Arithmetic::ScaledFloat, Arithmetic::SignedLog] {
                    let ws = draw_store(&mut Lcg::new(seed), num_vars, signed, arithmetic);
                    let mut acc = weighed(&eng, &Tdd::one(&tree), &ws);
                    for (i, c) in cnf.iter().enumerate() {
                        acc = eng.skip_and_clause(&acc, c).unwrap();
                        let targets = untouched(&tree, &vars_of(&cnf[i + 1..]), round >= 2);
                        acc = eng.skip_marginalize(&acc, &targets).unwrap();
                    }
                    assert!(acc.form.shape.kind[acc.form.shape.root as usize] >= super::SUMMED_VAR || acc.is_zero(), "{shape}: all summed out");
                    let got = eng.skip_weighted_value(&acc).unwrap().unwrap();
                    let tol = if signed { 1e-6 } else { 1e-12 };
                    assert!(agree(&got, &want, tol), "{shape}, {num_vars} vars, round {round}, {arithmetic:?}: {got:?} against {want:?}");
                    // The same walk joining each clause's diagram and summing
                    // out in one call.
                    let mut joined = weighed(&eng, &Tdd::one(&tree), &ws);
                    for (i, c) in cnf.iter().enumerate() {
                        let clause = eng.to_skip(&compile_clauses_on(&eng, &tree, std::slice::from_ref(c))).unwrap();
                        let targets = untouched(&tree, &vars_of(&cnf[i + 1..]), round >= 2);
                        joined = eng.skip_and_marginalizing(&joined, &clause, &targets).unwrap();
                    }
                    let got = eng.skip_weighted_value(&joined).unwrap().unwrap();
                    assert!(agree(&got, &want, tol), "{shape}, {num_vars} vars, round {round}, {arithmetic:?}, joined: {got:?} against {want:?}");
                    assert!(shared(&joined), "{shape}: joined shares its equal nodes");
                }
            }
        }
    }
}

/// Two diagrams that sum out the two sides of one node, each `true` over
/// the other's side, conjoin to the node summed out, and weigh their
/// conjunction.
#[test]
fn conjunction_of_two_summed_diagrams_sums_out_their_union() {
    use crate::diagram::Arithmetic;
    use crate::vtree::VtreeNode;
    let mut rng = Lcg::new(23);
    for round in 0..40 {
        let n = 4 + rng.below(8) as u32;
        for (shape, tree) in vtree_shapes(n) {
            let eng = Engine::new();
            let ws = draw_store(&mut rng, n, round % 2 == 1, Arithmetic::ExactRational);
            let inner: Vec<_> = tree.bottomup().filter(|&t| !tree.node(t).is_leaf()).collect();
            let a = inner[rng.below(inner.len() as u64) as usize];
            let VtreeNode::Internal { left, right, .. } = *tree.node(a) else { unreachable!() };
            let side = |t: crate::vtree::VtreeIdx| -> std::collections::HashSet<u32> {
                tree.subtree(t).filter_map(|x| match *tree.node(x) {
                    VtreeNode::Leaf { var, .. } => Some(var.0),
                    _ => None,
                }).collect()
            };
            let (lv, rv) = (side(left), side(right));
            let avoid = |cnf: Vec<Vec<i32>>, out: &std::collections::HashSet<u32>| -> Vec<Vec<i32>> {
                cnf.into_iter().filter(|c| c.iter().all(|l| !out.contains(&l.unsigned_abs()))).collect()
            };
            let shape_cnf = CnfShape { clauses: 2 * n as usize, width: 3 };
            let fc = avoid(rand_cnf(&mut rng, n, shape_cnf), &rv);
            let gc = avoid(rand_cnf(&mut rng, n, shape_cnf), &lv);
            let (f, g) = (compile_clauses_on(&eng, &tree, &fc), compile_clauses_on(&eng, &tree, &gc));
            let fm = eng.skip_marginalize(&weighed(&eng, &f, &ws), &[left]).unwrap();
            let gm = eng.skip_marginalize(&weighed(&eng, &g, &ws), &[right]).unwrap();
            let both = eng.skip_and(&fm, &gm).unwrap();
            assert!(shared(&both), "{shape}, round {round}: equal nodes");
            assert!(both.is_zero() || both.form.shape.kind[a.idx()] >= super::SUMMED_VAR, "{shape}: the node is summed out");
            let fg = eng.and(&f, &g).unwrap();
            assert!(agree(&eng.skip_weighted_value(&both).unwrap().unwrap(), &wmc(&eng, &fg, &ws), 0.0), "{shape}, round {round}");
            // From either side.
            let other = eng.skip_and(&gm, &fm).unwrap();
            assert!(agree(&eng.skip_weighted_value(&other).unwrap().unwrap(), &wmc(&eng, &fg, &ws), 0.0), "{shape}, round {round}, swapped");
            // And further conjoined with a CNF outside the node.
            let out: std::collections::HashSet<u32> = lv.union(&rv).copied().collect();
            let hc = avoid(rand_cnf(&mut rng, n, shape_cnf), &out);
            let h = compile_clauses_on(&eng, &tree, &hc);
            let all = eng.skip_and(&both, &weighed(&eng, &h, &ws)).unwrap();
            assert!(agree(&eng.skip_weighted_value(&all).unwrap().unwrap(), &wmc(&eng, &eng.and(&fg, &h).unwrap(), &ws), 0.0), "{shape}, round {round}, extended");
        }
    }
}

/// A summed-out subtree refuses what would read its variables.
#[test]
fn summed_out_subtrees_refuse_functions_of_their_variables() {
    use crate::diagram::Arithmetic;
    use crate::OperationError;
    let tree = Arc::new(crate::Vtree::balanced(4));
    let eng = Engine::new();
    let ws = draw_store(&mut Lcg::new(24), 4, false, Arithmetic::ExactRational);
    let f = compile_clauses_on(&eng, &tree, &[vec![1, 2], vec![-3, 4]]);
    let a = tree.bottomup().find(|&t| !tree.node(t).is_leaf()).unwrap();
    let m = eng.skip_marginalize(&weighed(&eng, &f, &ws), &[a]).unwrap();
    assert!(m.has_marginal_level());
    let var_in = tree.subtree(a).find_map(|x| match *tree.node(x) {
        crate::vtree::VtreeNode::Leaf { var, .. } => Some(var.0 as i32),
        _ => None,
    }).unwrap();
    assert_eq!(eng.skip_and_clause(&m, &[var_in]).unwrap_err(), OperationError::MarginalLevel(a));
    assert_eq!(eng.skip_negate(&m).unwrap_err(), OperationError::MarginalLevel(a));
    assert_eq!(eng.to_stored(&m).unwrap_err(), OperationError::MarginalLevel(a));
    assert!(matches!(eng.skip_model_count(&m), Err(OperationError::MarginalLevel(_))));
    // Both reading one subtree's values.
    assert_eq!(eng.skip_and(&m, &m).unwrap_err(), OperationError::MarginalLevel(a));
    // Without weights there is nothing to sum under.
    let bare = eng.to_skip(&f).unwrap();
    assert_eq!(eng.skip_marginalize(&bare, &[a]).unwrap_err(), OperationError::IncompatibleWeights);
    // A target outside the vtree.
    let far = crate::vtree::VtreeIdx(tree.num_nodes() as u32);
    assert_eq!(eng.skip_marginalize(&m, &[far]).unwrap_err(), OperationError::LevelNotInVtree(far));
    // Summing out what is summed out changes nothing.
    let again = eng.skip_marginalize(&m, &[a]).unwrap();
    assert_eq!(fingerprint(&again.form), fingerprint(&m.form));
}

/// A summed-out diagram embedded with weights weighs what the stored
/// `embed_over` of the stored diagram does, alone and conjoined with a
/// function of the variables outside the summed-out images.
#[test]
fn a_summed_embedding_is_the_stored_embed_over() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use crate::vtree::VarId;
    let mut rng = Lcg::new(25);
    let mut done = 0;
    while done < 200 {
        let n = 3 + rng.below(9) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.len() < 2 {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { continue };
            let small = eng.bind_vtree(small);
            let m = keep.len() as u32;
            let mut draw = || crate::test_helpers::rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64);
            let lits: Vec<LiteralWeights<_>> = (0..=n).map(|_| LiteralWeights { negative: draw(), positive: draw() }).collect();
            let big = WeightStore::new(RationalWeights::from_literals(&lits), Arithmetic::ExactRational);
            let kept: Vec<_> = keep.iter().map(|&k| lits[k as usize - 1].clone()).collect();
            let ws = WeightStore::new(RationalWeights::from_literals(&kept), Arithmetic::ExactRational);
            let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let inner: Vec<_> = small.bottomup().filter(|&t| !small.node(t).is_leaf()).filter(|_| rng.below(2) == 0).collect();
            let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
            let mut stored = f.clone();
            stored.set_weights(ws.empty_like()).unwrap();
            eng.marginalize_levels(&mut stored, &inner).unwrap();
            let (want, _) = eng.embed_over(&stored, &vtree, map, Some(big.empty_like())).unwrap();
            let s = eng.skip_marginalize(&weighed(&eng, &f, &ws), &inner).unwrap();
            let got = eng.skip_embed_over(&s, &vtree, map, Some(big.empty_like())).unwrap();
            let value = |d: &Tdd| eng.weighted_value(d).unwrap().unwrap();
            assert!(agree(&eng.skip_weighted_value(&got).unwrap().unwrap(), &value(&want), 0.0), "{shape}");
            // Conjoined with a CNF over the variables outside the images.
            let sh = &got.form.shape;
            let inside: std::collections::HashSet<u32> = (0..sh.num_nodes()).filter(|&v| sh.owner[v] != super::NONE && sh.var[v] != 0).map(|v| sh.var[v]).collect();
            let cnf: Vec<Vec<i32>> = rand_cnf(&mut rng, n, CnfShape { clauses: 4, width: 3 })
                .into_iter()
                .filter(|c| c.iter().all(|l| !inside.contains(&l.unsigned_abs())))
                .collect();
            let mut g = compile_clauses_on(&eng, &vtree, &cnf);
            let both = eng.skip_and(&got, &eng.to_skip(&g).unwrap()).unwrap();
            g.set_weights(big.empty_like()).unwrap();
            let fg = eng.and_marginalizing(&want, &g, &[]).unwrap();
            assert!(agree(&eng.skip_weighted_value(&both).unwrap().unwrap(), &value(&fg), 0.0), "{shape}, conjoined");
            if s.has_marginal_level() {
                assert!(matches!(eng.skip_embed_over(&s, &vtree, map, None), Err(crate::restructure::EmbedError::SourceWeights(_))));
            }
            done += 1;
        }
    }
}

/// A summed-out function that weighs as much as `true` is not read as
/// `true`: under signed weights `¬x ∧ y` weighs what `true` does over `x`
/// and `y`, and its conjunction with `x ∧ y`, summed out alike, is refused
/// or weighs nothing.
#[test]
fn a_function_weighing_as_much_as_true_is_not_true() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use crate::test_helpers::rat;
    use crate::vtree::VtreeNode;
    use crate::OperationError;
    let lits = [(1, 1), (-1, 2), (1, 1)].map(|(n, p)| LiteralWeights { negative: rat(n, 1), positive: rat(p, 1) });
    let ws = WeightStore::new(RationalWeights::from_literals(&lits), Arithmetic::ExactRational);
    let mut found = 0;
    for (shape, vtree) in vtree_shapes(3) {
        let eng = Engine::new();
        let vars = |t| {
            let mut out = Vec::new();
            let mut stack = vec![t];
            while let Some(u) = stack.pop() {
                match *vtree.node(u) {
                    VtreeNode::Leaf { var, .. } => out.push(var.0),
                    VtreeNode::Internal { left, right, .. } => stack.extend([left, right]),
                }
            }
            out.sort_unstable();
            out
        };
        let Some(m) = vtree.bottomup().find(|&t| vars(t) == [1, 2]) else { continue };
        found += 1;
        let f = compile_clauses_on(&eng, &vtree, &[vec![-1], vec![2]]);
        let g = compile_clauses_on(&eng, &vtree, &[vec![1], vec![2]]);
        assert_eq!(*wmc(&eng, &f, &ws).as_rational(), rat(4, 1), "{shape}: ¬x ∧ y weighs what true does");
        let fs = eng.skip_marginalize(&weighed(&eng, &f, &ws), &[m]).unwrap();
        let gs = eng.skip_marginalize(&weighed(&eng, &g, &ws), &[m]).unwrap();
        match eng.skip_and(&fs, &gs) {
            Err(OperationError::MarginalLevel(_)) => {}
            Ok(h) => assert_eq!(*eng.skip_weighted_value(&h).unwrap().unwrap().as_rational(), rat(0, 1), "{shape}"),
            Err(e) => panic!("{shape}: {e:?}"),
        }
    }
    assert!(found > 0);
}

/// A stored diagram with weight-marginal levels converts to skip form with
/// its value, and conjoins there as the stored form does with a function
/// of the variables outside its marginal subtrees.
#[test]
fn a_weight_marginal_stored_diagram_converts_with_its_value() {
    use crate::diagram::Arithmetic;
    let mut rng = Lcg::new(31);
    for round in 0..120 {
        let n = 3 + rng.below(8) as u32;
        let arithmetic = [Arithmetic::ExactRational, Arithmetic::ScaledFloat][round % 2];
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let signed = rng.below(2) == 0;
            let ws = draw_store(&mut rng, n, signed, arithmetic);
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            f.set_weights(ws.empty_like()).unwrap();
            let targets: Vec<_> = vtree.bottomup().filter(|&t| rng.below(if vtree.node(t).is_leaf() { 4 } else { 3 }) == 0).collect();
            eng.marginalize_levels(&mut f, &targets).unwrap();
            let s = eng.to_skip(&f).unwrap();
            let want = eng.weighted_value(&f).unwrap().unwrap();
            let tol = if arithmetic == Arithmetic::ExactRational { 0.0 } else { 1e-9 };
            assert!(agree(&eng.skip_weighted_value(&s).unwrap().unwrap(), &want, tol), "{shape}");
            let sh = &s.form.shape;
            let inside: std::collections::HashSet<u32> = (0..sh.num_nodes()).filter(|&v| sh.owner[v] != super::NONE && sh.var[v] != 0).map(|v| sh.var[v]).collect();
            let cnf: Vec<Vec<i32>> = rand_cnf(&mut rng, n, CnfShape { clauses: 3, width: 2 })
                .into_iter()
                .filter(|c| c.iter().all(|l| !inside.contains(&l.unsigned_abs())))
                .collect();
            let mut g = compile_clauses_on(&eng, &vtree, &cnf);
            let both = eng.skip_and(&s, &eng.to_skip(&g).unwrap()).unwrap();
            g.set_weights(ws.empty_like()).unwrap();
            let fg = eng.and_marginalizing(&f, &g, &[]).unwrap();
            let want = eng.weighted_value(&fg).unwrap().unwrap();
            assert!(agree(&eng.skip_weighted_value(&both).unwrap().unwrap(), &want, tol), "{shape}, conjoined");
        }
    }
}

/// A pair's sides as (level, index), for printing.
type Side2 = ((u32, u32), (u32, u32));

/// A skip-form result against the stored one: the canonical skip form of
/// its function, with its model count, and the same function stored.
/// Every node holds its pairs in their stored form, which [`super::normalize`]
/// keeps as they are: the fingerprint, which sorts them, does not see their
/// order.
fn assert_stored_order(s: &SkipForm, what: &str) {
    for &t in &s.shape.bottomup {
        let lvl = &s.levels[t as usize];
        for i in 0..lvl.width() {
            let node = lvl.node(i as u32);
            let mut x = node.to_vec();
            assert!(super::normalize(&s.shape, t, &mut x).is_ok() && x == node, "{what}: level {t} holds a node out of its stored form");
        }
    }
}

fn assert_canonical(eng: &Engine, s: &crate::SkipTdd, t: &Tdd, what: &str) {
    let want = eng.to_skip(t).unwrap();
    let held = built(eng, s);
    assert_stored_order(&held, what);
    if fingerprint(&held) != fingerprint(&want.form) {
        eprintln!("kids {:?}", held.shape.kids);
        eprintln!("mine {:?} out {:?}", dump(&held), (vt(held.out), ix(held.out)));
        eprintln!("want {:?} out {:?}", dump(&want.form), (vt(want.form.out), ix(want.form.out)));
        panic!("{what}: not the canonical skip form");
    }
    assert_eq!(eng.skip_model_count(s).unwrap(), eng.model_count(t).unwrap(), "{what}");
    assert!(eng.equivalent(&eng.to_stored(s).unwrap(), t).unwrap(), "{what}");
}

/// The overlay before its contraction is a skip form of the function: its
/// stored form, minimized, is the stored result.
fn assert_raw(eng: &Engine, op: super::overlay::Combine, f: &crate::SkipTdd, g: &crate::SkipTdd, t: &Tdd) {
    if f.is_zero() || g.is_zero() || f.vtree().node(f.vtree().root()).is_leaf() {
        return;
    }
    let raw = super::overlay::overlay_raw(op, &f.form, &g.form, eng.limits()).unwrap();
    assert_eq!(raw.model_count(eng.limits()).unwrap(), eng.model_count(t).unwrap(), "{op:?}: raw model count");
    let mut back = to_tdd(&raw, eng, f.vtree()).unwrap();
    eng.minimize(&mut back).unwrap();
    assert!(eng.equivalent(&back, t).unwrap(), "{op:?}: raw function");
}

/// `skip_or`, `skip_and_not` and `skip_and` (both ways) of random functions
/// against the stored `or`, `and_not` and `and`, on every vtree shape and on
/// wider balanced and random vtrees; and the overlay's raw result before its
/// contraction.
#[test]
fn overlay_is_the_canonical_skip_form() {
    use super::overlay::Combine;
    for (eng, f, g) in cases(21, 300).into_iter().chain(wide_cases(21, 30)) {
        let (sf, sg) = (eng.to_skip(&f).unwrap(), eng.to_skip(&g).unwrap());
        let or = eng.or(f.clone(), g.clone()).unwrap();
        assert_canonical(&eng, &eng.skip_or(&sf, &sg).unwrap(), &or, "or");
        assert_canonical(&eng, &eng.skip_or(&sg, &sf).unwrap(), &or, "or, swapped");
        assert_raw(&eng, Combine::Or, &sf, &sg, &or);
        let fg = eng.and_not(f.clone(), g.clone()).unwrap();
        assert_canonical(&eng, &eng.skip_and_not(&sf, &sg).unwrap(), &fg, "and_not");
        assert_raw(&eng, Combine::AndNot, &sf, &sg, &fg);
        let gf = eng.and_not(g.clone(), f.clone()).unwrap();
        assert_canonical(&eng, &eng.skip_and_not(&sg, &sf).unwrap(), &gf, "and_not, swapped");
        assert_raw(&eng, Combine::AndNot, &sg, &sf, &gf);
        let and = eng.and(&f, &g).unwrap();
        assert_canonical(&eng, &eng.skip_and(&sf, &sg).unwrap(), &and, "and");
        assert_canonical(&eng, &eng.skip_and(&sg, &sf).unwrap(), &and, "and, swapped");
    }
}

/// Pairs of random CNFs of 12 to 24 variables, each on a balanced, a random
/// and a right-linear vtree: wide enough for skips through many levels.
fn wide_cases(seed: u64, rounds: usize) -> Vec<(Engine, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut wide = Vec::new();
    for _ in 0..rounds {
        let n = 12 + rng.below(13) as u32;
        let shape = CnfShape { clauses: n as usize / 2 + rng.below(u64::from(n)) as usize, width: 3 };
        for vtree in [Arc::new(crate::vtree::Vtree::balanced(n)), Arc::new(crate::vtree::Vtree::random(n, rng.below(1000))), Arc::new(crate::vtree::Vtree::linear(n))] {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            wide.push((eng, f, g));
        }
    }
    wide
}

thread_local! {
    /// Whether the overlay's arena of pieces and its classes' lists of skips
    /// move their live lists to a new buffer at every growth, and how many
    /// times each has.
    static COMPACTIONS: std::cell::Cell<(bool, u64, u64)> = const { std::cell::Cell::new((false, 0, 0)) };
}

/// Whether the overlay moves its live lists at every growth.
pub(super) fn compactions_forced() -> bool {
    COMPACTIONS.with(|c| c.get().0)
}

/// Count a move of the overlay's live lists: its arena of pieces (`0`) or
/// its classes' lists of skips (`1`).
pub(super) fn note_compaction(lists: usize) {
    COMPACTIONS.with(|c| {
        let (forced, pieces, skips) = c.get();
        c.set(if lists == 0 { (forced, pieces + 1, skips) } else { (forced, pieces, skips + 1) });
    });
}

/// The overlay's and the conjunction's results where the pieces' arena and
/// the classes' lists of skips move their live lists to a new buffer at
/// every growth.
#[test]
fn overlays_that_compact_their_pieces_are_canonical() {
    COMPACTIONS.with(|c| c.set((true, 0, 0)));
    for (eng, f, g) in wide_cases(24, 20) {
        let (sf, sg) = (eng.to_skip(&f).unwrap(), eng.to_skip(&g).unwrap());
        assert_canonical(&eng, &eng.skip_or(&sf, &sg).unwrap(), &eng.or(f.clone(), g.clone()).unwrap(), "or");
        assert_canonical(&eng, &eng.skip_and_not(&sf, &sg).unwrap(), &eng.and_not(f.clone(), g.clone()).unwrap(), "and_not");
        assert_canonical(&eng, &eng.skip_and(&sf, &sg).unwrap(), &eng.and(&f, &g).unwrap(), "and");
    }
    let (_, pieces, lists) = COMPACTIONS.with(|c| c.replace((false, 0, 0)));
    assert!(pieces > 0 && lists > 0, "compacted {pieces} arenas and {lists} lists");
}

/// The overlay's edge cases on every vtree shape, one variable included:
/// constants, a literal operand, operands over disjoint variables, equal
/// operands and an operand with its negation.
#[test]
fn overlay_edge_cases_match_the_stored_form() {
    let mut rng = Lcg::new(22);
    for n in 1..=9u32 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            // Functions of the lower and the upper half of the variables.
            let half = |rng: &mut Lcg, lo: i32, hi: i32| -> Tdd {
                let cnf: Vec<Vec<i32>> = (0..2)
                    .map(|_| (0..2).map(|_| {
                        let v = lo + rng.below((hi - lo + 1) as u64) as i32;
                        if rng.below(2) == 0 { v } else { -v }
                    }).collect())
                    .collect();
                compile_clauses_on(&eng, &vtree, &cnf)
            };
            let m = (n as i32 / 2).max(1);
            let lower = half(&mut rng, 1, m);
            let upper = half(&mut rng, m.min(n as i32), n as i32);
            let not_f = eng.negate(&f).unwrap();
            let mut ops: Vec<Tdd> = vec![Tdd::zero(&vtree), Tdd::one(&vtree), f.clone(), not_f, lower, upper];
            for v in [1i32, -1, n as i32, -(n as i32), (n as i32 + 1) / 2] {
                ops.push(eng.literal(&vtree, v).unwrap());
            }
            let skips: Vec<crate::SkipTdd> = ops.iter().map(|t| eng.to_skip(t).unwrap()).collect();
            for (i, a) in ops.iter().enumerate() {
                for (j, b) in ops.iter().enumerate() {
                    let what = format!("{name} n {n}, operands {i} and {j}");
                    let (sa, sb) = (&skips[i], &skips[j]);
                    assert_canonical(&eng, &eng.skip_or(sa, sb).unwrap(), &eng.or(a.clone(), b.clone()).unwrap(), &format!("or, {what}"));
                    assert_canonical(&eng, &eng.skip_and_not(sa, sb).unwrap(), &eng.and_not(a.clone(), b.clone()).unwrap(), &format!("and_not, {what}"));
                }
            }
        }
    }
}

/// The conjunction's walk on the overlay's edge cases, both ways, on every
/// vtree shape: constants, literals (the smaller operand's output a leaf
/// label), operands over disjoint variables (the larger one's output beyond
/// the other's cone), equal operands (every level the larger operand's,
/// shared) and an operand with its negation (every slot dies). The raw
/// result is a skip form of the function that shares the larger operand's
/// levels beyond the cone it returns.
#[test]
fn conjunction_edge_cases_match_the_stored_form() {
    let mut rng = Lcg::new(24);
    let (mut shared, mut narrowed) = (0, 0);
    for n in 1..=9u32 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let lim = eng.limits();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let half = |rng: &mut Lcg, lo: i32, hi: i32| -> Tdd {
                let cnf: Vec<Vec<i32>> = (0..2)
                    .map(|_| (0..2).map(|_| {
                        let v = lo + rng.below((hi - lo + 1) as u64) as i32;
                        if rng.below(2) == 0 { v } else { -v }
                    }).collect())
                    .collect();
                compile_clauses_on(&eng, &vtree, &cnf)
            };
            let m = (n as i32 / 2).max(1);
            let lower = half(&mut rng, 1, m);
            let upper = half(&mut rng, m.min(n as i32), n as i32);
            let not_f = eng.negate(&f).unwrap();
            let mut ops: Vec<Tdd> = vec![Tdd::zero(&vtree), Tdd::one(&vtree), f.clone(), not_f, lower, upper];
            for v in [1i32, -1, n as i32, -(n as i32), (n as i32 + 1) / 2] {
                ops.push(eng.literal(&vtree, v).unwrap());
            }
            let skips: Vec<crate::SkipTdd> = ops.iter().map(|t| eng.to_skip(t).unwrap()).collect();
            for (i, a) in ops.iter().enumerate() {
                for (j, b) in ops.iter().enumerate() {
                    let what = format!("and, {name} n {n}, operands {i} and {j}");
                    let (sa, sb) = (&skips[i], &skips[j]);
                    let mut want = eng.and(a, b).unwrap();
                    eng.minimize(&mut want).unwrap();
                    assert_canonical(&eng, &eng.skip_and(sa, sb).unwrap(), &want, &what);
                    if sa.is_zero() || sb.is_zero() {
                        continue;
                    }
                    let (raw, cone) = conjoin::and_scoped(&sa.form, &sb.form, lim).unwrap();
                    assert_eq!(raw.model_count(lim).unwrap(), eng.model_count(&want).unwrap(), "{what}: raw model count");
                    let mut back = to_tdd(&raw, &eng, &vtree).unwrap();
                    eng.minimize(&mut back).unwrap();
                    assert!(eng.equivalent(&back, &want).unwrap(), "{what}: raw function");
                    let p = if sb.form.active.len() > sa.form.active.len() { &sb.form } else { &sa.form };
                    for &t in &raw.active {
                        let same = std::ptr::eq(&raw.levels[t as usize], &p.levels[t as usize]);
                        assert!(same || (cone[t as usize] && i != j), "{what}: level {t} is not shared");
                        shared += usize::from(same && cone[t as usize]);
                        narrowed += usize::from(raw.levels[t as usize].width() < p.levels[t as usize].width());
                    }
                    if i == j && !sa.form.active.is_empty() {
                        assert_eq!(raw.out, p.out, "{what}");
                        assert_eq!(raw.active.len(), p.active.len(), "{what}");
                    }
                    lim.discard(cone);
                }
            }
        }
    }
    assert!(shared > 0 && narrowed > 0, "shared {shared} narrowed {narrowed}");
}

/// Circuits compiled gate by gate, as a bit-blasted program is: each gate
/// the conjunction, disjunction or difference of two earlier gates or
/// literals, every gate the canonical skip form of the stored one.
#[test]
fn gate_by_gate_overlays_are_canonical() {
    let mut rng = Lcg::new(23);
    for round in 0..12 {
        let n = 4 + rng.below(13) as u32;
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut stored: Vec<Tdd> = Vec::new();
            let mut skips: Vec<crate::SkipTdd> = Vec::new();
            for v in 1..=n as i32 {
                for lit in [v, -v] {
                    let t = eng.literal(&vtree, lit).unwrap();
                    skips.push(eng.to_skip(&t).unwrap());
                    stored.push(t);
                }
            }
            for gate in 0..40 {
                // Recent gates more often, so that the circuit deepens.
                let pick = |rng: &mut Lcg, len: usize| if rng.below(2) == 0 { len - 1 - rng.below(len.min(6) as u64) as usize } else { rng.below(len as u64) as usize };
                let (i, j) = (pick(&mut rng, stored.len()), pick(&mut rng, stored.len()));
                let (a, b) = (&stored[i], &stored[j]);
                let (op, mut t, s) = match rng.below(3) {
                    0 => ("and", eng.and(a, b).unwrap(), eng.skip_and(&skips[i], &skips[j]).unwrap()),
                    1 => ("or", eng.or(a.clone(), b.clone()).unwrap(), eng.skip_or(&skips[i], &skips[j]).unwrap()),
                    _ => ("and_not", eng.and_not(a.clone(), b.clone()).unwrap(), eng.skip_and_not(&skips[i], &skips[j]).unwrap()),
                };
                eng.minimize(&mut t).unwrap();
                assert_canonical(&eng, &s, &t, &format!("round {round}, {name} n {n}, gate {gate} ({op} of {i} and {j})"));
                stored.push(t);
                skips.push(s);
            }
        }
    }
}

/// Random cubes over variables 1 to `n`: some repeat a literal, some name a
/// variable in both signs, and rarely one is empty.
fn rand_cubes(rng: &mut Lcg, n: u32) -> Vec<Vec<i32>> {
    let count = rng.below(2 * n as u64 + 2) as usize;
    (0..count)
        .map(|_| {
            if rng.below(60) == 0 {
                return Vec::new();
            }
            let w = 1 + rng.below(4) as usize;
            (0..w)
                .map(|_| {
                    let v = 1 + rng.below(n as u64) as i32;
                    if rng.below(2) == 0 { v } else { -v }
                })
                .collect()
        })
        .collect()
}

/// `¬c₁ ∧ … ∧ ¬cₙ` in stored form, a negated cube conjoined at a time.
fn none_of_stored(eng: &Engine, vtree: &Arc<crate::Vtree>, cubes: &[Vec<i32>]) -> Tdd {
    let mut f = Tdd::one(vtree);
    for cube in cubes {
        if cube.is_empty() {
            return Tdd::zero(vtree);
        }
        let clause: Vec<i32> = cube.iter().map(|&l| -l).collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

#[test]
fn none_of_is_the_canonical_conjunction_of_the_negated_cubes() {
    use crate::diagram::Arithmetic;
    let mut rng = Lcg::new(41);
    for round in 0..150 {
        let n = 1 + rng.below(10) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let cubes = rand_cubes(&mut rng, n);
            let want = none_of_stored(&eng, &vtree, &cubes);
            let got = eng.skip_none_of(&vtree, &cubes).unwrap();
            assert_eq!(fingerprint(&got.form), fingerprint(&eng.to_skip(&want).unwrap().form), "{shape}: {cubes:?}");
            let back = eng.to_stored(&got).unwrap();
            assert!(eng.equivalent(&back, &want).unwrap(), "{shape}: {cubes:?}");
            assert_eq!(eng.skip_model_count(&got).unwrap(), eng.model_count(&want).unwrap());
            // Signed weights value it as the stored form.
            let arithmetic = [Arithmetic::ExactRational, Arithmetic::ScaledFloat][round % 2];
            let ws = draw_store(&mut rng, n, true, arithmetic);
            let mut s = got;
            s.set_weights(ws.empty_like()).unwrap();
            let tol = if arithmetic == Arithmetic::ExactRational { 0.0 } else { 1e-9 };
            assert!(agree(&eng.skip_weighted_value(&s).unwrap().unwrap(), &wmc(&eng, &want, &ws), tol), "{shape}");
        }
    }
}

/// A table's rows over `k` binary parents, each with a switch of its own,
/// on a spine of switches above the parents (on either side), the parents
/// balanced or linear below: the skip form holds pairs in proportion to the
/// rows, where the stored form holds a node per row above each level.
#[test]
fn none_of_a_table_on_a_spine_of_switches_is_linear_in_its_rows() {
    use crate::vtree::VarId;
    for k in [3u32, 6] {
        let r = 1u32 << k;
        let rows: Vec<Vec<i32>> = (0..r)
            .map(|a| {
                let mut cube: Vec<i32> = (0..k).map(|j| if a >> j & 1 == 1 { (r + 1 + j) as i32 } else { -((r + 1 + j) as i32) }).collect();
                cube.push((a + 1) as i32);
                cube
            })
            .collect();
        for (mirror, balanced) in [(false, true), (false, false), (true, true), (true, false)] {
            let mut vars: Vec<VarId> = (1..=r).map(VarId).collect();
            let mut depths: Vec<u32> = (1..=r).collect();
            vars.extend((r + 1..=r + k).map(VarId));
            // The parents' subtree: halves by depth, or linear.
            let lca = |i: u32| if balanced { r + 1 + (i ^ (i + 1)).leading_zeros() } else { r + 1 + i };
            depths.extend((0..k - 1).map(lca));
            if mirror {
                vars.reverse();
                depths.reverse();
            }
            let vtree = Arc::new(crate::Vtree::from_in_order(&vars, &depths, r + k).unwrap());
            let eng = Engine::new();
            let want = none_of_stored(&eng, &vtree, &rows);
            let got = eng.skip_none_of(&vtree, &rows).unwrap();
            assert_eq!(fingerprint(&got.form), fingerprint(&eng.to_skip(&want).unwrap().form));
            let (skip, stored) = (got.pair_count(), want.pair_count());
            assert!(skip <= 6 * r as usize, "k {k}, mirror {mirror}, balanced {balanced}: {skip} pairs against {stored} stored");
            if k == 6 {
                assert!(stored >= (r * r / 2) as usize, "{stored} stored pairs");
            }
        }
    }
}

/// Weights read down chains longer than the valuer's blocks of products
/// (`super::eval`): a table's skip form on a spine of 256 switches weighs
/// what its stored form does, alone and conjoined with a clause where
/// random subtrees the clause does not read are summed out.
#[test]
fn weights_down_long_chains_are_those_of_the_stored_form() {
    use crate::diagram::Arithmetic;
    use crate::vtree::VarId;
    let k = 8u32;
    let r = 1u32 << k;
    let rows: Vec<Vec<i32>> = (0..r)
        .map(|a| {
            let mut cube: Vec<i32> = (0..k).map(|j| if a >> j & 1 == 1 { (r + 1 + j) as i32 } else { -((r + 1 + j) as i32) }).collect();
            cube.push((a + 1) as i32);
            cube
        })
        .collect();
    let mut rng = Lcg::new(47);
    for mirror in [false, true] {
        let mut vars: Vec<VarId> = (1..=r).map(VarId).collect();
        let mut depths: Vec<u32> = (1..=r).collect();
        vars.extend((r + 1..=r + k).map(VarId));
        depths.extend((0..k - 1).map(|i| r + 1 + i));
        if mirror {
            vars.reverse();
            depths.reverse();
        }
        let tree = Arc::new(crate::Vtree::from_in_order(&vars, &depths, r + k).unwrap());
        let eng = Engine::new();
        let want = none_of_stored(&eng, &tree, &rows);
        let mut got = eng.skip_none_of(&tree, &rows).unwrap();
        // Bounded values: exact rationals over hundreds of variables are slow.
        let arithmetic = if mirror { Arithmetic::SignedLog } else { Arithmetic::ScaledFloat };
        let ws = draw_store(&mut rng, r + k, mirror, arithmetic);
        got.set_weights(ws.empty_like()).unwrap();
        assert!(agree(&eng.skip_weighted_value(&got).unwrap().unwrap(), &wmc(&eng, &want, &ws), 1e-9), "mirror {mirror}");
        for round in 0..4 {
            let clause = vec![1 + rng.below(r as u64) as i32, -(1 + rng.below(r as u64) as i32), (r as u64 + 1 + rng.below(k as u64)) as i32];
            let live: std::collections::HashSet<u32> = clause.iter().map(|l| l.unsigned_abs()).collect();
            let targets: Vec<_> = untouched(&tree, &live, true).into_iter().filter(|_| rng.below(2) == 0).collect();
            let h = compile_clauses_on(&eng, &tree, &[clause]);
            let m = eng.skip_and_marginalizing(&got, &weighed(&eng, &h, &ws), &targets).unwrap();
            let both = wmc(&eng, &eng.and(&want, &h).unwrap(), &ws);
            assert!(agree(&eng.skip_weighted_value(&m).unwrap().unwrap(), &both, 1e-9), "mirror {mirror}, round {round}");
        }
    }
}

/// The stored form's nodes, counted on the skip form, are those
/// `to_stored` builds.
#[test]
fn the_stored_node_count_is_that_of_the_stored_form() {
    for (eng, f, g) in cases(44, 120) {
        for h in [&f, &g] {
            let s = eng.to_skip(h).unwrap();
            let back = eng.to_stored(&s).unwrap();
            assert_eq!(eng.skip_stored_node_count(&s).unwrap(), back.node_count() as u64);
        }
    }
    let mut rng = Lcg::new(45);
    for _ in 0..40 {
        let n = 1 + rng.below(12) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let got = eng.skip_none_of(&vtree, &rand_cubes(&mut rng, n)).unwrap();
            let back = eng.to_stored(&got).unwrap();
            assert_eq!(eng.skip_stored_node_count(&got).unwrap(), back.node_count() as u64, "{shape}");
        }
    }
}

/// Wider random cubes on wider vtrees than the test above, where classes
/// carried up a spine meet cubes of several leaves.
#[test]
fn none_of_on_wider_vtrees_is_the_canonical_conjunction() {
    let mut rng = Lcg::new(43);
    for _ in 0..12 {
        let n = 10 + rng.below(6) as u32;
        for (shape, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let cubes: Vec<Vec<i32>> = (0..2 * n)
                .map(|_| {
                    (0..2 + rng.below(5))
                        .map(|_| {
                            let v = 1 + rng.below(n as u64) as i32;
                            if rng.below(2) == 0 { v } else { -v }
                        })
                        .collect()
                })
                .collect();
            let want = none_of_stored(&eng, &vtree, &cubes);
            let got = eng.skip_none_of(&vtree, &cubes).unwrap();
            assert_eq!(fingerprint(&got.form), fingerprint(&eng.to_skip(&want).unwrap().form), "{shape}: {cubes:?}");
        }
    }
}

/// Integer weights per variable, exact: the positive literal `2v + 1`, the
/// negative one `5 + v` or, on every third variable, `-(v + 1)`, so that
/// `true` on a variable is neither 1 nor 2 and some weights are negative.
struct Exact;

impl crate::diagram::EvalAlgebra for Exact {
    type Value = i128;
    fn zero(&self) -> i128 {
        0
    }
    fn leaf(&self, var: crate::vtree::VarId, label: crate::diagram::LeafLabel) -> i128 {
        let v = var.0 as i128;
        let (p, q) = (2 * v + 1, if v % 3 == 0 { -(v + 1) } else { 5 + v });
        match label {
            crate::diagram::LeafLabel::Pos => p,
            crate::diagram::LeafLabel::Neg => q,
            _ => p + q,
        }
    }
    fn add_assign(&self, acc: &mut i128, other: &i128) {
        *acc += other;
    }
    fn mul(&self, a: &i128, b: &i128) -> i128 {
        a * b
    }
}

/// Every literal's joint count read off the passes against the count of
/// the conjunction, and a second diagram's.
fn assert_joint_counts<S>(eng: &Engine, e: &crate::SkipTdd, others: &[crate::SkipTdd], algebra: &S, what: &str)
where
    S: crate::diagram::EvalAlgebra,
    S::Value: PartialEq + std::fmt::Debug,
{
    use crate::Literal;
    let mut counted = eng.skip_count(e, algebra).unwrap();
    assert_eq!(*counted.total(), eng.skip_evaluate(e, algebra).unwrap(), "{what}: total");
    let n = e.vtree().num_leaves();
    for v in 1..=n {
        for sign in [true, false] {
            let lit = Literal::new(crate::vtree::VarId(v), sign);
            let got = eng.skip_count_literal(&mut counted, algebra, lit).unwrap();
            let want = eng.skip_evaluate(&eng.skip_and_clause(e, &[lit]).unwrap(), algebra).unwrap();
            assert_eq!(got, want, "{what}: literal {v} {sign}");
        }
    }
    for (k, g) in others.iter().enumerate() {
        let got = eng.skip_count_and(&mut counted, algebra, g).unwrap();
        let want = eng.skip_evaluate(&eng.skip_and(e, g).unwrap(), algebra).unwrap();
        assert_eq!(got, want, "{what}: diagram {k}");
    }
}

/// A few small formulas over `vtree`, as a query's formula is: clauses,
/// cubes and small CNFs over a handful of variables, and the constants.
fn small_formulas(eng: &Engine, rng: &mut Lcg, vtree: &Arc<crate::Vtree>) -> Vec<crate::SkipTdd> {
    let n = vtree.num_leaves();
    let lit = |rng: &mut Lcg| {
        let v = 1 + rng.below(n as u64) as i32;
        if rng.below(2) == 0 { v } else { -v }
    };
    let mut out = vec![eng.to_skip(&Tdd::one(vtree)).unwrap(), eng.to_skip(&Tdd::zero(vtree)).unwrap()];
    for _ in 0..3 {
        let clause: Vec<i32> = (0..1 + rng.below(3)).map(|_| lit(rng)).collect();
        out.push(eng.to_skip(&compile_clauses_on(eng, vtree, std::slice::from_ref(&clause))).unwrap());
        let cube: Vec<Vec<i32>> = clause.iter().map(|&l| vec![l]).collect();
        out.push(eng.to_skip(&compile_clauses_on(eng, vtree, &cube)).unwrap());
        let clauses = 2 + rng.below(3) as usize;
        let cnf = rand_cnf(rng, n, CnfShape { clauses, width: 2 });
        out.push(eng.to_skip(&compile_clauses_on(eng, vtree, &cnf)).unwrap());
    }
    out
}

#[test]
fn joint_counts_off_the_passes_are_the_counts_of_the_conjunctions() {
    let mut rng = Lcg::new(53);
    for (eng, f, g) in cases(17, 200) {
        let vtree = Arc::clone(f.vtree());
        let mut others = small_formulas(&eng, &mut rng, &vtree);
        others.push(eng.to_skip(&g).unwrap());
        others.push(eng.to_skip(&f).unwrap());
        others.push(eng.to_skip(&eng.negate(&f).unwrap()).unwrap());
        for (k, h) in [f.clone(), eng.and(&f, &g).unwrap(), eng.negate(&f).unwrap(), Tdd::one(&vtree), Tdd::zero(&vtree)]
            .into_iter()
            .enumerate()
        {
            let e = eng.to_skip(&h).unwrap();
            assert_joint_counts(&eng, &e, &others, &Exact, &format!("diagram {k}"));
            assert_joint_counts(&eng, &e, &others, &super::eval::Count, &format!("diagram {k}, models"));
        }
    }
}

/// Long chains: skips across hundreds of levels, from the root and from
/// below it, past the products read in blocks.
#[test]
fn joint_counts_down_long_chains_are_the_counts_of_the_conjunctions() {
    let mut rng = Lcg::new(59);
    for vtree in [Arc::new(crate::Vtree::linear(300)), Arc::new(crate::Vtree::reverse_linear(300))] {
        let eng = Engine::new();
        let mut e = eng.to_skip(&Tdd::one(&vtree)).unwrap();
        for clause in [vec![-7i32, 150, -290], vec![7i32, 151], vec![-150i32, 299, 3]] {
            e = eng.skip_and_clause(&e, &clause).unwrap();
        }
        let mut others = small_formulas(&eng, &mut rng, &vtree);
        others.push(eng.to_skip(&compile_clauses_on(&eng, &vtree, &[vec![2, -151], vec![150, 280]])).unwrap());
        assert_joint_counts(&eng, &e, &others, &super::eval::Count, "chain");
    }
}

#[test]
fn joint_count_edge_cases() {
    use crate::{Literal, OperationError};
    use crate::vtree::VarId;
    let eng = Engine::new();
    // One variable: the vtree's root is its leaf.
    let single = Arc::new(crate::Vtree::linear(1));
    let lits = [Tdd::one(&single), Tdd::zero(&single), compile_clauses_on(&eng, &single, &[vec![1]]), compile_clauses_on(&eng, &single, &[vec![-1]])];
    let all: Vec<crate::SkipTdd> = lits.iter().map(|f| eng.to_skip(f).unwrap()).collect();
    for e in &all {
        assert_joint_counts(&eng, e, &all, &Exact, "one variable");
    }
    let vtree = Arc::new(crate::Vtree::balanced(4));
    let e = eng.to_skip(&compile_clauses_on(&eng, &vtree, &[vec![1, -3]])).unwrap();
    let mut counted = eng.skip_count(&e, &Exact).unwrap();
    assert_eq!(
        eng.skip_count_literal(&mut counted, &Exact, Literal::pos(VarId(5))).unwrap_err(),
        OperationError::VariableNotInVtree(VarId(5))
    );
    let other = eng.to_skip(&Tdd::one(&Arc::new(crate::Vtree::balanced(4)))).unwrap();
    assert_eq!(eng.skip_count_and(&mut counted, &Exact, &other).unwrap_err(), OperationError::VtreeMismatch);
}

#[test]
fn merging_two_sorted_runs_sorts_them() {
    let eng = Engine::new();
    let lim = eng.limits();
    let mut rng = Lcg::new(17);
    let mut buf = Vec::new();
    for _ in 0..500 {
        let n = rng.below(40) as usize;
        let mid = if n == 0 { 0 } else { rng.below(n as u64 + 1) as usize };
        let mut a: Vec<Ref> = (0..n).map(|_| mk(rng.below(6) as u32, rng.below(50) as u32)).collect();
        a.sort_unstable();
        a.dedup();
        let mid = mid.min(a.len());
        // Two sorted runs of distinct keys, interleaved by a shuffle of the
        // split, as a tail's parts are.
        let mut left: Vec<Ref> = Vec::new();
        let mut right: Vec<Ref> = Vec::new();
        for &x in &a {
            if left.len() < mid && (right.len() >= a.len() - mid || rng.below(2) == 0) {
                left.push(x);
            } else {
                right.push(x);
            }
        }
        let mut got = left.clone();
        got.extend_from_slice(&right);
        let key = |m: Ref| ((vt(m) as u64) << 32) | ix(m) as u64;
        super::contract::merge_sorted(&mut got, left.len(), key, &mut buf, lim).unwrap();
        assert_eq!(got, a, "left {left:?} right {right:?}");
    }
}

/// A joint count refused at any of its reserves, the walk's product tables
/// included, returns `OverBudget` rather than aborting, and the same counted
/// evidence then gives the right joint.
#[test]
fn joint_count_refuses_at_every_reserve() {
    use crate::OperationError;
    for vtree in [Arc::new(crate::Vtree::balanced(16)), Arc::new(crate::Vtree::linear(16))] {
        let eng = Engine::new();
        let e = eng
            .to_skip(&compile_clauses_on(&eng, &vtree, &[vec![1, -3, 5], vec![2, 7, -9], vec![-4, 11, 13], vec![6, -15, 16]]))
            .unwrap();
        let g = eng.to_skip(&compile_clauses_on(&eng, &vtree, &[vec![-1, 8], vec![3, -10, 12], vec![5, 14], vec![-6, 9]])).unwrap();
        let want = eng.skip_count_and(&mut eng.skip_count(&e, &Exact).unwrap(), &Exact, &g).unwrap();
        let mut refused = 0;
        for k in 0..4096 {
            let mut counted = eng.skip_count(&e, &Exact).unwrap();
            eng.limits().refuse_nth_reserve(k);
            let got = eng.skip_count_and(&mut counted, &Exact, &g);
            eng.limits().grant_every_reserve();
            match got {
                Err(OperationError::OverBudget) => {
                    refused += 1;
                    assert_eq!(eng.skip_count_and(&mut counted, &Exact, &g).unwrap(), want, "after refusal {k}");
                }
                Ok(v) => {
                    assert_eq!(v, want);
                    break;
                }
                Err(other) => panic!("refusal {k}: {other:?}"),
            }
        }
        assert!(refused > 0, "the joint count reserved nothing");
    }
}

/// A negation is marked, not built ([`Engine::skip_negate`]): the
/// conjunction, disjunction and difference of operands marked or not read
/// the marks by De Morgan, and every other operation reads the function the
/// diagram marks. Each result is the canonical form of the stored one, on
/// every vtree shape, constants and double negations included.
#[test]
fn marked_negations_are_read_as_the_functions_they_mark() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(61);
    for (eng, f, g) in cases(61, 60) {
        let vtree = Arc::clone(f.vtree());
        let mut both: Vec<(Tdd, crate::SkipTdd)> = Vec::new();
        for t in [f.clone(), g.clone(), Tdd::one(&vtree), Tdd::zero(&vtree)] {
            let s = eng.to_skip(&t).unwrap();
            let marked = eng.skip_negate(&s).unwrap();
            // Marking shares the stored diagram.
            assert_eq!(fingerprint(&marked.form), fingerprint(&s.form));
            both.push((eng.negate(&t).unwrap(), marked.clone()));
            both.push((t, eng.skip_negate(&marked).unwrap()));
            both.push((eng.to_stored(&s).unwrap(), s));
        }
        for (i, (a, sa)) in both.iter().enumerate() {
            let what = format!("operand {i}");
            assert_eq!(sa.is_zero(), eng.model_count(a).unwrap() == 0u32.into(), "{what}");
            assert_canonical(&eng, sa, a, &what);
            for (j, (b, sb)) in both.iter().enumerate() {
                let what = format!("operands {i} and {j}");
                assert_canonical(&eng, &eng.skip_and(sa, sb).unwrap(), &eng.and(a, b).unwrap(), &format!("and, {what}"));
                assert_canonical(&eng, &eng.skip_or(sa, sb).unwrap(), &eng.or(a.clone(), b.clone()).unwrap(), &format!("or, {what}"));
                let diff = eng.and_not(a.clone(), b.clone()).unwrap();
                assert_canonical(&eng, &eng.skip_and_not(sa, sb).unwrap(), &diff, &format!("and_not, {what}"));
                assert_eq!(eng.skip_equivalent(sa, sb).unwrap(), eng.equivalent(a, b).unwrap(), "equivalent, {what}");
            }
            let lits = assignment(&mut rng, &f);
            let clause = eng.and_clause(a.clone(), &lits).unwrap();
            assert_canonical(&eng, &eng.skip_and_clause(sa, &lits).unwrap(), &clause, &format!("and_clause, {what}"));
            let cofactor = eng.condition(a.clone(), lits.iter().copied()).unwrap();
            assert_canonical(&eng, &eng.skip_condition(sa, &lits).unwrap(), &cofactor, &format!("condition, {what}"));
            let vars: Vec<VarId> = lits.iter().map(|l| VarId(l.unsigned_abs())).collect();
            let ex = eng.exists_vars(a.clone(), &vars).unwrap();
            assert_canonical(&eng, &eng.skip_exists(sa, &vars).unwrap(), &ex, &format!("exists, {what}"));
            let all = eng.forall_vars(a.clone(), &vars).unwrap();
            assert_canonical(&eng, &eng.skip_forall(sa, &vars).unwrap(), &all, &format!("forall, {what}"));
            assert_eq!(eng.skip_support(sa).unwrap(), eng.support(a).unwrap(), "support, {what}");
            let (want, got) = (eng.evaluate(a, &Weighted).unwrap(), eng.skip_evaluate(sa, &Weighted).unwrap());
            assert!(close(want, got), "evaluate, {what}: {want} against {got}");
            assert_eq!(*eng.skip_count(sa, &Exact).unwrap().total(), eng.skip_evaluate(&eng.to_skip(a).unwrap(), &Exact).unwrap());
            let (_, sb) = &both[(i + 1) % both.len()];
            let joint = eng.skip_count_and(&mut eng.skip_count(sa, &Exact).unwrap(), &Exact, sb).unwrap();
            assert_eq!(joint, eng.skip_evaluate(&eng.skip_and(sa, sb).unwrap(), &Exact).unwrap(), "joint count, {what}");
            match eng.skip_satisfying_assignment(sa).unwrap() {
                None => assert_eq!(eng.model_count(a).unwrap(), 0u32.into(), "{what}"),
                Some(model) => assert!(!eng.condition(a.clone(), model).unwrap().is_zero(), "model, {what}"),
            }
        }
    }
}

/// A marked negation placed on a larger vtree and restricted back to its
/// own variables is the stored embedding and projection of the negation,
/// on every vtree shape.
#[test]
fn marked_negations_embed_and_project_as_the_stored_ones() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(62);
    let mut done = 0;
    while done < 100 {
        let n = 3 + rng.below(8) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.is_empty() {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { continue };
            let small = eng.bind_vtree(small);
            let m = keep.len() as u32;
            let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
            let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let not = eng.negate(&f).unwrap();
            let marked = eng.skip_negate(&eng.to_skip(&f).unwrap()).unwrap();
            let (want, _) = eng.embed(&not, &vtree, map).unwrap();
            let placed = eng.skip_embed(&marked, &vtree, map).unwrap();
            assert_eq!(fingerprint(&built(&eng, &placed)), fingerprint(&canonical(&eng, want.clone())), "embed");
            assert_eq!(eng.skip_model_count(&placed).unwrap(), eng.model_count(&want).unwrap(), "embed");
            let back = eng.skip_project_to_vars(&placed, local, m).unwrap();
            assert_eq!(fingerprint(&built(&eng, &back)), fingerprint(&canonical(&eng, not.clone())), "project back");
            // A marked diagram that reads dropped variables is quantified.
            let wide = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            let wide_not = eng.negate(&wide).unwrap();
            let marked = eng.skip_negate(&eng.to_skip(&wide).unwrap()).unwrap();
            let want = eng.project_to_vars(&wide_not, local, m).unwrap();
            let got = eng.skip_project_to_vars(&marked, local, m).unwrap();
            assert_eq!(fingerprint(&built(&eng, &got)), fingerprint(&canonical(&eng, want.clone())), "project");
            assert_eq!(eng.skip_model_count(&got).unwrap(), eng.model_count(&want).unwrap(), "project");
            done += 1;
        }
    }
}

/// Reading a marked negation builds it: refused at any of its reserves,
/// the read returns `OverBudget`, and the marked diagram then gives the
/// right answer.
#[test]
fn a_marked_negation_is_built_at_every_reserve_or_refused() {
    use crate::OperationError;
    for vtree in [Arc::new(crate::Vtree::balanced(16)), Arc::new(crate::Vtree::linear(16))] {
        let eng = Engine::new();
        let f = compile_clauses_on(&eng, &vtree, &[vec![1, -3, 5], vec![2, 7, -9], vec![-4, 11, 13], vec![6, -15, 16]]);
        let marked = eng.skip_negate(&eng.to_skip(&f).unwrap()).unwrap();
        let want = eng.model_count(&eng.negate(&f).unwrap()).unwrap();
        let mut refused = 0;
        for k in 0..4096 {
            eng.limits().refuse_nth_reserve(k);
            let got = eng.skip_model_count(&marked);
            eng.limits().grant_every_reserve();
            match got {
                Err(OperationError::OverBudget) => {
                    refused += 1;
                    assert_eq!(eng.skip_model_count(&marked).unwrap(), want, "after refusal {k}");
                }
                Ok(v) => {
                    assert_eq!(v, want);
                    break;
                }
                Err(other) => panic!("refusal {k}: {other:?}"),
            }
        }
        assert!(refused > 0, "building the negation reserved nothing");
    }
}

/// A conjunction with a marked operand negates the smaller and conjoins, or
/// leaves the larger to the overlay; of two marked operands far apart it
/// negates and conjoins both. Each gives the stored conjunction, or refuses
/// and leaves the engine able to run it again.
#[test]
fn a_conjunction_with_a_marked_operand_is_built_at_every_reserve_or_refused() {
    use crate::OperationError;
    for vtree in [Arc::new(crate::Vtree::balanced(16)), Arc::new(crate::Vtree::linear(16))] {
        let eng = Engine::new();
        // A chain of clauses over every variable, on every level.
        let chain: Vec<Vec<i32>> = (1..16).map(|v| if v % 2 == 1 { vec![v, v + 1] } else { vec![-v, -(v + 1)] }).collect();
        let large = compile_clauses_on(&eng, &vtree, &chain);
        let sl = eng.to_skip(&large).unwrap();
        // A literal on under half the levels of `large`.
        let (small, ss) = (1..=16)
            .flat_map(|v| [v, -v])
            .map(|l| compile_clauses_on(&eng, &vtree, &[vec![l]]))
            .map(|s| (s.clone(), eng.to_skip(&s).unwrap()))
            .find(|(_, ss)| super::api::apart(&sl.form, &ss.form))
            .unwrap_or_else(|| panic!("no literal on under half the {} levels of the chain", sl.form.active.len()));
        let (nl, ns) = (eng.negate(&large).unwrap(), eng.negate(&small).unwrap());
        for (f, g, want) in [
            (sl.clone(), eng.skip_negate(&ss).unwrap(), eng.and(&large, &ns).unwrap()),
            (eng.skip_negate(&sl).unwrap(), ss.clone(), eng.and(&nl, &small).unwrap()),
            (eng.skip_negate(&sl).unwrap(), eng.skip_negate(&ss).unwrap(), eng.and(&nl, &ns).unwrap()),
        ] {
            let want = eng.model_count(&want).unwrap();
            let mut refused = 0;
            for k in 0..4096 {
                eng.limits().refuse_nth_reserve(k);
                let got = eng.skip_and(&f, &g);
                eng.limits().grant_every_reserve();
                match got {
                    Err(OperationError::OverBudget) => {
                        refused += 1;
                        assert_eq!(eng.skip_model_count(&eng.skip_and(&f, &g).unwrap()).unwrap(), want, "after refusal {k}");
                    }
                    Ok(h) => {
                        assert_eq!(eng.skip_model_count(&h).unwrap(), want);
                        break;
                    }
                    Err(other) => panic!("refusal {k}: {other:?}"),
                }
            }
            assert!(refused > 0, "the conjunction reserved nothing");
        }
    }
}
