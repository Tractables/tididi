use std::sync::Arc;

use super::convert::{from_tdd, to_tdd};
use super::{conjoin, contract::contract, ix, vt, Ref, Shape, SkipForm, ZREF};
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::{Engine, Tdd};

fn skip(f: &Tdd) -> SkipForm {
    from_tdd(f, Arc::new(Shape::of(f.vtree()))).expect("structural")
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
        assert_eq!(s.model_count(), eng.model_count(&f).unwrap());
        let mut back = to_tdd(&s, &eng, f.vtree()).unwrap();
        eng.minimize(&mut back).unwrap();
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
        assert_eq!(h.model_count(), eng.model_count(&want).unwrap());
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
        let h = contract(conjoin::and(&sf, &sg, eng.limits()).unwrap(), eng.limits()).unwrap();
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
            assert_eq!(h.model_count(), eng.model_count(&fun).unwrap());
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
    assert_eq!(one.model_count(), num_bigint::BigUint::from(1u32) << f.vtree().num_vars());
    assert!(super::negate::negate(&one, eng.limits()).unwrap().is_zero());
}

type Dump = Vec<(usize, Vec<Vec<((u32, u32, u32), (u32, u32, u32))>>)>;

fn dump(s: &SkipForm) -> Dump {
    s.levels
        .iter()
        .enumerate()
        .filter(|l| l.1.width() > 0)
        .map(|(i, l)| (i, (0..l.width()).map(|k| l.node(k as u32).iter().map(|&(a, b)| ((vt(a), ix(a), (a >> 62) as u32), (vt(b), ix(b), (b >> 62) as u32))).collect()).collect()))
        .collect()
}

#[test]
fn tagged_form_round_trips() {
    use super::tag::{tag_form, untag};
    for (eng, f, _) in cases(6, 300) {
        let s = skip(&f);
        let t = tag_form(&s);
        assert!(t.size_tagged().pairs <= s.size().pairs);
        assert_eq!(fingerprint(&untag(&t)), fingerprint(&s));
        assert_eq!(fingerprint(&tag_form(&untag(&t))), fingerprint(&t));
        let _ = eng;
    }
}

#[test]
fn tagged_conjunction_matches_the_untagged_one() {
    use super::tag::{self, tag_form, untag};
    for (eng, f, g) in cases(7, 300) {
        let (sf, sg) = (skip(&f), skip(&g));
        let sg = SkipForm { shape: Arc::clone(&sf.shape), ..sg };
        let want = contract(conjoin::and(&sf, &sg, eng.limits()).unwrap(), eng.limits()).unwrap();
        let (tf, tg) = (tag_form(&sf), tag_form(&sg));
        let raw = tag::and(&tf, &tg, eng.limits()).unwrap();
        // Before contraction: the same function.
        let ok = untag(&raw).model_count() == want.model_count();
        let h = tag::contract(&raw, eng.limits()).unwrap();
        if !ok || fingerprint(&untag(&h)) != fingerprint(&want) || fingerprint(&h) != fingerprint(&tag_form(&want)) {
            eprintln!("vtree {:?}", sf.shape.kids);
            eprintln!("untag(h) {:?} out {:x}", dump(&untag(&h)), untag(&h).out);
            eprintln!("want(untagged) {:?} out {:x}", dump(&want), want.out);
            eprintln!("tf {:?} out {:x}", dump(&tf), tf.out);
            eprintln!("tg {:?} out {:x}", dump(&tg), tg.out);
            eprintln!("raw {:?} out {:x}", dump(&raw), raw.out);
            eprintln!("mine {:?} out {:x}", dump(&h), h.out);
            eprintln!("want {:?} out {:x}", dump(&tag_form(&want)), tag_form(&want).out);
        }
        assert!(ok, "the raw conjunction's function");
        assert_eq!(fingerprint(&untag(&h)), fingerprint(&want));
        assert_eq!(fingerprint(&h), fingerprint(&tag_form(&want)));
        assert_eq!(fingerprint(&tag::contract(&h, eng.limits()).unwrap()), fingerprint(&h));
    }
}

#[test]
fn placement_matches_the_stored_embedding() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(8);
    let mut done = 0;
    while done < 300 {
        let n = 3 + rng.below(8) as u32;
        for (_, vtree) in vtree_shapes(n) {
            // A random subset of at least one variable, and its projection.
            let keep: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
            if keep.is_empty() {
                continue;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let Some(small) = vtree.project_to_vars(local, keep.len() as u32) else { continue };
            let eng = Engine::new();
            let small = eng.bind_vtree(small);
            let m = keep.len() as u32;
            let f = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, m, CnfShape { clauses: m as usize, width: 3 }));
            let map = |v: VarId| VarId(keep[v.0 as usize - 1]);
            let ish = Arc::new(Shape::of(&vtree));
            let placed = super::place::place(&skip(&f), &small, map, &vtree, &ish).expect("a projection places");
            let (mut want, _) = eng.embed(&f, &vtree, map).unwrap();
            eng.minimize(&mut want).unwrap();
            if fingerprint(&placed) != fingerprint(&skip(&want)) {
                eprintln!("keep {keep:?} big {:?}", ish.kids);
                eprintln!("small {:?}", Shape::of(&small).kids);
                let s = skip(&f);
                eprintln!("f {:?} out {:x}", dump(&s), s.out);
                eprintln!("placed {:?} out {:x}", dump(&placed), placed.out);
                let w = skip(&want);
                eprintln!("want {:?} out {:x}", dump(&w), w.out);
            }
            assert_eq!(fingerprint(&placed), fingerprint(&skip(&want)));
            done += 1;
        }
    }
}

/// Tagged-form invariants: a reference that names its own child level has
/// tag zero, and no stored node of one pair resolves to a reference.
fn check_tagged(s: &SkipForm) -> Result<(), String> {
    let sh = &*s.shape;
    for &u in &sh.topdown {
        let (l, r) = sh.kids[u as usize];
        let lvl = &s.levels[u as usize];
        for i in 0..lvl.width() {
            let node = lvl.node(i as u32);
            for &(a, b) in node {
                for (c, x) in [(l, a), (r, b)] {
                    if !sh.is_leaf(c) && vt(x) == c && x >> 62 != 0 {
                        return Err(format!("level {u} node {i}: direct reference to {c} tagged {}", x >> 62));
                    }
                }
            }
            if node.len() == 1 {
                let mut p = node.to_vec();
                if super::tag::normalize(sh, u, &mut p).is_err() {
                    return Err(format!("level {u} node {i}: {:?} resolves", ((vt(node[0].0), ix(node[0].0), node[0].0 >> 62), (vt(node[0].1), ix(node[0].1), node[0].1 >> 62))));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn tagged_forms_keep_their_invariants() {
    use super::tag::{self, tag_form};
    for (eng, f, g) in cases(9, 300) {
        let (sf, sg) = (skip(&f), skip(&g));
        let sg = SkipForm { shape: Arc::clone(&sf.shape), ..sg };
        let (tf, tg) = (tag_form(&sf), tag_form(&sg));
        check_tagged(&tf).unwrap();
        check_tagged(&tg).unwrap();
        let raw = tag::and(&tf, &tg, eng.limits()).unwrap();
        check_tagged(&raw).map_err(|e| format!("and: {e}")).unwrap();
        check_tagged(&tag::contract(&raw, eng.limits()).unwrap()).map_err(|e| format!("contract: {e}")).unwrap();
    }
}

#[test]
fn conjunction_of_placed_projections_is_canonical() {
    use crate::vtree::VarId;
    let mut rng = Lcg::new(10);
    let mut done = 0;
    while done < 300 {
        let n = 3 + rng.below(8) as u32;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let ish = Arc::new(Shape::of(&vtree));
            let mut placed = Vec::new();
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
                placed.push(super::place::place(&skip(&f), &small, map, &vtree, &ish).expect("a projection places"));
                embedded.push(eng.embed(&f, &vtree, map).unwrap().0);
            }
            if placed.len() < 2 {
                continue;
            }
            let h = contract(conjoin::and(&placed[0], &placed[1], eng.limits()).unwrap(), eng.limits()).unwrap();
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
            let h = contract(conjoin::and(&sf, &sg, eng.limits()).unwrap(), eng.limits()).unwrap();
            assert_eq!(fingerprint(&h), fingerprint(&want));
            done += 1;
        }
    }
}
