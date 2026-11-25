//! A conjunction that is false holds no node, by every route `and` takes.

use super::*;
use crate::test_helpers::check::check_no_false_nodes;
use crate::vtree::Home;

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

const N: u32 = 6;

/// Clauses over the variables `1..N`, the last one naming all of them, so
/// that on a home both operands are on one projection.
fn random_clauses(rng: &mut Lcg) -> Vec<Vec<i32>> {
    let mut clauses: Vec<Vec<i32>> = (0..2 + rng.below(4))
        .map(|_| {
            (0..1 + rng.below(2))
                .map(|_| {
                    let v = 1 + rng.below(u64::from(N - 1)) as i32;
                    if rng.below(2) == 0 { v } else { -v }
                })
                .collect()
        })
        .collect();
    clauses.push((1..N as i32).collect());
    clauses
}

fn conjunction(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut f = eng.cube(&crate::vtree::Home::unchecked(vtree), std::iter::empty::<i32>()).unwrap();
    for clause in clauses {
        f = eng.and(f, eng.clause(&crate::vtree::Home::unchecked(vtree), clause.as_slice()).unwrap()).unwrap();
    }
    f
}

fn check(label: &str, got: &Tdd, falses: &mut usize) {
    if got.is_zero() {
        *falses += 1;
        check_no_false_nodes(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    }
}

#[test]
fn false_conjunctions_hold_no_node() {
    let home = Home::new(Vtree::balanced(N));
    let mut falses = 0;
    for seed in 0..400u64 {
        let mut rng = Lcg(seed);
        let eng = Engine::new();
        let on_home = seed % 2 == 1;
        let vtree = if on_home { Arc::clone(&home) } else { Arc::new(Vtree::balanced(N)) };
        let (fc, gc) = (random_clauses(&mut rng), random_clauses(&mut rng));
        let (f, g) = (conjunction(&eng, &vtree, &fc), conjunction(&eng, &vtree, &gc));
        if f.is_zero() || g.is_zero() {
            continue;
        }
        let label = |route: &str| format!("seed {seed} f {fc:?} g {gc:?}: {route}");
        check(&label("owned"), &eng.and(f.clone(), g.clone()).unwrap(), &mut falses);
        check(&label("borrowed"), &eng.and(&f, &g).unwrap(), &mut falses);
        check(&label("one borrowed"), &eng.and(f.clone(), &g).unwrap(), &mut falses);
        check(&label("restoring"), &eng.and_restoring(f.clone(), g.clone()).map_err(|r| r.error).unwrap(), &mut falses);
        check(&label("restoring borrowed"), &eng.and_restoring(&f, &g).map_err(|r| r.error).unwrap(), &mut falses);
        check(&label("in place"), &eng.and_in_place(f.clone(), &g).unwrap(), &mut falses);
        check(&label("in place loose"), &eng.and_in_place_loose(f.clone(), &g).unwrap(), &mut falses);
        let mut clausal = f.clone();
        for clause in &gc {
            clausal = eng.and_clause(clausal, clause.as_slice()).unwrap();
        }
        check(&label("and_clause"), &clausal, &mut falses);
        if on_home {
            continue;
        }
        let index = crate::diagram::PairIndex::new(&g).unwrap();
        check(&label("indexed"), &eng.and_indexed(&f, &g, &index).unwrap(), &mut falses);
        let (left, _) = vtree.children(vtree.root());
        check(&label("marginalizing"), &crate::test_helpers::and_marginalizing_under(&eng, &f, &g, &[left]).unwrap(), &mut falses);
    }
    assert!(falses > 200, "the seeds give false conjunctions ({falses})");
}
