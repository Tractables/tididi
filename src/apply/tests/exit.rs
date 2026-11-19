//! What each operation returns without the closing minimization, and that
//! `canonical_results` certifies it.

use super::*;
use crate::diagram::Literal;
use crate::limits::LimitConfig;
use crate::test_helpers::check::{check_certified, check_implicit_levels, check_no_false_nodes, check_trimmed};

/// The invariants every stored node holds after any operation, minimized or
/// not; trimmed on a home only.
pub(crate) fn check_unminimized(f: &Tdd, label: &str) {
    let on_home = f.vtree().home().is_some() || f.vtree().is_home();
    for (checker, r) in [
        ("trimmed", if on_home { check_trimmed(f, false) } else { Ok(()) }),
        ("no_false_nodes", check_no_false_nodes(f)),
        ("implicit levels", check_implicit_levels(f)),
    ] {
        if let Err(e) = r {
            panic!("{label}: {checker}: {e}");
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

const N: u32 = 7;

/// An uncertified conjunction of random clauses over `N` variables.
fn random(eng: &Engine, vtree: &Arc<Vtree>, rng: &mut Lcg) -> Tdd {
    let mut f = eng.cube(vtree, std::iter::empty::<i32>()).unwrap();
    for _ in 0..1 + rng.next() % 4 {
        let width = 1 + rng.next() % 3;
        let clause: Vec<i32> = (0..width)
            .map(|_| {
                let v = 1 + (rng.next() % u64::from(N)) as i32;
                if rng.next().is_multiple_of(2) { v } else { -v }
            })
            .collect();
        f = eng.and(f, eng.clause(vtree, clause).unwrap()).unwrap();
    }
    f
}

/// Run `op` on an engine without the flag and on one with it: the first
/// result passes [`check_unminimized`], the second is certified, and both
/// denote the same function.
fn compare(label: &str, op: impl Fn(&Engine) -> Tdd) {
    let plain = Engine::new();
    let got = op(&plain);
    check_unminimized(&got, label);
    let canonical = Engine::new();
    let _prior = canonical.limits().install(LimitConfig::none().with_canonical_results(true));
    let want = op(&canonical);
    assert!(want.is_canonical(), "{label}: the flag certifies the result");
    if let Err(e) = check_certified(&want) {
        panic!("{label}: certified: {e}");
    }
    assert!(got.equivalent(&want).unwrap(), "{label}: the same function either way");
}

#[test]
fn every_changed_operation_keeps_the_invariants_and_the_flag_certifies() {
    let home = crate::vtree::Home::new(Vtree::balanced(N));
    let x = |v: u32| VarId(v);
    for seed in 0..120u64 {
        let vtree = if seed % 2 == 0 { Arc::new(Vtree::balanced(N)) } else { home.vtree().clone() };
        let mut rng = Lcg(seed);
        let e = Engine::new();
        let (f, g, h) = (random(&e, &vtree, &mut rng), random(&e, &vtree, &mut rng), random(&e, &vtree, &mut rng));
        let (a, b) = (x(1 + (rng.next() % u64::from(N)) as u32), x(1 + (rng.next() % u64::from(N)) as u32));
        let label = |op: &str| format!("seed {seed}: {op}");
        compare(&label("or"), |eng| eng.or(f.clone(), g.clone()).unwrap());
        compare(&label("and_not"), |eng| eng.and_not(f.clone(), g.clone()).unwrap());
        compare(&label("add"), |eng| {
            let only_g = eng.and_not(g.clone(), f.clone()).unwrap();
            eng.add(f.clone(), only_g).unwrap()
        });
        compare(&label("or_many"), |eng| eng.or_many(vec![f.clone(), g.clone(), h.clone()]).unwrap());
        compare(&label("nor_many"), |eng| eng.nor_many(vec![f.clone(), g.clone(), h.clone()]).unwrap());
        compare(&label("nor_many lone"), |eng| eng.nor_many(vec![f.clone()]).unwrap());
        compare(&label("xor"), |eng| eng.xor(f.clone(), g.clone()).unwrap());
        compare(&label("ite"), |eng| eng.ite(f.clone(), g.clone(), h.clone()).unwrap());
        compare(&label("and_exists"), |eng| eng.and_exists(f.clone(), g.clone(), &[a, b]).unwrap());
        compare(&label("negate"), |eng| eng.negate(&f).unwrap());
        compare(&label("negate certified"), |eng| {
            let mut m = f.clone();
            eng.minimize(&mut m).unwrap();
            eng.negate(m).unwrap()
        });
        compare(&label("condition"), |eng| eng.condition(f.clone(), [Literal::new(a, true), Literal::new(b, false)]).unwrap());
        compare(&label("condition_vars"), |eng| eng.condition_vars(f.clone(), &[a, b], true).unwrap());
        compare(&label("exists_vars"), |eng| eng.exists_vars(f.clone(), &[a, b]).unwrap());
        compare(&label("forall_vars"), |eng| eng.forall_vars(f.clone(), &[a, b]).unwrap());
        if a != b {
            compare(&label("rename_vars"), |eng| eng.rename_vars(f.clone(), &[(a, b), (b, a)]).unwrap());
        }
        compare(&label("substitute"), |eng| eng.substitute(f.clone(), &[(a, &g)]).unwrap());
        compare(&label("restrict_to_care"), |eng| eng.restrict_to_care(f.clone(), g.clone()).unwrap().into_tdd());
        compare(&label("filter_nodes"), |eng| {
            let mut i = 0u32;
            match eng.filter_nodes(&f, |_| { i += 1; !i.is_multiple_of(3) }).unwrap() {
                crate::apply::FilterOutcome::Filtered { tdd, .. } | crate::apply::FilterOutcome::Unsatisfiable { tdd, .. } => tdd,
                crate::apply::FilterOutcome::Unchanged => {
                    let mut m = f.clone();
                    eng.minimize(&mut m).unwrap();
                    m
                }
            }
        });
    }
}
