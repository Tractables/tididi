//! The conjunction's routes name no leaf `One` on operands that name none,
//! and agree level by level with the default route.

use std::sync::Arc;

use crate::test_helpers::check::reachable_true_labels;
use crate::test_helpers::{rand_conj, stored_levels, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::{Engine, Tdd};

use super::*;

/// Pairs of operands on projections of one home: random conjunctions over 6
/// to 10 variables on a balanced and a right-linear vtree, each minimized
/// and trimmed, so that it names no leaf `One`.
fn operands() -> Vec<(String, Tdd, Tdd)> {
    let mut out = Vec::new();
    for seed in 0..12u64 {
        let mut rng = Lcg::new(seed ^ 0x2e2);
        let n = 6 + rng.below(5) as u32;
        for (shape, home) in vtree_shapes(n).into_iter().filter(|(s, _)| matches!(*s, "balanced" | "linear")) {
            let hm = home;
            let home = Arc::clone(&hm);
            let make = |rng: &mut Lcg| {
                let mut f = rand_conj(&home, n, 2 + rng.below(6), 3, false, rng);
                f.minimize().unwrap();
                f.trim(&hm).unwrap();
                f
            };
            let (f, g) = (make(&mut rng), make(&mut rng));
            out.push((format!("seed {seed}, {n} variables, {shape}"), f, g));
        }
    }
    out
}

/// Every conjunction of `pairs` under `setting`.
fn conjoined(pairs: &[(String, Tdd, Tdd)], setting: impl FnOnce(&dyn Fn() -> Vec<Tdd>) -> Vec<Tdd>) -> Vec<Tdd> {
    let eng = Engine::new();
    setting(&|| pairs.iter().map(|(_, f, g)| eng.and(f, g).unwrap()).collect())
}

/// Each conjunction under `setting` names no leaf `One` and has the levels of
/// the default route's.
pub(in crate::apply::conjoin) fn check_route(name: &str, setting: impl FnOnce(&dyn Fn() -> Vec<Tdd>) -> Vec<Tdd>) {
    let eng = Engine::new();
    let cases = operands();
    let want = conjoined(&cases, |f| f());
    let got = conjoined(&cases, setting);
    for (((case, _, _), w), g) in cases.iter().zip(&want).zip(&got) {
        assert_eq!(reachable_true_labels(g), Ok(None), "{name}, {case}");
        assert_eq!((g.node_count(), g.pair_count()), (w.node_count(), w.pair_count()), "{name}, {case}");
        assert!(eng.xor(g.clone(), w.clone()).unwrap().is_zero(), "{name}, {case}");
    }
}

#[test]
fn audit_i1_routes_name_no_one() {
    assert!(conjoined(&operands(), |f| f()).iter().all(|t| reachable_true_labels(t) == Ok(None)));
    check_route("grid lookups", |f| grid_lookups(f));
    check_route("pair rows off", |f| pair_rows_off(f));
    check_route("word cells on", |f| word_cells_on(f));
    check_route("word cells off", |f| word_cells_off(f));
    check_route("single rows off", |f| single_rows_off(f));
    check_route("one-sided masks off", |f| one_sided_masks_off(f));
    check_route("lone columns off", |f| column_reads(f));
    check_route("relay off", |f| relay_off(f));
    check_route("stored levels", |f| stored_levels(f));
    check_route("complete sides listed", |f| list_complete_sides(f));
    check_route("lend off", |f| no_lend(f));
    check_route("probe forced", |f| always_probe(f));
    check_route("probe off", |f| no_probe(f));
}

#[test]
fn audit_i1_routes_operands_name_no_one() {
    for (case, f, g) in operands() {
        assert_eq!(reachable_true_labels(&f), Ok(None), "{case}");
        assert_eq!(reachable_true_labels(&g), Ok(None), "{case}");
        assert!(f.vtree().num_nodes() == 0 || Arc::ptr_eq(f.vtree().link_home().map(|home| &**home).unwrap_or(f.vtree()), g.vtree().link_home().map(|home| &**home).unwrap_or(g.vtree())) || g.vtree().num_nodes() == 0, "{case}: one home");
    }
}
