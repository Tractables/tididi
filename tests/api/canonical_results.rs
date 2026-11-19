//! A threshold comparator built as a chain of `ite`, as a linear-constraint
//! compiler builds one, conjoined with a cube on a variable it does not
//! read; with and without `canonical_results`.

use std::sync::Arc;

use tididi::limits::LimitConfig;
use tididi::vtree::VarId;
use tididi::{Engine, Tdd, Vtree};

/// Variables 2 to 21, most significant first: `true` where a set bit makes
/// the comparison true, `false` where a clear bit makes it false. Variable 22
/// is the last bit and variable 1 is read by neither branch.
const SET_DECIDES: [bool; 20] = [
    true, false, true, true, false, true, true, false, true, true, true, false, false, false, true, false, true, true, true, true,
];

fn comparator(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let lit = |v: u32| eng.literal(vtree, v as i32).unwrap();
    let mut f = lit(22);
    for (v, set_decides) in (2..22).zip(SET_DECIDES).rev() {
        f = match set_decides {
            true => eng.ite(lit(v), eng.one(vtree), f),
            false => eng.ite(lit(v), f, eng.zero(vtree)),
        }
        .unwrap();
    }
    eng.ite(lit(1), f.clone(), f).unwrap()
}

fn minimized(eng: &Engine, f: &Tdd) -> Tdd {
    let mut f = f.clone();
    eng.minimize(&mut f).unwrap();
    f
}

#[test]
fn conjoining_an_ite_chain_is_minimal_on_minimized_operands_and_under_the_flag() {
    let order: Vec<VarId> = (1..=22).map(VarId).collect();
    let vtree = Arc::new(Vtree::linear_from_order(&order).unwrap());
    let codes = Engine::new().cube(&vtree, [-1]).unwrap();

    let plain = Engine::new();
    let compared = comparator(&plain, &vtree);
    let got = plain.and(&compared, &codes).unwrap();
    let fewest = minimized(&plain, &got).pair_count();
    assert!(got.pair_count() >= fewest);
    let of_minimized = plain.and(minimized(&plain, &compared), minimized(&plain, &codes)).unwrap();
    assert_eq!(of_minimized.pair_count(), fewest);
    assert!(of_minimized.equivalent(&got).unwrap());

    let canonical = Engine::new();
    let _prior = canonical.limits().install(LimitConfig::none().with_canonical_results(true));
    let compared = comparator(&canonical, &vtree);
    assert!(compared.is_canonical());
    let both = canonical.and(&compared, &codes).unwrap();
    assert!(both.is_canonical());
    assert_eq!(both.pair_count(), fewest);
    assert!(both.equivalent(&got).unwrap());
}
