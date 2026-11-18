//! Embedding, projection and conjunction of canonical diagrams emit no level
//! that names a leaf child both as `One` and as a literal, which
//! [`TddBuilder::finish`](crate::diagram::TddBuilder::finish) refuses and
//! `trim` would read wrongly.

use crate::test_helpers::check::assert_no_mixed_leaf_side as assert_one_form;
use crate::test_helpers::{rand_conj, rand_conj_over, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::Engine;

fn subset(n: u32, rng: &mut Lcg) -> Vec<u32> {
    (1..=n).filter(|_| rng.coin()).collect()
}

#[test]
fn embedding_projection_and_conjunction_keep_one_leaf_form() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x51ce_0f0e);
    let (mut embedded, mut projected, mut on_projections) = (0, 0, 0);
    for n in [3u32, 5, 7] {
        let wider = vtree_shapes(n + 2);
        for (name, vtree) in vtree_shapes(n) {
            let home = Home::new(vtree.clone());
            for _ in 0..6 {
                let mut f = rand_conj(&vtree, n, 5, 3, false, &mut rng);
                let mut g = rand_conj(&vtree, n, 5, 3, false, &mut rng);
                f.minimize().unwrap();
                g.minimize().unwrap();
                assert_one_form(&f, "input");
                assert_one_form(&eng.and(&f, &g).unwrap(), &format!("and on {name}"));
                for (into_name, into) in &wider {
                    if let Ok((e, _)) = eng.embed(&f, into, |v| VarId(v.0 + 1)) {
                        assert_one_form(&e, &format!("embed {name} into {into_name}"));
                        embedded += 1;
                    }
                }
                let keep = subset(n, &mut rng);
                if !keep.is_empty() {
                    let local = |v: VarId| keep.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
                    let p = eng.project_to_vars(&f, local, keep.len() as u32).unwrap();
                    assert_one_form(&p, &format!("project {name} to {keep:?}"));
                    projected += 1;
                }
                let (a, b) = (subset(n, &mut rng), subset(n, &mut rng));
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                let own = |vars: &[u32]| Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap();
                let fa = rand_conj_over(&own(&a), &a, 4, 3, false, &mut rng);
                let fb = rand_conj_over(&own(&b), &b, 4, 3, false, &mut rng);
                assert_one_form(&fa, "input on a projection");
                assert_one_form(&eng.and(&fa, &fb).unwrap(), &format!("and on projections {a:?} {b:?} of {name}"));
                on_projections += 1;
            }
        }
    }
    assert!(embedded > 0 && projected > 0 && on_projections > 0);
}
