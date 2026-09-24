//! Variable support, conditioning and implied literals.
//!
//! Fixtures come from `crate::test_helpers`, re-exported by the parent.

use super::*;

use crate::Engine;

#[test]
fn support_mask_tracks_dependence() {
    // f = (x0 & x2): depends on x0, x2 but not x1.
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let x2 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true)]));
    let f = and2(&x0, &x2);
    let sup = support_mask(&f);
    assert_eq!(sup, vec![true, false, true], "support should be {{x0,x2}}");
    // Cross-check against the project-equality oracle: f independent of x iff
    // projecting x out leaves f equivalent (over the care of the other vars).
    for x in 0..3u32 {
        let projected = (f).clone().exists_var(VarId(x + 1)).unwrap();
        let unchanged = equiv(&f, &projected);
        assert_eq!(!unchanged, sup[x as usize], "support[{x}] mismatch vs oracle");
    }
}

#[test]
fn support_bits_covers_support_mask_and_detects_disjoint() {
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let x2 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true)]));
    let f = and2(&x0, &x2); // depends on {x0, x2}
    // `support_bits` OVER-approximates `support_mask` (never drops a real dependency);
    // on a minimized diagram like this it is exact.
    let mask = support_mask(&f);
    let bits = support_bits(&f);
    for (x, &m) in mask.iter().enumerate() {
        let b = (bits[x / 64] >> (x % 64)) & 1 == 1;
        assert!(!m || b, "support_bits must cover support_mask at var {x}");
        assert_eq!(b, m, "support_bits exact on minimized f at var {x}");
    }
    // Disjoint detection: g depends only on x1, sharing no variable with f.
    let g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(2, true)]));
    let bg = support_bits(&g);
    assert!(
        bits.iter().zip(bg.iter()).all(|(a, b)| a & b == 0),
        "f={{x0,x2}} and g={{x1}} must be detected disjoint"
    );
    // Overlapping support (shares x0) is not flagged disjoint.
    let bx0 = support_bits(&x0);
    assert!(
        !bits.iter().zip(bx0.iter()).all(|(a, b)| a & b == 0),
        "f={{x0,x2}} and x0 share x0 → must NOT be disjoint"
    );
}

#[test]
fn condition_var_detects_unit_forced_apply() {
    let vtree = Arc::new(Vtree::balanced(3));
    // (x0) AND (x0 v x1) AND (x1 v x2) -- x0 forced TRUE by the unit.
    let c0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));
    let t01 = apply_and(c0, f);
    let t = apply_and(t01, g);
    assert!(!count_is_zero(&t));
    // x0 forced true => x0=false is UNSAT (count 0), x0=true is SAT.
    assert!(count_is_zero(&(t).clone().condition_var(VarId(1), false).unwrap()));
    assert!(!count_is_zero(&(t).clone().condition_var(VarId(1), true).unwrap()));
}

// A conditioned diagram with no models must be CANONICALLY false: conditioning
// plus minimize can leave the output node holding pairs whose every path is
// dead (`model_count == 0`, `is_zero() == false`). Counting that is correct, but
// re-conjoining it revives the models the restriction killed, so `condition_*`
// collapses it to ZERO. Same diagram as `condition_var_detects_unit_forced_apply`.
#[test]
fn condition_var_canonicalizes_a_dead_result() {
    let vtree = Arc::new(Vtree::balanced(3));
    let c0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));
    let t01 = apply_and(c0, f);
    let t = apply_and(t01, g);
    let dead = (t).clone().condition_var(VarId(1), false).unwrap();
    assert!(count_is_zero(&dead), "x0 is forced true, so x0=false has no models");
    assert!(dead.is_zero(), "a model-count-0 conditioning result must be canonically ZERO");
    // Re-conjoining the canonical ⊥ stays ⊥ (the property the canonicalization buys).
    assert!(count_is_zero(&and2(&dead, &t)));
}

// Soundness contract: conditioning a leaf whose own level was marginalized must
// fail fast. `rewrite_for_restrict` matches the target-side ref against
// POS/NEG/one, and leaf-marginal rewrites exactly those refs into inline marginal counts
// in the same numeric space — so without the guard the variable is silently left
// unconditioned (miscount, no panic).
#[test]
fn condition_var_on_marginalized_leaf_fails_fast() {
    let eng = &crate::Engine::new();
    use crate::marginal::marginalize_leaf_inline;
    let vtree = Arc::new(Vtree::balanced(2));
    // x0 XOR x1 — depends on both vars, so the output sits at the root.
    let mut t = and2(
        &clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)])),
        &clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, false), (2, false)])),
    );
    let leaf = vtree.leaf_of(VarId(2)).expect("the vtree carries this variable");
    marginalize_leaf_inline(&mut t, leaf, &vtree);
    assert!(t.levels[leaf.idx()].is_marginal(), "test setup: leaf must be marginal");
    t.minimize().unwrap();
    assert_canonical(&t);
    assert_eq!(eng.condition_var(t, VarId(2), true).unwrap_err(), crate::OperationError::MarginalLevel(leaf));
}

// Same contract for the leaf's PARENT: a marginal parent holds marginal-slot refs
// (and no `nodes`), so the rewrite would read slot indices as leaf labels and
// then silently no-op.
#[test]
fn condition_var_through_marginal_parent_fails_fast() {
    let eng = Engine::new();
    use crate::marginal::marginalize_batch;
    let vtree = Arc::new(Vtree::balanced(2));
    // x0 XOR x1 — depends on both vars, so the output sits at the root.
    let mut t = and2(
        &clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)])),
        &clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, false), (2, false)])),
    );
    let leaf = vtree.leaf_of(VarId(1)).expect("the vtree carries this variable");
    let parent = vtree.node(leaf).parent().expect("leaf has a parent");
    marginalize_batch(&eng, &mut t, &[parent], &vtree).expect("no wall is installed in a test");
    assert!(!t.levels[leaf.idx()].is_marginal(), "test setup: only the parent is marginal");
    assert_canonical(&t);
    assert_eq!(eng.condition_var(t, VarId(1), true).unwrap_err(), crate::OperationError::MarginalLevel(parent));
}

#[test]
fn implied_literals_matches_condition_oracle() {
    use crate::diagram::Literal;


    // A literal is implied iff conditioning its variable to the opposite sign
    // leaves no satisfying assignments, including when f already has none.
    let oracle = |f: &Tdd, nvars: u32| -> Vec<Literal> {
        let mut out = Vec::new();
        for v in 0..nvars {
            for val in [false, true] {
                if count_is_zero(&(f).clone().condition_var(VarId(v + 1), !val).unwrap()) {
                    out.push(Literal::new(VarId(v + 1), val));
                }
            }
        }
        out
    };
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let nx0 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, false)]));
    let x1 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(2, true)]));
    let or12 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));

    // f = x0 & (x1 | x2): only x0 is backbone (x1,x2 each stay free).
    let mut f = and2(&x0, &or12);
    f.minimize().unwrap();
    let bb = f.implied_literals().unwrap();
    assert_eq!(bb, oracle(&f, 3));
    assert!(bb == [Literal::pos(VarId(1))]);

    // g = ~x0 & x1: x0 forced false, x1 forced true, x2 a pure don't-care (only
    // ever the One leaf) — must not appear.
    let mut g = and2(&nx0, &x1);
    g.minimize().unwrap();
    let bbg = g.implied_literals().unwrap();
    assert_eq!(bbg, oracle(&g, 3));
    assert!(bbg.iter().all(|lit| lit.var != VarId(3)));

    // UNSAT (x0 & ~x0) implies both signs, even for the unused variables.
    let mut z = and2(&x0, &nx0);
    z.minimize().unwrap();
    crate::test_helpers::assert_canonical(&z);
    assert_eq!(z.implied_literals().unwrap(), oracle(&z, 3));
    assert_eq!(z.implied_literals().unwrap().len(), 6);
}
