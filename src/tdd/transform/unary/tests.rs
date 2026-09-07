//! Integration tests for the unary transform family (project / condition /
//! restrict / demarginalize) together with the `query::support` structural
//! queries. Relocated verbatim from the former `project.rs`: the `mod tests`
//! body below is byte-for-byte unchanged, and the sibling symbols it reaches
//! via `super::` are re-bound into this file's module scope by the `use` block
//! below (repointed to the post-split module paths).

use crate::tdd::transform::unary::project::{
    project_var, project_var_scoped, project_vars_scoped,
    POS, NEG, ONE,
};
use crate::tdd::transform::unary::condition::condition_var;
use crate::tdd::transform::unary::demarginalize::demarginalize_to_indicator;
use crate::tdd::transform::unary::restrict::{restrict, Restricted, CareCanonical};
use crate::tdd::query::support::{support_mask, support_bits, implied_literals, reachable_pairs};
use crate::tdd::types::{ZERO, LocalNodeIdx};

mod tests {
    use std::sync::Arc;

    use num_bigint::BigUint;

    use super::{project_var, project_var_scoped, project_vars_scoped};
    use super::support_mask;
    use crate::vtree::Literal;
    use crate::tdd::transform::pairwise::conjoin::apply_and;
    use crate::tdd::build::{clause_to_tdd, constant_one, constant_zero};
    use crate::tdd::query::model_count;
    use crate::vtree::{VarId, Vtree, VtreeIdx};

    use super::condition_var;
    use crate::tdd::transform::pairwise::disjoin::apply_or;

    fn lit(var: u32, positive: bool) -> Literal {
        Literal::new(VarId(var), positive)
    }

    fn clause(lits: &[(u32, bool)]) -> Vec<Literal> {
        lits.iter().map(|&(v, p)| lit(v, p)).collect()
    }

    #[test]
    fn support_mask_tracks_dependence() {
        // f = (x0 & x2): depends on x0, x2 but NOT x1.
        let vtree = Arc::new(Vtree::balanced(3));
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let x2 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
        let f = and2(&x0, &x2);
        let sup = support_mask(&f);
        assert_eq!(sup, vec![true, false, true], "support should be {{x0,x2}}");
        // Cross-check against the project-equality oracle: f independent of x iff
        // projecting x out leaves f equivalent (over the care of the other vars).
        for x in 0..3u32 {
            let projected = project_var(&f, VarId(x));
            let unchanged = equiv(&f, &projected);
            assert_eq!(!unchanged, sup[x as usize], "support[{x}] mismatch vs oracle");
        }
    }

    #[test]
    fn support_bits_covers_support_mask_and_detects_disjoint() {
        use super::support_bits;
        let vtree = Arc::new(Vtree::balanced(3));
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let x2 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
        let f = and2(&x0, &x2); // depends on {x0, x2}
        // support_bits OVER-approximates support_mask (never drops a real dependency);
        // on a minimized diagram like this it is exact.
        let mask = support_mask(&f);
        let bits = support_bits(&f);
        for (x, &m) in mask.iter().enumerate() {
            let b = (bits[x / 64] >> (x % 64)) & 1 == 1;
            assert!(!m || b, "support_bits must cover support_mask at var {x}");
            assert_eq!(b, m, "support_bits exact on minimized f at var {x}");
        }
        // Disjoint detection: g depends only on x1, sharing no variable with f.
        let g = clause_to_tdd(&vtree, &clause(&[(1, true)]));
        let bg = support_bits(&g);
        assert!(
            bits.iter().zip(bg.iter()).all(|(a, b)| a & b == 0),
            "f={{x0,x2}} and g={{x1}} must be detected disjoint"
        );
        // Overlapping support (shares x0) is NOT flagged disjoint.
        let bx0 = support_bits(&x0);
        assert!(
            !bits.iter().zip(bx0.iter()).all(|(a, b)| a & b == 0),
            "f={{x0,x2}} and x0 share x0 → must NOT be disjoint"
        );
    }

    #[test]
    fn project_var_of_constant_one_is_one() {
        let vtree = Arc::new(Vtree::balanced(3));
        let tdd = constant_one(&vtree);
        let result = project_var(&tdd, VarId(0));
        assert!(!result.is_zero());
        assert_eq!(model_count(&result), BigUint::from(8u32));
    }

    #[test]
    fn project_var_of_constant_zero_is_zero() {
        let vtree = Arc::new(Vtree::balanced(3));
        let tdd = constant_zero(&vtree);
        let result = project_var(&tdd, VarId(0));
        assert!(result.is_zero());
    }

    #[test]
    fn project_var_of_literal_is_one() {
        let vtree = Arc::new(Vtree::balanced(1));
        let tdd = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        assert_eq!(model_count(&tdd), BigUint::from(1u32));

        let result = project_var(&tdd, VarId(0));
        assert!(!result.is_zero());
        assert_eq!(model_count(&result), BigUint::from(2u32));
    }

    // Regression: a unit-forced variable must be detectable by conditioning it to
    // the opposite value and finding the result UNSAT. NOTE: a false diagram is not
    // always `is_zero()` — conditioning or an apply can leave `model_count == 0` in a
    // non-canonical form (output node still has pairs). `condition_*` now
    // canonicalizes its own output, but an unarmed apply does not, so UNSAT detection
    // on a derived diagram uses `model_count == 0`, not `Tdd::is_zero()`. The
    // segment-restrict driver's `forced_literals` depends on this.
    fn count_is_zero(t: &Tdd) -> bool {
        model_count(t) == BigUint::from(0u32)
    }

    #[test]
    fn condition_var_detects_unit_forced_apply() {
        let vtree = Arc::new(Vtree::balanced(3));
        // (x0) AND (x0 v x1) AND (x1 v x2) -- x0 forced TRUE by the unit.
        let mut c0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let mut c1 = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
        let mut c2 = clause_to_tdd(&vtree, &clause(&[(1, true), (2, true)]));
        let mut t01 = apply_and(&mut c0, &mut c1);
        let t = apply_and(&mut t01, &mut c2);
        assert!(!count_is_zero(&t));
        // x0 forced true => x0=false is UNSAT (count 0), x0=true is SAT.
        assert!(count_is_zero(&condition_var(&t, VarId(0), false)));
        assert!(!count_is_zero(&condition_var(&t, VarId(0), true)));
    }

    // A conditioned diagram with no models must be CANONICALLY false: conditioning
    // plus minimize can leave the output node holding pairs whose every path is
    // dead (`model_count == 0`, `is_zero() == false`). Counting that is correct, but
    // re-conjoining it revives the models the restriction killed, so `condition_*`
    // collapses it to ZERO. Same diagram as `condition_var_detects_unit_forced_apply`
    // — the case the pre-fix code left non-canonical.
    #[test]
    fn condition_var_canonicalizes_a_dead_result() {
        let vtree = Arc::new(Vtree::balanced(3));
        let mut c0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let mut c1 = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
        let mut c2 = clause_to_tdd(&vtree, &clause(&[(1, true), (2, true)]));
        let mut t01 = apply_and(&mut c0, &mut c1);
        let t = apply_and(&mut t01, &mut c2);
        let dead = condition_var(&t, VarId(0), false);
        assert!(count_is_zero(&dead), "x0 is forced true, so x0=false has no models");
        assert!(dead.is_zero(), "a model-count-0 conditioning result must be canonically ZERO");
        // Re-conjoining the canonical ⊥ stays ⊥ (the property the canonicalization buys).
        assert!(count_is_zero(&and2(&dead, &t)));
    }

    // Soundness contract: conditioning a leaf whose own level was marginalized must
    // fail fast. `rewrite_for_restrict` matches the target-side ref against
    // POS/NEG/ONE, and leaf-marg rewrites exactly those refs into inline marg counts
    // in the SAME numeric space — so without the guard the variable is silently left
    // unconditioned (miscount, no panic).
    #[test]
    #[should_panic(expected = "leaf level")]
    fn condition_var_on_marginalized_leaf_fails_fast() {
        use crate::tdd::transform::unary::marginalize::marginalize_leaf_inline;
        let vtree = Arc::new(Vtree::balanced(2));
        // x0 XOR x1 — depends on both vars, so the output sits at the root.
        let mut t = and2(
            &clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)])),
            &clause_to_tdd(&vtree, &clause(&[(0, false), (1, false)])),
        );
        let leaf = vtree.leaf_of(VarId(1));
        marginalize_leaf_inline(&mut t, leaf, &vtree);
        assert!(t.levels[leaf.idx()].is_marginal(), "test setup: leaf must be marginal");
        let _ = condition_var(&t, VarId(1), true);
    }

    // Same contract for the leaf's PARENT: a marginal parent holds marg-slot refs
    // (and no `nodes`), so the rewrite would read slot indices as leaf labels and
    // then silently no-op.
    #[test]
    #[should_panic(expected = "parent level")]
    fn condition_var_through_marginal_parent_fails_fast() {
        use crate::tdd::transform::unary::marginalize::marginalize_batch;
        let vtree = Arc::new(Vtree::balanced(2));
        // x0 XOR x1 — depends on both vars, so the output sits at the root.
        let mut t = and2(
            &clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)])),
            &clause_to_tdd(&vtree, &clause(&[(0, false), (1, false)])),
        );
        let leaf = vtree.leaf_of(VarId(0));
        let parent = vtree.node(leaf).parent().expect("leaf has a parent");
        marginalize_batch(&mut t, &[parent], &vtree).expect("no wall is installed in a test");
        assert!(!t.levels[leaf.idx()].is_marginal(), "test setup: only the parent is marginal");
        let _ = condition_var(&t, VarId(0), true);
    }


    // ── restrict (generalized cofactor) ───────────────────────────────────────
    fn and2(a: &Tdd, b: &Tdd) -> Tdd {
        let mut a = a.clone();
        let mut b = b.clone();
        apply_and(&mut a, &mut b)
    }

    // f1 == f2 as Boolean functions over the shared vtree.
    fn equiv(a: &Tdd, b: &Tdd) -> bool {
        use crate::tdd::transform::unary::negate::negate;
        let a_not_b = and2(a, &negate(b));
        let not_a_b = and2(&negate(a), b);
        count_is_zero(&a_not_b) && count_is_zero(&not_a_b)
    }
    // Negate-free equivalence: `a∧b ⊆ a` and `a∧b ⊆ b` always, so equal model
    // counts on all three force `a == b` as sets. Uses only apply_and/model_count
    // (the restrict output is a valid TDD but NOT in `negate`'s t-full/complete
    // form, so the negate-based `equiv` above is the wrong oracle for it).
    fn equiv_nf(a: &Tdd, b: &Tdd) -> bool {
        let ca = model_count(a);
        let cb = model_count(b);
        ca == cb && model_count(&and2(a, b)) == ca
    }

    #[test]
    fn implied_literals_matches_condition_oracle() {
        use super::implied_literals;
        use crate::tdd::minimize::minimize;
        // Oracle: (v, val) is implied iff f is SAT but conditioning v := !val makes
        // it UNSAT — i.e. every model pins v = val.
        let oracle = |f: &Tdd, nvars: u32| -> std::collections::HashSet<(VarId, bool)> {
            let mut out = std::collections::HashSet::new();
            if count_is_zero(f) {
                return out;
            }
            for v in 0..nvars {
                for val in [true, false] {
                    if count_is_zero(&condition_var(f, VarId(v), !val)) {
                        out.insert((VarId(v), val));
                    }
                }
            }
            out
        };
        let vtree = Arc::new(Vtree::balanced(3));
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let nx0 = clause_to_tdd(&vtree, &clause(&[(0, false)]));
        let x1 = clause_to_tdd(&vtree, &clause(&[(1, true)]));
        let or12 = clause_to_tdd(&vtree, &clause(&[(1, true), (2, true)]));

        // f = x0 & (x1 | x2): only x0 is backbone (x1,x2 each stay free).
        let mut f = and2(&x0, &or12);
        minimize(&mut f);
        let bb = implied_literals(&f);
        assert_eq!(bb, oracle(&f, 3));
        assert!(bb.contains(&(VarId(0), true)) && bb.len() == 1);

        // g = ~x0 & x1: x0 forced false, x1 forced true, x2 a pure don't-care (only
        // ever the One leaf) — must NOT appear.
        let mut g = and2(&nx0, &x1);
        minimize(&mut g);
        let bbg = implied_literals(&g);
        assert_eq!(bbg, oracle(&g, 3));
        assert!(!bbg.contains(&(VarId(2), true)) && !bbg.contains(&(VarId(2), false)));

        // UNSAT (x0 & ~x0): no models, no implied literals.
        let mut z = and2(&x0, &nx0);
        minimize(&mut z);
        assert!(implied_literals(&z).is_empty());
    }

    // ── restrict: direct semantics evaluator (apply-independent ground truth) ──
    //
    // Walks the diagram by the TDD denotation `⋃ᵢ aᵢ×bᵢ` and evaluates a single
    // assignment. Independent of apply/model_count, so brute-forcing it over all
    // assignments is a soundness oracle that shares no machinery with the operator
    // OR with `equiv`.
    fn eval_label(l: super::LocalNodeIdx, x: bool) -> bool {
        if l == super::ONE {
            true
        } else if l == super::POS {
            x
        } else if l == super::NEG {
            !x
        } else {
            false // ZERO
        }
    }
    fn eval_node(t: &Tdd, v: crate::vtree::VtreeIdx, local: super::LocalNodeIdx, asn: &[bool]) -> bool {
        match *t.vtree.node(v) {
            crate::vtree::VtreeNode::Leaf { var, .. } => eval_label(local, asn[var.idx()]),
            crate::vtree::VtreeNode::Internal { left, right, .. } => {
                if local == super::ZERO {
                    return false;
                }
                for p in t.levels[v.idx()].pairs_of_idx(local.idx()) {
                    if eval_node(t, left, p.left, asn) && eval_node(t, right, p.right, asn) {
                        return true;
                    }
                }
                false
            }
        }
    }
    fn eval(t: &Tdd, asn: &[bool]) -> bool {
        if t.is_zero() {
            return false;
        }
        eval_node(t, t.output.vtree, t.output.local, asn)
    }

    // All-assignment validity bundle for a restrict result: brute-force soundness
    // against the direct evaluator + every structural invariant + exact determinism
    // + the never-larger gate.
    fn assert_restrict_ok(f: &Tdd, c: &Tdd, nvars: u32) {
        use super::{reachable_pairs, restrict};
        use crate::tdd::validate::{check_all_fast, check_determinism};
        let g = super::restrict(f, c.clone(), super::CareCanonical::No).into_tdd(f);
        let _ = restrict; // (re-export sanity)
        // (1) soundness over the FULL truth table, via the apply-free evaluator.
        for mask in 0..(1u32 << nvars) {
            let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
            let cv = eval(c, &asn);
            assert_eq!(
                eval(&g, &asn) && cv,
                eval(f, &asn) && cv,
                "restrict unsound at assignment {asn:?} (g∧c ≠ f∧c)"
            );
        }
        // (3) validity. restrict returns a sound subgraph of f that production uses
        // RAW — it may carry non-canonical false nodes that minimize removes. Soundness
        // is checked on raw g above; check structure/determinism on the canonical form.
        let mut gm = g.clone();
        crate::tdd::minimize::minimize(&mut gm);
        check_all_fast(&gm, "restrict-output");
        check_determinism(&gm).expect("restrict output must be deterministic (mutex pairs)");
        // (2) never larger than f — restrict returns a strict subgraph of f.
        assert!(
            reachable_pairs(&g) <= reachable_pairs(f),
            "restrict grew the diagram beyond f"
        );
    }

    #[test]
    fn restrict_tautological_care_is_identity() {
        // c = ⊤ pins g everywhere → g must equal f (no don't-cares).
        let vtree = Arc::new(Vtree::balanced(3));
        let x2 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
        let f = apply_or(&and2(&clause_to_tdd(&vtree, &clause(&[(0, true)])),
                               &clause_to_tdd(&vtree, &clause(&[(1, true)]))),
                         &x2);
        let c = constant_one(&vtree);
        let g = super::restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
        assert!(equiv_nf(&g, &f), "restrict(f, ⊤) must equal f");
        assert_restrict_ok(&f, &c, 3);
    }

    #[test]
    fn restrict_false_care_is_empty() {
        // c = ⊥: f∧c = ∅ for any g; restrict returns ⊥, the smallest sound answer.
        let vtree = Arc::new(Vtree::balanced(3));
        let f = apply_or(&clause_to_tdd(&vtree, &clause(&[(0, true)])),
                         &clause_to_tdd(&vtree, &clause(&[(1, true), (2, true)])));
        let c = constant_zero(&vtree);
        let g = super::restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
        assert!(count_is_zero(&g), "restrict(f, ⊥) must be ⊥ (f∧⊥ = ∅)");
        crate::tdd::validate::check_all_fast(&g, "restrict-false-care");
    }

    #[test]
    fn restrict_of_false_is_false() {
        let vtree = Arc::new(Vtree::balanced(3));
        let f = constant_zero(&vtree);
        let c = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let g = super::restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
        assert!(count_is_zero(&g), "restrict(⊥, c) must be ⊥");
        crate::tdd::validate::check_all_fast(&g, "restrict-of-false");
    }

    #[test]
    fn restrict_of_true_is_sound_and_valid() {
        // f = ⊤: g∧c must = c. g = ⊤ is the smallest sound answer.
        let vtree = Arc::new(Vtree::balanced(3));
        let f = constant_one(&vtree);
        let c = apply_or(&clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)])),
                         &clause_to_tdd(&vtree, &clause(&[(2, true)])));
        assert_restrict_ok(&f, &c, 3);
    }

    #[test]
    fn restrict_cube_care_shrinks_or_holds() {
        // f = (x0 ∧ x1) ∨ (¬x0 ∧ x2); care c = x0. On the care, f reduces to x1 and
        // the x2 branch is don't-care — the classic shrink. We assert the contract
        // (sound, never-larger, valid) and brute-force soundness directly.
        let vtree = Arc::new(Vtree::balanced(3));
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let nx0 = clause_to_tdd(&vtree, &clause(&[(0, false)]));
        let x1 = clause_to_tdd(&vtree, &clause(&[(1, true)]));
        let x2 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
        let f = apply_or(&and2(&x0, &x1), &and2(&nx0, &x2));
        let c = x0.clone();
        assert_restrict_ok(&f, &c, 3);
        // The restricted function must agree with x1 on the care set.
        let g = super::restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
        assert!(equiv(&and2(&g, &c), &and2(&x1, &c)));
    }

    #[test]
    fn restrict_drop_lever_sound_and_valid() {
        // f = (x0 ∧ x2) ∨ (x1 ∧ ¬x2); care c = x0. Wherever x0 = 0 the second term is
        // don't-care, so the DROP lever can prune it. Contract + brute force.
        let vtree = Arc::new(Vtree::balanced(3));
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let x1 = clause_to_tdd(&vtree, &clause(&[(1, true)]));
        let x2 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
        let nx2 = clause_to_tdd(&vtree, &clause(&[(2, false)]));
        let f = apply_or(&and2(&x0, &x2), &and2(&x1, &nx2));
        let c = x0;
        assert_restrict_ok(&f, &c, 3);
    }

    #[test]
    fn restrict_drops_dead_pair_of_alive_node() {
        // R3 pair-granular liveness: f = (x0 ∨ x1) has root pairs
        // [(x0,⊤), (¬x0,x1)]; care = (x0 ∨ ¬x1) kills every product of the second
        // pair ((¬x0∧x1)∧care = ∅) while the root NODE stays alive via the first.
        // Node-granular liveness alone would see an all-alive diagram; the
        // pair-granular probe must drop the dead pair: g ≡ x0, strictly smaller, sound.
        let vtree = Arc::new(Vtree::balanced(2));
        let f = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
        let c = clause_to_tdd(&vtree, &clause(&[(0, true), (1, false)]));
        assert_restrict_ok(&f, &c, 2);
        let g = match super::restrict(&f, c.clone(), super::CareCanonical::No) {
            super::Restricted::Shrunk(g) => g,
            super::Restricted::Unchanged => {
                panic!("pair-granular restrict must shrink: pair 2 is dead under care")
            }
            super::Restricted::False(_) => panic!("f∧care is SAT — must not collapse to ⊥"),
        };
        assert!(
            super::reachable_pairs(&g) < super::reachable_pairs(&f),
            "dropping the dead pair must strictly shrink"
        );
        let x0 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        assert!(equiv(&g, &x0), "g must be exactly x0 after the dead pair drops");
    }

    #[test]
    fn restrict_self_care_is_sound() {
        // c = f: g∧f must = f. g is free off f (the bulk of the cube) — a strong
        // don't-care stress, must stay sound and valid.
        let vtree = Arc::new(Vtree::balanced(4));
        let f = apply_or(&and2(&clause_to_tdd(&vtree, &clause(&[(0, true)])),
                               &clause_to_tdd(&vtree, &clause(&[(1, false)]))),
                         &clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)])));
        let c = f.clone();
        assert_restrict_ok(&f, &c, 4);
    }

    #[test]
    fn restrict_runs_on_assorted_small_circuits() {
        // "It runs" + stays valid on a spread of structured functions (xors, chains,
        // wide clauses), each against a couple of cube and non-cube cares.
        let vtree = Arc::new(Vtree::balanced(4));
        let lit = |v: u32, p: bool| clause_to_tdd(&vtree, &clause(&[(v, p)]));
        let xor = |a: u32, b: u32| {
            apply_or(&and2(&lit(a, true), &lit(b, false)), &and2(&lit(a, false), &lit(b, true)))
        };
        let fns = vec![
            xor(0, 1),
            and2(&xor(0, 1), &xor(2, 3)),
            apply_or(&lit(0, true), &and2(&lit(1, true), &lit(2, false))),
            clause_to_tdd(&vtree, &clause(&[(0, true), (1, false), (2, true), (3, true)])),
        ];
        let cares = vec![
            lit(0, true),
            apply_or(&lit(1, true), &lit(2, true)),
            xor(0, 2),
        ];
        for f in &fns {
            for c in &cares {
                if count_is_zero(c) {
                    continue;
                }
                assert_restrict_ok(f, c, 4);
            }
        }
    }

    // Re-home a diagram that depends only on vars under ONE child of its (global)
    // root to be rooted at that child — a genuinely low-rooted Boolean diagram.
    // `build`/`apply` ALWAYS root at the global vtree root, so re-homing is the
    // only way to manufacture the differing-root operand shape a tightly-rooted
    // segment would take. Requires the root level to be a single identity pair
    // (the `g ∧ ⊤` shape a single-region function compiles to — asserted).
    fn reroot_to_child(t: &Tdd, left_child: bool) -> Tdd {
        let root = t.output.vtree;
        let (lc, rc) = match *t.vtree.node(root) {
            crate::vtree::VtreeNode::Internal { left, right, .. } => (left, right),
            crate::vtree::VtreeNode::Leaf { .. } => panic!("reroot_to_child: root must be internal"),
        };
        let pairs = t.levels[root.idx()].pairs_of_idx(t.output.local.idx());
        assert_eq!(pairs.len(), 1, "reroot_to_child expects single-region g∧⊤ shape");
        let p = pairs[0];
        let (child, local) = if left_child { (lc, p.left) } else { (rc, p.right) };
        crate::tdd::types::Tdd::with_levels(
            t.vtree.clone(),
            t.levels.clone(),
            crate::tdd::types::TddNodeId { vtree: child, local },
        )
    }

    // restrict where `f` and `care` root at DIFFERENT vtree nodes (one contained
    // in the other, or disjoint). The un-generalized guard bailed `f.clone()` on
    // any root mismatch, so the productive containment cases below silently
    // returned f unchanged. `assert_restrict_ok` alone can't catch that (g == f is
    // always sound), so each productive case also asserts a STRICT prune — those
    // assertions FAIL on the un-generalized restrict and pass once it lifts the
    // lower operand to the common root.
    //
    // restrict drops dead NODES (then the pairs that point at them), so a real
    // prune needs an INTERNAL node of `f` to become unreachable under `care` — not
    // merely fewer satisfying assignments. That needs vtree depth ≥3, so we use
    // balanced(8): a "selector" `sel = (x0∧(x2∨x3)) ∨ (¬x0∧(x2∧x3))` has TWO
    // distinct nodes at the {2,3} level — forcing x0 makes one of them unreachable.
    // `sel` depends only on {0,2,3} ⊂ block {0..3} = L (a child of the global root
    // R), so it can be re-homed to L, and a care that forces x0 prunes it.
    //
    // Soundness is checked with the apply-free `eval` oracle over the full truth
    // table (the real `g∧c == f∧c` contract) plus the never-larger gate — NOT
    // `assert_restrict_ok`, whose `check_all_fast` enforces the global-root
    // *structural* convention (`validate_vtree_structure`: output.vtree == root).
    // That convention is what makes differing-root restrict a non-event in
    // production — every validated diagram (including marginalized ones, which mark
    // upper levels marginal rather than re-homing the root) is global-rooted. A
    // re-homed low-rooted operand is the not-yet-built "tightly-rooted segment"
    // shape; restrict computes the correct *function* for it (Case B's g is rooted
    // at f's own low node L, outside that structural convention, hence eval-only).
    #[test]
    fn restrict_differing_root_containment_difftest() {
        use super::{reachable_pairs, restrict};
        let vtree = Arc::new(Vtree::balanced(8));
        let lit = |v: u32, p: bool| clause_to_tdd(&vtree, &clause(&[(v, p)]));
        // Function-level soundness oracle: g∧c == f∧c over all 2^nvars assignments,
        // and g never larger than f. Apply-free (shares no machinery with restrict).
        let assert_sound = |f: &Tdd, c: &Tdd, nvars: u32| -> Tdd {
            let g = restrict(f, c.clone(), super::CareCanonical::No).into_tdd(f);
            for mask in 0..(1u32 << nvars) {
                let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                let cv = eval(c, &asn);
                assert_eq!(
                    eval(&g, &asn) && cv,
                    eval(f, &asn) && cv,
                    "restrict unsound at {asn:?} (g∧c ≠ f∧c)"
                );
            }
            assert!(reachable_pairs(&g) <= reachable_pairs(f), "restrict grew beyond f");
            g
        };
        // sel = (x0 ∧ (x2∨x3)) ∨ (¬x0 ∧ (x2∧x3)) — two nodes at {2,3}; forcing x0
        // kills the (x2∧x3) node, forcing ¬x0 kills the (x2∨x3) node.
        let x2or3 = apply_or(&lit(2, true), &lit(3, true));
        let x2and3 = and2(&lit(2, true), &lit(3, true));
        let sel = apply_or(&and2(&lit(0, true), &x2or3), &and2(&lit(0, false), &x2and3));
        // L = {0..3} (left child of the global root R); sel ⊆ L.
        let sel_l = reroot_to_child(&sel, true); // sel rooted at L
        let force_x0_l = reroot_to_child(&and2(&lit(0, true), &lit(1, true)), true); // (x0∧x1) at L

        // ── Case A: care BELOW f (care@L strictly below f@R) ──
        let f_a = sel.clone(); // global root R
        let care_a = force_x0_l.clone(); // (x0∧x1) at L: forces x0=true
        assert_ne!(care_a.output.vtree, f_a.output.vtree, "Case A must be differing-root");
        let g_a = assert_sound(&f_a, &care_a, 8);
        assert!(
            reachable_pairs(&g_a) < reachable_pairs(&f_a),
            "Case A (care below f): forcing x0 makes the (x2∧x3) node unreachable — must strictly prune"
        );

        // ── Case B: f BELOW care (f@L strictly below care@R) ──
        let f_b = sel_l.clone(); // sel rooted at L
        let care_b = and2(&lit(0, true), &lit(4, true)); // (x0∧x4) at R: forces x0=true
        assert_ne!(f_b.output.vtree, care_b.output.vtree, "Case B must be differing-root");
        let g_b = assert_sound(&f_b, &care_b, 8);
        assert!(
            reachable_pairs(&g_b) < reachable_pairs(&f_b),
            "Case B (f below care): forcing x0 makes the (x2∧x3) node unreachable — must strictly prune"
        );

        // ── Case C: disjoint supports, incomparable roots — sound no-op, g == f ──
        let rt = apply_or(&lit(4, true), &lit(5, true)); // (x4∨x5), depends on {4,5} ⊂ Rt
        let f_c = sel_l.clone(); // L = {0..3}
        let care_c = reroot_to_child(&rt, false); // (x4∨x5) at Rt = {4..7}
        assert_ne!(f_c.output.vtree, care_c.output.vtree, "Case C must be differing-root");
        let g_c = assert_sound(&f_c, &care_c, 8);
        assert_eq!(
            reachable_pairs(&g_c),
            reachable_pairs(&f_c),
            "Case C (disjoint): care cannot constrain f — must not prune"
        );
    }

    #[test]
    fn restrict_brute_force_randomized_multi_vtree() {
        // The exhaustive-soundness sweep: random (f, c) over several vtree SIZES, each
        // case checked by the apply-free evaluator over the full truth table PLUS all
        // invariants PLUS exact determinism PLUS never-larger. Small nvars keep the
        // 2^n brute force and the O(width²·apply) determinism check cheap.
        use super::{reachable_pairs, restrict};
        use crate::tdd::validate::{check_all_fast, check_determinism};
        let mut state: u64 = 0xfeed_face_cafe_d00d;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut total = 0;
        let mut shrinks = 0;
        for &nvars in &[2u32, 3, 4] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
                let nclauses = 1 + (rng() % 3) as usize;
                let mut acc: Option<Tdd> = None;
                for _ in 0..nclauses {
                    let width = 1 + (rng() % nvars as u64) as usize;
                    let mut lits: Vec<(u32, bool)> = Vec::new();
                    for _ in 0..width {
                        let v = (rng() % nvars as u64) as u32;
                        let pol = rng() % 2 == 0;
                        if lits.iter().any(|(u, _)| *u == v) {
                            continue;
                        }
                        lits.push((v, pol));
                    }
                    lits.sort_by_key(|&(v, _)| v);
                    lits.dedup_by_key(|&mut (v, _)| v);
                    let cl = clause_to_tdd(&vtree, &clause(&lits));
                    acc = Some(match acc {
                        None => cl,
                        Some(a) => and2(&a, &cl),
                    });
                }
                acc.unwrap()
            };
            for _ in 0..120 {
                let f = rand_fn(&mut rng);
                let c = rand_fn(&mut rng);
                if count_is_zero(&c) {
                    continue;
                }
                // Track the shrink count to keep the "levers inert" guard meaningful.
                let fp = reachable_pairs(&f);
                let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
                // exhaustive truth-table soundness (apply-free)
                for mask in 0..(1u32 << nvars) {
                    let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                    let cv = eval(&c, &asn);
                    assert_eq!(
                        eval(&g, &asn) && cv,
                        eval(&f, &asn) && cv,
                        "unsound: nvars={nvars} asn={asn:?}"
                    );
                }
                let mut gm = g.clone();
                crate::tdd::minimize::minimize(&mut gm);
                check_all_fast(&gm, "restrict-brute");
                check_determinism(&gm).expect("non-deterministic restrict output");
                // restrict returns a strict subgraph of the input f, so never larger
                // than f (un-minimized) — the raw-engine contract, not vs minimize(f).
                let gp = reachable_pairs(&g);
                assert!(gp <= fp, "grew beyond f: {gp} > {fp} (nvars={nvars})");
                if gp < fp {
                    shrinks += 1;
                }
                total += 1;
            }
        }
        assert!(total >= 200, "too few cases exercised: {total}");
        assert!(shrinks > 0, "no shrink across any vtree size — levers inert");
    }

    #[test]
    fn restrict_output_is_orphan_free() {
        // `reduce`/`restrict` must return an ARENA-COMPACT diagram: the rebuild is
        // demand-driven and emits a child before discovering its pair partner
        // collapsed to ZERO, which strands that child (an orphan: reachable_pairs
        // unchanged, but it lingers in the arena). The self-contained reduce prunes
        // its own output, so `size(g) == reachable_pairs(g)` for ANY caller. Bigger
        // vtrees (mixed liveness) are what surface the orphan; this FAILS on the
        // pre-prune engine and passes after. Soundness is asserted alongside so the
        // compactness numbers are trustworthy.
        use super::{reachable_pairs, restrict};
        let mut state: u64 = 0x0123_4567_89ab_cdef;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut total = 0u64;
        let mut shrinks = 0u64;
        for &nvars in &[5u32, 6, 7, 8] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
                let nclauses = 1 + (rng() % 4) as usize;
                let mut acc: Option<Tdd> = None;
                for _ in 0..nclauses {
                    let width = 1 + (rng() % nvars as u64) as usize;
                    let mut lits: Vec<(u32, bool)> = Vec::new();
                    for _ in 0..width {
                        let v = (rng() % nvars as u64) as u32;
                        let pol = rng() % 2 == 0;
                        if lits.iter().any(|(u, _)| *u == v) {
                            continue;
                        }
                        lits.push((v, pol));
                    }
                    lits.sort_by_key(|&(v, _)| v);
                    lits.dedup_by_key(|&mut (v, _)| v);
                    let cl = clause_to_tdd(&vtree, &clause(&lits));
                    acc = Some(match acc {
                        None => cl,
                        Some(a) => and2(&a, &cl),
                    });
                }
                acc.unwrap()
            };
            for _ in 0..150 {
                let mut f = rand_fn(&mut rng);
                let mut c = rand_fn(&mut rng);
                if count_is_zero(&c) {
                    continue;
                }
                // Inputs come from the test's non-minimizing `and2`/`clause_to_tdd`
                // builder and can carry their own orphans; minimize so we test
                // reduce's own compactness contract, not the builder's.
                crate::tdd::minimize::minimize(&mut f);
                crate::tdd::minimize::minimize(&mut c);
                let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
                // exhaustive truth-table soundness (apply-free)
                for mask in 0..(1u32 << nvars) {
                    let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                    let cv = eval(&c, &asn);
                    assert_eq!(
                        eval(&g, &asn) && cv,
                        eval(&f, &asn) && cv,
                        "unsound: nvars={nvars} asn={asn:?}"
                    );
                }
                // The contract: reduce's output is orphan-free — every arena pair
                // is reachable from the root.
                let (arena, reach) = (g.size(), reachable_pairs(&g));
                assert_eq!(
                    arena, reach,
                    "reduce left {} orphan pair(s) in the arena \
                     (nvars={nvars}): size={arena} reachable={reach}",
                    arena - reach
                );
                if reach < reachable_pairs(&f) {
                    shrinks += 1;
                }
                total += 1;
            }
        }
        assert!(total >= 400, "too few cases exercised: {total}");
        assert!(shrinks > 0, "no shrink across any vtree size — levers inert");
    }

    /// Randomized differing-root sweep with marginal-free diagrams: `care`
    /// re-homed strictly below `f`'s root (care ⊂ f's left block) and `f` re-homed
    /// strictly below `care`'s root (f ⊂ care's left block), each checked by the
    /// apply-free truth-table oracle plus never-larger. Complements the hand-built
    /// containment cases of `restrict_differing_root_containment_difftest`; the
    /// `shrinks > 0` guard keeps the sweep from passing on an all-`Unchanged` walk.
    #[test]
    fn restrict_differing_root_randomized() {
        use super::{reachable_pairs, restrict};
        let nvars = 6u32;
        let vtree = Arc::new(Vtree::balanced(nvars));
        // The left block of the global root: balanced(6) puts {0,1,2} under it.
        let left_vars: Vec<u32> = {
            let (lc, _) = match *vtree.node(vtree.root()) {
                crate::vtree::VtreeNode::Internal { left, right, .. } => (left, right),
                _ => unreachable!(),
            };
            (0..nvars)
                .filter(|&v| vtree.lca(vtree.leaf_of(VarId(v)), lc) == lc)
                .collect()
        };
        assert!(left_vars.len() >= 2, "left block too small: {left_vars:?}");
        let mut state: u64 = 0x5eed_0fd1_ff00_7a11;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_over = |rng: &mut dyn FnMut() -> u64, vars: &[u32]| -> Tdd {
            let nclauses = 1 + (rng() % 4) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = vars[(rng() as usize) % vars.len()];
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        // A minimized single-block function has the `g ∧ ⊤` root shape
        // `reroot_to_child` needs; skip the ones that collapsed to a constant.
        let rehome_left = |t: &Tdd| -> Option<Tdd> {
            let mut m = t.clone();
            crate::tdd::minimize::minimize(&mut m);
            if m.is_zero() || m.levels[m.output.vtree.idx()].pairs_of_idx(m.output.local.idx()).len() != 1 {
                return None;
            }
            Some(reroot_to_child(&m, true))
        };
        let all_vars: Vec<u32> = (0..nvars).collect();
        let check = |f: &Tdd, c: &Tdd| -> bool {
            let g = restrict(f, c.clone(), super::CareCanonical::No).into_tdd(f);
            for mask in 0..(1u32 << nvars) {
                let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                let cv = eval(c, &asn);
                assert_eq!(
                    eval(&g, &asn) && cv,
                    eval(f, &asn) && cv,
                    "differing-root restrict unsound at {asn:?} (g∧c ≠ f∧c)"
                );
            }
            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(f));
            assert!(gp <= fp, "restrict grew beyond f: {gp} > {fp}");
            gp < fp
        };
        let (mut total, mut shrinks) = (0u32, 0u32);
        for _ in 0..300 {
            // care strictly below f's root
            let f = rand_over(&mut rng, &all_vars);
            if let Some(care) = rehome_left(&rand_over(&mut rng, &left_vars)) {
                if !f.is_zero() && !count_is_zero(&care) {
                    assert_ne!(care.output.vtree, f.output.vtree);
                    shrinks += check(&f, &care) as u32;
                    total += 1;
                }
            }
            // f strictly below care's root
            let care = rand_over(&mut rng, &all_vars);
            if let Some(f) = rehome_left(&rand_over(&mut rng, &left_vars)) {
                if !f.is_zero() && !count_is_zero(&care) {
                    assert_ne!(care.output.vtree, f.output.vtree);
                    shrinks += check(&f, &care) as u32;
                    total += 1;
                }
            }
        }
        assert!(total >= 100, "too few differing-root cases exercised: {total}");
        assert!(shrinks > 0, "no shrink on any differing-root case — walk inert");
    }

    #[test]
    fn restrict_raw_output_is_apply_safe() {
        // Regression for the WS_FAST_REDUCE panic (prune.rs index-OOB): that lever
        // swaps in the RAW `restrict` output (un-minimized) and then conjoins
        // it — `apply_and(g, other)` followed by the conjoin's `minimize`. Public
        // `restrict` minimizes g first, so the raw-output → apply path is otherwise
        // untested. Assert (A) the raw g is a valid TDD and (B) conjoining it with an
        // arbitrary other member, then minimizing the product, stays valid and never
        // panics — over many random (f, care, other) across vtree sizes.
        use super::restrict;
        use crate::tdd::validate::check_all_fast;
        let mut state: u64 = 0x0bad_f00d_1337_c0de;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut conjoined = 0;
        for &nvars in &[2u32, 3, 4, 5] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
                let nclauses = 1 + (rng() % 4) as usize;
                let mut acc: Option<Tdd> = None;
                for _ in 0..nclauses {
                    let width = 1 + (rng() % nvars as u64) as usize;
                    let mut lits: Vec<(u32, bool)> = Vec::new();
                    for _ in 0..width {
                        let v = (rng() % nvars as u64) as u32;
                        let pol = rng() % 2 == 0;
                        if lits.iter().any(|(u, _)| *u == v) {
                            continue;
                        }
                        lits.push((v, pol));
                    }
                    lits.sort_by_key(|&(v, _)| v);
                    lits.dedup_by_key(|&mut (v, _)| v);
                    let cl = clause_to_tdd(&vtree, &clause(&lits));
                    acc = Some(match acc {
                        None => cl,
                        Some(a) => and2(&a, &cl),
                    });
                }
                acc.unwrap()
            };
            for _ in 0..150 {
                let f = rand_fn(&mut rng);
                let c = rand_fn(&mut rng);
                let other = rand_fn(&mut rng);
                if count_is_zero(&c) || f.is_zero() {
                    continue;
                }
                // (A) raw restrict output must be a valid TDD.
                let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
                check_all_fast(&g, "restrict-raw");
                // (B) the lever's path: conjoin raw g with another member, minimize.
                if other.is_zero() || g.is_zero() {
                    continue;
                }
                let mut ga = g.clone();
                let mut ob = other.clone();
                ga.vtree = f.vtree.clone();
                ob.vtree = f.vtree.clone();
                let mut p = apply_and(&mut ga, &mut ob);
                crate::tdd::minimize::minimize(&mut p);
                check_all_fast(&p, "apply(restrict-raw, other)+minimize");
                conjoined += 1;
            }
        }
        assert!(conjoined >= 100, "too few conjoin cases exercised: {conjoined}");
    }

    #[test]
    fn project_var_of_x_and_y_drops_x() {
        let vtree = Arc::new(Vtree::balanced(2));
        let mut tdd_x = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let mut tdd_y = clause_to_tdd(&vtree, &clause(&[(1, true)]));

        let tdd_xy = apply_and(&mut tdd_x, &mut tdd_y);
        assert_eq!(model_count(&tdd_xy), BigUint::from(1u32));

        let result = project_var(&tdd_xy, VarId(0));
        assert!(!result.is_zero());
        assert_eq!(model_count(&result), BigUint::from(2u32));
    }

    #[test]
    fn project_var_soundness_brute_force() {
        // F = (x ∨ y) ∧ (¬y ∨ z), vars 0=x 1=y 2=z. Project out y.
        let vtree = Arc::new(Vtree::balanced(3));

        let mut tdd1 = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
        let mut tdd2 = clause_to_tdd(&vtree, &clause(&[(1, false), (2, true)]));

        let tdd_f = apply_and(&mut tdd1, &mut tdd2);

        let brute = {
            let mut seen = std::collections::HashSet::new();
            for assign in 0u32..(1u32 << 3) {
                let x = (assign >> 0) & 1 == 1;
                let y = (assign >> 1) & 1 == 1;
                let z = (assign >> 2) & 1 == 1;
                let c1 = x || y;
                let c2 = !y || z;
                if c1 && c2 {
                    seen.insert(assign & !(1u32 << 1));
                }
            }
            seen.len() as u64
        };

        let result = project_var(&tdd_f, VarId(1));

        // The projected TDD still lives in the 3-var vtree. y's leaf becomes
        // "all One", so model_count counts over all 3 bits — each surviving
        // (x,z) pair appears twice (once for y=T, once for y=F).
        let result_count = model_count(&result);
        assert_eq!(
            result_count,
            BigUint::from(brute * 2),
            "brute projected count = {brute}; TDD model_count (3-var) = {result_count}"
        );
    }


    /// Regression for the 0-width marginal crash (production CNF mc2025_track1_189_bva).
    ///
    /// Root cause: `ensure_counts` in `cascade_marginalize` lacks the `width()==0`
    /// guard that `marginalize_batch` has at line 701. When an internal vtree level
    /// has 0 pair nodes, `ensure_counts` computes empty counts → `cascade_marginalize`
    /// calls `make_marginal(vec![], None)` → 0-width marginal. Later, `project_var`
    /// calls `apply_or(pos_cofactor, neg_cofactor)` where both cofactors inherit this
    /// 0-width marginal (the level is disjoint from the projected variable's leaf).
    /// `apply_and` then encounters k1=k2=0 with both levels marginal, which neither
    /// identity fast-path (both require k==1) handles — dense path panics at
    /// `pairs_view_into(0)` on an empty nodes Vec.
    ///
    /// Fix: 0-width marginal fast-path added to `apply_and_fallible` before the
    /// debug-assertions block (tididi/src/tdd/transform/pairwise/conjoin/mod.rs).
    ///
    /// This test constructs the crashing state directly and calls `apply_and`,
    /// because the state is unreachable through the public compile API with a
    /// static vtree: within one `run_marginalize_at` step, projection runs
    /// BEFORE `marginalize_batch`, and the marginal-carrying accumulator only
    /// merges with the other vtree half at their LCA — but any cross-half
    /// clause that schedules the projection trigger at that LCA step also
    /// delays the marginal's creation to the same step, where projection wins.
    /// Production reached the state via mid-compile vtree rotations
    /// (the marginal-cluster rotation pass), which aren't deterministic
    /// enough for a test.
    ///
    /// Construction: balanced(8), c1={var4}, c2={var5} — clauses confined to
    /// the right half, so the left half holds only trivial structure. Mirror
    /// production's marginalized left half in both operands:
    ///   A = Internal(var0,var1) → `make_marginal(vec![], None)` — the 0-width
    ///       orphan, exactly what `cascade_marginalize`/`ensure_counts` emits
    ///       for a 0-node level (it lacks `marginalize_batch`'s width()==0 guard);
    ///   B = Internal(var2,var3) → marginal [4]  (vars 2,3 free);
    ///   C = parent(A,B)         → marginal [16] (vars 0..3 free).
    /// C being marginal is what makes A a true orphan (marginal levels carry
    /// counts, not pair references), matching the production dump where the
    /// 0-width level had no live parents. `apply_and` then hits A with
    /// k1=k2=0, both marginal: without the fix the debug assert (debug builds)
    /// or `pairs_view_into(0)` (release) panics; with it the level passes
    /// through empty and the conjunction's count is unchanged.
    #[test]
    fn apply_and_zero_width_marginal_levels() {
        use crate::vtree::VtreeIdx;

        let vtree = Arc::new(Vtree::balanced(8));
        // Baseline: var4 ∧ var5 over 8 vars = 2^6 models.
        let mut b1 = clause_to_tdd(&vtree, &clause(&[(4, true)]));
        let mut b2 = clause_to_tdd(&vtree, &clause(&[(5, true)]));
        let baseline = model_count(&apply_and(&mut b1, &mut b2));
        assert_eq!(baseline, BigUint::from(64u32));

        // Locate Internal(varX,varY) nodes by their leaf children.
        let leaf_parent = |x: u32, y: u32| {
            (0..vtree.num_nodes() as u32)
                .map(VtreeIdx)
                .find(|&i| {
                    !vtree.node(i).is_leaf() && {
                        let (l, r) = vtree.children(i);
                        matches!(*vtree.node(l), crate::vtree::VtreeNode::Leaf { var, .. } if var == VarId(x))
                            && matches!(*vtree.node(r), crate::vtree::VtreeNode::Leaf { var, .. } if var == VarId(y))
                    }
                })
                .expect("balanced(8) must have this Internal(leaf,leaf) node")
        };
        let a = leaf_parent(0, 1);
        let b = leaf_parent(2, 3);
        let c = vtree.node(a).parent().expect("A has a parent");
        assert_eq!(vtree.node(b).parent(), Some(c), "C must be Internal(A,B)");

        let mut c1 = clause_to_tdd(&vtree, &clause(&[(4, true)]));
        let mut c2 = clause_to_tdd(&vtree, &clause(&[(5, true)]));
        for t in [&mut c1, &mut c2] {
            t.levels[a.idx()].make_marginal(vec![], None); // 0-width orphan
            t.levels[b.idx()].make_marginal(vec![4], None);
            t.levels[c.idx()].make_marginal(vec![16], None);
        }
        assert!(c1.levels[a.idx()].is_marginal() && c1.levels[a.idx()].width() == 0);
        assert!(c2.levels[a.idx()].is_marginal() && c2.levels[a.idx()].width() == 0);

        // Unfixed: panics inside apply_and_fallible at the 0-width marginal level.
        let result = apply_and(&mut c1, &mut c2);
        assert_eq!(
            model_count(&result),
            baseline,
            "orphan 0-width marginal level changed the conjunction's model count"
        );
    }



    // `marginal_constraint_lifted_to_free_indicator` moved to
    // tests/tdd_projection_compile.rs (`marginal_lift_indicator` mod) — it needs
    // CNF parsing, which lives in the CNF front end, and compilation, which
    // lives in the downstream driver crate — neither available here.

    /// REGRESSION: the WS_MARGINALIZE conjoin-loop sum-out can mint a summed-out
    /// leaf whose `marginal_counts` table is EMPTY (`Some(vec![])`) while its
    /// still-NON-marginal parent holds a bare-slot marg-side ref (e.g. slot 2)
    /// into it. `demarginalize_to_indicator`'s `map` closure reads `counts[raw]`
    /// ONLY for a satisfiability sanity-assert — the lift result is count-
    /// INDEPENDENT (always returns 0 == constant_one's true node). An
    /// out-of-range bare-slot ref must therefore NOT panic; it trusts the
    /// minimized-marginal invariant and lifts to a free cube. Pre-fix this
    /// panicked at `counts[raw as usize]` ("index 2 but len is 0").
    #[test]
    fn demarginalize_to_indicator_empty_marginal_counts_no_panic() {
        use super::demarginalize_to_indicator;
        use crate::vtree::VtreeNode;

        // 2-leaf vtree: root internal over two leaf children. constant_one is a
        // valid free cube whose root node is the inline pair {left:0, right:0}.
        let vtree = Arc::new(Vtree::balanced(2));
        let mut r = constant_one(&vtree);
        let root = vtree.root().idx();
        let VtreeNode::Internal { right, .. } = *vtree.node(VtreeIdx(root as u32)) else {
            panic!("balanced(2) root must be internal");
        };
        // Make the right leaf child a SUMMED-OUT marginal level with an EMPTY
        // count table — the orphaned state the WS_MARGINALIZE sum-out produces.
        r.levels[right.idx()].marginal_counts = Some(Vec::new());
        assert!(r.levels[right.idx()].is_marginal(), "right child must be marginal");
        // Point the root node's right (marg-side) ref at a BARE slot 2 — beyond
        // the (empty) table. Bare slot decode: neither bit 31 nor MARG_OVERFLOW_TAG
        // set, mirroring the `map` closure's bare-slot branch.
        assert!(r.levels[root].nodes[0].is_inline(), "root node must be inline");
        r.levels[root].nodes[0].b = 2;

        // Pre-fix: panics at the `counts[raw as usize]` bare-slot read (len 0).
        demarginalize_to_indicator(&mut r);

        // Post-fix: lifted to a satisfiable free cube (non-zero model count).
        assert_ne!(
            model_count(&r),
            BigUint::from(0u32),
            "lifted indicator must be satisfiable (a free cube), not empty"
        );
    }

    /// SOUNDNESS difftest for `restrict` on MARGINALIZED diagrams — the
    /// regime the segment-search fast reduction (`WS_FAST_REDUCE` + `WS_MARGINALIZE`)
    /// actually hits. `f` carries a marginalized bottom subtree (counts summed out);
    /// `care` is fully structural, so `apply_and(f, care)` only ever does the
    /// supported `structural ∧ marginal` (never `marginal ∧ marginal`). The
    /// marg-aware rebuild keeps every marginal level + marg-side ref VERBATIM and
    /// may prune only upper structural nodes that are dead under `care`. Contract:
    /// the marginal model count `#(f∧care)` is preserved bit-exactly. The
    /// `pruned > 0` assert guarantees the marg-aware rebuild branch is genuinely
    /// exercised (not a self-guard / OOM-fallback no-op that would pass trivially).
    #[test]
    fn restrict_marginal_soundness_difftest() {
        use super::{demarginalize_to_indicator, reachable_pairs, restrict};
        use crate::tdd::test_helpers::marginalize_subtree;
        use crate::vtree::{VtreeIdx, VtreeNode};
        let nvars = 6u32;
        let vtree = Arc::new(Vtree::balanced(nvars));
        // A non-root internal subtree → marginalizing it bottom-up yields a marginal
        // subtree under a still-structural root (the boundary the rebuild must handle).
        let marg_root = (0..vtree.num_nodes())
            .find(|&vi| matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) && vi != vtree.root().idx())
            .map(|vi| VtreeIdx(vi as u32))
            .expect("balanced(6) has a non-root internal node");
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
            let nclauses = 1 + (rng() % 5) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        let mut checked = 0;
        let mut pruned = 0;
        for case in 0..600 {
            let f = rand_fn(&mut rng);
            if f.is_zero() {
                continue;
            }
            let mut fm = f.clone();
            marginalize_subtree(&mut fm, marg_root);
            // Production marginal diagrams are canonical apply outputs; marginalize_subtree
            // is not (it hand-builds counts without re-minimizing), so minimize here to
            // match the real segment-search pool members restrict sees.
            crate::tdd::minimize::minimize(&mut fm);
            // Need a surviving internal marginal level, else nothing marg-specific is
            // exercised (skip the cases where the subtree collapsed away).
            let has_marg = (0..vtree.num_nodes()).any(|i| {
                matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && fm.levels[i].is_marginal()
            });
            if !has_marg {
                continue;
            }
            // care is a MARGINALIZED sibling too — exactly the real segment-search call
            // (pool[j] under WS_MARGINALIZE). restrict lifts it internally to
            // a satisfiability indicator; the supported reference product conjoins f
            // against that same indicator (`identity ∧ marginal`, never marginal²).
            let care_raw = rand_fn(&mut rng);
            if count_is_zero(&care_raw) {
                continue;
            }
            let mut care = care_raw.clone();
            marginalize_subtree(&mut care, marg_root);
            crate::tdd::minimize::minimize(&mut care);
            let mut care_ind = care.clone();
            demarginalize_to_indicator(&mut care_ind);
            // The reduction only ENGAGES when the (lifted) care shares f's output root
            // (else the self-guard returns f unchanged — a no-op we skip).
            if care_ind.is_zero() || care_ind.output.vtree != fm.output.vtree {
                continue;
            }
            // Soundness: the marginal count of f ∧ care(indicator) is invariant under
            // the prune — restrict's exact contract on the care it uses.
            let before = model_count(&and2(&fm, &care_ind));
            let g = restrict(&fm, care.clone(), super::CareCanonical::No).into_tdd(&fm);
            let after = model_count(&and2(&g, &care_ind));
            assert_eq!(
                before, after,
                "marg restrict changed #(f∧care) at case {case}: {before} != {after}"
            );
            // g is a strict subgraph of f (structural guarantee, holds with marg levels).
            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
            assert!(gp <= fp, "marg restrict larger than f at case {case}: {gp} > {fp}");
            if gp < fp {
                pruned += 1;
            }
            checked += 1;
        }
        assert!(checked >= 50, "too few marginal cases exercised: {checked}");
        assert!(
            pruned > 0,
            "reduction never pruned a marginal diagram — the marg-aware rebuild branch was untested"
        );
        println!("marg restrict: {pruned}/{checked} pruned, soundness held on all");
    }

    #[test]
    fn restrict_multiregion_marginal_soundness_difftest() {
        // Localization probe (task #11). The single-region marginal contract
        // (restrict_marginal_soundness_difftest) passes, yet the lever
        // empirically panics + miscounts under WS_MARGINALIZE on real CNFs — where
        // repeated conjoin+marginalize leaves MULTIPLE disjoint marginal regions
        // under a structural root. This marginalizes two disjoint non-root subtrees
        // and asserts the same contract: #(f∧care) invariant under the prune, no
        // panic. If this fails, the failure is inside restrict (fast repro
        // for the fix); if it passes, the real failure is downstream (the main
        // segment conjoin of the pruned output) and needs the CNF path.
        use super::{demarginalize_to_indicator, reachable_pairs, restrict};
        use crate::tdd::test_helpers::marginalize_subtree;
        use crate::vtree::{VtreeIdx, VtreeNode};
        let nvars = 8u32;
        let vtree = Arc::new(Vtree::balanced(nvars));
        // Two disjoint non-root internal subtrees (neither an ancestor of the other)
        // → two separate marginal regions under a still-structural root.
        let is_anc = |a: usize, mut b: usize| -> bool {
            while let Some(p) = vtree.node(VtreeIdx(b as u32)).parent() {
                if p.idx() == a {
                    return true;
                }
                b = p.idx();
            }
            false
        };
        let mut marg_roots: Vec<VtreeIdx> = Vec::new();
        for vi in 0..vtree.num_nodes() {
            if vi == vtree.root().idx() || !matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) {
                continue;
            }
            if marg_roots
                .iter()
                .all(|r| !is_anc(r.idx(), vi) && !is_anc(vi, r.idx()))
            {
                marg_roots.push(VtreeIdx(vi as u32));
                if marg_roots.len() == 2 {
                    break;
                }
            }
        }
        assert!(marg_roots.len() == 2, "need two disjoint internal subtrees");
        let mut state: u64 = 0xd1b5_4a32_d192_ed03;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
            let nclauses = 1 + (rng() % 6) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        let marg_all = |t: &mut Tdd| {
            for &r in &marg_roots {
                marginalize_subtree(t, r);
            }
            crate::tdd::minimize::minimize(t);
        };
        let mut checked = 0;
        let mut pruned = 0;
        for case in 0..800 {
            let f = rand_fn(&mut rng);
            if f.is_zero() {
                continue;
            }
            let mut fm = f.clone();
            marg_all(&mut fm);
            let n_marg = (0..vtree.num_nodes())
                .filter(|&i| {
                    matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && fm.levels[i].is_marginal()
                })
                .count();
            if n_marg < 2 {
                continue; // need both regions to survive minimize
            }
            let care_raw = rand_fn(&mut rng);
            if count_is_zero(&care_raw) {
                continue;
            }
            let mut care = care_raw.clone();
            marg_all(&mut care);
            let mut care_ind = care.clone();
            demarginalize_to_indicator(&mut care_ind);
            if care_ind.is_zero() || care_ind.output.vtree != fm.output.vtree {
                continue;
            }
            let before = model_count(&and2(&fm, &care_ind));
            let g = restrict(&fm, care.clone(), super::CareCanonical::No).into_tdd(&fm);
            let after = model_count(&and2(&g, &care_ind));
            assert_eq!(
                before, after,
                "multiregion marg restrict changed #(f∧care) at case {case}: {before} != {after}"
            );
            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
            assert!(gp <= fp, "multiregion marg restrict larger than f at case {case}: {gp} > {fp}");
            if gp < fp {
                pruned += 1;
            }
            checked += 1;
        }
        assert!(checked >= 30, "too few multiregion marginal cases: {checked}");
        println!("multiregion marg restrict: {pruned}/{checked} pruned, soundness held on all");
    }

    /// Restrict contract checked against the TRUE marginal `care`, not the
    /// self-consistent indicator. The two passing difftests above compute
    /// `#(f∧care_ind)` before AND after using the SAME demarginalized indicator on both
    /// sides — tautological (restrict prunes against that very `care_ind`, so
    /// `g∧care_ind == f∧care_ind` by construction), so they cannot catch a WRONG
    /// `care_ind`. This one compares against `care` itself.
    ///
    /// To compare against the TRUE marginal care we must be able to COUNT `f∧care` — but
    /// `marginal²` is unsupported. So keep the regions DISJOINT: `f` is marginal at region
    /// `R_f` and FREE over care's regions; `care` is marginal at TWO disjoint regions and
    /// FREE over `R_f`. Every conjoin is then `identity∧marginal` / `marginal∧identity`,
    /// never `marginal²`, so `model_count(f∧care)` is well-defined via the real apply.
    /// Contract: the marginal `#(f∧care)` is invariant under the prune.
    ///
    /// PASSES on current `restrict` (597/0 across the seed): the contract holds
    /// for DISJOINT multi-region marginal care — `demarginalize_to_indicator` is NOT wrong
    /// here. The production miscount needs OVERLAPPING regions (`f` AND `care` marginal at
    /// the SAME node), where the joint count over the summed region is unrecoverable, so no
    /// pure-`restrict` unit reference exists — that case is hunted at the fold level. This
    /// stays as the guard for the disjoint regime the older difftests left uncovered.
    #[test]
    fn restrict_true_marginal_care_multiregion_difftest() {
        use super::{reachable_pairs, restrict};
        use crate::tdd::test_helpers::marginalize_subtree;
        use crate::vtree::{VtreeIdx, VtreeNode};
        use std::collections::BTreeSet;
        let nvars = 8u32;
        let vtree = Arc::new(Vtree::balanced(nvars));

        // Leaf-var support under an internal subtree root (inclusive).
        let support_of = |root: VtreeIdx| -> BTreeSet<u32> {
            let mut s = BTreeSet::new();
            for vi in 0..vtree.num_nodes() {
                if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(vi as u32)) {
                    let mut cur = VtreeIdx(vi as u32);
                    let mut under = cur == root;
                    while let Some(p) = vtree.node(cur).parent() {
                        if p == root {
                            under = true;
                            break;
                        }
                        cur = p;
                    }
                    if under {
                        s.insert(var.0);
                    }
                }
            }
            s
        };

        // Three pairwise-disjoint non-root internal subtrees, smallest first, leaving at
        // least one shared/live variable outside all three (where restriction can bite).
        let mut internals: Vec<(VtreeIdx, BTreeSet<u32>)> = (0..vtree.num_nodes())
            .filter(|&vi| {
                matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) && vi != vtree.root().idx()
            })
            .map(|vi| {
                let r = VtreeIdx(vi as u32);
                let s = support_of(r);
                (r, s)
            })
            .collect();
        internals.sort_by_key(|(_, s)| s.len());
        let mut chosen: Vec<(VtreeIdx, BTreeSet<u32>)> = Vec::new();
        for (r, s) in internals {
            if chosen.iter().all(|(_, cs)| cs.is_disjoint(&s)) {
                chosen.push((r, s));
                if chosen.len() == 3 {
                    break;
                }
            }
        }
        assert_eq!(chosen.len(), 3, "need three disjoint internal subtrees");
        let (r_f, s_f) = chosen[0].clone();
        let (r_c1, s_c1) = chosen[1].clone();
        let (r_c2, s_c2) = chosen[2].clone();
        let mut union: BTreeSet<u32> = BTreeSet::new();
        union.extend(s_f.iter().cloned());
        union.extend(s_c1.iter().cloned());
        union.extend(s_c2.iter().cloned());
        let live: Vec<u32> = (0..nvars).filter(|v| !union.contains(v)).collect();
        assert!(!live.is_empty(), "need a shared/live variable");

        // f constrains (s_f ∪ live), FREE over care's regions; care constrains
        // (s_c1 ∪ s_c2 ∪ live), FREE over f's region. Disjoint marginal regions ⇒ the
        // conjoin is always identity∧marginal, never marginal².
        let f_vars: Vec<u32> = s_f.iter().chain(live.iter()).cloned().collect();
        let care_vars: Vec<u32> = s_c1
            .iter()
            .chain(s_c2.iter())
            .chain(live.iter())
            .cloned()
            .collect();

        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_over = |rng: &mut dyn FnMut() -> u64, vars: &[u32]| -> Tdd {
            let nclauses = 1 + (rng() % 6) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = vars[(rng() as usize) % vars.len()];
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };

        let mut checked = 0;
        let mut pruned = 0;
        let mut violations = 0;
        let mut first_violation: Option<(usize, String, String)> = None;
        // Fold-step (downstream conjoin+marginalize) repro counters.
        let only_pruned_fold = true;
        let mut fold_count_fail = 0;
        let mut first_fold_fail: Option<String> = None;
        for case in 0..800 {
            let mut fm = rand_over(&mut rng, &f_vars);
            if fm.is_zero() {
                continue;
            }
            marginalize_subtree(&mut fm, r_f);
            crate::tdd::minimize::minimize(&mut fm);

            let mut care = rand_over(&mut rng, &care_vars);
            if count_is_zero(&care) {
                continue;
            }
            marginalize_subtree(&mut care, r_c1);
            marginalize_subtree(&mut care, r_c2);
            crate::tdd::minimize::minimize(&mut care);

            // Need BOTH of care's marginal regions to survive minimize (multi-region).
            let n_marg = (0..vtree.num_nodes())
                .filter(|&i| {
                    matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && care.levels[i].is_marginal()
                })
                .count();
            if n_marg < 2 {
                continue;
            }
            // restrict only engages on a shared function root.
            if care.output.vtree != fm.output.vtree {
                continue;
            }

            // TRUE marginal care on both sides (disjoint regions ⇒ supported conjoin).
            let before = model_count(&and2(&fm, &care));
            let g = restrict(&fm, care.clone(), super::CareCanonical::No).into_tdd(&fm);
            let after = model_count(&and2(&g, &care));
            if before != after {
                violations += 1;
                if first_violation.is_none() {
                    first_violation = Some((case, before.to_string(), after.to_string()));
                }
            }
            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
            assert!(gp <= fp, "restrict larger than f at case {case}: {gp} > {fp}");
            if gp < fp {
                pruned += 1;
            }

            // ── FOLD STEP (task #11 f-side repro) ──────────────────────────────
            // The restrict contract (#(f∧care)) holds above, yet production panics
            // when the SHRUNK operand feeds the fold's apply_and THEN marginalize.
            // Mimic `merge_one_pair`: conjoin g with the care operand, then sum out
            // the now-private `live` vars via the PRODUCTION batch marginalizer, and
            // VALIDATE STRUCTURE (not just the count) at each stage — the dangling
            // marg-side ref the contract checks miss. care∧g == care∧fm (restrict
            // contract), so the post-marginalize counts must match; a structural
            // failure / OOB / mismatch on the g-path (while the fm-path stays clean)
            // localizes the defect to restrict's marginal-f output feeding the fold.
            if only_pruned_fold {
                if gp == fp {
                    checked += 1;
                    continue; // exercise the fold only where restrict actually shrank
                }
                let mut prod_g = and2(&care, &g);
                let mut prod_f = and2(&care, &fm);
                // care∧g == care∧fm, so they share support — filter the live (now
                // private) vars to those actually present, exactly as
                // marginalize_private_vars does (it derives dead vars from support).
                let supp = support_mask(&prod_g);
                let mut targets: Vec<VtreeIdx> = live
                    .iter()
                    .filter(|&&v| supp.get(v as usize).copied().unwrap_or(false))
                    .map(|&v| vtree.leaf_of(VarId(v)))
                    .collect();
                targets.sort_by_key(|vi| vtree.topo_pos(*vi));
                if targets.is_empty() {
                    checked += 1;
                    continue;
                }
                crate::tdd::transform::unary::marginalize::marginalize_batch(&mut prod_g, &targets, &vtree).expect("no wall is installed in a test");
                crate::tdd::transform::unary::marginalize::marginalize_batch(&mut prod_f, &targets, &vtree).expect("no wall is installed in a test");
                // The real production signal: model_count is the query that OOBs
                // (query.rs:607) on the corrupt fold structure, and the count must be
                // invariant (care∧g == care∧fm). A panic here IS the production bug;
                // a mismatch is a silent miscount. (validate_vtree_structure can't be
                // used post-marginalize — its Phase A unreachable!s on legitimate
                // inline marg refs, a false alarm, not corruption.)
                let cg = model_count(&prod_g);
                let cf = model_count(&prod_f);
                if cg != cf {
                    fold_count_fail += 1;
                    if first_fold_fail.is_none() {
                        first_fold_fail = Some(format!("case {case}: fold count {cg} != {cf}"));
                    }
                }
            }
            checked += 1;
        }
        println!(
            "true-marginal-care multiregion: {checked} checked, {pruned} pruned, {violations} violations; first={first_violation:?}"
        );
        println!(
            "  fold-after-conjoin: {fold_count_fail} count-fail; first_fold={first_fold_fail:?}"
        );
        assert_eq!(
            fold_count_fail, 0,
            "fold conjoin+marginalize MISCOUNTED on the restrict-shrunk operand in \
             {fold_count_fail}/{checked} cases (first {first_fold_fail:?})"
        );
        assert!(checked >= 30, "too few multi-region cases exercised: {checked}");
        assert!(
            pruned > 0,
            "reduction never pruned — the path is not exercised (test would pass vacuously)"
        );
        assert_eq!(
            violations, 0,
            "restrict changed #(f∧care) against the TRUE multi-region marginal \
             care in {violations}/{checked} cases (first {first_violation:?}) — the \
             multi-region demarginalize_to_indicator mis-approximates care"
        );
    }

    /// Restrict contract on a MARGINAL `f`, the production orientation the
    /// multi-region test above does NOT exercise (that one marginalizes `care`,
    /// leaving `f` free). The marginalized-pool restrict shrinks members
    /// that have themselves been marginalized — `restrict(f, care)`
    /// with `f` carrying summed-out (marginal) levels and `care` non-marginal —
    /// then conjoins the shrunk `g` with `care`. The contract `g ∧ care ==
    /// f ∧ care` must hold per-MODEL-COUNT for that marginal `f`.
    ///
    /// The liveness oracle inside `restrict` prunes a non-marginal node
    /// of `f` when the EMIT=false conjoin marks it dead. A non-marginal node
    /// routing into a marginal subtree is ALWAYS alive — marginal counts are >0,
    /// so every marginal node is alive. If the oracle killed such a node, `g`
    /// would lose models present in `f ∧ care` → miscount.
    ///
    /// This GUARD asserts no such miscount across synthesized marginal-`f`
    /// configs (random `f` over all vars, a random SCATTERED subset summed out,
    /// `care` over the complement). It PASSES — restrict is sound for every
    /// marginal-`f` shape reachable by this synthesis. The production miscount
    /// (proven on the blow-up instances) needs operand structure this synthesis
    /// does not reach (a large complex `care` against a tiny marginal `f` under
    /// the in-fold vtree graft); reproducing it needs captured real operands,
    /// not synthesis. Kept as the regression guard for the sound regime.
    #[test]
    fn restrict_marginal_f_difftest() {
        use super::{reachable_pairs, restrict};
        use crate::vtree::VtreeIdx;
        let nvars = 8u32;
        let vtree = Arc::new(Vtree::balanced(nvars));

        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_over = |rng: &mut dyn FnMut() -> u64, vars: &[u32]| -> Tdd {
            let nclauses = 1 + (rng() % 6) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = vars[(rng() as usize) % vars.len()];
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };

        let mut checked = 0usize;
        let mut pruned = 0usize;
        let mut fail = 0usize;
        let mut first_fail: Option<String> = None;
        for _trial in 0..600 {
            // f constrains ALL vars; then sum out a RANDOM SCATTERED subset — the
            // production private-var marginalize interleaves marginal and non-marginal
            // levels (unlike a contiguous subtree, where all marginal levels sit at the
            // bottom). That interleaving is what exercises a non-marginal node sitting
            // BELOW a marginal one.
            let all_vars: Vec<u32> = (0..nvars).collect();
            let mut f = rand_over(&mut rng, &all_vars);
            if f.is_zero() {
                continue;
            }
            let marg_vars: Vec<u32> = (0..nvars).filter(|_| rng() % 2 == 0).collect();
            if marg_vars.is_empty() || marg_vars.len() == nvars as usize {
                continue;
            }
            let care_vars: Vec<u32> = (0..nvars).filter(|v| !marg_vars.contains(v)).collect();
            if care_vars.is_empty() {
                continue;
            }
            let mut targets: Vec<VtreeIdx> =
                marg_vars.iter().map(|&v| vtree.leaf_of(VarId(v))).collect();
            targets.sort_by_key(|vi| vtree.topo_pos(*vi));
            crate::tdd::transform::unary::marginalize::marginalize_batch(&mut f, &targets, &vtree).expect("no wall is installed in a test");
            // care constrains only NON-marginal vars ⇒ identity at f's marginal levels,
            // so the conjoin stays legal and #(f∧care) is well-defined.
            let care = rand_over(&mut rng, &care_vars);
            if care.is_zero() {
                continue;
            }
            let prod_f = and2(&f, &care);
            let g = restrict(&f, care.clone(), super::CareCanonical::No).into_tdd(&f);
            if reachable_pairs(&g) < reachable_pairs(&f) {
                pruned += 1;
            }
            let prod_g = and2(&g, &care);
            let cf = model_count(&prod_f);
            let cg = model_count(&prod_g);
            if cf != cg {
                fail += 1;
                if first_fail.is_none() {
                    first_fail = Some(format!(
                        "marg_vars={marg_vars:?} care_vars={care_vars:?}: \
                         #(f∧care)={cf} != #(g∧care)={cg}"
                    ));
                }
            }
            checked += 1;
        }
        println!(
            "marginal-f restrict (scattered): {checked} checked, {pruned} pruned, {fail} miscount; first={first_fail:?}"
        );
        assert!(checked >= 30, "too few marginal-f cases exercised: {checked}");
        assert!(
            pruned > 0,
            "reduction never pruned a marginal f — path not exercised (test would pass vacuously)"
        );
        assert_eq!(
            fail, 0,
            "restrict MISCOUNTED #(f∧care) on a MARGINAL f in {fail}/{checked} \
             cases (first {first_fail:?}) — the liveness oracle killed a non-marginal node \
             that routes into an (always-alive) marginal subtree"
        );
    }

    /// P4 soundness gate. The ancestor-down-restriction prototype restricts a completed
    /// bottom-up accumulator under cares built from pending ancestor clauses. That
    /// operand has the VANILLA-COMPILE marginal shape, which differs from the
    /// task-#11 miscount shape (a marginal operand under the in-fold vtree GRAFT,
    /// where operand and care carry different marginal-level patterns over shared
    /// structure): here `b` is a bottom-up accumulator carrying marginal levels
    /// from DESCENDANT forgets — V2 summed out as a CONTIGUOUS subtree (marginal
    /// levels at the bottom), forgotten with the production `marginalize_batch` —
    /// and `care` is a NON-marginal TDD built purely from clauses over V1, whose
    /// support is DISJOINT from the forgotten V2 (a pending ancestor clause can
    /// never mention a var already forgotten below). Both share the global root and
    /// the SAME vtree `Arc` — no graft, so restrict takes its same-root fast path
    /// (never the marginal-lift fallback).
    ///
    /// Complements `restrict_marginal_f_difftest` (which forgets a SCATTERED subset,
    /// interleaving marginal/non-marginal levels): this one pins the contiguous
    /// descendant-forget shape P4 actually feeds restrict, and it counts the
    /// `Restricted::Shrunk` variant directly (not just a reachable-pair drop) plus
    /// deterministic contradiction cases, so it can never pass vacuously.
    ///
    /// Contract: `model_count(restrict(b,care) ∧ care) == model_count(b ∧ care)` —
    /// the exact invariant P4 relies on to down-restrict an accumulator in place.
    /// (`model_count` on a marginal TDD returns the summed count; that is precisely
    /// the semantics that must be preserved.)
    #[test]
    fn restrict_ancestor_marginal_operand_gate() {
        use super::{reachable_pairs, restrict, CareCanonical, Restricted};
        use crate::vtree::{VtreeIdx, VtreeNode};
        let nvars = 8u32;
        let vtree = Arc::new(Vtree::balanced(nvars));

        // Leaf-var support under an internal subtree root (inclusive).
        let support_of = |root: VtreeIdx| -> Vec<u32> {
            let mut s = Vec::new();
            for vi in 0..vtree.num_nodes() {
                if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(vi as u32)) {
                    let mut cur = VtreeIdx(vi as u32);
                    let mut under = cur == root;
                    while let Some(p) = vtree.node(cur).parent() {
                        if p == root {
                            under = true;
                            break;
                        }
                        cur = p;
                    }
                    if under {
                        s.push(var.0);
                    }
                }
            }
            s
        };
        // V2 = a small non-root internal subtree (contiguous vars, forgotten
        // bottom-up); V1 = the complement, leaving ≥2 vars for care to bite on.
        let marg_root = (0..vtree.num_nodes())
            .filter(|&vi| {
                matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) && vi != vtree.root().idx()
            })
            .map(|vi| VtreeIdx(vi as u32))
            .find(|&r| {
                let n = support_of(r).len();
                n >= 1 && (nvars as usize - n) >= 2
            })
            .expect("balanced(8) has a small non-root internal subtree");
        let v2: Vec<u32> = support_of(marg_root);
        let v1: Vec<u32> = (0..nvars).filter(|v| !v2.contains(v)).collect();
        let mut v2_targets: Vec<VtreeIdx> =
            v2.iter().map(|&v| vtree.leaf_of(VarId(v))).collect();
        v2_targets.sort_by_key(|vi| vtree.topo_pos(*vi));

        let mut state: u64 = 0xa5a5_5a5a_1234_9e37;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rand_over = |rng: &mut dyn FnMut() -> u64, vars: &[u32], maxc: u64| -> Tdd {
            let nclauses = 1 + (rng() % maxc) as usize;
            let mut acc: Option<Tdd> = None;
            for _ in 0..nclauses {
                let width = 1 + (rng() % 3) as usize;
                let mut lits: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = vars[(rng() as usize) % vars.len()];
                    let pol = rng() % 2 == 0;
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(&vtree, &clause(&lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        // Forget V2 (contiguous subtree) via the PRODUCTION batch marginalizer — the
        // same descendant-forget the bottom-up compile does; leaves marginal levels at
        // the subtree, non-marginal V1 structure above.
        let forget_v2 = |t: &mut Tdd| {
            crate::tdd::transform::unary::marginalize::marginalize_batch(t, &v2_targets, &vtree).expect("no wall is installed in a test");
        };
        // The production `marginalize_batch` marks the forgotten LEAF levels marginal
        // (a contiguous subtree summed out ⇒ its leaf levels carry the marginal counts),
        // so check any level, not just internal ones.
        let has_marg = |t: &Tdd| -> bool {
            (0..vtree.num_nodes()).any(|i| t.levels[i].is_marginal())
        };

        let mut checked = 0usize;
        let mut shrunk = 0usize; // Restricted::Shrunk outcomes (non-vacuity)
        let mut false_out = 0usize; // Restricted::False outcomes (care ⇒ ⊥)
        let mut fail = 0usize;
        let mut first_fail: Option<String> = None;

        // One soundness + non-vacuity check on a (b, care) pair.
        let mut check =
            |b: &Tdd, care: &Tdd, label: &str, fail: &mut usize, first_fail: &mut Option<String>| {
                assert_eq!(
                    care.output.vtree, b.output.vtree,
                    "{label}: care/b must share the global root (no graft)"
                );
                let before = model_count(&and2(b, care));
                let out = restrict(b, care.clone(), CareCanonical::No);
                match out {
                    Restricted::Shrunk(_) => shrunk += 1,
                    Restricted::False(_) => false_out += 1,
                    Restricted::Unchanged => {}
                }
                let g = out.into_tdd(b);
                let after = model_count(&and2(&g, care));
                if before != after {
                    *fail += 1;
                    if first_fail.is_none() {
                        *first_fail = Some(format!(
                            "{label}: V2={v2:?} #(b∧care)={before} != #(g∧care)={after}"
                        ));
                    }
                }
                assert!(
                    reachable_pairs(&g) <= reachable_pairs(b),
                    "{label}: g larger than b"
                );
                checked += 1;
            };

        // ── Deterministic cases: care contradicts part of b's V1 structure so
        // restrict provably shrinks (guards against an all-Unchanged vacuous pass).
        // b must ENTANGLE V1 and V2 (clauses mixing both) — else the forgotten V2
        // factors out as a scalar and minimize strips the marginal level, which is
        // NOT the production shape. p,q ∈ V1; z,z2 ∈ V2 keep the marginal level live. ──
        let p = v1[0];
        let q = v1[1];
        let z = v2[0];
        let z2 = *v2.last().unwrap();
        {
            // b = (p∨q) ∧ (p∨z) ∧ (q∨z2) ; forgetting V2 keeps (p,q) entangled with a
            // surviving marginal level. care=(¬p) forces p=false ⇒ prunes the p-branch.
            let c1 = clause_to_tdd(&vtree, &clause(&[(p, true), (q, true)]));
            let c2 = clause_to_tdd(&vtree, &clause(&[(p, true), (z, true)]));
            let c3 = clause_to_tdd(&vtree, &clause(&[(q, true), (z2, true)]));
            let mut b = and2(&and2(&c1, &c2), &c3);
            forget_v2(&mut b);
            assert!(has_marg(&b), "deterministic case 1 lost its marginal level");
            let care = clause_to_tdd(&vtree, &clause(&[(p, false)])); // ¬p
            check(&b, &care, "det1", &mut fail, &mut first_fail);
        }
        {
            // b = (¬p∨q) ∧ (p∨z) ∧ (q∨z2) ; care=(¬q) forces q=false ⇒ ¬p, prunes branches.
            let c1 = clause_to_tdd(&vtree, &clause(&[(p, false), (q, true)]));
            let c2 = clause_to_tdd(&vtree, &clause(&[(p, true), (z, true)]));
            let c3 = clause_to_tdd(&vtree, &clause(&[(q, true), (z2, true)]));
            let mut b = and2(&and2(&c1, &c2), &c3);
            forget_v2(&mut b);
            assert!(has_marg(&b), "deterministic case 2 lost its marginal level");
            let care = clause_to_tdd(&vtree, &clause(&[(q, false)])); // ¬q
            check(&b, &care, "det2", &mut fail, &mut first_fail);
        }

        // ── Randomized cases (≥50 checked across seeds) ──────────────────────────
        let all_vars: Vec<u32> = (0..nvars).collect();
        for _ in 0..500 {
            let mut b = rand_over(&mut rng, &all_vars, 6);
            if b.is_zero() {
                continue;
            }
            forget_v2(&mut b);
            if !has_marg(&b) {
                continue; // need a surviving marginal level to exercise the shape
            }
            let care = rand_over(&mut rng, &v1, 4);
            if care.is_zero() {
                continue;
            }
            check(&b, &care, "rand", &mut fail, &mut first_fail);
        }

        println!(
            "ancestor-marginal gate: {checked} checked, {shrunk} shrunk, {false_out} false, {fail} miscount; first={first_fail:?}"
        );
        assert!(checked >= 50, "too few gate cases exercised: {checked}");
        assert!(
            shrunk > 0,
            "restrict never produced a Shrunk outcome — the gate would pass vacuously \
             (all-Unchanged), so it proves nothing about the shrink path"
        );
        assert_eq!(
            fail, 0,
            "restrict MISCOUNTED #(b∧care) on the vanilla-compile marginal-operand \
             shape in {fail}/{checked} cases (first {first_fail:?}) — P4's down-restriction \
             invariant is UNSOUND here; STOP and coordinate the fix"
        );
    }

    /// Guard for the always-on marginalize-schedule invariant in `apply_and`
    /// (`conjoin/mod.rs`): conjoining a TDD that has marginalized a vtree node with
    /// one that still constrains a variable under that node is INVALID. Before the guard
    /// this dereferenced a bad marg-side reference and SIGSEGV'd in release (the debug
    /// assert that should have caught it had been compiled out); now it must panic
    /// cleanly so a marginalize-schedule bug surfaces loudly instead of corrupting the
    /// model count.
    ///
    /// Minimal hand-checkable case (6-var balanced vtree): `fm` = f with vtree node 7's
    /// subtree (vars {4,5}) marginalized via `marginalize_subtree` (production-faithful:
    /// mirrors `marginalize_batch`+`cascade_marginalize`, tags marg-side slots). `partner`
    /// still references x5, so `and2(partner, fm)` is the invalid conjoin and must be
    /// rejected. (In a correct run the schedule only marginalizes PRIVATE vars — vars no
    /// partner references — so this never arises; the test deliberately constructs it.)
    #[test]
    fn apply_and_rejects_marginalize_schedule_violation() {
        use crate::tdd::test_helpers::marginalize_subtree;
        use crate::vtree::VtreeIdx;
        let nvars = 6u32;
        let vtree = Arc::new(Vtree::balanced(nvars));
        let build = |cls: &[&[(u32, bool)]]| -> Tdd {
            let mut acc: Option<Tdd> = None;
            for lits in cls {
                let cl = clause_to_tdd(&vtree, &clause(lits));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        let ra = VtreeIdx(7);
        let f_raw = build(&[
            &[(0, false), (3, false), (5, true)],
            &[(0, true), (3, true)],
            &[(1, true), (4, false)],
        ]);
        let partner = build(&[
            &[(1, true), (5, true)],
            &[(0, true), (5, false)],
            &[(0, true), (2, true), (3, false)],
            &[(3, false)],
        ]);

        let mut fm = f_raw.clone();
        marginalize_subtree(&mut fm, ra);
        crate::tdd::minimize::minimize(&mut fm);
        let has_marg = (0..vtree.num_nodes())
            .any(|i| fm.levels[i].is_marginal());
        // Which variables does marginalizing node `ra` sum out (the leaves under ra)?
        // Production only marginalizes PRIVATE vars — vars no partner references. If any
        // of these is in partner's support, this is the de-marginalize-a-needed-var case,
        // which production does not create.
        let mut marg_vars: Vec<u32> = Vec::new();
        for vi in 0..vtree.num_nodes() {
            if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(vi as u32)) {
                let mut cur = VtreeIdx(vi as u32);
                let mut under = cur == ra;
                while let Some(p) = vtree.node(cur).parent() {
                    if p == ra {
                        under = true;
                        break;
                    }
                    cur = p;
                }
                if under {
                    marg_vars.push(var.0);
                }
            }
        }
        marg_vars.sort_unstable();
        let partner_support: std::collections::BTreeSet<u32> =
            [0u32, 1, 2, 3, 5].into_iter().collect();
        let overlap: Vec<u32> = marg_vars
            .iter()
            .copied()
            .filter(|v| partner_support.contains(v))
            .collect();
        println!(
            "MINIMAL setup: f.size={} fm.size={} fm_has_marg={} partner.size={} same_vtree={}",
            f_raw.size(),
            fm.size(),
            has_marg,
            partner.size(),
            partner.output.vtree == fm.output.vtree,
        );
        println!(
            "MINIMAL marg_vars(under node {})={:?} partner_support={:?} OVERLAP={:?}",
            ra.idx(),
            marg_vars,
            partner_support,
            overlap,
        );

        assert!(!overlap.is_empty(), "test fixture must marginalize a var partner uses");

        // The invalid conjoin: partner constrains x5, but fm summed x5 out. Before the
        // always-on marginalize-schedule guard this SIGSEGV'd (or silently miscounted)
        // in the apply's dense path. With the guard it must PANIC cleanly — catchable,
        // diagnostic, never a silent wrong answer. Silence the hook so the expected
        // panic's backtrace doesn't spam test output.
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| and2(&partner, &fm)));
        let _ = std::panic::take_hook();
        // An error MUST be thrown (the conjoin must not silently return a value).
        let payload = res.expect_err(
            "apply_and must REJECT conjoining a marginalized operand with one that still \
             constrains the summed-out var, but the conjoin returned a value",
        );
        // And it must be OUR invariant violation, not some incidental panic.
        let msg = payload
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        // apply_and may reject this invalid conjoin at either of two equivalent
        // marginal-conjoin guards: the deep "marginalize-schedule violation"
        // (conjoin/mod.rs:2587) or the earlier structural "Marginal pair
        // structure cannot conjoin with a non-trivial operand" sanity block
        // (conjoin/mod.rs:2166-2184), which catches this fixture first. Both
        // enforce the same property — a marginalized operand must not conjoin
        // with one still constraining the summed-out var — so accept either.
        assert!(
            msg.contains("marginalize-schedule violation")
                || msg.contains("Marginal pair structure cannot conjoin"),
            "expected apply_and to reject the marginal×constraining conjoin \
             (marg_vars={marg_vars:?}, overlap={overlap:?}), got a different panic: {msg:?}"
        );
    }

    // `vnode_split_with_marginal_child_is_sound` moved to
    // tests/tdd_projection_compile.rs (`vnode_split_marginal_child` mod) — it
    // needs CNF parsing, which lives in the CNF front end, and compilation,
    // which lives in the downstream driver crate — neither available here.


    // `streaming_projected_matches_brute_force_pmc` and
    // `streaming_projected_free_and_empty_vars` moved to
    // tests/tdd_projection_compile.rs (`streaming_pmc_component_spec` /
    // `streaming_pmc_free_and_empty_vars` mods) — they need CNF parsing and
    // preprocessing, which live in the CNF front end, and compilation, which
    // lives in the downstream driver crate — none of it available here.

    // ── brute-force PROJECTED-model-counting (PMC) oracle ────────────────────
    //
    // Guards the soundness identity used by the projected-count path:
    //
    //   PMC = model_count(project_vars(f, projected∩vtree)) >> |projected∩vtree|
    //
    // where `f = compile_cnf(formula, vtree)`, the projection is the real
    // OR-cofactor `project_vars` (NOT a leaf-2^k shortcut), and `free show vars`
    // (show vars absent from every clause / vtree) are zero here because we use
    // `Vtree::balanced(n)`, which places ALL n vars in the vtree. Hence
    // `projected∩vtree` = all non-show vars and `free show vars` = 0.

    use crate::tdd::types::Tdd;


    // ── project_var_scoped: direct unit tests ────────────────────────────────

    #[test]
    fn scoped_constant_one_is_one() {
        let vtree = Arc::new(Vtree::balanced(3));
        let tdd = constant_one(&vtree);
        let r = project_var_scoped(&tdd, VarId(0));
        assert!(!r.is_zero());
        assert_eq!(model_count(&r), BigUint::from(8u32));
    }

    #[test]
    fn scoped_constant_zero_is_zero() {
        let vtree = Arc::new(Vtree::balanced(3));
        let tdd = constant_zero(&vtree);
        let r = project_var_scoped(&tdd, VarId(0));
        assert!(r.is_zero());
    }

    #[test]
    fn scoped_single_literal_is_one() {
        let vtree = Arc::new(Vtree::balanced(1));
        let tdd = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let r = project_var_scoped(&tdd, VarId(0));
        assert!(!r.is_zero());
        assert_eq!(model_count(&r), BigUint::from(2u32));
    }

    #[test]
    fn scoped_x_and_y_drops_x() {
        let vtree = Arc::new(Vtree::balanced(2));
        let mut tx = clause_to_tdd(&vtree, &clause(&[(0, true)]));
        let mut ty = clause_to_tdd(&vtree, &clause(&[(1, true)]));
        let txy = apply_and(&mut tx, &mut ty);
        let r = project_var_scoped(&txy, VarId(0));
        assert_eq!(model_count(&r), BigUint::from(2u32));
    }




    /// Marginal-sibling case: `project_var` panics here; `project_var_scoped`
    /// must succeed and give the correct count.
    ///
    /// Build F over balanced(4) with two FULLY DISJOINT halves:
    ///   left half (a,b): {a∨b}      (3 models over {a,b})
    ///   right half (x,w): {x∨w}     (3 models over {x,w})
    /// so F = (a∨b) ∧ (x∨w), 9 models. Marginalize the LEFT half (a,b) — disjoint
    /// from x's leaf-to-root path — into a real width>1 marginal carrying the
    /// left subtree's per-node counts. Correctly marginalizing a disjoint
    /// subtree preserves the model count, so `project_var_scoped` on the
    /// marginalized TDD must equal `project_var` on the non-marginal TDD.
    #[test]
    fn scoped_marginal_sibling_succeeds() {
        use crate::vtree::{VtreeIdx, VtreeNode};

        let vtree = Arc::new(Vtree::balanced(4));
        let mut t1 = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)])); // a∨b
        let mut t2 = clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)])); // x∨w
        let f = apply_and(&mut t1, &mut t2);
        assert_eq!(model_count(&f), BigUint::from(9u32));

        // Reference: project x on the non-marginal TDD.
        let ref_count = model_count(&project_var(&f, VarId(2)));

        // Find Internal(a,b) — the root's left child, disjoint from x's path.
        let ab = (0..vtree.num_nodes() as u32)
            .map(VtreeIdx)
            .find(|&i| {
                !vtree.node(i).is_leaf() && {
                    let (l, r) = vtree.children(i);
                    matches!(*vtree.node(l), VtreeNode::Leaf { var, .. } if var == VarId(0))
                        && matches!(*vtree.node(r), VtreeNode::Leaf { var, .. } if var == VarId(1))
                }
            })
            .expect("balanced(4) has Internal(a,b)");

        // Marginalize the disjoint left subtree (a,b) via the test helper, which
        // installs the correct per-node counts. This produces a real width>1
        // marginal sibling on x's path (the root's left child) — exactly the
        // shape that crashes `project_var`'s cofactor-OR.
        let mut fm = f.clone();
        crate::tdd::test_helpers::marginalize_subtree(&mut fm, ab);
        assert!(fm.levels[ab.idx()].is_marginal());
        assert!(fm.levels[ab.idx()].width() > 0);
        // Marginalizing a disjoint subtree preserves the model count.
        assert_eq!(model_count(&fm), BigUint::from(9u32));

        // MUST NOT panic crossing the marginal sibling, and MUST match the count.
        let g = project_var_scoped(&fm, VarId(2));
        assert_eq!(
            model_count(&g),
            ref_count,
            "scoped projection over marginal sibling gave wrong count"
        );
    }

    /// Regression for the path-side `One` reference at an INTERNAL ancestor level.
    ///
    /// `regroup_internal` indexes `child_remap[path_child.idx()]`. On internal
    /// levels `LocalNodeIdx(0)` is the constant-true (One) representative, so a
    /// pair whose path-side (x's subtree) is UNCONSTRAINED references it as One.
    /// This test forces exactly that shape to confirm the scoped forget handles a
    /// path-side One ref at the root (not just at the leaf-parent).
    ///
    /// F = (v0∨v3) ∧ (v2∨v3) = (v0∧v2) ∨ v3 over balanced(4) [L=(v0,v1), R=(v2,v3)].
    /// When v3=1, F=1 regardless of the left half (v0,v1) → the root has a branch
    /// where the left (path) subtree is free = a path-side One ref. Project v0:
    /// ∃v0.F = v2∨v3; with v1 free and v0's leaf→One, model_count = 6·2 = 12, and
    /// PMC onto {v1,v2,v3} = 6.
    #[test]
    fn scoped_path_side_one_ref_at_root() {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut t1 = clause_to_tdd(&vtree, &clause(&[(0, true), (3, true)])); // v0∨v3
        let mut t2 = clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)])); // v2∨v3
        let f = apply_and(&mut t1, &mut t2);

        let g_scoped = project_var_scoped(&f, VarId(0));
        let g_ref = project_var(&f, VarId(0));
        assert_eq!(
            model_count(&g_scoped),
            model_count(&g_ref),
            "scoped != cofactor projecting v0 from (v0∨v3)∧(v2∨v3)"
        );
        assert_eq!(model_count(&g_scoped), BigUint::from(12u32));
        crate::tdd::validate::check_determinism(&g_scoped).unwrap();

        // PMC onto show={v1,v2,v3}: project v0, >>1, vs brute force.
        let clauses = vec![vec![1, 4], vec![3, 4]]; // DIMACS 1-indexed: (v0∨v3)∧(v2∨v3)
        let pmc = model_count(&project_vars_scoped(&f, &[VarId(0)])) >> 1usize;
        assert_eq!(pmc, brute_force_pmc(&clauses, 4, &[1, 2, 3]));
    }

    /// Brute-force PMC: number of DISTINCT projections (onto `show`) of the
    /// satisfying assignments of `clauses` over `n` 0-indexed variables.
    ///
    /// `clauses` are DIMACS-style (1-indexed literals); `show` lists 0-indexed
    /// "show" variables. Enumerates all `2^n` assignments, and for each that
    /// satisfies every clause records the tuple of values restricted to `show`
    /// in a `HashSet`; returns the set size. This is exactly PMC.
    fn brute_force_pmc(clauses: &[Vec<i32>], n: usize, show: &[usize]) -> BigUint {
        let mut seen: std::collections::HashSet<Vec<bool>> = std::collections::HashSet::new();
        for mask in 0u32..(1u32 << n) {
            let val = |i: usize| (mask >> i) & 1 == 1;
            let satisfied = clauses.iter().all(|clause| {
                clause.iter().any(|&lit| {
                    let var = (lit.unsigned_abs() as usize) - 1;
                    if lit > 0 { val(var) } else { !val(var) }
                })
            });
            if !satisfied {
                continue;
            }
            let proj: Vec<bool> = show.iter().map(|&v| val(v)).collect();
            seen.insert(proj);
        }
        BigUint::from(seen.len())
    }

















    // `driver_pmc`/`assert_driver_pmc`/`driver_pmc_forced_show_var`/
    // `driver_pmc_free_vs_forced_show`/`driver_pmc_unsat` moved to
    // tests/tdd_projection_compile.rs (`driver_pmc_bcp` mod, which carries its
    // own duplicated `brute_force_pmc` helper) — they need CNF parsing and
    // preprocessing, which live in the CNF front end, and the CLI and
    // compilation, which live in the downstream driver crate.


    // Pair count at a TDD's root (output) node — the "width" the worked example tracks.
    fn root_width(t: &Tdd) -> usize {
        if t.is_zero() {
            return 0;
        }
        t.levels[t.output.vtree.0 as usize].pair_count_at(t.output.local.0 as usize)
    }

    // A cube (conjunction of literals) as a TDD.
    fn cube(vtree: &Arc<Vtree>, lits: &[(u32, bool)]) -> Tdd {
        let mut acc = clause_to_tdd(vtree, &clause(&[lits[0]]));
        for &l in &lits[1..] {
            acc = and2(&acc, &clause_to_tdd(vtree, &clause(&[l])));
        }
        acc
    }

    #[test]
    #[ignore = "scaling probe: run via --ignored --nocapture to locate the wide-node wall"]
    fn restrict_scaling_wide_node() {
        // Worst-case stress: a single very wide root node. f = AND_i (x_i == x_{k+i})
        // over balanced(2k) — the root pairs each left-half value with its unique
        // matching right-half value, so root width = 2^k. This isolates the two
        // suspected scaling terms: lefts_disjoint is O(width^2) conj_empty calls, and
        // the care-set recursion fans out down the right subtree.
        use super::restrict;
        use std::time::Instant;
        println!("\n{:>4} {:>10} {:>12} {:>14}", "k", "rootW", "totalPairs", "restrict_ms");
        for k in [4u32, 6, 8, 10, 12, 14, 15, 16, 17, 18] {
            let n = 2 * k;
            let vtree = Arc::new(Vtree::balanced(n));
            // f = AND_i (x_i <-> x_{k+i}); each equiv is two clauses.
            let mut f: Option<Tdd> = None;
            for i in 0..k {
                let (a, b) = (i, k + i);
                let e1 = clause_to_tdd(&vtree, &clause(&[(a, true), (b, false)]));
                let e2 = clause_to_tdd(&vtree, &clause(&[(a, false), (b, true)]));
                let eq = and2(&e1, &e2);
                f = Some(match f {
                    None => eq,
                    Some(prev) => and2(&prev, &eq),
                });
            }
            let f = f.unwrap();
            // care = one clause spanning both halves (roots at the root node).
            let c = clause_to_tdd(&vtree, &clause(&[(0, true), (k, true)]));
            // Width/pair stats taken on f directly — no extra minimize pass (it's an
            // O(width) cost that would dominate the budget at high k, unrelated to restrict).
            let width = root_width(&f);
            let pairs = super::reachable_pairs(&f);
            let t0 = Instant::now();
            let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            // Soundness is exhaustively covered by restrict_heavy_correctness (1788 cases);
            // here we only spot-check the cheap low-k rows. The expensive part of the equiv
            // check is and2(_,c) at full width, which would swamp the restrict timing at high k.
            if k <= 14 {
                assert!(equiv(&and2(&g, &c), &and2(&f, &c)), "unsound at k={k}");
            }
            println!("{:>4} {:>10} {:>12} {:>14.2}", k, width, pairs, ms);
        }
        // Observed: restrict_ms ~2x per +1 k (pairs also 2x) => linear in pairs,
        // flat ~0.4us/pair through ~789k pairs (k=18). No O(width^2) term materializes
        // for large-f/small-care restriction.
        println!("(restrict_ms ~2x per +1 k tracks pair count => linear, ~0.4us/pair)");
    }

    #[test]
    #[ignore = "scaling probe: realistic large DNF TDD; --ignored --nocapture"]
    fn restrict_scaling_real_dnf() {
        // Realistic large TDD: f = OR of many random cubes (a DNF), which has coarse,
        // varied left-classes — unlike the EQ probe's singleton lefts, these DO make
        // restrict_node carry varied care-SETS down the recursion, so this exercises
        // the care-set fan-out term. The care c is itself a wide DNF (not a clause).
        // Reports time (scale) AND |g| vs |f∧c| (effectiveness: is restrict's
        // representative smaller than the naive conjunction?).
        use super::{reachable_pairs, restrict};
        use crate::tdd::transform::pairwise::disjoin::apply_or;
        use std::time::Instant;
        // Unbounded scaling ladder on a balanced vtree — hangs a plain
        // `--include-ignored` sweep for hours. Only run when explicitly armed.
        if std::env::var("TIDIDI_SCALING_PROBE").is_err() {
            eprintln!("skipped: set TIDIDI_SCALING_PROBE=1 to run this scaling probe");
            return;
        }
        let mut state: u64 = 0xda7a_5ca1_e000_1111;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let n = 24u32;
        let vtree = Arc::new(Vtree::balanced(n));
        // a random cube of `w` literals (covers 2^(n-w) models).
        let mut mk_cube = |w: usize, rng: &mut dyn FnMut() -> u64| -> Tdd {
            let mut lits: Vec<(u32, bool)> = Vec::new();
            while lits.len() < w {
                let v = (rng() % n as u64) as u32;
                if lits.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                lits.push((v, rng() % 2 == 0));
            }
            lits.sort_by_key(|&(v, _)| v);
            cube(&vtree, &lits)
        };
        // build a DNF of `m` cubes of width `w`.
        let build_dnf = |m: usize, w: usize, rng: &mut dyn FnMut() -> u64, mk: &mut dyn FnMut(usize, &mut dyn FnMut() -> u64) -> Tdd| -> Tdd {
            let mut acc = mk(w, rng);
            for _ in 1..m {
                acc = apply_or(&acc, &mk(w, rng));
            }
            acc
        };
        // wide care: a DNF of ~30 cubes, width 0.5n (covers ~half each).
        let c = build_dnf(30, (n as usize) / 2, &mut rng, &mut mk_cube);
        let mut cm = c.clone();
        crate::tdd::minimize::minimize(&mut cm);

        println!(
            "\nn={n} care|c|={}  (width 0.65n cubes)\n{:>6} {:>10} {:>10} {:>10} {:>8} {:>12}",
            reachable_pairs(&cm), "mCubes", "|f|", "|f∧c|", "|g|", "g/f∧c", "restrict_ms"
        );
        let w = ((n as f64) * 0.65).round() as usize;
        for &m in &[100usize, 300, 800, 2000, 5000] {
            let f = build_dnf(m, w, &mut rng, &mut mk_cube);
            let mut fm = f.clone();
            crate::tdd::minimize::minimize(&mut fm);
            let fc = and2(&f, &c);
            let mut fcm = fc.clone();
            crate::tdd::minimize::minimize(&mut fcm);
            let t0 = Instant::now();
            let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            assert!(equiv(&and2(&g, &c), &fc), "unsound at m={m}");
            let (sf, sfc, sg) = (reachable_pairs(&fm), reachable_pairs(&fcm), reachable_pairs(&g));
            let ratio = if sfc > 0 { sg as f64 / sfc as f64 } else { 0.0 };
            println!("{m:>6} {sf:>10} {sfc:>10} {sg:>10} {ratio:>8.2} {ms:>12.1}");
        }
        println!("(scale = restrict_ms vs |f|; effectiveness = g/f∧c < 1)");
    }

    #[test]
    #[ignore = "reporting: run via --ignored --nocapture for the effectiveness table"]
    fn restrict_effectiveness_conj_grows() {
        // The regime restrict is BUILT for: f and c whose conjunction GROWS
        // (|f∧c| ≫ |f|). f ranges over the low half of the variables, the care c over
        // the high half (a small shared band), so f∧c ≈ |f|·|c| blows up while
        // restrict returns g ≤ f. This is where restrict beats naive conjunction —
        // unlike `restrict_scaling_real_dnf`, whose restrictive care makes |f∧c| ≪ |f|.
        // Reports g/f∧c (the win vs conjunction), g/f, and restrict time (scaling).
        use super::{reachable_pairs, restrict};
        use crate::tdd::transform::pairwise::disjoin::apply_or;
        use crate::tdd::minimize::minimize as mini;
        use std::time::Instant;
        let mut state: u64 = 0xeff0_0011_2233_4455;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let n = 28u32;
        let vtree = Arc::new(Vtree::balanced(n));
        // An anchored random cube: `w` literals drawn from var range [lo, hi) plus the
        // `anchor` extreme literal, so the function roots at the vtree root (restrict's
        // same-root precondition; otherwise it no-ops).
        let mk = |lo: u32, hi: u32, w: usize, anchor: u32, rng: &mut dyn FnMut() -> u64| -> Tdd {
            let mut lits: Vec<(u32, bool)> = vec![(anchor, true)];
            while lits.len() < w + 1 {
                let v = lo + (rng() % (hi - lo) as u64) as u32;
                if v == anchor || lits.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                lits.push((v, rng() % 2 == 0));
            }
            lits.sort_by_key(|&(v, _)| v);
            cube(&vtree, &lits)
        };
        let dnf = |m: usize, lo: u32, hi: u32, w: usize, anchor: u32,
                   rng: &mut dyn FnMut() -> u64,
                   mk: &dyn Fn(u32, u32, usize, u32, &mut dyn FnMut() -> u64) -> Tdd|
         -> Tdd {
            let mut acc = mk(lo, hi, w, anchor, rng);
            for _ in 1..m {
                acc = apply_or(&acc, &mk(lo, hi, w, anchor, rng));
            }
            acc
        };
        // Care over the HIGH half [n/2, n), anchored at the top var.
        let c = dnf(40, n / 2, n, 4, n - 1, &mut rng, &mk);
        let mut cm = c.clone();
        mini(&mut cm);
        println!(
            "\n[effectiveness conj-grows] n={n} |c|={}\n{:>6} {:>8} {:>10} {:>8} {:>8} {:>8} {:>9} {:>9}",
            reachable_pairs(&cm), "mf", "|f|", "|f∧c|", "|g|", "g/f∧c", "g/f", "and2_ms", "restr_ms"
        );
        for &mf in &[20usize, 60, 150, 400] {
            // f over the LOW half [0, n/2+2) (small shared band), anchored at var 0.
            let f = dnf(mf, 0, n / 2 + 2, 4, 0, &mut rng, &mk);
            let mut fm = f.clone();
            mini(&mut fm);
            // Head-to-head: time the naive conjunction and2(f,c) against restrict(f,c).
            let t_and = Instant::now();
            let fc = and2(&f, &c);
            let and_ms = t_and.elapsed().as_secs_f64() * 1e3;
            let mut fcm = fc.clone();
            mini(&mut fcm);
            let t0 = Instant::now();
            let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            assert!(equiv(&and2(&g, &c), &fc), "unsound at mf={mf}");
            let (sf, sfc, sg) = (reachable_pairs(&fm), reachable_pairs(&fcm), reachable_pairs(&g));
            let r1 = if sfc > 0 { sg as f64 / sfc as f64 } else { 0.0 };
            let r2 = if sf > 0 { sg as f64 / sf as f64 } else { 0.0 };
            println!("{mf:>6} {sf:>8} {sfc:>10} {sg:>8} {r1:>8.3} {r2:>8.3} {and_ms:>9.1} {ms:>9.1}");
        }
        // Result (measured, not assumed): restrict wins decisively on SIZE (g/f∧c down
        // to ~0.01) but LOSES on compute time — computing the conjunction f∧c directly
        // (apply_and, a level-by-level grid product) is faster than restrict, whose
        // per-node conj_empty cell scan is O(width_f × width_c). So restrict is worth it
        // only when the smaller g is reused/stored enough to repay the extra build time;
        // it is NOT a faster drop-in for computing f∧c.
        println!("(restrict wins on size g/f∧c≪1, but computing f∧c directly is faster)");
    }


    // Random conjunction of clauses over `vtree`. When `span`, the support is forced
    // to include the two extreme variables so the function roots at the vtree root —
    // needed so `restrict`'s same-root precondition is met (otherwise it no-ops).
    fn rand_conj(
        vtree: &Arc<Vtree>,
        nvars: u32,
        nclauses_max: u64,
        width_max: u64,
        span: bool,
        rng: &mut dyn FnMut() -> u64,
    ) -> Tdd {
        let mut acc: Option<Tdd> = if span && nvars >= 2 {
            Some(clause_to_tdd(vtree, &clause(&[(0, true), (nvars - 1, true)])))
        } else {
            None
        };
        let nclauses = 1 + (rng() % nclauses_max) as usize;
        for _ in 0..nclauses {
            let width = 1 + (rng() % width_max) as usize;
            let mut lits: Vec<(u32, bool)> = Vec::new();
            for _ in 0..width {
                let v = (rng() % nvars as u64) as u32;
                let pol = rng() % 2 == 0;
                if lits.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                lits.push((v, pol));
            }
            lits.sort_by_key(|&(v, _)| v);
            lits.dedup_by_key(|&mut (v, _)| v);
            let cl = clause_to_tdd(vtree, &clause(&lits));
            acc = Some(match acc {
                None => cl,
                Some(a) => and2(&a, &cl),
            });
        }
        acc.unwrap()
    }

    #[test]
    #[ignore = "heavy: run explicitly via --ignored for extended correctness verification"]
    fn restrict_heavy_correctness() {
        // Extended verification: thousands of random (f, c) over vtree sizes 2..=8,
        // each checked by full-truth-table soundness (apply-free evaluator) + all
        // invariants + exact determinism + never-larger. 2^8 = 256 assignments keeps
        // the brute force tractable.
        use super::{reachable_pairs, restrict};
        use crate::tdd::validate::{check_all_fast, check_determinism};
        let mut state: u64 = 0x51ed_5eed_a5a5_1234;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut total = 0u64;
        let mut shrinks = 0u64;
        for &nvars in &[2u32, 3, 4, 5, 6, 7, 8] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            let cases = if nvars <= 4 { 400 } else { 200 };
            for _ in 0..cases {
                let f = rand_conj(&vtree, nvars, 4, nvars.max(2) as u64, false, &mut rng);
                let c = rand_conj(&vtree, nvars, 4, nvars.max(2) as u64, false, &mut rng);
                if count_is_zero(&c) {
                    continue;
                }
                let g = restrict(&f, c.clone(), super::CareCanonical::No).into_tdd(&f);
                for mask in 0..(1u32 << nvars) {
                    let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                    let cv = eval(&c, &asn);
                    assert_eq!(
                        eval(&g, &asn) && cv,
                        eval(&f, &asn) && cv,
                        "unsound: nvars={nvars} asn={asn:?}"
                    );
                }
                // restrict returns a sound subgraph of f that production uses RAW — it
                // may carry non-canonical false nodes that minimize removes. Soundness
                // is checked on raw g above; check structure/determinism on the canonical
                // form, and the never-larger gate against un-minimized input f (restrict
                // no longer minimizes — that cost is what the EMIT=false oracle avoids).
                let mut gm = g.clone();
                crate::tdd::minimize::minimize(&mut gm);
                check_all_fast(&gm, "heavy");
                check_determinism(&gm).expect("non-deterministic restrict output");
                let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&f));
                assert!(gp <= fp, "grew beyond input f: {gp} > {fp} nvars={nvars}");
                if gp < fp {
                    shrinks += 1;
                }
                total += 1;
            }
        }
        println!(
            "\n[restrict heavy correctness] {total} cases over nvars 2..=8 — \
             all sound, deterministic, valid; {shrinks} strict shrinks"
        );
        assert!(total >= 1500, "too few cases: {total}");
        assert!(shrinks > 0);
    }

    #[test]
    #[ignore = "reporting: run explicitly via --ignored --nocapture for the comparison table"]
    fn restrict_vs_conjunction_overview() {
        // Overview table: node-level restrict vs simple conjunction (apply_and).
        // For each (vtree size, care shape) cell, average over several random f over
        // the SAME spanning vtree: |f|, |f∧c| (conjunction), |restrict|, and the
        // wall time of each op. Soundness is asserted per case so the numbers are
        // trustworthy. Timings are single-process, --test-threads=1.
        use super::{reachable_pairs, restrict};
        use crate::tdd::transform::pairwise::conjoin::apply_and;
        use std::time::Instant;

        let mut state: u64 = 0xc0ffee_1234_5678;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let size = |t: &Tdd| -> usize {
            let mut m = t.clone();
            crate::tdd::minimize::minimize(&mut m);
            reachable_pairs(&m)
        };
        // Time an op over `reps`, returning (µs/op, one result).
        let bench = |reps: usize, op: &mut dyn FnMut() -> Tdd| -> (f64, Tdd) {
            let r0 = op();
            let t0 = Instant::now();
            for _ in 0..reps {
                let _ = op();
            }
            (t0.elapsed().as_secs_f64() / reps as f64 * 1e6, r0)
        };

        #[derive(Clone, Copy)]
        enum Care {
            Cube,
            Clause,
            Random,
        }
        let care_name = |c: Care| match c {
            Care::Cube => "cube(½ vars)",
            Care::Clause => "clause(3-lit)",
            Care::Random => "random-conj",
        };

        println!(
            "\n{:>5} {:>14} {:>8} {:>8} {:>10} {:>11} {:>11} {:>9}",
            "nvars", "care", "|f|", "|f∧c|", "|restrict|", "t_conj µs", "t_restr µs", "g/f∧c"
        );
        println!("{}", "-".repeat(82));

        for &nvars in &[8u32, 10, 12] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            for &care in &[Care::Cube, Care::Clause, Care::Random] {
                let seeds = 8;
                let (mut sf, mut sfc, mut sg, mut tc, mut tr) = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
                let mut counted = 0;
                for _ in 0..seeds {
                    let f = rand_conj(&vtree, nvars, 5, (nvars / 2).max(2) as u64, true, &mut rng);
                    let c = match care {
                        Care::Cube => {
                            // a cube over ~half the vars, spanning the extremes.
                            let mut lits: Vec<(u32, bool)> = vec![(0, true), (nvars - 1, false)];
                            let half = (nvars / 2).max(2);
                            for k in 1..half {
                                lits.push((k, rng() % 2 == 0));
                            }
                            lits.sort_by_key(|&(v, _)| v);
                            lits.dedup_by_key(|&mut (v, _)| v);
                            // a cube = conjunction of unit clauses
                            let mut acc = clause_to_tdd(&vtree, &clause(&[lits[0]]));
                            for &l in &lits[1..] {
                                acc = and2(&acc, &clause_to_tdd(&vtree, &clause(&[l])));
                            }
                            acc
                        }
                        Care::Clause => clause_to_tdd(
                            &vtree,
                            &clause(&[(0, true), (nvars / 2, false), (nvars - 1, true)]),
                        ),
                        Care::Random => rand_conj(&vtree, nvars, 4, (nvars / 2).max(2) as u64, true, &mut rng),
                    };
                    if count_is_zero(&c) {
                        continue;
                    }
                    // Same-root precondition (both span): if not met, skip (rare).
                    if f.output.vtree != c.output.vtree {
                        continue;
                    }
                    let reps = if nvars >= 12 { 12 } else { 30 };
                    let (t_conj, conj) = {
                        let f2 = f.clone();
                        let c2 = c.clone();
                        bench(reps, &mut || {
                            let mut a = f2.clone();
                            let mut b = c2.clone();
                            apply_and(&mut a, &mut b)
                        })
                    };
                    let (t_restr, g) = {
                        let f2 = f.clone();
                        let c2 = c.clone();
                        bench(reps, &mut || {
                            restrict(&f2, c2.clone(), super::CareCanonical::No).into_tdd(&f2)
                        })
                    };
                    // soundness so the row is trustworthy.
                    assert!(
                        equiv(&and2(&g, &c), &conj),
                        "restrict unsound in overview (nvars={nvars})"
                    );
                    sf += size(&f);
                    sfc += reachable_pairs(&conj);
                    sg += size(&g);
                    tc += t_conj;
                    tr += t_restr;
                    counted += 1;
                }
                if counted == 0 {
                    continue;
                }
                let (af, afc, ag) = (sf / counted, sfc / counted, sg / counted);
                let ratio = if afc > 0 { ag as f64 / afc as f64 } else { 0.0 };
                println!(
                    "{:>5} {:>14} {:>8} {:>8} {:>10} {:>11.1} {:>11.1} {:>9.2}",
                    nvars,
                    care_name(care),
                    af,
                    afc,
                    ag,
                    tc / counted as f64,
                    tr / counted as f64,
                    ratio
                );
            }
        }
        println!(
            "\nSizes are reachable pairs after minimize, averaged over seeds. \
             |restrict| ≤ |f| by gate; g∧c == f∧c (asserted). g/f∧c < 1 ⇒ restrict's \
             representative is smaller than the conjunction."
        );
    }

}

mod marginal_lift_indicator {
    use std::sync::Arc;

    use num_bigint::BigUint;

    use crate::tdd::build::constant_one;
    use crate::tdd::query::model_count;
    use crate::tdd::test_helpers::{compile_clauses, marginalize_subtree};
    use crate::tdd::transform::pairwise::conjoin::apply_and;
    use crate::tdd::transform::unary::demarginalize::demarginalize_to_indicator;
    use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

    /// A constraint carrying marginal levels is lifted to a fully non-marginal
    /// indicator, free over the summed-out variables, satisfiable iff the
    /// original count is non-zero.
    #[test]
    fn marginal_constraint_lifted_to_free_indicator() {
        let vtree = Arc::new(Vtree::balanced(5));
        let mut t = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![3, -4], vec![4, 5]]);
        let c = (0..vtree.num_nodes())
            .map(|vi| VtreeIdx(vi as u32))
            .find(|&vi| matches!(*vtree.node(vi), VtreeNode::Internal { .. }) && vi != vtree.root())
            .expect("balanced(5) has a non-root internal node");
        marginalize_subtree(&mut t, c);
        let zero = BigUint::from(0u32);
        let sat_before = model_count(&t) != zero;
        assert!(
            (0..vtree.num_nodes()).any(|i| {
                matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && t.levels[i].is_marginal()
            }),
            "setup must produce an internal marginal level"
        );

        let mut ind = t.clone();
        demarginalize_to_indicator(&mut ind);

        for i in 0..vtree.num_nodes() {
            if matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) {
                assert!(!ind.levels[i].is_marginal(), "internal level {i} must be lifted");
            }
        }
        assert_eq!(model_count(&ind) != zero, sat_before, "indicator must preserve satisfiability");
        let base = model_count(&ind);
        assert_eq!(model_count(&apply_and(&mut ind.clone(), &mut ind.clone())), base, "idempotent");
        let mut one = constant_one(&ind.vtree);
        assert_eq!(model_count(&apply_and(&mut one, &mut ind.clone())), base, "free ∧ indicator");
    }
}

