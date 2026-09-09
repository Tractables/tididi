//! Restriction against a care set: semantics and randomized difftests.
//!
//! Sibling of `unary_tests.rs`, which holds the fixtures these read.

use super::*;

use crate::engine::Engine;

#[test]
fn restrict_tautological_care_is_identity() {
    let eng = Engine::new();
    // c = ⊤ pins g everywhere → g must equal f (no don't-cares).
    let vtree = Arc::new(Vtree::balanced(3));
    let x2 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, true)]));
    let f = apply_or(and2(&clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)])),
                           &clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(1, true)]))),
                     x2);
    let c = constant_one(&eng, &vtree);
    let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
    assert!(equiv_nf(&eng, &g, &f), "crate::apply::restrict(f, ⊤) must equal f");
    assert_restrict_ok(&eng, &f, &c, 3);
}

#[test]
fn restrict_false_care_is_empty() {
    let eng = Engine::new();
    // c = ⊥: f∧c = ∅ for any g; restrict returns ⊥, the smallest sound answer.
    let vtree = Arc::new(Vtree::balanced(3));
    let f = apply_or(clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)])),
                     clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(1, true), (2, true)])));
    let c = constant_zero(&eng, &vtree);
    let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
    assert!(count_is_zero(&eng, &g), "crate::apply::restrict(f, ⊥) must be ⊥ (f∧⊥ = ∅)");
    crate::check::check_all_fast(&g, "restrict-false-care");
}

#[test]
fn restrict_of_false_is_false() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = constant_zero(&eng, &vtree);
    let c = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
    assert!(count_is_zero(&eng, &g), "crate::apply::restrict(⊥, c) must be ⊥");
    crate::check::check_all_fast(&g, "restrict-of-false");
}

#[test]
fn restrict_of_true_is_sound_and_valid() {
    let eng = Engine::new();
    // f = ⊤: g∧c must = c. g = ⊤ is the smallest sound answer.
    let vtree = Arc::new(Vtree::balanced(3));
    let f = constant_one(&eng, &vtree);
    let c = apply_or(clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)])),
                     clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, true)])));
    assert_restrict_ok(&eng, &f, &c, 3);
}

#[test]
fn restrict_cube_care_shrinks_or_holds() {
    let eng = Engine::new();
    // f = (x0 ∧ x1) ∨ (¬x0 ∧ x2); care c = x0. On the care, f reduces to x1 and
    // the x2 branch is don't-care — the classic shrink. We assert the contract
    // (sound, never-larger, valid) and brute-force soundness directly.
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let nx0 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, false)]));
    let x1 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));
    let x2 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, true)]));
    let f = apply_or(and2(&x0, &x1), and2(&nx0, &x2));
    let c = x0.clone();
    assert_restrict_ok(&eng, &f, &c, 3);
    // The restricted function must agree with x1 on the care set.
    let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
    assert!(equiv(&eng, &and2(&g, &c), &and2(&x1, &c)));
}

#[test]
fn restrict_drop_lever_sound_and_valid() {
    let eng = Engine::new();
    // f = (x0 ∧ x2) ∨ (x1 ∧ ¬x2); care c = x0. Wherever x0 = 0 the second term is
    // don't-care, so the DROP lever can prune it. Contract + brute force.
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let x1 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));
    let x2 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, true)]));
    let nx2 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, false)]));
    let f = apply_or(and2(&x0, &x2), and2(&x1, &nx2));
    let c = x0;
    assert_restrict_ok(&eng, &f, &c, 3);
}

#[test]
fn restrict_drops_dead_pair_of_alive_node() {
    let eng = Engine::new();
    // R3 pair-granular liveness: f = (x0 ∨ x1) has root pairs
    // [(x0,⊤), (¬x0,x1)]; care = (x0 ∨ ¬x1) kills every product of the second
    // pair ((¬x0∧x1)∧care = ∅) while the root NODE stays alive via the first.
    // Node-granular liveness alone would see an all-alive diagram; the
    // pair-granular probe must drop the dead pair: g ≡ x0, strictly smaller, sound.
    let vtree = Arc::new(Vtree::balanced(2));
    let f = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let c = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, false)]));
    assert_restrict_ok(&eng, &f, &c, 2);
    let g = match crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No) {
        crate::apply::Restricted::Shrunk(g) => g,
        crate::apply::Restricted::Unchanged => {
            panic!("pair-granular restrict must shrink: pair 2 is dead under care")
        }
        crate::apply::Restricted::False(_) => panic!("f∧care is SAT — must not collapse to ⊥"),
    };
    assert!(
        crate::test_helpers::reachable_pairs(&g) < crate::test_helpers::reachable_pairs(&f),
        "dropping the dead pair must strictly shrink"
    );
    let x0 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    assert!(equiv(&eng, &g, &x0), "g must be exactly x0 after the dead pair drops");
}

#[test]
fn restrict_self_care_is_sound() {
    let eng = Engine::new();
    // c = f: g∧f must = f. g is free off f (the bulk of the cube) — a strong
    // don't-care stress, must stay sound and valid.
    let vtree = Arc::new(Vtree::balanced(4));
    let f = apply_or(and2(&clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true)])),
                           &clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(1, false)]))),
                     clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)])));
    let c = f.clone();
    assert_restrict_ok(&eng, &f, &c, 4);
}

#[test]
fn restrict_runs_on_assorted_small_circuits() {
    let eng = Engine::new();
    // "It runs" + stays valid on a spread of structured functions (xors, chains,
    // wide clauses), each against a couple of cube and non-cube cares.
    let vtree = Arc::new(Vtree::balanced(4));
    let lit = |v: u32, p: bool| clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(v, p)]));
    let xor = |a: u32, b: u32| {
        apply_or(and2(&lit(a, true), &lit(b, false)), and2(&lit(a, false), &lit(b, true)))
    };
    let fns = vec![
        xor(0, 1),
        and2(&xor(0, 1), &xor(2, 3)),
        apply_or(lit(0, true), and2(&lit(1, true), &lit(2, false))),
        clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, false), (2, true), (3, true)])),
    ];
    let cares = vec![
        lit(0, true),
        apply_or(lit(1, true), lit(2, true)),
        xor(0, 2),
    ];
    for f in &fns {
        for c in &cares {
            if count_is_zero(&eng, c) {
                continue;
            }
            assert_restrict_ok(&eng, f, c, 4);
        }
    }
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
    let eng = &crate::engine::Engine::new();
        use crate::test_helpers::reachable_pairs;
    let vtree = Arc::new(Vtree::balanced(8));
    let lit = |v: u32, p: bool| clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(v, p)]));
    // Function-level soundness oracle: g∧c == f∧c over all 2^nvars assignments,
    // and g never larger than f. Apply-free (shares no machinery with restrict).
    let assert_sound = |f: &Tdd, c: &Tdd, nvars: u32| -> Tdd {
        let g = crate::apply::restrict(f, c.clone(), crate::apply::CareCanonical::No).into_tdd(f);
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
    let x2or3 = apply_or(lit(2, true), lit(3, true));
    let x2and3 = and2(&lit(2, true), &lit(3, true));
    let sel = apply_or(and2(&lit(0, true), &x2or3), and2(&lit(0, false), &x2and3));
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
    let rt = apply_or(lit(4, true), lit(5, true)); // (x4∨x5), depends on {4,5} ⊂ Rt
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
    let eng = Engine::new();
    // The exhaustive-soundness sweep: random (f, c) over several vtree SIZES, each
    // case checked by the apply-free evaluator over the full truth table PLUS all
    // invariants PLUS exact determinism PLUS never-larger. Small nvars keep the
    // 2^n brute force and the O(width²·apply) determinism check cheap.
        use crate::test_helpers::reachable_pairs;
    use crate::check::{check_all_fast, check_determinism};
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
                    let pol = rng().is_multiple_of(2);
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                lits.dedup_by_key(|&mut (v, _)| v);
                let cl = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&lits));
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
            if count_is_zero(&eng, &c) {
                continue;
            }
            // Track the shrink count to keep the "levers inert" guard meaningful.
            let fp = reachable_pairs(&f);
            let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
            crate::reduce::minimize(&mut gm);
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
    let eng = Engine::new();
    // `reduce`/`restrict` must return an ARENA-COMPACT diagram: the rebuild is
    // demand-driven and emits a child before discovering its pair partner
    // collapsed to ZERO, which strands that child (an orphan: reachable_pairs
    // unchanged, but it lingers in the arena). The self-contained reduce prunes
    // its own output, so `size(g) == reachable_pairs(g)` for ANY caller. Bigger
    // vtrees (mixed liveness) are what surface the orphan; this FAILS on the
    // pre-prune engine and passes after. Soundness is asserted alongside so the
    // compactness numbers are trustworthy.
        use crate::test_helpers::reachable_pairs;
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
                    let pol = rng().is_multiple_of(2);
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                lits.dedup_by_key(|&mut (v, _)| v);
                let cl = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&lits));
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
            if count_is_zero(&eng, &c) {
                continue;
            }
            // Inputs come from the test's non-minimizing `and2`/`Tdd::clause`
            // builder and can carry their own orphans; minimize so we test
            // reduce's own compactness contract, not the builder's.
            crate::reduce::minimize(&mut f);
            crate::reduce::minimize(&mut c);
            let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
    let eng = Engine::new();
        use crate::test_helpers::reachable_pairs;
    let nvars = 6u32;
    let vtree = Arc::new(Vtree::balanced(nvars));
    // The left block of the global root: balanced(6) puts {0,1,2} under it.
    let left_vars: Vec<u32> = {
        let (lc, _) = match *vtree.node(vtree.root()) {
            crate::vtree::VtreeNode::Internal { left, right, .. } => (left, right),
            _ => unreachable!(),
        };
        (0..nvars)
            .filter(|&v| vtree.lca(vtree.leaf_of(VarId(v)).expect("the vtree carries this variable"), lc) == lc)
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
                let pol = rng().is_multiple_of(2);
                if lits.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                lits.push((v, pol));
            }
            lits.sort_by_key(|&(v, _)| v);
            let cl = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&lits));
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
        crate::reduce::minimize(&mut m);
        if m.is_zero() || m.levels[m.output.vtree.idx()].pairs_of_idx(m.output.local.idx()).len() != 1 {
            return None;
        }
        Some(reroot_to_child(&m, true))
    };
    let all_vars: Vec<u32> = (0..nvars).collect();
    let check = |f: &Tdd, c: &Tdd| -> bool {
        let g = crate::apply::restrict(f, c.clone(), crate::apply::CareCanonical::No).into_tdd(f);
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
        if let Some(care) = rehome_left(&rand_over(&mut rng, &left_vars))
            && !f.is_zero() && !count_is_zero(&eng, &care) {
                assert_ne!(care.output.vtree, f.output.vtree);
                shrinks += check(&f, &care) as u32;
                total += 1;
            }
        // f strictly below care's root
        let care = rand_over(&mut rng, &all_vars);
        if let Some(f) = rehome_left(&rand_over(&mut rng, &left_vars))
            && !f.is_zero() && !count_is_zero(&eng, &care) {
                assert_ne!(care.output.vtree, f.output.vtree);
                shrinks += check(&f, &care) as u32;
                total += 1;
            }
    }
    assert!(total >= 100, "too few differing-root cases exercised: {total}");
    assert!(shrinks > 0, "no shrink on any differing-root case — walk inert");
}

#[test]
fn restrict_raw_output_is_apply_safe() {
    let eng = Engine::new();
    // Regression for the WS_FAST_REDUCE panic (prune.rs index-OOB): that lever
    // swaps in the RAW `restrict` output (un-minimized) and then conjoins
    // it — `apply_and(g, other)` followed by the conjoin's `minimize`. Public
    // `restrict` minimizes g first, so the raw-output → apply path is otherwise
    // untested. Assert (A) the raw g is a valid TDD and (B) conjoining it with an
    // arbitrary other member, then minimizing the product, stays valid and never
    // panics — over many random (f, care, other) across vtree sizes.
        use crate::check::check_all_fast;
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
                    let pol = rng().is_multiple_of(2);
                    if lits.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    lits.push((v, pol));
                }
                lits.sort_by_key(|&(v, _)| v);
                lits.dedup_by_key(|&mut (v, _)| v);
                let cl = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&lits));
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
            if count_is_zero(&eng, &c) || f.is_zero() {
                continue;
            }
            // (A) raw restrict output must be a valid TDD.
            let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
            check_all_fast(&g, "restrict-raw");
            // (B) the lever's path: conjoin raw g with another member, minimize.
            if other.is_zero() || g.is_zero() {
                continue;
            }
            let mut ga = g.clone();
            let mut ob = other.clone();
            ga.vtree = f.vtree.clone();
            ob.vtree = f.vtree.clone();
            let mut p = apply_and(ga, ob);
            crate::reduce::minimize(&mut p);
            check_all_fast(&p, "apply(restrict-raw, other)+minimize");
            conjoined += 1;
        }
    }
    assert!(conjoined >= 100, "too few conjoin cases exercised: {conjoined}");
}
