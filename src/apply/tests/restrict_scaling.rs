//! How far restriction shrinks a diagram, on the large operands.
//!
//! Fixtures come from `crate::test_helpers`, re-exported by the parent.
//!
//! Every test here is `#[ignore]`d: each builds diagrams far larger than the
//! rest of the suite, and the size is the point — these reach shapes the small
//! sweeps cannot. Run them with `--ignored`.

use super::*;

use crate::Engine;

#[test]
#[ignore = "heavy: a ladder of very wide root nodes"]
fn restrict_scaling_wide_node() {
    let eng = Engine::new();
    // Worst-case stress: a single very wide root node. f = AND_i (x_i == x_{k+i})
    // over balanced(2k) — the root pairs each left-half value with its unique
    // matching right-half value, so root width is exponential in k. This isolates
    // the two scaling terms: the disjoint-lefts scan over pairs of root nodes, and
    // the care-set recursion fanning out down the right subtree.
    for k in [4u32, 6, 8, 10, 12, 14, 15, 16, 17, 18] {
        let n = 2 * k;
        let vtree = Arc::new(Vtree::balanced(n));
        // f = AND_i (x_i <-> x_{k+i}); each equivalence is two clauses.
        let mut f: Option<Tdd> = None;
        for i in 0..k {
            let (a, b) = (i + 1, k + i + 1);
            let e1 = clause_to_tdd(&eng, &vtree, &clause(&[(a, true), (b, false)]));
            let e2 = clause_to_tdd(&eng, &vtree, &clause(&[(a, false), (b, true)]));
            let eq = and2(&e1, &e2);
            f = Some(match f {
                None => eq,
                Some(prev) => and2(&prev, &eq),
            });
        }
        let f = f.unwrap();
        // care = one clause spanning both halves (roots at the root node).
        let c = clause_to_tdd(&eng, &vtree, &clause(&[(1, true), (k, true)]));
        let g = (f.clone()).restrict_to_care(c.clone()).unwrap().into_tdd();
        // Soundness is exhaustively covered by `restrict_heavy_correctness`; the
        // equivalence check here conjoins at full root width, so it is affordable
        // only on the low-k rows.
        if k <= 14 {
            assert!(equiv(&and2(&g, &c), &and2(&f, &c)), "unsound at k={k}");
        }
        assert!(
            reachable_pairs(&g) <= reachable_pairs(&f),
            "restrict_to_care grew the diagram beyond f at k={k}"
        );
    }
}

#[test]
#[ignore = "heavy: an unbounded ladder of large disjunctive normal forms"]
fn restrict_scaling_real_dnf() {
    // Realistic large diagram: f is a disjunction of many random cubes, which has
    // coarse, varied left-classes — unlike the equivalence probe's singleton
    // lefts, these do make the node walk carry varied care-sets down the
    // recursion, so this exercises the care-set fan-out. The care is itself a
    // wide disjunction, not a clause.
    use crate::apply::apply_or;
    // The ladder has no bound, and hangs a plain `--include-ignored` sweep. Only
    // run when explicitly armed.
    if std::env::var("TIDIDI_SCALING_PROBE").is_err() {
        eprintln!("skipped: set TIDIDI_SCALING_PROBE=1 to run this scaling probe");
        return;
    }
    let mut rng = Lcg::new(0xda7a_5ca1_e000_1111);
    let n = 24u32;
    let vtree = Arc::new(Vtree::balanced(n));
    // A random cube of `w` literals.
    let mut mk_cube = |w: usize, rng: &mut Lcg| -> Tdd {
        let mut literals: Vec<(u32, bool)> = Vec::new();
        while literals.len() < w {
            let v = rng.below(u64::from(n)) as u32 + 1;
            if literals.iter().any(|(u, _)| *u == v) {
                continue;
            }
            literals.push((v, rng.coin()));
        }
        literals.sort_by_key(|&(v, _)| v);
        cube(&vtree, &literals)
    };
    // A disjunction of `m` cubes of width `w`.
    let build_dnf =
        |m: usize, w: usize, rng: &mut Lcg, mk: &mut dyn FnMut(usize, &mut Lcg) -> Tdd| -> Tdd {
            let mut acc = mk(w, rng);
            for _ in 1..m {
                acc = apply_or(acc, mk(w, rng));
            }
            acc
        };
    // Wide care: a disjunction of cubes each covering about half the space.
    let c = build_dnf(30, (n as usize) / 2, &mut rng, &mut mk_cube);

    let w = ((n as f64) * 0.65).round() as usize;
    for &m in &[100usize, 300, 800, 2000, 5000] {
        let f = build_dnf(m, w, &mut rng, &mut mk_cube);
        let fc = and2(&f, &c);
        let g = (f.clone()).restrict_to_care(c.clone()).unwrap().into_tdd();
        assert!(equiv(&and2(&g, &c), &fc), "unsound at m={m}");
        assert!(reachable_pairs(&g) <= reachable_pairs(&f), "restrict_to_care grew beyond f at m={m}");
    }
}

#[test]
#[ignore = "heavy: large operands whose conjunction grows"]
fn restrict_effectiveness_conj_grows() {
    // The regime restriction is built for: f and c whose conjunction grows.
    // f ranges over the low half of the variables and the care c over the high
    // half, sharing a small band, so the conjunction approaches the product of
    // the two sizes while restriction returns a subgraph of f. Unlike
    // `restrict_scaling_real_dnf`, whose restrictive care makes the conjunction
    // the smaller of the two.
    use crate::apply::apply_or;
    let mut rng = Lcg::new(0xeff0_0011_2233_4455);
    let n = 28u32;
    let vtree = Arc::new(Vtree::balanced(n));
    // An anchored random cube: `w` literals drawn from the variable range
    // `[lo, hi)` plus the `anchor` extreme literal, so the function roots at the
    // vtree root — restriction's same-root precondition, without which it no-ops.
    let mk = |lo: u32, hi: u32, w: usize, anchor: u32, rng: &mut Lcg| -> Tdd {
        let mut literals: Vec<(u32, bool)> = vec![(anchor + 1, true)];
        while literals.len() < w + 1 {
            let v = lo + rng.below(u64::from(hi - lo)) as u32 + 1;
            if v == anchor + 1 || literals.iter().any(|(u, _)| *u == v) {
                continue;
            }
            literals.push((v, rng.coin()));
        }
        literals.sort_by_key(|&(v, _)| v);
        cube(&vtree, &literals)
    };
    type MkCube<'a> = &'a dyn Fn(u32, u32, usize, u32, &mut Lcg) -> Tdd;
    let dnf =
        |m: usize, lo: u32, hi: u32, w: usize, anchor: u32, rng: &mut Lcg, mk: MkCube<'_>| -> Tdd {
            let mut acc = mk(lo, hi, w, anchor, rng);
            for _ in 1..m {
                acc = apply_or(acc, mk(lo, hi, w, anchor, rng));
            }
            acc
        };
    // Care over the high half `[n/2, n)`, anchored at the top variable.
    let c = dnf(40, n / 2, n, 4, n - 1, &mut rng, &mk);
    for &mf in &[20usize, 60, 150, 400] {
        // f over the low half `[0, n/2+2)` — the small shared band — anchored at
        // variable 0.
        let f = dnf(mf, 0, n / 2 + 2, 4, 0, &mut rng, &mk);
        let fc = and2(&f, &c);
        let g = (f.clone()).restrict_to_care(c.clone()).unwrap().into_tdd();
        assert!(equiv(&and2(&g, &c), &fc), "unsound at mf={mf}");
        assert!(reachable_pairs(&g) <= reachable_pairs(&f), "restrict_to_care grew beyond f at mf={mf}");
    }
}

#[test]
#[ignore = "heavy: thousands of randomized cases against a full truth-table oracle"]
fn restrict_heavy_correctness() {
    // Extended verification: thousands of random (f, c) over vtree sizes 2..=8,
    // each checked by full-truth-table soundness (the apply-free evaluator) plus
    // all invariants plus exact determinism plus never-larger. Eight variables
    // keeps the brute force tractable.
    use crate::test_helpers::check::{check_all_fast, check_determinism};
    let mut rng = Lcg::new(0x51ed_5eed_a5a5_1234);
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
            let g = (f.clone()).restrict_to_care(c.clone()).unwrap().into_tdd();
            for mask in 0..(1u32 << nvars) {
                let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
                let cv = eval(&c, &asn);
                assert_eq!(
                    eval(&g, &asn) && cv,
                    eval(&f, &asn) && cv,
                    "unsound: nvars={nvars} asn={asn:?}"
                );
            }
            // Restriction returns a sound subgraph of f that callers use raw — it
            // may carry non-canonical false nodes that minimizing removes.
            // Soundness is checked on raw g above; structure and determinism are
            // checked on the canonical form, and the never-larger gate against the
            // un-minimized input f.
            let mut gm = g.clone();
            gm.minimize().unwrap();
            check_all_fast(&gm, "heavy");
            check_determinism(&gm).expect("non-deterministic restrict_to_care output");
            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&f));
            assert!(gp <= fp, "grew beyond input f: {gp} > {fp} nvars={nvars}");
            if gp < fp {
                shrinks += 1;
            }
            total += 1;
        }
    }
    assert!(total >= 1500, "too few cases: {total}");
    assert!(shrinks > 0);
}

#[test]
#[ignore = "heavy: wide spanning operands across several vtree sizes"]
fn restrict_vs_conjunction_overview() {
    let eng = Engine::new();
    // A spread of (vtree size, care shape) cells, several random spanning f per
    // cell, each restriction checked against the conjunction it must agree with,
    // and against the never-larger gate. The three care shapes — a cube over half
    // the variables, a three-literal clause, and a random conjunction — are the
    // regimes restriction meets in a fold.
    use crate::apply::apply_and;

    let mut rng = Lcg::new(0x00c0_ffee_1234_5678);

    #[derive(Clone, Copy)]
    enum Care {
        Cube,
        Clause,
        Random,
    }

    for &nvars in &[8u32, 10, 12] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        for &care in &[Care::Cube, Care::Clause, Care::Random] {
            let seeds = 8;
            for _ in 0..seeds {
                let f = rand_conj(&vtree, nvars, 5, (nvars / 2).max(2) as u64, true, &mut rng);
                let c = match care {
                    Care::Cube => {
                        // A cube over about half the variables, spanning the extremes.
                        let mut literals: Vec<(u32, bool)> = vec![(1, true), (nvars, false)];
                        let half = (nvars / 2).max(2);
                        for k in 1..half {
                            literals.push((k + 1, rng.coin()));
                        }
                        literals.sort_by_key(|&(v, _)| v);
                        literals.dedup_by_key(|&mut (v, _)| v);
                        cube(&vtree, &literals)
                    }
                    Care::Clause => clause_to_tdd(
                        &eng,
                        &vtree,
                        &clause(&[(1, true), (nvars / 2, false), (nvars - 1, true)]),
                    ),
                    Care::Random => {
                        rand_conj(&vtree, nvars, 4, (nvars / 2).max(2) as u64, true, &mut rng)
                    }
                };
                if count_is_zero(&c) {
                    continue;
                }
                // Same-root precondition (both span); skip the rare case it fails.
                if f.output.vtree != c.output.vtree {
                    continue;
                }
                let conj = apply_and(f.clone(), c.clone());
                let g =
                    (f.clone()).restrict_to_care(c.clone()).unwrap().into_tdd();
                assert!(
                    equiv(&and2(&g, &c), &conj),
                    "restrict_to_care unsound in the overview (nvars={nvars})"
                );
                assert!(
                    reachable_pairs(&g) <= reachable_pairs(&f),
                    "restrict_to_care grew beyond f (nvars={nvars})"
                );
            }
        }
    }
}
