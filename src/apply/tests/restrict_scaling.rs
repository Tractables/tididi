//! How far restriction shrinks a diagram, and what it costs.
//!
//! Sibling of `unary_tests.rs`, which holds the fixtures these read.

use super::*;

#[test]
#[ignore = "scaling probe: run via --ignored --nocapture to locate the wide-node wall"]
fn restrict_scaling_wide_node() {
    // Worst-case stress: a single very wide root node. f = AND_i (x_i == x_{k+i})
    // over balanced(2k) — the root pairs each left-half value with its unique
    // matching right-half value, so root width = 2^k. This isolates the two
    // suspected scaling terms: lefts_disjoint is O(width^2) conj_empty calls, and
    // the care-set recursion fans out down the right subtree.
        use std::time::Instant;
    println!("\n{:>4} {:>10} {:>12} {:>14}", "k", "rootW", "totalPairs", "restrict_ms");
    for k in [4u32, 6, 8, 10, 12, 14, 15, 16, 17, 18] {
        let n = 2 * k;
        let vtree = Arc::new(Vtree::balanced(n));
        // f = AND_i (x_i <-> x_{k+i}); each equiv is two clauses.
        let mut f: Option<Tdd> = None;
        for i in 0..k {
            let (a, b) = (i, k + i);
            let e1 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(a, true), (b, false)]));
            let e2 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(a, false), (b, true)]));
            let eq = and2(&e1, &e2);
            f = Some(match f {
                None => eq,
                Some(prev) => and2(&prev, &eq),
            });
        }
        let f = f.unwrap();
        // care = one clause spanning both halves (roots at the root node).
        let c = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(0, true), (k, true)]));
        // Width/pair stats taken on f directly — no extra minimize pass (it's an
        // O(width) cost that would dominate the budget at high k, unrelated to restrict).
        let width = root_width(&f);
        let pairs = crate::test_helpers::reachable_pairs(&f);
        let t0 = Instant::now();
        let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
        use crate::test_helpers::reachable_pairs;
    use crate::apply::apply_or;
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
            acc = apply_or(acc, mk(w, rng));
        }
        acc
    };
    // wide care: a DNF of ~30 cubes, width 0.5n (covers ~half each).
    let c = build_dnf(30, (n as usize) / 2, &mut rng, &mut mk_cube);
    let mut cm = c.clone();
    crate::reduce::minimize(&mut cm);

    println!(
        "\nn={n} care|c|={}  (width 0.65n cubes)\n{:>6} {:>10} {:>10} {:>10} {:>8} {:>12}",
        reachable_pairs(&cm), "mCubes", "|f|", "|f∧c|", "|g|", "g/f∧c", "restrict_ms"
    );
    let w = ((n as f64) * 0.65).round() as usize;
    for &m in &[100usize, 300, 800, 2000, 5000] {
        let f = build_dnf(m, w, &mut rng, &mut mk_cube);
        let mut fm = f.clone();
        crate::reduce::minimize(&mut fm);
        let fc = and2(&f, &c);
        let mut fcm = fc.clone();
        crate::reduce::minimize(&mut fcm);
        let t0 = Instant::now();
        let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
        use crate::test_helpers::reachable_pairs;
    use crate::apply::apply_or;
    use crate::reduce::minimize as mini;
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
            acc = apply_or(acc, mk(lo, hi, w, anchor, rng));
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
        // Head-to-head: time the naive conjunction and2(f,c) against crate::apply::restrict(f,c).
        let t_and = Instant::now();
        let fc = and2(&f, &c);
        let and_ms = t_and.elapsed().as_secs_f64() * 1e3;
        let mut fcm = fc.clone();
        mini(&mut fcm);
        let t0 = Instant::now();
        let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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

#[test]
#[ignore = "heavy: run explicitly via --ignored for extended correctness verification"]
fn restrict_heavy_correctness() {
    // Extended verification: thousands of random (f, c) over vtree sizes 2..=8,
    // each checked by full-truth-table soundness (apply-free evaluator) + all
    // invariants + exact determinism + never-larger. 2^8 = 256 assignments keeps
    // the brute force tractable.
        use crate::test_helpers::reachable_pairs;
    use crate::check::{check_all_fast, check_determinism};
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
            let g = crate::apply::restrict(&f, c.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
            crate::reduce::minimize(&mut gm);
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
        use crate::test_helpers::reachable_pairs;
    use crate::apply::apply_and;
    use std::time::Instant;

    let mut state: u64 = 0xc0ffee_1234_5678;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 33
    };
    let size = |t: &Tdd| -> usize {
        let mut m = t.clone();
        crate::reduce::minimize(&mut m);
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
        Care::Clause => "crate::test_helpers::clause(3-lit)",
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
                        let mut acc = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[lits[0]]));
                        for &l in &lits[1..] {
                            acc = and2(&acc, &clause_to_tdd(&vtree, &crate::test_helpers::clause(&[l])));
                        }
                        acc
                    }
                    Care::Clause => clause_to_tdd(
                        &vtree,
                        &crate::test_helpers::clause(&[(0, true), (nvars / 2, false), (nvars - 1, true)]),
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
                        let a = f2.clone();
                        let b = c2.clone();
                        apply_and(a, b)
                    })
                };
                let (t_restr, g) = {
                    let f2 = f.clone();
                    let c2 = c.clone();
                    bench(reps, &mut || {
                        crate::apply::restrict(&f2, c2.clone(), crate::apply::CareCanonical::No).into_tdd(&f2)
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
