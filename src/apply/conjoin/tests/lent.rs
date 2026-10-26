//! The sparse leaf arm's levels whose nodes are f's as f stores them, moved
//! from f instead of written: the same diagram, stored alike, for the same
//! work, and both operands given back as they were on a refusal after such
//! a level; and the levels a conjunction builds as an operand holds them,
//! moved, copied or found once built, owed no twin search.

use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::diagram::Pass;
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::test_helpers::{compile_clauses_on, rand_cnf, same_storage, unnamed_levels, vtree_shapes, CnfShape, Lcg};
use crate::vtree::Vtree;

/// Every level the sparse route can take, takes it.
const SPARSE: SparseRoute = SparseRoute { sparsity: 1, min_grid: 0 };

/// Pairs of formulas on every vtree shape over `nvars` variables, the
/// first implying the second: a random formula, and it with a few more
/// clauses. Where the first's nodes refine the second's, the leaf arm's
/// product level is the first's.
fn cases(nvars: u32, seed: u64) -> Vec<(String, Arc<Vtree>, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for (name, vtree) in vtree_shapes(nvars) {
        for case in 0..6 {
            let eng = Engine::new();
            let base = rand_cnf(&mut rng, nvars, CnfShape { clauses: 2 * nvars as usize, width: 3 });
            let more = rand_cnf(&mut rng, nvars, CnfShape { clauses: 1 + case % 3, width: 2 });
            let g = compile_clauses_on(&eng, &vtree, &base);
            let f = compile_clauses_on(&eng, &vtree, &[base, more].concat());
            out.push((format!("{name}, {nvars} variables, case {case}"), Arc::clone(&vtree), f, g));
        }
    }
    out
}

/// Conjoin `f` and `g` on a fresh engine, every level on the sparse route
/// it can take, and the work it took.
fn conjoin(f: &Tdd, g: &Tdd) -> (Tdd, u64) {
    let eng = Engine::new();
    let _sparse = eng.limits().scope(LimitConfig::none().with_sparse_route(SPARSE));
    let mark = eng.limits().mark();
    let out = eng.and(f.clone(), g.clone()).unwrap();
    (out, eng.limits().work_since(mark))
}

/// Both ways round, the levels moved from f against the same levels
/// written: the same nodes and pairs on every level, and the same work.
#[test]
fn a_level_moved_from_f_is_the_level_written() {
    let before = lent_levels();
    let mut conjunctions = 0;
    for nvars in [10, 14] {
        for (what, _, f, g) in cases(nvars, 0x1e47_0f00 + u64::from(nvars)) {
            for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
                let (out, work) = conjoin(a, b);
                let (oracle, oracle_work) = no_lend(|| conjoin(a, b));
                assert!(same_storage(&out, &oracle), "{what}, {order}: the levels moved from f are not the levels written");
                assert_eq!(work, oracle_work, "{what}, {order}: the levels moved from f took other work");
                conjunctions += 1;
            }
        }
    }
    let lent = lent_levels() - before;
    eprintln!("levels moved from f: {lent} over {conjunctions} conjunctions");
    assert!(lent > 0, "expected levels moved from f, got none over {conjunctions} conjunctions");
}

/// A level built as an operand holds it is owed no twin search: the result
/// leaves it off the contraction's worklist unless an operand owed it, and
/// minimizing the result reaches the diagram minimizing it with every level
/// owed reaches, stored alike. On the sparse route wherever it can be taken
/// and on the routes the gate picks, both ways round, the levels found are
/// f's and g's.
#[test]
fn a_level_built_as_an_operand_holds_it_is_owed_no_twin_search() {
    let (mut of_f, mut of_g) = (0, 0);
    for nvars in [10, 14] {
        for (what, vtree, f, g) in cases(nvars, 0x5e1f_0a5f + u64::from(nvars)) {
            for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
                for sparse in [true, false] {
                    take_as_operand();
                    let out = if sparse {
                        conjoin(a, b).0
                    } else {
                        Engine::new().and(a.clone(), b.clone()).unwrap()
                    };
                    let found = take_as_operand();
                    let owed = |t: &Tdd| t.dirty.clone().take(Pass::Contract);
                    let (contract, operands) = (owed(&out), [owed(a), owed(b)].concat());
                    for &(t, _) in &found {
                        assert!(
                            !contract.contains(&t) || operands.contains(&t),
                            "{what}, {order}, sparse {sparse}: level {t}, built as an operand holds it, is owed a twin search",
                        );
                    }
                    of_f += found.iter().filter(|e| e.1).count();
                    of_g += found.iter().filter(|e| !e.1).count();
                    let eng = Engine::new();
                    let mut reduced = out.clone();
                    eng.minimize(&mut reduced).unwrap();
                    let mut oracle = out;
                    oracle.dirty.requeue(Pass::Contract, vtree.internal_bottomup().map(|(t, _, _)| t.0));
                    eng.minimize(&mut oracle).unwrap();
                    assert!(
                        same_storage(&reduced, &oracle),
                        "{what}, {order}, sparse {sparse}: minimized from the owed levels, not the diagram minimized from every level",
                    );
                }
            }
        }
    }
    eprintln!("levels built as f's: {of_f}, as g's: {of_g}");
    assert!(of_f > 0 && of_g > 0, "expected levels built as f's and as g's, got {of_f} and {of_g}");
}

/// What a refused conjunction gave back of an operand besides its
/// levels: whether they are closed, whether it is certified canonical, and
/// whether it keeps level counts.
fn kept_state(t: &Tdd) -> (bool, bool, bool) {
    (t.levels.is_closed(), t.levels.is_canonical(t.output), t.has_level_counts())
}

/// A restoring conjunction refused at each reserve and at each unit of
/// work in turn, on operands keeping level counts of which it moves levels
/// from f, against the same refusal with those levels written: every
/// refusal gives both operands back as they were given, levels and all,
/// and with what they keep besides as the written levels leave it, those
/// refused after a level was moved among them.
#[test]
fn a_refused_conjunction_gives_f_the_levels_moved_from_it_back() {
    let mut refused_after = 0;
    for (what, _, mut f, mut g) in cases(10, 0x1e47_0f0a) {
        let before = lent_levels();
        let _ = conjoin(&f, &g);
        if lent_levels() == before {
            continue;
        }
        let eng = Engine::new();
        eng.attach_level_counts(&mut f).unwrap();
        eng.attach_level_counts(&mut g).unwrap();
        let expected = conjoin(&f, &g).0;
        for arm in ["reserve", "work"] {
            for n in 0u64.. {
                // A fresh engine each time, so that both arms meet the
                // same reserves: a pooled scratch reserves on its first use.
                #[expect(clippy::result_large_err, reason = "the refusal hands back the operands")]
                let attempt = || {
                    let eng = Engine::new();
                    let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + n)), after_pairs: None };
                    let config = LimitConfig::none().with_sparse_route(SPARSE);
                    let _scope = eng.limits().scope(if arm == "work" { config.with_stop_rules(stop) } else { config });
                    if arm == "reserve" {
                        eng.limits().refuse_nth_reserve(n as u32);
                    }
                    eng.and_restoring(f.clone(), g.clone())
                };
                let lent = lent_levels();
                let outcome = attempt();
                let moved = lent_levels() > lent;
                match (outcome, no_lend(attempt)) {
                    (Ok(out), Ok(_)) => {
                        assert!(eng.equivalent(&out, &expected).unwrap(), "{what}: granted at {arm} {n}, a different function");
                        break;
                    }
                    (Err(refused), Err(oracle)) => {
                        assert!(
                            same_storage(&refused.f, &f) && same_storage(&refused.g, &g),
                            "{what}: refused at {arm} {n}, an operand changed",
                        );
                        assert_eq!(
                            (kept_state(&refused.f), kept_state(&refused.g)),
                            (kept_state(&oracle.f), kept_state(&oracle.g)),
                            "{what}: refused at {arm} {n}, an operand kept other than the written levels leave it",
                        );
                        refused_after += u64::from(moved);
                    }
                    (outcome, oracle) => panic!(
                        "{what}: at {arm} {n}, granted {} with levels moved from f and {} with them written",
                        outcome.is_ok(), oracle.is_ok(),
                    ),
                }
                assert!(n < 100_000, "{what}: never granted");
            }
        }
    }
    eprintln!("refusals after a level moved from f: {refused_after}");
    assert!(refused_after > 0, "expected refusals after a level moved from f, got none");
}

/// Whether every node of the internal level `c` of `out` is named by a pair
/// of its parent level.
fn named_whole(out: &Tdd, c: crate::vtree::VtreeIdx) -> bool {
    let p = out.vtree.node(c).parent().expect("a level below the root");
    let left = out.vtree.children(p).0 == c;
    let mut named = vec![false; out.levels[c.idx()].slot_count()];
    let dec = crate::diagram::ChildDecoder::structural();
    for (_, pairs) in out.levels[p.idx()].internal_inputs_iter() {
        for pair in pairs {
            named[dec.node(if left { pair.left } else { pair.right }).0 as usize] = true;
        }
    }
    named.iter().all(|&n| n)
}

/// A level under one built as an operand holds it is left off the loose
/// levels where the operand's is tight and it has as many nodes: every
/// level the result does not list is tight, and some under levels built as
/// an operand are not listed.
#[test]
fn a_level_under_one_built_as_an_operand_is_tight_where_the_operand_s_is() {
    let mut unlisted = 0;
    for nvars in [10, 14] {
        for (what, vtree, f, g) in cases(nvars, 0x7167_0a5f + u64::from(nvars)) {
            let eng = Engine::new();
            let (mut f, mut g) = (f, g);
            eng.minimize(&mut f).unwrap();
            eng.minimize(&mut g).unwrap();
            // A minimized diagram has no loose level.
            f.dirty.set_loose(Some(Vec::new()));
            g.dirty.set_loose(Some(Vec::new()));
            for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
                for sparse in [true, false] {
                    take_as_operand();
                    let out = if sparse {
                        conjoin(a, b).0
                    } else {
                        Engine::new().and(a.clone(), b.clone()).unwrap()
                    };
                    let found = take_as_operand();
                    if out.is_zero() {
                        continue;
                    }
                    let loose = out.dirty.loose().expect("both operands' loose levels are known").to_vec();
                    for (t, _, _) in vtree.internal_bottomup() {
                        if t == vtree.root() || loose.contains(&t.0) {
                            continue;
                        }
                        assert!(named_whole(&out, t), "{what}, {order}, sparse {sparse}: level {} is not listed loose but holds a node its parent does not name", t.0);
                    }
                    for &(p, _) in &found {
                        let (l, r) = vtree.children(crate::vtree::VtreeIdx(p));
                        unlisted += [l, r].iter().filter(|c| !vtree.node(**c).is_leaf() && !loose.contains(&c.0)).count();
                    }
                }
            }
        }
    }
    eprintln!("levels under levels built as an operand, not listed loose: {unlisted}");
    assert!(unlisted > 0, "expected levels under levels built as an operand left off the loose levels");
}

/// A level whose sibling is a vtree leaf is left off the loose levels where
/// f's level there is tight and the leaf arm found g's level holding each of
/// its live products for every label: every level the result does not list
/// is tight, and the leaf arm's held products leave some off.
#[test]
fn a_level_whose_products_the_leaf_arm_finds_held_for_every_label_is_tight() {
    let before = held_tight_levels();
    for nvars in [10, 14] {
        for (what, _, f, g) in cases(nvars, 0x4e1d_7a11 + u64::from(nvars)) {
            let eng = Engine::new();
            let (mut f, mut g) = (f, g);
            eng.minimize(&mut f).unwrap();
            eng.minimize(&mut g).unwrap();
            // A minimized diagram has no loose level.
            f.dirty.set_loose(Some(Vec::new()));
            g.dirty.set_loose(Some(Vec::new()));
            for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
                let (out, _) = conjoin(a, b);
                if out.is_zero() {
                    continue;
                }
                let loose = out.dirty.loose().expect("both operands' loose levels are known");
                for t in unnamed_levels(&out) {
                    assert!(loose.contains(&t), "{what}, {order}: level {t} is not listed loose but holds a node its parent does not name");
                }
            }
        }
    }
    let held = held_tight_levels() - before;
    eprintln!("levels left off the loose levels for the leaf arm's held products: {held}");
    assert!(held > 0, "expected levels left off the loose levels for the leaf arm's held products");
}
