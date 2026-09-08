//! Restriction where either side carries a marginal level.
//!
//! Sibling of `unary_tests.rs`, which holds the fixtures these read.

use super::*;

use crate::engine::Limits;

/// Restrict contract checked against the TRUE marginal `care`.
///
/// To compare against the TRUE marginal care we must be able to COUNT `f∧care` — but
/// `marginal²` is unsupported. So keep the regions DISJOINT: `f` is marginal at region
/// `R_f` and FREE over care's regions; `care` is marginal at TWO disjoint regions and
/// FREE over `R_f`. Every conjoin is then `identity∧marginal` / `marginal∧identity`,
/// never `marginal²`, so `model_count(f∧care)` is well-defined via the real apply.
/// Contract: the marginal `#(f∧care)` is invariant under the prune.
///
/// The contract holds for DISJOINT multi-region marginal care. OVERLAPPING
/// regions (`f` AND `care` marginal at the SAME node) have no pure-`restrict`
/// reference — the joint count over the summed region is unrecoverable — so
/// this guards the disjoint regime only.
#[test]
fn restrict_true_marginal_care_multiregion_difftest() {
    let lim = Limits::new();
        use crate::test_helpers::reachable_pairs;
    use crate::test_helpers::marginalize_subtree;
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
            let cl = clause_to_tdd(&vtree, &crate::test_helpers::clause(&lits));
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
        crate::reduce::minimize(&mut fm);

        let mut care = rand_over(&mut rng, &care_vars);
        if count_is_zero(&lim, &care) {
            continue;
        }
        marginalize_subtree(&mut care, r_c1);
        marginalize_subtree(&mut care, r_c2);
        crate::reduce::minimize(&mut care);

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
        let g = crate::apply::restrict(&fm, care.clone(), crate::apply::CareCanonical::No).into_tdd(&fm);
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

        // ── FOLD STEP ─────────────────────────────────────────────────────
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
                .map(|&v| vtree.leaf_of(VarId(v)).expect("the vtree carries this variable"))
                .collect();
            targets.sort_by_key(|vi| vtree.topo_pos(*vi));
            if targets.is_empty() {
                checked += 1;
                continue;
            }
            crate::marginal::marginalize_batch(&lim, &mut prod_g, &targets, &vtree).expect("no wall is installed in a test");
            crate::marginal::marginalize_batch(&lim, &mut prod_f, &targets, &vtree).expect("no wall is installed in a test");
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
         care in {violations}/{checked} cases (first {first_violation:?})"
    );
}

/// `f` and `care` marginal over ONE shared region.
#[test]
fn restrict_marginal_care_single_region_difftest() {
    let lim = Limits::new();
    restrict_marginal_care_same_regions(&lim, 0x9e37_79b9_7f4a_7c15, 6, 1, 50);
}

/// `f` and `care` marginal over TWO shared disjoint regions.
#[test]
fn restrict_marginal_care_two_regions_difftest() {
    let lim = Limits::new();
    restrict_marginal_care_same_regions(&lim, 0xd1b5_4a32_d192_ed03, 8, 2, 30);
}

/// Restrict contract on a MARGINAL `f`, the production orientation the
/// multi-region test above does NOT exercise (that one marginalizes `care`,
/// leaving `f` free). The marginalized-pool restrict shrinks members
/// that have themselves been marginalized — `crate::apply::restrict(f, care)`
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
    let lim = Limits::new();
        use crate::test_helpers::reachable_pairs;
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
            let cl = clause_to_tdd(&vtree, &crate::test_helpers::clause(&lits));
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
            marg_vars.iter().map(|&v| vtree.leaf_of(VarId(v)).expect("the vtree carries this variable")).collect();
        targets.sort_by_key(|vi| vtree.topo_pos(*vi));
        crate::marginal::marginalize_batch(&lim, &mut f, &targets, &vtree).expect("no wall is installed in a test");
        // care constrains only NON-marginal vars ⇒ identity at f's marginal levels,
        // so the conjoin stays legal and #(f∧care) is well-defined.
        let care = rand_over(&mut rng, &care_vars);
        if care.is_zero() {
            continue;
        }
        let prod_f = and2(&f, &care);
        let g = crate::apply::restrict(&f, care.clone(), crate::apply::CareCanonical::No).into_tdd(&f);
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
