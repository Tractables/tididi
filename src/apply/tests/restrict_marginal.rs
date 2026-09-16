//! Restriction where either side carries a marginal level.
//!
//! Fixtures come from `crate::test_helpers`, re-exported by the parent.

use super::*;

use crate::Engine;

/// Restrict contract checked against the TRUE marginal `care`.
///
/// To compare against the TRUE marginal care we must be able to COUNT `f∧care` — but
/// `marginal²` is unsupported. So keep the regions DISJOINT: `f` is marginal at region
/// `R_f` and FREE over care's regions; `care` is marginal at two disjoint regions and
/// FREE over `R_f`. Every conjoin is then `identity∧marginal` / `marginal∧identity`,
/// never `marginal²`, so `(f∧care).model_count()?` is well-defined via the real apply.
/// Contract: the marginal `#(f∧care)` is invariant under the prune.
///
/// The contract holds for DISJOINT multi-region marginal care. OVERLAPPING
/// regions (`f` AND `care` marginal at the same node) have no pure-`restrict_to_care`
/// reference — the joint count over the summed region is unrecoverable — so
/// this guards the disjoint regime only.
#[test]
fn restrict_true_marginal_care_multiregion_difftest() {
    let eng = Engine::new();
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

    let mut rng = Lcg::new(0x2545_f491_4f6c_dd1d);

    let mut checked = 0;
    let mut pruned = 0;
    let mut violations = 0;
    let mut first_violation: Option<(usize, String, String)> = None;
    // Fold-step (downstream conjoin+marginalize_levels) repro counters.
    let only_pruned_fold = true;
    let mut fold_count_fail = 0;
    let mut first_fold_fail: Option<String> = None;
    for case in 0..800 {
        let mut fm = rand_conj_over(&vtree, &f_vars, 6, 3, false, &mut rng);
        if fm.is_zero() {
            continue;
        }
        marginalize_subtree(&mut fm, r_f);
        fm.minimize().unwrap();

        let mut care = rand_conj_over(&vtree, &care_vars, 6, 3, false, &mut rng);
        if count_is_zero(&care) {
            continue;
        }
        marginalize_subtree(&mut care, r_c1);
        marginalize_subtree(&mut care, r_c2);
        care.minimize().unwrap();

        // Need both of care's marginal regions to survive (multi-region).minimize().unwrap().
        let n_marginal = (0..vtree.num_nodes())
            .filter(|&i| {
                matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && care.levels[i].is_marginal()
            })
            .count();
        if n_marginal < 2 {
            continue;
        }
        // restrict_to_care only engages on a shared function root.
        if care.output.vtree != fm.output.vtree {
            continue;
        }

        // TRUE marginal care on both sides (disjoint regions ⇒ supported conjoin).
        let before = (and2(&fm, &care)).model_count().unwrap();
        let g = (fm.clone()).restrict_to_care(care.clone()).unwrap().into_tdd();
        let after = (and2(&g, &care)).model_count().unwrap();
        if before != after {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some((case, before.to_string(), after.to_string()));
            }
        }
        let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
        assert!(gp <= fp, "restrict_to_care larger than f at case {case}: {gp} > {fp}");
        if gp < fp {
            pruned += 1;
        }

        // ── FOLD STEP ─────────────────────────────────────────────────────
        // The restrict_to_care contract (#(f∧care)) holds above, yet production panics
        // when the SHRUNK operand feeds the fold's apply_and THEN marginalize_levels.
        // Mimic `merge_one_pair`: conjoin g with the care operand, then sum out
        // the now-private `live` vars via the PRODUCTION batch marginalizer, and
        // VALIDATE STRUCTURE (not just the count) at each stage — the dangling
        // marginal-side ref the contract checks miss. care∧g == care∧fm (restrict_to_care
        // contract), so the post-marginalize_levels counts must match; a structural
        // failure / OOB / mismatch on the g-path (while the fm-path stays clean)
        // localizes the defect to restrict_to_care's marginal-f output feeding the fold.
        if only_pruned_fold {
            if gp == fp {
                checked += 1;
                continue; // exercise the fold only where restrict_to_care actually shrank
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
            crate::marginal::marginalize_batch(&eng, &mut prod_g, &targets, &vtree).expect("no wall is installed in a test");
            crate::marginal::marginalize_batch(&eng, &mut prod_f, &targets, &vtree).expect("no wall is installed in a test");
            // `model_count` is the query that reads out of bounds on a corrupt
            // fold structure, and the count must be invariant
            // (care∧g == care∧fm): a panic here is the bug, a mismatch a silent
            // miscount. `validate_vtree_structure` cannot be used after
            // marginalizing — it treats a legitimate inline marginal ref as a
            // violation.
            let cg = prod_g.model_count().unwrap();
            let cf = prod_f.model_count().unwrap();
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
        "fold conjoin+marginalize_levels MISCOUNTED on the restrict_to_care-shrunk operand in \
         {fold_count_fail}/{checked} cases (first {first_fold_fail:?})"
    );
    assert!(checked >= 30, "too few multi-region cases exercised: {checked}");
    assert!(
        pruned > 0,
        "reduction never pruned — the path is not exercised (test would pass vacuously)"
    );
    assert_eq!(
        violations, 0,
        "restrict_to_care changed #(f∧care) against the TRUE multi-region marginal \
         care in {violations}/{checked} cases (first {first_violation:?})"
    );
}

/// Restrict against care that is marginal at the same regions as `f`.
///
/// Restriction reads a care level that is marginal as ⊤ for liveness, so the
/// care it effectively applies is `∃R. care0` — the structural care with every
/// region's variables forgotten — and that projection is a structural diagram,
/// so `#(· ∧ care_proj)` is a supported conjunction on both sides even though
/// `f` is marginal over the same regions. The contract is
/// `#(g ∧ care_proj) == #(fm ∧ care_proj)`, plus: restricting against the
/// projection itself must produce the same subgraph.
fn restrict_marginal_care_same_regions(
    seed: u64,
    nvars: u32,
    want_regions: usize,
    min_checked: usize,
) {
    use crate::vtree::{VtreeIdx, VtreeNode};
    let vtree = Arc::new(Vtree::balanced(nvars));

    let vars_under = |root: VtreeIdx| -> Vec<VarId> {
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
                    s.push(var);
                }
            }
        }
        s
    };
    // Pairwise-disjoint non-root internal subtrees, smallest first.
    let mut internals: Vec<(VtreeIdx, Vec<VarId>)> = (0..vtree.num_nodes())
        .filter(|&vi| {
            matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) && vi != vtree.root().idx()
        })
        .map(|vi| (VtreeIdx(vi as u32), vars_under(VtreeIdx(vi as u32))))
        .collect();
    internals.sort_by_key(|(_, s)| s.len());
    let mut regions: Vec<(VtreeIdx, Vec<VarId>)> = Vec::new();
    for (r, s) in internals {
        if regions.iter().all(|(_, cs)| cs.iter().all(|v| !s.contains(v))) {
            regions.push((r, s));
            if regions.len() == want_regions {
                break;
            }
        }
    }
    assert_eq!(regions.len(), want_regions, "need {want_regions} disjoint internal subtrees");
    let region_vars: Vec<VarId> = regions.iter().flat_map(|(_, s)| s.iter().copied()).collect();

    let mut rng = Lcg::new(seed);
    let marginal_all = |t: &mut Tdd| {
        for (r, _) in &regions {
            marginalize_subtree(t, *r);
        }
        t.minimize().unwrap();
    };
    let n_marginal_internal = |t: &Tdd| {
        (0..vtree.num_nodes())
            .filter(|&i| {
                matches!(*vtree.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) && t.levels[i].is_marginal()
            })
            .count()
    };

    let mut checked = 0;
    let mut pruned = 0;
    for case in 0..800 {
        let f = rand_conj(&vtree, nvars, 6, 3, false, &mut rng);
        if f.is_zero() {
            continue;
        }
        let mut fm = f.clone();
        marginal_all(&mut fm);
        if n_marginal_internal(&fm) < want_regions {
            continue;
        }
        let care0 = rand_conj(&vtree, nvars, 6, 3, false, &mut rng);
        if count_is_zero(&care0) {
            continue;
        }
        let mut care = care0.clone();
        marginal_all(&mut care);
        if n_marginal_internal(&care) < want_regions {
            continue;
        }
        let care_proj = (care0).clone().exists_vars(&region_vars).unwrap();

        let before = (and2(&fm, &care_proj)).model_count().unwrap();
        let g = (fm.clone()).restrict_to_care(care.clone()).unwrap().into_tdd();
        let after = (and2(&g, &care_proj)).model_count().unwrap();
        assert_eq!(before, after, "restrict_to_care changed #(f ∧ ∃R.care) at case {case}: {before} != {after}");

        let g_proj = (fm.clone()).restrict_to_care(care_proj.clone()).unwrap().into_tdd();
        assert_eq!(
            normalized_levels(&g),
            normalized_levels(&g_proj),
            "marginal care and its projection restricted f differently at case {case}"
        );

        let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
        assert!(gp <= fp, "restrict_to_care larger than f at case {case}: {gp} > {fp}");
        if gp < fp {
            pruned += 1;
        }
        checked += 1;
    }
    assert!(checked >= min_checked, "too few cases exercised: {checked}");
    assert!(pruned > 0, "restrict_to_care never pruned a marginal diagram");
}

/// `f` and `care` marginal over one shared region.
#[test]
fn restrict_marginal_care_single_region_difftest() {
    restrict_marginal_care_same_regions(0x9e37_79b9_7f4a_7c15, 6, 1, 50);
}

/// `f` and `care` marginal over two shared disjoint regions.
#[test]
fn restrict_marginal_care_two_regions_difftest() {
    restrict_marginal_care_same_regions(0xd1b5_4a32_d192_ed03, 8, 2, 30);
}

/// Restrict contract on a MARGINAL `f`, the production orientation the
/// multi-region test above does not exercise (that one marginalizes `care`,
/// leaving `f` free). The marginalized-pool restrict_to_care shrinks members
/// that have themselves been marginalized — `f.restrict_to_care(care)?`
/// with `f` carrying summed-out (marginal) levels and `care` non-marginal —
/// then conjoins the shrunk `g` with `care`. The contract `g ∧ care ==
/// f ∧ care` must hold per-MODEL-COUNT for that marginal `f`.
///
/// The liveness oracle inside `restrict_to_care` prunes a non-marginal node
/// of `f` when the EMIT=false conjoin marks it dead. A non-marginal node
/// routing into a marginal subtree is always alive — marginal counts are >0,
/// so every marginal node is alive. If the oracle killed such a node, `g`
/// would lose models present in `f ∧ care` → miscount.
///
/// This GUARD asserts no such miscount across synthesized marginal-`f`
/// configs (random `f` over all vars, a random SCATTERED subset summed out,
/// `care` over the complement). It passes — restrict_to_care is sound for every
/// marginal-`f` shape reachable by this synthesis. The production miscount
/// (proven on the blow-up instances) needs operand structure this synthesis
/// does not reach (a large complex `care` against a tiny marginal `f` under
/// the in-fold vtree graft); reproducing it needs captured real operands,
/// not synthesis. Kept as the regression guard for the sound regime.
#[test]
fn restrict_marginal_f_difftest() {
    let eng = Engine::new();
    use crate::vtree::VtreeIdx;
    let nvars = 8u32;
    let vtree = Arc::new(Vtree::balanced(nvars));

    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15);

    let mut checked = 0usize;
    let mut pruned = 0usize;
    let mut fail = 0usize;
    let mut first_fail: Option<String> = None;
    for _trial in 0..600 {
        // f constrains all vars; then sum out a RANDOM SCATTERED subset — the
        // production private-var marginalize_levels interleaves marginal and non-marginal
        // levels (unlike a contiguous subtree, where all marginal levels sit at the
        // bottom). That interleaving is what exercises a non-marginal node sitting
        // BELOW a marginal one.
        let all_vars: Vec<u32> = (0..nvars).collect();
        let mut f = rand_conj_over(&vtree, &all_vars, 6, 3, false, &mut rng);
        if f.is_zero() {
            continue;
        }
        let marginal_vars: Vec<u32> = (0..nvars).filter(|_| rng.coin()).collect();
        if marginal_vars.is_empty() || marginal_vars.len() == nvars as usize {
            continue;
        }
        let care_vars: Vec<u32> = (0..nvars).filter(|v| !marginal_vars.contains(v)).collect();
        if care_vars.is_empty() {
            continue;
        }
        let mut targets: Vec<VtreeIdx> =
            marginal_vars.iter().map(|&v| vtree.leaf_of(VarId(v)).expect("the vtree carries this variable")).collect();
        targets.sort_by_key(|vi| vtree.topo_pos(*vi));
        crate::marginal::marginalize_batch(&eng, &mut f, &targets, &vtree).expect("no wall is installed in a test");
        // care constrains only NON-marginal vars ⇒ identity at f's marginal levels,
        // so the conjoin stays legal and #(f∧care) is well-defined.
        let care = rand_conj_over(&vtree, &care_vars, 6, 3, false, &mut rng);
        if care.is_zero() {
            continue;
        }
        let prod_f = and2(&f, &care);
        let g = (f.clone()).restrict_to_care(care.clone()).unwrap().into_tdd();
        if reachable_pairs(&g) < reachable_pairs(&f) {
            pruned += 1;
        }
        let prod_g = and2(&g, &care);
        let cf = prod_f.model_count().unwrap();
        let cg = prod_g.model_count().unwrap();
        if cf != cg {
            fail += 1;
            if first_fail.is_none() {
                first_fail = Some(format!(
                    "marginal_vars={marginal_vars:?} care_vars={care_vars:?}: \
                     #(f∧care)={cf} != #(g∧care)={cg}"
                ));
            }
        }
        checked += 1;
    }
    println!(
        "marginal-f scattered.restrict_to_care().unwrap(): {checked} checked, {pruned} pruned, {fail} miscount; first={first_fail:?}"
    );
    assert!(checked >= 30, "too few marginal-f cases exercised: {checked}");
    assert!(
        pruned > 0,
        "reduction never pruned a marginal f — path not exercised (test would pass vacuously)"
    );
    assert_eq!(
        fail, 0,
        "restrict_to_care MISCOUNTED #(f∧care) on a MARGINAL f in {fail}/{checked} \
         cases (first {first_fail:?}) — the liveness oracle killed a non-marginal node \
         that routes into an (always-alive) marginal subtree"
    );
}
