//! Integration tests for the unary transform family (project / condition /
//! restrict) together with the `query::support` structural
//! queries. Relocated verbatim from the former `project.rs`: the `mod tests`
//! body below is byte-for-byte unchanged, and the sibling symbols it reaches
//! via `super::` are re-bound into this file's module scope by the `use` block
//! below (repointed to the post-split module paths).

use crate::apply::project::{
    project_var, project_vars, Projection,
    POS, NEG, ONE,
};
use crate::apply::condition_var;
use crate::apply::restrict::{restrict, CareCanonical};
use crate::test_helpers::{reachable_pairs, support_mask};
use crate::diagram::{ZERO, NodeIdx};

mod tests {
    use crate::engine::Engine;
    use std::sync::Arc;

    use num_bigint::BigUint;

    use super::{project_var, project_vars, Projection};
    use super::support_mask;
    use crate::apply::apply_and;
    use crate::build::{clause_to_tdd, constant_one, constant_zero};
    use crate::query::model_count;
    use crate::vtree::{VarId, Vtree};

    use super::condition_var;
    use crate::apply::apply_or;

    // Regression: a unit-forced variable must be detectable by conditioning it to
    // the opposite value and finding the result UNSAT. NOTE: a false diagram is not
    // always `is_zero()` — conditioning or an apply can leave `model_count == 0` in a
    // non-canonical form (output node still has pairs). `condition_*` now
    // canonicalizes its own output, but an unarmed apply does not, so UNSAT detection
    // on a derived diagram uses `model_count == 0`, not `Tdd::is_zero()`. The
    // segment-restrict driver's `forced_literals` depends on this.
    fn count_is_zero(_eng: &Engine, t: &Tdd) -> bool {
        model_count(t) == BigUint::from(0u32)
    }

    // ── restrict (generalized cofactor) ───────────────────────────────────────
    fn and2(a: &Tdd, b: &Tdd) -> Tdd {
        let a = a.clone();
        let b = b.clone();
        apply_and(a, b)
    }

    // f1 == f2 as Boolean functions over the shared vtree.
    fn equiv(eng: &Engine, a: &Tdd, b: &Tdd) -> bool {
        use crate::apply::negate;
        let a_not_b = and2(a, &negate(b.clone()));
        let not_a_b = and2(&negate(a.clone()), b);
        count_is_zero(eng, &a_not_b) && count_is_zero(eng, &not_a_b)
    }
    // Negate-free equivalence: `a∧b ⊆ a` and `a∧b ⊆ b` always, so equal model
    // counts on all three force `a == b` as sets. Uses only apply_and/model_count
    // (the restrict output is a valid diagram but not in `negate`'s t-full/complete
    // form, so the negate-based `equiv` above is the wrong oracle for it).
    fn equiv_nf(_eng: &Engine, a: &Tdd, b: &Tdd) -> bool {
        let ca = model_count(a);
        let cb = model_count(b);
        ca == cb && model_count(&and2(a, b)) == ca
    }

    // ── restrict: direct semantics evaluator (apply-independent ground truth) ──
    //
    // Walks the diagram by the diagram denotation `⋃ᵢ aᵢ×bᵢ` and evaluates a single
    // assignment. Independent of apply/model_count, so brute-forcing it over all
    // assignments is a soundness oracle that shares no machinery with the operator
    // OR with `equiv`.
    fn eval_label(l: super::NodeIdx, x: bool) -> bool {
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
    fn eval_node(t: &Tdd, v: crate::vtree::VtreeIdx, local: super::NodeIdx, asn: &[bool]) -> bool {
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
    fn assert_restrict_ok(_eng: &Engine, f: &Tdd, c: &Tdd, nvars: u32) {
        use super::{reachable_pairs, restrict};
        use crate::check::{check_all_fast, check_determinism};
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
        crate::reduce::minimize(&mut gm);
        check_all_fast(&gm, "restrict-output");
        check_determinism(&gm).expect("restrict output must be deterministic (mutex pairs)");
        // (2) never larger than f — restrict returns a strict subgraph of f.
        assert!(
            reachable_pairs(&g) <= reachable_pairs(f),
            "restrict grew the diagram beyond f"
        );
    }

    // Re-home a diagram that depends only on vars under one child of its (global)
    // root to be rooted at that child — a genuinely low-rooted Boolean diagram.
    // `build`/`apply` always root at the global vtree root, so re-homing is the
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
        crate::diagram::Tdd::from_levels_unchecked(
            t.vtree.clone(),
            t.levels.clone(),
            crate::diagram::TddNodeId { vtree: child, local },
        )
    }

    // `marginal_constraint_lifted_to_free_indicator` moved to
    // tests/tdd_projection_compile.rs (`marginal_lift_indicator` mod) — it needs
    // CNF parsing and compilation, neither of which lives in this crate.

    /// Restrict against care that is marginal at the same regions as `f`.
    ///
    /// `restrict` reads a care level that is marginal as ⊤ for liveness, so the
    /// care it effectively applies is `∃R. care0` — the structural care with
    /// every region's variables forgotten — and that projection is a
    /// structural diagram, so `#(· ∧ care_proj)` is a supported conjoin on
    /// both sides even though `f` is marginal over the same regions. The
    /// contract is `#(g ∧ care_proj) == #(fm ∧ care_proj)`, plus: restricting
    /// against the projection itself must produce the same subgraph.
    fn restrict_marginal_care_same_regions(eng: &Engine, seed: u64, nvars: u32, want_regions: usize, min_checked: usize) {
        use super::{reachable_pairs, restrict};
        use crate::test_helpers::{marginalize_subtree, normalized_levels};
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

        let mut state: u64 = seed;
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
                let mut literals: Vec<(u32, bool)> = Vec::new();
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    let pol = rng().is_multiple_of(2);
                    if literals.iter().any(|(u, _)| *u == v) {
                        continue;
                    }
                    literals.push((v, pol));
                }
                literals.sort_by_key(|&(v, _)| v);
                let cl = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&literals));
                acc = Some(match acc {
                    None => cl,
                    Some(a) => and2(&a, &cl),
                });
            }
            acc.unwrap()
        };
        let marginal_all = |t: &mut Tdd| {
            for (r, _) in &regions {
                marginalize_subtree(t, *r);
            }
            crate::reduce::minimize(t);
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
            let f = rand_fn(&mut rng);
            if f.is_zero() {
                continue;
            }
            let mut fm = f.clone();
            marginal_all(&mut fm);
            if n_marginal_internal(&fm) < want_regions {
                continue;
            }
            let care0 = rand_fn(&mut rng);
            if count_is_zero(eng, &care0) {
                continue;
            }
            let mut care = care0.clone();
            marginal_all(&mut care);
            if n_marginal_internal(&care) < want_regions {
                continue;
            }
            let care_proj = project_vars(&care0, &region_vars, Projection::Automatic);

            let before = model_count(&and2(&fm, &care_proj));
            let g = restrict(&fm, care.clone(), super::CareCanonical::No).into_tdd(&fm);
            let after = model_count(&and2(&g, &care_proj));
            assert_eq!(before, after, "restrict changed #(f ∧ ∃R.care) at case {case}: {before} != {after}");

            let g_proj = restrict(&fm, care_proj.clone(), super::CareCanonical::No).into_tdd(&fm);
            assert_eq!(
                normalized_levels(&g),
                normalized_levels(&g_proj),
                "marginal care and its projection restricted f differently at case {case}"
            );

            let (gp, fp) = (reachable_pairs(&g), reachable_pairs(&fm));
            assert!(gp <= fp, "restrict larger than f at case {case}: {gp} > {fp}");
            if gp < fp {
                pruned += 1;
            }
            checked += 1;
        }
        assert!(checked >= min_checked, "too few cases exercised: {checked}");
        assert!(pruned > 0, "restrict never pruned a marginal diagram");
    }

    // `vnode_split_with_marginal_child_is_sound` moved to
    // tests/tdd_projection_compile.rs (`vnode_split_marginal_child` mod) — it
    // needs CNF parsing and compilation, neither of which lives in this
    // crate.

    // `streaming_projected_matches_brute_force_pmc` and
    // `streaming_projected_free_and_empty_vars` moved to
    // tests/tdd_projection_compile.rs (`streaming_pmc_component_spec` /
    // `streaming_pmc_free_and_empty_vars` mods) — they need CNF parsing and
    // preprocessing and compilation, none of which lives in this crate.

    // ── brute-force PROJECTED-model-counting (PMC) oracle ────────────────────
    //
    // Guards the soundness identity used by the projected-count path:
    //
    //   PMC = model_count(project_vars(f, projected∩vtree)) >> |projected∩vtree|
    //
    // where `f = compile_cnf(formula, vtree)`, the projection is the real
    // OR-cofactor `project_vars` (not a leaf-2^k shortcut), and `free show vars`
    // (show vars absent from every clause / vtree) are zero here because we use
    // `Vtree::balanced(n)`, which places all n vars in the vtree. Hence
    // `projected∩vtree` = all non-show vars and `free show vars` = 0.

    use crate::diagram::Tdd;

    // ── project_var_scoped: direct unit tests ────────────────────────────────

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
    // preprocessing and compilation, none of which lives in this crate.

    // Pair count at a diagram's root (output) node — the "width" the worked example tracks.
    fn root_width(t: &Tdd) -> usize {
        if t.is_zero() {
            return 0;
        }
        t.levels[t.output.vtree.0 as usize].pair_count_at(t.output.local.0 as usize)
    }

    // A cube (conjunction of literals) as a diagram.
    fn cube(vtree: &Arc<Vtree>, literals: &[(u32, bool)]) -> Tdd {
        let eng = &crate::engine::Engine::new();
        let mut acc = clause_to_tdd(eng, vtree, &crate::test_helpers::clause(&[literals[0]]));
        for &l in &literals[1..] {
            acc = and2(&acc, &clause_to_tdd(eng, vtree, &crate::test_helpers::clause(&[l])));
        }
        acc
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
        let eng = &crate::engine::Engine::new();
        let mut acc: Option<Tdd> = if span && nvars >= 2 {
            Some(clause_to_tdd(eng, vtree, &crate::test_helpers::clause(&[(0, true), (nvars - 1, true)])))
        } else {
            None
        };
        let nclauses = 1 + (rng() % nclauses_max) as usize;
        for _ in 0..nclauses {
            let width = 1 + (rng() % width_max) as usize;
            let mut literals: Vec<(u32, bool)> = Vec::new();
            for _ in 0..width {
                let v = (rng() % nvars as u64) as u32;
                let pol = rng().is_multiple_of(2);
                if literals.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                literals.push((v, pol));
            }
            literals.sort_by_key(|&(v, _)| v);
            literals.dedup_by_key(|&mut (v, _)| v);
            let cl = clause_to_tdd(eng, vtree, &crate::test_helpers::clause(&literals));
            acc = Some(match acc {
                None => cl,
                Some(a) => and2(&a, &cl),
            });
        }
        acc.unwrap()
    }

    #[path = "project.rs"]
    mod project;
    #[path = "restrict.rs"]
    mod restrict;
    #[path = "restrict_marginal.rs"]
    mod restrict_marginal;
    #[path = "restrict_marginal_gate.rs"]
    mod restrict_marginal_gate;
    #[path = "restrict_scaling.rs"]
    mod restrict_scaling;
    #[path = "support.rs"]
    mod support;
}
