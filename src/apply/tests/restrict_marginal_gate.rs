//! The gate that refuses a restriction under a marginal ancestor.
//!
//! Sibling of `restrict_marginal.rs`.

use super::*;

use crate::engine::Engine;

/// P4 soundness gate. The ancestor-down-restriction prototype restricts a completed
/// bottom-up accumulator under cares built from pending ancestor clauses. That
/// operand has the VANILLA-COMPILE marginal shape, which differs from the
/// miscounting shape (a marginal operand under the in-fold vtree GRAFT,
/// where operand and care carry different marginal-level patterns over shared
/// structure): here `b` is a bottom-up accumulator carrying marginal levels
/// from DESCENDANT forgets — V2 summed out as a CONTIGUOUS subtree (marginal
/// levels at the bottom), forgotten with the production `marginalize_batch` —
/// and `care` is a NON-marginal diagram built purely from clauses over V1, whose
/// support is DISJOINT from the forgotten V2 (a pending ancestor clause can
/// never mention a var already forgotten below). Both share the global root and
/// the same vtree `Arc` — no graft, so restrict takes its same-root fast path
/// (never the marginal-lift fallback).
///
/// Complements `restrict_marginal_f_difftest` (which forgets a SCATTERED subset,
/// interleaving marginal/non-marginal levels): this one pins the contiguous
/// descendant-forget shape P4 actually feeds restrict, and it counts the
/// `Restricted::Shrunk` variant directly (not just a reachable-pair drop) plus
/// deterministic contradiction cases, so it can never pass vacuously.
///
/// Contract: `model_count(crate::apply::restrict(b,care) ∧ care) == model_count(b ∧ care)` —
/// the exact invariant P4 relies on to down-restrict an accumulator in place.
/// (`model_count` on a marginal diagram returns the summed count; that is precisely
/// the semantics that must be preserved.)
#[test]
fn restrict_ancestor_marginal_operand_gate() {
    let eng = Engine::new();
        use crate::apply::{CareCanonical, Restricted};
    use crate::test_helpers::reachable_pairs;
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
    let marginal_root = (0..vtree.num_nodes())
        .filter(|&vi| {
            matches!(*vtree.node(VtreeIdx(vi as u32)), VtreeNode::Internal { .. }) && vi != vtree.root().idx()
        })
        .map(|vi| VtreeIdx(vi as u32))
        .find(|&r| {
            let n = support_of(r).len();
            n >= 1 && (nvars as usize - n) >= 2
        })
        .expect("balanced(8) has a small non-root internal subtree");
    let v2: Vec<u32> = support_of(marginal_root);
    let v1: Vec<u32> = (0..nvars).filter(|v| !v2.contains(v)).collect();
    let mut v2_targets: Vec<VtreeIdx> =
        v2.iter().map(|&v| vtree.leaf_of(VarId(v)).expect("the vtree carries this variable")).collect();
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
            let mut literals: Vec<(u32, bool)> = Vec::new();
            for _ in 0..width {
                let v = vars[(rng() as usize) % vars.len()];
                let pol = rng().is_multiple_of(2);
                if literals.iter().any(|(u, _)| *u == v) {
                    continue;
                }
                literals.push((v, pol));
            }
            literals.sort_by_key(|&(v, _)| v);
            let cl = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&literals));
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
        crate::marginal::marginalize_batch(&eng, t, &v2_targets, &vtree).expect("no wall is installed in a test");
    };
    // The production `marginalize_batch` marks the forgotten LEAF levels marginal
    // (a contiguous subtree summed out ⇒ its leaf levels carry the marginal counts),
    // so check any level, not just internal ones.
    let has_marginal = |t: &Tdd| -> bool {
        (0..vtree.num_nodes()).any(|i| t.levels[i].is_marginal())
    };

    let mut checked = 0usize;
    let mut shrunk = 0usize; // Restricted::Shrunk outcomes (non-vacuity)
    let mut false_out = 0usize; // Restricted::Unsatisfiable outcomes (care ⇒ ⊥)
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
            let out = crate::apply::restrict(b, care.clone(), CareCanonical::No);
            match out {
                Restricted::Shrunk(_) => shrunk += 1,
                Restricted::Unsatisfiable(_) => false_out += 1,
                Restricted::Unchanged(_) => {}
            }
            let g = out.into_tdd();
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
    // not the production shape. p,q ∈ V1; z,z2 ∈ V2 keep the marginal level live. ──
    let p = v1[0];
    let q = v1[1];
    let z = v2[0];
    let z2 = *v2.last().unwrap();
    {
        // b = (p∨q) ∧ (p∨z) ∧ (q∨z2) ; forgetting V2 keeps (p,q) entangled with a
        // surviving marginal level. care=(¬p) forces p=false ⇒ prunes the p-branch.
        let f = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(p, true), (q, true)]));
        let g = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(p, true), (z, true)]));
        let c3 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(q, true), (z2, true)]));
        let mut b = and2(&and2(&f, &g), &c3);
        forget_v2(&mut b);
        assert!(has_marginal(&b), "deterministic case 1 lost its marginal level");
        let care = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(p, false)])); // ¬p
        check(&b, &care, "det1", &mut fail, &mut first_fail);
    }
    {
        // b = (¬p∨q) ∧ (p∨z) ∧ (q∨z2) ; care=(¬q) forces q=false ⇒ ¬p, prunes branches.
        let f = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(p, false), (q, true)]));
        let g = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(p, true), (z, true)]));
        let c3 = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(q, true), (z2, true)]));
        let mut b = and2(&and2(&f, &g), &c3);
        forget_v2(&mut b);
        assert!(has_marginal(&b), "deterministic case 2 lost its marginal level");
        let care = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(q, false)])); // ¬q
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
        if !has_marginal(&b) {
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
