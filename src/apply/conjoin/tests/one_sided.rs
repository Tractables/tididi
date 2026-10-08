//! Levels with one multi-pair operand that build the dead-pair masks,
//! checked against the row loop without masks they replace: the same
//! diagram, and less work where a mask culls a cell.

use super::*;
use crate::Engine;
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;

/// The conjunction of `clauses` on `vtree`.
fn cnf(vtree: &Arc<Vtree>, clauses: &[[i32; 2]], ternary: &[[i32; 3]]) -> Tdd {
    let eng = Engine::new();
    let mut f = Tdd::one(vtree);
    for c in clauses {
        f = eng.and(f, Tdd::clause(vtree, *c).unwrap()).unwrap();
    }
    for c in ternary {
        f = eng.and(f, Tdd::clause(vtree, *c).unwrap()).unwrap();
    }
    f
}

/// Conjoin `f` and `g` on a fresh engine, with the one-sided masks or
/// without them: the result and the work units it took.
fn conjoin(f: &Tdd, g: &Tdd, masks: bool) -> (Tdd, u64) {
    let eng = Engine::new();
    let run = || eng.and(f.clone(), g.clone()).unwrap();
    let out = if masks { run() } else { one_sided_masks_off(run) };
    (out, eng.limits().work_units())
}

/// On a balanced vtree over 32 variables the node over 1 to 16 has children
/// over 1 to 8 and 9 to 16. Under `x_{24+k}`, `f` says `x_k ∧ (x_{3+k} ⊕
/// x_{8+k})` for k from 1 to 3: there it has a node for each set of the three
/// constraints, multi-pair where the set is not empty, and every prime of a
/// node holds `x_k` for each k of its set. `g` says `x_1 ↔ x_17`,
/// `x_2 ↔ x_18`, `x_9 ↔ x_19` and `x_10 ↔ x_20`: there it has 16 nodes of
/// one pair. Their grid there has over 64 cells, and a cell whose g prime
/// denies an `x_k` its f node needs has every candidate dead on the left.
/// Both ways round, so that the multi-pair operand is f once and g once.
#[test]
fn a_one_sided_level_masks_and_culls() {
    let vtree = Arc::new(Vtree::balanced(32));
    let mut binary = Vec::new();
    let mut ternary = Vec::new();
    for k in 1..=3 {
        let (on, a, b, c) = (24 + k, k, 3 + k, 8 + k);
        binary.push([-on, a]);
        ternary.push([-on, b, c]);
        ternary.push([-on, -b, -c]);
    }
    let f = cnf(&vtree, &binary, &ternary);
    let equalities: Vec<[i32; 2]> = [(1, 17), (2, 18), (9, 19), (10, 20)]
        .into_iter()
        .flat_map(|(a, b)| [[-a, b], [a, -b]])
        .collect();
    let g = cnf(&vtree, &equalities, &[]);
    for (x, y) in [(&f, &g), (&g, &f)] {
        let before = one_sided_masked_levels();
        let (oracle, oracle_work) = conjoin(x, y, false);
        assert_eq!(one_sided_masked_levels(), before, "the oracle built one-sided masks");
        let (out, work) = conjoin(x, y, true);
        assert_canonical(&out);
        assert!(is_self_conjunction(&out, &oracle), "the masks built another diagram");
        assert!(one_sided_masked_levels() > before, "no level built one-sided masks");
        assert!(work < oracle_work, "the masks culled no cell: {work} work units against {oracle_work}");
    }
}
