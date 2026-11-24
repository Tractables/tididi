//! Canonicity checking: the fixpoint of every pass `minimize` runs on a
//! structural diagram, and distinct functions by random-assignment signatures.

use std::collections::HashMap;
use std::collections::HashSet;
use crate::vtree::rng::Lcg;
use crate::diagram::{ChildSide, Tdd, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO};
use super::signature::{eval_all_signatures, random_var_assignments};

/// Check that `tdd` is canonical: `Ok` exactly when
/// [`minimize`](crate::Tdd::minimize), with every level owed to its passes,
/// would change nothing, up to node order within a level. The invariants are
/// stated in `docs/invariants.md`; the rules, in the order they are checked:
///
/// 1. Invariant 4, at the structural levels.
/// 2. Invariant 19, at the structural levels.
/// 3. Invariant 20. A level where one node does not pair up, or that names
///    the leaf `One`, passes.
/// 4. On a diagram with a summed-out level, invariants 7 to 10 and 24
///    (`marginal::check_marginal_canonical_form`) and 12
///    (`marginal::check_no_orphan_slots`), which stand in for rules 1 and 2
///    at the summed levels.
/// 5. On a diagram with no summed-out level, invariant 3
///    ([`check_distinct_functions`] over `rounds` rounds), which also fails
///    some diagrams whose levels are not deterministic.
///
/// Cost: O(pairs × log pairs + diagram size × rounds).
pub fn check_canonicity(tdd: &Tdd, rounds: u32) -> Result<(), String> {
    check_reachable(tdd)?;
    check_no_twins(tdd)?;
    check_no_leaf_twins(tdd)?;
    if tdd.has_marginal_level() {
        super::marginal::check_marginal_canonical_form(tdd)?;
        return super::marginal::check_no_orphan_slots(tdd);
    }
    check_distinct_functions(tdd, rounds)
}

fn check_reachable(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    let reach = tdd.reachable_nodes();
    for (t, _, _) in vtree.internal_bottomup().filter(|&(t, _, _)| !tdd.level(t).is_marginal()) {
        if let Some(j) = (0..tdd.levels[t.idx()].slot_count()).find(|&j| !reach[t.idx()][j]) {
            return Err(format!("vtree {t:?} node {j} is not reachable from the output"));
        }
    }
    Ok(())
}

fn check_no_twins(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    for (t, left, right) in vtree.internal_bottomup().filter(|&(t, _, _)| !tdd.level(t).is_marginal()) {
        let level = &tdd.levels[t.idx()];
        for (child, side) in [(left, ChildSide::Left), (right, ChildSide::Right)] {
            if vtree.node(child).is_leaf() || tdd.level(child).is_marginal() {
                continue;
            }
            let mut places: Vec<Vec<(usize, u32)>> = vec![Vec::new(); tdd.levels[child.idx()].slot_count()];
            for i in 0..level.slot_count() {
                for p in level.pairs_vec(i) {
                    let (me, partner) = side.pick((p.left, p.right), (p.right, p.left));
                    if me != ZERO.into() {
                        places[me.raw() as usize].push((i, partner.raw()));
                    }
                }
            }
            let mut seen: HashMap<Vec<(usize, u32)>, usize> = HashMap::with_capacity(places.len());
            for (j, mut at) in places.into_iter().enumerate() {
                at.sort_unstable();
                if let Some(k) = seen.insert(at, j) {
                    return Err(format!(
                        "vtree {child:?} nodes {k} and {j} are twins: level {t:?} names them in the same places"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn check_no_leaf_twins(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    for (t, left, right) in vtree.internal_bottomup().filter(|&(t, _, _)| !tdd.level(t).is_marginal()) {
        let level = &tdd.levels[t.idx()];
        for (child, side) in [(left, ChildSide::Left), (right, ChildSide::Right)] {
            if !vtree.node(child).is_var_leaf() || tdd.level(child).is_marginal() {
                continue;
            }
            let mut literal = false;
            let contractible = (0..level.slot_count()).all(|i| {
                let (mut pos, mut neg, mut one) = (Vec::new(), Vec::new(), false);
                for p in level.pairs_vec(i) {
                    let (label, partner) = side.pick((p.left, p.right), (p.right, p.left));
                    match label {
                        l if l == POS_LEAF_IDX.into() => pos.push(partner),
                        l if l == NEG_LEAF_IDX.into() => neg.push(partner),
                        l if l == ONE_LEAF_IDX.into() => one = true,
                        _ => return false,
                    }
                }
                let here = !pos.is_empty() || !neg.is_empty();
                literal |= here;
                pos.sort_unstable();
                neg.sort_unstable();
                !(one && here) && pos == neg
            });
            if contractible && literal {
                return Err(format!(
                    "level {t:?} pairs every Pos of leaf {child:?} with a Neg on the same partner"
                ));
            }
        }
    }
    Ok(())
}

/// Check invariant 3 by random-assignment signatures: a node's signature is
/// its value in the (Z_p, +, ×) semiring with random variable values,
/// p = 2^61 − 1, and a collision of two distinct functions has probability at
/// most `n/p` per round and pair. It does not decide canonical form, which
/// [`check_canonicity`] does. Cost: O(diagram size × rounds).
///
/// A diagram with a summed-out level is outside its domain, and it says so
/// with `Err`.
pub fn check_distinct_functions(tdd: &Tdd, rounds: u32) -> Result<(), String> {
    if tdd.has_marginal_level() {
        return Err("distinct functions is not decided on a diagram with a summed-out level".into());
    }
    let vtree = &tdd.vtree;
    let num_vars = vtree.num_vars() as usize;
    for round in 0..rounds {
        let mut rng = Lcg::new((round as u64).wrapping_mul(0x517cc1b727220a95));
        let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
        let signatures = eval_all_signatures(tdd, &pos_val, &neg_val);
        // The slots of a region not summed out are labels, of which only One
        // has a meaning.
        for t in vtree.bottomup().filter(|&t| !vtree.node(t).is_region()) {
            let level_sigs = &signatures[t.idx()];
            let mut seen = HashSet::with_capacity(level_sigs.len());
            for (i, &sig) in level_sigs.iter().enumerate() {
                if !seen.insert(sig) {
                    let j = level_sigs.iter().position(|&s| s == sig).unwrap();
                    return Err(format!(
                        "round {round}: vtree {t:?} nodes {j} and {i} have the same signature ({sig}) \
                         — likely equivalent (non-canonical)"
                    ));
                }
            }
        }
    }
    Ok(())
}
