//! How a test decides a diagram is right: enumeration, canonicity, structural
//! equality, the support oracles, and the deadline harness.

#[cfg(any(test, debug_assertions))]
use num_bigint::BigUint;

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::diagram::ChildSide;
#[cfg(any(test, debug_assertions))]
use crate::diagram::{LeafLabel, PairsIter};
use crate::diagram::{NodeIdx, Tdd, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO};
#[cfg(any(test, debug_assertions))]
use crate::engine::Engine;
#[cfg(test)]
use super::access::stopping_engine;
#[cfg(any(test, debug_assertions))]
use crate::query::count::leaf_seed;
#[cfg(any(test, debug_assertions))]
use crate::query::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};
#[cfg(any(test, debug_assertions))]
use crate::query::PinSemantics;
#[cfg(any(test, debug_assertions))]
use crate::value::{ColumnRetention, CountRead};
use crate::reduce::minimize;

#[cfg(test)]
use super::compile::and2;
#[cfg(any(test, debug_assertions))]
use crate::vtree::VarId;
use crate::vtree::{VtreeIdx, VtreeNode};

/// Model count of DIMACS-style clauses by enumeration.
pub fn brute_force_count(num_vars: u32, clauses: &[Vec<i32>]) -> u64 {
    (0..(1u64 << num_vars))
        .filter(|assignment| {
            clauses.iter().all(|clause| {
                clause.iter().any(|&lit| {
                    let val = (assignment >> (lit.unsigned_abs() - 1)) & 1 == 1;
                    (lit > 0) == val
                })
            })
        })
        .count() as u64
}

/// Per-level pair lists with node indices renamed to a canonical order, so two
/// diagrams over one vtree compare equal iff they are the same up to node
/// numbering. Leaf levels and marginal levels compare by width only; refs into
/// a marginal child are slot or inline refs and compare by raw value.
pub(crate) fn normalized_levels(tdd: &Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
    let vtree = &tdd.vtree;
    let n = vtree.num_nodes();
    let mut remap: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut out: Vec<Vec<Vec<(u32, u32)>>> = vec![Vec::new(); n];
    for (t, _) in vtree.leaf_bottomup() {
        remap[t.idx()] = (0..crate::diagram::LEAF_WIDTH as u32).collect();
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let level = &tdd.levels[t.idx()];
        if level.is_marginal() {
            remap[t.idx()] = (0..level.slot_count() as u32).collect();
            out[t.idx()] = vec![Vec::new(); level.slot_count()];
            continue;
        }
        let left_marginal = tdd.levels[left.idx()].is_marginal();
        let right_marginal = tdd.levels[right.idx()].is_marginal();
        let mut indexed: Vec<(usize, Vec<(u32, u32)>)> = (0..level.nodes.len())
            .map(|i| {
                let mut pairs: Vec<(u32, u32)> = level
                    .pairs_of_idx(i)
                    .iter()
                    .map(|p| {
                        (
                            if left_marginal { p.left.0 } else { remap[left.idx()][p.left.idx()] },
                            if right_marginal { p.right.0 } else { remap[right.idx()][p.right.idx()] },
                        )
                    })
                    .collect();
                pairs.sort();
                (i, pairs)
            })
            .collect();
        indexed.sort_by(|a, b| a.1.cmp(&b.1));
        remap[t.idx()] = vec![0; level.nodes.len()];
        for (new_i, (old_i, pairs)) in indexed.into_iter().enumerate() {
            remap[t.idx()][old_i] = new_i as u32;
            out[t.idx()].push(pairs);
        }
    }
    out
}

/// Per-node model counts in exact `BigUint`: `counts[vtree_idx][node_idx]` is
/// the number of satisfying assignments of each diagram node.
///
/// The full-precision oracle: no u128 fast path, one `BigUint` per node. It
/// shares the walk with [`ModelCounter`](crate::query::ModelCounter) and nothing else — its
/// arithmetic is independent, which is what makes the differential test
/// between the two worth running.
#[cfg(any(test, debug_assertions))]
pub fn node_counts(tdd: &Tdd) -> Vec<Vec<BigUint>> {
    count_big(tdd, &[], PinSemantics::Cofactor)
}

/// Full-precision pinned model count of `tdd` under `convention`.
///
/// The oracle the u128-hybrid pinned counter ([`ModelCounter`](crate::query::ModelCounter)) is
/// differentially tested against: one `BigUint` bottom-up pass with no u128
/// fast path, allocating a per-node count column for every level, per call.
///
/// Under [`PinSemantics::Evidence`] this is also the reference spelling of the
/// pinned readout: with the own-show leaves marginalized and the boundary vars
/// left Boolean, pinning a boundary assignment and counting yields that
/// assignment's boundary-function entry, marginal tagging decoded internally
/// (never read `marginal_counts` raw).
#[cfg(test)]
pub fn pinned_counts(tdd: &Tdd, pins: &[Option<bool>], convention: PinSemantics) -> BigUint {
    if tdd.is_zero() {
        return BigUint::ZERO;
    }
    let counts = count_big(tdd, pins, convention);
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
    counts[out_t][out_i].clone()
}

/// The walk behind [`node_counts`] and [`pinned_counts`], with per-variable
/// pins indexed by `VarId::idx()` (out-of-range or `None` entries leave the
/// variable free) and the seed convention the pinned leaves count under.
#[cfg(any(test, debug_assertions))]
fn count_big(tdd: &Tdd, pins: &[Option<bool>], convention: PinSemantics) -> Vec<Vec<BigUint>> {
    let eng = Engine::new();
    let fold = BigCounts { pins, convention };
    let mut cols: Vec<Vec<BigUint>> = (0..tdd.vtree.num_nodes())
        .map(|i| fold.alloc(&eng, tdd.reference_slot_count(VtreeIdx(i as u32))))
        .collect();
    fold_bottom_up_unpolled(&fold, &eng, tdd, &mut cols, ColumnRetention::All, |_, _| {});
    cols
}

/// The exact-`BigUint` counting fold.
#[cfg(any(test, debug_assertions))]
struct BigCounts<'a> {
    pins: &'a [Option<bool>],
    convention: PinSemantics,
}

#[cfg(any(test, debug_assertions))]
impl LevelFold for BigCounts<'_> {
    type Value = BigUint;
    type Col = Vec<BigUint>;

    fn alloc(&self, _eng: &Engine, width: usize) -> Vec<BigUint> {
        vec![BigUint::ZERO; width]
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<BigUint>, i: usize, v: BigUint) {
        col[i] = v;
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> BigUint {
        let pin = self.pins.get(var.idx()).copied().flatten();
        BigUint::from(leaf_seed(label, pin, self.convention))
    }

    /// A marginal level's counts are pin-independent: they were summed out before
    /// any pin existed, so they are read across verbatim.
    fn marginal_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<BigUint>) {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a marginal level carries counts");
        let big = level.marginal_counts_big();
        for (i, slot) in col[..counts.len()].iter_mut().enumerate() {
            match CountRead::from_slot(counts, big, i) {
                CountRead::Fast(c) => *slot = BigUint::from(c),
                CountRead::Big(b) => slot.clone_from(b),
            }
        }
    }

    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<BigUint>>,
        right: Side<'_, Vec<BigUint>>,
    ) -> BigUint {
        self.sum_over_pairs(pairs, left, right)
    }
}

#[cfg(any(test, debug_assertions))]
impl PairAlgebra for BigCounts<'_> {
    fn zero(&self) -> BigUint {
        BigUint::ZERO
    }
    fn read(&self, col: &Vec<BigUint>, i: usize) -> BigUint {
        col[i].clone()
    }
    fn inline(&self, count: u32) -> BigUint {
        BigUint::from(count)
    }
    fn add_assign(&self, acc: &mut BigUint, v: &BigUint) {
        *acc += v;
    }
    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        a * b
    }
}

/// `minimize` preserves the function: the output node's random-assignment
/// signature, in the semiring the checkers' signature evaluates in, is the
/// same before and after one `minimize` of `tdd`. `Err` names the round whose
/// signature moved.
///
/// Mutates `tdd` by that one `minimize`; safe on an already minimized
/// diagram, where it doubles as an idempotency check.
#[cfg(any(test, debug_assertions))]
pub fn check_minimize_soundness(tdd: &mut Tdd, rounds: u32) -> Result<(), String> {
    use crate::test_helpers::Lcg;
    use crate::test_helpers::check::signature::{eval_all_signatures, random_var_assignments};

    let num_vars = tdd.vtree.num_vars() as usize;
    let output_signature = |tdd: &Tdd, round: u32| -> u64 {
        if tdd.output.local == ZERO {
            return 0;
        }
        let mut rng = Lcg::new((round as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
        eval_all_signatures(tdd, &pos_val, &neg_val)[tdd.output.vtree.idx()][tdd.output.local.idx()]
    };
    let before: Vec<u64> = (0..rounds).map(|round| output_signature(tdd, round)).collect();
    minimize(tdd);
    for (round, &was) in before.iter().enumerate() {
        let now = output_signature(tdd, round as u32);
        if was != now {
            return Err(format!(
                "round {round}: output signature changed from {was} to {now} — minimize altered the function"
            ));
        }
    }
    Ok(())
}

/// `BigUint` → u128, panicking if the value exceeds 128 bits. Used by tests
/// that feed `node_counts` output into `become_marginal`, which
/// requires u128 counts.
#[cfg(test)]
pub fn big_to_u128(b: &BigUint) -> u128 {
    let digits = b.to_u64_digits();
    match digits.len() {
        0 => 0,
        1 => digits[0] as u128,
        2 => ((digits[1] as u128) << 64) | (digits[0] as u128),
        _ => panic!("test fixture count overflows u128: {b}"),
    }
}

/// Every invariant that holds of a finished diagram, in one call: the fast
/// family always, and the marginal family whenever the diagram carries a
/// marginal level. A test whose subject produces a diagram asserts this on the
/// result; only a test whose subject is mid-flight (an accumulator, a shrunk
/// operand) has cause to skip it.
#[cfg(any(test, debug_assertions))]
pub fn assert_canonical(tdd: &Tdd) {
    crate::test_helpers::check::check_all_fast(tdd, "assert_canonical");
    if tdd.has_marginal_level() {
        marginal_family(tdd, "assert_canonical");
    }
}

/// The invariants a marginalized diagram is held to: the vtree structure and
/// the marginal family.
///
/// Not [`assert_canonical`]: its canonicity check separates two nodes at a
/// level by a random-assignment signature, and a marginal child contributes
/// its stored count to that signature rather than anything structural, so the
/// check cannot decide a level that sits over summed-out storage — a
/// structural level whose two children are marginal signs `4 × 1` and `2 × 2`
/// identically, and a node with a repeated pair over a marginal subtree signs
/// as one pair over twice the count. What still holds after summing levels
/// out is the marginal family, and that is what this asserts.
#[cfg(any(test, debug_assertions))]
pub fn assert_marginal_canonical(tdd: &Tdd) {
    crate::test_helpers::check::validate_vtree_structure(tdd)
        .unwrap_or_else(|e| panic!("assert_marginal_canonical: vtree structure: {e}"));
    marginal_family(tdd, "assert_marginal_canonical");
}

/// The two marginal-form checkers, failing under `label`.
#[cfg(any(test, debug_assertions))]
fn marginal_family(tdd: &Tdd, label: &str) {
    type MarginalCheck = fn(&Tdd) -> Result<(), String>;
    let checks: [(&str, MarginalCheck); 2] = [
        ("no_orphan_slots", crate::test_helpers::check::marginal::check_no_orphan_slots),
        ("marginal_canonical_form", crate::test_helpers::check::marginal::check_marginal_canonical_form),
    ];
    for (name, check) in checks {
        check(tdd).unwrap_or_else(|e| panic!("{label}: {name}: {e}"));
    }
}

/// The checkers are compiled only under `cfg(test)` or `debug_assertions`, so
/// where they are absent this call has nothing to run. A build that wants the
/// invariants checked turns debug assertions on.
#[cfg(not(any(test, debug_assertions)))]
pub fn assert_canonical(_tdd: &Tdd) {}

/// As [`assert_canonical`]: nothing to run where the checkers are absent.
#[cfg(not(any(test, debug_assertions)))]
pub fn assert_marginal_canonical(_tdd: &Tdd) {}

/// Run `build` on an engine whose wall is already in the past, so the first
/// metered poll cuts. `stride` pins the reduce poll stride: `Some(1)` makes
/// every tick a poll, a larger value moves the cut past that much metered
/// work, `None` leaves the production stride.
#[cfg(test)]
pub fn deadline_probe<R>(stride: Option<u64>, build: impl FnOnce(&Engine) -> R) -> R {
    let eng = stopping_engine();
    eng.limits().pin_reduce_poll_stride(stride);
    build(&eng)
}


/// Two diagrams over one vtree are the same diagram: same root level, and
/// level-by-level equal pair lists once node numbering is normalized away.
pub fn assert_same_shape(a: &Tdd, b: &Tdd, what: &str) {
    assert_eq!(a.output().vtree, b.output().vtree, "{what}: root level differs");
    assert_eq!(a.pair_count(), b.pair_count(), "{what}: size differs");
    assert_eq!(normalized_levels(a), normalized_levels(b), "{what}: level shape differs");
}

/// Structural support of `t`: `out[x]` is true iff `t` depends on variable `x`
/// (some live pair references x's leaf with a Pos/Neg label, not just One).
/// Minimizes a clone first so every scanned node is reachable. O(size).
/// The exact oracle the `support_bits` over-approximation is tested against.
#[cfg(test)]
pub fn support_mask(t: &Tdd) -> Vec<bool> {
    let nvars = t.vtree.num_vars() as usize;
    let mut sup = vec![false; nvars];
    if t.is_zero() {
        return sup;
    }
    let mut mt = t.clone();
    minimize(&mut mt);
    if mt.is_zero() {
        return sup;
    }
    let vtree = Arc::clone(&mt.vtree);
    for (x, sup_x) in sup.iter_mut().enumerate() {
        let leaf = vtree.leaf_of(VarId(x as u32)).expect("the vtree carries this variable");
        // Output sits at the leaf itself: depends on x iff the label is Pos/Neg.
        if mt.output.vtree == leaf {
            *sup_x = mt.output.local == POS_LEAF_IDX || mt.output.local == NEG_LEAF_IDX;
            continue;
        }
        // The leaf has exactly one parent in the (tree) vtree; find the side and
        // scan that parent level for any Pos/Neg reference to it.
        for vi in 0..vtree.num_nodes() {
            let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
                VtreeNode::Internal { left, right, .. } => (left, right),
                VtreeNode::Leaf { .. } => continue,
            };
            let side = if left == leaf {
                ChildSide::Left
            } else if right == leaf {
                ChildSide::Right
            } else {
                continue;
            };
            let level = &mt.levels[vi];
            'scan: for ni in 0..level.nodes.len() {
                if level.nodes[ni].is_leaf() {
                    continue;
                }
                for p in level.pairs_of(&level.nodes[ni]) {
                    let child = match side {
                        ChildSide::Left => p.left,
                        ChildSide::Right => p.right,
                    };
                    if child == POS_LEAF_IDX || child == NEG_LEAF_IDX {
                        *sup_x = true;
                        break 'scan;
                    }
                }
            }
            break; // unique parent found; done with this var
        }
    }
    sup
}

/// Over-approximate structural support of `t`, packed into a `u64` bitmask
/// (bit `x` set ⇒ `t` may depend on variable `x`). Walks the diagram as-is in
/// one O(size) pass without minimizing first, so a dead node can set a bit for
/// a variable `t` does not depend on; [`support_mask`] is the exact form.
#[cfg(test)]
pub fn support_bits(t: &Tdd) -> Vec<u64> {
    let vtree = &t.vtree;
    let nvars = vtree.num_vars() as usize;
    let mut bits = vec![0u64; nvars.div_ceil(64)];
    if t.is_zero() {
        return bits;
    }
    // Output sitting directly at a leaf: set that leaf's var iff labelled Pos/Neg.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(t.output.vtree) {
        if t.output.local == POS_LEAF_IDX || t.output.local == NEG_LEAF_IDX {
            let x = var.idx();
            bits[x / 64] |= 1u64 << (x % 64);
        }
        return bits;
    }
    // Each variable is a vtree leaf whose single parent is an internal level; a Pos/Neg
    // reference to that leaf (on its side) in any of the level's pairs is a dependency.
    // Iterating levels once and reading each internal node's two leaf-children is O(size),
    // versus `support_mask`'s per-variable parent-find sweep.
    for vi in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        let lvar = match *vtree.node(left) {
            VtreeNode::Leaf { var, .. } => Some(var.idx()),
            _ => None,
        };
        let rvar = match *vtree.node(right) {
            VtreeNode::Leaf { var, .. } => Some(var.idx()),
            _ => None,
        };
        if lvar.is_none() && rvar.is_none() {
            continue;
        }
        let level = &t.levels[vi];
        let mut need_l = lvar.is_some();
        let mut need_r = rvar.is_some();
        'scan: for ni in 0..level.nodes.len() {
            if level.nodes[ni].is_leaf() {
                continue;
            }
            for p in level.pairs_of(&level.nodes[ni]) {
                if need_l && (p.left == POS_LEAF_IDX || p.left == NEG_LEAF_IDX) {
                    let x = lvar.unwrap();
                    bits[x / 64] |= 1u64 << (x % 64);
                    need_l = false;
                }
                if need_r && (p.right == POS_LEAF_IDX || p.right == NEG_LEAF_IDX) {
                    let x = rvar.unwrap();
                    bits[x / 64] |= 1u64 << (x % 64);
                    need_r = false;
                }
                if !need_l && !need_r {
                    break 'scan;
                }
            }
        }
    }
    bits
}

/// Total reachable input-pair count of a (preferably minimized) diagram — the honest
/// "size" for the never-larger gate (`Tdd::pair_count` counts dead arena pairs too).
pub(crate) fn reachable_pairs(t: &Tdd) -> usize {
    if t.is_zero() {
        return 0;
    }
    let reach = t.reachable_nodes();
    let mut n = 0;
    // Indexes the vtree, `t.levels` and `reach` at the same position.
    #[allow(clippy::needless_range_loop)]
    for vi in 0..t.vtree.num_nodes() {
        if t.vtree.node(VtreeIdx(vi as u32)).is_leaf() {
            continue;
        }
        let level = &t.levels[vi];
        for i in 0..level.nodes.len() {
            if reach[vi][i] && level.nodes[i].is_internal() {
                n += level.pair_count_at(i);
            }
        }
    }
    n
}


/// A diagram with no models. A false diagram is not always `is_zero()`:
/// an unarmed apply can leave `model_count == 0` in a non-canonical form,
/// with the output node still holding pairs. Conditioning canonicalizes its
/// own output, but an apply does not, so unsatisfiability on a derived
/// diagram is decided by the count.
#[cfg(test)]
pub fn count_is_zero(t: &Tdd) -> bool {
    crate::query::model_count(t) == BigUint::from(0u32)
}

/// `a` and `b` are the same Boolean function over their shared vtree, by the
/// two-way difference being empty.
#[cfg(test)]
pub fn equiv(a: &Tdd, b: &Tdd) -> bool {
    let a_not_b = and2(a, &crate::apply::negate(b.clone()));
    let not_a_b = and2(&crate::apply::negate(a.clone()), b);
    count_is_zero(&a_not_b) && count_is_zero(&not_a_b)
}

/// Negation-free equivalence: `a ∧ b ⊆ a` and `a ∧ b ⊆ b` always hold, so
/// equal model counts on all three force `a == b` as sets.
///
/// The oracle for an operand that is a valid diagram but not in the
/// complete form [`equiv`]'s negation needs — a restriction result, say.
#[cfg(test)]
pub fn equiv_nf(a: &Tdd, b: &Tdd) -> bool {
    let ca = crate::query::model_count(a);
    let cb = crate::query::model_count(b);
    ca == cb && crate::query::model_count(&and2(a, b)) == ca
}

/// One label under one variable's value.
fn eval_label(l: NodeIdx, x: bool) -> bool {
    if l == ONE_LEAF_IDX {
        true
    } else if l == POS_LEAF_IDX {
        x
    } else if l == NEG_LEAF_IDX {
        !x
    } else {
        false
    }
}

fn eval_node(t: &Tdd, v: VtreeIdx, local: NodeIdx, asn: &[bool]) -> bool {
    match *t.vtree.node(v) {
        VtreeNode::Leaf { var, .. } => eval_label(local, asn[var.idx()]),
        VtreeNode::Internal { left, right, .. } => {
            if local == ZERO {
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

/// One assignment, evaluated straight off the diagram denotation `⋃ᵢ aᵢ×bᵢ`.
///
/// Shares no machinery with apply or with the counting queries, so brute
/// forcing it over every assignment is a soundness oracle independent of the
/// operator under test.
pub fn eval(t: &Tdd, asn: &[bool]) -> bool {
    if t.is_zero() {
        return false;
    }
    eval_node(t, t.output.vtree, t.output.local, asn)
}

/// Every claim a restriction of `f` against care `c` makes, over `nvars`
/// variables: soundness across the full truth table by the apply-free
/// evaluator, structural validity and determinism of the canonical form, and
/// the never-larger gate.
///
/// Restriction returns a sound subgraph of `f` that callers use raw, so it may
/// carry non-canonical false nodes that minimizing removes. Soundness is
/// therefore checked on the raw result and structure on the minimized one.
pub fn assert_restrict_ok(f: &Tdd, c: &Tdd, nvars: u32) {
    let g = crate::apply::restrict_to_care(f.clone(), c.clone()).into_tdd();
    for mask in 0..(1u32 << nvars) {
        let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
        let cv = eval(c, &asn);
        assert_eq!(
            eval(&g, &asn) && cv,
            eval(f, &asn) && cv,
            "restrict_to_care unsound at assignment {asn:?} (g∧c ≠ f∧c)"
        );
    }
    let mut gm = g.clone();
    minimize(&mut gm);
    #[cfg(any(test, debug_assertions))]
    {
        crate::test_helpers::check::check_all_fast(&gm, "restrict_to_care-output");
        crate::test_helpers::check::check_determinism(&gm)
            .expect("restrict_to_care output must be deterministic (mutex pairs)");
    }
    assert!(reachable_pairs(&g) <= reachable_pairs(f), "restrict_to_care grew the diagram beyond f");
}

/// Brute-force projected model count: how many distinct projections onto
/// `show` the satisfying assignments of `clauses` have.
///
/// `clauses` are DIMACS-style over `n` variables and `show` lists them
/// 0-based. Every one of the `2^n` assignments is enumerated and each
/// satisfying one contributes its restriction to `show`; the answer is the
/// number of distinct restrictions. Distinct from [`brute_force_count`],
/// which counts satisfying assignments themselves.
#[cfg(test)]
pub fn brute_force_pmc(clauses: &[Vec<i32>], n: usize, show: &[usize]) -> BigUint {
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
        seen.insert(show.iter().map(|&v| val(v)).collect());
    }
    BigUint::from(seen.len())
}
