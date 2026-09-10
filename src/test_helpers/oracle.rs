//! How a test decides a diagram is right: enumeration, canonicity, structural
//! equality, the support oracles, and the deadline harness.

use num_bigint::BigUint;

use std::sync::Arc;

use crate::diagram::{ChildSide, NodeIdx, Tdd, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO};
use crate::engine::Engine;
use crate::reduce::minimize;

use super::compile::and2;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

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
pub fn normalized_levels(tdd: &Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
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
            remap[t.idx()] = (0..level.width() as u32).collect();
            out[t.idx()] = vec![Vec::new(); level.width()];
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

/// `BigUint` → u128, panicking if the value exceeds 128 bits. Used by tests
/// that feed `node_counts` output into `become_marginal`, which
/// requires u128 counts.
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
/// result; only a test whose subject is deliberately mid-flight (an
/// accumulator, a shrunk operand) has cause to skip it.
pub fn assert_canonical(tdd: &Tdd) {
    let fail = |name: &str, r: Result<(), String>| {
        if let Err(e) = r {
            panic!("assert_canonical: {name}: {e}");
        }
    };
    fail("vtree structure", crate::check::validate_vtree_structure(tdd));
    fail("no false nodes", crate::check::check_no_false_nodes(tdd));
    fail("canonicity", crate::check::check_canonicity(tdd, 3));
    if !tdd.has_marginal_level() {
        return;
    }
    type MarginalCheck = fn(&Tdd) -> Result<(), String>;
    let checks: [(&str, MarginalCheck); 5] = [
        ("inline_discipline", crate::check::marginal::check_inline_discipline),
        ("no_orphan_slots", crate::check::marginal::check_no_orphan_slots),
        ("slot_count_uniqueness", crate::check::marginal::check_slot_count_uniqueness),
        ("marginal_canonical_form", crate::check::marginal::check_marginal_canonical_form),
        ("no_fusion_redexes", crate::check::marginal::check_no_fusion_redexes),
    ];
    for (name, check) in checks {
        check(tdd).unwrap_or_else(|e| panic!("assert_canonical: {name}: {e}"));
    }
}

/// Run `build` on an engine whose wall is already in the past, so the first
/// metered poll cuts. `stride` pins the reduce poll stride: `Some(1)` makes
/// every tick a poll, a larger value moves the cut past that much metered
/// work, `None` leaves the production stride.
pub fn deadline_probe<R>(stride: Option<u64>, build: impl FnOnce(&Engine) -> R) -> R {
    let eng = Engine::with_stop_now();
    eng.limits().pin_reduce_poll_stride(stride);
    build(&eng)
}


/// Two diagrams over one vtree are the same diagram: same root level, and
/// level-by-level equal pair lists once node numbering is normalized away.
pub fn assert_same_shape(a: &Tdd, b: &Tdd, what: &str) {
    assert_eq!(a.output().vtree, b.output().vtree, "{what}: root level differs");
    assert_eq!(a.size(), b.size(), "{what}: size differs");
    assert_eq!(normalized_levels(a), normalized_levels(b), "{what}: level shape differs");
}

/// Structural support of `t`: `out[x]` is true iff `t` depends on variable `x`
/// (some live pair references x's leaf with a Pos/Neg label, not just One).
/// Minimizes a clone first so every scanned node is reachable. O(size).
///
/// Test-only: the exact `Vec<bool>` support oracle, kept as ground truth for the
/// `support_bits` over-approximation invariant tests (its former production
/// callers were removed).
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

/// Fast OVER-approximate structural support of `t`, packed into a `u64` bitmask
/// (bit `x` set ⇒ `t` MAY depend on variable `x`). Unlike [`support_mask`] this does
/// not clone or `minimize` first: it walks the diagram as-is in a single O(size)
/// pass, so a dead node can set a bit for a variable `t` no longer truly depends on.
/// That one-sided error is precisely what a disjoint-support pre-skip needs — if two
/// over-approximate supports are disjoint then the TRUE supports (subsets) are too,
/// so skipping the operation is sound; a spurious overlap merely runs the op that
/// would have run anyway. (`support_mask` is the EXACT `Vec<bool>` variant that
/// minimizes first — pick by whether you need exactness or raw speed.)
///
/// Test-only: the sole non-test caller was the retired segment-compile lane; the
/// projection unit tests keep it as a fast support oracle to check `support_mask`.
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
/// "size" for the never-larger gate (`Tdd::size` counts dead arena pairs too).
pub fn reachable_pairs(t: &Tdd) -> usize {
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
pub fn count_is_zero(t: &Tdd) -> bool {
    crate::query::model_count(t) == BigUint::from(0u32)
}

/// `a` and `b` are the same Boolean function over their shared vtree, by the
/// two-way difference being empty.
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
    let g = crate::apply::restrict(f.clone(), c.clone(), crate::apply::CareCanonical::No).into_tdd();
    for mask in 0..(1u32 << nvars) {
        let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
        let cv = eval(c, &asn);
        assert_eq!(
            eval(&g, &asn) && cv,
            eval(f, &asn) && cv,
            "restrict unsound at assignment {asn:?} (g∧c ≠ f∧c)"
        );
    }
    let mut gm = g.clone();
    minimize(&mut gm);
    crate::check::check_all_fast(&gm, "restrict-output");
    crate::check::check_determinism(&gm)
        .expect("restrict output must be deterministic (mutex pairs)");
    assert!(reachable_pairs(&g) <= reachable_pairs(f), "restrict grew the diagram beyond f");
}

/// Brute-force projected model count: how many distinct projections onto
/// `show` the satisfying assignments of `clauses` have.
///
/// `clauses` are DIMACS-style over `n` variables and `show` lists them
/// 0-based. Every one of the `2^n` assignments is enumerated and each
/// satisfying one contributes its restriction to `show`; the answer is the
/// number of distinct restrictions. Distinct from [`brute_force_count`],
/// which counts satisfying assignments themselves.
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
