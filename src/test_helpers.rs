//! Test-only helpers shared across the crate test modules.

use std::sync::Arc;

use num_bigint::BigUint;

use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::query::node_counts;
use crate::apply::apply_and;
use crate::diagram::{InputPair, NodeIdx, Tdd, TddLevel, TddNodeId, assert_can_make_marginal};
use crate::diagram::Literal;
use crate::apply::project::{NEG, POS};
use crate::diagram::ChildSide;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// DIMACS-style literals (`±(var+1)`) to `Literal`s.
pub fn literals(clause: &[i32]) -> Vec<Literal> {
    clause.iter().map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0)).collect()
}

/// `(var, polarity)` pairs to `Literal`s, for tests that name variables by
/// their 0-based index rather than in DIMACS.
pub fn clause(literals: &[(u32, bool)]) -> Vec<Literal> {
    literals.iter().map(|&(v, positive)| Literal::new(VarId(v), positive)).collect()
}

/// Conjoin DIMACS-style clauses one at a time, minimizing after each.
pub fn compile_clauses(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let eng = &crate::engine::Engine::new();
    let mut acc = constant_one(eng, vtree);
    for clause in clauses {
        let cl = clause_to_tdd(eng, vtree, &literals(clause));
        acc = apply_and(acc, cl);
        minimize(&mut acc);
    }
    acc
}

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

/// Formulas covering SAT, UNSAT, unit, wide, and don't-care shapes.
pub fn test_cases() -> Vec<(u32, Vec<Vec<i32>>)> {
    vec![
        (3, vec![vec![1, 2], vec![-2, 3], vec![-1, -3]]),
        (4, vec![vec![1, 2], vec![3, 4], vec![-1, -3], vec![-2, -4]]),
        (4, vec![vec![1], vec![2, -3], vec![-1, 3, 4], vec![-2, -4]]),
        (3, vec![vec![1, 2, 3]]),
        (2, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2]]),
        (1, vec![vec![1], vec![-1]]),
        (2, vec![vec![1], vec![2], vec![-1, -2]]),
        (3, vec![vec![1], vec![-1], vec![2, 3]]),
        (4, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2],
                 vec![3, 4], vec![-3, -4], vec![3, -4], vec![-3, 4]]),
        (3, vec![vec![1]]),
        (3, vec![vec![-2]]),
        (4, vec![vec![1, 2, 3, 4]]),
        (4, vec![vec![1, 2], vec![3], vec![-4]]),
        (4, vec![vec![1, 2], vec![-1, -2]]),
        (5, vec![vec![1]]),
        (3, vec![vec![1, 2], vec![-1, -2], vec![3]]),
        (6, vec![vec![1, 2], vec![-3, -4]]),
    ]
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

/// Bottom-up marginalize every internal, non-marginal, width≥1 level in
/// the subtree rooted at `root` (inclusive). Counts are derived from the
/// current diagram shape via `node_counts`. Mirrors production's
/// The marginalize pass's batch + cascade semantics for a single
/// subtree, without the streaming-marginal hooks.
pub fn marginalize_subtree(tdd: &mut Tdd, root: VtreeIdx) {
    let vtree = tdd.vtree.clone();
    let counts = node_counts(tdd);
    for &t in vtree.bottomup_slice() {
        let ti = t.idx();
        let mut under = t == root;
        if !under {
            let mut cur = t;
            while let Some(p) = vtree.node(cur).parent() {
                if p == root {
                    under = true;
                    break;
                }
                cur = p;
            }
        }
        if !under {
            continue;
        }
        if matches!(*vtree.node(VtreeIdx(ti as u32)), VtreeNode::Leaf { .. }) || tdd.levels[ti].is_marginal() {
            continue;
        }
        let w = tdd.levels[ti].width();
        if w == 0 {
            continue;
        }
        let u128_counts: Vec<u128> = (0..w).map(|i| big_to_u128(&counts[ti][i])).collect();
        assert_can_make_marginal(&tdd.levels, &vtree, t);
        tdd.levels[ti].become_marginal(u128_counts, None);
    }
    // Emulate production marginalization, which tags every persisted marginal-side
    // slot ref (bit 30) so the 0=inline decode invariant holds. Without this the
    // strict decode assert fires when a later reader hits a raw slot ref.
    crate::diagram::tag_all_marginal_side_slots(tdd, None);
}

// ── Marginal-invariant fixtures ──────────────────────────────────────────────
/// Non-inlinable count: forces the slot path so the inline-discipline check
/// stays out of the way of the other invariant tests.
pub(crate) const BIG: u128 = 1u128 << 40;

/// Minimal boundary-marginal diagram: `balanced(2)` vtree, right child
/// marginal with `counts`, root holding one internal node per entry of
/// `node_pair_lists` (pairs as raw `(left, right)` values; slot refs are
/// bare indices under the bare-is-slot polarity).
pub(crate) fn toy(counts: Vec<u128>, node_pair_lists: &[&[(u32, u32)]]) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    levels[right.idx()].set_counts_state(counts, None);
    for pl in node_pair_lists {
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: NodeIdx(l), right: NodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}

/// Weighted analogue of [`toy`]: the right child is a WEIGHT-marginal level
/// whose per-slot `BigRational` values live in the `WeightStore` attached to the
/// returned [`Tdd`] (not `marginal_counts`). The caller supplies the store; this
/// helper writes the values into it via `set_level` and attaches it. Parent pair
/// refs use the same bare-is-slot polarity as `toy`.
///
/// `balanced(3)`, not `balanced(2)` (which the integer [`toy`] still uses): its
/// root's right child is an INTERNAL node, so the marginal level here is an
/// ordinary internal one. A weight-marginal vtree LEAF is a different animal —
/// its `WeightStore` column is PINNED to the label-ordered 3-slot `leaf_val`
/// cache (`marginalize_leaf_weighted`), which no pass may compact, erase or
/// append to — so a leaf could not model an arbitrary-width, compactable
/// marginal store at all.
pub(crate) fn toy_weighted(
    mut ws: crate::diagram::WeightStore,
    values: Vec<num_rational::BigRational>,
    node_pair_lists: &[&[(u32, u32)]],
) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(3));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(3) root must be internal"),
    };
    debug_assert!(
        !vtree.node(right).is_leaf(),
        "toy_weighted's marginal side must be INTERNAL — a leaf column is pinned"
    );
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    levels[right.idx()].become_marginal_weighted(values.len() as u32);
    for pl in node_pair_lists {
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: NodeIdx(l), right: NodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let wvals: Vec<crate::diagram::WeightVal> =
        values.into_iter().map(crate::diagram::WeightVal::exact).collect();
    ws.set_level(right.idx(), wvals);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    tdd.set_weights(ws);
    tdd
}

// ── Structural support oracles ───────────────────────────────────────────────

/// Structural support of `t`: `out[x]` is true iff `t` depends on variable `x`
/// (some live pair references x's leaf with a Pos/Neg label, not just One).
/// Minimizes a clone first so every scanned node is reachable. O(size).
///
/// Test-only: the exact `Vec<bool>` support oracle, kept as ground truth for the
/// `support_bits` over-approximation invariant tests (its former production
/// callers were removed).
pub(crate) fn support_mask(t: &Tdd) -> Vec<bool> {
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
            *sup_x = mt.output.local == POS || mt.output.local == NEG;
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
                    if child == POS || child == NEG {
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
pub(crate) fn support_bits(t: &Tdd) -> Vec<u64> {
    let vtree = &t.vtree;
    let nvars = vtree.num_vars() as usize;
    let mut bits = vec![0u64; nvars.div_ceil(64)];
    if t.is_zero() {
        return bits;
    }
    // Output sitting directly at a leaf: set that leaf's var iff labelled Pos/Neg.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(t.output.vtree) {
        if t.output.local == POS || t.output.local == NEG {
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
                if need_l && (p.left == POS || p.left == NEG) {
                    let x = lvar.unwrap();
                    bits[x / 64] |= 1u64 << (x % 64);
                    need_l = false;
                }
                if need_r && (p.right == POS || p.right == NEG) {
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

/// Vtree shape shared by the fork-down fixtures below. Custom (not `balanced`)
/// so the marginal-carrying level `m` sits at an INTERNAL vtree node: an integer
/// marginal LEAF keeps an EMPTY store (bare refs are leaf-LABELS, decoded by
/// `read_marginal_count`), so it is not a legal fork-down scale target and the
/// mint that these tests exercise would be unsound there (see
/// `duplicate_pair_resolve.rs` `scale_leaf_marginal_label`). An internal marginal level exercises
/// the multiplicity-fork-down mechanics identically, with a real store to mint
/// into. Shape (left spine root → gp → bp; each 2-leaf subtree on the right):
///   root → (gp, σ);  gp → (bp, s);  bp → (x [leaf], m [INTERNAL]);
///   m → (m_l, m_r);  s → (s_l, s_r);  σ → (sig_l, sig_r).
pub fn boundary_internal_marginal_vtree() -> Vtree {
    // 7 vars; node ids reindexed bottom-up by `from_text` (root last),
    // so callers navigate via `children()` exactly as with `balanced`.
    //   x=0(leaf)  m=(1,2)  s=(3,4)  σ=(5,6);  bp=(x,m) gp=(bp,s) root=(gp,σ)
    Vtree::from_text(
        "vtree 13\n\
         L 0 1\nL 1 2\nL 2 3\nL 3 4\nL 4 5\nL 5 6\nL 6 7\n\
         I 7 1 2\nI 8 0 7\nI 9 3 4\nI 10 8 9\nI 11 5 6\nI 12 10 11\n",
    )
    .expect("boundary_internal_marginal_vtree parse")
}

