//! The implicit route checked against the route that writes every level's
//! pairs: the same nodes, the same pairs once written, the same arena
//! capacities, the same work, and the same stops, over products of functions on disjoint variables, whose
//! levels are complete products, over chains of such products, whose
//! operands are themselves implicit, and over such chains conditioned on a
//! variable, whose prunes keep some of the implicit levels described.

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, rand_conj_over, same_levels, stored_copies, stored_levels, Lcg};
use crate::vtree::{VarId, Vtree};

/// A random function of `vars`, minimized, on `vtree`, built on `eng`.
fn function_of(vtree: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj_over(vtree, vars, 6, 2, false, rng);
    f.minimize().unwrap();
    f
}

/// The conjunction of `parts` in order, `((p0 ∧ p1) ∧ p2) ∧ …`, or of four
/// parts as `(p0 ∧ p1) ∧ (p2 ∧ p3)` when `pairwise`, then conditioned on
/// each of `given`, a variable and its value, on one fresh engine, with
/// every level stored when `written` holds, under `rules`: the result and
/// the work it took.
fn chain(parts: &[Tdd], pairwise: bool, given: &[(u32, bool)], written: bool, rules: StopRules) -> (Result<Tdd, OperationError>, u64) {
    let eng = Engine::new();
    // The stored route conjoins the stored copies of operands a close or a
    // product left implicit.
    let parts: Vec<Tdd> = if written { parts.iter().map(stored_copies).collect() } else { parts.to_vec() };
    let parts = &parts[..];
    let run = || -> Result<Tdd, OperationError> {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        let mut acc = if let [a, b, c, d] = parts && pairwise {
            let left = eng.and(a.clone(), b.clone())?;
            let right = eng.and(c.clone(), d.clone())?;
            eng.and(left, right)?
        } else {
            let mut acc = parts[0].clone();
            for p in &parts[1..] {
                acc = eng.and(acc, p.clone())?;
            }
            acc
        };
        for &(v, value) in given {
            acc = eng.condition_var(acc, VarId(v), value)?;
        }
        Ok(acc)
    };
    let out = if written { stored_levels(run) } else { run() };
    (out, eng.limits().work_units())
}

/// What the implicit route left described: the levels that hold the
/// description of their pairs, those of them of one pair a node, which hold
/// none in an arena, and the levels a prune kept described.
#[derive(Clone, Copy, Debug, Default)]
struct Described {
    implicit: usize,
    one_pair: usize,
    kept: usize,
}

impl std::ops::AddAssign for Described {
    fn add_assign(&mut self, o: Described) {
        self.implicit += o.implicit;
        self.one_pair += o.one_pair;
        self.kept += o.kept;
    }
}

/// The levels of `t` that hold the description of their pairs, and those of
/// them of one pair a node.
fn implicit_levels(t: &Tdd) -> (usize, usize) {
    let described = t.levels.iter().filter_map(|l| l.pairs.implicit());
    (described.clone().count(), described.filter(|d| d.pairs_per_node() == 1).count())
}

/// The levels of `t` a prune kept described after dropping nodes: implicit,
/// with the slots of the dropped pairs past the described ones.
fn redescribed_levels(t: &Tdd) -> usize {
    t.levels.iter().filter(|l| l.implicit().is_some_and(|d| l.pairs.len() > d.pairs())).count()
}

/// Conjoin `parts` both ways, and condition on `given`, as [`chain`] does,
/// and require the same diagram, node for node, and the same work; then,
/// under a stop at each output-pair floor and at each work bound up to what
/// the chain took, the same stop at the same work. Returns what the result
/// holds described.
fn same_both_ways(parts: &[Tdd], pairwise: bool, given: &[(u32, bool)]) -> Described {
    let none = StopRules { unconditional: None, after_pairs: None };
    let (oracle, oracle_work) = chain(parts, pairwise, given, true, none);
    let (out, work) = chain(parts, pairwise, given, false, none);
    let (oracle, out) = (oracle.unwrap(), out.unwrap());
    assert_canonical(&out);
    same_levels(&out, &oracle);
    assert_eq!(work, oracle_work, "the implicit route did other work");
    let total: usize = out.levels.iter().map(|l| l.pairs.len()).sum();
    let mut floor = 1u64;
    while floor <= 4 * total as u64 {
        let rules = StopRules { unconditional: None, after_pairs: Some((floor, StopAt::WorkUnits(0))) };
        let (oracle, oracle_work) = chain(parts, pairwise, given, true, rules);
        let (stopped, work) = chain(parts, pairwise, given, false, rules);
        assert_eq!(stopped.is_ok(), oracle.is_ok(), "a stop at {floor} pairs fell differently");
        assert_eq!(work, oracle_work, "a stop at {floor} pairs fell at other work");
        floor = floor * 3 / 2 + 1;
    }
    let mut bound = 1u64;
    while bound <= oracle_work + 1 {
        let rules = StopRules { unconditional: Some(StopAt::WorkUnits(bound)), after_pairs: None };
        let (oracle, oracle_work) = chain(parts, pairwise, given, true, rules);
        let (stopped, work) = chain(parts, pairwise, given, false, rules);
        assert_eq!(stopped.is_ok(), oracle.is_ok(), "a stop at {bound} units fell differently");
        assert_eq!(work, oracle_work, "a stop at {bound} units fell at other work");
        if let (Ok(a), Ok(b)) = (&stopped, &oracle) {
            same_levels(a, b);
        }
        bound = bound * 2 + 1;
    }
    let (implicit, one_pair) = implicit_levels(&out);
    Described { implicit, one_pair, kept: redescribed_levels(&out) }
}

/// Random functions over the classes of the variables modulo `m`,
/// interleaved over a balanced vtree, conjoined in a chain: every product of
/// their nodes is satisfiable, so every level over two internal children is
/// complete, though few of their levels are affine. Returns the implicit
/// levels of the results.
fn chains_over_classes(n: u32, m: u32, seed: u64, rounds: usize) -> usize {
    let vtree = Arc::new(Vtree::balanced(n));
    let mut rng = Lcg::new(seed);
    let mut implicit = 0;
    for _ in 0..rounds {
        let parts: Vec<Tdd> = (0..m)
            .map(|c| {
                let vars: Vec<u32> = (1..=n).filter(|v| v % m == c).collect();
                function_of(&vtree, &vars, &mut rng)
            })
            .collect();
        implicit += same_both_ways(&parts, false, &[]).implicit;
    }
    implicit
}

/// A function of the variables `v` of class `j`, those with
/// `(v - 1) % m == j`, on `vtree`, every level of it affine: a level over a
/// child of fewer than four nodes holds every pair of a node of one child
/// and a node of the other, one pair a node; a level over two children of
/// an even number of nodes, four or more, holds a table, node `n0 + (a / e) · n1` with the pairs
/// `(n0 + (a / e) · x, n1 + (b / e) · x)` for `x < e`, `a` and `b` the
/// children's nodes and `e` two or four; the root holds one node, with the
/// pairs `(x, x)`. Distinct nodes of a level are disjoint, so the pairs of a
/// node are, and no two slots of a child share their contexts.
fn affine_function(eng: &Engine, vtree: &Arc<Vtree>, m: u32, j: u32, rng: &mut Lcg) -> Tdd {
    use crate::diagram::{TddBuilder, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    let mut b: TddBuilder = Tdd::builder(eng, vtree).unwrap();
    let mut nodes: Vec<Vec<NodeIdx>> = vec![Vec::new(); vtree.num_nodes()];
    // The height of each vtree node, and the tables' `e` at each height,
    // the same on both halves.
    let mut height = vec![0usize; vtree.num_nodes()];
    let e_at: Vec<usize> = (0..64).map(|_| if rng.below(2) == 0 { 4 } else { 2 }).collect();
    let root = vtree.root();
    for t in vtree.bottomup() {
        if !vtree.node(t).is_leaf() {
            let (l, r) = vtree.children(t);
            height[t.idx()] = 1 + height[l.idx()].max(height[r.idx()]);
        }
        if vtree.node(t).is_leaf() {
            let v = vtree.leaf_var(t).0;
            nodes[t.idx()] = if (v - 1) % m == j { vec![POS_LEAF_IDX, NEG_LEAF_IDX] } else { vec![ONE_LEAF_IDX] };
            continue;
        }
        let (l, r) = vtree.children(t);
        let (left, right) = (nodes[l.idx()].clone(), nodes[r.idx()].clone());
        let (a, c) = (left.len(), right.len());
        let mut push = |pairs: &[(usize, usize)]| {
            let pairs: Vec<ChildPair> = pairs.iter().map(|&(x, y)| ChildPair::new(left[x], right[y])).collect();
            b.push(eng, t, &pairs).unwrap()
        };
        nodes[t.idx()] = if t == root {
            assert_eq!(a, c, "the two halves are alike");
            vec![push(&(0..a).map(|x| (x, x)).collect::<Vec<_>>())]
        } else if a >= 4 && c >= 4 && a.is_multiple_of(2) && c.is_multiple_of(2) {
            let e = if a.is_multiple_of(4) && c.is_multiple_of(4) { e_at[height[t.idx()]] } else { 2 };
            let (sa, sc) = (a / e, c / e);
            (0..sc).flat_map(|n1| (0..sa).map(move |n0| (n0, n1))).map(|(n0, n1)| push(&(0..e).map(|x| (n0 + sa * x, n1 + sc * x)).collect::<Vec<_>>())).collect()
        } else {
            (0..c).flat_map(|y| (0..a).map(move |x| (x, y))).map(|(x, y)| push(&[(x, y)])).collect()
        };
    }
    let out = nodes[root.idx()][0];
    b.finish(TddNodeId { vtree: root, local: out }).unwrap()
}

/// The functions of [`affine_function`] over every class of `m`, on a
/// balanced vtree of `n` variables, conjoined in a chain, or pairwise for
/// four, then conditioned on `given` random variables. Returns what the
/// results hold described.
fn chains_of_tables(n: u32, m: u32, pairwise: bool, given: usize, seed: u64, rounds: usize) -> Described {
    let vtree = Arc::new(Vtree::balanced(n));
    let eng = Engine::new();
    let mut rng = Lcg::new(seed);
    let mut described = Described::default();
    for _ in 0..rounds {
        let parts: Vec<Tdd> = (0..m).map(|j| affine_function(&eng, &vtree, m, j, &mut rng)).collect();
        let given: Vec<(u32, bool)> = (0..given).map(|_| (1 + rng.below(u64::from(n)) as u32, rng.below(2) == 1)).collect();
        described += same_both_ways(&parts, pairwise, &given);
    }
    described
}

#[test]
fn random_functions_match_the_written_route() {
    chains_over_classes(16, 2, 0x11a7_0001, 8);
    chains_over_classes(18, 3, 0x11a7_0002, 6);
    chains_over_classes(24, 4, 0x11a7_0003, 4);
}

#[test]
fn products_of_two_tables_match_the_written_route() {
    assert!(chains_of_tables(16, 2, false, 0, 0x11a7_0011, 4).implicit > 0, "no level was implicit");
    assert!(chains_of_tables(32, 2, false, 0, 0x11a7_0012, 2).implicit > 0, "no level was implicit");
}

/// A chain over four classes: from the second conjunction on, one
/// operand's levels are implicit; pairwise, both operands' are. The
/// levels over eight variables are products of levels of one pair a node,
/// of 256 nodes of one pair: described, their nodes implied.
#[test]
fn implicit_times_implicit_matches_the_written_route() {
    for (pairwise, seed) in [(false, 0x11a7_0013), (true, 0x11a7_0014)] {
        let described = chains_of_tables(32, 4, pairwise, 0, seed, 3);
        assert!(described.implicit > 0, "no level was implicit");
        assert!(described.one_pair > 0, "no level of one pair a node was implicit");
    }
}

/// Products of tables conditioned on a variable or two: the levels off the
/// path to the variable lose the nodes no longer reached, or see their
/// children's renumbered, and those whose survivors are affine stay
/// described.
#[test]
fn conditioned_products_match_the_written_route() {
    let mut described = chains_of_tables(32, 2, false, 1, 0x11a7_0015, 6);
    described += chains_of_tables(32, 4, true, 2, 0x11a7_0016, 4);
    assert!(described.implicit > 0, "no level was implicit");
    assert!(described.kept > 0, "no prune kept a level described");
}
