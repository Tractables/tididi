//! `Tdd::trim`: the diagram moves to the projection of the leaves it names,
//! and every query over the home answers as before.

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::{and2, equiv, exact_weight, rand_conj};
use crate::vtree::rng::Lcg;
use crate::vtree::Home;
use crate::literal;

use super::*;

fn weights(num_vars: u32, rng: &mut Lcg) -> Weights {
    let mut w = || BigRational::new((rng.below(7) as i64 - 2).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..=num_vars).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// A diagram over `home` after random conditioning, forgetting,
/// conjunction and minimization.
fn random_diagram(home: &Arc<Vtree>, n: u32, rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj(home, n, 4, 3, false, rng);
    for _ in 0..rng.below(4) {
        let var = VarId(1 + rng.below(n as u64) as u32);
        f = match rng.below(4) {
            0 => f.condition_var(var, rng.coin()).unwrap(),
            1 => f.exists_vars(&[var]).unwrap(),
            2 => and2(&f, &rand_conj(home, n, 2, 3, false, rng)),
            _ => {
                f.minimize().unwrap();
                f
            }
        };
    }
    crate::test_helpers::padded(f, home)
}

fn var_subset(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    (1..=n).filter(|_| rng.coin()).map(VarId).collect()
}

/// Every query on `g`, a trimmed copy of `f`, answers as on `f`.
fn same_queries(f: &Tdd, g: &Tdd, n: u32, rng: &mut Lcg, label: &str) {
    assert_eq!(f.model_count().unwrap(), g.model_count().unwrap(), "{label}: model count");
    assert_eq!(f.is_sat().unwrap(), g.is_sat().unwrap(), "{label}: satisfiable");
    let vars = var_subset(n, rng);
    assert_eq!(
        f.projected_model_count(&vars).unwrap(),
        g.projected_model_count(&vars).unwrap(),
        "{label}: projected count over {vars:?}",
    );
    let ws = weights(n, rng);
    let algebra = ws.algebra().clone();
    assert_eq!(f.evaluate(&algebra).unwrap(), g.evaluate(&algebra).unwrap(), "{label}: evaluation");
    let weighted = Home::new(Vtree::clone(f.vtree()).with_weights(ws).unwrap());
    let (fw, gw) = (crate::test_helpers::embedded(f, &weighted, |v| v).unwrap().0, crate::test_helpers::embedded(g, &weighted, |v| v).unwrap().0);
    let value = |t: &Tdd| exact_weight(&t.weighted_value().unwrap().unwrap());
    assert_eq!(value(&fw), value(&gw), "{label}: weighted value");
}

#[test]
fn trimmed_copies_answer_as_the_original() {
    let mut trimmed_any = 0;
    for seed in 0..300u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(8) as u32;
        let hm = Home::new(Vtree::random(n, seed));
        let home = Arc::clone(&hm);
        let f = random_diagram(&home, n, &mut rng);
        let label = format!("seed {seed}, {n} variables");

        let mut g = f.clone();
        trimmed_any += usize::from(g.trim(&hm).unwrap());
        check_trimmed(&g, false).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(!g.trim(&hm).unwrap(), "{label}: a second trim moved the diagram");
        same_queries(&f, &g, n, &mut rng, &label);

        if g.vtree().num_nodes() > 0 {
            assert!(g.vtree().is_on(&hm), "{label}: another home");
            let (back, _) = crate::test_helpers::embedded(&g, &home, |v| v).unwrap();
            assert!(equiv(&f, &back), "{label}: the embedded copy is another function");
        }

        let mut h = g.clone();
        h.minimize().unwrap();
        h.trim(&hm).unwrap();
        check_trimmed(&h, true).unwrap_or_else(|e| panic!("{label}, minimized: {e}"));
        same_queries(&f, &h, n, &mut rng, &format!("{label}, minimized"));
    }
    assert!(trimmed_any > 100, "only {trimmed_any} diagrams moved");
}

#[test]
fn constants_have_no_leaf() {
    let hm = Home::new(Vtree::balanced(5));
    let home = Arc::clone(&hm);
    let mut zero = crate::test_helpers::padded(Tdd::zero(&crate::vtree::Home::unchecked(&home)), &home);
    assert!(zero.trim(&hm).unwrap());
    assert!(zero.is_zero() && zero.vtree().num_nodes() == 0);
    assert_eq!(zero.model_count().unwrap(), 0u32.into());

    let mut one = crate::test_helpers::padded(Tdd::one(&crate::vtree::Home::unchecked(&home)), &home);
    assert!(one.trim(&hm).unwrap());
    assert!(one.is_one() && !one.is_zero());
    assert_eq!(one.vtree().num_nodes(), 0);
    assert!(Arc::ptr_eq(&one.home(), &home));
    assert_eq!(one.model_count().unwrap(), 32u32.into());
    assert_eq!(one.projected_model_count(&[VarId(2), VarId(4), VarId(2)]).unwrap(), 4u32.into());
    assert!(one.is_sat().unwrap());
    assert!(!one.trim(&hm).unwrap());
    check_trimmed(&one, true).unwrap();

    let mut x = crate::test_helpers::padded(literal(&crate::vtree::Home::unchecked(&home), 3).unwrap(), &home);
    assert!(x.trim(&hm).unwrap());
    assert_eq!(x.vtree().num_nodes(), 1);
    assert_eq!(x.model_count().unwrap(), 16u32.into());
    check_trimmed(&x, true).unwrap();
}

/// The top own node whose home node is `h` or below it, by walking the home.
fn own_at_by_walk(own: &Vtree, home: &Vtree, h: VtreeIdx) -> Option<VtreeIdx> {
    let under = |mut a: VtreeIdx| loop {
        if a == h {
            return true;
        }
        match home.node(a).parent() {
            Some(p) => a = p,
            None => return false,
        }
    };
    (0..own.num_nodes() as u32)
        .map(VtreeIdx)
        .filter(|&t| under(own.home_node(t)))
        .find(|&t| own.node(t).parent().is_none_or(|p| !under(own.home_node(p))))
}

#[test]
fn own_at_finds_the_top_own_node_under_a_home_node() {
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed);
        let n = 1 + rng.below(12) as u32;
        let hm = Home::new(Vtree::random(n, seed));
        let home = Arc::clone(&hm);
        let mut vars = var_subset(n, &mut rng);
        if vars.is_empty() {
            vars.push(VarId(1));
        }
        let own = Vtree::projection(&hm, vars).unwrap();
        for h in (0..home.num_nodes() as u32).map(VtreeIdx) {
            assert_eq!(own.own_at(h), own_at_by_walk(&own, &home, h), "seed {seed}, home node {}", h.0);
        }
        assert_eq!(home.own_at(home.root()), Some(home.root()));
        let empty = Vtree::empty_projection(&hm);
        assert_eq!(empty.own_at(home.root()), None);
    }
}

#[test]
fn counts_on_a_projection_are_over_the_home() {
    let hm = Home::new(Vtree::balanced(6));
    let home = Arc::clone(&hm);
    let own = Vtree::projection(&hm, [VarId(2), VarId(5)]).unwrap();
    let f = Tdd::clause(&crate::vtree::Home::unchecked(&own), [2, 5]).unwrap();
    assert_eq!(f.model_count().unwrap(), (3u32 << 4).into());
    assert_eq!(f.projected_model_count(&[VarId(2), VarId(1)]).unwrap(), 4u32.into());
    assert_eq!(f.projected_model_count(&[VarId(5)]).unwrap(), 2u32.into());
    let all: Vec<VarId> = (1..=6).map(VarId).collect();
    assert_eq!(f.projected_model_count(&all).unwrap(), f.model_count().unwrap());
    assert!(matches!(f.projected_model_count(&[VarId(7)]), Err(crate::OperationError::VariableNotInVtree(_))));
    let mut on_home = Tdd::clause(&crate::vtree::Home::unchecked(&home), [2, 5]).unwrap();
    on_home.minimize().unwrap();
    assert_eq!(on_home.model_count().unwrap(), f.model_count().unwrap());
}

#[test]
fn merged_nodes_and_leaves_named_by_both_literals() {
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    let eng = Engine::new();
    // x1 ↔ x2, its two nodes under one root node beside true over x3, x4.
    let hm = Home::new(Vtree::balanced(4));
    let home = Arc::clone(&hm);
    let (l, r) = home.children(home.root());
    let mut b = Tdd::builder(&eng, &home).unwrap();
    let same = b.push(&eng, l, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let differ = b.push(&eng, l, &[ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let t = b.push(&eng, r, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let root = b.push(&eng, home.root(), &[ChildPair::new(same, t), ChildPair::new(differ, t)]).unwrap();
    // Seated unchecked: `finish` on a home would trim it.
    let f = unsafe { b.finish_unchecked(TddNodeId { vtree: home.root(), local: root }) };
    let mut g = f.clone();
    assert!(g.trim(&hm).unwrap());
    check_trimmed(&g, false).unwrap();
    assert_eq!(g.vtree().num_nodes(), 3);
    assert_eq!(g.level(g.vtree().root()).slot_count(), 1);
    assert_eq!(g.model_count().unwrap(), 8u32.into());
    assert!(equiv(&f, &crate::test_helpers::embedded(&g, &home, |v| v).unwrap().0));

    // (x1 ∨ ¬x1) beside true: x1 is named by both literals, and goes in a
    // second round.
    let hm = Home::new(Vtree::balanced(2));
    let two = Arc::clone(&hm);
    let mut b = Tdd::builder(&eng, &two).unwrap();
    let root = b
        .push(&eng, two.root(), &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)])
        .unwrap();
    let mut f = unsafe { b.finish_unchecked(TddNodeId { vtree: two.root(), local: root }) };
    assert!(f.trim(&hm).unwrap());
    assert!(f.is_one());
    assert_eq!(f.model_count().unwrap(), 4u32.into());
}

#[test]
fn counters_count_over_the_home() {
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(7) as u32;
        let hm = Home::new(Vtree::random(n, seed));
        let home = Arc::clone(&hm);
        let f = random_diagram(&home, n, &mut rng);
        let mut g = f.clone();
        g.trim(&hm).unwrap();
        let mut fc = f.counter().unwrap();
        let mut gc = g.counter().unwrap();
        assert_eq!(fc.model_count().unwrap(), gc.model_count().unwrap(), "seed {seed}");
        let pinned: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().leaf_of(v).is_some()).take(2).collect();
        for &var in &pinned {
            let value = Some(rng.coin());
            fc.set_pin(var, value).unwrap();
            gc.set_pin(var, value).unwrap();
        }
        assert_eq!(fc.model_count().unwrap(), gc.model_count().unwrap(), "seed {seed}, pinned {pinned:?}");
        fc.clear_pins();
        gc.clear_pins();
        assert_eq!(fc.count_table(&pinned).unwrap(), gc.count_table(&pinned).unwrap(), "seed {seed}, table");
    }
}

/// A deterministic diagram over `home` built level by level, its leaves in
/// `dropped` named true: each internal level's nodes split a random subset
/// of the pairs of its children's nodes, or, at random, hold the whole grid
/// one pair a node in row order, which a seat describes.
fn built_diagram(home: &Arc<Vtree>, dropped: &[bool], rng: &mut Lcg) -> Tdd {
    use crate::diagram::{ChildPair, NodeIdx, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    let eng = Engine::new();
    let mut b = Tdd::builder(&eng, home).unwrap();
    let mut refs: Vec<Vec<NodeIdx>> = vec![Vec::new(); home.num_nodes()];
    for t in home.bottomup() {
        let (l, r) = match *home.node(t) {
            crate::vtree::VtreeNode::Leaf { .. } => {
                refs[t.idx()] = if dropped[t.idx()] { vec![ONE_LEAF_IDX] } else { vec![POS_LEAF_IDX, NEG_LEAF_IDX] };
                continue;
            }
            crate::vtree::VtreeNode::Internal { left, right, .. } => (left, right),
        };
        let mut grid: Vec<ChildPair> =
            refs[l.idx()].iter().flat_map(|&a| refs[r.idx()].iter().map(move |&c| ChildPair::new(a, c))).collect();
        let mut nodes = Vec::new();
        if grid.len() > 1 && rng.below(4) == 0 {
            nodes.extend(grid.iter().map(|&p| vec![p]));
        } else {
            for i in (1..grid.len()).rev() {
                grid.swap(i, rng.below(i as u64 + 1) as usize);
            }
            grid.truncate(1 + rng.below(grid.len().min(8) as u64) as usize);
            let mut node = Vec::new();
            for p in grid {
                node.push(p);
                if rng.below(3) == 0 {
                    nodes.push(std::mem::take(&mut node));
                }
            }
            if !node.is_empty() {
                nodes.push(node);
            }
        }
        refs[t.idx()] = nodes.iter().map(|pairs| b.push(&eng, t, pairs).unwrap()).collect();
    }
    let root = home.root();
    let local = refs[root.idx()][rng.below(refs[root.idx()].len() as u64) as usize];
    // Seated unchecked: `finish` on a home would trim it.
    unsafe { b.finish_unchecked(TddNodeId { vtree: root, local }) }
}

fn leaf_vars(v: &Vtree) -> Vec<VarId> {
    let mut vars: Vec<VarId> =
        (0..v.num_nodes() as u32).map(VtreeIdx).filter(|&t| v.node(t).is_leaf()).map(|t| v.leaf_var(t)).collect();
    vars.sort_unstable();
    vars
}

fn column_values(column: &crate::value::CountVec) -> Vec<num_bigint::BigUint> {
    use crate::value::CountRead;
    (0..column.len())
        .map(|i| match column.get(i) {
            CountRead::Fast(v) => v.into(),
            CountRead::Big(v) => v.clone(),
        })
        .collect()
}

/// A copy of `f` owing every reduction pass every level, as a fresh seat
/// leaves a diagram.
fn owing_everything(f: &Tdd) -> Tdd {
    Tdd::from_levels_unchecked(Arc::clone(f.vtree()), f.levels().to_vec(), f.output())
}

/// One case of [`the_in_place_trim_is_the_rebuild`]: diagrams that dropped
/// one leaf, several, one beside a leaf, that owed contraction, that kept
/// level counts, and that stayed certified count in `seen`. Odd seeds build
/// diagrams by hand, with nodes of several pairs over a dropped side.
fn in_place_against_rebuild(seed: u64, seen: &mut [usize; 6]) {
    use crate::test_helpers::{assert_canonical, assert_same_shape, trim_by_rebuild};
    let mut rng = Lcg::new(seed.wrapping_mul(0x9e37_79b9) ^ 0x7121);
    let n = 2 + rng.below(9) as u32;
    let weighted = rng.below(4) == 0;
    let hm = match weighted {
        true => Home::weighted(Vtree::random(n, seed), weights(n, &mut rng)).unwrap(),
        false => Home::new(Vtree::random(n, seed)),
    };
    let home = Arc::clone(&hm);
    let mut f = if seed.is_multiple_of(2) {
        random_diagram(&home, n, &mut rng)
    } else {
        let dropped: Vec<bool> = (0..home.num_nodes()).map(|_| rng.below(3) == 0).collect();
        built_diagram(&home, &dropped, &mut rng)
    };
    if rng.coin() {
        f.attach_level_counts().unwrap();
    }
    let label = format!("seed {seed}, {n} variables");

    let (mut g, mut h) = (f.clone(), f.clone());
    let moved = g.trim(&hm).unwrap();
    assert_eq!(moved, trim_by_rebuild(&mut h, &hm).unwrap(), "{label}: whether the diagram moved");
    check_trimmed(&g, false).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(!g.clone().trim(&hm).unwrap(), "{label}: a second trim moved the diagram");
    assert_eq!(leaf_vars(g.vtree()), leaf_vars(h.vtree()), "{label}: leaves kept");
    assert_eq!(g.weights().is_some(), weighted, "{label}: weights carried");
    if weighted {
        let value = |t: &Tdd| exact_weight(&t.weighted_value().unwrap().unwrap());
        assert_eq!(value(&g), value(&f), "{label}: weighted value");
    }
    assert_eq!(g.is_zero(), h.is_zero(), "{label}: false");
    assert_eq!(g.is_one(), h.is_one(), "{label}: true");
    if g.vtree().num_nodes() > 0 {
        let back = |t: &Tdd| crate::test_helpers::embedded(t, &home, |v| v).unwrap().0;
        assert!(equiv(&back(&g), &back(&h)), "{label}: another function than the rebuild's");
        assert!(equiv(&back(&g), &f), "{label}: another function");
    }
    if !moved {
        return;
    }

    let dropped: Vec<VarId> = leaf_vars(f.vtree()).into_iter().filter(|v| !leaf_vars(g.vtree()).contains(v)).collect();
    let beside_leaf = dropped.iter().any(|&v| {
        let t = home.leaf_of(v).unwrap();
        home.node(t).parent().is_some_and(|p| {
            let (l, r) = home.children(p);
            home.node(if l == t { r } else { l }).is_leaf()
        })
    });
    seen[0] += usize::from(dropped.len() == 1);
    seen[1] += usize::from(dropped.len() > 1);
    seen[2] += usize::from(beside_leaf);
    seen[3] += usize::from(!f.dirty.is_empty());

    if let Some(kept) = g.levels.counts() {
        seen[4] += 1;
        let mut fresh = g.clone();
        fresh.attach_level_counts().unwrap();
        let fresh = fresh.levels.counts().unwrap();
        for t in (0..g.vtree().num_nodes() as u32).map(VtreeIdx) {
            if let Some(column) = kept.column(t) {
                let want = fresh.column(t).expect("a fresh count of every internal level");
                assert_eq!(column_values(column), column_values(want), "{label}: kept counts at {}", t.0);
            }
        }
    }

    let mut by_worklists = g.clone();
    by_worklists.minimize().unwrap();
    let mut by_fresh_pass = owing_everything(&g);
    by_fresh_pass.minimize().unwrap();
    let mut rebuilt = h.clone();
    rebuilt.minimize().unwrap();
    if g.vtree().num_nodes() > 0 {
        assert_canonical(&by_worklists);
        assert_same_shape(&by_worklists, &by_fresh_pass, &format!("{label}: the worklists against a fresh pass"));
        assert_same_shape(&by_worklists, &rebuilt, &format!("{label}: against the rebuild"));
    }
    if g.vtree().num_nodes() == 0 {
        assert!(g.levels.is_canonical(g.output), "{label}: a constant on no node is certified");
    } else if g.levels.is_canonical(g.output) {
        seen[5] += 1;
        assert!(f.levels.is_canonical(f.output), "{label}: certified by the trim");
        assert_canonical(&g);
        assert_same_shape(&g, &by_fresh_pass, &format!("{label}: certified"));
    }
}

#[test]
fn the_in_place_trim_is_the_rebuild() {
    use crate::test_helpers::{described_here, with_floor, LOW_FLOOR};
    let mut seen = [0usize; 6];
    let described = described_here();
    for seed in 0..800u64 {
        match seed % 4 {
            3 => with_floor(LOW_FLOOR, || in_place_against_rebuild(seed, &mut seen)),
            _ => in_place_against_rebuild(seed, &mut seen),
        }
    }
    assert!(described_here() > described, "no level was implicit");
    assert!(seen.iter().all(|&k| k >= 5), "cases seen: {seen:?}");
}

/// The trim the conjunction runs on its results; `Tdd::trim` refuses a
/// diagram with a marginal level.
fn trim_summed(f: &mut Tdd, home: &Home) -> Result<bool, OperationError> {
    crate::reduce::trim(&crate::Engine::new(), f, home, None, &[])
}

/// `f` on the projection of a home of four variables on 1, 2 and 3, its leaf
/// 2 summed out: the level over leaves 1 and 2 names 1 as true, so it is
/// dropped over the marginal leaf 2, which becomes a region.
fn summed_leaf_under_a_dropped_level(clause: &[i32], vars: &[u32], arithmetic: Option<Arithmetic>) -> (Tdd, Tdd, Home) {
    use crate::marginal::tests::region::weights;
    let home = match arithmetic {
        Some(arithmetic) => Home::weighted(Vtree::balanced(4), Weights::new(weights(4, &mut Lcg::new(7)), arithmetic)).unwrap(),
        None => Home::new(Vtree::balanced(4)),
    };
    let own = Vtree::projection(&home, vars.iter().map(|&v| crate::vtree::VarId(v))).unwrap();
    let mut f = crate::test_helpers::padded(Tdd::clause(&crate::vtree::Home::unchecked(&own), clause.iter().copied()).unwrap(), &own);
    let leaf = own.leaf_of(crate::vtree::VarId(2)).unwrap();
    crate::marginal::marginalize_levels(&Engine::new(), &mut f, &[leaf]).unwrap();
    (f.clone(), f, home)
}

#[test]
fn trim_drops_a_level_over_a_summed_out_leaf() {
    use crate::marginal::tests::region::{check_form, check_values};
    for arithmetic in [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)] {
        let (mut f, g, home) = summed_leaf_under_a_dropped_level(&[2, 3], &[1, 2, 3], arithmetic);
        assert!(trim_summed(&mut f, &home).unwrap(), "{arithmetic:?}");
        assert_eq!(f.vtree().num_leaves(), 1, "{arithmetic:?}");
        assert_eq!(f.vtree().regions().count(), 1, "{arithmetic:?}");
        check_form(&f, &format!("{arithmetic:?}"));
        check_values(&f, &g, 4, arithmetic, &format!("{arithmetic:?}"));
    }
}

#[test]
fn trim_drops_a_root_over_a_summed_out_leaf_when_the_output_names_a_slot() {
    use crate::marginal::tests::region::{check_form, check_values};
    // Over 1 to 4 the chain is two dropped levels long.
    for vars in [&[1, 2][..], &[1, 2, 3, 4]] {
        for arithmetic in [Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)] {
            let what = format!("{vars:?} {arithmetic:?}");
            let (mut f, g, home) = summed_leaf_under_a_dropped_level(&[2], vars, arithmetic);
            assert!(trim_summed(&mut f, &home).unwrap(), "{what}");
            assert_eq!(f.vtree().num_leaves(), 0, "{what}");
            assert!(f.vtree().node(f.vtree().root()).is_region(), "{what}");
            check_form(&f, &what);
            check_values(&f, &g, 4, arithmetic, &what);
        }
    }
    // An inline count at the root gets a level of one slot.
    let (mut f, g, home) = summed_leaf_under_a_dropped_level(&[2], &[1, 2], None);
    assert_eq!(trim_summed(&mut f, &home), Ok(true));
    assert!(f.vtree().node(f.vtree().root()).is_region());
    check_form(&f, "inline");
    check_values(&f, &g, 4, None, "inline");
}
