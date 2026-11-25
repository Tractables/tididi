//! `trim` on diagrams with regions: a leaf every pair names true goes, also
//! where the chain above it reaches a region and its nodes stand for sums of
//! that region's values. Checked against a pointwise evaluation of the
//! untrimmed diagram on homes of at most ten variables.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

use crate::diagram::{Arithmetic, ChildDecoder, ChildPair, EncodedChildRef, TddBuilder, TddLevel, ValueRef, Weights, WeightValue, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::marginal::tests::region::{operand, reweighted, under, weights};
use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::deadline_probe;
use crate::value::CountRead;
use crate::vtree::rng::Lcg;
use crate::vtree::Home;
use crate::OperationError;

use super::*;

fn rational(w: &WeightValue) -> BigRational {
    match w.as_scaled() {
        Some(s) => BigRational::from_float(s.to_f64()).expect("a finite weight"),
        None => w.as_rational().into_owned(),
    }
}

/// The value of `f` where its variable leaves take `x`: a node is the sum
/// over its pairs of the product of what they name, a leaf named true is 1,
/// a region reads its value.
fn value_at(f: &Tdd, x: &dyn Fn(VarId) -> bool) -> BigRational {
    let v = f.vtree();
    if v.num_nodes() == 0 {
        return BigRational::from_integer(BigInt::from(u8::from(f.is_one())));
    }
    let mut cols: Vec<Vec<BigRational>> = vec![Vec::new(); v.num_nodes()];
    let read = |cols: &[Vec<BigRational>], c: VtreeIdx, side: EncodedChildRef| -> BigRational {
        if !f.level(c).is_marginal() {
            return cols[c.idx()][side.raw() as usize].clone();
        }
        if side.is_reserved() {
            return BigRational::zero();
        }
        match ChildDecoder::marginal().value(side) {
            ValueRef::Inline(n) => BigRational::from_integer(BigInt::from(n)),
            ValueRef::Slot(s) => cols[c.idx()][s as usize].clone(),
        }
    };
    for t in v.bottomup() {
        let level = f.level(t);
        cols[t.idx()] = if level.is_marginal() {
            match level.count_column() {
                Some(c) => (0..c.len())
                    .map(|i| match c.get(i) {
                        CountRead::Fast(n) => BigRational::from_integer(BigInt::from(n)),
                        CountRead::Big(n) => BigRational::from_integer(BigInt::from(n.clone())),
                    })
                    .collect(),
                None => level.weight_values().expect("a summed level holds weights").iter().map(rational).collect(),
            }
        } else if v.node(t).is_leaf() {
            let on = x(v.leaf_var(t));
            let mut c = vec![BigRational::zero(); 3];
            c[POS_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(u8::from(on)));
            c[crate::diagram::NEG_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(u8::from(!on)));
            c[ONE_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(1));
            c
        } else {
            let (l, r) = v.children(t);
            (0..level.slot_count())
                .map(|j| level.pairs_iter_of_idx(j).map(|p| read(&cols, l, p.left) * read(&cols, r, p.right)).sum())
                .collect()
        };
    }
    cols[v.root().idx()][f.output().local.idx()].clone()
}

fn variables(f: &Tdd) -> Vec<VarId> {
    let v = f.vtree();
    v.bottomup().filter(|&t| v.node(t).is_var_leaf()).map(|t| v.leaf_var(t)).collect()
}

/// The value of `f` where the variables `vars` take `x`, summed over its
/// other variable leaves and counting each home variable it does not hold,
/// as a leaf or in a region, twice.
fn value_over(f: &Tdd, n: u32, vars: &[VarId], x: u32) -> BigRational {
    let v = f.vtree();
    let summed: u32 = v.regions().map(|t| v.region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum();
    let others: Vec<VarId> = variables(f).into_iter().filter(|w| !vars.contains(w)).collect();
    let free = n - variables(f).len() as u32 - summed;
    let mut total = BigRational::zero();
    for y in 0..1u32 << others.len() {
        let at = |var: VarId| match vars.iter().position(|&w| w == var) {
            Some(i) => x >> i & 1 == 1,
            None => others.iter().position(|&w| w == var).is_some_and(|i| y >> i & 1 == 1),
        };
        total += value_at(f, &at);
    }
    total * BigRational::from_integer(BigInt::from(1u64 << free))
}

/// `f` and `g`, `g`'s variable leaves a subset of `f`'s, agree at every
/// assignment of `g`'s variable leaves, the others summed; exactly or, in
/// floats, to rounding.
fn same_pointwise(f: &Tdd, g: &Tdd, n: u32, floats: bool, what: &str) {
    let vars = variables(g);
    for bits in 0..1u32 << vars.len() {
        let (a, b) = (value_over(f, n, &vars, bits), value_over(g, n, &vars, bits));
        if floats {
            let (a, b) = (a.to_f64().unwrap(), b.to_f64().unwrap());
            assert!((a - b).abs() <= 1e-9 * a.abs().max(b.abs()), "{what}: at {bits:b} {a} against {b}");
        } else {
            assert_eq!(a, b, "{what}: at {bits:b}");
        }
    }
}

/// `f` conditioned on each leaf of `leaves` taking its literal, as levels for
/// a builder, each leaf left named true: at its parent a pair naming the
/// literal names the leaf true and a pair naming the other literal goes. A
/// node that held the literal in two pairs stands for the sum of what they
/// named, the pairs' products being disjoint; two nodes of a level may
/// become equal. Then [`multiples`] with `seed`. `None` when a node is left
/// with no pair.
fn named_true(f: &Tdd, leaves: &[(VtreeIdx, u32)], seed: u64) -> Option<(TddBuilder, crate::diagram::TddNodeId)> {
    let eng = Engine::new();
    let v = Arc::clone(f.vtree());
    let mut levels = f.clone().levels.into_vec();
    for &(x, lit) in leaves {
        let p = v.node(x).parent().expect("a leaf under a node");
        let left = v.children(p).0 == x;
        let old = &levels[p.idx()];
        let mut level = TddLevel::new();
        for j in 0..old.slot_count() {
            let side = |q: &ChildPair| if left { q.left } else { q.right }.raw();
            let pairs: Vec<ChildPair> = old
                .pairs_iter_of_idx(j)
                .filter(|q| [lit, ONE_LEAF_IDX.0].contains(&side(q)))
                .map(|q| match left {
                    true => ChildPair::new(ONE_LEAF_IDX, q.right),
                    false => ChildPair::new(q.left, ONE_LEAF_IDX),
                })
                .collect();
            if pairs.is_empty() {
                return None;
            }
            level.push_node(eng.limits(), &pairs).unwrap();
        }
        if !summed_under(&v, &levels)[p.idx()] && has_equal_nodes(&level) {
            return None;
        }
        levels[p.idx()] = level;
    }
    multiples(&v, &mut levels, seed);
    Some((TddBuilder::from_levels(v, levels), f.output()))
}

/// Whether a summed-out level is at or under each vtree node: where none is,
/// a level holds no pair twice, in one node or across two.
fn summed_under(v: &Vtree, levels: &[TddLevel]) -> Vec<bool> {
    let mut under = vec![false; v.num_nodes()];
    for t in v.bottomup() {
        under[t.idx()] = levels[t.idx()].is_marginal()
            || (!v.node(t).is_leaf() && {
                let (l, r) = v.children(t);
                under[l.idx()] || under[r.idx()]
            });
    }
    under
}

fn has_equal_nodes(level: &TddLevel) -> bool {
    let mut nodes: Vec<Vec<(u32, u32)>> = (0..level.slot_count())
        .map(|j| {
            let mut n: Vec<(u32, u32)> = level.pairs_iter_of_idx(j).map(|q| (q.left.raw(), q.right.raw())).collect();
            n.sort_unstable();
            n
        })
        .collect();
    nodes.sort_unstable();
    nodes.windows(2).any(|w| w[0] == w[1])
}

/// The pairs of node `j` of `level`.
fn pairs_of(level: &TddLevel, j: usize) -> Vec<ChildPair> {
    level.pairs_iter_of_idx(j).collect()
}

/// `level` rebuilt node by node, `edit` changing each node's pairs.
fn rebuilt(level: &TddLevel, mut edit: impl FnMut(usize, &mut Vec<ChildPair>)) -> TddLevel {
    let eng = Engine::new();
    let mut out = TddLevel::new();
    for j in 0..level.slot_count() {
        let mut pairs = pairs_of(level, j);
        edit(j, &mut pairs);
        out.push_node(eng.limits(), &pairs).unwrap();
    }
    out
}

/// By `seed`, nothing, or at random one of: a node at a structural level with
/// a summed-out level under it holds one of its pairs once more, which adds
/// that pair's product again; a node of a structural level under another is
/// copied and one reference to it moves to the copy, which leaves two equal
/// nodes.
fn multiples(v: &Vtree, levels: &mut [TddLevel], seed: u64) {
    let mut rng = Lcg::new(seed);
    let summed_under = summed_under(v, levels);
    let structural: Vec<VtreeIdx> = v.bottomup().filter(|&t| !v.node(t).is_leaf() && !levels[t.idx()].is_marginal()).collect();
    if structural.is_empty() {
        return;
    }
    match rng.below(3) {
        0 => {
            let held: Vec<VtreeIdx> = structural.iter().copied().filter(|&t| summed_under[t.idx()]).collect();
            if held.is_empty() {
                return;
            }
            let t = held[rng.below(held.len() as u64) as usize];
            let j = rng.below(levels[t.idx()].slot_count() as u64) as usize;
            levels[t.idx()] = rebuilt(&levels[t.idx()], |i, pairs| {
                if i == j {
                    pairs.push(pairs[0]);
                }
            });
        }
        1 => {
            let copied: Vec<VtreeIdx> = structural
                .iter()
                .copied()
                .filter(|&t| summed_under[t.idx()] && v.node(t).parent().is_some_and(|p| !levels[p.idx()].is_marginal()))
                .collect();
            if copied.is_empty() {
                return;
            }
            let t = copied[rng.below(copied.len() as u64) as usize];
            let p = v.node(t).parent().unwrap();
            let left = v.children(p).0 == t;
            let j = rng.below(levels[t.idx()].slot_count() as u64) as u32;
            let copy = levels[t.idx()].slot_count() as u32;
            let pairs = pairs_of(&levels[t.idx()], j as usize);
            levels[t.idx()].push_node(Engine::new().limits(), &pairs).unwrap();
            let mut moved = false;
            levels[p.idx()] = rebuilt(&levels[p.idx()], |_, pairs| {
                for q in pairs.iter_mut() {
                    let side = if left { &mut q.left } else { &mut q.right };
                    if !moved && side.raw() == j {
                        *side = crate::diagram::NodeIdx(copy).into();
                        moved = true;
                    }
                }
            });
        }
        _ => {}
    }
}

/// One or two variable leaves of `f` at random, no two siblings: a node over
/// two leaves both named true would stand for the number of its pairs, which
/// no Boolean diagram holds.
/// Each with the literal it is conditioned on: one whose every node at the
/// parent holds a pair it names.
fn pick_leaves(f: &Tdd, rng: &mut Lcg) -> Vec<(VtreeIdx, u32)> {
    let v = f.vtree();
    let mut leaves: Vec<VtreeIdx> = v.bottomup().filter(|&t| v.node(t).is_var_leaf() && v.node(t).parent().is_some()).collect();
    let mut picked: Vec<(VtreeIdx, u32)> = Vec::new();
    for _ in 0..1 + rng.below(2) {
        if leaves.is_empty() {
            break;
        }
        let x = leaves.swap_remove(rng.below(leaves.len() as u64) as usize);
        let p = v.node(x).parent().unwrap();
        let left = v.children(p).0 == x;
        let level = f.level(p);
        let held = |lit: u32| {
            (0..level.slot_count()).all(|j| level.pairs_iter_of_idx(j).any(|q| [lit, ONE_LEAF_IDX.0].contains(&if left { q.left } else { q.right }.raw())))
        };
        let lits: Vec<u32> = [POS_LEAF_IDX.0, crate::diagram::NEG_LEAF_IDX.0].into_iter().filter(|&lit| held(lit)).collect();
        if !lits.is_empty() && picked.iter().all(|&(y, _)| v.node(y).parent() != Some(p)) {
            picked.push((x, lits[rng.below(lits.len() as u64) as usize]));
        }
    }
    picked
}

/// Whether a node at the parent of a leaf of `leaves` whose sibling is a
/// region holds two pairs: the trim must then add values.
fn sums_over_a_region(f: &Tdd, leaves: &[(VtreeIdx, u32)]) -> bool {
    let v = f.vtree();
    leaves.iter().any(|&(x, _)| {
        let p = v.node(x).parent().unwrap();
        let (l, r) = v.children(p);
        let sibling = if l == x { r } else { l };
        v.node(sibling).is_region() && (0..f.level(p).slot_count()).any(|j| f.level(p).pairs_iter_of_idx(j).len() > 1)
    })
}

/// A diagram with a region on a random home of at most ten variables, and
/// the arithmetic of its values.
fn with_region(case: u64, rng: &mut Lcg) -> Option<(Home, Tdd, Option<Arithmetic>)> {
    let eng = Engine::new();
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let arithmetic = modes[case as usize % modes.len()];
    let n = 3 + rng.below(8) as u32;
    let mut home = Home::new(Vtree::random(n, case));
    let all: Vec<VarId> = home.leaf_bottomup().map(|(_, var)| var).collect();
    let used: Vec<VarId> = all.iter().copied().filter(|_| rng.below(4) != 0).collect();
    let mut f = operand(&eng, &home, &used, rng);
    if let Some(arithmetic) = arithmetic {
        let weighted = Home::weighted(Vtree::random(n, case), Weights::new(weights(n, rng), arithmetic)).unwrap();
        f = reweighted(&f, &home, &weighted);
        home = weighted;
    }
    let h = VtreeIdx(rng.below(home.num_nodes() as u64) as u32);
    eng.marginalize(&mut f, under(&home, &[h])).ok()?;
    f.vtree().has_regions().then_some((home, f, arithmetic))
}

#[test]
fn a_dropped_leaf_over_a_region_adds_the_values_its_nodes_name() {
    let mut rng = Lcg::new(0x7a1d_5eed);
    let (mut cases, mut sums, mut multiples, mut built) = (0, 0, 0, 0);
    for case in 0..6000u64 {
        let Some((home, f, arithmetic)) = with_region(case, &mut rng) else { continue };
        let leaves = pick_leaves(&f, &mut rng);
        if leaves.is_empty() {
            continue;
        }
        let what = format!("case {case} leaves {leaves:?} arithmetic {arithmetic:?}");
        let Some((b, out)) = named_true(&f, &leaves, case) else { continue };
        if b.check(out).is_err() {
            continue;
        }
        let untrimmed = unsafe { b.finish_unchecked(out) };
        let n = home.num_leaves();
        let floats = arithmetic == Some(Arithmetic::ScaledFloat);
        let mut t = untrimmed.clone();
        Engine::new().trim(&mut t, &home).unwrap_or_else(|e| panic!("{what}: {e:?}"));        same_pointwise(&untrimmed, &t, n, floats, &what);
        check_trimmed(&t, false).unwrap_or_else(|e| panic!("{what}: {e}"));
        cases += 1;
        sums += usize::from(sums_over_a_region(&untrimmed, &leaves));
        multiples += usize::from(holds_a_multiple(&untrimmed));
        let Some((b, out)) = named_true(&f, &leaves, case) else { continue };
        let g = b.finish(out).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        same_pointwise(&untrimmed, &g, n, floats, &format!("{what}, built"));
        check_trimmed(&g, false).unwrap_or_else(|e| panic!("{what}, built: {e}"));
        built += 1;
    }
    println!("{cases} cases, {sums} adding values over a region, {multiples} with a repeated pair or equal nodes, {built} built");
    assert!(cases > 400 && sums > 40 && multiples > 100 && built > 400, "{cases} cases, {sums} adding values, {multiples} multiples, {built} built");
}

/// Whether a level of `f` holds a pair twice in a node, or two equal nodes.
fn holds_a_multiple(f: &Tdd) -> bool {
    f.vtree().bottomup().filter(|&t| !f.vtree().node(t).is_leaf() && !f.level(t).is_marginal()).any(|t| {
        let level = f.level(t);
        has_equal_nodes(level)
            || (0..level.slot_count()).any(|j| {
                let mut pairs = pairs_of(level, j);
                pairs.sort_unstable();
                pairs.windows(2).any(|w| w[0] == w[1])
            })
    })
}

/// A refused allocation or a stop at any point of a trim that adds values
/// leaves the diagram as it was.
#[test]
fn a_refused_trim_over_a_region_leaves_the_diagram() {
    let mut rng = Lcg::new(0x51ed);
    let mut checked = 0;
    for case in 0..3000u64 {
        if checked == 6 {
            break;
        }
        let Some((home, f, arithmetic)) = with_region(case, &mut rng) else { continue };
        let leaves = pick_leaves(&f, &mut rng);
        let Some((b, out)) = named_true(&f, &leaves, case) else { continue };
        if leaves.is_empty() || b.check(out).is_err() {
            continue;
        }
        let untrimmed = unsafe { b.finish_unchecked(out) };
        if !sums_over_a_region(&untrimmed, &leaves) {
            continue;
        }
        checked += 1;
        let n = home.num_leaves();
        let floats = arithmetic == Some(Arithmetic::ScaledFloat);
        let unchanged = |g: &Tdd, what: &str| {
            assert!(Arc::ptr_eq(g.vtree(), untrimmed.vtree()), "{what}: the vtree moved");
            same_pointwise(&untrimmed, g, n, floats, what);
        };
        let mut refused = 0;
        for cut in 0.. {
            let eng = Engine::new();
            let mut g = untrimmed.clone();
            eng.limits().refuse_nth_reserve(cut);
            let result = eng.trim(&mut g, &home);
            let pending = eng.limits().refusal_pending();
            match result {
                Err(e) => {
                    assert_eq!(e, OperationError::OverBudget, "case {case} cut {cut}");
                    unchanged(&g, &format!("case {case} refused at reserve {cut}"));
                    refused += 1;
                }
                Ok(_) if pending => break,
                Ok(_) => {}
            }
        }
        assert!(refused > 0, "case {case}: no reserve refused");
        for stride in [1, 2, 3, 5, 8] {
            let mut g = untrimmed.clone();
            if deadline_probe(Some(stride), |eng| eng.trim(&mut g, &home)).is_err() {
                unchanged(&g, &format!("case {case} stopped at stride {stride}"));
            }
        }
    }
    assert_eq!(checked, 6);
}

/// On `home`, whose root is `(q, n)` with `q = (r, s)` and `r` the region
/// over `{1, 2}`: the region naming 4 = 2^2 in every reference, the one node
/// of `q` holding `(4, side)` twice, `side` what `fill` puts under `s`, and
/// leaf `n` named true at the root. Dropping the region as true would leave
/// twice what `side` names, with no summed-out level under `s` to carry it:
/// the region stays and leaf `n` goes.
fn a_pair_held_twice_beside_a_full_region(home: &Home, n: u32, fill: impl FnOnce(VtreeIdx, &mut [TddLevel]) -> EncodedChildRef) {
    use crate::diagram::{NodeIdx, TddNodeId};
    let vars: Vec<VarId> = (1..=n).map(VarId).collect();
    let clause: Vec<i32> = (1..=n as i32).collect();
    let mut g = Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(home, vars).unwrap()), clause).unwrap();
    g.marginalize(|v| v.0 <= 2).unwrap();
    let v = Arc::clone(g.vtree());
    let root = v.root();
    let (q, last) = v.children(root);
    let (r, s) = v.children(q);
    assert!(v.node(r).is_region() && v.leaf_var(last) == VarId(n));
    let eng = Engine::new();
    let mut levels = g.clone().levels.into_vec();
    let mut region = TddLevel::new();
    region.become_marginal(Vec::new(), None);
    levels[r.idx()] = region;
    let side = fill(s, &mut levels);
    let full = EncodedChildRef::from_raw(ValueRef::inline_raw(4).unwrap());
    levels[q.idx()] = TddLevel::new();
    levels[q.idx()].push_node(eng.limits(), &[ChildPair::new(full, side); 2]).unwrap();
    levels[root.idx()] = TddLevel::new();
    levels[root.idx()].push_node(eng.limits(), &[ChildPair::new(NodeIdx(0), ONE_LEAF_IDX)]).unwrap();
    let out = TddNodeId { vtree: root, local: NodeIdx(0) };
    let b = TddBuilder::from_levels(Arc::clone(&v), levels.clone());
    b.check(out).unwrap();
    let untrimmed = unsafe { b.finish_unchecked(out) };
    let mut t = untrimmed.clone();
    assert_eq!(eng.trim(&mut t, home), Ok(true));
    check_trimmed(&t, false).unwrap();
    assert_eq!(t.vtree().regions().count(), 1);
    assert!(t.vtree().leaf_of(VarId(n)).is_none());
    same_pointwise(&untrimmed, &t, n, false, "trimmed");
    let built = TddBuilder::from_levels(v, levels).finish(out).unwrap();
    check_trimmed(&built, false).unwrap();
    same_pointwise(&untrimmed, &built, n, false, "built");
}

fn leaf(v: u32) -> Vtree {
    Vtree::leaf(VarId(v))
}

fn join(a: &Vtree, b: &Vtree) -> Vtree {
    Vtree::join(a, b).unwrap()
}

/// `s` is leaf 3, named `x3`: twice a label.
#[test]
fn a_pair_held_twice_over_a_leaf_beside_a_full_region_keeps_the_region() {
    let home = Home::new(join(&join(&join(&leaf(1), &leaf(2)), &leaf(3)), &leaf(4)));
    a_pair_held_twice_beside_a_full_region(&home, 4, |_, _| POS_LEAF_IDX.into());
}

/// `s` is `(3, 4)` holding `x3 ∧ x4`: twice a node with no summed-out level
/// under it.
#[test]
fn a_pair_held_twice_over_a_level_beside_a_full_region_keeps_the_region() {
    let home = Home::new(join(&join(&join(&leaf(1), &leaf(2)), &join(&leaf(3), &leaf(4))), &leaf(5)));
    a_pair_held_twice_beside_a_full_region(&home, 5, |s, levels| {
        levels[s.idx()] = TddLevel::new();
        levels[s.idx()].push_node(Engine::new().limits(), &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
        crate::diagram::NodeIdx(0).into()
    });
}

/// Home `(((1, 2), 3), 4)` with no summed-out level; `(1, 2)` holds `x1 ∧
/// x2`, its parent two equal nodes naming it with leaf 3 true, and the root
/// `x4` to the one and `¬x4` to the other. Both nodes of the dropped level
/// keep `x1 ∧ x2`.
#[test]
fn two_equal_nodes_of_a_dropped_level_both_keep_what_they_name() {
    use crate::diagram::{NodeIdx, TddNodeId, NEG_LEAF_IDX};
    let home = Home::new(join(&join(&join(&leaf(1), &leaf(2)), &leaf(3)), &leaf(4)));
    let g = Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [1, 2, 3, 4].map(VarId)).unwrap()), [1, 2, 3, 4]).unwrap();
    let v = Arc::clone(g.vtree());
    let root = v.root();
    let (t, _) = v.children(root);
    let (e, _) = v.children(t);
    let eng = Engine::new();
    let mut levels = g.clone().levels.into_vec();
    let mut node = |at: VtreeIdx, nodes: &[&[ChildPair]]| {
        levels[at.idx()] = TddLevel::new();
        for pairs in nodes {
            levels[at.idx()].push_node(eng.limits(), pairs).unwrap();
        }
    };
    node(e, &[&[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]]);
    let named = [ChildPair::new(NodeIdx(0), ONE_LEAF_IDX)];
    node(t, &[&named, &named]);
    node(root, &[&[ChildPair::new(NodeIdx(0), POS_LEAF_IDX), ChildPair::new(NodeIdx(1), NEG_LEAF_IDX)]]);
    let out = TddNodeId { vtree: root, local: NodeIdx(0) };
    let b = TddBuilder::from_levels(v, levels);
    b.check(out).unwrap();
    let untrimmed = unsafe { b.finish_unchecked(out) };
    let mut trimmed = untrimmed.clone();
    assert_eq!(eng.trim(&mut trimmed, &home), Ok(true));
    check_trimmed(&trimmed, false).unwrap();
    assert!(trimmed.vtree().leaf_of(VarId(3)).is_none());
    same_pointwise(&untrimmed, &trimmed, 4, false, "trimmed");
    assert_eq!(trimmed.model_count().unwrap(), num_bigint::BigUint::from(4u32));
}

/// Home `(((1, (2, 3)), 4), 5)` with `(2, 3)` summed out into a region `r`;
/// `p = (1, r)` holds `x1` and `¬x1`, each with value 1 at `r`. Leaf 4 is
/// named true at `q = (p, 4)`, whose two nodes name `{x1, ¬x1}` and `{x1}`,
/// and the root holds `¬x5` to the one and `x5` to the other. The end node
/// `x1` is named by both nodes of the dropped level, and each keeps it.
#[test]
fn an_end_node_two_dropped_nodes_name_beside_a_region_stays_in_both() {
    use crate::diagram::{NodeIdx, TddNodeId, NEG_LEAF_IDX};
    let home = Home::new(join(&join(&join(&leaf(1), &join(&leaf(2), &leaf(3))), &leaf(4)), &leaf(5)));
    let mut g = Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [1, 2, 3, 4, 5].map(VarId)).unwrap()), [1, 2, 3, 4, 5]).unwrap();
    g.marginalize(|v| [2, 3].contains(&v.0)).unwrap();
    let v = Arc::clone(g.vtree());
    let root = v.root();
    let (q, _) = v.children(root);
    let (p, _) = v.children(q);
    let (_, r) = v.children(p);
    assert!(v.node(r).is_region());
    let eng = Engine::new();
    let mut levels = g.clone().levels.into_vec();
    let mut node = |at: VtreeIdx, nodes: &[&[ChildPair]]| {
        levels[at.idx()] = TddLevel::new();
        for pairs in nodes {
            levels[at.idx()].push_node(eng.limits(), pairs).unwrap();
        }
    };
    let one = EncodedChildRef::from_raw(ValueRef::inline_raw(1).unwrap());
    node(p, &[&[ChildPair::new(POS_LEAF_IDX, one)], &[ChildPair::new(NEG_LEAF_IDX, one)]]);
    node(q, &[&[ChildPair::new(NodeIdx(0), ONE_LEAF_IDX), ChildPair::new(NodeIdx(1), ONE_LEAF_IDX)], &[ChildPair::new(NodeIdx(0), ONE_LEAF_IDX)]]);
    node(root, &[&[ChildPair::new(NodeIdx(0), NEG_LEAF_IDX), ChildPair::new(NodeIdx(1), POS_LEAF_IDX)]]);
    let mut region = TddLevel::new();
    region.become_marginal(Vec::new(), None);
    levels[r.idx()] = region;
    let out = TddNodeId { vtree: root, local: NodeIdx(0) };
    let b = TddBuilder::from_levels(v, levels);
    b.check(out).unwrap();
    let untrimmed = unsafe { b.finish_unchecked(out) };
    let mut trimmed = untrimmed.clone();
    assert_eq!(eng.trim(&mut trimmed, &home), Ok(true));
    check_trimmed(&trimmed, false).unwrap();
    assert!(trimmed.vtree().leaf_of(VarId(4)).is_none());
    same_pointwise(&untrimmed, &trimmed, 5, false, "trimmed");
}
