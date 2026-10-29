//! Summed levels taken off the description of their pairs checked against
//! the row loop's cells: the same values bit for bit, the same diagram,
//! the same work, and the same stops. The operands are the affine
//! functions of the implicit route's tests, over disjoint classes of
//! variables, so that their products' levels are complete and described;
//! conditioned and contradictory ones leave child grids with dead cells,
//! which the row loop sums.

use super::implicit::affine_function;
use super::*;
use super::{FloatRows, DESCRIBED_SUMS, FLOAT_ROWS};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{rat, with_floor, Lcg};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use num_rational::BigRational;

/// Literal weights for `num_vars` variables: random positive ones, or,
/// with `wide`, weights hundreds of binades apart, whose columns the
/// sums read scaled.
fn weights(rng: &mut Lcg, num_vars: u32, wide: bool) -> RationalWeights {
    let power = |k: i32| {
        let p = BigRational::from_integer(num_bigint::BigInt::from(1) << k.unsigned_abs());
        if k < 0 { p.recip() } else { p }
    };
    RationalWeights::from_literals(
        &(0..num_vars)
            .map(|v| match wide {
                false => LiteralWeights {
                    negative: rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64),
                    positive: rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64),
                },
                true => LiteralWeights {
                    negative: power(600 * ((v % 3) as i32 - 1)),
                    positive: power(-550 * (v % 2) as i32) * rat(3, 1 + v as i64),
                },
            })
            .collect::<Vec<_>>(),
    )
}

/// The internal vtree nodes of height at most `h`, bottom up: every
/// subtree of that height, summed out.
fn targets_up_to(vtree: &Vtree, h: usize) -> Vec<VtreeIdx> {
    let mut height = vec![0usize; vtree.num_nodes()];
    let mut out = Vec::new();
    for t in vtree.bottomup() {
        if vtree.node(t).is_leaf() {
            continue;
        }
        let (l, r) = vtree.children(t);
        height[t.idx()] = 1 + height[l.idx()].max(height[r.idx()]);
        if height[t.idx()] <= h {
            out.push(t);
        }
    }
    out
}

/// `parts` conjoined as `p0 ∧ p1`, or `(p0 ∧ p1) ∧ (p2 ∧ p3)` for four,
/// the last conjunction summing out `targets`, on a fresh engine under
/// `rules`, with the streamed float levels summed as `rows` says: the
/// result and the work it took.
fn sum(parts: &[Tdd], targets: &[VtreeIdx], rows: FloatRows, rules: StopRules) -> (Result<Tdd, OperationError>, u64) {
    let eng = Engine::new();
    FLOAT_ROWS.with(|m| m.set(rows));
    let out = {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        match parts {
            [a, b] => eng.and_marginalizing(a, b, targets),
            [a, b, c, d] => eng
                .and(a, b)
                .and_then(|l| Ok((l, eng.and(c, d)?)))
                .and_then(|(l, r)| eng.and_marginalizing(l, r, targets)),
            _ => unreachable!("two or four parts"),
        }
    };
    FLOAT_ROWS.with(|m| m.set(FloatRows::Sum));
    (out, eng.limits().work_units())
}

/// The bits of a weight value, as the scaled-float arithmetic holds it.
fn bits(v: &crate::diagram::WeightValue) -> (u64, i64) {
    let s = v.as_scaled().expect("a scaled value");
    (s.mantissa().to_bits(), s.exponent())
}

/// Require `out` to be `oracle`: the same root, the same nodes and pairs
/// on every structural level, and the same slots and values, bit for bit,
/// on every summed one.
fn same_sums(out: &Tdd, oracle: &Tdd) {
    assert_eq!(out.output, oracle.output, "another root");
    let (wa, wb) = (out.weights.as_ref().unwrap(), oracle.weights.as_ref().unwrap());
    let (mut a_buf, mut b_buf) = (Vec::new(), Vec::new());
    for (t, (a, b)) in out.levels.iter().zip(&oracle.levels).enumerate() {
        assert_eq!(a.is_marginal(), b.is_marginal(), "level {t}");
        assert_eq!(a.slot_count(), b.slot_count(), "level {t} holds other slots");
        let values = |w: &WeightStore| w.level(t).map(|vs| vs.iter().map(bits).collect::<Vec<_>>());
        assert_eq!(values(wa), values(wb), "level {t} holds other values");
        if !a.is_marginal() {
            for i in 0..a.nodes().len() {
                assert_eq!(a.pairs_read(i, &mut a_buf), b.pairs_read(i, &mut b_buf), "level {t}, node {i}");
            }
        }
    }
}

/// Sum out `targets` of `parts`' conjunction both ways and require the same
/// result and the same work; then, under a stop at each work bound up to
/// what it took, the same stop at the same work.
fn same_both_ways(parts: &[Tdd], targets: &[VtreeIdx]) {
    let none = StopRules { unconditional: None, after_pairs: None };
    let (oracle, oracle_work) = sum(parts, targets, FloatRows::Cells, none);
    let (out, work) = sum(parts, targets, FloatRows::Sum, none);
    same_sums(&out.unwrap(), &oracle.unwrap());
    assert_eq!(work, oracle_work, "the sums off the description did other work");
    let mut bound = 1u64;
    while bound <= oracle_work + 1 {
        let rules = StopRules { unconditional: Some(StopAt::WorkUnits(bound)), after_pairs: None };
        let (oracle, oracle_work) = sum(parts, targets, FloatRows::Cells, rules);
        let (stopped, work) = sum(parts, targets, FloatRows::Sum, rules);
        assert_eq!(stopped.is_ok(), oracle.is_ok(), "a stop at {bound} units fell differently");
        assert_eq!(work, oracle_work, "a stop at {bound} units fell at other work");
        if let (Ok(a), Ok(b)) = (&stopped, &oracle) {
            same_sums(a, b);
        }
        bound = bound * 2 + 1;
    }
}

/// The affine functions of `m` classes on a balanced vtree of `n`
/// variables, weighted, each conditioned on the variables `given` names.
fn parts(n: u32, m: u32, wide: bool, given: &[(u32, bool)], rng: &mut Lcg) -> (Arc<Vtree>, Vec<Tdd>) {
    let vtree = Arc::new(Vtree::balanced(n));
    let eng = Engine::new();
    let store = WeightStore::new(weights(rng, n, wide), Arithmetic::ScaledFloat);
    let parts = (0..m)
        .map(|j| {
            let mut f = affine_function(&eng, &vtree, m, j, rng);
            for &(v, value) in given {
                f = eng.condition_var(f, VarId(v), value).unwrap();
            }
            f.set_weights(store.empty_like()).unwrap();
            f
        })
        .collect();
    (vtree, parts)
}

/// Every height of subtree summed out of the product of two tables or of
/// two products of two, up to the whole product, whose root is one cell.
fn sums_at_every_height(n: u32, m: u32, wide: bool, given: usize, seed: u64) {
    let mut rng = Lcg::new(seed);
    let given: Vec<(u32, bool)> = (0..given).map(|_| (1 + rng.below(u64::from(n)) as u32, rng.below(2) == 1)).collect();
    let (vtree, parts) = parts(n, m, wide, &given, &mut rng);
    let depth = n.ilog2() as usize + 1;
    for h in 1..=depth {
        same_both_ways(&parts, &targets_up_to(&vtree, h));
    }
}

fn described_sums() -> u64 {
    DESCRIBED_SUMS.with(std::cell::Cell::get)
}

/// Complete levels, their child grids every cell's slot in cell order: the
/// products of two tables and of two products of two, at the floor and
/// below it, where small levels are described too and the root's one cell
/// is summed off its description.
#[test]
fn complete_levels_match_the_cells() {
    let before = described_sums();
    sums_at_every_height(16, 2, false, 0, 0x5d_0001);
    sums_at_every_height(32, 2, false, 0, 0x5d_0002);
    sums_at_every_height(32, 4, false, 0, 0x5d_0003);
    assert!(described_sums() > before, "no level was summed off its description");
    let before = described_sums();
    with_floor(2, || {
        sums_at_every_height(8, 2, false, 0, 0x5d_0004);
        sums_at_every_height(16, 4, false, 0, 0x5d_0005);
    });
    assert!(described_sums() > before, "no small level was summed off its description");
}

/// Weights hundreds of binades apart: columns the plain products do not
/// cover are summed by the row loop, the others off their descriptions.
#[test]
fn wide_weights_match_the_cells() {
    sums_at_every_height(16, 2, true, 0, 0x5d_0011);
    sums_at_every_height(32, 4, true, 0, 0x5d_0012);
}

/// Conditioned operands, whose products the condition renumbers, and a
/// contradiction, every cell of whose product is empty: child grids with
/// dead cells, or levels no longer described, are summed by the row loop.
#[test]
fn conditioned_and_empty_products_match_the_cells() {
    with_floor(2, || {
        sums_at_every_height(16, 2, false, 1, 0x5d_0021);
        sums_at_every_height(32, 4, false, 2, 0x5d_0022);
    });
    let mut rng = Lcg::new(0x5d_0023);
    let (vtree, mut parts) = parts(16, 2, false, &[], &mut rng);
    let eng = Engine::new();
    let mut not = eng.negate(&parts[0]).unwrap();
    not.set_weights(parts[0].weights.as_ref().unwrap().empty_like()).unwrap();
    parts[1] = not;
    for h in 1..=5 {
        same_both_ways(&parts, &targets_up_to(&vtree, h));
    }
}
