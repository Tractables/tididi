//! A level whose one marginal child is a pass-through — the other operand is
//! the identity there, so that side kills no pair — is a join on its other
//! child alone. The sparse route scatters from that child's live products and
//! carries the marginal side across. A grid walk would first densify the
//! joined child's grid, the product of the operands' widths there, and then
//! visit every cell of the level with every pair of its row.
//!
//! This is the shape of the second step of a chain of conjunctions that sums
//! variables out as it goes: the running conjunction has summed a variable
//! out, and the next operand, which does not mention it, is the identity
//! over that subtree.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use num_bigint::BigUint;
use num_rational::BigRational;

use super::{ForcedThresholds, SparseThresholds};
use super::inner_index::{block, pack};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Tdd, WeightStore};
use crate::limits::{LimitConfig, MemoryHooks};
use crate::test_helpers::{assert_marginal_canonical, assert_same_shape};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

/// The blocks `a`, `b`, `c` of `bits` variables each, and the vtree
/// `((a b) c)` over them — `((b a) c)` with `b_first` — each block a balanced
/// subtree.
fn blocks_and_vtree(bits: u32, b_first: bool) -> ([Vec<VarId>; 3], Arc<Vtree>) {
    let (a, b, c) = (block(1, bits), block(1 + bits, bits), block(1 + 2 * bits, bits));
    let (first, second) = if b_first { (&b, &a) } else { (&a, &b) };
    let ab = Vtree::join(
        &Vtree::balanced_over(first).unwrap(),
        &Vtree::balanced_over(second).unwrap(),
    )
    .unwrap();
    let vtree = Arc::new(Vtree::join(&ab, &Vtree::balanced_over(&c).unwrap()).unwrap());
    ([a, b, c], vtree)
}

/// The root of the subtree over `block`.
fn subtree(vtree: &Vtree, block: &[VarId]) -> VtreeIdx {
    let leaf = |v: VarId| vtree.leaf_of(v).expect("a block variable is in the vtree");
    vtree.lca(leaf(block[0]), leaf(block[block.len() - 1]))
}

/// Which keys `e` and `k` keep, and the `c` value `k` gives a key.
#[derive(Clone, Copy)]
struct Rows {
    in_e: fn(u64) -> bool,
    in_k: fn(u64) -> bool,
    c_of: fn(u64) -> u64,
}

/// The star `f(a, b) ∧ e(a) ∧ k(a, c)` over `keys` values of `a`, each with
/// its own `b` and one `c`. Every key `e` and `k` both keep is one model.
struct Star {
    vtree: Arc<Vtree>,
    blocks: [Vec<VarId>; 3],
    f: Tdd,
    e: Tdd,
    k: Tdd,
    models: BigUint,
}

fn star(eng: &Engine, bits: u32, keys: u64, b_first: bool, rows: Rows) -> Star {
    let (blocks, vtree) = blocks_and_vtree(bits, b_first);
    let [a, b, c] = &blocks;
    let modulus = 1u64 << bits;
    let Rows { in_e, in_k, c_of } = rows;
    let f_rows: Vec<Vec<u64>> = (0..keys).map(|x| vec![x, (x * 5 + 1) % modulus]).collect();
    let e_rows: Vec<Vec<u64>> = (0..keys).filter(|&x| in_e(x)).map(|x| vec![x]).collect();
    let k_rows: Vec<Vec<u64>> = (0..keys).filter(|&x| in_k(x)).map(|x| vec![x, c_of(x) % modulus]).collect();
    let (fv, fr) = pack(&[a, b], &f_rows);
    let (ev, er) = pack(&[a], &e_rows);
    let (kv, kr) = pack(&[a, c], &k_rows);
    let models = (0..keys).filter(|&x| in_e(x) && in_k(x)).count();
    Star {
        f: eng.from_models(&vtree, &fv, &fr).unwrap(),
        e: eng.from_models(&vtree, &ev, &er).unwrap(),
        k: eng.from_models(&vtree, &kv, &kr).unwrap(),
        models: BigUint::from(models),
        vtree,
        blocks,
    }
}

impl Star {
    /// The same star with `store` attached to every relation.
    fn weighted(&self, store: &WeightStore) -> Star {
        let weigh = |t: &Tdd| {
            let mut t = t.clone();
            t.set_weights(store.clone()).unwrap();
            t
        };
        Star {
            vtree: Arc::clone(&self.vtree),
            blocks: self.blocks.clone(),
            f: weigh(&self.f),
            e: weigh(&self.e),
            k: weigh(&self.k),
            models: self.models.clone(),
        }
    }

    /// `f ∧ e` with `b` summed out: the running conjunction, marginal over `b`.
    fn summed(&self, eng: &Engine) -> Tdd {
        let b = subtree(&self.vtree, &self.blocks[1]);
        let summed = eng.and_marginalizing(self.f.clone(), self.e.clone(), &[b]).unwrap();
        assert!(summed.level(b).is_marginal());
        summed
    }

    /// The second join, summing `c` out. At the level over `a` and `b`, `b` is
    /// marginal in `summed` and `k` is the identity there.
    fn second(&self, eng: &Engine, summed: Tdd) -> Tdd {
        let c = subtree(&self.vtree, &self.blocks[2]);
        eng.and_marginalizing(summed, self.k.clone(), &[c]).unwrap()
    }
}

#[test]
fn a_pass_through_level_never_densifies_its_joined_child() {
    let eng = Engine::new();
    let rows = Rows { in_e: |x| x % 10 != 0, in_k: |x| x % 3 != 0, c_of: |x| x * 7 + 3 };
    let s = star(&eng, 11, 2000, false, rows);
    let summed = s.summed(&eng);
    // Both operands hold a node per key at the root of `a`: the grid the walk
    // would have filled under the level over `a` and `b`.
    let a = subtree(&s.vtree, &s.blocks[0]);
    let child_grid_bytes = (summed.level(a).slot_count() * s.k.level(a).slot_count() * std::mem::size_of::<u32>()) as u64;

    let largest = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&largest);
    let hooks = MemoryHooks::new(
        move |bytes| { seen.fetch_max(bytes, Ordering::Relaxed); },
        || 0, || None, || {},
    );
    let mut out = {
        let _installed = eng.limits().scope(LimitConfig::none().with_memory_hooks(hooks));
        s.second(&eng, summed)
    };
    let largest = largest.load(Ordering::Relaxed);
    assert!(
        largest < child_grid_bytes,
        "a {largest}-byte reservation: the pass-through level densified a {child_grid_bytes}-byte child grid",
    );
    assert_eq!(out.model_count().unwrap(), s.models);
    let plain = eng.and(eng.and(s.f.clone(), s.e.clone()).unwrap(), s.k.clone()).unwrap();
    assert_eq!(plain.model_count().unwrap(), s.models, "the plain conjunction counts the same models");
    out.minimize().unwrap();
    assert_marginal_canonical(&out);
}

/// Exact weights in eighths for the variables `1..=n`, no two literals alike.
fn weights(n: u32) -> WeightStore {
    let eighths = |k: u32| BigRational::new((1 + k % 7).into(), 8.into());
    let literals: Vec<LiteralWeights<BigRational>> = (0..n)
        .map(|i| LiteralWeights { negative: eighths(i), positive: eighths(3 * i + 2) })
        .collect();
    WeightStore::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// Forced sparse routing against the grid walk, with the marginal side on
/// either side of the level and carried by either operand, counted and
/// weighted.
#[test]
fn the_pass_through_scatter_builds_what_the_grid_walk_builds() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let store = weights(18);
    // A dense `e` leaves the running conjunction wider than `k`, which keeps
    // it on the left and makes it the carrier; a sparse one narrows it, so
    // the conjunction swaps the operands and `k` holds the left side. One `c`
    // for every key leaves `k` a single node at the level, where the scatter
    // writes the one product's pairs straight into the level.
    let cases = [
        (Rows { in_e: |x| x % 7 != 0, in_k: |x| x % 3 != 0, c_of: |x| x * 7 + 3 }, true),
        (Rows { in_e: |x| x % 3 == 0, in_k: |x| x % 5 != 0, c_of: |x| x * 7 + 3 }, false),
        (Rows { in_e: |x| x % 7 != 0, in_k: |x| x % 3 != 0, c_of: |_| 9 }, true),
    ];
    for b_first in [false, true] {
        for (rows, summed_wider) in cases {
            let what = format!("b_first={b_first} summed_wider={summed_wider}");
            let s = star(&eng, 6, 60, b_first, rows);
            let summed = s.summed(&eng);
            assert_eq!(summed.max_width() > s.k.max_width(), summed_wider, "{what}");
            let build = |thresholds| {
                let _forced = ForcedThresholds::install(thresholds);
                let mut out = s.second(&eng, summed.clone());
                out.minimize().unwrap();
                assert_marginal_canonical(&out);
                assert_eq!(out.model_count().unwrap(), s.models, "{what}");
                out
            };
            let (scattered, walked) = (build(sparse), build(dense));
            assert_same_shape(&scattered, &walked, &what);

            // The weighted value of the plain conjunction, against both routes.
            let w = s.weighted(&store);
            let plain = eng.and(eng.and(w.f.clone(), w.e.clone()).unwrap(), w.k.clone()).unwrap();
            let want = eng.weighted_value(&plain).unwrap().unwrap().into_rational();
            let w_summed = w.summed(&eng);
            for thresholds in [sparse, dense] {
                let _forced = ForcedThresholds::install(thresholds);
                let mut out = w.second(&eng, w_summed.clone());
                out.minimize().unwrap();
                assert_marginal_canonical(&out);
                assert_eq!(eng.weighted_value(&out).unwrap().unwrap().into_rational(), want, "{what}");
            }
        }
    }
}
