//! Levels priced before they are built: a level whose output cannot fit is
//! refused there and named, and the count a level is priced by is the
//! pairs the level holds (a debug build checks every level it can count).

use std::sync::Arc;

use super::*;
use crate::limits::{LevelRefusal, LimitConfig, SparseRoute};
use crate::test_helpers::{assert_canonical, rand_conj_over, vtree_shapes, Lcg};
use crate::vtree::{VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

/// The sparse route on every level it can take.
const SPARSE: SparseRoute = SparseRoute { sparsity: 1, min_grid: 0 };

/// `bits` variables from `first`.
fn block(first: u32, bits: u32) -> Vec<VarId> {
    (first..first + bits).map(VarId).collect()
}

/// A level of every product: under the node joining `x` and `y`, `f`'s
/// nodes are `x = k` and `g`'s `y = k`, one for each code `k` of `z`, so
/// every product of a node of `f` and a node of `g` is satisfiable. With
/// `halved`, `f` also asks for an even `y` and `g` for an even `x`: a
/// product is live only on even codes, the children are then not complete,
/// and the level is counted through its children's live products. Returns
/// the vtree, the node joining `x` and `y`, the two operands, and the pairs
/// that node's level holds.
fn every_product(bits: u32, halved: bool) -> (Arc<Vtree>, VtreeIdx, Tdd, Tdd, u128) {
    let (x, y, z) = (block(1, bits), block(1 + bits, bits), block(1 + 2 * bits, bits));
    let xy = Vtree::join(&Vtree::balanced_over(&x).unwrap(), &Vtree::balanced_over(&y).unwrap()).unwrap();
    let vtree = Arc::new(Vtree::join(&xy, &Vtree::balanced_over(&z).unwrap()).unwrap());
    let joined = vtree.children(vtree.root()).0;
    let codes = 1u64 << bits;
    // Rows over `z`, the block `z` selects, and the other block.
    let equal = |selected: &[VarId], other: &[VarId]| {
        let vars: Vec<VarId> = z.iter().chain(selected).chain(other).copied().collect();
        let others: Vec<u64> = (0..codes).filter(|o| !halved || o % 2 == 0).collect();
        let rows: Vec<u64> = (0..codes)
            .flat_map(|k| others.iter().map(move |&o| k | k << bits | o << (2 * bits)))
            .collect();
        let mut tdd = Tdd::from_models(&vtree, &vars, &rows).unwrap();
        tdd.minimize().unwrap();
        tdd
    };
    let (f, g) = (equal(&x, &y), equal(&y, &x));
    let live = if halved { codes / 2 } else { codes } as u128;
    (vtree, joined, f, g, live * live)
}

/// A conjunction of `f` and `g` on a fresh engine under `budget`, on the
/// sparse route where it can take it when `sparse` holds: the result and
/// the level it refused, if it refused one.
fn conjoin(f: &Tdd, g: &Tdd, budget: Option<u64>, sparse: bool) -> (Result<Tdd, OperationError>, Option<LevelRefusal>) {
    let eng = Engine::new();
    let mut config = LimitConfig::none().with_memory_budget_bytes(budget);
    if sparse {
        config = config.with_sparse_route(SPARSE);
    }
    let _scope = eng.limits().scope(config);
    let out = eng.and(f.clone(), g.clone());
    (out, eng.limits().meters().refused_level)
}

/// A level of every product is refused before it is built, and named, on
/// either route and whether its children are complete or listed; without
/// the budget the same conjunction is built. The price is at least the
/// level's pairs, and the grid besides where the level takes the dense
/// route.
#[test]
fn a_level_of_every_product_is_refused_before_it_is_built() {
    let mut sparse_priced = false;
    for halved in [false, true] {
        for sparse in [false, true] {
            // 128 live codes either way: a level of 16 384 pairs, 128 KiB.
            let (_, joined, f, g, pairs) = every_product(if halved { 8 } else { 7 }, halved);
            let case = format!("halved {halved}, sparse {sparse}");

            let (out, refused) = conjoin(&f, &g, None, sparse);
            let mut out = out.unwrap_or_else(|e| panic!("{case}: {e:?}"));
            out.minimize().unwrap();
            assert_canonical(&out);
            assert_eq!(out.model_count().unwrap(), 128u32.into(), "{case}");
            assert_eq!(refused, None, "{case}: no budget prices nothing");

            let need = output_bytes(pairs);
            let (out, refused) = conjoin(&f, &g, Some(need / 2), sparse);
            assert_eq!(out.err(), Some(OperationError::OverBudget), "{case}");
            let refused = refused.unwrap_or_else(|| panic!("{case}: the level was not priced"));
            assert_eq!(refused.level, joined, "{case}");
            assert!(refused.needed_bytes >= need && refused.needed_bytes > refused.headroom_bytes, "{case}: {refused:?}");
            sparse_priced |= sparse && refused.needed_bytes == need;
        }
    }
    assert!(sparse_priced, "some level took the sparse route and was priced by its pairs alone");
}

/// A budget the level fits under prices it and builds it: the price is a
/// floor on what the level claims, never a refusal of a level that fits.
#[test]
fn a_level_that_fits_is_built_under_its_price() {
    let (_, _, f, g, pairs) = every_product(6, true);
    for sparse in [false, true] {
        let (out, refused) = conjoin(&f, &g, Some(64 * output_bytes(pairs)), sparse);
        let mut out = out.unwrap();
        out.minimize().unwrap();
        assert_canonical(&out);
        assert_eq!(out.model_count().unwrap(), 32u32.into());
        assert_eq!(refused, None);
    }
}

/// Random conjunctions on every vtree shape, on both routes, unbounded and
/// under a budget: a debug build counts every level it can read and checks
/// the count against the level built, and every result is the one the
/// default route builds unbounded.
#[test]
fn the_count_a_level_is_priced_by_is_the_pairs_it_holds() {
    let mut rng = Lcg::new(0x9e37);
    for (shape, vtree) in vtree_shapes(9) {
        let vars: Vec<u32> = (1..=vtree.num_vars()).collect();
        for _ in 0..6 {
            let f = rand_conj_over(&vtree, &vars, 8, 3, false, &mut rng);
            let g = rand_conj_over(&vtree, &vars, 8, 3, false, &mut rng);
            let (expected, _) = conjoin(&f, &g, None, false);
            let expected = expected.unwrap();
            for sparse in [false, true] {
                for budget in [None, Some(1 << 40)] {
                    let (out, refused) = conjoin(&f, &g, budget, sparse);
                    let mut out = out.unwrap_or_else(|e| panic!("{shape}: {e:?}"));
                    out.minimize().unwrap();
                    assert_canonical(&out);
                    assert_eq!(out.model_count().unwrap(), expected.model_count().unwrap(), "{shape}");
                    assert_eq!(refused, None, "{shape}");
                }
            }
        }
    }
}
