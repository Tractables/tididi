//! Cost of quantifying a whole block of variables, by vtree shape.
//!
//! Composes a sparse relation with itself: `∃y. R(x, y) ∧ R(y, z)`, where each
//! of x, y and z is a block of nine variables. The two conjuncts are built with
//! [`Tdd::from_models`], on three vtrees over the three blocks — right-linear,
//! blocks chained under their own subtrees, and balanced. Conjoining them is
//! the same work on all three; forgetting the middle block is not.
//!
//! Run with `cargo bench --bench exists_block`.

use std::sync::Arc;
use std::time::Instant;

use tididi::apply::QuantificationStrategy;
use tididi::vtree::{VarId, Vtree};
use tididi::{and, OperationError, Tdd};

/// Variables one block holds; a block ranges over `1 << BITS` values.
const BITS: u32 = 9;
/// Values the relation relates.
const VALUES: u64 = 475;
/// Pairs the relation holds.
const PAIRS: usize = 13_289;
/// Blocks in the vtree: the composition's x, y and z.
const BLOCKS: u32 = 3;

/// The variables of block `b`, most significant bit first.
fn block(b: u32) -> Vec<VarId> {
    (0..BITS).map(|i| VarId(b * BITS + i + 1)).collect()
}

/// One model of a conjunct: `hi` in its first block, `lo` in its second.
fn model(hi: u64, lo: u64) -> u64 {
    let mut bits = 0u64;
    for i in 0..BITS {
        let shift = u64::from(BITS - 1 - i);
        bits |= ((hi >> shift) & 1) << i;
        bits |= ((lo >> shift) & 1) << (u64::from(BITS) + u64::from(i));
    }
    bits
}

/// A sparse relation with a heavy-tailed degree distribution, deduplicated and
/// sorted. Reproducible: the generator is a fixed-seed linear congruence.
fn relation() -> Vec<u64> {
    // Rank `r` is drawn with weight proportional to `1 / (r + 1) ^ 1.2`,
    // tabulated once as a cumulative distribution over the domain.
    let weights: Vec<f64> = (0..VALUES).map(|r| 1.0 / ((r + 1) as f64).powf(1.2)).collect();
    let total: f64 = weights.iter().sum();
    let mut cumulative = Vec::with_capacity(VALUES as usize);
    let mut running = 0.0;
    for w in &weights {
        running += w / total;
        cumulative.push(running);
    }
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut draw = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        cumulative.partition_point(|&c| c < u).min(VALUES as usize - 1) as u64
    };
    let mut seen = std::collections::HashSet::with_capacity(PAIRS);
    while seen.len() < PAIRS {
        let (u, v) = (draw(), draw());
        if u != v {
            seen.insert((u, v));
        }
    }
    let mut models: Vec<u64> = seen.into_iter().map(|(u, v)| model(u, v)).collect();
    models.sort_unstable();
    models
}

/// The three vtrees over `BLOCKS * BITS` variables.
fn vtrees() -> Vec<(&'static str, Arc<Vtree>)> {
    let linear = Vtree::linear(BLOCKS * BITS);
    let balanced = Vtree::balanced(BLOCKS * BITS);
    // Each block is a balanced subtree and the blocks chain left to right, so
    // the quantified block is a subtree of its own.
    let mut chained = Vtree::balanced_over(&block(0)).expect("distinct variables");
    for b in 1..BLOCKS {
        let next = Vtree::balanced_over(&block(b)).expect("distinct variables");
        chained = Vtree::join(&chained, &next).expect("disjoint blocks");
    }
    vec![
        ("linear", Arc::new(linear)),
        ("chained", Arc::new(chained)),
        ("balanced", Arc::new(balanced)),
    ]
}

/// Conjoin the two conjuncts on `vtree` and report the product's size and time.
fn product(vtree: &Arc<Vtree>, models: &[u64]) -> Result<(Tdd, f64, usize), OperationError> {
    let first: Vec<VarId> = block(0).into_iter().chain(block(1)).collect();
    let second: Vec<VarId> = block(1).into_iter().chain(block(2)).collect();
    let f = Tdd::from_models(vtree, &first, models)?;
    let g = Tdd::from_models(vtree, &second, models)?;
    let start = Instant::now();
    let joined = and(f, g)?;
    let seconds = start.elapsed().as_secs_f64();
    let pairs = joined.pair_count();
    Ok((joined, seconds, pairs))
}

fn main() -> Result<(), OperationError> {
    let models = relation();
    println!("relation: {} pairs over {VALUES} values, {BITS} variables each", models.len());
    let head: Vec<VarId> = block(0).into_iter().chain(block(2)).collect();
    let middle = block(1);
    let mut expected: Option<num_bigint::BigUint> = None;

    for (name, vtree) in vtrees() {
        let (joined, and_seconds, and_pairs) = product(&vtree, &models)?;
        let mut line = format!("{name:9} and {and_seconds:8.3}s {and_pairs:9} pairs");

        for (label, how) in [
            ("cofactor-or", QuantificationStrategy::CofactorOr),
            ("structural", QuantificationStrategy::Structural),
        ] {
            let copy = joined.clone();
            let start = Instant::now();
            let forgotten = copy.exists_vars_with_strategy(&middle, how)?;
            let seconds = start.elapsed().as_secs_f64();
            let count = forgotten.model_count()? >> middle.len();
            line.push_str(&format!("   exists/{label} {seconds:8.3}s {:9} pairs", forgotten.pair_count()));
            match &expected {
                None => expected = Some(count),
                Some(want) => assert_eq!(&count, want, "{name}/{label}: answer changed"),
            }
        }

        let start = Instant::now();
        let count = joined.projected_model_count(&head)?;
        let seconds = start.elapsed().as_secs_f64();
        line.push_str(&format!("   count {seconds:8.3}s"));
        assert_eq!(Some(&count), expected.as_ref(), "{name}: projected count disagrees");
        println!("{line}");
    }
    println!("composed relation: {} pairs", expected.expect("one vtree ran"));
    Ok(())
}
