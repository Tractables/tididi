//! Find the minimum cost of a valid backup configuration.
//! Run with `cargo run --example minimum_cost`.

use std::sync::Arc;
use tididi::diagram::{EvalAlgebra, LeafLabel};
use tididi::vtree::VarId;
use tididi::{and, literal, Tdd, Vtree};

/// Enabling costs for local backups, remote backups, encryption and notifications.
struct Costs([u32; 4]);

impl EvalAlgebra for Costs {
    type Value = Option<u64>;

    fn zero(&self) -> Self::Value { None }

    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value {
        match label {
            LeafLabel::Pos => Some(u64::from(self.0[var.idx()])),
            LeafLabel::Neg | LeafLabel::One => Some(0),
            LeafLabel::Zero => None,
        }
    }

    fn add_assign(&self, best: &mut Self::Value, candidate: &Self::Value) {
        *best = match (*best, *candidate) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }

    fn mul(&self, left: &Self::Value, right: &Self::Value) -> Self::Value {
        // A valid circuit combines disjoint variables; four u32 costs fit in u64.
        Some((*left)? + (*right)?)
    }
}

fn main() -> Result<(), tididi::OperationError> {
    let vtree = Arc::new(Vtree::balanced(4));
    let configurations = and(
        Tdd::clause(&vtree, [1, 2])?,
        Tdd::clause(&vtree, [-2, 3])?,
    )?;
    let costs = Costs([5, 2, 1, 0]);
    assert_eq!(configurations.evaluate(&costs)?, Some(3));
    println!("Minimum configuration cost: 3");

    let local_discount = Costs([1, 2, 1, 0]);
    assert_eq!(configurations.evaluate(&local_discount)?, Some(1));

    let with_remote = and(configurations.clone(), literal(&vtree, 2)?)?;
    assert_eq!(with_remote.evaluate(&local_discount)?, Some(3));
    let conflicting = and(with_remote, literal(&vtree, -3)?)?;
    assert_eq!(conflicting.evaluate(&costs)?, None);

    // Independently enumerate this small model to verify both cost scenarios.
    for prices in [&costs, &local_discount] {
        let expected = (0..16u32).filter(|&bits| {
            let local = bits & 1 != 0;
            let remote = bits & 2 != 0;
            let encrypted = bits & 4 != 0;
            (local || remote) && (!remote || encrypted)
        }).map(|bits| {
            prices.0.iter().enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, &cost)| u64::from(cost)).sum::<u64>()
        }).min();
        assert_eq!(configurations.evaluate(prices)?, expected);
    }
    Ok(())
}
