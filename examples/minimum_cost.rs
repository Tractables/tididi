//! Find the minimum cost of a valid backup configuration.
//! Run with `cargo run --example minimum_cost`.

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;
    use tididi::{literal, Vtree};

    let vtree = Arc::new(Vtree::balanced(4));
    let local = literal(&vtree, 1)?;
    let remote = literal(&vtree, 2)?;
    let encrypted = literal(&vtree, 3)?;
    let configurations = (local | remote.clone()) & (!remote.clone() | encrypted.clone());

    use tididi::diagram::{EvalAlgebra, LeafLabel};
    use tididi::vtree::VarId;

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

    let costs = Costs([5, 2, 1, 0]);
    let minimum = configurations.evaluate(&costs)?.expect("the rules have a solution");
    println!("Minimum configuration cost: {minimum}");

    let local_discount = Costs([1, 2, 1, 0]);
    println!("Minimum with discount: {:?}", configurations.evaluate(&local_discount)?);

    let with_remote = configurations.clone() & remote;
    println!("Minimum with remote backups: {:?}", with_remote.evaluate(&local_discount)?);
    let remote_without_encryption = with_remote & !encrypted;
    println!("Minimum without encryption: {:?}", remote_without_encryption.evaluate(&costs)?);

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
