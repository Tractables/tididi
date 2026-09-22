// scenario: docs/scenarios.md#execution

//! Bound operations, retain inputs for a retry, and release idle scratch.
//! Run with `cargo run --example execution_limits`.

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;

    use tididi::limits::LimitConfig;
    use tididi::{literal, OperationError, Tdd, Vtree};

    let vtree = Arc::new(Vtree::balanced(4));
    let context = Arc::clone(vtree.context());
    let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
    let attempt = context.with_limits(limit, |operations| {
        operations.clause(&vtree, [1, 2])
    });
    match attempt {
        Ok(diagram) => println!("Destination choices: {}", diagram.model_count()?),
        Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
        Err(error) => return Err(error),
    }

    // Limits end with the batch, so an ordinary call can build the same rule.
    let destination = Tdd::clause(&vtree, [1, 2])?;
    println!("Destination choices: {}", destination.model_count()?);

    // Supply copies so the originals survive a refused operation.
    let encrypted = literal(&vtree, 3)?;
    let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
    let attempt = context.with_limits(limit, |operations| {
        operations.and(destination.clone(), encrypted.clone())
    });
    let secured = match attempt {
        Ok(diagram) => diagram,
        Err(OperationError::OverBudget) => {
            println!("Not enough budget; retrying with the original operands");
            tididi::and(destination, encrypted)?
        }
        Err(error) => return Err(error),
    };
    println!("Secured choices: {}", secured.model_count()?);

    // A borrowed counter keeps its cached counts after the bounded query.
    let mut counter = secured.counter()?;
    let query_limit = LimitConfig::none().with_memory_budget_bytes(Some(1_000_000));
    let bounded_count = context.with_limits(query_limit, |operations| {
        counter.bind(operations).model_count()
    })?;
    println!("Count with a budget: {bounded_count}");

    context.clear_scratch();
    println!("Circuit remains usable: {}", secured.is_sat()?);
    Ok(())
}
