// scenario: docs/scenarios.md#reusable-components

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use tididi::{literal, Engine, Tdd, Vtree};
    use tididi::vtree::{Home, VarId};

    let component = Home::new(Vtree::balanced(2));
    let backup = Tdd::clause(&component, [1, 2])?;
    println!("Choices for one server: {}", backup.model_count()?);

    let engine = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let (server_a, _) = engine.transport(&backup, &home, |var| var, 0).map_err(|r| r.error)?;
    let (server_b, _) = engine.transport(&backup, &home, |var| VarId(var.0 + 2), 0).map_err(|r| r.error)?;
    println!("Server A rule over both servers: {}", server_a.model_count()?);

    let independent = server_a & server_b;
    println!("Independent backup choices: {}", independent.model_count()?);

    let remote_a = literal(&home, 2)?;
    let remote_b = literal(&home, 4)?;
    let shared_capacity = !(remote_a & remote_b);
    let system = independent & shared_capacity;
    println!("Choices with shared capacity: {}", system.model_count()?);
    println!("Reusable component still available: {}", backup.model_count()?);
    let place_b = |rule: &Tdd| engine.transport(rule, &home, |var| VarId(var.0 + 2), 0)
        .map(|(placed, _)| placed).map_err(|r| r.error);
    let local_only = Tdd::cube(&component, [1, -2])?;
    let original_b = place_b(&backup)?;
    let updated_b = place_b(&local_only)?;
    println!("Server B choices before and after: {} -> {}",
        original_b.model_count()?, updated_b.model_count()?);
    Ok(())
}
