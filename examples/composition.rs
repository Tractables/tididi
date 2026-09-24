// scenario: docs/scenarios.md#reusable-components

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use tididi::{literal, Tdd, Vtree};
    use tididi::vtree::VarId;

    let component_vtree = Arc::new(Vtree::balanced(2));
    let backup = Tdd::clause(&component_vtree, [1, 2])?;
    println!("Choices for one server: {}", backup.model_count()?);

    let vtree = Arc::new(Vtree::balanced(4));
    let (server_a, _) = backup.embed(&vtree, |var| var)?;
    let (server_b, _) = backup.embed(&vtree, |var| VarId(var.0 + 2))?;
    println!("Server A rule over both servers: {}", server_a.model_count()?);

    let independent = server_a & server_b;
    println!("Independent backup choices: {}", independent.model_count()?);

    let remote_a = literal(&vtree, 2)?;
    let remote_b = literal(&vtree, 4)?;
    let shared_capacity = !(remote_a & remote_b);
    let system = independent & shared_capacity;
    println!("Choices with shared capacity: {}", system.model_count()?);
    println!("Reusable component still available: {}", backup.model_count()?);
    use tididi::restructure::EmbeddingPlan;
    let place_b = EmbeddingPlan::new(&component_vtree, &vtree, |var| VarId(var.0 + 2))?;
    let local_only = Tdd::cube(&component_vtree, [1, -2])?;
    let original_b = place_b.apply(&backup)?;
    let updated_b = place_b.apply(&local_only)?;
    println!("Server B choices before and after: {} -> {}",
        original_b.model_count()?, updated_b.model_count()?);
    Ok(())
}
