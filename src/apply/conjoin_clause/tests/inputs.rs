use super::*;
use std::sync::{Arc, Mutex};
use crate::limits::{LimitConfig, MemoryHooks};
use crate::test_helpers::assert_canonical;

#[test]
fn typed_clause_adapter_preserves_kernel_allocation_requests() {
    let tree = Arc::new(Vtree::balanced(8));
    let input = Tdd::one(&tree);
    assert_canonical(&input);
    let literals = [Literal::try_from(1).unwrap(), Literal::try_from(-7).unwrap()];
    let mut requests = Vec::new();
    for use_adapter in [false, true] {
        let engine = Engine::new();
        let allocations = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&allocations);
        let hooks = MemoryHooks::new(move |bytes| observed.lock().unwrap().push(bytes), || 0, || None, || {});
        let _limits = engine.limits().scope(LimitConfig::none().with_memory_hooks(hooks));
        let mut result = if use_adapter {
            engine.and_clause(input.clone(), literals.as_slice()).unwrap()
        } else {
            conjoin_clause_owned(&engine, input.clone(), &literals).unwrap()
        };
        requests.push(allocations.lock().unwrap().clone());
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), 192u32.into());
    }
    assert!(!requests[0].is_empty());
    assert_eq!(requests[0], requests[1]);
}
