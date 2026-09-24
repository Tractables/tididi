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
            conjoin_clause_on(&engine, input.clone(), &literals).unwrap()
        };
        requests.push(allocations.lock().unwrap().clone());
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), 192u32.into());
    }
    assert!(!requests[0].is_empty());
    assert_eq!(requests[0], requests[1]);
}

#[test]
fn repeated_and_conflicting_literals_read_as_the_normalized_clause_in_both_modes() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let f = Tdd::clause(&vtree, [1, 3]).unwrap();
    assert_eq!(f.model_count().unwrap(), 12u32.into());
    let count = |mut g: Tdd| {
        g.minimize().unwrap();
        assert_canonical(&g);
        g.model_count().unwrap()
    };

    // `f ∧ (x2 ∨ ¬x4)`, with the repeats and without.
    let repeated = count(eng.and_clause(f.clone(), &[2, 2, -4, 2][..]).unwrap());
    assert_eq!(repeated, 9u32.into());
    assert_eq!(repeated, count(eng.and_clause(f.clone(), &[2, -4][..]).unwrap()));
    // A variable in both polarities: the clause is true and `f` is the answer.
    assert_eq!(count(eng.and_clause(f.clone(), &[2, -2, 4][..]).unwrap()), 12u32.into());

    // `f ∨ (¬x1 ∧ x2 ∧ ¬x3 ∧ x4)` on the chain route once the repeat is gone,
    // and `f ∨ (x2 ∧ ¬x4)` on the complement route.
    let full = count(eng.or_cube(f.clone(), &[-1, 2, -3, 4, 2][..]).unwrap());
    assert_eq!(full, 13u32.into());
    assert_eq!(full, count(eng.or_cube(f.clone(), &[-1, 2, -3, 4][..]).unwrap()));
    let partial = count(eng.or_cube(f.clone(), &[2, 2, -4][..]).unwrap());
    assert_eq!(partial, 13u32.into());
    assert_eq!(partial, count(eng.or_cube(f.clone(), &[2, -4][..]).unwrap()));
    // A variable in both polarities: the cube is false and `f` is the answer.
    assert_eq!(count(eng.or_cube(f.clone(), &[2, -2, 4][..]).unwrap()), 12u32.into());
}
