//! What the `testing` feature guarantees a consumer's test suite.

use std::sync::Arc;

use tididi::diagram::{ChildPair, POS_LEAF_IDX, TddNodeId, ValueRef};
use tididi::test_helpers::assert_canonical;
use tididi::{Engine, Tdd, Vtree};

/// The invariant checkers walk the diagram in a release build too.
#[test]
fn invariant_checkers_are_active_in_release_integration_tests() {
    let vtree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = vtree.root();
    let x2 = vtree.children(root).1;
    let mut free = Tdd::cube(&vtree, std::iter::empty::<i32>()).unwrap();
    eng.marginalize_levels(&mut free, &[x2]).unwrap();
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    builder.replace_level(&eng, x2, free.level_view(x2)).unwrap();
    let pair = ChildPair::new(POS_LEAF_IDX, ValueRef::Inline(2).side().unwrap());
    let output = builder.push(&eng, root, &[pair]).unwrap();
    builder.push(&eng, root, &[pair]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: output }).unwrap();
    // Storage-valid but deliberately noncanonical: two nodes have identical
    // content, which finish accepts over a summed-out child.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| assert_canonical(&f))).is_err());
    let valid = Tdd::cube(&vtree, [1, -2]).unwrap();
    assert_canonical(&valid);
}
