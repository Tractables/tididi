//! What the `testing` feature guarantees a consumer's test suite.

use std::sync::Arc;

use tididi::diagram::{ChildPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
use tididi::test_helpers::assert_canonical;
use tididi::{Engine, Tdd, Vtree};

/// The invariant checkers walk the diagram in a release build too.
#[test]
fn invariant_checkers_are_active_in_release_integration_tests() {
    let vtree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let root = vtree.root();
    let output = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: output }).unwrap();
    // Storage-valid but deliberately noncanonical: two nodes have identical content.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| assert_canonical(&f))).is_err());
    let valid = Tdd::cube(&vtree, [1, -2]).unwrap();
    assert_canonical(&valid);
}
