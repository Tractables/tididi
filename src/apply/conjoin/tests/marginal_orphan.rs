use super::*;
use crate::Engine;
use crate::vtree::Vtree;
use std::sync::Arc;

/// A conjunction over a diagram carrying a zero-width marginal orphan level.
///
/// Marginalizing a whole subtree leaves the levels *under* its root with no
/// node any pair can reference; the in-apply cascade hands those on as
/// zero-width marginals. The orphan's own contribution is already inside its
/// marginal ancestor's count, so emptying it must not move the model count —
/// and the level must never reach the dense route, which would read pairs out
/// of an empty node list.
#[test]
fn a_zero_width_marginal_orphan_conjoins_to_the_same_count() {
    let vtree = Arc::new(Vtree::balanced(8));
    let root = vtree.root();
    let (_, right) = vtree.children(root);
    let (right_left, right_right) = vtree.children(right);

    // Both operands constrain only the left half, so marginalizing the right
    // subtree gives them the same mass there.
    let fixture = |clauses: &[Vec<i32>]| {
        let mut t = crate::test_helpers::compile_clauses(&vtree, clauses);
        crate::test_helpers::marginalize_subtree(&mut t, right);
        t
    };
    let f = || fixture(&[vec![1, 2], vec![-2, 3]]);
    let g = || fixture(&[vec![2, 4], vec![-1, -3]]);

    let eng = Engine::new();
    let want = (apply_and(f(), g())).model_count().unwrap();

    let (mut f, mut g) = (f(), g());
    for t in [&mut f, &mut g] {
        for orphan in [right_left, right_right] {
            t.levels[orphan.idx()].set_counts_state(vec![], None);
        }
    }
    let out = eng.and(f, g)
        .expect("a zero-width orphan must not fail the conjunction");

    assert_eq!(
        out.model_count().unwrap(), want,
        "emptying an orphan under a marginal ancestor must not move the count",
    );
}
