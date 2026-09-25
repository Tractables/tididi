use std::sync::Arc;
use crate::{Tdd, Vtree};
use crate::io::vtree_to_dot;
use crate::test_helpers::assert_canonical;
use crate::vtree::VarId;

#[test]
fn plain_and_annotated_views_use_their_own_vtree() {
    for vtree in [Vtree::balanced_over(&[VarId(10), VarId(3), VarId(8)]).unwrap(),
                  Vtree::linear_from_order(&[VarId(8), VarId(10), VarId(3)]).unwrap()] {
        let vtree = Arc::new(vtree);
        let f = Tdd::clause(&vtree, [10, -3]).unwrap();
        assert_canonical(&f);
        let plain = vtree_to_dot(&vtree);
        let overlay = f.level_sizes_to_dot();
        assert!(!plain.contains("fillcolor="));
        assert!(overlay.contains("fillcolor="));
        for label in ["X₁₀", "X₃", "X₈"] {
            assert!(plain.contains(label));
            assert!(overlay.contains(label));
        }
        let edges = |text: &str| text.lines().filter(|line| line.contains(" -- "))
            .map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(edges(&plain), edges(&overlay));
        assert_eq!(edges(&plain).len(), 4);
        assert_eq!(overlay.matches("w=").count(), 2);
    }
}

#[test]
fn vtree_overlays_handle_constants_leaves_and_marginal_levels() {
    for vtree in [Vtree::leaf(VarId(1)), Vtree::balanced(3)] {
        let vtree = Arc::new(vtree);
        for mut f in [Tdd::zero(&vtree), Tdd::one(&vtree)] {
            assert_canonical(&f);
            assert!(f.level_sizes_to_dot().starts_with("graph vtree"));
            f.marginalize_levels(&[vtree.root()]).unwrap();
            assert_canonical(&f);
            let dot = f.level_sizes_to_dot();
            assert!(dot.starts_with("graph vtree"));
            assert!(!dot.contains("NaN"));
        }
    }
}
