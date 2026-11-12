//! Restriction on a vtree with a long run of levels whose products are
//! decided only when asked.
//!
//! A level is decided on demand when neither child is listed or a leaf. On a
//! right-linear vtree where `f` is marginal on every left leaf and `care`
//! holds two nodes a level, every level of the spine is such a level, and
//! asking for the root product asks for the one below it, down the whole
//! spine. The operands are built and restricted on a thread with the default
//! stack.

use super::*;

use crate::apply::RestrictionOutcome;
use crate::vtree::VtreeIdx;

/// Leaves summed out of `f`, one level of the spine each.
const SPINE: u32 = 20_000;

/// The stack of the thread that restricts: the default of a spawned thread.
const STACK: usize = 2 << 20;

/// `f = y1 ∨ y2` with `x1 … xm` summed out, and `care = (x1 ∧ … ∧ xm) ∨
/// (¬x1 ∧ … ∧ ¬xm)`, on the right-linear vtree over `x1 … xm, y1, y2`.
fn spine_operands(m: u32) -> (Tdd, Tdd) {
    let engine = Engine::new();
    let vtree = Arc::new(Vtree::linear(m + 2));
    let xs = 1..=m as i32;
    let mut f = engine.clause(&vtree, [m as i32 + 1, m as i32 + 2]).unwrap();
    let leaves: Vec<VtreeIdx> = (1..=m).map(|x| vtree.leaf_of(VarId(x)).unwrap()).collect();
    engine.marginalize_levels(&mut f, &leaves).unwrap();
    let all = engine.cube(&vtree, xs.clone()).unwrap();
    let none = engine.cube(&vtree, xs.map(|x| -x)).unwrap();
    let care = engine.or(all, none).unwrap();
    (f, care)
}

/// Nothing of `f` dies under a care it meets everywhere, however long the
/// run of levels decided on demand.
#[test]
fn restrict_over_a_deep_run_of_levels_decided_on_demand() {
    let outcome = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(|| {
            let (f, care) = spine_operands(SPINE);
            let lazy = f.vtree.root();
            assert!(f.level(f.vtree.children(lazy).0).is_marginal());
            assert_eq!(care.level(f.vtree.children(lazy).1).nodes().len(), 2);
            let before = crate::test_helpers::restrict_paths();
            let out = Engine::new().restrict_to_care_bounded(f, &care, u64::MAX).unwrap();
            let after = crate::test_helpers::restrict_paths();
            (out, [2, 3, 4].map(|k| after[k] - before[k]))
        })
        .expect("the thread starts")
        .join()
        .expect("the restriction returns");
    let (outcome, listed) = outcome;
    assert_eq!(listed, [0, 0, 0], "levels were listed, not decided on demand");
    assert!(matches!(outcome, RestrictionOutcome::Unchanged(_)), "{outcome:?}");
}
