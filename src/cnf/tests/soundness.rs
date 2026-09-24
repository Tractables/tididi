//! The encoding against exhaustive evaluation, on seeded diagrams with and
//! without summed-out levels.

use num_bigint::BigUint;

use super::*;
use crate::diagram::LeafLabel;

/// Check every node that has a literal against a direct evaluation of the
/// diagram. Under the activation literal and any assignment of the free
/// variables, the clauses hold, and they force each node's variable to
/// whether the node's value is positive there. The variable can be true
/// exactly when the node has a model.
pub(super) fn assert_sound(f: &Tdd, encoding: &CnfEncoding, store: &ClauseStore) {
    let activation = encoding.activation();
    let free = free_vars(f);
    let vars: Vec<usize> = (0..free.len()).filter(|&v| free[v]).collect();
    let nodes: Vec<(TddNodeId, i32)> = internal_nodes(f).into_iter().filter_map(|id| Some((id, encoding.literal(id)?))).collect();
    let mut alive = vec![false; nodes.len()];
    for bits in 0u32..1 << vars.len() {
        let mut assignment = vec![false; free.len()];
        let mut assumptions = vec![activation];
        for (k, &v) in vars.iter().enumerate() {
            assignment[v] = bits >> k & 1 == 1;
            let literal = v as i32 + 1;
            assumptions.push(if assignment[v] { literal } else { -literal });
        }
        assert!(store.satisfiable(&assumptions), "the clauses exclude {assumptions:?}");
        for (k, &(id, y)) in nodes.iter().enumerate() {
            let holds = node_value(f, id, &|_| false, &assignment) > BigUint::ZERO;
            alive[k] |= holds;
            assumptions.push(if holds { -y } else { y });
            assert!(!store.satisfiable(&assumptions), "{id:?} is {holds} under {assumptions:?}, but its variable can differ");
            assumptions.pop();
        }
    }
    for (k, &(id, y)) in nodes.iter().enumerate() {
        assert_eq!(store.satisfiable(&[activation, y]), alive[k], "{id:?}: whether its variable can be true");
    }
}

/// Encode `f` in full and check it: the reachable structural internal nodes
/// and only those have literals, each agrees with evaluation, the output's
/// literal can be true exactly when the diagram has a model, and retiring
/// the activation literal frees every variable.
fn check(f: &Tdd) {
    let (encoding, sink) = encode(f, None);
    let reachable = f.reachable_nodes();
    assert_eq!(encoding.reachable_nodes(), reachable);
    let mut expected = 0;
    for id in internal_nodes(f) {
        let encoded = reachable[id.vtree.idx()][id.local.idx()] && f.levels[id.vtree.idx()].nodes()[id.local.idx()].is_internal();
        assert_eq!(encoding.literal(id).is_some(), encoded, "{id:?}");
        expected += u64::from(encoded);
    }
    assert_eq!((encoding.encoded_nodes(), encoding.skipped(), encoding.stop()), (expected, 0, None));
    assert_sound(f, &encoding, &sink.store);
    let activation = encoding.activation();
    let Some(output) = encoding.literal(f.output()) else {
        assert!(f.is_zero() || f.levels[f.output().vtree.idx()].is_marginal());
        return;
    };
    let has_model = f.model_count().unwrap() > BigUint::ZERO;
    assert_eq!(sink.store.satisfiable(&[activation, output]), has_model);
    let mut store = sink.store;
    store.retire();
    assert!(store.satisfiable(&[output]) && store.satisfiable(&[-output]));
}

#[test]
fn node_variables_match_evaluation_on_seeded_diagrams() {
    for f in random_diagrams(0xc0de, 40, 3..7) { check(&f); }
}

#[test]
fn node_variables_match_evaluation_with_summed_out_levels() {
    for (f, _) in marginal_diagrams(0x5eed, 40, 4..7) { check(&f); }
}

#[test]
fn node_variables_match_evaluation_on_the_hand_built_diagrams() {
    // The boundary fixtures are the ones that read count slots.
    check(&chain().0);
    check(&golden::inline_marginal().0);
    check(&golden::marginal_boundary(LeafLabel::Neg).0);
    check(&golden::marginal_boundary(LeafLabel::Pos).0);
}
