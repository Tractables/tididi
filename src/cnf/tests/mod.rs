//! Tests of the CNF encoding: its exact call stream, its agreement with
//! exhaustive search, its stops, and what it refuses.

use std::ops::ControlFlow;
use std::sync::Arc;

use super::*;
use crate::diagram::NodeIdx;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::*;
use crate::vtree::Vtree;

mod golden;
mod soundness;
mod stops;

/// One call an encoding made on its sink.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Fresh(i32),
    Leaf(u32),
    Clause(Vec<i32>),
    Poll(EncodePoint),
}

/// A [`ClauseStore`] that records every call and answers `Break` to poll
/// number `stop_at`, counting from zero.
struct Recorder {
    store: ClauseStore,
    calls: Vec<Call>,
    polls: usize,
    stop_at: Option<usize>,
}

impl ClauseSink for Recorder {
    fn fresh_var(&mut self) -> i32 {
        let var = self.store.fresh_var();
        self.calls.push(Call::Fresh(var));
        var
    }

    fn leaf_literal(&mut self, var: VarId) -> i32 {
        self.calls.push(Call::Leaf(var.0));
        self.store.leaf_literal(var)
    }

    fn clause(&mut self, body: &[i32]) {
        self.calls.push(Call::Clause(body.to_vec()));
        self.store.clause(body);
    }

    fn poll(&mut self, at: EncodePoint) -> ControlFlow<()> {
        self.calls.push(Call::Poll(at));
        self.polls += 1;
        if self.stop_at == Some(self.polls - 1) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }
}

impl Recorder {
    /// A recorder over a store that holds `f`'s variables and the activation
    /// literal after them, which gates every clause.
    fn over(f: &Tdd, stop_at: Option<usize>) -> (Self, i32) {
        let mut store = ClauseStore::new(f.vtree().num_vars());
        let activation = store.activate();
        (Recorder { store, calls: Vec::new(), polls: 0, stop_at }, activation)
    }

    /// The polls among the calls.
    fn polls(&self) -> Vec<EncodePoint> {
        self.calls.iter().filter_map(|call| match call { Call::Poll(at) => Some(*at), _ => None }).collect()
    }
}

/// Encode `f` with a recorder that breaks at poll `stop_at`.
fn encode(f: &Tdd, stop_at: Option<usize>) -> (CnfEncoding, Recorder) {
    let (mut sink, activation) = Recorder::over(f, stop_at);
    let encoding = f.encode_cnf(CnfScheme::Equivalence, activation, &mut sink).unwrap();
    assert_eq!(encoding.activation(), activation);
    (encoding, sink)
}

/// Every stored node of the internal levels of `f`, bottom-up and by index.
fn internal_nodes(f: &Tdd) -> Vec<TddNodeId> {
    f.vtree().internal_bottomup()
        .flat_map(|(t, _, _)| (0..f.levels[t.idx()].nodes().len()).map(move |i| TddNodeId { vtree: t, local: NodeIdx(i as u32) }))
        .collect()
}

/// A sink that fails the test on any call.
struct Untouched;

impl ClauseSink for Untouched {
    fn fresh_var(&mut self) -> i32 { panic!("a refused encoding allocated a variable") }
    fn leaf_literal(&mut self, _: VarId) -> i32 { panic!("a refused encoding asked for a leaf literal") }
    fn clause(&mut self, _: &[i32]) { panic!("a refused encoding gave a clause") }
    fn poll(&mut self, _: EncodePoint) -> ControlFlow<()> { panic!("a refused encoding polled") }
}

#[test]
fn a_weight_marginal_level_is_refused_before_the_sink_is_called() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::clause(&tree, [1, 3, 5, 7]).unwrap();
    let weights = vec![LiteralWeights { negative: rat(2, 1), positive: rat(3, 1) }; 8];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    let forgotten = tree.children(tree.root()).0;
    f.marginalize_levels(&[forgotten]).unwrap();
    assert_canonical(&f);
    let first = f.levels.iter().position(|level| level.is_weight_marginal()).expect("a weighted level");
    assert!(f.levels[forgotten.idx()].is_weight_marginal());
    let refused = f.encode_cnf(CnfScheme::Equivalence, 9, &mut Untouched).err();
    assert_eq!(refused, Some(OperationError::MarginalLevel(VtreeIdx(first as u32))));
}

#[test]
fn an_activation_the_table_cannot_hold_is_refused() {
    let (f, _) = chain();
    for activation in [0, i32::MIN] {
        assert_eq!(f.encode_cnf(CnfScheme::Equivalence, activation, &mut Untouched).err(), Some(OperationError::InvalidLiteral(activation)));
    }
}

#[test]
fn a_sink_literal_the_table_cannot_hold_is_refused() {
    /// Numbers variables after the diagram's, except that fresh variable
    /// number `bad_fresh` or leaf literal number `bad_leaf`, counting from
    /// zero, is `bad`.
    struct Bad { last: i32, fresh: usize, leaves: usize, bad_fresh: Option<usize>, bad_leaf: Option<usize>, bad: i32 }
    impl ClauseSink for Bad {
        fn fresh_var(&mut self) -> i32 {
            self.last += 1;
            self.fresh += 1;
            if self.bad_fresh == Some(self.fresh - 1) { self.bad } else { self.last }
        }
        fn leaf_literal(&mut self, var: VarId) -> i32 {
            self.leaves += 1;
            if self.bad_leaf == Some(self.leaves - 1) { self.bad } else { var.0 as i32 }
        }
        fn clause(&mut self, _: &[i32]) {}
    }
    let (f, _) = chain();
    for bad in [0, i32::MIN] {
        // The true variable, a node's variable, a pair's variable, and two
        // leaf literals.
        for (bad_fresh, bad_leaf) in [(Some(0), None), (Some(3), None), (Some(9), None), (None, Some(0)), (None, Some(2))] {
            let mut sink = Bad { last: 5, fresh: 0, leaves: 0, bad_fresh, bad_leaf, bad };
            let result = f.encode_cnf(CnfScheme::Equivalence, 5, &mut sink);
            assert_eq!(result.err(), Some(OperationError::InvalidLiteral(bad)), "{bad_fresh:?} {bad_leaf:?}");
        }
    }
}

#[test]
fn cancellation_stops_between_nodes_with_the_calls_so_far_given() {
    let (f, _) = chain();
    let (_, full) = encode(&f, None);
    let mut stopped_inside = false;
    for units in 0.. {
        let eng = Engine::new();
        let _guard = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(units)), ..StopRules::default()
        }));
        eng.limits().pin_reduce_poll_stride(Some(1));
        let (mut sink, activation) = Recorder::over(&f, None);
        match eng.encode_cnf(&f, CnfScheme::Equivalence, activation, &mut sink) {
            Err(error) => {
                assert_eq!(error, OperationError::Stopped);
                assert_eq!(sink.calls, full.calls[..sink.calls.len()], "a stop after {units} units");
                stopped_inside |= sink.polls > 0;
            }
            Ok(_) => {
                assert_eq!(sink.calls, full.calls);
                break;
            }
        }
    }
    assert!(stopped_inside, "some stop must fall after the first poll");
}

#[test]
fn a_refused_allocation_leaves_the_calls_so_far_given() {
    let fixtures: Vec<Tdd> = [chain().0].into_iter().chain(marginal_diagrams(0x51de, 6, 4..7).into_iter().map(|(f, _)| f)).collect();
    for f in &fixtures {
        let (expected, full) = encode(f, None);
        let mut completed = false;
        for nth in 0..4096 {
            let eng = Engine::new();
            eng.limits().refuse_nth_reserve(nth);
            let (mut sink, activation) = Recorder::over(f, None);
            let result = eng.encode_cnf(f, CnfScheme::Equivalence, activation, &mut sink);
            eng.limits().grant_every_reserve();
            match result {
                Err(error) => {
                    assert_eq!(error, OperationError::OverBudget);
                    assert_eq!(sink.calls, full.calls[..sink.calls.len()], "refusal {nth}");
                }
                Ok(encoding) => {
                    assert_eq!(sink.calls, full.calls);
                    assert_eq!(encoding.literals, expected.literals);
                    completed = true;
                    break;
                }
            }
        }
        assert!(completed, "every reservation must be covered");
    }
}

#[test]
fn a_false_diagram_gets_only_the_true_variable() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::zero(&tree);
    assert_canonical(&f);
    let (encoding, sink) = encode(&f, None);
    let t = 6;
    let mut expected = vec![Call::Fresh(t), Call::Clause(vec![t])];
    expected.extend((1..=4).map(Call::Leaf));
    expected.extend(f.vtree().internal_bottomup().enumerate().flat_map(|(complete, (level, _, _))| {
        [Call::Poll(EncodePoint::Level { level, complete }), Call::Poll(EncodePoint::Node { level, slot: 0 })]
    }).filter(|call| !matches!(call, Call::Poll(EncodePoint::Node { level, .. }) if f.levels[level.idx()].nodes().is_empty())));
    assert_eq!(sink.calls, expected);
    assert_eq!((encoding.encoded_nodes(), encoding.skipped(), encoding.stop()), (0, 0, None));
    assert_eq!(encoding.literal(f.output()), None);
    assert!(encoding.erasure_certified());
}
