//! Replay the same operation traces as the Python and C consumers.

use std::sync::Arc;
use tididi::{and, or, xor, ite, literal, Tdd, Vtree};
use tididi::vtree::VarId;

#[test]
fn shared_behavioral_traces() {
    let mut vtree = Arc::new(Vtree::balanced(1));
    let mut circuits = Vec::<Tdd>::new();
    let mut n = 1;
    for (line_number, line) in include_str!("fixtures/conformance.txt").lines().enumerate() {
        if line.starts_with('#') { continue; }
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 5);
        let [a, b, c] = [1, 2, 3].map(|i| fields[i].parse::<i32>().unwrap());
        if fields[0] == "case" {
            n = a as u32;
            vtree = Arc::new(if b == 0 { Vtree::balanced(n) } else { Vtree::linear(n) });
            circuits.clear();
            continue;
        }
        let copy = |id: i32| circuits[id as usize].clone();
        let mut f = match fields[0] {
            "literal" => literal(&vtree, a).unwrap(),
            "one" => Tdd::one(&vtree),
            "zero" => Tdd::zero(&vtree),
            "and" => and(copy(a), copy(b)).unwrap(),
            "or" => or(copy(a), copy(b)).unwrap(),
            "xor" => xor(copy(a), copy(b)).unwrap(),
            "ite" => ite(copy(a), copy(b), copy(c)).unwrap(),
            "negate" => copy(a).negate().unwrap(),
            "condition" => copy(a).condition([b]).unwrap(),
            "exists" => copy(a).exists_var(VarId(b as u32)).unwrap(),
            "swap" => copy(a).rename_vars(&[(VarId(b as u32), VarId(c as u32)), (VarId(c as u32), VarId(b as u32))]).unwrap(),
            "rename" => copy(a).rename_vars(&[(VarId(b as u32), VarId(c as u32))]).unwrap(),
            "minimize" => { let mut f = copy(a); f.minimize().unwrap(); f },
            "roundtrip" => {
                let mut bytes = Vec::new();
                tididi::io::write_tdd(&mut bytes, &circuits[a as usize]).unwrap();
                tididi::io::read_tdd(&mut bytes.as_slice(), &vtree).unwrap()
            }
            op => panic!("unknown trace operation {op}"),
        };
        let truth = u64::from_str_radix(fields[4], 16).unwrap();
        let context = format!("line {}: {line}", line_number + 1);
        assert_eq!(f.model_count().unwrap(), truth.count_ones().into(), "{context}");
        let mut counter = f.counter().unwrap();
        for bits in 0..(1 << n) {
            let pins: Vec<i32> = (1..=n).map(|v| if bits & (1 << (v - 1)) != 0 { v as i32 } else { -(v as i32) }).collect();
            counter.observe(&pins).unwrap();
            assert_eq!(counter.model_count().unwrap(), ((truth >> bits) & 1).into(), "{context}, assignment {bits}");
        }
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), truth.count_ones().into(), "{context}");
        drop(counter);
        let expected_support: Vec<_> = (1..=n).filter(|v| (0..(1 << n)).any(|x| ((truth >> x) ^ (truth >> (x ^ (1 << (v - 1))))) & 1 != 0)).map(VarId).collect();
        assert_eq!(f.support().unwrap(), expected_support, "{context}");
        let mut implied: Vec<_> = f.implied_literals().unwrap().into_iter().map(|l| if l.sign { l.var.0 as i32 } else { -(l.var.0 as i32) }).collect();
        implied.sort_unstable();
        let expected_implied: Vec<_> = (-(n as i32)..=n as i32).filter(|&v| v != 0 && (0..(1 << n)).all(|x| (truth >> x) & 1 == 0 || (x & (1 << (v.unsigned_abs() - 1)) != 0) == (v > 0))).collect();
        assert_eq!(implied, expected_implied, "{context}");
        f.minimize().unwrap();
        tididi::test_helpers::assert_canonical(&f);
        circuits.push(f);
    }
}

/// Both cached query types replay the same ownership and refusal script.
enum Cached {
    Count(tididi::query::OwnedModelCounter),
    Weight(tididi::query::OwnedEvaluator<tididi::diagram::RationalWeights>),
}

impl Cached {
    fn observe(&mut self, values: &[i32]) -> Result<(), tididi::OperationError> {
        match self { Self::Count(q) => q.observe(values), Self::Weight(q) => q.observe(values) }
    }
    fn clear(&mut self, var: Option<VarId>) {
        match (self, var) {
            (Self::Count(q), Some(v)) => q.set_pin(v, None).unwrap(),
            (Self::Weight(q), Some(v)) => q.set_pin(v, None).unwrap(),
            (Self::Count(q), None) => q.clear_pins(),
            (Self::Weight(q), None) => q.clear_pins(),
        }
    }
    fn read(&mut self, engine: &tididi::Engine) -> Result<String, tididi::OperationError> {
        match self {
            Self::Count(q) => q.bind(engine).model_count().map(|n| n.to_string()),
            Self::Weight(q) => q.bind(engine).value().map(|n| n.to_string()),
        }
    }
    fn finish(self) -> Tdd {
        match self { Self::Count(q) => q.into_inner(), Self::Weight(q) => q.into_inner() }
    }
}

#[test]
fn shared_stateful_sessions() {
    use tididi::{Engine, OperationError};
    use tididi::limits::{LimitConfig, StopCallback, StopDecision};
    use tididi::diagram::{LiteralWeights, RationalWeights};
    let engine = Engine::new();
    let stopped = || LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Stop)));
    let mut vtree = Arc::new(Vtree::balanced(1));
    let mut circuit: Option<Tdd> = None;
    let mut saved: Option<Tdd> = None;
    let mut query: Option<Cached> = None;
    let mut weighted = false;
    let mut original_truth = 0;
    for (line_number, line) in include_str!("fixtures/conformance_sessions.txt").lines().enumerate() {
        if line.starts_with('#') { continue; }
        let fields: Vec<_> = line.split_whitespace().collect();
        let [a, b, c] = [1, 2, 3].map(|i| fields[i].parse::<i32>().unwrap());
        let expected = u64::from_str_radix(fields[4], 16).unwrap();
        let context = format!("line {}: {line}", line_number + 1);
        let pins: Vec<_> = [a, b].into_iter().filter(|v| *v != 0).collect();
        match fields[0] {
            "case" => {
                vtree = Arc::new(if b == 0 { Vtree::balanced(a as u32) } else { Vtree::linear(a as u32) });
                circuit = Some(Tdd::clause(&vtree, [1, 2]).unwrap());
                saved = None; query = None; weighted = c != 0; original_truth = expected;
                continue;
            }
            "save" => saved = circuit.clone(),
            "open" => {
                let f = circuit.take().unwrap();
                tididi::test_helpers::assert_canonical(&f);
                query = Some(if weighted {
                    let one = num_rational::BigRational::from_integer(1.into());
                    let weights = RationalWeights::from_literals(&vec![LiteralWeights { negative: one.clone(), positive: one }; vtree.num_leaves() as usize]);
                    Cached::Weight(f.into_evaluator(weights).unwrap())
                } else { Cached::Count(f.into_counter().unwrap()) });
            }
            "observe" => query.as_mut().unwrap().observe(&pins).unwrap(),
            "reject_observe" => assert!(matches!(query.as_mut().unwrap().observe(&pins), Err(OperationError::VariableNotInVtree(_))), "{context}"),
            "refuse_read" | "refuse_dirty" => {
                let q = query.as_mut().unwrap();
                if fields[0] == "refuse_dirty" { q.observe(&pins).unwrap(); }
                let _limit = engine.limits().scope(stopped());
                assert!(matches!(q.read(&engine), Err(OperationError::Stopped)), "{context}");
            }
            "clear_one" => query.as_mut().unwrap().clear(Some(VarId(a as u32))),
            "clear" => query.as_mut().unwrap().clear(None),
            "finish" => circuit = Some(query.take().unwrap().finish()),
            "refuse_transform" => {
                let _limit = engine.limits().scope(stopped());
                let result = engine.and(circuit.take().unwrap(), saved.as_ref().unwrap().clone());
                assert!(matches!(result, Err(OperationError::Stopped)), "{context}");
            }
            "recover" => circuit = saved.clone(),
            "roundtrip" => {
                let mut bytes = Vec::new();
                tididi::io::write_tdd(&mut bytes, circuit.as_ref().unwrap()).unwrap();
                circuit = Some(tididi::io::read_tdd(&mut bytes.as_slice(), &vtree).unwrap());
            }
            "minimize" => circuit.as_mut().unwrap().minimize().unwrap(),
            op => panic!("unknown session operation {op}"),
        }
        let answer = if let Some(q) = query.as_mut() { q.read(&engine).unwrap() }
            else { circuit.as_ref().or(saved.as_ref()).unwrap().model_count().unwrap().to_string() };
        assert_eq!(answer, expected.to_string(), "{context}");
        if let Some(f) = &circuit { tididi::test_helpers::assert_canonical(f); }
        if let Some(backup) = &saved {
            assert_eq!(backup.model_count().unwrap(), original_truth.count_ones().into(), "{context}: saved circuit");
        }
    }
}
