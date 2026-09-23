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
