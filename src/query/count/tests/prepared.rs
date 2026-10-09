use super::*;
use crate::{Vtree, query::{Retention, PinSemantics}};
use crate::vtree::VarId;
use crate::test_helpers::assert_canonical;
use num_bigint::BigUint;
use std::sync::Arc;

fn table(tree: &Arc<Vtree>, vars: &[VarId]) -> (Tdd, Vec<u64>) {
    let rows: Vec<_> = (0..8192u64).map(|i| (i * 40503) & 65535).collect();
    let f = Tdd::from_models(tree, vars, &rows).unwrap();
    assert_canonical(&f);
    (f, rows)
}

#[test]
fn prepared_updates_match_model_enumeration() {
    for sparse in [false, true] {
        let vars: Vec<_> = (1..=16).map(|i| VarId(if sparse { i * 101 } else { i })).collect();
        let tree = Arc::new(Vtree::balanced_over(&vars).unwrap());
        let (f, rows) = table(&tree, &vars);
        let plan = Prepared::new(&Engine::new(), &f).unwrap();
        assert!(plan.levels.iter().any(Option::is_some));
        for retention in [Retention::All, Retention::Frontier] {
            for convention in [PinSemantics::Evidence, PinSemantics::Cofactor] {
                let mut counter = f.counter_with(retention, convention).unwrap();
                counter.observe([crate::Literal { var: vars[0], sign: true }]).unwrap();
                counter.model_count().unwrap();
                counter.prepare().unwrap();
                let mut pins = [None; 16];
                pins[0] = Some(true);
                for step in 0..192usize {
                    let v = (step * 7) % 16;
                    let pin = match step % 5 { 0 => None, 1 | 2 => Some(true), _ => Some(false) };
                    pins[v] = pin;
                    counter.set_pin(vars[v], pin).unwrap();
                    let count = rows.iter().filter(|&&row| pins.iter().enumerate().all(|(i, &pin)| pin.is_none_or(|p| p == (row & (1 << i) != 0)))).count();
                    let free = if convention == PinSemantics::Cofactor { pins.iter().filter(|pin| pin.is_some()).count() } else { 0 };
                    assert_eq!(counter.model_count().unwrap(), BigUint::from(count) << free, "step {step}");
                    if step % 19 == 0 { counter.prepare().unwrap(); }
                }
                counter.clear_pins();
                assert_eq!(counter.model_count().unwrap(), BigUint::from(rows.len()));
            }
        }
    }
}

#[test]
fn prepared_counts_promote_and_retry_after_every_reservation_failure() {
    let tree = Arc::new(Vtree::balanced(160));
    let vars: Vec<_> = (0..16).map(|i| VarId(1 + 10 * i)).collect();
    let free: Vec<_> = (1..=160).map(VarId).filter(|v| !vars.contains(v)).collect();
    let (f, rows) = table(&tree, &vars);
    let eng = Engine::new();
    let mut counter = f.counter().unwrap();
    counter.prepare().unwrap();
    let plan = Prepared::new(&eng, &f).unwrap();
    assert!(plan.levels[tree.root().idx()].is_some(), "overflowing root uses packed pairs");
    assert_eq!(counter.model_count().unwrap(), BigUint::from(rows.len()) << 144);
    for pinned in [144, 100, 30, 0, 144, 0] {
        counter.clear_pins();
        let pins: Vec<_> = free.iter().take(pinned).map(|&v| (v, Some(false))).collect();
        counter.set_pins(&pins).unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(rows.len()) << (144 - pinned));
    }
    let mut finished = false;
    for refusal in 0..4096 {
        let mut counter = f.counter().unwrap();
        counter.observe([-1]).unwrap();
        let before = counter.model_count().unwrap();
        eng.limits().refuse_nth_reserve(refusal);
        let result = counter.bind(&eng).prepare();
        eng.limits().grant_every_reserve();
        assert_eq!(counter.model_count().unwrap(), before);
        match result {
            Err(OperationError::OverBudget) => (),
            Ok(()) => { finished = true; break; }
            Err(error) => panic!("unexpected preparation refusal: {error}"),
        }
    }
    assert!(finished);
}

#[test]
fn packed_word_boundaries_and_exact_overflow() {
    let eng = Engine::new();
    for width in [16, 32, 64] {
        let mut words = Words::new(&eng, width, 2).unwrap();
        words.push(0);
        words.push(low(width));
        assert_eq!(words.charged_bytes(), u64::from(width / 8) * 2);
    }
    let left = 31;
    let raw = u64::from(u32::MAX >> 1) | (u64::from(u32::MAX >> 1) << left);
    let pairs = [raw];
    let decoded: Vec<_> = Inputs { inline: None, pairs: pairs.iter(), left }.collect();
    assert_eq!(decoded[0].left.raw(), u32::MAX >> 1);
    assert_eq!(decoded[0].right.raw(), u32::MAX >> 1);
    let pair = [0u16, 0, 0];
    let input = Inputs { inline: None, pairs: pair.iter(), left: 1 };
    let total = IntFold::fold(input, |_| crate::value::CountRead::Fast(u128::MAX), |_| crate::value::CountRead::Fast(u128::MAX));
    match total { Count::Big(v) => assert_eq!(v, BigUint::from(u128::MAX).pow(2) * 3u32), _ => panic!("expected overflow") }
}

#[test]
fn preparation_stops_atomically_and_count_tables_restore_pins() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tree = Arc::new(Vtree::balanced(16));
    let vars: Vec<_> = (1..=16).map(VarId).collect();
    let (f, _) = table(&tree, &vars);
    let eng = Engine::new();
    let mut c = f.counter().unwrap();
    c.observe([-1, 2]).unwrap();
    let before = c.model_count().unwrap();
    let total_polls = Arc::new(AtomicUsize::new(0));
    {
        let seen = Arc::clone(&total_polls);
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
            seen.fetch_add(1, Ordering::Relaxed);
            StopDecision::Continue
        }))));
        f.counter().unwrap().bind(&eng).prepare().unwrap();
    }
    assert!(total_polls.load(Ordering::Relaxed) > 2);
    for stop_at in 0..total_polls.load(Ordering::Relaxed) {
        let polls = Arc::new(AtomicUsize::new(0));
        {
            let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
                if polls.fetch_add(1, Ordering::Relaxed) >= stop_at { StopDecision::Stop } else { StopDecision::Continue }
            }))));
            assert_eq!(c.bind(&eng).prepare(), Err(OperationError::Stopped));
        }
        assert_eq!(c.model_count().unwrap(), before);
    }
    c.bind(&eng).prepare().unwrap();
    let mut oracle = f.counter().unwrap();
    oracle.observe([-1, 2]).unwrap();
    assert_eq!(c.count_table(&vars[2..6]).unwrap(), oracle.count_table(&vars[2..6]).unwrap());
    assert_eq!(c.model_count().unwrap(), before);
    {
        let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Stop))));
        assert_eq!(c.bind(&eng).prepare(), Err(OperationError::Stopped));
        assert_eq!(c.bind(&eng).model_count(), Err(OperationError::Stopped));
    }
    assert_eq!(c.model_count().unwrap(), before);
}

#[test]
fn marginal_implicit_and_constant_levels_keep_their_readers() {
    let tree = Arc::new(Vtree::balanced(16));
    let vars: Vec<_> = (1..=16).map(VarId).collect();
    let (mut f, _) = table(&tree, &vars);
    f.marginalize_levels(&[tree.leaf_of(VarId(1)).unwrap()]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    let mut c = f.counter().unwrap();
    c.prepare().unwrap();
    for var in &vars[1..] {
        c.clear_pins();
        c.set_pin(*var, Some(true)).unwrap();
        let mut oracle = f.counter().unwrap();
        oracle.set_pin(*var, Some(true)).unwrap();
        assert_eq!(c.model_count().unwrap(), oracle.model_count().unwrap());
    }
    assert!(matches!(c.set_pin(VarId(1), None), Err(OperationError::MarginalLevel(_))));
    for f in [Tdd::one(&tree), Tdd::zero(&tree)] {
        assert_canonical(&f);
        let eng = Engine::new();
        let plan = Prepared::new(&eng, &f).unwrap();
        assert!(plan.ready && plan.levels.is_empty());
        let mut c = f.counter().unwrap();
        c.prepare().unwrap();
        assert_eq!(c.model_count().unwrap(), f.model_count().unwrap());
        eng.limits().refuse_nth_reserve(0);
        c.bind(&eng).prepare().unwrap();
        eng.limits().grant_every_reserve();
    }
}

#[test]
fn full_width_headers_and_packed_overflow_use_the_shared_fold() {
    use super::super::column::QueryCounts;
    let eng = Engine::new();
    let mut left = QueryCounts::try_with_width(&eng, 1).unwrap();
    let mut right = QueryCounts::try_with_width(&eng, 1).unwrap();
    left.set(&eng, 0, Count::from_u128(u128::MAX)).unwrap();
    right.set(&eng, 0, Count::Fast(2)).unwrap();
    let mut out = QueryCounts::try_with_width(&eng, 3).unwrap();
    // Inline, empty range, and one arena pair, with no bits left for a mask.
    let nodes = [0u64, 1, 1 | (1 << 33)];
    let pairs = [0u16];
    let packed = Packed {
        nodes: Words::U64(nodes.to_vec()), pairs: Words::U16(pairs.to_vec()),
        layout: Layout { left: 31, start: 32, payload: 64, node_word: 64, pair_word: 16, pairs: 1 },
        vars: [VtreeIdx(0); TRACKED], tracked: 0,
    };
    packed.fold(&eng, &nodes, &pairs, &left, &right, &mut out, 0, &mut eng.limits().gate()).unwrap();
    let expected = BigUint::from(u128::MAX) * 2u32;
    assert!(matches!(out.get(0), crate::value::CountRead::Big(v) if v == &expected));
    assert!(matches!(out.get(1), crate::value::CountRead::Fast(0)));
    assert!(matches!(out.get(2), crate::value::CountRead::Big(v) if v == &expected));
}

#[test]
fn prepared_model_tables_on_random_trees_match_enumeration() {
    let vars: Vec<_> = (1..=24).map(VarId).collect();
    for seed in [7, 91, 0x7ead_beef] {
        let tree = Arc::new(Vtree::random(24, seed));
        let rows: Vec<_> = (0..32768u64).map(|i| {
            let value = (i * 517807 + seed) & ((1 << 24) - 1);
            let value = value ^ (value >> 7);
            (value ^ (value << 9)) & ((1 << 24) - 1)
        }).collect();
        let f = Tdd::from_models(&tree, &vars, &rows).unwrap();
        assert_canonical(&f);
        let plan = Prepared::new(&Engine::new(), &f).unwrap();
        assert!(plan.levels.iter().any(Option::is_some));
        let mut counter = f.counter().unwrap();
        counter.prepare().unwrap();
        for step in 0..64u64 {
            counter.clear_pins();
            let mask = ((step * 32771) ^ seed) & ((1 << 24) - 1);
            let value = step.wrapping_mul(104729) & mask;
            let pins: Vec<_> = (0..24).filter(|&i| mask & (1 << i) != 0).map(|i| (VarId(i + 1), Some(value & (1 << i) != 0))).collect();
            counter.set_pins(&pins).unwrap();
            let expected = rows.iter().filter(|&&row| row & mask == value).count();
            assert_eq!(counter.model_count().unwrap(), BigUint::from(expected));
        }
    }
}
