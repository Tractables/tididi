//! The reserved emit sink against the emit sink it stands in for.

use super::*;
use crate::Engine;
use crate::test_helpers::pair;

/// [`doubled_pairs_capacity`] is the capacity a full pair arena grows to
/// when one more pair is pushed through the fallible push.
#[test]
fn doubled_pairs_capacity_is_the_arenas_growth() {
    let eng = Engine::new();
    for cap in [0usize, 1, 2, 3, 4, 5, 7, 8, 1000, 8192, 12_345] {
        let mut v: Vec<ChildPair> = Vec::with_capacity(cap);
        let cap = v.capacity();
        v.resize(cap, pair(0, 0));
        eng.limits().try_push(&mut v, pair(1, 1)).unwrap();
        assert_eq!(v.capacity(), doubled_pairs_capacity(cap), "a full arena of {cap}");
    }
}

/// A reserved sink charges the output-pair meter after every pair where an
/// emit sink seeded at the same capacity does, and holds the same pairs; with
/// no schedule of its own it is an emit sink.
#[test]
fn a_reserved_sink_charges_where_an_emit_sink_grows() {
    let eng_emit = Engine::new();
    let eng_reserved = Engine::new();
    let _ops = (eng_emit.limits().begin_operation(), eng_reserved.limits().begin_operation());
    let cands: Vec<(u32, u32)> = (0..300u32).map(|k| (k, k + 7)).collect();
    for (seed, total) in [(0usize, 300usize), (4, 300), (8, 300), (64, 300), (512, 300)] {
        for unscheduled in [false, true] {
            let mut emit = TddLevel::new();
            emit.pairs.reserve_exact(seed);
            let mut reserved = TddLevel::new();
            reserved.pairs.reserve_exact(seed);
            let charged = if unscheduled { usize::MAX } else { reserved.pairs.capacity() };
            if !unscheduled {
                reserved.pairs.reserve_exact(total);
            }
            let mut a = EmitSink { level: &mut emit };
            let mut b = ReservedEmitSink { level: &mut reserved, charged };
            // Runs of uneven length, so room runs out mid-run and between runs.
            let mut at = 0;
            for len in [1usize, 3, 2, 7, 50, 1, 100, 136] {
                let run = &cands[at..at + len];
                at += len;
                push_kept(&eng_emit, &mut a, run, false, |&(l, r)| (l, r)).unwrap();
                push_kept(&eng_reserved, &mut b, run, false, |&(l, r)| (l, r)).unwrap();
                assert_eq!(
                    eng_reserved.limits().meters().pairs_in_flight,
                    eng_emit.limits().meters().pairs_in_flight,
                    "seed {seed}, unscheduled {unscheduled}, after {at} pairs"
                );
            }
            assert_eq!(at, total);
            assert_eq!(b.buf()[..], a.buf()[..]);
            if unscheduled {
                assert_eq!(b.buf().capacity(), a.buf().capacity(), "an unscheduled sink grows as an emit sink");
            } else {
                assert_eq!(b.buf().capacity(), seed.max(total), "a reserved arena never grows");
            }
            eng_emit.limits().level_settled(0);
            eng_reserved.limits().level_settled(0);
        }
    }
}
