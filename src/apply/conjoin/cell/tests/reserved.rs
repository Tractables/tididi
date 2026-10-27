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
            emit.pairs.stored_mut().reserve_exact(seed);
            let mut reserved = TddLevel::new();
            reserved.pairs.stored_mut().reserve_exact(seed);
            let charged = if unscheduled { usize::MAX } else { reserved.pairs.capacity() };
            if !unscheduled {
                reserved.pairs.stored_mut().reserve_exact(total);
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

/// A projected sink charges the output-pair meter after every pair where an
/// emit sink seeded at the same capacity does, and holds the same pairs,
/// whether its reservation ahead covers the level, falls short of it, is cut
/// by its bound or finds the room there already; a reservation that covers
/// the level leaves the arena at its size.
#[test]
fn a_projected_sink_charges_where_an_emit_sink_grows() {
    let eng_emit = Engine::new();
    let eng_projected = Engine::new();
    let _ops = (eng_emit.limits().begin_operation(), eng_projected.limits().begin_operation());
    let cands: Vec<(u32, u32)> = (0..3000u32).map(|k| (k, k + 7)).collect();
    let runs = [1usize, 3, 2, 7, 50, 1, 100, 136, 700, 2000];
    // The rows done and in all when the reservation is made, after the
    // first four runs (13 pairs), and the most it may reserve.
    for seed in [0usize, 8, 64] {
        for (done, rows, most) in [(1, 512, usize::MAX), (1, 64, usize::MAX), (1, 512, 100), (1, 2, usize::MAX)] {
            let mut emit = TddLevel::new();
            emit.pairs.stored_mut().reserve_exact(seed);
            let mut projected = TddLevel::new();
            projected.pairs.stored_mut().reserve_exact(seed);
            let mut a = EmitSink { level: &mut emit };
            let mut b = ProjectedEmitSink::new(eng_projected.limits(), (most, most), rows, &mut projected);
            let mut at = 0;
            let mut reserved = 0;
            for (k, len) in runs.into_iter().enumerate() {
                if k == 4 {
                    b.reserve_ahead(&eng_projected, done).unwrap();
                    reserved = b.buf().capacity();
                }
                let run = &cands[at..at + len];
                at += len;
                push_kept(&eng_emit, &mut a, run, false, |&(l, r)| (l, r)).unwrap();
                push_kept(&eng_projected, &mut b, run, false, |&(l, r)| (l, r)).unwrap();
                assert_eq!(
                    eng_projected.limits().meters().pairs_in_flight,
                    eng_emit.limits().meters().pairs_in_flight,
                    "seed {seed}, {done} of {rows} rows, most {most}, after {at} pairs"
                );
            }
            assert_eq!(b.buf()[..], a.buf()[..]);
            // 13 pairs in one of 512 rows project 7488, an eighth over.
            if rows == 512 && most == usize::MAX {
                assert_eq!(reserved, 7488, "seed {seed}");
                assert_eq!(b.buf().capacity(), 7488, "a level within its reservation does not grow");
            }
            if most == 100 {
                assert_eq!(reserved, 100.max(seed), "a reservation within its bound");
            }
            eng_emit.limits().level_settled(0);
            eng_projected.limits().level_settled(0);
        }
    }
}
