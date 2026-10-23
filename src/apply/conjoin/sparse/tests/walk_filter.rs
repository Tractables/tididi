//! The counted root's filtered walk: a round that opened few inner-g
//! children tests each key the walk reads against a filter over them and
//! holds the keys that pass until their exact test's reads have arrived,
//! and a level whose filter passes too many keys walks the rest unfiltered.
//! The count must be the one the unfiltered walk, which tests every key
//! against the round's bit set, returns, and the model count of the
//! conjunction built in full: on random joins, on skewed ones whose rounds
//! hold more hits than the ring, with a filter pinned to one word, which
//! every key passes, so every test is the exact one, with a level switched
//! to the unfiltered walk after a few keys and never, and with the levels
//! whose count column is under a size walked unfiltered.
use super::inner_index::{block, pack};
use super::{random_rows, vtrees, walks_run, FilterPin, ForcedFilter, ForcedThresholds, SparseThresholds};
use crate::test_helpers::assert_canonical;
use crate::Engine;

#[test]
fn a_filtered_walk_counts_what_the_unfiltered_walk_counts() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let (a, b, c) = (block(1, 7), block(8, 7), block(15, 7));
    let pins = [
        None,
        Some(FilterPin { keys: 0, bits_log: None, sample: u64::MAX, min_bytes: 0 }),
        Some(FilterPin { keys: 1 << 15, bits_log: Some(6), sample: u64::MAX, min_bytes: 0 }),
        Some(FilterPin { keys: 1 << 15, bits_log: Some(6), sample: 16, min_bytes: 0 }),
        Some(FilterPin { keys: 1 << 15, bits_log: Some(9), sample: u64::MAX, min_bytes: 0 }),
        Some(FilterPin { keys: 1 << 15, bits_log: Some(9), sample: 0, min_bytes: 0 }),
        Some(FilterPin { keys: 1 << 15, bits_log: None, sample: u64::MAX, min_bytes: 1 << 12 }),
        Some(FilterPin { keys: 1 << 15, bits_log: None, sample: u64::MAX, min_bytes: usize::MAX }),
    ];
    let before = walks_run();
    for (seed, n, skew) in [(0u64, 400usize, false), (1, 1500, false), (2, 3000, true), (3, 6000, true)] {
        let f3 = random_rows(seed + 200, n, &[7; 3], skew);
        let mut g3 = random_rows(seed + 300, n / 2, &[7; 3], skew);
        g3.extend(f3.iter().step_by(2).cloned());
        let (f2, g2) = (random_rows(seed, n, &[7; 2], skew), random_rows(seed + 100, n, &[7; 2], skew));
        let joins = [
            (pack(&[&a, &b], &f2), pack(&[&a, &c], &g2)),
            (pack(&[&a, &b], &f2), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b, &c], &f3), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b, &c], &f3), pack(&[&a, &b, &c], &g3)),
        ];
        for vtree in vtrees(&a, &b, &c) {
            for ((fv, fr), (gv, gr)) in &joins {
                let f = eng.from_models(&vtree, fv, fr).unwrap();
                let g = eng.from_models(&vtree, gv, gr).unwrap();
                assert_canonical(&f);
                assert_canonical(&g);
                let built = eng.and(f.clone(), g.clone()).unwrap();
                assert_canonical(&built);
                let expected = eng.model_count(&built).unwrap();
                let _forced = ForcedThresholds::install(sparse);
                for pin in pins {
                    let _pinned = ForcedFilter::install(pin);
                    for (l, r) in [(&f, &g), (&g, &f)] {
                        let counted = eng.and_model_count(l.clone(), r.clone(), &[]).unwrap();
                        assert_eq!(counted, expected, "seed {seed}, {pin:?}");
                    }
                }
            }
        }
    }
    // Every way the walk runs was reached: filtered, unfiltered, filtered
    // with the ring of held hits wrapping, unfiltered on a level whose
    // filter passed too many keys, and on a level whose count column is too
    // narrow to filter, the walk the level ran before the filter.
    let after = walks_run();
    for kind in 0..5 {
        assert!(after[kind] > before[kind], "walk kind {kind} never ran: {before:?} -> {after:?}");
    }
}
