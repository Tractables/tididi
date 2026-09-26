//! The reverse indexes and the candidate grouping are counting sorts with
//! 32-bit offsets. More items than `u32::MAX` must refuse with
//! `IndexOverflow` rather than wrap the total and scatter past the entries.
use crate::limits::OperationError;
use crate::Engine;

use super::super::index::{counting_sort, Grouped};

/// Histograms whose total passes `u32::MAX` refuse before any entry is
/// sized, whichever key carries it past.
#[test]
fn a_histogram_past_u32_refuses() {
    let eng = Engine::new();
    for counts in [[u32::MAX, 1], [1, u32::MAX], [u32::MAX / 2 + 1, u32::MAX / 2 + 1]] {
        let mut out = Grouped::<u32>::default();
        let got = counting_sort(eng.limits(), 2, std::iter::empty::<(usize, u32)>(), |s| s, Some(&counts), 0, &mut out);
        assert_eq!(got, Err(OperationError::IndexOverflow), "{counts:?}");
        assert!(out.entries.is_empty());
    }
}

/// Small sorts still group by key, stably.
#[test]
fn items_are_grouped_by_key_in_order() {
    let eng = Engine::new();
    let items = [(2usize, 20u32), (0, 1), (2, 21), (1, 10), (0, 2)];
    let mut out = Grouped::<u32>::default();
    counting_sort(eng.limits(), 3, items.iter().copied(), |s| s, None, 0, &mut out).unwrap();
    assert_eq!(out.offsets, [0, 2, 3, 5]);
    assert_eq!(out.entries, [1, 2, 10, 20, 21]);
}

