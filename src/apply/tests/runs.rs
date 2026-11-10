use super::*;

/// Runs closed one after another, empty ones among them, read back as
/// pushed.
#[test]
fn runs_read_back_as_closed() {
    let lim = Limits::new();
    let mut runs: Runs<u32> = Runs::default();
    assert_eq!(runs.runs(), 0);
    runs.begin(&lim).unwrap();
    assert_eq!(runs.runs(), 0);
    runs.push(&lim, 3).unwrap();
    runs.push(&lim, 1).unwrap();
    runs.open_run_mut().sort_unstable();
    runs.close(&lim).unwrap();
    runs.close_until(&lim, 3).unwrap();
    runs.push(&lim, 7).unwrap();
    runs.close(&lim).unwrap();
    runs.close_until(&lim, 2).unwrap();
    assert_eq!(runs.runs(), 4);
    let read: Vec<&[u32]> = (0..runs.runs()).map(|k| runs.run(k)).collect();
    assert_eq!(read, [&[1, 3][..], &[], &[], &[7]]);
    runs.begin_exact(&lim, 2).unwrap();
    assert_eq!((runs.runs(), runs.items.len()), (0, 0));
}

/// A run that would end past the offsets is refused as `IndexOverflow`,
/// here under a pinned list cap, and leaves the runs closed before it.
#[test]
fn a_run_past_the_offsets_is_refused() {
    let lim = Limits::new();
    let mut runs: Runs<u32> = Runs::default();
    runs.begin(&lim).unwrap();
    lim.pin_list_cap(Some(3));
    runs.push(&lim, 0).unwrap();
    runs.push(&lim, 1).unwrap();
    runs.close(&lim).unwrap();
    runs.push(&lim, 2).unwrap();
    assert_eq!(runs.close(&lim), Err(OperationError::IndexOverflow));
    assert_eq!(runs.close_until(&lim, 3), Err(OperationError::IndexOverflow));
    lim.pin_list_cap(None);
    assert_eq!(runs.runs(), 1);
    assert_eq!(runs.run(0), [0, 1]);
}
