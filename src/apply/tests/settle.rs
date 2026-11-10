use super::*;

/// A node read with more pairs than its hits number in 32 bits, here past
/// a pinned list cap, is refused with `IndexOverflow` rather than read with
/// wrapped positions.
#[test]
fn a_node_past_the_pair_positions_is_refused() {
    let lim = Limits::new();
    let mut st = Settle {
        width: 2,
        hits: Vec::new(),
        pieces: Runs::default(),
        made: Runs::default(),
    };
    st.pieces.begin(&lim).unwrap();
    st.made.begin(&lim).unwrap();
    lim.pin_list_cap(Some(4));
    st.node(&lim, 0, 3).unwrap();
    assert_eq!(st.node(&lim, 1, 4), Err(OperationError::IndexOverflow));
    lim.pin_list_cap(None);
    st.node(&lim, 1, 4).unwrap();
    assert_eq!(st.pieces.runs(), 2);
}
