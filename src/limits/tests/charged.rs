use crate::limits::{Limits, Transient};

fn in_flight(lim: &Limits) -> u64 {
    lim.meters().in_flight_bytes
}

#[test]
fn a_dropped_transient_hands_its_capacity_back() {
    let lim = Limits::new();
    let _op = lim.begin_operation();
    let mut buf = Transient::new(&lim, Vec::<u64>::new());
    lim.reserve_exact(&mut buf, 100).unwrap();
    assert_eq!(in_flight(&lim), 800);
    drop(buf);
    assert_eq!(in_flight(&lim), 0);
}

#[test]
fn a_kept_buffer_stays_charged() {
    let lim = Limits::new();
    let _op = lim.begin_operation();
    let mut buf = Transient::new(&lim, Vec::<u64>::new());
    lim.reserve_exact(&mut buf, 100).unwrap();
    let kept = buf.keep();
    assert_eq!(in_flight(&lim), 800);
    lim.discard(kept);
    assert_eq!(in_flight(&lim), 0);
}
