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

#[test]
fn an_empty_buffer_grows_to_the_capacity_vec_would_give_it() {
    use crate::limits::growth::amortized_capacity;
    fn check<T: Clone + Default>() {
        for capacity in [0usize, 1, 3, 4, 7, 8, 100, 1000] {
            for additional in [1usize, 2, 4, 5, 9, 16, 150, 3000] {
                if additional <= capacity {
                    continue;
                }
                let mut by_vec: Vec<T> = Vec::with_capacity(capacity);
                by_vec.try_reserve(additional).unwrap();
                assert_eq!(amortized_capacity::<T>(capacity, additional), by_vec.capacity(), "{capacity} {additional}");
                // Through the limits, an empty buffer with a block, cleared
                // or never filled, ends where `Vec` would have it and is
                // charged the same growth.
                for exact in [false, true] {
                    let lim = Limits::new();
                    let _op = lim.begin_operation();
                    let mut v: Vec<T> = vec![T::default(); capacity];
                    v.clear();
                    let mut by_vec: Vec<T> = Vec::with_capacity(capacity);
                    if exact { by_vec.reserve_exact(additional) } else { by_vec.reserve(additional) }
                    if exact { lim.reserve_exact(&mut v, additional).unwrap() } else { lim.reserve(&mut v, additional).unwrap() }
                    assert_eq!(v.capacity(), by_vec.capacity(), "{capacity} {additional} {exact}");
                    assert!(v.is_empty());
                    let grown = (v.capacity() - capacity) * std::mem::size_of::<T>();
                    assert_eq!(in_flight(&lim), grown as u64);
                }
            }
        }
    }
    check::<u8>();
    check::<u64>();
    // Wider than the 1 KiB past which `Vec` allocates one element at least.
    check::<[[u64; 32]; 5]>();
}

/// What `charge_as_reserved` gives and charges, with nothing allocated, is
/// what `reserve_exact` gives an empty buffer of that capacity and charges
/// for it.
#[test]
fn a_charge_as_reserved_is_an_exact_reserve_without_the_buffer() {
    use crate::diagram::ChildPair;
    for capacity in [0usize, 1, 7, 100] {
        for additional in [0usize, 1, 7, 50, 3000] {
            let by_reserve = Limits::new();
            let _op = by_reserve.begin_operation();
            let mut v: Vec<ChildPair> = Vec::with_capacity(capacity);
            let before = v.capacity();
            by_reserve.reserve_exact(&mut v, additional).unwrap();
            let by_charge = Limits::new();
            let _op = by_charge.begin_operation();
            let grown = by_charge.charge_as_reserved::<ChildPair>(before, additional).unwrap();
            assert_eq!(grown, v.capacity(), "{capacity} {additional}");
            assert_eq!(in_flight(&by_charge), in_flight(&by_reserve), "{capacity} {additional}");
        }
    }
}

/// What `reserve_as_pushed` gives and charges is what as many pushes give
/// and charge, from an empty buffer or one partly filled.
#[test]
fn a_reserve_as_pushed_is_a_run_of_pushes() {
    for capacity in [0usize, 1, 4, 7, 100] {
        for len in [0usize, 1, 4] {
            for additional in [0usize, 1, 3, 4, 5, 64, 3000] {
                let by_push = Limits::new();
                let _op = by_push.begin_operation();
                let mut pushed: Vec<u64> = Vec::with_capacity(capacity.max(len));
                pushed.extend(0..len as u64);
                let mut reserved: Vec<u64> = Vec::with_capacity(capacity.max(len));
                reserved.extend(0..len as u64);
                assert_eq!(reserved.capacity(), pushed.capacity());
                for x in 0..additional as u64 {
                    by_push.try_push(&mut pushed, x).unwrap();
                }
                let by_reserve = Limits::new();
                let _op = by_reserve.begin_operation();
                by_reserve.reserve_as_pushed(&mut reserved, additional).unwrap();
                assert_eq!(reserved.capacity(), pushed.capacity(), "{capacity} {len} {additional}");
                assert_eq!(in_flight(&by_reserve), in_flight(&by_push), "{capacity} {len} {additional}");
            }
        }
    }
}

#[test]
fn a_refused_doubling_asks_for_a_quarter_more_or_for_what_is_needed() {
    use crate::limits::growth::grow_less;
    // Room for a quarter more than the length where that covers the request;
    // the request alone where it does not. Either way no more than that.
    for (len, additional, room) in [(1000usize, 1usize, 250usize), (1000, 249, 250), (1000, 250, 250), (1000, 251, 251), (1000, 4000, 4000), (3, 1, 1), (64, 1, 16)] {
        let mut v: Vec<u64> = vec![7; len];
        v.shrink_to_fit();
        grow_less(&mut v, additional).unwrap();
        assert!(v.capacity() >= len + room, "{len} {additional}: {}", v.capacity());
        assert!(v.capacity() < 2 * len.max(1) + additional, "{len} {additional}: grew to {}", v.capacity());
        assert_eq!(v.len(), len);
        assert!(v.iter().all(|&x| x == 7));
    }
}
