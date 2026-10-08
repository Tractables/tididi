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
