use super::*;
use crate::vtree::rng::Lcg;

/// The bytes every row holds, walked rather than kept.
fn walked_bytes<T>(rows: &Rows<T>) -> u64 {
    rows.rows.charged_bytes() + row_bytes(&rows.rows)
}

#[test]
fn the_kept_byte_count_of_the_rows_is_the_count_a_walk_of_every_row_gives() {
    let eng = Engine::new();
    let mut rng = Lcg::new(11);
    let mut rows: Rows<u32> = Rows::default();
    for _ in 0..300 {
        let n = rng.below(64) as usize;
        rows.reset(eng.limits(), n).unwrap();
        assert_eq!(rows.len(), n);
        assert!(rows.iter().all(Vec::is_empty));
        for _ in 0..rng.below(40) {
            if n == 0 { break; }
            let at = rng.below(n as u64) as usize;
            if rng.below(8) == 0 {
                // The chunked dedup takes a consumed row's allocation away.
                drop(std::mem::take(&mut rows[at]));
            } else {
                rows[at].push(rng.next_u64() as u32);
            }
        }
        assert_eq!(rows.charged_bytes(), walked_bytes(&rows));
    }
}

#[test]
fn a_row_past_a_narrower_level_keeps_its_allocation_for_a_wider_one() {
    let eng = Engine::new();
    let mut rows: Rows<u32> = Rows::default();
    rows.reset(eng.limits(), 10).unwrap();
    rows[7].extend(0..100);
    rows.reset(eng.limits(), 3).unwrap();
    rows.reset(eng.limits(), 10).unwrap();
    assert!(rows[7].is_empty());
    assert!(rows[7].capacity() >= 100);
    assert_eq!(rows.charged_bytes(), walked_bytes(&rows));
}
