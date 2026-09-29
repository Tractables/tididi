use super::*;
use crate::vtree::rng::Lcg;
use crate::Engine;

/// Triples whose fields stay under the given widths, sorted by
/// `sort_triples` and by a comparison sort.
fn sorted_both_ways(rng: &mut Lcg, width: [u32; 4], n: usize) -> (Vec<u128>, Vec<u128>) {
    let field = |rng: &mut Lcg, w: u32| if w == 0 { 0 } else { (rng.next_u64() as u32) >> (32 - w) };
    let triples: Vec<u128> = (0..n)
        .map(|_| {
            let f: Vec<u32> = width.iter().map(|&w| field(rng, w)).collect();
            ((f[0] as u128) << 96) | ((f[1] as u128) << 64) | ((f[2] as u128) << 32) | f[3] as u128
        })
        .collect();
    let mut want = triples.clone();
    want.sort_unstable();
    let mut got = triples;
    let eng = Engine::new();
    sort_triples(eng.limits(), &mut got, &mut Vec::new()).unwrap();
    (got, want)
}

#[test]
fn cutting_the_fields_to_one_word_keeps_the_order_of_the_triples() {
    let mut rng = Lcg::new(11);
    let n = RADIX_MIN_ROWS + 101;
    // One word, with a field empty, and past one word (the comparison sort).
    for width in [[12, 12, 20, 20], [0, 17, 30, 17], [1, 1, 1, 1], [20, 20, 20, 20], [32, 32, 32, 32]] {
        let (got, want) = sorted_both_ways(&mut rng, width, n);
        assert_eq!(got, want, "fields {width:?}");
    }
    let (got, want) = sorted_both_ways(&mut rng, [8, 8, 8, 8], 100);
    assert_eq!(got, want, "a short run");
}
