use super::*;
use crate::vtree::rng::Lcg;
use crate::Engine;

/// Triples whose fields stay under the given widths, in ascending order of
/// their src as `collect_triples` leaves them: their fields after a sort in
/// the fitted word (or the wide one past 64 bits) and after a comparison
/// sort of the wide words.
fn sorted_both_ways(rng: &mut Lcg, width: [u32; 4], n: usize) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
    let field = |rng: &mut Lcg, w: u32| if w == 0 { 0 } else { (rng.next_u64() as u32) >> (32 - w) };
    let wide = &Layout::WIDE;
    let mut triples: Vec<u128> = (0..n)
        .map(|_| {
            let f: Vec<u32> = width.iter().map(|&w| field(rng, w)).collect();
            let inner = ChildPair::new(EncodedChildRef::from_raw(f[0]), EncodedChildRef::from_raw(f[1]));
            u128::pack(wide, inner, f[3], EncodedChildRef::from_raw(f[2]))
        })
        .collect();
    triples.sort_by_key(|&t| tri_src(t, wide));
    let fields = |p: u128| std::array::from_fn(|k| p.field(wide, k));
    let mut want = triples.clone();
    want.sort_unstable();
    let want = want.into_iter().map(fields).collect();
    let eng = Engine::new();
    let fitted = Layout::fitted(width.map(|w| if w == 0 { 0 } else { u32::MAX >> (32 - w) }));
    let got = if fitted.bits() <= 64 {
        let mut words: Vec<u64> = triples.iter().map(|&p| u64::pack(&fitted, tri_inner(p, wide), tri_src(p, wide), tri_axis(p, wide))).collect();
        u64::sort(eng.limits(), &fitted, &mut words).unwrap();
        words.into_iter().map(|w| std::array::from_fn(|k| w.field(&fitted, k))).collect()
    } else {
        u128::sort(eng.limits(), wide, &mut triples).unwrap();
        triples.into_iter().map(fields).collect()
    };
    (got, want)
}

#[test]
fn a_fitted_word_keeps_the_order_of_the_triples() {
    let mut rng = Lcg::new(11);
    let n = RADIX_MIN_ROWS + 101;
    // One word, with a field empty, with the inner pair empty (a field at bit
    // 64), and past one word (the wide word).
    for width in [[12, 12, 20, 20], [0, 17, 30, 17], [1, 1, 1, 1], [0, 0, 32, 32], [20, 20, 20, 20], [32, 32, 32, 32]] {
        let (got, want) = sorted_both_ways(&mut rng, width, n);
        assert_eq!(got, want, "fields {width:?}");
    }
    let (got, want) = sorted_both_ways(&mut rng, [8, 8, 8, 8], 100);
    assert_eq!(got, want, "a short run");
}
