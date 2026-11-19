use super::*;
use super::data::{rebuild_level, reserve_rebuilt};

/// The remap a merge rewrites the parent through, where node `i` merges
/// into `targets[i]` (itself for a survivor) and the members in
/// `redirected` are content-equal redirects, which keep their parent pairs
/// and are renamed onto their survivor's new number; every other member
/// keeps its survivor's old index under [`DROPPED`].
fn remap_of(targets: &[u32], redirected: &[usize]) -> MergeRemap {
    let mut remap = MergeRemap {
        final_remap: vec![NodeIdx(0); targets.len()],
        absorbed: vec![0; targets.len().div_ceil(64)],
        ..Default::default()
    };
    let mut next = 0;
    for (i, &t) in targets.iter().enumerate() {
        if t as usize == i {
            remap.final_remap[i] = NodeIdx(next);
            next += 1;
        } else {
            remap.absorbed[i / 64] |= 1 << (i % 64);
            remap.final_remap[i] = if redirected.contains(&i) { remap.final_remap[t as usize] } else { NodeIdx(t | DROPPED) };
        }
    }
    remap
}

mod sweep;

/// The vector remap keeps and rewrites the pairs the scalar one does, on
/// either side, from any start and for any length.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[test]
fn vector_remap_matches_scalar() {
    use super::rewrite::{remap_avx2, remap_pairs};
    use crate::diagram::ChildPair;
    let Some(avx2) = crate::isa::Avx2::detect() else { return };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |m: u32| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % m
    };
    let width = 300u32;
    let final_remap: Vec<NodeIdx> =
        (0..width).map(|_| NodeIdx(next(width) | if next(3) == 0 { DROPPED } else { 0 })).collect();
    for len in 0..40 {
        for from in [0, 1, 3, len / 2] {
            if from > len {
                continue;
            }
            let seg: Vec<ChildPair> = (0..len).map(|_| ChildPair::new(NodeIdx(next(width)), NodeIdx(next(width)))).collect();
            for left in [true, false] {
                let (mut a, mut b) = (seg.clone(), seg.clone());
                let (ka, kb) = if left {
                    // Safety: `avx2` was detected.
                    (remap_pairs::<true>(&mut a, from, 0, &final_remap), unsafe {
                        remap_avx2::<true>(avx2, &mut b, from, &final_remap)
                    })
                } else {
                    (remap_pairs::<false>(&mut a, from, 0, &final_remap), unsafe {
                        remap_avx2::<false>(avx2, &mut b, from, &final_remap)
                    })
                };
                assert_eq!(ka, kb, "kept count, len {len} from {from} left {left}");
                assert_eq!(a[..ka], b[..kb], "kept pairs, len {len} from {from} left {left}");
            }
        }
    }
}
