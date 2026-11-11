use super::*;
use crate::Engine;
use crate::vtree::rng::Lcg;

/// A slice as a context scatter writes it: parents ascending in the high
/// half, each parent's entries drawn from `siblings` values, in draw order.
/// Runs reach `max_run` entries, so some exceed any small sort's bound.
fn context_slice(rng: &mut Lcg, parents: u64, max_run: u64, siblings: u64) -> Vec<u64> {
    let mut slice = Vec::new();
    let mut parent = rng.below(4);
    for _ in 0..parents {
        for _ in 0..rng.below(max_run + 1) {
            slice.push((parent << 32) | rng.below(siblings));
        }
        parent += 1 + rng.below(3);
    }
    slice
}

/// One parent's run as a marginal level's pair lists leave it: ascending
/// blocks of siblings, each block written `copies` times over.
fn repeated_blocks(rng: &mut Lcg, parent: u64, blocks: u64, copies: u64) -> Vec<u64> {
    let mut run = Vec::new();
    let mut sibling = 0;
    for _ in 0..blocks {
        let block: Vec<u64> = (0..1 + rng.below(12))
            .map(|_| {
                sibling += rng.below(40);
                (parent << 32) | sibling
            })
            .collect();
        for _ in 0..copies {
            run.extend_from_slice(&block);
        }
    }
    run
}

/// Sort `slice` with the shared `tables`, and check it against a full sort.
fn assert_sorts(eng: &Engine, tables: &mut groups::SortCounts, mut slice: Vec<u64>) {
    let mut want = slice.clone();
    want.sort_unstable();
    groups::sort_signature_slice(eng.limits(), &mut slice, tables).expect("no limit is set");
    assert_eq!(slice, want);
}

/// Every slice comes out as a full sort leaves it: slices out of order only
/// within one parent's run, short and long runs, runs of few values (sorted
/// by counting), runs over many values (by the general sort), slices whose
/// parents descend somewhere, and slices in no order at all. One set of
/// tables serves every slice, so a count left behind would show in a later
/// one.
#[test]
fn signature_slices_sort_as_a_full_sort_does() {
    let eng = Engine::new();
    let tables = &mut groups::SortCounts::default();
    let mut rng = Lcg::new(0x5157_2026);
    let rounds = if cfg!(miri) { 6 } else { 2000 };
    for round in 0..rounds {
        let max_run = [1, 2, 3, 8, 40, 300][round % 6];
        let siblings = [2, 5, 1 << 20, 60][round % 4];
        let parents = 1 + rng.below(30);
        let slice = context_slice(&mut rng, parents, max_run, siblings);
        assert_sorts(&eng, tables, slice.clone());

        // Parents out of order: two entries of different runs exchanged.
        if slice.len() >= 2 {
            let mut swapped = slice.clone();
            let i = rng.below(slice.len() as u64) as usize;
            let j = rng.below(slice.len() as u64) as usize;
            swapped.swap(i, j);
            assert_sorts(&eng, tables, swapped);
        }

        let mut runs = Vec::new();
        for parent in 0..1 + rng.below(4) {
            let (blocks, copies) = (1 + rng.below(20), 1 + rng.below(6));
            runs.extend(repeated_blocks(&mut rng, parent, blocks, copies));
        }
        assert_sorts(&eng, tables, runs);

        // No order at all, over a grid of 64 cells.
        let len = rng.below(200);
        assert_sorts(&eng, tables, (0..len).map(|_| (rng.below(8) << 32) | rng.below(8)).collect());
    }
    // Entries either side of a high half boundary, sorted whole.
    let around: Vec<u64> = (0..300).map(|_| (1 << 32) - 100 + rng.below(200)).collect();
    assert_sorts(&eng, tables, around);
    assert_sorts(&eng, tables, Vec::new());
    assert_sorts(&eng, tables, vec![7]);
    assert_sorts(&eng, tables, vec![(1 << 32) | 3, (1 << 32) | 2]);
    assert_sorts(&eng, tables, vec![(2 << 32) | 1, (1 << 32) | 2]);
    assert_sorts(&eng, tables, vec![u64::MAX, 0, u64::MAX, 0]);
}
