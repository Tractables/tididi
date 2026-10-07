use rustc_hash::FxHashMap;

use super::*;
use crate::test_helpers::Lcg;

/// At most `n` parent values of `width` bits, sorted and distinct, each in
/// one of at most `classes` atoms, which are numbered by their smallest value.
fn parent_values(rng: &mut Lcg, width: usize, n: usize, classes: u64) -> Values<'static> {
    let mut data: Vec<u64> = (0..n).map(|_| rng.next_u64() & ((1u64 << width) - 1)).collect();
    data.sort_unstable();
    data.dedup();
    let mut number = FxHashMap::default();
    let atom: Vec<u32> = data
        .iter()
        .map(|_| {
            let next = number.len() as u32;
            *number.entry(rng.below(classes)).or_insert(next)
        })
        .collect();
    Values { words: 1, data: data.into(), atom, atoms: number.len() as u32 }
}

/// The atom count and the triples of a decomposition, whichever way it
/// lists them.
fn triples_of(split: &Decomposition) -> (usize, Vec<[u32; 3]>) {
    match split {
        Decomposition::Triples { atoms, triples } => (*atoms, triples.clone()),
        Decomposition::Grouped { ends, lows } => {
            let mut triples = Vec::new();
            let mut start = 0;
            for (high, &end) in ends.iter().enumerate() {
                assert!(start < end as usize, "every high atom has a pair");
                triples.extend(lows[start..end as usize].iter().map(|&low| [0, high as u32, low]));
                start = end as usize;
            }
            assert_eq!(start, lows.len());
            (1, triples)
        }
        Decomposition::Pairs { pairs } => {
            // The pairs name child atoms, as the nodes they will be.
            let triples: Vec<[u32; 3]> = pairs.iter().map(|pair| [0, pair.left.raw(), pair.right.raw()]).collect();
            assert!(triples.windows(2).all(|t| t[0] < t[1]), "the pairs ascend");
            (1, triples)
        }
        Decomposition::ByAtom { ends, pairs } => {
            let mut triples = Vec::new();
            let mut start = 0;
            for (atom, &end) in ends.iter().enumerate() {
                assert!(start < end as usize, "every atom has a pair");
                triples.extend(pairs[start..end as usize].iter().map(|&pair| [atom as u32, (pair >> 32) as u32, pair as u32]));
                start = end as usize;
            }
            assert_eq!(start, pairs.len());
            (ends.len(), triples)
        }
    }
}

#[test]
fn a_key_without_the_index_splits_as_one_with_it() {
    // A key too wide for the index finds a value through its high value's
    // run instead. Both must decompose as the split through materialized
    // parts does, whichever side the triples are read from.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x5eed_2026);
    let mut sides = [0usize; 2];
    for round in 0..400 {
        let (low_width, high_width) = (1 + rng.below(8) as usize, 1 + rng.below(8) as usize);
        let n = 2 + rng.below(300) as usize;
        let classes = 1 + rng.below(n as u64 / 2);
        let parent = parent_values(&mut rng, low_width + high_width, n, classes);
        if parent.atoms as usize == parent.len() {
            continue;
        }
        let high_runs = count_high_runs(&lim, &parent, low_width).unwrap();
        let indexed = KeyLayout::new(parent.len(), low_width, high_runs, parent.atoms, false).unwrap();
        assert!(indexed.indexed);
        let bare = KeyLayout { indexed: false, index_bits: 0, ..indexed };
        let mut s = Scratch::default();
        let want = split_parts(&lim, &mut s, &parent, (low_width, high_width), false).unwrap();
        for key in [indexed, bare] {
            let got = split_words(&lim, &mut s, &parent, low_width, key).unwrap();
            sides[usize::from(from_low(parent.len(), &s))] += 1;
            assert_eq!(triples_of(&got.0), triples_of(&want.0), "round {round}");
            for (got, want) in [(&got.1, &want.1), (&got.2, &want.2)] {
                assert_eq!((&got.data, &got.atom, got.atoms), (&want.data, &want.atom, want.atoms), "round {round}");
            }
        }
    }
    assert!(sides.iter().all(|&count| count > 0), "both emission sides are taken: {sides:?}");
}

/// The values of a relation whose high values fall into classes that share
/// their low parts, and whose low parts come in groups that always occur
/// together: both sides then have values that merge into one atom. Every
/// value lies in one atom, as at the node over every constrained variable.
fn merging_values(rng: &mut Lcg, low_width: usize, high_width: usize) -> Values<'static> {
    let groups: Vec<Vec<u64>> = (0..1 + rng.below(12))
        .map(|_| (0..1 + rng.below(3)).map(|_| rng.next_u64() & ((1u64 << low_width) - 1)).collect())
        .collect();
    let classes: Vec<Vec<usize>> = (0..1 + rng.below(6))
        .map(|_| (0..1 + rng.below(4)).map(|_| rng.below(groups.len() as u64) as usize).collect())
        .collect();
    let mut data = Vec::new();
    for _ in 0..1 + rng.below(40) {
        let high = rng.next_u64() & ((1u64 << high_width) - 1);
        for &group in &classes[rng.below(classes.len() as u64) as usize] {
            data.extend(groups[group].iter().map(|&low| high << low_width | low));
        }
    }
    data.sort_unstable();
    data.dedup();
    let atom = vec![0; data.len()];
    Values { words: 1, data: data.into(), atom, atoms: 1 }
}

thread_local! {
    /// Whether this thread hashes runs by the parity of their keys alone,
    /// so that most runs collide and only the confirming comparison tells
    /// them apart.
    pub(super) static WEAK_HASH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Many distinct values of a wide or a narrow low part, enough that the
/// radix sort runs.
fn many_values(rng: &mut Lcg, low_width: usize, high_width: usize) -> Values<'static> {
    let highs = 1 + rng.below(1 << high_width.min(12));
    let spread = 1 + rng.below(1 << low_width.min(20));
    let mut data = Vec::new();
    for _ in 0..highs {
        let high = rng.next_u64() & ((1u64 << high_width) - 1);
        let base = rng.next_u64() & ((1u64 << low_width) - 1);
        for _ in 0..1 + rng.below(8) {
            let low = (base + rng.below(spread)) & ((1u64 << low_width) - 1);
            data.push(high << low_width | low);
        }
    }
    data.sort_unstable();
    data.dedup();
    let atom = vec![0; data.len()];
    Values { words: 1, data: data.into(), atom, atoms: 1 }
}

/// `values` with parent atoms: each value's a function of its low part when
/// `by_low`, else drawn at random, among `classes`, and numbered by their
/// smallest value as atoms are.
fn with_atoms(mut values: Values<'static>, rng: &mut Lcg, low_width: usize, classes: u64, by_low: bool) -> Values<'static> {
    let (mut class_of_low, mut number) = (FxHashMap::default(), FxHashMap::default());
    let mut atom = Vec::new();
    for &value in values.data.iter() {
        let class = match by_low {
            true => *class_of_low.entry(value & ((1u64 << low_width) - 1)).or_insert_with(|| rng.below(classes)),
            false => rng.below(classes),
        };
        let next = number.len() as u32;
        atom.push(*number.entry(class).or_insert(next));
    }
    values.atom = atom;
    values.atoms = number.len() as u32;
    values
}

/// Split `parent` through materialized parts and through every other split
/// that takes it: the sorted single-atom split for one parent atom, and the
/// hashed split for a low part narrow enough to address, writing child
/// atoms and, for one parent atom, the node's pairs. Requires the same
/// atoms and triples of all, and returns the reference's children, low then
/// high, and which of the others ran.
fn check_split(lim: &Limits, parent: &Values, widths: (usize, usize), what: &str) -> (Values<'static>, Values<'static>, [bool; 2]) {
    let mut s = Scratch::default();
    let distinct = parent.atoms as usize == parent.len();
    let want = split_parts(lim, &mut s, parent, widths, distinct).unwrap();
    let mut copy = Values { words: 1, data: parent.data.clone(), atom: parent.atom.clone(), atoms: parent.atoms };
    let sorted = match parent.atoms == 1 && !distinct {
        true => split_single(lim, &mut s, &mut copy, widths).unwrap(),
        false => None,
    };
    let hashed = split_hashed(lim, &mut s, parent, widths.0).unwrap();
    s.direct = true;
    let direct = split_hashed(lim, &mut s, parent, widths.0).unwrap();
    s.direct = false;
    if parent.atoms == 1 {
        assert!(direct.as_ref().is_none_or(|got| matches!(got.0, Decomposition::Pairs { .. })), "{what}");
    }
    let ran = [sorted.is_some(), hashed.is_some()];
    for got in sorted.into_iter().chain(hashed).chain(direct) {
        assert_eq!(triples_of(&got.0), triples_of(&want.0), "{what}");
        for (got, want) in [(&got.1, &want.1), (&got.2, &want.2)] {
            assert_eq!((&got.data, &got.atom, got.atoms), (&want.data, &want.atom, want.atoms), "{what}");
        }
    }
    (want.1, want.2, ran)
}

/// A random parent of one of several shapes, with widths that the hashed
/// split addresses unless `wide`.
fn random_parent(rng: &mut Lcg, round: usize, wide: bool) -> (Values<'static>, (usize, usize)) {
    let (low_width, high_width) = (1 + rng.below(10) as usize, 1 + rng.below(12) as usize);
    let (values, low_width) = match round % 4 {
        0 => {
            let n = 2 + rng.below(400) as usize;
            (parent_values(rng, low_width + high_width, n, 1), low_width)
        }
        1 if wide => (many_values(rng, 4 * low_width, high_width), 4 * low_width),
        1 => (many_values(rng, low_width, high_width), low_width),
        _ => (merging_values(rng, low_width, high_width), low_width),
    };
    let values = match round / 4 % 3 {
        0 => values,
        1 => {
            let classes = 1 + rng.below(5);
            with_atoms(values, rng, low_width, classes, true)
        }
        _ => {
            let classes = 1 + rng.below(values.len() as u64);
            with_atoms(values, rng, low_width, classes, false)
        }
    };
    (values, (low_width, high_width))
}

#[test]
fn hashed_and_single_atom_splits_split_as_the_general_split_does() {
    // The single-atom split and the hashed split number hashed runs; the
    // split through materialized parts sorts and hashes every key. All must
    // give the same atoms and triples, when high values merge, when low
    // values merge, and when neither does, with one parent atom or several,
    // and whether the low part is narrow enough to address or not.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x51_2026);
    let (mut merges, mut ran) = ([0usize; 2], [0usize; 3]);
    for round in 0..1200 {
        let (parent, widths) = random_parent(&mut rng, round, round % 8 == 1);
        if parent.len() < 2 {
            continue;
        }
        let (low, high, took) = check_split(&lim, &parent, widths, &format!("round {round}"));
        merges[0] += usize::from(high.atoms as usize != high.len());
        merges[1] += usize::from(low.atoms as usize != low.len());
        ran[0] += usize::from(took[0]);
        ran[1] += usize::from(took[1] && parent.atoms == 1);
        ran[2] += usize::from(took[1] && parent.atoms > 1);
    }
    assert!(merges.iter().all(|&count| count > 50), "high and low values merge: {merges:?}");
    assert!(ran.iter().all(|&count| count > 100), "every split runs: {ran:?}");
}

#[test]
fn runs_whose_hashes_collide_split_as_the_general_split_does() {
    // Hashing runs by the parity of their keys makes most of them collide,
    // so every numbering confirms against the runs themselves, and the low
    // side lists each low value's completions to do so.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x0dd_2026);
    WEAK_HASH.with(|weak| weak.set(true));
    let mut ran = [0usize; 2];
    for round in 0..400 {
        let (parent, widths) = random_parent(&mut rng, round, round % 8 == 1);
        if parent.len() >= 2 {
            let (_, _, took) = check_split(&lim, &parent, widths, &format!("round {round}"));
            ran[0] += usize::from(took[0]);
            ran[1] += usize::from(took[1]);
        }
    }
    WEAK_HASH.with(|weak| weak.set(false));
    assert!(ran.iter().all(|&count| count > 50), "both splits run: {ran:?}");
}

/// Values of a few high values over a wide low part, most low parts under
/// one high value only and the rest under several: low values of one key
/// then share a few atoms, and the single-atom split reads its pairs off
/// them. Enough that the radix sort runs, when `many`.
fn few_highs(rng: &mut Lcg, low_width: usize, high_width: usize, many: bool) -> Values<'static> {
    let highs: Vec<u64> = (0..1 + rng.below(6)).map(|_| rng.next_u64() & ((1u64 << high_width) - 1)).collect();
    let shared: Vec<u64> = (0..rng.below(20)).map(|_| rng.next_u64() & ((1u64 << low_width) - 1)).collect();
    let per_high = if many { 400 + rng.below(3000) } else { 1 + rng.below(60) };
    let mut data = Vec::new();
    for &high in &highs {
        for _ in 0..per_high {
            data.push(high << low_width | (rng.next_u64() & ((1u64 << low_width) - 1)));
        }
        for &low in &shared {
            if rng.below(2) == 0 {
                data.push(high << low_width | low);
            }
        }
    }
    data.sort_unstable();
    data.dedup();
    let atom = vec![0; data.len()];
    Values { words: 1, data: data.into(), atom, atoms: 1 }
}

#[test]
fn a_single_atom_split_reads_its_pairs_off_either_side() {
    // Few high values over many low values of one key each: the low side's
    // first runs hold a key or two apiece, and the pairs are read off them.
    // Either side must give the pairs the general split gives, and both are
    // taken.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x10_51de);
    let mut sides = [0usize; 2];
    for round in 0..600 {
        let (low_width, high_width) = (20 + rng.below(30) as usize, 1 + rng.below(6) as usize);
        let (parent, widths) = match round % 3 {
            0 => random_parent(&mut rng, round, true),
            _ => (few_highs(&mut rng, low_width, high_width, round % 3 == 2), (low_width, high_width)),
        };
        if parent.len() < 2 || parent.atoms != 1 {
            continue;
        }
        let mut s = Scratch::default();
        let want = split_parts(&lim, &mut s, &parent, widths, false).unwrap();
        let mut copy = Values { words: 1, data: parent.data.clone(), atom: parent.atom.clone(), atoms: 1 };
        let Some(got) = split_single(&lim, &mut s, &mut copy, widths).unwrap() else { continue };
        sides[usize::from(single_from_low(parent.len(), &s).is_some())] += 1;
        assert_eq!(triples_of(&got.0), triples_of(&want.0), "round {round}");
        for (got, want) in [(&got.1, &want.1), (&got.2, &want.2)] {
            assert_eq!((&got.data, &got.atom, got.atoms), (&want.data, &want.atom, want.atoms), "round {round}");
        }
    }
    assert!(sides.iter().all(|&count| count > 50), "both sides are read: {sides:?}");
}

#[test]
fn a_key_too_wide_for_a_word_leaves_the_single_atom_split() {
    // Fifty low bits leave the high value's index thirteen: more high runs
    // than that indexes fall back, and fewer take the single-atom split.
    let lim = Limits::new();
    let (low_width, high_width) = (50, 14);
    for (runs, fits) in [(1usize << 13, true), ((1 << 13) + 1, false)] {
        let data: Vec<u64> = (0..runs as u64).flat_map(|h| [h << low_width | 1, h << low_width | 2]).collect();
        let mut parent = Values { words: 1, atom: vec![0; data.len()], data: data.into(), atoms: 1 };
        let mut s = Scratch::default();
        let got = split_single(&lim, &mut s, &mut parent, (low_width, high_width)).unwrap();
        assert_eq!(got.is_some(), fits, "{runs} runs");
    }
}

#[test]
fn a_leading_zero_key_changes_a_runs_hash() {
    // Runs that differ in a leading zero key hash apart, so those two tell
    // apart without reading the runs.
    assert_ne!(mix(RUN_SEED, 0), RUN_SEED);
    assert_ne!(mix(mix(RUN_SEED, 0), 5), mix(RUN_SEED, 5));
}

#[test]
fn hashed_runs_number_by_content_whatever_their_hashes() {
    // A table that numbers runs by hash must still tell apart runs whose
    // hashes collide, and must merge runs whose content is equal.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x0c01_11de);
    for round in 0..200 {
        let (runs, spread) = (1 + rng.below(300), 1 + rng.below(40));
        let contents: Vec<u64> = (0..runs).map(|_| rng.below(spread)).collect();
        let mut want = Vec::new();
        let mut firsts_want = Vec::new();
        let mut seen = FxHashMap::default();
        for (k, &content) in contents.iter().enumerate() {
            let next = seen.len() as u32;
            let id = *seen.entry(content).or_insert(next);
            if id == next {
                firsts_want.push(k as u32);
            }
            want.push(id);
        }
        let same = |a: u32, b: u32| contents[a as usize] == contents[b as usize];
        let degenerate: Vec<u64> = contents.iter().map(|&content| content % 3).collect();
        for hashes in [contents.clone(), degenerate, vec![7; contents.len()]] {
            let (mut slots, mut firsts) = (Vec::new(), Vec::new());
            let (ids, count) = number_hashed(&lim, &mut slots, &hashes, &mut firsts, same).unwrap();
            assert_eq!((&ids, count, &firsts), (&want, seen.len() as u32, &firsts_want), "round {round}");
        }
    }
}

#[test]
fn the_runs_between_two_parts_are_at_most_the_keys_and_the_parts() {
    assert_eq!(runs_between(3, 3, 10), 1);
    assert_eq!(runs_between(4, 9, 100), 6);
    assert_eq!(runs_between(0, 5, 3), 3);
    assert_eq!(runs_between(0, u64::MAX, 7), 7);
}

#[test]
fn counting_the_products_tells_merged_low_values_from_shared_atoms() {
    // One parent atom of the values (0, 0), (0, 1), (1, 0): high 0 pairs
    // with lows 0 and 1, high 1 with low 0 alone. Each child value is its
    // own atom, and the products of the pairs hold the three values once
    // each. Lows 0 and 1 merged, as a hash collision would merge them, make
    // the product of high 1 with the merged atom hold (1, 1) too.
    let lim = Limits::new();
    let exact = Decomposition::Grouped { ends: vec![2, 3], lows: vec![0, 1, 0] };
    assert!(covers_exactly(&lim, &exact, 3, (&[0, 1], 2), (&[0, 1], 2)).unwrap());
    let merged = Decomposition::Grouped { ends: vec![1, 2], lows: vec![0, 0] };
    assert!(!covers_exactly(&lim, &merged, 3, (&[0, 1], 2), (&[0, 0], 1)).unwrap());
    // Lows 0 and 1 under highs 0 and 1 alike share an atom, and so do the
    // highs: one pair whose product is the four values.
    let shared = Decomposition::Grouped { ends: vec![1], lows: vec![0] };
    assert!(covers_exactly(&lim, &shared, 4, (&[0, 0], 1), (&[0, 0], 1)).unwrap());
    // Two parent atoms, (0, 0) and (0, 1): lows 0 and 1 complete high 0 to
    // different atoms. Merged, the pair of each parent atom claims both.
    let atoms = Decomposition::ByAtom { ends: vec![1, 2], pairs: vec![0, 1] };
    assert!(covers_exactly(&lim, &atoms, 2, (&[0], 1), (&[0, 1], 2)).unwrap());
    let claimed = Decomposition::ByAtom { ends: vec![1, 2], pairs: vec![0, 0] };
    assert!(!covers_exactly(&lim, &claimed, 2, (&[0], 1), (&[0, 0], 1)).unwrap());
}

/// Values of many high values over a wide low part of few values, each
/// high value taking a few of them, as a wide block of few combinations
/// under a near-unique one gives. Every value lies in one atom.
fn sparse_values(rng: &mut Lcg, low_width: usize, high_width: usize) -> Values<'static> {
    let palette: Vec<u64> = (0..1 + rng.below(10)).map(|_| rng.next_u64() & ((1u64 << low_width) - 1)).collect();
    let mut data = Vec::new();
    for _ in 0..200 + rng.below(400) {
        let high = rng.next_u64() & ((1u64 << high_width) - 1);
        for _ in 0..1 + rng.below(palette.len() as u64) {
            data.push(high << low_width | palette[rng.below(palette.len() as u64) as usize]);
        }
    }
    data.sort_unstable();
    data.dedup();
    let atom = vec![0; data.len()];
    Values { words: 1, data: data.into(), atom, atoms: 1 }
}

#[test]
fn a_wide_low_part_of_few_values_splits_by_rank_as_the_general_split_does() {
    // Ranks keep the values ascending and stand for their low parts alone,
    // so the split through them is the values' own: with one parent atom
    // (its pairs written as the node's) or several, over the caller's
    // values or owned ones, and declined where the low parts are many.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x5a_2026);
    let (mut ran, mut declined) = (0, 0);
    for round in 0..300 {
        let (low_width, high_width) = (13 + rng.below(36) as usize, 9 + rng.below(8) as usize);
        let values = match round % 3 {
            0 => sparse_values(&mut rng, low_width, high_width),
            1 => {
                let values = sparse_values(&mut rng, low_width, high_width);
                let classes = 1 + rng.below(5);
                with_atoms(values, &mut rng, low_width, classes, round % 2 == 0)
            }
            _ => many_values(&mut rng, low_width, high_width),
        };
        let distinct = values.atoms as usize == values.len();
        if distinct || values.len() < 2 {
            continue;
        }
        let mut s = Scratch::default();
        let want = split_parts(&lim, &mut s, &values, (low_width, high_width), distinct).unwrap();
        let data = match round % 2 {
            0 => Cow::Borrowed(&values.data[..]),
            _ => Cow::Owned(values.data.to_vec()),
        };
        let mut copy = Values { words: 1, data, atom: values.atom.clone(), atoms: values.atoms };
        (s.sparse, s.direct) = (true, values.atoms == 1);
        match split_sparse(&lim, &mut s, &mut copy, low_width).unwrap() {
            None => declined += 1,
            Some(got) => {
                ran += 1;
                assert_eq!(triples_of(&got.0), triples_of(&want.0), "round {round}");
                for (got, want) in [(&got.1, &want.1), (&got.2, &want.2)] {
                    assert_eq!((&got.data, &got.atom, got.atoms), (&want.data, &want.atom, want.atoms), "round {round}");
                }
            }
        }
    }
    assert!(ran > 100 && declined > 30, "ran {ran}, declined {declined}");
}

#[test]
fn a_sample_repeats_a_low_part_where_the_low_parts_are_few() {
    // Few low parts under many high values repeat in any even sample as
    // often as that few would; low parts all distinct never do, whatever
    // the sample's size, and many with repeats too rarely.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x5e_2026);
    let mut table = Vec::new();
    let mask = (1u64 << 40) - 1;
    for count in [2usize, 3, 100, SPARSE_SAMPLE] {
        let distinct: Vec<u64> = (0..50_000u64).map(|k| k << 40 | ((k * 0x9e37_79b9) & mask)).collect();
        let most = 50_000 / SPARSE_LOWS;
        assert!(!sample_repeats(&lim, &mut table, &distinct, mask, count, most).unwrap(), "{count}");
        let palette: Vec<u64> = (0..1 + count as u64 / 4).map(|_| rng.next_u64() & mask).collect();
        let few: Vec<u64> = (0..50_000u64).map(|k| k << 40 | palette[(k % palette.len() as u64) as usize]).collect();
        assert!(sample_repeats(&lim, &mut table, &few, mask, count, most).unwrap(), "{count}");
        // Ten times as many low parts as `most`, drawn at random: a sample
        // repeats some, a tenth as often as `most` would make it.
        let many: Vec<u64> = (0..50_000u64).map(|k| k << 40 | rng.below(10 * most as u64)).collect();
        if count == SPARSE_SAMPLE {
            assert!(!sample_repeats(&lim, &mut table, &many, mask, count, most).unwrap(), "{count}");
        }
    }
}

#[test]
fn ranks_found_through_a_table_that_grows_split_as_the_general_split_does() {
    // Enough distinct low parts that the table of `split_sparse` grows past
    // its first size, where they appear early and where they appear late,
    // and one more than it takes, which it declines.
    let lim = Limits::new();
    let mut rng = Lcg::new(0x9a_2026);
    let low_width = 30;
    for (lows, late, take) in [(3_000u64, false, true), (4_000, true, true), (5_001, false, false)] {
        let palette: Vec<u64> = (0..lows).map(|_| rng.next_u64() & ((1u64 << low_width) - 1)).collect();
        let n = 16 * 5_000;
        // Values of high value `h`, one low part each, a few palette entries
        // early and the whole palette once they appear late.
        let mut data: Vec<u64> = (0..n as u64)
            .map(|h| {
                let pick = match late && h < n as u64 / 2 {
                    true => rng.below(8),
                    false => rng.below(lows),
                };
                h << low_width | palette[pick as usize]
            })
            .collect();
        // Every low part at least once, so the distinct count is the
        // palette's: first, or last where they appear late.
        let from = if late { n - palette.len() } else { 0 };
        for (k, &low) in palette.iter().enumerate() {
            data[from + k] = ((from + k) as u64) << low_width | low;
        }
        data.sort_unstable();
        let values = Values { words: 1, data: data.into(), atom: vec![0; n], atoms: 1 };
        let mut s = Scratch::default();
        let want = split_parts(&lim, &mut s, &values, (low_width, 17), false).unwrap();
        let mut copy = Values { words: 1, data: Cow::Borrowed(&values.data[..]), atom: values.atom.clone(), atoms: 1 };
        (s.sparse, s.direct) = (true, true);
        let got = split_sparse(&lim, &mut s, &mut copy, low_width).unwrap();
        assert_eq!(got.is_some(), take, "{lows} low parts, late {late}");
        if let Some(got) = got {
            assert!(s.exact.len() > 1 << GROWING_BITS, "the table grew");
            assert_eq!(triples_of(&got.0), triples_of(&want.0), "{lows} low parts");
            for (got, want) in [(&got.1, &want.1), (&got.2, &want.2)] {
                assert_eq!((&got.data, &got.atom, got.atoms), (&want.data, &want.atom, want.atoms), "{lows} low parts");
            }
        }
    }
}
