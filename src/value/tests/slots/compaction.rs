use super::*;

#[test]
fn compaction_merges_values_and_rekeys_only_referenced_overflows() {
    let sentinel = u128::MAX;
    let exact_max = BigUint::from(sentinel);
    let fast = vec![sentinel, 7, sentinel, 7, sentinel, 0];
    let overflow: CountOverflow = [
        (0, &exact_max + 100u32),
        (2, exact_max.clone()),
        (4, exact_max.clone()),
    ].into_iter().collect();
    for kept in [vec![1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4, 5], vec![]] {
        let mut counts = fast.clone();
        let mut big = Some(overflow.clone());
        let mut remap = vec![u32::MAX; counts.len()];
        let (new_len, merged) = compact_count_slots(
            &mut counts, &mut big, kept.iter().copied(), &mut remap,
        );
        counts.truncate(new_len);
        assert_eq!(merged, if kept.is_empty() { 0 } else { 2 });
        assert_eq!(new_len, kept.len() - merged);
        for (old, &new) in remap.iter().enumerate() {
            if kept.contains(&old) {
                let actual = count_key_at(&counts, big.as_ref(), new as usize);
                assert_eq!(actual, count_key_at(&fast, Some(&overflow), old));
            } else {
                assert_eq!(new, u32::MAX);
            }
        }
        let surviving_big = counts.iter().filter(|&&c| c == sentinel).count();
        assert_eq!(big.as_ref().unwrap().len(), surviving_big);
    }
}
