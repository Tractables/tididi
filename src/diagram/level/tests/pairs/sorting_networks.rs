/// Verify all sorting networks (n=3..=8) sort every permutation correctly.
#[test]
fn sorting_networks_all_permutations() {
    fn test_network(n: usize) {
        // Generate all permutations of 0..n and verify sorted output.
        let mut perm: Vec<u32> = (0..n as u32).collect();
        let mut count = 0u64;
        loop {
            let mut s = perm.clone();
            sorting_network!(s, n);
            for i in 1..n {
                assert!(
                    s[i - 1] <= s[i],
                    "sorting_network n={} failed on {:?} → {:?} (s[{}]={} > s[{}]={})",
                    n, perm, s, i - 1, s[i - 1], i, s[i],
                );
            }
            count += 1;
            // Next permutation (Heap's algorithm is simpler but this is fine)
            if !next_permutation(&mut perm) {
                break;
            }
        }
        // Sanity: we tested n! permutations
        let expected: u64 = (1..=n as u64).product();
        assert_eq!(count, expected, "n={}: tested {} perms, expected {}", n, count, expected);
    }

    for n in 3..=8 {
        test_network(n);
    }
}

/// Generate the next lexicographic permutation. Returns false when wrapped.
fn next_permutation(a: &mut [u32]) -> bool {
    let n = a.len();
    if n < 2 { return false; }
    let mut i = n - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 { return false; }
    let mut j = n - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}
