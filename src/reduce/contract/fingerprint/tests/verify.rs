use super::*;
use crate::Engine;

/// Context entries listed outright, `(node, (parent node, sibling))` in
/// parent order.
struct Listed(Vec<(u32, (u32, u32))>);

impl TwinEntries for Listed {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for &(node, (p, s)) in &self.0 {
            f(node, pack(p, s));
        }
    }

    fn owner_keyed(&self) -> bool {
        true
    }
}

/// Verify `nominated`, each a group of nodes taken as sharing a fingerprint
/// (every node in no group stands alone), and return the groups written.
fn verify(entries: &Listed, width: usize, nominated: &[&[u32]]) -> Option<Vec<Vec<u32>>> {
    let eng = Engine::new();
    let mut scratch = ContractScratch {
        reps: (0..width as u32).collect(),
        is_candidate: vec![false; width],
        ..ContractScratch::default()
    };
    for group in nominated {
        for &n in *group {
            scratch.reps[n as usize] = group[0];
            scratch.is_candidate[n as usize] = true;
        }
    }
    let found = groups::verify_fingerprint_groups(&eng, entries, width, false, &mut scratch).expect("verify")?;
    let mut got = Vec::new();
    for (g, &start) in scratch.group_starts.iter().enumerate() {
        let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
        got.push(scratch.flat_groups[start as usize..end].to_vec());
    }
    assert_eq!(found, !got.is_empty());
    Some(got)
}

/// Nominated groups whose members share every context are written as twin
/// groups, by lowest member, members ascending.
#[test]
fn groups_with_equal_contexts_are_verified() {
    // Nodes 0, 2 and 4 share (0, 7) and (1, 3); nodes 1 and 3 share (0, 5).
    let entries = Listed(vec![
        (0, (0, 7)), (2, (0, 7)), (1, (0, 5)), (4, (0, 7)), (3, (0, 5)),
        (4, (1, 3)), (0, (1, 3)), (2, (1, 3)),
    ]);
    assert_eq!(verify(&entries, 5, &[&[0, 2, 4], &[1, 3]]), Some(vec![vec![0, 2, 4], vec![1, 3]]));
}

/// A nominated group is refused where some member misses a context another
/// member has, in one parent node or across two, or where a member no pair
/// names is grouped with named ones.
#[test]
fn groups_with_different_contexts_are_refused() {
    // Node 1 lacks (1, 3).
    let missing = Listed(vec![(0, (0, 7)), (1, (0, 7)), (0, (1, 3))]);
    assert_eq!(verify(&missing, 2, &[&[0, 1]]), None);
    // Nodes 0 and 1 each have one context, in different parent nodes.
    let apart = Listed(vec![(0, (0, 7)), (1, (1, 7))]);
    assert_eq!(verify(&apart, 2, &[&[0, 1]]), None);
    // Nodes 0 and 1 share (0, 7), node 2 does not.
    let partial = Listed(vec![(0, (0, 7)), (1, (0, 7)), (2, (0, 6))]);
    assert_eq!(verify(&partial, 3, &[&[0, 1, 2]]), None);
    // Node 2 is named by no pair.
    let unnamed = Listed(vec![(0, (0, 7)), (1, (0, 7))]);
    assert_eq!(verify(&unnamed, 3, &[&[0, 1, 2]]), None);
    // Equal counts of contexts across the two siblings do not stand in for
    // equal contexts: 0 has (0, 5), (0, 6); 1 has (0, 5), (0, 7).
    let swapped = Listed(vec![(0, (0, 5)), (1, (0, 5)), (0, (0, 6)), (1, (0, 7))]);
    assert_eq!(verify(&swapped, 2, &[&[0, 1]]), None);
}

/// Nodes no pair names have equal, empty contexts.
#[test]
fn unnamed_nodes_are_twins() {
    let entries = Listed(vec![(1, (0, 7))]);
    assert_eq!(verify(&entries, 4, &[&[0, 2, 3]]), Some(vec![vec![0, 2, 3]]));
}
