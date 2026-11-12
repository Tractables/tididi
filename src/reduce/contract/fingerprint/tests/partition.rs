use super::*;
use crate::Engine;
use std::collections::HashMap;

/// Entries listed outright, `(node, entry)`.
struct Listed(Vec<(u32, u64)>);

impl TwinEntries for Listed {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for &(node, entry) in &self.0 {
            f(node, entry);
        }
    }
}

/// The groups of nodes with equal entry multisets, members ascending, groups
/// by their lowest member.
fn listed_groups(entries: &Listed, width: usize) -> Vec<Vec<u32>> {
    let mut of: Vec<Vec<u64>> = vec![Vec::new(); width];
    for &(node, entry) in &entries.0 {
        of[node as usize].push(entry);
    }
    let mut by: HashMap<Vec<u64>, Vec<u32>> = HashMap::new();
    for (node, mut e) in of.into_iter().enumerate() {
        e.sort_unstable();
        by.entry(e).or_default().push(node as u32);
    }
    let mut groups: Vec<Vec<u32>> = by.into_values().filter(|g| g.len() > 1).collect();
    groups.sort();
    groups
}

/// A level wide enough for its fingerprints to be partitioned groups as a
/// listing of its nodes' entries does: planted groups found whole, every
/// other node alone, and a level with no twin reported as such.
#[test]
#[cfg_attr(miri, ignore = "reaches no unsafe code, at a width Miri takes over an hour to interpret")]
fn a_partitioned_wide_level_groups_as_its_entries_do() {
    let eng = Engine::new();
    let mut scratch = ContractScratch::default();
    let width = PARTITION_MIN_WIDTH + 4_321;
    for planted in [0u32, 1, 7, 300] {
        let mut entries = Listed(Vec::new());
        for node in 0..width as u32 {
            let k = node % 1_000;
            if k < planted {
                // Groups of nodes a thousand apart share their entries, one
                // of them twice.
                let g = u64::from(k);
                entries.0.extend([(node, pack(1 << 20 | k, 5)), (node, g << 40 | 3), (node, g << 40 | 3)]);
            } else {
                entries.0.extend([(node, pack(node, 1)), (node, pack(node / 2, 2))]);
            }
        }
        let want = listed_groups(&entries, width);
        let found = group_twins_by_entries(&eng, &entries, width, &mut scratch).expect("group_twins_by_entries");
        let mut got: Vec<Vec<u32>> = Vec::new();
        if found {
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
        }
        assert_eq!(found, planted > 0, "{planted} groups planted");
        assert_eq!(got, want, "{planted} groups planted");
    }
}

fn listed_found(scratch: &ContractScratch, found: bool) -> Vec<Vec<u32>> {
    let mut got: Vec<Vec<u32>> = Vec::new();
    if found {
        for (g, &start) in scratch.group_starts.iter().enumerate() {
            let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
            got.push(scratch.flat_groups[start as usize..end].to_vec());
        }
    }
    got
}

/// Nodes whose entries come parent by parent, each parent's siblings in an
/// order that differs between twins and that repeats a sibling, group as a
/// listing of their entries does: twins found whole, near-twins (one
/// sibling or one parent apart) apart, on a narrow, a partitioned and a
/// two-node level, with runs short enough for the general sort and long
/// enough for the counting sort.
#[test]
#[cfg_attr(miri, ignore = "reaches no unsafe code, at a width Miri takes over an hour to interpret")]
fn entries_out_of_order_within_a_parent_group_as_listed() {
    let eng = Engine::new();
    let mut scratch = ContractScratch::default();
    let shapes = [(2usize, 1u32), (3, 1), (1_000, 1), (PARTITION_MIN_WIDTH + 77, 1), (2, 12), (3, 12), (1_000, 12)];
    for (width, spread) in shapes {
        let mut by_node: Vec<Vec<(u32, u32)>> = vec![Vec::new(); width];
        for node in 0..width as u32 {
            let class = node % 5;
            // Parents 1, 4 and 9, each with up to `6 * spread` siblings; twins
            // share the class, rotated so their order within a parent differs.
            for parent in [1u32, 4, 9] {
                let mut sibs: Vec<u32> = (0..(2 + class) * spread).map(|s| 10 * s + parent).collect();
                if class == 3 {
                    sibs.push(sibs[0]);
                }
                let r = node as usize % sibs.len();
                sibs.rotate_left(r);
                if node % 2 == 1 {
                    sibs.reverse();
                }
                by_node[node as usize].extend(sibs.into_iter().map(|s| (parent, s)));
            }
            // Class 4 splits on its node's parity: one sibling apart.
            if class == 4 {
                by_node[node as usize].push((12, 1 + node % 2));
            }
        }
        // Parent by parent, as the scatter walks them.
        let mut listed = Vec::new();
        for parent in 0..16 {
            for (node, es) in by_node.iter().enumerate() {
                for &(p, sib) in es {
                    if p == parent {
                        listed.push((node as u32, pack(p, sib)));
                    }
                }
            }
        }
        let entries = Listed(listed);
        let want = listed_groups(&entries, width);
        let found = group_twins_by_entries(&eng, &entries, width, &mut scratch).expect("group_twins_by_entries");
        assert_eq!(listed_found(&scratch, found), want, "width {width}, spread {spread}");
        assert_eq!(found, !want.is_empty(), "width {width}, spread {spread}");
    }
}
