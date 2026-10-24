use super::*;
use super::data::{rebuild_level, remap_and_reserve};

/// The remap a merge rewrites the parent through, where node `i` merges
/// into `targets[i]` (itself for a survivor) and the members in
/// `redirected` are content-equal redirects, which keep their parent pairs.
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
            let dropped = if redirected.contains(&i) { 0 } else { DROPPED };
            remap.final_remap[i] = NodeIdx(remap.final_remap[t as usize].0 | dropped);
        }
    }
    remap
}

mod sweep;
