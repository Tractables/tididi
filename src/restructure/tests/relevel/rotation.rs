use super::*;
use crate::Engine;
use crate::vtree::{RotationKind, Vtree};
use crate::test_helpers::{rotate_left, rotate_right};
use crate::restructure::relevel::restructure_inner_search;



/// The size-objective descent of `Engine::rotation_search` under
/// [`MinimizePairs`](crate::restructure::search::local::MinimizePairs).
fn size_descent(tdd: &mut Tdd) {
    use crate::restructure::search::local::MinimizePairs;
    use crate::restructure::search::RotationSearchConfig;
    Engine::new()
        .rotation_search(tdd, &mut MinimizePairs, &RotationSearchConfig::default())
        .expect("an unarmed engine stops nothing");
}
use std::sync::Arc;
use crate::test_helpers::{assert_canonical, compile_clauses};

#[test]
fn left_rotation_preserves_model_count() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![-3, 4]]);
    let mc_before = tdd.model_count().unwrap();

    let mut vt = (*vtree).clone();
    let root = vt.root();
    let info = rotate_left(&mut vt, root).unwrap();
    tdd.vtree = Arc::new(vt);
    restructure_inner_search(&mut tdd, &info, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
    tdd.minimize().unwrap();
    assert_canonical(&tdd);
    assert_eq!(mc_before, tdd.model_count().unwrap());
}

/// A left rotation and the right rotation that undoes it both hold the count,
/// so the count after the round trip is the count before it — not merely the
/// count reached midway.
#[test]
fn right_rotation_preserves_model_count() {
    // Right rotation is applicable only after a left rotation has been made,
    // so start from a linear vtree and make one.
    let vtree = Arc::new(Vtree::linear(4));
    let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![-3, 4]]);
    let mc_before = tdd.model_count().unwrap();

    let mut vt = (*vtree).clone();
    let root = vt.root();
    let left_idx = rotate_left(&mut vt, root).unwrap();
    tdd.vtree = Arc::new(vt.clone());
    restructure_inner_search(&mut tdd, &left_idx, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
    tdd.minimize().unwrap();
    assert_eq!(mc_before, tdd.model_count().unwrap(), "the left rotation moved the count");

    let right_idx = rotate_right(&mut vt, root).unwrap();
    tdd.vtree = Arc::new(vt);
    restructure_inner_search(&mut tdd, &right_idx, RotationKind::Right, &mut RestructureScratch::default(), usize::MAX);
    tdd.minimize().unwrap();
    assert_canonical(&tdd);
    assert_eq!(mc_before, tdd.model_count().unwrap(), "the round trip moved the count");
}

#[test]
fn left_rotation_unsat_stays_unsat() {
    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = compile_clauses(&vtree, &[vec![1], vec![-1]]);
    let mc_before = tdd.model_count().unwrap();
    assert_eq!(mc_before, num_bigint::BigUint::ZERO);
    let mut vt = (*vtree).clone();
    let root = vt.root();
    if let Some(info) = rotate_left(&mut vt, root) {
        tdd.vtree = Arc::new(vt);
        restructure_inner_search(&mut tdd, &info, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
        tdd.minimize().unwrap();
        assert_canonical(&tdd);
        assert_eq!(tdd.model_count().unwrap(), num_bigint::BigUint::ZERO);
    }
}

/// Count safety of the parent-of-marginal rotation. A vtree rotation is a pure
/// variable reorder, so `model_count` must be invariant whether or not a
/// regrouped subtree is collapsed to a marginal level — the property the
/// structural cascade test above does not check. The
/// closure-after-rotation must leave the count exactly equal to the
/// marginalization-first count (which equals the Boolean count).
#[test]
fn parent_of_marginal_rotation_preserves_model_count() {
    let eng = Engine::new();
    use crate::marginal::{marginalize_batch, marginalize_closure};

    let vt_str = "vtree 9\n\
        L 0 1\nL 1 2\nI 2 0 1\n\
        L 3 3\nL 4 4\nI 5 3 4\n\
        L 6 5\nI 7 5 6\n\
        I 8 2 7\n";
    let vtree = Arc::new(Vtree::from_text(vt_str).unwrap());
    let root = vtree.root();
    let (a_idx, w_idx) = vtree.children(root);
    let (b_idx, _c_idx) = vtree.children(w_idx);

    let mut tdd = compile_clauses(
        &vtree,
        &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![1, -5]],
    );
    let mc_bool = tdd.model_count().unwrap();

    // marginalization-first: collapse A and B, then rotate the parent (gc=1).
    let mut targets = vec![a_idx, b_idx];
    targets.sort_by_key(|t| t.idx());
    marginalize_batch(&eng, &mut tdd, &targets, &vtree).expect("no wall is installed in a test");
    let mc_marginal = tdd.model_count().unwrap();
    assert_eq!(mc_bool, mc_marginal, "marginalize_levels must preserve count");

    let mut vt = (*vtree).clone();
    let info = rotate_left(&mut vt, root).unwrap();
    let new_vtree = Arc::new(vt);
    tdd.vtree = Arc::clone(&new_vtree);
    restructure_inner_search(&mut tdd, &info, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
    // Close clusters (the production path runs marginalize_closure after search).
    marginalize_closure(&eng, &mut tdd).expect("no wall is installed in a test");
    tdd.minimize().unwrap();
    assert_canonical(&tdd);
    let mc_after = tdd.model_count().unwrap();
    assert_eq!(
        mc_marginal, mc_after,
        "gc=1 (parent-of-marginal) rotation must preserve model_count"
    );
}

/// MEMORY-RECLAIM regression. When a cluster-rotation marginalizes a new
/// parent over two already-marginal children, those children become interior
/// to a marginal region: their count stores are subsumed by the parent's
/// aggregate and unreachable from the root. The reclaim-on-marginalize_levels
/// (`free_subsumed_marginal_children`, fired when the new parent marginalizes
/// inside `marginalize_closure`) must FREE them (width → 0) while preserving
/// #F exactly. Fails without that free (children keep their stores resident).
#[test]
fn cluster_rotation_frees_subsumed_child_stores() {
    let eng = Engine::new();
    use crate::marginal::{marginalize_batch, marginalize_closure};

    // Same shape as the cascade-up test: v=(A,w), w=(B,C); A,B internal.
    let vt_str = "vtree 9\n\
        L 0 1\nL 1 2\nI 2 0 1\n\
        L 3 3\nL 4 4\nI 5 3 4\n\
        L 6 5\nI 7 5 6\n\
        I 8 2 7\n";
    let vtree = Arc::new(Vtree::from_text(vt_str).unwrap());
    let root = vtree.root();
    let (a_idx, w_idx) = vtree.children(root);
    let (b_idx, _c_idx) = vtree.children(w_idx);

    let mut tdd = compile_clauses(
        &vtree,
        &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![1, -5]],
    );
    let mc_bool = tdd.model_count().unwrap();

    // Collapse A and B — each becomes the top of its own marginal region
    // (parent still structural) with a non-empty count store.
    let mut targets = vec![a_idx, b_idx];
    targets.sort_by_key(|t| t.idx());
    marginalize_batch(&eng, &mut tdd, &targets, &vtree).expect("no wall is installed in a test");
    assert!(
        tdd.levels[a_idx.idx()].slot_count() > 0 && tdd.levels[b_idx.idx()].slot_count() > 0,
        "test setup: A and B carry count stores before clustering"
    );

    // Cluster A,B under a new parent and close — the new parent marginalizes.
    let mut vt = (*vtree).clone();
    let info = rotate_left(&mut vt, root).unwrap();
    let new_vtree = Arc::new(vt);
    tdd.vtree = Arc::clone(&new_vtree);
    restructure_inner_search(&mut tdd, &info, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
    marginalize_closure(&eng, &mut tdd).expect("no wall is installed in a test");

    assert!(
        tdd.levels[info.w_idx.idx()].is_marginal(),
        "new parent over two marginal children must marginalize_levels"
    );
    // A and B are now interior to the new parent's marginal region — freed.
    assert!(
        tdd.levels[a_idx.idx()].is_marginal() && tdd.levels[a_idx.idx()].slot_count() == 0,
        "subsumed child A must be freed (width 0, still marginal)"
    );
    assert!(
        tdd.levels[b_idx.idx()].is_marginal() && tdd.levels[b_idx.idx()].slot_count() == 0,
        "subsumed child B must be freed (width 0, still marginal)"
    );
    // #F is unchanged by the reclaim.
    assert_eq!(
        mc_bool,
        tdd.model_count().unwrap(),
        "freeing subsumed interior stores must preserve model_count"
    );
}

/// A sweep of parent-of-marginal rotations preserves the count, not just a
/// single one (the test above). Generate small random CNFs, marginalize a
/// random subtree, run the public greedy rotation search + closure, assert
/// model_count is preserved. Parent-of-marginal rotations commit
/// unconditionally, so the sweep exercises that path under a plain
/// `cargo test`. Driven by the library's
/// [`Engine::rotation_search`](crate::Engine::rotation_search) under the
/// size objective (single source of truth for the size-descent sweep).
#[test]
fn fuzz_search_preserves_marginal_count() {
    let eng = Engine::new();
    use crate::marginal::{marginalize_batch, marginalize_closure};
    use crate::vtree::VtreeIdx;

    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut first_fail: Option<String> = None;
    for seed in 0..600u64 {
        let num_vars = 6 + (rng() % 4) as u32; // 6..=9
        let nc = num_vars as usize + (rng() % num_vars as u64) as usize + 2;
        let mut clauses: Vec<Vec<i32>> = Vec::new();
        for _ in 0..nc {
            let k = 2 + (rng() % 2) as usize; // 2 or 3 literals
            let mut literals = Vec::new();
            for _ in 0..k {
                let v = 1 + (rng() % num_vars as u64) as i32;
                let s = if rng() & 1 == 0 { 1 } else { -1 };
                // A real clause has each variable at most once (Clause::new
                // debug-asserts this); skip a var already in this clause.
                if literals.iter().any(|l: &i32| l.unsigned_abs() == v as u32) {
                    continue;
                }
                literals.push(v * s);
            }
            if literals.is_empty() {
                continue;
            }
            clauses.push(literals);
        }
        let vtree = Arc::new(Vtree::balanced(num_vars));
        let mut tdd = compile_clauses(&vtree, &clauses);

        let internals: Vec<VtreeIdx> = vtree
            .internal_bottomup()
            .map(|(t, _, _)| t)
            .filter(|t| *t != vtree.root())
            .collect();
        if internals.is_empty() {
            continue;
        }
        let tgt = internals[(rng() as usize) % internals.len()];
        marginalize_batch(&eng, &mut tdd, &[tgt], &vtree).expect("no wall is installed in a test");
        let mc_before = tdd.model_count().unwrap();

        size_descent(&mut tdd);
        marginalize_closure(&eng, &mut tdd).expect("no wall is installed in a test");
        let mc_after = tdd.model_count().unwrap();

        if mc_before != mc_after {
            first_fail = Some(format!(
                "seed={seed} vars={num_vars} tgt={} clauses={clauses:?} before={mc_before} after={mc_after}",
                tgt.idx()
            ));
            break;
        }
    }
    assert!(
        first_fail.is_none(),
        "search must preserve marginalized count through parent-of-marginal rotations: {}",
        first_fail.unwrap()
    );
}

/// The deterministic minimal case the fuzz above found: marginalize a subtree,
/// then run a greedy rotation sweep and the closure. The count must survive
/// both. Parent-of-marginal rotations commit unconditionally under a plain
/// `cargo test`, so this runs the same path the fuzz does, without the search
/// for a witness.
#[test]
fn gc1_sweep_undercount_repro() {
    let eng = Engine::new();
    use crate::marginal::{marginalize_batch, marginalize_closure};
    use crate::vtree::VtreeIdx;

    // Two clauses the fuzz drew — [6,3,-6] and [-4,4,2] — are tautologies (a
    // var and its negation), always-true and thus semantically no-ops; dropped
    // here to satisfy the Clause::new "each variable at most once"
    // precondition without changing #F.
    let clauses = vec![
        vec![-5, 8, -6], vec![3, -6, -5], vec![-2, -6, -7],
        vec![-4, -2], vec![4, -7], vec![-4, 6, 3],
        vec![8, 1, 2], vec![-6, -4],
    ];
    let vtree = Arc::new(Vtree::balanced(8));
    let mut tdd = compile_clauses(&vtree, &clauses);
    marginalize_batch(&eng, &mut tdd, &[VtreeIdx(9)], &vtree).expect("no wall is installed in a test");
    // marginalize_batch collapses distinct equal-count children to the same
    // inline ref, manufacturing twin nodes (n0≡n2) that are count-correct but
    // non-canonical. A rotation that regroups them by content would collapse
    // the multiplicity (undercount) unless it keeps the multiset.
    // Preserving the count through the sweep and the closure below therefore
    // rests on the regrouping keeping the multiset rather than the content set.
    let mc_marginal = tdd.model_count().unwrap();

    size_descent(&mut tdd);
    let mc_search = tdd.model_count().unwrap();

    marginalize_closure(&eng, &mut tdd).expect("no wall is installed in a test");
    let mc_closure = tdd.model_count().unwrap();

    assert_eq!(
        mc_marginal, mc_search,
        "the search SWEEP changed the marginalized count"
    );
    assert_eq!(
        mc_marginal, mc_closure,
        "the closure changed the marginalized count"
    );
}

// ─── Rotation Locality ─────────────────────────────────────────────────
//
// Under canonicity, `restructure_inner_search` followed by the
// locality check and `clear_worklists` mutates only `levels[v_idx]` and
// `levels[w_idx]`. Every other level is bit-for-bit identical pre and
// post. These tests assert that property directly on snapshotted level
// contents.

/// Snapshot the content of every level (nodes + pairs + multi_pairs). Dirty-tracking
/// state may legitimately differ post-rotation; only content is invariant.
fn snapshot_levels(tdd: &Tdd) -> Vec<(Vec<EncodedNode>, Vec<ChildPair>, Vec<MultiPairRange>)> {
    tdd.levels
        .iter()
        .map(|l| (l.nodes.clone(), l.pairs.clone(), l.multi_pairs.clone()))
        .collect()
}

fn assert_locality(
    tdd: &Tdd,
    snap: &[(Vec<EncodedNode>, Vec<ChildPair>, Vec<MultiPairRange>)],
    v_idx: usize,
    w_idx: usize,
) {
    for (i, level) in tdd.levels.iter().enumerate() {
        if i == v_idx || i == w_idx { continue; }
        assert_eq!(
            level.nodes, snap[i].0,
            "rotation-locality: level {i} nodes changed (v={v_idx}, w={w_idx})",
        );
        assert_eq!(
            level.pairs, snap[i].1,
            "rotation-locality: level {i} pairs changed (v={v_idx}, w={w_idx})",
        );
        assert_eq!(
            level.multi_pairs, snap[i].2,
            "rotation-locality: level {i} multi_pairs changed (v={v_idx}, w={w_idx})",
        );
    }
}

/// Helper: apply a left rotation at `target`, run the locality check and
/// clear the worklists, then assert locality. Returns the (v_idx, w_idx) used so the caller can
/// chain further rotations.
fn rotate_left_and_check_locality(eng: &Engine, tdd: &mut Tdd, target: crate::vtree::VtreeIdx) -> Option<(usize, usize)> {
    let mut vt = (*tdd.vtree).clone();
    let info = rotate_left(&mut vt, target)?;
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let snap = snapshot_levels(tdd);
    tdd.vtree = Arc::new(vt);
    let _ = restructure_inner_search(tdd, &info, RotationKind::Left, &mut RestructureScratch::default(), usize::MAX);
    crate::test_helpers::check::debug_assert_rotation_locality(eng, tdd, info.w_idx);
    tdd.clear_worklists();
    assert_locality(tdd, &snap, v_idx, w_idx);
    assert_canonical(tdd);
    Some((v_idx, w_idx))
}

fn rotate_right_and_check_locality(eng: &Engine, tdd: &mut Tdd, target: crate::vtree::VtreeIdx) -> Option<(usize, usize)> {
    let mut vt = (*tdd.vtree).clone();
    let info = rotate_right(&mut vt, target)?;
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let snap = snapshot_levels(tdd);
    tdd.vtree = Arc::new(vt);
    let _ = restructure_inner_search(tdd, &info, RotationKind::Right, &mut RestructureScratch::default(), usize::MAX);
    crate::test_helpers::check::debug_assert_rotation_locality(eng, tdd, info.w_idx);
    tdd.clear_worklists();
    assert_locality(tdd, &snap, v_idx, w_idx);
    assert_canonical(tdd);
    Some((v_idx, w_idx))
}

#[test]
fn rotation_locality_left_at_root_balanced5() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(5));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2, 3], vec![-2, 4], vec![3, -4, 5],
            vec![-1, 5], vec![1, -3, 4],
        ],
    );
    let root = tdd.vtree.root();
    let _ = rotate_left_and_check_locality(&eng, &mut tdd, root)
        .expect("left rotation applicable at root of balanced(5)");
}

#[test]
fn rotation_locality_right_after_left_balanced5() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(5));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2, 3], vec![-2, 4], vec![3, -4, 5],
            vec![-1, 5], vec![1, -3, 4],
        ],
    );
    let root = tdd.vtree.root();
    let _ = rotate_left_and_check_locality(&eng, &mut tdd, root)
        .expect("left rotation applicable");
    let root = tdd.vtree.root();
    let _ = rotate_right_and_check_locality(&eng, &mut tdd, root)
        .expect("right rotation applicable after left");
}

#[test]
fn rotation_locality_linear_left_chain() {
    let eng = Engine::new();
    // Linear vtree on 6 vars — every internal node admits a left rotation.
    // Walks down the right spine applying rotations and re-checking
    // locality at each step. Covers cascade scenarios where successive
    // rotations interact.
    let vtree = Arc::new(Vtree::linear(6));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2], vec![-2, 3], vec![-3, 4],
            vec![-4, 5], vec![-5, 6], vec![1, -6],
        ],
    );
    // Apply 3 successive left rotations at the (current) root; each must
    // satisfy rotation locality independently.
    for _ in 0..3 {
        let root = tdd.vtree.root();
        if rotate_left_and_check_locality(&eng, &mut tdd, root).is_none() {
            break;
        }
    }
}

#[test]
fn rotation_locality_unsat_left() {
    let eng = Engine::new();
    // UNSAT formula: post-rotation diagram has an empty output but the
    // locality invariant still applies (vacuously: every level is empty,
    // and empty == empty).
    let vtree = Arc::new(Vtree::balanced(3));
    let mut tdd = compile_clauses(&vtree, &[vec![1], vec![-1], vec![2, 3]]);
    let root = tdd.vtree.root();
    if rotate_left_and_check_locality(&eng, &mut tdd, root).is_some() {
        // Asserted internally.
    }
}
