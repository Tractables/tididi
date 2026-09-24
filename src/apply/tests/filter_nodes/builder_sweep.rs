//! The filter against a replica written with the public builder alone: one
//! bottom-up pass pushing every surviving node, an unchecked finish, then the
//! same reduction. A caller that filters this way today gets the same
//! diagram from the library, down to the reduction worklists the finish
//! seeds.

use super::*;
use crate::diagram::{TddBuilder, TddLevel};

/// The replica. Returns the unfinished builder, the output or `None` when it
/// was removed, and the two counts.
///
/// The public builder has no way to set a level's inlined-side markers, so
/// every structural level of the replica leaves them clear.
pub(super) fn replica(eng: &Engine, f: &Tdd, keep: impl Fn(TddNodeId) -> bool) -> (TddBuilder, Option<TddNodeId>, FilterStats) {
    let vtree = f.vtree();
    let mut remap: Vec<Vec<u32>> = (0..vtree.num_nodes())
        .map(|i| (0..f.reference_slot_count(VtreeIdx(i as u32)) as u32).collect())
        .collect();
    let mut out = Tdd::builder(eng, vtree).unwrap();
    for i in 0..vtree.num_nodes() {
        let t = VtreeIdx(i as u32);
        if f.levels()[i].is_marginal() { out.replace_level(eng, t, f.level_view(t)).unwrap(); }
    }
    let side = |child: VtreeIdx, raw: EncodedChildRef, remap: &[Vec<u32>]| -> u32 {
        if f.levels()[child.idx()].is_marginal() {
            match ChildDecoder::marginal().value(raw) {
                ValueRef::Inline(0) => DEAD,
                ValueRef::Inline(_) => raw.raw(),
                ValueRef::Slot(slot) => remap[child.idx()][slot as usize],
            }
        } else {
            remap[child.idx()][ChildDecoder::structural().node(raw).idx()]
        }
    };
    let mut stats = FilterStats::default();
    let mut pairs = Vec::new();
    for (t, left, right) in vtree.internal_bottomup() {
        let source = &f.levels()[t.idx()];
        if source.is_marginal() { continue; }
        let mut next = 0;
        for i in 0..source.nodes().len() {
            if !source.nodes()[i].is_internal() || !keep(TddNodeId { vtree: t, local: NodeIdx(i as u32) }) {
                remap[t.idx()][i] = DEAD;
                continue;
            }
            pairs.clear();
            for pair in source.pairs_of_idx(i) {
                if pair.left == ZERO.into() || pair.right == ZERO.into() {
                    stats.pairs_dropped += 1;
                    continue;
                }
                let (l, r) = (side(left, pair.left, &remap), side(right, pair.right, &remap));
                if l == DEAD || r == DEAD {
                    stats.pairs_dropped += 1;
                    continue;
                }
                pairs.push(ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)));
            }
            if pairs.is_empty() {
                remap[t.idx()][i] = DEAD;
                stats.emptied_nodes += 1;
                continue;
            }
            out.push(eng, t, &pairs).unwrap();
            remap[t.idx()][i] = next;
            next += 1;
        }
    }
    let root = f.output();
    let local = remap[root.vtree.idx()][root.local.idx()];
    (out, (local != DEAD).then_some(TddNodeId { vtree: root.vtree, local: NodeIdx(local) }), stats)
}

/// Everything a later operation reads of `f`: its levels, its output and the
/// levels its reduction still has to visit.
pub(super) fn image(f: &Tdd) -> String {
    format!("{:?}\n{:?}\n{:?}", &*f.levels, f.output, f.dirty)
}

/// [`image`] of `f` with every inlined-side marker cleared.
pub(super) fn shape(f: &Tdd) -> String {
    let mut g = f.clone();
    for level in g.levels.iter_mut() { level.inlined_sides = 0; }
    image(&g)
}

/// Give `g`'s structural levels the inlined-side markers of `f`'s.
fn copy_markers(g: &mut Tdd, f: &Tdd) {
    for (level, source) in g.levels.iter_mut().zip(f.levels.iter()) {
        if !source.is_marginal() { level.inlined_sides = source.inlined_sides; }
    }
}

/// Satisfiable structural and marginal operands, each with a sparse
/// rejection.
fn cases() -> impl Iterator<Item = (Tdd, impl Fn(TddNodeId) -> bool)> {
    let marginal = marginal_diagrams(0xb5e2, 80, 8..12).into_iter().map(|(f, _)| f);
    random_diagrams(0xb5e1, 60, 8..13).into_iter().chain(marginal).filter(|f| !f.is_zero()).enumerate()
        .map(|(k, f)| { let keep = sparing_output(&f, k as u64, 10); (f, keep) })
}

#[test]
fn the_sweep_and_its_finish_match_a_builder_sweep() {
    let mut compared = [0usize; 2];
    for (f, keep) in cases() {
        let eng = Engine::new();
        let Some(mut remap) = ask(&eng, &f, &keep).unwrap() else { continue };
        let (assembly, stats) = sweep(&eng, &f, &mut remap).unwrap();
        let (builder, output, expected) = replica(&eng, &f, &keep);
        assert_eq!(stats, expected);
        // The same levels, node for node and pair for pair; only the markers
        // differ, and only where the operand has them.
        for (i, source) in f.levels.iter().enumerate() {
            let t = VtreeIdx(i as u32);
            let (mut ours, theirs) = (assembly.level(t).clone(), builder.level(t).clone());
            if !source.is_marginal() {
                assert_eq!(ours.inlined_sides, source.inlined_sides);
                ours.inlined_sides = theirs.inlined_sides;
            }
            assert_eq!(format!("{ours:?}"), format!("{theirs:?}"), "level {t:?}");
        }
        let root = f.output();
        let local = if f.vtree.node(root.vtree).is_leaf() || f.levels[root.vtree.idx()].is_marginal() {
            root.local.0
        } else {
            remap[root.vtree.idx()][root.local.idx()]
        };
        assert_eq!((local != DEAD).then_some(TddNodeId { vtree: root.vtree, local: NodeIdx(local) }), output);
        let Some(output) = output else { continue };
        // Finishing seeds the same reduction work, and the same reduction then
        // leaves the same diagram.
        let mut ours = assembly.finish(output).unwrap();
        // Safety: every pushed pair is a pair of the valid operand with its
        // sides remapped onto surviving nodes.
        let mut theirs = unsafe { builder.finish_unchecked(output) };
        copy_markers(&mut theirs, &f);
        assert_eq!(image(&ours), image(&theirs));
        eng.reduce(&mut ours, ReductionPlan::default()).unwrap();
        eng.reduce(&mut theirs, ReductionPlan::default()).unwrap();
        assert_eq!(image(&ours), image(&theirs));
        compared[usize::from(f.levels.iter().any(TddLevel::any_inlined_side))] += 1;
    }
    assert!(compared.iter().all(|&n| n >= 20), "too few operands compared: {compared:?}");
}

#[test]
fn every_outcome_matches_a_builder_sweep_under_either_reduction() {
    let mut outcomes = [0usize; 3];
    for (f, keep) in cases() {
        let marked = f.levels.iter().any(TddLevel::any_inlined_side);
        for full in [false, true] {
            let plan = || if full { ReductionPlan::default() } else { ReductionPlan::Prune };
            let eng = Engine::new();
            let outcome = eng.filter_nodes_with(&f, &keep, plan()).unwrap();
            let (builder, output, stats) = replica(&eng, &f, &keep);
            match outcome {
                FilterOutcome::Unchanged => {
                    assert_eq!(stats, FilterStats::default());
                    assert!(f.levels.iter().enumerate().all(|(i, level)| level.is_marginal() || level.nodes().iter().enumerate()
                        .all(|(j, node)| !node.is_internal() || keep(TddNodeId { vtree: VtreeIdx(i as u32), local: NodeIdx(j as u32) }))));
                    outcomes[0] += 1;
                }
                FilterOutcome::Filtered { tdd, stats: ours } => {
                    assert_eq!(ours, stats);
                    // Safety: as in the test above.
                    let mut theirs = unsafe { builder.finish_unchecked(output.expect("the output survived")) };
                    eng.reduce(&mut theirs, plan()).unwrap();
                    // The replica leaves the markers clear and is otherwise the
                    // same diagram; with no markers to copy it is the same.
                    assert_eq!(shape(&tdd), shape(&theirs));
                    if !marked { assert_eq!(image(&tdd), image(&theirs)); }
                    outcomes[1] += 1;
                }
                FilterOutcome::Unsatisfiable { tdd, stats: ours } => {
                    assert_eq!(ours, stats);
                    assert!(output.is_none());
                    builder.abandon(&eng);
                    assert_eq!(image(&tdd), image(&Tdd::zero(f.vtree())));
                    outcomes[2] += 1;
                }
            }
        }
    }
    assert!(outcomes.iter().all(|&n| n > 0), "every outcome occurs: {outcomes:?}");
}
