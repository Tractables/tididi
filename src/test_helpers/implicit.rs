//! Implicit levels for the tests of the passes that read or change them: a
//! diagram with an implicit x-decision level, and the stored copy
//! of a level, written out by enumerating its description, the oracle the
//! pass on the implicit level is checked against.

use std::cell::Cell;
use std::sync::Arc;

use crate::Engine;
use crate::diagram::{ChildPair, ImplicitLevel, NodeIdx, Tdd, TddLevel, TddNodeId, FLOOR, NEG_LEAF_IDX, POS_LEAF_IDX};
use crate::vtree::{Vtree, VtreeIdx};

/// A diagram over the right-linear vtree of four variables, `r = (x1, v)`,
/// `v = (x2, w)`, `w = (x3, x4)`, with an x-decision level of `n` nodes at
/// `v`: node `i` holds `(x2, 4i)`, `(¬x2, 4i + 1)`, `(x2, 4i + 2)` and
/// `(¬x2, 4i + 3)`, over the `4n` nodes `(x3 ∧ x4)` of `w`'s level, and the
/// root's one node holds `(x1, i)` for every node `i` of `v`'s, which is
/// held as the description of its pairs. Returns the diagram, `v` and `w`.
pub(crate) fn x_decision_diagram(n: usize) -> (Tdd, VtreeIdx, VtreeIdx) {
    let vtree = Arc::new(Vtree::linear(4));
    let mut tdd = crate::build::constant_one(&Engine::new(), &vtree);
    let r = vtree.root();
    let (_, v) = vtree.children(r);
    let (_, w) = vtree.children(v);
    assert!(!vtree.node(w).is_leaf() && vtree.node(vtree.children(w).0).is_leaf());
    let below = &mut tdd.levels[w.idx()];
    below.clear();
    for _ in 0..4 * n {
        below.push_internal_node(&[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]);
    }
    let level = &mut tdd.levels[v.idx()];
    level.clear();
    for i in 0..n {
        let label = |m: usize| if m.is_multiple_of(2) { POS_LEAF_IDX } else { NEG_LEAF_IDX };
        let pairs: Vec<ChildPair> = (0..4).map(|m| ChildPair::new(label(m), NodeIdx((4 * i + m) as u32))).collect();
        level.push_internal_node(&pairs);
    }
    describe(level);
    let root = &mut tdd.levels[r.idx()];
    root.clear();
    let pairs: Vec<ChildPair> = (0..n).map(|i| ChildPair::new(POS_LEAF_IDX, NodeIdx(i as u32))).collect();
    root.push_internal_node(&pairs);
    tdd.output = TddNodeId { vtree: r, local: NodeIdx(0) };
    (tdd, v, w)
}

/// Hold `level`, stored and affine, as the description of its pairs, at the
/// capacity of its arena.
pub(crate) fn describe(level: &mut TddLevel) {
    let d = ImplicitLevel::fit(level).expect("an affine level");
    let capacity = level.pairs.capacity();
    let node_capacity = level.nodes.imply();
    level.ranges.clear();
    level.pairs.clear();
    level.pairs.describe(d, capacity, node_capacity);
}

/// The stored copy of `level`: its pairs written where its nodes' ranges
/// put them, in an arena of the length and capacity the description stands
/// for, with the same dead slots.
pub(crate) fn stored_copy(level: &TddLevel) -> TddLevel {
    let mut copy = level.clone();
    copy.store_if_implicit();
    copy
}

/// `tdd` with the level at `v` replaced by its [`stored_copy`].
pub(crate) fn with_stored_copy(tdd: &Tdd, v: VtreeIdx) -> Tdd {
    let mut stored = tdd.clone();
    stored.levels[v.idx()] = stored_copy(&tdd.levels[v.idx()]);
    stored
}

/// `tdd` with every level replaced by its [`stored_copy`].
pub(crate) fn stored_copies(tdd: &Tdd) -> Tdd {
    let mut stored = tdd.clone();
    for level in &mut stored.levels {
        *level = stored_copy(level);
    }
    stored
}

thread_local! {
    /// Whether every level on this thread is built stored
    /// ([`stored_levels`]).
    static STORED_LEVELS: Cell<bool> = const { Cell::new(false) };
    /// The fewest pairs a level on this thread holds as their description
    /// ([`with_floor`]).
    static FLOOR_HERE: Cell<usize> = const { Cell::new(FLOOR) };
    /// The arenas on this thread held as a description so far.
    static DESCRIBED_HERE: Cell<u64> = const { Cell::new(0) };
}

/// The fewest pairs a level holds as their description on this thread:
/// [`FLOOR`], unless [`with_floor`] lowered it.
pub(crate) fn floor() -> usize {
    FLOOR_HERE.with(Cell::get)
}

/// Run `f` with the floor at `floor` pairs, so that the levels of small
/// diagrams can be implicit and the passes on them reached.
pub(crate) fn with_floor<R>(floor: usize, f: impl FnOnce() -> R) -> R {
    struct Reset(usize);
    impl Drop for Reset {
        fn drop(&mut self) {
            FLOOR_HERE.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(FLOOR_HERE.with(|c| c.replace(floor)));
    f()
}

/// Count an arena held as a description on this thread.
pub(crate) fn note_described() {
    DESCRIBED_HERE.with(|c| c.set(c.get() + 1));
}

/// The arenas on this thread held as a description so far.
pub(crate) fn described_here() -> u64 {
    DESCRIBED_HERE.with(Cell::get)
}

/// Whether every level on this thread is built stored.
pub(crate) fn stored_levels_forced() -> bool {
    STORED_LEVELS.with(Cell::get)
}

/// Run `f` with every level built stored: no conjunction level takes the
/// implicit route and no close describes a level, so that a pass on stored
/// operands meets no implicit level. The oracle implicit levels are checked
/// against.
pub(crate) fn stored_levels<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            STORED_LEVELS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(STORED_LEVELS.with(|c| c.replace(true)));
    f()
}

/// Require `out` to be `oracle` node for node: the same nodes in the same
/// order, each with the same pairs, every pair arena of the same length,
/// capacity and dead slots, every node arena of the same capacity, stored
/// or implied, and `oracle` stored throughout. A level whose
/// nodes are the oracle's word for word has its arena once written; a
/// level a prune or a close described numbers its nodes' ranges from the
/// start of the arena instead, where the stored route left them where they
/// were.
pub(crate) fn same_levels(out: &Tdd, oracle: &Tdd) {
    assert_eq!(out.output, oracle.output);
    let mut buf = Vec::new();
    for (t, (a, b)) in out.levels.iter().zip(oracle.levels.iter()).enumerate() {
        assert!(b.pairs.implicit().is_none(), "the stored route made a level implicit");
        assert_eq!(a.is_marginal(), b.is_marginal());
        assert_eq!(a.nodes().len(), b.nodes().len(), "the implicit route kept other nodes");
        if !a.is_marginal() {
            for i in 0..a.nodes().len() {
                assert_eq!(a.pair_count_at(i), b.pair_count_at(i));
                assert_eq!(a.pairs_read(i, &mut buf), b.pairs_vec(i), "a node has other pairs");
            }
        }
        assert_eq!(a.pairs.len(), b.pairs.len(), "the implicit route holds another arena length");
        assert_eq!(a.pairs.capacity(), b.pairs.capacity(), "the implicit route holds another capacity");
        assert_eq!(a.dead_pairs, b.dead_pairs, "the implicit route counts other dead slots");
        assert_eq!(
            a.node_capacity(),
            b.node_capacity(),
            "level {t}: the implicit route holds another node capacity, {} nodes, {:?} pairs a node",
            b.nodes().len(),
            a.implicit().map(|d| d.pairs_per_node())
        );
        if a.nodes == b.nodes {
            // Node for node the same ranges: the description stands for the
            // stored arena's pairs where they lie. A level built stored from
            // a description holds other words in its dead slots.
            assert_eq!(a.ranges, b.ranges);
            if a.pairs.implicit().is_none() && a.dead_pairs == 0 {
                assert_eq!(a.pairs, b.pairs, "a stored level holds another arena");
            }
        }
    }
}

/// The floor [`same_as_stored`] runs its operation under: every affine
/// level of two pairs or more is implicit.
pub(crate) const LOW_FLOOR: usize = 2;

/// Run `op` with the floor lowered to [`LOW_FLOOR`], so that the small
/// diagrams of the tests hold implicit levels, and again with every level
/// stored ([`stored_levels`]); require that the first run described a
/// level, and of every diagram it returns, the canonical form of implicit
/// levels and the stored run's levels, node for node ([`same_levels`]).
/// `op` builds its operands itself, so that the stored run's are stored.
/// Returns the diagrams of the first run.
pub(crate) fn same_as_stored(op: impl Fn() -> Vec<Tdd>) -> Vec<Tdd> {
    let oracle = stored_levels(&op);
    let before = described_here();
    let out = with_floor(LOW_FLOOR, || {
        let out = op();
        for t in &out {
            assert_implicit_canonical(t);
        }
        out
    });
    assert!(described_here() > before, "no level was implicit");
    assert_eq!(out.len(), oracle.len());
    for (a, b) in out.iter().zip(&oracle) {
        same_levels(a, b);
    }
    out
}

/// Require every level of `t` to be in the canonical form of implicit
/// levels: described exactly when it can be.
pub(crate) fn assert_implicit_canonical(t: &Tdd) {
    if let Err(e) = crate::test_helpers::check::check_implicit_levels(t) {
        panic!("a level is out of canonical form: {e}");
    }
}

/// The pairs of every structural internal level's nodes, each node's
/// sorted: what a pass on an implicit level and the same pass on its stored
/// copy agree on.
pub(crate) fn sorted_pairs(tdd: &Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
    tdd.vtree
        .internal_bottomup()
        .map(|(t, _, _)| {
            let level = &tdd.levels[t.idx()];
            if level.is_marginal() {
                return Vec::new();
            }
            (0..level.nodes().len())
                .map(|i| {
                    let mut pairs: Vec<(u32, u32)> = level.pairs_iter_of_idx(i).map(|p| (p.left.raw(), p.right.raw())).collect();
                    pairs.sort_unstable();
                    pairs
                })
                .collect()
        })
        .collect()
}
