//! A diagram's pairs, looked up by their sides.
//!
//! For each internal level, the pairs of the nodes the output reaches are
//! kept twice: grouped by their left side, each group sorted by the right
//! side, and grouped by their right side, each group sorted by the left. A
//! group is a run of one array, found through an array of run starts with
//! a slot per node of the child level. A pair of two sides is found by a
//! binary search in its left side's run, and the pairs that share one side
//! are that side's run.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::FxHasher;

use crate::Engine;
use crate::limits::{Limits, OperationError};
use crate::vtree::{Vtree, VtreeIdx};

use super::{ChildPair, NodeIdx, Tdd};

/// For every internal level of a diagram `g`, the pairs of `g`'s nodes,
/// looked up by their sides: which node owns a pair `(l, r)`, and which
/// pairs, with their owners, have a given left side or a given right side.
///
/// A node of `g` at vtree node `t` is the disjoint union of its pairs, each
/// pair `(l, r)` the product of node `l` of `t`'s left child level and node
/// `r` of its right child level; distinct nodes at one level are disjoint,
/// so a pair belongs to at most one node ([architecture, invariant
/// 1](crate::guide::architecture)). The index answers, for a pair of
/// children, the one node built from them, without reading the level.
/// [`Engine::and_indexed`] and [`Engine::sum_indexed`] read it to conjoin
/// another diagram with `g` while reading only the parts of `g` the other
/// diagram meets.
///
/// Only the nodes `g`'s output reaches are indexed. A leaf child's sides
/// are its labels ([`LeafLabel`](super::LeafLabel)).
///
/// # Size and cost
///
/// Each indexed pair is held twice, as two 32-bit words in each direction,
/// and each level holds one 32-bit run start per node of each child level:
/// about `16 · pairs + 4 · nodes` bytes in all ([`heap_bytes`](Self::heap_bytes)
/// gives the exact figure). Building it reads every pair of the reachable
/// nodes twice and sorts each run: `O(pairs · log(pairs per run))` time.
/// It keeps no reference to `g`; it holds the vtree.
///
/// # Validity
///
/// An index describes the diagram it was built from as that diagram stood.
/// Any operation that rebuilds or changes `g` (an apply, a minimization, a
/// reduction, a renumbering, a conditioning) makes it stale: build a new
/// one. An operation reading an index first compares a record the index
/// keeps of `g`'s shape (its vtree, output, level widths and a sample of
/// each level's pairs) with its operand, and refuses one that differs
/// ([`OperationError::IndexMismatch`]); a change that keeps every recorded
/// feature is not detected.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::diagram::{NodeIdx, PairIndex};
///
/// let vtree = Arc::new(Vtree::balanced(2));
/// let g = Tdd::clause(&vtree, [1, 2])?;
/// let index = PairIndex::new(&g)?;
/// let root = vtree.root();
/// // Every pair of the output is found by its two sides, and its owner is the output.
/// for pair in g.level(root).pairs_iter_of_idx(g.output().local.idx()) {
///     assert_eq!(index.owner(root, pair), Some(g.output().local));
/// }
/// assert_eq!(index.pair_count(), g.pair_count());
/// # Ok::<(), tididi::OperationError>(())
/// ```
#[derive(Clone, Debug)]
pub struct PairIndex {
    /// The vtree of the diagram indexed, which an operand must share.
    vtree: Arc<Vtree>,
    /// The record of the diagram's shape, compared with an operand's.
    shape: u64,
    /// One entry per vtree node; a leaf's is empty.
    levels: Vec<Sides>,
    /// The pairs indexed, over every level.
    pairs: usize,
}

/// One level's pairs, by each side.
#[derive(Clone, Debug, Default)]
struct Sides {
    /// The run of each left side `l` in `by_left`: `left_at[l]..left_at[l + 1]`.
    left_at: Vec<u32>,
    /// `(right side, owner)`, grouped by left side, each run sorted.
    by_left: Vec<(u32, u32)>,
    /// The run of each right side `r` in `by_right`.
    right_at: Vec<u32>,
    /// `(left side, owner)`, grouped by right side, each run sorted.
    by_right: Vec<(u32, u32)>,
}

impl Sides {
    fn left_run(&self, l: u32) -> &[(u32, u32)] {
        run(&self.left_at, &self.by_left, l)
    }

    fn right_run(&self, r: u32) -> &[(u32, u32)] {
        run(&self.right_at, &self.by_right, r)
    }
}

/// Run `side` of a grouped array; empty for a side past the starts.
#[inline]
fn run<'a>(at: &[u32], entries: &'a [(u32, u32)], side: u32) -> &'a [(u32, u32)] {
    let s = side as usize;
    match (at.get(s), at.get(s + 1)) {
        (Some(&from), Some(&to)) => &entries[from as usize..to as usize],
        _ => &[],
    }
}

impl PairIndex {
    /// Index the pairs of `g`'s reachable nodes, using the execution
    /// context of `g`'s vtree ([`Engine::pair_index`] under a caller's
    /// limits).
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] when `g` has a marginal level,
    /// [`OperationError::IndexOverflow`] when one level holds `2^32` pairs
    /// or more, and [`OperationError::OverBudget`] when an allocation is
    /// refused.
    pub fn new(g: &Tdd) -> Result<PairIndex, OperationError> {
        let context = Arc::clone(g.context());
        context.run(|eng| eng.pair_index(g))
    }

    /// The node of level `t` that owns `pair`, if one of the indexed nodes
    /// does; `None` at a leaf or a vtree node outside the vtree.
    pub fn owner(&self, t: VtreeIdx, pair: ChildPair) -> Option<NodeIdx> {
        self.owner_of(t, pair.left.raw(), pair.right.raw()).map(NodeIdx)
    }

    /// The indexed pairs of level `t` whose left side is `left`, as
    /// `(right side, owner)` in ascending order of the right side.
    pub fn with_left(&self, t: VtreeIdx, left: NodeIdx) -> impl ExactSizeIterator<Item = (NodeIdx, NodeIdx)> + '_ {
        self.left_run(t, left.0).iter().map(|&(r, o)| (NodeIdx(r), NodeIdx(o)))
    }

    /// The indexed pairs of level `t` whose right side is `right`, as
    /// `(left side, owner)` in ascending order of the left side.
    pub fn with_right(&self, t: VtreeIdx, right: NodeIdx) -> impl ExactSizeIterator<Item = (NodeIdx, NodeIdx)> + '_ {
        self.right_run(t, right.0).iter().map(|&(l, o)| (NodeIdx(l), NodeIdx(o)))
    }

    /// The pairs indexed, over every level: the pairs of the reachable
    /// nodes, which is [`Tdd::pair_count`] for a diagram whose every node
    /// is reachable.
    pub fn pair_count(&self) -> usize {
        self.pairs
    }

    /// The bytes the index holds on the heap: its arrays' capacities.
    pub fn heap_bytes(&self) -> usize {
        let word = std::mem::size_of::<u32>();
        let entry = std::mem::size_of::<(u32, u32)>();
        self.levels.capacity() * std::mem::size_of::<Sides>()
            + self.levels.iter().map(|s| {
                (s.left_at.capacity() + s.right_at.capacity()) * word + (s.by_left.capacity() + s.by_right.capacity()) * entry
            }).sum::<usize>()
    }

    /// The vtree of the diagram indexed.
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// Refuse a diagram this index was not built from: another vtree, or
    /// another recorded shape.
    pub(crate) fn check(&self, g: &Tdd) -> Result<(), OperationError> {
        if !Arc::ptr_eq(&self.vtree, g.vtree()) {
            return Err(OperationError::VtreeMismatch);
        }
        if self.shape != shape_of(g) {
            return Err(OperationError::IndexMismatch);
        }
        Ok(())
    }

    /// The owner of the pair `(l, r)` at level `t`.
    #[inline]
    pub(crate) fn owner_of(&self, t: VtreeIdx, l: u32, r: u32) -> Option<u32> {
        let run = self.left_run(t, l);
        run.binary_search_by_key(&r, |e| e.0).ok().map(|k| run[k].1)
    }

    /// `(right, owner)` of the pairs at `t` whose left side is `l`.
    #[inline]
    pub(crate) fn left_run(&self, t: VtreeIdx, l: u32) -> &[(u32, u32)] {
        self.levels.get(t.idx()).map_or(&[], |s| s.left_run(l))
    }

    /// `(left, owner)` of the pairs at `t` whose right side is `r`.
    #[inline]
    pub(crate) fn right_run(&self, t: VtreeIdx, r: u32) -> &[(u32, u32)] {
        self.levels.get(t.idx()).map_or(&[], |s| s.right_run(r))
    }

    /// Every indexed pair at `t`, as `(left, right, owner)`.
    pub(crate) fn all_at(&self, t: VtreeIdx) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        let sides = self.levels.get(t.idx());
        let lefts = sides.map_or(0, |s| s.left_at.len().saturating_sub(1));
        (0..lefts as u32).flat_map(move |l| {
            sides.map_or(&[][..], |s| s.left_run(l)).iter().map(move |&(r, o)| (l, r, o))
        })
    }
}

/// The record of `g`'s shape an index keeps: its output, each level's
/// width, and at each internal level the pair counts and first pairs of
/// four nodes spread over it. Read in time linear in the vtree, whatever
/// the diagram's size.
fn shape_of(g: &Tdd) -> u64 {
    let mut h = FxHasher::default();
    g.output().hash(&mut h);
    let vtree = g.vtree();
    for t in vtree.bottomup() {
        let level = g.level(t);
        level.slot_count().hash(&mut h);
        if vtree.node(t).is_leaf() || level.is_marginal() {
            continue;
        }
        let width = level.node_count();
        if width == 0 {
            continue;
        }
        let mut last = usize::MAX;
        for k in [0, width / 3, 2 * width / 3, width - 1] {
            if k == last {
                continue;
            }
            last = k;
            level.pair_count_at(k).hash(&mut h);
            for p in level.pairs_iter_of_idx(k).take(2) {
                p.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// Which nodes `g`'s output reaches, per level over its reference slots.
fn reached(lim: &Limits, g: &Tdd) -> Result<Vec<Vec<bool>>, OperationError> {
    let vtree = g.vtree();
    let mut reach: Vec<Vec<bool>> = Vec::new();
    lim.reserve_exact(&mut reach, vtree.num_nodes())?;
    for i in 0..vtree.num_nodes() {
        let mut level = Vec::new();
        lim.try_resize(&mut level, g.reference_slot_count(VtreeIdx(i as u32)), false)?;
        reach.push(level);
    }
    if g.is_zero() {
        return Ok(reach);
    }
    reach[g.output().vtree.idx()][g.output().local.idx()] = true;
    let mut gate = lim.gate();
    for (t, l, r) in vtree.internal_bottomup().rev() {
        let level = g.level(t);
        for i in 0..level.node_count() {
            if !reach[t.idx()][i] {
                continue;
            }
            let pairs = level.pairs_iter_of_idx(i);
            gate.poll(pairs.len() as u64)?;
            for p in pairs {
                reach[l.idx()][p.left.raw() as usize] = true;
                reach[r.idx()][p.right.raw() as usize] = true;
            }
        }
    }
    gate.flush()?;
    Ok(reach)
}

/// Run starts over the sides of one child level, and the runs.
type Runs = (Vec<u32>, Vec<(u32, u32)>);

/// Group `(side, other, owner)` entries by `side` into run starts over
/// `width` sides and `(other, owner)` runs sorted by `other`.
fn grouped(
    lim: &Limits,
    width: usize,
    count: usize,
    fill: impl Fn(&mut dyn FnMut(u32, u32, u32)),
) -> Result<Runs, OperationError> {
    let mut at: Vec<u32> = Vec::new();
    lim.try_resize(&mut at, width + 1, 0)?;
    fill(&mut |side, _, _| at[side as usize + 1] += 1);
    for s in 0..width {
        at[s + 1] += at[s];
    }
    let mut entries: Vec<(u32, u32)> = Vec::new();
    lim.try_resize(&mut entries, count, (0, 0))?;
    let mut next: Vec<u32> = Vec::new();
    lim.reserve_exact(&mut next, width)?;
    next.extend_from_slice(&at[..width]);
    fill(&mut |side, other, owner| {
        let k = &mut next[side as usize];
        entries[*k as usize] = (other, owner);
        *k += 1;
    });
    drop(next);
    for s in 0..width {
        let run = &mut entries[at[s] as usize..at[s + 1] as usize];
        run.sort_unstable();
        debug_assert!(
            run.windows(2).all(|w| w[0].0 != w[1].0),
            "a pair belongs to one node (invariant 1)",
        );
    }
    Ok((at, entries))
}

impl Engine {
    /// [`PairIndex::new`] under this engine's limits.
    ///
    /// # Errors
    ///
    /// As [`PairIndex::new`], and the engine's refusals
    /// ([`OperationError::Stopped`]).
    pub fn pair_index(&self, g: &Tdd) -> Result<PairIndex, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        g.require_structure()?;
        let vtree = Arc::clone(g.vtree());
        let reach = reached(lim, g)?;
        let mut levels: Vec<Sides> = Vec::new();
        lim.reserve_exact(&mut levels, vtree.num_nodes())?;
        levels.resize_with(vtree.num_nodes(), Sides::default);
        let mut pairs = 0usize;
        let mut gate = lim.gate();
        for (t, l, r) in vtree.internal_bottomup() {
            let level = g.level(t);
            let live = &reach[t.idx()];
            let each = |emit: &mut dyn FnMut(ChildPair, u32)| {
                for (i, &on) in live.iter().enumerate() {
                    if on {
                        for p in level.pairs_iter_of_idx(i) {
                            emit(p, i as u32);
                        }
                    }
                }
            };
            let mut count = 0usize;
            each(&mut |_, _| count += 1);
            gate.poll(count as u64)?;
            if u32::try_from(count).is_err() {
                return Err(OperationError::IndexOverflow);
            }
            pairs += count;
            let (left_at, by_left) = grouped(lim, g.reference_slot_count(l), count, |put| {
                each(&mut |p, o| put(p.left.raw(), p.right.raw(), o));
            })?;
            let (right_at, by_right) = grouped(lim, g.reference_slot_count(r), count, |put| {
                each(&mut |p, o| put(p.right.raw(), p.left.raw(), o));
            })?;
            levels[t.idx()] = Sides { left_at, by_left, right_at, by_right };
        }
        gate.flush()?;
        Ok(PairIndex { vtree, shape: shape_of(g), levels, pairs })
    }
}
