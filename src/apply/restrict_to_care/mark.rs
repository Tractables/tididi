//! The liveness walk over `f × care`.

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::limits::OperationError;

use crate::apply::CONJOIN_GRID;
use crate::apply::conjoin::budget::NO_PRODUCT;
use crate::diagram::{ChildPair, NodeIdx, Tdd, LEAF_WIDTH, ZERO};
use crate::vtree::VtreeIdx;

use super::Marking;

/// One operand's reference into a vtree level: `None` = the operand is `⊤` here
/// (not yet rooted, or `care` marginal at this level), `Some(l)` = its node `l`.
type Ref = Option<NodeIdx>;
/// A walked pair: `f`'s reference and `care`'s reference at one vtree level.
type Key = (Ref, Ref);

/// What a child reference pair resolves to before it becomes a walked pair.
enum Child {
    Dead,
    Live,
    Pair(Key),
}

/// The pairs walked at one internal vtree level, in discovery order, with their
/// liveness (filled bottom-up once every child level is evaluated).
#[derive(Default)]
struct LevelPairs {
    index: FxHashMap<Key, u32>,
    keys: Vec<Key>,
    live: Vec<bool>,
}

impl Marking {
    /// Walk the reachable pairs of `f × care` from vtree node `r` (a root of one
    /// operand) and mark every live f-node and f-pair. Two phases: discover the
    /// pairs top-down with a work stack, then evaluate their liveness bottom-up.
    ///
    /// `remaining` is the allowance of product-pair probes across both phases;
    /// spending it returns `None` with the marks abandoned.
    pub(super) fn walk(eng: &Engine, f: &Tdd, care: &Tdd, r: VtreeIdx, mut remaining: u64) -> Result<Option<Marking>, OperationError> {
        let mut poll = eng.limits().gate();
        let vtree = &f.vtree;
        let nlev = vtree.num_nodes();
        let ctx = WalkCtx { f, care };
        let mut levels = Vec::new();
        eng.limits().reserve_exact(&mut levels, nlev)?;
        levels.resize_with(nlev, LevelPairs::default);

        // Phase 1: discover. Each internal pair enumerates its child references;
        // a child that is neither dead nor trivially live is a new pair to walk.
        let root_key = match ctx.child(r, None, None) {
            Child::Pair(k) => k,
            Child::Live => {
                // A compatible leaf or a marginal scalar: nothing died.
                return Marking::trivial(eng, f, true).map(Some);
            }
            Child::Dead => return Marking::trivial(eng, f, false).map(Some),
        };
        let mut stack = Vec::new();
        eng.limits().try_push(&mut stack, (r, root_key))?;
        levels[r.idx()].push(eng, root_key)?;
        while let Some((v, (fo, co))) = stack.pop() {
            let (lc, rc) = vtree.children(v);
            for (fl, fr) in refs(f, v, fo) {
                for (cl, cr) in refs(care, v, co) {
                    if !spend(&mut remaining) {
                        poll.flush()?;
                        return Ok(None);
                    }
                    poll.poll(1)?;
                    for (cv, a, b) in [(lc, fl, cl), (rc, fr, cr)] {
                        if let Child::Pair(k) = ctx.child(cv, a, b)
                            && levels[cv.idx()].push(eng, k)? {
                                eng.limits().try_push(&mut stack, (cv, k))?;
                            }
                    }
                }
            }
        }

        // Phase 2: evaluate bottom-up (children before parents) and mark.
        let mut alive = mark_rows(eng, f, false)?;
        let mut pair_alive = mark_rows(eng, f, 0u64)?;
        for (v, lc, rc) in vtree.internal_bottomup() {
            for i in 0..levels[v.idx()].keys.len() {
                let (fo, co) = levels[v.idx()].keys[i];
                let mut any = false;
                for (k, (fl, fr)) in refs(f, v, fo).enumerate() {
                    let mut live = false;
                    for (cl, cr) in refs(care, v, co) {
                        if !spend(&mut remaining) {
                            poll.flush()?;
                            return Ok(None);
                        }
                        poll.poll(1)?;
                        if ctx.live_of(&levels, lc, fl, cl) && ctx.live_of(&levels, rc, fr, cr) {
                            live = true;
                            break;
                        }
                    }
                    if live {
                        any = true;
                        if let Some(fnode) = fo {
                            alive[v.idx()][fnode.idx()] = true;
                            if k < 64 {
                                pair_alive[v.idx()][fnode.idx()] |= 1 << k;
                            }
                        }
                    }
                }
                if any
                    && let Some(fnode) = fo
                        && f.levels[v.idx()].pairs_of_idx(fnode.idx()).len() > 64 {
                            pair_alive[v.idx()][fnode.idx()] = u64::MAX;
                        }
                levels[v.idx()].live[i] = any;
            }
        }
        let root_live = levels[r.idx()].live[0];
        poll.flush()?;
        Ok(Some(Marking { alive, pair_alive, root_live }))
    }

    /// Marks for a walk that never examined a pair: nothing dies (every reachable
    /// node is reported alive, so `nothing_reachable_died` holds).
    pub(super) fn trivial(eng: &Engine, f: &Tdd, root_live: bool) -> Result<Marking, OperationError> {
        Ok(Marking {
            alive: mark_rows(eng, f, true)?,
            pair_alive: mark_rows(eng, f, u64::MAX)?,
            root_live,
        })
    }

    /// Is every node and pair reachable from `f`'s root marked live? Then the
    /// rebuild would reproduce `f` pair-for-pair, so `g == f` and the caller can
    /// reuse `f` verbatim. Stack-driven traversal of `f`'s reachable subgraph.
    pub(super) fn nothing_reachable_died(&self, eng: &Engine, f: &Tdd) -> Result<bool, OperationError> {
        let mut poll = eng.limits().gate();
        let vtree = &f.vtree;
        let mut seen = mark_rows(eng, f, false)?;
        let mut stack = Vec::new();
        eng.limits().try_push(&mut stack, (f.output.vtree, f.output.local))?;
        while let Some((v, l)) = stack.pop() {
            if l == ZERO || vtree.node(v).is_leaf() || f.levels[v.idx()].is_marginal() {
                continue;
            }
            if std::mem::replace(&mut seen[v.idx()][l.idx()], true) {
                continue;
            }
            if !self.alive[v.idx()][l.idx()] {
                return Ok(false);
            }
            let pairs = f.levels[v.idx()].pairs_of_idx(l.idx());
            let mask = self.pair_alive[v.idx()][l.idx()];
            if mask != u64::MAX && (mask.count_ones() as usize) < pairs.len() {
                return Ok(false);
            }
            let (lc, rc) = vtree.children(v);
            for p in pairs {
                for (child, side) in [(lc, p.left), (rc, p.right)] {
                    let decoder = f.levels[child.idx()].child_decoder();
                    if !decoder.is_marginal() {
                        eng.limits().try_push(&mut stack, (child, decoder.node(side)))?;
                    }
                }
                poll.poll(1)?;
            }
        }
        poll.flush()?;
        Ok(true)
    }
}

impl LevelPairs {
    /// Record `k` if new; true iff it was.
    fn push(&mut self, eng: &Engine, k: Key) -> Result<bool, OperationError> {
        if self.index.len() == self.index.capacity() { eng.limits().reserve_map(&mut self.index, 1)?; }
        let std::collections::hash_map::Entry::Vacant(slot) = self.index.entry(k) else {
            return Ok(false);
        };
        slot.insert(self.keys.len() as u32);
        eng.limits().try_push(&mut self.keys, k)?;
        eng.limits().try_push(&mut self.live, false)?;
        Ok(true)
    }
}

/// The operands of one walk plus the child-resolution rules shared by both phases.
struct WalkCtx<'a> {
    f: &'a Tdd,
    care: &'a Tdd,
}

impl WalkCtx<'_> {
    /// Resolve the references `(fo, co)` that a parent pair hands to child level
    /// `cv`: root an operand whose root is `cv`, then apply the terminal rules
    /// (`ZERO` dead; `f` marginal live; `care` marginal ⊤; leaf table), else it
    /// is a pair to walk.
    fn child(&self, cv: VtreeIdx, fo: Ref, co: Ref) -> Child {
        let fo = if fo.is_none() && cv == self.f.output.vtree { Some(self.f.output.local) } else { fo };
        let co = if co.is_none() && cv == self.care.output.vtree { Some(self.care.output.local) } else { co };
        if fo == Some(ZERO) || co == Some(ZERO) {
            return Child::Dead;
        }
        if self.f.levels[cv.idx()].is_marginal() {
            // A marginal f level is a count (> 0), never a node: always live.
            return Child::Live;
        }
        // A marginal care level is a satisfiability indicator: ⊤ for liveness.
        let co = if self.care.levels[cv.idx()].is_marginal() { None } else { co };
        if self.f.vtree.node(cv).is_leaf() {
            return if leaf_dead(fo, co) { Child::Dead } else { Child::Live };
        }
        Child::Pair((fo, co))
    }

    /// Liveness of a child reference pair, reading an already-evaluated walked
    /// pair from `levels` (phase 1 recorded every pair `child` can name).
    fn live_of(&self, levels: &[LevelPairs], cv: VtreeIdx, fo: Ref, co: Ref) -> bool {
        match self.child(cv, fo, co) {
            Child::Dead => false,
            Child::Live => true,
            Child::Pair(k) => {
                let lp = &levels[cv.idx()];
                lp.live[lp.index[&k] as usize]
            }
        }
    }
}

/// Whether two leaf references conflict: only `{Pos, Neg}` does, and `⊤`
/// (`None`) is a wildcard. `ZERO` never reaches here.
fn leaf_dead(fo: Ref, co: Ref) -> bool {
    match (fo, co) {
        (Some(a), Some(b)) => {
            debug_assert!(a.idx() < LEAF_WIDTH && b.idx() < LEAF_WIDTH, "a leaf reference names a label");
            CONJOIN_GRID[a.idx()][b.idx()] == NO_PRODUCT
        }
        _ => false,
    }
}

/// Take one probe from the allowance; false once it is spent.
#[inline]
fn spend(remaining: &mut u64) -> bool {
    if *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    true
}

/// The (left, right) child references of one operand at level `v`: its node's
/// pairs, or the single `(⊤, ⊤)` pair when the operand is `⊤` there.
fn refs(t: &Tdd, v: VtreeIdx, o: Ref) -> impl Iterator<Item = (Ref, Ref)> + '_ {
    let pairs: &[ChildPair] = match o {
        Some(l) => t.levels[v.idx()].pairs_of_idx(l.idx()),
        None => &[],
    };
    let top = o.is_none();
    let (left, right) = t.vtree.children(v);
    let decode = move |child: VtreeIdx, side| {
        let decoder = t.levels[child.idx()].child_decoder();
        if decoder.is_marginal() { None } else { Some(decoder.node(side)) }
    };
    pairs
        .iter()
        .map(move |p| (decode(left, p.left), decode(right, p.right)))
        .chain(std::iter::once((None, None)).filter(move |_| top))
}

/// Allocate one initialized marking row per diagram level through the engine.
fn mark_rows<T: Clone>(eng: &Engine, f: &Tdd, value: T) -> Result<Vec<Vec<T>>, OperationError> {
    let mut rows = Vec::new();
    eng.limits().reserve_exact(&mut rows, f.levels.len())?;
    for level in &f.levels {
        let mut row = Vec::new();
        eng.limits().try_resize(&mut row, level.nodes.len(), value.clone())?;
        rows.push(row);
    }
    Ok(rows)
}
