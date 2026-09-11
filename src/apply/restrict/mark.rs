//! The liveness walk over `f × care`.

use std::collections::HashMap;

use crate::diagram::{InputPair, NodeIdx, Tdd, ZERO};
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
    index: HashMap<Key, u32>,
    keys: Vec<Key>,
    live: Vec<bool>,
}

impl Marking {
    /// Walk the reachable pairs of `f × care` from vtree node `r` (a root of one
    /// operand) and mark every live f-node and f-pair. Two phases: discover the
    /// pairs top-down with a work stack, then evaluate their liveness bottom-up.
    pub(super) fn walk(f: &Tdd, care: &Tdd, r: VtreeIdx) -> Marking {
        let vtree = &f.vtree;
        let nlev = vtree.num_nodes();
        let ctx = WalkCtx { f, care };
        let mut levels: Vec<LevelPairs> = (0..nlev).map(|_| LevelPairs::default()).collect();

        // Phase 1: discover. Each internal pair enumerates its child references;
        // a child that is neither dead nor trivially live is a new pair to walk.
        let root_key = match ctx.child(r, None, None) {
            Child::Pair(k) => k,
            Child::Live => {
                // `r` is a root, so this is `f` marginal at its own root (a scalar):
                // nothing to mark, nothing died.
                return Marking::trivial(f, true);
            }
            Child::Dead => return Marking::trivial(f, false),
        };
        let mut stack: Vec<(VtreeIdx, Key)> = vec![(r, root_key)];
        levels[r.idx()].push(root_key);
        while let Some((v, (fo, co))) = stack.pop() {
            let (lc, rc) = vtree.children(v);
            for (fl, fr) in refs(f, v, fo) {
                for (cl, cr) in refs(care, v, co) {
                    for (cv, a, b) in [(lc, fl, cl), (rc, fr, cr)] {
                        if let Child::Pair(k) = ctx.child(cv, a, b)
                            && levels[cv.idx()].push(k) {
                                stack.push((cv, k));
                            }
                    }
                }
            }
        }

        // Phase 2: evaluate bottom-up (children before parents) and mark.
        let mut alive: Vec<Vec<bool>> =
            (0..nlev).map(|vi| vec![false; f.levels[vi].nodes.len()]).collect();
        let mut pair_alive: Vec<Vec<u64>> =
            (0..nlev).map(|vi| vec![0u64; f.levels[vi].nodes.len()]).collect();
        for (v, lc, rc) in vtree.internal_bottomup() {
            for i in 0..levels[v.idx()].keys.len() {
                let (fo, co) = levels[v.idx()].keys[i];
                let mut any = false;
                for (k, (fl, fr)) in refs(f, v, fo).enumerate() {
                    let live = refs(care, v, co).any(|(cl, cr)| {
                        ctx.live_of(&levels, lc, fl, cl) && ctx.live_of(&levels, rc, fr, cr)
                    });
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
        Marking { alive, pair_alive, root_live }
    }

    /// Marks for a walk that never examined a pair: nothing dies (every reachable
    /// node is reported alive, so `nothing_reachable_died` holds).
    fn trivial(f: &Tdd, root_live: bool) -> Marking {
        let nlev = f.vtree.num_nodes();
        Marking {
            alive: (0..nlev).map(|vi| vec![true; f.levels[vi].nodes.len()]).collect(),
            pair_alive: (0..nlev).map(|vi| vec![u64::MAX; f.levels[vi].nodes.len()]).collect(),
            root_live,
        }
    }

    /// Is every node and pair reachable from `f`'s root marked live? Then the
    /// rebuild would reproduce `f` pair-for-pair, so `g == f` and the caller can
    /// reuse `f` verbatim. Stack-driven traversal of `f`'s reachable subgraph.
    pub(super) fn nothing_reachable_died(&self, f: &Tdd) -> bool {
        let vtree = &f.vtree;
        let mut seen: Vec<Vec<bool>> =
            (0..vtree.num_nodes()).map(|vi| vec![false; f.levels[vi].nodes.len()]).collect();
        let mut stack = vec![(f.output.vtree, f.output.local)];
        while let Some((v, l)) = stack.pop() {
            if l == ZERO || vtree.node(v).is_leaf() || f.levels[v.idx()].is_marginal() {
                continue;
            }
            if std::mem::replace(&mut seen[v.idx()][l.idx()], true) {
                continue;
            }
            if !self.alive[v.idx()][l.idx()] {
                return false;
            }
            let pairs = f.levels[v.idx()].pairs_of_idx(l.idx());
            let mask = self.pair_alive[v.idx()][l.idx()];
            if mask != u64::MAX && (mask.count_ones() as usize) < pairs.len() {
                return false;
            }
            let (lc, rc) = vtree.children(v);
            for p in pairs {
                stack.push((lc, p.left));
                stack.push((rc, p.right));
            }
        }
        true
    }
}

impl LevelPairs {
    /// Record `k` if new; true iff it was.
    fn push(&mut self, k: Key) -> bool {
        if self.index.contains_key(&k) {
            return false;
        }
        self.index.insert(k, self.keys.len() as u32);
        self.keys.push(k);
        self.live.push(false);
        true
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

/// The 3×3 leaf table: two literals conflict only as `{Pos, Neg}`; `One` and
/// `⊤` (`None`) are wildcards. Index 3 (`LeafLabel::Zero`) is never stored.
fn leaf_dead(fo: Ref, co: Ref) -> bool {
    match (fo, co) {
        (Some(a), Some(b)) => a.0 >= 3 || b.0 >= 3 || a.0 + b.0 == 3,
        _ => false,
    }
}

/// The (left, right) child references of one operand at level `v`: its node's
/// pairs, or the single `(⊤, ⊤)` pair when the operand is `⊤` there.
fn refs(t: &Tdd, v: VtreeIdx, o: Ref) -> impl Iterator<Item = (Ref, Ref)> + '_ {
    let pairs: &[InputPair] = match o {
        Some(l) => t.levels[v.idx()].pairs_of_idx(l.idx()),
        None => &[],
    };
    let top = o.is_none();
    pairs
        .iter()
        .map(|p| (Some(p.left), Some(p.right)))
        .chain(std::iter::once((None, None)).filter(move |_| top))
}
