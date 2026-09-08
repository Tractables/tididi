//! Restrict-to-care: prune `f` to the subgraph that survives under a "care" TDD.
//!
//! `restrict(f, care)` returns `g`, a **structural subgraph of `f`** — every node
//! of `g` is a node of `f` keeping a subset of its pairs — with
//! `g ∧ care == f ∧ care`. It never grows the diagram (`g.size() ≤ f.size()`)
//! and it is the *drop lever only*:
//! it deletes pairs and nodes that produce no model under `care`, nothing else
//! (not Coudert–Madre `constrain`, no sibling substitution). The result is raw:
//! orphan-free but otherwise non-canonical, so a caller that needs a reduced
//! diagram runs `minimize` on it.
//!
//! Algorithm — a memoized top-down walk over node pairs of `f × care`, then a
//! rebuild:
//! 1. Start at `r = lca(root(f), root(care))`. An operand not rooted at `r` is
//!    `⊤` there (`None`) and becomes its root node once the walk reaches that
//!    vtree node. Incomparable roots cover disjoint variables, so `care` cannot
//!    constrain `f`: `Unchanged`.
//! 2. A pair `(v, f_node, care_node)` is *live* iff some `f`-pair × `care`-pair
//!    has both child pairs live. A `ZERO` child is dead; a leaf is dead only for
//!    `{Pos, Neg}`; a level that is marginal in `f` is a count (always live, no
//!    descent); a level that is marginal in `care` is `⊤` for liveness, so `f` is
//!    walked under `⊤` below it. Every pair is scanned (no early exit): an
//!    `f`-pair is marked live when it is live against *some* care pair, and an
//!    `f`-node when some pair of it is.
//! 3. If the root pair is dead, `f ∧ care ≡ ⊥` → `False`. If every node and pair
//!    reachable from `f`'s root is live → `Unchanged`. Otherwise `DeadRebuilder`
//!    re-emits the live subgraph (marginal levels verbatim) and the orphan prune
//!    reclaims children stranded by a collapsed partner → `Shrunk`.
//!
//! The walk is stack-driven (no recursion), visits at most `|f| · |care|` node
//! pairs, and is not budgeted or deadline-aware: unlike the apply engine it
//! never returns `OverBudget`. It has no solver caller, so simplicity wins here;
//! the apply engine carries no restrict-specific code.

use crate::engine::Engine;
use std::collections::HashMap;
use std::sync::Arc;

use crate::reduce::{minimize, try_minimize, MinimizeOptions, MinimizePasses};
use crate::diagram::{InputPair, LocalNodeIdx, Tdd, TddLevel, TddNodeId, ZERO, take_levels};
use crate::utils::sort_pairs;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Whether the caller guarantees `care` is already canonical (reduced/minimized).
/// `Yes` skips the O(|care|) `minimize(care)` prologue — sound because
/// `g ∧ care == f ∧ care` holds for ANY representation of care; minimize only
/// shrinks the walk. The flag is live in every mode (it always selects the
/// prologue), never inert.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CareCanonical {
    /// Caller guarantees `care` is canonical — skip `minimize(care)`.
    Yes,
    /// `care` may be non-canonical — run `minimize(care)` first.
    No,
}

/// Outcome of [`restrict`]. Lets the caller skip the dead epilogue (canonicalize +
/// size-compare + commit) on the common no-shrink case (`Unchanged`).
pub enum Restricted {
    /// Provably `g == f` (nothing reachable died, zero-/leaf-f early-out, or
    /// incomparable roots). No new diagram was built.
    Unchanged,
    /// A strict subgraph `g ⊊ f` (some pair died and the rebuild produced a
    /// smaller, count-correct-but-non-canonical `g`; caller canonicalizes).
    Shrunk(Tdd),
    /// `care` killed every model of `f` (`care ≡ ⊥` or `f ∧ care = ∅`): the
    /// canonical `⊥` is the smallest sound representative. A CHANGE, not a no-op.
    False(Tdd),
}

impl Restricted {
    /// Collapse to a concrete `g`, cloning `f` on `Unchanged`.
    pub fn into_tdd(self, f: &Tdd) -> Tdd {
        match self {
            Restricted::Unchanged => f.clone(),
            Restricted::Shrunk(g) | Restricted::False(g) => g,
        }
    }
}

/// The implementation behind [`Engine::restrict`](crate::Engine::restrict).
pub(crate) fn restrict_on(eng: &Engine, f: &Tdd, care: Tdd, care_canonical: CareCanonical) -> Restricted {
    if f.is_zero() {
        return Restricted::Unchanged;
    }
    let mut care = care;
    if care_canonical == CareCanonical::No {
        minimize(&mut care);
    }
    if care.is_zero() {
        // care ≡ ∅ ⇒ f ∧ care = ∅ ⇒ ⊥ is the smallest sound representative.
        return Restricted::False(Tdd::zero(&f.vtree));
    }
    let v0 = f.output.vtree;
    if f.vtree.node(v0).is_leaf() {
        // A literal has no internal pairs to drop.
        return Restricted::Unchanged;
    }
    // Both operands must share vtree structure; the walk reads indices in `f.vtree`.
    let r = f.vtree.lca(v0, care.output.vtree);
    if r != v0 && r != care.output.vtree {
        // Incomparable roots ⇒ disjoint variable regions ⇒ care can't constrain f.
        return Restricted::Unchanged;
    }
    let marks = Marking::walk(f, &care, r);
    if !marks.root_live {
        // care killed every model of f ⇒ f ∧ care = ∅.
        return Restricted::False(Tdd::zero(&f.vtree));
    }
    if marks.nothing_reachable_died(f) {
        return Restricted::Unchanged;
    }
    match marks.rebuild(eng, f) {
        Some(g) => Restricted::Shrunk(g),
        None => Restricted::Unchanged,
    }
}

/// One operand's reference into a vtree level: `None` = the operand is `⊤` here
/// (not yet rooted, or `care` marginal at this level), `Some(l)` = its node `l`.
type Ref = Option<LocalNodeIdx>;
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

/// Liveness marks over `f` produced by the `f × care` walk.
struct Marking {
    /// `[v.idx()][f-local]` — does this f-node survive under care?
    alive: Vec<Vec<bool>>,
    /// `[v.idx()][f-local]` — bit `k` set iff pair `k` of the f-node is live
    /// against some care pair; `u64::MAX` for a live node with more than 64
    /// pairs (no pair info: the rebuild keeps every pair of it).
    pair_alive: Vec<Vec<u64>>,
    /// Is the root pair live, i.e. is `f ∧ care` structurally non-false?
    root_live: bool,
}

impl Marking {
    /// Walk the reachable pairs of `f × care` from vtree node `r` (a root of one
    /// operand) and mark every live f-node and f-pair. Two phases: discover the
    /// pairs top-down with a work stack, then evaluate their liveness bottom-up.
    fn walk(f: &Tdd, care: &Tdd, r: VtreeIdx) -> Marking {
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
            let (lc, rc) = children(vtree, v);
            for (fl, fr) in refs(f, v, fo) {
                for (cl, cr) in refs(care, v, co) {
                    for (cv, a, b) in [(lc, fl, cl), (rc, fr, cr)] {
                        if let Child::Pair(k) = ctx.child(cv, a, b) {
                            if levels[cv.idx()].push(k) {
                                stack.push((cv, k));
                            }
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
                if any {
                    if let Some(fnode) = fo {
                        if f.levels[v.idx()].pairs_of_idx(fnode.idx()).len() > 64 {
                            pair_alive[v.idx()][fnode.idx()] = u64::MAX;
                        }
                    }
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
    fn nothing_reachable_died(&self, f: &Tdd) -> bool {
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
            let (lc, rc) = children(vtree, v);
            for p in pairs {
                stack.push((lc, p.left));
                stack.push((rc, p.right));
            }
        }
        true
    }

    /// Re-emit the live subgraph of `f` as a new diagram: `DeadRebuilder` keeps
    /// alive nodes and live pairs, marginal levels carry through verbatim, and
    /// the orphan prune makes the result arena-compact. `None` only when the
    /// prune runs out of memory (the caller then keeps `f`, which is sound).
    fn rebuild(self, eng: &Engine, f: &Tdd) -> Option<Tdd> {
        let nlev = f.vtree.num_nodes();
        let v0 = f.output.vtree;
        let marg: Vec<bool> = (0..nlev).map(|vi| f.levels[vi].is_marginal()).collect();
        // Dense per-level memo (both keys — vtree level, f-local index — are dense), a
        // per-level `Vec<u32>` with an UNVISITED sentinel replacing a hash map. Sized to
        // each level's f-node width; leaf levels are never indexed.
        let memo: Vec<Vec<u32>> = (0..nlev)
            .map(|vi| vec![DeadRebuilder::UNVISITED; f.levels[vi].nodes.len()])
            .collect();
        let mut rb = DeadRebuilder {
            f,
            vtree: &f.vtree,
            alive: self.alive,
            pair_alive: self.pair_alive.into_iter().map(Some).collect(),
            marg,
            out: take_levels(eng, nlev),
            memo,
        };
        let root = rb.rebuild(v0, f.output.local);
        let mut out = std::mem::take(&mut rb.out);
        // Marginalization fidelity: the rebuild only emits non-marginal levels (the
        // top of the diagram). Marginal levels (the bottom subtree — counts, no nodes)
        // are untouched by restriction (care constrains only counted vars), so carry
        // them through verbatim; their `marginal_counts`/`_big` slots back the marg-side
        // refs the rebuilt parents kept verbatim. Restore each rebuilt parent's
        // marg-inlined flags (push_internal_node starts them clear) so downstream count
        // decoders read its marg-side refs with the same inline/slot polarity as f.
        for vi in 0..nlev {
            if f.levels[vi].is_marginal() {
                out[vi] = f.levels[vi].clone();
            } else {
                out[vi].set_marg_inlined_left(f.levels[vi].marg_inlined_left());
                out[vi].set_marg_inlined_right(f.levels[vi].marg_inlined_right());
            }
        }
        let mut g = Tdd::with_levels(Arc::clone(&f.vtree), out, TddNodeId { vtree: v0, local: root });
        // The demand-driven rebuild emits a child before learning its pair partner
        // collapsed to ZERO, stranding that child as an arena orphan. Reclaim them so
        // the result is orphan-free (`size == reachable_pairs`) for any caller. Cheap
        // downward GC only (O(|g|)); reachable-twin contraction is `minimize`'s job.
        let prune_only = MinimizeOptions { passes: MinimizePasses::PruneOnly, ..Default::default() };
        if try_minimize(eng, &mut g, prune_only).is_err() {
            return None;
        }
        Some(g)
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

fn children(vtree: &Vtree, v: VtreeIdx) -> (VtreeIdx, VtreeIdx) {
    match *vtree.node(v) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => unreachable!("children() on a leaf vtree node"),
    }
}

/// Rebuild arena for [`restrict`]: keep each alive f-node, emitting the subset of
/// its pairs whose children both survive AND which produced ≥1 live product under
/// care. The `memo` keeps the map 1:1 with alive f-nodes, so f's DAG sharing
/// carries over and the result is a strict subgraph of f. Recursive: the depth
/// is bounded by the vtree height.
struct DeadRebuilder<'a> {
    f: &'a Tdd,
    vtree: &'a Vtree,
    /// `[v.idx()][f-local]` — does this f-node survive under care?
    alive: Vec<Vec<bool>>,
    /// `[v.idx()]` → per-node alive-PAIR bitmasks, or `None` = no pair info
    /// for the level (keep every pair of an alive node). Bit `k` of
    /// `pair_alive[v][i]` = pair `k` of f-node `i` produced ≥1 live product under
    /// care; `u64::MAX` = no info for that node.
    pair_alive: Vec<Option<Vec<u64>>>,
    /// `[v.idx()]` — is this level MARGINAL in f (counts, not nodes)? On a marg
    /// level a pair's child ref on that side is an inline/slot COUNT, not a node
    /// index — so it is kept verbatim, never recursed into or `alive`-indexed.
    marg: Vec<bool>,
    out: Vec<TddLevel>,
    /// `[v.idx()][f-local]` → rebuilt output-local index for that alive f-node, or
    /// `UNVISITED`. Dense per-level table (both keys dense) replacing a hash map.
    memo: Vec<Vec<u32>>,
}

impl DeadRebuilder<'_> {
    /// Memo "not yet rebuilt" sentinel. Must differ from every value `emit` can
    /// return — small output-local indices AND `ZERO` (= `u32::MAX`, minted for an
    /// alive f-node whose pairs all collapsed) — so it is `u32::MAX - 1`, a value no
    /// real level width can reach.
    const UNVISITED: u32 = u32::MAX - 1;

    fn is_leaf(&self, v: VtreeIdx) -> bool {
        self.vtree.node(v).is_leaf()
    }
    /// A child reference is kept iff it is a leaf label (always) or an alive internal
    /// node. `ZERO` is never kept.
    fn alive_child(&self, v: VtreeIdx, l: LocalNodeIdx) -> bool {
        if l == ZERO {
            return false;
        }
        self.is_leaf(v) || self.alive[v.idx()][l.idx()]
    }
    fn emit(&mut self, v: VtreeIdx, mut pairs: Vec<InputPair>) -> LocalNodeIdx {
        if pairs.is_empty() {
            return ZERO;
        }
        sort_pairs(&mut pairs);
        self.out[v.idx()].push_internal_node(&pairs)
    }
    fn rebuild(&mut self, v: VtreeIdx, fl: LocalNodeIdx) -> LocalNodeIdx {
        if self.is_leaf(v) || fl == ZERO {
            return fl;
        }
        let cached = self.memo[v.idx()][fl.idx()];
        if cached != Self::UNVISITED {
            return LocalNodeIdx(cached);
        }
        let (lc, rc) = children(self.vtree, v);
        // A child on a MARGINAL level is an inline/slot COUNT, not a node: it is
        // always present (carries the marginalized subtree's multiplicity) and is
        // copied verbatim — never `alive`-indexed (the count value would alias a
        // wild node index) and never recursed into (there are no child nodes).
        let l_marg = self.marg[lc.idx()];
        let r_marg = self.marg[rc.idx()];
        // `fr` is a Copy of the `&'a Tdd`, so `fp` borrows f (lifetime 'a), NOT self —
        // letting the recursive `self.rebuild` mutate while we iterate f's pairs.
        let fr = self.f;
        let fp = fr.levels[v.idx()].pairs_of_idx(fl.idx());
        // Pair-granular drop: a pair that produced no live product under care is
        // dead even when BOTH its children stay alive via other parents. Only
        // trusted when the mask is a real ≤64-pair mask (`u64::MAX` = no info). A
        // live node with a zero mask is impossible by construction.
        let pmask: Option<u64> = match self.pair_alive[v.idx()].as_deref() {
            Some(masks) if masks[fl.idx()] != u64::MAX && fp.len() <= 64 => {
                debug_assert!(
                    masks[fl.idx()] != 0,
                    "alive f-node with an all-dead pair mask at level {} idx {}",
                    v.idx(),
                    fl.idx()
                );
                Some(masks[fl.idx()])
            }
            _ => None,
        };
        let mut np: Vec<InputPair> = Vec::with_capacity(fp.len());
        for (k, p) in fp.iter().enumerate() {
            if let Some(m) = pmask {
                if (m >> k) & 1 == 0 {
                    continue;
                }
            }
            let l_ok = if l_marg { true } else { self.alive_child(lc, p.left) };
            let r_ok = if r_marg { true } else { self.alive_child(rc, p.right) };
            if l_ok && r_ok {
                let l = if l_marg { p.left } else { self.rebuild(lc, p.left) };
                let r = if r_marg { p.right } else { self.rebuild(rc, p.right) };
                // ZERO only arises on a rebuilt (non-marg) side; a marg-side count
                // ref never equals ZERO (bit 31 is reserved clear), so guard only
                // the sides we actually rebuilt.
                if (!l_marg && l == ZERO) || (!r_marg && r == ZERO) {
                    continue;
                }
                np.push(InputPair { left: l, right: r });
            }
        }
        let local = self.emit(v, np);
        debug_assert_ne!(local.0, Self::UNVISITED, "emitted local collided with the memo sentinel");
        self.memo[v.idx()][fl.idx()] = local.0;
        local
    }
}

/// Restrict `f` to the region `care` names, on a transient engine.
///
/// [`Engine::restrict`] is this operation on a caller's engine.
#[must_use]
pub fn restrict(f: &Tdd, care: Tdd, care_canonical: CareCanonical) -> Restricted {
    restrict_on(&Engine::new(), f, care, care_canonical)
}
