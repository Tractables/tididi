//! Re-emitting the live subgraph as a new diagram.

use std::sync::Arc;

use crate::engine::Engine;
use crate::limits::ApplyError;
use crate::reduce::{try_minimize, MinimizeOptions, MinimizeScope};
use crate::diagram::{InputPair, NodeIdx, Tdd, TddLevel, TddNodeId, ZERO, take_levels};
use crate::diagram::sort_pairs;
use crate::vtree::{Vtree, VtreeIdx};

use super::Marking;

impl Marking {
    /// Re-emit the live subgraph of `f` as a new diagram: `DeadRebuilder` keeps
    /// alive nodes and live pairs, marginal levels carry through verbatim, and
    /// the orphan prune makes the result arena-compact. The prune is the one
    /// step an armed limit can cut, and its error is the operation's.
    pub(super) fn rebuild(self, eng: &Engine, f: &Tdd) -> Result<Tdd, ApplyError> {
        let nlev = f.vtree.num_nodes();
        let v0 = f.output.vtree;
        let marginal: Vec<bool> = (0..nlev).map(|vi| f.levels[vi].is_marginal()).collect();
        // Dense per-level memo, sized to each level's f-node width; leaf
        // levels are never indexed.
        let memo: Vec<Vec<u32>> = (0..nlev)
            .map(|vi| vec![DeadRebuilder::UNVISITED; f.levels[vi].nodes.len()])
            .collect();
        let mut rb = DeadRebuilder {
            f,
            vtree: &f.vtree,
            alive: self.alive,
            pair_alive: self.pair_alive,
            marginal,
            out: take_levels(eng, nlev),
            memo,
        };
        let root = rb.rebuild(v0, f.output.local);
        let mut out = std::mem::take(&mut rb.out);
        // Marginal levels carry through verbatim: their stores back the
        // marginal-side refs the rebuilt parents kept. Each rebuilt parent
        // gets its marginal-inlined flags back (`push_internal_node` starts
        // them clear) so readers decode its marginal-side refs as in `f`.
        #[allow(clippy::needless_range_loop)]
        for vi in 0..nlev {
            if f.levels[vi].is_marginal() {
                out[vi] = f.levels[vi].clone();
            } else {
                out[vi].set_marginal_inlined_left(f.levels[vi].marginal_inlined_left());
                out[vi].set_marginal_inlined_right(f.levels[vi].marginal_inlined_right());
            }
        }
        let mut g = Tdd::from_levels_unchecked(Arc::clone(&f.vtree), out, TddNodeId { vtree: v0, local: root });
        // The rebuild emits a child before learning its pair partner collapsed
        // to `ZERO`, stranding that child as an arena orphan; the prune
        // reclaims them so the result is orphan-free.
        let prune_only = MinimizeOptions { passes: MinimizeScope::PruneOnly, ..Default::default() };
        try_minimize(eng, &mut g, prune_only)?;
        Ok(g)
    }
}

/// Rebuild arena for [`restrict()`](super::restrict): keep each alive f-node, emitting the subset of
/// its pairs whose children both survive and which produced ≥1 live product under
/// care. The `memo` keeps the map 1:1 with alive f-nodes, so the sharing structure of f
/// carries over and the result is a strict subgraph of f. Recursive: the depth
/// is bounded by the vtree height.
struct DeadRebuilder<'a> {
    f: &'a Tdd,
    vtree: &'a Vtree,
    /// `[v.idx()][f-local]` — does this f-node survive under care?
    alive: Vec<Vec<bool>>,
    /// `[v.idx()][f-local]` — bit `k` set iff pair `k` of the f-node produced
    /// at least one live product under care; `u64::MAX` = no info for that
    /// node (keep every pair of it).
    pair_alive: Vec<Vec<u64>>,
    /// `[v.idx()]` — is this level marginal in f (counts, not nodes)? On a marginal
    /// level a pair's child ref on that side is an inline/slot count, not a node
    /// index — so it is kept verbatim, never recursed into or `alive`-indexed.
    marginal: Vec<bool>,
    out: Vec<TddLevel>,
    /// `[v.idx()][f-local]` → rebuilt output-local index for that alive f-node, or
    /// `UNVISITED`.
    memo: Vec<Vec<u32>>,
}

impl DeadRebuilder<'_> {
    /// Memo "not yet rebuilt" sentinel: differs from every value `emit` can
    /// return, output-local indices and `ZERO` (`u32::MAX`) included.
    const UNVISITED: u32 = u32::MAX - 1;

    fn is_leaf(&self, v: VtreeIdx) -> bool {
        self.vtree.node(v).is_leaf()
    }
    /// A child reference is kept iff it is a leaf label (always) or an alive internal
    /// node. `ZERO` is never kept.
    fn alive_child(&self, v: VtreeIdx, l: NodeIdx) -> bool {
        if l == ZERO {
            return false;
        }
        self.is_leaf(v) || self.alive[v.idx()][l.idx()]
    }
    fn emit(&mut self, v: VtreeIdx, mut pairs: Vec<InputPair>) -> NodeIdx {
        if pairs.is_empty() {
            return ZERO;
        }
        // Sort but do not dedup: once any level is marginal a pair list is a
        // multiset, and equal pairs carry the multiplicity the count
        // recurrence needs.
        sort_pairs(&mut pairs);
        self.out[v.idx()].push_internal_node(&pairs)
    }
    fn rebuild(&mut self, v: VtreeIdx, fl: NodeIdx) -> NodeIdx {
        if self.is_leaf(v) || fl == ZERO {
            return fl;
        }
        let cached = self.memo[v.idx()][fl.idx()];
        if cached != Self::UNVISITED {
            return NodeIdx(cached);
        }
        let (lc, rc) = self.vtree.children(v);
        // A child ref on a marginal level is a value ref, not a node index: it
        // is copied verbatim, never `alive`-indexed and never recursed into.
        let l_marginal = self.marginal[lc.idx()];
        let r_marginal = self.marginal[rc.idx()];
        // `fr` is a Copy of the `&'a Tdd`, so `fp` borrows f (lifetime 'a), not self —
        // letting the recursive `self.rebuild` mutate while we iterate f's pairs.
        let fr = self.f;
        let fp = fr.levels[v.idx()].pairs_of_idx(fl.idx());
        // A pair that produced no live product under care is dead even when
        // both its children stay alive via other parents; the mask is only
        // trusted for a node of at most 64 pairs (`u64::MAX` = no info).
        let mask = self.pair_alive[v.idx()][fl.idx()];
        let pmask: Option<u64> = if mask != u64::MAX && fp.len() <= 64 {
            debug_assert!(
                mask != 0,
                "alive f-node with an all-dead pair mask at level {} idx {}",
                v.idx(),
                fl.idx()
            );
            Some(mask)
        } else {
            None
        };
        let mut np: Vec<InputPair> = Vec::with_capacity(fp.len());
        for (k, p) in fp.iter().enumerate() {
            if let Some(m) = pmask
                && (m >> k) & 1 == 0 {
                    continue;
                }
            let l_ok = if l_marginal { true } else { self.alive_child(lc, p.left) };
            let r_ok = if r_marginal { true } else { self.alive_child(rc, p.right) };
            if l_ok && r_ok {
                let l = if l_marginal { p.left } else { self.rebuild(lc, p.left) };
                let r = if r_marginal { p.right } else { self.rebuild(rc, p.right) };
                // `ZERO` only arises on a rebuilt side; a marginal-side ref
                // keeps bit 31 clear.
                if (!l_marginal && l == ZERO) || (!r_marginal && r == ZERO) {
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
