//! Disjoining a cube: the spine walk's other emit.
//!
//! A cube `M` and its negated clause `¬M` name the same two functions at every
//! level, with the lanes swapped: the clause's `d_t` — none of its literals
//! satisfied inside subtree `t` — *is* the cube's restriction `M_t`, and its
//! `c_t` is the complement. So one walk computes, for every node `a` of every
//! spine level, both `a ∖ M_t` and `a ∩ M_t`, and
//!
//! ```text
//! f ∨ M  =  (f ∖ M)  ⊎  M
//! ```
//!
//! is a **disjoint** union: the conjunction's own output plus one more pair at
//! the output node, naming `M`. The conjunction emits the `(c,c)`, `(c,d)` and
//! `(d,c)` cells of each node and never `(d,d)`, so that cell is free at every
//! level and the level stays a partition.
//!
//! What `M` needs is a node per level, which is the **cube chain**. It is not
//! `M_t` itself — that would contain the `a ∩ M_t` nodes the conjunction has
//! just emitted and break structural determinism. The canonical blocks are
//! `a_i ∖ M_t`, `a_i ∩ M_t` and `M_t ∖ ⋃a_i`, so the chain's node at level `t`
//! is the `d_t` node that already denotes `M_t` when some node meets the cube,
//! and a fresh single-pair node otherwise.
//!
//! This is exact when the cube fixes every variable of the vtree: `M_t` is one
//! value, so by structural determinism at most one node of a level meets it and
//! the chain adds at most one node and one pair per level. A cube that leaves a
//! variable free has `M_t = ⊤` over the free subtrees, which is not a node of a
//! diagram whose nodes are disjoint but not exhaustive; expressing it needs the
//! free subtrees made full, so those cubes take the complement route instead.

use super::*;

use crate::apply::negate::negate_on;
use crate::reduce::ReductionPlan;

/// The cube's node at each level, built bottom-up beside the spine walk.
pub(super) struct CubeChain {
    /// Per level, the index of the node denoting the cube's restriction to
    /// that subtree, or [`NO_PRODUCT`] until the level is closed. Leaf entries
    /// are the cube's leaf labels and are seeded by [`CubeChain::new`].
    m_idx: Vec<u32>,
}

impl CubeChain {
    /// Seed the chain from the cube's leaves. `clause` is the cube's negation,
    /// so each literal's complement is the label the cube assigns.
    pub(super) fn new(
        lim: &crate::limits::Limits,
        vtree: &Vtree,
        clause: &[(Literal, VtreeIdx)],
    ) -> Result<Self, OperationError> {
        let mut m_idx = Vec::new();
        lim.try_resize(&mut m_idx, vtree.num_nodes(), NO_PRODUCT)?;
        for &(lit, t) in clause {
            m_idx[t.idx()] = if lit.sign { NEG_LEAF_IDX.0 } else { POS_LEAF_IDX.0 };
        }
        Ok(CubeChain { m_idx })
    }

    /// The pair denoting the cube at level `t`: its two children's chain nodes.
    /// Both are closed before `t` is reached, since the walk is bottom-up.
    pub(super) fn pair_at(&self, vtree: &Vtree, t: VtreeIdx) -> ChildPair {
        let (left, right) = vtree.children(t);
        debug_assert!(self.m_idx[left.idx()] != NO_PRODUCT && self.m_idx[right.idx()] != NO_PRODUCT,
            "the cube chain reached level {t:?} before its children");
        ChildPair::new(
            EncodedChildRef::from_raw(self.m_idx[left.idx()]),
            EncodedChildRef::from_raw(self.m_idx[right.idx()]),
        )
    }

    /// Record level `t`'s chain node once the level has been rebuilt.
    ///
    /// `lanes` is the level's own `cd_map` block, one entry per node the level
    /// held before the rebuild. A non-empty `d_t` lane there denotes `M_t`,
    /// and there is at most one: the nodes of a level are disjoint and a
    /// complete cube's `M_t` is a single value. With none, no node of the
    /// level meets the cube and `M_t` is minted as a fresh single-pair node.
    pub(super) fn close_level(
        &mut self,
        eng: &Engine,
        t: VtreeIdx,
        vtree: &Vtree,
        level: &mut TddLevel,
        lanes: &[[u32; 2]],
    ) -> Result<(), OperationError> {
        let mut found = NO_PRODUCT;
        for entry in lanes {
            if entry[1] == NO_PRODUCT { continue; }
            debug_assert_eq!(found, NO_PRODUCT,
                "two nodes of level {t:?} meet the cube; structural determinism broken");
            found = entry[1];
            if !cfg!(debug_assertions) { break; }
        }
        if found == NO_PRODUCT {
            let pair = self.pair_at(vtree, t);
            found = level.push_node(eng.limits(), &[pair])?.0;
        }
        self.m_idx[t.idx()] = found;
        Ok(())
    }
}

/// Disjoin `cube` into `f`, consuming the operand. The implementation behind
/// [`Engine::or_cube`](crate::Engine::or_cube).
///
/// # Errors
///
/// Returns the [`OperationError`] the disjunction stopped on.
pub(crate) fn disjoin_cube_owned(eng: &Engine, f: Tdd, cube: &[Literal]) -> Result<Tdd, OperationError> {
    spine_walk(eng, f, cube, true)
}

/// `f ∨ M` as `¬(¬f ∧ ¬M)` — two make-full passes rather than the three a
/// general disjunction runs, and the route for a cube the spine walk's chain
/// cannot name. `clause` is `¬M` normalized and non-empty, and `f` has
/// structure at its leaves. The result is minimized.
pub(super) fn disjoin_cube_by_complement(eng: &Engine, f: Tdd, clause: &[(Literal, VtreeIdx)]) -> Result<Tdd, OperationError> {
    f.require_structure()?;
    let not_f = negate_on(eng, f)?;
    let mut rest = conjoin_normalized(eng, not_f, clause)?;
    eng.reduce(&mut rest, ReductionPlan::default())?;
    let mut out = negate_on(eng, rest)?;
    eng.reduce(&mut out, ReductionPlan::default())?;
    Ok(out)
}

#[cfg(test)]
#[path = "tests/cube.rs"]
mod tests;
