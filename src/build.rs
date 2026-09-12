//! Constants, literals and clauses as diagrams.
//!
//! These are the leaves of every compilation: everything else is built by
//! combining them with [`crate::apply`] and reducing with [`crate::reduce`].
//!
//! Entry points: [`Tdd::one`] and [`Tdd::zero`] are the two constants;
//! [`Engine::cube`] builds a conjunction of literals. A single clause is
//! [`Tdd::clause`], the clause conjoined into ⊤ by
//! [`crate::apply::apply_and_clause`].

use std::sync::Arc;

use crate::vtree::Vtree;
use crate::engine::Engine;

use crate::diagram::{self, *};

/// Build a diagram computing the constant-false function (no assignment satisfies it).
/// Output points to the `ZERO` sentinel (`u32::MAX`) — no actual nodes are created.
pub(crate) fn constant_zero(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let levels = diagram::take_levels(eng, vtree.num_nodes());
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: ZERO },
    )
}

/// Build a diagram computing the constant-true function (all assignments satisfy it).
/// Width 1 at every internal vtree level (only one node at index 0).
/// Leaf levels are marginal (no stored nodes); One is at index 0 (`ONE_LEAF_IDX`).
pub(crate) fn constant_one(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let mut levels = diagram::take_levels(eng, vtree.num_nodes());

    // Internal levels: each has one node pairing the child's "true" node.
    // Leaf children reference One (`ONE_LEAF_IDX`); internal children reference
    // their single node at index 0.
    for (t, left, right) in vtree.internal_bottomup() {
        let left_child_idx = if vtree.node(left).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let right_child_idx = if vtree.node(right).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let pair = InputPair { left: left_child_idx, right: right_child_idx };
        levels[t.idx()].push_internal_node(&[pair]);
    }

    // Output: One (for single-variable vtrees) or the sole internal node (index 0).
    let out_local = if vtree.node(vtree.root()).is_leaf() {
        ONE_LEAF_IDX
    } else {
        NodeIdx(0)
    };
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

/// Build the cube diagram: one width-1 node per internal vtree node, whose
/// pair names the leaf label the cube assigns to each side's subtree.
///
/// Bottom-up, so the pair a node writes names children that already exist.
fn cube_to_tdd(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    literals: impl IntoIterator<Item = impl Into<Literal>>,
) -> Tdd {
    let mut label = vec![ONE_LEAF_IDX; vtree.num_nodes()];
    for lit in literals {
        let lit: Literal = lit.into();
        let leaf = vtree
            .leaf_of(lit.var)
            .expect("the cube names a variable this vtree has no leaf for");
        assert_eq!(
            label[leaf.idx()], ONE_LEAF_IDX,
            "the cube names variable {:?} twice",
            lit.var,
        );
        label[leaf.idx()] = if lit.positive { POS_LEAF_IDX } else { NEG_LEAF_IDX };
    }
    let mut b = Tdd::build(eng, vtree);
    for (t, left, right) in vtree.internal_bottomup() {
        label[t.idx()] = b.push(t, &[InputPair {
            left: label[left.idx()],
            right: label[right.idx()],
        }]);
    }
    let root = vtree.root();
    b.finish(TddNodeId { vtree: root, local: label[root.idx()] })
        .expect("a cube names one node per internal level and seats the root on it")
}

impl Tdd {
    /// The constant-true function over `vtree`: every assignment satisfies it.
    ///
    /// Width 1 at every internal vtree level; the leaf levels are marginal.
    pub fn one(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().one(vtree)
    }

    /// The constant-false function over `vtree`: no assignment satisfies it.
    ///
    /// The output points at the `ZERO` sentinel, so no nodes are created.
    pub fn zero(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().zero(vtree)
    }
}

/// The construction entry points on a caller's engine.
impl crate::engine::Engine {
    /// The constant-true function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn one(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_one(self, vtree)
    }

    /// The constant-false function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn zero(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_zero(self, vtree)
    }

    /// The conjunction of `literals` over `vtree`: one width-1 node per
    /// internal vtree node, so the whole diagram is one path.
    ///
    /// A variable no literal mentions is free — the cube says nothing about
    /// it, so both of its values satisfy the result. Each item is converted
    /// with [`Into<Literal>`], so plain integers use the 1-based DIMACS sign
    /// convention.
    ///
    /// # Panics
    ///
    /// If `literals` names a variable twice, or names one `vtree` has no leaf
    /// for.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Engine;
    /// use tididi::vtree::Vtree;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = eng.cube(&vtree, [1, -2]); // x1 ∧ ¬x2, with x3 free
    /// assert_eq!(f.model_count(), 2u32.into());
    /// ```
    #[must_use]
    pub fn cube(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Tdd {
        crate::build::cube_to_tdd(self, vtree, literals)
    }
}

#[cfg(test)]
mod tests;
