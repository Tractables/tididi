//! Levels, nodes, pairs, the reference encodings, the level pool, weights.
//!
//! This is storage, not algorithm: the operations over these types live in
//! [`crate::apply`], [`crate::marginal`], [`crate::reduce`] and
//! [`crate::restructure`], and the tree the levels are seated on is
//! [`crate::vtree`].
//!
//! Entry points: [`Tdd`] is the diagram, [`TddLevel`] one vtree node's storage
//! in it, [`ChildPair`] one element of a node's decomposition, and [`ChildDecoder`]
//! the decoding a pair side goes through when its child level is marginal.
//! [`TddBuilder`] assembles a diagram level by level, and [`EvalAlgebra`] defines
//! how a structural diagram is evaluated.
//!
//! The stored encoding is the traversal contract: a reader walks the levels
//! and pairs directly, with no view layer in between. Everything a reader may
//! rely on is stated on the types themselves; this section is the map.
//!
//! # Traversing a diagram
//!
//! - A [`Tdd`] has one [`TddLevel`] per vtree node, at `levels[t.idx()]`
//!   ([`Tdd::level`]); `output` names the node at the vtree root that denotes
//!   the function, or the [`ZERO`] sentinel for the constant-false function
//!   ([`Tdd::is_zero`]).
//! - Visit children before parents with `vtree.internal_bottomup()`, which
//!   yields each internal vtree node with its two children; the child levels of
//!   `t` are the levels of `vtree.children(t)`.
//! - A **leaf level** stores nothing. Its three nodes are implicit, at local
//!   indices [`ONE_LEAF_IDX`] (⊤), [`POS_LEAF_IDX`] (the variable) and
//!   [`NEG_LEAF_IDX`] (its negation), which are the [`LeafLabel`] values in
//!   that order.
//!   [`Tdd::reference_slot_count`] reports [`LEAF_WIDTH`] there. A leaf level that
//!   has been summed out reports [`TddLevel::is_marginal`] like any other
//!   marginal level, and a parent's side into it decodes through its
//!   [`TddLevel::child_decoder`].
//! - A **structural level** stores its nodes in slots; walk them with
//!   [`TddLevel::internal_inputs_iter`], which yields `(local index, pairs)` and
//!   skips tombstones, or read one node's pairs with [`TddLevel::pairs_of`].
//!   Each [`ChildPair`] indexes a node in the left child level and one in the
//!   right child level; the node denotes the disjoint union of its pairs'
//!   products.
//! - A **marginal level** has dropped its structure: it stores no nodes and
//!   [`TddLevel::marginal_counts`] holds one model count per node. In weighted
//!   mode ([`Tdd::set_weights`]) the level is weight-marginal instead
//!   ([`TddLevel::is_weight_marginal`]): `marginal_counts` is `None` and the
//!   values are [`WeightStore::level`]`(t.idx())` of the diagram's store. A
//!   pair whose child level is marginal does not hold a plain index on that
//!   side; decode it with the child's [`TddLevel::child_decoder`], which yields
//!   either the count itself or an index into the child's values.
//!
//! Invariants a reader may rely on:
//!
//! - every child index is in range for the child level's `reference_slot_count`
//!   (after [`ChildDecoder::child`] on a marginal side, a slot index is in range
//!   for the child's values);
//! - [`ZERO`] never appears in a pair — every live node in a structural diagram is satisfiable;
//! - marginality is downward-closed: every level below a marginal level is
//!   marginal or a leaf;
//! - in a diagram an operation produced, distinct nodes at a level denote
//!   pairwise disjoint functions, so a node's pairs denote disjoint products
//!   and its count is the sum over its pairs; a diagram built level by level
//!   ([`TddBuilder`]) has that property only if its author kept it;
//! - after [`minimize`](crate::Tdd::minimize), distinct nodes at a
//!   structural level have no remaining context twins and every live node is
//!   reachable from `output`; minimization does not repair a violation of the
//!   builder's determinism precondition.
//!
//! # Inspect structural pairs
//!
//! To inspect the representation without decoding children, walk the vtree and
//! ask each level for its live nodes:
//!
//! ```
//! use std::sync::Arc;
//! use tididi::{Tdd, Vtree};
//!
//! let tree = Arc::new(Vtree::balanced(3));
//! let f = Tdd::clause(&tree, [1, 2])?;
//! let mut total = 0;
//! for t in tree.bottomup() {
//!     for (_node, pairs) in f.level(t).internal_inputs_iter() {
//!         total += pairs.len();
//!     }
//! }
//! assert_eq!(total, f.pair_count());
//! # Ok::<(), tididi::OperationError>(())
//! ```
//!
//! # Decode child values
//!
//! The following worked count handles structural and count-marginal levels;
//! weight-marginal values require the diagram's attached store instead. It has one
//! marginalized subtree, so the walk decodes all four kinds of pair side: a
//! node of a leaf level, a node of a structural level, a value carried inline
//! at the reference, and a value held in the child's slot table.
//!
//! ```
//! use std::sync::Arc;
//! use num_bigint::BigUint;
//! use tididi::{or, Tdd};
//! use tididi::diagram::{ChildRef, EncodedChildRef, ChildDecoder, ValueRef};
//! use tididi::vtree::{Vtree, VtreeIdx};
//!
//! /// Which kinds of pair side the walk decoded.
//! #[derive(Default)]
//! struct Seen { leaf: bool, node: bool, inline: bool, slot: bool }
//!
//! fn side(s: EncodedChildRef, view: ChildDecoder, child: &[BigUint], child_is_leaf: bool, seen: &mut Seen)
//!     -> BigUint
//! {
//!     match view.child(s) {
//!         ChildRef::Value(ValueRef::Inline(k)) => { seen.inline = true; BigUint::from(k) }
//!         ChildRef::Value(ValueRef::Slot(j)) => { seen.slot = true; child[j as usize].clone() }
//!         ChildRef::Node(n) => {
//!             if child_is_leaf { seen.leaf = true } else { seen.node = true }
//!             child[n.idx()].clone()
//!         }
//!     }
//! }
//!
//! fn count(t: &Tdd, seen: &mut Seen) -> BigUint {
//!     if t.is_zero() { return BigUint::ZERO; }
//!     let vtree = t.vtree();
//!     let mut c: Vec<Vec<BigUint>> = (0..vtree.num_nodes())
//!         .map(|i| vec![BigUint::ZERO; t.reference_slot_count(VtreeIdx(i as u32))])
//!         .collect();
//!     for (leaf, _var) in vtree.leaf_bottomup() {
//!         c[leaf.idx()] = vec![2u32.into(), 1u32.into(), 1u32.into()]; // One, Pos, Neg
//!     }
//!     for (v, l, r) in vtree.internal_bottomup() {
//!         let lvl = t.level(v);
//!         if lvl.is_marginal() {
//!             // The level's structure was summed out: read its values instead.
//!             let counts = lvl.marginal_counts().unwrap();
//!             for (i, &n) in counts.iter().enumerate() {
//!                 c[v.idx()][i] = if n != u128::MAX { n.into() } else {
//!                     lvl.marginal_counts_big().unwrap().get(i).unwrap().clone()
//!                 };
//!             }
//!             continue;
//!         }
//!         let (lv, rv) = (t.level(l).child_decoder(), t.level(r).child_decoder());
//!         let (l_leaf, r_leaf) = (vtree.node(l).is_leaf(), vtree.node(r).is_leaf());
//!         for (i, pairs) in lvl.internal_inputs_iter() {
//!             let mut total = BigUint::ZERO;
//!             for p in pairs {
//!                 total += side(p.left, lv, &c[l.idx()], l_leaf, seen)
//!                     * side(p.right, rv, &c[r.idx()], r_leaf, seen);
//!             }
//!             c[v.idx()][i] = total;
//!         }
//!     }
//!     c[t.output().vtree.idx()][t.output().local.idx()].clone()
//! }
//!
//! // Sixty-four variables, so the left subtree carries thirty-two of them and
//! // its node values straddle the width a reference can carry inline.
//! let vtree = Arc::new(Vtree::balanced(64));
//! let mut f = or(Tdd::cube(&vtree, [1, 2, 3])?, Tdd::cube(&vtree, [4, 33])?)?;
//! f.minimize()?;
//! let expected = f.model_count()?;
//!
//! // Sum out the root's left subtree, bottom-up.
//! let (left, _right) = vtree.children(vtree.root());
//! let under = |mut t: VtreeIdx| loop {
//!     if t == left { return true }
//!     match vtree.node(t).parent() { Some(p) => t = p, None => return false }
//! };
//! let levels: Vec<VtreeIdx> =
//!     vtree.internal_bottomup_slice().iter().copied().filter(|&t| under(t)).collect();
//! f.marginalize_levels(&levels)?;
//!
//! let mut seen = Seen::default();
//! assert_eq!(count(&f, &mut seen), expected);
//! assert!(seen.leaf && seen.node && seen.inline && seen.slot);
//! # Ok::<(), tididi::OperationError>(())
//! ```

mod literal;
mod primitives;
mod packed;
pub(crate) mod marginal_ref;
mod build_error;
mod leaf_column;
pub(crate) mod builder;
mod level;
pub(crate) mod pool;
pub(crate) mod semiring;
mod tdd;
mod weights;

// primitives
pub use literal::Literal;
pub(crate) use literal::is_tautological;
pub use primitives::{
    ChildPair, EncodedChildRef, LeafLabel, NodeIdx, EncodedNode, TddNodeId,
    LEAF_WIDTH, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX, ZERO,
};
pub(crate) use primitives::{MultiPairRange, CHILD_PAIR_BYTES};

pub use packed::PairsIter;

// marginal
pub use marginal_ref::{CountOverflow, ChildRef, ChildDecoder, ValueRef, ValueRefError};
pub(crate) use marginal_ref::{
    MarginalSide,
    boundary_marginal_levels, boundary_marginal_levels_into, boundary_marginal_levels_of,
    for_each_side_ref_mut, remap_refs_into, ChildSide, Sides,
    MARGINAL_INLINE_MAX,
    tag_all_marginal_side_slots,
    assert_can_make_marginal, resolve_swapped_marginal_side,
};

// semiring
pub use semiring::{EvalAlgebra, LiteralWeights, RationalWeights, SignedLog, WeightValue};
pub use weights::{Arithmetic, WeightStore};

// level
pub use level::TddLevel;
pub(crate) use level::sort_pairs;

// pool
pub(crate) use pool::{return_levels, take_levels, try_take_levels};
pub(crate) use pool::{drop_pools, PoolSlot};
pub(crate) use pool::LevelPool;

// tdd
pub use build_error::TddBuildError;
pub use builder::{LevelView, TddBuilder};
pub use tdd::Tdd;
pub(crate) use leaf_column::{
    find_leaf_slot_by_value, leaf_canon_map, leaf_column_vals, leaf_count, LEAF_COUNTS,
};
pub(crate) use tdd::{Changed, Dirty};

#[cfg(test)]
mod tests;
