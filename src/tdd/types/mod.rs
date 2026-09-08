//! The diagram's storage types: [`Tdd`], [`TddLevel`], [`TddNodeData`],
//! [`InputPair`], and the marginal-reference decoding ([`resolve_marg_ref`]).
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
//!   [`NEG_LEAF_IDX`] (its negation); [`LeafLabel::from_idx`] names them.
//!   [`Tdd::effective_width`] reports [`LEAF_WIDTH`] there.
//! - A **structural level** stores its nodes in `nodes`; walk them with
//!   [`TddLevel::internal_inputs_iter`], which yields `(local index, pairs)` and
//!   skips tombstones, or read one node's pairs with [`TddLevel::pairs_of`].
//!   Each [`InputPair`] indexes a node in the left child level and one in the
//!   right child level; the node denotes the disjoint union of its pairs'
//!   products.
//! - A **marginal level** has dropped its structure: `nodes` and `pairs` are
//!   empty and `marginal_counts` holds one model count per node. A pair whose
//!   child level is marginal does not hold a plain index on that side; decode
//!   it with [`resolve_marg_ref`], which yields either the count itself or an
//!   index into the child's `marginal_counts`.
//!
//! Invariants a reader may rely on:
//!
//! - every child index is in range for the child level's `effective_width`
//!   (after `resolve_marg_ref` on a marginal side, the index is in range for
//!   `marginal_counts`);
//! - [`ZERO`] never appears in a pair — every stored node is satisfiable;
//! - marginality is downward-closed: every level below a marginal level is
//!   marginal or a leaf;
//! - after [`minimize`](crate::tdd::minimize::minimize), distinct nodes at a
//!   level denote distinct functions and every node is reachable from
//!   `output`; a diagram built by hand ([`Tdd::try_from_levels`]) has neither
//!   guarantee until minimized.
//!
//! A bottom-up model count written against this contract (the same walk, with
//! comments, is `examples/traverse_count.rs`):
//!
//! ```
//! use std::sync::Arc;
//! use num_bigint::BigUint;
//! use tididi::tdd::Tdd;
//! use tididi::tdd::types::{MargResolved, resolve_marg_ref};
//! use tididi::vtree::{Vtree, VtreeIdx};
//!
//! fn count(t: &Tdd) -> BigUint {
//!     if t.is_zero() { return BigUint::ZERO; }
//!     let mut c: Vec<Vec<BigUint>> = (0..t.vtree.num_nodes())
//!         .map(|i| vec![BigUint::ZERO; t.effective_width(VtreeIdx(i as u32))])
//!         .collect();
//!     for (leaf, _var) in t.vtree.leaf_bottomup() {
//!         c[leaf.idx()] = vec![2u32.into(), 1u32.into(), 1u32.into()]; // One, Pos, Neg
//!     }
//!     for (v, l, r) in t.vtree.internal_bottomup() {
//!         let lvl = t.level(v);
//!         if lvl.is_marginal() {
//!             let counts = lvl.marginal_counts.as_ref().unwrap();
//!             for (i, &n) in counts.iter().enumerate() {
//!                 c[v.idx()][i] = if n != u128::MAX { n.into() } else {
//!                     lvl.marginal_counts_big.as_ref().unwrap().get(i).unwrap().clone()
//!                 };
//!             }
//!             continue;
//!         }
//!         let (lm, rm) = (t.level(l).is_marginal(), t.level(r).is_marginal());
//!         for (i, pairs) in lvl.internal_inputs_iter() {
//!             let mut total = BigUint::ZERO;
//!             for p in pairs {
//!                 let side = |raw, marg, child: &Vec<BigUint>| match resolve_marg_ref(raw, marg) {
//!                     MargResolved::Inline(k) => BigUint::from(k),
//!                     MargResolved::Index(j) => child[j].clone(),
//!                 };
//!                 total += side(p.left.0, lm, &c[l.idx()]) * side(p.right.0, rm, &c[r.idx()]);
//!             }
//!             c[v.idx()][i] = total;
//!         }
//!     }
//!     c[t.output.vtree.idx()][t.output.local.idx()].clone()
//! }
//!
//! let vtree = Arc::new(Vtree::balanced(3));
//! let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
//! assert_eq!(count(&f), f.model_count());
//! ```

mod primitives;
mod packed;
mod marg;
mod level;
mod pool;
mod tdd;

// primitives
pub use primitives::{
    InputPair, LeafLabel, LocalNodeIdx, TddNodeData, TddNodeId,
    LEAF_WIDTH, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX, ZERO,
};
pub(crate) use primitives::{ExtMulti, INPUT_PAIR_BYTES};

pub use packed::PairsIter;

// marg
pub use marg::{
    BigSide, MARG_INLINE_MAX, MARG_OVERFLOW_TAG, MARG_VALUE_MASK,
    MargRef, MargResolved, resolve_marg_ref,
};
pub(crate) use marg::{
    decode_marg_coord, marg_inline_max,
    tag_all_marg_side_slots, tag_all_marg_side_slots_at,
    assert_can_make_marginal, resolve_swapped_marg_side,
};

// marg test-only override hooks — dev/test profiles only (compiled out of
// plain release); `pub` so the downstream compiler crate's tests can reach
// them across the crate boundary.
#[cfg(any(test, debug_assertions))]
#[doc(hidden)]
pub use marg::set_marg_inline_max;

// level
pub use level::TddLevel;

// pool
pub use pool::{return_levels, take_levels};
pub(crate) use pool::{MAX_LEVEL_ARENA_BYTES, drop_pools, return_levels2};
#[cfg(test)]
pub(crate) use pool::{reset_level, LEVELS_POOL, LEVELS_POOL2};

// tdd
pub use tdd::{Tdd, TddBuildError};

#[cfg(test)]
#[path = "../types_tests.rs"]
mod tests;
