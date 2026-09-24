//! Collapsing a fully quantified subtree while the product is built.
//!
//! This is [`Quantification::FusedSubtrees`](crate::Quantification::FusedSubtrees),
//! which is not the default. It reaches a level by walking its whole product
//! grid, so it wins exactly where that grid was going to be walked and
//! materialized anyway, and loses where the level had a cheaper route — the
//! sparse one, or an identity. Making the choice per level is the open problem
//! stated at the bottom of this comment.
//!
//! [`Engine::and_exists`](crate::Engine::and_exists) removes a set of leaves
//! from `f ∧ g`. A vtree node every leaf of which is removed contributes one
//! bit to the answer and nothing else: over its variables, `∃(f_i ∧ g_j)` is
//! `⊤` when the two are jointly satisfiable and `⊥` when they are not. So the
//! sweep does not build that subtree at all. It decides each of the level's
//! product cells by one satisfiability test — which stops at the first
//! surviving pair, because `∃` absorbs everything after it — and writes a
//! level holding the single `⊤` node that every live cell names.
//!
//! Every level of the subtree is built this way, bottom-up, so a collapsed
//! level's own pair `(⊤, ⊤)` always names a child that holds a `⊤` at index 0
//! — a collapsed level does, and a leaf level always does. What the subtree
//! costs is therefore one early-exiting satisfiability test per product cell
//! and one node per level, in place of the emit of every surviving product.
//!
//! What the sweep leaves behind is the product with those subtrees replaced by
//! `⊤`, which is what the unfused route's quantification sweep reaches at
//! those levels. The rest of the quantification is therefore unchanged: the
//! owner-set regroup in [`crate::apply::project`] runs over the ancestors
//! exactly as it would have, and re-establishes the level-wide disjointness
//! the collapse breaks. The one thing that sweep cannot see for itself is that
//! a collapsed level *had* more than one node, so it is told which levels were
//! collapsed: a level already reduced to `⊤` still owes its ancestors the
//! regroup.
//!
//! **Open.** The choice is all-or-nothing per quantified subtree, because a
//! level built here reads its children's grids positionally and a level built
//! by the sparse route has no grid to read. Deciding per level would mean
//! writing the satisfiability bit through the same route selection and child
//! lookup the emitting build uses, rather than beside them; until that is
//! done, a caller choosing this setting is choosing the dense walk for every
//! level of every subtree it collapses.

use crate::Engine;
use crate::diagram::{ChildPair, EncodedChildRef, ONE_LEAF_IDX, Tdd};
use crate::limits::{OperationError, PollGate};

use super::{ApplyRun, LevelShape, NO_PRODUCT};
use super::child_lookup::{ChildLookup, DenseLookup};
use crate::diagram::Sides;

/// The pair of a `⊤` node: the constant-true node sits at local index 0 of
/// every level, leaf or internal, so both sides name it.
const TRUE_PAIR: ChildPair = ChildPair {
    left: EncodedChildRef::from_raw(ONE_LEAF_IDX.0),
    right: EncodedChildRef::from_raw(ONE_LEAF_IDX.0),
};

/// Build a level every leaf below which is quantified: one satisfiability test
/// per product cell, and a level holding the single `⊤` node the live cells
/// name.
///
/// Both children are inside the same quantified subtree, so both were built by
/// this route or are leaves; either way their grids are dense and this reads
/// them positionally, as the dense cell walk does.
///
/// # Errors
///
/// A refused grid claim or node push, or the armed stop, polled per cell and
/// per visited operand pair.
pub(super) fn build_level_quantified(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let cells = fw.here * gw.here;
    let grid = run.products.arena.alloc(eng, ti, cells)?;
    run.products.arena.set_dense(ti, grid);
    let base = grid.idx();
    // The children's grids, read positionally as the dense cell walk reads
    // them.
    let sides = Sides {
        left: DenseLookup {
            base: run.products.arena.materialized(li).expect("a collapsed level's left child has a grid").idx(),
            stride: gw.left as u32,
        },
        right: DenseLookup {
            base: run.products.arena.materialized(ri).expect("a collapsed level's right child has a grid").idx(),
            stride: gw.right as u32,
        },
    };
    // This route emits at most one pair, so it arms no emit-growth bound; the
    // routed path's `open_level_arenas` makes the same call for the routes
    // that do not emit into `level.pairs`.
    lim.begin_level(None);

    let (f_level, g_level) = (f.level(t), g.level(t));
    let ApplyRun { levels, products, .. } = run;
    let level = &mut levels[ti];
    let slab = products.arena.slab_mut();
    slab[base..base + cells].fill(NO_PRODUCT);

    let mut gate = lim.gate();
    for i in 0..fw.here {
        let f_pairs = f_level.pairs_of_idx(i);
        if f_pairs.is_empty() {
            continue;
        }
        let row = base + i * gw.here;
        for j in 0..gw.here {
            gate.poll(1)?;
            let g_pairs = g_level.pairs_of_idx(j);
            if !cell_is_satisfiable(f_pairs, g_pairs, slab, &sides, &mut gate)? {
                continue;
            }
            if level.nodes.is_empty() {
                // The one node this level ever holds, minted on the first live
                // cell so a level with none stays empty and reads as `⊥`.
                level.push_node(eng.limits(), &[TRUE_PAIR])?;
            }
            slab[row + j] = ONE_LEAF_IDX.0;
        }
    }
    gate.flush()?;
    products.record_live(ti, level.slot_count());
    level.shrink_arrays();
    lim.level_settled(level.pairs.len() as u64);
    Ok(())
}

/// Whether `f_pairs ∧ g_pairs` has a model, which is what one cell of a
/// collapsed level contributes.
///
/// The walk returns on the first surviving product: a quantified subtree asks
/// only whether one exists, so everything after it is absorbed. That is the
/// whole difference between this and the emit walk, which has to visit every
/// product because every one of them is part of the answer.
#[inline]
fn cell_is_satisfiable(
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
    slab: &[u32],
    sides: &Sides<DenseLookup>,
    gate: &mut PollGate<'_>,
) -> Result<bool, OperationError> {
    for p1 in f_pairs {
        gate.poll(g_pairs.len() as u64)?;
        for p2 in g_pairs {
            if sides.left.get(slab, p1.left.0, p2.left.0) == NO_PRODUCT {
                continue;
            }
            if sides.right.get(slab, p1.right.0, p2.right.0) == NO_PRODUCT {
                continue;
            }
            return Ok(true);
        }
    }
    Ok(false)
}
