//! Conjunction of two diagrams on different projections of one home vtree.
//!
//! The result lives on the smallest projection holding both leaf sets: the
//! larger operand's vtree when one set holds the other, the union projection
//! otherwise. The sweep is [`conjoin_onto`]'s.

use std::borrow::Cow;
use std::sync::Arc;

use super::{conjoin_cow, conjoin_onto, prepare_owned_weights, VtreeMask};
use crate::diagram::{return_levels, try_take_levels, Assembly, ChildPair, Dirty, Tdd, TddNodeId};
use crate::limits::OperationError;
use crate::restructure::EmbedError;
use crate::vtree::{LeafSets, Vtree, VtreeIdx};
use crate::Engine;

/// [`Engine::and`] of operands on different vtree allocations.
pub(super) fn conjoin_projections<'a>(eng: &Engine, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>) -> Result<Tdd, OperationError> {
    let into = match Vtree::leaf_sets(f.vtree(), g.vtree()).ok_or(OperationError::VtreeMismatch)? {
        LeafSets::Equal => {
            reseat_either(eng, &mut f, &mut g)?;
            prepare_owned_weights(&mut f, &mut g)?;
            return conjoin_cow(eng, f, g, VtreeMask::default());
        }
        LeafSets::AInB => Arc::clone(g.vtree()),
        LeafSets::BInA => Arc::clone(f.vtree()),
        LeafSets::Union(into) => {
            if let Some((f_map, g_map, f_left)) = Vtree::split_at_root(&into, f.vtree(), g.vtree())
                && joinable(&f)
                && joinable(&g)
            {
                return join_at_root(eng, f, &f_map, g, &g_map, f_left, &into);
            }
            into
        }
    };
    conjoin_onto(eng, f, |v| v, g, |v| v, &into, false).map_err(|refused| operation_error(refused.error))
}

/// Put one operand on the other's vtree, of the same layout: an owned one is
/// reseated, else a copy of `g` is.
fn reseat_either<'a>(eng: &Engine, f: &mut Cow<'a, Tdd>, g: &mut Cow<'a, Tdd>) -> Result<(), OperationError> {
    let (from, onto) = match (matches!(f, Cow::Owned(_)), matches!(g, Cow::Owned(_))) {
        (_, true) => (g, Arc::clone(f.vtree())),
        (true, false) => (f, Arc::clone(g.vtree())),
        (false, false) => {
            *g = Cow::Owned(g.try_clone_on(eng)?);
            (g, Arc::clone(f.vtree()))
        }
    };
    // Safety: equal leaf sets of one home give the same layout, node for
    // node, with the same variable ids.
    unsafe { from.to_mut().reseat_vtree_unchecked(&onto) };
    Ok(())
}

/// The conjunction's error for a refusal of [`conjoin_onto`] on operands of
/// one home, where the placement maps are the identity and cannot fail.
fn operation_error(error: EmbedError) -> OperationError {
    match error {
        EmbedError::Operation(error) => error,
        EmbedError::SourceWeights(_) | EmbedError::DestinationWeights(_) => OperationError::IncompatibleWeights,
        _ => OperationError::VtreeMismatch,
    }
}

/// Whether [`join_at_root`] takes `tdd`: it moves levels and pairs two
/// outputs, so the operand owes no reduction and carries no weights or
/// summed-out level.
fn joinable(tdd: &Tdd) -> bool {
    tdd.weights.is_none() && tdd.dirty.is_empty() && !tdd.is_zero() && !tdd.has_marginal_level()
}

/// The conjunction of `f` and `g` whose leaf sets lie under the two children
/// of `into`'s root: their levels moved (owned) or copied (borrowed) to their
/// images by `f_map` and `g_map`, and one root node pairing the two outputs.
fn join_at_root(
    eng: &Engine,
    f: Cow<'_, Tdd>,
    f_map: &[VtreeIdx],
    g: Cow<'_, Tdd>,
    g_map: &[VtreeIdx],
    f_left: bool,
    into: &Arc<Vtree>,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let closed = f.levels.is_closed() && g.levels.is_closed();
    let pair = match f_left {
        true => ChildPair::new(f.output.local, g.output.local),
        false => ChildPair::new(g.output.local, f.output.local),
    };
    let mut assembly = Assembly::from_levels(eng, Arc::clone(into), try_take_levels(eng, into.num_nodes())?, None);
    let (levels, _) = assembly.parts_mut();
    for (operand, map) in [(f, f_map), (g, g_map)] {
        match operand {
            Cow::Owned(mut moved) => {
                let source = &mut moved.levels[..];
                for (from, &to) in map.iter().enumerate() {
                    std::mem::swap(&mut levels[to.idx()], &mut source[from]);
                }
                return_levels(eng, std::mem::take(&mut moved.levels).into_vec());
            }
            Cow::Borrowed(source) => {
                for (from, &to) in map.iter().enumerate() {
                    source.levels[from].try_copy_over(lim, &mut levels[to.idx()])?;
                }
            }
        }
    }
    let root = into.root();
    let local = levels[root.idx()].push_node(lim, &[pair])?;
    let changed = [root];
    assembly.finish_with(TddNodeId { vtree: root, local }, Dirty::default(), &[], closed.then_some(&changed[..]))
}
