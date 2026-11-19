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
use crate::apply::extend;
use crate::build::constant_without_leaf;
use crate::vtree::{LeafSets, SameLayout, VarId, Vtree, VtreeIdx};
use crate::Engine;

/// [`Engine::and`] of operands on different vtree allocations.
pub(super) fn conjoin_projections<'a>(eng: &Engine, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>) -> Result<Tdd, OperationError> {
    if f.vtree().num_nodes() == 0 || g.vtree().num_nodes() == 0 {
        return conjoin_constant(eng, f, g);
    }
    let sets = Vtree::leaf_sets(f.vtree(), g.vtree())?;
    if f.has_marginal_level() || g.has_marginal_level() {
        check_summed_leaves(&f, &g)?;
    }
    let into = match sets {
        LeafSets::Equal(same) => {
            reseat_either(eng, &mut f, &mut g, &same)?;
            prepare_owned_weights(&mut f, &mut g)?;
            let mut out = conjoin_cow(eng, f, g, VtreeMask::default())?;
            crate::reduce::close_constant(eng, &mut out);
            return Ok(out);
        }
        LeafSets::AInB => Arc::clone(g.vtree()),
        LeafSets::BInA => Arc::clone(f.vtree()),
        LeafSets::Union(into) => {
            if let Some((f_map, g_map, f_left)) = Vtree::split_at_root(&into, f.vtree(), g.vtree())
                && joinable(&f)
                && joinable(&g)
            {
                crate::apply::shared_weights([&*f, &*g].into_iter())?;
                return join_at_root(eng, f, &f_map, g, &g_map, f_left, &into);
            }
            into
        }
    };
    let mut out = conjoin_onto(eng, f, |v| v, g, |v| v, &into).map_err(|refused| extend::operation_error(refused.error))?;
    crate::reduce::close_constant(eng, &mut out);
    Ok(out)
}

/// Refuse operands of which one has summed out a variable the other has a
/// leaf for; [`Vtree::leaf_sets`] refuses one inside the other's extent.
#[cold]
pub(super) fn check_summed_leaves(f: &Tdd, g: &Tdd) -> Result<(), OperationError> {
    check_summed_leaves_mapped(f, &|v| v, g, &|v| v)
}

/// [`check_summed_leaves`] with the variables of `f` and `g` renamed by
/// `f_map` and `g_map`.
#[cold]
pub(super) fn check_summed_leaves_mapped(
    f: &Tdd,
    f_map: &dyn Fn(VarId) -> VarId,
    g: &Tdd,
    g_map: &dyn Fn(VarId) -> VarId,
) -> Result<(), OperationError> {
    for ((a, a_map), (b, b_map)) in [((f, f_map), (g, g_map)), ((g, g_map), (f, f_map))] {
        if !a.has_marginal_level() {
            continue;
        }
        let mut vars: Vec<VarId> = b.vtree().leaf_bottomup().map(|(_, var)| b_map(var)).collect();
        vars.sort_unstable();
        for (leaf, var) in a.vtree().leaf_bottomup() {
            if a.level(leaf).is_marginal() && vars.binary_search(&a_map(var)).is_ok() {
                return Err(OperationError::MarginalLevel(leaf));
            }
        }
    }
    Ok(())
}

/// Refuse operands on one projection that have both summed out a region.
#[cold]
pub(super) fn check_shared_regions(f: &Tdd, g: &Tdd) -> Result<(), OperationError> {
    match f.vtree().regions().find(|&t| f.level(t).is_marginal() && g.level(t).is_marginal()) {
        Some(t) => Err(OperationError::MarginalLevel(t)),
        None => Ok(()),
    }
}

/// `f ∧ g` where one operand is a constant on the empty projection: false,
/// or the other operand.
pub(super) fn conjoin_constant(eng: &Engine, f: Cow<'_, Tdd>, g: Cow<'_, Tdd>) -> Result<Tdd, OperationError> {
    Vtree::leaf_sets(f.vtree(), g.vtree())?;
    let (constant, other) = match f.vtree().num_nodes() {
        0 => (f, g),
        _ => (g, f),
    };
    let weights = crate::apply::shared_weights([&*constant, &*other].into_iter())?;
    if constant.is_zero() || other.is_zero() {
        let mut zero = constant_without_leaf(&Vtree::empty_projection(constant.vtree().link_home().expect("a vtree with no node is a projection")), false);
        zero.weights = weights;
        return Ok(zero);
    }
    let mut out = extend::into_owned(eng, other)?;
    if out.weights.is_none() {
        out.weights = weights;
    }
    Ok(out)
}

/// Put one operand on the other's vtree, of the same layout: an owned one is
/// reseated, else a copy of `g` is.
fn reseat_either<'a>(eng: &Engine, f: &mut Cow<'a, Tdd>, g: &mut Cow<'a, Tdd>, same: &SameLayout) -> Result<(), OperationError> {
    let from = match (matches!(f, Cow::Owned(_)), matches!(g, Cow::Owned(_))) {
        (_, true) => g,
        (true, false) => f,
        (false, false) => {
            *g = Cow::Owned(g.try_clone_on(eng)?);
            g
        }
    };
    from.to_mut().reseat(same);
    Ok(())
}

/// Whether [`join_at_root`] takes `tdd`: it moves levels and pairs two
/// outputs as node references, so the operand's output is not a summed-out
/// level's value.
fn joinable(tdd: &Tdd) -> bool {
    !tdd.is_zero() && (!tdd.has_marginal_level() || !tdd.level(tdd.output.vtree).is_marginal())
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
    let copied = |op: &Cow<'_, Tdd>| match op {
        Cow::Borrowed(source) => source.weights.as_ref().map(|w| w.try_clone_on(lim)).transpose(),
        Cow::Owned(_) => Ok(None),
    };
    let stores = [copied(&f)?, copied(&g)?];
    let mut assembly = Assembly::from_levels(eng, Arc::clone(into), try_take_levels(eng, into.num_nodes())?, None);
    let (levels, weights) = assembly.parts_mut();
    // A moved level owes the reductions it owed in its operand, at its image.
    let mut carried = Dirty::default();
    for ((operand, map), store) in [(f, f_map), (g, g_map)].into_iter().zip(stores) {
        let mut store = store;
        let mut owed = match operand {
            Cow::Owned(mut moved) => {
                store = moved.weights.take();
                let source = &mut moved.levels[..];
                for (from, &to) in map.iter().enumerate() {
                    std::mem::swap(&mut levels[to.idx()], &mut source[from]);
                }
                return_levels(eng, std::mem::take(&mut moved.levels).into_vec());
                moved.take_worklists()
            }
            Cow::Borrowed(source) => {
                for (from, &to) in map.iter().enumerate() {
                    source.levels[from].try_copy_over(lim, &mut levels[to.idx()])?;
                }
                source.dirty.clone_on(eng)?
            }
        };
        owed.remap(map);
        carried.merge_under(owed);
        // A store keys its columns by level, so they follow the levels' images.
        if let Some(mut store) = store {
            store.rekey_levels(|level| Some(map[level].idx()));
            match weights {
                Some(merged) => merged.absorb(store),
                None => *weights = Some(store),
            }
        }
    }
    carried.dedup_above(into.num_nodes());
    let root = into.root();
    let local = levels[root.idx()].push_node(lim, &[pair])?;
    let changed = [root];
    assembly.finish_with(TddNodeId { vtree: root, local }, carried, &[], closed.then_some(&changed[..]))
}
