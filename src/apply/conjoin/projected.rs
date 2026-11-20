//! Conjunction of two diagrams on different projections of one home vtree.
//!
//! The result lives on the smallest projection holding both leaf sets: the
//! larger operand's vtree when one set holds the other, the union projection
//! otherwise. The sweep is [`conjoin_onto`]'s.

use std::borrow::Cow;
use std::sync::Arc;

use super::drive;
use super::union;
use super::held::Held;
use super::view::View;
use super::{conjoin_cow, conjoin_kept, conjoin_onto, recycle, AndRefused, Operands, VtreeMask};
use crate::restructure::embed::Plan;
use crate::diagram::{return_levels, try_take_levels, Assembly, ChildPair, Dirty, Tdd, TddNodeId};
use crate::limits::OperationError;
use crate::apply::extend;
use crate::build::constant_without_leaf;
use crate::vtree::{LeafSets, VarId, Vtree, VtreeIdx};
use crate::Engine;

/// [`Engine::and`] of operands on different vtree allocations.
pub(super) fn conjoin_projections<'a>(eng: &Engine, mut f: Cow<'a, Tdd>, mut g: Cow<'a, Tdd>) -> Result<Tdd, OperationError> {
    if f.vtree().num_nodes() == 0 || g.vtree().num_nodes() == 0 {
        return conjoin_constant(eng, f, g);
    }
    let g_larger = g.vtree().num_leaves() > f.vtree().num_leaves();
    if matches!(g, Cow::Owned(_)) && (g_larger || !matches!(f, Cow::Owned(_))) {
        std::mem::swap(&mut f, &mut g);
    }
    if let Cow::Owned(owned) = f {
        (f, g) = match conjoin_nested(eng, owned, g)? {
            Ok(out) => return Ok(out),
            Err(operands) => operands,
        };
    }
    let mut reader = (union::readable(&f) && union::readable(&g)).then(|| union::Reader::new(eng, (f.vtree(), g.vtree())));
    let sets = match reader.as_mut() {
        Some(reader) => Vtree::leaf_sets_reading(f.vtree(), g.vtree(), Some(reader))?,
        None => Vtree::leaf_sets(f.vtree(), g.vtree())?,
    };
    let mut tables = reader.and_then(union::Reader::into_tables);
    if f.has_marginal_level() || g.has_marginal_level() {
        check_summed_leaves(&f, &g)?;
    }
    let into = match sets {
        // The sweep reads the vtree of `f` only, and the two have one layout.
        LeafSets::Equal(_) => {
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
                return join_at_root(eng, f, &f_map, g, &g_map, f_left, &into);
            }
            if let Some(tables) = tables.as_deref_mut() {
                return union::conjoin_union(eng, f, g, &into, tables);
            }
            into
        }
    };
    let mut out = conjoin_onto(eng, f, |v| v, g, |v| v, &into).map_err(|refused| extend::operation_error(refused.error))?;
    crate::reduce::close_constant(eng, &mut out);
    Ok(out)
}

/// Operands [`conjoin_nested`] does not take, as given.
type NotNested<'a> = (Cow<'a, Tdd>, Cow<'a, Tdd>);

/// `f ∧ g` where `f` is owned, both are plain (no summed level, no region,
/// not false) and the leaves of `g` are among fewer than
/// all of `f`'s: the sweep reads `g` through a view of the nodes of `f`'s
/// vtree with a leaf of `g` under them, found from `g`'s leaves upward, and
/// the result takes `f`'s level array, every other level left as `f` has
/// it. Other operands come back, as given.
fn conjoin_nested<'a>(eng: &Engine, f: Tdd, g: Cow<'a, Tdd>) -> Result<Result<Tdd, NotNested<'a>>, OperationError> {
    let plain = |t: &Tdd| !t.is_zero() && !t.vtree().has_regions() && !t.has_marginal_level();
    let lim = eng.limits();
    let mut leaves = eng.scratch.apply.cone_leaves.checkout(eng);
    if !plain(&f) || !plain(&g) || !f.vtree().nested_leaves(g.vtree(), &mut leaves) {
        return Ok(Err((Cow::Owned(f), g)));
    }
    let into = Arc::clone(f.vtree());
    let mut targets = eng.scratch.apply.cone_targets.checkout_preserving(eng);
    let plan = match Plan::cone(lim, g.vtree(), &into, &leaves, std::mem::take(&mut *targets)) {
        Ok(plan) => plan,
        Err((error, back)) => {
            *targets = back;
            return Err(error);
        }
    };
    let mut slots = eng.scratch.apply.cone_slots.checkout_preserving(eng);
    let mut view = match View::build_in(eng, &g, &into, &plan, (std::mem::take(&mut *slots), Vec::new())) {
        Ok(view) => view,
        Err((error, (back, _))) => {
            *slots = back;
            *targets = plan.release();
            return Err(error);
        }
    };
    let (mut f, mut g) = (Cow::Owned(f), g);
    let out = drive::apply_and_adopting(
        eng,
        &mut Held::of(&mut f, None),
        &mut Held::of(&mut g, Some(&mut view)),
        VtreeMask::free_of(&plan),
        plan.cone_nodes(),
    );
    recycle(eng, f);
    recycle(eng, g);
    *slots = view.release(&plan).0;
    *targets = plan.release();
    let mut out = out?;
    crate::reduce::close_constant(eng, &mut out);
    out.levels.know_no_marginal();
    Ok(Ok(out))
}

/// [`Engine::and_restoring`] of operands not on one vtree allocation, or on
/// the empty projection. A plain operand, with no summed level, goes through
/// as given where the leaf sets are equal or one holds the other, as those
/// paths hand it back unchanged; otherwise both are read borrowed.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
pub(super) fn conjoin_projections_kept<'a>(eng: &Engine, f: Cow<'a, Tdd>, g: Cow<'a, Tdd>) -> Result<Tdd, AndRefused<'a>> {
    let plain = |t: &Tdd| !t.has_marginal_level();
    if f.vtree().num_nodes() != 0 && g.vtree().num_nodes() != 0 && plain(&f) && plain(&g) {
        let into = match Vtree::leaf_sets(f.vtree(), g.vtree()) {
            Err(error) => return Err(AndRefused { error, f, g }),
            Ok(LeafSets::Equal(_)) => None,
            Ok(LeafSets::AInB) => Some(Arc::clone(g.vtree())),
            Ok(LeafSets::BInA) => Some(Arc::clone(f.vtree())),
            Ok(LeafSets::Union(_)) => return conjoin_borrowing(eng, f, g),
        };
        let mut out = match into {
            None => conjoin_kept(eng, f, g, Operands::default(), Operands::default(), Operands::default(), Operands::default())?,
            Some(into) => conjoin_onto(eng, f, |v| v, g, |v| v, &into)
                .map_err(|r| AndRefused { error: extend::operation_error(r.error), f: r.f, g: r.g })?,
        };
        crate::reduce::close_constant(eng, &mut out);
        return Ok(out);
    }
    conjoin_borrowing(eng, f, g)
}

/// [`conjoin_projections`] of borrows of `f` and `g`, which a refusal hands
/// back.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn conjoin_borrowing<'a>(eng: &Engine, f: Cow<'a, Tdd>, g: Cow<'a, Tdd>) -> Result<Tdd, AndRefused<'a>> {
    conjoin_projections(eng, Cow::Borrowed(&*f), Cow::Borrowed(&*g)).map_err(|error| AndRefused { error, f, g })
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
    if constant.is_zero() || other.is_zero() {
        return Ok(constant_without_leaf(&Vtree::empty_projection(constant.vtree().link_home().expect("a vtree with no node is a projection")), false));
    }
    extend::into_owned(eng, other)
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
    let mut assembly = Assembly::from_levels(eng, Arc::clone(into), try_take_levels(eng, into.num_nodes())?);
    let levels = assembly.levels_mut();
    // A moved level owes the reductions it owed in its operand, at its image.
    let mut carried = Dirty::default();
    for (operand, map) in [(f, f_map), (g, g_map)] {
        let mut owed = match operand {
            Cow::Owned(mut moved) => {
                let source = &mut moved.levels[..];
                for (from, &to) in map.iter().enumerate() {
                    levels[to.idx()].swap_with(&mut source[from]);
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
    }
    carried.dedup_above(into.num_nodes());
    let root = into.root();
    let local = levels[root.idx()].push_node(lim, &[pair])?;
    let changed = [root];
    assembly.finish_with(TddNodeId { vtree: root, local }, carried, &[], closed.then_some(&changed[..]))
}
