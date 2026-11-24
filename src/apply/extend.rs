//! Operands on projections of one home vtree, brought to one vtree.
//!
//! An operation that reads two or more diagrams level by level runs on the
//! projection of their home on the union of their leaf sets. Each operand
//! not on it is placed there, so the kernels read the operands as they read
//! operands of one vtree. An added leaf, variable or region, is named `One`:
//! the operand is true over its variables and has not summed them out.
//! Beside another operand's region that `One` is the identity of a product
//! with its sum, which is how the conjunction and the product count read
//! it; an entry for which a summed operand has no meaning refuses it before
//! anything is placed. A result on a projection is trimmed to the leaves it
//! names.

use std::borrow::Cow;
use std::sync::Arc;

use crate::diagram::Tdd;
use crate::limits::OperationError;
use crate::restructure::EmbedError;
use crate::restructure::embed::{place, Free};
use crate::vtree::{LeafSets, Vtree, VtreeIdx};
use crate::Engine;

/// Whether `operands` share the first one's vtree allocation and it has a
/// node, so that the kernels run on them as they are.
#[inline]
fn on_one_vtree<'t>(first: &Tdd, rest: impl IntoIterator<Item = &'t Tdd>) -> bool {
    rest.into_iter().all(|g| Arc::ptr_eq(&first.vtree, &g.vtree)) && first.vtree.num_nodes() != 0
}

/// Whether `operands` need [`on_one`] before a kernel reads them.
#[inline]
pub(crate) fn needed<const N: usize>(operands: [&Tdd; N]) -> bool {
    match operands.split_first() {
        Some((first, rest)) => !on_one_vtree(first, rest.iter().copied()),
        None => false,
    }
}

/// [`OperationError::VtreeMismatch`] for `operands` on more than one home,
/// which an operation refuses before any refusal that reads their levels.
#[inline]
pub(crate) fn refuse_other_homes<'t>(operands: impl IntoIterator<Item = &'t Tdd>) -> Result<(), OperationError> {
    let mut operands = operands.into_iter();
    let Some(first) = operands.next() else { return Ok(()) };
    operands
        .filter(|g| !Arc::ptr_eq(&first.vtree, &g.vtree))
        .try_for_each(|g| Vtree::refuse_other_home(&first.vtree, &g.vtree))
}

/// `operands`, an array or a `Vec`, on one vtree: each one already there as
/// it is, the others placed. An owned operand stays owned.
#[inline]
pub(crate) fn on_one<'a, O>(eng: &Engine, operands: O) -> Result<O, OperationError>
where
    O: AsRef<[Cow<'a, Tdd>]> + IntoIterator<Item = Cow<'a, Tdd>> + TryFrom<Vec<Cow<'a, Tdd>>>,
{
    match operands.as_ref().split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest.iter().map(|g| &**g)) => {
            let extended = extend(eng, operands.into_iter().collect())?;
            Ok(O::try_from(extended).unwrap_or_else(|_| unreachable!("one item per operand")))
        }
        _ => Ok(operands),
    }
}

/// `f` owned: a borrowed one copied under the engine's limits.
pub(crate) fn into_owned(eng: &Engine, f: Cow<'_, Tdd>) -> Result<Tdd, OperationError> {
    match f {
        Cow::Borrowed(f) => f.try_clone_on(eng),
        Cow::Owned(f) => Ok(f),
    }
}

/// `f` placed on `into`, which holds its leaf set: for an operand read with
/// something built for one vtree.
pub(crate) fn onto<'a>(eng: &Engine, f: &'a Tdd, into: &Arc<Vtree>) -> Result<Cow<'a, Tdd>, OperationError> {
    if Arc::ptr_eq(&f.vtree, into) {
        return Ok(Cow::Borrowed(f));
    }
    match Vtree::leaf_sets(&f.vtree, into)? {
        LeafSets::Equal(_) | LeafSets::AInB if into.num_nodes() != 0 => {}
        _ => return Err(OperationError::VtreeMismatch),
    }
    extend_one(eng, Cow::Borrowed(f), into)
}

/// `result` of an operation on projections, trimmed to the leaves it names;
/// elsewhere [`close_constant`](crate::reduce::close_constant).
#[inline]
pub(crate) fn trimmed(eng: &Engine, mut result: Tdd) -> Result<Tdd, OperationError> {
    match result.vtree.on_home() {
        true => trim_result(eng, &mut result)?,
        false => crate::reduce::close_constant(eng, &mut result),
    }
    Ok(result)
}

#[cold]
fn trim_result(eng: &Engine, result: &mut Tdd) -> Result<(), OperationError> {
    let home = result.vtree.home().expect("a vtree on a home");
    crate::reduce::trim(eng, result, &home, None, &[])?;
    Ok(())
}

/// Nodes `targets` of `from`, which every one names, as nodes of `into`, a
/// projection of the same home holding `from`'s leaves; `None` when `into`
/// is `from`.
pub(crate) fn nodes_on(eng: &Engine, from: &Arc<Vtree>, into: &Arc<Vtree>, targets: &[VtreeIdx]) -> Result<Option<Vec<VtreeIdx>>, OperationError> {
    if Arc::ptr_eq(from, into) || targets.is_empty() {
        return Ok(None);
    }
    let mut nodes = Vec::new();
    eng.limits().reserve_exact(&mut nodes, targets.len())?;
    nodes.extend(targets.iter().map(|&t| {
        into.node_at_home(from.home_node(t)).expect("a node of a projection inside `into` is a node of `into`")
    }));
    Ok(Some(nodes))
}

/// Place every operand on the projection on the union of their leaf sets.
/// The empty projection has no level for a kernel to read, so operands that
/// are all constants go to the projection on one leaf of the home, which
/// the closing trim drops.
#[cold]
fn extend<'a>(eng: &Engine, operands: Vec<Cow<'a, Tdd>>) -> Result<Vec<Cow<'a, Tdd>>, OperationError> {
    let Some((first, rest)) = operands.split_first() else { return Ok(operands) };
    let mut into = Arc::clone(&first.vtree);
    for g in rest {
        match Vtree::leaf_sets(&into, &g.vtree)? {
            LeafSets::Equal(_) | LeafSets::BInA => {}
            LeafSets::AInB => into = Arc::clone(&g.vtree),
            LeafSets::Union(union) => into = union,
        }
    }
    if into.num_nodes() == 0 {
        let home = into.link_home().expect("a vtree with no node is a projection");
        let (_, var) = home.vtree().leaf_bottomup().next().expect("a home has a leaf");
        into = Vtree::projection(home, [var]).expect("a variable of the home");
    }
    operands.into_iter().map(|f| extend_one(eng, f, &into)).collect()
}

/// Whether `f` is on a projection with fewer leaves than `into`, which
/// holds its leaf set.
fn moves(f: &Tdd, into: &Vtree) -> bool {
    f.vtree.num_nodes() != into.num_nodes()
}

/// `f` on `into`, which holds its leaf set: reseated when the leaf sets are
/// equal, placed otherwise.
fn extend_one<'a>(eng: &Engine, f: Cow<'a, Tdd>, into: &Arc<Vtree>) -> Result<Cow<'a, Tdd>, OperationError> {
    if Arc::ptr_eq(&f.vtree, into) {
        return Ok(f);
    }
    if !moves(&f, into) {
        let Ok(LeafSets::Equal(same)) = Vtree::leaf_sets(&f.vtree, into) else {
            unreachable!("a projection holding the leaf set of another with as many nodes has that leaf set")
        };
        let mut f = into_owned(eng, f)?;
        f.reseat(&same);
        return Ok(Cow::Owned(f));
    }
    if f.vtree.num_nodes() == 0 {
        return Ok(Cow::Owned(crate::build::constant_padded_on(eng, into, f.is_one())?));
    }
    match place(eng, f, into, |var| var, Free::Operand) {
        Ok((placed, _)) => Ok(Cow::Owned(placed)),
        Err((error, _)) => Err(operation_error(error)),
    }
}

/// The error of a placement onto a projection of the operand's own home,
/// where the identity map cannot fail.
pub(crate) fn operation_error(error: EmbedError) -> OperationError {
    match error {
        EmbedError::Operation(error) => error,
        _ => OperationError::VtreeMismatch,
    }
}
