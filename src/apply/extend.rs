//! Operands on projections of one home vtree, brought to one vtree.
//!
//! An operation that reads two or more diagrams level by level runs on the
//! projection of their home on the union of their leaf sets. Each operand
//! not on it is placed there, every added leaf a true level, so the kernels
//! read the operands as they read operands of one vtree. A result on a
//! projection is trimmed to the leaves it names.

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

/// Whether `operands` need [`borrowed`] or [`owned`] before a kernel reads
/// them.
#[inline]
pub(crate) fn needed<const N: usize>(operands: [&Tdd; N]) -> bool {
    match operands.split_first() {
        Some((first, rest)) => !on_one_vtree(first, rest.iter().copied()),
        None => false,
    }
}

/// `operands`, owned, on one vtree.
#[inline]
pub(crate) fn owned<const N: usize>(eng: &Engine, operands: [Tdd; N]) -> Result<[Tdd; N], OperationError> {
    match operands.split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest) => {
            let extended = extend(eng, operands.into_iter().map(Cow::Owned).collect())?;
            Ok(array(extended.into_iter().map(Cow::into_owned).collect()))
        }
        _ => Ok(operands),
    }
}

/// `operands`, borrowed, on one vtree: each one already there as it is, the
/// others placed.
#[inline]
pub(crate) fn borrowed<'a, const N: usize>(eng: &Engine, operands: [&'a Tdd; N]) -> Result<[Cow<'a, Tdd>; N], OperationError> {
    match operands.split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest.iter().copied()) => {
            Ok(array(extend(eng, operands.into_iter().map(Cow::Borrowed).collect())?))
        }
        _ => Ok(operands.map(Cow::Borrowed)),
    }
}

/// Owned or borrowed `operands` on one vtree.
#[inline]
pub(crate) fn cows<'a, const N: usize>(eng: &Engine, operands: [Cow<'a, Tdd>; N]) -> Result<[Cow<'a, Tdd>; N], OperationError> {
    match operands.split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest.iter().map(|g| &**g)) => {
            Ok(array(extend(eng, operands.into_iter().collect())?))
        }
        _ => Ok(operands),
    }
}

/// An owned operand and a borrowed one on one vtree.
#[inline]
pub(crate) fn owned_borrowed<'a>(eng: &Engine, f: Tdd, g: &'a Tdd) -> Result<(Tdd, Cow<'a, Tdd>), OperationError> {
    if !needed([&f, g]) {
        return Ok((f, Cow::Borrowed(g)));
    }
    let [f, g] = array(extend(eng, vec![Cow::Owned(f), Cow::Borrowed(g)])?);
    Ok((f.into_owned(), g))
}

/// The variadic [`owned`].
#[inline]
pub(crate) fn owned_vec(eng: &Engine, operands: Vec<Tdd>) -> Result<Vec<Tdd>, OperationError> {
    match operands.split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest) => {
            let extended = extend(eng, operands.into_iter().map(Cow::Owned).collect())?;
            Ok(extended.into_iter().map(Cow::into_owned).collect())
        }
        _ => Ok(operands),
    }
}

/// `operands`, a diagram and the diagrams it reads, on one vtree.
pub(crate) fn mixed<'a>(eng: &Engine, operands: Vec<Cow<'a, Tdd>>) -> Result<Vec<Cow<'a, Tdd>>, OperationError> {
    match operands.split_first() {
        Some((first, rest)) if !on_one_vtree(first, rest.iter().map(|g| &**g)) => extend(eng, operands),
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

fn array<T, const N: usize>(items: Vec<T>) -> [T; N] {
    items.try_into().unwrap_or_else(|_| unreachable!("one item per operand"))
}

/// `f` placed on `into`, which holds its leaf set: for an operand read with
/// something built for one vtree.
pub(crate) fn onto<'a>(eng: &Engine, f: &'a Tdd, into: &Arc<Vtree>) -> Result<Cow<'a, Tdd>, OperationError> {
    if Arc::ptr_eq(&f.vtree, into) {
        return Ok(Cow::Borrowed(f));
    }
    match Vtree::leaf_sets(&f.vtree, into)? {
        LeafSets::Equal | LeafSets::AInB if into.num_nodes() != 0 => {}
        _ => return Err(OperationError::VtreeMismatch),
    }
    extend_one(eng, Cow::Borrowed(f), into)
}

/// `result` of an operation on projections, trimmed to the leaves it names.
#[inline]
pub(crate) fn trimmed(eng: &Engine, mut result: Tdd) -> Result<Tdd, OperationError> {
    if result.vtree.home().is_some() {
        trim_result(eng, &mut result)?;
    }
    Ok(result)
}

#[cold]
fn trim_result(eng: &Engine, result: &mut Tdd) -> Result<(), OperationError> {
    let home = result.vtree.home().expect("a projection").clone();
    crate::reduce::trim(eng, result, &home, &[])?;
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
            LeafSets::Equal | LeafSets::BInA => {}
            LeafSets::AInB => into = Arc::clone(&g.vtree),
            LeafSets::Union(union) => into = union,
        }
    }
    if into.num_nodes() == 0 {
        let home = into.home().expect("a vtree with no node is a projection");
        let (_, var) = home.vtree().leaf_bottomup().next().expect("a home has a leaf");
        into = Vtree::projection(home, [var]).expect("a variable of the home");
    }
    if operands.iter().any(|f| moves(f, &into)) {
        crate::apply::shared_weights(operands.iter().map(|f| &**f)).map_err(|_| OperationError::WeightsAcrossProjections)?;
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
        let mut f = into_owned(eng, f)?;
        // Safety: equal leaf sets of one home give one layout, node for node,
        // with the same variable ids.
        unsafe { f.reseat_vtree_unchecked(into) };
        return Ok(Cow::Owned(f));
    }
    if f.vtree.num_nodes() == 0 {
        let mut constant = crate::build::constant_on(eng, into, f.is_one())?;
        constant.weights = f.weights.as_ref().map(crate::diagram::WeightStore::empty_like);
        return Ok(Cow::Owned(constant));
    }
    match place(eng, f, into, |var| var, Free::Build, false) {
        Ok((placed, plan)) => {
            plan.recycle(eng);
            Ok(Cow::Owned(placed))
        }
        Err((error, _)) => Err(operation_error(error)),
    }
}

/// The error of a placement onto a projection of the operand's own home,
/// where the identity map cannot fail.
pub(crate) fn operation_error(error: EmbedError) -> OperationError {
    match error {
        EmbedError::Operation(error) => error,
        EmbedError::SourceWeights(_) | EmbedError::DestinationWeights(_) => OperationError::IncompatibleWeights,
        _ => OperationError::VtreeMismatch,
    }
}
