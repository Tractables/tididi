//! Validation and bit positions shared by column queries.

use std::borrow::Cow;

use crate::diagram::Tdd;
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

use super::columns::MAX_COLUMN_BITS;

/// Each variable's column and bit (`0` the least significant) in
/// `columns`.
pub(super) fn roles(vtree: &Vtree, columns: &[&[VarId]]) -> Result<Vec<Option<(u32, u32)>>, OperationError> {
    let mut role: Vec<Option<(u32, u32)>> = vec![None; vtree.num_vars() as usize + 1];
    for (j, vars) in columns.iter().enumerate() {
        if vars.len() > MAX_COLUMN_BITS {
            return Err(OperationError::ColumnTooWide { column: j, bits: vars.len() });
        }
        for (i, &v) in vars.iter().enumerate() {
            if vtree.leaf_of(v).is_none() {
                return Err(OperationError::VariableNotInVtree(v));
            }
            let slot = &mut role[v.0 as usize];
            if slot.is_some() {
                return Err(OperationError::DuplicateVariable(v));
            }
            *slot = Some((j as u32, (vars.len() - 1 - i) as u32));
        }
    }
    Ok(role)
}

/// `tdd` on a vtree holding every variable `columns` lists that its home
/// has: as it is when its own vtree holds them, otherwise placed on the
/// union of its projection with those variables. The function depends on no
/// variable its projection dropped, so each comes out free.
pub(super) fn with_listed<'a>(eng: &Engine, tdd: &'a Tdd, columns: &[&[VarId]]) -> Result<Cow<'a, Tdd>, OperationError> {
    let vtree = tdd.vtree();
    let Some(home) = vtree.link_home() else { return Ok(Cow::Borrowed(tdd)) };
    let dropped: Vec<VarId> = columns
        .iter()
        .flat_map(|vars| vars.iter().copied())
        .filter(|&v| vtree.leaf_of(v).is_none() && home.rank(v).is_some())
        .collect();
    if dropped.is_empty() {
        return Ok(Cow::Borrowed(tdd));
    }
    let listed = Vtree::projection(home, dropped).expect("variables of the home");
    crate::apply::extend::onto(eng, tdd, &Vtree::union(vtree, &listed)?)
}
