//! Validation and bit positions shared by column queries.

use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree};

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
