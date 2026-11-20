//! Operations followed by a chosen reduction, for tests that compare plans.

use std::borrow::Cow;

use crate::apply::FilterOutcome;
use crate::diagram::TddNodeId;
use crate::reduce::ReductionPlan;
use crate::{Engine, OperationError, Tdd};

pub(crate) fn negate_with<'f>(eng: &Engine, f: impl Into<Cow<'f, Tdd>>, plan: ReductionPlan<'_>) -> Result<Tdd, OperationError> {
    let mut g = eng.negate(f)?;
    eng.reduce(&mut g, plan)?;
    Ok(g)
}

pub(crate) fn filter_nodes_with(
    eng: &Engine,
    f: &Tdd,
    keep: impl FnMut(TddNodeId) -> bool,
    plan: ReductionPlan<'_>,
) -> Result<FilterOutcome, OperationError> {
    Ok(match eng.filter_nodes(f, keep)? {
        FilterOutcome::Filtered { mut tdd, stats } => {
            eng.reduce(&mut tdd, plan)?;
            FilterOutcome::Filtered { tdd, stats }
        }
        outcome => outcome,
    })
}
