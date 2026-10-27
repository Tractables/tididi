use super::*;

mod certify;
mod expand_full;
mod leaf_form;
mod plan;
mod prune_scope;

/// [`expand_full`](crate::apply::negate::expand_full), for a caller that
/// does not read which levels it added a node to. No negation's close
/// follows, so the live pairs it left the levels are forgotten.
fn expand_full_unmarked(eng: &crate::Engine, tdd: &mut crate::Tdd) -> Result<(), crate::OperationError> {
    let mut filled = vec![false; tdd.vtree.num_nodes()];
    crate::apply::negate::expand_full(eng, tdd, &mut filled)?;
    tdd.levels.forget_left_pairs();
    Ok(())
}
