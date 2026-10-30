use super::*;

mod borrowed;
mod certify;
mod expand_full;
mod leaf_form;
mod marginal;
mod plan;
mod prune_scope;

/// [`expand_full`](crate::apply::negate::expand_full) on a diagram's own
/// levels, for a caller that does not read which levels it added a node to.
/// No negation's close follows, so the live pairs it left the levels are
/// forgotten.
fn expand_full_unmarked(eng: &crate::Engine, tdd: &mut crate::Tdd) -> Result<(), crate::OperationError> {
    use crate::apply::negate::{expand_full, Target};
    let mut filled = vec![false; tdd.vtree.num_nodes()];
    let (canonical, closed) = (tdd.levels.is_canonical(tdd.output), tdd.levels.is_closed());
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    expand_full(eng, &vtree, &mut Target { levels: &mut tdd.levels, source: None }, canonical, closed, &mut filled)?;
    tdd.levels.forget_left_pairs();
    Ok(())
}
