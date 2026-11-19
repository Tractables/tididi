//! The move of a clause's accumulator onto the projection that adds the
//! clause's variables, built from the accumulator's own link.

use std::sync::Arc;

use super::{place_on, ClauseWork};
use crate::apply::conjoin::view::{chain_readers, moved_dirty, read_chain_nodes, reserve_moved};
use crate::diagram::{return_levels, try_take_levels, ChildSide, NodeIdx, Sides, Tdd, TddNodeId};
use crate::limits::OperationError;
use crate::restructure::embed::{literal_chain, Chain};
use crate::restructure::placement::{first_ref, push_free_level, PassThrough};
use crate::vtree::{Reading, VarId, Vtree, VtreeIdx};
use crate::Engine;

/// The clauses of at most this many literals sort their added variables on
/// the stack.
const ON_STACK: usize = 16;

/// `f`, on a projection with no region, on the projection that adds the
/// variables of `lits` with a leaf in the home and none in `f`'s vtree, of
/// which there is one at least. `f` is consumed on every outcome.
///
/// # Errors
///
/// The placement's, where it is taken: on a constant, a diagram with a
/// summed level, or a result on every leaf of the home, which is the home's
/// own vtree with its node ids. Otherwise only the limits'.
#[inline(never)]
pub(super) fn widen<L: crate::LiteralInput>(eng: &Engine, f: Tdd, lits: &[L]) -> Result<Tdd, OperationError> {
    let mut work = eng.scratch.clause.checkout(eng);
    let ClauseWork { widen_added, widen_map, widen_fresh, .. } = &mut *work;
    let mut stack = [(0, VarId(0)); ON_STACK];
    let added: &mut [(u32, VarId)] = {
        let home = f.vtree.link_home().expect("only a projection is widened");
        let mut found = lits
            .iter()
            .filter_map(|lit| lit.literal().ok())
            .filter(|lit| f.vtree.home_has(lit.var))
            .map(|lit| (home.rank(lit.var).expect("a variable the home has has a rank"), lit.var));
        match lits.len() <= ON_STACK {
            true => {
                let mut m = 0;
                for (slot, found) in stack.iter_mut().zip(&mut found) {
                    *slot = found;
                    m += 1;
                }
                &mut stack[..m]
            }
            false => {
                eng.limits().reserve_exact(widen_added, lits.len())?;
                widen_added.extend(found);
                &mut widen_added[..]
            }
        }
    };
    added.sort_unstable_by_key(|&(rank, _)| rank);
    let m = dedup(added);
    let added = &added[..m];
    let home = f.vtree.link_home().expect("only a projection is widened");
    let full = f.vtree.num_leaves() as usize + m == home.vtree().num_leaves() as usize;
    let constant = f.vtree.num_nodes() == 0 || f.is_zero() || f.is_one();
    if full || constant || f.has_marginal_level() {
        let own = f.vtree.leaf_bottomup().map(|(_, var)| var);
        let into = Vtree::projection(home, own.chain(added.iter().map(|&(_, var)| var))).expect("every variable has a leaf in the home");
        return place_on(eng, f, &into);
    }
    let lim = eng.limits();
    widen_map.clear();
    lim.try_resize(widen_map, f.vtree.num_nodes(), VtreeIdx(0))?;
    lim.reserve_exact(widen_fresh, m)?;
    let source = Arc::clone(&f.vtree);
    let into = Arc::new(source.extended(added, widen_map, |t, reading| widen_fresh.push((t, reading))));
    moved_onto(eng, f, (&source, &into), widen_map, widen_fresh)
}

/// Keep the first of each run of equal ranks of `added`, at its front;
/// returns how many there are.
fn dedup(added: &mut [(u32, VarId)]) -> usize {
    let mut m = 0;
    for i in 0..added.len() {
        if m == 0 || added[i].0 != added[m - 1].0 {
            added[m] = added[i];
            m += 1;
        }
    }
    m
}

/// `f`, on `source`, on `into`, which holds it: each level moved to its
/// node's image by `map`, and each node of `fresh` given the level a
/// placement gives it, children before parents.
fn moved_onto(
    eng: &Engine,
    mut f: Tdd,
    (source, into): (&Arc<Vtree>, &Arc<Vtree>),
    map: &[VtreeIdx],
    fresh: &[(VtreeIdx, Reading)],
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let carries_leaf = |reading| matches!(reading, Reading::Pass(c) if into.node(c).is_var_leaf());
    let chains = fresh.iter().any(|&(_, reading)| carries_leaf(reading));
    let mut gate = lim.gate();
    gate.poll(fresh.len() as u64)?;
    gate.flush()?;
    let listed = reserve_moved(eng, &mut f.dirty, source, chains)?;
    let mut levels = try_take_levels(eng, into.num_nodes())?;
    let closed = f.levels.is_closed();
    let moved = &mut f.levels[..];
    for (s, &d) in map.iter().enumerate() {
        std::mem::swap(&mut levels[d.idx()], &mut moved[s]);
    }
    let is_image = |t: VtreeIdx| fresh.binary_search_by_key(&t, |&(u, _)| u).is_err();
    let mut tops = Vec::new();
    let mut local = f.output.local;
    for &(t, reading) in fresh {
        let Reading::Pass(c) = reading else {
            push_free_level(&mut levels[t.idx()], into, t);
            continue;
        };
        let free_side = if into.children(t).0 == c { ChildSide::Right } else { ChildSide::Left };
        let through = PassThrough::new(into, t, free_side, |c| levels[c.idx()].slot_count());
        let chain = through
            .carries_leaf
            .then(|| literal_chain(into, (is_image, |_| false), |p, side| first_ref(&levels[p.idx()], side), f.output.local, t));
        match chain {
            Some(Chain::Literals(top)) => lim.try_push(&mut tops, top)?,
            Some(Chain::Output(_)) => local = NodeIdx(0),
            _ => {}
        }
        through.fill(lim, &mut levels[t.idx()], chain.as_ref().map(Chain::labels))?;
    }
    for (parent, moving) in chain_readers(lim, into, &tops)? {
        let (left, right) = into.children(parent);
        let decoders = Sides { left: levels[left.idx()].child_decoder(), right: levels[right.idx()].child_decoder() };
        read_chain_nodes(lim, &mut levels[parent.idx()], moving, decoders)?;
    }
    tops.sort_unstable();
    let dirty = moved_dirty(f.take_worklists(), listed, (source, into), (map, |_| false), &tops);
    let output = TddNodeId { vtree: into.root(), local };
    let mut result = Tdd { vtree: Arc::clone(into), levels: levels.into(), output, dirty };
    match closed {
        true => result.close_changed_levels(&[]),
        false => result.close_levels(),
    }
    return_levels(eng, std::mem::take(&mut f.levels).into_vec());
    Ok(result)
}
