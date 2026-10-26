//! The fixtures every file here reads through `use super::*`.

use super::*;
use super::contract::{contract_all_twins, contract_leaf::contract_leaf_twins};
fn canonicalize_content_twins(eng: &Engine, tdd: &mut crate::Tdd) -> Result<(), OperationError> {
    super::driver::Reduction::new(eng, tdd).content_twins()
}
use crate::apply::apply_and;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;

use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use std::sync::Arc;

/// A count-marginal diagram straight out of a conjunction, still owing every
/// pass.
fn edited_marginal_diagram(eng: &Engine) -> crate::Tdd {
    let vtree = Arc::new(Vtree::balanced(8));
    let mut g = eng.clause(&vtree, [1, 5]).unwrap();
    let (left, _) = vtree.children(vtree.root());
    let summed: Vec<VtreeIdx> = vtree.internal_bottomup_slice().iter().copied()
        .filter(|&t| { let mut cur = t; loop {
            if cur == left { break true; }
            match vtree.node(cur).parent() { Some(p) => cur = p, None => break false }
        } })
        .collect();
    eng.marginalize_levels(&mut g, &summed).unwrap();
    let f = eng.and(g, eng.literal(&vtree, 8).unwrap()).unwrap();
    assert!(f.has_marginal_level() && !f.dirty.is_empty());
    f
}

mod canonicity;
mod fixpoint;
mod marginal;
mod pair_arena;
mod plans;
mod pruning;
mod twins;
mod twins_budget;
mod twins_inline;
mod whole_diagrams;

// The prune's test hooks: the sides of stored levels marked as bytes and as
// bits, the stored levels a settle read whole and node by node and the
// implicit levels it read off their digits, and whether a settle reads every
// level node by node, to compare the work the other reads charge with it.
thread_local! {
    static SIDES_MARKED: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
    static SETTLE_READS: std::cell::Cell<[u64; 3]> = const { std::cell::Cell::new([0; 3]) };
    static SETTLE_BY_NODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn note_sides_marked(bytes: u64, bits: u64) {
    SIDES_MARKED.with(|c| {
        let [a, b] = c.get();
        c.set([a + bytes, b + bits]);
    });
}

pub(super) fn note_settle_read(how: usize) {
    SETTLE_READS.with(|c| {
        let mut n = c.get();
        n[how] += 1;
        c.set(n);
    });
}

pub(super) fn settle_by_node() -> bool {
    SETTLE_BY_NODE.with(std::cell::Cell::get)
}
