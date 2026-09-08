//! The writers' marginal-level refusal.
//!
//! `.tdd` and DOT are both structural formats: a pair names its two children by
//! local node index. A marginal level holds per-node model counts instead of
//! nodes, so a pair pointing into one carries an inline count and there is no
//! index to emit. That used to reach an `unreachable!()` inside the emit loop —
//! a panic any library caller could trigger by rendering a diagram it had
//! marginalized. These pin the refusal as an error at the entry point.

use crate::engine::Limits;
use std::sync::Arc;

use crate::build::clause_to_tdd;
use crate::write::tdd_to_dot;
use crate::write::save::{save_tdd, write_tdd};
use crate::reduce::minimize;
use crate::apply::apply_and;
use crate::marginal::marginalize_batch;
use crate::diagram::Tdd;
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};

/// A small diagram with one level marginalized away, i.e.
/// `Tdd::has_marginal_level()` holds.
fn tdd_with_a_marginal_level() -> Tdd {
    let lim = Limits::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let lit = |v: i32| Literal::new(VarId(v.unsigned_abs() - 1), v > 0);
    let clauses = [
        vec![lit(1), lit(2)],
        vec![lit(-2), lit(3)],
        vec![lit(3), lit(4)],
    ];
    let mut acc: Option<Tdd> = None;
    for c in &clauses {
        let clause = clause_to_tdd(&vtree, c);
        acc = Some(match acc {
            Some(prev) => {
                let mut r = apply_and(prev, clause);
                minimize(&mut r);
                r
            }
            None => clause,
        });
    }
    let mut tdd = acc.expect("three clauses build a diagram");
    minimize(&mut tdd);

    // Forget one variable's leaf level: the sibling of the root's left child
    // subtree bottom. Any single leaf level will do — marginalizing it makes
    // that level count-bearing.
    let (left, _right) = vtree.children(vtree.root());
    let (target, _) = vtree.children(left);
    marginalize_batch(&lim, &mut tdd, &[target], &vtree).expect("no wall is installed here");

    assert!(
        tdd.has_marginal_level(),
        "test setup: marginalize_batch must leave a marginal level behind",
    );
    tdd
}

#[test]
fn write_tdd_refuses_a_marginal_diagram() {
    let tdd = tdd_with_a_marginal_level();
    let mut buf: Vec<u8> = Vec::new();
    let err = write_tdd(&mut buf, &tdd).expect_err("a marginal diagram has no .tdd encoding");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("marginal"),
        "the message must say what is wrong: {err}",
    );
    assert!(buf.is_empty(), "nothing may be written before the refusal");
}

#[test]
fn save_tdd_refuses_a_marginal_diagram_without_creating_the_file() {
    let tdd = tdd_with_a_marginal_level();
    let path = std::env::temp_dir().join("tididi_io_marginal_refusal.tdd");
    std::fs::remove_file(&path).ok();

    let err = save_tdd(&tdd, path.to_str().expect("a UTF-8 temp path"))
        .expect_err("a marginal diagram has no .tdd encoding");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        !path.exists(),
        "a refused save must not leave a stray file behind",
    );
}

#[test]
fn tdd_to_dot_refuses_a_marginal_diagram() {
    let tdd = tdd_with_a_marginal_level();
    let err = tdd_to_dot(&tdd).expect_err("a marginal level has no edges to draw");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("marginal"),
        "the message must say what is wrong: {err}",
    );
}

#[test]
fn the_writers_still_accept_an_explicit_diagram() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
    assert!(!f.has_marginal_level(), "setup: no level may be marginal");

    let mut buf: Vec<u8> = Vec::new();
    write_tdd(&mut buf, &f).expect("an explicit diagram serializes");
    assert!(!buf.is_empty());
    let dot = tdd_to_dot(&f).expect("an explicit diagram renders");
    assert!(dot.starts_with("graph tdd {"));
}
