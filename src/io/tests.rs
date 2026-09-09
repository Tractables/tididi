//! The writers' marginal-level refusal.
//!
//! `.tdd` and DOT are both structural formats: a pair names its two children by
//! local node index. A marginal level holds per-node model counts instead of
//! nodes, so a pair pointing into one carries an inline count and there is no
//! index to emit. These pin the refusal as an error at the entry point, so
//! rendering a marginalized diagram cannot panic inside the emit loop.

use crate::engine::Engine;
use std::sync::Arc;

use crate::build::clause_to_tdd;
use crate::io::tdd_to_dot;
use crate::io::{save_tdd, write_tdd};
use crate::reduce::minimize;
use crate::apply::apply_and;
use crate::marginal::marginalize_batch;
use crate::diagram::Tdd;
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};

/// A small diagram with one level marginalized away, i.e.
/// `Tdd::has_marginal_level()` holds.
fn tdd_with_a_marginal_level() -> Tdd {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let lit = |v: i32| Literal::new(VarId(v.unsigned_abs() - 1), v > 0);
    let clauses = [
        vec![lit(1), lit(2)],
        vec![lit(-2), lit(3)],
        vec![lit(3), lit(4)],
    ];
    let mut acc: Option<Tdd> = None;
    for c in &clauses {
        let clause = clause_to_tdd(&eng, &vtree, c);
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
    marginalize_batch(&eng, &mut tdd, &[target], &vtree).expect("no wall is installed here");

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
    assert!(matches!(err, crate::io::IoError::Format(_)), "a refusal is a format error, not an io one: {err:?}");
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
    assert!(matches!(err, crate::io::IoError::Format(_)), "a refusal is a format error, not an io one: {err:?}");
    assert!(
        !path.exists(),
        "a refused save must not leave a stray file behind",
    );
}

#[test]
fn tdd_to_dot_refuses_a_marginal_diagram() {
    let tdd = tdd_with_a_marginal_level();
    let err = tdd_to_dot(&tdd).expect_err("a marginal level has no edges to draw");
    assert!(matches!(err, crate::io::IoError::Format(_)), "a refusal is a format error, not an io one: {err:?}");
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

// ── Round trip ───────────────────────────────────────────────────────────────

/// What the writer emits, the reader must accept, and the diagram that comes
/// back must be the one that went in: the same function, and the same
/// structure level by level once the writer's node renumbering is normalized
/// away.
///
/// Randomized over small CNFs from fixed seeds, and over the two shapes the
/// format handles specially — the unsatisfiable diagram, whose problem line
/// carries the `ZERO` token and no records, and the tautology, whose output is
/// a leaf's implicit `one` node.
#[test]
fn writing_a_diagram_and_reading_it_back_returns_the_same_diagram() {
    use crate::build::constant_one;
    use crate::io::{read_tdd, write_tdd};
    use crate::query::model_count;
    use crate::test_helpers::normalized_levels;

    let eng = Engine::new();
    let round_trip = |f: &Tdd| -> Tdd {
        let mut bytes: Vec<u8> = Vec::new();
        write_tdd(&mut bytes, f).expect("an explicit diagram writes");
        read_tdd(&mut bytes.as_slice(), &f.vtree).expect("what the writer emits, the reader reads")
    };

    for &seed in &[0x5eed_0001u64, 0x5eed_0002, 0x5eed_0003] {
        let mut state = seed;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        for &nvars in &[2u32, 3, 4, 5] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            for _ in 0..10 {
                let mut f = constant_one(&eng, &vtree);
                for _ in 0..1 + rng() % 4 {
                    let width = 1 + (rng() % 3) as usize;
                    let mut literals = Vec::new();
                    let mut seen = vec![false; nvars as usize];
                    for _ in 0..width {
                        let v = (rng() % u64::from(nvars)) as u32;
                        if seen[v as usize] {
                            continue;
                        }
                        seen[v as usize] = true;
                        literals.push(if rng() % 2 == 0 {
                            Literal::pos(VarId(v))
                        } else {
                            Literal::neg(VarId(v))
                        });
                    }
                    if literals.is_empty() {
                        continue;
                    }
                    f = apply_and(f, clause_to_tdd(&eng, &vtree, &literals));
                }
                minimize(&mut f);
                let back = round_trip(&f);
                assert_eq!(model_count(&back), model_count(&f), "the round trip changed the function");
                assert_eq!(
                    normalized_levels(&back),
                    normalized_levels(&f),
                    "the round trip changed the structure"
                );
            }
        }
    }

    let vtree = Arc::new(Vtree::balanced(3));
    let zero = Tdd::zero(&vtree);
    assert!(round_trip(&zero).is_zero(), "the ZERO token must read back as the zero diagram");
    let one = constant_one(&eng, &vtree);
    assert_eq!(model_count(&round_trip(&one)), model_count(&one), "the tautology must survive");
}

/// The reader's refusals: a file that is not a diagram, and a file that is a
/// diagram over a different vtree. Both are the caller's mistake, not the
/// stream's, so both are `Format`.
#[test]
fn the_reader_refuses_what_is_not_this_diagram() {
    use crate::build::constant_one;
    use crate::io::{read_tdd, write_tdd, IoError};

    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = apply_and(
        constant_one(&eng, &vtree),
        clause_to_tdd(&eng, &vtree, &[Literal::pos(VarId(0)), Literal::neg(VarId(1))]),
    );
    let mut bytes: Vec<u8> = Vec::new();
    write_tdd(&mut bytes, &f).expect("an explicit diagram writes");

    let wider = Arc::new(Vtree::balanced(4));
    assert!(
        matches!(read_tdd(&mut bytes.as_slice(), &wider), Err(IoError::Format(_))),
        "a file written over another vtree must be refused, not silently reinterpreted"
    );

    for bad in [
        "c only a comment\n",
        "p tdd 3 5 4 0\nI 4 0 1 0\n",
        "p tdd 3 5 4 0\nL 0 9\n",
        "p tdd 3 5 4 7\n",
    ] {
        assert!(
            matches!(read_tdd(&mut bad.as_bytes(), &vtree), Err(IoError::Format(_))),
            "a malformed file must be refused: {bad:?}"
        );
    }
}
