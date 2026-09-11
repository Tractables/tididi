//! The writers' marginal-level refusal, and the shape of what they emit.
//!
//! `.tdd` and DOT are both structural formats: a pair names its two children by
//! local node index. A marginal level holds per-node model counts instead of
//! nodes, so a pair pointing into one carries an inline count and there is no
//! index to emit. These pin the refusal as an error at the entry point, so
//! rendering a marginalized diagram cannot panic inside the emit loop.
//!
//! The text-format tests at the end read the emitted lines directly, because a
//! reader outside this crate parses them by prefix and would not survive a
//! silent change to the header block.
//!
//! Operands come from [`crate::test_helpers::compile_clauses`], which conjoins
//! the clauses one at a time against a fixed vtree; a driver that preprocesses
//! the formula first reaches these diagrams by other routes, and that variety
//! belongs to the driver's own tests.

use crate::engine::Engine;
use std::sync::Arc;

use crate::apply::conjoin_clause::clause_to_tdd;
use crate::io::tdd_to_dot;
use crate::io::{save_tdd, write_tdd};
use crate::reduce::minimize;
use crate::apply::apply_and;
use crate::marginal::marginalize_batch;
use crate::diagram::Tdd;
use crate::diagram::Literal;
use crate::test_helpers::{
    assert_canonical, compile_clauses_on, literals, rand_cnf, CnfShape, Lcg,
};
use crate::vtree::{VarId, Vtree};

/// A small diagram with one level marginalized away, i.e.
/// `Tdd::has_marginal_level()` holds.
fn tdd_with_a_marginal_level() -> Tdd {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut tdd =
        compile_clauses_on(&eng, &vtree, &[vec![1, 2], vec![-2, 3], vec![3, 4]]);

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
    // No canonical-form assertion here: pair fusion saturates in a later
    // reduction, not in the forget itself, so this fixture is mid-pipeline by
    // design. What the tests below need of it is that it is marginal.
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

    let err = save_tdd(&tdd, &path)
        .expect_err("a marginal diagram has no .tdd encoding");
    assert!(matches!(err, crate::io::IoError::Format(_)), "a refusal is a format error, not an io one: {err:?}");
    assert!(
        !path.exists(),
        "a refused save must not leave a stray file behind",
    );
}

/// The pre-sizing over-allocates, so the saved file must be trimmed to what was
/// written: a tail of padding bytes is not a record and the reader refuses it.
#[test]
fn a_saved_diagram_reads_back_without_a_tail_of_padding() {
    let vtree = Arc::new(crate::vtree::Vtree::balanced(8));
    let mut tdd = compile_clauses_on(&Engine::new(), &vtree, &[vec![1, -2], vec![2, 3]]);
    minimize(&mut tdd);

    let path = std::env::temp_dir().join("tididi_io_save_round_trip.tdd");
    save_tdd(&tdd, &path).expect("a structural diagram writes");
    let bytes = std::fs::read(&path).expect("the file is readable");
    let back = crate::io::load_tdd(&path, &vtree).expect("what was written reads back");
    std::fs::remove_file(&path).ok();

    assert!(!bytes.contains(&0), "the file must hold no padding bytes");
    assert_eq!(back.model_count(), tdd.model_count());
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
    assert_canonical(&f);

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
        let mut rng = Lcg::new(seed);
        for &nvars in &[2u32, 3, 4, 5] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            for _ in 0..10 {
                let mut f = constant_one(&eng, &vtree);
                for clause in rand_cnf(&mut rng, nvars, CnfShape { clauses: 4, width: 3 }) {
                    f = apply_and(f, clause_to_tdd(&eng, &vtree, &literals(&clause)));
                }
                minimize(&mut f);
                assert_canonical(&f);
                let back = round_trip(&f);
                assert_canonical(&back);
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
    let one_back = round_trip(&one);
    assert_canonical(&one_back);
    assert_eq!(model_count(&one_back), model_count(&one), "the tautology must survive");
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
        "p tdd 1 3 5 4 0\nI 4 0 1 0\n",
        "p tdd 1 3 5 4 0\nL 0 9\n",
        "p tdd 1 3 5 4 7\n",
    ] {
        assert!(
            matches!(read_tdd(&mut bad.as_bytes(), &vtree), Err(IoError::Format(_))),
            "a malformed file must be refused: {bad:?}"
        );
    }
}

// ── The format version ───────────────────────────────────────────────────────

/// The version in the problem line is what lets a file outlive the build that
/// wrote it: a reader loads a file up to its own version and says so when it
/// will not.
///
/// A problem line with no version is a file from before the format carried one.
/// Its first field is the leaf count, so reading it as a version would accept
/// the file and rebuild a different diagram from it — the refusal has to name
/// the cause rather than fail later on a field that no longer lines up.
#[test]
fn a_problem_line_without_a_version_is_refused_as_predating_the_version() {
    use crate::io::{read_tdd, IoError};

    let vtree = Arc::new(Vtree::balanced(3));
    let unversioned = "p tdd 3 5 4 0\n";
    let Err(IoError::Format(msg)) = read_tdd(&mut unversioned.as_bytes(), &vtree) else {
        panic!("a file with no format version must be refused");
    };
    assert!(
        msg.contains("no format version") && msg.contains("before the format was versioned"),
        "the refusal must say the file predates the version: {msg}"
    );
}

/// A file from a later version may use records this build would misread, so it
/// is refused — and the message names both versions, because the reader's own
/// version is half of why the file will not load.
#[test]
fn a_file_from_a_later_version_is_refused_naming_both_versions() {
    use crate::io::{read_tdd, IoError};

    let vtree = Arc::new(Vtree::balanced(3));
    let newer = "p tdd 2 3 5 4 0\n";
    let Err(IoError::Format(msg)) = read_tdd(&mut newer.as_bytes(), &vtree) else {
        panic!("a file from a later format version must be refused");
    };
    assert!(
        msg.contains("version 2") && msg.contains("version 1"),
        "the refusal must name the file's version and the reader's: {msg}"
    );
}

/// The two halves of what a version buys: a comment line a reader does not
/// recognize is ignored, so a writer may annotate a file freely, while a record
/// letter it does not recognize is refused, so a new record has to raise the
/// version rather than pass unnoticed.
#[test]
fn an_unknown_comment_is_ignored_and_an_unknown_record_is_refused() {
    use crate::io::{read_tdd, write_tdd, IoError};
    use crate::query::model_count;

    let vtree = Arc::new(Vtree::balanced(3));
    let f = crate::test_helpers::compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3]]);
    let mut bytes: Vec<u8> = Vec::new();
    write_tdd(&mut bytes, &f).expect("an explicit diagram writes");
    let text = String::from_utf8(bytes).expect("the format is text");

    let annotated = format!("c written by something else\nc\n{text}");
    let back = read_tdd(&mut annotated.as_bytes(), &vtree)
        .expect("a comment line a reader does not know is ignored");
    assert_eq!(model_count(&back), model_count(&f), "a comment changed the function");

    let with_record = format!("{text}X 4 0 1 0\n");
    let Err(IoError::Format(msg)) = read_tdd(&mut with_record.as_bytes(), &vtree) else {
        panic!("a record letter the reader does not know must be refused");
    };
    assert!(msg.contains("unknown record type"), "the refusal must name the record: {msg}");
}

/// The header block a reader keys off: comment lines, then the problem line
/// naming the variable count, then the vtree leaves and the internal nodes.
#[test]
fn the_text_format_carries_a_header_leaves_and_internal_nodes() {
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = crate::test_helpers::compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3]]);
    assert_canonical(&tdd);
    let mut out = Vec::new();
    write_tdd(&mut out, &tdd).unwrap();
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.lines().collect();

    assert!(lines[0].starts_with("c "), "the file opens with a comment line");
    let problem = lines
        .iter()
        .find(|l| l.starts_with("p tdd "))
        .expect("the header carries a problem line");
    assert!(
        problem.starts_with("p tdd 1 3 "),
        "the problem line names the format version and then the variable count"
    );
    assert!(lines.iter().any(|l| l.starts_with("L ")), "vtree leaves are emitted");
    assert!(lines.iter().any(|l| l.starts_with("I ")), "internal nodes are emitted");
}

/// An unsatisfiable formula has no nodes to emit, so the output names the
/// constant and stops there.
#[test]
fn the_text_format_names_the_zero_constant_and_emits_no_nodes() {
    let vtree = Arc::new(Vtree::balanced(2));
    let tdd = crate::test_helpers::compile_clauses(&vtree, &[vec![1], vec![-1], vec![2]]);
    assert_canonical(&tdd);
    let mut out = Vec::new();
    write_tdd(&mut out, &tdd).unwrap();
    let text = String::from_utf8(out).unwrap();

    assert!(text.contains("ZERO"), "the output names the zero constant");
    assert_eq!(text.lines().filter(|l| l.starts_with("I ")).count(), 0);
}
