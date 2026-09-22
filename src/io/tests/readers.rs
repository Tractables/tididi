use std::sync::Arc;
use crate::{Tdd, Vtree};
use crate::io::{read_tdd, write_tdd, IoError};
use crate::test_helpers::assert_canonical;

/// A nonconstant diagram serialized without optional comments.
fn fixture() -> (Arc<Vtree>, Tdd, String) {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let mut bytes = Vec::new();
    write_tdd(&mut bytes, &f).unwrap();
    let text = String::from_utf8(bytes).unwrap().lines()
        .filter(|line| !line.starts_with('c')).map(|line| format!("{line}\n")).collect();
    (tree, f, text)
}

/// A malformed record reports its line through the format error.
fn refused(text: &str, tree: &Arc<Vtree>) {
    match read_tdd(&mut text.as_bytes(), tree) {
        Err(IoError::Format(message)) => assert!(!message.is_empty()),
        other => panic!("expected a format error for {text:?}, got {other:?}"),
    }
}

#[test]
fn duplicate_headers_leaf_records_and_trailing_fields_are_refused() {
    let (tree, _, text) = fixture();
    let lines: Vec<_> = text.lines().collect();
    for line in lines.iter().filter(|line| line.starts_with('p') || line.starts_with('L')) {
        let bad = format!("{text}{line}\n");
        refused(&bad, &tree);
        let bad = text.replacen(line, &format!("{line} ignored"), 1);
        refused(&bad, &tree);
    }
    let late_header = format!("{}\n{}\n", lines[1..].join("\n"), lines[0]);
    refused(&late_header, &tree);
}

#[test]
fn missing_records_and_truncated_fields_are_refused() {
    let (tree, _, text) = fixture();
    let lines: Vec<_> = text.lines().collect();
    for removed in 0..lines.len() {
        let bad = lines.iter().enumerate().filter(|(i, _)| *i != removed)
            .map(|(_, line)| format!("{line}\n")).collect::<String>();
        refused(&bad, &tree);
    }
    for end in 0..text.trim_end().len() {
        // Removing only a complete final pair can describe another valid function.
        let prefix = &text[..end];
        if let Ok(f) = read_tdd(&mut prefix.as_bytes(), &tree) {
            assert_canonical(&f);
            assert!(f.model_count().unwrap() <= 4u32.into());
        }
    }
}

#[test]
fn invalid_numbers_and_references_are_refused_without_panicking() {
    let (tree, _, text) = fixture();
    let lines: Vec<Vec<&str>> = text.lines().map(|line| line.split_whitespace().collect()).collect();
    for (line_index, tokens) in lines.iter().enumerate() {
        let start = if tokens[0] == "p" { 2 } else { 1 };
        for token_index in start..tokens.len() {
            for replacement in ["-1", "4294967296", "nope", "2147483648"] {
                let mut changed = lines.clone();
                changed[line_index][token_index] = replacement;
                let bad = changed.iter().map(|fields| format!("{}\n", fields.join(" "))).collect::<String>();
                refused(&bad, &tree);
            }
        }
    }
}

#[test]
fn zero_outputs_require_the_zero_token_and_no_node_records() {
    let (tree, _, text) = fixture();
    let header = format!("p tdd 1 2 3 {}", tree.root().idx());
    let zero = format!("{header} ZERO\n");
    let f = read_tdd(&mut zero.as_bytes(), &tree).unwrap();
    assert_canonical(&f);
    assert!(f.is_zero());
    refused(&format!("{header} 4294967295\n"), &tree);
    refused(&format!("{zero}L 0 1\n"), &tree);
    let records = text.lines().filter(|line| !line.starts_with('p')).collect::<Vec<_>>().join("\n");
    refused(&format!("{zero}{records}"), &tree);
}

#[test]
fn line_endings_comments_and_short_reads_preserve_the_diagram() {
    use std::io::BufReader;
    let (tree, original, text) = fixture();
    for variant in [text.clone(), text.replace('\n', "\r\n"), text.trim_end().to_owned(),
        format!("c optional comment\n\n{}", text.replace('\n', "\n c another comment\n"))] {
        let mut reader = BufReader::with_capacity(1, variant.as_bytes());
        let restored = read_tdd(&mut reader, &tree).unwrap();
        assert_canonical(&restored);
        assert!(crate::Engine::new().equivalent(&original, &restored).unwrap());
    }
}

#[test]
fn read_errors_and_invalid_utf8_propagate_as_io_errors() {
    use std::io::{self, Read, BufReader};
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Err(io::Error::other("injected failure")) }
    }
    let (tree, _, text) = fixture();
    let mut broken = BufReader::new(text.as_bytes().chain(Broken));
    assert!(matches!(read_tdd(&mut broken, &tree), Err(IoError::Io(_))));
    let mut invalid = text.into_bytes();
    invalid.push(255);
    assert!(matches!(read_tdd(&mut invalid.as_slice(), &tree), Err(IoError::Io(_))));
}

#[test]
fn undefined_format_versions_are_refused_even_below_the_current_version() {
    let (tree, _, text) = fixture();
    for version in [0, 2, u32::MAX] {
        let invalid = text.replacen("p tdd 1 ", &format!("p tdd {version} "), 1);
        match read_tdd(&mut invalid.as_bytes(), &tree) {
            Err(IoError::Format(message)) => {
                assert!(message.contains(&format!("version {version}")));
                assert!(message.contains("version 1"));
            }
            other => panic!("expected an unsupported-version error, got {other:?}"),
        }
    }
}

#[test]
fn file_local_ids_and_record_order_do_not_change_the_function() {
    let vtree = Arc::new(Vtree::balanced(2));
    // The root is file node 0; its children and leaf declarations arrive later.
    let text = "p tdd 1 2 3 0 0\nI 0 2 1 1 2\nL 1 2\nL 2 1\n";
    let loaded = read_tdd(&mut text.as_bytes(), &vtree).unwrap();
    let expected = Tdd::cube(&vtree, [1, -2]).unwrap();
    assert_canonical(&loaded);
    assert_canonical(&expected);
    assert!(loaded.equivalent(&expected).unwrap());
    for bad in [
        text.replace("I 0 2 1", "I 0 2 2"), // one leaf reached twice
        text.replace("I 0 2 1", "I 0 0 1"), // cycle
        text.replace("L 1 2", "L 1 1"),     // duplicate variable
        text.replace("L 1 2", "L 1 0"),     // zero variable
        text.replace("L 1 2", "L 1 3"),     // wrong variable
        text.replace("L 1 2", "L 0 2"),     // leaf/internal collision
        format!("{text}I 0 1 2 1 2\n"),    // inconsistent child declaration
        text.replace("p tdd 1 2 3 0", "p tdd 1 2 3 1"), // wrong root
    ] { refused(&bad, &vtree); }
}

#[test]
fn rotated_version_one_file_with_original_indices_still_loads() {
    // Older writers used in-memory IDs, which differ from the separately saved vtree.
    let vtree = Arc::new(Vtree::from_text(
        "vtree 7\nL 0 1\nL 1 2\nL 2 3\nL 3 4\nI 4 2 3\nI 5 1 4\nI 6 0 5\n"
    ).unwrap());
    let text = "p tdd 1 4 7 6 0\nL 0 1\nL 1 2\nL 2 3\nL 3 4\n\
                I 5 2 3 2 0\nI 5 2 3 1 0\nI 4 1 5 0 1\nI 4 1 5 0 0\n\
                I 6 0 4 1 0 1 1 2 1\n";
    let loaded = read_tdd(&mut text.as_bytes(), &vtree).unwrap();
    let expected = Tdd::clause(&vtree, [1, -3]).unwrap();
    assert_canonical(&loaded);
    assert_canonical(&expected);
    assert!(loaded.equivalent(&expected).unwrap());
    refused(text, &Arc::new(Vtree::balanced(4)));
}

/// Unknown layouts must be rejected before parsing their version-specific fields.
#[test]
fn header_errors_check_version_before_interpreting_fields() {
    let vtree = Arc::new(Vtree::balanced(2));
    for (header, diagnostic) in [
        ("p tdd 2 unknown fields may differ", "file is format version 2"),
        ("p tdd 0 unknown fields may differ extra", "file is format version 0"),
        ("p tdd nope 2 3 2 ZERO", "format version is not a number"),
        ("p tdd 1 nope 3 2 ZERO", "leaf count is not a number"),
        ("p tdd 1 2 3 2 ZERO extra", "unexpected trailing field"),
        ("p tdd 2 3 2 ZERO", "no format version"),
    ] {
        let text = format!("c header follows\n{header}\n");
        let Err(IoError::Format(message)) = read_tdd(&mut text.as_bytes(), &vtree) else {
            panic!("expected a format error for {header:?}");
        };
        assert!(message.contains("line 2"), "{message}");
        assert!(message.contains(diagnostic), "{header:?}: {message}");
    }
}
