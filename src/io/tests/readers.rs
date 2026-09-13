use std::sync::Arc;
use crate::{Tdd, Vtree};
use crate::io::{read_tdd, write_tdd, IoError};
use crate::test_helpers::assert_canonical;

/// A nonconstant diagram serialized without optional comments.
fn fixture() -> (Arc<Vtree>, Tdd, String) {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]);
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
            assert!(f.model_count() <= 4u32.into());
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
