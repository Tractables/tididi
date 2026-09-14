//! The comment rules in `CONTRIBUTING.md`, enforced on the crate's source.
//!
//! Four checks over the files under `src/`; rules 1, 3 and 4 read the files
//! that are not themselves test modules, and rule 2 reads every one, since a
//! test's comments point a reader at the same files. The first two read the
//! prose: comment lines (`//`, `///`, `//!`) and the messages of `unreachable!`,
//! `panic!` and `assert*!` invocations, which a reader meets in the same way
//! as a comment.
//!
//! 1. No all-caps word in prose. Emphasis is carried by sentence structure;
//!    a name that is genuinely upper case is a code item and belongs in
//!    backticks, which this check strips before looking.
//! 2. A cited `something.rs` path names a file that exists. A citation is
//!    read wherever it appears, backticked spans included, and a cited
//!    `dir/file.rs` must match that much of a real path and not merely the
//!    file name.
//! 3. Production files carry no `#[cfg(test)]` item other than the module
//!    declaration for their test file and the imports it needs.
//! 4. Every `pub mod` in `src/lib.rs` has a row in the module table in
//!    `docs/architecture.md`.
//!
//! A fenced block inside a doc comment is source the reader compiles, so both
//! checks skip from one fence line to the next.
//!
//! Each check carries an allowlist of what is outstanding, so the rule holds
//! from here on while the existing prose is rewritten. An allowlist entry
//! names one file and one token, so a new violation cannot hide behind an old
//! one, and the lists only shrink. The prose lists are empty; what remains is
//! the test-only items production files still carry.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// Upper-case words that are the ordinary spelling of the thing they name.
const ACRONYMS: &[&str] = &[
    "API", "BDD", "CNF", "DFS", "DIMACS", "DOT", "LCA", "OOM", "RSS", "SAT", "SDD", "TDD", "UNSAT",
];

/// All-caps prose still to be rewritten, as `(file, word)`.
// generated:ALL_CAPS:begin
const ALL_CAPS_ALLOW: &[(&str, &str)] = &[];
// generated:ALL_CAPS:end

/// Citations of a file that is not under `src/` still to be repaired, as
/// `(file, cited)`.
// generated:CITED_PATH:begin
const CITED_PATH_ALLOW: &[(&str, &str)] = &[];
// generated:CITED_PATH:end

/// Outstanding test-only items in production files, as `(file, item)`.
// generated:CFG_TEST:begin
const CFG_TEST_ALLOW: &[(&str, &str)] = &[];
// generated:CFG_TEST:end

/// The one public module with no row in the boundary table: a doc-hidden shim
/// that carries the README into the reference and holds nothing else.
const UNTABLED_MODULES: &[&str] = &["readme"];

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A file whose contents are themselves tests, and so outside these rules.
fn is_test_file(rel: &str) -> bool {
    rel.contains("/tests/") || rel.starts_with("test_helpers/")
}

/// Every `.rs` file under `src/`, as a path relative to `src/`, sorted.
fn source_files() -> Vec<(String, PathBuf)> {
    let src = crate_dir().join("src");
    let mut out = Vec::new();
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("src/ is readable") {
            let path = entry.expect("a readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let rel = path
                    .strip_prefix(&src)
                    .expect("under src/")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((rel, path));
            }
        }
    }
    out.sort();
    out
}

fn non_test_sources() -> Vec<(String, PathBuf)> {
    source_files().into_iter().filter(|(rel, _)| !is_test_file(rel)).collect()
}

/// What a rule does with a backticked span: a prose rule reads the span as
/// the code it is and drops it; the citation rule reads the path inside it.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Code {
    Strip,
    Keep,
}

/// `text` with every backticked span dropped or kept, per `code`. The
/// backticks themselves always become spaces, so a span never runs into the
/// word beside it.
fn code_spans(text: &str, code: Code) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for part in text.split('`') {
        if !in_code || code == Code::Keep {
            out.push_str(part);
        }
        out.push(' ');
        in_code = !in_code;
    }
    out
}

/// The comment body of a line, or `None` if the line is not a comment. The
/// leading marker is removed and backticked spans are handled per `code`.
fn comment_prose(line: &str, code: Code) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("//") {
        return None;
    }
    let body = trimmed.trim_start_matches('/').trim_start_matches('!');
    Some(code_spans(body, code))
}

/// The maximal runs of three or more upper-case letters in `prose`.
fn all_caps_words(prose: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut run = String::new();
    for ch in prose.chars() {
        if ch.is_ascii_uppercase() {
            run.push(ch);
        } else {
            if run.len() >= 3 {
                out.push(std::mem::take(&mut run));
            } else {
                run.clear();
            }
        }
    }
    if run.len() >= 3 {
        out.push(run);
    }
    out
}

/// The `something.rs` paths cited in `prose`, with any URL removed first. A
/// citation may name directories (`apply/conjoin/sparse/mod.rs`); a leading
/// `src/` is dropped, since that is where the crate's sources are.
fn cited_paths(prose: &str) -> Vec<String> {
    let without_urls: String = prose
        .split_whitespace()
        .filter(|w| !w.contains("://"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut out = Vec::new();
    for word in
        without_urls.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '/'))
    {
        let name = word.trim_matches(['.', '/']);
        if !name.ends_with(".rs") {
            continue;
        }
        let stem = &name[..name.len() - 3];
        let segment_ok = |s: &str| {
            !s.is_empty()
                && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        };
        if stem.split('/').all(segment_ok) {
            out.push(name.strip_prefix("src/").unwrap_or(name).to_string());
        }
    }
    out
}

/// The prose of every comment line of `text`, as `(line number, prose)`. A
/// fenced block of a doc comment is source rather than prose, so the fence
/// lines and everything between them are left out.
fn prose_lines(text: &str, code: Code) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (n, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("//") {
            continue;
        }
        let body = trimmed.trim_start_matches('/').trim_start_matches('!');
        if body.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        if let Some(prose) = comment_prose(line, code) {
            out.push((n + 1, prose));
        }
    }
    out
}

/// The macros whose string arguments are a message the reader reads as prose.
const MESSAGE_MACROS: &[&str] =
    &["unreachable", "panic", "assert", "assert_eq", "assert_ne", "debug_assert", "debug_assert_eq", "debug_assert_ne"];

/// The name of a message macro invoked at `i`, with the index of the character
/// after its opening delimiter. `None` if no invocation starts here.
fn message_macro_at(chars: &[char], i: usize) -> Option<usize> {
    if i > 0 && (chars[i - 1].is_ascii_alphanumeric() || chars[i - 1] == '_') {
        return None;
    }
    let name = MESSAGE_MACROS.iter().find(|name| {
        let end = i + name.chars().count();
        chars.get(i..end).is_some_and(|w| w.iter().copied().eq(name.chars()))
            && chars.get(end) == Some(&'!')
    })?;
    let mut j = i + name.chars().count() + 1;
    while chars.get(j).is_some_and(|c| c.is_whitespace()) {
        j += 1;
    }
    matches!(chars.get(j), Some('(' | '[' | '{')).then_some(j + 1)
}

/// The contents of the string literal opening at `i`, with the index just past
/// its closing quote. Escapes are unwrapped so the prose reads as it prints.
fn read_string(chars: &[char], i: usize, line: &mut usize) -> (String, usize) {
    let mut out = String::new();
    let mut j = i + 1;
    while j < chars.len() && chars[j] != '"' {
        if chars[j] == '\\' {
            j += 1;
            match chars.get(j) {
                // A continuation swallows the newline and the next line's indent.
                Some('\n') => {
                    *line += 1;
                    j += 1;
                    while chars.get(j).is_some_and(|c| c.is_whitespace() && *c != '\n') {
                        j += 1;
                    }
                    if !out.ends_with(' ') {
                        out.push(' ');
                    }
                    continue;
                }
                Some(c) => out.push(*c),
                None => break,
            }
        } else {
            if chars[j] == '\n' {
                *line += 1;
            }
            out.push(chars[j]);
        }
        j += 1;
    }
    (out, j + 1)
}

/// The prose of every panic message in `text`, as `(line number, prose)`: the
/// string literals of a message macro's argument list, backticked spans
/// removed the same way a comment's are.
fn message_prose(text: &str, code: Code) -> Vec<(usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut line = 1usize;
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                line += usize::from(chars[i] == '\n');
                i += 1;
            }
            i = (i + 2).min(chars.len());
        } else if c == '"' {
            let at = line;
            let (lit, next) = read_string(&chars, i, &mut line);
            if depth > 0 {
                out.push((at, code_spans(&lit, code)));
            }
            i = next;
        } else if c == '\'' && (chars.get(i + 1) == Some(&'\\') || chars.get(i + 2) == Some(&'\'')) {
            // A character literal, not a lifetime: skip past its closing quote.
            i += 2;
            while i < chars.len() && chars[i] != '\'' {
                i += usize::from(chars[i] == '\\') + 1;
            }
            i += 1;
        } else if depth > 0 {
            depth += usize::from(matches!(c, '(' | '[' | '{'));
            depth -= usize::from(matches!(c, ')' | ']' | '}'));
            i += 1;
        } else if let Some(after) = message_macro_at(&chars, i) {
            depth = 1;
            i = after;
        } else {
            i += 1;
        }
    }
    out
}

/// Every line of `text` a reader reads as prose: its comments and its panic
/// messages.
fn file_prose(text: &str, code: Code) -> Vec<(usize, String)> {
    let mut out = prose_lines(text, code);
    out.extend(message_prose(text, code));
    out
}

#[test]
fn prose_carries_emphasis_by_structure_not_by_capitals() {
    let allowed: HashSet<(&str, &str)> = ALL_CAPS_ALLOW.iter().copied().collect();
    let acronyms: HashSet<&str> = ACRONYMS.iter().copied().collect();
    let mut new_hits: Vec<String> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for (rel, path) in non_test_sources() {
        let text = fs::read_to_string(&path).expect("a readable source file");
        for (n, prose) in file_prose(&text, Code::Strip) {
            for word in all_caps_words(&prose) {
                if acronyms.contains(word.as_str())
                    || allowed.contains(&(rel.as_str(), word.as_str()))
                {
                    continue;
                }
                if seen.insert((rel.clone(), word.clone())) {
                    new_hits.push(format!("{rel}:{n}: {word}"));
                }
            }
        }
    }
    assert!(new_hits.is_empty(), "all-caps words in prose:\n{}", new_hits.join("\n"));
}

#[test]
fn a_comment_cites_only_a_file_that_exists() {
    let allowed: HashSet<(&str, &str)> = CITED_PATH_ALLOW.iter().copied().collect();
    // Every source path relative to `src/`, plus the `dir.rs` spelling of a
    // `dir/mod.rs`, which is how a module is named in prose.
    let mut known: Vec<String> = Vec::new();
    for (rel, _) in source_files() {
        known.push(rel.clone());
        if let Some(dir) = rel.strip_suffix("/mod.rs") {
            known.push(format!("{dir}.rs"));
        }
    }
    // A citation names a suffix of a real path, at whole path segments.
    let resolves = |cited: &str| {
        known.iter().any(|k| k == cited || k.ends_with(&format!("/{cited}")))
    };
    let mut new_hits: Vec<String> = Vec::new();
    for (rel, path) in source_files() {
        let text = fs::read_to_string(&path).expect("a readable source file");
        for (n, prose) in file_prose(&text, Code::Keep) {
            for cited in cited_paths(&prose) {
                if resolves(&cited) || allowed.contains(&(rel.as_str(), cited.as_str())) {
                    continue;
                }
                new_hits.push(format!("{rel}:{n}: cites {cited}, which is not under src/"));
            }
        }
    }
    assert!(new_hits.is_empty(), "comments citing a file that does not exist:\n{}", new_hits.join("\n"));
}

/// The item a `#[cfg(test)]` guards, excluding fields and conditional hook calls.
fn guarded_item(lines: &[&str], at: usize) -> Option<String> {
    let decl = lines[at + 1..]
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with("#["))?;
    if decl.starts_with("use ") {
        return None;
    }
    let words: Vec<&str> = decl.split_whitespace().collect();
    if words.contains(&"mod") && decl.ends_with(';') {
        return None;
    }
    for (i, w) in words.iter().enumerate() {
        if matches!(*w, "fn" | "struct" | "enum" | "const" | "static" | "impl" | "trait" | "type")
            && let Some(next) = words.get(i + 1)
        {
            let name: String =
                next.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    if decl.starts_with("if ") {
        return None;
    }
    if let Some((field, _)) = decl.split_once(':')
        && !field.is_empty()
        && field.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    Some(decl.chars().take(40).collect())
}

#[test]
fn cfg_scan_distinguishes_hook_fields_and_calls_from_test_items() {
    for decl in [
        "refuse_after: Cell<Option<u32>>,",
        "refuse_after: Cell::new(None),",
        "if self.refuses_reserve() { return Err(OperationError::OverBudget); }",
    ] {
        assert_eq!(guarded_item(&["#[cfg(test)]", decl], 0), None);
    }
    for decl in [
        "pub(crate) fn helper() {}",
        "struct Helper;",
        "impl Limits {",
        "mod tests {",
    ] {
        assert!(guarded_item(&["#[cfg(test)]", decl], 0).is_some());
    }
}

#[test]
fn a_production_file_holds_no_test_only_item() {
    let allowed: HashSet<(&str, &str)> = CFG_TEST_ALLOW.iter().copied().collect();
    let mut new_hits: Vec<String> = Vec::new();
    for (rel, path) in non_test_sources() {
        let text = fs::read_to_string(&path).expect("a readable source file");
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            if !line.trim().starts_with("#[cfg(test)]") {
                continue;
            }
            let Some(item) = guarded_item(&lines, n) else { continue };
            if allowed.contains(&(rel.as_str(), item.as_str())) {
                continue;
            }
            new_hits.push(format!("{rel}:{}: test-only item {item}", n + 1));
        }
    }
    assert!(new_hits.is_empty(), "test-only items in production files:\n{}", new_hits.join("\n"));
}

#[test]
fn every_public_module_has_a_row_in_the_module_table() {
    // The table's module names are intra-doc links, so the brackets come out
    // before the row is matched.
    let table = fs::read_to_string(crate_dir().join("docs/architecture.md"))
        .expect("the architecture document is readable")
        .replace(['[', ']'], "");
    let lib = fs::read_to_string(crate_dir().join("src/lib.rs")).expect("lib.rs is readable");
    let untabled: HashSet<&str> = UNTABLED_MODULES.iter().copied().collect();
    let mut missing: Vec<String> = Vec::new();
    for line in lib.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub mod ") else { continue };
        let name: String =
            rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        if name.is_empty() || untabled.contains(name.as_str()) {
            continue;
        }
        if !table.contains(&format!("| `{name}` |")) {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "public modules with no row in docs/architecture.md: {}",
        missing.join(", ")
    );
}

/// A guard on the message scan: it reads the message of a panic macro, follows
/// a continuation onto the next line, and reads nothing outside such a macro —
/// not a plain call, and not a comment that names one of the macros.
#[test]
fn the_message_scan_reads_panic_messages_and_nothing_else() {
    let text = "\
fn f() {
    // panic!(\"in a comment\")
    let s = format!(\"in a call\");
    assert!(c, \"first half \\
             second half\");
    unreachable!(\"lone message\");
}
";
    let found: Vec<String> =
        message_prose(text, Code::Strip).into_iter().map(|(_, prose)| prose.trim().to_string()).collect();
    assert_eq!(found, vec!["first half second half", "lone message"]);
    assert_eq!(message_prose(text, Code::Strip)[1].0, 6, "the message is reported at its own line");
}

/// A guard on the lint itself: the source walk finds the crate, and reads more
/// than a handful of files.
#[test]
fn the_lint_walks_the_whole_crate() {
    let all = source_files();
    assert!(all.len() > 100, "the source walk found only {} files", all.len());
    assert!(
        all.iter().any(|(rel, _)| rel == "lib.rs"),
        "the source walk did not find the crate root",
    );
    assert!(
        non_test_sources().len() < all.len(),
        "no file was recognized as a test module",
    );
}
