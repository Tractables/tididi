//! The comment rules in `CONTRIBUTING.md`, enforced on non-test source.
//!
//! Four checks, each on the comment lines (`//`, `///`, `//!`) of every file
//! under `src/` that is not itself a test module:
//!
//! 1. No all-caps word in prose. Emphasis is carried by sentence structure;
//!    a name that is genuinely upper case is a code item and belongs in
//!    backticks, which this check strips before looking.
//! 2. A cited `something.rs` path names a file that exists.
//! 3. Production files carry no `#[cfg(test)]` item other than the module
//!    declaration for their test file and the imports it needs.
//! 4. Every `pub mod` in `src/lib.rs` has a row in the module table in
//!    `docs/architecture.md`.
//!
//! The first two read prose only. A fenced block inside a doc comment is the
//! multi-line form of a backticked span: it is source the reader compiles, not
//! prose, so both checks skip from one fence line to the next.
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
const CFG_TEST_ALLOW: &[(&str, &str)] = &[
    ("check/canonicity.rs", "LevelAnalysis"),
    ("check/canonicity.rs", "analyze_ray_classes"),
    ("check/canonicity.rs", "check_canonicity_projective"),
    ("check/marginal_counts.rs", "check_store_counts_c3"),
    ("check/signature.rs", "eval_mass_vector"),
    ("check/signature.rs", "mod_inv"),
    ("check/signature.rs", "mod_pow"),
    ("diagram/level/mod.rs", "set_counts_state"),
    ("diagram/marginal_ref/mod.rs", "bytes"),
    ("diagram/marginal_ref/mod.rs", "try_clone"),
    ("diagram/mod.rs", "pub(crate) use pool::reset_level;"),
    ("diagram/primitives.rs", "leaf"),
    ("diagram/primitives.rs", "tombstone"),
    ("diagram/tdd.rs", "contract_worklist"),
    ("diagram/tdd.rs", "reachable_from_root_level"),
    ("diagram/tdd.rs", "seed_contract_worklist"),
    ("diagram/tdd.rs", "seed_leaf_worklist"),
    ("engine/limits/mod.rs", "grant_every_reserve"),
    ("engine/limits/mod.rs", "pin_reduce_poll_stride"),
    ("engine/limits/mod.rs", "refuse_nth_reserve"),
    ("engine/mod.rs", "pub(crate) use limits::DENSE_GROWTH_DECI"),
    ("engine/mod.rs", "pub(crate) use memory::{vas_headroom_wit"),
    ("query/count/mod.rs", "node_counts_pinned_mode"),
    ("query/count/mod.rs", "pinned_counts"),
    ("query/mod.rs", "pub(crate) use count::pinned_counts;"),
    ("reduce/contract/mod.rs", "pub(crate) use strategies::contract_all_"),
    ("reduce/contract/pair_fusion/mod.rs", "fuse_pairs"),
    ("reduce/mod.rs", "pub(crate) use content_twins::canonicali"),
    ("session.rs", "with_stop_now"),
    ("session.rs", "with_tuning"),
    ("value_fold/mod.rs", "all_u64"),
    ("value_fold/mod.rs", "clone_guarded"),
    ("value_fold/mod.rs", "has_big"),
    ("value_fold/mod.rs", "push_i"),
    ("value_fold/mod.rs", "try_clone"),
    ("vtree/rotate.rs", "abandon"),
    ("vtree/rotate.rs", "rotate_left"),
    ("vtree/rotate.rs", "rotate_right"),
    ("vtree/rotate.rs", "unrotate_left"),
    ("vtree/rotate.rs", "unrotate_right"),
    ("vtree/topo.rs", "rebuild"),
    ("vtree/topo.rs", "rebuild_topo"),
];
// generated:CFG_TEST:end

/// The one public module with no row in the boundary table: a doc-hidden shim
/// that carries the README into the reference and holds nothing else.
const UNTABLED_MODULES: &[&str] = &["readme"];

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A file whose contents are themselves tests, and so outside these rules.
fn is_test_file(rel: &str) -> bool {
    rel.ends_with("_tests.rs")
        || rel.ends_with("/tests.rs")
        || rel.contains("/tests/")
        || rel.starts_with("test_helpers/")
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

/// The comment body of a line, or `None` if the line is not a comment. The
/// leading marker and every backticked span are removed, so a code name in
/// backticks is not read as prose.
fn comment_prose(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("//") {
        return None;
    }
    let body = trimmed.trim_start_matches('/').trim_start_matches('!');
    let mut out = String::new();
    let mut in_code = false;
    for part in body.split('`') {
        if !in_code {
            out.push_str(part);
            out.push(' ');
        }
        in_code = !in_code;
    }
    Some(out)
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

/// The `something.rs` names cited in `prose`, with any URL removed first.
fn cited_paths(prose: &str) -> Vec<String> {
    let without_urls: String = prose
        .split_whitespace()
        .filter(|w| !w.contains("://"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut out = Vec::new();
    for word in without_urls.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.')) {
        let name = word.trim_matches('.');
        if !name.ends_with(".rs") {
            continue;
        }
        let stem = &name[..name.len() - 3];
        if !stem.is_empty()
            && stem.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && stem.chars().next().is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        {
            out.push(name.to_string());
        }
    }
    out
}

/// The prose of every comment line of `text`, as `(line number, prose)`. A
/// fenced block of a doc comment is source rather than prose, so the fence
/// lines and everything between them are left out.
fn prose_lines(text: &str) -> Vec<(usize, String)> {
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
        if let Some(prose) = comment_prose(line) {
            out.push((n + 1, prose));
        }
    }
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
        for (n, prose) in prose_lines(&text) {
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
    let mut known: HashSet<String> = HashSet::new();
    for (rel, _) in source_files() {
        let file = rel.rsplit('/').next().expect("a file name").to_string();
        known.insert(file);
        if rel.ends_with("mod.rs")
            && let Some(dir) = rel.rsplit('/').nth(1)
        {
            known.insert(format!("{dir}.rs"));
        }
    }
    let mut new_hits: Vec<String> = Vec::new();
    for (rel, path) in non_test_sources() {
        let text = fs::read_to_string(&path).expect("a readable source file");
        for (n, prose) in prose_lines(&text) {
            for cited in cited_paths(&prose) {
                if known.contains(&cited) || allowed.contains(&(rel.as_str(), cited.as_str())) {
                    continue;
                }
                new_hits.push(format!("{rel}:{n}: cites {cited}, which is not under src/"));
            }
        }
    }
    assert!(new_hits.is_empty(), "comments citing a file that does not exist:\n{}", new_hits.join("\n"));
}

/// The identifier a `#[cfg(test)]` guards, skipping any further attributes.
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
    Some(decl.chars().take(40).collect())
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
