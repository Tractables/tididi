//! Keep the walkthrough excerpts tied to the programs exercised by CI.

/// Compare Rust excerpts without indentation introduced by their surrounding scope.
fn normalized(source: &str) -> String {
    source.lines().map(str::trim).collect::<Vec<_>>().join("\n")
}

#[test]
fn walkthrough_code_comes_from_the_runnable_examples() {
    let examples = [
        ("configurations", include_str!("../docs/examples/configurations.md"), include_str!("../examples/build_minimize_count.rs")),
        ("execution", include_str!("../docs/examples/execution.md"), include_str!("../examples/build_minimize_count.rs")),
        ("probability", include_str!("../docs/examples/probability.md"), include_str!("../examples/probabilistic_query.rs")),
        ("reachability", include_str!("../docs/examples/reachability.md"), include_str!("../examples/symbolic_reachability.rs")),
        ("persistence", include_str!("../docs/examples/persistence.md"), include_str!("../examples/save_reload.rs")),
        ("vtrees", include_str!("../docs/examples/vtrees.md"), include_str!("../examples/vtree_grouping.rs")),
        ("statistics", include_str!("../docs/examples/statistics.md"), include_str!("../examples/statistic.rs")),
    ];
    for (name, markdown, source) in examples {
        let source = normalized(source);
        let mut excerpts = 0;
        let mut lines = markdown.lines();
        while let Some(line) = lines.next() {
            let Some(attributes) = line.strip_prefix("```") else { continue };
            if !attributes.starts_with("rust") && !attributes.contains("tested-example") {
                continue;
            }
            assert_eq!(line, "```rust,ignore,{class=tested-example}", "{name}: excerpts are checked against the executable example");
            let mut snippet = Vec::new();
            let mut closed = false;
            for line in lines.by_ref() {
                if line == "```" {
                    closed = true;
                    break;
                }
                snippet.push(line);
            }
            assert!(closed, "{name}: unclosed Rust excerpt");
            let snippet = normalized(&snippet.join("\n"));
            assert!(!snippet.is_empty(), "{name}: empty Rust excerpt");
            assert!(source.contains(&snippet), "{name}: excerpt does not occur in the runnable example:\n{snippet}");
            excerpts += 1;
        }
        assert!(excerpts > 0, "{name}: walkthrough has no checked excerpts");
    }
}

/// Keep links within the documentation version rustdoc is rendering.
#[test]
fn crate_documentation_uses_intra_doc_links() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut directories = vec![root.join("docs"), root.join("src")];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                directories.push(path);
            } else if matches!(path.extension().and_then(|s| s.to_str()), Some("md" | "rs")) {
                let source = std::fs::read_to_string(&path).unwrap();
                assert!(!source.contains("https://docs.rs/tididi/"),
                    "{}: use a crate:: link so rustdoc resolves the current version", path.display());
            }
        }
    }
}
