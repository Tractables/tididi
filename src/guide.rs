//! The [API overview](api) groups circuit operations and links to their specifications.
//! The [worked examples](examples) combine these operations in applications.
//! Read the [data model](model) for the representation or the
//! [architecture reference](architecture) for the implementation.

/// Worked examples, from Boolean constraints to custom diagram traversal.
pub mod examples {
    // The documentation integration test checks these excerpts against runnable programs.
    // Suppress the ignored-doctest badge only on blocks covered by that check.
    macro_rules! walkthrough {
        ($name:ident, $path:literal) => {
            #[doc = "<style>.example-wrap.ignore:has(> pre.tested-example) > .tooltip { display: none; }</style>"]
            #[doc = include_str!($path)]
            pub mod $name {}
        };
    }

    walkthrough!(configurations, "../docs/examples/configurations.md");
    walkthrough!(execution, "../docs/examples/execution.md");
    walkthrough!(probability, "../docs/examples/probability.md");
    walkthrough!(reachability, "../docs/examples/reachability.md");
    walkthrough!(persistence, "../docs/examples/persistence.md");
    walkthrough!(vtrees, "../docs/examples/vtrees.md");
    walkthrough!(optimization, "../docs/examples/optimization.md");
    walkthrough!(statistics, "../docs/examples/statistics.md");
}

#[doc = include_str!("../docs/api-guide.md")]
pub mod api {}

#[doc = include_str!("../docs/tdd.md")]
pub mod model {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}
