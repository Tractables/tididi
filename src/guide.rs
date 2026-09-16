//! The [API overview](api) groups circuit operations and links to their specifications.
//! The [worked examples](examples) combine these operations in applications.
//! Read the [data model](model) for the representation or the
//! [architecture reference](architecture) for the implementation.

/// Worked examples, from Boolean constraints to custom diagram traversal.
pub mod examples {
    #[doc = include_str!("../docs/examples/configurations.md")]
    pub mod configurations {}

    #[doc = include_str!("../docs/examples/execution.md")]
    pub mod execution {}

    #[doc = include_str!("../docs/examples/probability.md")]
    pub mod probability {}

    #[doc = include_str!("../docs/examples/reachability.md")]
    pub mod reachability {}

    #[doc = include_str!("../docs/examples/persistence.md")]
    pub mod persistence {}

    #[doc = include_str!("../docs/examples/vtrees.md")]
    pub mod vtrees {}

    #[doc = include_str!("../docs/examples/statistics.md")]
    pub mod statistics {}
}

#[doc = include_str!("../docs/api-guide.md")]
pub mod api {}

#[doc = include_str!("../docs/tdd.md")]
pub mod model {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}
