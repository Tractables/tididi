//! The [API overview](api) groups circuit operations and links to their specifications.
//! The [worked examples](examples) combine these operations in applications.
//! Read the [data model](model) for the representation or the
//! [architecture reference](architecture) for the implementation.

// Keep the index, sidebar and module declarations in one reading order.
macro_rules! walkthroughs {
    ($($name:ident => ($title:literal, $description:literal)),+ $(,)?) => {
        macro_rules! navigation {
            () => {
                concat!(
                    "<script>(() => { const examples = [",
                    $("[\"", stringify!($name), "\",\"", $title, "\"],",)+
                    "];", include_str!("../docs/example-navigation.js"), "})();</script>"
                )
            };
        }

        #[doc = concat!(
            "Worked examples, from basic circuit operations to custom calculations.\n\n",
            $("1. [", $title, "](examples::", stringify!($name), ") — ", $description, "\n",)+
        )]
        #[doc = navigation!()]
        pub mod examples {
            $(
                #[doc = include_str!(concat!("../docs/examples/", stringify!($name), ".md"))]
                // Only excerpts checked against runnable programs suppress the badge.
                #[doc = "<style>.example-wrap.ignore:has(> pre.tested-example) > .tooltip { display: none; }</style>"]
                #[doc = navigation!()]
                pub mod $name {}
            )+
        }
    };
}

walkthroughs! {
    configurations => ("Configurations", "Build rules, count solutions, and update a user's choices."),
    probability => ("Probabilities", "Evaluate events under changing probabilities."),
    reachability => ("Reachability", "Explore the states reachable through a transition system."),
    persistence => ("Saving and loading", "Save circuits and restore them for later use."),
    vtrees => ("Variable grouping", "Compare circuit sizes under different vtrees."),
    execution => ("Execution limits", "Bound work and release retained scratch buffers."),
    optimization => ("Minimum costs", "Define a custom evaluation algebra."),
    statistics => ("Circuit statistics", "Traverse nodes and pairs for a custom measurement."),
}

#[doc = include_str!("../docs/api-guide.md")]
pub mod api {}

#[doc = include_str!("../docs/tdd.md")]
pub mod model {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}
