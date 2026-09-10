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
//! Each check carries an allowlist of what is outstanding, so the rule holds
//! from here on while the existing prose is rewritten. An allowlist entry
//! names one file and one token, so a new violation cannot hide behind an old
//! one, and the lists only shrink.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// Upper-case words that are the ordinary spelling of the thing they name.
const ACRONYMS: &[&str] = &[
    "API", "BDD", "CNF", "DFS", "DIMACS", "DOT", "LCA", "OOM", "RSS", "SAT", "SDD", "TDD", "UNSAT",
];

/// Outstanding all-caps prose, as `(file, word)`.
// generated:ALL_CAPS:begin
const ALL_CAPS_ALLOW: &[(&str, &str)] = &[
    ("apply/condition.rs", "ALREADY"),
    ("apply/condition.rs", "AND"),
    ("apply/condition.rs", "CONJOINING"),
    ("apply/condition.rs", "GROWING"),
    ("apply/condition.rs", "NON"),
    ("apply/condition.rs", "OWN"),
    ("apply/condition.rs", "PLACE"),
    ("apply/condition.rs", "PREFIX"),
    ("apply/condition.rs", "SET"),
    ("apply/condition.rs", "SINGLE"),
    ("apply/condition.rs", "WEIGHTED"),
    ("apply/conjoin/budget.rs", "CHARGE"),
    ("apply/conjoin/budget.rs", "LLVM"),
    ("apply/conjoin/cell/columns.rs", "CELL"),
    ("apply/conjoin/cell/columns.rs", "LEVEL"),
    ("apply/conjoin/cell/columns.rs", "ONCE"),
    ("apply/conjoin/cell/columns.rs", "OUTPUT"),
    ("apply/conjoin/cell/columns.rs", "OWN"),
    ("apply/conjoin/cell/columns.rs", "READ"),
    ("apply/conjoin/cell/columns.rs", "SAFETY"),
    ("apply/conjoin/cell/columns.rs", "THE"),
    ("apply/conjoin/cell/kernel.rs", "PRODUCT"),
    ("apply/conjoin/cell/mod.rs", "ONCE"),
    ("apply/conjoin/cell/rows.rs", "CHILD"),
    ("apply/conjoin/cell/rows.rs", "DENSE"),
    ("apply/conjoin/cell/rows.rs", "FALSE"),
    ("apply/conjoin/cell/rows.rs", "MAX"),
    ("apply/conjoin/cell/rows.rs", "ONCE"),
    ("apply/conjoin/cell/rows.rs", "PRODUCT"),
    ("apply/conjoin/cell/rows.rs", "TLS"),
    ("apply/conjoin/cell/rows.rs", "TRUE"),
    ("apply/conjoin/cell/rows.rs", "ZERO"),
    ("apply/conjoin/cell/rows_stream.rs", "PRODUCT"),
    ("apply/conjoin/cell/rows_stream.rs", "VIEWS"),
    ("apply/conjoin/child_lookup.rs", "OOB"),
    ("apply/conjoin/child_lookup.rs", "SAFETY"),
    ("apply/conjoin/drive/level.rs", "EXACT"),
    ("apply/conjoin/drive/level.rs", "ONCE"),
    ("apply/conjoin/drive/level.rs", "PLACE"),
    ("apply/conjoin/drive/level.rs", "PRODUCT"),
    ("apply/conjoin/drive/level.rs", "THE"),
    ("apply/conjoin/drive/mod.rs", "CONSUMED"),
    ("apply/conjoin/drive/mod.rs", "COUNT"),
    ("apply/conjoin/drive/mod.rs", "MARGINAL"),
    ("apply/conjoin/drive/mod.rs", "OUTPUT"),
    ("apply/conjoin/drive/mod.rs", "PRODUCT"),
    ("apply/conjoin/drive/mod.rs", "START"),
    ("apply/conjoin/drive/mod.rs", "STRUCTURAL"),
    ("apply/conjoin/drive/mod.rs", "ZERO"),
    ("apply/conjoin/grid_arena.rs", "PRODUCT"),
    ("apply/conjoin/grid_arena.rs", "VALIDITY"),
    ("apply/conjoin/identity.rs", "KEEP"),
    ("apply/conjoin/identity.rs", "MAX"),
    ("apply/conjoin/identity.rs", "TRUE"),
    ("apply/conjoin/liveness.rs", "PRODUCT"),
    ("apply/conjoin/marginal_plan.rs", "AND"),
    ("apply/conjoin/marginal_plan.rs", "CARRIER"),
    ("apply/conjoin/marginal_plan.rs", "COUNT"),
    ("apply/conjoin/marginal_plan.rs", "DIRECTLY"),
    ("apply/conjoin/marginal_plan.rs", "ENTRY"),
    ("apply/conjoin/marginal_plan.rs", "IDENTITY"),
    ("apply/conjoin/marginal_plan.rs", "INLINE"),
    ("apply/conjoin/marginal_plan.rs", "INTO"),
    ("apply/conjoin/marginal_plan.rs", "MARGINAL"),
    ("apply/conjoin/marginal_plan.rs", "MASK"),
    ("apply/conjoin/marginal_plan.rs", "MAX"),
    ("apply/conjoin/marginal_plan.rs", "MODEL"),
    ("apply/conjoin/marginal_plan.rs", "OOB"),
    ("apply/conjoin/marginal_plan.rs", "OPERAND"),
    ("apply/conjoin/marginal_plan.rs", "OUT"),
    ("apply/conjoin/marginal_plan.rs", "OUTPUT"),
    ("apply/conjoin/marginal_plan.rs", "THIS"),
    ("apply/conjoin/marginal_plan.rs", "THREE"),
    ("apply/conjoin/marginal_plan.rs", "VALUE"),
    ("apply/conjoin/marginal_plan.rs", "WAS"),
    ("apply/conjoin/marginal_plan.rs", "WHEN"),
    ("apply/conjoin/marginal_plan.rs", "ZERO"),
    ("apply/conjoin/mod.rs", "AND"),
    ("apply/conjoin/mod.rs", "CONSUMED"),
    ("apply/conjoin/mod.rs", "HERE"),
    ("apply/conjoin/mod.rs", "NODE"),
    ("apply/conjoin/output.rs", "START"),
    ("apply/conjoin/plan.rs", "DESCENDANT"),
    ("apply/conjoin/plan.rs", "FULL"),
    ("apply/conjoin/plan.rs", "RESTRICTED"),
    ("apply/conjoin/plan.rs", "SWAP"),
    ("apply/conjoin/restrict.rs", "NEITHER"),
    ("apply/conjoin/restrict.rs", "THE"),
    ("apply/conjoin/restrict_plan.rs", "LEAF"),
    ("apply/conjoin/restrict_plan.rs", "MISSING"),
    ("apply/conjoin/restrict_plan.rs", "NARROWER"),
    ("apply/conjoin/restrict_plan.rs", "OFF"),
    ("apply/conjoin/restrict_plan.rs", "REFERENCES"),
    ("apply/conjoin/restrict_plan.rs", "ZERO"),
    ("apply/conjoin/route.rs", "ASYMMETRIC"),
    ("apply/conjoin/route.rs", "EITHER"),
    ("apply/conjoin/route.rs", "OTHER"),
    ("apply/conjoin/route.rs", "OUTPUT"),
    ("apply/conjoin/setup.rs", "DENSE"),
    ("apply/conjoin/setup.rs", "VAS"),
    ("apply/conjoin/sparse/config.rs", "PAIRS"),
    ("apply/conjoin/sparse/config.rs", "SPARSE"),
    ("apply/conjoin/sparse/config.rs", "THRESHOLD"),
    ("apply/conjoin/sparse/index.rs", "OTHER"),
    ("apply/conjoin/sparse/level.rs", "CONJOIN"),
    ("apply/conjoin/sparse/level.rs", "COUNT"),
    ("apply/conjoin/sparse/level.rs", "FALSE"),
    ("apply/conjoin/sparse/level.rs", "GRID"),
    ("apply/conjoin/sparse/level.rs", "STRUCTURAL"),
    ("apply/conjoin/sparse/level.rs", "THE"),
    ("apply/conjoin/sparse/level.rs", "TRUE"),
    ("apply/conjoin/sparse/level.rs", "ZERO"),
    ("apply/conjoin/sparse/mod.rs", "SPARSE"),
    ("apply/conjoin/sparse/mod.rs", "THRESHOLD"),
    ("apply/conjoin/sparse/scatter.rs", "CONJOIN"),
    ("apply/conjoin/sparse/scatter.rs", "FILTERED"),
    ("apply/conjoin/sparse/scatter.rs", "GRID"),
    ("apply/conjoin/sparse/scatter.rs", "THE"),
    ("apply/conjoin/streaming_marginal/count.rs", "CLEAR"),
    ("apply/conjoin/streaming_marginal/count.rs", "COUNT"),
    ("apply/conjoin/streaming_marginal/count.rs", "INLINE"),
    ("apply/conjoin/streaming_marginal/count.rs", "LEAF"),
    ("apply/conjoin/streaming_marginal/count.rs", "LLVM"),
    ("apply/conjoin/streaming_marginal/count.rs", "MARGINAL"),
    ("apply/conjoin/streaming_marginal/count.rs", "MAX"),
    ("apply/conjoin/streaming_marginal/count.rs", "OVERFLOW"),
    ("apply/conjoin/streaming_marginal/count.rs", "PRIOR"),
    ("apply/conjoin/streaming_marginal/count.rs", "SAFETY"),
    ("apply/conjoin/streaming_marginal/count.rs", "SET"),
    ("apply/conjoin/streaming_marginal/fold.rs", "BORROW"),
    ("apply/conjoin/streaming_marginal/fold.rs", "FREE"),
    ("apply/conjoin/streaming_marginal/level.rs", "FALLIBLE"),
    ("apply/conjoin/streaming_marginal/weight.rs", "HERE"),
    ("apply/conjoin/streaming_marginal/weight.rs", "LEAF"),
    ("apply/conjoin/streaming_marginal/weight.rs", "THIS"),
    ("apply/conjoin/streaming_marginal/weight.rs", "WIDTH"),
    ("apply/conjoin_clause/emit.rs", "PRODUCT"),
    ("apply/conjoin_clause/mod.rs", "AND"),
    ("apply/conjoin_clause/mod.rs", "LENGTH"),
    ("apply/conjoin_clause/mod.rs", "MOVES"),
    ("apply/conjoin_clause/mod.rs", "ZERO"),
    ("apply/conjoin_clause/pairs.rs", "FUSED"),
    ("apply/conjoin_clause/rebuild.rs", "DIRECTLY"),
    ("apply/conjoin_clause/rebuild.rs", "FUSED"),
    ("apply/conjoin_clause/rebuild.rs", "HERE"),
    ("apply/conjoin_clause/rebuild.rs", "INPUT"),
    ("apply/conjoin_clause/rebuild.rs", "PAIR"),
    ("apply/conjoin_clause/rebuild.rs", "PER"),
    ("apply/conjoin_clause/rebuild.rs", "PRODUCT"),
    ("apply/conjoin_clause/spine.rs", "INTERNAL"),
    ("apply/conjoin_clause/spine.rs", "PRODUCT"),
    ("apply/conjoin_clause/spine.rs", "UNION"),
    ("apply/disjoin.rs", "AND"),
    ("apply/disjoin.rs", "DPLL"),
    ("apply/leaf.rs", "PRODUCT"),
    ("apply/negate.rs", "AIG"),
    ("apply/negate.rs", "SET"),
    ("apply/negate.rs", "XOR"),
    ("apply/project/structural.rs", "ANCESTOR"),
    ("apply/project/structural.rs", "AND"),
    ("apply/project/structural.rs", "DISJOINT"),
    ("apply/project/structural.rs", "GRANDPARENT"),
    ("apply/project/structural.rs", "MISCOUNT"),
    ("apply/project/structural.rs", "MULTIPLICITY"),
    ("apply/project/structural.rs", "NEW"),
    ("apply/project/structural.rs", "PATH"),
    ("apply/project/structural.rs", "REWRITTEN"),
    ("apply/project/structural.rs", "SAFE"),
    ("apply/project/structural.rs", "SETS"),
    ("apply/project/structural.rs", "SIBLING"),
    ("apply/project/structural.rs", "UNORDERED"),
    ("apply/restrict.rs", "AND"),
    ("apply/restrict.rs", "CHANGE"),
    ("apply/restrict.rs", "COUNT"),
    ("apply/restrict.rs", "DAG"),
    ("apply/restrict.rs", "MARGINAL"),
    ("apply/restrict.rs", "PAIR"),
    ("apply/restrict.rs", "UNVISITED"),
    ("apply/restrict.rs", "VALUE"),
    ("apply/restrict.rs", "ZERO"),
    ("build.rs", "IDX"),
    ("build.rs", "LEAF"),
    ("build.rs", "ONE"),
    ("build.rs", "ZERO"),
    ("check/canonicity.rs", "AND"),
    ("check/canonicity.rs", "ATDD"),
    ("check/marginal.rs", "AND"),
    ("check/marginal.rs", "ARE"),
    ("check/marginal.rs", "MARGINAL"),
    ("check/marginal.rs", "OUT"),
    ("check/marginal.rs", "PARENTS"),
    ("check/marginal.rs", "SCOPE"),
    ("check/marginal_counts.rs", "VALUES"),
    ("check/signature.rs", "INLINE"),
    ("check/signature.rs", "LEAF"),
    ("check/signature.rs", "MARGINAL"),
    ("check/signature.rs", "MASS"),
    ("check/signature.rs", "MAX"),
    ("check/signature.rs", "PRIME"),
    ("check/signature.rs", "WIDTH"),
    ("diagram/builder.rs", "IDX"),
    ("diagram/builder.rs", "LEAF"),
    ("diagram/builder.rs", "MARGINALIZED"),
    ("diagram/builder.rs", "NEG"),
    ("diagram/builder.rs", "POS"),
    ("diagram/level/arena.rs", "AND"),
    ("diagram/level/arena.rs", "BIT"),
    ("diagram/level/arena.rs", "COMPACT"),
    ("diagram/level/arena.rs", "DEAD"),
    ("diagram/level/arena.rs", "ENTIRE"),
    ("diagram/level/arena.rs", "LEAF"),
    ("diagram/level/arena.rs", "MIN"),
    ("diagram/level/arena.rs", "MULTI"),
    ("diagram/level/arena.rs", "OWN"),
    ("diagram/level/arena.rs", "PAIRS"),
    ("diagram/level/marginal.rs", "CELL"),
    ("diagram/level/marginal.rs", "INLINABLE"),
    ("diagram/level/marginal.rs", "INLINE"),
    ("diagram/level/marginal.rs", "MARGINAL"),
    ("diagram/level/marginal.rs", "MAX"),
    ("diagram/level/marginal.rs", "OPTIMISATION"),
    ("diagram/level/marginal.rs", "SLOT"),
    ("diagram/level/marginal.rs", "TAGGED"),
    ("diagram/level/marginal.rs", "ZERO"),
    ("diagram/level/mod.rs", "AND"),
    ("diagram/level/mod.rs", "COUNTS"),
    ("diagram/level/mod.rs", "EXPLICIT"),
    ("diagram/level/mod.rs", "INLINE"),
    ("diagram/level/mod.rs", "INLINED"),
    ("diagram/level/mod.rs", "LEFT"),
    ("diagram/level/mod.rs", "MARGINAL"),
    ("diagram/level/mod.rs", "METRIC"),
    ("diagram/level/mod.rs", "MODEL"),
    ("diagram/level/mod.rs", "NOTE"),
    ("diagram/level/mod.rs", "THIS"),
    ("diagram/level/mod.rs", "TRIGGER"),
    ("diagram/level/pairs.rs", "SAFETY"),
    ("diagram/level/pairs.rs", "TAOCP"),
    ("diagram/marginal_ref/mod.rs", "ASCENDING"),
    ("diagram/marginal_ref/mod.rs", "MANY"),
    ("diagram/marginal_ref/mod.rs", "OPTIMISATION"),
    ("diagram/marginal_ref/mod.rs", "OVERFLOW"),
    ("diagram/marginal_ref/mod.rs", "SAFE"),
    ("diagram/marginal_ref/mod.rs", "ZERO"),
    ("diagram/marginal_ref/swap.rs", "INFALLIBLE"),
    ("diagram/marginal_ref/swap.rs", "NOTHING"),
    ("diagram/marginal_ref/swap.rs", "OOB"),
    ("diagram/marginal_ref/swap.rs", "OUTPUT"),
    ("diagram/marginal_ref/swap.rs", "OVERFLOW"),
    ("diagram/marginal_ref/swap.rs", "PER"),
    ("diagram/marginal_ref/swap.rs", "SINGLE"),
    ("diagram/marginal_ref/swap.rs", "SOURCE"),
    ("diagram/marginal_ref/swap.rs", "SPARSE"),
    ("diagram/marginal_ref/swap.rs", "WEIGHTED"),
    ("diagram/marginal_ref/swap.rs", "ZERO"),
    ("diagram/marginal_ref/tag.rs", "ALREADY"),
    ("diagram/marginal_ref/tag.rs", "OOB"),
    ("diagram/marginal_ref/tag.rs", "THIS"),
    ("diagram/mod.rs", "MAX"),
    ("diagram/mod.rs", "ZERO"),
    ("diagram/pool.rs", "LEVEL"),
    ("diagram/pool.rs", "RESIZED"),
    ("diagram/pool.rs", "TALLY"),
    ("diagram/primitives.rs", "BIT"),
    ("diagram/primitives.rs", "INLINE"),
    ("diagram/primitives.rs", "LEAF"),
    ("diagram/primitives.rs", "MULTI"),
    ("diagram/primitives.rs", "RANGE"),
    ("diagram/primitives.rs", "SENTINEL"),
    ("diagram/semiring/mod.rs", "WMC"),
    ("diagram/semiring/rational.rs", "PWMC"),
    ("diagram/semiring/weight.rs", "ALREADY"),
    ("diagram/semiring/weight.rs", "AND"),
    ("diagram/semiring/weight.rs", "WMC"),
    ("diagram/tdd.rs", "CALLER"),
    ("diagram/tdd.rs", "EVERYWHERE"),
    ("diagram/tdd.rs", "INPUT"),
    ("diagram/tdd.rs", "INTO"),
    ("diagram/tdd.rs", "KNOWS"),
    ("diagram/tdd.rs", "LEAF"),
    ("diagram/tdd.rs", "MULTISET"),
    ("diagram/tdd.rs", "NOTE"),
    ("diagram/tdd.rs", "SET"),
    ("diagram/tdd.rs", "SHAPE"),
    ("diagram/tdd.rs", "SUPPLIED"),
    ("diagram/tdd.rs", "THRESHOLD"),
    ("diagram/tdd.rs", "WHAT"),
    ("diagram/tdd.rs", "ZERO"),
    ("diagram/weights.rs", "COMPACTION"),
    ("diagram/weights.rs", "DELIBERATELY"),
    ("diagram/weights.rs", "DOMAIN"),
    ("diagram/weights.rs", "PLACE"),
    ("diagram/weights.rs", "VALUE"),
    ("diagram/weights.rs", "WITHOUT"),
    ("engine/limits/growth.rs", "BEGINNING"),
    ("engine/limits/growth.rs", "CONCLUDE"),
    ("engine/limits/growth.rs", "FLOOR"),
    ("engine/limits/growth.rs", "LLVM"),
    ("engine/limits/growth.rs", "NEXT"),
    ("engine/limits/growth.rs", "OUTPUT"),
    ("engine/limits/growth.rs", "OVER"),
    ("engine/limits/growth.rs", "REACHING"),
    ("engine/limits/growth.rs", "SCHEDULE"),
    ("engine/limits/growth.rs", "SOUND"),
    ("engine/limits/mod.rs", "ASKED"),
    ("engine/limits/mod.rs", "REFUSED"),
    ("engine/limits/policy.rs", "SIGABRT"),
    ("engine/memory.rs", "SIGABRT"),
    ("engine/poll.rs", "WITHOUT"),
    ("engine/pool.rs", "LAST"),
    ("engine/stop.rs", "PAIRS"),
    ("engine/stop.rs", "PLACE"),
    ("engine/stop.rs", "SIZE"),
    ("io/dot.rs", "ZERO"),
    ("io/mod.rs", "COUNTS"),
    ("io/tdd_format.rs", "AND"),
    ("io/tdd_format.rs", "ARGUMENT"),
    ("io/tdd_format.rs", "CARRY"),
    ("io/tdd_format.rs", "DOES"),
    ("io/tdd_format.rs", "FORMAT"),
    ("io/tdd_format.rs", "LOCAL"),
    ("io/tdd_format.rs", "SAFETY"),
    ("io/tdd_format.rs", "THE"),
    ("io/tdd_format.rs", "VTREE"),
    ("io/tdd_format.rs", "WHAT"),
    ("io/tdd_format.rs", "ZERO"),
    ("marginal/column.rs", "DEDUP"),
    ("marginal/column.rs", "EMIT"),
    ("marginal/column.rs", "FORBIDDEN"),
    ("marginal/column.rs", "HERE"),
    ("marginal/column.rs", "SHARED"),
    ("marginal/column.rs", "SITE"),
    ("marginal/fold.rs", "BETWEEN"),
    ("marginal/fold.rs", "FIRST"),
    ("marginal/fold.rs", "LAST"),
    ("marginal/leaf.rs", "ASCENDING"),
    ("marginal/leaf.rs", "BELOW"),
    ("marginal/leaf.rs", "CANONICAL"),
    ("marginal/leaf.rs", "CANONICALIZATION"),
    ("marginal/leaf.rs", "CHECK"),
    ("marginal/leaf.rs", "CREATES"),
    ("marginal/leaf.rs", "DOES"),
    ("marginal/leaf.rs", "EQUAL"),
    ("marginal/leaf.rs", "INVARIANT"),
    ("marginal/leaf.rs", "LABEL"),
    ("marginal/leaf.rs", "LEAF"),
    ("marginal/leaf.rs", "LEAVES"),
    ("marginal/leaf.rs", "LOOKUP"),
    ("marginal/leaf.rs", "MARGINAL"),
    ("marginal/leaf.rs", "NEIGHBOURING"),
    ("marginal/leaf.rs", "OTHER"),
    ("marginal/leaf.rs", "OUTPUT"),
    ("marginal/leaf.rs", "PIN"),
    ("marginal/leaf.rs", "PRIVATE"),
    ("marginal/leaf.rs", "REF"),
    ("marginal/leaf.rs", "SHARED"),
    ("marginal/leaf.rs", "SMALLEST"),
    ("marginal/leaf.rs", "SOUNDNESS"),
    ("marginal/leaf.rs", "STRUCTURAL"),
    ("marginal/leaf.rs", "SUBSUMED"),
    ("marginal/leaf.rs", "SUM"),
    ("marginal/leaf.rs", "THE"),
    ("marginal/leaf.rs", "VALUE"),
    ("marginal/leaf.rs", "ZERO"),
    ("marginal/mod.rs", "LEAF"),
    ("marginal/mod.rs", "THIS"),
    ("marginal/mod.rs", "ZERO"),
    ("marginal/store.rs", "ANOTHER"),
    ("marginal/store.rs", "CLEAR"),
    ("marginal/store.rs", "COMPACTED"),
    ("marginal/store.rs", "EQUAL"),
    ("marginal/store.rs", "EXCEPT"),
    ("marginal/store.rs", "FIRST"),
    ("marginal/store.rs", "INTERNAL"),
    ("marginal/store.rs", "INVARIANT"),
    ("marginal/store.rs", "LABEL"),
    ("marginal/store.rs", "LEAF"),
    ("marginal/store.rs", "MIRRORS"),
    ("marginal/store.rs", "PIN"),
    ("marginal/store.rs", "PLACE"),
    ("marginal/store.rs", "PRE"),
    ("marginal/store.rs", "READ"),
    ("marginal/store.rs", "SET"),
    ("marginal/store.rs", "SHARED"),
    ("marginal/store.rs", "SOUNDNESS"),
    ("marginal/store.rs", "STORE"),
    ("marginal/store.rs", "STRUCTURAL"),
    ("marginal/store.rs", "THE"),
    ("marginal/store.rs", "THIS"),
    ("marginal/store.rs", "WEIGHTED"),
    ("marginal/store.rs", "ZERO"),
    ("query/count/incremental.rs", "OWNS"),
    ("query/count/incremental.rs", "ZERO"),
    ("query/count/mod.rs", "WALK"),
    ("query/count/mod.rs", "ZERO"),
    ("query/reduction.rs", "AND"),
    ("query/reduction.rs", "OTHER"),
    ("query/reduction.rs", "ZERO"),
    ("query/sat.rs", "AND"),
    ("query/sat.rs", "NON"),
    ("query/sat.rs", "ZERO"),
    ("query/semiring/mod.rs", "LEAF"),
    ("reduce/content_twins.rs", "ELIGIBILITY"),
    ("reduce/content_twins.rs", "END"),
    ("reduce/content_twins.rs", "FORCES"),
    ("reduce/content_twins.rs", "GALLOPING"),
    ("reduce/content_twins.rs", "POLICY"),
    ("reduce/content_twins.rs", "PROBE"),
    ("reduce/content_twins.rs", "TERMINATION"),
    ("reduce/content_twins.rs", "THIS"),
    ("reduce/content_twins.rs", "VALUE"),
    ("reduce/content_twins.rs", "WEIGHTED"),
    ("reduce/contract/content_twin.rs", "AND"),
    ("reduce/contract/content_twin.rs", "KEYS"),
    ("reduce/contract/content_twin.rs", "MECHANISM"),
    ("reduce/contract/content_twin.rs", "NOTE"),
    ("reduce/contract/content_twin.rs", "PASS"),
    ("reduce/contract/content_twin.rs", "THIS"),
    ("reduce/contract/content_twin.rs", "XOR"),
    ("reduce/contract/contract_leaf.rs", "PLACE"),
    ("reduce/contract/contract_leaf.rs", "STRUCTURAL"),
    ("reduce/contract/contract_leaf.rs", "TRIGGER"),
    ("reduce/contract/contract_leaf.rs", "WEIGHT"),
    ("reduce/contract/duplicate_pair_resolve.rs", "CLONED"),
    ("reduce/contract/duplicate_pair_resolve.rs", "INLINE"),
    ("reduce/contract/duplicate_pair_resolve.rs", "LEAF"),
    ("reduce/contract/duplicate_pair_resolve.rs", "NODES"),
    ("reduce/contract/duplicate_pair_resolve.rs", "PUSHED"),
    ("reduce/contract/duplicate_pair_resolve.rs", "REUSED"),
    ("reduce/contract/duplicate_pair_resolve.rs", "SIZE"),
    ("reduce/contract/duplicate_pair_resolve.rs", "WHOLE"),
    ("reduce/contract/duplicate_pair_scale.rs", "ASCENDING"),
    ("reduce/contract/duplicate_pair_scale.rs", "CANONICAL"),
    ("reduce/contract/duplicate_pair_scale.rs", "DECLINING"),
    ("reduce/contract/duplicate_pair_scale.rs", "EMPTY"),
    ("reduce/contract/duplicate_pair_scale.rs", "FALLBACK"),
    ("reduce/contract/duplicate_pair_scale.rs", "GLOBAL"),
    ("reduce/contract/duplicate_pair_scale.rs", "HERE"),
    ("reduce/contract/duplicate_pair_scale.rs", "INLINE"),
    ("reduce/contract/duplicate_pair_scale.rs", "INTEGER"),
    ("reduce/contract/duplicate_pair_scale.rs", "INTERNAL"),
    ("reduce/contract/duplicate_pair_scale.rs", "INVARIANT"),
    ("reduce/contract/duplicate_pair_scale.rs", "LABEL"),
    ("reduce/contract/duplicate_pair_scale.rs", "LEAF"),
    ("reduce/contract/duplicate_pair_scale.rs", "LEFT"),
    ("reduce/contract/duplicate_pair_scale.rs", "LOOKUP"),
    ("reduce/contract/duplicate_pair_scale.rs", "MARGINAL"),
    ("reduce/contract/duplicate_pair_scale.rs", "MAX"),
    ("reduce/contract/duplicate_pair_scale.rs", "NEIGHBOURING"),
    ("reduce/contract/duplicate_pair_scale.rs", "OOB"),
    ("reduce/contract/duplicate_pair_scale.rs", "OTHER"),
    ("reduce/contract/duplicate_pair_scale.rs", "PIN"),
    ("reduce/contract/duplicate_pair_scale.rs", "THE"),
    ("reduce/contract/duplicate_pair_scale.rs", "WEIGHT"),
    ("reduce/contract/duplicate_pair_scale.rs", "WIDTH"),
    ("reduce/contract/duplicate_pair_scale.rs", "ZERO"),
    ("reduce/contract/fingerprint/groups.rs", "DIST"),
    ("reduce/contract/fingerprint/groups.rs", "EMPTY"),
    ("reduce/contract/fingerprint/groups.rs", "MAX"),
    ("reduce/contract/fingerprint/groups.rs", "MEMBERS"),
    ("reduce/contract/fingerprint/groups.rs", "SET"),
    ("reduce/contract/fingerprint/groups.rs", "SIGABRT"),
    ("reduce/contract/fingerprint/groups.rs", "UNCONDITIONAL"),
    ("reduce/contract/fingerprint/mod.rs", "DIST"),
    ("reduce/contract/fingerprint/mod.rs", "FREE"),
    ("reduce/contract/fingerprint/mod.rs", "INDEX"),
    ("reduce/contract/fingerprint/mod.rs", "PRELUDE"),
    ("reduce/contract/fingerprint/mod.rs", "XOR"),
    ("reduce/contract/merge/data.rs", "GROWS"),
    ("reduce/contract/merge/data.rs", "HERE"),
    ("reduce/contract/merge/data.rs", "NODE"),
    ("reduce/contract/merge/data.rs", "ONCE"),
    ("reduce/contract/merge/data.rs", "WHOLE"),
    ("reduce/contract/merge/plan.rs", "AND"),
    ("reduce/contract/merge/plan.rs", "CAN"),
    ("reduce/contract/merge/plan.rs", "CHILD"),
    ("reduce/contract/merge/plan.rs", "DECIDED"),
    ("reduce/contract/merge/plan.rs", "FALSE"),
    ("reduce/contract/merge/plan.rs", "FIRST"),
    ("reduce/contract/merge/plan.rs", "ONCE"),
    ("reduce/contract/merge/plan.rs", "OVERLAP"),
    ("reduce/contract/merge/plan.rs", "OWN"),
    ("reduce/contract/merge/plan.rs", "PARENT"),
    ("reduce/contract/merge/plan.rs", "PLAIN"),
    ("reduce/contract/merge/plan.rs", "POD"),
    ("reduce/contract/merge/plan.rs", "SUMMED"),
    ("reduce/contract/merge/plan.rs", "SURVIVOR"),
    ("reduce/contract/merge/plan.rs", "WHOLE"),
    ("reduce/contract/merge/rewrite.rs", "ONCE"),
    ("reduce/contract/pair_fusion/mod.rs", "AND"),
    ("reduce/contract/pair_fusion/mod.rs", "APPLIED"),
    ("reduce/contract/pair_fusion/mod.rs", "BOUNDARY"),
    ("reduce/contract/pair_fusion/mod.rs", "BOXED"),
    ("reduce/contract/pair_fusion/mod.rs", "DROPS"),
    ("reduce/contract/pair_fusion/mod.rs", "EXACT"),
    ("reduce/contract/pair_fusion/mod.rs", "GLOBAL"),
    ("reduce/contract/pair_fusion/mod.rs", "INLINE"),
    ("reduce/contract/pair_fusion/mod.rs", "LABEL"),
    ("reduce/contract/pair_fusion/mod.rs", "LEAF"),
    ("reduce/contract/pair_fusion/mod.rs", "LESS"),
    ("reduce/contract/pair_fusion/mod.rs", "LOOKUP"),
    ("reduce/contract/pair_fusion/mod.rs", "NEIGHBOURING"),
    ("reduce/contract/pair_fusion/mod.rs", "PINNED"),
    ("reduce/contract/pair_fusion/mod.rs", "RLIMIT"),
    ("reduce/contract/pair_fusion/mod.rs", "SIGNED"),
    ("reduce/contract/pair_fusion/mod.rs", "SLOT"),
    ("reduce/contract/pair_fusion/mod.rs", "WEIGHTED"),
    ("reduce/contract/pair_fusion/mod.rs", "WIDTH"),
    ("reduce/contract/pair_fusion/plan.rs", "ACTION"),
    ("reduce/contract/pair_fusion/plan.rs", "AND"),
    ("reduce/contract/pair_fusion/plan.rs", "ARE"),
    ("reduce/contract/pair_fusion/plan.rs", "CANONICAL"),
    ("reduce/contract/pair_fusion/plan.rs", "DEFINITION"),
    ("reduce/contract/pair_fusion/plan.rs", "DENSE"),
    ("reduce/contract/pair_fusion/plan.rs", "DROP"),
    ("reduce/contract/pair_fusion/plan.rs", "EXCEPTION"),
    ("reduce/contract/pair_fusion/plan.rs", "EXPLICIT"),
    ("reduce/contract/pair_fusion/plan.rs", "FULL"),
    ("reduce/contract/pair_fusion/plan.rs", "INLINE"),
    ("reduce/contract/pair_fusion/plan.rs", "INVARIANT"),
    ("reduce/contract/pair_fusion/plan.rs", "LABEL"),
    ("reduce/contract/pair_fusion/plan.rs", "LEAF"),
    ("reduce/contract/pair_fusion/plan.rs", "MISS"),
    ("reduce/contract/pair_fusion/plan.rs", "MULTISET"),
    ("reduce/contract/pair_fusion/plan.rs", "OCCURRENCES"),
    ("reduce/contract/pair_fusion/plan.rs", "PIN"),
    ("reduce/contract/pair_fusion/plan.rs", "PINNED"),
    ("reduce/contract/pair_fusion/plan.rs", "SUPERSET"),
    ("reduce/contract/pair_fusion/plan.rs", "SURVIVING"),
    ("reduce/contract/pair_fusion/plan.rs", "THE"),
    ("reduce/contract/pair_fusion/plan.rs", "WEIGHTED"),
    ("reduce/contract/pair_fusion/rewrite.rs", "ALIASING"),
    ("reduce/contract/pair_fusion/rewrite.rs", "CONTIGUOUS"),
    ("reduce/contract/pair_fusion/rewrite.rs", "DOWN"),
    ("reduce/contract/pair_fusion/rewrite.rs", "INLINE"),
    ("reduce/contract/pair_fusion/rewrite.rs", "ONCE"),
    ("reduce/contract/pair_fusion/rewrite.rs", "OWN"),
    ("reduce/contract/pair_fusion/rewrite.rs", "PLACE"),
    ("reduce/contract/pair_fusion/rewrite.rs", "SHRINKS"),
    ("reduce/contract/pair_fusion/slots.rs", "EQUAL"),
    ("reduce/contract/pair_fusion/slots.rs", "EXACT"),
    ("reduce/contract/pair_fusion/slots.rs", "FALSE"),
    ("reduce/contract/pair_fusion/slots.rs", "INTEGER"),
    ("reduce/contract/pair_fusion/slots.rs", "LEAF"),
    ("reduce/contract/pair_fusion/slots.rs", "REAL"),
    ("reduce/contract/pair_fusion/slots.rs", "SIGNED"),
    ("reduce/contract/pair_fusion/slots.rs", "SLOT"),
    ("reduce/contract/pair_fusion/slots.rs", "SOUNDNESS"),
    ("reduce/contract/pair_fusion/slots.rs", "VALUES"),
    ("reduce/contract/pair_fusion/slots.rs", "WITHIN"),
    ("reduce/contract/pair_fusion/slots.rs", "ZERO"),
    ("reduce/contract/pair_fusion_weighted_tests/leaf.rs", "ASYMMETRIC"),
    ("reduce/contract/pair_fusion_weighted_tests/leaf.rs", "DEFINITION"),
    ("reduce/contract/pair_fusion_weighted_tests/leaf.rs", "DROPPED"),
    ("reduce/contract/pair_fusion_weighted_tests/leaf.rs", "DUPLICATE"),
    ("reduce/contract/pair_fusion_weighted_tests/leaf.rs", "LABEL"),
    ("reduce/contract/scratch.rs", "ARE"),
    ("reduce/contract/scratch.rs", "CANDIDATE"),
    ("reduce/contract/scratch.rs", "DENSE"),
    ("reduce/contract/scratch.rs", "EMPTY"),
    ("reduce/contract/scratch.rs", "FIRST"),
    ("reduce/contract/scratch.rs", "INDEPENDENTLY"),
    ("reduce/contract/scratch.rs", "KEEPS"),
    ("reduce/contract/scratch.rs", "MASS"),
    ("reduce/contract/scratch.rs", "MAX"),
    ("reduce/contract/scratch.rs", "MULTISET"),
    ("reduce/contract/scratch.rs", "NODE"),
    ("reduce/contract/scratch.rs", "OUT"),
    ("reduce/contract/scratch.rs", "PERSISTENCE"),
    ("reduce/contract/scratch.rs", "POD"),
    ("reduce/contract/scratch.rs", "REUSED"),
    ("reduce/contract/scratch.rs", "SLOT"),
    ("reduce/contract/scratch.rs", "SURVIVOR"),
    ("reduce/contract/scratch.rs", "XOR"),
    ("reduce/contract/strategies.rs", "BETWEEN"),
    ("reduce/contract/strategies.rs", "DELETED"),
    ("reduce/contract/strategies.rs", "DOWNWARD"),
    ("reduce/contract/strategies.rs", "NODE"),
    ("reduce/contract/strategies.rs", "RAISE"),
    ("reduce/contract/strategies.rs", "REBUILD"),
    ("reduce/contract/strategies.rs", "SECOND"),
    ("reduce/contract/strategies.rs", "SET"),
    ("reduce/contract/strategies.rs", "WEIGHTED"),
    ("reduce/contract/strategies.rs", "WITHOUT"),
    ("reduce/mod.rs", "AND"),
    ("reduce/mod.rs", "CANONICAL"),
    ("reduce/mod.rs", "EXCEPTION"),
    ("reduce/mod.rs", "PANICS"),
    ("reduce/prune.rs", "LEAF"),
    ("reduce/prune.rs", "OOB"),
    ("reduce/prune.rs", "STORE"),
    ("reduce/prune.rs", "VAS"),
    ("reduce/prune.rs", "WIDTH"),
    ("reduce/prune.rs", "ZERO"),
    ("reduce/slot_prune.rs", "ASCENDING"),
    ("reduce/slot_prune.rs", "ASSIGN"),
    ("reduce/slot_prune.rs", "COMMON"),
    ("reduce/slot_prune.rs", "EQUAL"),
    ("reduce/slot_prune.rs", "INCREMENT"),
    ("reduce/slot_prune.rs", "INTEGER"),
    ("reduce/slot_prune.rs", "LEAF"),
    ("reduce/slot_prune.rs", "MOVE"),
    ("reduce/slot_prune.rs", "NOTE"),
    ("reduce/slot_prune.rs", "OOB"),
    ("reduce/slot_prune.rs", "PLACE"),
    ("reduce/slot_prune.rs", "POST"),
    ("reduce/slot_prune.rs", "PRE"),
    ("reduce/slot_prune.rs", "REKEYED"),
    ("reduce/slot_prune.rs", "RETIREMENT"),
    ("reduce/slot_prune.rs", "SEMANTIC"),
    ("reduce/slot_prune.rs", "SOUNDNESS"),
    ("reduce/slot_prune.rs", "STORE"),
    ("reduce/slot_prune.rs", "SWAP"),
    ("reduce/slot_prune.rs", "TAGGER"),
    ("reduce/slot_prune.rs", "TALLY"),
    ("reduce/slot_prune.rs", "VALUES"),
    ("reduce/slot_prune.rs", "WEIGHTED"),
    ("reduce/slot_prune_stores.rs", "CURRENT"),
    ("reduce/slot_prune_stores.rs", "EMPTY"),
    ("reduce/slot_prune_stores.rs", "FAST"),
    ("reduce/slot_prune_stores.rs", "FIRST"),
    ("reduce/slot_prune_stores.rs", "IDENTITY"),
    ("reduce/slot_prune_stores.rs", "INVARIANT"),
    ("reduce/slot_prune_stores.rs", "LABEL"),
    ("reduce/slot_prune_stores.rs", "LEAF"),
    ("reduce/slot_prune_stores.rs", "PATH"),
    ("reduce/slot_prune_stores.rs", "PIN"),
    ("reduce/slot_prune_stores.rs", "POSITION"),
    ("reduce/slot_prune_stores.rs", "REFS"),
    ("reduce/slot_prune_stores.rs", "REMAP"),
    ("reduce/slot_prune_stores.rs", "SHARED"),
    ("reduce/slot_prune_stores.rs", "SLOT"),
    ("reduce/slot_prune_stores.rs", "STORE"),
    ("reduce/slot_prune_stores.rs", "ZERO"),
    ("reduce/slots.rs", "COUNT"),
    ("reduce/slots.rs", "INDEPENDENTLY"),
    ("reduce/slots.rs", "NEW"),
    ("reduce/slots.rs", "OOB"),
    ("reduce/slots.rs", "OVERFLOW"),
    ("reduce/slots.rs", "ZERO"),
    ("restructure/graft.rs", "OOB"),
    ("restructure/relevel.rs", "CONSTRUCTION"),
    ("restructure/relevel.rs", "CONTEXT"),
    ("restructure/relevel.rs", "DISTINCT"),
    ("restructure/relevel.rs", "EVERYWHERE"),
    ("restructure/relevel.rs", "EXPAND"),
    ("restructure/relevel.rs", "EXPANSION"),
    ("restructure/relevel.rs", "FULL"),
    ("restructure/relevel.rs", "FULLY"),
    ("restructure/relevel.rs", "IDENTICAL"),
    ("restructure/relevel.rs", "KEEP"),
    ("restructure/relevel.rs", "KEEPS"),
    ("restructure/relevel.rs", "MARGINAL"),
    ("restructure/relevel.rs", "SUM"),
    ("restructure/search/cluster.rs", "BETWEEN"),
    ("restructure/search/cluster.rs", "LEFT"),
    ("restructure/search/cluster.rs", "RIGHT"),
    ("restructure/search/core.rs", "PAIRS"),
    ("restructure/search/core.rs", "PARENT"),
    ("restructure/search/local.rs", "CANONICAL"),
    ("restructure/search/local.rs", "NON"),
    ("value_fold/domain.rs", "DEEP"),
    ("value_fold/domain.rs", "INLINE"),
    ("value_fold/domain.rs", "INVERTED"),
    ("value_fold/domain.rs", "LEAF"),
    ("value_fold/domain.rs", "MARGINAL"),
    ("value_fold/domain.rs", "MAX"),
    ("value_fold/domain.rs", "PRODUCT"),
    ("value_fold/domain.rs", "READ"),
    ("value_fold/domain.rs", "VALUES"),
    ("value_fold/domain.rs", "VIEW"),
    ("value_fold/fold.rs", "AND"),
    ("value_fold/fold.rs", "COLUMN"),
    ("value_fold/fold.rs", "CONTRACT"),
    ("value_fold/fold.rs", "EMPTY"),
    ("value_fold/fold.rs", "IGNORED"),
    ("value_fold/fold.rs", "INHERENT"),
    ("value_fold/fold.rs", "NOTHING"),
    ("value_fold/fold.rs", "READERS"),
    ("value_fold/fold.rs", "ROOT"),
    ("value_fold/fold.rs", "ZERO"),
    ("value_fold/fold.rs", "ZST"),
    ("value_fold/mod.rs", "MAX"),
    ("value_fold/mod.rs", "OLD"),
    ("value_fold/mod.rs", "SCRATCH"),
    ("value_fold/mod.rs", "SPARSE"),
    ("vtree/build.rs", "BFS"),
    ("vtree/build.rs", "DVE"),
    ("vtree/build.rs", "OBDD"),
    ("vtree/build.rs", "RNG"),
    ("vtree/project.rs", "KEEP"),
];
// generated:ALL_CAPS:end

/// Outstanding citations of a file that is not in `src/`, as `(file, cited)`.
// generated:CITED_PATH:begin
const CITED_PATH_ALLOW: &[(&str, &str)] = &[
    ("reduce/contract/merge/data.rs", "types.rs"),
    ("reduce/contract/scratch.rs", "types.rs"),
    ("reduce/prune.rs", "types.rs"),
];
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

/// Modules that are public for a downstream driver or for tests, and so have
/// no row in a table describing the compilation boundary.
const UNTABLED_MODULES: &[&str] = &["check", "compiler_seam", "readme"];

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

#[test]
fn prose_carries_emphasis_by_structure_not_by_capitals() {
    let allowed: HashSet<(&str, &str)> = ALL_CAPS_ALLOW.iter().copied().collect();
    let acronyms: HashSet<&str> = ACRONYMS.iter().copied().collect();
    let mut new_hits: Vec<String> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for (rel, path) in non_test_sources() {
        let text = fs::read_to_string(&path).expect("a readable source file");
        for (n, line) in text.lines().enumerate() {
            let Some(prose) = comment_prose(line) else { continue };
            for word in all_caps_words(&prose) {
                if acronyms.contains(word.as_str())
                    || allowed.contains(&(rel.as_str(), word.as_str()))
                {
                    continue;
                }
                if seen.insert((rel.clone(), word.clone())) {
                    new_hits.push(format!("{rel}:{}: {word}", n + 1));
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
        for (n, line) in text.lines().enumerate() {
            let Some(prose) = comment_prose(line) else { continue };
            for cited in cited_paths(&prose) {
                if known.contains(&cited) || allowed.contains(&(rel.as_str(), cited.as_str())) {
                    continue;
                }
                new_hits.push(format!("{rel}:{}: cites {cited}, which is not under src/", n + 1));
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
