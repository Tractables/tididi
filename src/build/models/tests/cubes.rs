use std::sync::Arc;

use crate::diagram::Tdd;
use crate::limits::OperationError;
use crate::test_helpers::{assert_canonical, assert_same_shape, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree};

/// One cube over `width` variables: its values and the variables it fixes,
/// as bit lists.
type Cube = (Vec<bool>, Vec<bool>);

/// `cubes` packed as `from_cubes` reads them.
fn packed_cubes(width: usize, cubes: &[Cube]) -> Vec<u64> {
    let w = width.div_ceil(64).max(1);
    let mut out = Vec::new();
    for (value, fixed) in cubes {
        let start = out.len();
        out.resize(start + 2 * w, 0u64);
        for i in 0..width {
            out[start + i / 64] |= u64::from(value[i]) << (i % 64);
            out[start + w + i / 64] |= u64::from(fixed[i]) << (i % 64);
        }
    }
    out
}

/// Every assignment the cubes cover, packed as `from_models` reads rows.
fn expanded_rows(width: usize, cubes: &[Cube]) -> Vec<u64> {
    let w = width.div_ceil(64).max(1);
    let mut out = Vec::new();
    for (value, fixed) in cubes {
        let free: Vec<usize> = (0..width).filter(|&i| !fixed[i]).collect();
        for assignment in 0..1u64 << free.len() {
            let start = out.len();
            out.resize(start + w, 0u64);
            for i in 0..width {
                if fixed[i] && value[i] {
                    out[start + i / 64] |= 1 << (i % 64);
                }
            }
            for (k, &i) in free.iter().enumerate() {
                out[start + i / 64] |= ((assignment >> k) & 1) << (i % 64);
            }
        }
    }
    out
}

/// `from_cubes` against `from_models` over the covered assignments.
fn check(vtree: &Arc<Vtree>, vars: &[VarId], cubes: &[Cube], what: &str) {
    let built = Tdd::from_cubes(vtree, vars, &packed_cubes(vars.len(), cubes)).unwrap();
    let oracle = Tdd::from_models(vtree, vars, &expanded_rows(vars.len(), cubes)).unwrap();
    assert_canonical(&built);
    assert_same_shape(&built, &oracle, what);
    assert_eq!(built.model_count().unwrap(), oracle.model_count().unwrap(), "{what}");
    // Both store each level's nodes by their least pairs, so the levels
    // are the same node for node.
    for (t, (x, y)) in built.levels().iter().zip(oracle.levels()).enumerate() {
        assert_eq!(x.nodes().len(), y.nodes().len(), "{what}: level {t}'s nodes");
        for i in 0..x.nodes().len() {
            assert_eq!(x.pairs_vec(i), y.pairs_vec(i), "{what}: level {t}, node {i}");
        }
    }
}

/// A cube over `width` variables that fixes each with probability `fix`
/// in 256, and at most `max_free` free.
fn random_cube(rng: &mut Lcg, width: usize, fix: u64, max_free: usize) -> Cube {
    let value: Vec<bool> = (0..width).map(|_| rng.coin()).collect();
    let mut fixed: Vec<bool> = (0..width).map(|_| rng.below(256) < fix).collect();
    let mut free = fixed.iter().filter(|&&f| !f).count();
    for f in fixed.iter_mut() {
        if free <= max_free {
            break;
        }
        if !*f {
            *f = true;
            free -= 1;
        }
    }
    (value, fixed)
}

/// `n` distinct variables of `1..=total`, in a random order.
fn random_vars(rng: &mut Lcg, total: u32, n: usize) -> Vec<VarId> {
    let mut all: Vec<VarId> = (1..=total).map(VarId).collect();
    for i in (1..all.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        all.swap(i, j);
    }
    all.truncate(n);
    all
}

#[test]
fn no_cubes_is_false_and_no_variables_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let none = Tdd::from_cubes(&vtree, &[VarId(1), VarId(2)], &[]).unwrap();
    assert!(none.is_zero());
    let all = Tdd::from_cubes(&vtree, &[], &[0, 0]).unwrap();
    assert_canonical(&all);
    assert_eq!(all.model_count().unwrap(), 8u32.into());
}

#[test]
fn a_ragged_cube_buffer_and_a_bad_variable_are_refused() {
    let vtree = Arc::new(Vtree::balanced(3));
    assert_eq!(
        Tdd::from_cubes(&vtree, &[VarId(1)], &[0, 0, 0]).unwrap_err(),
        OperationError::RaggedRows { words: 3, per_row: 2 },
    );
    assert_eq!(
        Tdd::from_cubes(&vtree, &[VarId(1), VarId(9)], &[0, 0]).unwrap_err(),
        OperationError::VariableNotInVtree(VarId(9)),
    );
    assert_eq!(
        Tdd::from_cubes(&vtree, &[VarId(2), VarId(2)], &[0, 0]).unwrap_err(),
        OperationError::DuplicateVariable(VarId(2)),
    );
}

#[test]
fn a_cube_free_everywhere_is_true_and_fixed_everywhere_is_a_row() {
    for (name, vtree) in vtree_shapes(6) {
        let vars: Vec<VarId> = (1..=5).map(VarId).collect();
        let all = Tdd::from_cubes(&vtree, &vars, &[0b10110, 0]).unwrap();
        assert_canonical(&all);
        assert_same_shape(&all, &Tdd::one(&vtree), name);
        let row = Tdd::from_cubes(&vtree, &vars, &[0b10110, 0b11111]).unwrap();
        assert_same_shape(&row, &Tdd::from_models(&vtree, &vars, &[0b10110]).unwrap(), name);
    }
}

#[test]
fn value_bits_under_a_free_variable_and_past_the_last_are_ignored() {
    let vtree = Arc::new(Vtree::balanced(4));
    let vars = [VarId(1), VarId(2), VarId(3)];
    let clean = Tdd::from_cubes(&vtree, &vars, &[0b001, 0b101, 0b000, 0b010]).unwrap();
    let noisy = Tdd::from_cubes(&vtree, &vars, &[!0b100, 0b101 | !0b111, 0b101, 0b010, 0b000, 0b010]).unwrap();
    assert_canonical(&noisy);
    assert_same_shape(&clean, &noisy, "ignored bits and a repeated cube");
}

#[test]
fn random_cubes_build_what_their_assignments_build() {
    // Arbitrary masks, overlapping and repeated cubes, the constrained
    // variables scattered among free ones on random vtrees.
    let mut rng = Lcg::new(0xc0be_5eed);
    for case in 0..600 {
        let width = 1 + rng.below(9) as usize;
        let total = width as u32 + rng.below(4) as u32;
        let vars = random_vars(&mut rng, total, width);
        let fix = [64, 128, 200, 240][rng.below(4) as usize];
        let n = 1 + rng.below(12) as usize;
        let mut cubes: Vec<Cube> = (0..n).map(|_| random_cube(&mut rng, width, fix, 6)).collect();
        if case % 5 == 0 {
            let again = cubes[0].clone();
            cubes.push(again);
        }
        let vtree = Arc::new(Vtree::random(total, 0x70 + case));
        check(&vtree, &vars, &cubes, &format!("case {case}: {cubes:?}"));
    }
}

#[test]
fn random_cubes_on_every_vtree_shape() {
    let mut rng = Lcg::new(0x005b_a9e5);
    for round in 0..40 {
        let width = 2 + rng.below(8) as usize;
        let n = 1 + rng.below(10) as usize;
        let cubes: Vec<Cube> = (0..n).map(|_| random_cube(&mut rng, width, 160, 5)).collect();
        for (name, vtree) in vtree_shapes(width as u32 + 1) {
            let vars = random_vars(&mut rng, width as u32 + 1, width);
            check(&vtree, &vars, &cubes, &format!("round {round} {name}"));
        }
    }
}

/// Aligned intervals of two blocks, `bits` and `rank_bits` wide, and their
/// products, as a rank relation's boxes make them: the cubes fix a leading
/// part of each block and leave at most `max_free` of its bits free.
fn block_cubes(rng: &mut Lcg, bits: usize, rank_bits: usize, n: usize, max_free: usize) -> Vec<Cube> {
    let piece = |rng: &mut Lcg, width: usize| {
        let free = rng.below(width.min(max_free) as u64 + 1) as usize;
        let value: Vec<bool> = (0..width).map(|i| i < width - free && rng.coin()).collect();
        let fixed: Vec<bool> = (0..width).map(|i| i < width - free).collect();
        (value, fixed)
    };
    (0..n)
        .map(|_| {
            let (mut value, mut fixed) = piece(rng, bits);
            let (v, f) = piece(rng, rank_bits);
            value.extend(v);
            fixed.extend(f);
            (value, fixed)
        })
        .collect()
}

#[test]
fn products_of_aligned_intervals_build_what_their_assignments_build() {
    let mut rng = Lcg::new(0xb0c5_2026);
    for case in 0..300 {
        let (bits, rank_bits) = (1 + rng.below(7) as usize, 1 + rng.below(5) as usize);
        let total = (bits + rank_bits) as u32 + rng.below(3) as u32;
        let vars = random_vars(&mut rng, total, bits + rank_bits);
        let n = 1 + rng.below(20) as usize;
        let cubes = block_cubes(&mut rng, bits, rank_bits, n, 7);
        let vtree = Arc::new(Vtree::random(total, 0xb00 + case));
        check(&vtree, &vars, &cubes, &format!("case {case}"));
        // Blocks laid out as the vtree's leaves, most significant first.
        let ordered = Arc::new(Vtree::balanced(total));
        let in_order: Vec<VarId> = (1..=(bits + rank_bits) as u32).map(VarId).collect();
        check(&ordered, &in_order, &cubes, &format!("case {case}, blocks in leaf order"));
    }
}

#[test]
fn cubes_wider_than_a_word_read_every_word() {
    // Seventy variables, a few free per cube, so the assignments stay few.
    let mut rng = Lcg::new(0x70_c0be);
    for case in 0..40 {
        let width = 65 + rng.below(10) as usize;
        let total = width as u32 + rng.below(3) as u32;
        let vars = random_vars(&mut rng, total, width);
        let n = 1 + rng.below(8) as usize;
        let mut cubes: Vec<Cube> = (0..n).map(|_| random_cube(&mut rng, width, 250, 4)).collect();
        // Cubes that differ only past the first word, and overlap there.
        let mut twin = cubes[0].clone();
        twin.1[width - 1] = false;
        cubes.push(twin);
        let vtree = Arc::new(Vtree::random(total, 0x700 + case));
        check(&vtree, &vars, &cubes, &format!("case {case}"));
    }
}

#[test]
fn thousands_of_cubes_sort_through_the_radix_passes() {
    // Enough cubes that a node's parts, lists and triples pass the radix
    // sort's size floor; narrow enough that a cube packs into one word, and
    // wide enough that it does not.
    let mut rng = Lcg::new(0x00ad_1c5e);
    for (case, (width, n)) in [(12, 5000), (18, 6000), (20, 4000), (40, 3000)].into_iter().enumerate() {
        let total = width as u32 + 2;
        let vars = random_vars(&mut rng, total, width);
        let cubes: Vec<Cube> = (0..n).map(|_| random_cube(&mut rng, width, 230, 3)).collect();
        for seed in 0..3 {
            let vtree = Arc::new(Vtree::random(total, 0x5eed + 16 * case as u64 + seed));
            check(&vtree, &vars, &cubes, &format!("width {width}, seed {seed}"));
        }
        let rows = block_cubes(&mut rng, width / 2, width - width / 2, n, 3);
        let in_order: Vec<VarId> = (1..=width as u32).map(VarId).collect();
        check(&Arc::new(Vtree::balanced(total)), &in_order, &rows, &format!("width {width}, blocks"));
    }
}

#[test]
fn every_refusal_point_answers_over_budget_and_returns_the_buffers() {
    use crate::limits::LimitConfig;
    use crate::Engine;
    let vtree = Arc::new(Vtree::balanced(8));
    let vars: Vec<VarId> = (1..=8).map(VarId).collect();
    let mut rng = Lcg::new(7);
    let cubes: Vec<Cube> = (0..30).map(|_| random_cube(&mut rng, 8, 180, 4)).collect();
    let packed = packed_cubes(8, &cubes);
    let refused = vtree.context().with_limits(
        LimitConfig::none().with_memory_budget_bytes(Some(64)),
        |eng| eng.from_cubes(&vtree, &vars, &packed),
    );
    assert_eq!(refused.unwrap_err(), OperationError::OverBudget);
    let mut refusals = 0;
    for cut in 0..200u32 {
        let eng = Engine::new();
        Tdd::builder(&eng, &vtree).unwrap().abandon(&eng);
        eng.limits().refuse_nth_reserve(cut);
        match eng.from_cubes(&vtree, &vars, &packed) {
            Ok(f) => assert_canonical(&f),
            Err(e) => {
                assert_eq!(e, OperationError::OverBudget, "cut {cut}");
                assert_eq!(eng.scratch.levels.occupancy(), 1, "cut {cut} lost a level buffer");
                refusals += 1;
            }
        }
        eng.limits().grant_every_reserve();
    }
    assert!(refusals > 0, "no reservation was refused across the sweep");
}
