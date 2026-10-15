use super::*;

const MIB: u64 = 1024 * 1024;

/// A bumping arena whose slab is full at `cells` cells.
fn full_bump_arena(cells: usize) -> GridArena {
    let mut arena = GridArena {
        cells: Vec::with_capacity(cells),
        grids: vec![None; 2],
        bump: Some(Bump { end: 0, free: Vec::new() }),
    };
    arena.cells.resize(arena.cells.capacity(), NO_PRODUCT);
    arena.bump.as_mut().unwrap().end = arena.cells.len();
    arena
}

#[test]
fn frontier_growth_doubles_with_room_to_spare() {
    let eng = Engine::new();
    let mut arena = full_bump_arena(1 << 20);
    let cap0 = arena.cells.capacity();
    let base = arena.alloc(&eng, 1, 1000).expect("an unbounded engine grows the slab");
    assert_eq!(base.idx(), cap0);
    assert!(arena.cells.capacity() >= 2 * cap0, "growth with room to spare doubles");
    assert_eq!(arena.cells.len(), cap0 + 1000);
}

#[test]
fn frontier_growth_near_the_cap_is_exact() {
    // A full 16 MiB slab under a 12 MiB soft budget: doubling would charge
    // 16 MiB and be refused, though the frontier needs only 4,000 bytes
    // more. The exact growth charges those, which fit.
    let eng = Engine::new();
    let lim = eng.limits();
    let mut arena = full_bump_arena(4 << 20);
    let cap0 = arena.cells.capacity();
    lim.set_budget(Some(12 * MIB));
    let base = arena.alloc(&eng, 1, 1000).expect("the frontier fits the headroom");
    assert_eq!(base.idx(), cap0);
    assert!(arena.cells.capacity() < 2 * cap0, "growth near the cap stays below doubling");
    assert!(lim.meters().in_flight_bytes < MIB);
    assert!(arena.cells[cap0..].iter().all(|&cell| cell == NO_PRODUCT));
}
