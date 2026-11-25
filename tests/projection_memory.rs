//! A diagram on a projection of a large vtree allocates nothing sized by
//! that vtree, and comparing two projections allocates nothing. Its own
//! binary, because it counts every allocation the process makes; the tests
//! take turns.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tididi::vtree::{Home, VarId, Vtree};
use tididi::{Literal, Tdd};

struct Largest;

static COUNTING: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TURN: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            LARGEST.fetch_max(new_size, Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Largest = Largest;

#[test]
fn clause_on_a_projection_of_a_million_leaves() {
    let _turn = TURN.lock().unwrap();
    const N: u32 = 1_000_000;
    let home = Home::new(Vtree::balanced(N));
    let vars = [17, 4_242, 99_999, 500_000, 777_777, 999_999].map(VarId);
    let k = vars.len();

    COUNTING.store(true, Ordering::Relaxed);
    let f = Tdd::clause(&home, vars.map(Literal::pos)).unwrap();
    COUNTING.store(false, Ordering::Relaxed);
    // The count is over the home, an integer of about N bits.
    let count = f.model_count().unwrap();

    assert_eq!(f.vtree().num_nodes(), 2 * k - 1);
    assert_eq!(f.vtree().num_vars(), N);
    assert_eq!(count, num_bigint::BigUint::from((1u32 << k) - 1) << (N as usize - k));
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest < N as usize / 16, "an allocation of {largest} bytes");
}

/// The union of two projections on one leaf set, or of one inside the
/// other, is an operand, found by comparing rank lists without allocating.
/// Debug builds validate the operands, which allocates.
#[test]
#[cfg_attr(debug_assertions, ignore = "debug builds validate the operands")]
fn a_union_of_equal_or_nested_leaf_sets_allocates_nothing() {
    let _turn = TURN.lock().unwrap();
    let home = Home::new(Vtree::balanced(1 << 16));
    let vars: Vec<VarId> = (0..200u32).map(|i| VarId(1 + i * 300)).collect();
    let a = Vtree::projection(&home, vars.iter().copied()).unwrap();
    let same = Vtree::projection(&home, vars.iter().rev().copied()).unwrap();
    let inside = Vtree::projection(&home, vars.iter().step_by(7).copied()).unwrap();
    let few = Vtree::projection(&home, [vars[3], vars[150]]).unwrap();
    assert!(!Arc::ptr_eq(&a, &same));

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let unions = [
        Vtree::union(&a, &same).unwrap(),
        Vtree::union(&inside, &a).unwrap(),
        Vtree::union(&a, &few).unwrap(),
        Vtree::union(&a, &home).unwrap(),
    ];
    COUNTING.store(false, Ordering::Relaxed);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(Arc::ptr_eq(&unions[0], &a) && Arc::ptr_eq(&unions[1], &a) && Arc::ptr_eq(&unions[2], &a));
    assert!(Arc::ptr_eq(&unions[3], &home));
}
