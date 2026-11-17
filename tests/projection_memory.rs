//! A diagram on a projection of a large vtree allocates nothing sized by
//! that vtree. Its own binary, because it counts every allocation the
//! process makes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tididi::vtree::{VarId, Vtree};
use tididi::{Literal, Tdd};

struct Largest;

static COUNTING: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
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
        }
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Largest = Largest;

#[test]
fn clause_on_a_projection_of_a_million_leaves() {
    const N: u32 = 1_000_000;
    let home = Arc::new(Vtree::balanced(N));
    let vars = [17, 4_242, 99_999, 500_000, 777_777, 999_999].map(VarId);
    let k = vars.len();
    // The first projection builds the home's index, linear in the home.
    Vtree::projection(&home, [VarId(1)]).unwrap();

    COUNTING.store(true, Ordering::Relaxed);
    let own = Vtree::projection(&home, vars).unwrap();
    let f = Tdd::clause(&own, vars.map(Literal::pos)).unwrap();
    COUNTING.store(false, Ordering::Relaxed);
    // The count is over the home, an integer of about N bits.
    let count = f.model_count().unwrap();

    assert_eq!(f.vtree().num_nodes(), 2 * k - 1);
    assert_eq!(f.vtree().num_vars(), N);
    assert_eq!(count, num_bigint::BigUint::from((1u32 << k) - 1) << (N as usize - k));
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest < N as usize / 16, "an allocation of {largest} bytes");
}
