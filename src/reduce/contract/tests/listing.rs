//! The listing a contraction sweep keeps of the parents to search whole
//! and of the nodes each level changed.

use smallvec::SmallVec;

use super::super::scratch::{ContractScratch, Listing};
use crate::execution::pool::{Buffers, PooledScratch};
use crate::limits::Limits;

/// A sweep reads only what it wrote: the parents on its own worklist, and
/// the lists of the levels it changed. A later sweep reads every level
/// unset without clearing one, a wrapped stamp unsets them all, and
/// released buffers regrow unset.
#[test]
fn a_listing_sweep_reads_only_its_own_entries() {
    let lim = Limits::new();
    let mut listing = Listing::default();
    listing.start(&lim, 10, &[3, 7]).unwrap();
    assert!(listing.whole(3) && listing.whole(7) && !listing.whole(4) && !listing.whole(12));
    listing.list_mut(&lim, 5).unwrap().extend([1, 2]);
    listing.list_mut(&lim, 5).unwrap().push(4);
    listing.list_mut(&lim, 9).unwrap().push(0);
    let five = listing.take(5);
    assert_eq!(five, [1, 2, 4]);
    listing.put(5, five);
    assert_eq!(listing.take(5), [1, 2, 4]);
    assert!(listing.take(6).is_empty());
    listing.put(6, vec![8]);
    assert!(listing.take(6).is_empty(), "a list is given back only to a level that has one");

    listing.start(&lim, 10, &[4]).unwrap();
    assert!(!listing.whole(3) && listing.whole(4));
    assert!(listing.take(5).is_empty() && listing.take(9).is_empty());
    listing.list_mut(&lim, 9).unwrap().push(6);
    assert_eq!(listing.take(9), [6]);

    // A stamp that wraps unsets every level before it is used again.
    listing.sweep = u32::MAX;
    listing.start(&lim, 10, &[2]).unwrap();
    listing.at[1] = (1, 0);
    listing.whole[8] = 1;
    listing.sweep = u32::MAX;
    listing.start(&lim, 10, &[2]).unwrap();
    assert_eq!(listing.sweep, 1);
    assert!(listing.whole(2) && !listing.whole(8) && listing.take(1).is_empty());

    // Released buffers regrow unset, on a larger diagram too.
    listing.list_mut(&lim, 3).unwrap().push(5);
    listing.buffers(&mut |buf| buf.release());
    listing.start(&lim, 16, &[15]).unwrap();
    assert!(listing.whole(15) && !listing.whole(2) && listing.take(3).is_empty());
    listing.list_mut(&lim, 14).unwrap().push(1);
    assert_eq!(listing.take(14), [1]);
}

/// A parked set keeps the lists of its last sweep and the groups of its
/// last node alone: the walk that prices it on every return reads each
/// inner list it keeps, so those of one wide sweep would make every later
/// return as slow as that sweep.
#[test]
fn a_parked_contract_scratch_keeps_only_what_its_last_call_filed() {
    let lim = Limits::new();
    let mut sc = ContractScratch::default();
    sc.listing.start(&lim, 64, &[]).unwrap();
    for t in 0..64 {
        lim.try_push(sc.listing.list_mut(&lim, t).unwrap(), 1).unwrap();
    }
    for g in 0..8 {
        lim.try_push(&mut sc.pair_fusion.groups, SmallVec::new()).unwrap();
        // Five refs spill the group out of its inline four.
        sc.pair_fusion.groups[g].extend([0, 1, 2, 3, 4]);
    }
    sc.pair_fusion.touched.extend([7, 9]);
    let spilled = sc.pair_fusion.groups[0].capacity() * 4;
    let wide = sc.retained_bytes();

    // A narrow sweep files two levels, reusing the first two lists.
    sc.listing.start(&lim, 64, &[]).unwrap();
    for t in 0..2 {
        sc.listing.list_mut(&lim, t).unwrap().push(1);
    }
    let kept = sc.retain(&lim);
    assert_eq!(kept, sc.retained_bytes());
    assert_eq!(kept, wide - 62 * 16 - 6 * spilled, "the 62 lists and 6 groups past the last call's are shed");
    assert_eq!(sc.pair_fusion.groups.len(), 2);
    assert_eq!(sc.listing.take(1), [1]);
    assert!(sc.listing.take(2).is_empty());
    // The next sweep files past the kept lists afresh.
    sc.listing.start(&lim, 64, &[]).unwrap();
    for t in 0..5 {
        sc.listing.list_mut(&lim, t).unwrap().push(t as u32);
    }
    assert_eq!(sc.listing.take(4), [4]);
}
