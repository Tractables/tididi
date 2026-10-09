//! The listing a contraction sweep keeps of the parents to search whole
//! and of the nodes each level changed.

use super::super::scratch::Listing;
use crate::execution::pool::Buffers;
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
