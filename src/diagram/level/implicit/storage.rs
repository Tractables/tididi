//! Stored and described pair arenas, including allocation accounting.

use crate::limits::{Charged, Limits, OperationError};
use super::{ChildPair, ImplicitLevel, DESCRIBED, REDESCRIBED};

/// A level's pair arena: the pairs, or, on an implicit level, the
/// description of the pairs in their place.
///
/// The two are told apart wherever pairs are read or changed:
/// [`stored`](Self::stored) gives a stored arena's pairs and
/// [`implicit`](Self::implicit) an implicit one's description, and nothing
/// writes an implicit arena's pairs out. [`len`](Self::len) and
/// [`capacity`](Self::capacity) are those of the arena the level would have
/// stored, which the meters, the sweeps and the level pool read as they read
/// a stored one's.
///
/// A prune that keeps an implicit level's survivors as a description
/// ([`redescribe`](Self::redescribe)) renumbers them from the start of the
/// arena and leaves its length where it was: the slots past the described
/// pairs stand for those of the nodes it dropped, which a stored arena keeps
/// until a sweep reclaims them, so that the length, the capacity and the
/// sweeps are those of the stored arena. Nothing reads those slots.
///
/// An implicit arena's vector of pairs is empty, so that the bounds check
/// that guards a read of a stored node's pairs ([`slots`](Self::slots)) is
/// also what tells an implicit level apart there, and a stored level's reads
/// and writes cost what they cost on a plain vector. The description is
/// boxed, with the capacity of the node arena of a level whose nodes it
/// implies.
#[derive(Debug, Default)]
pub(crate) struct PairArena {
    /// The pairs of a stored arena; empty on an implicit one.
    stored: Vec<ChildPair>,
    /// The description of the pairs, on an implicit arena.
    described: Option<Box<Described>>,
}

/// What an implicit arena holds in place of its pairs, and its level in
/// place of its nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Described {
    level: ImplicitLevel,
    /// The arena's length: the described pairs, then the slots of pairs a
    /// prune dropped.
    len: usize,
    /// The capacity the arena would have.
    capacity: usize,
    /// The capacity the level's node arena would have, where the
    /// description implies its nodes.
    node_capacity: usize,
}

impl PairArena {
    /// The arena's length, stored or described.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.described {
            None => self.stored.len(),
            Some(d) => d.len,
        }
    }

    /// Whether the arena holds no pairs, stored or described.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The arena's capacity: on an implicit arena, the capacity it would
    /// have, which the level pool and the meters read as they would read the
    /// stored one's.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        match &self.described {
            None => self.stored.capacity(),
            Some(d) => d.capacity,
        }
    }

    /// The description of the pairs, on an implicit arena.
    #[inline]
    pub(crate) fn implicit(&self) -> Option<&ImplicitLevel> {
        self.described.as_deref().map(|d| &d.level)
    }

    /// The capacity the node arena of a level whose nodes this description
    /// implies would have; 0 on a stored arena.
    #[inline]
    pub(crate) fn node_capacity(&self) -> usize {
        self.described.as_ref().map_or(0, |d| d.node_capacity)
    }

    /// Set the capacity [`node_capacity`](Self::node_capacity) reads, on
    /// an implicit arena.
    #[inline]
    pub(crate) fn set_node_capacity(&mut self, capacity: usize) {
        if let Some(d) = self.described.as_mut() {
            d.node_capacity = capacity;
        }
    }

    /// The pairs of a stored arena; `None` on an implicit one.
    #[inline]
    pub(crate) fn stored(&self) -> Option<&[ChildPair]> {
        match self.described {
            None => Some(&self.stored),
            Some(_) => None,
        }
    }

    /// The length of the vector of stored pairs: a stored arena's length,
    /// and zero on an implicit one, whose vector is empty.
    #[inline]
    pub(crate) fn stored_vec_len(&self) -> usize {
        debug_assert!(self.described.is_none() || self.stored.is_empty(), "an implicit arena stores no pairs");
        self.stored.len()
    }

    /// The length of a stored arena, which a level being built extends.
    /// Not valid on an implicit arena (a debug build panics).
    #[inline]
    pub(crate) fn stored_len(&self) -> usize {
        debug_assert!(self.described.is_none(), "an implicit level's pairs are not stored");
        self.stored.len()
    }

    /// The stored pairs at `range`, which must be a node's: `None` on an
    /// implicit arena, whose vector of pairs is empty, so that the bounds
    /// check of a stored node's read is the only test it takes. A node of
    /// an implicit level of one pair a node holds it inline and is read off
    /// its word.
    #[inline(always)]
    pub(crate) fn slots(&self, range: std::ops::Range<usize>) -> Option<&[ChildPair]> {
        self.stored.get(range)
    }

    /// The pairs of a stored arena, to change or extend.
    ///
    /// Not valid on an implicit arena, whose pairs are not stored (a debug
    /// build panics): code that changes a level's pairs in place takes an
    /// implicit level through its description, and code that builds a level
    /// starts from a cleared one.
    #[inline]
    #[track_caller]
    pub(crate) fn stored_mut(&mut self) -> &mut Vec<ChildPair> {
        debug_assert!(self.described.is_none(), "an implicit level's pairs are not stored");
        &mut self.stored
    }

    /// Hold `described`'s pairs as their description, at the capacity the
    /// stored arena would have, and `node_capacity` as the capacity of the
    /// level's node arena where the description implies its nodes. The
    /// arena is empty; its allocation is dropped.
    pub(crate) fn describe(&mut self, described: ImplicitLevel, capacity: usize, node_capacity: usize) {
        debug_assert!(self.is_empty() && described.per_node >= 1);
        let len = described.arena_len();
        DESCRIBED.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        self.stored = Vec::new();
        self.described = Some(Box::new(Described { level: described, len, capacity, node_capacity }));
    }

    /// Hold a stored arena's pairs as their description `described`, keeping
    /// the arena's length and capacity: the slots past the described pairs
    /// stand for the dead slots a sweep would drop. `node_capacity` is as
    /// for [`describe`](Self::describe).
    pub(crate) fn describe_stored(&mut self, described: ImplicitLevel, node_capacity: usize) {
        debug_assert!(self.stored().is_some() && described.per_node >= 1 && described.arena_len() <= self.len());
        let (len, capacity) = (self.len(), self.capacity());
        DESCRIBED.fetch_add(described.arena_len() as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        self.stored = Vec::new();
        self.described = Some(Box::new(Described { level: described, len, capacity, node_capacity }));
    }

    /// Hold `described` in place of an implicit arena's description: what a
    /// prune leaves of the level, its nodes renumbered from the start of the
    /// arena. The arena keeps its length and capacity, the slots past the
    /// described pairs standing for those the prune dropped.
    pub(crate) fn redescribe(&mut self, described: ImplicitLevel) {
        let d = self.described.as_mut().expect("redescribe on a stored arena");
        debug_assert!(described.per_node >= 1 && described.arena_len() <= d.len);
        REDESCRIBED.fetch_add(described.arena_len() as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        d.level = described;
    }

    /// Exchange the two sides of an implicit arena's description.
    pub(crate) fn swap_described_sides(&mut self) {
        let d = self.described.as_mut().expect("a stored arena swaps its pairs");
        d.level = d.level.swapped();
    }

    /// Shorten the arena to `len`, as [`Vec::truncate`] does. An implicit
    /// arena drops the slots past its described pairs.
    ///
    /// # Panics
    ///
    /// Panics on an implicit arena cut shorter than its described pairs.
    #[track_caller]
    pub(crate) fn truncate(&mut self, len: usize) {
        match &mut self.described {
            None => self.stored.truncate(len),
            Some(d) => {
                assert!(len >= d.level.arena_len(), "a truncation into an implicit level's pairs");
                d.len = d.len.min(len);
            }
        }
    }

    /// Empty the arena, keeping its capacity, as [`Vec::clear`] does: an
    /// implicit arena becomes an empty stored one of the capacity it would
    /// have had.
    #[inline]
    pub(crate) fn clear(&mut self) {
        match &self.described {
            None => self.stored.clear(),
            Some(_) => self.clear_described(),
        }
    }

    /// [`clear`](Self::clear) on an implicit arena, out of line so that a
    /// stored one's inlines where it is called.
    #[inline(never)]
    fn clear_described(&mut self) {
        if let Some(d) = self.described.take() {
            self.stored = Vec::with_capacity(d.capacity);
        }
    }

    /// Drop the capacity past the arena's length, as [`Vec::shrink_to_fit`]
    /// does.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        match &mut self.described {
            None => self.stored.shrink_to_fit(),
            Some(d) => d.capacity = d.len,
        }
    }

    /// A copy of the arena, reserved through `lim` as
    /// [`TddLevel::try_clone_on`] reserves the others: a stored arena is
    /// copied at its length, an implicit one keeps its description, its
    /// length as its capacity and its nodes as their capacity, and has that
    /// length charged.
    pub(crate) fn try_clone_on(&self, lim: &Limits) -> Result<PairArena, OperationError> {
        match &self.described {
            None => {
                let mut vec = Vec::new();
                lim.reserve_exact(&mut vec, self.stored.len())?;
                vec.extend_from_slice(&self.stored);
                Ok(PairArena::from(vec))
            }
            Some(d) => {
                let len = d.len;
                lim.charge_bytes((len as u64).saturating_mul(std::mem::size_of::<ChildPair>() as u64))?;
                let described = Described { level: d.level.clone(), len, capacity: len, node_capacity: d.level.nodes };
                Ok(PairArena { stored: Vec::new(), described: Some(Box::new(described)) })
            }
        }
    }
}

impl Clone for PairArena {
    /// A copy as [`Vec::clone`] makes one, at the arena's length: a stored
    /// arena's pairs, or the description with its length as its capacity
    /// and its nodes as theirs.
    #[inline]
    fn clone(&self) -> Self {
        match &self.described {
            None => PairArena { stored: self.stored.clone(), described: None },
            Some(_) => self.clone_described(),
        }
    }
}

impl PairArena {
    /// [`clone`](Clone::clone) on an implicit arena, out of line so that a
    /// stored one's inlines where it is called.
    #[inline(never)]
    fn clone_described(&self) -> PairArena {
        let described = self.described.as_ref().map(|d| Box::new(Described { level: d.level.clone(), len: d.len, capacity: d.len, node_capacity: d.level.nodes }));
        PairArena { stored: self.stored.clone(), described }
    }
}

impl From<Vec<ChildPair>> for PairArena {
    #[inline]
    fn from(vec: Vec<ChildPair>) -> Self {
        PairArena { stored: vec, described: None }
    }
}

impl PartialEq for PairArena {
    /// Whether the two arenas are the same: the same stored pairs, or the
    /// same description at the same length. A level is implicit exactly when
    /// its pairs as numbered can be described (see [`ImplicitLevel`]), so a
    /// stored arena and an implicit one never hold the same level's pairs.
    fn eq(&self, other: &PairArena) -> bool {
        match (&self.described, &other.described) {
            (None, None) => self.stored == other.stored,
            (Some(a), Some(b)) => a.len == b.len && a.level == b.level,
            _ => false,
        }
    }
}

impl Charged for PairArena {
    #[inline]
    fn charged_bytes(&self) -> u64 {
        (self.capacity() as u64).saturating_mul(std::mem::size_of::<ChildPair>() as u64)
    }
}
