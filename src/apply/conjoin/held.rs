//! An operand as the conjunction sweep holds it: owned, or borrowed from
//! the caller.
//!
//! The sweep takes some of an operand's levels into its result: a level an
//! identity fast path or the relabelling route carries, the levels of a
//! region under a free level, and the sparse leaf arm's level that is f's
//! own. An owned operand's level is moved there, and its levels are dropped
//! once read; a borrowed operand's is copied, and nothing of it is dropped
//! or changed. Both build the same diagram for the same work: a copy charges
//! the memory budget but not the work clock, as a move charges neither.

use std::borrow::Borrow;
use std::ops::Deref;

use crate::diagram::{Tdd, TddLevel};
use crate::limits::{Limits, OperationError};

/// An operand of the sweep, owned or borrowed. Reads go through `Deref`.
pub(crate) enum Held<'a> {
    Owned(&'a mut Tdd),
    Borrowed(&'a Tdd),
}

impl Deref for Held<'_> {
    type Target = Tdd;
    #[inline]
    fn deref(&self) -> &Tdd {
        match self {
            Held::Owned(tdd) => tdd,
            Held::Borrowed(tdd) => tdd,
        }
    }
}

impl Borrow<Tdd> for Held<'_> {
    #[inline]
    fn borrow(&self) -> &Tdd {
        self
    }
}

impl<'s> Held<'s> {
    /// The sweep's hold on an operand the caller gave owned or borrowed:
    /// an owned one through `tdd`, a borrowed one through its borrow.
    #[inline]
    pub(crate) fn of(tdd: &'s mut std::borrow::Cow<'_, Tdd>) -> Self {
        match tdd {
            std::borrow::Cow::Owned(tdd) => Held::Owned(tdd),
            std::borrow::Cow::Borrowed(tdd) => Held::Borrowed(tdd),
        }
    }
}

impl Held<'_> {
    /// Whether the operand is the caller's, which the sweep only reads.
    #[inline]
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self, Held::Borrowed(_))
    }

    /// The operand, where the sweep may change it.
    #[inline]
    pub(crate) fn owned(&mut self) -> Option<&mut Tdd> {
        match self {
            Held::Owned(tdd) => Some(tdd),
            Held::Borrowed(_) => None,
        }
    }

    /// Put the operand's level `t` into `out`: moved where the operand is
    /// owned, leaving it what `out` held, and copied where it is borrowed.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the copy is refused; `out` is
    /// then as it was.
    pub(crate) fn carry(&mut self, lim: &Limits, t: usize, out: &mut TddLevel) -> Result<(), OperationError> {
        match self {
            Held::Owned(tdd) => std::mem::swap(out, &mut tdd.levels[t]),
            Held::Borrowed(tdd) => *out = tdd.levels[t].try_clone_on(lim)?,
        }
        Ok(())
    }

    /// Undo [`carry`](Self::carry) for a refused sweep that gives its
    /// operands back: the owned operand takes its level back from `out`; a
    /// borrowed one was only copied, and keeps what it has.
    #[inline]
    pub(crate) fn uncarry(&mut self, t: usize, out: &mut TddLevel) {
        if let Held::Owned(tdd) = self {
            std::mem::swap(out, &mut tdd.levels[t]);
        }
    }
}
