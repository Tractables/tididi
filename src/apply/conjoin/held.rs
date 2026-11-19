//! An operand as the conjunction sweep holds it: owned, or borrowed from
//! the caller, and read on the sweep's vtree as it is or through a
//! [`View`].
//!
//! The sweep takes some of an operand's levels into its result: a level an
//! identity fast path or the relabelling route carries, the levels of a
//! region under a free level, and the sparse leaf arm's level that is f's
//! own. An owned operand's level is moved there, and its levels are dropped
//! once read; a borrowed operand's is copied, and nothing of it is dropped
//! or changed. Both build the same diagram for the same work: a copy charges
//! the memory budget but not the work clock, as a move charges neither. A
//! level a view built is the view's, and moves as an owned one does.

use std::borrow::Cow;
use std::sync::Arc;

use crate::diagram::{Dirty, Tdd, TddLevel, TddNodeId};
use crate::limits::{Limits, OperationError};
use crate::restructure::placement::push_free_level;
use crate::vtree::{Vtree, VtreeIdx};

use super::setup::LevelShape;
use super::view::View;

enum Hold<'a> {
    Owned(&'a mut Tdd),
    Borrowed(&'a Tdd),
}

/// An operand of the sweep, owned or borrowed, its levels indexed by the
/// nodes of the sweep's vtree.
pub(crate) struct Held<'a> {
    tdd: Hold<'a>,
    view: Option<&'a mut View>,
}

impl<'s> Held<'s> {
    /// The sweep's hold on an operand the caller gave owned or borrowed:
    /// an owned one through `tdd`, a borrowed one through its borrow; on the
    /// vtree of `view` where there is one, else on its own.
    #[inline]
    pub(crate) fn of(tdd: &'s mut Cow<'_, Tdd>, view: Option<&'s mut View>) -> Self {
        let tdd = match tdd {
            Cow::Owned(tdd) => Hold::Owned(tdd),
            Cow::Borrowed(tdd) => Hold::Borrowed(tdd),
        };
        Held { tdd, view }
    }

    /// A borrowed operand on its own vtree.
    #[inline]
    pub(crate) fn reading(tdd: &'s Tdd) -> Self {
        Held { tdd: Hold::Borrowed(tdd), view: None }
    }

    /// An owned operand on its own vtree.
    #[inline]
    pub(crate) fn owning(tdd: &'s mut Tdd) -> Self {
        Held { tdd: Hold::Owned(tdd), view: None }
    }
}

impl Held<'_> {
    /// The operand as the caller gave it, its levels on its own vtree.
    #[inline(always)]
    pub(crate) fn tdd(&self) -> &Tdd {
        match &self.tdd {
            Hold::Owned(tdd) => tdd,
            Hold::Borrowed(tdd) => tdd,
        }
    }

    /// Whether the operand is read through a view.
    #[inline]
    pub(crate) fn is_viewed(&self) -> bool {
        self.view.is_some()
    }

    /// The operand's levels where it is read on its own vtree, `None`
    /// through a view.
    #[inline(always)]
    pub(crate) fn own_levels(&self) -> Option<&[TddLevel]> {
        match self.view {
            None => Some(&self.tdd().levels),
            Some(_) => None,
        }
    }

    /// The operand's level at node `t` of the sweep's vtree.
    #[inline(always)]
    pub(crate) fn level(&self, t: impl LevelAt) -> &TddLevel {
        let t = t.at();
        match &self.view {
            None => &self.tdd().levels[t],
            Some(view) => view.level(self.tdd(), t),
        }
    }

    /// The operand's levels at the sweep's level `shape.t` and its two
    /// children.
    #[inline(always)]
    pub(super) fn here(&self, shape: LevelShape) -> Here<'_> {
        let (t, left, right) = (shape.t.idx(), shape.left.idx(), shape.right.idx());
        let tdd = self.tdd();
        match &self.view {
            None => {
                let levels = &tdd.levels;
                Here { t: &levels[t], left: &levels[left], right: &levels[right] }
            }
            Some(view) => Here { t: view.level(tdd, t), left: view.level(tdd, left), right: view.level(tdd, right) },
        }
    }

    /// The sweep's vtree.
    #[inline]
    pub(crate) fn vtree(&self) -> &Arc<Vtree> {
        match &self.view {
            None => &self.tdd().vtree,
            Some(view) => view.vtree(),
        }
    }

    /// The operand's output on the sweep's vtree.
    #[inline]
    pub(crate) fn output(&self) -> TddNodeId {
        match &self.view {
            None => self.tdd().output,
            Some(view) => view.output(),
        }
    }

    /// The operand's worklists on the sweep's vtree.
    #[inline]
    pub(crate) fn dirty(&self) -> &Dirty {
        match &self.view {
            None => &self.tdd().dirty,
            Some(view) => view.dirty(),
        }
    }

    /// Whether the operand is the caller's, which the sweep only reads.
    #[inline]
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self.tdd, Hold::Borrowed(_))
    }

    /// The operand, where the sweep may change it, its levels on its own
    /// vtree.
    #[inline]
    pub(crate) fn owned(&mut self) -> Option<&mut Tdd> {
        match &mut self.tdd {
            Hold::Owned(tdd) => Some(tdd),
            Hold::Borrowed(_) => None,
        }
    }

    /// Call `each` on the levels at `ts` where the sweep may change them:
    /// an owned operand's, or ones its view built.
    #[inline(always)]
    pub(crate) fn each_level_mut(&mut self, ts: [usize; 2], mut each: impl FnMut(&mut TddLevel)) {
        match (self.view.as_deref_mut(), &mut self.tdd) {
            (None, Hold::Owned(tdd)) => {
                // One mutable access to the levels: each forgets the
                // diagram's summary.
                let levels = &mut *tdd.levels;
                for t in ts {
                    each(&mut levels[t]);
                }
            }
            (None, Hold::Borrowed(_)) => {}
            (Some(view), tdd) => {
                for t in ts {
                    match (view.held_mut(t), &mut *tdd) {
                        (Ok(Some(built)), _) => each(built),
                        (Err(own), Hold::Owned(tdd)) => each(&mut tdd.levels[own]),
                        _ => {}
                    }
                }
            }
        }
    }

    /// Swap the stored nodes and pair arena of the level at `t` with
    /// `level`'s ([`crate::diagram::LevelStorage::swap_arenas`]) where the
    /// sweep may change it, and say whether it did.
    #[inline]
    pub(crate) fn swap_arenas(&mut self, t: usize, level: &mut TddLevel) -> bool {
        let own = match self.view.as_deref_mut().map(|view| view.held_mut(t)) {
            None => t,
            Some(Ok(Some(built))) => {
                std::mem::swap(&mut built.nodes, &mut level.nodes);
                std::mem::swap(&mut built.pairs, &mut level.pairs);
                return true;
            }
            Some(Ok(None)) => return false,
            Some(Err(own)) => own,
        };
        match &mut self.tdd {
            Hold::Owned(tdd) => {
                tdd.levels.swap_arenas(own, level);
                true
            }
            Hold::Borrowed(_) => false,
        }
    }

    /// Have the view build its pass-through at `t` ([`View::pass_through`]).
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the level cannot grow.
    pub(super) fn pass_through(&mut self, lim: &Limits, t: VtreeIdx) -> Result<(), OperationError> {
        let Held { tdd, view } = self;
        let tdd: &Tdd = match tdd {
            Hold::Owned(tdd) => tdd,
            Hold::Borrowed(tdd) => tdd,
        };
        view.as_deref_mut().expect("a pass-through is built by a view").pass_through(lim, tdd, t)
    }

    /// Make the operand's level at the free node `t` the level it stands
    /// for, where an owned operand on its own vtree holds it empty; a view
    /// builds it when it is carried.
    #[inline]
    pub(crate) fn build_free(&mut self, t: VtreeIdx) {
        if self.view.is_none()
            && let Hold::Owned(tdd) = &mut self.tdd
        {
            let vtree = Arc::clone(&tdd.vtree);
            push_free_level(&mut tdd.levels[t.idx()], &vtree, t);
        }
    }

    /// Put the operand's level `t` into `out`: moved where the operand is
    /// owned or the view built it, leaving it what `out` held, copied where
    /// the operand is borrowed, and built at a free node of a view, where
    /// `out` is empty.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the copy is refused; `out` is
    /// then as it was.
    #[inline(always)]
    pub(crate) fn carry(&mut self, lim: &Limits, t: usize, out: &mut TddLevel) -> Result<(), OperationError> {
        match (self.view.is_none(), &mut self.tdd) {
            (true, Hold::Owned(tdd)) => {
                std::mem::swap(out, &mut tdd.levels[t]);
                Ok(())
            }
            _ => self.carry_other(lim, t, out),
        }
    }

    #[inline(never)]
    fn carry_other(&mut self, lim: &Limits, t: usize, out: &mut TddLevel) -> Result<(), OperationError> {
        let own = match self.view.as_deref_mut() {
            None => t,
            Some(view) => match view.held_mut(t) {
                Ok(Some(built)) => {
                    std::mem::swap(out, built);
                    return Ok(());
                }
                Ok(None) => {
                    view.build_free(t, out);
                    return Ok(());
                }
                Err(own) => own,
            },
        };
        match &mut self.tdd {
            Hold::Owned(tdd) => std::mem::swap(out, &mut tdd.levels[own]),
            Hold::Borrowed(tdd) => tdd.levels[own].try_clone_over(lim, 0, 0, out)?,
        }
        Ok(())
    }

    /// Undo [`carry`](Self::carry) for a refused sweep that gives its
    /// operands back: the owned operand, or the view, takes its level back
    /// from `out`; a borrowed one was only copied, and keeps what it has.
    #[inline]
    pub(crate) fn uncarry(&mut self, t: usize, out: &mut TddLevel) {
        let own = match self.view.as_deref_mut().map(|view| view.held_mut(t)) {
            None => t,
            Some(Ok(Some(built))) => {
                std::mem::swap(out, built);
                return;
            }
            Some(Ok(None)) => {
                *out = TddLevel::new();
                return;
            }
            Some(Err(own)) => own,
        };
        if let Hold::Owned(tdd) = &mut self.tdd {
            std::mem::swap(out, &mut tdd.levels[own]);
        }
    }
}

/// One operand's levels at a level of the sweep and its two children
/// ([`Held::here`]).
#[derive(Clone, Copy)]
pub(super) struct Here<'a> {
    pub(super) t: &'a TddLevel,
    pub(super) left: &'a TddLevel,
    pub(super) right: &'a TddLevel,
}

/// An operand's levels by node of the sweep's vtree. A loop over many levels
/// is instantiated on [`Held::own_levels`] where there is no view, so that it
/// does not test for one at each level.
pub(crate) trait Levels {
    fn level_at(&self, t: usize) -> &TddLevel;
}

impl Levels for [TddLevel] {
    #[inline(always)]
    fn level_at(&self, t: usize) -> &TddLevel {
        &self[t]
    }
}

impl Levels for Held<'_> {
    #[inline(always)]
    fn level_at(&self, t: usize) -> &TddLevel {
        self.level(t)
    }
}

/// A node of the sweep's vtree, by index or by id.
pub(crate) trait LevelAt {
    fn at(self) -> usize;
}

impl LevelAt for usize {
    #[inline(always)]
    fn at(self) -> usize {
        self
    }
}

impl LevelAt for VtreeIdx {
    #[inline(always)]
    fn at(self) -> usize {
        self.idx()
    }
}
