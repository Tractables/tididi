use std::ops::Range;

use super::home::{Home, HomeLink};
use super::{VarId, Vtree, VtreeIdx};

/// The home variables a diagram on a projection does not have, as an
/// evaluation passes them to [`EvalAlgebra::free`] and [`ColumnAlgebra::free`].
///
/// It is a view of the projection: making one, [`len`](Self::len) and
/// [`intervals`](Self::intervals) cost nothing of the home's size, so an
/// algebra that values the free variables without naming them (a count, a
/// table whose weights sum to one) pays for its own diagram only.
///
/// The variables are those of the home leaves whose ranks ([`Home::rank`])
/// lie in [`intervals`](Self::intervals). There are at most one more
/// intervals than the projection has childless nodes, plus one for each
/// interval of a region's extent and each variable pinned by an
/// [`Evaluation`](crate::query::Evaluation).
///
/// [`EvalAlgebra::free`]: crate::diagram::EvalAlgebra::free
/// [`ColumnAlgebra::free`]: crate::diagram::ColumnAlgebra::free
#[derive(Clone, Debug)]
pub struct FreeVars<'a> {
    link: &'a HomeLink,
    window: (u32, u32),
    extents_free: bool,
    /// Home ranks taken out of the set, ascending, each in it.
    pins: &'a [u32],
    len: u32,
}

impl<'a> FreeVars<'a> {
    /// The number of variables.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether there is no variable.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The home the ranks are of.
    pub fn home(&self) -> &'a Home {
        &self.link.home
    }

    /// The variables as intervals of home ranks, ascending, none empty, no
    /// two touching.
    pub fn intervals(&self) -> impl Iterator<Item = Range<u32>> + 'a {
        let link = self.link;
        let (lo, hi) = self.window;
        let skip_anchors = self.extents_free && !link.regions.is_empty();
        let start = link.ranks().partition_point(|&r| r < lo);
        let mut own = (start..link.ranks().len())
            .map(move |i| (i, link.ranks()[i]))
            .take_while(move |&(_, r)| r < hi)
            .filter(move |&(i, _)| !skip_anchors || link.regions.binary_search_by_key(&(i as u32), |region| region.at).is_err())
            .map(|(_, r)| r);
        let mut next = lo;
        let mut gaps = std::iter::from_fn(move || loop {
            let end = own.next();
            let gap = next..end.unwrap_or(hi);
            next = end.map_or(hi, |r| r + 1);
            if !gap.is_empty() || end.is_none() {
                return (!gap.is_empty()).then_some(gap);
            }
        });
        let mut removed: Vec<(u32, u32)> = self.pins.iter().map(|&r| (r, r + 1)).collect();
        if !self.extents_free {
            removed.extend(link.regions.iter().flat_map(|region| region.extent.iter().copied()));
        }
        removed.sort_unstable();
        let (mut j, mut rest) = (0, None::<Range<u32>>);
        std::iter::from_fn(move || loop {
            let mut gap = match rest.take() {
                Some(gap) => gap,
                None => gaps.next()?,
            };
            while removed.get(j).is_some_and(|&(_, end)| end <= gap.start) {
                j += 1;
            }
            match removed.get(j) {
                Some(&(start, end)) if start <= gap.start => {
                    gap.start = end;
                    rest = Some(gap).filter(|gap| !gap.is_empty());
                }
                Some(&(start, _)) if start < gap.end => {
                    rest = Some(start..gap.end);
                    return Some(gap.start..start);
                }
                _ => return Some(gap),
            }
        })
    }

    /// The variables in rank order, one step per variable.
    pub fn vars(&self) -> impl Iterator<Item = VarId> + 'a {
        let var_at = &self.link.home.index().var_at;
        self.intervals().flat_map(move |r| var_at[r.start as usize..r.end as usize].iter().copied())
    }

    /// The set without `pins`, ranks of variables in it, ascending.
    pub(crate) fn without<'b>(&self, pins: &'b [u32]) -> FreeVars<'b>
    where
        'a: 'b,
    {
        debug_assert!(self.pins.is_empty());
        FreeVars { pins, len: self.len - pins.len() as u32, ..self.clone() }
    }

    /// The first variable and the set without it.
    pub(crate) fn split_first(&self) -> Option<(VarId, FreeVars<'a>)> {
        let rank = self.intervals().next()?.start;
        let var = self.link.home.index().var_at[rank as usize];
        Some((var, FreeVars { window: (rank + 1, self.window.1), len: self.len - 1, ..self.clone() }))
    }
}

impl Vtree {
    /// The home variables this vtree has neither as a leaf nor in an extent;
    /// `None` when there is none.
    pub(crate) fn free_set(&self) -> Option<FreeVars<'_>> {
        let link = self.home.as_deref()?;
        let len = self.free_leaves();
        let window = (0, link.home.vtree().num_leaves());
        (len > 0).then_some(FreeVars { link, window, extents_free: false, pins: &[], len })
    }

    /// The variables of the home leaves under own node `t`'s home node that
    /// this vtree does not have as a leaf, those in extents included; `None`
    /// when there is none or when every home variable outside the extents is
    /// a leaf.
    pub(crate) fn free_set_under(&self, t: VtreeIdx) -> Option<FreeVars<'_>> {
        let link = self.home.as_deref().filter(|_| self.free_leaves() > 0)?;
        let Range { start: lo, end: hi } = link.home.rank_range(self.home_node(t));
        let (a, b) = (link.ranks().partition_point(|&r| r < lo), link.ranks().partition_point(|&r| r < hi));
        let anchors = link.regions.partition_point(|region| (region.at as usize) < b)
            - link.regions.partition_point(|region| (region.at as usize) < a);
        let len = hi - lo - (b - a - anchors) as u32;
        (len > 0).then_some(FreeVars { link, window: (lo, hi), extents_free: true, pins: &[], len })
    }

    /// The ranks of those of `vars` in [`free_set`](Self::free_set),
    /// ascending.
    pub(crate) fn free_ranks(&self, vars: impl Iterator<Item = VarId>) -> Vec<u32> {
        let Some(link) = self.home.as_deref() else { return Vec::new() };
        let mut ranks: Vec<u32> = vars
            .filter_map(|var| link.home.rank(var))
            .filter(|&r| link.ranks().binary_search(&r).is_err() && link.region_over(r, r + 1).is_none())
            .collect();
        ranks.sort_unstable();
        ranks.dedup();
        ranks
    }
}
