//! Conjunction of two diagrams whose leaf sets are neither equal nor nested,
//! each read where it is through the readings the union's merge gives
//! ([`Vtree::leaf_sets_reading`]).

use std::borrow::Cow;
use std::sync::Arc;

use super::view::{Slot, View};
use super::{conjoin_kept, Operands, VtreeMask};
use crate::diagram::{LevelBuffer, Tdd};
use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use crate::limits::{Limits, OperationError};
use crate::restructure::embed::{Plan, Target};
use crate::vtree::{Reading, UnionReader, Vtree, VtreeIdx};
use crate::Engine;

/// One operand's tables, by node of the union: its target, every entry
/// [`Target::Free`] and every slot [`Slot::Free`] between conjunctions; the
/// image of each of its nodes; the internal nodes it is not free at,
/// children before parents.
#[derive(Default)]
struct Side {
    targets: Vec<Target>,
    slots: Vec<Slot>,
    /// The levels of its view, every one empty between conjunctions.
    built: LevelBuffer,
    embedding: Vec<VtreeIdx>,
    cone: Vec<VtreeIdx>,
    pass_throughs: usize,
}

impl Side {
    /// Size the tables for a union of at most `nodes` nodes and an operand
    /// of `own` nodes.
    fn prepare(&mut self, lim: &Limits, nodes: usize, own: usize) -> Result<(), OperationError> {
        if self.targets.len() < nodes {
            lim.try_resize(&mut self.targets, nodes, Target::Free)?;
        }
        if self.slots.len() < nodes {
            lim.try_resize(&mut self.slots, nodes, Slot::Free)?;
        }
        self.embedding.clear();
        lim.try_resize(&mut self.embedding, own, VtreeIdx(0))?;
        self.cone.clear();
        lim.reserve_exact(&mut self.cone, nodes / 2)?;
        self.pass_throughs = 0;
        Ok(())
    }

    #[inline]
    fn read(&mut self, u: VtreeIdx, reading: Reading, leaves: usize) {
        match reading {
            Reading::Own(s) => {
                self.targets[u.idx()] = Target::Image { source: s, mirrored: false };
                self.embedding[s.idx()] = u;
                if s.idx() >= leaves {
                    self.cone.push(u);
                }
            }
            Reading::Pass(c) => {
                let carried = match self.targets[c.idx()] {
                    Target::Image { source, .. } | Target::PassThrough(source) => source,
                    Target::Free => unreachable!("a pass-through carries a side with a leaf"),
                };
                self.targets[u.idx()] = Target::PassThrough(carried);
                self.cone.push(u);
                self.pass_throughs += 1;
            }
            Reading::Free => {}
        }
    }

    /// Every target written free again.
    fn clear(&mut self) {
        for &u in self.embedding.iter().chain(&self.cone) {
            if let Some(target) = self.targets.get_mut(u.idx()) {
                *target = Target::Free;
            }
        }
        self.embedding.clear();
        self.cone.clear();
    }

    fn plan(&mut self) -> Plan {
        let parts = (std::mem::take(&mut self.targets), std::mem::take(&mut self.embedding), std::mem::take(&mut self.cone));
        Plan::from_parts(parts.0, parts.1, parts.2, self.pass_throughs)
    }

    fn put_back(&mut self, plan: Plan, view: Option<View>) {
        if let Some(view) = view {
            (self.slots, self.built.levels) = view.release(&plan);
        }
        (self.targets, self.embedding, self.cone) = plan.into_parts();
    }
}

/// Both operands' tables, kept between conjunctions.
#[derive(Default)]
pub(crate) struct UnionTables {
    sides: [Side; 2],
}

impl Buffers for UnionTables {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        for side in &mut self.sides {
            visit(&mut side.targets);
            visit(&mut side.slots);
            visit(&mut side.embedding);
            visit(&mut side.cone);
            side.built.buffers(visit);
        }
    }
}

impl PooledScratch for UnionTables {
    fn prepare(&mut self) {}

    fn retain(&mut self, lim: &Limits) -> usize {
        for side in &mut self.sides {
            side.clear();
            side.built.retain(lim);
        }
        self.release_oversized(lim)
    }
}

/// Whether the conjunction reads `tdd` through its readings: no weight
/// store, summed level or region, not false, and structural.
pub(super) fn readable(tdd: &Tdd) -> bool {
    tdd.weights.is_none()
        && !tdd.is_zero()
        && !tdd.vtree().has_regions()
        && !tdd.has_marginal_level()
        && tdd.require_structure().is_ok()
}

/// What fills [`UnionTables`] from [`Vtree::leaf_sets_reading`].
pub(super) struct Reader<'t> {
    lim: &'t Limits,
    tables: &'t mut UnionTables,
    /// Each operand's node count and leaf count.
    sizes: [(usize, usize); 2],
}

impl<'t> Reader<'t> {
    pub(super) fn new(lim: &'t Limits, tables: &'t mut UnionTables, (f, g): (&Vtree, &Vtree)) -> Self {
        let size = |v: &Vtree| (v.num_nodes(), v.num_leaves() as usize);
        Reader { lim, tables, sizes: [size(f), size(g)] }
    }
}

impl UnionReader for Reader<'_> {
    fn begin(&mut self, k: usize) -> Result<(), OperationError> {
        for (side, &(own, _)) in self.tables.sides.iter_mut().zip(&self.sizes) {
            side.prepare(self.lim, 2 * k - 1, own)?;
        }
        Ok(())
    }

    #[inline]
    fn read(&mut self, u: VtreeIdx, [rf, rg]: [Reading; 2]) {
        let [sf, sg] = &mut self.tables.sides;
        sf.read(u, rf, self.sizes[0].1);
        sg.read(u, rg, self.sizes[1].1);
    }
}

/// `f ∧ g` on `into`, their union, with `tables` as the reader of
/// [`prepare`] left them: each operand read through a view of its plan.
pub(super) fn conjoin_union<'a>(
    eng: &Engine,
    f: Cow<'a, Tdd>,
    g: Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    tables: &mut UnionTables,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let [sf, sg] = &mut tables.sides;
    lim.gate().poll_each((sf.cone.len() + sg.cone.len()) as u64)?;
    let (fp, gp) = (sf.plan(), sg.plan());
    let tables = |side: &mut Side| (std::mem::take(&mut side.slots), std::mem::take(&mut side.built.levels));
    let fv = View::build_in(eng, &f, into, &fp, tables(sf));
    let gv = View::build_in(eng, &g, into, &gp, tables(sg));
    let (mut fv, mut gv) = match (fv, gv) {
        (Ok(fv), Ok(gv)) => (fv, gv),
        (fv, gv) => {
            let mut error = None;
            for (side, plan, view) in [(&mut *sf, fp, fv), (&mut *sg, gp, gv)] {
                match view {
                    Ok(view) => side.put_back(plan, Some(view)),
                    Err((e, (slots, built))) => {
                        (side.slots, side.built.levels) = (slots, built);
                        side.put_back(plan, None);
                        error = Some(e);
                    }
                }
            }
            return Err(error.expect("a view was refused"));
        }
    };
    let out = conjoin_kept(
        eng,
        f,
        g,
        Operands { f: VtreeMask::free_of(&fp), g: VtreeMask::free_of(&gp) },
        Operands { f: Some(&mut fv), g: Some(&mut gv) },
        Operands { f: Some(&fp), g: Some(&gp) },
    )
    .map_err(|refused| refused.error);
    sf.put_back(fp, Some(fv));
    sg.put_back(gp, Some(gv));
    let mut out = out?;
    crate::reduce::close_constant(eng, &mut out);
    Ok(out)
}
