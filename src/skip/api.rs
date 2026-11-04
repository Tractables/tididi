//! The public skip-form diagram, [`SkipTdd`], and the operations on it.

use std::borrow::Cow;
use std::sync::Arc;

use num_bigint::BigUint;

use super::conjoin;
use super::contract::{contract, contract_in};
use super::convert::{from_tdd, stored_nodes, summed_roots, to_tdd};
use super::embed::{embed, embed_summed};
use super::negate::{is_true, negate};
use super::overlay::{overlay, Combine};
use super::project::project;
use super::quantify::{exists, free};
use super::pass::SkipCount;
use super::{filled, sum, Shape, SkipForm, SUMMED, SUMMED_VAR};
use crate::diagram::{EvalAlgebra, Literal, Tdd, TddBuildError, WeightStore, WeightValue};
use crate::restructure::embed::Plan;
use crate::restructure::{EmbedError, EmbeddingPlan};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, LiteralInput, OperationError};

/// A Boolean diagram whose pairs may skip levels.
///
/// A [`Tdd`] stores a node at every level a function's paths cross. Where a
/// level's other child is a single variable that the function ignores, that
/// node is `true` on the variable and the rest of the function below, and a
/// `SkipTdd` does not store it: the pair side above names the node below
/// directly. On a right-linear vtree this is the reduced form of an ordered
/// BDD, where a [`Tdd`] is the quasi-reduced one, and it can hold many times
/// fewer pairs.
///
/// A `SkipTdd` without summed-out subtrees is always canonical: two
/// diagrams over the same vtree denote the same function exactly when they
/// are equal, and no operation on it needs a reduction. It may hold its
/// function as the negation of the diagram it stores: [`Engine::skip_negate`]
/// only marks it, the conjunction, disjunction and difference read the mark
/// by De Morgan, and the other operations build the negation first.
///
/// With weights attached ([`SkipTdd::set_weights`]), subtrees can be summed
/// out ([`Engine::skip_marginalize`], [`Engine::skip_and_marginalizing`]),
/// as for a [`Tdd`]: a summed-out subtree keeps a column of values in the
/// weight store, and the diagram is then valued by
/// [`Engine::skip_weighted_value`]. Such a diagram conjoins with another
/// that is `true` over its summed-out subtrees; the other operations refuse
/// it with [`OperationError::MarginalLevel`].
///
/// It is a type of its own because stored readers walk [`Tdd`] levels
/// directly, and a level a pair skips is not stored. The operations that
/// take a `SkipTdd` are named `skip_*` on [`Engine`], with
/// [`Engine::to_skip`] and [`Engine::to_stored`] converting between the
/// forms; every other operation takes the stored form.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Literal, Tdd, Vtree};
/// use tididi::vtree::VarId;
///
/// let vtree = Arc::new(Vtree::linear(3));
/// let eng = Engine::new();
/// let f = eng.to_skip(&Tdd::clause(&vtree, [Literal::pos(VarId(1)), Literal::pos(VarId(3))])?)?;
/// let g = eng.skip_and_clause(&f, &[Literal::neg(VarId(1)), Literal::pos(VarId(2))])?;
/// assert_eq!(eng.skip_model_count(&g)?, 4u32.into());
/// let stored = eng.to_stored(&g)?;
/// assert_eq!(eng.model_count(&stored)?, 4u32.into());
/// assert!(g.pair_count() < stored.pair_count());
/// # Ok::<(), tididi::OperationError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SkipTdd {
    vtree: Arc<Vtree>,
    pub(super) form: SkipForm,
    /// Whether the diagram is `¬form`: a negation not built. Never with a
    /// summed-out subtree.
    neg: bool,
    /// The weights the summed-out subtrees are valued under, with their
    /// columns.
    weights: Option<WeightStore>,
}

impl SkipTdd {
    /// `form` on `self`'s vtree, with `self`'s weights.
    fn with(&self, form: SkipForm) -> SkipTdd {
        SkipTdd { vtree: Arc::clone(&self.vtree), form, neg: false, weights: self.weights.clone() }
    }

    /// `¬self`, marked: the same stored diagram.
    fn negated(&self) -> SkipTdd {
        SkipTdd { neg: !self.neg, ..self.clone() }
    }

    /// `self` with its negation built where it holds one unbuilt, for the
    /// operations that read the stored diagram as the function.
    pub(super) fn built(&self, lim: &crate::limits::Limits) -> Result<Cow<'_, SkipTdd>, OperationError> {
        if !self.neg {
            return Ok(Cow::Borrowed(self));
        }
        Ok(Cow::Owned(self.with(negate(&self.form, lim)?)))
    }

    /// `Ok` where nothing is summed out, for an operation on functions.
    fn boolean(&self) -> Result<(), OperationError> {
        let sh = &*self.form.shape;
        match (0..sh.num_nodes()).find(|&v| sh.kind[v] >= SUMMED_VAR) {
            Some(v) => Err(OperationError::MarginalLevel(VtreeIdx(v as u32))),
            None => Ok(()),
        }
    }

    /// Whether a subtree is summed out.
    pub fn has_marginal_level(&self) -> bool {
        self.form.shape.is_weighted()
    }

    /// The attached weights, with the columns of the summed-out subtrees.
    pub fn weights(&self) -> Option<&WeightStore> {
        self.weights.as_ref()
    }

    /// Attach `store`'s weights, as [`Tdd::set_weights`] attaches them.
    /// Where a subtree is summed out, the store must interpret the weights
    /// as the attached one does, whose columns it then takes over.
    ///
    /// # Errors
    ///
    /// [`TddBuildError::MissingVariableWeight`] when `store` does not weigh
    /// a variable of the vtree, and [`TddBuildError::IncompatibleWeights`]
    /// for a summed-out subtree and a store that weighs differently.
    pub fn set_weights(&mut self, mut store: WeightStore) -> Result<(), TddBuildError> {
        store.check_variables(self.vtree.leaf_bottomup().map(|(_, var)| var))?;
        if let Some(old) = self.weights.as_ref().filter(|_| self.has_marginal_level()) {
            if !old.compatible(&store) {
                return Err(TddBuildError::IncompatibleWeights);
            }
            let sh = &*self.form.shape;
            for v in 0..sh.num_nodes() {
                if sh.kind[v] == SUMMED {
                    store.set_level(v, old.level(v).expect("a summed-out subtree has its column").to_vec());
                }
            }
        }
        self.weights = Some(store);
        Ok(())
    }

    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`).
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// Whether the diagram is the constant false.
    pub fn is_zero(&self) -> bool {
        if self.neg { is_true(&self.form) } else { self.form.is_zero() }
    }

    /// The pairs stored over all levels; a negation not built counts as
    /// the diagram it negates.
    pub fn pair_count(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].pairs.len()).sum()
    }

    /// The nodes stored over all levels.
    pub fn node_count(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].width()).sum()
    }

    /// The most nodes stored at one level.
    pub fn max_width(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].width()).max().unwrap_or(0)
    }
}

/// The Boolean combinations of two diagrams that [`Engine::skip_and`],
/// [`Engine::skip_or`] and [`Engine::skip_and_not`] build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Boolean {
    And,
    Or,
    AndNot,
}

impl Engine {
    /// The skip form of `f`, with its weights.
    ///
    /// `f` need not be reduced. Converting a canonical `f` (a certified
    /// result) costs one pass over its pairs. Each level `f` holds as
    /// weights under a level that is not marginal becomes a summed-out
    /// subtree; nothing in its column is read as `true`, so a conjunction
    /// needs the other operand `true` there.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] when a level of `f` holds model
    /// counts rather than weights, [`OperationError::OverBudget`] when the
    /// skip form does not fit, and [`OperationError::Stopped`] for an armed
    /// stop.
    pub fn to_skip(&self, f: &Tdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let vtree = f.vtree();
        let roots = summed_roots(f);
        let Some(fw) = f.weights.as_ref().filter(|_| !roots.is_empty()) else {
            let canonical = f.levels.is_canonical(f.output);
            let form = from_tdd(f, Arc::new(Shape::of(vtree)), lim)?;
            let form = if canonical { form } else { contract(form, lim)? };
            return Ok(SkipTdd { vtree: Arc::clone(vtree), form, neg: false, weights: f.weights.clone() });
        };
        // Each internal subtree's column behind `One`; nothing else of the
        // stored columns is read in skip form.
        let mut ws = fw.empty_like();
        for &(m, _) in &roots {
            if !vtree.node(VtreeIdx(m)).is_leaf() {
                let mut col = Vec::new();
                let stored = fw.level(m as usize).unwrap_or(&[]);
                lim.reserve_exact(&mut col, stored.len() + 1)?;
                col.push(sum::top_weight(vtree, fw, VtreeIdx(m)));
                col.extend_from_slice(stored);
                ws.set_level(m as usize, col);
            }
        }
        let form = from_tdd(f, Arc::new(Shape::summed(vtree, &roots)), lim)?;
        let form = match sum::sum_out(&form, vtree, &mut ws, &[], true, None, lim)? {
            Some(shared) => {
                form.discard(lim);
                shared
            }
            None => form,
        };
        Ok(SkipTdd { vtree: Arc::clone(vtree), form, neg: false, weights: Some(ws) })
    }

    /// The stored form of `f`, with its weights: canonical, with a node at
    /// every level a pair of `f` skips.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] where `f` sums a subtree out, and
    /// [`OperationError::OverBudget`] or [`OperationError::IndexOverflow`]
    /// when the stored levels do not fit.
    pub fn to_stored(&self, f: &SkipTdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        f.boolean()?;
        let f = &*f.built(self.limits())?;
        let mut tdd = to_tdd(&f.form, self, &f.vtree)?;
        if let Some(ws) = &f.weights {
            tdd.set_weights(ws.clone())?;
        }
        Ok(tdd)
    }

    /// The nodes [`Engine::to_stored`] would give `f`, counted without
    /// building them: `f`'s own, and a node at every level a pair of `f`
    /// skips, one per (level, target). It costs a pass over `f`'s pairs,
    /// where the stored form can hold the square of them.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] where `f` sums a subtree out, and
    /// [`OperationError::OverBudget`] where the count's tables do not fit.
    pub fn skip_stored_node_count(&self, f: &SkipTdd) -> Result<u64, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        stored_nodes(&f.built(lim)?.form, lim)
    }

    /// `f ∧ g`, with the weights of either; where either sums a subtree
    /// out, the other must be `true` over it, and a node both of whose
    /// children are then summed out is summed out too.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the operands do not share a
    /// vtree allocation, [`OperationError::IncompatibleWeights`] when they
    /// weigh differently, [`OperationError::MarginalLevel`] at a subtree
    /// one sums out where the other constrains its variables or both read
    /// its values, [`OperationError::OverBudget`] when the result does not
    /// fit, and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_and(&self, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !Arc::ptr_eq(&f.vtree, &g.vtree) {
            return Err(OperationError::VtreeMismatch);
        }
        let weights = match (&f.weights, &g.weights) {
            (Some(a), Some(b)) if !a.compatible(b) => return Err(OperationError::IncompatibleWeights),
            (Some(a), _) => Some(a),
            (None, b) => b.as_ref(),
        };
        if f.has_marginal_level() || g.has_marginal_level() {
            // An operand that sums nothing out may hold a negation unbuilt.
            let (f, g) = (&*f.built(lim)?, &*g.built(lim)?);
            let mut ws = weights.expect("a summed-out subtree has weights").clone();
            let empty = ws.empty_like();
            let gw = g.weights.as_ref().unwrap_or(&empty);
            let form = sum::and(&f.form, &g.form, &f.vtree, &mut ws, gw, &[], lim)?;
            return Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form, neg: false, weights: Some(ws) });
        }
        self.combine(Boolean::And, f, g, weights)
    }

    /// `f ∧ g`, `f ∨ g` or `f ∧ ¬g` for operands that sum nothing out, on
    /// one vtree, with `weights`: by De Morgan one conjunction, difference
    /// or disjunction of the diagrams they store, its result's negation
    /// left unbuilt where the operands' negations ask for it.
    fn combine(&self, op: Boolean, f: &SkipTdd, g: &SkipTdd, weights: Option<&WeightStore>) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        // `f op g` is `a ∧ b`, or its negation where `neg`, with `a` and `b`
        // the stored diagrams of `f` and `g` or their negations (`na`, `nb`).
        let (na, nb, neg) = match op {
            Boolean::And => (f.neg, g.neg, false),
            Boolean::AndNot => (f.neg, !g.neg, false),
            Boolean::Or => (!f.neg, !g.neg, true),
        };
        let (a, b) = (&f.form, &g.form);
        let (form, neg) = match (na, nb) {
            (false, false) => (self.conjoin(&f.vtree, a, b)?, neg),
            (false, true) | (true, false) => {
                // `p ∧ ¬n`. The overlay walks every level of both; the
                // conjunction only the cone of the smaller, so a smaller `n`
                // is cheaper negated and conjoined, and a larger one cheaper
                // left to the overlay than negated.
                let (p, n) = if nb { (a, b) } else { (b, a) };
                if size(n) < size(p) {
                    let not = negate(n, lim)?;
                    let res = self.conjoin(&f.vtree, p, &not);
                    not.discard(lim);
                    (res?, neg)
                } else {
                    (overlay(Combine::AndNot, p, n, lim)?, neg)
                }
            }
            (true, true) if (f.neg || g.neg) && apart(a, b) => {
                // `¬a ∧ ¬b` of a marked operand: as a disjunction it keeps
                // the mark, which a chain of such conjunctions then carries
                // through a walk of the whole accumulated diagram at each
                // step. Where one is on under half the other's levels the
                // two negations are built and conjoined instead.
                let x = negate(a, lim)?;
                let res = negate(b, lim).and_then(|y| {
                    let res = self.conjoin(&f.vtree, &x, &y);
                    y.discard(lim);
                    res
                });
                x.discard(lim);
                (res?, neg)
            }
            (true, true) => (overlay(Combine::Or, a, b, lim)?, !neg),
        };
        Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form, neg, weights: weights.cloned() })
    }

    /// `f ∧ g` for the stored diagrams of two operands on `vtree` that sum
    /// nothing out.
    fn conjoin(&self, vtree: &Arc<Vtree>, f: &SkipForm, g: &SkipForm) -> Result<SkipForm, OperationError> {
        let lim = self.limits();
        // The walk is the larger operand, so that the levels it indexes are
        // the smaller one's.
        let (a, b) = if size(f) < size(g) { (g, f) } else { (f, g) };
        let (raw, cone) = match conjoin::hybrid_on() {
            true => conjoin::and_hybrid(a, b, self, vtree)?,
            false => conjoin::and_scoped(a, b, lim)?,
        };
        let res = contract_traced("and", a, b, raw, &cone, lim);
        lim.discard(cone);
        res
    }

    /// `f ∧ g` with the subtrees of `targets` summed out under the
    /// operands' weights, as [`Engine::and_marginalizing`] sums them out
    /// of a [`Tdd`]: [`Engine::skip_and`], then [`Engine::skip_marginalize`],
    /// in one rebuild where an operand sums a subtree out.
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_and`] and [`Engine::skip_marginalize`].
    pub fn skip_and_marginalizing(&self, f: &SkipTdd, g: &SkipTdd, targets: &[VtreeIdx]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !(f.has_marginal_level() || g.has_marginal_level()) {
            let both = self.skip_and(f, g)?;
            return self.skip_marginalize(&both, targets);
        }
        if !Arc::ptr_eq(&f.vtree, &g.vtree) {
            return Err(OperationError::VtreeMismatch);
        }
        if let Some(&t) = targets.iter().find(|t| t.idx() >= f.vtree.num_nodes()) {
            return Err(OperationError::LevelNotInVtree(t));
        }
        let (f, g) = (&*f.built(lim)?, &*g.built(lim)?);
        let mut ws = match (&f.weights, &g.weights) {
            (Some(a), Some(b)) if !a.compatible(b) => return Err(OperationError::IncompatibleWeights),
            (Some(a), _) | (None, Some(a)) => a.clone(),
            (None, None) => unreachable!("a summed-out subtree has weights"),
        };
        let empty = ws.empty_like();
        let gw = g.weights.as_ref().unwrap_or(&empty);
        let targets: Vec<u32> = targets.iter().map(|t| t.0).collect();
        let form = sum::and(&f.form, &g.form, &f.vtree, &mut ws, gw, &targets, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form, neg: false, weights: Some(ws) })
    }

    /// `f` with the subtree of each of `targets` summed out under its
    /// weights, and every node both of whose children are then summed out:
    /// the function of the other variables that weighs each of their
    /// assignments by the weight of its extensions.
    ///
    /// # Errors
    ///
    /// [`OperationError::LevelNotInVtree`] for a target outside the vtree,
    /// [`OperationError::IncompatibleWeights`] when `f` has no weights,
    /// [`OperationError::OverBudget`] when the result does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_marginalize(&self, f: &SkipTdd, targets: &[VtreeIdx]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let f = &*f.built(lim)?;
        if let Some(&t) = targets.iter().find(|t| t.idx() >= f.vtree.num_nodes()) {
            return Err(OperationError::LevelNotInVtree(t));
        }
        let Some(ws) = &f.weights else {
            return Err(OperationError::IncompatibleWeights);
        };
        let mut ws = ws.clone();
        let targets: Vec<u32> = targets.iter().map(|t| t.0).collect();
        // `f` is canonical: only the new summed subtrees' ancestors change.
        let unchanged = filled(lim, f.vtree.num_nodes(), false)?;
        let summed = sum::sum_out(&f.form, &f.vtree, &mut ws, &targets, false, Some(&unchanged), lim);
        lim.discard(unchanged);
        Ok(match summed? {
            Some(form) => SkipTdd { vtree: Arc::clone(&f.vtree), form, neg: false, weights: Some(ws) },
            None => f.clone(),
        })
    }

    /// The weight of `f`, its summed-out subtrees' values included, as
    /// [`Engine::weighted_value`] gives it for a [`Tdd`]; `None` without
    /// weights.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the values' table does not fit,
    /// and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_weighted_value(&self, f: &SkipTdd) -> Result<Option<WeightValue>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let Some(ws) = &f.weights else {
            return Ok(None);
        };
        sum::weighted_value(&f.built(lim)?.form, ws, lim).map(Some)
    }

    /// `f ∧ (l₁ ∨ … ∨ lₖ)` for the literals of `clause`, signed integers or
    /// typed literals as [`Engine::and_clause`] takes them; an empty clause
    /// gives false.
    ///
    /// # Errors
    ///
    /// As [`Engine::and_clause`].
    pub fn skip_and_clause<L: LiteralInput>(&self, f: &SkipTdd, clause: &[L]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if f.has_marginal_level() {
            // The clause on the vtree's own shape, conjoined as a diagram.
            let shape = Arc::new(Shape::of(&f.vtree));
            let Some(c) = conjoin::clause_form(&f.vtree, &shape, clause, lim)? else {
                return Ok(f.clone());
            };
            let g = SkipTdd { vtree: Arc::clone(&f.vtree), form: c, neg: false, weights: None };
            return self.skip_and(f, &g);
        }
        let Some(c) = conjoin::clause_form(&f.vtree, &f.form.shape, clause, lim)? else {
            return Ok(f.clone());
        };
        if c.is_zero() || f.is_zero() {
            c.discard(lim);
            return Ok(f.with(SkipForm::zero(Arc::clone(&f.form.shape))));
        }
        if f.neg {
            // The clause less the stored diagram.
            let g = SkipTdd { vtree: Arc::clone(&f.vtree), form: c, neg: false, weights: None };
            let res = self.combine(Boolean::And, f, &g, f.weights.as_ref());
            g.form.discard(lim);
            return res;
        }
        let (raw, cone) = conjoin::and_scoped(&f.form, &c, lim)?;
        let res = contract_traced("clause", &f.form, &c, raw, &cone, lim);
        c.discard(lim);
        lim.discard(cone);
        Ok(f.with(res?))
    }

    /// `¬c₁ ∧ … ∧ ¬cₙ` over `vtree`: that none of the cubes holds, a table's
    /// rows ruled out at once. Built from the cubes level by level, a class
    /// of each level's assignments per set of cubes they leave open, where
    /// conjoining the negated cubes one at a time pays for the whole
    /// diagram at each. A cube with a variable in both signs never holds;
    /// an empty cube always does.
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable outside the
    /// vtree, [`OperationError::OverBudget`] when the result does not fit,
    /// and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_none_of<L: LiteralInput>(&self, vtree: &Arc<Vtree>, cubes: &[Vec<L>]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let shape = Arc::new(Shape::of(vtree));
        let form = super::none_of::none_of(vtree, &shape, cubes, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(vtree), form, neg: false, weights: None })
    }

    /// `¬f`, marked rather than built: it shares `f`'s stored diagram, and
    /// the operations that read it build the negation only where De Morgan
    /// does not reach it.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] where `f` sums a subtree out, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_negate(&self, f: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let _op = self.limits().enter()?;
        f.boolean()?;
        Ok(f.negated())
    }

    /// `f ∨ g`, with the weights of either. Built by overlaying the
    /// operands, as [`Engine::or`] is: no level is complemented.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the operands do not share a
    /// vtree allocation, [`OperationError::IncompatibleWeights`] when they
    /// weigh differently, [`OperationError::MarginalLevel`] where either sums
    /// a subtree out, [`OperationError::OverBudget`] when the result does not
    /// fit, and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_or(&self, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        self.skip_overlay(Boolean::Or, f, g)
    }

    /// `f ∧ ¬g`, the models of `f` that are not models of `g`, with the
    /// weights of either. As [`Engine::and_not`], it never builds `¬g`.
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_or`].
    pub fn skip_and_not(&self, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        self.skip_overlay(Boolean::AndNot, f, g)
    }

    /// [`Engine::skip_or`] or [`Engine::skip_and_not`].
    fn skip_overlay(&self, op: Boolean, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !Arc::ptr_eq(&f.vtree, &g.vtree) {
            return Err(OperationError::VtreeMismatch);
        }
        let weights = match (&f.weights, &g.weights) {
            (Some(a), Some(b)) if !a.compatible(b) => return Err(OperationError::IncompatibleWeights),
            (Some(a), _) => Some(a),
            (None, b) => b.as_ref(),
        };
        f.boolean()?;
        g.boolean()?;
        self.combine(op, f, g, weights)
    }

    /// The number of assignments to the vtree's variables that satisfy `f`.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the counts do not fit.
    pub fn skip_model_count(&self, f: &SkipTdd) -> Result<BigUint, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        f.built(lim)?.form.model_count(lim)
    }

    /// The value of `f` in `algebra`, as [`Engine::evaluate`] gives it for
    /// the stored form: a variable a pair skips is valued as `true` on it.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the values' table does not fit,
    /// and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_evaluate<S: EvalAlgebra>(&self, f: &SkipTdd, algebra: &S) -> Result<S::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        f.built(lim)?.form.evaluate(algebra, lim)
    }

    /// The value of `f` in `algebra`, as [`Engine::skip_evaluate`] gives
    /// it, with the values of `f`'s nodes kept: a [`SkipCount`], from which
    /// [`Engine::skip_count_literal`] and [`Engine::skip_count_and`] read
    /// the value of `f` with a literal or with another diagram without
    /// building the conjunction.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] where `f` sums a subtree out,
    /// [`OperationError::OverBudget`] where the values do not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_count<S: EvalAlgebra>(&self, f: &SkipTdd, algebra: &S) -> Result<SkipCount<S::Value>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        SkipCount::new(&*f.built(lim)?, algebra, lim)
    }

    /// The value of `f ∧ literal` in `algebra`, `f` the diagram `counted`
    /// counts: the sum, over the sides that name the literal's variable or
    /// skip past it, of their contexts times the literal's weight and the
    /// rest of the side's value. The first joint count takes the pass down,
    /// one pass over `f`'s pairs, which every later one reads; this one
    /// then reads the skips that cross the variable's parent level.
    /// `algebra` is the one `counted` was taken in.
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable outside the
    /// vtree, [`OperationError::OverBudget`] where the pass down does not
    /// fit, and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_count_literal<S: EvalAlgebra>(
        &self,
        counted: &mut SkipCount<S::Value>,
        algebra: &S,
        literal: Literal,
    ) -> Result<S::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        counted.literal(algebra, literal, lim)
    }

    /// The value of `f ∧ g` in `algebra`, `f` the diagram `counted` counts
    /// and `g` a diagram over its vtree, without building the conjunction:
    /// the states of `f` at the level of `g`'s output node, its nodes there
    /// and the skips that cross the level, each with its context from the
    /// pass down, against `g`'s node, and below that the products of
    /// their nodes the states reach, each valued once, down to where
    /// either diagram is `true`. `algebra` is the one `counted` was taken
    /// in.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] unless `g` shares `f`'s vtree,
    /// [`OperationError::MarginalLevel`] where `g` sums a subtree out,
    /// [`OperationError::OverBudget`] where the products' values do not
    /// fit, and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_count_and<S: EvalAlgebra>(
        &self,
        counted: &mut SkipCount<S::Value>,
        algebra: &S,
        g: &SkipTdd,
    ) -> Result<S::Value, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        g.boolean()?;
        counted.and(algebra, &*g.built(lim)?, lim)
    }

    /// `f` with the literals of `assignment` set, signed integers or typed
    /// literals as [`Engine::and_clause`] takes them: a function that no
    /// longer depends on their variables. As [`Engine::condition`], equal
    /// literals count once and opposite ones give false.
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_and_clause`].
    pub fn skip_condition<L: LiteralInput>(&self, f: &SkipTdd, assignment: &[L]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        let mut set: Vec<(u32, Literal)> = Vec::new();
        lim.reserve_exact(&mut set, assignment.len())?;
        for &raw in assignment {
            let lit = raw.literal()?;
            let leaf = f.vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
            set.push((leaf.0, lit));
        }
        set.sort_unstable_by_key(|&(leaf, lit)| (leaf, lit.sign));
        set.dedup();
        let opposite = set.windows(2).any(|w| w[0].0 == w[1].0);
        // Conditioning commutes with negation: a marked diagram is
        // conditioned as stored and keeps its mark.
        let mut g = f.with(f.form.clone());
        for &(leaf, lit) in &set {
            if opposite || g.is_zero() {
                break;
            }
            let fixed = self.skip_and_clause(&g, &[lit])?;
            g = f.with(free(fixed.form, leaf, lim)?);
        }
        lim.discard(set);
        if opposite {
            return Ok(f.with(SkipForm::zero(Arc::clone(&f.form.shape))));
        }
        Ok(SkipTdd { neg: f.neg, ..g })
    }

    /// `∃ vars. f`: `f` with each variable of `vars` set either way, in one
    /// bottom-up regroup of the levels above them for the whole set, as
    /// [`Engine::exists_vars`] quantifies a [`Tdd`].
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable outside the
    /// vtree, [`OperationError::MarginalLevel`] where a subtree is summed
    /// out, [`OperationError::OverBudget`] when the result does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_exists(&self, f: &SkipTdd, vars: &[VarId]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        let f = &*f.built(lim)?;
        let leaves = quantified_leaves(f, vars, lim)?;
        let form = exists(&f.form, &leaves, false, lim);
        lim.discard(leaves);
        Ok(f.with(form?))
    }

    /// `∀ vars. f`: `f` with each variable of `vars` set both ways, as
    /// `¬∃ vars. ¬f`.
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_exists`].
    pub fn skip_forall(&self, f: &SkipTdd, vars: &[VarId]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        let leaves = quantified_leaves(f, vars, lim)?;
        if leaves.is_empty() {
            return Ok(f.clone());
        }
        let not = f.negated();
        let not = not.built(lim)?;
        let form = exists(&not.form, &leaves, true, lim);
        if let Cow::Owned(built) = not {
            built.form.discard(lim);
        }
        lim.discard(leaves);
        Ok(f.with(form?).negated())
    }

    /// `∃ vars. f ∧ g`: [`Engine::skip_and`], then [`Engine::skip_exists`].
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_and`] and [`Engine::skip_exists`].
    pub fn skip_and_exists(&self, f: &SkipTdd, g: &SkipTdd, vars: &[VarId]) -> Result<SkipTdd, OperationError> {
        let _op = self.limits().enter()?;
        self.skip_exists(&self.skip_and(f, g)?, vars)
    }

    /// Whether `f` and `g` denote the same function: their canonical forms
    /// are equal up to the numbering of nodes, compared in one pass up.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] where the operands' vtrees differ,
    /// [`OperationError::MarginalLevel`] where a subtree is summed out,
    /// [`OperationError::OverBudget`] where the pass does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_equivalent(&self, f: &SkipTdd, g: &SkipTdd) -> Result<bool, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !Arc::ptr_eq(&f.vtree, &g.vtree) {
            return Err(OperationError::VtreeMismatch);
        }
        f.boolean()?;
        g.boolean()?;
        if f.neg == g.neg {
            return f.form.same(&g.form, lim);
        }
        f.built(lim)?.form.same(&g.built(lim)?.form, lim)
    }

    /// `f` on the larger vtree `into`, each variable renamed through `map`
    /// and every other variable of `into` free, placed as [`Tdd::embed`]
    /// places it. Where a free side is a single variable the pairs skip it,
    /// so a skip-form copy adds a node only where a level's free side holds
    /// several variables.
    ///
    /// # Errors
    ///
    /// As [`Tdd::embed`].
    pub fn skip_embed(&self, f: &SkipTdd, into: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<SkipTdd, EmbedError> {
        let plan = self.embedding_plan(&f.vtree, into, map)?;
        self.skip_embed_with(f, &plan)
    }

    /// [`Engine::skip_embed`] up to mirrors, as [`Tdd::embed_mirrored`]
    /// places.
    ///
    /// # Errors
    ///
    /// As [`Tdd::embed_mirrored`].
    pub fn skip_embed_mirrored(&self, f: &SkipTdd, into: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<SkipTdd, EmbedError> {
        let plan = self.embedding_plan_mirrored(&f.vtree, into, map)?;
        self.skip_embed_with(f, &plan)
    }

    /// [`Engine::skip_embed`] with a prepared placement.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when `plan` is not from `f`'s
    /// vtree allocation; otherwise as [`EmbeddingPlan::apply`].
    pub fn skip_embed_with(&self, f: &SkipTdd, plan: &EmbeddingPlan) -> Result<SkipTdd, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !Arc::ptr_eq(plan.source(), &f.vtree) {
            return Err(OperationError::VtreeMismatch.into());
        }
        f.boolean()?;
        let layout = plan.layout();
        let shape = Arc::new(Shape::of(plan.destination()));
        // Placing a function on a larger vtree commutes with negation.
        let form = embed(&f.form, shape, &layout.embedding.levels, &layout.mirrored, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(plan.destination()), form, neg: f.neg, weights: None })
    }

    /// `f` restricted to some of its variables, every other variable
    /// existentially quantified, on the restricted vtree, as
    /// [`Tdd::project_to_vars`] restricts a diagram; the inverse of
    /// [`Engine::skip_embed`]. A diagram that does not depend on the dropped
    /// variables is moved in one pass over its levels; one that does is
    /// quantified with [`Engine::skip_exists`] first.
    ///
    /// # Errors
    ///
    /// As [`Tdd::project_to_vars`].
    pub fn skip_project_to_vars(&self, f: &SkipTdd, local_of: impl Fn(VarId) -> Option<VarId>, num_local: u32) -> Result<SkipTdd, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        let crate::restructure::project::Scope { original, dropped, into } = self.projection_scope(&f.vtree, &local_of, num_local)?;
        let plan = Plan::build(self, &into, &f.vtree, |local| original[local.idx()].expect("a leaf of the restriction"), false)?;
        let result = self.skip_project_along(f, &into, &plan, &dropped);
        plan.recycle(self);
        lim.discard(original);
        lim.discard(dropped);
        result
    }

    /// [`Engine::skip_project_to_vars`] once `into` is matched in `f`'s
    /// vtree as `plan`, `dropped` the variables it leaves out.
    fn skip_project_along(&self, f: &SkipTdd, into: &Arc<Vtree>, plan: &Plan, dropped: &[VarId]) -> Result<SkipTdd, EmbedError> {
        let lim = self.limits();
        let shape = Arc::new(Shape::of(into));
        let done = |form, neg| SkipTdd { vtree: Arc::clone(into), form, neg, weights: None };
        // A diagram that reads no dropped variable moves as it is, and so
        // does its negation.
        if let Some(form) = project(&f.form, &shape, plan, lim)? {
            return Ok(done(form, f.neg));
        }
        // The diagram reads a dropped variable: quantify those it reads.
        let f = &*f.built(lim)?;
        let support = f.form.support(lim)?;
        let read: Vec<VarId> = dropped.iter().copied().filter(|v| support.binary_search(v).is_ok()).collect();
        let g = self.skip_exists(f, &read)?;
        match project(&g.form, &shape, plan, lim)? {
            Some(form) => Ok(done(form, false)),
            None => unreachable!("a canonical diagram over the kept variables alone restricts"),
        }
    }

    /// [`Engine::skip_embed`] for a diagram that may sum subtrees out, with
    /// `weights` as the result's store, as [`Engine::embed_over`] places a
    /// [`Tdd`]: the image of a summed-out subtree is summed out, and a
    /// variable `into` adds under it multiplies its values by the
    /// variable's weight of `true`.
    ///
    /// # Errors
    ///
    /// Those of [`Engine::skip_embed`]; [`EmbedError::DestinationWeights`]
    /// for a table that does not cover `into`'s variables;
    /// [`EmbedError::SourceWeights`] for a summed-out subtree without
    /// `weights` or a table that disagrees with `f`'s under the renaming;
    /// [`OperationError::MarginalLevel`] at a summed-out subtree whose image
    /// holds another variable of `f`.
    pub fn skip_embed_over(
        &self,
        f: &SkipTdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        weights: Option<WeightStore>,
    ) -> Result<SkipTdd, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let plan = self.embedding_plan(&f.vtree, into, &map)?;
        if let Some(store) = &weights {
            store.check_variables(into.leaf_bottomup().map(|(_, var)| var)).map_err(EmbedError::DestinationWeights)?;
        }
        if !f.has_marginal_level() {
            let mut out = self.skip_embed_with(f, &plan)?;
            out.weights = weights;
            return Ok(out);
        }
        let source = f.weights.as_ref().expect("a summed-out subtree has weights");
        let Some(store) = weights else {
            let level = (0..f.form.shape.num_nodes()).find(|&v| f.form.shape.kind[v] >= SUMMED_VAR).expect("a summed-out subtree");
            return Err(EmbedError::SourceWeights(TddBuildError::WeightedLevelWithoutStore { level: VtreeIdx(level as u32) }));
        };
        if !source.compatible_after_rename(&store, f.vtree.leaf_bottomup().map(|(_, var)| (var, map(var)))) {
            return Err(EmbedError::SourceWeights(TddBuildError::IncompatibleWeights));
        }
        let layout = plan.layout();
        let (form, store) = embed_summed(&f.form, source, plan.destination(), &layout.embedding.levels, &layout.mirrored, store, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(plan.destination()), form, neg: false, weights: Some(store) })
    }

    /// The variables `f` depends on, sorted, as [`Engine::support`] gives
    /// them for the stored diagram.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] where a subtree is summed out,
    /// [`OperationError::OverBudget`] where the pass does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_support(&self, f: &SkipTdd) -> Result<Vec<VarId>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        f.form.support(lim)
    }

    /// A model of `f`, a literal for every variable of the vtree, or `None`
    /// for false; as [`Engine::satisfying_assignment`] gives it, a variable
    /// left free is false.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the walk does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_satisfying_assignment(&self, f: &SkipTdd) -> Result<Option<Vec<Literal>>, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.boolean()?;
        f.built(lim)?.form.satisfying_assignment(lim)
    }
}

/// The distinct leaves of `vars` in `f`'s vtree, sorted.
fn quantified_leaves(f: &SkipTdd, vars: &[VarId], lim: &crate::limits::Limits) -> Result<Vec<u32>, OperationError> {
    let mut leaves = Vec::new();
    lim.reserve_exact(&mut leaves, vars.len())?;
    for &x in vars {
        leaves.push(f.vtree.leaf_of(x).ok_or(OperationError::VariableNotInVtree(x))?.0);
    }
    leaves.sort_unstable();
    leaves.dedup();
    Ok(leaves)
}

/// The contraction after a conjunction, scoped to its cone; under
/// `TIDIDI_SKIP_TRACE` (experiment) one line per call on standard error
/// with the operands' sizes, the walk's counts (the raw result's pairs kept
/// from the larger operand, its dead slots, shared levels and rewritten
/// slots) and the time of each phase.
fn contract_traced(tag: &str, f: &SkipForm, g: &SkipForm, raw: SkipForm, cone: &[bool], lim: &crate::limits::Limits) -> Result<SkipForm, OperationError> {
    if !super::trace::on() {
        return contract_in(raw, Some(cone), lim);
    }
    let pairs = |s: &SkipForm, inside: bool| -> usize {
        s.active.iter().filter(|&&t| !inside || cone[t as usize]).map(|&t| s.levels[t as usize].pairs.len()).sum()
    };
    let (fp, fc, gp, rp) = (pairs(f, false), pairs(f, true), pairs(g, false), pairs(&raw, false));
    let lv = cone.iter().filter(|&&c| c).count();
    let mut clock = super::trace::Clock::start();
    let res = contract_in(raw, Some(cone), lim);
    clock.lap(7);
    let ph = super::trace::take();
    let cn = super::trace::take_counts();
    if let Ok(r) = &res {
        let (same, _) = super::overlay::copied(r, if fp >= gp { f } else { g });
        eprintln!(
            "SKIPTRACE {tag} f {fp} fcone {fc} g {gp} raw {rp} out {} copied {same} conelv {lv} kept {} dead {} shared {} rewritten {} us {}",
            pairs(r, false),
            cn[0],
            cn[1],
            cn[2],
            cn[3],
            ph.iter().map(|n| (n / 1000).to_string()).collect::<Vec<_>>().join(" ")
        );
    }
    res
}

/// An operand's size for the conjunction's choice of its walk: the levels it
/// is on, then its pairs.
fn size(s: &SkipForm) -> (usize, usize) {
    (s.active.len(), s.active.iter().map(|&t| s.levels[t as usize].pairs.len()).sum())
}

/// Whether one of two nonzero operands is on under half the other's levels.
pub(super) fn apart(a: &SkipForm, b: &SkipForm) -> bool {
    let (x, y) = (a.active.len(), b.active.len());
    !(a.is_zero() || b.is_zero()) && 2 * x.min(y) < x.max(y)
}
