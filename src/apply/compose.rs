//! Boolean combinations built from the shared apply and quantification kernels.

use crate::limits::{Charged, Transient};
use crate::reduce::ReductionPlan;
use crate::vtree::{VarId, VtreeIdx};
use crate::{Engine, OperationError, Tdd};
use std::borrow::Cow;

/// A composition owns one result until its last consumer can take the storage.
/// Earlier consumers get fallible copies; no scratch survives the operation.
#[derive(Default)]
pub(super) struct SharedCircuit {
    value: Option<Tdd>,
    remaining: usize,
}

impl SharedCircuit {
    pub(super) fn new(value: Tdd, uses: usize) -> Self {
        debug_assert!(uses > 0);
        Self {
            value: Some(value),
            remaining: uses,
        }
    }

    pub(super) fn take(&mut self, eng: &Engine) -> Result<Tdd, OperationError> {
        debug_assert!(self.remaining > 0);
        let result = if self.remaining == 1 {
            self.value.take().expect("live composition operand")
        } else {
            self.value
                .as_ref()
                .expect("live composition operand")
                .try_clone_on(eng)?
        };
        self.remaining -= 1;
        Ok(result)
    }
}

/// Exclusive disjunction: exactly one operand holds.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are consumed, must be structural, and must share a vtree
/// allocation. The result is minimized; weight handling and intermediate
/// storage follow [`ite`](crate::ite).
///
/// # Errors
///
/// The structural-input, compatibility and resource errors of [`ite`](crate::ite).
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, xor, Vtree};
/// let vtree = Arc::new(Vtree::balanced(2));
/// let parity = xor(literal(&vtree, 1)?, literal(&vtree, 2)?)?;
/// assert_eq!(parity.model_count()?, 2u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn xor(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.xor(f, g))
}

/// If `condition` holds, use `then_branch`; otherwise use `else_branch`.
///
/// Uses the shared vtree's execution context automatically.
///
/// The condition is itself a Boolean function, evaluated on each assignment:
/// `(condition ∧ then_branch) ∨ (¬condition ∧ else_branch)`. All three operands
/// must be structural, on one vtree or on projections of one home, with
/// compatible weights;
/// an unweighted operand inherits the agreed weights. The result is minimized.
///
/// Operands are consumed on success or error. Composition copies the condition
/// and builds intermediate diagrams using the shared vtree context; temporary
/// storage can exceed the result's size.
///
/// # Errors
///
/// Returns a vtree, root, weight or marginal-level error before composition,
/// or a resource error from a component operation.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, ite, Vtree};
/// let vtree = Arc::new(Vtree::balanced(3));
/// let select = literal(&vtree, 1)?;
/// let yes = literal(&vtree, 2)?;
/// let no = literal(&vtree, 3)?;
/// let choice = ite(select, yes, no)?;
/// assert_eq!(choice.model_count()?, 4u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn ite(condition: Tdd, then_branch: Tdd, else_branch: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(condition.context());
    context.run(|eng| eng.ite(condition, then_branch, else_branch))
}

/// How [`and_exists`] removes the quantified variables.
///
/// Every setting computes the same function and returns it in the same
/// canonical form; they differ in what is built on the way, and a disagreement
/// between any two of them is a defect.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Quantification {
    /// Quantify each variable one operand leaves free out of the *other*
    /// operand, before the product. The default: it is the rewrite that can
    /// remove a variable from the problem rather than from the answer, and it
    /// costs one pass over an operand that is about to be conjoined anyway.
    ///
    /// An operand with more than sixteen times the other's pairs is not
    /// quantified before the product: the product reaches only the part
    /// of it that the smaller operand leads to, and quantifying after the
    /// product sweeps only that part, where a pass over the large operand
    /// would sweep all of it. The smaller operand is still quantified first.
    #[default]
    Fused,
    /// Build the whole conjunction, then quantify it. Useful when neither
    /// operand offers variables that can be eliminated before conjunction.
    Product,
}

impl Quantification {
    /// Whether this setting rewrites the operands before the product.
    #[inline]
    fn pushes_through(self) -> bool {
        matches!(self, Quantification::Fused)
    }

    /// The setting's name, for an error that names it.
    fn name(self) -> &'static str {
        match self {
            Quantification::Fused => "Quantification::Fused",
            Quantification::Product => "Quantification::Product",
        }
    }
}

/// Existential conjunction: `exists vars. (f AND g)`.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are structural and consumed; the result is minimized and
/// keeps their shared vtree and agreed weights. Quantified variables remain
/// free in that universe, as in [`Tdd::exists_vars`]. Summing counts with
/// [`Engine::and_marginalizing`] is a different operation.
///
/// Quantification follows [`Quantification::Fused`] on unweighted operands
/// and [`Quantification::Product`] on weighted ones; [`Engine::and_exists_with`]
/// takes the choice.
///
/// # Errors
///
/// Validates operand compatibility, structure and every variable before
/// applying; component resource errors propagate.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, and_exists, xor, Vtree};
/// use tididi::vtree::VarId;
/// let vtree = Arc::new(Vtree::balanced(2));
/// let current = literal(&vtree, -1)?; // current state x is false
/// let transition = xor(literal(&vtree, 1)?, literal(&vtree, 2)?)?;
/// // The relation flips x to next-state y; forget the current-state variable.
/// let next = and_exists(current, transition, &[VarId(1)])?;
/// assert!(next.equivalent(&literal(&vtree, 2)?)?);
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn and_exists(f: Tdd, g: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.and_exists(f, g, vars))
}

impl Engine {
    /// Run [`ite`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn ite(
        &self,
        condition: Tdd,
        then_branch: Tdd,
        else_branch: Tdd,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        for f in [&condition, &then_branch, &else_branch] {
            f.require_structure()?;
        }
        let [mut condition, mut then_branch, mut else_branch] =
            super::extend::on_one(self, [condition, then_branch, else_branch].map(Cow::Owned))?.map(Cow::into_owned);
        super::prepare_weights(&mut [&mut condition, &mut then_branch, &mut else_branch])?;
        let mut condition = SharedCircuit::new(condition, 2);
        let yes = self.and(condition.take(self)?, then_branch)?;
        let no = self.and_not(else_branch, condition.take(self)?)?;
        self.or(yes, no)
    }

    /// Run [`xor`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn xor(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        f.require_structure()?;
        g.require_structure()?;
        let [mut f, mut g] = super::extend::on_one(self, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
        super::prepare_weights(&mut [&mut f, &mut g])?;
        let mut f = SharedCircuit::new(f, 2);
        let mut g = SharedCircuit::new(g, 2);
        let only_f = self.and_not(f.take(self)?, g.take(self)?)?;
        let only_g = self.and_not(g.take(self)?, f.take(self)?)?;
        self.or(only_f, only_g)
    }

    /// Run [`and_exists`] using this batch's scratch and resource limits.
    ///
    /// Unweighted operands take [`Quantification::Fused`]; weighted operands
    /// take [`Quantification::Product`], the one setting they admit.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_exists(&self, f: Tdd, g: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        let how = if f.weights.is_some() || g.weights.is_some() {
            Quantification::Product
        } else {
            Quantification::default()
        };
        self.and_exists_with(f, g, vars, how)
    }

    /// Run [`and_exists`] with a chosen [`Quantification`], using this batch's
    /// scratch and resource limits.
    ///
    /// All settings return the same canonical function.
    ///
    /// # Errors
    ///
    /// [`OperationError::InertOption`] when a fused setting is chosen for
    /// weighted operands: quantifying a variable out of one of them before the
    /// product would leave the product's weights unresolved, so weighted
    /// operands admit only [`Quantification::Product`]. Otherwise the
    /// operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_exists_with(
        &self,
        f: Tdd,
        g: Tdd,
        vars: &[VarId],
        how: Quantification,
    ) -> Result<Tdd, OperationError> {
        self.and_exists_with_reduction(f, g, vars, how, ReductionPlan::default())
    }

    /// [`and_exists_with`](Self::and_exists_with) with the reduction it ends
    /// with chosen explicitly.
    ///
    /// Every plan returns the same function. [`ReductionPlan::Prune`] leaves
    /// every node reachable without establishing canonical form: it skips the
    /// twin contraction, so the result can be larger than the minimized one,
    /// and a later [`minimize`](Self::minimize) finishes the job. An operand
    /// quantified before the product is minimized whatever the plan.
    ///
    /// # Errors
    ///
    /// As [`and_exists_with`](Self::and_exists_with).
    pub fn and_exists_with_reduction(
        &self,
        f: Tdd,
        g: Tdd,
        vars: &[VarId],
        how: Quantification,
        plan: ReductionPlan<'_>,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        f.require_structure()?;
        g.require_structure()?;
        let [mut f, mut g] = super::extend::on_one(self, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
        super::prepare_weights(&mut [&mut f, &mut g])?;
        // Quantifying a weighted operand away entirely would replace its store
        // by an empty one, and which of the two stores the product then carries
        // is a question the rewrites below do not answer.
        if how.pushes_through() && f.weights.is_some() {
            return Err(OperationError::InertOption {
                option: how.name(),
                needs: "unweighted operands",
            });
        }
        let targets = super::project::quantification_targets(self, f.vtree(), vars)?;
        if how.pushes_through() {
            (f, g) = push_local_targets(self, f, g, &targets)?;
        }
        let product = self.and(f, g)?;
        // The product's reduction can leave a target's leaf.
        let targets = super::project::quantification_targets(self, product.vtree(), vars)?;
        if targets.is_empty() || product.is_zero() {
            // No sweep runs to reduce the product, so it is reduced here.
            let mut product = product;
            self.reduce(&mut product, plan)?;
            return super::extend::trimmed(self, product);
        }
        super::extend::trimmed(self, super::project::exists_targets_on(self, product, &targets, plan)?)
    }
}

/// Quantify out of each operand the targets the other one does not constrain,
/// before the product.
///
/// `∃S.(f ∧ g) = (∃S.f) ∧ g` when `g` is constant over every leaf of `S` that
/// `f` is not (there, quantifying `f` is the identity), and symmetrically. The
/// unit is such a *covered* subtree — every leaf a target that at most one
/// operand constrains — taken whole, so that the operand's own sweep replaces
/// it by `⊤` and regroups only its ancestors.
///
/// A covered subtree that is a whole quantified subtree of its own, under a
/// parent that is not, is always pushed: the product then never builds it,
/// and the ancestors the push regroups are levels the quantification after
/// the product regroups anyway. One inside a larger quantified subtree is
/// different, because that quantification takes it for free, with the rest of
/// the larger subtree: it is pushed only when the operand has more pairs
/// inside it than on its ancestors, whose regroup is what the push costs. That
/// separates a variable only one operand has, most of whose operand vanishes,
/// from the low bits of a code the other operand happens to be constant over —
/// a relation over a dictionary range whose size ends in zero bits constrains
/// the variable but not those bits — whose leaf levels hold nothing, where a
/// push would regroup every level of the operand above them. Leaf-constancy
/// is decided by the conjunction's own identity precompute, which reads it off
/// the references into the leaf's level: sound, and deliberately incomplete —
/// a missed target only stays in the product.
///
/// Nothing is pushed into an operand with more than [`PUSH_RATIO`] times the
/// other's pairs ([`far_larger`]). A push sweeps the whole operand it goes
/// into, and the argument above — the ancestors it regroups are levels the
/// quantification after the product regroups anyway — assumes the product is
/// about as large as that operand. Beside a far smaller operand it is not: the
/// product reaches only the cells the smaller operand's pairs lead to, and the
/// quantification after it sweeps only those, so the push would pay a pass
/// over the large operand to remove variables from cells the product never
/// builds. The smaller operand is still pushed into: its pass is the cheap
/// one, and it shrinks the product.
///
/// Every target stays a target. A leaf removed here leaves both operands
/// constant over it, so quantifying it again is the identity; keeping it is
/// what holds each subtree's target set down-closed, and a subtree that is not
/// down-closed is one the quantification after the product regroups instead
/// of freeing whole.
///
/// # Errors
///
/// A refused reservation or a component quantification's error; both operands
/// are consumed on every outcome.
pub(super) fn push_local_targets(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: &[VtreeIdx],
) -> Result<(Tdd, Tdd), OperationError> {
    if targets.is_empty() || f.is_zero() || g.is_zero() {
        return Ok((f, g));
    }
    let lim = eng.limits();
    let vtree = std::sync::Arc::clone(f.vtree());
    let num_nodes = vtree.num_nodes();
    let mut into_f = Transient::new(lim, Vec::<VtreeIdx>::new());
    let mut into_g = Transient::new(lim, Vec::<VtreeIdx>::new());
    {
        let mut free_in_f = eng.scratch.apply.f_identity.checkout(eng);
        let mut free_in_g = eng.scratch.apply.g_identity.checkout(eng);
        super::conjoin::init_leaf_identity(eng, &mut free_in_f, &f)?;
        super::conjoin::init_leaf_identity(eng, &mut free_in_g, &g)?;
        // Bottom-up: whether every leaf below a node is a target, and a
        // covered one; an operand's flag at a covered node, its constancy over
        // all of it (a flag off a covered node is never read).
        let mut whole = Transient::new(lim, Vec::new());
        lim.try_resize(&mut whole, num_nodes, false)?;
        let mut covered = Transient::new(lim, Vec::new());
        lim.try_resize(&mut covered, num_nodes, false)?;
        for &leaf in targets {
            whole[leaf.idx()] = true;
            covered[leaf.idx()] = free_in_f[leaf.idx()] || free_in_g[leaf.idx()];
        }
        for (t, left, right) in vtree.internal_bottomup() {
            let (l, r) = (left.idx(), right.idx());
            whole[t.idx()] = whole[l] && whole[r];
            covered[t.idx()] = covered[l] && covered[r];
            free_in_f[t.idx()] = free_in_f[l] && free_in_f[r];
            free_in_g[t.idx()] = free_in_g[l] && free_in_g[r];
        }
        // Each maximal covered subtree, and the operands it is pushed into:
        // bit 0 for `f`, bit 1 for `g`.
        let mut push = Transient::new(lim, Vec::new());
        lim.try_resize(&mut push, num_nodes, 0u8)?;
        let mut costs: [Option<Transient<'_, LevelPairs>>; 2] = [None, None];
        let large = far_larger(&f, &g);
        for &t in vtree.bottomup_slice() {
            let parent = vtree.node(t).parent();
            if !covered[t.idx()] || parent.is_some_and(|p| covered[p.idx()]) {
                continue;
            }
            let alone = parent.is_none_or(|p| !whole[p.idx()]);
            for (side, free, operand) in [(0, &*free_in_f, &f), (1, &*free_in_g, &g)] {
                if free[t.idx()] || large == Some(side) {
                    continue;
                }
                let pays = alone || {
                    if costs[side].is_none() {
                        costs[side] =
                            Some(Transient::new(lim, LevelPairs::of(eng, operand, &vtree)?));
                    }
                    let pairs = costs[side].as_ref().expect("computed above");
                    pairs.inside[t.idx()] > pairs.above[t.idx()]
                };
                if pays {
                    push[t.idx()] |= 1 << side;
                }
            }
        }
        // Top-down: a covered subtree's leaves go where it goes.
        for (t, left, right) in vtree.internal_bottomup().rev() {
            if covered[t.idx()] {
                push[left.idx()] |= push[t.idx()];
                push[right.idx()] |= push[t.idx()];
            }
        }
        for &leaf in targets {
            if push[leaf.idx()] & 1 != 0 {
                lim.try_push(&mut into_f, leaf)?;
            }
            if push[leaf.idx()] & 2 != 0 {
                lim.try_push(&mut into_g, leaf)?;
            }
        }
    }
    if !into_f.is_empty() {
        f = super::project::exists_targets_on(eng, f, &into_f, ReductionPlan::default())?;
    }
    if !into_g.is_empty() {
        g = super::project::exists_targets_on(eng, g, &into_g, ReductionPlan::default())?;
    }
    Ok((f, g))
}

/// How many times the other operand's pairs an operand of
/// [`Quantification::Fused`] may hold and still be quantified before the
/// product ([`push_local_targets`]).
const PUSH_RATIO: usize = 16;

/// The pair count of each of `tdd`'s nodes, level by level.
fn pair_stream(tdd: &Tdd) -> impl Iterator<Item = usize> + '_ {
    tdd.levels
        .iter()
        .flat_map(crate::diagram::TddLevel::pair_counts)
}

/// Which operand, `0` for `f` and `1` for `g`, holds more than
/// [`PUSH_RATIO`] times the other's pairs, if either does.
///
/// The two are read node by node, always from the one read less far in
/// pairs, until one is read whole: that one is the smaller, its count is
/// exact, and the other is then read only as far as the bound. The cost is a
/// small multiple of the smaller operand's size, never a pass over a large
/// one.
fn far_larger(f: &Tdd, g: &Tdd) -> Option<usize> {
    let mut sides = [pair_stream(f), pair_stream(g)];
    let mut sums = [0usize; 2];
    loop {
        let side = usize::from(sums[1] < sums[0]);
        match sides[side].next() {
            Some(k) => sums[side] += k,
            None => {
                let other = if side == 0 { g } else { f };
                let bound = sums[side].saturating_mul(PUSH_RATIO);
                return (!other.pair_count_at_most(bound)).then_some(1 - side);
            }
        }
    }
}

/// An operand's pairs by vtree node: `inside[t]`, those of the levels in
/// `t`'s subtree; `above[t]`, those of `t`'s proper ancestors.
struct LevelPairs {
    inside: Vec<u64>,
    above: Vec<u64>,
}

impl LevelPairs {
    /// Sum `tdd`'s live pairs level by level over `vtree`, both ways.
    fn of(eng: &Engine, tdd: &Tdd, vtree: &crate::vtree::Vtree) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let mut inside = Transient::new(lim, Vec::new());
        lim.try_resize(&mut inside, vtree.num_nodes(), 0u64)?;
        let mut above = Transient::new(lim, Vec::new());
        lim.try_resize(&mut above, vtree.num_nodes(), 0u64)?;
        for (t, left, right) in vtree.internal_bottomup() {
            inside[t.idx()] =
                tdd.levels[t.idx()].live_pairs() as u64 + inside[left.idx()] + inside[right.idx()];
        }
        for (t, left, right) in vtree.internal_bottomup().rev() {
            let here = above[t.idx()] + tdd.levels[t.idx()].live_pairs() as u64;
            above[left.idx()] = here;
            above[right.idx()] = here;
        }
        Ok(Self {
            inside: inside.keep(),
            above: above.keep(),
        })
    }
}

impl Charged for LevelPairs {
    fn charged_bytes(&self) -> u64 {
        self.inside.charged_bytes() + self.above.charged_bytes()
    }
}
