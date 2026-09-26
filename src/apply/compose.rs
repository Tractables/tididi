//! Boolean combinations built from the shared apply and quantification kernels.

use crate::limits::{Charged, Transient};
use crate::reduce::ReductionPlan;
use crate::vtree::{VarId, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

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
        Self { value: Some(value), remaining: uses }
    }

    pub(super) fn take(&mut self, eng: &Engine) -> Result<Tdd, OperationError> {
        debug_assert!(self.remaining > 0);
        let result = if self.remaining == 1 {
            self.value.take().expect("live composition operand")
        } else {
            self.value.as_ref().expect("live composition operand").try_clone_on(eng)?
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
/// must be structural and share a vtree allocation and compatible weights;
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
    #[default]
    Fused,
    /// [`Fused`](Quantification::Fused), and additionally do not build a vtree
    /// subtree every leaf of which is quantified: decide each of its product
    /// cells by one satisfiability test and write the `⊤` that quantifying it
    /// would have left.
    ///
    /// This is the deeper rewrite and it is *not* the default, because it wins
    /// only where the collapsed subtree would otherwise have been materialized
    /// in full. Where the conjunction would have reached that subtree by a
    /// route cheaper than a walk of its product grid — the sparse route, or an
    /// identity level — the collapse bypasses that route and pays more than the
    /// build it replaces.
    FusedSubtrees,
    /// Build the whole conjunction, then quantify it. Useful when neither
    /// operand offers variables that can be eliminated before conjunction.
    Product,
}

impl Quantification {
    /// Whether this setting rewrites the operands before the product.
    #[inline]
    fn pushes_through(self) -> bool {
        matches!(self, Quantification::Fused | Quantification::FusedSubtrees)
    }

    /// Whether this setting collapses a fully quantified subtree.
    #[inline]
    fn collapses_subtrees(self) -> bool {
        matches!(self, Quantification::FusedSubtrees)
    }

    /// The setting's name, for an error that names it.
    fn name(self) -> &'static str {
        match self {
            Quantification::Fused => "Quantification::Fused",
            Quantification::FusedSubtrees => "Quantification::FusedSubtrees",
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
        mut condition: Tdd,
        mut then_branch: Tdd,
        mut else_branch: Tdd,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        super::check_vtree(&condition, &then_branch)?;
        super::check_vtree(&condition, &else_branch)?;
        for f in [&condition, &then_branch, &else_branch] {
            f.require_structure()?;
        }
        super::prepare_weights(&mut [&mut condition, &mut then_branch, &mut else_branch])?;
        let mut condition = SharedCircuit::new(condition, 2);
        let otherwise = self.negate(condition.take(self)?)?;
        let yes = self.and(condition.take(self)?, then_branch)?;
        let no = self.and(otherwise, else_branch)?;
        self.or(yes, no)
    }

    /// Run [`xor`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn xor(&self, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights(&mut [&mut f, &mut g])?;
        let mut g = SharedCircuit::new(g, 2);
        let not_g = self.negate(g.take(self)?)?;
        self.ite(f, not_g, g.take(self)?)
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
        mut f: Tdd,
        mut g: Tdd,
        vars: &[VarId],
        how: Quantification,
        plan: ReductionPlan<'_>,
    ) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights(&mut [&mut f, &mut g])?;
        // Quantifying a weighted operand away entirely would replace its store
        // by an empty one, and which of the two stores the product then carries
        // is a question the rewrites below do not answer.
        if how.pushes_through() && f.weights.is_some() {
            return Err(OperationError::InertOption { option: how.name(), needs: "unweighted operands" });
        }
        let targets = super::project::quantification_targets(self, f.vtree(), vars)?;
        let (product, collapsed) = if how.pushes_through() {
            (f, g) = push_local_targets(self, f, g, &targets)?;
            if how.collapses_subtrees() {
                let vtree = std::sync::Arc::clone(f.vtree());
                let whole = Transient::new(self.limits(), quantified_subtrees(self, &vtree, &targets)?);
                super::conjoin::conjoin_on(self, f, g, super::conjoin::VtreeMask::new(Some(&whole[..])))?
            } else {
                (self.and(f, g)?, false)
            }
        } else {
            (self.and(f, g)?, false)
        };
        if targets.is_empty() || product.is_zero() {
            // No sweep runs to reduce the product, so it is reduced here.
            let mut product = product;
            self.reduce(&mut product, plan)?;
            return Ok(product);
        }
        super::project::exists_targets_on(self, product, &targets, collapsed, plan)
    }
}

/// Label every vtree node all of whose leaves are quantified, bottom-up: what
/// the conjunction collapses instead of building.
///
/// # Errors
///
/// A refused reservation for the label array.
fn quantified_subtrees(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    targets: &[VtreeIdx],
) -> Result<Vec<bool>, OperationError> {
    let mut whole = Vec::new();
    eng.limits().try_resize(&mut whole, vtree.num_nodes(), false)?;
    for &leaf in targets {
        whole[leaf.idx()] = true;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        whole[t.idx()] = whole[left.idx()] && whole[right.idx()];
    }
    Ok(whole)
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
/// Every target stays a target. A leaf removed here leaves both operands
/// constant over it, so quantifying it again is the identity; keeping it is
/// what holds each subtree's target set down-closed, and a subtree that is not
/// down-closed is one the conjunction cannot collapse.
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
        for &t in vtree.bottomup_slice() {
            let parent = vtree.node(t).parent();
            if !covered[t.idx()] || parent.is_some_and(|p| covered[p.idx()]) {
                continue;
            }
            let alone = parent.is_none_or(|p| !whole[p.idx()]);
            for (side, free, operand) in [(0, &*free_in_f, &f), (1, &*free_in_g, &g)] {
                if free[t.idx()] {
                    continue;
                }
                let pays = alone || {
                    if costs[side].is_none() {
                        costs[side] = Some(Transient::new(lim, LevelPairs::of(eng, operand, &vtree)?));
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
        f = super::project::exists_targets_on(eng, f, &into_f, false, ReductionPlan::default())?;
    }
    if !into_g.is_empty() {
        g = super::project::exists_targets_on(eng, g, &into_g, false, ReductionPlan::default())?;
    }
    Ok((f, g))
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
            inside[t.idx()] = tdd.levels[t.idx()].live_pairs() as u64 + inside[left.idx()] + inside[right.idx()];
        }
        for (t, left, right) in vtree.internal_bottomup().rev() {
            let here = above[t.idx()] + tdd.levels[t.idx()].live_pairs() as u64;
            above[left.idx()] = here;
            above[right.idx()] = here;
        }
        Ok(Self { inside: inside.keep(), above: above.keep() })
    }
}

impl Charged for LevelPairs {
    fn charged_bytes(&self) -> u64 {
        self.inside.charged_bytes() + self.above.charged_bytes()
    }
}
