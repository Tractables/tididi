//! Boolean combinations built from the shared apply and quantification kernels.

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
/// Quantification follows [`Quantification::Fused`]; see
/// [`Engine::and_exists_with`] for the reference route.
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
        super::check_vtree(&condition, &then_branch)?;
        super::check_vtree(&condition, &else_branch)?;
        for f in [&condition, &then_branch, &else_branch] {
            f.require_structure()?;
        }
        super::prepare_weights(&mut [&mut condition, &mut then_branch, &mut else_branch])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
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
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights(&mut [&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        let mut g = SharedCircuit::new(g, 2);
        let not_g = self.negate(g.take(self)?)?;
        self.ite(f, not_g, g.take(self)?)
    }

    /// Run [`and_exists`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_exists(&self, f: Tdd, g: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        self.and_exists_with(f, g, vars, Quantification::default())
    }

    /// Run [`and_exists`] with a chosen [`Quantification`], using this batch's
    /// scratch and resource limits.
    ///
    /// All settings return the same canonical function. Weighted operands
    /// always use [`Quantification::Product`] to preserve their interpretation,
    /// even when a fused setting is requested.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_exists_with(
        &self,
        mut f: Tdd,
        mut g: Tdd,
        vars: &[VarId],
        how: Quantification,
    ) -> Result<Tdd, OperationError> {
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights(&mut [&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        let targets = super::project::quantification_targets(self, f.vtree(), vars)?;
        // A weighted operand is left on the reference route. Quantifying one of
        // them away entirely would replace its store by an empty one, and which
        // of the two stores the product then carries is a question the rewrites
        // below do not answer.
        let fused = how.pushes_through() && f.weights.is_none();
        if !fused {
            let product = self.and(f, g)?;
            let identity = targets.is_empty() || product.is_zero();
            let mut result = super::project::exists_targets_on(self, product, &targets, &[])?;
            if identity { self.minimize(&mut result)?; }
            return Ok(result);
        }
        (f, g) = push_local_targets(self, f, g, &targets)?;
        let (product, collapsed) = if how.collapses_subtrees() {
            let vtree = std::sync::Arc::clone(f.vtree());
            let subtrees = quantified_subtrees(self, &vtree, &targets)?;
            let (product, swept) =
                super::conjoin::conjoin_quantifying(self, f, g, &subtrees.whole)?;
            (product, if swept { subtrees.maximal } else { Vec::new() })
        } else {
            (self.and(f, g)?, Vec::new())
        };
        // A nonempty quantification minimizes a non-false product.
        let identity = targets.is_empty() || product.is_zero();
        let mut result = super::project::exists_targets_on(self, product, &targets, &collapsed)?;
        if identity { self.minimize(&mut result)?; }
        Ok(result)
    }
}

/// Which vtree nodes a quantification takes whole, and which of those are the
/// tops of their subtrees.
struct Quantified {
    /// Every node all of whose leaves are quantified: what the conjunction
    /// collapses instead of building.
    whole: Vec<bool>,
    /// The maximal ones among the internal nodes: the levels whose parents the
    /// quantification sweep must regroup even though they now hold one node.
    maximal: Vec<bool>,
}

/// Label the vtree by [`Quantified`]'s two rules, bottom-up.
///
/// A leaf that is its subtree's top is not recorded: the sweep hands its
/// parent a map for the three leaf labels whatever the level looks like, so
/// the regroup there already runs.
///
/// # Errors
///
/// A refused reservation for either label array.
fn quantified_subtrees(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    targets: &[VtreeIdx],
) -> Result<Quantified, OperationError> {
    let lim = eng.limits();
    let num_nodes = vtree.num_nodes();
    let mut whole = Vec::new();
    lim.try_resize(&mut whole, num_nodes, false)?;
    let mut maximal = Vec::new();
    lim.try_resize(&mut maximal, num_nodes, false)?;
    for &leaf in targets {
        whole[leaf.idx()] = true;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        whole[t.idx()] = whole[left.idx()] && whole[right.idx()];
    }
    for (t, _, _) in vtree.internal_bottomup() {
        maximal[t.idx()] = whole[t.idx()]
            && vtree.node(t).parent().is_none_or(|p| !whole[p.idx()]);
    }
    Ok(Quantified { whole, maximal })
}

/// Quantify the targets one operand does not constrain out of the other one,
/// before the product.
///
/// `∃ℓ.(f ∧ g) = f ∧ (∃ℓ.g)` whenever `f` is constant over `ℓ`, and
/// symmetrically. Leaf-constancy is decided by the conjunction's own identity
/// precompute, which reads it off the references into the leaf's level: sound,
/// and deliberately incomplete — a missed target only stays in the product.
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
fn push_local_targets(
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
    let mut free_in_f = eng.apply().left_identity.checkout(lim);
    let mut free_in_g = eng.apply().right_identity.checkout(lim);
    let mut into_f: Vec<VtreeIdx> = Vec::new();
    let mut into_g: Vec<VtreeIdx> = Vec::new();
    let split = (|| -> Result<(), OperationError> {
        super::conjoin::init_leaf_identity(eng, &mut free_in_f, &f, &vtree, num_nodes)?;
        super::conjoin::init_leaf_identity(eng, &mut free_in_g, &g, &vtree, num_nodes)?;
        for &leaf in targets {
            // A leaf both operands are constant over needs no pass at all.
            if free_in_f[leaf.idx()] && !free_in_g[leaf.idx()] {
                lim.try_push(&mut into_g, leaf)?;
            } else if free_in_g[leaf.idx()] && !free_in_f[leaf.idx()] {
                lim.try_push(&mut into_f, leaf)?;
            }
        }
        Ok(())
    })();
    drop(free_in_f);
    drop(free_in_g);
    split?;
    if !into_f.is_empty() {
        f = super::project::exists_targets_on(eng, f, &into_f, &[])?;
    }
    if !into_g.is_empty() {
        g = super::project::exists_targets_on(eng, g, &into_g, &[])?;
    }
    lim.discard(into_f);
    lim.discard(into_g);
    Ok((f, g))
}
