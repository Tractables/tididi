//! Fold a value block over the prefix of an order block.
//!
//! [`Engine::prefix_fold`] pairs each point of one function with the fold
//! of the values a second function holds at the point's key, at every order
//! up to the point's (or from it on, [`FoldRange`]): a running sum, a
//! running maximum or a rank, computed by one walk down both diagrams
//! rather than by joining the second with itself.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::apply::CONJOIN_GRID;
use crate::apply::conjoin::budget::NO_PRODUCT;
use crate::diagram::{
    Assembly, ChildPair, ColumnAlgebra, LeafLabel, NodeIdx, SlotPairs, Tdd, TddNodeId, LEAF_WIDTH, NEG_LEAF_IDX, ONE_LEAF_IDX,
    POS_LEAF_IDX,
};
use crate::limits::{Limits, OperationError, PollGate};
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx};

/// The values a prefix fold combines ([`Engine::prefix_fold`]): each model
/// of the value block stands for a value, and `⊕` combines two values. `⊕`
/// is commutative and associative, so the fold of a set of models does not
/// depend on the order it is taken in.
///
/// No identity is needed: a point whose range holds no model is left out
/// of the result, so the fold of an empty set is never formed.
///
/// ```
/// use tididi::apply::FoldMonoid;
///
/// /// The largest value seen.
/// struct Max;
/// impl FoldMonoid for Max {
///     type Value = u64;
///     fn value(&self, code: u64) -> u64 { code }
///     fn combine(&self, a: &u64, b: &u64) -> u64 { *a.max(b) }
/// }
/// assert_eq!(Max.combine(&Max.value(3), &Max.value(5)), 5);
///
/// /// How many models: every model is worth one, so the fold reads only
/// /// how many each node of the value block holds ([`count`](FoldMonoid::count)).
/// struct Count;
/// impl FoldMonoid for Count {
///     type Value = u128;
///     fn value(&self, _: u64) -> u128 { 1 }
///     fn combine(&self, a: &u128, b: &u128) -> u128 { a + b }
///     fn count(&self, n: u128) -> Option<u128> { Some(n) }
/// }
/// assert_eq!(Count.count(3), Some(Count.combine(&Count.value(4), &Count.combine(&Count.value(0), &Count.value(7)))));
/// ```
pub trait FoldMonoid {
    /// A value, or the fold of several. Equal folds share one code.
    type Value: Clone + Eq + std::hash::Hash;
    /// The value of one model of the value block, read as the unsigned
    /// number whose bit `i` is the value block's variable `i`.
    fn value(&self, code: u64) -> Self::Value;
    /// `a ⊕ b`.
    fn combine(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
    /// The fold of any `n ≥ 1` models, where every model of the value block
    /// has one value (a count of models); `None`, the default, where a
    /// model's value depends on its code. A monoid returns `Some` for every
    /// `n` or for none.
    ///
    /// Where it returns `Some`, the fold reads the value block by the number
    /// of models each of its nodes holds, through one model count of the
    /// value subtree, and never reads a code: the block may then hold up to
    /// 127 variables, where a block read by its codes holds at most 64.
    fn count(&self, n: u128) -> Option<Self::Value> {
        let _ = n;
        None
    }
}

/// Which orders each point's fold reads ([`Engine::prefix_fold`]): at a
/// point of order `e`, the values at the orders `e'` that stand so to `e`.
///
/// ```
/// use tididi::apply::FoldRange;
///
/// // A rank counts the values above a point; a running total those up to it.
/// assert_ne!(FoldRange::Above, FoldRange::AtMost);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoldRange {
    /// `e' ≤ e`: a running fold up to the point, its own order in.
    AtMost,
    /// `e' < e`: the orders before the point's.
    Below,
    /// `e' ≥ e`: a running fold from the top order down to the point, its
    /// own order in.
    AtLeast,
    /// `e' > e`: the orders after the point's, as a rank counts them.
    Above,
}

impl FoldRange {
    /// Whether the fold runs from the top order down.
    fn descending(self) -> bool {
        matches!(self, Self::AtLeast | Self::Above)
    }

    /// Whether the point's own order is left out.
    fn strict(self) -> bool {
        matches!(self, Self::Below | Self::Above)
    }
}

/// How a prefix fold codes its folds ([`Engine::prefix_fold`]): the vtree
/// the result lies on, the code block, and a code per fold.
#[derive(Clone, Debug)]
pub struct FoldCodes {
    /// The operands' vtree with the value subtree replaced by a subtree
    /// whose leaves are exactly `code_vars`; the code block may reuse the
    /// value block's variables.
    pub into: Arc<Vtree>,
    /// The code block: bit `i` of a code is the value of `code_vars[i]`.
    pub code_vars: Vec<VarId>,
    /// One code per fold, in the order the folds were given. Two folds may
    /// share a code.
    pub codes: Vec<u64>,
}

/// Why a prefix fold was refused ([`Engine::prefix_fold`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrefixFoldError {
    /// The blocks do not lie on the vtrees as the fold reads them: a block
    /// is empty, the order block, the code block or a value block read by
    /// its codes is wider than 64 variables, a value block read by its
    /// counts ([`FoldMonoid::count`]) wider than 127, the order block's leaves are not
    /// the left children of a right-linear chain whose last right child
    /// holds exactly the value block, or the coding's `into` is not the
    /// vtree with that value subtree replaced by one whose leaves are exactly
    /// its code variables.
    Layout(String),
    /// A variable named twice: within one block, or in both the order block
    /// and the value block.
    Variables(VtreeError),
    /// The codes do not fit the folds: not one code per fold, or a code past
    /// the range of the code variables.
    Codes(String),
    /// An operation the fold runs was refused: operands on two vtrees, a
    /// diagram that has discarded the structure at a level, a refused
    /// allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for PrefixFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout(message) => write!(f, "prefix fold layout: {message}"),
            Self::Variables(error) => write!(f, "prefix fold variables: {error}"),
            Self::Codes(message) => write!(f, "prefix fold codes: {message}"),
            Self::Operation(error) => write!(f, "prefix fold: {error}"),
        }
    }
}

impl std::error::Error for PrefixFoldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Variables(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Layout(_) | Self::Codes(_) => None,
        }
    }
}

impl From<OperationError> for PrefixFoldError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// No node, set, state or fold.
const NONE: u32 = u32::MAX;

impl Engine {
    /// Each point of `points` with the fold of the values `values` holds at
    /// its key up to its order, or from it on: a running sum, a running
    /// maximum, or a rank.
    ///
    /// `points` and `values` share one vtree, whose variables fall in three
    /// blocks. The order block `order` is read as an unsigned number `e`,
    /// `order[0]` its most significant bit; the value block `value_vars` as
    /// an unsigned number `v` whose bit `i` is `value_vars[i]`; every other
    /// variable is the key `k`. A point is an assignment `(k, e)` that some
    /// model of `points` extends. Its fold is
    ///
    /// `F(k, e) = ⊕ { monoid.value(v) : values(k, e', v) and e' ~ e }`,
    ///
    /// where `~` is `range`'s comparison, `≤`, `<`, `≥` or `>`
    /// ([`FoldRange`]): every model of `values` at key `k` and an order in
    /// the range is folded in, several at one `e'` included. A point whose
    /// range holds no model of `values` is left out; a point at an order
    /// where `values` holds none carries the fold of the orders past it.
    /// Where `monoid` counts ([`FoldMonoid::count`]), the value block is
    /// read by how many models each node holds, not by their codes, so
    /// `F(k, e)` is `count(n)` for the `n` models in range.
    ///
    /// `coding` is called once with the distinct folds, in an unspecified
    /// order (none where no point keeps a fold), and returns how they are
    /// coded ([`FoldCodes`]): a vtree `into`, a code block and one code per
    /// fold, so the block can be as wide as the folds need. The result, on
    /// `into`, holds `(k, e, c)` for each point that keeps a fold, `c` the
    /// code of its fold, and nothing else.
    ///
    /// The fold reads its blocks off the vtree: the leaves of the order block
    /// are the left children of a right-linear chain, `order[0]` at the
    /// chain's top node `t`, and the right child of its last node is the
    /// subtree of the value block, whose leaves are exactly `value_vars`. The
    /// key lies outside `t`'s subtree, on either side of the path to it.
    /// The coding's `into` is the same vtree with the value subtree replaced
    /// by a subtree whose leaves are exactly its code variables.
    ///
    /// The walk reads `points` and `values` together from the root: above
    /// `t` as their conjunction on the key (built from the key's leaves up
    /// over the nodes each diagram reaches, so only pairs of nodes whose
    /// conjunction is satisfiable are formed), then down the chain with the set
    /// of nodes each diagram holds there and the fold of the orders already
    /// passed below the point's. Each such state is visited once, and a level
    /// of the chain holds at most one state per distinct prefix of a point's
    /// order, so the chain takes at most `n · order.len()` states for `n`
    /// points, each reading the pairs of its nodes. Before the walk, one pass
    /// over `values` in `t`'s subtree folds each node's models, reading every
    /// model of each node at the value subtree's root (for a count, one
    /// model count of the subtree reads how many it holds). After it, a cube
    /// is built per code and the result is minimized.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::apply::{FoldCodes, FoldMonoid, FoldRange};
    /// use tididi::vtree::VarId;
    ///
    /// /// A running total: a model of the value block is its number.
    /// struct Sum;
    /// impl FoldMonoid for Sum {
    ///     type Value = u64;
    ///     fn value(&self, code: u64) -> u64 { code }
    ///     fn combine(&self, a: &u64, b: &u64) -> u64 { a + b }
    /// }
    ///
    /// // A store x1 beside the chain of a day x2 x3 (x2 most significant)
    /// // over an amount x4 x5 (x4 its low bit). Each total is coded as
    /// // itself on x4 x5 x6 in the amount's place.
    /// let leaf = |v| Vtree::leaf(VarId(v));
    /// let chain = |low: Vtree| Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &Vtree::join(&leaf(3), &low)?)?);
    /// let vtree = Arc::new(chain(Vtree::join(&leaf(4), &leaf(5))?)?);
    /// let into = Arc::new(chain(Vtree::join(&leaf(4), &Vtree::join(&leaf(5), &leaf(6))?)?)?);
    /// let row = |store: u64, day: u64, amount: u64| store | (day >> 1) << 1 | (day & 1) << 2 | amount << 3;
    ///
    /// // Store 0 sells 2 on day 0 and 3 on day 2; store 1 sells 1 on days 1 and 3.
    /// let vars: Vec<VarId> = (1..=5).map(VarId).collect();
    /// let sales = Tdd::from_models(&vtree, &vars, &[row(0, 0, 2), row(0, 2, 3), row(1, 1, 1), row(1, 3, 1)])?;
    /// // Store 0 is open every day, store 1 on days 0 to 2.
    /// let open: Vec<u64> = (0..4).map(|d| row(0, d, 0)).chain((0..3).map(|d| row(1, d, 0))).collect();
    /// let open = Tdd::from_models(&vtree, &vars[..3], &open)?;
    ///
    /// let engine = Engine::new();
    /// let (order, amount) = ([VarId(2), VarId(3)], [VarId(4), VarId(5)]);
    /// let coding = |totals: &[u64]| FoldCodes {
    ///     into: into.clone(),
    ///     code_vars: vec![VarId(4), VarId(5), VarId(6)],
    ///     codes: totals.to_vec(),
    /// };
    /// let totals = engine.prefix_fold(&open, &sales, &order, &amount, FoldRange::AtMost, &Sum, coding)?;
    ///
    /// // Store 0 runs 2, 2, 5, 5. Store 1 has no sale up to day 0, then 1
    /// // on day 1 and on day 2, which sells nothing.
    /// let out: Vec<VarId> = (1..=6).map(VarId).collect();
    /// let rows = [row(0, 0, 2), row(0, 1, 2), row(0, 2, 5), row(0, 3, 5), row(1, 1, 1), row(1, 2, 1)];
    /// assert!(engine.equivalent(&totals, &Tdd::from_models(&into, &out, &rows)?)?);
    /// # tididi::test_helpers::assert_canonical(&totals);
    ///
    /// // How many sales each open day has after it, from the top day down:
    /// // a count, which reads how many models a node holds, not their codes.
    /// struct Count;
    /// impl FoldMonoid for Count {
    ///     type Value = u64;
    ///     fn value(&self, _: u64) -> u64 { 1 }
    ///     fn combine(&self, a: &u64, b: &u64) -> u64 { a + b }
    ///     fn count(&self, n: u128) -> Option<u64> { u64::try_from(n).ok() }
    /// }
    /// let coding = |counts: &[u64]| FoldCodes {
    ///     into: into.clone(),
    ///     code_vars: vec![VarId(4), VarId(5), VarId(6)],
    ///     codes: counts.to_vec(),
    /// };
    /// let after = engine.prefix_fold(&open, &sales, &order, &amount, FoldRange::Above, &Count, coding)?;
    /// // Store 0 sells after days 0 and 1 (on day 2), not after day 2 or 3;
    /// // store 1 twice after day 0, once after days 1 and 2.
    /// let rows = [row(0, 0, 1), row(0, 1, 1), row(1, 0, 2), row(1, 1, 1), row(1, 2, 1)];
    /// assert!(engine.equivalent(&after, &Tdd::from_models(&into, &out, &rows)?)?);
    /// # tididi::test_helpers::assert_canonical(&after);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`PrefixFoldError::Layout`] when a block is empty, the order or code
    /// block or a value block read by its codes holds more than 64
    /// variables, a counted value block more than 127, an order variable is not the left child of the chain's
    /// next node, the value subtree's leaves are not exactly `value_vars`, or
    /// the coding's `into` differs from the vtree outside the value subtree
    /// or has other leaves than its code variables in its place (checked once
    /// the folds are known); [`PrefixFoldError::Variables`] when a variable is
    /// named twice in `order` and `value_vars` together, or twice in the code
    /// block; [`PrefixFoldError::Codes`] when the coding has other than one
    /// code per fold, or a code with a bit at or past its code block's
    /// width; [`PrefixFoldError::Operation`] with
    /// [`OperationError::VtreeMismatch`] for operands on different vtree
    /// allocations, [`OperationError::MarginalLevel`] for an operand that
    /// has discarded the structure at a level, and for a refused allocation
    /// or an armed stop.
    #[allow(clippy::too_many_arguments)]
    pub fn prefix_fold<M: FoldMonoid>(
        &self,
        points: &Tdd,
        values: &Tdd,
        order: &[VarId],
        value_vars: &[VarId],
        range: FoldRange,
        monoid: &M,
        coding: impl FnOnce(&[M::Value]) -> FoldCodes,
    ) -> Result<Tdd, PrefixFoldError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        crate::apply::check_vtree(points, values)?;
        let vtree = points.vtree();
        let counted = monoid.count(1).is_some();
        let layout = Layout::new(lim, vtree, order, value_vars, counted)?;
        points.require_structure()?;
        values.require_structure()?;
        if points.is_zero() || values.is_zero() {
            let FoldCodes { into, code_vars, codes } = coding(&[]);
            Coding::new(lim, vtree, layout.value_root, &into, &code_vars, &codes, 0)?;
            return Ok(crate::build::constant_zero(self, &into));
        }
        // A count reads how many models each node of the value root holds.
        let counts = match counted {
            true => Some(self.evaluate_columns_at(values, layout.value_root, &ModelCounts)?),
            false => None,
        };
        let mut fold = Fold {
            lim,
            gate: lim.gate(),
            monoid,
            ops: [points, values],
            layout: &layout,
            range,
            counts,
            vals: Vec::new(),
            val_ids: FxHashMap::default(),
            totals: Vec::new(),
            levels: Vec::new(),
            products: Vec::new(),
            slot: Vec::new(),
            reached: Vec::new(),
            reached_at: FxHashMap::default(),
        };
        fold.total()?;
        fold.descend()?;
        let mut folds = Vec::new();
        lim.reserve(&mut folds, fold.reached.len())?;
        folds.extend(fold.reached.iter().map(|&v| fold.vals[v as usize].clone()));
        let FoldCodes { into, code_vars, codes } = coding(&folds);
        let mapped = Coding::new(lim, vtree, layout.value_root, &into, &code_vars, &codes, folds.len())?;
        drop(folds);
        Ok(fold.emit(self, &into, &mapped, &codes)?)
    }
}

/// Where the blocks lie on the two vtrees.
struct Layout {
    /// The nodes from the root down to `t`, each with its child off the
    /// path, the key's, and whether the path goes on to the right child.
    path: Vec<(VtreeIdx, VtreeIdx, bool)>,
    /// The chain from `t` down, the node whose left child is `order[j]` at `j`.
    chain: Vec<VtreeIdx>,
    /// The value subtree's root.
    value_root: VtreeIdx,
    /// Per node of the vtree, the bit of a value code its leaf reads.
    value_bit: Vec<u8>,
}

/// Where the coding puts the folds' codes on `into`.
struct Coding {
    /// Per node of the vtree outside the value subtree, its node in
    /// `into`; at the value root, the code subtree's root.
    image: Vec<VtreeIdx>,
    /// Per node of `into`, the bit of a code its leaf reads.
    code_bit: Vec<u8>,
}

/// A block's refusal: empty, or past `most` variables.
fn check_block(name: &str, block: &[VarId], most: usize) -> Result<(), PrefixFoldError> {
    if block.is_empty() {
        return Err(PrefixFoldError::Layout(format!("the {name} block is empty")));
    }
    if block.len() > most {
        return Err(PrefixFoldError::Layout(format!("the {name} block has {} variables, more than {most}", block.len())));
    }
    Ok(())
}

/// Model counts in `u128`, one column per level: exact on a subtree of at
/// most 127 variables, whose nodes hold at most `2^127` models each.
struct ModelCounts;

impl ColumnAlgebra for ModelCounts {
    type Column = Vec<u128>;
    type Value = u128;

    fn zero(&self) -> u128 {
        0
    }

    fn column(&self, _: VtreeIdx, width: usize) -> Vec<u128> {
        vec![0; width]
    }

    fn leaf(&self, _: VtreeIdx, _: VarId, label: LeafLabel, col: &mut Vec<u128>) {
        col[label as usize] = if label == LeafLabel::One { 2 } else { 1 };
    }

    fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<u128>, right: &Vec<u128>, out: &mut Vec<u128>) {
        out[slot] = pairs.map(|(l, r)| left[l] * right[r]).sum();
    }

    fn read(&self, _: VtreeIdx, col: Vec<u128>, slot: usize) -> u128 {
        col[slot]
    }
}

impl Layout {
    fn new(lim: &Limits, vtree: &Vtree, order: &[VarId], value_vars: &[VarId], counted: bool) -> Result<Self, PrefixFoldError> {
        let layout = |message: String| PrefixFoldError::Layout(message);
        check_block("order", order, 64)?;
        check_block("value", value_vars, if counted { 127 } else { 64 })?;
        // Each variable once in the order and value blocks.
        let mut role: FxHashMap<VarId, u8> = FxHashMap::default();
        lim.reserve_map(&mut role, order.len() + value_vars.len())?;
        for (i, &v) in order.iter().chain(value_vars).enumerate() {
            let bit = i.checked_sub(order.len()).map_or(u8::MAX, |b| b as u8);
            if role.insert(v, bit).is_some() {
                return Err(PrefixFoldError::Variables(VtreeError::OverlappingVariable(v)));
            }
        }

        // The chain: each order variable the left child of the next node.
        let leaf = |v: VarId| vtree.leaf_of(v).ok_or_else(|| layout(format!("order variable {} is not a leaf of the vtree", v.0)));
        let Some(mut d) = vtree.node(leaf(order[0])?).parent() else {
            return Err(layout("the first order variable is the whole vtree".to_string()));
        };
        let mut chain = Vec::new();
        lim.reserve(&mut chain, order.len())?;
        let mut value_root = d;
        for (j, &v) in order.iter().enumerate() {
            let (l, r) = vtree.children(d);
            if l != leaf(v)? {
                return Err(layout(format!("order variable {} is not the left child of the chain's node {j}", v.0)));
            }
            chain.push(d);
            match order.get(j + 1) {
                Some(next) if vtree.node(r).is_leaf() => {
                    return Err(layout(format!("the chain ends before order variable {}", next.0)));
                }
                Some(_) => d = r,
                None => value_root = r,
            }
        }

        // The value subtree: exactly the value block.
        let mut value_bit = Vec::new();
        lim.try_resize(&mut value_bit, vtree.num_nodes(), u8::MAX)?;
        let mut leaves = 0usize;
        let mut stack = Vec::new();
        lim.try_push(&mut stack, value_root)?;
        while let Some(s) = stack.pop() {
            if vtree.node(s).is_leaf() {
                let v = vtree.leaf_var(s);
                match role.get(&v) {
                    Some(&bit) if bit != u8::MAX => value_bit[s.idx()] = bit,
                    _ => return Err(layout(format!("variable {} under the value subtree is not a value variable", v.0))),
                }
                leaves += 1;
            } else {
                let (l, r) = vtree.children(s);
                lim.try_push(&mut stack, l)?;
                lim.try_push(&mut stack, r)?;
            }
        }
        if leaves != value_vars.len() {
            return Err(layout("a value variable is not a leaf of the value subtree".to_string()));
        }

        // The path from the root down to `t`.
        let mut path = Vec::new();
        let mut child = chain[0];
        while let Some(u) = vtree.node(child).parent() {
            let (l, r) = vtree.children(u);
            let right = r == child;
            lim.try_push(&mut path, (u, if right { l } else { r }, right))?;
            child = u;
        }
        path.reverse();
        Ok(Layout { path, chain, value_root, value_bit })
    }
}

impl Coding {
    /// `into` checked against `vtree`, whose value subtree is at
    /// `value_root`, and `codes` against `code_vars`: `folds` codes, each
    /// within the block.
    fn new(
        lim: &Limits,
        vtree: &Vtree,
        value_root: VtreeIdx,
        into: &Vtree,
        code_vars: &[VarId],
        codes: &[u64],
        folds: usize,
    ) -> Result<Self, PrefixFoldError> {
        let layout = |message: String| PrefixFoldError::Layout(message);
        check_block("code", code_vars, 64)?;
        let mut code_role: FxHashMap<VarId, u8> = FxHashMap::default();
        lim.reserve_map(&mut code_role, code_vars.len())?;
        for (i, &v) in code_vars.iter().enumerate() {
            if code_role.insert(v, i as u8).is_some() {
                return Err(PrefixFoldError::Variables(VtreeError::OverlappingVariable(v)));
            }
        }
        if codes.len() != folds {
            return Err(PrefixFoldError::Codes(format!("{} codes for {folds} folds", codes.len())));
        }
        let width = code_vars.len();
        if let Some((i, c)) = codes.iter().enumerate().find(|&(_, &c)| width < 64 && c >> width != 0) {
            return Err(PrefixFoldError::Codes(format!("code {c} of fold {i} is past the range of {width} code variables")));
        }

        // `into`: the vtree outside the value subtree, the code subtree in its place.
        let mut stack = Vec::new();
        let mut image = Vec::new();
        lim.try_resize(&mut image, vtree.num_nodes(), VtreeIdx(NONE))?;
        let mut code_root = None;
        let mut pairs = Vec::new();
        lim.try_push(&mut pairs, (vtree.root(), into.root()))?;
        while let Some((s, d)) = pairs.pop() {
            image[s.idx()] = d;
            if s == value_root {
                code_root = Some(d);
                continue;
            }
            match (vtree.node(s).is_leaf(), into.node(d).is_leaf()) {
                (true, true) if vtree.leaf_var(s) == into.leaf_var(d) => {}
                (true, true) => {
                    let (a, b) = (vtree.leaf_var(s).0, into.leaf_var(d).0);
                    return Err(layout(format!("`into` has variable {b} where the vtree has {a}")));
                }
                (false, false) => {
                    let ((sl, sr), (dl, dr)) = (vtree.children(s), into.children(d));
                    lim.try_push(&mut pairs, (sl, dl))?;
                    lim.try_push(&mut pairs, (sr, dr))?;
                }
                _ => return Err(layout("`into` differs in shape from the vtree outside the value subtree".to_string())),
            }
        }
        let code_root = code_root.expect("the walk reaches every node of a matching shape");
        let mut code_bit = Vec::new();
        lim.try_resize(&mut code_bit, into.num_nodes(), u8::MAX)?;
        let mut leaves = 0usize;
        lim.try_push(&mut stack, code_root)?;
        while let Some(s) = stack.pop() {
            if into.node(s).is_leaf() {
                let v = into.leaf_var(s);
                let Some(&bit) = code_role.get(&v) else {
                    return Err(layout(format!("variable {} under the code subtree is not a code variable", v.0)));
                };
                code_bit[s.idx()] = bit;
                leaves += 1;
            } else {
                let (l, r) = into.children(s);
                lim.try_push(&mut stack, l)?;
                lim.try_push(&mut stack, r)?;
            }
        }
        if leaves != code_vars.len() {
            return Err(layout("a code variable is not a leaf of the code subtree".to_string()));
        }
        Ok(Coding { image, code_bit })
    }
}

/// Sets of nodes of one level, each held once: sorted node indices, laid
/// end to end.
#[derive(Default)]
struct Sets {
    items: Vec<u32>,
    ranges: Vec<[u32; 2]>,
    /// The first set with each hash, and per set the next one with its hash.
    first: FxHashMap<u64, u32>,
    next: Vec<u32>,
}

impl Sets {
    fn get(&self, id: u32) -> &[u32] {
        let [start, len] = self.ranges[id as usize];
        &self.items[start as usize..(start + len) as usize]
    }

    /// The id of sorted, distinct `nodes`.
    fn intern(&mut self, lim: &Limits, nodes: &[u32]) -> Result<u32, OperationError> {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        nodes.hash(&mut hasher);
        let hash = hasher.finish();
        let (mut at, mut last) = (self.first.get(&hash).copied().unwrap_or(NONE), NONE);
        while at != NONE {
            if self.get(at) == nodes {
                return Ok(at);
            }
            last = at;
            at = self.next[at as usize];
        }
        let id = self.ranges.len() as u32;
        let start = self.items.len() as u32;
        lim.reserve(&mut self.items, nodes.len())?;
        self.items.extend_from_slice(nodes);
        lim.try_push(&mut self.ranges, [start, nodes.len() as u32])?;
        lim.try_push(&mut self.next, NONE)?;
        match last {
            NONE => {
                lim.reserve_map(&mut self.first, 1)?;
                self.first.insert(hash, id);
            }
            last => self.next[last as usize] = id,
        }
        Ok(id)
    }
}

/// One level of the walk: a node on the path to `t`, or of the chain.
#[derive(Default)]
struct Level {
    /// Its states: a set of nodes of `points`, one of `values`, and the
    /// fold of the orders passed below the point's (`NONE` above the chain,
    /// and while no model is passed).
    states: Vec<[u32; 3]>,
    index: FxHashMap<[u32; 3], u32>,
    /// The sets the states name, per operand.
    sets: [Sets; 2],
    /// Per state, its moves to the next level: a label (a literal or a
    /// product on the key's side) and the next level's state, or below the
    /// chain the fold's place in `Fold::reached`.
    moves: Vec<[u32; 2]>,
    ranges: Vec<[u32; 2]>,
}

impl Level {
    fn add(&mut self, lim: &Limits, state: [u32; 3]) -> Result<u32, OperationError> {
        if let Some(&id) = self.index.get(&state) {
            return Ok(id);
        }
        let id = self.states.len() as u32;
        lim.try_push(&mut self.states, state)?;
        lim.reserve_map(&mut self.index, 1)?;
        self.index.insert(state, id);
        Ok(id)
    }

    fn moves_of(&self, s: usize) -> &[[u32; 2]] {
        let [start, len] = self.ranges[s];
        &self.moves[start as usize..(start + len) as usize]
    }
}

/// The conjunction of nodes of `points` and of `values` under one level of
/// the key's side, built from its children's: one product per pair of
/// nodes whose conjunction is satisfiable, with the pairs of products (or
/// literals) below it.
#[derive(Default)]
struct Products {
    /// The products, those of one node of `points` together, each with its
    /// pairs and, once it is built, its node in the result.
    index: FxHashMap<[u32; 2], u32>,
    keys: Vec<[u32; 2]>,
    pairs: Vec<[u32; 2]>,
    ranges: Vec<[u32; 2]>,
    /// Per node of `points`, its run of products.
    of_a: FxHashMap<u32, [u32; 2]>,
    /// The products a move reaches, and those below them.
    used: Vec<bool>,
    out: Vec<u32>,
}

impl Products {
    fn pairs_of(&self, p: usize) -> &[[u32; 2]] {
        let [start, len] = self.ranges[p];
        &self.pairs[start as usize..(start + len) as usize]
    }

    /// The products of node `a` of `points`.
    fn run(&self, a: u32) -> std::ops::Range<usize> {
        match self.of_a.get(&a) {
            Some(&[start, len]) => start as usize..(start + len) as usize,
            None => 0..0,
        }
    }
}

/// The key's side of one path level: its internal nodes bottom-up, each
/// with its products.
#[derive(Default)]
struct KeySide {
    order: Vec<VtreeIdx>,
    at: Vec<Products>,
}

struct Fold<'a, M: FoldMonoid> {
    lim: &'a Limits,
    gate: PollGate<'a>,
    monoid: &'a M,
    /// `points` and `values`.
    ops: [&'a Tdd; 2],
    layout: &'a Layout,
    range: FoldRange,
    /// Under a count, the number of models of each node at the value root.
    counts: Option<Vec<u128>>,
    /// Every value met, once.
    vals: Vec<M::Value>,
    val_ids: FxHashMap<M::Value, u32>,
    /// Per level of the chain, and last the value root, the fold of every
    /// model of each node of `values` there.
    totals: Vec<Vec<u32>>,
    /// The levels of the path, then of the chain.
    levels: Vec<Level>,
    /// Per level of the path, the products on its key's side.
    products: Vec<KeySide>,
    /// Per vtree node on a key's side, its place in that side's `order`.
    slot: Vec<u32>,
    /// The folds reached below the chain, and each one's place there.
    reached: Vec<u32>,
    reached_at: FxHashMap<u32, u32>,
}

impl<'a, M: FoldMonoid> Fold<'a, M> {
    /// The id of `v`.
    fn intern(&mut self, v: M::Value) -> Result<u32, OperationError> {
        if let Some(&id) = self.val_ids.get(&v) {
            return Ok(id);
        }
        let id = self.vals.len() as u32;
        self.lim.try_push(&mut self.vals, v.clone())?;
        self.lim.reserve_map(&mut self.val_ids, 1)?;
        self.val_ids.insert(v, id);
        Ok(id)
    }

    /// `a ⊕ b` on ids, `NONE` the empty fold.
    fn plus(&mut self, a: u32, b: u32) -> Result<u32, OperationError> {
        match (a, b) {
            (NONE, x) | (x, NONE) => Ok(x),
            (a, b) => {
                self.gate.poll(1)?;
                let v = self.monoid.combine(&self.vals[a as usize], &self.vals[b as usize]);
                self.intern(v)
            }
        }
    }

    /// The fold of the nodes `set` of `values`, from `totals[k]`.
    fn fold_of(&mut self, k: usize, set: &[u32]) -> Result<u32, OperationError> {
        let mut acc = NONE;
        for &n in set {
            acc = self.plus(acc, self.totals[k][n as usize])?;
        }
        Ok(acc)
    }

    /// `totals`: at the value root each node's models, read off its
    /// subtree (under a count, how many it holds); up the chain each node's
    /// pairs, a ⊤ literal's twice.
    fn total(&mut self) -> Result<(), PrefixFoldError> {
        let values: &'a Tdd = self.ops[1];
        let vtree = values.vtree();
        let root = self.layout.value_root;
        let m = self.layout.chain.len();
        self.lim.try_resize(&mut self.totals, m + 1, Vec::new())?;
        let width = match vtree.node(root).is_leaf() {
            true => LEAF_WIDTH,
            false => values.level(root).slot_count(),
        };
        let mut memo: FxHashMap<(u32, u32), std::rc::Rc<Vec<u64>>> = FxHashMap::default();
        let mut at_root = Vec::new();
        self.lim.reserve(&mut at_root, width)?;
        if let Some(counts) = self.counts.take() {
            for n in 0..width {
                self.gate.poll(1)?;
                let Some(v) = counts.get(n).and_then(|&c| self.monoid.count(c)) else {
                    return Err(PrefixFoldError::Codes("the monoid counts one model but not every number".to_string()));
                };
                let id = self.intern(v)?;
                at_root.push(id);
            }
        }
        for n in (at_root.len() as u32)..width as u32 {
            let codes = self.models(root, n, &mut memo)?;
            let mut acc: Option<M::Value> = None;
            for &c in codes.iter() {
                self.gate.poll(1)?;
                let v = self.monoid.value(c);
                acc = Some(match acc {
                    None => v,
                    Some(a) => self.monoid.combine(&a, &v),
                });
            }
            let id = self.intern(acc.expect("every node has a model"))?;
            at_root.push(id);
        }
        drop(memo);
        self.totals[m] = at_root;
        for j in (0..m).rev() {
            let d = self.layout.chain[j];
            let level = values.level(d);
            let mut here = Vec::new();
            self.lim.reserve(&mut here, level.slot_count())?;
            for n in 0..level.slot_count() {
                let mut acc = NONE;
                for pair in level.pairs_iter_of_idx(n) {
                    self.gate.poll(1)?;
                    let below = self.totals[j + 1][pair.right.raw() as usize];
                    acc = self.plus(acc, below)?;
                    if pair.left.raw() == ONE_LEAF_IDX.0 {
                        acc = self.plus(acc, below)?;
                    }
                }
                here.push(acc);
            }
            self.totals[j] = here;
        }
        Ok(())
    }

    /// The value codes of node `n` of `values`'s level `t`, a node of the
    /// value subtree, one per model.
    fn models(
        &mut self,
        t: VtreeIdx,
        n: u32,
        memo: &mut FxHashMap<(u32, u32), std::rc::Rc<Vec<u64>>>,
    ) -> Result<std::rc::Rc<Vec<u64>>, OperationError> {
        let values: &'a Tdd = self.ops[1];
        let vtree = values.vtree();
        if vtree.node(t).is_leaf() {
            let bit = 1u64 << self.layout.value_bit[t.idx()];
            return Ok(std::rc::Rc::new(match n {
                x if x == POS_LEAF_IDX.0 => vec![bit],
                x if x == NEG_LEAF_IDX.0 => vec![0],
                _ => vec![0, bit],
            }));
        }
        if let Some(found) = memo.get(&(t.0, n)) {
            return Ok(found.clone());
        }
        let (l, r) = vtree.children(t);
        let mut out = Vec::new();
        for pair in values.level(t).pairs_iter_of_idx(n as usize) {
            let left = self.models(l, pair.left.raw(), memo)?;
            let right = self.models(r, pair.right.raw(), memo)?;
            self.gate.poll((left.len() * right.len()) as u64)?;
            self.lim.reserve(&mut out, left.len() * right.len())?;
            for &x in left.iter() {
                out.extend(right.iter().map(|&y| x | y));
            }
        }
        let out = std::rc::Rc::new(out);
        self.lim.reserve_map(memo, 1)?;
        memo.insert((t.0, n), out.clone());
        Ok(out)
    }

    /// The walk from the root down the path and the chain: each level's
    /// states and their moves.
    fn descend(&mut self) -> Result<(), OperationError> {
        let (r, m) = (self.layout.path.len(), self.layout.chain.len());
        self.lim.reserve(&mut self.levels, r + m)?;
        self.levels.extend((0..r + m).map(|_| Level::default()));
        self.lim.reserve(&mut self.products, r)?;
        self.products.extend((0..r).map(|_| KeySide::default()));
        self.lim.try_resize(&mut self.slot, self.ops[0].vtree().num_nodes(), NONE)?;
        let start = [self.ops[0].output().local.0, self.ops[1].output().local.0];
        let top = &mut self.levels[0];
        let a = top.sets[0].intern(self.lim, &start[..1])?;
        let b = top.sets[1].intern(self.lim, &start[1..])?;
        top.add(self.lim, [a, b, NONE])?;
        for at in 0..r {
            self.path_level(at)?;
        }
        for j in 0..m {
            self.chain_level(r + j, j)?;
        }
        Ok(())
    }

    /// The moves of path level `at`: per key on its key's side, the nodes
    /// each operand holds on the path side.
    fn path_level(&mut self, at: usize) -> Result<(), OperationError> {
        let (u, x, right) = self.layout.path[at];
        let ops = self.ops;
        let vtree = ops[0].vtree();
        let leaf_key = vtree.node(x).is_leaf();
        let lim = self.lim;
        // Per state and operand, its groups: a key and the set it leads to.
        let mut groups: Vec<[u32; 2]> = Vec::new();
        let mut spans: Vec<[u32; 4]> = Vec::new();
        let mut scratch: Vec<[u32; 2]> = Vec::new();
        let mut buf: Vec<u32> = Vec::new();
        {
            let (here, below) = self.levels.split_at_mut(at + 1);
            let (here, below) = (&here[at], &mut below[0]);
            lim.reserve(&mut spans, here.states.len())?;
            for state in &here.states {
                let mut span = [0u32; 4];
                for side in 0..2 {
                    let start = groups.len();
                    let nodes = here.sets[side].get(state[side]);
                    scratch.clear();
                    for &n in nodes {
                        for pair in ops[side].level(u).pairs_iter_of_idx(n as usize) {
                            self.gate.poll(1)?;
                            let (key, next) = match right {
                                true => (pair.left.raw(), pair.right.raw()),
                                false => (pair.right.raw(), pair.left.raw()),
                            };
                            // A ⊤ leaf is both of its literals.
                            if leaf_key && key == ONE_LEAF_IDX.0 {
                                lim.try_push(&mut scratch, [POS_LEAF_IDX.0, next])?;
                                lim.try_push(&mut scratch, [NEG_LEAF_IDX.0, next])?;
                            } else {
                                lim.try_push(&mut scratch, [key, next])?;
                            }
                        }
                    }
                    scratch.sort_unstable();
                    scratch.dedup();
                    let mut i = 0;
                    while i < scratch.len() {
                        let key = scratch[i][0];
                        buf.clear();
                        while i < scratch.len() && scratch[i][0] == key {
                            lim.try_push(&mut buf, scratch[i][1])?;
                            i += 1;
                        }
                        let set = below.sets[side].intern(lim, &buf)?;
                        lim.try_push(&mut groups, [key, set])?;
                    }
                    span[2 * side] = start as u32;
                    span[2 * side + 1] = (groups.len() - start) as u32;
                }
                spans.push(span);
            }
        }
        let range = |span: &[u32; 4], side: usize| {
            let start = span[2 * side] as usize;
            start..start + span[2 * side + 1] as usize
        };
        // The products on an internal key's side.
        let top = match leaf_key {
            true => NONE,
            false => {
                let mut roots: [Vec<u32>; 2] = Default::default();
                for span in &spans {
                    for (side, roots) in roots.iter_mut().enumerate() {
                        lim.reserve(roots, span[2 * side + 1] as usize)?;
                        roots.extend(groups[range(span, side)].iter().map(|g| g[0]));
                    }
                }
                self.conjoin(at, x, roots)?;
                self.slot[x.idx()]
            }
        };
        let (here, below) = self.levels.split_at_mut(at + 1);
        let (here, below) = (&mut here[at], &mut below[0]);
        let mut used: Vec<u32> = Vec::new();
        lim.reserve(&mut here.ranges, spans.len())?;
        for span in &spans {
            let start = here.moves.len();
            let gbs = &groups[range(span, 1)];
            for ga in &groups[range(span, 0)] {
                // The groups of `values` whose key meets `ga`'s, each with
                // its label: on a leaf the literal both hold, else their
                // product, from the products of `ga`'s key or from the
                // groups, whichever are fewer.
                if leaf_key {
                    if let Ok(i) = gbs.binary_search_by_key(&ga[0], |g| g[0]) {
                        self.gate.poll(1)?;
                        let next = below.add(lim, [ga[1], gbs[i][1], NONE])?;
                        lim.try_push(&mut here.moves, [ga[0], next])?;
                    }
                    continue;
                }
                let side = &self.products[at].at[top as usize];
                let run = side.run(ga[0]);
                if run.len() <= gbs.len() {
                    for p in run {
                        self.gate.poll(1)?;
                        if let Ok(i) = gbs.binary_search_by_key(&side.keys[p][1], |g| g[0]) {
                            let next = below.add(lim, [ga[1], gbs[i][1], NONE])?;
                            lim.try_push(&mut here.moves, [p as u32, next])?;
                            lim.try_push(&mut used, p as u32)?;
                        }
                    }
                } else {
                    for gb in gbs {
                        self.gate.poll(1)?;
                        if let Some(&p) = side.index.get(&[ga[0], gb[0]]) {
                            let next = below.add(lim, [ga[1], gb[1], NONE])?;
                            lim.try_push(&mut here.moves, [p, next])?;
                            lim.try_push(&mut used, p)?;
                        }
                    }
                }
            }
            here.ranges.push([start as u32, (here.moves.len() - start) as u32]);
        }
        if !leaf_key {
            self.mark_used(at, top as usize, &used)?;
        }
        Ok(())
    }

    /// The products of path level `at`'s key side `x`: per internal node of
    /// its subtree, children first, every satisfiable pair of a node each
    /// operand reaches from `roots` (its nodes at `x`), with its pairs below.
    ///
    /// A product `(a, b)` at a node is found from a pair `(l, r)` of `a`:
    /// each product `(l, l')` below on the left whose `l'` begins a pair
    /// `(l', r')` of `values` there, the pair `b`'s, and `(r, r')` a product
    /// on the right. Per `(l, l')` it reads the pairs that begin with `l'`
    /// or the products of `r`, whichever are fewer, so no pair of nodes is
    /// tried that no satisfiable conjunction holds.
    fn conjoin(&mut self, at: usize, x: VtreeIdx, mut roots: [Vec<u32>; 2]) -> Result<(), OperationError> {
        let (lim, ops) = (self.lim, self.ops);
        let vtree = ops[0].vtree();
        let side = &mut self.products[at];
        // The internal nodes under `x`, children before parents.
        let mut stack = Vec::new();
        lim.try_push(&mut stack, (x, false))?;
        while let Some((t, done)) = stack.pop() {
            if vtree.node(t).is_leaf() {
                continue;
            }
            if done {
                self.slot[t.idx()] = side.order.len() as u32;
                lim.try_push(&mut side.order, t)?;
                continue;
            }
            let (l, r) = vtree.children(t);
            lim.try_push(&mut stack, (t, true))?;
            lim.try_push(&mut stack, (r, false))?;
            lim.try_push(&mut stack, (l, false))?;
        }
        let n = side.order.len();
        lim.reserve(&mut side.at, n)?;
        side.at.extend((0..n).map(|_| Products::default()));
        let slot = &self.slot;
        let child = |c: VtreeIdx| match vtree.node(c).is_leaf() {
            true => None,
            false => Some(slot[c.idx()] as usize),
        };
        // Top-down: the nodes each operand reaches, per internal node.
        let mut reach: [Vec<Vec<u32>>; 2] = Default::default();
        for (side, reach) in reach.iter_mut().enumerate() {
            lim.try_resize(reach, n, Vec::new())?;
            reach[n - 1] = std::mem::take(&mut roots[side]);
        }
        for k in (0..n).rev() {
            let (cl, cr) = {
                let (tl, tr) = vtree.children(side.order[k]);
                (child(tl), child(tr))
            };
            for (s, reach) in reach.iter_mut().enumerate() {
                let mut nodes = std::mem::take(&mut reach[k]);
                nodes.sort_unstable();
                nodes.dedup();
                for &node in &nodes {
                    for pair in ops[s].level(side.order[k]).pairs_iter_of_idx(node as usize) {
                        self.gate.poll(1)?;
                        if let Some(c) = cl {
                            lim.try_push(&mut reach[c], pair.left.raw())?;
                        }
                        if let Some(c) = cr {
                            lim.try_push(&mut reach[c], pair.right.raw())?;
                        }
                    }
                }
                reach[k] = nodes;
            }
        }
        // Bottom-up: the products.
        let mut b_pairs: Vec<[u32; 3]> = Vec::new();
        let mut found: Vec<[u32; 3]> = Vec::new();
        let mut lefts: Vec<[u32; 2]> = Vec::new();
        for (k, &t) in side.order.iter().enumerate() {
            let (tl, tr) = vtree.children(t);
            let (cl, cr) = (child(tl), child(tr));
            let (below, here) = side.at.split_at_mut(k);
            let here = &mut here[0];
            // The pairs of `values` there, by their left child.
            b_pairs.clear();
            for &nb in &reach[1][k] {
                for pair in ops[1].level(t).pairs_iter_of_idx(nb as usize) {
                    self.gate.poll(1)?;
                    lim.try_push(&mut b_pairs, [pair.left.raw(), pair.right.raw(), nb])?;
                }
            }
            b_pairs.sort_unstable();
            for &na in &reach[0][k] {
                found.clear();
                for pair in ops[0].level(t).pairs_iter_of_idx(na as usize) {
                    let (l, r) = (pair.left.raw(), pair.right.raw());
                    // The products on the left, each with its other node.
                    lefts.clear();
                    match cl {
                        None => {
                            for lit in [ONE_LEAF_IDX.0, POS_LEAF_IDX.0, NEG_LEAF_IDX.0] {
                                if CONJOIN_GRID[l as usize][lit as usize] != NO_PRODUCT {
                                    lim.try_push(&mut lefts, [lit, CONJOIN_GRID[l as usize][lit as usize]])?;
                                }
                            }
                        }
                        Some(c) => {
                            for p in below[c].run(l) {
                                lim.try_push(&mut lefts, [below[c].keys[p][1], p as u32])?;
                            }
                        }
                    }
                    let rights = match cr {
                        None => 3,
                        Some(c) => below[c].run(r).len(),
                    };
                    if rights == 0 {
                        continue;
                    }
                    for &[lb, pl] in &lefts {
                        self.gate.poll(1)?;
                        let first = b_pairs.partition_point(|q| q[0] < lb);
                        let width = b_pairs[first..].partition_point(|q| q[0] == lb);
                        if width == 0 {
                            continue;
                        }
                        if width <= rights {
                            for &[_, rb, nb] in &b_pairs[first..first + width] {
                                self.gate.poll(1)?;
                                let pr = match cr {
                                    None => match CONJOIN_GRID[r as usize][rb as usize] {
                                        NO_PRODUCT => continue,
                                        label => label,
                                    },
                                    Some(c) => match below[c].index.get(&[r, rb]) {
                                        Some(&p) => p,
                                        None => continue,
                                    },
                                };
                                lim.try_push(&mut found, [nb, pl, pr])?;
                            }
                        } else {
                            let run = &b_pairs[first..first + width];
                            let mut meet = |rb: u32, pr: u32| -> Result<(), OperationError> {
                                self.gate.poll(1)?;
                                // A pair of `values` belongs to one node at most.
                                if let Ok(i) = run.binary_search_by_key(&rb, |q| q[1]) {
                                    lim.try_push(&mut found, [run[i][2], pl, pr])?;
                                }
                                Ok(())
                            };
                            match cr {
                                None => {
                                    for rb in [ONE_LEAF_IDX.0, POS_LEAF_IDX.0, NEG_LEAF_IDX.0] {
                                        if CONJOIN_GRID[r as usize][rb as usize] != NO_PRODUCT {
                                            meet(rb, CONJOIN_GRID[r as usize][rb as usize])?;
                                        }
                                    }
                                }
                                Some(c) => {
                                    for p in below[c].run(r) {
                                        meet(below[c].keys[p][1], p as u32)?;
                                    }
                                }
                            }
                        }
                    }
                }
                if found.is_empty() {
                    continue;
                }
                found.sort_unstable();
                found.dedup();
                let first = here.keys.len() as u32;
                let mut i = 0;
                while i < found.len() {
                    let nb = found[i][0];
                    let start = here.pairs.len();
                    while i < found.len() && found[i][0] == nb {
                        lim.try_push(&mut here.pairs, [found[i][1], found[i][2]])?;
                        i += 1;
                    }
                    let id = here.keys.len() as u32;
                    lim.try_push(&mut here.keys, [na, nb])?;
                    lim.try_push(&mut here.ranges, [start as u32, (here.pairs.len() - start) as u32])?;
                    lim.reserve_map(&mut here.index, 1)?;
                    here.index.insert([na, nb], id);
                }
                lim.reserve_map(&mut here.of_a, 1)?;
                here.of_a.insert(na, [first, here.keys.len() as u32 - first]);
            }
            let products = here.keys.len();
            lim.try_resize(&mut here.used, products, false)?;
        }
        Ok(())
    }

    /// Marks the products `used` at slot `top` of path level `at`'s key
    /// side, and those below them, for the result.
    fn mark_used(&mut self, at: usize, top: usize, used: &[u32]) -> Result<(), OperationError> {
        let vtree = self.ops[0].vtree();
        let side = &mut self.products[at];
        for &p in used {
            side.at[top].used[p as usize] = true;
        }
        for k in (0..=top).rev() {
            let (tl, tr) = vtree.children(side.order[k]);
            let child = |c: VtreeIdx| match vtree.node(c).is_leaf() {
                true => None,
                false => Some(self.slot[c.idx()] as usize),
            };
            let (cl, cr) = (child(tl), child(tr));
            let (below, here) = side.at.split_at_mut(k);
            let here = &here[0];
            for p in 0..here.keys.len() {
                if !here.used[p] {
                    continue;
                }
                self.gate.poll(1)?;
                for &[l, r] in here.pairs_of(p) {
                    if let Some(c) = cl {
                        below[c].used[l as usize] = true;
                    }
                    if let Some(c) = cr {
                        below[c].used[r as usize] = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// The moves of chain level `at`, the chain's `j`-th node: the points
    /// with `order[j]` false keep the fold; those with it true add the
    /// fold of the models with it false. Below the last node, the fold the
    /// point keeps: with the models at its own order unless `strict`.
    fn chain_level(&mut self, at: usize, j: usize) -> Result<(), OperationError> {
        let d = self.layout.chain[j];
        let last = j + 1 == self.layout.chain.len();
        let (lim, ops) = (self.lim, self.ops);
        let mut halves: [[Vec<u32>; 2]; 2] = Default::default();
        let n = self.levels[at].states.len();
        lim.reserve(&mut self.levels[at].ranges, n)?;
        for s in 0..n {
            let state = self.levels[at].states[s];
            // Per operand, the nodes below with `order[j]` false, then true.
            for side in 0..2 {
                let [zero, one] = &mut halves[side];
                zero.clear();
                one.clear();
                let level = ops[side].level(d);
                for &node in self.levels[at].sets[side].get(state[side]) {
                    for pair in level.pairs_iter_of_idx(node as usize) {
                        self.gate.poll(1)?;
                        let (lit, next) = (pair.left.raw(), pair.right.raw());
                        if lit != POS_LEAF_IDX.0 {
                            lim.try_push(zero, next)?;
                        }
                        if lit != NEG_LEAF_IDX.0 {
                            lim.try_push(one, next)?;
                        }
                    }
                }
                for half in [zero, one] {
                    half.sort_unstable();
                    half.dedup();
                }
            }
            let start = self.levels[at].moves.len();
            let descending = self.range.descending();
            for (bit, lit) in [(0, NEG_LEAF_IDX.0), (1, POS_LEAF_IDX.0)] {
                if halves[0][bit].is_empty() {
                    continue;
                }
                // The values this bit passes: those whose order is below the
                // point's from here (a 0 where the point has a 1), or above
                // it from the top down (a 1 where the point has a 0).
                let passed = match (bit, descending) {
                    (1, false) | (0, true) => {
                        let past = self.fold_of(j + 1, &halves[1][1 - bit])?;
                        self.plus(state[2], past)?
                    }
                    _ => state[2],
                };
                self.gate.poll(1)?;
                let next = match last {
                    true => {
                        let fold = match self.range.strict() {
                            true => passed,
                            false => {
                                let here = self.fold_of(j + 1, &halves[1][bit])?;
                                self.plus(passed, here)?
                            }
                        };
                        if fold == NONE {
                            continue;
                        }
                        match self.reached_at.get(&fold) {
                            Some(&i) => i,
                            None => {
                                let i = self.reached.len() as u32;
                                lim.try_push(&mut self.reached, fold)?;
                                lim.reserve_map(&mut self.reached_at, 1)?;
                                self.reached_at.insert(fold, i);
                                i
                            }
                        }
                    }
                    // No model left below and none passed: no fold.
                    false if halves[1][bit].is_empty() && passed == NONE => continue,
                    false => {
                        let below = &mut self.levels[at + 1];
                        let a = below.sets[0].intern(lim, &halves[0][bit])?;
                        let b = below.sets[1].intern(lim, &halves[1][bit])?;
                        below.add(lim, [a, b, passed])?
                    }
                };
                lim.try_push(&mut self.levels[at].moves, [lit, next])?;
            }
            let here = &mut self.levels[at];
            here.ranges.push([start as u32, (here.moves.len() - start) as u32]);
        }
        Ok(())
    }

    /// The result on `into`: a cube per code, the products on the keys'
    /// sides, then from the chain's bottom up one node per distinct pair a
    /// state's moves lead to, the root holding the top state's pairs;
    /// minimized.
    fn emit(mut self, eng: &Engine, into: &Arc<Vtree>, coding: &Coding, codes: &[u64]) -> Result<Tdd, OperationError> {
        let (lim, layout, ops) = (self.lim, self.layout, self.ops);
        let vtree = ops[0].vtree();
        let mut assembly = Assembly::new(eng, into)?;
        let code_root = coding.image[layout.value_root.idx()];
        let mut cubes = Vec::new();
        lim.reserve(&mut cubes, codes.len())?;
        for &code in codes {
            let node = cube(eng, &mut assembly, &mut self.gate, into, code_root, code, &coding.code_bit)?;
            cubes.push(node);
        }
        // The products, children first.
        for side in &mut self.products {
            for k in 0..side.order.len() {
                let t = side.order[k];
                let (tl, tr) = vtree.children(t);
                let child = |c: VtreeIdx| match vtree.node(c).is_leaf() {
                    true => None,
                    false => Some(self.slot[c.idx()] as usize),
                };
                let (cl, cr) = (child(tl), child(tr));
                let n = side.at[k].keys.len();
                let mut out = Vec::new();
                lim.reserve(&mut out, n)?;
                let mut pairs = Vec::new();
                for p in 0..n {
                    if !side.at[k].used[p] {
                        out.push(NONE);
                        continue;
                    }
                    pairs.clear();
                    for &[l, r] in side.at[k].pairs_of(p) {
                        let l = cl.map_or(l, |c| side.at[c].out[l as usize]);
                        let r = cr.map_or(r, |c| side.at[c].out[r as usize]);
                        lim.try_push(&mut pairs, ChildPair::new(NodeIdx(l), NodeIdx(r)))?;
                    }
                    self.gate.poll(pairs.len() as u64)?;
                    out.push(assembly.push(eng, coding.image[t.idx()], &pairs)?.0);
                }
                side.at[k].out = out;
            }
        }
        // The levels, bottom-up: per state, the nodes its moves lead to.
        let (r, m) = (layout.path.len(), layout.chain.len());
        let mut root = Vec::new();
        let mut below: (Vec<u32>, Vec<[u32; 2]>) = (Vec::new(), Vec::new());
        for at in (0..r + m).rev() {
            let (t, label_side, right) = match at.checked_sub(r) {
                Some(j) => (layout.chain[j], None, true),
                None => {
                    let (u, x, right) = layout.path[at];
                    let side = match vtree.node(x).is_leaf() {
                        true => None,
                        false => Some(self.slot[x.idx()] as usize),
                    };
                    (u, Some(side), right)
                }
            };
            let bottom = at + 1 == r + m;
            let level = &self.levels[at];
            let mut index: FxHashMap<[u32; 2], u32> = FxHashMap::default();
            let mut items = Vec::new();
            let mut ranges = Vec::new();
            lim.reserve(&mut ranges, level.states.len())?;
            for s in 0..level.states.len() {
                let start = items.len();
                for &[label, next] in level.moves_of(s) {
                    let label = match label_side {
                        Some(Some(k)) => self.products[at].at[k].out[label as usize],
                        _ => label,
                    };
                    let children: &[u32] = match bottom {
                        true => std::slice::from_ref(&cubes[next as usize]),
                        false => {
                            let [start, len] = below.1[next as usize];
                            &below.0[start as usize..(start + len) as usize]
                        }
                    };
                    for &c in children {
                        self.gate.poll(1)?;
                        let pair = match right {
                            true => [label, c],
                            false => [c, label],
                        };
                        let pair_of = ChildPair::new(NodeIdx(pair[0]), NodeIdx(pair[1]));
                        if at == 0 {
                            lim.try_push(&mut root, pair_of)?;
                            continue;
                        }
                        let node = match index.get(&pair) {
                            Some(&node) => node,
                            None => {
                                let node = assembly.push(eng, coding.image[t.idx()], &[pair_of])?.0;
                                lim.reserve_map(&mut index, 1)?;
                                index.insert(pair, node);
                                node
                            }
                        };
                        lim.try_push(&mut items, node)?;
                    }
                }
                ranges.push([start as u32, (items.len() - start) as u32]);
            }
            below = (items, ranges);
        }
        if root.is_empty() {
            return Ok(crate::build::constant_zero(eng, into));
        }
        let top = assembly.push(eng, into.root(), &root)?;
        self.gate.flush()?;
        let mut result = assembly.finish(TddNodeId { vtree: into.root(), local: top })?;
        eng.minimize(&mut result)?;
        Ok(result)
    }
}

/// The cube of `code` on the subtree of `into` at `t`, whose leaves read
/// the bits `code_bit` names: a literal at a leaf, else one node of one
/// pair, shared with an equal one already built.
fn cube(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    gate: &mut PollGate<'_>,
    into: &Vtree,
    t: VtreeIdx,
    code: u64,
    code_bit: &[u8],
) -> Result<u32, OperationError> {
    if into.node(t).is_leaf() {
        return Ok(match code >> code_bit[t.idx()] & 1 {
            1 => POS_LEAF_IDX.0,
            _ => NEG_LEAF_IDX.0,
        });
    }
    let (l, r) = into.children(t);
    let l = cube(eng, assembly, gate, into, l, code, code_bit)?;
    let r = cube(eng, assembly, gate, into, r, code, code_bit)?;
    gate.poll(1)?;
    Ok(assembly.intern(eng, t, &[ChildPair::new(NodeIdx(l), NodeIdx(r))])?.0)
}

#[cfg(test)]
#[path = "tests/prefix_fold.rs"]
mod tests;
