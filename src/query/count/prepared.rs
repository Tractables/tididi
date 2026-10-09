//! Immutable compact reads for a retained counter.

use crate::{Engine, Tdd, OperationError};
use crate::diagram::{ChildPair, EncodedChildRef};
use crate::limits::{Charged, PollGate, Transient};
use crate::vtree::VtreeIdx;
use crate::value::{Count, IntFold};
use super::column::CountColumn;
use crate::query::cache::PinState;

// Below 32 KiB of native words, decoding can cost more than the smaller reads save.
const MIN_WORDS: usize = 4096;
const TRACKED: usize = 8;

fn bits(max: u64) -> u32 { (64 - max.leading_zeros()).max(1) }
fn low(n: u32) -> u64 { u64::MAX >> (64 - n) }
fn word_bits(n: u32) -> u32 { if n <= 16 { 16 } else if n <= 32 { 32 } else { 64 } }
fn translated(map: &[u32], slot: u32) -> u32 { if map.is_empty() { slot } else { map[slot as usize] } }

#[derive(Default)]
pub(super) struct Prepared {
    pub(super) levels: Vec<Option<Packed>>,
    pub(super) ready: bool,
    pub(super) output: Option<usize>,
}

impl Charged for Prepared {
    fn charged_bytes(&self) -> u64 {
        self.levels.charged_bytes() + self.levels.iter().flatten().map(Charged::charged_bytes).sum::<u64>()
    }
}

/// A low tag distinguishes an inline pair from an arena range. The remaining
/// payload holds either two child slots or a start and length; spare high bits
/// hold two implication polarities per selected variable. Word widths are
/// chosen before adding implications, so a summary never widens a node.
#[derive(Clone, Copy)]
struct Layout {
    left: u32,
    start: u32,
    payload: u32,
    node_word: u32,
    pair_word: u32,
    pairs: usize,
}

impl Layout {
    fn read(tdd: &Tdd, t: VtreeIdx, gate: &mut PollGate) -> Result<Option<Self>, OperationError> {
        if tdd.vtree.node(t).is_leaf() || tdd.levels[t.idx()].is_marginal() { return Ok(None); }
        let (l, r) = tdd.vtree.children(t);
        if tdd.levels[l.idx()].is_marginal() || tdd.levels[r.idx()].is_marginal() { return Ok(None); }
        let Some(stored) = tdd.levels[t.idx()].stored() else { return Ok(None); };
        let (mut pairs, mut max_len) = (0usize, 0usize);
        for node in stored.nodes() {
            let inputs = stored.of(node);
            gate.poll(inputs.len() as u64 + 1)?;
            if inputs.len() != 1 {
                let Some(n) = pairs.checked_add(inputs.len()) else { return Ok(None); };
                pairs = n;
                max_len = max_len.max(inputs.len());
            }
        }
        if stored.nodes().len().saturating_add(pairs) < MIN_WORDS { return Ok(None); }
        // Reordering preserves the child slot domain but may move a referenced
        // slot above the largest reference this particular parent used.
        let left = bits(tdd.reference_slot_count(l).saturating_sub(1) as u64);
        let right = bits(tdd.reference_slot_count(r).saturating_sub(1) as u64);
        let start = bits(pairs as u64);
        let payload = 1 + (left + right).max(start + bits(max_len as u64));
        if payload > 64 || left + right > 64 { return Ok(None); }
        let layout = Self { left, start, payload, node_word: word_bits(payload), pair_word: word_bits(left + right), pairs };
        if layout.node_word == 64 && layout.pair_word == 64 { return Ok(None); }
        Ok(Some(layout))
    }
}

enum Words { U16(Vec<u16>), U32(Vec<u32>), U64(Vec<u64>) }

impl Charged for Words {
    fn charged_bytes(&self) -> u64 {
        match self { Self::U16(v) => v.charged_bytes(), Self::U32(v) => v.charged_bytes(), Self::U64(v) => v.charged_bytes() }
    }
}

impl Words {
    fn new(eng: &Engine, width: u32, len: usize) -> Result<Self, OperationError> {
        fn buffer<T>(eng: &Engine, len: usize) -> Result<Vec<T>, OperationError> {
            let mut v = Transient::new(eng.limits(), Vec::new());
            eng.limits().reserve_exact(&mut v, len)?;
            Ok(v.keep())
        }
        Ok(match width { 16 => Self::U16(buffer(eng, len)?), 32 => Self::U32(buffer(eng, len)?), _ => Self::U64(buffer(eng, len)?) })
    }

    fn push(&mut self, value: u64) {
        match self {
            Self::U16(v) => { debug_assert!(u16::try_from(value).is_ok()); v.push(value as u16); }
            Self::U32(v) => { debug_assert!(u32::try_from(value).is_ok()); v.push(value as u32); }
            Self::U64(v) => v.push(value),
        }
    }
}

pub(super) struct Packed {
    nodes: Words,
    pairs: Words,
    layout: Layout,
    vars: [VtreeIdx; TRACKED],
    tracked: usize,
}

impl Charged for Packed {
    fn charged_bytes(&self) -> u64 { self.nodes.charged_bytes() + self.pairs.charged_bytes() }
}

/// At most eight implications per node, in original slot order. A parent's
/// candidates come from its two disjoint child maps; dropping a candidate loses
/// pruning opportunities but cannot assert a new implication.
struct Summary {
    vars: [VtreeIdx; TRACKED],
    masks: Vec<u16>,
    remap: Vec<u32>,
}

impl Default for Summary {
    fn default() -> Self {
        Self { vars: [VtreeIdx(0); TRACKED], masks: Vec::new(), remap: Vec::new() }
    }
}

impl Charged for Summary {
    fn charged_bytes(&self) -> u64 { self.masks.charged_bytes() + self.remap.charged_bytes() }
}

struct Summaries(Vec<Summary>);
impl Charged for Summaries {
    fn charged_bytes(&self) -> u64 { self.0.charged_bytes() + self.0.iter().map(Charged::charged_bytes).sum::<u64>() }
}

impl Summary {
    fn mask(&self, slot: u32) -> u32 { self.masks.get(slot as usize).copied().unwrap_or(0) as u32 }
}

fn project(mask: u32, selected: &[usize]) -> u16 {
    selected.iter().enumerate().fold(0, |out, (i, &from)| out | (((mask >> (2 * from)) & 3) as u16) << (2 * i))
}

impl Prepared {
    pub(super) fn new(eng: &Engine, tdd: &Tdd) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let mut gate = lim.gate();
        let n = tdd.vtree.num_nodes();
        let mut result = Transient::new(lim, Self::default());
        let mut layouts = Transient::new(lim, Vec::new());
        lim.reserve_exact(&mut layouts, n)?;
        layouts.resize(n, None);
        for t in tdd.vtree.bottomup() {
            gate.poll(1)?;
            layouts[t.idx()] = Layout::read(tdd, t, &mut gate)?;
        }
        if layouts.iter().all(Option::is_none) {
            gate.finish()?;
            return Ok(Self { ready: true, ..Self::default() });
        }
        lim.reserve_exact(&mut result.levels, n)?;
        result.levels.resize_with(n, || None);
        let mut summaries = Transient::new(lim, Summaries(Vec::new()));
        lim.reserve_exact(&mut summaries.0, n)?;
        summaries.0.resize_with(n, Summary::default);
        for t in tdd.vtree.bottomup() {
            gate.poll(1)?;
            let ti = t.idx();
            let level = &tdd.levels[ti];
            if level.is_marginal() { continue; }
            if tdd.vtree.node(t).is_leaf() {
                let s = &mut summaries.0[ti];
                s.vars[0] = t;
                lim.reserve_exact(&mut s.masks, 3)?;
                s.masks.extend([0, 1, 2]);
                continue;
            }
            let (l, r) = tdd.vtree.children(t);
            // Arithmetic levels already avoid storing their products. Do not
            // expand them just to collect optional implications.
            if let Some(stored) = level.stored() {
                let left = &summaries.0[l.idx()];
                let right = &summaries.0[r.idx()];
                let mut masks = Transient::new(lim, Vec::<u32>::new());
                lim.reserve_exact(&mut masks, stored.nodes().len())?;
                let mut frequency = [0u64; 2 * TRACKED];
                for node in stored.nodes() {
                    let pairs = stored.of(node);
                    gate.poll(pairs.len() as u64 + 1)?;
                    let mut mask = if pairs.is_empty() { 0 } else { u32::MAX };
                    for pair in pairs {
                        mask &= left.mask(pair.left.raw()) | (right.mask(pair.right.raw()) << 16);
                    }
                    masks.push(mask);
                    let mut implied = (mask | (mask >> 1)) & 0x5555_5555;
                    while implied != 0 {
                        let i = (implied.trailing_zeros() / 2) as usize;
                        frequency[i] = frequency[i].saturating_add(pairs.len() as u64);
                        implied &= implied - 1;
                    }
                }
                let mut candidates = std::array::from_fn::<_, { 2 * TRACKED }, _>(|i| i);
                candidates.sort_unstable_by_key(|&i| (std::cmp::Reverse(frequency[i]), i));
                let tracked = candidates.iter().take(TRACKED).take_while(|&&i| frequency[i] != 0).count();
                let selected = &candidates[..tracked];
                let mut summary = Transient::new(lim, Summary::default());
                for (i, &from) in selected.iter().enumerate() {
                    summary.vars[i] = if from < TRACKED { left.vars[from] } else { right.vars[from - TRACKED] };
                }
                lim.reserve_exact(&mut summary.masks, masks.len())?;
                summary.masks.extend(masks.iter().map(|&mask| project(mask, selected)));
                if let Some(layout) = layouts[ti] {
                    let header_tracked = tracked.min(((layout.node_word - layout.payload) / 2) as usize);
                    let mut order = Transient::new(lim, Vec::<u32>::new());
                    let reorder = header_tracked != 0 && tdd.vtree.node(t).parent().is_none_or(|parent| layouts[parent.idx()].is_some());
                    if reorder {
                        lim.reserve_exact(&mut order, masks.len())?;
                        order.extend(0..u32::try_from(masks.len()).map_err(|_| OperationError::IndexOverflow)?);
                        let mask = (1u32 << (2 * header_tracked)) - 1;
                        order.sort_unstable_by_key(|&i| (u32::from(summary.masks[i as usize]) & mask, i));
                        lim.try_resize(&mut summary.remap, masks.len(), 0)?;
                        for (new, &old) in order.iter().enumerate() { summary.remap[old as usize] = new as u32; }
                    }
                    let nodes = Transient::new(lim, Words::new(eng, layout.node_word, masks.len())?);
                    let pairs = Words::new(eng, layout.pair_word, layout.pairs)?;
                    let mut packed = Transient::new(lim, Packed { nodes: nodes.keep(), pairs, layout, vars: summary.vars, tracked: header_tracked });
                    let mut at = 0usize;
                    for new in 0..masks.len() {
                        let old = if order.is_empty() { new } else { order[new] as usize };
                        let pairs = stored.of_idx(old);
                        gate.poll(pairs.len() as u64 + 1)?;
                        let encode = |p: ChildPair| u64::from(translated(&left.remap, p.left.raw()))
                            | (u64::from(translated(&right.remap, p.right.raw())) << layout.left);
                        let raw = if pairs.len() == 1 { encode(pairs[0]) << 1 } else {
                            let start = at;
                            for &pair in pairs { packed.pairs.push(encode(pair)); }
                            at += pairs.len();
                            1 | (((start as u64) | ((pairs.len() as u64) << layout.start)) << 1)
                        };
                        let mask = u64::from(summary.masks[old]) & ((1u64 << (2 * header_tracked)) - 1);
                        packed.nodes.push(if header_tracked == 0 { raw } else { raw | (mask << layout.payload) });
                    }
                    debug_assert_eq!(at, layout.pairs);
                    if t == tdd.output.vtree && !tdd.is_zero() {
                        result.output = Some(translated(&summary.remap, tdd.output.local.0) as usize);
                    }
                    result.levels[ti] = Some(packed.keep());
                }
                summaries.0[ti] = summary.keep();
            }
            for child in [l, r] { lim.discard(std::mem::take(&mut summaries.0[child.idx()])); }
        }
        gate.finish()?;
        result.ready = true;
        Ok(result.keep())
    }

    pub(super) fn fold_level<C: CountColumn>(
        &self, eng: &Engine, tdd: &Tdd, cols: &mut [C], t: VtreeIdx,
        pins: &[PinState], gate: &mut PollGate,
    ) -> Result<bool, OperationError> {
        let Some(Some(packed)) = self.levels.get(t.idx()) else { return Ok(false); };
        let (l, r) = tdd.vtree.children(t);
        let [out, left, right] = cols.get_disjoint_mut([t.idx(), l.idx(), r.idx()]).expect("distinct query columns");
        let mut bad = 0u64;
        for i in 0..packed.tracked {
            bad |= match pins.get(packed.vars[i].idx()).and_then(|pin| pin.value) {
                Some(false) => 1 << (2 * i), Some(true) => 2 << (2 * i), None => 0,
            };
        }
        let bad = if bad == 0 { 0 } else { bad << packed.layout.payload };
        macro_rules! pairs { ($nodes:expr) => { match &packed.pairs {
            Words::U16(v) => packed.fold(eng, $nodes, v, left, right, out, bad, gate),
            Words::U32(v) => packed.fold(eng, $nodes, v, left, right, out, bad, gate),
            Words::U64(v) => packed.fold(eng, $nodes, v, left, right, out, bad, gate),
        } }; }
        match &packed.nodes { Words::U16(v) => pairs!(v), Words::U32(v) => pairs!(v), Words::U64(v) => pairs!(v) }?;
        Ok(true)
    }
}

/// A node's inline pair or a range in a typed packed arena.
#[derive(Clone)]
struct Inputs<'a, P> {
    inline: Option<u64>,
    pairs: std::slice::Iter<'a, P>,
    left: u32,
}

impl<P: Copy + Into<u64>> Iterator for Inputs<'_, P> {
    type Item = ChildPair;
    #[inline]
    fn next(&mut self) -> Option<ChildPair> {
        let word = self.inline.take().or_else(|| self.pairs.next().map(|&p| p.into()))?;
        Some(ChildPair::new(EncodedChildRef::from_raw((word & low(self.left)) as u32), EncodedChildRef::from_raw((word >> self.left) as u32)))
    }
}

impl Packed {
    #[allow(clippy::too_many_arguments)]
    fn fold<N: Copy + Into<u64>, P: Copy + Into<u64>, C: CountColumn>(
        &self, eng: &Engine, nodes: &[N], pairs: &[P], left: &C, right: &C,
        out: &mut C, bad: u64, gate: &mut PollGate,
    ) -> Result<(), OperationError> {
        let batch = eng.limits().reduce_poll_stride().clamp(1, 256) as usize;
        for (chunk, nodes) in nodes.chunks(batch).enumerate() {
            gate.poll(nodes.len() as u64)?;
            for (offset, &word) in nodes.iter().enumerate() {
                let raw = word.into();
                let value = if raw & bad != 0 { Count::Fast(0) } else {
                    let data = (raw & low(self.layout.payload)) >> 1;
                    let inputs = if raw & 1 == 0 {
                        Inputs { inline: Some(data), pairs: pairs[..0].iter(), left: self.layout.left }
                    } else {
                        let start = (data & low(self.layout.start)) as usize;
                        let len = (data >> self.layout.start) as usize;
                        Inputs { inline: None, pairs: pairs[start..start + len].iter(), left: self.layout.left }
                    };
                    match C::fold_structural(inputs.clone(), left, right) {
                        Some(total) => Count::from_u128(total),
                        None => IntFold::fold(inputs, |k| left.get(k.raw() as usize), |k| right.get(k.raw() as usize)),
                    }
                };
                out.set(eng, chunk * batch + offset, value)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/prepared.rs"]
mod tests;
