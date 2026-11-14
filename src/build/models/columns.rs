//! Canonical construction from a table's columns and a selection of its rows.
//!
//! [`Tdd::from_columns`] is [`Tdd::from_models`] for a caller that holds its
//! table as columns of codes, the shape [`Tdd::model_columns`] writes, and
//! wants the diagram of some of its rows: the rows are packed straight into
//! the build's own layout, so the caller never writes a table of the chosen
//! rows, nor packs one into words the build then re-encodes. Everything
//! after the packing is `from_models`'s build.

use std::borrow::Cow;
use std::sync::Arc;

use crate::diagram::{NodeIdx, Tdd};
use crate::limits::{Limits, OperationError};
use crate::query::MAX_COLUMN_BITS;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

use super::layout::Layout;
use super::rows::{sorted_distinct, words_per_row};

/// The rows of a table that [`Tdd::from_columns`] reads.
#[derive(Clone, Copy, Debug)]
pub enum RowSelection<'a> {
    /// Every row.
    All,
    /// The rows at these indices, in any order; a repeat is one model.
    Listed(&'a [u32]),
    /// The rows whose flag is set, one flag per row.
    Marked(&'a [bool]),
}

impl Tdd {
    /// The canonical diagram whose models are the chosen rows of a table held
    /// as columns, with every variable of `vtree` that no column lists free.
    ///
    /// Column `j` writes its codes onto the variables `columns[j]`, read as
    /// [`model_columns`](Self::model_columns) reads them: the first variable
    /// is the code's most significant bit, so a row sets `columns[j][i]`
    /// exactly when bit `columns[j].len() - 1 - i` of its code `codes[j][r]`
    /// is set. Bits of a code past its column's width are ignored. `rows`
    /// chooses the rows; a repeated row, or two rows with the same codes,
    /// denote one model.
    ///
    /// This is [`from_models`](Self::from_models) on the chosen rows packed
    /// with the variables `columns` lists, column after column, and returns
    /// the same diagram. The rows are packed straight into the layout the
    /// build sorts and splits in, so a table whose rows are not stored
    /// apart, or not in that layout, need not be copied first. Rows that
    /// come in leaf order (each column's variables left to right in the
    /// vtree, the columns in order, and the rows sorted on the columns) are
    /// only checked, as `from_models` checks rows already in order.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::diagram::RowSelection;
    /// use tididi::vtree::VarId;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// // A two-bit code on variables 1 and 2, a one-bit code on variable 3.
    /// let columns: [&[VarId]; 2] = [&[VarId(1), VarId(2)], &[VarId(3)]];
    /// let codes: [&[u32]; 2] = [&[3, 0, 3, 2], &[1, 0, 1, 1]];
    /// // Rows 0, 2 and 3: row 2 repeats row 0.
    /// let f = Tdd::from_columns(&vtree, &columns, &codes, RowSelection::Marked(&[true, false, true, true]))?;
    /// assert_eq!(f.model_count()?, 2u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # let g = Tdd::from_models(&vtree, &[VarId(1), VarId(2), VarId(3)], &[0b111, 0b101])?;
    /// # assert!(f.equivalent(&g)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a listed variable that is
    /// not a leaf of `vtree`, [`OperationError::DuplicateVariable`] for one
    /// listed twice, [`OperationError::ColumnTooWide`] for a column of more
    /// than [`MAX_COLUMN_BITS`] variables, [`OperationError::RaggedColumns`]
    /// when `codes` does not hold one column per entry of `columns`, of one
    /// length, or a [`RowSelection::Marked`] holds a flag count other than
    /// that length, [`OperationError::RowOutOfRange`] for a listed row past
    /// it, and otherwise as [`from_models`](Self::from_models).
    pub fn from_columns(
        vtree: &Arc<Vtree>,
        columns: &[&[VarId]],
        codes: &[&[u32]],
        rows: RowSelection<'_>,
    ) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.from_columns(vtree, columns, codes, rows))
    }
}

impl Engine {
    /// Run [`Tdd::from_columns`] using this batch's scratch and resource
    /// limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_columns`].
    pub fn from_columns(
        &self,
        vtree: &Arc<Vtree>,
        columns: &[&[VarId]],
        codes: &[&[u32]],
        rows: RowSelection<'_>,
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let len = table_len(columns, codes, rows)?;
        let mut vars = Vec::new();
        lim.reserve_exact(&mut vars, columns.iter().map(|c| c.len()).sum())?;
        for column in columns {
            vars.extend_from_slice(column);
        }
        let mut layout = self.scratch.model_layout.checkout(self);
        layout.prepare_for(lim, vtree, &vars)?;
        let chosen = match rows {
            RowSelection::All => len,
            RowSelection::Listed(listed) => listed.len(),
            RowSelection::Marked(marked) => marked.iter().filter(|&&m| m).count(),
        };
        if chosen == 0 || vars.is_empty() {
            return super::super::constant_on(self, vtree, chosen > 0);
        }
        if chosen > NodeIdx::MAX_LIVE {
            // As `from_models`: a level holds at most one node per row.
            return Err(OperationError::IndexOverflow);
        }
        let w = words_per_row(vars.len());
        let (packed, ascending) = pack(lim, &layout, columns, codes, rows, chosen, w)?;
        lim.discard(vars);
        let mut radix = crate::sort::Radix::default();
        let sorted = sorted_distinct(lim, &mut radix, layout.position.len(), &layout, packed, w, ascending)?;
        super::build_sorted(self, vtree, &layout, Cow::Owned(sorted), w, radix)
    }
}

/// The rows a table of `codes` holds, checked against `columns` and `rows`.
fn table_len(columns: &[&[VarId]], codes: &[&[u32]], rows: RowSelection<'_>) -> Result<usize, OperationError> {
    if let Some(column) = columns.iter().position(|c| c.len() > MAX_COLUMN_BITS) {
        return Err(OperationError::ColumnTooWide { column, bits: columns[column].len() });
    }
    let len = codes.first().map_or(0, |c| c.len());
    if codes.len() != columns.len() {
        return Err(OperationError::RaggedColumns { column: codes.len().min(columns.len()), len: 0, expected: len });
    }
    if let Some(column) = codes.iter().position(|c| c.len() != len) {
        return Err(OperationError::RaggedColumns { column, len: codes[column].len(), expected: len });
    }
    match rows {
        RowSelection::All => Ok(len),
        RowSelection::Marked(marked) if marked.len() != len => {
            Err(OperationError::RaggedColumns { column: codes.len(), len: marked.len(), expected: len })
        }
        RowSelection::Marked(_) => Ok(len),
        RowSelection::Listed(listed) => match listed.iter().find(|&&r| r as usize >= len) {
            Some(&row) => Err(OperationError::RowOutOfRange { row: row as usize, rows: len }),
            None => Ok(len),
        },
    }
}

/// How one column's code lands in a packed row: shifted into place where
/// its bits sit next to each other, most significant highest, within one
/// word, and otherwise spread a byte at a time through tables.
enum Placement {
    /// The code shifted left by `shift` bits into word `word`.
    Shift { word: usize, shift: u32, mask: u32 },
    /// One table of 256 rows of `w` words per byte of the code: row `v` of
    /// byte `k`'s table holds the bits a code whose byte `k` is `v` sets.
    Spread { bytes: usize, table: Vec<u64> },
}

/// The chosen rows packed in the layout's order, `w` words each, and
/// whether one-word rows ascend strictly as packed.
fn pack(
    lim: &Limits,
    layout: &Layout,
    columns: &[&[VarId]],
    codes: &[&[u32]],
    rows: RowSelection<'_>,
    chosen: usize,
    w: usize,
) -> Result<(Vec<u64>, bool), OperationError> {
    let mut placements = Vec::new();
    lim.reserve_exact(&mut placements, columns.len())?;
    let mut first = 0;
    for column in columns {
        let positions = &layout.position[first..first + column.len()];
        first += column.len();
        placements.push(place(lim, positions, w)?);
    }
    let mut packed = Vec::new();
    lim.reserve_exact(&mut packed, chosen * w)?;
    let mut gate = lim.gate();
    gate.poll((chosen * columns.len().max(1)) as u64)?;
    let ascending = if w == 1 {
        let one = |r: usize| -> u64 {
            let mut word = 0u64;
            for (placement, column) in placements.iter().zip(codes) {
                let code = column[r];
                word |= match placement {
                    Placement::Shift { shift, mask, .. } => u64::from(code & mask) << shift,
                    Placement::Spread { bytes, table } => {
                        (0..*bytes).fold(0, |acc, k| acc | table[k * 256 + ((code >> (8 * k)) & 0xff) as usize])
                    }
                };
            }
            word
        };
        let mut ascending = true;
        let mut last: Option<u64> = None;
        let mut push = |word: u64| {
            ascending &= last.is_none_or(|last| last < word);
            last = Some(word);
            packed.push(word);
        };
        match rows {
            RowSelection::All => (0..codes[0].len()).for_each(|r| push(one(r))),
            RowSelection::Listed(listed) => listed.iter().for_each(|&r| push(one(r as usize))),
            RowSelection::Marked(marked) => {
                marked.iter().enumerate().filter(|&(_, &m)| m).for_each(|(r, _)| push(one(r)));
            }
        }
        ascending
    } else {
        let mut row = vec![0u64; w];
        let mut one = |r: usize| {
            row.iter_mut().for_each(|word| *word = 0);
            for (placement, column) in placements.iter().zip(codes) {
                let code = column[r];
                match placement {
                    Placement::Shift { word, shift, mask } => row[*word] |= u64::from(code & mask) << shift,
                    Placement::Spread { bytes, table } => {
                        for k in 0..*bytes {
                            let entry = &table[(k * 256 + ((code >> (8 * k)) & 0xff) as usize) * w..][..w];
                            row.iter_mut().zip(entry).for_each(|(word, &bits)| *word |= bits);
                        }
                    }
                }
            }
            packed.extend_from_slice(&row);
        };
        match rows {
            RowSelection::All => (0..codes[0].len()).for_each(&mut one),
            RowSelection::Listed(listed) => listed.iter().for_each(|&r| one(r as usize)),
            RowSelection::Marked(marked) => {
                marked.iter().enumerate().filter(|&(_, &m)| m).for_each(|(r, _)| one(r));
            }
        }
        false
    };
    gate.flush()?;
    for placement in placements {
        if let Placement::Spread { table, .. } = placement {
            lim.discard(table);
        }
    }
    Ok((packed, ascending))
}

/// The placement of a column whose bits, most significant first, land at
/// `positions` of a `w`-word row.
fn place(lim: &Limits, positions: &[u32], w: usize) -> Result<Placement, OperationError> {
    let bits = positions.len();
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    if bits == 0 {
        return Ok(Placement::Shift { word: 0, shift: 0, mask: 0 });
    }
    let low = positions[bits - 1];
    let adjacent = positions.iter().enumerate().all(|(i, &p)| p == low + (bits - 1 - i) as u32);
    if adjacent && (low % 64) as usize + bits <= 64 {
        return Ok(Placement::Shift { word: low as usize / 64, shift: low % 64, mask });
    }
    let bytes = bits.div_ceil(8);
    let mut table = Vec::new();
    lim.try_resize(&mut table, bytes * 256 * w, 0u64)?;
    for (i, &to) in positions.iter().enumerate() {
        // Bit `bits - 1 - i` of the code: byte `bit / 8`, bit `bit % 8` in it.
        let bit = bits - 1 - i;
        let (byte, within) = (bit / 8, bit % 8);
        for value in 0..256usize {
            if (value >> within) & 1 == 1 {
                table[(byte * 256 + value) * w + to as usize / 64] |= 1u64 << (to % 64);
            }
        }
    }
    Ok(Placement::Spread { bytes, table })
}

#[cfg(test)]
#[path = "tests/columns.rs"]
mod tests;
