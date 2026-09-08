//! Leaf-level apply constants: `CONJOIN_GRID` for leaf product lookup.
//!
//! With implicit leaf representation, leaf levels have no stored nodes — the
//! index IS the label (0=One, 1=Pos, 2=Neg). The `CONJOIN_GRID` constant
//! gives the leaf product as a static 3×3 lookup table used in `apply_and`'s
//! inner loop.

/// Sentinel for dead (unsatisfiable) leaf product cells.
const DEAD: u32 = u32::MAX;

/// Static 3×3 conjunction grid for implicit leaf product.
///
/// `CONJOIN_GRID[i][j]` = output label index when conjoining leaf label `i`
/// with leaf label `j`, or `DEAD` (`u32::MAX`) if the conjunction is Zero.
///
/// ```text
///        j=One(0)  j=Pos(1)  j=Neg(2)
/// i=One:    0         1        2
/// i=Pos:    1         1       DEAD
/// i=Neg:    2        DEAD      2
/// ```

pub(crate) const CONJOIN_GRID: [[u32; 3]; 3] = [
    [0,    1,    2   ],  // One ∧ {One, Pos, Neg}
    [1,    1,    DEAD],  // Pos ∧ {One, Pos, Neg}
    [2,    DEAD, 2   ],  // Neg ∧ {One, Pos, Neg}
];

