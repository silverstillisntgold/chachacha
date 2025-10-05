/*!
Module containing useful constants/structs and the core [`Machine`] trait.
*/

/// Size (in 8-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U8: usize = MATRIX_SIZE_U8 * DEPTH;
/// Size (in 64-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U64: usize = BUF_LEN_U8 / size_of::<u64>();
/// Columns present in a standard ChaCha matrix.
pub const COLUMNS: usize = 4;
/// Rows present in a standard ChaCha matrix.
pub const ROWS: usize = 4;
/// Size (in 8-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U8: usize = (ROWS - 1) * size_of::<Row>();
/// Size (in 32-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U32: usize = SEED_LEN_U8 / size_of::<u32>();
/// Size (in 64-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U64: usize = SEED_LEN_U8 / size_of::<u64>();
/// Size (in 8-bit integers) of a reference ChaCha matrix.
pub const MATRIX_SIZE_U8: usize = MATRIX_SIZE_U32 * size_of::<u32>();
/// Size (in 32-bit integers) of a reference ChaCha matrix.
pub const MATRIX_SIZE_U32: usize = COLUMNS * ROWS;

/// The amount of distinct ChaCha blocks we process in parallel.
pub const DEPTH: usize = 4;
/// Standard constant used in all ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};

/// Wrapper for the raw data of a ChaCha row. In a reference
/// implementation this would just be the `u32x4` field, but having
/// `u64x2` is useful for working with a 64-bit counter and `u8x16`
/// is useful for some tests. `u16x8` is included for completeness.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub union Row {
    pub u8x16: [u8; 16],
    pub u16x8: [u16; 8],
    pub u32x4: [u32; 4],
    pub u64x2: [u64; 2],
}

/// `ChaChaCore` without the `PhantomData` types.
///
/// Makes implementation in `Machine` less verbose.
#[repr(C)]
pub struct ChaChaNaked {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
}
