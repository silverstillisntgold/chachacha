use crate::variations::*;
use core::ops::Add;

/// Size (in 8-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U8: usize = MATRIX_SIZE_U8 * DEPTH;
/// Size (in 64-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U64: usize = BUF_LEN_U8 / size_of::<u64>();
pub const COLUMNS: usize = 4;
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

/// `ChaChaCore` without the `PhantomData` types. Makes concrete
/// implementations of `Machine` less verbose.
#[repr(C)]
pub struct ChaChaNaked {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
}

/// Core trait which must be implemented for all supported architectures.
pub trait Machine: Add<Output = Self> + Clone {
    /// Creates a new `Machine` by broadcasting the provided `ChaChaNaked`
    /// to `DEPTH` instances and incrementing the counters accordingly.
    #[inline]
    fn new<V: Variant>(state: &ChaChaNaked) -> Self {
        match V::VAR {
            Variants::Djb => Self::new_djb(state),
            Variants::Ietf => Self::new_ietf(state),
        }
    }

    /// Not to be used directly.
    fn new_djb(state: &ChaChaNaked) -> Self;

    /// Not to be used directly.
    fn new_ietf(state: &ChaChaNaked) -> Self;

    /// Increments the counter of each ChaCha instance in the current `Machine`.
    #[inline]
    fn increment<V: Variant>(&mut self) {
        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }
    }

    /// Not to be used directly.
    fn increment_djb(&mut self);

    /// Not to be used directly.
    fn increment_ietf(&mut self);

    /// Performs the standard ChaCha double round operation on all underlying instances.
    fn double_round(&mut self);

    /// Turns the current state of the `Machine` into it's byte representation.
    fn fetch_result(self, buf: &mut [u8; BUF_LEN_U8]);
}
