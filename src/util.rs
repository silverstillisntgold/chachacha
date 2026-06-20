/*!
Module containing useful constants/structs.
*/

#[cfg(target_feature = "neon")]
use core::arch::neon;
#[cfg(target_arch = "x86")]
use core::arch::x86::__m128i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;

/// The amount of distinct ChaCha blocks we process in parallel.
pub const DEPTH: usize = 4;
/// Standard constant used in all ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};

/// Columns in a ChaCha matrix.
const COLUMNS: usize = 4;
/// Rows in a ChaCha matrix.
const ROWS: usize = 4;

/// Size (in bytes) of a reference ChaCha matrix.
const MATRIX_SIZE_U8: usize = COLUMNS * ROWS * size_of::<u32>();

/// Size (in 8-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U8: usize = MATRIX_SIZE_U8 * DEPTH;
/// Size (in 16-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U16: usize = BUF_LEN_U8 / size_of::<u16>();
/// Size (in 32-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U32: usize = BUF_LEN_U8 / size_of::<u32>();
/// Size (in 64-bit integers) of a single ChaCha computation.
pub const BUF_LEN_U64: usize = BUF_LEN_U8 / size_of::<u64>();

/// Size (in 8-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U8: usize = (ROWS - 1) * size_of::<Row>();
/// Size (in 16-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U16: usize = SEED_LEN_U8 / size_of::<u16>();
/// Size (in 32-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U32: usize = SEED_LEN_U8 / size_of::<u32>();
/// Size (in 64-bit integers) of the raw seed for a ChaCha instance.
pub const SEED_LEN_U64: usize = SEED_LEN_U8 / size_of::<u64>();

pub trait Machine {
    fn get(&mut self, buf: &mut [u8; BUF_LEN_U8]);
}

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
    // Used for broadcasting in avx2 and avx512 backends.
    #[cfg(target_feature = "sse2")]
    pub u128x1: __m128i,
}

pub trait DoubleRounds {
    const COUNT: usize;
}

pub struct R8;
impl DoubleRounds for R8 {
    const COUNT: usize = 4;
}

pub struct R12;
impl DoubleRounds for R12 {
    const COUNT: usize = 6;
}

pub struct R20;
impl DoubleRounds for R20 {
    const COUNT: usize = 10;
}

pub enum Variants {
    /// Original variant proposed by the author of the salsa
    /// and chacha algorithms: Daniel J. Bernstein.
    Djb,
    /// Alternative variation specified by the IETF, most often
    /// used in conjunction with Poly1305.
    Ietf,
}

pub trait Variant {
    const VAR: Variants;
}

pub struct Djb;
impl Variant for Djb {
    const VAR: Variants = Variants::Djb;
}

pub struct Ietf;
impl Variant for Ietf {
    const VAR: Variants = Variants::Ietf;
}
