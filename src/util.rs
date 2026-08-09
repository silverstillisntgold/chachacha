use crate::chacha::ChaChaCore;
#[cfg(target_arch = "x86")]
use core::arch::x86::__m128i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;

/// Columns in a reference ChaCha matrix.
pub const COLUMNS: usize = 4;
/// Rows in a reference ChaCha matrix.
pub const ROWS: usize = 4;
/// Size (in 32-bit ints) of a reference ChaCha matrix.
pub const SIZE: usize = COLUMNS * ROWS;
/// Size (in bytes) of a reference ChaCha matrix.
pub const MATRIX_SIZE: usize = SIZE * size_of::<u32>();

/// The amount of ChaCha instances processed in parallel.
pub const BLOCKS: usize = 4;
/// The amount of bytes generated in parallel.
pub const BATCH_BYTES: usize = BLOCKS * MATRIX_SIZE;

/// Standard constant used in all ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};

/// Trait which represents an implementation for a specific hardware architecture.
pub trait Backend: Sized {
    /// Creates a new instance of [`Self`], which is used for holding
    /// initial state and tracking the running counter.
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self;

    /// Fills `buffer` with the output stream of `self`.
    ///
    /// TODO: When rust gets full const-generics it might be benefical to
    /// specialize the length of `buffer` to better optimize specific backends.
    fn fill<B: Backend, const ROUNDS: usize, V: Variant>(&mut self, buffer: &mut [u8; BATCH_BYTES]);
}

/// Wrapper for the raw data of a ChaCha row. In a reference
/// implementation this would just be the `u32x4` field, but having
/// `u64x2` is useful for working with a 64-bit counter and `u8x16`
/// is useful for some tests. `u16x8` is included for completeness.
///
/// The size and aligment of this struct are both 16 bytes to enable
/// the compiler to generate aligned operations wherever possible.
#[repr(C, align(16))]
pub union Row {
    pub u8x16: [u8; 16],
    pub u16x8: [u16; 8],
    pub u32x4: [u32; 4],
    pub u64x2: [u64; 2],
    // Useful in x86 backends.
    #[cfg(target_feature = "sse2")]
    pub u128x1: __m128i,
}

impl Clone for Row {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Self { u16x8: self.u16x8 } }
    }
}

pub enum Variants {
    /// Original variant proposed by the author of the salsa
    /// and chacha algorithms: Daniel J. Bernstein.
    Djb,
    /// Alternative variant specified by the IETF, most often
    /// used in conjunction with Poly1305.
    Ietf,
}

pub trait Variant {
    const VAR: Variants;
}

/// Type-level representation of [`Variants::Djb`].
pub struct Djb;
impl Variant for Djb {
    const VAR: Variants = Variants::Djb;
}

/// Type-level representation of [`Variants::Ietf`].
pub struct Ietf;
impl Variant for Ietf {
    const VAR: Variants = Variants::Ietf;
}
