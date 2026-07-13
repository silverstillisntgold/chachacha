use crate::chacha::ChaChaCore;
#[cfg(target_arch = "x86")]
use core::arch::x86::__m128i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;

/// Columns in a reference ChaCha matrix.
const COLUMNS: usize = 4;
/// Rows in a reference ChaCha matrix.
const ROWS: usize = 4;
/// Size (in bytes) of a reference ChaCha matrix.
pub const MATRIX_SIZE: usize = COLUMNS * ROWS * size_of::<u32>();

/// Standard constant used in all ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};

pub trait Backend: Sized {
    const BATCH_BYTES: usize = Self::BLOCKS * MATRIX_SIZE;
    const BLOCKS: usize;

    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    );
}

/// Wrapper for the raw data of a ChaCha row. In a reference
/// implementation this would just be the `u32x4` field, but having
/// `u64x2` is useful for working with a 64-bit counter and `u8x16`
/// is useful for some tests. `u16x8` is included for completeness.
///
/// The size and aligment of this struct are both 16 bytes so that
/// we can use aligned loads in the vectorized backends.
#[repr(C, align(16))]
pub union Row {
    pub u8x16: [u8; 16],
    pub u16x8: [u16; 8],
    pub u32x4: [u32; 4],
    pub u64x2: [u64; 2],
    // Useful in the avx2 and avx512 backends.
    #[cfg(target_feature = "sse2")]
    pub u128x1: __m128i,
}

impl Clone for Row {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Self { u8x16: self.u8x16 } }
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
