use crate::rounds::*;
use crate::variations::*;
use core::ops::Add;

/// Size (in 8-bit integers) of a ChaCha computation result.
pub const BUF_LEN: usize = CHACHA_SIZE * DEPTH * (size_of::<u32>() / size_of::<u8>());
/// Size (in 64-bit integers) of a ChaCha computation result.
pub const BUF_LEN_U64: usize = BUF_LEN / size_of::<u64>();
pub const CHACHA_COLUMNS: usize = 4;
pub const CHACHA_ROWS: usize = 4;
pub const CHACHA_SEED_LEN: usize = (CHACHA_ROWS - 1) * size_of::<Row>();
/// Size (in 32-bit integers) of a reference ChaCha matrix.
pub const CHACHA_SIZE: usize = CHACHA_ROWS * CHACHA_COLUMNS;
/// The amount of distinct Chacha blocks we process in parallel.
pub const DEPTH: usize = 4;
/// Standard constant used in all ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};

/// Wrapper for the raw data of a `ChaCha` row. In a reference
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

#[repr(C)]
pub struct ChaChaNaked {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
}

/// Defines the interface that concrete implementations need to
/// implement to process the state of a `ChaCha` instance.
pub trait Machine
where
    Self: Add<Output = Self> + Clone,
{
    #[inline(always)]
    fn new<V: Variant>(state: &ChaChaNaked) -> Self {
        match V::VAR {
            Variants::Djb => Self::new_djb(state),
            Variants::Ietf => Self::new_ietf(state),
        }
    }

    fn new_djb(state: &ChaChaNaked) -> Self;

    fn new_ietf(state: &ChaChaNaked) -> Self;

    #[inline(always)]
    fn increment<V: Variant>(&mut self) {
        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }
    }

    fn increment_djb(&mut self);

    fn increment_ietf(&mut self);

    fn double_round(&mut self);

    fn fetch_result(self, buf: &mut [u8; BUF_LEN]);

    #[inline(always)]
    fn chacha<const INCREMENT: bool, R: DoubleRounds, V: Variant>(
        &mut self,
        buf: &mut [u8; BUF_LEN],
    ) {
        let mut cur = self.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + self.clone();
        if INCREMENT {
            self.increment::<V>();
        }
        result.fetch_result(buf);
    }
}
