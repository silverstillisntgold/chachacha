use crate::variations::*;
use core::{
    mem::{MaybeUninit, transmute},
    ops::Add,
};

/// Standard constant used in ChaCha implementations.
pub const ROW_A: Row = Row {
    u8x16: *b"expand 32-byte k",
};
/// Size (in 32-bit integers) of a reference ChaCha matrix.
pub const CHACHA_SIZE: usize = 16;
/// Size (in 8-bit integers) of a ChaCha computation result.
pub const BUF_LEN: usize = CHACHA_SIZE * WIDTH * (size_of::<u32>() / size_of::<u8>());
/// Since we process in chunks of 4, the counter of the base
/// ChaCha instance needs to be incremented by 4.
pub const DEPTH: usize = 4;
pub const WIDTH: usize = 4;
pub const CHACHA_SEED_LEN: usize = 3 * size_of::<Row>();

/// Defines the interface that concrete implementations needed to
/// implement to process the state of a `ChaCha` instance.
pub trait Machine
where
    Self: Add<Output = Self> + Clone,
{
    /// Uses the provided [`ChaChaSmall`] state to create a new `Machine`,
    /// which will internally handle it's own counters.
    #[inline(always)]
    fn new<V: Variant>(state: &ChaChaSmall) -> Self {
        match V::VAR {
            Variants::Djb => Self::new_djb(state),
            Variants::Ietf => Self::new_ietf(state),
        }
    }

    fn new_djb(state: &ChaChaSmall) -> Self;

    fn new_ietf(state: &ChaChaSmall) -> Self;

    fn increment_djb(&mut self);

    fn increment_ietf(&mut self);

    /// Process a standard double round of the ChaCha algorithm.
    fn double_round(&mut self);

    /// Fills `buf` with the output of 4 processed ChaCha blocks.
    fn fill_block(self, buf: &mut [u8; BUF_LEN]);

    #[inline(always)]
    fn get_block(self) -> [u8; BUF_LEN] {
        #[allow(invalid_value)]
        let mut data = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        self.fill_block(&mut data);
        data
    }
}

/// Wrapper for the data of a `ChaCha` row. In a reference
/// implementation this would just be the `u32x4` field, but having
/// `u64x2` is useful for working with a 64-bit counter and `u8x16`
/// is useful for some tests. `u16x8` is included for completeness.
#[allow(unused)]
#[repr(align(16))]
#[derive(Clone, Copy)]
pub union Row {
    pub u8x16: [u8; 16],
    pub u16x8: [u16; 8],
    pub u32x4: [u32; 4],
    pub u64x2: [u64; 2],
}

#[derive(Clone)]
pub struct ChaChaSmall {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
}

impl Default for ChaChaSmall {
    #[inline(always)]
    fn default() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

impl From<[u8; CHACHA_SEED_LEN]> for ChaChaSmall {
    #[inline(always)]
    fn from(value: [u8; CHACHA_SEED_LEN]) -> Self {
        unsafe { transmute(value) }
    }
}

impl ChaChaSmall {
    #[inline(always)]
    pub fn increment<V: Variant>(&mut self) {
        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }
    }

    #[inline(always)]
    fn increment_djb(&mut self) {
        unsafe {
            self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(DEPTH as u64);
        }
    }

    #[inline(always)]
    fn increment_ietf(&mut self) {
        unsafe { self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(DEPTH as u32) }
    }
}
