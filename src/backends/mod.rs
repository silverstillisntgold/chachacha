/*!
TODO: Module docs.
*/

mod avx2;
mod avx512;
mod soft;
mod sse2;

use crate::util::ChaChaNaked;
use core::{
    marker::PhantomData,
    ops::{Add, BitOr, BitXor, Deref, DerefMut},
};

const AVX512_REG_SIZE: usize = 512;
const BUF_SIZE: usize = AVX512_REG_SIZE / i32::BITS as usize;

pub trait VectorType {}

/// Represents a single row of four side-by-side ChaCha instances.
#[repr(C, align(64))]
pub struct Vector<T> {
    inner: [i32; BUF_SIZE],
    _phantom: PhantomData<T>,
}

impl<T> Vector<T> {
    #[inline(always)]
    pub fn reverse(&mut self) {
        self.inner.reverse();
    }
}

macro_rules! rotate_left_epi32 {
    ($vector: expr, $LEFT_SHIFT: expr) => {{
        const LEFT_SHIFT: i64 = $LEFT_SHIFT;
        const RIGHT_SHIFT: i64 = 32 - LEFT_SHIFT;
        let left_shift = $vector.shift_left::<LEFT_SHIFT>();
        let right_shift = $vector.shift_right::<RIGHT_SHIFT>();
        left_shift | right_shift
    }};
}

pub trait VectorOps: Add + BitOr + BitXor + Sized {
    fn shift_left<const IMM8: i64>(self) -> Self;

    fn shift_right<const IMM8: i64>(self) -> Self;

    /// Shuffles the four internal 128-bit lanes using `IMM8` as a mask.
    fn shuffle_128<const IMM8: i32>(self) -> Self;
}

#[repr(C)]
pub struct Machine<T> {
    row_a: Vector<T>,
    row_b: Vector<T>,
    row_c: Vector<T>,
    row_d: Vector<T>,
}

impl<T> Machine<T>
where
    T: VectorType,
    Vector<T>: VectorOps,
{
}
