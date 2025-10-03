//#![allow(unused)]

mod avx2;
mod avx512;
mod neon;
mod soft;
mod sse2;

use core::{
    marker::PhantomData,
    ops::{Add, BitOr, BitXor, Deref, DerefMut},
};

const AVX512_REG_SIZE: usize = 512;
const BUF_SIZE: usize = AVX512_REG_SIZE / i32::BITS as usize;

pub struct ChaChaInstance<T: VectorType = avx2::AVX2> {
    row_a: Vector<T>,
    row_b: Vector<T>,
    row_c: Vector<T>,
    rod_d: Vector<T>,
}

pub trait VectorType {}

#[derive(Clone)]
#[repr(C, align(64))]
pub struct Vector<T: VectorType> {
    inner: [i32; BUF_SIZE],
    _phantom: PhantomData<T>,
}

impl<T: VectorType> Deref for Vector<T> {
    type Target = [i32; BUF_SIZE];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: VectorType> DerefMut for Vector<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

macro_rules! rotate_left_epi32 {
    ($value: expr, $LEFT_SHIFT: expr) => {{
        const LEFT_SHIFT: i64 = $LEFT_SHIFT;
        const RIGHT_SHIFT: i64 = 32 - LEFT_SHIFT;
        let left_shift = $value.shift_left::<LEFT_SHIFT>();
        let right_shift = $value.shift_right::<RIGHT_SHIFT>();
        left_shift | right_shift
    }};
}

pub trait VectorOps: Add + BitOr + BitXor + Sized {
    fn shift_left<const IMM8: i64>(self) -> Self;

    fn shift_right<const IMM8: i64>(self) -> Self;

    /// Shuffles the four internal 128-bit lanes using `IMM8` as a mask.
    fn shuffle_128<const IMM8: i32>(self) -> Self;
}
