use super::{Vector, VectorOps, VectorType};
use core::arch::x86_64::*;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

const LOCAL_SIZE: usize = 4;

#[repr(C, align(64))]
pub struct Internal([__m128i; LOCAL_SIZE]);

pub struct SSE2;
impl VectorType for SSE2 {}

impl From<Internal> for Vector<SSE2> {
    #[inline(always)]
    fn from(value: Internal) -> Self {
        unsafe { transmute(value) }
    }
}

impl From<Vector<SSE2>> for Internal {
    #[inline(always)]
    fn from(value: Vector<SSE2>) -> Self {
        unsafe { transmute(value) }
    }
}

impl Add for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_add_epi32(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitOr for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_or_si128(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitXor for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_xor_si128(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl VectorOps for Vector<SSE2> {
    #[inline(always)]
    fn shift_left<const IMM8: i64>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(IMM8);
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_sll_epi32(lhs.0[i], count);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shift_right<const IMM8: i64>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(IMM8);
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_srl_epi32(lhs.0[i], count);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shuffle_128<const IMM8: i32>(self) -> Self {
        unsafe {
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm_shuffle_epi32(lhs.0[i], IMM8);
            }
            lhs.into()
        }
    }
}
