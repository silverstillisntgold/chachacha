use super::{Vector, VectorOps, VectorType};
use crate::util::Row;
use core::arch::x86_64::*;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

const LOCAL_SIZE: usize = 2;

#[repr(C, align(64))]
struct Internal([__m256i; LOCAL_SIZE]);

#[derive(Clone, Copy)]
pub struct AVX2;
impl VectorType for AVX2 {}

impl From<Internal> for Vector<AVX2> {
    #[inline(always)]
    fn from(value: Internal) -> Self {
        unsafe { transmute(value) }
    }
}

impl From<Vector<AVX2>> for Internal {
    #[inline(always)]
    fn from(value: Vector<AVX2>) -> Self {
        unsafe { transmute(value) }
    }
}

impl Add for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_add_epi32(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitOr for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_or_si256(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitXor for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_xor_si256(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl VectorOps for Vector<AVX2> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        unsafe {
            let tmp = transmute(value);
            Internal([_mm256_broadcastd_epi32(tmp); LOCAL_SIZE]).into()
        }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_sll_epi32(lhs.0[i], count);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_srl_epi32(lhs.0[i], count);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(self) -> Self {
        unsafe {
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = _mm256_shuffle_epi32::<MASK>(lhs.0[i]);
            }
            lhs.into()
        }
    }
}
