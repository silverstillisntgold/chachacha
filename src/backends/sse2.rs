use super::soft::Soft;
use super::{BUF_SIZE_U128, Vector, VectorOps};
use crate::util::Row;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct SSE2;

impl Add for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_add_epi32(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl BitOr for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_or_si128(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl BitXor for Vector<SSE2> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_xor_si128(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl VectorOps for Vector<SSE2> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        Vector::<Soft>::broadcast_row(value).cast()
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_sll_epi32(self.u128x4[i], count);
            }
            self
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_srl_epi32(self.u128x4[i], count);
            }
            self
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = _mm_shuffle_epi32::<MASK>(self.u128x4[i]);
            }
            self
        }
    }
}
