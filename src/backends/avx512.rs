use super::{Internal, Vector, VectorOps};
use crate::util::Row;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct AVX512;

impl Add for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            self.u512x1 = _mm512_add_epi32(self.u512x1, rhs.u512x1);
            self
        }
    }
}

impl BitOr for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            self.u512x1 = _mm512_or_si512(self.u512x1, rhs.u512x1);
            self
        }
    }
}

impl BitXor for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            self.u512x1 = _mm512_xor_si512(self.u512x1, rhs.u512x1);
            self
        }
    }
}

impl VectorOps for Vector<AVX512> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        let tmp = unsafe { _mm512_broadcast_i32x4(value.u128x1) };
        Self {
            inner: Internal { u512x1: tmp },
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            self.u512x1 = _mm512_sll_epi32(self.u512x1, count);
            self
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            self.u512x1 = _mm512_srl_epi32(self.u512x1, count);
            self
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(mut self) -> Self {
        unsafe {
            self.u512x1 = _mm512_shuffle_epi32::<MASK>(self.u512x1);
            self
        }
    }
}
