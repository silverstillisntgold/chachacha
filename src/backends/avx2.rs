use super::{BUF_SIZE_U256, Internal, Vector, VectorOps};
use crate::util::Row;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct AVX2;

impl Add for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_add_epi32(self.u256x2[i], rhs.u256x2[i]);
            }
            self
        }
    }
}

impl BitOr for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_or_si256(self.u256x2[i], rhs.u256x2[i]);
            }
            self
        }
    }
}

impl BitXor for Vector<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_xor_si256(self.u256x2[i], rhs.u256x2[i]);
            }
            self
        }
    }
}

impl VectorOps for Vector<AVX2> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        let tmp = unsafe { _mm256_broadcastsi128_si256(value.u128x1) };
        Self {
            inner: Internal { u256x2: [tmp; _] },
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_sll_epi32(self.u256x2[i], count);
            }
            self
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(K as i64);
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_srl_epi32(self.u256x2[i], count);
            }
            self
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U256 {
                self.u256x2[i] = _mm256_shuffle_epi32::<MASK>(self.u256x2[i]);
            }
            self
        }
    }
}
