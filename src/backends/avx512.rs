use super::{Vector, VectorOps, VectorType};
use core::arch::x86_64::*;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

pub struct AVX512;
impl VectorType for AVX512 {}

impl From<__m512i> for Vector<AVX512> {
    #[inline(always)]
    fn from(value: __m512i) -> Self {
        unsafe { transmute(value) }
    }
}

impl From<Vector<AVX512>> for __m512i {
    #[inline(always)]
    fn from(value: Vector<AVX512>) -> Self {
        unsafe { transmute(value) }
    }
}

impl Add for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { _mm512_add_epi32(self.into(), rhs.into()).into() }
    }
}

impl BitOr for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { _mm512_or_si512(self.into(), rhs.into()).into() }
    }
}

impl BitXor for Vector<AVX512> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { _mm512_xor_si512(self.into(), rhs.into()).into() }
    }
}

impl VectorOps for Vector<AVX512> {
    #[inline(always)]
    fn shift_left<const IMM8: i64>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(IMM8);
            _mm512_sll_epi32(self.into(), count).into()
        }
    }

    #[inline(always)]
    fn shift_right<const IMM8: i64>(self) -> Self {
        unsafe {
            let count = _mm_set1_epi64x(IMM8);
            _mm512_srl_epi32(self.into(), count).into()
        }
    }

    #[inline(always)]
    fn shuffle_128<const IMM8: i32>(self) -> Self {
        unsafe { _mm512_shuffle_epi32(self.into(), IMM8).into() }
    }
}
