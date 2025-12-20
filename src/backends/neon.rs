use super::soft::Soft;
use super::{BUF_SIZE_U128, Vector, VectorOps};
use crate::util::Row;
use core::arch::aarch64::*;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct Neon;

impl Add for Vector<Neon> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = vaddq_u32(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl BitOr for Vector<Neon> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = vorrq_u32(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl BitXor for Vector<Neon> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = veorq_u32(self.u128x4[i], rhs.u128x4[i]);
            }
            self
        }
    }
}

impl VectorOps for Vector<Neon> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        Vector::<Soft>::broadcast_row(value).cast()
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = vshlq_n_u32::<K>(self.u128x4[i]);
            }
            self
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U128 {
                self.u128x4[i] = vshrq_n_u32::<K>(self.u128x4[i]);
            }
            self
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(self) -> Self {
        // Delegate lane shuffling to the implementation used for soft targets,
        // since doing this with neon intrinsics when our input is `MASK` is kind
        // of aids.
        // The soft implementation is written in such a way that the compiler
        // still emits optimal shuffling assembly in release mode.
        self.cast::<Soft>().shuffle_internal::<MASK>().cast()
    }
}
