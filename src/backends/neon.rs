use super::{Vector, VectorOps};
use crate::util::Row;
use core::arch::aarch64::*;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

const LOCAL_SIZE: usize = 4;

#[repr(C, align(64))]
struct Internal([int32x4_t; LOCAL_SIZE]);

#[derive(Clone, Copy)]
pub struct NEON;

impl From<Internal> for Vector<NEON> {
    #[inline(always)]
    fn from(value: Internal) -> Self {
        unsafe { transmute(value) }
    }
}

impl From<Vector<NEON>> for Internal {
    #[inline(always)]
    fn from(value: Vector<NEON>) -> Self {
        unsafe { transmute(value) }
    }
}

impl Add for Vector<NEON> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = vaddq_s32(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitOr for Vector<NEON> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = vorrq_s32(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl BitXor for Vector<NEON> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut lhs = Internal::from(self);
            let rhs = Internal::from(rhs);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = veorq_s32(lhs.0[i], rhs.0[i]);
            }
            lhs.into()
        }
    }
}

impl VectorOps for Vector<NEON> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        unsafe {
            let tmp = transmute(value);
            Internal([tmp; LOCAL_SIZE]).into()
        }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(self) -> Self {
        unsafe {
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = vshlq_n_s32::<K>(lhs.0[i]);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(self) -> Self {
        unsafe {
            let mut lhs = Internal::from(self);
            for i in 0..LOCAL_SIZE {
                lhs.0[i] = vshrq_n_s32::<K>(lhs.0[i]);
            }
            lhs.into()
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(self) -> Self {
        // Delegate lane shuffling to the implementation used for soft targets,
        // since doing this with neon intrinsics when our input is `MASK` is kind
        // of aids.
        // The soft implementation is written in such a way that the compiler still
        // emits optimal assembly in release mode.
        self.cast::<super::soft::Soft>()
            .shuffle_internal::<MASK>()
            .cast()
    }
}
