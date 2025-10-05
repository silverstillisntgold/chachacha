use super::{Vector, VectorOps, VectorType};
use crate::util::Row;
use core::arch::aarch64::*;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

const LOCAL_SIZE: usize = 4;

#[repr(C, align(64))]
struct Internal([int32x4_t; LOCAL_SIZE]);

#[derive(Clone, Copy)]
pub struct Neon;
impl VectorType for Neon {}

impl From<Internal> for Vector<Neon> {
    #[inline(always)]
    fn from(value: Internal) -> Self {
        unsafe { transmute(value) }
    }
}

impl From<Vector<Neon>> for Internal {
    #[inline(always)]
    fn from(value: Vector<Neon>) -> Self {
        unsafe { transmute(value) }
    }
}

impl Add for Vector<Neon> {
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

impl BitOr for Vector<Neon> {
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

impl BitXor for Vector<Neon> {
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

impl VectorOps for Vector<Neon> {
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
    fn shuffle_128<const MASK: i32>(self) -> Self {
        // Defer shuffling to the soft method because doing this with
        // neon intrinsics when when are taking `MASK` is aids. Maybe when const
        // generics are more fleshed out this won't be the case.
        // The optimizer still compiles this to `vext` neon instructions in
        // release mode, but in debug mode it'll be slower.
        Vector::<super::soft::Soft>::shuffle_128::<MASK>(self.cast()).cast()
    }
}
