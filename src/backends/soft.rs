use super::{BUF_SIZE_U32, Internal, Vector, VectorOps};
use crate::util::Row;
use core::marker::PhantomData;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct Soft;

impl Add for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U32 {
                // Need to use `wrapping_add` or debug builds will lose their shit.
                self.u32x16[i] = self.u32x16[i].wrapping_add(rhs.u32x16[i]);
            }
            self
        }
    }
}

impl BitOr for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U32 {
                self.u32x16[i] |= rhs.u32x16[i];
            }
            self
        }
    }
}

impl BitXor for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..BUF_SIZE_U32 {
                self.u32x16[i] ^= rhs.u32x16[i];
            }
            self
        }
    }
}

impl VectorOps for Vector<Soft> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        Self {
            inner: Internal { rowx4: [value; _] },
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U32 {
                self.u32x16[i] <<= K;
            }
            self
        }
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        unsafe {
            for i in 0..BUF_SIZE_U32 {
                self.u32x16[i] >>= K;
            }
            self
        }
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(mut self) -> Self {
        const fn select<const MASK: i32, const K: i32>() -> usize {
            (MASK as usize >> K) & 0b_11
        }
        unsafe {
            // Testing demonstrates that LLVM has no problem turning this into optimal
            // shuffling operations on targets which support them.
            self.u32x16 = [
                // First 128-bit lane.
                self.u32x16[select::<MASK, 0>()],
                self.u32x16[select::<MASK, 2>()],
                self.u32x16[select::<MASK, 4>()],
                self.u32x16[select::<MASK, 6>()],
                // Second 128-bit lane.
                self.u32x16[4 + select::<MASK, 0>()],
                self.u32x16[4 + select::<MASK, 2>()],
                self.u32x16[4 + select::<MASK, 4>()],
                self.u32x16[4 + select::<MASK, 6>()],
                // Third 128-bit lane.
                self.u32x16[8 + select::<MASK, 0>()],
                self.u32x16[8 + select::<MASK, 2>()],
                self.u32x16[8 + select::<MASK, 4>()],
                self.u32x16[8 + select::<MASK, 6>()],
                // Fourth 128-bit lane.
                self.u32x16[12 + select::<MASK, 0>()],
                self.u32x16[12 + select::<MASK, 2>()],
                self.u32x16[12 + select::<MASK, 4>()],
                self.u32x16[12 + select::<MASK, 6>()],
            ];
            self
        }
    }
}
