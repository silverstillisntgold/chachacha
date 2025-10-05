use super::{BUF_SIZE, Vector, VectorOps};
use crate::util::Row;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct Soft;

impl Add for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            // Explicitly use `wrapping_add` to avoid debug builds losing their shit.
            self.inner[i] = self.inner[i].wrapping_add(rhs.inner[i]);
        }
        self
    }
}

impl BitOr for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            self.inner[i] |= rhs.inner[i];
        }
        self
    }
}

impl BitXor for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            self.inner[i] ^= rhs.inner[i];
        }
        self
    }
}

impl VectorOps for Vector<Soft> {
    #[inline(always)]
    fn broadcast_row(value: Row) -> Self {
        const SIZE: usize = size_of::<[i32; BUF_SIZE]>() / size_of::<Row>();
        let tmp = [value; SIZE];
        unsafe { transmute(tmp) }
    }

    #[inline(always)]
    fn shift_left<const K: i32>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] <<= K;
        }
        self
    }

    #[inline(always)]
    fn shift_right<const K: i32>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] >>= K;
        }
        self
    }

    #[inline(always)]
    fn shuffle_internal<const MASK: i32>(mut self) -> Self {
        const fn select<const MASK: i32, const K: i32>() -> usize {
            (MASK as usize >> K) & 0b_11
        }
        self.inner.i32x16 = [
            // First 128-bit lane.
            self.inner[select::<MASK, 0>()],
            self.inner[select::<MASK, 2>()],
            self.inner[select::<MASK, 4>()],
            self.inner[select::<MASK, 6>()],
            // Second 128-bit lane.
            self.inner[4 + select::<MASK, 0>()],
            self.inner[4 + select::<MASK, 2>()],
            self.inner[4 + select::<MASK, 4>()],
            self.inner[4 + select::<MASK, 6>()],
            // Third 128-bit lane.
            self.inner[8 + select::<MASK, 0>()],
            self.inner[8 + select::<MASK, 2>()],
            self.inner[8 + select::<MASK, 4>()],
            self.inner[8 + select::<MASK, 6>()],
            // Fourth 128-bit lane.
            self.inner[12 + select::<MASK, 0>()],
            self.inner[12 + select::<MASK, 2>()],
            self.inner[12 + select::<MASK, 4>()],
            self.inner[12 + select::<MASK, 6>()],
        ];
        self
    }
}
