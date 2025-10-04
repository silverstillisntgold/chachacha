use super::{BUF_SIZE, Vector, VectorOps, VectorType};
use crate::util::Row;
use core::mem::transmute;
use core::ops::{Add, BitOr, BitXor};

#[derive(Clone, Copy)]
pub struct Soft;
impl VectorType for Soft {}

impl Add for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            // Explicitly use `wrapping_add` to avoid debug builds
            // throwing a hissy fit.
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
        const SIZE: usize = size_of::<Vector<Soft>>() / size_of::<Row>();
        let tmp = [value; SIZE];
        unsafe { transmute(tmp) }
    }

    #[inline(always)]
    fn shift_left<const K: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] <<= K;
        }
        self
    }

    #[inline(always)]
    fn shift_right<const K: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] >>= K;
        }
        self
    }

    #[inline(always)]
    fn shuffle_128<const MASK: i32>(mut self) -> Self {
        #[inline(always)]
        const fn select<const MASK: i32, const K: i32>() -> usize {
            ((MASK >> K) & 3) as usize
        }

        let old = unsafe { self.inner.i32x16 };

        // First lane
        self.inner[0] = old[select::<MASK, 0>()];
        self.inner[1] = old[select::<MASK, 2>()];
        self.inner[2] = old[select::<MASK, 4>()];
        self.inner[3] = old[select::<MASK, 6>()];

        // Second lane
        self.inner[4] = old[4 + select::<MASK, 0>()];
        self.inner[5] = old[4 + select::<MASK, 2>()];
        self.inner[6] = old[4 + select::<MASK, 4>()];
        self.inner[7] = old[4 + select::<MASK, 6>()];

        // Third lane
        self.inner[8] = old[8 + select::<MASK, 0>()];
        self.inner[9] = old[8 + select::<MASK, 2>()];
        self.inner[10] = old[8 + select::<MASK, 4>()];
        self.inner[11] = old[8 + select::<MASK, 6>()];

        // Fourth lane
        self.inner[12] = old[12 + select::<MASK, 0>()];
        self.inner[13] = old[12 + select::<MASK, 2>()];
        self.inner[14] = old[12 + select::<MASK, 4>()];
        self.inner[15] = old[12 + select::<MASK, 6>()];

        self
    }
}
