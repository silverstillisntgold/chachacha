use super::{BUF_SIZE, Vector, VectorOps, VectorType};
use crate::util::Row;
use core::ops::{Add, BitOr, BitXor};
use core::{marker::PhantomData, mem::transmute};

pub struct Soft;
impl VectorType for Soft {}

impl Add for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
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
    fn shift_left<const IMM8: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] <<= IMM8;
        }
        self
    }

    #[inline(always)]
    fn shift_right<const IMM8: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self.inner[i] >>= IMM8;
        }
        self
    }

    #[inline(always)]
    fn shuffle_128<const IMM8: i32>(mut self) -> Self {
        #[inline(always)]
        const fn select<const IMM8: i32, const SHIFT: i32>() -> usize {
            ((IMM8 >> SHIFT) & 3) as usize
        }

        let old = unsafe { self.inner.i32x16 };

        // First lane
        self.inner[0] = old[select::<IMM8, 0>()];
        self.inner[1] = old[select::<IMM8, 2>()];
        self.inner[2] = old[select::<IMM8, 4>()];
        self.inner[3] = old[select::<IMM8, 6>()];

        // Second lane
        self.inner[4] = old[4 + select::<IMM8, 0>()];
        self.inner[5] = old[4 + select::<IMM8, 2>()];
        self.inner[6] = old[4 + select::<IMM8, 4>()];
        self.inner[7] = old[4 + select::<IMM8, 6>()];

        // Third lane
        self.inner[8] = old[8 + select::<IMM8, 0>()];
        self.inner[9] = old[8 + select::<IMM8, 2>()];
        self.inner[10] = old[8 + select::<IMM8, 4>()];
        self.inner[11] = old[8 + select::<IMM8, 6>()];

        // Fourth lane
        self.inner[12] = old[12 + select::<IMM8, 0>()];
        self.inner[13] = old[12 + select::<IMM8, 2>()];
        self.inner[14] = old[12 + select::<IMM8, 4>()];
        self.inner[15] = old[12 + select::<IMM8, 6>()];

        self
    }
}
