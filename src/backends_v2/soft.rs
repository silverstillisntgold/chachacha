use super::{BUF_SIZE, Vector, VectorOps, VectorType};
use core::ops::{Add, BitOr, BitXor};

pub struct Soft;
impl VectorType for Soft {}

impl Add for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            self[i] = self[i].wrapping_add(rhs[i]);
        }
        self
    }
}

impl BitOr for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            self[i] |= rhs[i];
        }
        self
    }
}

impl BitXor for Vector<Soft> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        for i in 0..BUF_SIZE {
            self[i] ^= rhs[i];
        }
        self
    }
}

impl VectorOps for Vector<Soft> {
    #[inline(always)]
    fn shift_left<const IMM8: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self[i] <<= IMM8;
        }
        self
    }

    #[inline(always)]
    fn shift_right<const IMM8: i64>(mut self) -> Self {
        for i in 0..BUF_SIZE {
            self[i] >>= IMM8;
        }
        self
    }

    #[inline(always)]
    fn shuffle_128<const IMM8: i32>(mut self) -> Self {
        #[inline(always)]
        const fn select<const IMM8: i32, const SHIFT: i32>() -> usize {
            ((IMM8 >> SHIFT) & 3) as usize
        }

        // Don't want implicit copies.
        #[allow(clippy::clone_on_copy)]
        let old = self.clone();

        // First lane
        self[0] = old[select::<IMM8, 0>()];
        self[1] = old[select::<IMM8, 2>()];
        self[2] = old[select::<IMM8, 4>()];
        self[3] = old[select::<IMM8, 6>()];

        // Second lane
        self[4] = old[4 + select::<IMM8, 0>()];
        self[5] = old[4 + select::<IMM8, 2>()];
        self[6] = old[4 + select::<IMM8, 4>()];
        self[7] = old[4 + select::<IMM8, 6>()];

        // Third lane
        self[8] = old[8 + select::<IMM8, 0>()];
        self[9] = old[8 + select::<IMM8, 2>()];
        self[10] = old[8 + select::<IMM8, 4>()];
        self[11] = old[8 + select::<IMM8, 6>()];

        // Fourth lane
        self[12] = old[12 + select::<IMM8, 0>()];
        self[13] = old[12 + select::<IMM8, 2>()];
        self[14] = old[12 + select::<IMM8, 4>()];
        self[15] = old[12 + select::<IMM8, 6>()];

        self
    }
}
