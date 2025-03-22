use crate::util::*;
use core::{mem::transmute, ops::Add};

#[derive(Clone, Copy)]
union InternalMatrix {
    raw: [u32; CHACHA_SIZE],
    rows: [Row; DEPTH],
}

#[derive(Clone)]
pub struct Matrix {
    state: [InternalMatrix; DEPTH],
}

impl Add for Matrix {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..self.state.len() {
                for j in 0..self.state[i].raw.len() {
                    self.state[i].raw[j] = self.state[i].raw[j].wrapping_add(rhs.state[i].raw[j]);
                }
            }
            self
        }
    }
}

impl Matrix {
    #[inline]
    fn quarter_round(&mut self, a: usize, b: usize, c: usize, d: usize) {
        unsafe {
            for matrix in self.state.iter_mut() {
                matrix.raw[a] = matrix.raw[a].wrapping_add(matrix.raw[b]);
                matrix.raw[d] ^= matrix.raw[a];
                matrix.raw[d] = matrix.raw[d].rotate_left(16);

                matrix.raw[c] = matrix.raw[c].wrapping_add(matrix.raw[d]);
                matrix.raw[b] ^= matrix.raw[c];
                matrix.raw[b] = matrix.raw[b].rotate_left(12);

                matrix.raw[a] = matrix.raw[a].wrapping_add(matrix.raw[b]);
                matrix.raw[d] ^= matrix.raw[a];
                matrix.raw[d] = matrix.raw[d].rotate_left(8);

                matrix.raw[c] = matrix.raw[c].wrapping_add(matrix.raw[d]);
                matrix.raw[b] ^= matrix.raw[c];
                matrix.raw[b] = matrix.raw[b].rotate_left(7);
            }
        }
    }
}

impl Machine for Matrix {
    #[inline]
    fn new_djb(state: &ChaChaSmall) -> Self {
        unsafe {
            let mut result = [[ROW_A, state.row_b, state.row_c, state.row_d]; DEPTH];
            result[1][3].u64x2[0] = result[1][3].u64x2[0].wrapping_add(1);
            result[2][3].u64x2[0] = result[2][3].u64x2[0].wrapping_add(2);
            result[3][3].u64x2[0] = result[3][3].u64x2[0].wrapping_add(3);
            Self {
                state: transmute(result),
            }
        }
    }

    #[inline]
    fn new_ietf(state: &ChaChaSmall) -> Self {
        unsafe {
            let mut result = [[ROW_A, state.row_b, state.row_c, state.row_d]; DEPTH];
            result[1][3].u32x4[0] = result[1][3].u32x4[0].wrapping_add(1);
            result[2][3].u32x4[0] = result[2][3].u32x4[0].wrapping_add(2);
            result[3][3].u32x4[0] = result[3][3].u32x4[0].wrapping_add(3);
            Self {
                state: transmute(result),
            }
        }
    }

    #[inline]
    fn double_round(&mut self) {
        // Column rounds
        self.quarter_round(0, 4, 8, 12);
        self.quarter_round(1, 5, 9, 13);
        self.quarter_round(2, 6, 10, 14);
        self.quarter_round(3, 7, 11, 15);
        // Diagonal rounds
        self.quarter_round(0, 5, 10, 15);
        self.quarter_round(1, 6, 11, 12);
        self.quarter_round(2, 7, 8, 13);
        self.quarter_round(3, 4, 9, 14);
    }

    #[inline]
    fn fill_block(self, buf: &mut [u8; BUF_LEN]) {
        unsafe {
            *buf = transmute(self);
        }
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            self.state[0].rows[3].u64x2[0] =
                self.state[0].rows[3].u64x2[0].wrapping_add(DEPTH_U8.into());
            self.state[1].rows[3].u64x2[0] =
                self.state[1].rows[3].u64x2[0].wrapping_add(DEPTH_U8.into());
            self.state[2].rows[3].u64x2[0] =
                self.state[2].rows[3].u64x2[0].wrapping_add(DEPTH_U8.into());
            self.state[3].rows[3].u64x2[0] =
                self.state[3].rows[3].u64x2[0].wrapping_add(DEPTH_U8.into());
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            self.state[0].rows[3].u32x4[0] =
                self.state[0].rows[3].u32x4[0].wrapping_add(DEPTH_U8.into());
            self.state[1].rows[3].u32x4[0] =
                self.state[1].rows[3].u32x4[0].wrapping_add(DEPTH_U8.into());
            self.state[2].rows[3].u32x4[0] =
                self.state[2].rows[3].u32x4[0].wrapping_add(DEPTH_U8.into());
            self.state[3].rows[3].u32x4[0] =
                self.state[3].rows[3].u32x4[0].wrapping_add(DEPTH_U8.into());
        }
    }
}
