use crate::util::*;
use core::mem::transmute;
use core::ops::Add;

#[derive(Clone)]
#[repr(C)]
pub struct Matrix {
    state: [InternalMatrix; DEPTH],
}

#[derive(Clone, Copy)]
#[repr(C)]
union InternalMatrix {
    raw: [u32; MATRIX_SIZE_U32],
    rows: [Row; ROWS],
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
    fn new_djb(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [InternalMatrix {
                    rows: [ROW_A, state.row_b, state.row_c, state.row_d],
                }; DEPTH],
            };
            result.state[1].rows[3].u64x2[0] = result.state[1].rows[3].u64x2[0].wrapping_add(1);
            result.state[2].rows[3].u64x2[0] = result.state[2].rows[3].u64x2[0].wrapping_add(2);
            result.state[3].rows[3].u64x2[0] = result.state[3].rows[3].u64x2[0].wrapping_add(3);
            result
        }
    }

    #[inline]
    fn new_ietf(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [InternalMatrix {
                    rows: [ROW_A, state.row_b, state.row_c, state.row_d],
                }; DEPTH],
            };
            result.state[1].rows[3].u32x4[0] = result.state[1].rows[3].u32x4[0].wrapping_add(1);
            result.state[2].rows[3].u32x4[0] = result.state[2].rows[3].u32x4[0].wrapping_add(2);
            result.state[3].rows[3].u32x4[0] = result.state[3].rows[3].u32x4[0].wrapping_add(3);
            result
        }
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            let increment = DEPTH as u64;
            self.state[0].rows[3].u64x2[0] = self.state[0].rows[3].u64x2[0].wrapping_add(increment);
            self.state[1].rows[3].u64x2[0] = self.state[1].rows[3].u64x2[0].wrapping_add(increment);
            self.state[2].rows[3].u64x2[0] = self.state[2].rows[3].u64x2[0].wrapping_add(increment);
            self.state[3].rows[3].u64x2[0] = self.state[3].rows[3].u64x2[0].wrapping_add(increment);
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            let increment = DEPTH as u32;
            self.state[0].rows[3].u32x4[0] = self.state[0].rows[3].u32x4[0].wrapping_add(increment);
            self.state[1].rows[3].u32x4[0] = self.state[1].rows[3].u32x4[0].wrapping_add(increment);
            self.state[2].rows[3].u32x4[0] = self.state[2].rows[3].u32x4[0].wrapping_add(increment);
            self.state[3].rows[3].u32x4[0] = self.state[3].rows[3].u32x4[0].wrapping_add(increment);
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
    fn fetch_result(self, buf: &mut [u8; BUF_LEN_U8]) {
        unsafe {
            *buf = transmute(self);
        }
    }
}
