use crate::util::*;
use core::arch::aarch64::*;
use core::{mem::transmute, ops::Add};

#[derive(Clone)]
pub struct Matrix {
    state: [[uint32x4_t; WIDTH]; DEPTH],
}

impl Add for Matrix {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..self.state.len() {
                for j in 0..self.state[i].len() {
                    self.state[i][j] = vaddq_u32(self.state[i][j], rhs.state[i][j]);
                }
            }
            self
        }
    }
}

macro_rules! rotate_left_epi32 {
    ($value:expr, $LEFT_SHIFT:expr) => {{
        const RIGHT_SHIFT: i32 = 32 - $LEFT_SHIFT;
        let left_shift = vshlq_n_u32($value, $LEFT_SHIFT);
        let right_shift = vshrq_n_u32($value, RIGHT_SHIFT);
        vorrq_u32(left_shift, right_shift)
    }};
}

impl Matrix {
    #[inline(always)]
    fn quarter_round(&mut self) {
        unsafe {
            for [a, b, c, d] in self.state.iter_mut() {
                *a = vaddq_u32(*a, *b);
                *d = veorq_u32(*d, *a);
                *d = rotate_left_epi32!(*d, 16);

                *c = vaddq_u32(*c, *d);
                *b = veorq_u32(*b, *c);
                *b = rotate_left_epi32!(*b, 12);

                *a = vaddq_u32(*a, *b);
                *d = veorq_u32(*d, *a);
                *d = rotate_left_epi32!(*d, 8);

                *c = vaddq_u32(*c, *d);
                *b = veorq_u32(*b, *c);
                *b = rotate_left_epi32!(*b, 7);
            }
        }
    }

    #[inline(always)]
    fn make_diagonal(&mut self) {
        unsafe {
            for [a, _, c, d] in self.state.iter_mut() {
                *a = vextq_u32(*a, *a, 3);
                *c = vextq_u32(*c, *c, 1);
                *d = vextq_u32(*d, *d, 2);
            }
        }
    }

    #[inline(always)]
    fn unmake_diagonal(&mut self) {
        unsafe {
            for [a, _, c, d] in self.state.iter_mut() {
                *c = vextq_u32(*c, *c, 3);
                *d = vextq_u32(*d, *d, 2);
                *a = vextq_u32(*a, *a, 1);
            }
        }
    }
}

impl Machine for Matrix {
    #[inline(always)]
    fn new(state: &ChaCha<Self>) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [[
                    transmute(ROW_A),
                    transmute(state.row_b),
                    transmute(state.row_c),
                    transmute(state.row_d),
                ]; DEPTH],
            };
            // Look what they need to mimic a fraction of my power.
            result.state[1][3] = vaddq_u32(
                result.state[1][3],
                vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(1), vcreate_u64(0))),
            );
            result.state[2][3] = vaddq_u32(
                result.state[2][3],
                vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(2), vcreate_u64(0))),
            );
            result.state[3][3] = vaddq_u32(
                result.state[3][3],
                vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(3), vcreate_u64(0))),
            );
            result
        }
    }

    #[inline(always)]
    fn double_round(&mut self) {
        // Column rounds
        self.quarter_round();
        // Diagonal rounds
        self.make_diagonal();
        self.quarter_round();
        self.unmake_diagonal();
    }

    #[inline(always)]
    fn fill_block(self, buf: &mut [u64; BUF_LEN]) {
        unsafe {
            *buf = transmute(self);
        }
    }
}
