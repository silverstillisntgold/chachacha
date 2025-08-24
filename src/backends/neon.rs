use crate::util::*;
use core::arch::aarch64::*;
use core::mem::transmute;
use core::ops::Add;

#[derive(Clone)]
#[repr(C)]
pub struct Matrix {
    state: [[InternalRow; ROWS]; DEPTH],
}

#[derive(Clone, Copy)]
#[repr(C)]
union InternalRow {
    u32x4: uint32x4_t,
    u64x2: uint64x2_t,
}

impl Add for Matrix {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..self.state.len() {
                for j in 0..self.state[i].len() {
                    self.state[i][j].u32x4 =
                        vaddq_u32(self.state[i][j].u32x4, rhs.state[i][j].u32x4);
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
    #[inline]
    fn quarter_round(&mut self) {
        unsafe {
            for [a, b, c, d] in self.state.iter_mut().map(|v| {
                let u32x4_4: &mut [uint32x4_t; ROWS] = transmute(v);
                u32x4_4
            }) {
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

    #[inline]
    fn make_diagonal(&mut self) {
        unsafe {
            for [a, _, c, d] in self.state.iter_mut().map(|v| {
                let u32x4_4: &mut [uint32x4_t; ROWS] = transmute(v);
                u32x4_4
            }) {
                *a = vextq_u32(*a, *a, 3);
                *c = vextq_u32(*c, *c, 1);
                *d = vextq_u32(*d, *d, 2);
            }
        }
    }

    #[inline]
    fn unmake_diagonal(&mut self) {
        unsafe {
            for [a, _, c, d] in self.state.iter_mut().map(|v| {
                let u32x4_4: &mut [uint32x4_t; ROWS] = transmute(v);
                u32x4_4
            }) {
                *c = vextq_u32(*c, *c, 3);
                *d = vextq_u32(*d, *d, 2);
                *a = vextq_u32(*a, *a, 1);
            }
        }
    }
}

impl Machine for Matrix {
    #[inline]
    fn new_djb(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [[
                    transmute(ROW_A),
                    transmute(state.row_b),
                    transmute(state.row_c),
                    transmute(state.row_d),
                ]; DEPTH],
            };
            result.state[1][3].u64x2 = vaddq_u64(
                result.state[1][3].u64x2,
                vcombine_u64(vcreate_u64(1), vcreate_u64(0)),
            );
            result.state[2][3].u64x2 = vaddq_u64(
                result.state[2][3].u64x2,
                vcombine_u64(vcreate_u64(2), vcreate_u64(0)),
            );
            result.state[3][3].u64x2 = vaddq_u64(
                result.state[3][3].u64x2,
                vcombine_u64(vcreate_u64(3), vcreate_u64(0)),
            );
            result
        }
    }

    #[inline]
    fn new_ietf(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [[
                    transmute(ROW_A),
                    transmute(state.row_b),
                    transmute(state.row_c),
                    transmute(state.row_d),
                ]; DEPTH],
            };
            result.state[1][3].u32x4 = vaddq_u32(
                result.state[1][3].u32x4,
                vcombine_u32(vcreate_u32(1), vcreate_u32(0)),
            );
            result.state[2][3].u32x4 = vaddq_u32(
                result.state[2][3].u32x4,
                vcombine_u32(vcreate_u32(2), vcreate_u32(0)),
            );
            result.state[3][3].u32x4 = vaddq_u32(
                result.state[3][3].u32x4,
                vcombine_u32(vcreate_u32(3), vcreate_u32(0)),
            );
            result
        }
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            let increment = vcombine_u64(vcreate_u64(DEPTH as u64), vcreate_u64(0));
            self.state[0][3].u64x2 = vaddq_u64(self.state[0][3].u64x2, increment);
            self.state[1][3].u64x2 = vaddq_u64(self.state[1][3].u64x2, increment);
            self.state[2][3].u64x2 = vaddq_u64(self.state[2][3].u64x2, increment);
            self.state[3][3].u64x2 = vaddq_u64(self.state[3][3].u64x2, increment);
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            let increment = vcombine_u32(vcreate_u32(DEPTH as u64), vcreate_u32(0));
            self.state[0][3].u32x4 = vaddq_u32(self.state[0][3].u32x4, increment);
            self.state[1][3].u32x4 = vaddq_u32(self.state[1][3].u32x4, increment);
            self.state[2][3].u32x4 = vaddq_u32(self.state[2][3].u32x4, increment);
            self.state[3][3].u32x4 = vaddq_u32(self.state[3][3].u32x4, increment);
        }
    }

    #[inline]
    fn double_round(&mut self) {
        // Column rounds
        self.quarter_round();
        // Diagonal rounds
        self.make_diagonal();
        self.quarter_round();
        self.unmake_diagonal();
    }

    #[inline]
    fn fetch_result(self, buf: &mut [u8; BUF_LEN_U8]) {
        unsafe {
            *buf = transmute(self);
        }
    }
}
