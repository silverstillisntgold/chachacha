use crate::util::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem::transmute;
use core::ops::Add;

#[derive(Clone)]
#[repr(C)]
pub struct Matrix {
    state: [__m512i; ROWS],
}

impl Add for Matrix {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        unsafe {
            for i in 0..self.state.len() {
                self.state[i] = _mm512_add_epi32(self.state[i], rhs.state[i]);
            }
            self
        }
    }
}

impl Matrix {
    #[inline]
    fn quarter_round(&mut self) {
        unsafe {
            self.state[0] = _mm512_add_epi32(self.state[0], self.state[1]);
            self.state[3] = _mm512_xor_si512(self.state[3], self.state[0]);
            self.state[3] = _mm512_rol_epi32(self.state[3], 16);

            self.state[2] = _mm512_add_epi32(self.state[2], self.state[3]);
            self.state[1] = _mm512_xor_si512(self.state[1], self.state[2]);
            self.state[1] = _mm512_rol_epi32(self.state[1], 12);

            self.state[0] = _mm512_add_epi32(self.state[0], self.state[1]);
            self.state[3] = _mm512_xor_si512(self.state[3], self.state[0]);
            self.state[3] = _mm512_rol_epi32(self.state[3], 8);

            self.state[2] = _mm512_add_epi32(self.state[2], self.state[3]);
            self.state[1] = _mm512_xor_si512(self.state[1], self.state[2]);
            self.state[1] = _mm512_rol_epi32(self.state[1], 7);
        }
    }

    #[inline]
    fn make_diagonal(&mut self) {
        unsafe {
            self.state[0] = _mm512_shuffle_epi32(self.state[0], 0b_10_01_00_11);
            self.state[2] = _mm512_shuffle_epi32(self.state[2], 0b_00_11_10_01);
            self.state[3] = _mm512_shuffle_epi32(self.state[3], 0b_01_00_11_10);
        }
    }

    #[inline]
    fn unmake_diagonal(&mut self) {
        unsafe {
            self.state[2] = _mm512_shuffle_epi32(self.state[2], 0b_10_01_00_11);
            self.state[3] = _mm512_shuffle_epi32(self.state[3], 0b_01_00_11_10);
            self.state[0] = _mm512_shuffle_epi32(self.state[0], 0b_00_11_10_01);
        }
    }
}

impl Machine for Matrix {
    #[inline]
    fn new_djb(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [
                    _mm512_broadcast_i32x4(transmute(ROW_A)),
                    _mm512_broadcast_i32x4(transmute(state.row_b)),
                    _mm512_broadcast_i32x4(transmute(state.row_c)),
                    _mm512_broadcast_i32x4(transmute(state.row_d)),
                ],
            };
            result.state[3] =
                _mm512_add_epi64(result.state[3], _mm512_set_epi64(0, 0, 0, 1, 0, 2, 0, 3));
            result
        }
    }

    #[inline]
    fn new_ietf(state: &ChaChaNaked) -> Self {
        unsafe {
            let mut result = Matrix {
                state: [
                    _mm512_broadcast_i32x4(transmute(ROW_A)),
                    _mm512_broadcast_i32x4(transmute(state.row_b)),
                    _mm512_broadcast_i32x4(transmute(state.row_c)),
                    _mm512_broadcast_i32x4(transmute(state.row_d)),
                ],
            };
            result.state[3] = _mm512_add_epi32(
                result.state[3],
                _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3),
            );
            result
        }
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            self.state[3] = _mm512_add_epi64(
                self.state[3],
                _mm512_set_epi64(
                    0,
                    DEPTH as i64,
                    0,
                    DEPTH as i64,
                    0,
                    DEPTH as i64,
                    0,
                    DEPTH as i64,
                ),
            );
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            self.state[3] = _mm512_add_epi32(
                self.state[3],
                _mm512_set_epi32(
                    0,
                    0,
                    0,
                    DEPTH as i32,
                    0,
                    0,
                    0,
                    DEPTH as i32,
                    0,
                    0,
                    0,
                    DEPTH as i32,
                    0,
                    0,
                    0,
                    DEPTH as i32,
                ),
            );
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
            *buf = transmute([
                [
                    _mm512_extracti32x4_epi32(self.state[0], 3),
                    _mm512_extracti32x4_epi32(self.state[1], 3),
                    _mm512_extracti32x4_epi32(self.state[2], 3),
                    _mm512_extracti32x4_epi32(self.state[3], 3),
                ],
                [
                    _mm512_extracti32x4_epi32(self.state[0], 2),
                    _mm512_extracti32x4_epi32(self.state[1], 2),
                    _mm512_extracti32x4_epi32(self.state[2], 2),
                    _mm512_extracti32x4_epi32(self.state[3], 2),
                ],
                [
                    _mm512_extracti32x4_epi32(self.state[0], 1),
                    _mm512_extracti32x4_epi32(self.state[1], 1),
                    _mm512_extracti32x4_epi32(self.state[2], 1),
                    _mm512_extracti32x4_epi32(self.state[3], 1),
                ],
                [
                    _mm512_extracti32x4_epi32(self.state[0], 0),
                    _mm512_extracti32x4_epi32(self.state[1], 0),
                    _mm512_extracti32x4_epi32(self.state[2], 0),
                    _mm512_extracti32x4_epi32(self.state[3], 0),
                ],
            ]);
        }
    }
}
