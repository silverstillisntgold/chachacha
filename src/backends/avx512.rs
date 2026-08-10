#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Variant, Variants},
};
use arch::{
    __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_broadcast_i32x4, _mm512_rol_epi32,
    _mm512_setr_epi32, _mm512_setr_epi64, _mm512_shuffle_epi32, _mm512_shuffle_i32x4,
    _mm512_xor_si512,
};

const VECTOR_WIDTH: usize = BATCH_BYTES / size_of::<__m512i>();

pub struct Avx512 {
    row_a: __m512i,
    row_b: __m512i,
    row_c: __m512i,
    row_d: __m512i,
}

impl Backend for Avx512 {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        unsafe {
            let row_a = _mm512_broadcast_i32x4(ROW_A.u128x1);
            let row_b = _mm512_broadcast_i32x4(core.row_b.u128x1);
            let row_c = _mm512_broadcast_i32x4(core.row_c.u128x1);
            let row_d = match V::VAR {
                Variants::Djb => _mm512_add_epi64(
                    _mm512_broadcast_i32x4(core.row_d.u128x1),
                    _mm512_setr_epi64(0, 0, 1, 0, 2, 0, 3, 0),
                ),
                Variants::Ietf => _mm512_add_epi32(
                    _mm512_broadcast_i32x4(core.row_d.u128x1),
                    _mm512_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0),
                ),
            };
            Self {
                row_a,
                row_b,
                row_c,
                row_d,
            }
        }
    }

    #[inline(never)]
    fn fill<B: Backend, const ROUNDS: usize, V: Variant, const XOR: bool>(
        &mut self,
        buffer: &mut [u8; BATCH_BYTES],
    ) {
        unsafe {
            let mut a = self.row_a;
            let mut b = self.row_b;
            let mut c = self.row_c;
            let mut d = self.row_d;

            double_rounds::<ROUNDS>(&mut a, &mut b, &mut c, &mut d);

            a = _mm512_add_epi32(a, self.row_a);
            b = _mm512_add_epi32(b, self.row_b);
            c = _mm512_add_epi32(c, self.row_c);
            d = _mm512_add_epi32(d, self.row_d);

            match V::VAR {
                Variants::Djb => {
                    self.row_d = _mm512_add_epi64(
                        self.row_d,
                        _mm512_setr_epi64(
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                        ),
                    )
                }
                Variants::Ietf => {
                    self.row_d = _mm512_add_epi32(
                        self.row_d,
                        _mm512_setr_epi32(
                            BLOCKS as i32,
                            0,
                            0,
                            0,
                            BLOCKS as i32,
                            0,
                            0,
                            0,
                            BLOCKS as i32,
                            0,
                            0,
                            0,
                            BLOCKS as i32,
                            0,
                            0,
                            0,
                        ),
                    )
                }
            }

            let permuted = core::mem::transmute::<[__m512i; VECTOR_WIDTH], [u8; BATCH_BYTES]>(
                permute_blocks(a, b, c, d),
            );

            for i in 0..BATCH_BYTES {
                if XOR {
                    buffer[i] ^= permuted[i];
                } else {
                    buffer[i] = permuted[i];
                }
            }
        }
    }
}

#[inline(always)]
fn permute_blocks(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; VECTOR_WIDTH] {
    unsafe {
        let ab01 = _mm512_shuffle_i32x4::<0x44>(a, b);
        let ab23 = _mm512_shuffle_i32x4::<0xee>(a, b);
        let cd01 = _mm512_shuffle_i32x4::<0x44>(c, d);
        let cd23 = _mm512_shuffle_i32x4::<0xee>(c, d);
        [
            _mm512_shuffle_i32x4::<0x88>(ab01, cd01),
            _mm512_shuffle_i32x4::<0xdd>(ab01, cd01),
            _mm512_shuffle_i32x4::<0x88>(ab23, cd23),
            _mm512_shuffle_i32x4::<0xdd>(ab23, cd23),
        ]
    }
}

#[inline(always)]
fn double_rounds<const ROUNDS: usize>(
    a: &mut __m512i,
    b: &mut __m512i,
    c: &mut __m512i,
    d: &mut __m512i,
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(a, b, c, d);
        rows_to_cols(a, c, d);
        add_xor_rotate(a, b, c, d);
        cols_to_rows(a, c, d);
    }
}

#[inline(always)]
fn add_xor_rotate(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        *a = _mm512_add_epi32(*a, *b);
        *d = _mm512_xor_si512(*d, *a);
        *d = _mm512_rol_epi32::<16>(*d);

        *c = _mm512_add_epi32(*c, *d);
        *b = _mm512_xor_si512(*b, *c);
        *b = _mm512_rol_epi32::<12>(*b);

        *a = _mm512_add_epi32(*a, *b);
        *d = _mm512_xor_si512(*d, *a);
        *d = _mm512_rol_epi32::<8>(*d);

        *c = _mm512_add_epi32(*c, *d);
        *b = _mm512_xor_si512(*b, *c);
        *b = _mm512_rol_epi32::<7>(*b);
    }
}

#[inline(always)]
fn rows_to_cols(a: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        *a = _mm512_shuffle_epi32::<0x93>(*a);
        *c = _mm512_shuffle_epi32::<0x39>(*c);
        *d = _mm512_shuffle_epi32::<0x4e>(*d);
    }
}

#[inline(always)]
fn cols_to_rows(a: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        *a = _mm512_shuffle_epi32::<0x39>(*a);
        *c = _mm512_shuffle_epi32::<0x93>(*c);
        *d = _mm512_shuffle_epi32::<0x4e>(*d);
    }
}
