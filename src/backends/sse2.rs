#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Variant, Variants},
};
use arch::{
    __m128i, _mm_add_epi32, _mm_add_epi64, _mm_or_si128, _mm_set_epi32, _mm_set_epi64x,
    _mm_shuffle_epi32, _mm_slli_epi32, _mm_srli_epi32, _mm_xor_si128,
};

const VECTOR_WIDTH: usize = BATCH_BYTES / size_of::<__m128i>();

pub struct Sse2 {
    row_a: __m128i,
    row_b: __m128i,
    row_c: __m128i,
    row_d0: __m128i,
    row_d1: __m128i,
    row_d2: __m128i,
    row_d3: __m128i,
}

impl Backend for Sse2 {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        unsafe {
            let row_a = ROW_A.u128x1;
            let row_b = core.row_b.u128x1;
            let row_c = core.row_c.u128x1;
            let (row_d0, row_d1, row_d2, row_d3) = {
                let row_d = core.row_d.u128x1;
                match V::VAR {
                    Variants::Djb => {
                        let d0 = _mm_add_epi64(row_d, _mm_set_epi64x(0, 0));
                        let d1 = _mm_add_epi64(row_d, _mm_set_epi64x(0, 1));
                        let d2 = _mm_add_epi64(row_d, _mm_set_epi64x(0, 2));
                        let d3 = _mm_add_epi64(row_d, _mm_set_epi64x(0, 3));
                        (d0, d1, d2, d3)
                    }
                    Variants::Ietf => {
                        let d0 = _mm_add_epi32(row_d, _mm_set_epi32(0, 0, 0, 0));
                        let d1 = _mm_add_epi32(row_d, _mm_set_epi32(0, 0, 0, 1));
                        let d2 = _mm_add_epi32(row_d, _mm_set_epi32(0, 0, 0, 2));
                        let d3 = _mm_add_epi32(row_d, _mm_set_epi32(0, 0, 0, 3));
                        (d0, d1, d2, d3)
                    }
                }
            };
            Self {
                row_a,
                row_b,
                row_c,
                row_d0,
                row_d1,
                row_d2,
                row_d3,
            }
        }
    }

    #[inline(never)]
    fn fill<B: Backend, const ROUNDS: usize, V: Variant, const XOR: bool>(
        &mut self,
        buffer: &mut [u8; BATCH_BYTES],
    ) {
        unsafe {
            let mut a0 = self.row_a;
            let mut b0 = self.row_b;
            let mut c0 = self.row_c;
            let mut d0 = self.row_d0;
            let mut a1 = self.row_a;
            let mut b1 = self.row_b;
            let mut c1 = self.row_c;
            let mut d1 = self.row_d1;
            let mut a2 = self.row_a;
            let mut b2 = self.row_b;
            let mut c2 = self.row_c;
            let mut d2 = self.row_d2;
            let mut a3 = self.row_a;
            let mut b3 = self.row_b;
            let mut c3 = self.row_c;
            let mut d3 = self.row_d3;

            double_rounds::<ROUNDS>(
                &mut a0, &mut b0, &mut c0, &mut d0, &mut a1, &mut b1, &mut c1, &mut d1, &mut a2,
                &mut b2, &mut c2, &mut d2, &mut a3, &mut b3, &mut c3, &mut d3,
            );

            a0 = _mm_add_epi32(a0, self.row_a);
            b0 = _mm_add_epi32(b0, self.row_b);
            c0 = _mm_add_epi32(c0, self.row_c);
            d0 = _mm_add_epi32(d0, self.row_d0);
            a1 = _mm_add_epi32(a1, self.row_a);
            b1 = _mm_add_epi32(b1, self.row_b);
            c1 = _mm_add_epi32(c1, self.row_c);
            d1 = _mm_add_epi32(d1, self.row_d1);
            a2 = _mm_add_epi32(a2, self.row_a);
            b2 = _mm_add_epi32(b2, self.row_b);
            c2 = _mm_add_epi32(c2, self.row_c);
            d2 = _mm_add_epi32(d2, self.row_d2);
            a3 = _mm_add_epi32(a3, self.row_a);
            b3 = _mm_add_epi32(b3, self.row_b);
            c3 = _mm_add_epi32(c3, self.row_c);
            d3 = _mm_add_epi32(d3, self.row_d3);

            match V::VAR {
                Variants::Djb => {
                    let increment = _mm_set_epi64x(0, BLOCKS as i64);
                    self.row_d0 = _mm_add_epi64(self.row_d0, increment);
                    self.row_d1 = _mm_add_epi64(self.row_d1, increment);
                    self.row_d2 = _mm_add_epi64(self.row_d2, increment);
                    self.row_d3 = _mm_add_epi64(self.row_d3, increment);
                }
                Variants::Ietf => {
                    let increment = _mm_set_epi32(0, 0, 0, BLOCKS as i32);
                    self.row_d0 = _mm_add_epi32(self.row_d0, increment);
                    self.row_d1 = _mm_add_epi32(self.row_d1, increment);
                    self.row_d2 = _mm_add_epi32(self.row_d2, increment);
                    self.row_d3 = _mm_add_epi32(self.row_d3, increment);
                }
            }

            let bytes = core::mem::transmute::<[__m128i; VECTOR_WIDTH], [u8; BATCH_BYTES]>([
                a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3,
            ]);

            for i in 0..BATCH_BYTES {
                if XOR {
                    buffer[i] ^= bytes[i];
                } else {
                    buffer[i] = bytes[i];
                }
            }
        }
    }
}

#[inline]
fn double_rounds<const ROUNDS: usize>(
    a0: &mut __m128i,
    b0: &mut __m128i,
    c0: &mut __m128i,
    d0: &mut __m128i,
    a1: &mut __m128i,
    b1: &mut __m128i,
    c1: &mut __m128i,
    d1: &mut __m128i,
    a2: &mut __m128i,
    b2: &mut __m128i,
    c2: &mut __m128i,
    d2: &mut __m128i,
    a3: &mut __m128i,
    b3: &mut __m128i,
    c3: &mut __m128i,
    d3: &mut __m128i,
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(
            a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3,
        );
        rows_to_cols(a0, c0, d0, a1, c1, d1, a2, c2, d2, a3, c3, d3);
        add_xor_rotate(
            a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3,
        );
        cols_to_rows(a0, c0, d0, a1, c1, d1, a2, c2, d2, a3, c3, d3);
    }
}

#[inline]
fn add_xor_rotate(
    a0: &mut __m128i,
    b0: &mut __m128i,
    c0: &mut __m128i,
    d0: &mut __m128i,
    a1: &mut __m128i,
    b1: &mut __m128i,
    c1: &mut __m128i,
    d1: &mut __m128i,
    a2: &mut __m128i,
    b2: &mut __m128i,
    c2: &mut __m128i,
    d2: &mut __m128i,
    a3: &mut __m128i,
    b3: &mut __m128i,
    c3: &mut __m128i,
    d3: &mut __m128i,
) {
    unsafe {
        *a0 = _mm_add_epi32(*a0, *b0);
        *d0 = _mm_xor_si128(*d0, *a0);
        *d0 = _mm_or_si128(_mm_slli_epi32::<16>(*d0), _mm_srli_epi32::<16>(*d0));

        *a1 = _mm_add_epi32(*a1, *b1);
        *d1 = _mm_xor_si128(*d1, *a1);
        *d1 = _mm_or_si128(_mm_slli_epi32::<16>(*d1), _mm_srli_epi32::<16>(*d1));

        *a2 = _mm_add_epi32(*a2, *b2);
        *d2 = _mm_xor_si128(*d2, *a2);
        *d2 = _mm_or_si128(_mm_slli_epi32::<16>(*d2), _mm_srli_epi32::<16>(*d2));

        *a3 = _mm_add_epi32(*a3, *b3);
        *d3 = _mm_xor_si128(*d3, *a3);
        *d3 = _mm_or_si128(_mm_slli_epi32::<16>(*d3), _mm_srli_epi32::<16>(*d3));

        *c0 = _mm_add_epi32(*c0, *d0);
        *b0 = _mm_xor_si128(*b0, *c0);
        *b0 = _mm_or_si128(_mm_slli_epi32::<12>(*b0), _mm_srli_epi32::<20>(*b0));

        *c1 = _mm_add_epi32(*c1, *d1);
        *b1 = _mm_xor_si128(*b1, *c1);
        *b1 = _mm_or_si128(_mm_slli_epi32::<12>(*b1), _mm_srli_epi32::<20>(*b1));

        *c2 = _mm_add_epi32(*c2, *d2);
        *b2 = _mm_xor_si128(*b2, *c2);
        *b2 = _mm_or_si128(_mm_slli_epi32::<12>(*b2), _mm_srli_epi32::<20>(*b2));

        *c3 = _mm_add_epi32(*c3, *d3);
        *b3 = _mm_xor_si128(*b3, *c3);
        *b3 = _mm_or_si128(_mm_slli_epi32::<12>(*b3), _mm_srli_epi32::<20>(*b3));

        *a0 = _mm_add_epi32(*a0, *b0);
        *d0 = _mm_xor_si128(*d0, *a0);
        *d0 = _mm_or_si128(_mm_slli_epi32::<8>(*d0), _mm_srli_epi32::<24>(*d0));

        *a1 = _mm_add_epi32(*a1, *b1);
        *d1 = _mm_xor_si128(*d1, *a1);
        *d1 = _mm_or_si128(_mm_slli_epi32::<8>(*d1), _mm_srli_epi32::<24>(*d1));

        *a2 = _mm_add_epi32(*a2, *b2);
        *d2 = _mm_xor_si128(*d2, *a2);
        *d2 = _mm_or_si128(_mm_slli_epi32::<8>(*d2), _mm_srli_epi32::<24>(*d2));

        *a3 = _mm_add_epi32(*a3, *b3);
        *d3 = _mm_xor_si128(*d3, *a3);
        *d3 = _mm_or_si128(_mm_slli_epi32::<8>(*d3), _mm_srli_epi32::<24>(*d3));

        *c0 = _mm_add_epi32(*c0, *d0);
        *b0 = _mm_xor_si128(*b0, *c0);
        *b0 = _mm_or_si128(_mm_slli_epi32::<7>(*b0), _mm_srli_epi32::<25>(*b0));

        *c1 = _mm_add_epi32(*c1, *d1);
        *b1 = _mm_xor_si128(*b1, *c1);
        *b1 = _mm_or_si128(_mm_slli_epi32::<7>(*b1), _mm_srli_epi32::<25>(*b1));

        *c2 = _mm_add_epi32(*c2, *d2);
        *b2 = _mm_xor_si128(*b2, *c2);
        *b2 = _mm_or_si128(_mm_slli_epi32::<7>(*b2), _mm_srli_epi32::<25>(*b2));

        *c3 = _mm_add_epi32(*c3, *d3);
        *b3 = _mm_xor_si128(*b3, *c3);
        *b3 = _mm_or_si128(_mm_slli_epi32::<7>(*b3), _mm_srli_epi32::<25>(*b3));
    }
}

#[inline]
fn rows_to_cols(
    a0: &mut __m128i,
    c0: &mut __m128i,
    d0: &mut __m128i,
    a1: &mut __m128i,
    c1: &mut __m128i,
    d1: &mut __m128i,
    a2: &mut __m128i,
    c2: &mut __m128i,
    d2: &mut __m128i,
    a3: &mut __m128i,
    c3: &mut __m128i,
    d3: &mut __m128i,
) {
    unsafe {
        *a0 = _mm_shuffle_epi32::<0x93>(*a0);
        *c0 = _mm_shuffle_epi32::<0x39>(*c0);
        *d0 = _mm_shuffle_epi32::<0x4e>(*d0);

        *a1 = _mm_shuffle_epi32::<0x93>(*a1);
        *c1 = _mm_shuffle_epi32::<0x39>(*c1);
        *d1 = _mm_shuffle_epi32::<0x4e>(*d1);

        *a2 = _mm_shuffle_epi32::<0x93>(*a2);
        *c2 = _mm_shuffle_epi32::<0x39>(*c2);
        *d2 = _mm_shuffle_epi32::<0x4e>(*d2);

        *a3 = _mm_shuffle_epi32::<0x93>(*a3);
        *c3 = _mm_shuffle_epi32::<0x39>(*c3);
        *d3 = _mm_shuffle_epi32::<0x4e>(*d3);
    }
}

#[inline]
fn cols_to_rows(
    a0: &mut __m128i,
    c0: &mut __m128i,
    d0: &mut __m128i,
    a1: &mut __m128i,
    c1: &mut __m128i,
    d1: &mut __m128i,
    a2: &mut __m128i,
    c2: &mut __m128i,
    d2: &mut __m128i,
    a3: &mut __m128i,
    c3: &mut __m128i,
    d3: &mut __m128i,
) {
    unsafe {
        *a0 = _mm_shuffle_epi32::<0x39>(*a0);
        *c0 = _mm_shuffle_epi32::<0x93>(*c0);
        *d0 = _mm_shuffle_epi32::<0x4e>(*d0);

        *a1 = _mm_shuffle_epi32::<0x39>(*a1);
        *c1 = _mm_shuffle_epi32::<0x93>(*c1);
        *d1 = _mm_shuffle_epi32::<0x4e>(*d1);

        *a2 = _mm_shuffle_epi32::<0x39>(*a2);
        *c2 = _mm_shuffle_epi32::<0x93>(*c2);
        *d2 = _mm_shuffle_epi32::<0x4e>(*d2);

        *a3 = _mm_shuffle_epi32::<0x39>(*a3);
        *c3 = _mm_shuffle_epi32::<0x93>(*c3);
        *d3 = _mm_shuffle_epi32::<0x4e>(*d3);
    }
}
