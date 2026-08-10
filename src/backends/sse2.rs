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

pub struct Sse2 {
    row_a: __m128i,
    row_b: __m128i,
    row_c: __m128i,
    row_d: __m128i,
}

impl Backend for Sse2 {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        unsafe {
            let row_a = ROW_A.u128x1;
            let row_b = core.row_b.u128x1;
            let row_c = core.row_c.u128x1;
            let row_d = core.row_d.u128x1;
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
        const BLOCK_LEN: usize = BATCH_BYTES / BLOCKS;
        let (blocks, _) = buffer.as_chunks_mut::<BLOCK_LEN>();
        for block in blocks {
            unsafe {
                let mut a = self.row_a;
                let mut b = self.row_b;
                let mut c = self.row_c;
                let mut d = self.row_d;

                double_rounds::<ROUNDS>(&mut a, &mut b, &mut c, &mut d);

                a = _mm_add_epi32(a, self.row_a);
                b = _mm_add_epi32(b, self.row_b);
                c = _mm_add_epi32(c, self.row_c);
                d = _mm_add_epi32(d, self.row_d);

                match V::VAR {
                    Variants::Djb => self.row_d = _mm_add_epi64(self.row_d, _mm_set_epi64x(0, 1)),
                    Variants::Ietf => {
                        self.row_d = _mm_add_epi32(self.row_d, _mm_set_epi32(0, 0, 0, 1))
                    }
                }

                let bytes =
                    core::mem::transmute::<[__m128i; BLOCKS], [u8; BLOCK_LEN]>([a, b, c, d]);

                for i in 0..BLOCK_LEN {
                    if XOR {
                        block[i] ^= bytes[i];
                    } else {
                        block[i] = bytes[i];
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn double_rounds<const ROUNDS: usize>(
    a: &mut __m128i,
    b: &mut __m128i,
    c: &mut __m128i,
    d: &mut __m128i,
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(a, b, c, d);
        rows_to_cols(a, c, d);
        add_xor_rotate(a, b, c, d);
        cols_to_rows(a, c, d);
    }
}

#[inline(always)]
fn add_xor_rotate(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
    unsafe {
        *a = _mm_add_epi32(*a, *b);
        *d = _mm_xor_si128(*d, *a);
        *d = _mm_or_si128(_mm_slli_epi32::<16>(*d), _mm_srli_epi32::<16>(*d));

        *c = _mm_add_epi32(*c, *d);
        *b = _mm_xor_si128(*b, *c);
        *b = _mm_or_si128(_mm_slli_epi32::<12>(*b), _mm_srli_epi32::<20>(*b));

        *a = _mm_add_epi32(*a, *b);
        *d = _mm_xor_si128(*d, *a);
        *d = _mm_or_si128(_mm_slli_epi32::<8>(*d), _mm_srli_epi32::<24>(*d));

        *c = _mm_add_epi32(*c, *d);
        *b = _mm_xor_si128(*b, *c);
        *b = _mm_or_si128(_mm_slli_epi32::<7>(*b), _mm_srli_epi32::<25>(*b));
    }
}

#[inline(always)]
fn rows_to_cols(a: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
    unsafe {
        *a = _mm_shuffle_epi32::<0x93>(*a);
        *c = _mm_shuffle_epi32::<0x39>(*c);
        *d = _mm_shuffle_epi32::<0x4e>(*d);
    }
}

#[inline(always)]
fn cols_to_rows(a: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
    unsafe {
        *a = _mm_shuffle_epi32::<0x39>(*a);
        *c = _mm_shuffle_epi32::<0x93>(*c);
        *d = _mm_shuffle_epi32::<0x4e>(*d);
    }
}
