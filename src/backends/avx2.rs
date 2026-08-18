#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Variant, Variants},
};
use arch::{
    __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_broadcastsi128_si256, _mm256_or_si256,
    _mm256_permute2x128_si256, _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_shuffle_epi32,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_xor_si256,
};

const VECTOR_WIDTH: usize = BATCH_BYTES / size_of::<__m256i>();

pub struct Avx2 {
    row_a: __m256i,
    row_b: __m256i,
    row_c: __m256i,
    row_d0: __m256i,
    row_d1: __m256i,
}

impl Backend for Avx2 {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        unsafe {
            let row_a = _mm256_broadcastsi128_si256(ROW_A.u128x1);
            let row_b = _mm256_broadcastsi128_si256(core.row_b.u128x1);
            let row_c = _mm256_broadcastsi128_si256(core.row_c.u128x1);
            let (row_d0, row_d1) = {
                let row_d = _mm256_broadcastsi128_si256(core.row_d.u128x1);
                match V::VAR {
                    Variants::Djb => {
                        let d0 = _mm256_add_epi64(row_d, _mm256_setr_epi64x(0, 0, 1, 0));
                        let d1 = _mm256_add_epi64(row_d, _mm256_setr_epi64x(2, 0, 3, 0));
                        (d0, d1)
                    }
                    Variants::Ietf => {
                        let d0 = _mm256_add_epi32(row_d, _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0));
                        let d1 = _mm256_add_epi32(row_d, _mm256_setr_epi32(2, 0, 0, 0, 3, 0, 0, 0));
                        (d0, d1)
                    }
                }
            };
            Self {
                row_a,
                row_b,
                row_c,
                row_d0,
                row_d1,
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

            double_rounds::<ROUNDS>(
                &mut a0, &mut b0, &mut c0, &mut d0, &mut a1, &mut b1, &mut c1, &mut d1,
            );

            a0 = _mm256_add_epi32(a0, self.row_a);
            b0 = _mm256_add_epi32(b0, self.row_b);
            c0 = _mm256_add_epi32(c0, self.row_c);
            d0 = _mm256_add_epi32(d0, self.row_d0);
            a1 = _mm256_add_epi32(a1, self.row_a);
            b1 = _mm256_add_epi32(b1, self.row_b);
            c1 = _mm256_add_epi32(c1, self.row_c);
            d1 = _mm256_add_epi32(d1, self.row_d1);

            match V::VAR {
                Variants::Djb => {
                    let increment = _mm256_setr_epi64x(BLOCKS as i64, 0, BLOCKS as i64, 0);
                    self.row_d0 = _mm256_add_epi64(self.row_d0, increment);
                    self.row_d1 = _mm256_add_epi64(self.row_d1, increment);
                }
                Variants::Ietf => {
                    let increment =
                        _mm256_setr_epi32(BLOCKS as i32, 0, 0, 0, BLOCKS as i32, 0, 0, 0);
                    self.row_d0 = _mm256_add_epi32(self.row_d0, increment);
                    self.row_d1 = _mm256_add_epi32(self.row_d1, increment);
                }
            }

            let permuted = core::mem::transmute::<[__m256i; VECTOR_WIDTH], [u8; BATCH_BYTES]>(
                permute_blocks(a0, b0, c0, d0, a1, b1, c1, d1),
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

#[inline]
fn permute_blocks(
    a0: __m256i,
    b0: __m256i,
    c0: __m256i,
    d0: __m256i,
    a1: __m256i,
    b1: __m256i,
    c1: __m256i,
    d1: __m256i,
) -> [__m256i; VECTOR_WIDTH] {
    unsafe {
        [
            _mm256_permute2x128_si256::<0x20>(a0, b0),
            _mm256_permute2x128_si256::<0x20>(c0, d0),
            _mm256_permute2x128_si256::<0x31>(a0, b0),
            _mm256_permute2x128_si256::<0x31>(c0, d0),
            _mm256_permute2x128_si256::<0x20>(a1, b1),
            _mm256_permute2x128_si256::<0x20>(c1, d1),
            _mm256_permute2x128_si256::<0x31>(a1, b1),
            _mm256_permute2x128_si256::<0x31>(c1, d1),
        ]
    }
}

#[inline]
fn double_rounds<const ROUNDS: usize>(
    a0: &mut __m256i,
    b0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    b1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(a0, b0, c0, d0, a1, b1, c1, d1);
        rows_to_cols(a0, c0, d0, a1, c1, d1);
        add_xor_rotate(a0, b0, c0, d0, a1, b1, c1, d1);
        cols_to_rows(a0, c0, d0, a1, c1, d1);
    }
}

#[inline]
fn add_xor_rotate(
    a0: &mut __m256i,
    b0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    b1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    unsafe {
        *a0 = _mm256_add_epi32(*a0, *b0);
        *a1 = _mm256_add_epi32(*a1, *b1);
        *d0 = _mm256_xor_si256(*d0, *a0);
        *d1 = _mm256_xor_si256(*d1, *a1);
        *d0 = _mm256_or_si256(_mm256_slli_epi32::<16>(*d0), _mm256_srli_epi32::<16>(*d0));
        *d1 = _mm256_or_si256(_mm256_slli_epi32::<16>(*d1), _mm256_srli_epi32::<16>(*d1));

        *c0 = _mm256_add_epi32(*c0, *d0);
        *c1 = _mm256_add_epi32(*c1, *d1);
        *b0 = _mm256_xor_si256(*b0, *c0);
        *b1 = _mm256_xor_si256(*b1, *c1);
        *b0 = _mm256_or_si256(_mm256_slli_epi32::<12>(*b0), _mm256_srli_epi32::<20>(*b0));
        *b1 = _mm256_or_si256(_mm256_slli_epi32::<12>(*b1), _mm256_srli_epi32::<20>(*b1));

        *a0 = _mm256_add_epi32(*a0, *b0);
        *a1 = _mm256_add_epi32(*a1, *b1);
        *d0 = _mm256_xor_si256(*d0, *a0);
        *d1 = _mm256_xor_si256(*d1, *a1);
        *d0 = _mm256_or_si256(_mm256_slli_epi32::<8>(*d0), _mm256_srli_epi32::<24>(*d0));
        *d1 = _mm256_or_si256(_mm256_slli_epi32::<8>(*d1), _mm256_srli_epi32::<24>(*d1));

        *c0 = _mm256_add_epi32(*c0, *d0);
        *c1 = _mm256_add_epi32(*c1, *d1);
        *b0 = _mm256_xor_si256(*b0, *c0);
        *b1 = _mm256_xor_si256(*b1, *c1);
        *b0 = _mm256_or_si256(_mm256_slli_epi32::<7>(*b0), _mm256_srli_epi32::<25>(*b0));
        *b1 = _mm256_or_si256(_mm256_slli_epi32::<7>(*b1), _mm256_srli_epi32::<25>(*b1));
    }
}

#[inline]
fn rows_to_cols(
    a0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    unsafe {
        *a0 = _mm256_shuffle_epi32::<0x93>(*a0);
        *a1 = _mm256_shuffle_epi32::<0x93>(*a1);
        *c0 = _mm256_shuffle_epi32::<0x39>(*c0);
        *c1 = _mm256_shuffle_epi32::<0x39>(*c1);
        *d0 = _mm256_shuffle_epi32::<0x4e>(*d0);
        *d1 = _mm256_shuffle_epi32::<0x4e>(*d1);
    }
}

#[inline]
fn cols_to_rows(
    a0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    unsafe {
        *a0 = _mm256_shuffle_epi32::<0x39>(*a0);
        *a1 = _mm256_shuffle_epi32::<0x39>(*a1);
        *c0 = _mm256_shuffle_epi32::<0x93>(*c0);
        *c1 = _mm256_shuffle_epi32::<0x93>(*c1);
        *d0 = _mm256_shuffle_epi32::<0x4e>(*d0);
        *d1 = _mm256_shuffle_epi32::<0x4e>(*d1);
    }
}
