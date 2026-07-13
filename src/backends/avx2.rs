#![allow(clippy::too_many_arguments)]

use crate::util::{Backend, MATRIX_SIZE, ROW_A, Variant, Variants};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_broadcastsi128_si256, _mm256_extract_epi32,
    _mm256_extract_epi64, _mm256_loadu_si256, _mm256_or_si256, _mm256_permute2x128_si256,
    _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_shuffle_epi32, _mm256_slli_epi32,
    _mm256_srli_epi32, _mm256_storeu_si256, _mm256_xor_si256,
};

pub struct Avx2;

impl Backend for Avx2 {
    const BLOCKS: usize = 4;

    #[inline]
    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut crate::chacha::ChaChaCore<Self, ROUNDS, V>,
        mut buffer: &mut [u8],
    ) {
        const {
            assert!(ROUNDS.is_multiple_of(2));
        }
        unsafe {
            let base_row_a = _mm256_broadcastsi128_si256(ROW_A.u128x1);
            let base_row_b = _mm256_broadcastsi128_si256(core.row_b.u128x1);
            let base_row_c = _mm256_broadcastsi128_si256(core.row_c.u128x1);
            let mut base_row_d = _mm256_broadcastsi128_si256(core.row_d.u128x1);

            let _x = buffer.as_chunks_mut::<{ Self::BATCH_BYTES }>();

            while buffer.len() >= Self::BATCH_BYTES {
                let (cur_batch, remainder) = buffer.split_at_mut(Self::BATCH_BYTES);

                let (base_d0, base_d1) = match V::VAR {
                    Variants::Djb => {
                        let d0 = _mm256_add_epi64(base_row_d, _mm256_setr_epi64x(0, 0, 1, 0));
                        let d1 = _mm256_add_epi64(base_row_d, _mm256_setr_epi64x(2, 0, 3, 0));
                        (d0, d1)
                    }
                    Variants::Ietf => {
                        let d0 =
                            _mm256_add_epi32(base_row_d, _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0));
                        let d1 =
                            _mm256_add_epi32(base_row_d, _mm256_setr_epi32(2, 0, 0, 0, 3, 0, 0, 0));
                        (d0, d1)
                    }
                };

                let mut a0 = base_row_a;
                let mut b0 = base_row_b;
                let mut c0 = base_row_c;
                let mut d0 = base_d0;
                let mut a1 = base_row_a;
                let mut b1 = base_row_b;
                let mut c1 = base_row_c;
                let mut d1 = base_d1;

                double_rounds::<ROUNDS>(
                    &mut a0, &mut b0, &mut c0, &mut d0, &mut a1, &mut b1, &mut c1, &mut d1,
                );

                a0 = _mm256_add_epi32(a0, base_row_a);
                b0 = _mm256_add_epi32(b0, base_row_b);
                c0 = _mm256_add_epi32(c0, base_row_c);
                d0 = _mm256_add_epi32(d0, base_d0);
                a1 = _mm256_add_epi32(a1, base_row_a);
                b1 = _mm256_add_epi32(b1, base_row_b);
                c1 = _mm256_add_epi32(c1, base_row_c);
                d1 = _mm256_add_epi32(d1, base_d1);

                let block0_ab = _mm256_permute2x128_si256::<0x20>(a0, b0);
                let block0_cd = _mm256_permute2x128_si256::<0x20>(c0, d0);
                let block1_ab = _mm256_permute2x128_si256::<0x31>(a0, b0);
                let block1_cd = _mm256_permute2x128_si256::<0x31>(c0, d0);
                let block2_ab = _mm256_permute2x128_si256::<0x20>(a1, b1);
                let block2_cd = _mm256_permute2x128_si256::<0x20>(c1, d1);
                let block3_ab = _mm256_permute2x128_si256::<0x31>(a1, b1);
                let block3_cd = _mm256_permute2x128_si256::<0x31>(c1, d1);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(0), block0_ab);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(32), block0_cd);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(64), block1_ab);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(96), block1_cd);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(128), block2_ab);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(160), block2_cd);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(192), block3_ab);
                store_or_xor::<XOR>(cur_batch.as_mut_ptr().add(224), block3_cd);

                base_row_d = match V::VAR {
                    Variants::Djb => _mm256_add_epi64(
                        base_row_d,
                        _mm256_setr_epi64x(Self::BLOCKS as i64, 0, Self::BLOCKS as i64, 0),
                    ),
                    Variants::Ietf => _mm256_add_epi32(
                        base_row_d,
                        _mm256_setr_epi32(
                            Self::BLOCKS as i32,
                            0,
                            0,
                            0,
                            Self::BLOCKS as i32,
                            0,
                            0,
                            0,
                        ),
                    ),
                };

                buffer = remainder;
            }

            if !buffer.is_empty() {
                let (base_d0, base_d1) = match V::VAR {
                    Variants::Djb => {
                        let d0 = _mm256_add_epi64(base_row_d, _mm256_setr_epi64x(0, 0, 1, 0));
                        let d1 = _mm256_add_epi64(base_row_d, _mm256_setr_epi64x(2, 0, 3, 0));
                        (d0, d1)
                    }
                    Variants::Ietf => {
                        let d0 =
                            _mm256_add_epi32(base_row_d, _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0));
                        let d1 =
                            _mm256_add_epi32(base_row_d, _mm256_setr_epi32(2, 0, 0, 0, 3, 0, 0, 0));
                        (d0, d1)
                    }
                };

                let mut a0 = base_row_a;
                let mut b0 = base_row_b;
                let mut c0 = base_row_c;
                let mut d0 = base_d0;
                let mut a1 = base_row_a;
                let mut b1 = base_row_b;
                let mut c1 = base_row_c;
                let mut d1 = base_d1;

                double_rounds::<ROUNDS>(
                    &mut a0, &mut b0, &mut c0, &mut d0, &mut a1, &mut b1, &mut c1, &mut d1,
                );

                a0 = _mm256_add_epi32(a0, base_row_a);
                b0 = _mm256_add_epi32(b0, base_row_b);
                c0 = _mm256_add_epi32(c0, base_row_c);
                d0 = _mm256_add_epi32(d0, base_d0);
                a1 = _mm256_add_epi32(a1, base_row_a);
                b1 = _mm256_add_epi32(b1, base_row_b);
                c1 = _mm256_add_epi32(c1, base_row_c);
                d1 = _mm256_add_epi32(d1, base_d1);

                let block0_ab = _mm256_permute2x128_si256::<0x20>(a0, b0);
                let block0_cd = _mm256_permute2x128_si256::<0x20>(c0, d0);
                let block1_ab = _mm256_permute2x128_si256::<0x31>(a0, b0);
                let block1_cd = _mm256_permute2x128_si256::<0x31>(c0, d0);
                let block2_ab = _mm256_permute2x128_si256::<0x20>(a1, b1);
                let block2_cd = _mm256_permute2x128_si256::<0x20>(c1, d1);
                let block3_ab = _mm256_permute2x128_si256::<0x31>(a1, b1);
                let block3_cd = _mm256_permute2x128_si256::<0x31>(c1, d1);
                let mut tail = [0u8; Self::BATCH_BYTES];
                let tail_ptr = tail.as_mut_ptr();
                store_or_xor::<false>(tail_ptr.add(0), block0_ab);
                store_or_xor::<false>(tail_ptr.add(32), block0_cd);
                store_or_xor::<false>(tail_ptr.add(64), block1_ab);
                store_or_xor::<false>(tail_ptr.add(96), block1_cd);
                store_or_xor::<false>(tail_ptr.add(128), block2_ab);
                store_or_xor::<false>(tail_ptr.add(160), block2_cd);
                store_or_xor::<false>(tail_ptr.add(192), block3_ab);
                store_or_xor::<false>(tail_ptr.add(224), block3_cd);

                if XOR {
                    for (destination, keystream) in buffer.iter_mut().zip(tail.iter()) {
                        *destination ^= *keystream;
                    }
                } else {
                    buffer.copy_from_slice(&tail[..buffer.len()]);
                }
            }

            let tail_blocks = buffer.len().div_ceil(MATRIX_SIZE);
            match V::VAR {
                Variants::Djb => {
                    let counter = _mm256_extract_epi64::<0>(base_row_d) as u64;
                    core.row_d.u64x2[0] = counter.wrapping_add(tail_blocks as u64);
                }

                Variants::Ietf => {
                    let counter = _mm256_extract_epi32::<0>(base_row_d) as u32;
                    core.row_d.u32x4[0] = counter.wrapping_add(tail_blocks as u32);
                }
            }
        }
    }
}

#[inline]
fn store_or_xor<const XOR: bool>(dst: *mut u8, keystream: __m256i) {
    unsafe {
        let output = if XOR {
            let input = _mm256_loadu_si256(dst.cast());
            _mm256_xor_si256(input, keystream)
        } else {
            keystream
        };
        _mm256_storeu_si256(dst.cast(), output);
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
        rows_to_cols(a0, b0, c0, d0, a1, b1, c1, d1);
        add_xor_rotate(a0, b0, c0, d0, a1, b1, c1, d1);
        cols_to_rows(a0, b0, c0, d0, a1, b1, c1, d1);
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
        // a += b
        *a0 = _mm256_add_epi32(*a0, *b0);
        *a1 = _mm256_add_epi32(*a1, *b1);

        // d ^= a
        *d0 = _mm256_xor_si256(*d0, *a0);
        *d1 = _mm256_xor_si256(*d1, *a1);

        // d <<<= 16
        *d0 = _mm256_or_si256(_mm256_slli_epi32::<16>(*d0), _mm256_srli_epi32::<16>(*d0));
        *d1 = _mm256_or_si256(_mm256_slli_epi32::<16>(*d1), _mm256_srli_epi32::<16>(*d1));

        // c += d
        *c0 = _mm256_add_epi32(*c0, *d0);
        *c1 = _mm256_add_epi32(*c1, *d1);

        // b ^= c
        *b0 = _mm256_xor_si256(*b0, *c0);
        *b1 = _mm256_xor_si256(*b1, *c1);

        // b <<<= 12
        *b0 = _mm256_or_si256(_mm256_slli_epi32::<12>(*b0), _mm256_srli_epi32::<20>(*b0));
        *b1 = _mm256_or_si256(_mm256_slli_epi32::<12>(*b1), _mm256_srli_epi32::<20>(*b1));

        // a += b
        *a0 = _mm256_add_epi32(*a0, *b0);
        *a1 = _mm256_add_epi32(*a1, *b1);

        // d ^= a
        *d0 = _mm256_xor_si256(*d0, *a0);
        *d1 = _mm256_xor_si256(*d1, *a1);

        // d <<<= 8
        *d0 = _mm256_or_si256(_mm256_slli_epi32::<8>(*d0), _mm256_srli_epi32::<24>(*d0));
        *d1 = _mm256_or_si256(_mm256_slli_epi32::<8>(*d1), _mm256_srli_epi32::<24>(*d1));

        // c += d
        *c0 = _mm256_add_epi32(*c0, *d0);
        *c1 = _mm256_add_epi32(*c1, *d1);

        // b ^= c
        *b0 = _mm256_xor_si256(*b0, *c0);
        *b1 = _mm256_xor_si256(*b1, *c1);

        // b <<<= 7
        *b0 = _mm256_or_si256(_mm256_slli_epi32::<7>(*b0), _mm256_srli_epi32::<25>(*b0));
        *b1 = _mm256_or_si256(_mm256_slli_epi32::<7>(*b1), _mm256_srli_epi32::<25>(*b1));
    }
}

#[inline]
fn rows_to_cols(
    a0: &mut __m256i,
    _b0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    _b1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    unsafe {
        // A: rotate right by one u32.
        *a0 = _mm256_shuffle_epi32::<0x93>(*a0);
        *a1 = _mm256_shuffle_epi32::<0x93>(*a1);

        // B remains unchanged.

        // C: rotate left by one u32.
        *c0 = _mm256_shuffle_epi32::<0x39>(*c0);
        *c1 = _mm256_shuffle_epi32::<0x39>(*c1);

        // D: rotate left by two u32s.
        *d0 = _mm256_shuffle_epi32::<0x4e>(*d0);
        *d1 = _mm256_shuffle_epi32::<0x4e>(*d1);
    }
}

#[inline]
fn cols_to_rows(
    a0: &mut __m256i,
    _b0: &mut __m256i,
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
    _b1: &mut __m256i,
    c1: &mut __m256i,
    d1: &mut __m256i,
) {
    unsafe {
        // Undo A's right-one rotation with a left-one rotation.
        *a0 = _mm256_shuffle_epi32::<0x39>(*a0);
        *a1 = _mm256_shuffle_epi32::<0x39>(*a1);

        // B remains unchanged.

        // Undo C's left-one rotation with a right-one rotation.
        *c0 = _mm256_shuffle_epi32::<0x93>(*c0);
        *c1 = _mm256_shuffle_epi32::<0x93>(*c1);

        // A rotation by two is its own inverse.
        *d0 = _mm256_shuffle_epi32::<0x4e>(*d0);
        *d1 = _mm256_shuffle_epi32::<0x4e>(*d1);
    }
}
