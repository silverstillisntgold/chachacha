#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, MATRIX_SIZE, ROW_A, Variant, Variants},
};
use arch::{
    __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_broadcastsi128_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_setr_epi32, _mm256_setr_epi64x,
    _mm256_shuffle_epi32, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256,
    _mm256_xor_si256,
};

const STREAM_VECTORS: usize = BATCH_BYTES / size_of::<__m256i>();

pub struct Avx2;

impl Backend for Avx2 {
    #[inline]
    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        let (batches, remainder) = buffer.as_chunks_mut::<{ BATCH_BYTES }>();
        let batches_len = batches.len();

        unsafe {
            let row_a = _mm256_broadcastsi128_si256(ROW_A.u128x1);
            let row_b = _mm256_broadcastsi128_si256(core.row_b.u128x1);
            let row_c = _mm256_broadcastsi128_si256(core.row_c.u128x1);
            let (mut d0, mut d1) = {
                // Placed within this scope to make it obvious this is the only place it's used.
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

            for dst in batches {
                let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, d0, d1);
                store_or_xor::<XOR>(dst, keystream);
                match V::VAR {
                    Variants::Djb => {
                        let increment = _mm256_setr_epi64x(BLOCKS as i64, 0, BLOCKS as i64, 0);
                        d0 = _mm256_add_epi64(d0, increment);
                        d1 = _mm256_add_epi64(d1, increment);
                    }
                    Variants::Ietf => {
                        let increment =
                            _mm256_setr_epi32(BLOCKS as i32, 0, 0, 0, BLOCKS as i32, 0, 0, 0);
                        d0 = _mm256_add_epi32(d0, increment);
                        d1 = _mm256_add_epi32(d1, increment);
                    }
                }
            }

            if !remainder.is_empty() {
                process_tail::<ROUNDS, XOR>(row_a, row_b, row_c, d0, d1, remainder);
            }
        }

        let full_blocks = batches_len * BLOCKS;
        let tail_blocks = remainder.len().div_ceil(MATRIX_SIZE);
        let consumed_blocks = full_blocks + tail_blocks;
        core.advance_blocks(consumed_blocks);
    }
}

#[inline(never)]
fn process_tail<const ROUNDS: usize, const XOR: bool>(
    row_a: __m256i,
    row_b: __m256i,
    row_c: __m256i,
    d0: __m256i,
    d1: __m256i,
    remainder: &mut [u8],
) {
    let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, d0, d1);
    let mut tail = [0; BATCH_BYTES];
    store_or_xor::<false>(&mut tail, keystream);
    for (destination, keystream) in remainder.iter_mut().zip(tail) {
        if XOR {
            *destination ^= keystream;
        } else {
            *destination = keystream;
        }
    }
}

#[inline]
fn generate_batch<const ROUNDS: usize>(
    row_a: __m256i,
    row_b: __m256i,
    row_c: __m256i,
    base_d0: __m256i,
    base_d1: __m256i,
) -> [__m256i; 8] {
    unsafe {
        let mut a0 = row_a;
        let mut b0 = row_b;
        let mut c0 = row_c;
        let mut d0 = base_d0;
        let mut a1 = row_a;
        let mut b1 = row_b;
        let mut c1 = row_c;
        let mut d1 = base_d1;

        double_rounds::<ROUNDS>(
            &mut a0, &mut b0, &mut c0, &mut d0, &mut a1, &mut b1, &mut c1, &mut d1,
        );

        a0 = _mm256_add_epi32(a0, row_a);
        b0 = _mm256_add_epi32(b0, row_b);
        c0 = _mm256_add_epi32(c0, row_c);
        d0 = _mm256_add_epi32(d0, base_d0);
        a1 = _mm256_add_epi32(a1, row_a);
        b1 = _mm256_add_epi32(b1, row_b);
        c1 = _mm256_add_epi32(c1, row_c);
        d1 = _mm256_add_epi32(d1, base_d1);

        permute_blocks(a0, b0, c0, d0, a1, b1, c1, d1)
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
) -> [__m256i; 8] {
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
fn store_or_xor<const XOR: bool>(dst: &mut [u8; BATCH_BYTES], keystream: [__m256i; 8]) {
    unsafe {
        const CHUNK_SIZE: usize = size_of::<__m256i>();
        let (chunks, remainder) = dst.as_chunks_mut::<CHUNK_SIZE>();
        debug_assert!(remainder.is_empty(), "there should be no remainder");
        for (chunk, stream) in chunks.iter_mut().zip(keystream) {
            let ptr = chunk.as_mut_ptr().cast::<__m256i>();
            let output = if XOR {
                let input = _mm256_loadu_si256(ptr.cast_const());
                _mm256_xor_si256(input, stream)
            } else {
                stream
            };
            _mm256_storeu_si256(ptr, output);
        }
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
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
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
    c0: &mut __m256i,
    d0: &mut __m256i,
    a1: &mut __m256i,
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
