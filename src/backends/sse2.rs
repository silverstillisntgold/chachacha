#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, MATRIX_SIZE, ROW_A, Variant, Variants},
};
use arch::{
    __m128i, _mm_add_epi32, _mm_add_epi64, _mm_loadu_si128, _mm_or_si128, _mm_set_epi64x,
    _mm_setr_epi32, _mm_shuffle_epi32, _mm_slli_epi32, _mm_srli_epi32, _mm_storeu_si128,
    _mm_xor_si128,
};

const STREAM_VECTORS: usize = BATCH_BYTES / size_of::<__m128i>();

pub struct Sse2;

impl Backend for Sse2 {
    #[inline]
    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        let (batches, remainder) = buffer.as_chunks_mut::<{ BATCH_BYTES }>();
        let batches_len = batches.len();

        unsafe {
            let row_a = ROW_A.u128x1;
            let row_b = core.row_b.u128x1;
            let row_c = core.row_c.u128x1;
            let mut d = {
                // Placed within this scope to make it obvious this is the only place it's used.
                let row_d = core.row_d.u128x1;
                match V::VAR {
                    Variants::Djb => [
                        row_d,
                        _mm_add_epi64(row_d, _mm_set_epi64x(0, 1)),
                        _mm_add_epi64(row_d, _mm_set_epi64x(0, 2)),
                        _mm_add_epi64(row_d, _mm_set_epi64x(0, 3)),
                    ],
                    Variants::Ietf => [
                        row_d,
                        _mm_add_epi32(row_d, _mm_setr_epi32(1, 0, 0, 0)),
                        _mm_add_epi32(row_d, _mm_setr_epi32(2, 0, 0, 0)),
                        _mm_add_epi32(row_d, _mm_setr_epi32(3, 0, 0, 0)),
                    ],
                }
            };

            for dst in batches {
                let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, d);
                store_or_xor::<XOR>(dst, keystream);
                match V::VAR {
                    Variants::Djb => {
                        let increment = _mm_set_epi64x(0, BLOCKS as i64);
                        for row_d in &mut d {
                            *row_d = _mm_add_epi64(*row_d, increment);
                        }
                    }
                    Variants::Ietf => {
                        let increment = _mm_setr_epi32(BLOCKS as i32, 0, 0, 0);
                        for row_d in &mut d {
                            *row_d = _mm_add_epi32(*row_d, increment);
                        }
                    }
                }
            }

            if !remainder.is_empty() {
                process_tail::<ROUNDS, XOR>(row_a, row_b, row_c, d, remainder);
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
    row_a: __m128i,
    row_b: __m128i,
    row_c: __m128i,
    d: [__m128i; BLOCKS],
    remainder: &mut [u8],
) {
    let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, d);
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
    row_a: __m128i,
    row_b: __m128i,
    row_c: __m128i,
    base_d: [__m128i; BLOCKS],
) -> [__m128i; STREAM_VECTORS] {
    unsafe {
        let mut a = [row_a; BLOCKS];
        let mut b = [row_b; BLOCKS];
        let mut c = [row_c; BLOCKS];
        let mut d = base_d;

        double_rounds::<ROUNDS>(&mut a, &mut b, &mut c, &mut d);

        for block in 0..BLOCKS {
            a[block] = _mm_add_epi32(a[block], row_a);
            b[block] = _mm_add_epi32(b[block], row_b);
            c[block] = _mm_add_epi32(c[block], row_c);
            d[block] = _mm_add_epi32(d[block], base_d[block]);
        }

        [
            a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1], a[2], b[2], c[2], d[2], a[3], b[3],
            c[3], d[3],
        ]
    }
}

#[inline]
fn store_or_xor<const XOR: bool>(
    dst: &mut [u8; BATCH_BYTES],
    keystream: [__m128i; STREAM_VECTORS],
) {
    unsafe {
        const CHUNK_SIZE: usize = size_of::<__m128i>();
        let (chunks, remainder) = dst.as_chunks_mut::<CHUNK_SIZE>();
        debug_assert!(remainder.is_empty(), "there should be no remainder");
        for (chunk, stream) in chunks.iter_mut().zip(keystream) {
            let ptr = chunk.as_mut_ptr().cast::<__m128i>();
            let output = if XOR {
                let input = _mm_loadu_si128(ptr.cast_const());
                _mm_xor_si128(input, stream)
            } else {
                stream
            };
            _mm_storeu_si128(ptr, output);
        }
    }
}

#[inline]
fn double_rounds<const ROUNDS: usize>(
    a: &mut [__m128i; BLOCKS],
    b: &mut [__m128i; BLOCKS],
    c: &mut [__m128i; BLOCKS],
    d: &mut [__m128i; BLOCKS],
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(a, b, c, d);
        rows_to_cols(a, c, d);
        add_xor_rotate(a, b, c, d);
        cols_to_rows(a, c, d);
    }
}

#[inline]
fn add_xor_rotate(
    a: &mut [__m128i; BLOCKS],
    b: &mut [__m128i; BLOCKS],
    c: &mut [__m128i; BLOCKS],
    d: &mut [__m128i; BLOCKS],
) {
    unsafe {
        // a += b
        for block in 0..BLOCKS {
            a[block] = _mm_add_epi32(a[block], b[block]);
        }

        // d ^= a
        for block in 0..BLOCKS {
            d[block] = _mm_xor_si128(d[block], a[block]);
        }

        // d <<<= 16
        for block in 0..BLOCKS {
            d[block] = _mm_or_si128(
                _mm_slli_epi32::<16>(d[block]),
                _mm_srli_epi32::<16>(d[block]),
            );
        }

        // c += d
        for block in 0..BLOCKS {
            c[block] = _mm_add_epi32(c[block], d[block]);
        }

        // b ^= c
        for block in 0..BLOCKS {
            b[block] = _mm_xor_si128(b[block], c[block]);
        }

        // b <<<= 12
        for block in 0..BLOCKS {
            b[block] = _mm_or_si128(
                _mm_slli_epi32::<12>(b[block]),
                _mm_srli_epi32::<20>(b[block]),
            );
        }

        // a += b
        for block in 0..BLOCKS {
            a[block] = _mm_add_epi32(a[block], b[block]);
        }

        // d ^= a
        for block in 0..BLOCKS {
            d[block] = _mm_xor_si128(d[block], a[block]);
        }

        // d <<<= 8
        for block in 0..BLOCKS {
            d[block] = _mm_or_si128(
                _mm_slli_epi32::<8>(d[block]),
                _mm_srli_epi32::<24>(d[block]),
            );
        }

        // c += d
        for block in 0..BLOCKS {
            c[block] = _mm_add_epi32(c[block], d[block]);
        }

        // b ^= c
        for block in 0..BLOCKS {
            b[block] = _mm_xor_si128(b[block], c[block]);
        }

        // b <<<= 7
        for block in 0..BLOCKS {
            b[block] = _mm_or_si128(
                _mm_slli_epi32::<7>(b[block]),
                _mm_srli_epi32::<25>(b[block]),
            );
        }
    }
}

#[inline]
fn rows_to_cols(a: &mut [__m128i; BLOCKS], c: &mut [__m128i; BLOCKS], d: &mut [__m128i; BLOCKS]) {
    unsafe {
        for block in 0..BLOCKS {
            // A: rotate right by one u32.
            a[block] = _mm_shuffle_epi32::<0x93>(a[block]);

            // B remains unchanged.

            // C: rotate left by one u32.
            c[block] = _mm_shuffle_epi32::<0x39>(c[block]);

            // D: rotate left by two u32s.
            d[block] = _mm_shuffle_epi32::<0x4e>(d[block]);
        }
    }
}

#[inline]
fn cols_to_rows(a: &mut [__m128i; BLOCKS], c: &mut [__m128i; BLOCKS], d: &mut [__m128i; BLOCKS]) {
    unsafe {
        for block in 0..BLOCKS {
            // Undo A's right-one rotation with a left-one rotation.
            a[block] = _mm_shuffle_epi32::<0x39>(a[block]);

            // B remains unchanged.

            // Undo C's left-one rotation with a right-one rotation.
            c[block] = _mm_shuffle_epi32::<0x93>(c[block]);

            // A rotation by two is its own inverse.
            d[block] = _mm_shuffle_epi32::<0x4e>(d[block]);
        }
    }
}
