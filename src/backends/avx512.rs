#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, MATRIX_SIZE, ROW_A, Variant, Variants},
};
use arch::{
    __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_broadcast_i32x4, _mm512_loadu_si512,
    _mm512_rol_epi32, _mm512_setr_epi32, _mm512_setr_epi64, _mm512_shuffle_epi32,
    _mm512_shuffle_i32x4, _mm512_storeu_si512, _mm512_xor_si512,
};

const STREAM_VECTORS: usize = BATCH_BYTES / size_of::<__m512i>();

pub struct Avx512;

impl Backend for Avx512 {
    #[inline]
    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        let (batches, remainder) = buffer.as_chunks_mut::<{ BATCH_BYTES }>();
        let batches_len = batches.len();

        unsafe {
            let row_a = _mm512_broadcast_i32x4(ROW_A.u128x1);
            let row_b = _mm512_broadcast_i32x4(core.row_b.u128x1);
            let row_c = _mm512_broadcast_i32x4(core.row_c.u128x1);
            let mut row_d = {
                // Placed within this scope to make it obvious this is the only place it's used.
                let row_d = _mm512_broadcast_i32x4(core.row_d.u128x1);
                match V::VAR {
                    Variants::Djb => {
                        _mm512_add_epi64(row_d, _mm512_setr_epi64(0, 0, 1, 0, 2, 0, 3, 0))
                    }
                    Variants::Ietf => _mm512_add_epi32(
                        row_d,
                        _mm512_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0),
                    ),
                }
            };

            for dst in batches {
                let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, row_d);
                store_or_xor::<XOR>(dst, keystream);
                match V::VAR {
                    Variants::Djb => {
                        let increment = _mm512_setr_epi64(
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                            BLOCKS as i64,
                            0,
                        );
                        row_d = _mm512_add_epi64(row_d, increment);
                    }
                    Variants::Ietf => {
                        let increment = _mm512_setr_epi32(
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
                        );
                        row_d = _mm512_add_epi32(row_d, increment);
                    }
                }
            }

            if !remainder.is_empty() {
                process_tail::<ROUNDS, XOR>(row_a, row_b, row_c, row_d, remainder);
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
    row_a: __m512i,
    row_b: __m512i,
    row_c: __m512i,
    row_d: __m512i,
    remainder: &mut [u8],
) {
    let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, row_d);
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
    row_a: __m512i,
    row_b: __m512i,
    row_c: __m512i,
    base_row_d: __m512i,
) -> [__m512i; STREAM_VECTORS] {
    unsafe {
        let mut a = row_a;
        let mut b = row_b;
        let mut c = row_c;
        let mut d = base_row_d;

        double_rounds::<ROUNDS>(&mut a, &mut b, &mut c, &mut d);

        a = _mm512_add_epi32(a, row_a);
        b = _mm512_add_epi32(b, row_b);
        c = _mm512_add_epi32(c, row_c);
        d = _mm512_add_epi32(d, base_row_d);

        permute_blocks(a, b, c, d)
    }
}

#[inline]
fn permute_blocks(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; STREAM_VECTORS] {
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

#[inline]
fn store_or_xor<const XOR: bool>(
    dst: &mut [u8; BATCH_BYTES],
    keystream: [__m512i; STREAM_VECTORS],
) {
    unsafe {
        const CHUNK_SIZE: usize = size_of::<__m512i>();
        let (chunks, remainder) = dst.as_chunks_mut::<CHUNK_SIZE>();
        debug_assert!(remainder.is_empty(), "there should be no remainder");
        for (chunk, stream) in chunks.iter_mut().zip(keystream) {
            let ptr = chunk.as_mut_ptr().cast::<__m512i>();
            let output = if XOR {
                let input = _mm512_loadu_si512(ptr.cast_const());
                _mm512_xor_si512(input, stream)
            } else {
                stream
            };
            _mm512_storeu_si512(ptr, output);
        }
    }
}

#[inline]
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

#[inline]
fn add_xor_rotate(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        // a += b
        *a = _mm512_add_epi32(*a, *b);

        // d ^= a
        *d = _mm512_xor_si512(*d, *a);

        // d <<<= 16
        *d = _mm512_rol_epi32::<16>(*d);

        // c += d
        *c = _mm512_add_epi32(*c, *d);

        // b ^= c
        *b = _mm512_xor_si512(*b, *c);

        // b <<<= 12
        *b = _mm512_rol_epi32::<12>(*b);

        // a += b
        *a = _mm512_add_epi32(*a, *b);

        // d ^= a
        *d = _mm512_xor_si512(*d, *a);

        // d <<<= 8
        *d = _mm512_rol_epi32::<8>(*d);

        // c += d
        *c = _mm512_add_epi32(*c, *d);

        // b ^= c
        *b = _mm512_xor_si512(*b, *c);

        // b <<<= 7
        *b = _mm512_rol_epi32::<7>(*b);
    }
}

#[inline]
fn rows_to_cols(a: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        // A: rotate right by one u32.
        *a = _mm512_shuffle_epi32::<0x93>(*a);

        // B remains unchanged.

        // C: rotate left by one u32.
        *c = _mm512_shuffle_epi32::<0x39>(*c);

        // D: rotate left by two u32s.
        *d = _mm512_shuffle_epi32::<0x4e>(*d);
    }
}

#[inline]
fn cols_to_rows(a: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
    unsafe {
        // Undo A's right-one rotation with a left-one rotation.
        *a = _mm512_shuffle_epi32::<0x39>(*a);

        // B remains unchanged.

        // Undo C's left-one rotation with a right-one rotation.
        *c = _mm512_shuffle_epi32::<0x93>(*c);

        // A rotation by two is its own inverse.
        *d = _mm512_shuffle_epi32::<0x4e>(*d);
    }
}
