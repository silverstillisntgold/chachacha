use core::ptr::{read_unaligned, write_unaligned};

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, COLUMNS, MATRIX_SIZE, ROW_A, Variant, Variants},
};

/// A single row of a ChaCha block. Keeping the four words together mirrors the
/// SSE2 backend and gives the optimizer a natural 128-bit vectorization unit.
type Vector = [u32; COLUMNS];

const STREAM_VECTORS: usize = BATCH_BYTES / size_of::<Vector>();

pub struct Soft;

impl Backend for Soft {
    #[inline]
    fn process<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        const {
            assert!(ROUNDS > 0 && ROUNDS.is_multiple_of(2));
        }

        let (batches, remainder) = buffer.as_chunks_mut::<{ BATCH_BYTES }>();
        let batches_len = batches.len();

        unsafe {
            // ROW_A is initialized from bytes, so convert its native-endian
            // union view back into the little-endian ChaCha words.
            let row_a = from_le(ROW_A.u32x4);
            let row_b = core.row_b.u32x4;
            let row_c = core.row_c.u32x4;
            let mut d = {
                // Placed within this scope to make it obvious this is the only place it's used.
                let row_d = core.row_d.u32x4;
                match V::VAR {
                    Variants::Djb => [
                        row_d,
                        add_counter_djb(row_d, 1),
                        add_counter_djb(row_d, 2),
                        add_counter_djb(row_d, 3),
                    ],
                    Variants::Ietf => [
                        row_d,
                        add_counter_ietf(row_d, 1),
                        add_counter_ietf(row_d, 2),
                        add_counter_ietf(row_d, 3),
                    ],
                }
            };

            for dst in batches {
                let keystream = generate_batch::<ROUNDS>(row_a, row_b, row_c, d);
                store_or_xor::<XOR>(dst, keystream);
                match V::VAR {
                    Variants::Djb => {
                        for row_d in &mut d {
                            *row_d = add_counter_djb(*row_d, BLOCKS as u64);
                        }
                    }
                    Variants::Ietf => {
                        for row_d in &mut d {
                            *row_d = add_counter_ietf(*row_d, BLOCKS as u32);
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

#[cold]
#[inline(never)]
fn process_tail<const ROUNDS: usize, const XOR: bool>(
    row_a: Vector,
    row_b: Vector,
    row_c: Vector,
    d: [Vector; BLOCKS],
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
    row_a: Vector,
    row_b: Vector,
    row_c: Vector,
    base_d: [Vector; BLOCKS],
) -> [Vector; STREAM_VECTORS] {
    let mut a = [row_a; BLOCKS];
    let mut b = [row_b; BLOCKS];
    let mut c = [row_c; BLOCKS];
    let mut d = base_d;

    double_rounds::<ROUNDS>(&mut a, &mut b, &mut c, &mut d);

    for block in 0..BLOCKS {
        a[block] = add(a[block], row_a);
        b[block] = add(b[block], row_b);
        c[block] = add(c[block], row_c);
        d[block] = add(d[block], base_d[block]);
    }

    [
        a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1], a[2], b[2], c[2], d[2], a[3], b[3], c[3],
        d[3],
    ]
}

#[inline]
fn store_or_xor<const XOR: bool>(dst: &mut [u8; BATCH_BYTES], keystream: [Vector; STREAM_VECTORS]) {
    const CHUNK_SIZE: usize = size_of::<Vector>();
    let (chunks, remainder) = dst.as_chunks_mut::<CHUNK_SIZE>();
    debug_assert!(remainder.is_empty(), "there should be no remainder");

    for (chunk, stream) in chunks.iter_mut().zip(keystream) {
        unsafe {
            let ptr = chunk.as_mut_ptr().cast::<Vector>();
            let output = if XOR {
                let input = from_le(read_unaligned(ptr.cast_const()));
                xor(input, stream)
            } else {
                stream
            };
            write_unaligned(ptr, to_le(output));
        }
    }
}

#[inline]
fn double_rounds<const ROUNDS: usize>(
    a: &mut [Vector; BLOCKS],
    b: &mut [Vector; BLOCKS],
    c: &mut [Vector; BLOCKS],
    d: &mut [Vector; BLOCKS],
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
    a: &mut [Vector; BLOCKS],
    b: &mut [Vector; BLOCKS],
    c: &mut [Vector; BLOCKS],
    d: &mut [Vector; BLOCKS],
) {
    // a += b
    for block in 0..BLOCKS {
        a[block] = add(a[block], b[block]);
    }

    // d ^= a
    for block in 0..BLOCKS {
        d[block] = xor(d[block], a[block]);
    }

    // d <<<= 16
    for block in 0..BLOCKS {
        d[block] = rotate_left::<16>(d[block]);
    }

    // c += d
    for block in 0..BLOCKS {
        c[block] = add(c[block], d[block]);
    }

    // b ^= c
    for block in 0..BLOCKS {
        b[block] = xor(b[block], c[block]);
    }

    // b <<<= 12
    for block in 0..BLOCKS {
        b[block] = rotate_left::<12>(b[block]);
    }

    // a += b
    for block in 0..BLOCKS {
        a[block] = add(a[block], b[block]);
    }

    // d ^= a
    for block in 0..BLOCKS {
        d[block] = xor(d[block], a[block]);
    }

    // d <<<= 8
    for block in 0..BLOCKS {
        d[block] = rotate_left::<8>(d[block]);
    }

    // c += d
    for block in 0..BLOCKS {
        c[block] = add(c[block], d[block]);
    }

    // b ^= c
    for block in 0..BLOCKS {
        b[block] = xor(b[block], c[block]);
    }

    // b <<<= 7
    for block in 0..BLOCKS {
        b[block] = rotate_left::<7>(b[block]);
    }
}

#[inline]
fn rows_to_cols(a: &mut [Vector; BLOCKS], c: &mut [Vector; BLOCKS], d: &mut [Vector; BLOCKS]) {
    for block in 0..BLOCKS {
        // A: rotate right by one u32.
        a[block] = rotate_words_right_one(a[block]);

        // B remains unchanged.

        // C: rotate left by one u32.
        c[block] = rotate_words_left_one(c[block]);

        // D: rotate left by two u32s.
        d[block] = rotate_words_two(d[block]);
    }
}

#[inline]
fn cols_to_rows(a: &mut [Vector; BLOCKS], c: &mut [Vector; BLOCKS], d: &mut [Vector; BLOCKS]) {
    for block in 0..BLOCKS {
        // Undo A's right-one rotation with a left-one rotation.
        a[block] = rotate_words_left_one(a[block]);

        // B remains unchanged.

        // Undo C's left-one rotation with a right-one rotation.
        c[block] = rotate_words_right_one(c[block]);

        // A rotation by two is its own inverse.
        d[block] = rotate_words_two(d[block]);
    }
}

#[inline(always)]
fn add(a: Vector, b: Vector) -> Vector {
    [
        a[0].wrapping_add(b[0]),
        a[1].wrapping_add(b[1]),
        a[2].wrapping_add(b[2]),
        a[3].wrapping_add(b[3]),
    ]
}

#[inline(always)]
fn xor(a: Vector, b: Vector) -> Vector {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[inline(always)]
fn rotate_left<const ROTATION: u32>(value: Vector) -> Vector {
    [
        value[0].rotate_left(ROTATION),
        value[1].rotate_left(ROTATION),
        value[2].rotate_left(ROTATION),
        value[3].rotate_left(ROTATION),
    ]
}

#[inline(always)]
fn rotate_words_right_one(value: Vector) -> Vector {
    [value[3], value[0], value[1], value[2]]
}

#[inline(always)]
fn rotate_words_left_one(value: Vector) -> Vector {
    [value[1], value[2], value[3], value[0]]
}

#[inline(always)]
fn rotate_words_two(value: Vector) -> Vector {
    [value[2], value[3], value[0], value[1]]
}

#[inline(always)]
fn add_counter_djb(mut row_d: Vector, increment: u64) -> Vector {
    let counter = u64::from(row_d[0]) | (u64::from(row_d[1]) << 32);
    let counter = counter.wrapping_add(increment);
    row_d[0] = counter as u32;
    row_d[1] = (counter >> 32) as u32;
    row_d
}

#[inline(always)]
fn add_counter_ietf(mut row_d: Vector, increment: u32) -> Vector {
    row_d[0] = row_d[0].wrapping_add(increment);
    row_d
}

#[inline(always)]
fn from_le(value: Vector) -> Vector {
    [
        u32::from_le(value[0]),
        u32::from_le(value[1]),
        u32::from_le(value[2]),
        u32::from_le(value[3]),
    ]
}

#[inline(always)]
fn to_le(value: Vector) -> Vector {
    [
        value[0].to_le(),
        value[1].to_le(),
        value[2].to_le(),
        value[3].to_le(),
    ]
}
