use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Row, Variant, Variants},
};
use core::arch::aarch64::{
    uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, veorq_u32, vextq_u32, vshrq_n_u32, vsliq_n_u32,
};

const VECTOR_WIDTH: usize = BATCH_BYTES / size_of::<NeonRow>();

#[derive(Clone, Copy)]
#[repr(C, align(64))]
union NeonRow {
    u32: [uint32x4_t; BLOCKS],
    u64: [uint64x2_t; BLOCKS],
}

pub struct Neon {
    row_a: NeonRow,
    row_b: NeonRow,
    row_c: NeonRow,
    row_d: NeonRow,
}

impl Backend for Neon {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        let row_a = broadcast(ROW_A);
        let row_b = broadcast(core.row_b);
        let row_c = broadcast(core.row_c);
        let row_d = match V::VAR {
            Variants::Djb => add_epi64(
                broadcast(core.row_d),
                setr_epi64([[0, 0], [1, 0], [2, 0], [3, 0]]),
            ),

            Variants::Ietf => add_epi32(
                broadcast(core.row_d),
                setr_epi32([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]),
            ),
        };
        Self {
            row_a,
            row_b,
            row_c,
            row_d,
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

            a = add_epi32(a, self.row_a);
            b = add_epi32(b, self.row_b);
            c = add_epi32(c, self.row_c);
            d = add_epi32(d, self.row_d);

            match V::VAR {
                Variants::Djb => {
                    self.row_d = add_epi64(
                        self.row_d,
                        setr_epi64([
                            [BLOCKS as u64, 0],
                            [BLOCKS as u64, 0],
                            [BLOCKS as u64, 0],
                            [BLOCKS as u64, 0],
                        ]),
                    );
                }
                Variants::Ietf => {
                    self.row_d = add_epi32(
                        self.row_d,
                        setr_epi32([
                            [BLOCKS as u32, 0, 0, 0],
                            [BLOCKS as u32, 0, 0, 0],
                            [BLOCKS as u32, 0, 0, 0],
                            [BLOCKS as u32, 0, 0, 0],
                        ]),
                    );
                }
            }

            let permuted = core::mem::transmute::<[NeonRow; VECTOR_WIDTH], [u8; BATCH_BYTES]>(
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
fn broadcast(row: Row) -> NeonRow {
    let row = unsafe { core::mem::transmute(row.u32x4) };
    NeonRow { u32: [row; BLOCKS] }
}

#[inline(always)]
fn setr_epi32(values: [[u32; 4]; BLOCKS]) -> NeonRow {
    unsafe {
        NeonRow {
            u32: [
                core::mem::transmute(values[0]),
                core::mem::transmute(values[1]),
                core::mem::transmute(values[2]),
                core::mem::transmute(values[3]),
            ],
        }
    }
}

#[inline(always)]
fn setr_epi64(values: [[u64; 2]; BLOCKS]) -> NeonRow {
    unsafe {
        NeonRow {
            u64: [
                core::mem::transmute(values[0]),
                core::mem::transmute(values[1]),
                core::mem::transmute(values[2]),
                core::mem::transmute(values[3]),
            ],
        }
    }
}

#[inline(always)]
fn add_epi32(a: NeonRow, b: NeonRow) -> NeonRow {
    unsafe {
        let a = a.u32;
        let b = b.u32;
        NeonRow {
            u32: [
                vaddq_u32(a[0], b[0]),
                vaddq_u32(a[1], b[1]),
                vaddq_u32(a[2], b[2]),
                vaddq_u32(a[3], b[3]),
            ],
        }
    }
}

#[inline(always)]
fn add_epi64(a: NeonRow, b: NeonRow) -> NeonRow {
    unsafe {
        let a = a.u64;
        let b = b.u64;
        NeonRow {
            u64: [
                vaddq_u64(a[0], b[0]),
                vaddq_u64(a[1], b[1]),
                vaddq_u64(a[2], b[2]),
                vaddq_u64(a[3], b[3]),
            ],
        }
    }
}

#[inline(always)]
fn xor(a: NeonRow, b: NeonRow) -> NeonRow {
    unsafe {
        let a = a.u32;
        let b = b.u32;
        NeonRow {
            u32: [
                veorq_u32(a[0], b[0]),
                veorq_u32(a[1], b[1]),
                veorq_u32(a[2], b[2]),
                veorq_u32(a[3], b[3]),
            ],
        }
    }
}

#[inline(always)]
fn rol_lane<const LEFT: i32, const RIGHT: i32>(value: uint32x4_t) -> uint32x4_t {
    unsafe { vsliq_n_u32::<LEFT>(vshrq_n_u32::<RIGHT>(value), value) }
}

#[inline(always)]
fn rol_epi32<const LEFT: i32, const RIGHT: i32>(value: NeonRow) -> NeonRow {
    unsafe {
        let value = value.u32;
        NeonRow {
            u32: [
                rol_lane::<LEFT, RIGHT>(value[0]),
                rol_lane::<LEFT, RIGHT>(value[1]),
                rol_lane::<LEFT, RIGHT>(value[2]),
                rol_lane::<LEFT, RIGHT>(value[3]),
            ],
        }
    }
}

#[inline(always)]
fn shuffle<const OFFSET: i32>(value: NeonRow) -> NeonRow {
    unsafe {
        let value = value.u32;
        NeonRow {
            u32: [
                vextq_u32::<OFFSET>(value[0], value[0]),
                vextq_u32::<OFFSET>(value[1], value[1]),
                vextq_u32::<OFFSET>(value[2], value[2]),
                vextq_u32::<OFFSET>(value[3], value[3]),
            ],
        }
    }
}

#[inline(always)]
fn permute_blocks(a: NeonRow, b: NeonRow, c: NeonRow, d: NeonRow) -> [NeonRow; VECTOR_WIDTH] {
    unsafe {
        let a = a.u32;
        let b = b.u32;
        let c = c.u32;
        let d = d.u32;
        [
            NeonRow {
                u32: [a[0], b[0], c[0], d[0]],
            },
            NeonRow {
                u32: [a[1], b[1], c[1], d[1]],
            },
            NeonRow {
                u32: [a[2], b[2], c[2], d[2]],
            },
            NeonRow {
                u32: [a[3], b[3], c[3], d[3]],
            },
        ]
    }
}

#[inline(always)]
fn double_rounds<const ROUNDS: usize>(
    a: &mut NeonRow,
    b: &mut NeonRow,
    c: &mut NeonRow,
    d: &mut NeonRow,
) {
    for _ in 0..(ROUNDS / 2) {
        add_xor_rotate(a, b, c, d);
        rows_to_cols(a, c, d);
        add_xor_rotate(a, b, c, d);
        cols_to_rows(a, c, d);
    }
}

#[inline(always)]
fn add_xor_rotate(a: &mut NeonRow, b: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    *a = add_epi32(*a, *b);
    *d = xor(*d, *a);
    *d = rol_epi32::<16, 16>(*d);

    *c = add_epi32(*c, *d);
    *b = xor(*b, *c);
    *b = rol_epi32::<12, 20>(*b);

    *a = add_epi32(*a, *b);
    *d = xor(*d, *a);
    *d = rol_epi32::<8, 24>(*d);

    *c = add_epi32(*c, *d);
    *b = xor(*b, *c);
    *b = rol_epi32::<7, 25>(*b);
}

#[inline(always)]
fn rows_to_cols(a: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    *a = shuffle::<3>(*a);
    *c = shuffle::<1>(*c);
    *d = shuffle::<2>(*d);
}

#[inline(always)]
fn cols_to_rows(a: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    *a = shuffle::<1>(*a);
    *c = shuffle::<3>(*c);
    *d = shuffle::<2>(*d);
}
