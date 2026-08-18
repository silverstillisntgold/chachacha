use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Row, Variant, Variants},
};
use core::arch::aarch64::{
    uint32x4_t, uint32x4x4_t, uint64x2_t, uint64x2x4_t, vaddq_u32, vaddq_u64, veorq_u32, vextq_u32,
    vorrq_u32, vshlq_n_u32, vshrq_n_u32,
};

const VECTOR_WIDTH: usize = BATCH_BYTES / size_of::<NeonRow>();

#[derive(Clone, Copy)]
union NeonRow {
    u32: uint32x4x4_t,
    u64: uint64x2x4_t,
}

impl NeonRow {
    #[inline]
    fn add_epi32(&mut self, other: Self) {
        unsafe {
            self.u32.0 = vaddq_u32(self.u32.0, other.u32.0);
            self.u32.1 = vaddq_u32(self.u32.1, other.u32.1);
            self.u32.2 = vaddq_u32(self.u32.2, other.u32.2);
            self.u32.3 = vaddq_u32(self.u32.3, other.u32.3);
        }
    }

    #[inline]
    fn add_epi64(&mut self, other: Self) {
        unsafe {
            self.u64.0 = vaddq_u64(self.u64.0, other.u64.0);
            self.u64.1 = vaddq_u64(self.u64.1, other.u64.1);
            self.u64.2 = vaddq_u64(self.u64.2, other.u64.2);
            self.u64.3 = vaddq_u64(self.u64.3, other.u64.3);
        }
    }

    #[inline]
    fn broadcast_u32(value: uint32x4_t) -> Self {
        Self {
            u32: uint32x4x4_t(value, value, value, value),
        }
    }

    #[inline]
    fn broadcast_u64(value: uint64x2_t) -> Self {
        Self {
            u64: uint64x2x4_t(value, value, value, value),
        }
    }

    #[inline]
    fn rotl_epi32<const LEFT: i32, const RIGHT: i32>(&mut self) {
        unsafe {
            self.u32.0 = vorrq_u32(
                vshlq_n_u32::<LEFT>(self.u32.0),
                vshrq_n_u32::<RIGHT>(self.u32.0),
            );
            self.u32.1 = vorrq_u32(
                vshlq_n_u32::<LEFT>(self.u32.1),
                vshrq_n_u32::<RIGHT>(self.u32.1),
            );
            self.u32.2 = vorrq_u32(
                vshlq_n_u32::<LEFT>(self.u32.2),
                vshrq_n_u32::<RIGHT>(self.u32.2),
            );
            self.u32.3 = vorrq_u32(
                vshlq_n_u32::<LEFT>(self.u32.3),
                vshrq_n_u32::<RIGHT>(self.u32.3),
            );
        }
    }

    #[inline]
    fn shuffle<const SHUFFLE: i32>(&mut self) {
        unsafe {
            self.u32.0 = vextq_u32::<SHUFFLE>(self.u32.0, self.u32.0);
            self.u32.1 = vextq_u32::<SHUFFLE>(self.u32.1, self.u32.1);
            self.u32.2 = vextq_u32::<SHUFFLE>(self.u32.2, self.u32.2);
            self.u32.3 = vextq_u32::<SHUFFLE>(self.u32.3, self.u32.3);
        }
    }

    #[inline]
    fn xor(&mut self, other: Self) {
        unsafe {
            self.u32.0 = veorq_u32(self.u32.0, other.u32.0);
            self.u32.1 = veorq_u32(self.u32.1, other.u32.1);
            self.u32.2 = veorq_u32(self.u32.2, other.u32.2);
            self.u32.3 = veorq_u32(self.u32.3, other.u32.3);
        }
    }
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
        unsafe {
            let row_a = NeonRow::broadcast_u32(ROW_A.u32x4_neon);
            let row_b = NeonRow::broadcast_u32(core.row_b.u32x4_neon);
            let row_c = NeonRow::broadcast_u32(core.row_c.u32x4_neon);
            let row_d = match V::VAR {
                Variants::Djb => {
                    let mut d1 = core.row_d;
                    d1.u64x2[0] = d1.u64x2[0].wrapping_add(1);
                    let mut d2 = core.row_d;
                    d2.u64x2[0] = d2.u64x2[0].wrapping_add(2);
                    let mut d3 = core.row_d;
                    d3.u64x2[0] = d3.u64x2[0].wrapping_add(3);
                    NeonRow {
                        u64: uint64x2x4_t(
                            core.row_d.u64x2_neon,
                            d1.u64x2_neon,
                            d2.u64x2_neon,
                            d3.u64x2_neon,
                        ),
                    }
                }
                Variants::Ietf => {
                    let mut d1 = core.row_d;
                    d1.u32x4[0] = d1.u32x4[0].wrapping_add(1);
                    let mut d2 = core.row_d;
                    d2.u32x4[0] = d2.u32x4[0].wrapping_add(2);
                    let mut d3 = core.row_d;
                    d3.u32x4[0] = d3.u32x4[0].wrapping_add(3);
                    NeonRow {
                        u32: uint32x4x4_t(
                            core.row_d.u32x4_neon,
                            d1.u32x4_neon,
                            d2.u32x4_neon,
                            d3.u32x4_neon,
                        ),
                    }
                }
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

            a.add_epi32(self.row_a);
            b.add_epi32(self.row_b);
            c.add_epi32(self.row_c);
            d.add_epi32(self.row_d);

            match V::VAR {
                Variants::Djb => {
                    let increment = Row {
                        u64x2: [BLOCKS as u64, 0],
                    };
                    self.row_d
                        .add_epi64(NeonRow::broadcast_u64(increment.u64x2_neon));
                }
                Variants::Ietf => {
                    let increment = Row {
                        u32x4: [BLOCKS as u32, 0, 0, 0],
                    };
                    self.row_d
                        .add_epi32(NeonRow::broadcast_u32(increment.u32x4_neon));
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

#[inline]
fn permute_blocks(a: NeonRow, b: NeonRow, c: NeonRow, d: NeonRow) -> [NeonRow; VECTOR_WIDTH] {
    unsafe {
        [
            NeonRow {
                u32: uint32x4x4_t(a.u32.0, b.u32.0, c.u32.0, d.u32.0),
            },
            NeonRow {
                u32: uint32x4x4_t(a.u32.1, b.u32.1, c.u32.1, d.u32.1),
            },
            NeonRow {
                u32: uint32x4x4_t(a.u32.2, b.u32.2, c.u32.2, d.u32.2),
            },
            NeonRow {
                u32: uint32x4x4_t(a.u32.3, b.u32.3, c.u32.3, d.u32.3),
            },
        ]
    }
}

#[inline]
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

#[inline]
fn add_xor_rotate(a: &mut NeonRow, b: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    a.add_epi32(*b);
    d.xor(*a);
    d.rotl_epi32::<16, 16>();

    c.add_epi32(*d);
    b.xor(*c);
    b.rotl_epi32::<12, 20>();

    a.add_epi32(*b);
    d.xor(*a);
    d.rotl_epi32::<8, 24>();

    c.add_epi32(*d);
    b.xor(*c);
    b.rotl_epi32::<7, 25>();
}

#[inline]
fn rows_to_cols(a: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    a.shuffle::<3>();
    c.shuffle::<1>();
    d.shuffle::<2>();
}

#[inline]
fn cols_to_rows(a: &mut NeonRow, c: &mut NeonRow, d: &mut NeonRow) {
    a.shuffle::<1>();
    c.shuffle::<3>();
    d.shuffle::<2>();
}
