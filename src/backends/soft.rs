#![allow(clippy::missing_transmute_annotations)]

use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Row, Variant, Variants},
};

const VECTOR_SIZE: usize = 2;

#[derive(Clone)]
#[repr(C, align(32))]
struct Vector([Row; VECTOR_SIZE]);

impl From<[Row; VECTOR_SIZE]> for Vector {
    #[inline]
    fn from(value: [Row; VECTOR_SIZE]) -> Self {
        Self(value)
    }
}

pub struct Soft {
    // row_a: Vector,
    // row_b: Vector,
    // row_c: Vector,
    // row_d0: Vector,
    // row_d1: Vector,
    inner: super::avx2::Avx2,
}

impl Backend for Soft {
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        Self {
            inner: super::avx2::Avx2::new(core),
        }
    }

    fn fill<B: Backend, const ROUNDS: usize, V: Variant, const XOR: bool>(
        &mut self,
        buffer: &mut [u8; BATCH_BYTES],
    ) {
        self.inner.fill::<super::avx2::Avx2, ROUNDS, V, XOR>(buffer);
    }
}
