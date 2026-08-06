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
    fn process_internal<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        todo!()
    }
}
