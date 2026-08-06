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
    fn process_internal<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        todo!()
    }
}
