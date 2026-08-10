// #[cfg(target_arch = "x86")]
// use core::arch::x86 as arch;
// #[cfg(target_arch = "x86_64")]
// use core::arch::x86_64 as arch;

use crate::{
    backends::soft::Soft,
    chacha::ChaChaCore,
    util::{BATCH_BYTES, Backend, Variant},
};
// use arch::{
//     __m128i, _mm_add_epi32, _mm_add_epi64, _mm_loadu_si128, _mm_or_si128, _mm_set_epi64x,
//     _mm_setr_epi32, _mm_shuffle_epi32, _mm_slli_epi32, _mm_srli_epi32, _mm_storeu_si128,
//     _mm_xor_si128,
// };

pub struct Sse2 {
    // row_a: Vector,
    // row_b: Vector,
    // row_c: Vector,
    // row_d0: Vector,
    // row_d1: Vector,
    inner: Soft,
}

impl Backend for Sse2 {
    #[inline(always)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        Self {
            inner: Soft::new(core),
        }
    }

    #[inline(always)]
    fn fill<B: Backend, const ROUNDS: usize, V: Variant, const XOR: bool>(
        &mut self,
        buffer: &mut [u8; BATCH_BYTES],
    ) {
        self.inner.fill::<Soft, ROUNDS, V, XOR>(buffer);
    }
}
