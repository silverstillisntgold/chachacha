use crate::util::*;
use core::marker::PhantomData;
use core::mem::transmute;

#[repr(C)]
pub struct ChaChaCore<B, const ROUNDS: usize, V> {
    pub(crate) row_b: Row,
    pub(crate) row_c: Row,
    pub(crate) row_d: Row,
    _pd: PhantomData<(B, V)>,
}

impl<B, const ROUNDS: usize> ChaChaCore<B, ROUNDS, Djb>
where
    B: Backend,
{
    pub fn new(key: [u32; 8], nonce: [u32; 2]) -> Self {
        Self::new_with_counter(key, 0, nonce)
    }

    pub fn new_with_counter(key: [u32; 8], counter: u64, nonce: [u32; 2]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }

    pub fn seek(&mut self, position: u64) {
        unsafe {
            self.row_d.u64x2[0] = position;
        }
    }

    pub fn position(&self) -> u64 {
        unsafe { self.row_d.u64x2[0] }
    }
}

impl<B, const ROUNDS: usize> ChaChaCore<B, ROUNDS, Ietf>
where
    B: Backend,
{
    pub fn new(key: [u32; 8], nonce: [u32; 3]) -> Self {
        Self::new_with_counter(key, 0, nonce)
    }

    pub fn new_with_counter(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }

    pub fn seek(&mut self, position: u32) {
        unsafe {
            self.row_d.u32x4[0] = position;
        }
    }

    pub fn position(&self) -> u32 {
        unsafe { self.row_d.u32x4[0] }
    }
}

impl<B, const ROUNDS: usize, V> ChaChaCore<B, ROUNDS, V>
where
    B: Backend,
    V: Variant,
{
    /// Creates a new [`ChaChaCore`] instance from it's byte representation.
    ///
    /// Unless you know that you need to use this, you shouldn't.
    pub fn from_bytes(bytes: [u8; 48]) -> Self {
        unsafe { transmute(bytes) }
    }

    /// XORs the entirety of `buffer` with output from `self`.
    #[inline(never)]
    pub fn apply_keystream(&mut self, buffer: &mut [u8]) {
        self.inner::<true>(buffer);
    }

    /// Fills the entirety of `buffer` with output from `self`.
    #[inline(never)]
    pub fn fill(&mut self, buffer: &mut [u8]) {
        self.inner::<false>(buffer);
    }

    #[inline]
    fn inner<const XOR: bool>(&mut self, buffer: &mut [u8]) {
        const {
            assert!(ROUNDS > 0);
            assert!(ROUNDS.is_multiple_of(2));
        }

        let mut backend = B::new(self);
        let (chunks, remainder) = buffer.as_chunks_mut::<BATCH_BYTES>();

        // We're done with using `self`, so we can just go ahead and compute how
        // many blocks our upcoming parallel computation will consume and increment
        // the counter here.
        let blocks = (chunks.len() as u64)
            .wrapping_mul(BLOCKS as u64)
            .wrapping_add(remainder.len().div_ceil(MATRIX_SIZE) as u64);
        unsafe {
            match V::VAR {
                Variants::Djb => {
                    self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(blocks);
                }
                Variants::Ietf => {
                    self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(blocks as u32);
                }
            }
        }

        for chunk in chunks {
            backend.fill::<B, ROUNDS, V, XOR>(chunk);
        }

        if !remainder.is_empty() {
            let mut tmp = [0; BATCH_BYTES];
            backend.fill::<B, ROUNDS, V, XOR>(&mut tmp);
            for (r, t) in remainder.iter_mut().zip(tmp) {
                if XOR {
                    *r ^= t;
                } else {
                    *r = t;
                }
            }
        }
    }
}
