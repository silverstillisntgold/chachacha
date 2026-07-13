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

    #[inline(never)]
    pub fn apply_keystream(&mut self, buffer: &mut [u8]) {
        B::process::<ROUNDS, V, true>(self, buffer);
    }

    #[inline(never)]
    pub fn fill(&mut self, buffer: &mut [u8]) {
        B::process::<ROUNDS, V, false>(self, buffer);
    }

    #[inline]
    pub fn get_block(&mut self) -> [u8; 256] {
        let mut buf = [0; 256];
        self.fill(&mut buf);
        buf
    }
}
