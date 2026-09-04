use crate::util::*;
use core::{marker::PhantomData, mem::transmute};

#[repr(C)]
pub struct ChaChaCore<B, const ROUNDS: usize, V> {
    pub(crate) row_b: Row,
    pub(crate) row_c: Row,
    pub(crate) row_d: Row,
    _pd: PhantomData<(B, V)>,
}

impl<B: Backend, const ROUNDS: usize> ChaChaCore<B, ROUNDS, Djb> {
    /// Creates a new ChaCha instance from `key` and `nonce`.
    pub fn new(key: [u32; 8], nonce: [u32; 2]) -> Self {
        Self::new_with_counter(key, 0, nonce)
    }

    /// Creates a new ChaCha instance from `key` and `nonce`,
    /// starting at position `counter`.
    pub fn new_with_counter(key: [u32; 8], counter: u64, nonce: [u32; 2]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }

    /// Seeks to ChaCha stream position at `position`.
    pub fn seek(&mut self, position: u64) {
        unsafe {
            self.row_d.u64x2[0] = position;
        }
    }

    /// Returns the current ChaCha stream position.
    pub fn position(&self) -> u64 {
        unsafe { self.row_d.u64x2[0] }
    }
}

impl<B: Backend, const ROUNDS: usize> ChaChaCore<B, ROUNDS, Ietf> {
    /// Creates a new ChaCha instance from `key` and `nonce`.
    pub fn new(key: [u32; 8], nonce: [u32; 3]) -> Self {
        Self::new_with_counter(key, 0, nonce)
    }

    /// Creates a new ChaCha instance from `key` and `nonce`,
    /// starting at position `counter`.
    pub fn new_with_counter(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }

    /// Seeks to ChaCha stream position at `position`.
    pub fn seek(&mut self, position: u32) {
        unsafe {
            self.row_d.u32x4[0] = position;
        }
    }

    /// Returns the current ChaCha stream position.
    pub fn position(&self) -> u32 {
        unsafe { self.row_d.u32x4[0] }
    }
}

impl<B: Backend, const ROUNDS: usize, V: Variant> ChaChaCore<B, ROUNDS, V> {
    /// Creates a new instance of [`Self`] from it's byte representation.
    ///
    /// Unless you know that you need to use this, you shouldn't.
    pub fn from_bytes(bytes: [u8; 48]) -> Self {
        unsafe { transmute(bytes) }
    }

    /// XORs the entirety of `buffer` with output from `self`.
    ///
    /// If you are planning to provide a `buffer` which is all zeros, use [`Self::fill`] instead.
    #[inline(never)]
    pub fn apply_keystream(&mut self, buffer: &mut [u8]) {
        self.inner::<true>(buffer);
    }

    /// [`Self::apply_keystream`], but for a fixed-length `buffer`.
    #[inline(never)]
    pub fn apply_keystream_exact(&mut self, buffer: &mut [u8; BATCH_BYTES]) {
        self.inner::<true>(buffer);
    }

    /// Fills the entirety of `buffer` with output from `self`.
    #[inline(never)]
    pub fn fill(&mut self, buffer: &mut [u8]) {
        self.inner::<false>(buffer);
    }

    /// [`Self::fill`], but for a fixed-length `buffer`.
    #[inline(never)]
    pub fn fill_exact(&mut self, buffer: &mut [u8; BATCH_BYTES]) {
        self.inner::<false>(buffer);
    }

    /// ChaCha real smooth.
    #[inline]
    fn inner<const XOR: bool>(&mut self, buffer: &mut [u8]) {
        const {
            assert!(ROUNDS > 0);
            assert!(ROUNDS.is_multiple_of(2));
        }

        let mut backend = B::new(self);
        let (chunks, remainder) = buffer.as_chunks_mut::<BATCH_BYTES>();

        // We have no further use for `self`, so compute how many blocks our
        // parallel computation will consume and increment the counter here.
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

        // A remainder can't be passed directly to Backend::fill because it requires a full
        // `BATCH_BYTES` buffer, so we need a tempory buffer.
        //
        // We explicitly are choosing to propagate `XOR` instead of passing `false` so that
        // only one instance of Backend::fill is generated. The cost of XORing our all-zeros
        // `tmp` buffer is not significant enough to matter, since it only happens once.
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
