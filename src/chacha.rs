/*!
Module containing the [`ChaChaCore`] type, which as it's name suggests, is the core type used
to abstract the ChaCha algorithm to the most powerful vectorization model available.
*/

// Pointless to zero memory we're going to immediately overwrite,
// but rust complains about leaving it uninitialized because it can't
// tell we're filling it before it's eventually used.
#![allow(invalid_value)]

use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};
use core::ptr::copy_nonoverlapping;

#[repr(C)]
pub struct ChaChaCore<M, R, V> {
    row_b: Row,
    row_c: Row,
    row_d: Row,
    _m: PhantomData<M>,
    _r: PhantomData<R>,
    _v: PhantomData<V>,
}

impl<M, R, V> From<u8> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: u8) -> Self {
        [value; SEED_LEN_U8].into()
    }
}

impl<M, R, V> From<[u8; SEED_LEN_U8]> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: [u8; SEED_LEN_U8]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> From<u32> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: u32) -> Self {
        [value; SEED_LEN_U32].into()
    }
}

impl<M, R, V> From<[u32; SEED_LEN_U32]> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: [u32; SEED_LEN_U32]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> From<u64> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: u64) -> Self {
        [value; SEED_LEN_U64].into()
    }
}

impl<M, R, V> From<[u64; SEED_LEN_U64]> for ChaChaCore<M, R, V> {
    #[inline]
    fn from(value: [u64; SEED_LEN_U64]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline]
    pub fn new(key: [u32; 8], counter: u64, nonce: [u32; 3]) -> Self {
        let row_b = Row {
            u32x4: [key[0], key[1], key[2], key[3]],
        };
        let row_c = Row {
            u32x4: [key[4], key[5], key[6], key[7]],
        };
        let row_d = match V::VAR {
            Variants::Djb => {
                let nonce = unsafe { transmute([nonce[0], nonce[1]]) };
                Row {
                    u64x2: [counter, nonce],
                }
            }
            Variants::Ietf => {
                let counter = counter as u32;
                Row {
                    u32x4: [counter, nonce[0], nonce[1], nonce[2]],
                }
            }
        };
        Self {
            row_b,
            row_c,
            row_d,
            _m: PhantomData,
            _r: PhantomData,
            _v: PhantomData,
        }
    }

    /// Fills `dst` with bytes from the output of `self`.
    #[inline(never)]
    pub fn fill(&mut self, dst: &mut [u8]) {
        let mut machine = M::new::<V>(self.get_naked());
        dst.chunks_exact_mut(BUF_LEN_U8).for_each(|chunk| {
            let buf: &mut [u8; BUF_LEN_U8] = chunk.try_into().unwrap();
            self.chacha::<true>(&mut machine, buf)
        });
        let rem = dst.chunks_exact_mut(BUF_LEN_U8).into_remainder();
        if !rem.is_empty() {
            let mut buf: [u8; BUF_LEN_U8] = unsafe { MaybeUninit::uninit().assume_init() };
            self.chacha::<false>(&mut machine, &mut buf);
            unsafe {
                copy_nonoverlapping(buf.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
            self.determine_new_counter(rem.len());
        }
    }

    #[inline]
    pub fn get_block_u64(&mut self) -> [u64; BUF_LEN_U64] {
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block_u64(&mut result);
        result
    }

    #[inline]
    pub fn get_block(&mut self) -> [u8; BUF_LEN_U8] {
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block(&mut result);
        result
    }

    #[inline]
    pub fn fill_block_u64(&mut self, buf: &mut [u64; BUF_LEN_U64]) {
        let temp = unsafe { transmute(buf) };
        self.chacha_once(temp);
    }

    #[inline]
    pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
        self.chacha_once(buf);
    }

    #[inline(never)]
    fn chacha_once(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
        let mut machine = M::new::<V>(self.get_naked());
        self.chacha::<false>(&mut machine, buf);
        self.increment();
    }

    #[inline]
    fn chacha<const INCREMENT: bool>(&mut self, machine: &mut M, buf: &mut [u8; BUF_LEN_U8]) {
        let mut cur = machine.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + machine.clone();
        result.fetch_result(buf);
        if INCREMENT {
            machine.increment::<V>();
            self.increment();
        }
    }

    #[inline]
    fn determine_new_counter(&mut self, len: usize) {
        let increment = len.div_ceil(MATRIX_SIZE_U8);
        unsafe {
            match V::VAR {
                Variants::Djb => {
                    self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(increment as u64);
                }
                Variants::Ietf => {
                    self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(increment as u32);
                }
            }
        }
    }

    #[inline]
    fn increment(&mut self) {
        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(DEPTH as u64);
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(DEPTH as u32);
        }
    }

    #[inline]
    fn get_naked(&self) -> &ChaChaNaked {
        // Sanity checks to ensure the `PhantomData` members in
        // `ChaChaCore` don't cause any issues with transmutation.
        const {
            assert!(size_of::<Self>() == size_of::<ChaChaNaked>());
            assert!(align_of::<Self>() == align_of::<ChaChaNaked>());
        }
        unsafe { transmute(self) }
    }
}
