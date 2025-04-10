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
    #[inline(always)]
    fn from(value: u8) -> Self {
        [value; SEED_LEN].into()
    }
}

impl<M, R, V> From<[u8; SEED_LEN]> for ChaChaCore<M, R, V> {
    #[inline(always)]
    fn from(value: [u8; SEED_LEN]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> From<[u32; SEED_LEN_U32]> for ChaChaCore<M, R, V> {
    #[inline(always)]
    fn from(value: [u32; SEED_LEN_U32]) -> Self {
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
    pub fn new(state: impl Into<Self>) -> Self {
        state.into()
    }

    #[inline(never)]
    pub fn fill(&mut self, dst: &mut [u8]) {
        let mut machine = M::new::<V>(self.get_naked());
        dst.chunks_exact_mut(BUF_LEN).for_each(|chunk| {
            let buf: &mut [u8; BUF_LEN] = chunk.try_into().unwrap();
            self.chacha(&mut machine, buf)
        });
        let rem = dst.chunks_exact_mut(BUF_LEN).into_remainder();
        if !rem.is_empty() {
            let mut buf: [u8; 256] = unsafe { MaybeUninit::uninit().assume_init() };
            self.chacha(&mut machine, &mut buf);
            // This assertion is optimized out in release mode.
            assert!(rem.len() < buf.len());
            // SAFETY: We've just guaranteed that the length of `rem` is less
            // than that of `buf`, so this can never overflow.
            unsafe {
                copy_nonoverlapping(buf.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }

    #[inline]
    pub fn get_block_u64(&mut self) -> [u64; BUF_LEN_U64] {
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block_u64(&mut result);
        result
    }

    #[inline]
    pub fn get_block(&mut self) -> [u8; BUF_LEN] {
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
    pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN]) {
        self.chacha_once(buf);
    }

    #[inline(never)]
    fn chacha_once(&mut self, buf: &mut [u8; BUF_LEN]) {
        let mut machine = M::new::<V>(self.get_naked());
        self.chacha(&mut machine, buf);
    }

    #[inline(always)]
    fn chacha(&mut self, machine: &mut M, buf: &mut [u8; BUF_LEN]) {
        let mut cur = machine.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + machine.clone();
        machine.increment::<V>();
        self.increment();
        result.fetch_result(buf);
    }

    #[inline(always)]
    fn increment(&mut self) {
        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }
    }

    #[inline(always)]
    fn increment_djb(&mut self) {
        unsafe {
            self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(DEPTH as u64);
        }
    }

    #[inline(always)]
    fn increment_ietf(&mut self) {
        unsafe { self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(DEPTH as u32) }
    }

    #[inline(always)]
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
