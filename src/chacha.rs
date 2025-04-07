use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};

#[repr(C)]
pub struct ChaCha<M, R, V> {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
    _m: PhantomData<M>,
    _r: PhantomData<R>,
    _v: PhantomData<V>,
}

impl<M, R, V> Clone for ChaCha<M, R, V> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            row_b: self.row_b,
            row_c: self.row_c,
            row_d: self.row_d,
            _m: PhantomData,
            _r: PhantomData,
            _v: PhantomData,
        }
    }
}

impl<M, R, V> Default for ChaCha<M, R, V> {
    #[inline(always)]
    fn default() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

impl<M, R, V> From<u8> for ChaCha<M, R, V> {
    #[inline(always)]
    fn from(value: u8) -> Self {
        [value; CHACHA_SEED_LEN].into()
    }
}

impl<M, R, V> From<[u8; CHACHA_SEED_LEN]> for ChaCha<M, R, V> {
    #[inline(always)]
    fn from(value: [u8; CHACHA_SEED_LEN]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> ChaCha<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    pub fn new(state: impl Into<Self>) -> Self {
        state.into()
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

    #[inline(never)]
    pub fn get_block(&mut self) -> [u8; BUF_LEN] {
        #[allow(invalid_value)]
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block(&mut result);
        result
    }

    #[inline(always)]
    pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN]) {
        self.fill_block_noincrement(buf);
        self.increment();
    }

    #[inline(never)]
    fn fill_block_noincrement(&mut self, buf: &mut [u8; BUF_LEN]) {
        let mut cur = match V::VAR {
            Variants::Djb => M::new_djb(self.as_ref()),
            Variants::Ietf => M::new_ietf(self.as_ref()),
        };
        let old = cur.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + old;
        result.fetch_result(buf);
    }
}
