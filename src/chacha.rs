use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};

#[derive(Clone)]
pub struct ChaChaSmall {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
}

impl Default for ChaChaSmall {
    #[inline(always)]
    fn default() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

impl From<[u8; CHACHA_SEED_LEN]> for ChaChaSmall {
    #[inline(always)]
    fn from(value: [u8; CHACHA_SEED_LEN]) -> Self {
        unsafe { transmute(value) }
    }
}

impl ChaChaSmall {
    #[inline(always)]
    pub fn increment<V: Variant>(&mut self) {
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
}

pub struct ChaCha<M, R, V> {
    matrix: M,
    _pd1: PhantomData<R>,
    _pd2: PhantomData<V>,
}

impl<M, R, V> ChaCha<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    pub fn new<T>(state: T) -> Self
    where
        T: Into<ChaChaSmall>,
    {
        let chacha = state.into();
        let matrix = M::new::<V>(&chacha);
        Self {
            matrix,
            _pd1: PhantomData,
            _pd2: PhantomData,
        }
    }

    #[inline(never)]
    pub fn fill_block_once(state: &mut ChaChaSmall, buf: &mut [u8; BUF_LEN]) {
        Self::new(state.clone()).fill_block_noincrement(buf);
        state.increment::<V>();
    }

    #[inline(always)]
    pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN]) {
        self.fill_block_noincrement(buf);
        self.increment();
    }

    #[inline(never)]
    pub fn get_block(&mut self) -> [u8; BUF_LEN] {
        #[allow(invalid_value)]
        let mut result = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        self.fill_block(&mut result);
        result
    }

    #[inline(always)]
    fn fill_block_noincrement(&mut self, buf: &mut [u8; BUF_LEN]) {
        let mut cur = self.matrix.clone();
        let old = self.matrix.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + old;
        result.fill_block(buf);
    }

    #[inline(always)]
    fn increment(&mut self) {
        match V::VAR {
            Variants::Djb => self.matrix.increment_djb(),
            Variants::Ietf => self.matrix.increment_ietf(),
        }
    }
}
