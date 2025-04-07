use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};

#[repr(C)]
pub struct ChaChaCore<M, R, V> {
    pub row_b: Row,
    pub row_c: Row,
    pub row_d: Row,
    _m: PhantomData<M>,
    _r: PhantomData<R>,
    _v: PhantomData<V>,
}

impl<M, R, V> AsRef<ChaChaNaked> for ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    fn as_ref(&self) -> &ChaChaNaked {
        // Sanity checks to ensure the `PhantomData` members in
        // `ChaChaCore` don't cause any issues with transmutation.
        const {
            assert!(size_of::<Self>() == size_of::<ChaChaNaked>());
            assert!(align_of::<Self>() == align_of::<ChaChaNaked>());
        }
        unsafe { transmute(self) }
    }
}

impl<M, R, V> From<u8> for ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    fn from(value: u8) -> Self {
        [value; CHACHA_SEED_LEN].into()
    }
}

impl<M, R, V> From<[u8; CHACHA_SEED_LEN]> for ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    fn from(value: [u8; CHACHA_SEED_LEN]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<M, R, V> ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    #[inline(always)]
    pub fn new(state: impl Into<Self>) -> Self {
        state.into()
    }

    pub fn fill(&mut self, dest: &mut [u8]) {
        let mut m = M::new::<V>(self.as_ref());
        dest.chunks_exact_mut(BUF_LEN).for_each(|chunk| {
            let chunk: &mut [u8; BUF_LEN] = chunk.try_into().unwrap();
        });
    }

    #[inline(never)]
    fn process_chacha(m: &mut M, buf: &mut [u8; BUF_LEN]) {
        let mut cur = m.clone();
        let old = cur.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + old;
        result.fetch_result(buf);
        m.increment::<V>();
    }

    #[inline(always)]
    pub fn get_block_u64(&mut self) -> [u64; BUF_LEN_U64] {
        #[allow(invalid_value)]
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block_u64(&mut result);
        result
    }

    #[inline(always)]
    pub fn fill_block_u64(&mut self, buf: &mut [u64; BUF_LEN_U64]) {
        let temp = unsafe { transmute(buf) };
        self.fill_block(temp);
    }

    #[inline(always)]
    pub fn get_block(&mut self) -> [u8; BUF_LEN] {
        #[allow(invalid_value)]
        let mut result = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_block(&mut result);
        result
    }

    #[inline(never)]
    pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN]) {
        self.process_chacha_once(buf);
        self.increment();
    }

    #[inline(always)]
    fn process_chacha_once(&mut self, buf: &mut [u8; BUF_LEN]) {
        let mut cur = M::new::<V>(self.as_ref());
        let old = cur.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + old;
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
}
