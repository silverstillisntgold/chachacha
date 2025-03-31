use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;

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
    #[inline]
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

    #[inline]
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

    #[inline]
    fn fill_block_noincrement(&mut self, buf: &mut [u8; BUF_LEN]) {
        let mut cur = self.matrix.clone();
        let old = self.matrix.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + old;
        result.fill_block(buf);
    }

    #[inline]
    fn increment(&mut self) {
        match V::VAR {
            Variants::Djb => self.matrix.increment_djb(),
            Variants::Ietf => self.matrix.increment_ietf(),
        }
    }
}
