use crate::{
    rounds::DoubleRounds,
    util::{BUF_LEN, ChaChaSmall, Machine},
    variations::{Variant, Variants},
};
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

    #[inline]
    pub fn block_fill(&mut self, buf: &mut [u8; BUF_LEN]) {
        self.block_fill_noincrement(buf);
        self.increment();
    }

    #[inline]
    fn block_fill_noincrement(&mut self, buf: &mut [u8; BUF_LEN]) {
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

    #[inline(never)]
    pub fn get_block(&mut self) -> [u8; BUF_LEN] {
        #[allow(invalid_value)]
        let mut result = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        self.block_fill(&mut result);
        result
    }

    #[inline(never)]
    pub fn new_and_block_fill<T>(state: T, buf: &mut [u8; BUF_LEN])
    where
        T: Into<ChaChaSmall>,
    {
        let mut temp = Self::new(state);
        temp.block_fill_noincrement(buf);
    }
}
