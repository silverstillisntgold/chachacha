// Pointless to zero memory we're going to immediately overwrite,
// but rust complains about leaving it uninitialized because it can't
// tell we're filling it before it's eventually used.
#![allow(clippy::uninit_assumed_init, invalid_value)]

use crate::backends::*;
use crate::util::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};
use core::ptr::copy_nonoverlapping;

#[repr(C)]
pub struct ChaChaCore<M, R, V> {
    row_b: Row,
    row_c: Row,
    row_d: Row,
    _pd: PhantomData<(M, R, V)>,
}

impl<M, V> ChaChaCore<M, Djb, V>
where
    M: Machine,
    V: Variant,
{
    pub fn new(key: [u32; 8], nonce: [u32; 2]) -> Self {
        let counter = 0;
        Self::new_with_counter(key, counter, nonce)
    }

    pub fn new_with_counter(key: [u32; 8], counter: u64, nonce: [u32; 2]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }
}

impl<M, V> ChaChaCore<M, Ietf, V>
where
    M: Machine,
    V: Variant,
{
    pub fn new(key: [u32; 8], nonce: [u32; 3]) -> Self {
        let counter = 0;
        Self::new_with_counter(key, counter, nonce)
    }

    pub fn new_with_counter(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> Self {
        unsafe { transmute((key, counter, nonce)) }
    }
}

impl<M, R, V> ChaChaCore<M, R, V>
where
    M: Machine,
    R: DoubleRounds,
    V: Variant,
{
    /* ******** PUBLIC METHODS ******** */

    #[inline(never)]
    pub fn xor(&mut self, buf: &[u8]) {
        let _ = buf;
        unimplemented!()
    }

    #[inline(never)]
    pub fn fill(&mut self, buf: &[u8]) {
        let _ = buf;
        unimplemented!()
    }

    #[inline(never)]
    pub fn fill_exact_u8(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
        let _ = buf;
        unimplemented!()
    }

    #[inline]
    pub fn get_exact_u8(&mut self) -> [u8; BUF_LEN_U8] {
        let mut buf = unsafe { MaybeUninit::uninit().assume_init() };
        self.fill_exact_u8(&mut buf);
        buf
    }

    /* ******** PRIVATE METHODS ******** */

    #[inline]
    fn slice(&mut self, buf: &[u8]) {
        let _ = buf;
        unimplemented!()
    }
}

// #[repr(C)]
// pub struct ChaChaCore<R, T, V> {
//     row_b: Row,
//     row_c: Row,
//     row_d: Row,
//     _phantom: PhantomData<(R, T, V)>,
// }

// impl<R, T, V> ChaChaCore<R, T, V>
// where
//     R: DoubleRounds,
//     T: Copy,
//     Vector<T>: VectorOps,
//     V: Variant,
// {
//     /// Xors `dst` with bytes from the output of `self`.
//     #[inline(never)]
//     pub fn xor(&mut self, dst: &mut [u8]) {
//         self.slice::<true>(dst);
//     }

//     /// Fills `dst` with bytes from the output of `self`.
//     #[inline(never)]
//     pub fn fill(&mut self, dst: &mut [u8]) {
//         self.slice::<false>(dst);
//     }

//     #[inline]
//     fn slice<const XOR: bool>(&mut self, dst: &mut [u8]) {
//         let mut machine = Machine::<T>::new::<V>(self.get_naked());
//         dst.chunks_exact_mut(BUF_LEN_U8).for_each(|chunk| {
//             // FUCKING JUST GIVE US ARRAY WINDOWS OR SOMETHING DAMNIT.
//             let buf: &mut [u8; BUF_LEN_U8] = chunk.try_into().unwrap();
//             self.chacha::<true, XOR>(&mut machine, buf)
//         });
//         let rem = dst.chunks_exact_mut(BUF_LEN_U8).into_remainder();
//         if !rem.is_empty() {
//             let mut buf: [u8; BUF_LEN_U8] = unsafe { MaybeUninit::uninit().assume_init() };
//             self.chacha::<false, XOR>(&mut machine, &mut buf);
//             unsafe {
//                 copy_nonoverlapping(buf.as_ptr(), rem.as_mut_ptr(), rem.len());
//             }
//             // Normally, `ChaChaCore` is incremented by `DEPTH` after each call to ChaChaCore::chacha, but
//             // this approach fails to maintain parity with reference ChaCha implementations when `dst` has
//             // a length which isn't a perfect multiple of `BUF_LEN_U8`.
//             // Because we are processesing four ChaCha instances at once, we meed to make sure the counter
//             // is set to the value just beyond the instance whose data we (even just partially) consumed.
//             // For values of rem.len(), these are the mappings we need:
//             // (0,64] --> 1 (only data from the first ChaCha instance was used)
//             // (64,128] --> 2 (data from the first two ChaCha instances was used)
//             // (128,192] --> 3 (data from the first three ChaCha instances was used)
//             // (192,256] --> 4 (data from all ChaCha instances was used)
//             let increment = rem.len().div_ceil(MATRIX_SIZE_U8);
//             unsafe {
//                 match V::VAR {
//                     Variants::Djb => {
//                         self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(increment as u64);
//                     }
//                     Variants::Ietf => {
//                         self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(increment as u32);
//                     }
//                 }
//             }
//         }
//     }

//     /// Computes the result of a ChaCha computation and uses it to fill
//     /// the returned array with `u64` values.
//     #[inline]
//     pub fn get_block_u64(&mut self) -> [u64; BUF_LEN_U64] {
//         let mut result = unsafe { MaybeUninit::uninit().assume_init() };
//         self.fill_block_u64(&mut result);
//         result
//     }

//     /// Computes the result of a ChaCha computation and uses it to fill
//     /// the returned array with `u8` values.
//     #[inline]
//     pub fn get_block(&mut self) -> [u8; BUF_LEN_U8] {
//         let mut result = unsafe { MaybeUninit::uninit().assume_init() };
//         self.fill_block(&mut result);
//         result
//     }

//     /// Computes the result of a ChaCha computation and uses it to fill
//     /// `buf` with `u64` values.
//     #[inline]
//     pub fn fill_block_u64(&mut self, buf: &mut [u64; BUF_LEN_U64]) {
//         let temp = unsafe { transmute(buf) };
//         self.chacha_once::<false>(temp);
//     }

//     /// Computes the result of a ChaCha computation and uses it to fill
//     /// `buf` with `u8` values.
//     #[inline]
//     pub fn fill_block(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
//         self.chacha_once::<false>(buf);
//     }

//     /// Computes the result of a ChaCha computation and xors it with the data in `buf`.
//     #[inline]
//     pub fn xor_block(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
//         self.chacha_once::<true>(buf);
//     }

//     #[inline(never)]
//     fn chacha_once<const XOR: bool>(&mut self, buf: &mut [u8; BUF_LEN_U8]) {
//         let mut machine = Machine::<T>::new::<V>(self.get_naked());
//         self.chacha::<false, XOR>(&mut machine, buf);
//         self.increment();
//     }

//     #[inline(always)]
//     fn chacha<const INCREMENT: bool, const XOR: bool>(
//         &mut self,
//         machine: &mut Machine<T>,
//         buf: &mut [u8; BUF_LEN_U8],
//     ) {
//         let mut cur = machine.clone();
//         for _ in 0..R::COUNT {
//             cur.double_round();
//         }
//         let result = cur + machine.clone();
//         if XOR {
//             result.xor_inner(buf);
//         } else {
//             result.get_inner(buf);
//         }
//         if INCREMENT {
//             machine.increment::<V>();
//             self.increment();
//         }
//     }

//     #[inline(always)]
//     fn increment(&mut self) {
//         unsafe {
//             match V::VAR {
//                 Variants::Djb => {
//                     self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(DEPTH as u64);
//                 }
//                 Variants::Ietf => {
//                     self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(DEPTH as u32);
//                 }
//             }
//         }
//     }

//     #[inline(always)]
//     fn get_naked(&self) -> &ChaChaNaked {
//         const {
//             assert!(align_of::<Self>() == align_of::<ChaChaNaked>());
//             assert!(size_of::<Self>() == size_of::<ChaChaNaked>());
//         }
//         unsafe { transmute(self) }
//     }
// }

// impl<R, T, V> From<u8> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: u8) -> Self {
//         [value; SEED_LEN_U8].into()
//     }
// }

// impl<R, T, V> From<u16> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: u16) -> Self {
//         [value; SEED_LEN_U16].into()
//     }
// }

// impl<R, T, V> From<u32> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: u32) -> Self {
//         [value; SEED_LEN_U32].into()
//     }
// }

// impl<R, T, V> From<u64> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: u64) -> Self {
//         [value; SEED_LEN_U64].into()
//     }
// }

// impl<R, T, V> From<[u8; SEED_LEN_U8]> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: [u8; SEED_LEN_U8]) -> Self {
//         unsafe { transmute(value) }
//     }
// }

// impl<R, T, V> From<[u16; SEED_LEN_U16]> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: [u16; SEED_LEN_U16]) -> Self {
//         unsafe { transmute(value) }
//     }
// }

// impl<R, T, V> From<[u32; SEED_LEN_U32]> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: [u32; SEED_LEN_U32]) -> Self {
//         unsafe { transmute(value) }
//     }
// }

// impl<R, T, V> From<[u64; SEED_LEN_U64]> for ChaChaCore<R, T, V> {
//     #[inline]
//     fn from(value: [u64; SEED_LEN_U64]) -> Self {
//         unsafe { transmute(value) }
//     }
// }
