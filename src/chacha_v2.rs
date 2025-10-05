/*!
Module containing the [`ChaChaCore`] type, which as it's name suggests, is the core type used
to abstract the ChaCha algorithm to the most powerful vectorization model available.
*/

// Pointless to zero memory we're going to immediately overwrite,
// but rust complains about leaving it uninitialized because it can't
// tell we're filling it before it's eventually used.
#![allow(clippy::uninit_assumed_init, invalid_value)]

use crate::backends::*;
use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, transmute};
use core::ptr::copy_nonoverlapping;

#[repr(C)]
pub struct ChaChaCore<R, T, V> {
    row_b: Row,
    row_c: Row,
    row_d: Row,
    _phantom: PhantomData<(R, T, V)>,
}

impl<R, T> ChaChaCore<R, T, Djb> {
    pub fn new(key: [u32; 8], counter: u64, nonce: [u32; 2]) -> Self {
        let row_b = Row {
            u32x4: [key[0], key[1], key[2], key[3]],
        };
        let row_c = Row {
            u32x4: [key[4], key[5], key[6], key[7]],
        };
        let row_d = Row {
            u64x2: [counter, unsafe { transmute(nonce) }],
        };
        Self {
            row_b,
            row_c,
            row_d,
            _phantom: PhantomData,
        }
    }
}

impl<R, T> ChaChaCore<R, T, Ietf> {
    pub fn new(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> Self {
        let row_b = Row {
            u32x4: [key[0], key[1], key[2], key[3]],
        };
        let row_c = Row {
            u32x4: [key[4], key[5], key[6], key[7]],
        };
        let row_d = Row {
            u32x4: [counter, nonce[0], nonce[1], nonce[2]],
        };
        Self {
            row_b,
            row_c,
            row_d,
            _phantom: PhantomData,
        }
    }
}

impl<R, T, V> ChaChaCore<R, T, V>
where
    R: DoubleRounds,
    T: Copy,
    Vector<T>: VectorOps,
    V: Variant,
{
    #[inline(always)]
    fn chacha<const INCREMENT: bool, const XOR: bool>(
        &mut self,
        machine: &mut MachineV2<T>,
        buf: &mut [u8; BUF_LEN_U8],
    ) {
        let mut cur = machine.clone();
        for _ in 0..R::COUNT {
            cur.double_round();
        }
        let result = cur + machine.clone();
        if XOR {
            result.xor_inner(buf);
        } else {
            result.get_inner(buf);
        }
        if INCREMENT {
            machine.increment::<V>();
            //self.increment();
        }
    }

    // #[inline]
    // pub fn get_counter(&self) -> u64 {
    //     unsafe {
    //         match V::VAR {
    //             Variants::Djb => self.row_d.u64x2[0],
    //             Variants::Ietf => self.row_d.u32x4[0] as u64,
    //         }
    //     }
    // }

    // #[inline]
    // pub fn set_counter(&mut self, new_counter: u64) {
    //     unsafe {
    //         match V::VAR {
    //             Variants::Djb => self.row_d.u64x2[0] = new_counter,
    //             Variants::Ietf => self.row_d.u32x4[0] = new_counter as u32,
    //         }
    //     }
    // }
}

impl<R, T, V> From<u8> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: u8) -> Self {
        [value; SEED_LEN_U8].into()
    }
}

impl<R, T, V> From<u32> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: u32) -> Self {
        [value; SEED_LEN_U32].into()
    }
}

impl<R, T, V> From<u64> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: u64) -> Self {
        [value; SEED_LEN_U64].into()
    }
}

impl<R, T, V> From<[u8; SEED_LEN_U8]> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: [u8; SEED_LEN_U8]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<R, T, V> From<[u32; SEED_LEN_U32]> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: [u32; SEED_LEN_U32]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<R, T, V> From<[u64; SEED_LEN_U64]> for ChaChaCore<R, T, V> {
    #[inline]
    fn from(value: [u64; SEED_LEN_U64]) -> Self {
        unsafe { transmute(value) }
    }
}
