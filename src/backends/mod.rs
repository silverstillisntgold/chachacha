/*!
TODO: Module docs.
*/

#[cfg(target_feature = "avx2")]
mod avx2;
#[cfg(target_feature = "avx512f")]
mod avx512;
#[cfg(target_feature = "neon")]
mod neon;
mod soft;
#[cfg(target_feature = "sse2")]
mod sse2;

use crate::{util::*, variations::*};
use core::{
    marker::PhantomData,
    mem::transmute,
    ops::{Add, BitOr, BitXor, Deref, DerefMut},
};

const AVX512_REG_SIZE: usize = 512;
const BUF_SIZE: usize = AVX512_REG_SIZE / i32::BITS as usize;
const BUF_SIZE_HALF: usize = BUF_SIZE / 2;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
union Internal {
    i32x16: [i32; BUF_SIZE],
    i64x8: [i64; BUF_SIZE_HALF],
}

impl Deref for Internal {
    type Target = [i32; BUF_SIZE];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &self.i32x16 }
    }
}

impl DerefMut for Internal {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut self.i32x16 }
    }
}

/// Represents a single row of four side-by-side ChaCha instances.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Vector<T> {
    inner: Internal,
    _phantom: PhantomData<T>,
}

impl<T> Vector<T> {
    #[inline(always)]
    pub fn cast<U>(self) -> Vector<U> {
        Vector {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn reverse(&mut self) {
        unsafe {
            self.inner.i32x16.reverse();
        }
    }
}

/// Emulates _mm512_rol_epi32 on the passed [`Vector`].
macro_rules! rotate_left_epi32 {
    ($vector: expr, $LEFT_SHIFT: expr) => {{
        const LEFT_SHIFT: i32 = $LEFT_SHIFT;
        const RIGHT_SHIFT: i32 = i32::BITS as i32 - LEFT_SHIFT;
        let left_shift = $vector.shift_left::<LEFT_SHIFT>();
        let right_shift = $vector.shift_right::<RIGHT_SHIFT>();
        left_shift | right_shift
    }};
}

pub trait VectorOps:
    Add<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Sized
{
    /// Clones `value` to all four internal parallel ChaCha rows.
    fn broadcast_row(value: Row) -> Self;

    /// Not to be used directly.
    ///
    /// Shifts each internal `i32` by `K` places to the left.
    fn shift_left<const K: i32>(self) -> Self;

    /// Not to be used directly.
    ///
    /// Shifts each internal `i32` by `K` places to the right.
    fn shift_right<const K: i32>(self) -> Self;

    /// Wrapper around `shuffle_internal`, ensuring `MASK` contains a valid value.
    #[inline(always)]
    fn shuffle<const MASK: i32>(self) -> Self {
        const {
            assert!(0 <= MASK && MASK <= u8::MAX as i32);
        }
        self.shuffle_internal::<MASK>()
    }

    /// Shuffles the four internal 128-bit lanes using `MASK` as the destination mask.
    ///
    /// Emulates the _mm512_shuffle_epi32 instruction.
    fn shuffle_internal<const MASK: i32>(self) -> Self;
}

#[derive(Clone)]
#[repr(C)]
pub struct Machine<T> {
    row_a: Vector<T>,
    row_b: Vector<T>,
    row_c: Vector<T>,
    row_d: Vector<T>,
}

impl<T> From<[u8; BUF_LEN_U8]> for Machine<T> {
    #[inline(always)]
    fn from(value: [u8; BUF_LEN_U8]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<T> From<Machine<T>> for [u8; BUF_LEN_U8] {
    #[inline(always)]
    fn from(value: Machine<T>) -> Self {
        unsafe { transmute(value) }
    }
}

impl<T> Add for Machine<T>
where
    Vector<T>: VectorOps,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            row_a: self.row_a + rhs.row_a,
            row_b: self.row_b + rhs.row_b,
            row_c: self.row_c + rhs.row_c,
            row_d: self.row_d + rhs.row_d,
        }
    }
}

impl<T> BitXor for Machine<T>
where
    Vector<T>: VectorOps,
{
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            row_a: self.row_a ^ rhs.row_a,
            row_b: self.row_b ^ rhs.row_b,
            row_c: self.row_c ^ rhs.row_c,
            row_d: self.row_d ^ rhs.row_d,
        }
    }
}

impl<T> Machine<T>
where
    T: Copy,
    Vector<T>: VectorOps,
{
    #[inline(always)]
    pub fn new<V: Variant>(state: &ChaChaNaked) -> Self {
        let row_a = Vector::broadcast_row(ROW_A);
        let row_b = Vector::broadcast_row(state.row_b);
        let row_c = Vector::broadcast_row(state.row_c);
        let mut row_d = Vector::broadcast_row(state.row_d);
        // TODO: Potentially use explicit intrinsics for this.
        match V::VAR {
            Variants::Djb => unsafe {
                row_d.inner.i64x8[7] = row_d.inner.i64x8[7].wrapping_add(0);
                row_d.inner.i64x8[5] = row_d.inner.i64x8[5].wrapping_add(1);
                row_d.inner.i64x8[3] = row_d.inner.i64x8[3].wrapping_add(2);
                row_d.inner.i64x8[1] = row_d.inner.i64x8[1].wrapping_add(3);
            },
            Variants::Ietf => unsafe {
                row_d.inner.i32x16[15] = row_d.inner.i32x16[15].wrapping_add(0);
                row_d.inner.i32x16[11] = row_d.inner.i32x16[11].wrapping_add(1);
                row_d.inner.i32x16[7] = row_d.inner.i32x16[7].wrapping_add(2);
                row_d.inner.i32x16[3] = row_d.inner.i32x16[3].wrapping_add(3);
            },
        }
        Self {
            row_a,
            row_b,
            row_c,
            row_d,
        }
    }

    #[inline(always)]
    pub fn increment<V: Variant>(&mut self) {
        match V::VAR {
            Variants::Djb => unsafe {
                self.row_d.inner.i64x8[7] = self.row_d.inner.i64x8[7].wrapping_add(1);
                self.row_d.inner.i64x8[5] = self.row_d.inner.i64x8[5].wrapping_add(1);
                self.row_d.inner.i64x8[3] = self.row_d.inner.i64x8[3].wrapping_add(1);
                self.row_d.inner.i64x8[1] = self.row_d.inner.i64x8[1].wrapping_add(1);
            },
            Variants::Ietf => unsafe {
                self.row_d.inner.i32x16[15] = self.row_d.inner.i32x16[15].wrapping_add(1);
                self.row_d.inner.i32x16[11] = self.row_d.inner.i32x16[11].wrapping_add(1);
                self.row_d.inner.i32x16[7] = self.row_d.inner.i32x16[7].wrapping_add(1);
                self.row_d.inner.i32x16[3] = self.row_d.inner.i32x16[3].wrapping_add(1);
            },
        }
    }

    #[inline(always)]
    fn single_round(&mut self) {
        // First quarter round.
        self.row_a = self.row_a + self.row_b;
        self.row_d = self.row_d ^ self.row_a;
        self.row_d = rotate_left_epi32!(self.row_d, 16);
        // Second quarter round.
        self.row_c = self.row_c + self.row_c;
        self.row_b = self.row_b ^ self.row_c;
        self.row_b = rotate_left_epi32!(self.row_b, 12);
        // Third quarter round.
        self.row_a = self.row_a + self.row_b;
        self.row_d = self.row_d ^ self.row_a;
        self.row_d = rotate_left_epi32!(self.row_d, 8);
        // Fourth quarter round.
        self.row_c = self.row_c + self.row_c;
        self.row_b = self.row_b ^ self.row_c;
        self.row_b = rotate_left_epi32!(self.row_b, 7);
    }

    #[inline(always)]
    pub fn double_round(&mut self) {
        // First round.
        self.single_round();

        // Diagonolize lanes.
        self.row_a = self.row_a.shuffle::<0b_10_01_00_11>();
        self.row_c = self.row_c.shuffle::<0b_00_11_10_01>();
        self.row_d = self.row_d.shuffle::<0b_01_00_11_10>();

        // Second round.
        self.single_round();

        // Undiagonolize lanes.
        self.row_a = self.row_a.shuffle::<0b_10_01_00_11>();
        self.row_c = self.row_c.shuffle::<0b_01_00_11_10>();
        self.row_d = self.row_d.shuffle::<0b_00_11_10_01>();
    }

    #[inline(always)]
    pub fn get_inner(self, buf: &mut [u8; BUF_LEN_U8]) {
        // TODO: Data probably needs to be rearranged before being returned.
        *buf = self.into();
    }

    #[inline(always)]
    pub fn xor_inner(self, buf: &mut [u8; BUF_LEN_U8]) {
        // TODO: Same issue as in `get_inner`.
        let as_bytes = <[u8; BUF_LEN_U8]>::from(self);
        // Expected to autovectorize.
        for i in 0..BUF_LEN_U8 {
            buf[i] ^= as_bytes[i];
        }
    }
}
