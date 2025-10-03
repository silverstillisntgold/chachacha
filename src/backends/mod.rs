/*!
TODO: Module docs.
*/

mod avx2;
mod avx512;
mod soft;
mod sse2;

use crate::{
    rounds::*,
    util::{ChaChaNaked, Row},
    variations::*,
};
use core::{
    marker::PhantomData,
    mem::transmute,
    ops::{Add, BitOr, BitXor, Deref, DerefMut},
};

const AVX512_REG_SIZE: usize = 512;
const BUF_SIZE: usize = AVX512_REG_SIZE / i32::BITS as usize;

pub trait VectorType {}

/// Represents a single row of four side-by-side ChaCha instances.
#[repr(C, align(64))]
pub struct Vector<T> {
    inner: [i32; BUF_SIZE],
    _phantom: PhantomData<T>,
}

impl<T> Vector<T> {
    #[inline(always)]
    pub fn reverse(&mut self) {
        self.inner.reverse();
    }

    #[inline(always)]
    pub fn increment_idx<const IDX: usize>(&mut self) {
        todo!()
    }
}

macro_rules! rotate_left_epi32 {
    ($vector: expr, $LEFT_SHIFT: expr) => {{
        const LEFT_SHIFT: i64 = $LEFT_SHIFT;
        const RIGHT_SHIFT: i64 = 32 - LEFT_SHIFT;
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
    /// Shifts each internal `i32` by `IMM8` places to the left.
    fn shift_left<const IMM8: i64>(self) -> Self;

    /// Not to be used directly.
    ///
    /// Shifts each internal `i32` by `IMM8` places to the right.
    fn shift_right<const IMM8: i64>(self) -> Self;

    /// Shuffles the four internal 128-bit lanes using `IMM8` as a destination mask.
    fn shuffle_128<const IMM8: i32>(self) -> Self;
}

#[repr(C)]
pub struct Machine<T> {
    row_a: Vector<T>,
    row_b: Vector<T>,
    row_c: Vector<T>,
    row_d: Vector<T>,
}

impl<T> From<[u8; 256]> for Machine<T> {
    #[inline(always)]
    fn from(value: [u8; 256]) -> Self {
        unsafe { transmute(value) }
    }
}

impl<T> From<Machine<T>> for [u8; 256] {
    #[inline(always)]
    fn from(value: Machine<T>) -> Self {
        unsafe { transmute(value) }
    }
}

impl<T> Machine<T>
where
    T: VectorType,
    Vector<T>: VectorOps,
{
    #[inline(always)]
    pub fn new<V: Variant>(state: &ChaChaNaked) -> Self {
        todo!()
    }

    #[inline(always)]
    pub fn increment<V: Variant>(&mut self) {
        todo!()
    }

    #[inline(always)]
    pub fn double_round(&mut self) {
        todo!()
    }

    #[inline(always)]
    pub fn get_inner(self) -> [u8; 256] {
        // TODO: Data probably needs to be rearranged before being returned.
        unsafe { transmute(self) }
    }

    #[inline(always)]
    pub fn xor_inner(self, other: [u8; 256]) -> [u8; 256] {
        // TODO: Same issue as in `Self::get_inner`.
        let other = Self::from(other);
        Self {
            row_a: self.row_a ^ other.row_a,
            row_b: self.row_b ^ other.row_b,
            row_c: self.row_c ^ other.row_c,
            row_d: self.row_d ^ other.row_d,
        }
        .into()
    }
}
