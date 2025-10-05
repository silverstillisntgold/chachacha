/*!
TODO: Module docs.
*/

mod avx2;
mod avx512;
mod neon;
mod soft;
mod sse2;

use crate::{
    util::{ChaChaNaked, ROW_A, Row},
    variations::*,
};
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

pub trait VectorType: Copy {}

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
        self.inner.reverse();
    }

    #[inline(always)]
    pub fn increment_idx<const IDX: usize>(&mut self) {
        todo!()
    }
}

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

    /// Shuffles the four internal 128-bit lanes using `MASK` as a destination mask.
    fn shuffle_128<const MASK: i32>(self) -> Self;
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

    // for [a, b, c, d] in self.state.iter_mut() {
    //     *a = _mm256_add_epi32(*a, *b);
    //     *d = _mm256_xor_si256(*d, *a);
    //     *d = rotate_left_epi32!(*d, 16);

    //     *c = _mm256_add_epi32(*c, *d);
    //     *b = _mm256_xor_si256(*b, *c);
    //     *b = rotate_left_epi32!(*b, 12);

    //     *a = _mm256_add_epi32(*a, *b);
    //     *d = _mm256_xor_si256(*d, *a);
    //     *d = rotate_left_epi32!(*d, 8);

    //     *c = _mm256_add_epi32(*c, *d);
    //     *b = _mm256_xor_si256(*b, *c);
    //     *b = rotate_left_epi32!(*b, 7);
    // }

    #[inline(always)]
    pub fn double_round(&mut self) {
        // First round
        self.row_a = self.row_a + self.row_b;
        self.row_d = self.row_d ^ self.row_a;
        self.row_d = rotate_left_epi32!(self.row_d, 16);

        // Diagonolize lanes
        self.row_a = self.row_a.shuffle_128::<0b_10_01_00_11>();
        self.row_c = self.row_c.shuffle_128::<0b_00_11_10_01>();
        self.row_d = self.row_d.shuffle_128::<0b_01_00_11_10>();

        // Second round
        // TODO

        // Undiagonolize lanes
        self.row_a = self.row_a.shuffle_128::<0b_10_01_00_11>();
        self.row_c = self.row_c.shuffle_128::<0b_01_00_11_10>();
        self.row_d = self.row_d.shuffle_128::<0b_00_11_10_01>();
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
