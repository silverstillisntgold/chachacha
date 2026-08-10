use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, ROW_A, Row, SIZE, Variant, Variants},
};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
union InternalMatrix {
    u32x16: [u32; SIZE],
    u64x8: [u64; SIZE / 2],
}

impl InternalMatrix {
    #[inline(always)]
    fn increment<const INCREMENT: usize, V: Variant>(&mut self) {
        const {
            assert!(INCREMENT <= u32::MAX as usize);
        }
        unsafe {
            match V::VAR {
                Variants::Djb => self.u64x8[6] = self.u64x8[6].wrapping_add(INCREMENT as u64),
                Variants::Ietf => self.u32x16[12] = self.u32x16[12].wrapping_add(INCREMENT as u32),
            }
        }
    }

    #[inline(always)]
    fn add(&mut self, other: Self) {
        unsafe {
            for (s, o) in self.u32x16.iter_mut().zip(other.u32x16) {
                *s = s.wrapping_add(o);
            }
        }
    }

    #[inline(always)]
    fn quarter_round(&mut self, a: usize, b: usize, c: usize, d: usize) {
        unsafe {
            self.u32x16[a] = self.u32x16[a].wrapping_add(self.u32x16[b]);
            self.u32x16[d] ^= self.u32x16[a];
            self.u32x16[d] = self.u32x16[d].rotate_left(16);

            self.u32x16[c] = self.u32x16[c].wrapping_add(self.u32x16[d]);
            self.u32x16[b] ^= self.u32x16[c];
            self.u32x16[b] = self.u32x16[b].rotate_left(12);

            self.u32x16[a] = self.u32x16[a].wrapping_add(self.u32x16[b]);
            self.u32x16[d] ^= self.u32x16[a];
            self.u32x16[d] = self.u32x16[d].rotate_left(8);

            self.u32x16[c] = self.u32x16[c].wrapping_add(self.u32x16[d]);
            self.u32x16[b] ^= self.u32x16[c];
            self.u32x16[b] = self.u32x16[b].rotate_left(7);
        }
    }
}

#[derive(Clone)]
pub struct Soft {
    inner: [InternalMatrix; BLOCKS],
}

impl Soft {
    #[inline(always)]
    fn increment<const INCREMENT: usize, V: Variant>(&mut self) {
        for s in self.inner.iter_mut() {
            s.increment::<INCREMENT, V>();
        }
    }

    #[inline(always)]
    fn add(&mut self, other: Self) {
        for (s, o) in self.inner.iter_mut().zip(other.inner) {
            s.add(o);
        }
    }

    #[inline(always)]
    fn quarter_round(&mut self, a: usize, b: usize, c: usize, d: usize) {
        for internal_matrix in self.inner.iter_mut() {
            internal_matrix.quarter_round(a, b, c, d);
        }
    }
}

impl Backend for Soft {
    #[inline(never)]
    fn new<B: Backend, const ROUNDS: usize, V: Variant>(core: &ChaChaCore<B, ROUNDS, V>) -> Self {
        let src = [ROW_A, core.row_b, core.row_c, core.row_d];
        let inner0 = unsafe { core::mem::transmute::<[Row; BLOCKS], InternalMatrix>(src) };
        let mut inner1 = inner0;
        let mut inner2 = inner0;
        let mut inner3 = inner0;
        inner1.increment::<1, V>();
        inner2.increment::<2, V>();
        inner3.increment::<3, V>();
        let inner = [inner0, inner1, inner2, inner3];
        Self { inner }
    }

    #[inline(never)]
    fn fill<B: Backend, const ROUNDS: usize, V: Variant, const XOR: bool>(
        &mut self,
        buffer: &mut [u8; BATCH_BYTES],
    ) {
        let mut out = self.clone();

        for _ in 0..(ROUNDS / 2) {
            // column rounds
            out.quarter_round(0, 4, 8, 12);
            out.quarter_round(1, 5, 9, 13);
            out.quarter_round(2, 6, 10, 14);
            out.quarter_round(3, 7, 11, 15);

            // diagonal rounds
            out.quarter_round(0, 5, 10, 15);
            out.quarter_round(1, 6, 11, 12);
            out.quarter_round(2, 7, 8, 13);
            out.quarter_round(3, 4, 9, 14);
        }

        out.add(self.clone());

        self.increment::<BLOCKS, V>();

        let tmp = unsafe { core::mem::transmute::<Soft, [u8; BATCH_BYTES]>(out) };

        for i in 0..BATCH_BYTES {
            if XOR {
                buffer[i] ^= tmp[i];
            } else {
                buffer[i] = tmp[i];
            }
        }
    }
}
