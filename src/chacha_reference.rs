/*!
Module containing a reference Chacha implementation, which is verified against the reference
Chacha test vectors available [here].

The tests are only run against the original [`Djb`] variant, but the difference in a simple
reference implementation like this is trivial (literally a single line of code), so we assume passing
all these tests means we would also pass the equivalent [`Ietf`] variant tests.

[here]: https://github.com/secworks/chacha_testvectors/blob/master/src/chacha_testvectors.txt
*/

use crate::rounds::*;
use crate::util::*;
use crate::variations::*;
use core::iter::repeat_with;
use core::mem::transmute;
use core::ops::Add;

const CHACHA_RESULT_SIZE: usize = CHACHA_SIZE * size_of::<u32>();

type ChaChaMatrix = [u32; CHACHA_SIZE];
type ChaChaResult = [u8; CHACHA_RESULT_SIZE];

#[derive(Clone)]
#[repr(C)]
pub struct ChaCha {
    row_a: Row,
    row_b: Row,
    row_c: Row,
    row_d: Row,
}

impl Add for ChaCha {
    type Output = ChaChaMatrix;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut a: ChaChaMatrix = unsafe { transmute(self) };
        let b: ChaChaMatrix = unsafe { transmute(rhs) };
        a.iter_mut()
            .zip(b)
            .for_each(|(a, b)| *a = a.wrapping_add(b));
        a
    }
}

impl From<u8> for ChaCha {
    #[inline]
    fn from(value: u8) -> Self {
        let mut result = ChaCha::from([value; CHACHA_SEED_LEN]);
        unsafe {
            // Tests expect the counter to start at 0.
            result.row_d.u64x2[0] = 0;
        }
        result
    }
}

impl From<[u8; CHACHA_SEED_LEN]> for ChaCha {
    #[inline]
    fn from(value: [u8; CHACHA_SEED_LEN]) -> Self {
        unsafe {
            const CHACHA_SEED_LEN_ROW: usize = CHACHA_SEED_LEN / size_of::<Row>();
            let rows: [Row; CHACHA_SEED_LEN_ROW] = transmute(value);
            Self {
                row_a: ROW_A,
                row_b: rows[0],
                row_c: rows[1],
                row_d: rows[2],
            }
        }
    }
}

impl ChaCha {
    #[inline]
    fn quarter_round(&mut self, a: usize, b: usize, c: usize, d: usize) {
        let matrix: &mut ChaChaMatrix = unsafe { transmute(self) };

        matrix[a] = matrix[a].wrapping_add(matrix[b]);
        matrix[d] ^= matrix[a];
        matrix[d] = matrix[d].rotate_left(16);

        matrix[c] = matrix[c].wrapping_add(matrix[d]);
        matrix[b] ^= matrix[c];
        matrix[b] = matrix[b].rotate_left(12);

        matrix[a] = matrix[a].wrapping_add(matrix[b]);
        matrix[d] ^= matrix[a];
        matrix[d] = matrix[d].rotate_left(8);

        matrix[c] = matrix[c].wrapping_add(matrix[d]);
        matrix[b] ^= matrix[c];
        matrix[b] = matrix[b].rotate_left(7);
    }

    #[inline]
    fn increment_djb(&mut self) {
        unsafe {
            // Index 12 and 13 of the chacha matrix are treated as a
            // single 64-bit integer and incremented.
            self.row_d.u64x2[0] = self.row_d.u64x2[0].wrapping_add(1);
        }
    }

    #[inline]
    fn increment_ietf(&mut self) {
        unsafe {
            // Index 12 of the chacha matrix is incremented in isolation.
            self.row_d.u32x4[0] = self.row_d.u32x4[0].wrapping_add(1);
        }
    }

    #[inline]
    pub fn fill<R: DoubleRounds, V: Variant>(&mut self, dst: &mut [u8]) {
        let src = repeat_with(|| self.get_block::<R, V>()).flatten();
        dst.iter_mut().zip(src).for_each(|(dst_val, src_val)| {
            *dst_val = src_val;
        });
    }

    #[inline(never)]
    pub fn get_block<R: DoubleRounds, V: Variant>(&mut self) -> ChaChaResult {
        let mut cur = self.clone();

        for _ in 0..R::COUNT {
            // Column rounds
            cur.quarter_round(0, 4, 8, 12);
            cur.quarter_round(1, 5, 9, 13);
            cur.quarter_round(2, 6, 10, 14);
            cur.quarter_round(3, 7, 11, 15);
            // Diagonal rounds
            cur.quarter_round(0, 5, 10, 15);
            cur.quarter_round(1, 6, 11, 12);
            cur.quarter_round(2, 7, 8, 13);
            cur.quarter_round(3, 4, 9, 14);
        }

        let result = cur + self.clone();

        match V::VAR {
            Variants::Djb => self.increment_djb(),
            Variants::Ietf => self.increment_ietf(),
        }

        unsafe { transmute(result) }
    }
}

#[test]
fn reference_8_rounds() {
    let next_block = |c: &mut ChaCha| c.get_block::<R8, Djb>();

    // TC1: All zero key and IV.
    let mut chacha = ChaCha::from(0);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x3e, 0x00, 0xef, 0x2f, 0x89, 0x5f, 0x40, 0xd6, 0x7f, 0x5b, 0xb8, 0xe8, 0x1f, 0x09,
            0xa5, 0xa1, 0x2c, 0x84, 0x0e, 0xc3, 0xce, 0x9a, 0x7f, 0x3b, 0x18, 0x1b, 0xe1, 0x88,
            0xef, 0x71, 0x1a, 0x1e, 0x98, 0x4c, 0xe1, 0x72, 0xb9, 0x21, 0x6f, 0x41, 0x9f, 0x44,
            0x53, 0x67, 0x45, 0x6d, 0x56, 0x19, 0x31, 0x4a, 0x42, 0xa3, 0xda, 0x86, 0xb0, 0x01,
            0x38, 0x7b, 0xfd, 0xb8, 0x0e, 0x0c, 0xfe, 0x42,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xd2, 0xae, 0xfa, 0x0d, 0xea, 0xa5, 0xc1, 0x51, 0xbf, 0x0a, 0xdb, 0x6c, 0x01, 0xf2,
            0xa5, 0xad, 0xc0, 0xfd, 0x58, 0x12, 0x59, 0xf9, 0xa2, 0xaa, 0xdc, 0xf2, 0x0f, 0x8f,
            0xd5, 0x66, 0xa2, 0x6b, 0x50, 0x32, 0xec, 0x38, 0xbb, 0xc5, 0xda, 0x98, 0xee, 0x0c,
            0x6f, 0x56, 0x8b, 0x87, 0x2a, 0x65, 0xa0, 0x8a, 0xbf, 0x25, 0x1d, 0xeb, 0x21, 0xbb,
            0x4b, 0x56, 0xe5, 0xd8, 0x82, 0x1e, 0x68, 0xaa,
        ],
    );

    // TC2: Single bit in key set. All zero IV.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_b.u8x16[0] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xcf, 0x5e, 0xe9, 0xa0, 0x49, 0x4a, 0xa9, 0x61, 0x3e, 0x05, 0xd5, 0xed, 0x72, 0x5b,
            0x80, 0x4b, 0x12, 0xf4, 0xa4, 0x65, 0xee, 0x63, 0x5a, 0xcc, 0x3a, 0x31, 0x1d, 0xe8,
            0x74, 0x04, 0x89, 0xea, 0x28, 0x9d, 0x04, 0xf4, 0x3c, 0x75, 0x18, 0xdb, 0x56, 0xeb,
            0x44, 0x33, 0xe4, 0x98, 0xa1, 0x23, 0x8c, 0xd8, 0x46, 0x4d, 0x37, 0x63, 0xdd, 0xbb,
            0x92, 0x22, 0xee, 0x3b, 0xd8, 0xfa, 0xe3, 0xc8,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xb4, 0x35, 0x5a, 0x7d, 0x93, 0xdd, 0x88, 0x67, 0x08, 0x9e, 0xe6, 0x43, 0x55, 0x8b,
            0x95, 0x75, 0x4e, 0xfa, 0x2b, 0xd1, 0xa8, 0xa1, 0xe2, 0xd7, 0x5b, 0xcd, 0xb3, 0x20,
            0x15, 0x54, 0x26, 0x38, 0x29, 0x19, 0x41, 0xfe, 0xb4, 0x99, 0x65, 0x58, 0x7c, 0x4f,
            0xdf, 0xe2, 0x19, 0xcf, 0x0e, 0xc1, 0x32, 0xa6, 0xcd, 0x4d, 0xc0, 0x67, 0x39, 0x2e,
            0x67, 0x98, 0x2f, 0xe5, 0x32, 0x78, 0xc0, 0xb4,
        ],
    );

    // TC3: Single bit in IV set. All zero key.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_d.u8x16[8] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x2b, 0x8f, 0x4b, 0xb3, 0x79, 0x83, 0x06, 0xca, 0x51, 0x30, 0xd4, 0x7c, 0x4f, 0x8d,
            0x4e, 0xd1, 0x3a, 0xa0, 0xed, 0xcc, 0xc1, 0xbe, 0x69, 0x42, 0x09, 0x0f, 0xae, 0xec,
            0xa0, 0xd7, 0x59, 0x9b, 0x7f, 0xf0, 0xfe, 0x61, 0x6b, 0xb2, 0x5a, 0xa0, 0x15, 0x3a,
            0xd6, 0xfd, 0xc8, 0x8b, 0x95, 0x49, 0x03, 0xc2, 0x24, 0x26, 0xd4, 0x78, 0xb9, 0x7b,
            0x22, 0xb8, 0xf9, 0xb1, 0xdb, 0x00, 0xcf, 0x06,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x47, 0x0b, 0xdf, 0xfb, 0xc4, 0x88, 0xa8, 0xb7, 0xc7, 0x01, 0xeb, 0xf4, 0x06, 0x1d,
            0x75, 0xc5, 0x96, 0x91, 0x86, 0x49, 0x7c, 0x95, 0x36, 0x78, 0x09, 0xaf, 0xa8, 0x0b,
            0xd8, 0x43, 0xb0, 0x40, 0xa7, 0x9a, 0xbc, 0x6e, 0x73, 0xa9, 0x17, 0x57, 0xf1, 0xdb,
            0x73, 0xc8, 0xea, 0xcf, 0xa5, 0x43, 0xb3, 0x8f, 0x28, 0x9d, 0x06, 0x5a, 0xb2, 0xf3,
            0x03, 0x2d, 0x37, 0x7b, 0x8c, 0x37, 0xfe, 0x46,
        ],
    );

    // TC4: All bits in key and IV are set.
    chacha = ChaCha::from(0xFF);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xe1, 0x63, 0xbb, 0xf8, 0xc9, 0xa7, 0x39, 0xd1, 0x89, 0x25, 0xee, 0x83, 0x62, 0xda,
            0xd2, 0xcd, 0xc9, 0x73, 0xdf, 0x05, 0x22, 0x5a, 0xfb, 0x2a, 0xa2, 0x63, 0x96, 0xf2,
            0xa9, 0x84, 0x9a, 0x4a, 0x44, 0x5e, 0x05, 0x47, 0xd3, 0x1c, 0x16, 0x23, 0xc5, 0x37,
            0xdf, 0x4b, 0xa8, 0x5c, 0x70, 0xa9, 0x88, 0x4a, 0x35, 0xbc, 0xbf, 0x3d, 0xfa, 0xb0,
            0x77, 0xe9, 0x8b, 0x0f, 0x68, 0x13, 0x5f, 0x54,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x81, 0xd4, 0x93, 0x3f, 0x8b, 0x32, 0x2a, 0xc0, 0xcd, 0x76, 0x2c, 0x27, 0x23, 0x5c,
            0xe2, 0xb3, 0x15, 0x34, 0xe0, 0x24, 0x4a, 0x9a, 0x2f, 0x1f, 0xd5, 0xe9, 0x44, 0x98,
            0xd4, 0x7f, 0xf1, 0x08, 0x79, 0x0c, 0x00, 0x9c, 0xf9, 0xe1, 0xa3, 0x48, 0x03, 0x2a,
            0x76, 0x94, 0xcb, 0x28, 0x02, 0x4c, 0xd9, 0x6d, 0x34, 0x98, 0x36, 0x1e, 0xdb, 0x17,
            0x85, 0xaf, 0x75, 0x2d, 0x18, 0x7a, 0xb5, 0x4b,
        ],
    );

    // TC5: Every even bit set in key and IV.
    chacha = ChaCha::from(0x55);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x7c, 0xb7, 0x82, 0x14, 0xe4, 0xd3, 0x46, 0x5b, 0x6d, 0xc6, 0x2c, 0xf7, 0xa1, 0x53,
            0x8c, 0x88, 0x99, 0x69, 0x52, 0xb4, 0xfb, 0x72, 0xcb, 0x61, 0x05, 0xf1, 0x24, 0x3c,
            0xe3, 0x44, 0x2e, 0x29, 0x75, 0xa5, 0x9e, 0xbc, 0xd2, 0xb2, 0xa5, 0x98, 0x29, 0x0d,
            0x75, 0x38, 0x49, 0x1f, 0xe6, 0x5b, 0xdb, 0xfe, 0xfd, 0x06, 0x0d, 0x88, 0x79, 0x81,
            0x20, 0xa7, 0x0d, 0x04, 0x9d, 0xc2, 0x67, 0x7d,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xd4, 0x8f, 0xf5, 0xa2, 0x51, 0x3e, 0x49, 0x7a, 0x5d, 0x54, 0x80, 0x2d, 0x74, 0x84,
            0xc4, 0xf1, 0x08, 0x39, 0x44, 0xd8, 0xd0, 0xd1, 0x4d, 0x64, 0x82, 0xce, 0x09, 0xf7,
            0xe5, 0xeb, 0xf2, 0x0b, 0x29, 0x80, 0x7d, 0x62, 0xc3, 0x18, 0x74, 0xd0, 0x2f, 0x5d,
            0x3c, 0xc8, 0x53, 0x81, 0xa7, 0x45, 0xec, 0xbc, 0x60, 0x52, 0x52, 0x05, 0xe3, 0x00,
            0xa7, 0x69, 0x61, 0xbf, 0xe5, 0x1a, 0xc0, 0x7c,
        ],
    );

    // TC6: Every odd bit set in key and IV.
    chacha = ChaCha::from(0xAA);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x40, 0xf9, 0xab, 0x86, 0xc8, 0xf9, 0xa1, 0xa0, 0xcd, 0xc0, 0x5a, 0x75, 0xe5, 0x53,
            0x1b, 0x61, 0x2d, 0x71, 0xef, 0x7f, 0x0c, 0xf9, 0xe3, 0x87, 0xdf, 0x6e, 0xd6, 0x97,
            0x2f, 0x0a, 0xae, 0x21, 0x31, 0x1a, 0xa5, 0x81, 0xf8, 0x16, 0xc9, 0x0e, 0x8a, 0x99,
            0xde, 0x99, 0x0b, 0x6b, 0x95, 0xaa, 0xc9, 0x24, 0x50, 0xf4, 0xe1, 0x12, 0x71, 0x26,
            0x67, 0xb8, 0x04, 0xc9, 0x9e, 0x9c, 0x6e, 0xda,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xf8, 0xd1, 0x44, 0xf5, 0x60, 0xc8, 0xc0, 0xea, 0x36, 0x88, 0x0d, 0x3b, 0x77, 0x87,
            0x4c, 0x9a, 0x91, 0x03, 0xd1, 0x47, 0xf6, 0xde, 0xd3, 0x86, 0x28, 0x48, 0x01, 0xa4,
            0xee, 0x15, 0x8e, 0x5e, 0xa4, 0xf9, 0xc0, 0x93, 0xfc, 0x55, 0xfd, 0x34, 0x4c, 0x33,
            0x34, 0x9d, 0xc5, 0xb6, 0x99, 0xe2, 0x1d, 0xc8, 0x3b, 0x42, 0x96, 0xf9, 0x2e, 0xe3,
            0xec, 0xab, 0xf3, 0xd5, 0x1f, 0x95, 0xfe, 0x3f,
        ],
    );

    // TC7: Sequence patterns in key and IV.
    chacha = ChaCha::from([
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
        0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b,
        0x5a, 0x69, 0x78,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xdb, 0x43, 0xad, 0x9d, 0x1e, 0x84, 0x2d, 0x12, 0x72, 0xe4, 0x53, 0x0e, 0x27, 0x6b,
            0x3f, 0x56, 0x8f, 0x88, 0x59, 0xb3, 0xf7, 0xcf, 0x6d, 0x9d, 0x2c, 0x74, 0xfa, 0x53,
            0x80, 0x8c, 0xb5, 0x15, 0x7a, 0x8e, 0xbf, 0x46, 0xad, 0x3d, 0xcc, 0x4b, 0x6c, 0x7d,
            0xad, 0xde, 0x13, 0x17, 0x84, 0xb0, 0x12, 0x0e, 0x0e, 0x22, 0xf6, 0xd5, 0xf9, 0xff,
            0xa7, 0x40, 0x7d, 0x4a, 0x21, 0xb6, 0x95, 0xd9,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xc5, 0xdd, 0x30, 0xbf, 0x55, 0x61, 0x2f, 0xab, 0x9b, 0xdd, 0x11, 0x89, 0x20, 0xc1,
            0x98, 0x16, 0x47, 0x0c, 0x7f, 0x5d, 0xcd, 0x42, 0x32, 0x5d, 0xbb, 0xed, 0x8c, 0x57,
            0xa5, 0x62, 0x81, 0xc1, 0x44, 0xcb, 0x0f, 0x03, 0xe8, 0x1b, 0x30, 0x04, 0x62, 0x4e,
            0x06, 0x50, 0xa1, 0xce, 0x5a, 0xfa, 0xf9, 0xa7, 0xcd, 0x81, 0x63, 0xf6, 0xdb, 0xd7,
            0x26, 0x02, 0x25, 0x7d, 0xd9, 0x6e, 0x47, 0x1e,
        ],
    );

    // TC8: key: 'All your base are belong to us!, IV: 'IETF2013'
    chacha = ChaCha::from([
        0xc4, 0x6e, 0xc1, 0xb1, 0x8c, 0xe8, 0xa8, 0x78, 0x72, 0x5a, 0x37, 0xe7, 0x80, 0xdf, 0xb7,
        0x35, 0x1f, 0x68, 0xed, 0x2e, 0x19, 0x4c, 0x79, 0xfb, 0xc6, 0xae, 0xbe, 0xe1, 0xa6, 0x67,
        0x97, 0x5d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0xda, 0x31, 0xd5, 0xcf,
        0x68, 0x82, 0x21,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x83, 0x87, 0x51, 0xb4, 0x2d, 0x8d, 0xdd, 0x8a, 0x3d, 0x77, 0xf4, 0x88, 0x25, 0xa2,
            0xba, 0x75, 0x2c, 0xf4, 0x04, 0x7c, 0xb3, 0x08, 0xa5, 0x97, 0x8e, 0xf2, 0x74, 0x97,
            0x3b, 0xe3, 0x74, 0xc9, 0x6a, 0xd8, 0x48, 0x06, 0x58, 0x71, 0x41, 0x7b, 0x08, 0xf0,
            0x34, 0xe6, 0x81, 0xfe, 0x46, 0xa9, 0x3f, 0x7d, 0x5c, 0x61, 0xd1, 0x30, 0x66, 0x14,
            0xd4, 0xaa, 0xf2, 0x57, 0xa7, 0xcf, 0xf0, 0x8b,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x16, 0xf2, 0xfd, 0xa1, 0x70, 0xcc, 0x18, 0xa4, 0xb5, 0x8a, 0x26, 0x67, 0xed, 0x96,
            0x27, 0x74, 0xaf, 0x79, 0x2a, 0x6e, 0x7f, 0x3c, 0x77, 0x99, 0x25, 0x40, 0x71, 0x1a,
            0x7a, 0x13, 0x6d, 0x7e, 0x8a, 0x2f, 0x8d, 0x3f, 0x93, 0x81, 0x67, 0x09, 0xd4, 0x5a,
            0x3f, 0xa5, 0xf8, 0xce, 0x72, 0xfd, 0xe1, 0x5b, 0xe7, 0xb8, 0x41, 0xac, 0xba, 0x3a,
            0x2a, 0xbd, 0x55, 0x72, 0x28, 0xd9, 0xfe, 0x4f,
        ],
    );
}

#[test]
fn reference_12_rounds() {
    let next_block = |c: &mut ChaCha| c.get_block::<R12, Djb>();

    // TC1: All zero key and IV.
    let mut chacha = ChaCha::from(0);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x9b, 0xf4, 0x9a, 0x6a, 0x07, 0x55, 0xf9, 0x53, 0x81, 0x1f, 0xce, 0x12, 0x5f, 0x26,
            0x83, 0xd5, 0x04, 0x29, 0xc3, 0xbb, 0x49, 0xe0, 0x74, 0x14, 0x7e, 0x00, 0x89, 0xa5,
            0x2e, 0xae, 0x15, 0x5f, 0x05, 0x64, 0xf8, 0x79, 0xd2, 0x7a, 0xe3, 0xc0, 0x2c, 0xe8,
            0x28, 0x34, 0xac, 0xfa, 0x8c, 0x79, 0x3a, 0x62, 0x9f, 0x2c, 0xa0, 0xde, 0x69, 0x19,
            0x61, 0x0b, 0xe8, 0x2f, 0x41, 0x13, 0x26, 0xbe,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x0b, 0xd5, 0x88, 0x41, 0x20, 0x3e, 0x74, 0xfe, 0x86, 0xfc, 0x71, 0x33, 0x8c, 0xe0,
            0x17, 0x3d, 0xc6, 0x28, 0xeb, 0xb7, 0x19, 0xbd, 0xcb, 0xcc, 0x15, 0x15, 0x85, 0x21,
            0x4c, 0xc0, 0x89, 0xb4, 0x42, 0x25, 0x8d, 0xcd, 0xa1, 0x4c, 0xf1, 0x11, 0xc6, 0x02,
            0xb8, 0x97, 0x1b, 0x8c, 0xc8, 0x43, 0xe9, 0x1e, 0x46, 0xca, 0x90, 0x51, 0x51, 0xc0,
            0x27, 0x44, 0xa6, 0xb0, 0x17, 0xe6, 0x93, 0x16,
        ],
    );

    // TC2: Single bit in key set. All zero IV.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_b.u8x16[0] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x12, 0x05, 0x6e, 0x59, 0x5d, 0x56, 0xb0, 0xf6, 0xee, 0xf0, 0x90, 0xf0, 0xcd, 0x25,
            0xa2, 0x09, 0x49, 0x24, 0x8c, 0x27, 0x90, 0x52, 0x5d, 0x0f, 0x93, 0x02, 0x18, 0xff,
            0x0b, 0x4d, 0xdd, 0x10, 0xa6, 0x00, 0x22, 0x39, 0xd9, 0xa4, 0x54, 0xe2, 0x9e, 0x10,
            0x7a, 0x7d, 0x06, 0xfe, 0xfd, 0xfe, 0xf0, 0x21, 0x0f, 0xeb, 0xa0, 0x44, 0xf9, 0xf2,
            0x9b, 0x17, 0x72, 0xc9, 0x60, 0xdc, 0x29, 0xc0,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x0c, 0x73, 0x66, 0xc5, 0xcb, 0xc6, 0x04, 0x24, 0x0e, 0x66, 0x5e, 0xb0, 0x2a, 0x69,
            0x37, 0x2a, 0x7a, 0xf9, 0x79, 0xb2, 0x6f, 0xbb, 0x78, 0x09, 0x2a, 0xc7, 0xc4, 0xb8,
            0x80, 0x29, 0xa7, 0xc8, 0x54, 0x51, 0x3b, 0xc2, 0x17, 0xbb, 0xfc, 0x7d, 0x90, 0x43,
            0x2e, 0x30, 0x8e, 0xba, 0x15, 0xaf, 0xc6, 0x5a, 0xeb, 0x48, 0xef, 0x10, 0x0d, 0x56,
            0x01, 0xe6, 0xaf, 0xba, 0x25, 0x71, 0x17, 0xa9,
        ],
    );

    // TC3: Single bit in IV set. All zero key.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_d.u8x16[8] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x64, 0xb8, 0xbd, 0xf8, 0x7b, 0x82, 0x8c, 0x4b, 0x6d, 0xba, 0xf7, 0xef, 0x69, 0x8d,
            0xe0, 0x3d, 0xf8, 0xb3, 0x3f, 0x63, 0x57, 0x14, 0x41, 0x8f, 0x98, 0x36, 0xad, 0xe5,
            0x9b, 0xe1, 0x29, 0x69, 0x46, 0xc9, 0x53, 0xa0, 0xf3, 0x8e, 0xcf, 0xfc, 0x9e, 0xcb,
            0x98, 0xe8, 0x1d, 0x5d, 0x99, 0xa5, 0xed, 0xfc, 0x8f, 0x9a, 0x0a, 0x45, 0xb9, 0xe4,
            0x1e, 0xf3, 0xb3, 0x1f, 0x02, 0x8f, 0x1d, 0x0f,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x55, 0x9d, 0xb4, 0xa7, 0xf2, 0x22, 0xc4, 0x42, 0xfe, 0x23, 0xb9, 0xa2, 0x59, 0x6a,
            0x88, 0x28, 0x51, 0x22, 0xee, 0x4f, 0x13, 0x63, 0x89, 0x6e, 0xa7, 0x7c, 0xa1, 0x50,
            0x91, 0x2a, 0xc7, 0x23, 0xbf, 0xf0, 0x4b, 0x02, 0x6a, 0x2f, 0x80, 0x7e, 0x03, 0xb2,
            0x9c, 0x02, 0x07, 0x7d, 0x7b, 0x06, 0xfc, 0x1a, 0xb9, 0x82, 0x7c, 0x13, 0xc8, 0x01,
            0x3a, 0x6d, 0x83, 0xbd, 0x3b, 0x52, 0xa2, 0x6f,
        ],
    );

    // TC4: All bits in key and IV are set.
    chacha = ChaCha::from(0xFF);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x04, 0xbf, 0x88, 0xda, 0xe8, 0xe4, 0x7a, 0x22, 0x8f, 0xa4, 0x7b, 0x7e, 0x63, 0x79,
            0x43, 0x4b, 0xa6, 0x64, 0xa7, 0xd2, 0x8f, 0x4d, 0xab, 0x84, 0xe5, 0xf8, 0xb4, 0x64,
            0xad, 0xd2, 0x0c, 0x3a, 0xca, 0xa6, 0x9c, 0x5a, 0xb2, 0x21, 0xa2, 0x3a, 0x57, 0xeb,
            0x5f, 0x34, 0x5c, 0x96, 0xf4, 0xd1, 0x32, 0x2d, 0x0a, 0x2f, 0xf7, 0xa9, 0xcd, 0x43,
            0x40, 0x1c, 0xd5, 0x36, 0x63, 0x9a, 0x61, 0x5a,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x5c, 0x94, 0x29, 0xb5, 0x5c, 0xa3, 0xc1, 0xb5, 0x53, 0x54, 0x55, 0x96, 0x69, 0xa1,
            0x54, 0xac, 0xa4, 0x6c, 0xd7, 0x61, 0xc4, 0x1a, 0xb8, 0xac, 0xe3, 0x85, 0x36, 0x3b,
            0x95, 0x67, 0x5f, 0x06, 0x8e, 0x18, 0xdb, 0x5a, 0x67, 0x3c, 0x11, 0x29, 0x1b, 0xd4,
            0x18, 0x78, 0x92, 0xa9, 0xa3, 0xa3, 0x35, 0x14, 0xf3, 0x71, 0x2b, 0x26, 0xc1, 0x30,
            0x26, 0x10, 0x32, 0x98, 0xed, 0x76, 0xbc, 0x9a,
        ],
    );

    // TC5: Every even bit set in key and IV.
    chacha = ChaCha::from(0x55);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xa6, 0x00, 0xf0, 0x77, 0x27, 0xff, 0x93, 0xf3, 0xda, 0x00, 0xdd, 0x74, 0xcc, 0x3e,
            0x8b, 0xfb, 0x5c, 0xa7, 0x30, 0x2f, 0x6a, 0x0a, 0x29, 0x44, 0x95, 0x3d, 0xe0, 0x04,
            0x50, 0xee, 0xcd, 0x40, 0xb8, 0x60, 0xf6, 0x60, 0x49, 0xf2, 0xea, 0xed, 0x63, 0xb2,
            0xef, 0x39, 0xcc, 0x31, 0x0d, 0x2c, 0x48, 0x8f, 0x5d, 0x9a, 0x24, 0x1b, 0x61, 0x5d,
            0xc0, 0xab, 0x70, 0xf9, 0x21, 0xb9, 0x1b, 0x95,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x14, 0x0e, 0xff, 0x4a, 0xa4, 0x95, 0xac, 0x61, 0x28, 0x9b, 0x6b, 0xc5, 0x7d, 0xe0,
            0x72, 0x41, 0x9d, 0x09, 0xda, 0xa7, 0xa7, 0x24, 0x39, 0x90, 0xda, 0xf3, 0x48, 0xa8,
            0xf2, 0x83, 0x1e, 0x59, 0x7c, 0xf3, 0x79, 0xb3, 0xb2, 0x84, 0xf0, 0x0b, 0xda, 0x27,
            0xa4, 0xc6, 0x80, 0x85, 0x37, 0x4a, 0x8a, 0x5c, 0x38, 0xde, 0xd6, 0x2d, 0x11, 0x41,
            0xca, 0xe0, 0xbb, 0x83, 0x8d, 0xdc, 0x22, 0x32,
        ],
    );

    // TC6: Every odd bit set in key and IV.
    chacha = ChaCha::from(0xAA);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x85, 0x65, 0x05, 0xb0, 0x1d, 0x3b, 0x47, 0xaa, 0xe0, 0x3d, 0x6a, 0x97, 0xaa, 0x0f,
            0x03, 0x3a, 0x9a, 0xdc, 0xc9, 0x43, 0x77, 0xba, 0xbd, 0x86, 0x08, 0x86, 0x4f, 0xb3,
            0xf6, 0x25, 0xb6, 0xe3, 0x14, 0xf0, 0x86, 0x15, 0x8f, 0x9f, 0x72, 0x5d, 0x81, 0x1e,
            0xeb, 0x95, 0x3b, 0x7f, 0x74, 0x70, 0x76, 0xe4, 0xc3, 0xf6, 0x39, 0xfa, 0x84, 0x1f,
            0xad, 0x6c, 0x9a, 0x70, 0x9e, 0x62, 0x13, 0x97,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x6d, 0xd6, 0xee, 0x9b, 0x5e, 0x1e, 0x2e, 0x67, 0x6b, 0x1c, 0x9e, 0x2b, 0x82, 0xc2,
            0xe9, 0x6c, 0x16, 0x48, 0x43, 0x7b, 0xff, 0x2f, 0x01, 0x26, 0xb7, 0x4e, 0x8c, 0xe0,
            0xa9, 0xb0, 0x6d, 0x17, 0x20, 0xac, 0x0b, 0x6f, 0x09, 0x08, 0x6f, 0x28, 0xbc, 0x20,
            0x15, 0x87, 0xf0, 0x53, 0x5e, 0xd9, 0x38, 0x52, 0x70, 0xd0, 0x8b, 0x4a, 0x93, 0x82,
            0xf1, 0x8f, 0x82, 0xdb, 0xde, 0x18, 0x21, 0x0e,
        ],
    );

    // TC7: Sequence patterns in key and IV.
    chacha = ChaCha::from([
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
        0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b,
        0x5a, 0x69, 0x78,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x7e, 0xd1, 0x2a, 0x3a, 0x63, 0x91, 0x2a, 0xe9, 0x41, 0xba, 0x6d, 0x4c, 0x0d, 0x5e,
            0x86, 0x2e, 0x56, 0x8b, 0x0e, 0x55, 0x89, 0x34, 0x69, 0x35, 0x50, 0x5f, 0x06, 0x4b,
            0x8c, 0x26, 0x98, 0xdb, 0xf7, 0xd8, 0x50, 0x66, 0x7d, 0x8e, 0x67, 0xbe, 0x63, 0x9f,
            0x3b, 0x4f, 0x6a, 0x16, 0xf9, 0x2e, 0x65, 0xea, 0x80, 0xf6, 0xc7, 0x42, 0x94, 0x45,
            0xda, 0x1f, 0xc2, 0xc1, 0xb9, 0x36, 0x50, 0x40,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xe3, 0x2e, 0x50, 0xc4, 0x10, 0x6f, 0x3b, 0x3d, 0xa1, 0xce, 0x7c, 0xcb, 0x1e, 0x71,
            0x40, 0xb1, 0x53, 0x49, 0x3c, 0x0f, 0x3a, 0xd9, 0xa9, 0xbc, 0xff, 0x07, 0x7e, 0xc4,
            0x59, 0x6f, 0x1d, 0x0f, 0x29, 0xbf, 0x9c, 0xba, 0xa5, 0x02, 0x82, 0x0f, 0x73, 0x2a,
            0xf5, 0xa9, 0x3c, 0x49, 0xee, 0xe3, 0x3d, 0x1c, 0x4f, 0x12, 0xaf, 0x3b, 0x42, 0x97,
            0xaf, 0x91, 0xfe, 0x41, 0xea, 0x9e, 0x94, 0xa2,
        ],
    );

    // TC8: key: 'All your base are belong to us!, IV: 'IETF2013'
    chacha = ChaCha::from([
        0xc4, 0x6e, 0xc1, 0xb1, 0x8c, 0xe8, 0xa8, 0x78, 0x72, 0x5a, 0x37, 0xe7, 0x80, 0xdf, 0xb7,
        0x35, 0x1f, 0x68, 0xed, 0x2e, 0x19, 0x4c, 0x79, 0xfb, 0xc6, 0xae, 0xbe, 0xe1, 0xa6, 0x67,
        0x97, 0x5d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0xda, 0x31, 0xd5, 0xcf,
        0x68, 0x82, 0x21,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x14, 0x82, 0x07, 0x27, 0x84, 0xbc, 0x6d, 0x06, 0xb4, 0xe7, 0x3b, 0xdc, 0x11, 0x8b,
            0xc0, 0x10, 0x3c, 0x79, 0x76, 0x78, 0x6c, 0xa9, 0x18, 0xe0, 0x69, 0x86, 0xaa, 0x25,
            0x1f, 0x7e, 0x9c, 0xc1, 0xb2, 0x74, 0x9a, 0x0a, 0x16, 0xee, 0x83, 0xb4, 0x24, 0x2d,
            0x2e, 0x99, 0xb0, 0x8d, 0x7c, 0x20, 0x09, 0x2b, 0x80, 0xbc, 0x46, 0x6c, 0x87, 0x28,
            0x3b, 0x61, 0xb1, 0xb3, 0x9d, 0x0f, 0xfb, 0xab,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xd9, 0x4b, 0x11, 0x6b, 0xc1, 0xeb, 0xdb, 0x32, 0x9b, 0x9e, 0x4f, 0x62, 0x0d, 0xb6,
            0x95, 0x54, 0x4a, 0x8e, 0x3d, 0x9b, 0x68, 0x47, 0x3d, 0x0c, 0x97, 0x5a, 0x46, 0xad,
            0x96, 0x6e, 0xd6, 0x31, 0xe4, 0x2a, 0xff, 0x53, 0x0a, 0xd5, 0xea, 0xc7, 0xd8, 0x04,
            0x7a, 0xdf, 0xa1, 0xe5, 0x11, 0x3c, 0x91, 0xf3, 0xe3, 0xb8, 0x83, 0xf1, 0xd1, 0x89,
            0xac, 0x1c, 0x8f, 0xe0, 0x7b, 0xa5, 0xa4, 0x2b,
        ],
    );
}

#[test]
fn reference_20_rounds() {
    let next_block = |c: &mut ChaCha| c.get_block::<R20, Djb>();

    // TC1: All zero key and IV.
    let mut chacha = ChaCha::from(0);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90, 0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86,
            0xbd, 0x28, 0xbd, 0xd2, 0x19, 0xb8, 0xa0, 0x8d, 0xed, 0x1a, 0xa8, 0x36, 0xef, 0xcc,
            0x8b, 0x77, 0x0d, 0xc7, 0xda, 0x41, 0x59, 0x7c, 0x51, 0x57, 0x48, 0x8d, 0x77, 0x24,
            0xe0, 0x3f, 0xb8, 0xd8, 0x4a, 0x37, 0x6a, 0x43, 0xb8, 0xf4, 0x15, 0x18, 0xa1, 0x1c,
            0xc3, 0x87, 0xb6, 0x69, 0xb2, 0xee, 0x65, 0x86,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x9f, 0x07, 0xe7, 0xbe, 0x55, 0x51, 0x38, 0x7a, 0x98, 0xba, 0x97, 0x7c, 0x73, 0x2d,
            0x08, 0x0d, 0xcb, 0x0f, 0x29, 0xa0, 0x48, 0xe3, 0x65, 0x69, 0x12, 0xc6, 0x53, 0x3e,
            0x32, 0xee, 0x7a, 0xed, 0x29, 0xb7, 0x21, 0x76, 0x9c, 0xe6, 0x4e, 0x43, 0xd5, 0x71,
            0x33, 0xb0, 0x74, 0xd8, 0x39, 0xd5, 0x31, 0xed, 0x1f, 0x28, 0x51, 0x0a, 0xfb, 0x45,
            0xac, 0xe1, 0x0a, 0x1f, 0x4b, 0x79, 0x4d, 0x6f,
        ],
    );

    // TC2: Single bit in key set. All zero IV.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_b.u8x16[0] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xc5, 0xd3, 0x0a, 0x7c, 0xe1, 0xec, 0x11, 0x93, 0x78, 0xc8, 0x4f, 0x48, 0x7d, 0x77,
            0x5a, 0x85, 0x42, 0xf1, 0x3e, 0xce, 0x23, 0x8a, 0x94, 0x55, 0xe8, 0x22, 0x9e, 0x88,
            0x8d, 0xe8, 0x5b, 0xbd, 0x29, 0xeb, 0x63, 0xd0, 0xa1, 0x7a, 0x5b, 0x99, 0x9b, 0x52,
            0xda, 0x22, 0xbe, 0x40, 0x23, 0xeb, 0x07, 0x62, 0x0a, 0x54, 0xf6, 0xfa, 0x6a, 0xd8,
            0x73, 0x7b, 0x71, 0xeb, 0x04, 0x64, 0xda, 0xc0,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x10, 0xf6, 0x56, 0xe6, 0xd1, 0xfd, 0x55, 0x05, 0x3e, 0x50, 0xc4, 0x87, 0x5c, 0x99,
            0x30, 0xa3, 0x3f, 0x6d, 0x02, 0x63, 0xbd, 0x14, 0xdf, 0xd6, 0xab, 0x8c, 0x70, 0x52,
            0x1c, 0x19, 0x33, 0x8b, 0x23, 0x08, 0xb9, 0x5c, 0xf8, 0xd0, 0xbb, 0x7d, 0x20, 0x2d,
            0x21, 0x02, 0x78, 0x0e, 0xa3, 0x52, 0x8f, 0x1c, 0xb4, 0x85, 0x60, 0xf7, 0x6b, 0x20,
            0xf3, 0x82, 0xb9, 0x42, 0x50, 0x0f, 0xce, 0xac,
        ],
    );

    // TC3: Single bit in IV set. All zero key.
    chacha = ChaCha::from(0);
    unsafe {
        chacha.row_d.u8x16[8] = 1;
    }
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xef, 0x3f, 0xdf, 0xd6, 0xc6, 0x15, 0x78, 0xfb, 0xf5, 0xcf, 0x35, 0xbd, 0x3d, 0xd3,
            0x3b, 0x80, 0x09, 0x63, 0x16, 0x34, 0xd2, 0x1e, 0x42, 0xac, 0x33, 0x96, 0x0b, 0xd1,
            0x38, 0xe5, 0x0d, 0x32, 0x11, 0x1e, 0x4c, 0xaf, 0x23, 0x7e, 0xe5, 0x3c, 0xa8, 0xad,
            0x64, 0x26, 0x19, 0x4a, 0x88, 0x54, 0x5d, 0xdc, 0x49, 0x7a, 0x0b, 0x46, 0x6e, 0x7d,
            0x6b, 0xbd, 0xb0, 0x04, 0x1b, 0x2f, 0x58, 0x6b,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x53, 0x05, 0xe5, 0xe4, 0x4a, 0xff, 0x19, 0xb2, 0x35, 0x93, 0x61, 0x44, 0x67, 0x5e,
            0xfb, 0xe4, 0x40, 0x9e, 0xb7, 0xe8, 0xe5, 0xf1, 0x43, 0x0f, 0x5f, 0x58, 0x36, 0xae,
            0xb4, 0x9b, 0xb5, 0x32, 0x8b, 0x01, 0x7c, 0x4b, 0x9d, 0xc1, 0x1f, 0x8a, 0x03, 0x86,
            0x3f, 0xa8, 0x03, 0xdc, 0x71, 0xd5, 0x72, 0x6b, 0x2b, 0x6b, 0x31, 0xaa, 0x32, 0x70,
            0x8a, 0xfe, 0x5a, 0xf1, 0xd6, 0xb6, 0x90, 0x58,
        ],
    );

    // TC4: All bits in key and IV are set.
    chacha = ChaCha::from(0xFF);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xd9, 0xbf, 0x3f, 0x6b, 0xce, 0x6e, 0xd0, 0xb5, 0x42, 0x54, 0x55, 0x77, 0x67, 0xfb,
            0x57, 0x44, 0x3d, 0xd4, 0x77, 0x89, 0x11, 0xb6, 0x06, 0x05, 0x5c, 0x39, 0xcc, 0x25,
            0xe6, 0x74, 0xb8, 0x36, 0x3f, 0xea, 0xbc, 0x57, 0xfd, 0xe5, 0x4f, 0x79, 0x0c, 0x52,
            0xc8, 0xae, 0x43, 0x24, 0x0b, 0x79, 0xd4, 0x90, 0x42, 0xb7, 0x77, 0xbf, 0xd6, 0xcb,
            0x80, 0xe9, 0x31, 0x27, 0x0b, 0x7f, 0x50, 0xeb,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x5b, 0xac, 0x2a, 0xcd, 0x86, 0xa8, 0x36, 0xc5, 0xdc, 0x98, 0xc1, 0x16, 0xc1, 0x21,
            0x7e, 0xc3, 0x1d, 0x3a, 0x63, 0xa9, 0x45, 0x13, 0x19, 0xf0, 0x97, 0xf3, 0xb4, 0xd6,
            0xda, 0xb0, 0x77, 0x87, 0x19, 0x47, 0x7d, 0x24, 0xd2, 0x4b, 0x40, 0x3a, 0x12, 0x24,
            0x1d, 0x7c, 0xca, 0x06, 0x4f, 0x79, 0x0f, 0x1d, 0x51, 0xcc, 0xaf, 0xf6, 0xb1, 0x66,
            0x7d, 0x4b, 0xbc, 0xa1, 0x95, 0x8c, 0x43, 0x06,
        ],
    );

    // TC5: Every even bit set in key and IV.
    chacha = ChaCha::from(0x55);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xbe, 0xa9, 0x41, 0x1a, 0xa4, 0x53, 0xc5, 0x43, 0x4a, 0x5a, 0xe8, 0xc9, 0x28, 0x62,
            0xf5, 0x64, 0x39, 0x68, 0x55, 0xa9, 0xea, 0x6e, 0x22, 0xd6, 0xd3, 0xb5, 0x0a, 0xe1,
            0xb3, 0x66, 0x33, 0x11, 0xa4, 0xa3, 0x60, 0x6c, 0x67, 0x1d, 0x60, 0x5c, 0xe1, 0x6c,
            0x3a, 0xec, 0xe8, 0xe6, 0x1e, 0xa1, 0x45, 0xc5, 0x97, 0x75, 0x01, 0x7b, 0xee, 0x2f,
            0xa6, 0xf8, 0x8a, 0xfc, 0x75, 0x80, 0x69, 0xf7,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xe0, 0xb8, 0xf6, 0x76, 0xe6, 0x44, 0x21, 0x6f, 0x4d, 0x2a, 0x34, 0x22, 0xd7, 0xfa,
            0x36, 0xc6, 0xc4, 0x93, 0x1a, 0xca, 0x95, 0x0e, 0x9d, 0xa4, 0x27, 0x88, 0xe6, 0xd0,
            0xb6, 0xd1, 0xcd, 0x83, 0x8e, 0xf6, 0x52, 0xe9, 0x7b, 0x14, 0x5b, 0x14, 0x87, 0x1e,
            0xae, 0x6c, 0x68, 0x04, 0xc7, 0x00, 0x4d, 0xb5, 0xac, 0x2f, 0xce, 0x4c, 0x68, 0xc7,
            0x26, 0xd0, 0x04, 0xb1, 0x0f, 0xca, 0xba, 0x86,
        ],
    );

    // TC6: Every odd bit set in key and IV.
    chacha = ChaCha::from(0xAA);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x9a, 0xa2, 0xa9, 0xf6, 0x56, 0xef, 0xde, 0x5a, 0xa7, 0x59, 0x1c, 0x5f, 0xed, 0x4b,
            0x35, 0xae, 0xa2, 0x89, 0x5d, 0xec, 0x7c, 0xb4, 0x54, 0x3b, 0x9e, 0x9f, 0x21, 0xf5,
            0xe7, 0xbc, 0xbc, 0xf3, 0xc4, 0x3c, 0x74, 0x8a, 0x97, 0x08, 0x88, 0xf8, 0x24, 0x83,
            0x93, 0xa0, 0x9d, 0x43, 0xe0, 0xb7, 0xe1, 0x64, 0xbc, 0x4d, 0x0b, 0x0f, 0xb2, 0x40,
            0xa2, 0xd7, 0x21, 0x15, 0xc4, 0x80, 0x89, 0x06,
        ],
    );
    assert_eq!(
        block_2,
        [
            0x72, 0x18, 0x44, 0x89, 0x44, 0x05, 0x45, 0xd0, 0x21, 0xd9, 0x7e, 0xf6, 0xb6, 0x93,
            0xdf, 0xe5, 0xb2, 0xc1, 0x32, 0xd4, 0x7e, 0x6f, 0x04, 0x1c, 0x90, 0x63, 0x65, 0x1f,
            0x96, 0xb6, 0x23, 0xe6, 0x2a, 0x11, 0x99, 0x9a, 0x23, 0xb6, 0xf7, 0xc4, 0x61, 0xb2,
            0x15, 0x30, 0x26, 0xad, 0x5e, 0x86, 0x6a, 0x2e, 0x59, 0x7e, 0xd0, 0x7b, 0x84, 0x01,
            0xde, 0xc6, 0x3a, 0x09, 0x34, 0xc6, 0xb2, 0xa9,
        ],
    );

    // TC7: Sequence patterns in key and IV.
    chacha = ChaCha::from([
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
        0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b,
        0x5a, 0x69, 0x78,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0x9f, 0xad, 0xf4, 0x09, 0xc0, 0x08, 0x11, 0xd0, 0x04, 0x31, 0xd6, 0x7e, 0xfb, 0xd8,
            0x8f, 0xba, 0x59, 0x21, 0x8d, 0x5d, 0x67, 0x08, 0xb1, 0xd6, 0x85, 0x86, 0x3f, 0xab,
            0xbb, 0x0e, 0x96, 0x1e, 0xea, 0x48, 0x0f, 0xd6, 0xfb, 0x53, 0x2b, 0xfd, 0x49, 0x4b,
            0x21, 0x51, 0x01, 0x50, 0x57, 0x42, 0x3a, 0xb6, 0x0a, 0x63, 0xfe, 0x4f, 0x55, 0xf7,
            0xa2, 0x12, 0xe2, 0x16, 0x7c, 0xca, 0xb9, 0x31,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xfb, 0xfd, 0x29, 0xcf, 0x7b, 0xc1, 0xd2, 0x79, 0xed, 0xdf, 0x25, 0xdd, 0x31, 0x6b,
            0xb8, 0x84, 0x3d, 0x6e, 0xde, 0xe0, 0xbd, 0x1e, 0xf1, 0x21, 0xd1, 0x2f, 0xa1, 0x7c,
            0xbc, 0x2c, 0x57, 0x4c, 0xcc, 0xab, 0x5e, 0x27, 0x51, 0x67, 0xb0, 0x8b, 0xd6, 0x86,
            0xf8, 0xa0, 0x9d, 0xf8, 0x7e, 0xc3, 0xff, 0xb3, 0x53, 0x61, 0xb9, 0x4e, 0xbf, 0xa1,
            0x3f, 0xec, 0x0e, 0x48, 0x89, 0xd1, 0x8d, 0xa5,
        ],
    );

    // TC8: key: 'All your base are belong to us!, IV: 'IETF2013'
    chacha = ChaCha::from([
        0xc4, 0x6e, 0xc1, 0xb1, 0x8c, 0xe8, 0xa8, 0x78, 0x72, 0x5a, 0x37, 0xe7, 0x80, 0xdf, 0xb7,
        0x35, 0x1f, 0x68, 0xed, 0x2e, 0x19, 0x4c, 0x79, 0xfb, 0xc6, 0xae, 0xbe, 0xe1, 0xa6, 0x67,
        0x97, 0x5d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0xda, 0x31, 0xd5, 0xcf,
        0x68, 0x82, 0x21,
    ]);
    let block_1 = next_block(&mut chacha);
    let block_2 = next_block(&mut chacha);
    assert_eq!(
        block_1,
        [
            0xf6, 0x3a, 0x89, 0xb7, 0x5c, 0x22, 0x71, 0xf9, 0x36, 0x88, 0x16, 0x54, 0x2b, 0xa5,
            0x2f, 0x06, 0xed, 0x49, 0x24, 0x17, 0x92, 0x30, 0x2b, 0x00, 0xb5, 0xe8, 0xf8, 0x0a,
            0xe9, 0xa4, 0x73, 0xaf, 0xc2, 0x5b, 0x21, 0x8f, 0x51, 0x9a, 0xf0, 0xfd, 0xd4, 0x06,
            0x36, 0x2e, 0x8d, 0x69, 0xde, 0x7f, 0x54, 0xc6, 0x04, 0xa6, 0xe0, 0x0f, 0x35, 0x3f,
            0x11, 0x0f, 0x77, 0x1b, 0xdc, 0xa8, 0xab, 0x92,
        ],
    );
    assert_eq!(
        block_2,
        [
            0xe5, 0xfb, 0xc3, 0x4e, 0x60, 0xa1, 0xd9, 0xa9, 0xdb, 0x17, 0x34, 0x5b, 0x0a, 0x40,
            0x27, 0x36, 0x85, 0x3b, 0xf9, 0x10, 0xb0, 0x60, 0xbd, 0xf1, 0xf8, 0x97, 0xb6, 0x29,
            0x0f, 0x01, 0xd1, 0x38, 0xae, 0x2c, 0x4c, 0x90, 0x22, 0x5b, 0xa9, 0xea, 0x14, 0xd5,
            0x18, 0xf5, 0x59, 0x29, 0xde, 0xa0, 0x98, 0xca, 0x7a, 0x6c, 0xcf, 0xe6, 0x12, 0x27,
            0x05, 0x3c, 0x84, 0xe4, 0x9a, 0x4a, 0x33, 0x32,
        ],
    );
}
