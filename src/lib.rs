/*!
TODO
*/

#![no_std]
#![cfg_attr(
    all(
        feature = "nightly",
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]
#![warn(missing_docs)]

#[cfg(test)]
mod chacha_reference;

mod backends;
mod chacha;
mod rounds;
mod util;
mod variations;

use self::chacha::ChaChaCore;
use backends::Matrix;
use rounds::*;
use variations::*;

type ChaCha<R, V> = ChaChaCore<Matrix, R, V>;

pub type ChaCha8Djb = ChaCha<R8, Djb>;
pub type ChaCha12Djb = ChaCha<R12, Djb>;
pub type ChaCha20Djb = ChaCha<R20, Djb>;

pub type ChaCha8Ietf = ChaCha<R8, Ietf>;
pub type ChaCha12Ietf = ChaCha<R12, Ietf>;
pub type ChaCha20Ietf = ChaCha<R20, Ietf>;

pub use util::{BUF_LEN_U8, BUF_LEN_U64, SEED_LEN_U8, SEED_LEN_U32, SEED_LEN_U64};

#[cfg(test)]
mod tests {
    use super::backends::*;
    use super::chacha::ChaChaCore;
    use super::chacha_reference::ChaCha as ChaChaRef;
    use super::rounds::*;
    use super::util::*;
    use super::variations::*;
    use core::iter::repeat_with;
    use core::mem::transmute;

    const TEST_COUNT: usize = 50;
    const TEST_LEN: usize = 16;
    /// Reference implementation needs 4 times the runs since it
    /// produces a quarter of the output per block operation.
    const TEST_LEN_REF: usize = TEST_LEN * 4;

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_8_djb_neon() {
        test_chacha::<neon::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_8_ietf_neon() {
        test_chacha::<neon::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_djb_neon() {
        test_chacha::<neon::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_ietf_neon() {
        test_chacha::<neon::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_djb_neon() {
        test_chacha::<neon::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_ietf_neon() {
        test_chacha::<neon::Matrix, R20, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_8_djb_avx512() {
        test_chacha::<avx512::Matrix, R8, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_8_ietf_avx512() {
        test_chacha::<avx512::Matrix, R8, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_12_djb_avx512() {
        test_chacha::<avx512::Matrix, R12, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_12_ietf_avx512() {
        test_chacha::<avx512::Matrix, R12, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_20_djb_avx512() {
        test_chacha::<avx512::Matrix, R20, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_20_ietf_avx512() {
        test_chacha::<avx512::Matrix, R20, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_djb_avx2() {
        test_chacha::<avx2::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_ietf_avx2() {
        test_chacha::<avx2::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_djb_avx2() {
        test_chacha::<avx2::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_ietf_avx2() {
        test_chacha::<avx2::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_djb_avx2() {
        test_chacha::<avx2::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_ietf_avx2() {
        test_chacha::<avx2::Matrix, R20, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_djb_sse2() {
        test_chacha::<sse2::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_ietf_sse2() {
        test_chacha::<sse2::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_djb_sse2() {
        test_chacha::<sse2::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_ietf_sse2() {
        test_chacha::<sse2::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_djb_sse2() {
        test_chacha::<sse2::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_ietf_sse2() {
        test_chacha::<sse2::Matrix, R20, Ietf>();
    }

    #[test]
    fn chacha_8_djb_soft() {
        test_chacha::<soft::Matrix, R8, Djb>();
    }

    #[test]
    fn chacha_8_ietf_soft() {
        test_chacha::<soft::Matrix, R8, Ietf>();
    }

    #[test]
    fn chacha_12_djb_soft() {
        test_chacha::<soft::Matrix, R12, Djb>();
    }

    #[test]
    fn chacha_12_ietf_soft() {
        test_chacha::<soft::Matrix, R12, Ietf>();
    }

    #[test]
    fn chacha_20_djb_soft() {
        test_chacha::<soft::Matrix, R20, Djb>();
    }

    #[test]
    fn chacha_20_ietf_soft() {
        test_chacha::<soft::Matrix, R20, Ietf>();
    }

    fn test_chacha<M: Machine, R: DoubleRounds, V: Variant>() {
        for i in 0..TEST_COUNT {
            let mut seed = [0; SEED_LEN_U8];
            getrandom::fill(&mut seed).unwrap();
            // The difference between the djb/ietf variants is only apparent
            // when index 12 crosses the `u32::MAX` threshold, since that's the
            // point where ietf would only wrap index 12 around to 0, but the
            // djb variant would also increment index 13.
            if i >= (TEST_COUNT / 2) {
                let seed_ref: &mut [u32; SEED_LEN_U32] = unsafe { transmute(&mut seed) };
                seed_ref[8] = u32::MAX - 7;
            }
            let mut chacha = ChaChaCore::<M, R, V>::from(seed);
            let mut chacha_ref = ChaChaRef::from(seed);

            let chacha_iter = repeat_with(|| chacha.get_block()).take(TEST_LEN).flatten();
            let chacha_ref_iter = repeat_with(|| chacha_ref.get_block::<R, V>())
                .take(TEST_LEN_REF)
                .flatten();
            chacha_iter
                .zip(chacha_ref_iter)
                .for_each(|(a, b)| assert_eq!(a, b));

            const BIG_IF_TRU: usize = BUF_LEN_U8 * 2;
            let size = getrandom::u32().unwrap() as usize % BIG_IF_TRU;
            for _ in 0..TEST_COUNT {
                let mut buf = [0; BIG_IF_TRU];
                let mut buf_ref = [0; BIG_IF_TRU];
                chacha.fill(&mut buf[..size]);
                chacha_ref.fill::<R, V>(&mut buf_ref[..size]);
                assert_eq!(buf, buf_ref);
            }
        }
    }
}
