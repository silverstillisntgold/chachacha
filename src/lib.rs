/*!
# ChaChaCha: ChaCha with a little extra Cha

Extremely fast ChaCha implementation. Primarily made for use as a CRNG in the [`ya-rand`] crate,
but should be just as usable anywhere else you might want to use ChaCha.

## Examples

```ignore
use chachacha::{BUF_LEN_U64, BUF_LEN_U8, ChaCha12Djb};

// Create a new `ChaCha12Djb` instance with a key that is all ones,
// a counter starting at 69, and a nonce of 0 and 1 (the last nonce
// value is discarded in the `Djb` variants).
let mut chacha = ChaCha12Djb::new([u32::MAX; 8],
                                   69,
                                  [0, 1, 2]);
// 256 bytes of output
let block_output: [u8; BUF_LEN_U8] = chacha.get_block();
let all_zeros = block_output.into_iter().all(|v| v == 0);
assert!(!all_zeros);
```

[`ya-rand`]: https://crates.io/crates/ya-rand
*/

//#![deny(missing_docs)]
#![no_std]

// The reference implementation is only used for testing the vectorized implementations
// to ensure they're correct; don't bother compiling it when not testing.
#[cfg(test)]
mod chacha_reference;

mod backends;
mod chacha;
mod util;

mod internal {
    //use crate::backends::soft::Soft as TargetMachine;
    use crate::backends::TargetMachine;
    use crate::chacha::ChaChaCore;
    use crate::util::{Djb, Ietf};

    /// ChaCha with a custom amount of rounds, a 64-bit counter, and a 64-bit nonce.
    ///
    /// This is non-standard, and should generally not be used.
    pub type ChaChaDjb<const ROUNDS: usize> = ChaChaCore<TargetMachine, ROUNDS, Djb>;

    /// ChaCha with a custom amount of rounds, a 32-bit counter, and a 96-bit nonce.
    ///
    /// This is non-standard, and should generally not be used.
    pub type ChaChaIetf<const ROUNDS: usize> = ChaChaCore<TargetMachine, ROUNDS, Ietf>;
}

#[cfg(feature = "_internal")]
pub use internal::{ChaChaDjb, ChaChaIetf};
#[cfg(not(feature = "_internal"))]
use internal::{ChaChaDjb, ChaChaIetf};

/// ChaCha with 8 rounds, a 64-bit counter, and a 64-bit nonce.
pub type ChaCha8Djb = ChaChaDjb<8>;
/// ChaCha with 12 rounds, a 64-bit counter, and a 64-bit nonce.
pub type ChaCha12Djb = ChaChaDjb<12>;
/// ChaCha with 20 rounds, a 64-bit counter, and a 64-bit nonce.
pub type ChaCha20Djb = ChaChaDjb<20>;

/// ChaCha with 8 rounds, a 32-bit counter, and a 96-bit nonce.
pub type ChaCha8Ietf = ChaChaIetf<8>;
/// ChaCha with 12 rounds, a 32-bit counter, and a 96-bit nonce.
pub type ChaCha12Ietf = ChaChaIetf<12>;
/// ChaCha with 20 rounds, a 32-bit counter, and a 96-bit nonce.
pub type ChaCha20Ietf = ChaChaIetf<20>;

#[cfg(test)]
mod tests {
    use crate::{backends::*, chacha::ChaChaCore, chacha_reference::ChaCha as ChaChaRef, util::*};
    use core::mem::transmute;

    const TEST_ITERS: usize = 1 << 4;
    const TEST_LENGTHS: &[usize] = &[
        0, 1, 2, 3, 4, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383,
        384, 385, 447, 448, 449, 511, 512, 513, 575, 576, 577, 639, 640, 641, 703, 704, 705, 767,
        768, 769, 831, 832, 833, 895, 896, 897, 959, 960, 961, 1023, 1024, 1025, 1087, 1088, 1089,
        1151, 1152, 1153, 1215, 1216, 1217, 1279, 1280, 1281, 1343, 1344, 1345, 1407, 1408, 1409,
        1471, 1472, 1473, 1535, 1536, 1537, 1599, 1600, 1601, 1663, 1664, 1665, 1727, 1728, 1729,
        1791, 1792, 1793, 1855, 1856, 1857, 1919, 1920, 1921, 1983, 1984, 1985, 2047, 2048, 2049,
    ];

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_8_djb_neon() {
        test_chacha::<neon::Neon, 8, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_8_ietf_neon() {
        test_chacha::<neon::Neon, 8, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_djb_neon() {
        test_chacha::<neon::Neon, 12, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_ietf_neon() {
        test_chacha::<neon::Neon, 12, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_djb_neon() {
        test_chacha::<neon::Neon, 20, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_ietf_neon() {
        test_chacha::<neon::Neon, 20, Ietf>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_8_djb_avx512() {
        test_chacha::<avx512::Avx512, 8, Djb>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_8_ietf_avx512() {
        test_chacha::<avx512::Avx512, 8, Ietf>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_12_djb_avx512() {
        test_chacha::<avx512::Avx512, 12, Djb>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_12_ietf_avx512() {
        test_chacha::<avx512::Avx512, 12, Ietf>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_20_djb_avx512() {
        test_chacha::<avx512::Avx512, 20, Djb>();
    }

    #[cfg(target_feature = "avx512f")]
    #[test]
    fn chacha_20_ietf_avx512() {
        test_chacha::<avx512::Avx512, 20, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_djb_avx2() {
        test_chacha::<avx2::Avx2, 8, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_ietf_avx2() {
        test_chacha::<avx2::Avx2, 8, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_djb_avx2() {
        test_chacha::<avx2::Avx2, 12, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_ietf_avx2() {
        test_chacha::<avx2::Avx2, 12, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_djb_avx2() {
        test_chacha::<avx2::Avx2, 20, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_ietf_avx2() {
        test_chacha::<avx2::Avx2, 20, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_djb_sse2() {
        test_chacha::<sse2::Sse2, 8, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_ietf_sse2() {
        test_chacha::<sse2::Sse2, 8, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_djb_sse2() {
        test_chacha::<sse2::Sse2, 12, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_ietf_sse2() {
        test_chacha::<sse2::Sse2, 12, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_djb_sse2() {
        test_chacha::<sse2::Sse2, 20, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_ietf_sse2() {
        test_chacha::<sse2::Sse2, 20, Ietf>();
    }

    #[test]
    fn chacha_8_djb_soft() {
        test_chacha::<soft::Soft, 8, Djb>();
    }

    #[test]
    fn chacha_8_ietf_soft() {
        test_chacha::<soft::Soft, 8, Ietf>();
    }

    #[test]
    fn chacha_12_djb_soft() {
        test_chacha::<soft::Soft, 12, Djb>();
    }

    #[test]
    fn chacha_12_ietf_soft() {
        test_chacha::<soft::Soft, 12, Ietf>();
    }

    #[test]
    fn chacha_20_djb_soft() {
        test_chacha::<soft::Soft, 20, Djb>();
    }

    #[test]
    fn chacha_20_ietf_soft() {
        test_chacha::<soft::Soft, 20, Ietf>();
    }

    fn test_chacha<B: Backend, const ROUNDS: usize, V: Variant>() {
        let mut seed = [0; 48];
        let mut buf = [0; 4096];
        let mut buf_ref = [0; 4096];

        for i in 0..TEST_ITERS {
            // We iterate in reverse to fill the upper bits of `buf` and `buf_ref`
            // with garbage. We expect the garbage in both of them to be the same.
            for &length in TEST_LENGTHS.iter().rev() {
                getrandom::fill(&mut seed).unwrap();
                // The difference between the djb/ietf variants is only apparent
                // when index 12 crosses the `u32::MAX` threshold, since that's the
                // point where ietf would only wrap index 12 around to 0, but the
                // djb variant would also increment index 13.
                if i.is_multiple_of(2) {
                    let seed_ref: &mut [u32; 12] = unsafe { transmute(&mut seed) };
                    seed_ref[8] = match i % 8 {
                        0 => u32::MAX,
                        2 => u32::MAX - 1,
                        4 => u32::MAX - 2,
                        6 => u32::MAX - 3,
                        _ => unreachable!(),
                    };
                }

                // We want to test filling a buffer from the output stream of ChaCha
                // and we want to test XORing a buffer's contents with the output stream.
                //
                // So we fill each buffer, then XOR its contents with the output stream.

                let buf = &mut buf[..length];
                let buf_ref = &mut buf_ref[..length];
                let mut chacha = ChaChaCore::<B, ROUNDS, V>::from_bytes(seed);
                let mut chacha_ref = ChaChaRef::<ROUNDS, V>::from(seed);

                chacha.fill(buf);
                chacha_ref.fill(buf_ref);
                assert_eq!(buf, buf_ref);

                chacha.apply_keystream(buf);
                chacha_ref.apply_keystream(buf_ref);
                assert_eq!(buf, buf_ref);
            }
        }
    }
}
