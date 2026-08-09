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

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
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
    use crate::backends::*;
    use crate::chacha::ChaChaCore;
    use crate::chacha_reference::ChaCha as ChaChaRef;
    use crate::util::*;
    use core::iter::repeat_with;
    use core::mem::transmute;

    const TEST_COUNT: usize = 1 << 6;
    const TEST_LEN: usize = 1 << 4;
    /// Reference implementation needs 4 times the runs since it
    /// produces a quarter of the output per block operation.
    const TEST_LEN_REF: usize = TEST_LEN * 4;

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

    fn test_chacha<B, const ROUNDS: usize, V>()
    where
        B: Backend,
        V: Variant,
    {
        for i in 0..TEST_COUNT {
            let mut seed = [0; 48];
            seed.fill_with(|| getrandom::u64().unwrap() as u8);
            // The difference between the djb/ietf variants is only apparent
            // when index 12 crosses the `u32::MAX` threshold, since that's the
            // point where ietf would only wrap index 12 around to 0, but the
            // djb variant would also increment index 13.
            if i >= (TEST_COUNT / 2) {
                let seed_ref: &mut [u32; 12] = unsafe { transmute(&mut seed) };
                seed_ref[8] = u32::MAX - 7;
            }
            let mut chacha = ChaChaCore::<B, ROUNDS, V>::from_bytes(seed);
            let mut chacha_ref = ChaChaRef::<ROUNDS, V>::from(seed);

            let chacha_iter = repeat_with(|| {
                let mut buffer = [0; 256];
                chacha.fill(&mut buffer);
                buffer
            })
            .take(TEST_LEN)
            .flatten();
            let chacha_ref_iter = repeat_with(|| chacha_ref.get_block())
                .take(TEST_LEN_REF)
                .flatten();
            chacha_iter
                .zip(chacha_ref_iter)
                .for_each(|(a, b)| assert_eq!(a, b));

            const BIG_IF_TRU: usize = 512 * 2;
            for _ in 0..TEST_COUNT {
                let mut buf = [0; BIG_IF_TRU];
                let mut buf_ref = [0; BIG_IF_TRU];
                let size = getrandom::u64().unwrap() as usize % BIG_IF_TRU;
                chacha.fill(&mut buf[..size]);
                chacha_ref.fill(&mut buf_ref[..size]);
                assert_eq!(buf, buf_ref);
            }
        }
    }
}
