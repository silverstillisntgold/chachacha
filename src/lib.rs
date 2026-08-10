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

    const BUFFER_LEN: usize = (1 << 12) + 7;
    const TEST_COUNT: usize = 1 << 6;

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
        let mut buf = [0; BUFFER_LEN];
        let mut buf_ref = [0; BUFFER_LEN];

        for i in 0..TEST_COUNT {
            getrandom::fill(&mut seed).unwrap();
            // The difference between the djb/ietf variants is only apparent
            // when index 12 crosses the `u32::MAX` threshold, since that's the
            // point where ietf would only wrap index 12 around to 0, but the
            // djb variant would also increment index 13.
            if i >= (TEST_COUNT / 2) {
                let seed_ref: &mut [u32; 12] = unsafe { transmute(&mut seed) };
                seed_ref[8] = u32::MAX - 7;
            }

            let mut chacha = ChaChaCore::<B, ROUNDS, V>::from_bytes(seed);
            chacha.fill(&mut buf);
            chacha.apply_keystream(&mut buf);

            let mut chacha_ref = ChaChaRef::<ROUNDS, V>::from(seed);
            chacha_ref.fill(&mut buf_ref);
            chacha_ref.apply_keystream(&mut buf_ref);

            assert_eq!(buf, buf_ref);
        }
    }
}
