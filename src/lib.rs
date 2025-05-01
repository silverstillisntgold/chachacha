/*!
"Alright you disgusting rat bastard, so how does all this garbage generate cryptographic random values?"
Take a seat little Timmy, and allow me to blow your mind.

Before anything else, it's important to have a general understanding of the structure of the
reference ChaCha algorithm. A ChaCha instance typically holds 16 32-bit integers (their signedness
is irrelevant), in the form of a 4-by-4 matrix. This being a flat or 2d array is an implementation
detail that shouldn't impact performance or output at all, and is something you don't need to
worry about.

The first 4 integers are constant values from the string "`expand 32-byte k`", and exist to ensure a
base amount of entropy for instances with shitty key values. The next 8 integers are the key/seed values.
Of the last 4 integers, the first 2 together represent a 64-bit integer that functions as the counter
for the instance. **This counter is the only thing that changes between invocations of a given ChaCha
instance.** Say you run a ChaCha round with a given state, where the 64-bit counter happens to 69. After
it has returned the result, the counter of that instance will then be 70, which will impact the next execution
of a ChaCha round. The last 2 integers (nonce values) are used as a way of differentiating between instances
that might have the same key/seed values.

```text
"expa"   "nd 3"  "2-by"  "te k"
Key      Key      Key    Key
Key      Key      Key    Key
Counter  Counter  Nonce  Nonce
```

Since we are only using ChaCha as an RNG, we randomize everything when creating instances. Meaning that we
treat the nonce values as extra key/seed values, and the counter can start anywhere in it's cycle.
This is fast, simple, and means we effectively have 2<sup>320</sup> unique data streams that can be generated.
Each of these streams can provide 1 ZiB before repeating (when the counter is incremented 2<sup>64</sup>
times back to where it started on initialization).

All implementations process four instances of chacha per invocation.

The soft implementation is the [reference implementation], but structured in a way that allows the compiler
to easily auto-vectorize the rounds. The result isn't as fast as the manually vectorized variants, but seems
to be about twice as fast the equivalent non-vectorized code. The vectorized variants were developed using
[this paper].

The process of generating data using `[SecureRng]` is as follows:

1. Take the internal `ChaCha` instance and turn it into a `Machine`. A `Machine` serves as the abstraction
layer for different architectures, and it's contents will vary depending on the flags used to compile the
final binary (this crate **does not** use runtime dispatch). But it's size will always be 256 bytes,
since it will always contain 4 distinct chacha matrixs, despite their representations being different.
This `Machine` handles incrementing the counter values of it's internal chacha blocks by 0, 1, 2, and 3.
The underlying `ChaCha` struct doesn't bother storing the constants directly, they are instead directly
loaded from static memory when creating `Machine` instances.

2. The newly created `Machine` is cloned, and the original `ChaCha` instance has it's counter incremented by
4, so next time it's called we don't get overlap in any of the internal chacha instances.

3. 4 double rounds are performed (making this a ChaCha8 implemetation). In the soft implementation this is
straightforward, but the vectorized variants take a different approach. A double round performs two rounds,
the first operates on one of the four columns, and the second operates on one of the four diagonals. To make
the vectorized approaches faster, we tranform the `Machine` state so we only ever need to perform column
rounds. Column rounds don't change much, but before each "diagonal" round, we shuffle the contents of the vectors.
This is done so we can again perform a column round, but now the column we are operating on contains the data
that just a moment ago was layed out in a diagonal. After the round is completed, this transformation is
reverted.

4. The `Machine` which has just had chacha rounds performed on it is then added to the cloned `Machine` from
step 2. The resulting `Machine` then contains the output of four independent chacha matrix computed in parallel.

5. For the soft, sse2, and neon implementations the `Machine` state is already in the layout we need it and can be
transmuted (bit-cast) directly into an array for end-user consumption. But due to how vector register indexing works,
we have to do additional work for avx2 and avx512 to make sure the layout of the results are correct. It looks a
bit convoluted but all we're doing is moving the internal 128-bit components of the 256/512 bit registers around
to make their ordering match that of the sse2 variant (which directly uses 128-bit vectors).

[reference implementation]: https://en.wikipedia.org/wiki/Salsa20#ChaCha_variant
[this paper]: https://eprint.iacr.org/2013/759

## Security

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
    fn chacha_8_neon_djb() {
        test_chacha::<neon::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_8_neon_ietf() {
        test_chacha::<neon::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_neon_djb() {
        test_chacha::<neon::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_12_neon_ietf() {
        test_chacha::<neon::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_neon_djb() {
        test_chacha::<neon::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "neon")]
    #[test]
    fn chacha_20_neon_ietf() {
        test_chacha::<neon::Matrix, R20, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_8_avx512_djb() {
        test_chacha::<avx512::Matrix, R8, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_8_avx512_ietf() {
        test_chacha::<avx512::Matrix, R8, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_12_avx512_djb() {
        test_chacha::<avx512::Matrix, R12, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_12_avx512_ietf() {
        test_chacha::<avx512::Matrix, R12, Ietf>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_20_avx512_djb() {
        test_chacha::<avx512::Matrix, R20, Djb>();
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    #[test]
    fn chacha_20_avx512_ietf() {
        test_chacha::<avx512::Matrix, R20, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_avx2_djb() {
        test_chacha::<avx2::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_8_avx2_ietf() {
        test_chacha::<avx2::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_avx2_djb() {
        test_chacha::<avx2::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_12_avx2_ietf() {
        test_chacha::<avx2::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_avx2_djb() {
        test_chacha::<avx2::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn chacha_20_avx2_ietf() {
        test_chacha::<avx2::Matrix, R20, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_sse2_djb() {
        test_chacha::<sse2::Matrix, R8, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_8_sse2_ietf() {
        test_chacha::<sse2::Matrix, R8, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_sse2_djb() {
        test_chacha::<sse2::Matrix, R12, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_12_sse2_ietf() {
        test_chacha::<sse2::Matrix, R12, Ietf>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_sse2_djb() {
        test_chacha::<sse2::Matrix, R20, Djb>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn chacha_20_sse2_ietf() {
        test_chacha::<sse2::Matrix, R20, Ietf>();
    }

    #[test]
    fn chacha_8_soft_djb() {
        test_chacha::<soft::Matrix, R8, Djb>();
    }

    #[test]
    fn chacha_8_soft_ietf() {
        test_chacha::<soft::Matrix, R8, Ietf>();
    }

    #[test]
    fn chacha_12_soft_djb() {
        test_chacha::<soft::Matrix, R12, Djb>();
    }

    #[test]
    fn chacha_12_soft_ietf() {
        test_chacha::<soft::Matrix, R12, Ietf>();
    }

    #[test]
    fn chacha_20_soft_djb() {
        test_chacha::<soft::Matrix, R20, Djb>();
    }

    #[test]
    fn chacha_20_soft_ietf() {
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
