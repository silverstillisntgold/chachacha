/*!
Module containing all the non-portable shit. Only explicitly re-exports the widest
implementation available as the definitive `Matrix` for the entire submodule, but still enables
whatever other modules are available on the target system. This is done for testing purposes,
and none of it is accessible by the end-user of this crate.

A ChaCha instance holds 16 32-bit integers (their signedness is irrelevant),
in the form of a 4-by-4 matrix. The first 4 integers are constant values from the string "`expand 32-byte k`",
and exist to ensure a base amount of entropy for instances with shitty key values. The next 8 integers are
the key/seed values. Of the last 4 integers, the first 2 together represent a 64-bit integer that functions
as the counter for the instance. **This counter is the only thing that changes between invocations of a
given ChaCha instance.** Say you run a ChaCha round with a given state, where the 64-bit counter happens to 69.
After it has returned the result, the counter of that instance will then be 70, which will impact the next execution
of a ChaCha round. The last 2 integers are used as a way of differentiating between instances that might
have the same key/seed values, and are called the "nonce".

This is the layout of the original variant proposed by the author of ChaCha, Daniel J. Bernstein.
Below is a visual representation. This layout enables 2<sup>320</sup> unique key/nonce combinations,
each capable of generating 1 ZiB of output before repeating.

```text
"expa"   "nd 3"  "2-by"  "te k"
Key      Key      Key    Key
Key      Key      Key    Key
Counter  Counter  Nonce  Nonce
```

An alternative layout, suggested by the IETF, uses only a single 32-bit integer for the counter
and three of them for nonces. Both implementations are provided by this crate.

The soft implementation is the [reference implementation], but batched to (in theory) increase performance and maintain
API compatability with the other impls. The result isn't as fast as the manually vectorized variants, but is
better than running a pure reference implementation four times sequentially.

The vectorized variants all use [this paper] as a general guide, with lots of experimentation/testing to fill
in the gaps. [This commit] is used as a reference for ordering in the diagonalization methods. Ordering doesn't
seem to make any difference on modern machines, but this should hopefully prevent issues with older CPUs.

TLDR is we process four ChaCha instances at once, working on them in terms of their rows instead of individual elements.
SSE2/Neon are only wide enough for individual instances to be processed, but AVX2 allows for processesing two instances at once
and AVX512 allows processesing all four at once.

[reference implementation]: https://en.wikipedia.org/wiki/Salsa20#ChaCha_variant
[this paper]: https://eprint.iacr.org/2013/759
[this commit]: https://github.com/cryptocorrosion/cryptocorrosion/commit/8608f02b1fd8847cdaeb09c965f7ea26faa2039c
*/

pub mod soft;

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        #[cfg(target_feature = "avx512f")]
        pub mod avx512;
        #[cfg(target_feature = "avx2")]
        pub mod avx2;
        #[cfg(target_feature = "sse2")]
        pub mod sse2;

        cfg_if::cfg_if! {
            if #[cfg(target_feature = "avx512f")] {
                pub use avx512::Matrix;
            } else if #[cfg(target_feature = "avx2")] {
                pub use avx2::Matrix;
            } else if #[cfg(target_feature = "sse2")] {
                pub use sse2::Matrix;
            } else {
                compile_error!("targeting x86 without sse2 is unsupported");
            }
        }
    } else if #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))] {
        cfg_if::cfg_if! {
            if #[cfg(target_feature = "neon")] {
                pub mod neon;
                pub use neon::Matrix;
            } else {
                compile_error!("neon is a default feature of arm64");
            }
        }
    } else {
        pub use soft::Matrix;
    }
}
