/*!
A single chacha instance looks like the following, where each `*` represents a 32-bit integer.

```text
* * * *
* * * *
* * * *
* * * *
```

So when we process four "blocks" of ChaCha, one can think of it as looking like this:

```text
* * * * | * * * * | * * * * | * * * *
* * * * | * * * * | * * * * | * * * *
* * * * | * * * * | * * * * | * * * *
* * * * | * * * * | * * * * | * * * *
```

The key distinction between the backends is how exactly these four "blocks" are partitioned
and how their processing is parallelized.

TODO: Soft explanation after it gets another optimization pass.

The sse2 backend is parallelized per block, so we process a block at a time 4 times in a row.

The avx2 backend is parallelized per block pair, so we process 2 blocks twice in a row.

And the avx512 backend is processed all at once.

The exact pipeline for each backend is dependent in part on how many registers are expected to be available.
Normally sse2 and avx2 expose 16 registers. Avx2 makes use of all of them and avoids any spilling during the core
ChaCha double rounds. Sse2 over-uses registers and causes quite a bit of spilling, but because of how tight the loop
is and thanks to the nature of modern processes, the spilled data just lives in L1 and incurs less of a performance
penalty than different non-spilling approaches. Avx512 not only exposes 32 registers, but also is written such that
there is zero spilling throughout the entire fill/xor procedure. I don't have a computer to test the avx512 speedup
but I imagine it's immense.

Neon is a bit weird. It provides 128-bit wide registers, just like sse2, but provides 32 of them, just like avx512.
So we actually structure the neon backend to resemble the pipeline of the avx512 backend, with the expectation that newer
ARM64 cores which support fancier packed instruction sets (SVE), may be able to spot what we're doing and excute in a
manner similar to avx512. Even with regular neon instructions the current design already seems to trounce previous
neon ChaCha implementations.
*/

#![allow(clippy::too_many_arguments, unused)]

#[cfg(target_feature = "avx2")]
pub mod avx2;
#[cfg(target_feature = "avx512f")]
pub mod avx512;
#[cfg(target_feature = "neon")]
pub mod neon;
pub mod soft;
#[cfg(target_feature = "sse2")]
pub mod sse2;

// Choose the widest available machine as the default.
cfg_select! {
    target_feature = "avx512f" => {
        pub use avx512::Avx512 as TargetMachine;
    }
    target_feature = "avx2" => {
        pub use avx2::Avx2 as TargetMachine;
    }
    target_feature = "sse2" => {
        pub use sse2::Sse2 as TargetMachine;
    }
    target_feature = "neon" => {
        pub use neon::Neon as TargetMachine;
    }
    _ => {
        pub use soft::Soft as TargetMachine;
    }
}
