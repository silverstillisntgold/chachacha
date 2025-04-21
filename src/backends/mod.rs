/*!
Submodule containing all the non-portable shit. Only explicitly re-exports the widest
implementation available as the definitive `Matrix` for the entire submodule, but still enables
whatever other modules are available on the target system. This is done for testing purposes,
and none of it is accessible by the end-user of this crate.
*/

pub mod soft;

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        pub mod avx512;
        #[cfg(target_feature = "avx2")]
        pub mod avx2;
        #[cfg(target_feature = "sse2")]
        pub mod sse2;

        cfg_if::cfg_if! {
            if #[cfg(all(feature = "nightly", target_feature = "avx512f"))] {
                pub use avx512::Matrix;
            } else if #[cfg(target_feature = "avx2")] {
                pub use avx2::Matrix;
            } else if #[cfg(target_feature = "sse2")] {
                pub use sse2::Matrix;
            } else {
                compile_error!(
                    "building x86 programs without support for sse2 may introduce undefined behavior"
                );
            }
        }
    // NEON on ARM32 is both unsound and gated behind nightly.
    } else if #[cfg(all(
        target_feature = "neon",
        any(target_arch = "aarch64", target_arch = "arm64ec")
    ))] {
        pub mod neon;
        pub use neon::Matrix;
    } else {
        pub use soft::Matrix;
    }
}
