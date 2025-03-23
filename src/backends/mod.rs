pub mod soft;

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        #[cfg(feature = "nightly")]
        pub mod avx512;
        pub mod avx2;
        pub mod sse2;

        cfg_if! {
            if #[cfg(all(feature = "nightly", target_feature = "avx512f"))] {
                pub use avx512::Matrix;
            } else if #[cfg(target_feature = "avx2")] {
                pub use avx2::Matrix;
            } else if #[cfg(target_feature = "sse2")] {
                pub use sse2::Matrix;
            } else {
                compile_error!(
                    "building programs on x86 without support for sse2 may introduce undefined behavior"
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
