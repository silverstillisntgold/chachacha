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
