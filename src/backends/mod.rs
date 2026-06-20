/*!
TODO: Module docs.
*/

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
        pub use avx512::Machine as TargetMachine;
    }
    target_feature = "avx2" => {
        pub use avx2::Machine as TargetMachine;
    }
    target_feature = "sse2" => {
        pub use sse2::Machine as TargetMachine;
    }
    target_feature = "neon" => {
        pub use neon::Machine as TargetMachine;
    }
    _ => {
        pub use soft::Machine as TargetMachine;
    }
}
