# ChaChaCha: ChaCha with a little extra Cha

Extremely fast ChaCha implementation. Primarily made for use as a CRNG in the [`ya-rand`] crate,
but should be just as usable anywhere else you might want to use ChaCha. This is intended to be
a low-level primitive, and should generally not be used directly. When used as a cipher it should
be paired with something like Poly1305, and when used as an RNG it's output should be consumed
and fed to the user batches.

[`ya-rand`]: https://crates.io/crates/ya-rand
