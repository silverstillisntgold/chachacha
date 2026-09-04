# ChaChaCha: ChaCha with a little extra Cha

Extremely fast (the fastest?) ChaCha implementation. Primarily made for use as a CRNG in the
[`ya-rand`] crate, but should be just as usable anywhere else you might want to use ChaCha.

This is as a low-level primitive, and should generally **not** be used directly.
When used as a cipher it should be paired with something like Poly1305, and when used as an
RNG it's output should be batched to amortize the cost of generating new entropy.

[`ya-rand`]: https://crates.io/crates/ya-rand
