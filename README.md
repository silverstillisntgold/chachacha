# ChaChaCha: ChaCha with a little extra Cha

Extremely fast chacha implementation. Primarily made for use in the [`ya-rand`] crate,
but just as usable anywhere else you might want to use Chacha. Generally speaking you don't want
to use Chacha directly, as it's normally paired with Poly1305 for authentication.

Documentation is minimal at the moment (working on it), and the public API needs significant
improvement (it's kind of dogshit right now).

[`ya-rand`]: https://crates.io/crates/ya-rand
