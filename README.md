# ChaChaCha: ChaCha with a little extra Cha

Extremely fast ChaCha implementation. Primarily made for use as a CRNG in the [`ya-rand`] crate,
but should be just as usable anywhere else you might want to use ChaCha.

## Examples

```rust
use chachacha::{BUF_LEN_U64, BUF_LEN_U8, ChaCha12Djb};

// Create a new `ChaCha12Djb` instance with a key that is all ones,
// a counter starting at 69, and a nonce of 0 and 1 (the last nonce
// value is discarded in the `Djb` variants).
let mut chacha = ChaCha12Djb::new([u32::MAX; 8],
                                   69,
                                  [0, 1, 2]);
// 256 bytes of output
let block_output: [u8; BUF_LEN_U8] = chacha.get_block();
let all_zeros = block_output.into_iter().all(|v| v == 0);
assert!(!all_zeros);
```

[`ya-rand`]: https://crates.io/crates/ya-rand
