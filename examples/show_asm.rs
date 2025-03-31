//! Used for viewing assembly of all the implementations via `cargo-show-asm`.

use chachacha::*;
use core::hint::black_box;

fn main() {
    let empty_state = [0; 48];
    let mut chacha = ChaCha8Ietf::new(empty_state);
    black_box(chacha.get_block());
    let mut chacha = ChaCha12Ietf::new(empty_state);
    black_box(chacha.get_block());
    let mut chacha = ChaCha20Ietf::new(empty_state);
    black_box(chacha.get_block());
    let mut chacha = ChaCha8Djb::new(empty_state);
    black_box(chacha.get_block());
    let mut chacha = ChaCha12Djb::new(empty_state);
    black_box(chacha.get_block());
    let mut chacha = ChaCha20Djb::new(empty_state);
    black_box(chacha.get_block());
}
