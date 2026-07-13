use crate::util::Backend;

pub struct Sse2;

impl Backend for Sse2 {
    const BLOCKS: usize = 4;

    fn process<const ROUNDS: usize, V: crate::util::Variant, const XOR: bool>(
        core: &mut crate::chacha::ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        todo!()
    }
}
