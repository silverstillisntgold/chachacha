use crate::{
    chacha::ChaChaCore,
    util::{BATCH_BYTES, BLOCKS, Backend, COLUMNS, MATRIX_SIZE, ROW_A, Variant, Variants},
};

pub struct Soft;

impl Backend for Soft {
    #[inline]
    fn process_internal<const ROUNDS: usize, V: Variant, const XOR: bool>(
        core: &mut ChaChaCore<Self, ROUNDS, V>,
        buffer: &mut [u8],
    ) {
        todo!()
    }
}
