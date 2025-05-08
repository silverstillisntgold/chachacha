/*!
Module containing the variants of ChaCha (awfully descriptive, I know).
*/

pub enum Variants {
    /// Original variant proposed by the author of the salsa
    /// and chacha algorithms: Daniel J. Bernstein.
    Djb,
    /// Alternative variation specified by the IETF, most often
    /// used in conjunction with Poly1305.
    Ietf,
}

pub trait Variant {
    const VAR: Variants;
}

pub struct Djb;
impl Variant for Djb {
    const VAR: Variants = Variants::Djb;
}

pub struct Ietf;
impl Variant for Ietf {
    const VAR: Variants = Variants::Ietf;
}
