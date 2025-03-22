pub enum Variants {
    Djb,
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
