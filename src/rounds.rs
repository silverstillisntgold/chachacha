/*!
Module containing the standard ChaCha round counts.
*/

pub trait DoubleRounds {
    const COUNT: usize;
}

pub struct R8;
impl DoubleRounds for R8 {
    const COUNT: usize = 4;
}

pub struct R12;
impl DoubleRounds for R12 {
    const COUNT: usize = 6;
}

pub struct R20;
impl DoubleRounds for R20 {
    const COUNT: usize = 10;
}
