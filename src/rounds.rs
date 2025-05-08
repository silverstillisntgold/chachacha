/*!
Module containing the most commonly used ChaCha round counts. It would technically be possible to use
any amount of rounds, but it's just not reasonable or worthwhile for any application.
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
