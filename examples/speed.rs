use chachacha::*;
use std::{hint::black_box, time::Instant};

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::{ChaCha8Rng, rand_core::RngCore};

use chacha20::ChaCha8Rng as Wow;

const SIZE: usize = 8;
const BUF_LEN: usize = SIZE * GB;
const GB: usize = 1 << 30;

fn main() {
    let mut buf = vec![u8::MAX; BUF_LEN];
    let mut seed = [0; SEED_LEN_U8];
    getrandom::fill(&mut seed).unwrap();
    let mut c1 = ChaCha8Djb::new(seed.clone());
    let mut c2 = ChaCha8Rng::from_seed(Default::default());
    let mut c3 = Wow::from_seed(Default::default());

    let start = Instant::now();
    c1.fill(&mut buf);
    let mut delta = Instant::now().duration_since(start).as_secs_f64();
    let t = delta;

    delta /= BUF_LEN as f64;
    delta *= 1e9;
    println!("{:.2} ns per byte", delta);
    let speed1 = SIZE as f64 / t;
    println!("{:.2} GiB/s", speed1);

    let start = Instant::now();
    c2.fill_bytes(buf.as_mut_slice());
    let mut delta = Instant::now().duration_since(start).as_secs_f64();
    let t = delta;

    delta /= BUF_LEN as f64;
    delta *= 1e9;
    println!("{:.2} ns per byte", delta);
    let speed2 = SIZE as f64 / t;
    println!("{:.2} GiB/s", speed2);

    let start = Instant::now();
    c3.fill_bytes(buf.as_mut_slice());
    let mut delta = Instant::now().duration_since(start).as_secs_f64();
    let t = delta;
    black_box(buf);

    delta /= BUF_LEN as f64;
    delta *= 1e9;
    println!("{:.2} ns per byte", delta);
    let speed3 = SIZE as f64 / t;
    println!("{:.2} GiB/s", speed3);
    println!();

    let mut wow1 = speed1 / speed2;
    wow1 -= 1.0;
    wow1 *= 100.0;
    println!("Local over `rand`: {:.2}%", wow1);
    let mut wow2 = speed1 / speed3;
    wow2 -= 1.0;
    wow2 *= 100.0;
    println!("Local over `chacha20`: {:.2}%", wow2);
    println!();
}
