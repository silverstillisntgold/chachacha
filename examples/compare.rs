use rand::{Rng, SeedableRng};
use std::time::Instant;

const SIZE: usize = 7 * (1 << 30);

fn main() {
    let mut buf = [0; 48];
    getrandom::fill(&mut buf).unwrap();
    let mut buffer = vec![69_u8; SIZE];

    let mut c1 = chachacha::ChaCha8Djb::from_bytes(buf);
    let start = Instant::now();
    c1.fill(&mut buffer);
    let delta = start.elapsed().as_secs_f64();
    println!("time: {:.4}", delta);
    println!("GB/s: {:.4}", SIZE as f64 / delta / 1e9);

    let mut c2 = chachacha_042::ChaCha8Djb::from(buf);
    let start = Instant::now();
    c2.fill(&mut buffer);
    let delta = start.elapsed().as_secs_f64();
    println!("time: {:.4}", delta);
    println!("GB/s: {:.4}", SIZE as f64 / delta / 1e9);

    let mut c3 = chacha20::ChaCha8Rng::from_rng(&mut rand::rng());
    let start = Instant::now();
    c3.fill_bytes(&mut buffer);
    let delta = start.elapsed().as_secs_f64();
    println!("time: {:.4}", delta);
    println!("GB/s: {:.4}", SIZE as f64 / delta / 1e9);

    let mut c4 = rand_chacha::ChaCha8Rng::from_rng(&mut rand::rng());
    let start = Instant::now();
    c4.fill_bytes(&mut buffer);
    let delta = start.elapsed().as_secs_f64();
    println!("time: {:.4}", delta);
    println!("GB/s: {:.4}", SIZE as f64 / delta / 1e9);
}
