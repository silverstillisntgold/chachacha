use rand::{Rng, SeedableRng};
use std::time::Instant;

const SIZE: usize = 11 * (1 << 30) + 7;

fn main() {
    let mut buf = [0; _];
    let mut buffer = vec![u8::MAX; SIZE];

    for i in 1..=3 {
        println!("Iteration #{}", i);
        getrandom::fill(&mut buf).unwrap();

        let mut c1 = chachacha::ChaCha8Djb::from_bytes(buf);
        let start = Instant::now();
        c1.fill(&mut buffer);
        let delta = start.elapsed().as_secs_f64();
        println!("Current implementation:");
        println!("  time: {:.4}", delta);
        println!("  GB/s: {:.4}", SIZE as f64 / delta / 1e9);

        let mut c2 = chachacha_042::ChaCha8Djb::from(buf);
        let start = Instant::now();
        c2.fill(&mut buffer);
        let delta = start.elapsed().as_secs_f64();
        println!("Previous implementation:");
        println!("  time: {:.4}", delta);
        println!("  GB/s: {:.4}", SIZE as f64 / delta / 1e9);

        let mut c3 = chacha20::ChaCha8Rng::from_rng(&mut rand::rng());
        let start = Instant::now();
        c3.fill_bytes(&mut buffer);
        let delta = start.elapsed().as_secs_f64();
        println!("ChaCha20 library:");
        println!("  time: {:.4}", delta);
        println!("  GB/s: {:.4}", SIZE as f64 / delta / 1e9);

        let mut c4 = rand_chacha::ChaCha8Rng::from_rng(&mut rand::rng());
        let start = Instant::now();
        c4.fill_bytes(&mut buffer);
        let delta = start.elapsed().as_secs_f64();
        println!("RandChaCha library:");
        println!("  time: {:.4}", delta);
        println!("  GB/s: {:.4}", SIZE as f64 / delta / 1e9);
        println!();
    }
}
