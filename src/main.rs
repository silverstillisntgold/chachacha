#![allow(unused)]

use chacha20::cipher::*;
use chacha20::*;
use chachacha::*;
use std::hint::black_box;
use std::time;

const GB: usize = 1 << 30;
const SIZE: usize = 8 * GB;

fn main() {
    let mut data = vec![u8::MAX; black_box(SIZE)];

    let mut cc_og = ChaCha20Legacy::new(&[0x42; 32].into(), &[0x24; 8].into());
    let start = time::Instant::now();
    cc_og.apply_keystream(&mut data);
    let delta = time::Instant::now().duration_since(start);
    black_box(&data);
    println!("time to fill 1: {:.2} seconds", delta.as_secs_f64());
    let perf1 = SIZE as f64 / GB as f64 / delta.as_secs_f64();
    println!("perf 1: {:.2} GB/s", perf1);

    let mut cc = ChaCha20Djb::from([69; SEED_LEN_U8]);
    let start = time::Instant::now();
    cc.xor(&mut data);
    let delta = time::Instant::now().duration_since(start);
    black_box(&data);
    println!("time to fill 2: {:.2} seconds", delta.as_secs_f64());
    let perf2 = SIZE as f64 / GB as f64 / delta.as_secs_f64();
    println!("perf 2: {:.2} GB/s", perf2);
    println!(
        "perf delta: {:.2}%",
        perf1.max(perf2) / perf1.min(perf2) * 100.0 - 100.0
    );

    let mut cc_wowie = chacha::ChaCha20::new(&[0x42; 32], &[0x24; 8]);
    let start = time::Instant::now();
    cc_wowie.xor_keystream(&mut data);
    let delta: time::Duration = time::Instant::now().duration_since(start);
    black_box(&data);
    println!("time to fill 3: {:.2} seconds", delta.as_secs_f64());
    let perf1 = SIZE as f64 / GB as f64 / delta.as_secs_f64();
    println!("perf 3: {:.2} GB/s", perf1);
}
