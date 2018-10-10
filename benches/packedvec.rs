#![feature(test)]

extern crate packed_vec;
extern crate rand;
extern crate test;

use packed_vec::*;
use test::Bencher;

#[bench]
fn creation(bench: &mut Bencher) {
    let mut v = vec![];
    for i in 0..10000u16 {
        v.push(i);
    }
    bench.iter(|| PackedVec::new(v.clone()));
}

#[bench]
fn iteration(bench: &mut Bencher) {
    let mut v = vec![];
    for i in 0..10000u16 {
        v.push(i);
    }
    let pv = PackedVec::new(v);
    bench.iter(|| pv.iter().collect::<Vec<_>>());
}
