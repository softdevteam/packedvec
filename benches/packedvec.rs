#![feature(test)]

extern crate packedvec;
extern crate rand;
extern crate test;

use packedvec::*;
use test::{black_box, Bencher};

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

#[bench]
fn get(bench: &mut Bencher) {
    let mut v = vec![];
    for _ in 0..100 {
        for i in 0..1000u16 {
            v.push(i);
        }
    }
    let pv = PackedVec::new(v);
    bench.iter(|| {
        for i in 0..pv.len() {
            black_box(pv.get(i));
        }
    });
}

#[bench]
fn get_unchecked(bench: &mut Bencher) {
    let mut v = vec![];
    for _ in 0..100 {
        for i in 0..1000u16 {
            v.push(i);
        }
    }
    let pv = PackedVec::new(v);
    bench.iter(|| {
        for i in 0..pv.len() {
            black_box(unsafe { pv.get_unchecked(i) });
        }
    });
}
