[![Build Status](https://api.travis-ci.org/softdevteam/packedvec.svg?branch=master)](https://travis-ci.org/softdevteam/packedvec)
[![Latest version](https://img.shields.io/crates/v/packedvec.svg)](https://crates.io/crates/packedvec)
[![Documentation](https://docs.rs/packedvec/badge.svg)](https://docs.rs/packedvec)

# PackedVec

A [`PackedVec`](https://docs.rs/packedvec/) stores vectors of integers
efficiently while providing an API similar to `Vec`. The basic idea is to store
each element using the minimum number of bits needed to represent every element
in the `Vec`. For example, if we have a `Vec<u64>` with elements [20, 30, 140],
every element wastes most of its 64 bits: 7 bits is sufficient to represent the
range of elements in the vector. Given this input vector, `PackedVec` stores
each elements using exactly 7 bits, saving substantial memory. For vectors which
often contain small ranges of numbers, and which are created rarely, but read
from frequently, this can be a significant memory and performance win.

# Examples

`PackedVec` has two main API differences from `Vec`: a `PackedVec` is created
from a `Vec`; and a `PackedVec` returns values rather than references. Both
points can be seen in this example:

```rust
use packedvec::PackedVec;
let v = vec![-1, 30, 120];
let pv = PackedVec::new(v.clone());
assert_eq!(pv.get(0), Some(-1));
assert_eq!(pv.get(2), Some(120));
assert_eq!(pv.get(3), None);
assert_eq!(v.iter().cloned().collect::<Vec<_>>(), pv.iter().collect::<Vec<_>>());
```

