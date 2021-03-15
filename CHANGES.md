# packedvec 1.2.3 (2021-03-15)

* `PackedVec` now implements `Clone`.


# packedvec 1.2.2 (2019-11-21)

* License as dual Apache-2.0/MIT (instead of a more complex, and little
  understood, triple license of Apache-2.0/MIT/UPL-1.0).


# packedvec 1.2.1 (2019-07-31)

* Add `is_empty` function to better match expectations from `Vec`.


# packedvec 1.2.0 (2018-12-29)

* Add an unsafe `get_unchecked` function. In the best case (a linear scan
  through a PackedVec), this is roughly 10% faster than `get`. Fortuitously,
  `iter()` hits this best case naturally, and thus is roughly 10% faster than
  previously.


# packedvec 1.1.0 (2018-10-17)

* Add `bwidth` function.
* Add `Eq`, `PartialEq`, and `Hash` implementations for `PackedVec`.


# packedvec 1.0.0 (2018-10-15)

First stable release.
