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
