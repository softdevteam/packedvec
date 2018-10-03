# Smaller immutable `Vec<u*>`s
This library provides a more efficient way of storing immutable
lists of unsigned integer values.

A `PackedVec` is a `struct` that stores the elements of a given `Vec<u64>` as
a series of`N` bit values, where `N` is the number of bits needed to represent
the largest number from the original list.

The `struct` stores the elements of the original `vec` as a smaller collection
of `u*` values. This is achieved by storing multiple elements inside a single
`u*` value.

```
    use packed_vec::PackedVec;

    let v = vec![1, 4294967296, 2, 3, 4294967296, 5];
    let mut small_vec = PackedVec::new(v);
    assert_eq!(small_vec.get(1), 4294967296);
    assert_eq!(small_vec.iter().next(), Some(1));
```
