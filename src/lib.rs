// Copyright (c) 2017 Garbiela Alexandra Moldovan
// Copyright (c) 2018 King's College London created by the Software Development Team
// <http://soft-dev.org/>
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0>, or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, or the UPL-1.0 license <http://opensource.org/licenses/UPL>
// at your option. This file may not be copied, modified, or distributed except according to those
// terms.

//! A [`PackedVec`](struct.PackedVec.html) stores vectors of integers efficiently while providing
//! a API similar to `Vec`.
//!
//! The main documentation for this crate can be found in the [`PackedVec`](struct.PackedVec.html)
//! struct.

extern crate num_traits;
extern crate rand;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use num_traits::{cast::FromPrimitive, AsPrimitive, PrimInt, ToPrimitive, Unsigned};
use std::{
    cmp::Ord,
    fmt::Debug,
    hash::{Hash, Hasher},
    mem::size_of,
};

/// A [`PackedVec`](struct.PackedVec.html) stores vectors of integers efficiently while providing
/// an API similar to `Vec`. The basic idea is to store each element using the minimum number of
/// bits needed to represent every element in the `Vec`. For example, if we have a `Vec<u64>` with
/// elements [20, 30, 140], every element wastes most of its 64 bits: 7 bits is sufficient to
/// represent the range of elements in the vector. Given this input vector, `PackedVec` stores each
/// elements using exactly 7 bits, saving substantial memory. For vectors which often contain small
/// ranges of numbers, and which are created rarely, but read from frequently, this can be a
/// significant memory and performance win.
///
/// # Examples
///
/// `PackedVec` has two main API differences from `Vec`: a `PackedVec` is created from a `Vec`; and
/// a `PackedVec` returns values rather than references. Both points can be seen in this example:
///
/// ```rust
/// use packedvec::PackedVec;
/// let v = vec![-1, 30, 120];
/// let pv = PackedVec::new(v.clone());
/// assert_eq!(pv.get(0), Some(-1));
/// assert_eq!(pv.get(2), Some(120));
/// assert_eq!(pv.get(3), None);
/// assert_eq!(v.iter().cloned().collect::<Vec<_>>(), pv.iter().collect::<Vec<_>>());
/// ```
///
/// ## Storage backing type
///
/// `PackedVec` defaults to using `usize` as a storage backing type. You can choose your own
/// storage type with the [`new_with_storaget()`](struct.PackedVec.html#method.new_with_storaget)
/// constructor. In general we recommend using the default `usize` backing storage unless you have
/// rigorously benchmarked your particular use case and are sure that a different storage type is
/// superior.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackedVec<T, StorageT = usize> {
    len: usize,
    bits: Vec<StorageT>,
    bwidth: usize,
    min: T,
}

impl<'a, T> PackedVec<T, usize>
where
    T: 'static + AsPrimitive<usize> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    usize: AsPrimitive<T>,
{
    /// Constructs a new `PackedVec` from `vec`. The `PackedVec` has `usize` backing storage, which
    /// is likely to be the best choice in nearly all situations.
    pub fn new(vec: Vec<T>) -> Self {
        PackedVec::new_with_storaget(vec)
    }
}

impl<'a, T, StorageT> PackedVec<T, StorageT>
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    /// Constructs a new `PackedVec` from `vec` (with a user-defined backing storage type).
    pub fn new_with_storaget(vec: Vec<T>) -> PackedVec<T, StorageT> {
        if size_of::<T>() > size_of::<StorageT>() {
            panic!("The backing storage type must be the same size or bigger as the stored integer size.");
        }

        // If the input vector was empty, we don't need to fill the backing storage with anything.
        if vec.is_empty() {
            return PackedVec {
                len: vec.len(),
                bits: vec![],
                bwidth: 0,
                min: T::zero(), // This value will never be used
            };
        };

        // We now want to find the difference between the biggest and smallest elements so that we
        // can pack things down as far as possible. In other words, if we have a vec [0, 4] the
        // range is 4 and we need 2 bits to represent that range; if we have a vec [-1, 1] the
        // range is 2 and we need 1 bit to represent that range.
        //
        // Perhaps surprisingly, it's faster to call iter().max() and iter().min() separately than
        // it is to write a single for loop over the input vector. This is probably because of the
        // overhead of an iterator, and perhaps because max() and min() use CPU vector
        // instructions.
        let max = *vec.iter().max().unwrap();
        let min = *vec.iter().min().unwrap();
        // If the input vector consisted entirely of the same element, we don't need to fill the
        // backing storage with anything.
        if max == min {
            return PackedVec {
                len: vec.len(),
                bits: vec![],
                bwidth: 0,
                // By definition "min" will be the value
                min,
            };
        };
        let bwidth = i_log2(delta(min, max));

        let mut bit_vec = vec![];
        let mut buf = StorageT::zero();
        let mut bit_count: usize = 0;
        for &e in &vec {
            let e = delta(min, e);
            if bit_count + bwidth < num_bits::<StorageT>() {
                let shifted_e = e << num_bits::<StorageT>() - (bwidth + bit_count);
                buf = buf | shifted_e;
                bit_count += bwidth;
            } else {
                let remaining_bits = num_bits::<StorageT>() - bit_count;
                // add as many bits as possible before adding the remaining
                // bits to the next u64
                let first_half = e >> (bwidth - remaining_bits);
                // for example if width = 5 and remaining_bits = 3
                // e = 00101 -> add 001 to the buffer, insert buffer into
                // bit array and create a new buffer containing 01 00000000...
                buf = buf | first_half;
                bit_vec.push(buf);
                buf = StorageT::zero();
                bit_count = 0;
                if bwidth - remaining_bits > 0 {
                    let mask = (StorageT::one() << (bwidth - remaining_bits)) - StorageT::one();
                    let mut second_half = e & mask;
                    bit_count += bwidth - remaining_bits;
                    // add the second half of the number to the buffer
                    second_half = second_half << num_bits::<StorageT>() - bit_count;
                    buf = buf | second_half;
                }
            }
        }
        if bit_count != 0 {
            bit_vec.push(buf);
        }

        PackedVec {
            len: vec.len(),
            bits: bit_vec,
            bwidth,
            min,
        }
    }

    /// Return the value at the specified `index`.
    ///
    /// # Example
    /// ```
    /// use packedvec::PackedVec;
    /// let v: Vec<u8> = vec![1, 2, 3, 4];
    /// let packedvec = PackedVec::new(v);
    /// let val: Option<u8> = packedvec.get(3);
    /// assert_eq!(val, Some(4));
    /// ```
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let min = self.min;
        if self.bwidth == 0 {
            // The original vector consisted entirely of the same element.
            return Some(min);
        }

        let item_index = (index * self.bwidth) / num_bits::<StorageT>();
        let start = (index * self.bwidth) % num_bits::<StorageT>();
        if start + self.bwidth < num_bits::<StorageT>() {
            let mask = ((StorageT::one() << self.bwidth) - StorageT::one())
                << (num_bits::<StorageT>() - self.bwidth - start);
            let item =
                (self.bits[item_index] & mask) >> (num_bits::<StorageT>() - self.bwidth - start);
            Some(inv_delta(min, item))
        } else if self.bwidth == num_bits::<StorageT>() {
            Some(inv_delta(min, self.bits[item_index]))
        } else {
            let bits_written = num_bits::<StorageT>() - start;
            let mask = ((StorageT::one() << bits_written) - StorageT::one())
                << (num_bits::<StorageT>() - bits_written - start);
            let first_half =
                (self.bits[item_index] & mask) >> (num_bits::<StorageT>() - bits_written - start);
            // second half
            let remaining_bits = self.bwidth - bits_written;
            if remaining_bits > 0 {
                let mask = ((StorageT::one() << remaining_bits) - StorageT::one())
                    << (num_bits::<StorageT>() - remaining_bits);
                let second_half =
                    (self.bits[item_index + 1] & mask) >> (num_bits::<StorageT>() - remaining_bits);
                Some(inv_delta(min, (first_half << remaining_bits) + second_half))
            } else {
                Some(inv_delta(min, first_half))
            }
        }
    }

    /// Return the number of elements in this `PackedVec`.
    ///
    /// # Example
    /// ```
    /// use packedvec::PackedVec;
    /// let v: Vec<u16> = vec![1, 2, 3, 4];
    /// let packedvec = PackedVec::new(v);
    /// assert_eq!(packedvec.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns an iterator over the `PackedVec`.
    pub fn iter(&'a self) -> PackedVecIter<'a, T, StorageT> {
        PackedVecIter {
            packedvec: self,
            idx: 0,
        }
    }
}

impl<T: PartialEq> PartialEq for PackedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len || self.min != other.min || self.bwidth != other.bwidth {
            return false;
        }
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(b1, b2)| b1 == b2)
    }
}

impl<T: Debug + PartialEq> Eq for PackedVec<T> {}

impl<T: Debug + Hash> Hash for PackedVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for blk in self.bits.iter() {
            blk.hash(state);
        }
    }
}

#[derive(Debug)]
pub struct PackedVecIter<'a, T: 'a, StorageT: 'a> {
    packedvec: &'a PackedVec<T, StorageT>,
    idx: usize,
}

impl<'a, T, StorageT> Iterator for PackedVecIter<'a, T, StorageT>
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        self.packedvec.get(self.idx - 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let ub = self.packedvec.len() - self.idx; // upper bound
        (ub, Some(ub))
    }
}

/// How many bits does this type represent?
fn num_bits<T>() -> usize {
    size_of::<T>() * 8
}

/// Convert (possibly signed) `x` of type `T` into an unsigned `StorageT`.
fn abs<T, StorageT>(x: T) -> StorageT
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    debug_assert!(size_of::<StorageT>() >= size_of::<T>());
    // These three clauses are not easy to understand, because they can deal with both signed and
    // unsigned types. The first clause *always* matches against unsigned types. It also matches
    // sometimes against signed types. The second and third clauses can only ever match against
    // signed types.
    if x >= T::zero() {
        x.as_()
    } else if x == T::min_value() {
        T::max_value().as_() + StorageT::one()
    } else {
        (T::zero() - x).as_()
    }
}

/// Return the delta between `min` and `max` as an unsigned integer (i.e. delta(-2, 2) == 4).
fn delta<T, StorageT>(min: T, max: T) -> StorageT
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    debug_assert!(size_of::<StorageT>() >= size_of::<T>());
    // These three clauses are not easy to understand, because they can deal with both signed and
    // unsigned types. The first clause *always* matches against unsigned types. It also matches
    // sometimes against signed types. The second and third clauses can only ever match against
    // signed types.
    if min >= T::zero() {
        (max - min).as_()
    } else if min < T::zero() && max < T::zero() {
        abs(max) - abs(min)
    } else {
        debug_assert!(min < T::zero());
        max.as_() + abs(min)
    }
}

/// Given a (possibly signed) minimum value `min` and an absolute delta `d`, return a (possibly)
/// signed inverted value (i.e. inv_delta(-2, 4) == 2).
fn inv_delta<T, StorageT>(min: T, d: StorageT) -> T
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    debug_assert!(size_of::<StorageT>() >= size_of::<T>());
    // These three clauses are not easy to understand, because they can deal with both signed and
    // unsigned types. The first clause *always* matches against unsigned types. It also matches
    // sometimes against signed types. The second and third clauses can only ever match against
    // signed types.
    if d <= T::max_value().as_() {
        min + d.as_()
    } else if d == abs(T::min_value()) + abs(T::max_value()) {
        debug_assert!(min == T::min_value());
        T::max_value()
    } else {
        min + T::max_value() + (d - T::max_value().as_()).as_()
    }
}

fn i_log2<T: PrimInt + Unsigned>(n: T) -> usize {
    size_of::<T>() * 8 - n.leading_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rand;

    #[test]
    fn empty_vec() {
        let v: Vec<u16> = vec![];
        let packed_v = PackedVec::new(v);
        assert_eq!(packed_v.len(), 0);
        assert_eq!(packed_v.bwidth, 0);
        assert_eq!(packed_v.bits, vec![]);
        let mut iter = packed_v.iter();
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn all_values_fit_in_one_item() {
        let v: Vec<u16> = vec![0, 2, 3];
        let v_len = v.len();
        let packed_v = PackedVec::<u16, u64>::new_with_storaget(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.bwidth, 2);
        assert_eq!(packed_v.bits, vec![3170534137668829184]);
        let mut iter = packed_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn value_spanning_two_items() {
        let v = vec![0, 4294967296, 2, 3, 4294967296, 5];
        let v_len = v.len();
        let packed_v = PackedVec::<u64, u64>::new_with_storaget(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.bwidth, 33);
        assert_eq!(
            packed_v.bits,
            vec![
                1073741824,
                1073741824,
                4035225266123964416,
                1441151880758558720
            ]
        );
        for (&orig, packed) in v.iter().zip(packed_v.iter()) {
            assert_eq!(orig, packed);
        }
    }

    #[test]
    fn values_fill_bwidth() {
        let v: Vec<usize> = vec![1, 2, 9223372036854775808, 100, 0, 3];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.bwidth, 64);
        assert_eq!(packed_v.bits, v);
        let mut iter = packed_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn vec_u8() {
        let v: Vec<u8> = vec![255, 2, 255];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        let mut iter = packed_v.iter();
        for number in v {
            let value: Option<u8> = iter.next();
            assert_eq!(value, Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn vec_u16() {
        let v: Vec<u16> = vec![1, 2, 65535];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        let mut iter = packed_v.iter();
        for number in v {
            let value: Option<u16> = iter.next();
            assert_eq!(value, Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn vec_u32() {
        let v: Vec<u32> = vec![1, 4294967295, 2, 100, 65535];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        let mut iter = packed_v.iter();
        for number in v {
            let value: Option<u32> = iter.next();
            assert_eq!(value, Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn vec_u64() {
        let v: Vec<u64> = vec![
            1,
            4294967295,
            18446744073709551615,
            100,
            18446744073709551613,
            65535,
        ];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        let mut iter = packed_v.iter();
        for number in v {
            let value: Option<u64> = iter.next();
            assert_eq!(value, Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    fn random_unsigned_ints<T, StorageT>()
    where
        T: 'static
            + AsPrimitive<StorageT>
            + FromPrimitive
            + Ord
            + PrimInt
            + Rand
            + ToPrimitive
            + Unsigned,
        StorageT: AsPrimitive<T> + PrimInt + Unsigned,
    {
        const LENGTH: usize = 100000;
        let mut v: Vec<T> = Vec::with_capacity(LENGTH);
        for _ in 0..(LENGTH + 1) {
            v.push(rand::random::<T>());
        }
        let packed_v = PackedVec::<T, StorageT>::new_with_storaget(v.clone());
        assert_eq!(packed_v.len(), v.len());
        assert!(packed_v.iter().zip(v.iter()).all(|(x, y)| x == *y));
    }

    #[test]
    fn random_vec() {
        random_unsigned_ints::<u8, u8>();
        random_unsigned_ints::<u8, u16>();
        random_unsigned_ints::<u8, u32>();
        random_unsigned_ints::<u8, u64>();
        random_unsigned_ints::<u16, u16>();
        random_unsigned_ints::<u16, u32>();
        random_unsigned_ints::<u16, u64>();
        random_unsigned_ints::<u32, u32>();
        random_unsigned_ints::<u32, u64>();
        random_unsigned_ints::<u64, u64>();
    }

    #[test]
    #[should_panic]
    fn t_must_not_be_bigger_than_storaget() {
        PackedVec::<u16, u8>::new_with_storaget(vec![0]);
    }

    #[test]
    fn vecs_with_only_one_value() {
        let pv = PackedVec::new(vec![0u16, 0u16]);
        assert_eq!(pv.bits.len(), 0);
        assert_eq!(pv.get(0), Some(0));
        assert_eq!(pv.get(1), Some(0));
        assert_eq!(pv.get(2), None);
        assert_eq!(pv.iter().collect::<Vec<u16>>(), vec![0, 0]);

        let pv = PackedVec::new(vec![u16::max_value(), u16::max_value()]);
        assert_eq!(pv.bits.len(), 0);
        assert_eq!(
            pv.iter().collect::<Vec<u16>>(),
            vec![u16::max_value(), u16::max_value()]
        );
    }

    #[test]
    fn zero_bits_spanning_across_elements() {
        let v: Vec<u64> = vec![0, 5, 17, 17, 6, 3, 25, 29, 10, 0, 10, 0, 2];
        let v_len = v.len();
        let pv = PackedVec::new(v.clone());
        assert_eq!(pv.len(), v_len);
        let mut iter = pv.iter();
        for number in v {
            let value: Option<u64> = iter.next();
            assert_eq!(value, Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_delta() {
        // The trickiness here almost entirely relates to negative maximum values for signed
        // integer types
        assert_eq!(delta::<u8, u8>(0, 2), 2);
        assert_eq!(delta::<u8, u8>(0, u8::max_value()), u8::max_value());
        assert_eq!(delta::<i8, u8>(-2, 0), 2);
        assert_eq!(delta::<i8, u8>(-2, 2), 4);
        assert_eq!(delta::<i8, u8>(i8::min_value(), 0), 128);
        assert_eq!(delta::<i8, u8>(0, i8::max_value()), 127);
        assert_eq!(
            delta::<i8, u8>(i8::min_value(), i8::max_value()),
            u8::max_value()
        );
        assert_eq!(delta::<i8, u8>(i8::min_value(), i8::min_value()), 0);
        assert_eq!(
            delta::<i32, u32>(i32::min_value(), i32::max_value()),
            (i32::max_value() as u32) * 2 + 1
        );
        assert_eq!(
            delta::<i32, usize>(i32::min_value(), i32::max_value()),
            (i32::max_value() as usize) * 2 + 1
        );
    }

    #[test]
    fn test_inv_delta() {
        // These tests are, in essence, those from delta with the last two values swapped (i.e.
        // assert_eq!(delta(x, y), z) becomes assert_eq!(inv_delta(x, z), y).
        assert_eq!(inv_delta::<u8, u8>(0, 2), 2);
        assert_eq!(inv_delta::<u8, u8>(0, u8::max_value()), u8::max_value());
        assert_eq!(inv_delta::<i8, u8>(-2, 2), 0);
        assert_eq!(inv_delta::<i8, u8>(-2, 4), 2);
        assert_eq!(inv_delta::<i8, u8>(i8::min_value(), 128), 0);
        assert_eq!(inv_delta::<i8, u8>(0, 127), i8::max_value());
        assert_eq!(
            inv_delta::<i8, u8>(i8::min_value(), u8::max_value()),
            i8::max_value()
        );
        assert_eq!(inv_delta::<i8, u8>(i8::min_value(), 0), i8::min_value());
        assert_eq!(
            inv_delta::<i32, u32>(i32::min_value(), ((i32::max_value() as u32) * 2 + 1).as_()),
            i32::max_value()
        );
        assert_eq!(
            inv_delta::<i32, usize>(
                i32::min_value(),
                ((i32::max_value() as usize) * 2 + 1).as_()
            ),
            i32::max_value()
        );
    }

    #[test]
    fn efficient_range() {
        let pv = PackedVec::new(vec![9998, 9999, 10000]);
        assert_eq!(pv.bwidth, 2);
        assert_eq!(pv.iter().collect::<Vec<_>>(), vec![9998, 9999, 10000]);
    }

    #[test]
    fn negative_values() {
        let pv = PackedVec::new(vec![-1, 1]);
        assert_eq!(pv.iter().collect::<Vec<_>>(), vec![-1, 1]);

        let pv = PackedVec::new(vec![i32::min_value(), 1]);
        assert_eq!(pv.iter().collect::<Vec<_>>(), vec![i32::min_value(), 1]);

        let pv = PackedVec::<i32, u32>::new_with_storaget(vec![i32::min_value(), i32::max_value()]);
        assert_eq!(
            pv.iter().collect::<Vec<_>>(),
            vec![i32::min_value(), i32::max_value()]
        );
    }

    #[test]
    fn test_eq() {
        assert_eq!(PackedVec::<u8>::new(vec![]), PackedVec::new(vec![]));
        assert_eq!(PackedVec::new(vec![0]), PackedVec::new(vec![0]));
        assert_eq!(PackedVec::new(vec![4, 10]), PackedVec::new(vec![4, 10]));
        assert_eq!(
            PackedVec::new(vec![u32::max_value(), u32::min_value()]),
            PackedVec::new(vec![u32::max_value(), u32::min_value()])
        );
        assert_ne!(PackedVec::new(vec![1, 4]), PackedVec::new(vec![0, 3]));
    }
}
