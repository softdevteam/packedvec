extern crate num_traits;
extern crate rand;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use num_traits::{cast::FromPrimitive, AsPrimitive, PrimInt, ToPrimitive, Unsigned};
use std::{cmp::Ord, marker::PhantomData, mem::size_of};

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackedVec<T, StorageT = usize> {
    len: usize,
    bits: Vec<StorageT>,
    item_width: usize,
    phantom: PhantomData<T>,
}

impl<'a, T> PackedVec<T, usize>
where
    T: 'static + AsPrimitive<usize> + FromPrimitive + Ord + PrimInt + ToPrimitive + Unsigned,
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
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive + Unsigned,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    /// Constructs a new `PackedVec` from `vec` (with a user-defined backing storage type).
    pub fn new_with_storaget(vec: Vec<T>) -> PackedVec<T, StorageT> {
        if size_of::<T>() > size_of::<StorageT>() {
            panic!("The backing storage type must be the same size or bigger as the stored integer size.");
        }
        let m = vec.iter().max();
        // If the input vector was empty, or if it consisted entirely of zeros, we don't need to
        // fill the backing storage with anything.
        if m.is_none() || *m.unwrap() == T::zero() {
            return PackedVec {
                len: vec.len(),
                bits: vec![],
                item_width: 0,
                phantom: PhantomData,
            };
        };

        let item_width = i_log2(*m.unwrap());
        let mut bit_vec = vec![];
        let mut buf = StorageT::zero();
        let mut bit_count: usize = 0;
        for item in &vec {
            let item: StorageT = (*item).as_();
            if bit_count + item_width < num_bits::<StorageT>() {
                let shifted_item = item << num_bits::<StorageT>() - (item_width + bit_count);
                buf = buf | shifted_item;
                bit_count += item_width;
            } else {
                let remaining_bits = num_bits::<StorageT>() - bit_count;
                // add as many bits as possible before adding the remaining
                // bits to the next u64
                let first_half = item >> (item_width - remaining_bits);
                // for example if width = 5 and remaining_bits = 3
                // item = 00101 -> add 001 to the buffer, insert buffer into
                // bit array and create a new buffer containing 01 00000000...
                buf = buf | first_half;
                bit_vec.push(buf);
                buf = StorageT::zero();
                bit_count = 0;
                if item_width - remaining_bits > 0 {
                    let mask = (StorageT::one() << (item_width - remaining_bits)) - StorageT::one();
                    let mut second_half = item & mask;
                    bit_count += item_width - remaining_bits;
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
            item_width,
            phantom: PhantomData,
        }
    }

    /// Return the value at the specified `index`
    /// # Example
    /// ```
    /// use packed_vec::PackedVec;
    /// let v: Vec<u8> = vec![1, 2, 3, 4];
    /// let packed_vec = PackedVec::new(v);
    /// let val: Option<u8> = packed_vec.get(3);
    /// assert_eq!(val, Some(4));
    /// ```
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        if self.item_width == 0 {
            // The original vector consisted entirely of zeros.
            return Some(T::zero());
        }

        let item_index = (index * self.item_width) / num_bits::<StorageT>();
        let start = (index * self.item_width) % num_bits::<StorageT>();
        if start + self.item_width < num_bits::<StorageT>() {
            let mask = ((StorageT::one() << self.item_width) - StorageT::one())
                << (num_bits::<StorageT>() - self.item_width - start);
            let item = (self.bits[item_index] & mask)
                >> (num_bits::<StorageT>() - self.item_width - start);
            Some(item.as_())
        } else if self.item_width == num_bits::<StorageT>() {
            Some(self.bits[item_index].as_())
        } else {
            let bits_written = num_bits::<StorageT>() - start;
            let mask = ((StorageT::one() << bits_written) - StorageT::one())
                << (num_bits::<StorageT>() - bits_written - start);
            let first_half =
                (self.bits[item_index] & mask) >> (num_bits::<StorageT>() - bits_written - start);
            // second half
            let remaining_bits = self.item_width - bits_written;
            if remaining_bits > 0 {
                let mask = ((StorageT::one() << remaining_bits) - StorageT::one())
                    << (num_bits::<StorageT>() - remaining_bits);
                let second_half =
                    (self.bits[item_index + 1] & mask) >> (num_bits::<StorageT>() - remaining_bits);
                Some(((first_half << remaining_bits) + second_half).as_())
            } else {
                Some(first_half.as_())
            }
        }
    }

    /// Return the number of elements in this.
    /// # Example
    /// ```
    /// use packed_vec::PackedVec;
    /// let v: Vec<u16> = vec![1, 2, 3, 4];
    /// let packed_vec = PackedVec::new(v);
    /// assert_eq!(packed_vec.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// An iterator over the elements of the vector.
    pub fn iter(&'a self) -> PackedVecIter<'a, T, StorageT> {
        PackedVecIter {
            packed_vec: self,
            idx: 0,
        }
    }
}

#[derive(Debug)]
pub struct PackedVecIter<'a, T: 'a, StorageT: 'a> {
    packed_vec: &'a PackedVec<T, StorageT>,
    idx: usize,
}

impl<'a, T, StorageT> Iterator for PackedVecIter<'a, T, StorageT>
where
    T: 'static + AsPrimitive<StorageT> + FromPrimitive + Ord + PrimInt + ToPrimitive + Unsigned,
    StorageT: AsPrimitive<T> + PrimInt + Unsigned,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        self.packed_vec.get(self.idx - 1)
    }
}

/// How many bits does this type represent?
fn num_bits<T>() -> usize {
    size_of::<T>() * 8
}

fn i_log2<T: PrimInt + Unsigned>(n: T) -> usize {
    debug_assert!(n > T::zero());
    let mut bits = num_bits::<T>() - 1;
    while n & (T::one() << bits) == T::zero() {
        bits -= 1;
    }
    bits + 1
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
        assert_eq!(packed_v.item_width, 0);
        assert_eq!(packed_v.bits, vec![]);
        let mut iter = packed_v.iter();
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn all_values_fit_in_one_item() {
        let v: Vec<u16> = vec![1, 2, 3];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.item_width, 2);
        assert_eq!(packed_v.bits, vec![7782220156096217088]);
        let mut iter = packed_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn value_spanning_two_items() {
        let v: Vec<u64> = vec![1, 4294967296, 2, 3, 4294967296, 5];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.item_width, 33);
        assert_eq!(
            packed_v.bits,
            vec![
                3221225472,
                1073741824,
                4035225266123964416,
                1441151880758558720
            ]
        );
        let mut iter = packed_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn values_fill_item_width() {
        let v: Vec<usize> = vec![1, 2, 9223372036854775808, 100, 0, 3];
        let v_len = v.len();
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v_len);
        assert_eq!(packed_v.item_width, 64);
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
    fn vecs_with_only_zeros() {
        let pv = PackedVec::new(vec![0u16, 0u16]);
        assert_eq!(pv.bits.len(), 0);
        assert_eq!(pv.get(0), Some(0));
        assert_eq!(pv.get(1), Some(0));
        assert_eq!(pv.get(2), None);
        assert_eq!(pv.iter().collect::<Vec<u16>>(), vec![0, 0]);
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
}
