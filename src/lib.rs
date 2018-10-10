extern crate num;
extern crate rand;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use num::{cast::FromPrimitive, ToPrimitive, Unsigned};
use std::{cmp::Ord, marker::PhantomData};

const WORD_SIZE: usize = 64;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackedVec<T> {
    len: usize,
    bits: Vec<u64>,
    item_width: usize,
    phantom: PhantomData<T>,
}

impl<'a, T> PackedVec<T>
where
    T: FromPrimitive + Ord + ToPrimitive + Unsigned,
{
    /// Return a `PackedVec` containing a compressed version of the elements of
    /// `vec`.
    pub fn new(vec: Vec<T>) -> PackedVec<T> {
        let len = vec.len();
        let item_width = if let Some(v) = vec.iter().max() {
            i_log2((*v).to_u64().unwrap())
        } else {
            return PackedVec {
                len: 0,
                bits: vec![],
                item_width: 0,
                phantom: PhantomData,
            };
        };
        let mut bit_vec = vec![];
        let mut buf: u64 = 0;
        let mut bit_count: usize = 0;
        for ref item in vec.iter() {
            let item = item.to_u64().unwrap();
            if bit_count + item_width < WORD_SIZE {
                let shifted_item = item << WORD_SIZE - (item_width + bit_count);
                buf |= shifted_item;
                bit_count += item_width;
            } else {
                let remaining_bits = WORD_SIZE - bit_count;
                // add as many bits as possible before adding the remaining
                // bits to the next u64
                let first_half = item >> (item_width - remaining_bits);
                // for example if width = 5 and remaining_bits = 3
                // item = 00101 -> add 001 to the buffer, insert buffer into
                // bit array and create a new buffer containing 01 00000000...
                buf |= first_half;
                bit_vec.push(buf);
                buf = 0;
                bit_count = 0;
                if item_width - remaining_bits > 0 {
                    let mask = (1 << (item_width - remaining_bits)) - 1;
                    let mut second_half = item & mask;
                    bit_count += item_width - remaining_bits;
                    // add the second half of the number to the buffer
                    second_half <<= WORD_SIZE - bit_count;
                    buf |= second_half;
                }
            }
        }
        if buf != 0 {
            bit_vec.push(buf);
        }
        PackedVec {
            len,
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
        let item_index = (index * self.item_width) / WORD_SIZE;
        let start = (index * self.item_width) % WORD_SIZE;
        if start + self.item_width < WORD_SIZE {
            let mask = ((1 << self.item_width) - 1) << (WORD_SIZE - self.item_width - start);
            let item = (self.bits[item_index].to_u64().unwrap() & mask)
                >> (WORD_SIZE - self.item_width - start);
            Some(FromPrimitive::from_u64(item).unwrap())
        } else if self.item_width == WORD_SIZE {
            Some(FromPrimitive::from_u64(self.bits[item_index].to_u64().unwrap()).unwrap())
        } else {
            let bits_written = WORD_SIZE - start;
            let mask = ((1 << bits_written) - 1) << (WORD_SIZE - bits_written - start);
            let first_half = (self.bits[item_index].to_u64().unwrap() & mask)
                >> (WORD_SIZE - bits_written - start);
            // second half
            let remaining_bits = self.item_width - bits_written;
            if remaining_bits > 0 {
                let mask = ((1 << remaining_bits) - 1) << (WORD_SIZE - remaining_bits);
                let second_half = (self.bits[item_index + 1].to_u64().unwrap() & mask)
                    >> (WORD_SIZE - remaining_bits);
                Some(FromPrimitive::from_u64((first_half << remaining_bits) + second_half).unwrap())
            } else {
                Some(FromPrimitive::from_u64(first_half).unwrap())
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
    pub fn iter(&'a self) -> PackedVecIter<'a, T> {
        PackedVecIter {
            packed_vec: &self,
            idx: 0,
        }
    }
}

#[derive(Debug)]
pub struct PackedVecIter<'a, T>
where
    T: 'a + FromPrimitive + Ord + ToPrimitive + Unsigned,
{
    packed_vec: &'a PackedVec<T>,
    idx: usize,
}

impl<'a, T> Iterator for PackedVecIter<'a, T>
where
    T: 'a + FromPrimitive + Ord + ToPrimitive + Unsigned,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        self.packed_vec.get(self.idx - 1)
    }
}

fn i_log2(number: u64) -> usize {
    let mut bits = 63;
    while number & (1 << bits) == 0 {
        bits -= 1;
    }
    bits + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

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
        let v: Vec<u64> = vec![1, 2, 9223372036854775808, 100, 0, 3];
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

    fn random_unsigned_ints<T>()
    where
        T: Unsigned + ToPrimitive + FromPrimitive + Ord + Clone + std::fmt::Debug + rand::Rand,
    {
        const LENGTH: usize = 100000;
        let mut v: Vec<T> = Vec::with_capacity(LENGTH);
        for _ in 0..(LENGTH + 1) {
            v.push(rand::random::<T>());
        }
        let packed_v = PackedVec::new(v.clone());
        assert_eq!(packed_v.len(), v.len());
        assert!(packed_v.iter().zip(v.iter()).all(|(x, y)| x == *y));
    }

    #[test]
    fn random_vec() {
        random_unsigned_ints::<u8>();
        random_unsigned_ints::<u16>();
        random_unsigned_ints::<u32>();
        random_unsigned_ints::<u64>();
    }
}
