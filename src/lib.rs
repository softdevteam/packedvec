const WORD_SIZE:usize = 64;

#[derive(Debug)]
pub struct PackedVec {
    len: usize,
    bits: Vec<u64>,
    item_width: usize,
}

#[derive(Debug)]
pub struct PackedVecIter<'a> {
    packed_vec: &'a PackedVec,
    idx: usize,
}

impl<'a> Iterator for PackedVecIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        self.packed_vec.get(self.idx - 1)
    }
}

impl <'a> PackedVec {
    /// Return the value at the specified `index`
    /// # Example
    /// ```
    /// use compressed_dst::PackedVec;
    /// let v = vec![1, 2, 3, 4];
    /// let small_vec = PackedVec::from_u64_vec(v);
    /// assert_eq!(small_vec.get(3), Some(4));
    /// ```
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        let item_index = (index * self.item_width) / WORD_SIZE;
        let start = (index * self.item_width) % WORD_SIZE;
        if start + self.item_width < WORD_SIZE {
            let mask = ((1 << self.item_width) - 1)
                << (WORD_SIZE - self.item_width - start);
            let item = (self.bits[item_index] & mask) >>
                (WORD_SIZE - self.item_width - start);
            Some(item)
        } else if self.item_width == WORD_SIZE {
            Some(self.bits[item_index])
        } else {
            let bits_written = WORD_SIZE - start;
            let mask = ((1 << bits_written) - 1)
                << (WORD_SIZE - bits_written - start);
            let first_half = (self.bits[item_index] & mask)
                >> (WORD_SIZE - bits_written - start);
            // second half
            let remaining_bits = self.item_width - bits_written;
            if remaining_bits > 0 {
                let mask = ((1 << remaining_bits) - 1)
                    << (WORD_SIZE - remaining_bits);
                let second_half = (self.bits[item_index + 1] & mask)
                    >> (WORD_SIZE - remaining_bits);
                Some((first_half << remaining_bits) + second_half)
            } else {
                Some(first_half)
            }
        }
    }

    /// Return a `PackedVec` containing a compressed version of the elements of
    /// `vec`.
    pub fn from_u64_vec(vec: Vec<u64>) -> PackedVec {
        let len = vec.len();
        let item_width = if let Some(v) = vec.iter().max() {
            let log = (*v as f64).log2();
            if log == log.ceil() {
                (log + 1.0) as usize
            } else {
                log.ceil() as usize
            }
        } else {
            return PackedVec {
                len: 0,
                bits: vec![],
                item_width: 0,
            };
        };
        let mut bit_vec = vec![];
        let mut buf: u64 = 0;
        let mut bit_count: usize = 0;
        for &item in vec.iter() {
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
            len: len,
            bits: bit_vec,
            item_width: item_width,
        }
    }

    /// Return the number of elements in this.
    /// # Example
    /// ```
    /// use compressed_dst::PackedVec;
    /// let v = vec![1, 2, 3, 4];
    /// let small_vec = PackedVec::from_u64_vec(v);
    /// assert_eq!(small_vec.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// An iterator over the elements of the vector.
    pub fn iter(&'a self) -> PackedVecIter<'a> {
        PackedVecIter {
            packed_vec: &self,
            idx: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vec() {
        let v = vec![];
        let small_v = PackedVec::from_u64_vec(v);
        assert_eq!(small_v.len(), 0);
        assert_eq!(small_v.item_width, 0);
        assert_eq!(small_v.bits, vec![]);
        let mut iter = small_v.iter();
        assert_eq!(iter.idx, 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn all_values_fit_in_one_item() {
        let v = vec![1, 2, 3];
        let v_len = v.len();
        let small_v = PackedVec::from_u64_vec(v.clone());
        assert_eq!(small_v.len(), v_len);
        assert_eq!(small_v.item_width, 2);
        assert_eq!(small_v.bits, vec![7782220156096217088]);
        let mut iter = small_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn value_spanning_two_items() {
        let v = vec![1, 4294967296, 2, 3, 4294967296, 5];
        let v_len = v.len();
        let small_v = PackedVec::from_u64_vec(v.clone());
        assert_eq!(small_v.len(), v_len);
        assert_eq!(small_v.item_width, 33);
        assert_eq!(small_v.bits, vec![3221225472, 1073741824,
                                      4035225266123964416,
                                      1441151880758558720]);
        let mut iter = small_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn values_fill_item_width() {
        let v = vec![1, 2, 9223372036854775808, 100, 0, 3];
        let v_len = v.len();
        let small_v = PackedVec::from_u64_vec(v.clone());
        assert_eq!(small_v.len(), v_len);
        assert_eq!(small_v.item_width, 64);
        assert_eq!(small_v.bits, v);
        let mut iter = small_v.iter();
        for number in v {
            assert_eq!(iter.next(), Some(number));
        }
        assert_eq!(iter.next(), None);
    }
}
