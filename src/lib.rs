use crate::bn256::Fr;
pub use halo2curves::bn256;
use halo2curves::ff::{Field, PrimeField};
use num_bigint::BigUint;

pub enum ReduceRule {
    Bytes(Vec<u8>, usize),
    Field(Fr, usize), // F * shiftbits
    U64(u64),
}

impl ReduceRule {
    fn nb_inputs(&self) -> usize {
        match self {
            ReduceRule::Bytes(_, a) => *a, // a * u64
            ReduceRule::Field(_, _) => 4,  // 4 * u64
            ReduceRule::U64(_) => 1,       // 1 * u64
        }
    }
    fn reduce(&mut self, v: u64, offset: usize) {
        match self {
            ReduceRule::Bytes(ref mut x, _) => {
                let mut bytes: Vec<u8> = v.to_le_bytes().to_vec();
                x.append(&mut bytes);
            } // a * u64
            ReduceRule::Field(ref mut x, shift) => {
                let mut acc = Fr::from_u128(v as u128);
                for _ in 0..offset {
                    acc = acc * Fr::from_u128(1u128 << *shift)
                }
                *x = *x + acc
            } // 4 * u64
            ReduceRule::U64(ref mut x) => {
                *x = v;
            } // 1 * u64
        }
    }

    fn reset(&mut self) {
        match self {
            ReduceRule::Bytes(ref mut x, _) => x.clear(), // a * u64
            ReduceRule::Field(ref mut x, _shift) => *x = Fr::ZERO, // 4 * u64
            ReduceRule::U64(ref mut x) => {
                *x = 0;
            } // 1 * u64
        }
    }

    pub fn field_value(&self) -> Option<Fr> {
        match self {
            ReduceRule::Bytes(_, _) => None,
            ReduceRule::Field(f, _) => Some(*f), // 4 * u64
            ReduceRule::U64(_) => None,          // 4 * u64
        }
    }
    pub fn bytes_value(&self) -> Option<Vec<u8>> {
        match self {
            ReduceRule::Bytes(b, _) => Some(b.clone()),
            ReduceRule::Field(_, _) => None, // 4 * u64
            ReduceRule::U64(_) => None,      // 4 * u64
        }
    }
    pub fn u64_value(&self) -> Option<u64> {
        match self {
            ReduceRule::Bytes(_, _) => None,
            ReduceRule::Field(_, _) => None, // 4 * u64
            ReduceRule::U64(v) => Some(*v),  // 4 * u64
        }
    }
}

pub struct Reduce {
    pub cursor: usize,
    pub rules: Vec<ReduceRule>,
}

impl Reduce {
    pub fn new(rules: Vec<ReduceRule>) -> Self {
        Reduce { cursor: 0, rules }
    }
    pub fn total_len(&self) -> usize {
        self.rules.iter().fold(0, |acc, x| acc + x.nb_inputs())
    }
}

impl Reduce {
    /// take in a u64 value and update all the reduce rule accordingly
    pub fn reduce(&mut self, v: u64) {
        let mut cursor = self.cursor;
        let total = self.total_len();
        if cursor == 0 {
            for rule in self.rules.iter_mut() {
                rule.reset()
            }
        }
        for index in 0..self.rules.len() {
            if cursor >= self.rules[index].nb_inputs() {
                cursor = cursor - self.rules[index].nb_inputs();
            } else {
                self.rules[index].reduce(v, cursor);
                break;
            }
        }
        self.cursor += 1;
        if self.cursor == total {
            self.cursor = 0;
        }
    }
}

pub fn bytes_to_u64(bytes: &[u8; 32]) -> [u64; 4] {
    let r = bytes
        .to_vec()
        .chunks_exact(8)
        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    r.try_into().unwrap()
}

pub fn field_to_bn(f: &Fr) -> BigUint {
    let bytes = f.to_bytes();
    BigUint::from_bytes_le(&bytes[..])
}

pub fn bn_to_field(bn: &BigUint) -> Fr {
    let mut bytes = bn.to_bytes_le();
    bytes.resize(48, 0);
    let bytes = &bytes[..];
    Fr::from_repr(bytes.try_into().unwrap()).unwrap()
}

pub fn bytes_to_field(bytes: &[u8; 32]) -> Fr {
    Fr::from_repr(bytes.clone()).unwrap()
}

pub fn field_to_u32(f: &Fr) -> u32 {
    let bytes = f.to_bytes();
    u32::from_le_bytes(bytes[0..4].try_into().unwrap())
}

pub fn field_to_u64(f: &Fr) -> u64 {
    let bytes = f.to_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::Reduce;
    use super::ReduceRule;
    use crate::bn256::Fr;
    use halo2curves::ff::PrimeField;
    fn new_reduce(rules: Vec<ReduceRule>) -> Reduce {
        Reduce { cursor: 0, rules }
    }

    #[test]
    fn test_reduce_bytes() {
        let reducerule = ReduceRule::Bytes(vec![], 4);
        let mut reduce = Reduce {
            cursor: 0,
            rules: vec![reducerule],
        };
        reduce.reduce(1);
    }

    #[test]
    fn test_reduce_bytes_twice() {
        let reducerule = ReduceRule::Bytes(vec![], 1);
        let mut reduce = Reduce {
            cursor: 0,
            rules: vec![reducerule],
        };
        reduce.reduce(1);
        reduce.reduce(2);
        assert_eq!(
            reduce.rules[0].bytes_value().unwrap(),
            vec![2, 0, 0, 0, 0, 0, 0, 0]
        )
    }

    #[test]
    fn test_reduce_u64() {
        let mut get = new_reduce(vec![
            ReduceRule::U64(0),
            ReduceRule::U64(0),
            ReduceRule::U64(0),
            ReduceRule::U64(0),
        ]);
        get.reduce(12);
        assert_eq!(get.cursor, 1);
        assert_eq!(get.rules[0].u64_value().unwrap(), 12);
    }

    #[test]
    fn test_reduce_fr() {
        let mut get = new_reduce(vec![ReduceRule::Field(Fr::zero(), 64)]);
        get.reduce(1);
        get.reduce(1);
        get.reduce(0);
        get.reduce(0);
        assert_eq!(
            get.rules[0].field_value().unwrap(),
            Fr::from_u128((1u128 << 64) + 1)
        );
    }
}
