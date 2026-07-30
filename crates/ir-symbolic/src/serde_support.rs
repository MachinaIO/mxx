use num_bigint::{BigInt, BigUint};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
use std::str::FromStr;

pub(crate) mod hex32 {
    use super::*;

    pub fn encode(value: &[u8; 32]) -> String {
        let mut encoded = String::with_capacity(64);
        for byte in value {
            use std::fmt::Write;
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        encoded
    }

    pub fn decode<E: Error>(encoded: &str) -> Result<[u8; 32], E> {
        if encoded.len() != 64 {
            return Err(E::custom("expected a 64-character lowercase hex digest"));
        }
        let mut result = [0_u8; 32];
        for (index, byte) in result.iter_mut().enumerate() {
            let offset = index * 2;
            let pair = &encoded[offset..offset + 2];
            if pair.bytes().any(|value| !value.is_ascii_digit() && !(b'a'..=b'f').contains(&value))
            {
                return Err(E::custom("expected lowercase hexadecimal digits"));
            }
            *byte = u8::from_str_radix(pair, 16).map_err(E::custom)?;
        }
        Ok(result)
    }

    pub fn serialize<S: Serializer>(value: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode(value))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 32], D::Error> {
        let encoded = String::deserialize(deserializer)?;
        decode::<D::Error>(&encoded)
    }
}

pub(crate) mod optional_hex32 {
    use super::*;

    pub fn serialize<S: Serializer>(
        value: &Option<[u8; 32]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&HexDigest(value)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<[u8; 32]>, D::Error> {
        Option::<HexDigestOwned>::deserialize(deserializer).map(|value| value.map(|value| value.0))
    }

    struct HexDigest<'a>(&'a [u8; 32]);

    impl Serialize for HexDigest<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            super::hex32::serialize(self.0, serializer)
        }
    }

    struct HexDigestOwned([u8; 32]);

    impl<'de> Deserialize<'de> for HexDigestOwned {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            super::hex32::deserialize(deserializer).map(Self)
        }
    }
}

pub(crate) mod hex32_set {
    use super::*;
    use serde::ser::SerializeSeq;
    use std::collections::BTreeSet;

    pub fn serialize<S: Serializer>(
        value: &BTreeSet<[u8; 32]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(value.len()))?;
        for digest in value {
            sequence.serialize_element(&HexDigest(digest))?;
        }
        sequence.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeSet<[u8; 32]>, D::Error> {
        let values = Vec::<HexDigestOwned>::deserialize(deserializer)?;
        let input_len = values.len();
        let result = values.into_iter().map(|value| value.0).collect::<BTreeSet<_>>();
        if result.len() == input_len {
            Ok(result)
        } else {
            Err(D::Error::custom("duplicate digest"))
        }
    }

    struct HexDigest<'a>(&'a [u8; 32]);

    impl Serialize for HexDigest<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            super::hex32::serialize(self.0, serializer)
        }
    }

    struct HexDigestOwned([u8; 32]);

    impl<'de> Deserialize<'de> for HexDigestOwned {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            super::hex32::deserialize(deserializer).map(Self)
        }
    }
}

pub(crate) mod bigint {
    use super::*;

    pub fn serialize<S: Serializer>(value: &BigInt, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<BigInt, D::Error> {
        let value = String::deserialize(deserializer)?;
        BigInt::from_str(&value).map_err(D::Error::custom)
    }
}

pub(crate) mod biguint {
    use super::*;

    pub fn serialize<S: Serializer>(value: &BigUint, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<BigUint, D::Error> {
        let value = String::deserialize(deserializer)?;
        BigUint::from_str(&value).map_err(D::Error::custom)
    }
}
