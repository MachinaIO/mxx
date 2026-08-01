use num_bigint::BigInt;
use serde::{Deserialize, Deserializer, Serializer, de::Error};
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

pub(crate) mod optional_bigint {
    use super::*;

    pub fn serialize<S: Serializer>(
        value: &Option<BigInt>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&value.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<BigInt>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|value| BigInt::from_str(&value).map_err(D::Error::custom))
            .transpose()
    }
}

pub(crate) mod bigint_vec {
    use super::*;
    use serde::Serialize;

    pub fn serialize<S: Serializer>(values: &[BigInt], serializer: S) -> Result<S::Ok, S::Error> {
        values.iter().map(ToString::to_string).collect::<Vec<_>>().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<BigInt>, D::Error> {
        Vec::<String>::deserialize(deserializer)?
            .into_iter()
            .map(|value| BigInt::from_str(&value).map_err(D::Error::custom))
            .collect()
    }
}
