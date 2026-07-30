use num_bigint::{BigInt, BigUint};
use serde::{Deserialize, Deserializer, Serializer, de::Error};
use std::str::FromStr;

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
