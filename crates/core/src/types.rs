//! Value types: what the core knows about a value.
//!
//! A [`CoreType`] says how many bytes a value has and how they are grouped, and whether the core
//! itself reads the value. It says nothing about what the bytes mean: a matrix over a ring, a key,
//! and a ciphertext of the same size have the same type here. A DSL keeps the meaning on its own
//! handles and checks it before lowering.
//!
//! Sizes are compile expressions, so after parameters are bound every value has a known number of
//! bytes before anything runs ([`ConcreteType`]). A family element type may depend on the member
//! index through [`crate::expr::MEMBER_INDEX`]; a family whose members resolve to different types
//! is a [`ConcreteFamily::Dependent`] family.
//!
//! In memory and in the byte encoding of inputs and outputs, a buffer is its bytes, an `Int` of
//! `w` words is `8 w` bytes of little-endian two's complement, a `Bool` is one little-endian `u64`
//! that is `0` or `1`, and a family is its members in order.

use crate::expr::{Env, ExprError, IntExpr, MEMBER_INDEX};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreType {
    /// `count` elements of `element_bytes` bytes. The core stores every value 8-byte aligned.
    Buffer {
        element_bytes: IntExpr,
        count: IntExpr,
    },
    /// `count` members of one type; the element type may depend on the member index.
    Family {
        element: Box<CoreType>,
        count: IntExpr,
    },
    /// A signed integer of `words` 64-bit words that the core may read.
    Int {
        words: u32,
    },
    Bool,
    /// An opaque value a store can hold, identified by its schema. It has no size known to the
    /// core, so it cannot appear in a proof image.
    TypedBlob {
        type_name: String,
        schema_hash: [u8; 32],
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConcreteType {
    Buffer { bytes: u64 },
    Family(ConcreteFamily),
    Int { words: u32 },
    Bool,
    TypedBlob { type_name: String, schema_hash: [u8; 32] },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConcreteFamily {
    Uniform { element: Box<ConcreteType>, count: u64 },
    Dependent(Vec<ConcreteType>),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TypeError {
    #[error(transparent)]
    Expr(#[from] ExprError),
    #[error("negative or oversized size {0}")]
    BadSize(i128),
    #[error("a dependent family of {0} members is too large to resolve member by member")]
    DependentFamilyTooLarge(u64),
}

/// The most members a dependent family may have: each member's type is stored.
pub const MAX_DEPENDENT_MEMBERS: u64 = 1 << 20;

impl CoreType {
    pub fn buffer(element_bytes: impl Into<IntExpr>, count: impl Into<IntExpr>) -> Self {
        Self::Buffer { element_bytes: element_bytes.into(), count: count.into() }
    }

    pub fn bytes(bytes: impl Into<IntExpr>) -> Self {
        Self::buffer(1, bytes)
    }

    pub fn family(element: CoreType, count: impl Into<IntExpr>) -> Self {
        Self::Family { element: Box::new(element), count: count.into() }
    }

    /// The concrete type in `env`.
    pub fn resolve(&self, env: &Env) -> Result<ConcreteType, TypeError> {
        Ok(match self {
            Self::Buffer { element_bytes, count } => {
                let bytes = element_bytes
                    .eval(env)?
                    .checked_mul(count.eval(env)?)
                    .ok_or(ExprError::Overflow)?;
                ConcreteType::Buffer { bytes: size(bytes)? }
            }
            Self::Family { element, count } => {
                let count = size(count.eval(env)?)?;
                if element.uses_member_index() {
                    if count > MAX_DEPENDENT_MEMBERS {
                        return Err(TypeError::DependentFamilyTooLarge(count));
                    }
                    let members = (0..count)
                        .map(|i| element.resolve(&env.with_index(MEMBER_INDEX, i128::from(i))))
                        .collect::<Result<Vec<_>, _>>()?;
                    ConcreteFamily::from_members(members).into()
                } else {
                    ConcreteType::Family(ConcreteFamily::Uniform {
                        element: Box::new(element.resolve(env)?),
                        count,
                    })
                }
            }
            Self::Int { words } => ConcreteType::Int { words: *words },
            Self::Bool => ConcreteType::Bool,
            Self::TypedBlob { type_name, schema_hash } => {
                ConcreteType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
            }
        })
    }

    /// This type with every parameter in `values` replaced by its expression.
    pub fn substitute_vars(&self, values: &std::collections::BTreeMap<String, IntExpr>) -> Self {
        match self {
            Self::Buffer { element_bytes, count } => Self::Buffer {
                element_bytes: element_bytes.substitute_vars(values),
                count: count.substitute_vars(values),
            },
            Self::Family { element, count } => Self::Family {
                element: Box::new(element.substitute_vars(values)),
                count: count.substitute_vars(values),
            },
            Self::Int { .. } | Self::Bool | Self::TypedBlob { .. } => self.clone(),
        }
    }

    fn uses_member_index(&self) -> bool {
        match self {
            Self::Buffer { element_bytes, count, .. } => {
                element_bytes.uses_index(MEMBER_INDEX) || count.uses_index(MEMBER_INDEX)
            }
            Self::Family { element, count } => {
                element.uses_member_index() || count.uses_index(MEMBER_INDEX)
            }
            Self::Int { .. } | Self::Bool | Self::TypedBlob { .. } => false,
        }
    }
}

fn size(value: i128) -> Result<u64, TypeError> {
    u64::try_from(value).map_err(|_| TypeError::BadSize(value))
}

impl From<ConcreteFamily> for ConcreteType {
    fn from(family: ConcreteFamily) -> Self {
        Self::Family(family)
    }
}

impl ConcreteFamily {
    /// The family of these members: uniform when they all have one type.
    pub fn from_members(members: Vec<ConcreteType>) -> Self {
        match members.first() {
            Some(first) if members.iter().all(|m| m == first) => {
                Self::Uniform { element: Box::new(first.clone()), count: members.len() as u64 }
            }
            None => Self::Dependent(Vec::new()),
            Some(_) => Self::Dependent(members),
        }
    }

    pub fn len(&self) -> u64 {
        match self {
            Self::Uniform { count, .. } => *count,
            Self::Dependent(members) => members.len() as u64,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn member(&self, index: u64) -> Option<&ConcreteType> {
        match self {
            Self::Uniform { element, count } => (index < *count).then_some(element),
            Self::Dependent(members) => members.get(usize::try_from(index).ok()?),
        }
    }

    /// The byte offset of member `index` (at most the member count) in the family's encoding, or
    /// `None` when a member before it has no size.
    pub fn member_offset(&self, index: u64) -> Option<u64> {
        match self {
            Self::Uniform { element, count } => {
                if index > *count {
                    return None;
                }
                index.checked_mul(element.size()?)
            }
            Self::Dependent(members) => members
                .get(..usize::try_from(index).ok()?)?
                .iter()
                .try_fold(0u64, |total, m| total.checked_add(m.size()?)),
        }
    }
}

impl ConcreteType {
    /// The number of bytes of the value's encoding, or `None` for a typed blob.
    pub fn size(&self) -> Option<u64> {
        match self {
            Self::Buffer { bytes, .. } => Some(*bytes),
            Self::Family(ConcreteFamily::Uniform { element, count }) => {
                element.size()?.checked_mul(*count)
            }
            Self::Family(ConcreteFamily::Dependent(members)) => {
                members.iter().try_fold(0u64, |total, m| total.checked_add(m.size()?))
            }
            Self::Int { words } => Some(8 * u64::from(*words)),
            Self::Bool => Some(8),
            Self::TypedBlob { .. } => None,
        }
    }

    pub fn family(&self) -> Option<&ConcreteFamily> {
        match self {
            Self::Family(family) => Some(family),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Bindings;

    #[test]
    fn test_dependent_family_resolves_member_by_member() {
        let element = CoreType::bytes(IntExpr::LoopIndex(MEMBER_INDEX) * 8 + 8);
        let ty = CoreType::family(element, "n")
            .resolve(&Env::new(Bindings::from([("n".into(), 3)])))
            .unwrap();
        let family = ty.family().unwrap();
        assert_eq!(family.len(), 3);
        assert_eq!(family.member_offset(2), Some(8 + 16));
        assert_eq!(ty.size(), Some(8 + 16 + 24));
    }

    #[test]
    fn test_uniform_family_from_equal_members() {
        let family = ConcreteFamily::from_members(vec![ConcreteType::Bool; 4]);
        assert!(matches!(family, ConcreteFamily::Uniform { count: 4, .. }));
    }
}
