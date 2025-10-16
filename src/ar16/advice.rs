use crate::{
    ar16::{AR16Encoding, AR16PublicKey},
    circuit::gate::GateId,
    poly::Poly,
};
use std::collections::HashMap;

/// Logical label for advice encodings within a level set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdviceLabel(pub String);

impl AdviceLabel {
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }
}

impl From<&str> for AdviceLabel {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Descriptor identifying level-specific advice requirements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AdviceRole {
    /// Encoding of `E_k(s^2)`.
    SecretSquare,
    /// Encoding of `E_k(s)`.
    SecretPlain,
    /// Encoding of `E_k(c_{k-1}·s)` for a given gate.
    LiftedCipher { gate: GateId },
    /// Encoding of `E_k(c_{k-1})` replicated at this level.
    LiftedPlain { gate: GateId },
    /// Custom label originating from higher-level algorithms.
    Custom(AdviceLabel),
}

/// Advice set supplied by the encryptor for a single circuit level.
#[derive(Debug, Clone, Default)]
pub struct AdviceSet<P: Poly> {
    level: usize,
    encodings: HashMap<AdviceRole, AR16Encoding<P>>,
    public_keys: HashMap<AdviceRole, AR16PublicKey<P>>,
}

impl<P: Poly> AdviceSet<P> {
    pub fn new(level: usize) -> Self {
        Self { level, encodings: HashMap::new(), public_keys: HashMap::new() }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn insert_encoding(&mut self, role: AdviceRole, encoding: AR16Encoding<P>) {
        self.encodings.insert(role, encoding);
    }

    pub fn insert_public_key(&mut self, role: AdviceRole, key: AR16PublicKey<P>) {
        self.public_keys.insert(role, key);
    }

    pub fn get_encoding(&self, role: &AdviceRole) -> Option<&AR16Encoding<P>> {
        self.encodings.get(role)
    }

    pub fn get_public_key(&self, role: &AdviceRole) -> Option<&AR16PublicKey<P>> {
        self.public_keys.get(role)
    }
}
