use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::poly::Poly;

/// Circuit level tag used to index modulus towers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(pub u32);

impl Level {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn to_index(self) -> usize {
        self.0 as usize
    }

    pub fn next(self) -> Self {
        Level(self.0 + 1)
    }
}

impl From<usize> for Level {
    fn from(value: usize) -> Self {
        Level(value as u32)
    }
}

impl From<u32> for Level {
    fn from(value: u32) -> Self {
        Level(value)
    }
}

impl From<Level> for usize {
    fn from(level: Level) -> Self {
        level.to_index()
    }
}

/// Functional public key / label for an AR16 encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AR16PublicKey<P: Poly> {
    level: Level,
    u: P,
}

impl<P: Poly> AR16PublicKey<P> {
    pub fn new(level: Level, u: P) -> Self {
        Self { level, u }
    }

    pub fn level(&self) -> Level {
        self.level
    }

    pub fn u(&self) -> &P {
        &self.u
    }

    pub fn into_inner(self) -> P {
        self.u
    }

    pub fn scale(&self, factor: &P) -> Self {
        Self { level: self.level, u: self.u.clone() * factor.clone() }
    }
}

impl<P: Poly> Add for AR16PublicKey<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.level, rhs.level);
        Self { level: self.level, u: self.u + rhs.u }
    }
}

impl<P: Poly> AddAssign<&AR16PublicKey<P>> for AR16PublicKey<P> {
    fn add_assign(&mut self, rhs: &AR16PublicKey<P>) {
        assert_eq!(self.level, rhs.level);
        self.u = self.u.clone() + rhs.u.clone();
    }
}

impl<P: Poly> Sub for AR16PublicKey<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.level, rhs.level);
        Self { level: self.level, u: self.u - rhs.u }
    }
}

impl<P: Poly> SubAssign<&AR16PublicKey<P>> for AR16PublicKey<P> {
    fn sub_assign(&mut self, rhs: &AR16PublicKey<P>) {
        assert_eq!(self.level, rhs.level);
        self.u = self.u.clone() - rhs.u.clone();
    }
}

/// AR16 ciphertext encoding `c = u · s + noise + y`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AR16Encoding<P: Poly> {
    label: AR16PublicKey<P>,
    c: P,
}

impl<P: Poly> AR16Encoding<P> {
    pub fn new(label: AR16PublicKey<P>, c: P) -> Self {
        Self { label, c }
    }

    pub fn level(&self) -> Level {
        self.label.level()
    }

    pub fn label(&self) -> &AR16PublicKey<P> {
        &self.label
    }

    pub fn body(&self) -> &P {
        &self.c
    }

    pub fn into_parts(self) -> (AR16PublicKey<P>, P) {
        (self.label, self.c)
    }

    pub fn scale(&self, factor: &P) -> Self {
        Self { label: self.label.scale(factor), c: self.c.clone() * factor.clone() }
    }

    pub fn evaluate(&self, secret: &P) -> P {
        self.c.clone() - self.label.u().clone() * secret.clone()
    }
}

impl<P: Poly> Add for AR16Encoding<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.level(), rhs.level());
        Self { label: self.label + rhs.label, c: self.c + rhs.c }
    }
}

impl<P: Poly> AddAssign<&AR16Encoding<P>> for AR16Encoding<P> {
    fn add_assign(&mut self, rhs: &AR16Encoding<P>) {
        assert_eq!(self.level(), rhs.level());
        self.label += rhs.label();
        self.c = self.c.clone() + rhs.c.clone();
    }
}

impl<P: Poly> Sub for AR16Encoding<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.level(), rhs.level());
        Self { label: self.label - rhs.label, c: self.c - rhs.c }
    }
}

impl<P: Poly> SubAssign<&AR16Encoding<P>> for AR16Encoding<P> {
    fn sub_assign(&mut self, rhs: &AR16Encoding<P>) {
        assert_eq!(self.level(), rhs.level());
        self.label -= rhs.label();
        self.c = self.c.clone() - rhs.c.clone();
    }
}
