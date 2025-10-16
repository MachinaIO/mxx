//! AR16 public-key and ciphertext evaluation helpers following ePrint 2016/361 §5.
//!
//! This module provides the data structures and orchestration logic for the `EvalPK`
//! and `EvalCT` algorithms described in Section 5. Concrete arithmetic is expressed in
//! terms of the repository’s `PolyMatrix` abstraction so that encodings keep the usual
//! Regev form `(a, b)` without embedding plaintext or noise information.
//!
//! ## Layout
//! * `advice` – strongly typed access to level-specific advice encodings and public keys.
//! * `state` – `EvalContext` that caches memoized gate evaluations together with advice.
//! * `eval_pk` / `eval_encoding` – recursive evaluators that follow the paper’s recurrence,
//!   including multiplication layers realised via the quadratic method.
//! * `quadratic` – helper routines implementing Equations (5.6)–(5.24).
//!
//! ## Usage
//! ```ignore
//! # use mxx::ar16::{eval_encoding, eval_public_key, EvalContext};
//! # use mxx::ar16::advice::AdviceSet;
//! # use mxx::ar16::{AR16Encoding, AR16PublicKey, Ar16Error};
//! # use mxx::circuit::PolyCircuit;
//! # use mxx::poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly};
//! # use mxx::matrix::dcrt::DCRTPolyMatrix;
//! # use num_bigint::BigUint;
//! # use std::collections::HashMap;
//! let circuit = PolyCircuit::<DCRTPoly>::new();
//! let advice_sets: Vec<AdviceSet<DCRTPolyMatrix>> = vec![];
//! let mut ctx = EvalContext::new(advice_sets);
//! let input_keys: HashMap<_, AR16PublicKey<DCRTPolyMatrix>> = HashMap::new();
//! let input_encodings: HashMap<_, AR16Encoding<DCRTPolyMatrix>> = HashMap::new();
//! let pk = eval_public_key(&circuit, &mut ctx, &input_keys)?;
//! let ct = eval_encoding(&circuit, &pk, &mut ctx, &input_encodings)?;
//! # Ok::<(), Ar16Error>(())
//! ```
//! See `agents/ar16.md` for additional implementation guidance.

pub mod advice;
pub mod error;
pub mod eval_encoding;
pub mod eval_pk;
pub mod quadratic;
pub mod state;

pub use error::Ar16Error;
pub use eval_encoding::eval_encoding;
pub use eval_pk::{EvalPkOutput, eval_public_key};
pub use state::EvalContext;

use crate::poly::Poly;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// AR16 public key container `a`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AR16PublicKey<P: Poly> {
    pub a: P,
}

impl<P: Poly> AR16PublicKey<P> {
    pub fn new(a: P) -> Self {
        Self { a }
    }

    pub fn scale(&self, factor: &P) -> Self {
        Self { a: self.a.clone() * factor.clone() }
    }
}

impl<P: Poly> Add for AR16PublicKey<P> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { a: self.a + rhs.a }
    }
}

impl<'a, P: Poly> Add<&'a AR16PublicKey<P>> for AR16PublicKey<P> {
    type Output = Self;
    fn add(self, rhs: &'a AR16PublicKey<P>) -> Self::Output {
        Self { a: self.a + rhs.a.clone() }
    }
}

impl<P: Poly> AddAssign<&AR16PublicKey<P>> for AR16PublicKey<P> {
    fn add_assign(&mut self, rhs: &AR16PublicKey<P>) {
        self.a = self.a.clone() + rhs.a.clone();
    }
}

impl<P: Poly> Sub for AR16PublicKey<P> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self { a: self.a - rhs.a }
    }
}

impl<'a, P: Poly> Sub<&'a AR16PublicKey<P>> for AR16PublicKey<P> {
    type Output = Self;
    fn sub(self, rhs: &'a AR16PublicKey<P>) -> Self::Output {
        Self { a: self.a - rhs.a.clone() }
    }
}

impl<P: Poly> SubAssign<&AR16PublicKey<P>> for AR16PublicKey<P> {
    fn sub_assign(&mut self, rhs: &AR16PublicKey<P>) {
        self.a = self.a.clone() - rhs.a.clone();
    }
}

/// AR16 ciphertext encoding `(a, b)` in Regev form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AR16Encoding<P: Poly> {
    pub a: AR16PublicKey<P>,
    pub b: P,
}

impl<P: Poly> AR16Encoding<P> {
    pub fn new(a: AR16PublicKey<P>, b: P) -> Self {
        Self { a, b }
    }

    pub fn scale(&self, factor: &P) -> Self {
        Self { a: self.a.scale(factor), b: self.b.clone() * factor.clone() }
    }

    pub fn evaluate(&self, secret: &P) -> P {
        self.b.clone() - self.a.a.clone() * secret.clone()
    }
}

impl<P: Poly> Add for AR16Encoding<P> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}

impl<'a, P: Poly> Add<&'a AR16Encoding<P>> for AR16Encoding<P> {
    type Output = Self;
    fn add(self, rhs: &'a AR16Encoding<P>) -> Self::Output {
        Self { a: self.a + rhs.a.clone(), b: self.b + rhs.b.clone() }
    }
}

impl<P: Poly> AddAssign<&AR16Encoding<P>> for AR16Encoding<P> {
    fn add_assign(&mut self, rhs: &AR16Encoding<P>) {
        self.a += &rhs.a;
        self.b = self.b.clone() + rhs.b.clone();
    }
}

impl<P: Poly> Sub for AR16Encoding<P> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl<'a, P: Poly> Sub<&'a AR16Encoding<P>> for AR16Encoding<P> {
    type Output = Self;
    fn sub(self, rhs: &'a AR16Encoding<P>) -> Self::Output {
        Self { a: self.a - rhs.a.clone(), b: self.b - rhs.b.clone() }
    }
}

impl<P: Poly> SubAssign<&AR16Encoding<P>> for AR16Encoding<P> {
    fn sub_assign(&mut self, rhs: &AR16Encoding<P>) {
        self.a -= &rhs.a;
        self.b = self.b.clone() - rhs.b.clone();
    }
}
