//! What the refresh composition ([`super::compose`]) computes on: the keys alone (the
//! obfuscator, which decodes the keys whose preimages it samples) or the rows with their keys
//! (the evaluator). Both carriers build every key with the same `mxx-exponent-lut` key helpers
//! (`RnsBggParameters::retarget_key`, `RnsBggParameters::mul_public_key`); a row carrier adds
//! the row half of each operation (`retarget_encoding`, `mul_public_rows`, which call those key
//! helpers on the same arguments). The key subexpressions of the two graphs are therefore
//! identical, which the agreement of the obfuscator's preimages with the evaluator's decoders
//! rests on.
//!
//! A carrier is one encoding at the root modulus `Q_0`: its key `K = (K_(kj))` and, for a row
//! carrier, its row `c = (c_0, c_1)`. Its *component* `j` lists the matrices that a
//! per-component operation (a public product or a refresh combination, spec §2.1) acts on alike:
//! `[K_(0j), K_(1j)]`, followed by `c_j` for a row carrier.

use mxx_dsl::{GraphValue, Mat};
use mxx_exponent_lut::rns_bgg::{
    RnsBggError, RnsBggForm, RnsBggMatrix, RnsBggParameters, RnsBggRow,
};
use mxx_ir_core::{IntExpr, node::IndexRange};

/// An encoding the refresh composition re-targets, multiplies and recombines.
pub trait Carrier: GraphValue {
    /// A public encoding: the carrier and, for a row carrier, its clear plaintext.
    type Public: GraphValue;

    /// The matrices of a component ([`Self::component`]).
    const WIDTH: usize;

    /// The component entries a decoder of this carrier produces: the decoded key rows
    /// `[K_0, K_1]` for a key, the decoded row `d` for a row.
    const DECODED: &'static [usize];

    /// `Gamma` of this encoding in `form` at `level` to the form `target` (a row in the ring of
    /// `target_level`) at `target_level` (spec §2.1).
    fn retarget(
        self,
        chain: &RnsBggParameters,
        from: (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self, RnsBggError>;

    /// [`Self::retarget`] of a public encoding, whose plaintext it keeps.
    fn retarget_public(
        public: Self::Public,
        chain: &RnsBggParameters,
        from: (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self::Public, RnsBggError>;

    /// `MulPub(P, c)` at `level` of the public encoding `public` by this encoding.
    fn mul_public(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        public: &Self::Public,
    ) -> Result<Self, RnsBggError>;

    /// `MulPub(P, Q)` at `level` of two public encodings: a public encoding of the product
    /// `alpha_P alpha_Q`.
    fn public_product(
        chain: &RnsBggParameters,
        level: usize,
        left: &Self::Public,
        right: &Self::Public,
    ) -> Result<Self::Public, RnsBggError>;

    /// The carrier of a public encoding.
    fn of_public(public: Self::Public) -> Self;

    /// The columns `range` of every matrix.
    fn slice_columns(&self, range: IndexRange) -> Self;

    /// Component `j` (`0` or `1`): `[K_(0j), K_(1j)]`, then `c_j` for a row carrier.
    fn component(&self, j: usize) -> Vec<Mat>;

    /// The carrier of the components `first` (`j = 0`) and `second` (`j = 1`).
    fn from_components(first: Vec<Mat>, second: Vec<Mat>) -> Self;

    /// The zero encoding (zero row and key) of `columns` columns in `ring`.
    fn zero(ring: &mxx_dsl::Ring, columns: usize) -> Self {
        let zero = vec![ring.zero((1, columns)); Self::WIDTH];
        Self::from_components(zero.clone(), zero)
    }
}

/// The key `K` of an encoding, without its row.
pub type Key = RnsBggMatrix;

/// The row `c` of an encoding and its key `K`.
pub type Row = (RnsBggRow, RnsBggMatrix);

/// A public row encoding with its clear plaintext `alpha`.
pub type PublicRow = (RnsBggRow, RnsBggMatrix, Mat);

fn slice(matrix: &Mat, range: &IndexRange) -> Mat {
    matrix.clone().slice(None, Some(range.clone()))
}

fn slice_key(key: &Key, range: &IndexRange) -> Key {
    (
        (slice(&key.0.0, range), slice(&key.0.1, range)),
        (slice(&key.1.0, range), slice(&key.1.1, range)),
    )
}

fn key_component(key: &Key, j: usize) -> Vec<Mat> {
    match j {
        0 => vec![key.0.0.clone(), key.1.0.clone()],
        _ => vec![key.0.1.clone(), key.1.1.clone()],
    }
}

fn key_of_components(first: &[Mat], second: &[Mat]) -> Key {
    ((first[0].clone(), second[0].clone()), (first[1].clone(), second[1].clone()))
}

impl Carrier for Key {
    type Public = Key;
    const WIDTH: usize = 2;
    const DECODED: &'static [usize] = &[0, 1];

    fn retarget(
        self,
        chain: &RnsBggParameters,
        (level, form): (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self, RnsBggError> {
        chain.retarget_key(level, form, target_level, target, self)
    }

    fn retarget_public(
        public: Self::Public,
        chain: &RnsBggParameters,
        from: (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self::Public, RnsBggError> {
        public.retarget(chain, from, target_level, target)
    }

    fn mul_public(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        public: &Self::Public,
    ) -> Result<Self, RnsBggError> {
        chain.mul_public_key(level, public, self)
    }

    fn public_product(
        chain: &RnsBggParameters,
        level: usize,
        left: &Self::Public,
        right: &Self::Public,
    ) -> Result<Self::Public, RnsBggError> {
        chain.mul_public_key(level, left, right)
    }

    fn of_public(public: Self::Public) -> Self {
        public
    }

    fn slice_columns(&self, range: IndexRange) -> Self {
        slice_key(self, &range)
    }

    fn component(&self, j: usize) -> Vec<Mat> {
        key_component(self, j)
    }

    fn from_components(first: Vec<Mat>, second: Vec<Mat>) -> Self {
        key_of_components(&first, &second)
    }
}

impl Carrier for Row {
    type Public = PublicRow;
    const WIDTH: usize = 3;
    const DECODED: &'static [usize] = &[2];

    fn retarget(
        self,
        chain: &RnsBggParameters,
        (level, form): (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self, RnsBggError> {
        chain.retarget_encoding(level, form, target_level, target, self)
    }

    fn retarget_public(
        (row, key, alpha): Self::Public,
        chain: &RnsBggParameters,
        from: (usize, RnsBggForm),
        target_level: usize,
        target: Mat,
    ) -> Result<Self::Public, RnsBggError> {
        let (row, key) = (row, key).retarget(chain, from, target_level, target)?;
        Ok((row, key, alpha))
    }

    fn mul_public(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        (row, key, alpha): &Self::Public,
    ) -> Result<Self, RnsBggError> {
        chain.mul_public_rows(level, (row, key, alpha), (&self.0, &self.1))
    }

    fn public_product(
        chain: &RnsBggParameters,
        level: usize,
        left: &Self::Public,
        right: &Self::Public,
    ) -> Result<Self::Public, RnsBggError> {
        let (row, key) = (right.0.clone(), right.1.clone()).mul_public(chain, level, left)?;
        Ok((row, key, left.2.clone() * right.2.clone()))
    }

    fn of_public((row, key, _): Self::Public) -> Self {
        (row, key)
    }

    fn slice_columns(&self, range: IndexRange) -> Self {
        ((slice(&self.0.0, &range), slice(&self.0.1, &range)), slice_key(&self.1, &range))
    }

    fn component(&self, j: usize) -> Vec<Mat> {
        let mut component = key_component(&self.1, j);
        component.push(if j == 0 { self.0.0.clone() } else { self.0.1.clone() });
        component
    }

    fn from_components(first: Vec<Mat>, second: Vec<Mat>) -> Self {
        ((first[2].clone(), second[2].clone()), key_of_components(&first, &second))
    }
}

/// The column range `[start w, (start + blocks) w)` of `blocks` blocks of `w` columns from
/// block `start` (an index expression).
pub fn block_columns(start: IntExpr, blocks: usize, width: usize) -> IndexRange {
    let start = IntExpr::Mul(Box::new(start), Box::new(IntExpr::constant(width))).canonicalize();
    let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(blocks * width)))
        .canonicalize();
    IndexRange { start, end }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;
    use mxx_dsl::DslContext;
    use mxx_exponent_lut::rns_bgg::RnsBggRoot;

    /// The key of a row carrier's re-targeting and public product is the very subexpression the
    /// key carrier builds: both graphs freeze to the same canonical key outputs.
    #[test]
    fn row_and_key_carriers_share_key_subexpressions() {
        let (parameters, _) = toy();
        let root = parameters.root(rand::random()).unwrap();
        let chain = parameters.chain(&root).unwrap();
        let ring = root.ring();
        let hash_key = root.hash_key_input();
        let key = |tag: &str| root.key(hash_key.clone(), RnsBggRoot::input_tag(tag, None));
        let row = |name: &str| {
            (
                ring.input(format!("{name}.0"), (1, root.columns())),
                ring.input(format!("{name}.1"), (1, root.columns())),
            )
        };
        let level = parameters.decoding_level();
        let form = chain.form(level, RnsBggForm::State).unwrap();
        let public_key = key("public");
        let private_key = key("private");
        let keys = private_key
            .clone()
            .retarget(&chain, (0, RnsBggForm::State), level, form.clone())
            .unwrap();
        let keys = keys
            .mul_public(
                &chain,
                level,
                &Key::retarget_public(
                    public_key.clone(),
                    &chain,
                    (0, RnsBggForm::Public),
                    level,
                    chain.form(level, RnsBggForm::Public).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let rows = (row("private"), private_key)
            .retarget(&chain, (0, RnsBggForm::State), level, form)
            .unwrap();
        let public = Row::retarget_public(
            (row("public"), public_key, ring.identity(1)),
            &chain,
            (0, RnsBggForm::Public),
            level,
            chain.form(level, RnsBggForm::Public).unwrap(),
        )
        .unwrap();
        let rows = rows.mul_public(&chain, level, &public).unwrap();
        let freeze = |key: Key| {
            DslContext::new("carrier-key").output("key", key).unwrap().build().unwrap().graph
        };
        let (from_keys, from_rows) = (freeze(keys), freeze(rows.1));
        assert_eq!(
            mxx_ir_core::encoding::spec_hash(&from_keys, &Default::default()).unwrap(),
            mxx_ir_core::encoding::spec_hash(&from_rows, &Default::default()).unwrap()
        );
    }
}
