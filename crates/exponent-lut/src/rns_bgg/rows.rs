//! Operations on the rows of a Static chain outside a Program: the re-targeting `Gamma` of an
//! encoding to another form and level, and the public multiplication of a general row. Key-only
//! operations are shared with the full encodings, so their key subgraphs are identical.
//!
//! A Program's lowered output is an encoding in the form its carrier names: the state form
//! `Delta_l G^st_l` of a private wire at level `l` (a raw coefficient wire is one too), or the
//! public gadget `G_(Q_l)` of a public wire. A protocol that combines outputs of several
//! Programs, or rows that are not Program wires, moves them to one form by `Gamma` and
//! multiplies them by public encodings here.

use super::*;
use mxx_dsl::GraphValue;
use mxx_ir_core::node::IndexRange;

impl RnsBggParameters {
    /// The level whose gadget the form `form` of `level` is a multiple of: `level + 1` for the
    /// state form (whose gadget is the public gadget of the next level), `level` for the
    /// public form.
    fn gadget_level(level: usize, form: RnsBggForm) -> usize {
        match form {
            RnsBggForm::State => level + 1,
            RnsBggForm::Public => level,
        }
    }

    /// `Gamma` of an encoding (`row` and its key `key`) in `form` at `level` to the target form
    /// `target` (a `1 x w` row in the ring of `target_level`) at `target_level`:
    ///
    /// `Gamma(c) = ModDown_(Q_l -> Q_t)(c G_N^-1((N / Q_t) target))`
    ///
    /// for the gadget modulus `N` of the source form (`Q_(l+1)` for the state form, `Q_l` for the
    /// public form), whose source form is `(Q_l / N) G_N`. The target `T` of the encoding is kept
    /// and its form becomes `target`: `(Q_l / N) G_N G_N^-1((N / Q_t) target) = (Q_l / Q_t)
    /// target`, which the ModDown divides exactly (`Rel.modDown`). It needs `Q_t | N`, that is
    /// `target_level` at or below the gadget level. The key's blocks are acted on alike, and the
    /// error becomes `(e D - r_c + s_bar r_0 + r_1) / (Q_l / Q_t)` for the digits `D` and the
    /// rounding remainders of the row and the key blocks (no rounding when `Q_t = Q_l`).
    pub fn retarget_encoding(
        &self,
        level: usize,
        form: RnsBggForm,
        target_level: usize,
        target: Mat,
        (row, key): (RnsBggRow, RnsBggMatrix),
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let (digits, target_ring) = self.retarget_action(level, form, target_level, target)?;
        let row = map_row(row, |entry| entry.mul_small_rhs(digits.clone()));
        let key = map_matrix(key, |entry| entry.mul_small_rhs(digits.clone()));
        match target_ring {
            Some(ring) => {
                Ok((checked_mod_down_encoding(row, &ring)?, checked_mod_down_public(key, &ring)?))
            }
            None => Ok((row, key)),
        }
    }

    /// The right-action digits of `Gamma` from `form` at `level` to `target` at
    /// `target_level`, and the destination ring when a rounded modulus drop is needed.
    /// The bounded compact digits live in the source ring; the target modulus must divide the
    /// form's gadget. They retain their coefficient bound for `mul_small_rhs` consumers.
    pub fn retarget_action(
        &self,
        level: usize,
        form: RnsBggForm,
        target_level: usize,
        target: Mat,
    ) -> Result<(SmallMatrix, Option<Ring>), RnsBggError> {
        let gadget_level = Self::gadget_level(level, form);
        let gadget = self.level(gadget_level)?;
        let target_parameters = self.level(target_level)?;
        let gadget_modulus: BigUint = gadget.modulus().as_ref().clone();
        let target_modulus: BigUint = target_parameters.modulus().as_ref().clone();
        if target_level < gadget_level || !(&gadget_modulus % &target_modulus).is_zero() {
            return Err(RnsBggError::Parameters("a re-targeting needs a divisor of its gadget"));
        }
        let gadget_ring = self.ring(gadget_level)?;
        let ratio = &gadget_modulus / &target_modulus;
        let scaled = target.centered_rebase(&gadget_ring) *
            gadget_ring.polynomial([IntExpr::from(BigInt::from(ratio))]);
        let digits = scaled
            .decompose(public_gadget_base(gadget.base_bits())?, gadget.modulus_digits())
            .centered_rebase(&self.ring(level)?);
        let target_ring = if self.level(level)?.modulus() == target_parameters.modulus() {
            None
        } else {
            Some(self.ring(target_level)?)
        };
        Ok((digits, target_ring))
    }

    /// The key of [`Self::retarget_encoding`], computed without encoding rows or secrets.
    /// Every block uses [`Self::retarget_action`] and the same rounded modulus drop as a row.
    pub fn retarget_key(
        &self,
        level: usize,
        form: RnsBggForm,
        target_level: usize,
        target: Mat,
        key: RnsBggMatrix,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let (digits, target_ring) = self.retarget_action(level, form, target_level, target)?;
        let key = map_matrix(key, |entry| entry.mul_small_rhs(digits.clone()));
        match target_ring {
            Some(ring) => checked_mod_down_public(key, &ring),
            None => Ok(key),
        }
    }

    /// The public multiplication `MulPub(P, c)` at `level` of the public encoding `public`
    /// (row `P`, key `K^P`, in the public form of `level`) of the clear plaintext `alpha` by the
    /// general row `c` with key `K^c` (any form at `level`, `w` columns per component): for each
    /// component `j`, with `D_j = [G^-1(K^c_(0j)); G^-1(K^c_(1j))]` under the public gadget of
    /// `level`,
    ///
    /// `out_j = [P_0 | P_1] D_j + alpha c_j`, `K^out_(kj) = [K^P_(k0) | K^P_(k1)] D_j`,
    ///
    /// an encoding of `alpha T_j` in the form of `c` for every target `T` of `c` (the general
    /// component relation `mul_public` of `ExponentPublic.lean`: the mask term
    /// `alpha (s_bar K^c_(0j) + K^c_(1j))` of the public product cancels against `alpha c_j`).
    pub fn mul_public_rows(
        &self,
        level: usize,
        public: (&RnsBggRow, &RnsBggMatrix, &Mat),
        encoding: (&RnsBggRow, &RnsBggMatrix),
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        self.mul_public_rows_scaled(level, public, encoding, 0)
    }

    /// [`Self::mul_public_rows`] of a public encoding whose plaintext `alpha` is `2^k` times a
    /// signed monomial (`k = scale_log2`, a contract of the caller): `D_j` decomposes `2^-k
    /// K^c_(.j)` and the row term is `2^-k alpha c_j`, so the output encodes `2^-k alpha T_j`
    /// and the error of `c` is multiplied by the signed monomial `2^-k alpha` only (the carrier
    /// twin of `ExponentLutProgramBuilder::mul_public_scaled`).
    pub fn mul_public_rows_scaled(
        &self,
        level: usize,
        (public_row, public_key, alpha): (&RnsBggRow, &RnsBggMatrix, &Mat),
        (row, key): (&RnsBggRow, &RnsBggMatrix),
        scale_log2: u32,
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let digits = self.mul_public_digits(level, key, scale_log2)?;
        let output_key = (
            Self::mul_public_row_action(&public_key.0, &digits),
            Self::mul_public_row_action(&public_key.1, &digits),
        );
        let alpha = match self.inverse_scale(level, scale_log2)? {
            Some(weight) => weight * alpha.clone(),
            None => alpha.clone(),
        };
        let product = Self::mul_public_row_action(public_row, &digits);
        Ok((
            (product.0 + alpha.clone() * row.0.clone(), product.1 + alpha * row.1.clone()),
            output_key,
        ))
    }

    /// The key of [`Self::mul_public_rows`], depending only on the two public keys.
    /// Neither the public plaintext nor either encoding row enters this subgraph.
    pub fn mul_public_key(
        &self,
        level: usize,
        public_key: &RnsBggMatrix,
        key: &RnsBggMatrix,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        self.mul_public_key_scaled(level, public_key, key, 0)
    }

    /// The key of [`Self::mul_public_rows_scaled`].
    pub fn mul_public_key_scaled(
        &self,
        level: usize,
        public_key: &RnsBggMatrix,
        key: &RnsBggMatrix,
        scale_log2: u32,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let digits = self.mul_public_digits(level, key, scale_log2)?;
        Ok((
            Self::mul_public_row_action(&public_key.0, &digits),
            Self::mul_public_row_action(&public_key.1, &digits),
        ))
    }

    /// Decomposes each key column (times `2^-scale_log2`) once for reuse across the two
    /// public-key row contractions.
    fn mul_public_digits(
        &self,
        level: usize,
        key: &RnsBggMatrix,
        scale_log2: u32,
    ) -> Result<RnsBggDigits, RnsBggError> {
        let scaled;
        let key = match self.inverse_scale(level, scale_log2)? {
            Some(weight) => {
                scaled = super::map_matrix(key.clone(), |entry| weight.clone() * entry);
                &scaled
            }
            None => key,
        };
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        let digits = parameters.modulus_digits();
        Ok((
            (key.0.0.clone().decompose(base, digits), key.1.0.clone().decompose(base, digits)),
            (key.0.1.clone().decompose(base, digits), key.1.1.clone().decompose(base, digits)),
        ))
    }

    /// Contracts a public row with the gadget-decomposed key columns, without new decompositions.
    fn mul_public_row_action(public_row: &RnsBggRow, digits: &RnsBggDigits) -> RnsBggRow {
        let product = |column: &(Preimage, Preimage)| {
            public_row.0.clone().mul_small_rhs(column.0.clone()) +
                public_row.1.clone().mul_small_rhs(column.1.clone())
        };
        (product(&digits.0), product(&digits.1))
    }

    /// The integer entries of the form `form` of `level` ([`Self::form`]) modulo `Q_level`.
    ///
    /// The public gadget of a level is the exact CRT gadget: entry `t d + k` is `B^k` modulo
    /// tower `t` and zero modulo the other towers. The state form is `Delta_l` times the public
    /// gadget of level `l + 1` (centered there), or `Delta_l` alone on the last level.
    pub fn form_weights(
        &self,
        level: usize,
        form: RnsBggForm,
    ) -> Result<Vec<BigUint>, RnsBggError> {
        let modulus: BigUint = self.level(level)?.modulus().as_ref().clone();
        let gadget = |gadget_level: usize| -> Result<Vec<BigUint>, RnsBggError> {
            let parameters = self.level(gadget_level)?;
            let (moduli, _, _) = parameters.to_crt();
            let gadget_modulus: BigUint = parameters.modulus().as_ref().clone();
            let digits = parameters.modulus_digits() / moduli.len();
            let base = BigUint::one() << parameters.base_bits();
            let mut weights = Vec::with_capacity(parameters.modulus_digits());
            for tower in &moduli {
                let tower = BigUint::from(*tower);
                let cofactor = &gadget_modulus / &tower;
                let idempotent = (&cofactor % &tower)
                    .modinv(&tower)
                    .ok_or(RnsBggError::Parameters("CRT primes are not coprime"))? *
                    &cofactor;
                let mut power = BigUint::one();
                for _ in 0..digits {
                    weights.push(&idempotent * &power % &gadget_modulus);
                    power = power * &base % &tower;
                }
            }
            Ok(weights)
        };
        Ok(match form {
            RnsBggForm::Public => gadget(level)?,
            RnsBggForm::State => {
                let delta = BigInt::from(self.message_scale(level)?);
                let modulus = BigInt::from(modulus);
                match self.levels.get(level + 1) {
                    Some(next) => {
                        let next_modulus = BigInt::from(next.modulus().as_ref().clone());
                        gadget(level + 1)?
                            .into_iter()
                            .map(|weight| {
                                let weight = BigInt::from(weight);
                                let centered = if &weight * 2 > next_modulus {
                                    weight - &next_modulus
                                } else {
                                    weight
                                };
                                canonical_residue(&delta * centered, &modulus)
                            })
                            .collect()
                    }
                    None => vec![canonical_residue(delta, &modulus)],
                }
            }
        })
    }
}

/// `value` modulo `modulus` in `[0, modulus)`.
fn canonical_residue(value: BigInt, modulus: &BigInt) -> BigUint {
    let residue = value % modulus;
    let residue =
        if residue.sign() == num_bigint::Sign::Minus { residue + modulus } else { residue };
    residue.to_biguint().expect("a nonnegative residue")
}

/// An encoding a composition re-targets, multiplies and recombines.
pub trait Carrier: GraphValue {
    /// A public encoding: the carrier and, for a row carrier, its clear plaintext.
    type Public: GraphValue;

    /// The matrices of a component ([`Self::component`]).
    const WIDTH: usize;

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
    ) -> Result<Self, RnsBggError> {
        self.mul_public_scaled(chain, level, public, 0)
    }

    /// [`Self::mul_public`] of a public encoding whose plaintext is `2^scale_log2` times a
    /// signed monomial, with the scale removed inside the decomposition
    /// ([`RnsBggParameters::mul_public_rows_scaled`]).
    fn mul_public_scaled(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        public: &Self::Public,
        scale_log2: u32,
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

    fn mul_public_scaled(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        public: &Self::Public,
        scale_log2: u32,
    ) -> Result<Self, RnsBggError> {
        chain.mul_public_key_scaled(level, public, self, scale_log2)
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

    fn mul_public_scaled(
        &self,
        chain: &RnsBggParameters,
        level: usize,
        (row, key, alpha): &Self::Public,
        scale_log2: u32,
    ) -> Result<Self, RnsBggError> {
        chain.mul_public_rows_scaled(level, (row, key, alpha), (&self.0, &self.1), scale_log2)
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
    use crate::test_support::{execute_cpu, full, validate};
    use mxx_backends::{RuntimeValue, poly::Poly};
    use mxx_dsl::{BuiltGraph, DslContext};

    /// Small overridable parameters for key-action tests, with two genuine modulus drops.
    fn parameters() -> RnsBggParameters {
        let dimension = std::env::var("MXX_ROWS_TEST_RING_DIMENSION")
            .map(|value| value.parse().unwrap())
            .unwrap_or(8);
        let bits = std::env::var("MXX_ROWS_TEST_PRIME_BITS")
            .map(|value| value.parse().unwrap())
            .unwrap_or(17);
        let base_bits = std::env::var("MXX_ROWS_TEST_BASE_BITS")
            .map(|value| value.parse().unwrap())
            .unwrap_or(4);
        let pool = DCRTPolyParams::new(dimension, 3, bits, base_bits, None, None);
        let primes = pool.to_crt().0;
        let middle = pool.select_modulus(&(BigUint::from(primes[0]) * primes[1])).unwrap();
        let bottom = pool.select_modulus(&BigUint::from(primes[0])).unwrap();
        RnsBggParameters::new(vec![pool, middle, bottom], 0, 1).unwrap()
    }

    /// Hash-derived random rows and blocks, with a fresh runtime hash key for every run.
    fn random_row(ring: &Ring, tag: &str, columns: usize) -> RnsBggRow {
        let hash = ring.bytes_input("hash-key", 32);
        let entry = |component| {
            ring.hash_matrix(hash.clone(), format!("{tag}-{component}").into_bytes(), (1, columns))
        };
        (entry(0), entry(1))
    }

    /// Checks canonical key subgraph identity and executes both versions on random inputs.
    fn check_keys(parameters: &RnsBggParameters, expected: RnsBggMatrix, actual: RnsBggMatrix) {
        let build =
            |key| DslContext::new("key-action").output("key", key).unwrap().build().unwrap();
        let expected_graph = build(expected.clone());
        let actual_graph = build(actual.clone());
        assert_eq!(
            mxx_ir_core::encoding::spec_hash(&expected_graph.graph, &Default::default()).unwrap(),
            mxx_ir_core::encoding::spec_hash(&actual_graph.graph, &Default::default()).unwrap(),
            "key-only and full key subgraphs differ",
        );
        let difference = (
            (actual.0.0 - expected.0.0, actual.0.1 - expected.0.1),
            (actual.1.0 - expected.1.0, actual.1.1 - expected.1.1),
        );
        check_zero(parameters, &build(difference));
    }

    /// Runs a difference graph and checks every CRT polynomial coefficient is zero.
    fn check_zero(parameters: &RnsBggParameters, graph: &BuiltGraph) {
        let result = execute_cpu(
            &validate(graph),
            parameters.levels().to_vec(),
            BTreeMap::from([(
                "hash-key".to_owned(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().as_slice().into()),
            )]),
        );
        for (name, output) in &result.outputs {
            let matrix = full(output);
            let (rows, columns) = matrix.size();
            for row in 0..rows {
                for column in 0..columns {
                    assert!(
                        matrix.entry(row, column).coeffs_biguints().iter().all(Zero::is_zero),
                        "{name}"
                    );
                }
            }
        }
    }

    /// Public and state forms share key graphs, both with and without a modulus drop.
    #[test]
    fn retarget_key_matches_full_encoding() {
        let parameters = parameters();
        let ring = parameters.ring(0).unwrap();
        for (form, target_level) in [
            (RnsBggForm::Public, 0),
            (RnsBggForm::Public, 1),
            (RnsBggForm::Public, 2),
            (RnsBggForm::State, 1),
            (RnsBggForm::State, 2),
        ] {
            let columns = match form {
                RnsBggForm::Public => parameters.level(0).unwrap().modulus_digits(),
                RnsBggForm::State => parameters.component_columns(0).unwrap(),
            };
            let row = random_row(&ring, "row", columns);
            let key = (random_row(&ring, "top", columns), random_row(&ring, "bottom", columns));
            for target_form in [RnsBggForm::Public, RnsBggForm::State] {
                let target = parameters.form(target_level, target_form).unwrap();
                let expected = parameters
                    .retarget_key(0, form, target_level, target.clone(), key.clone())
                    .unwrap();
                let actual = parameters
                    .retarget_encoding(0, form, target_level, target, (row.clone(), key.clone()))
                    .unwrap();
                check_keys(&parameters, expected, actual.1);
            }
        }
    }

    /// Public multiplication keys never depend on rows or the public plaintext, for either form.
    #[test]
    fn mul_public_key_matches_full_rows() {
        let parameters = parameters();
        for level in 0..parameters.levels().len() {
            let ring = parameters.ring(level).unwrap();
            let public_columns = parameters.level(level).unwrap().modulus_digits();
            let public_row = random_row(&ring, "public-row", public_columns);
            let public_key = (
                random_row(&ring, "public-top", public_columns),
                random_row(&ring, "public-bottom", public_columns),
            );
            let alpha = random_row(&ring, "alpha", 1).0;
            for columns in [public_columns, parameters.component_columns(level).unwrap()] {
                let row = random_row(&ring, "row", columns);
                let key = (random_row(&ring, "top", columns), random_row(&ring, "bottom", columns));
                let expected = parameters.mul_public_key(level, &public_key, &key).unwrap();
                let actual = parameters
                    .mul_public_rows(level, (&public_row, &public_key, &alpha), (&row, &key))
                    .unwrap();
                check_keys(&parameters, expected, actual.1);
            }
        }
    }

    /// A scaled public product is the unscaled product of the plaintext `2^-k alpha` with the
    /// encoding whose key is `2^-k K^c` (and whose row is unchanged): the decomposition reads
    /// `2^-k K^c` and the row term is `2^-k alpha c`, for the row and the key.
    #[test]
    fn scaled_public_products_decompose_the_scaled_key() {
        let parameters = parameters();
        for level in 0..parameters.levels().len() {
            let ring = parameters.ring(level).unwrap();
            let public_columns = parameters.level(level).unwrap().modulus_digits();
            let public_row = random_row(&ring, "public-row", public_columns);
            let public_key = (
                random_row(&ring, "public-top", public_columns),
                random_row(&ring, "public-bottom", public_columns),
            );
            let alpha = random_row(&ring, "alpha", 1).0;
            let columns = parameters.component_columns(level).unwrap();
            let row = random_row(&ring, "row", columns);
            let key = (random_row(&ring, "top", columns), random_row(&ring, "bottom", columns));
            for scale in [1, 5] {
                let weight = parameters.inverse_scale(level, scale).unwrap().unwrap();
                let scaled_key =
                    super::super::map_matrix(key.clone(), |entry| weight.clone() * entry);
                let (actual_row, actual_key) = parameters
                    .mul_public_rows_scaled(
                        level,
                        (&public_row, &public_key, &alpha),
                        (&row, &key),
                        scale,
                    )
                    .unwrap();
                let (expected_row, expected_key) = parameters
                    .mul_public_rows(
                        level,
                        (&public_row, &public_key, &(weight.clone() * alpha.clone())),
                        (&row, &scaled_key),
                    )
                    .unwrap();
                check_keys(&parameters, expected_key, actual_key.clone());
                check_keys(
                    &parameters,
                    parameters.mul_public_key_scaled(level, &public_key, &key, scale).unwrap(),
                    actual_key,
                );
                let difference = (actual_row.0 - expected_row.0, actual_row.1 - expected_row.1);
                check_zero(
                    &parameters,
                    &DslContext::new("scaled-row")
                        .output("row-0", difference.0)
                        .unwrap()
                        .output("row-1", difference.1)
                        .unwrap()
                        .build()
                        .unwrap(),
                );
            }
        }
    }

    /// The extracted action preserves the divisor restriction of full re-targeting.
    #[test]
    fn retarget_action_rejects_upward_state_target() {
        let parameters = parameters();
        assert!(
            parameters
                .retarget_action(0, RnsBggForm::State, 0, parameters.public_gadget(0).unwrap())
                .is_err()
        );
    }
}
