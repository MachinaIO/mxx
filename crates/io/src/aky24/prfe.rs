//! Appendix B.1 private pseudorandom functional encryption.
//!
//! This module is intentionally private to `mxx-io`. It implements the exact
//! dual-use equations used by the AKY24 prMIFE cascade; it is not the disabled
//! public AKY24 FE API.

use super::{
    circuits::{
        CanonicalMatrix, CanonicalPolynomial, CanonicalResidue, PrescribedCoinLayout,
        PrescribedGaussianGroup,
    },
    config::{Aky24ConfigError, Aky24IoConfig},
};
use mxx_bgg::{
    AttributeEncodingCompiler, AttributeEncodingWire, AttributeEvaluationError,
    AttributeMatrixEvaluation,
};
use mxx_dsl::{Bool, Family, Int, Mat, Ring, Trapdoor};
#[cfg(test)]
use mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::GoldreichGraph;
use mxx_gadgets::{
    Poly,
    circuit::{
        ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
        PolyCircuit, PolyGateKind, PublicLookupLowering, SlotOperationLowering, gate::GateId,
        lower_circuit,
    },
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    node::{ConcatAxis, IndexRange},
};
use num_bigint::BigInt;
use thiserror::Error;

const SECRET_ROWS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PrivatePrfeLayerDimensions {
    pub plaintext_bits: usize,
    pub coefficient_bits: usize,
    pub x_columns: usize,
    pub attribute_count: usize,
    pub ciphertext_bits: usize,
    pub prescribed_tape_bits: usize,
}

impl PrivatePrfeLayerDimensions {
    pub fn new(config: &Aky24IoConfig, plaintext_bits: usize) -> Result<Self, Aky24ConfigError> {
        if plaintext_bits == 0 {
            return Err(Aky24ConfigError::NonPositiveParameter);
        }
        let coefficient_bits = config
            .modulus
            .to_biguint()
            .map(|value| value.bits() as usize)
            .filter(|bits| *bits > 0)
            .ok_or(Aky24ConfigError::NonPositiveParameter)?;
        let gsw_columns = SECRET_ROWS
            .checked_mul(config.digit_count)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let encrypted_bits = plaintext_bits
            .checked_add(config.security_parameter_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let x_columns = gsw_columns
            .checked_mul(encrypted_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let attribute_bits = SECRET_ROWS
            .checked_mul(x_columns)
            .and_then(|count| count.checked_mul(config.ring_dimension))
            .and_then(|count| count.checked_mul(coefficient_bits))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let attribute_count =
            attribute_bits.checked_add(1).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let c_b_polynomials = SECRET_ROWS
            .checked_mul(config.digit_count + 2)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let x_polynomials =
            SECRET_ROWS.checked_mul(x_columns).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let attribute_polynomials = attribute_count
            .checked_mul(gsw_columns)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let ciphertext_bits = c_b_polynomials
            .checked_add(x_polynomials)
            .and_then(|count| count.checked_add(attribute_polynomials))
            .and_then(|count| count.checked_mul(config.ring_dimension))
            .and_then(|count| count.checked_mul(coefficient_bits))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;

        let r_bits = gsw_columns
            .checked_mul(x_columns)
            .and_then(|count| count.checked_mul(config.ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let binary_tape = config
            .security_parameter_bits
            .checked_add(r_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let uniform_coefficients = (SECRET_ROWS - 1)
            .checked_mul(gsw_columns)
            .and_then(|count| count.checked_mul(config.ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let uniform_sample_bits = coefficient_bits
            .checked_add(config.uniform_statistical_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let uniform_tape = uniform_coefficients
            .checked_mul(uniform_sample_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let b_error_coefficients = c_b_polynomials
            .checked_mul(config.ring_dimension)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let fhe_error_coefficients = gsw_columns
            .checked_mul(config.ring_dimension)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let attribute_error_coefficients = attribute_polynomials
            .checked_mul(config.ring_dimension)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let gaussian_coefficients = config
            .ring_dimension
            .checked_add(b_error_coefficients)
            .and_then(|count| count.checked_add(fhe_error_coefficients))
            .and_then(|count| count.checked_add(attribute_error_coefficients))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let gaussian_tape = gaussian_coefficients
            .checked_mul(config.gaussian_sample_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let prescribed_tape_bits = binary_tape
            .checked_add(uniform_tape)
            .and_then(|count| count.checked_add(gaussian_tape))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        Ok(Self {
            plaintext_bits,
            coefficient_bits,
            x_columns,
            attribute_count,
            ciphertext_bits,
            prescribed_tape_bits,
        })
    }
}

#[derive(Clone)]
pub(crate) struct PrivatePrfeLayerWires {
    pub b_trapdoor: Trapdoor,
    pub b_public: Mat,
    pub attribute_public: Vec<Mat>,
    compiler: AttributeEncodingCompiler,
    config: Aky24IoConfig,
    x_rows: usize,
    x_columns: usize,
    coefficient_bits: usize,
    plaintext_bits: usize,
}

#[derive(Clone)]
pub(crate) struct PrivatePrfeCiphertextWire {
    pub c_b: Mat,
    pub x: Mat,
    pub attributes: Vec<AttributeEncodingWire>,
}

/// Exact Appendix B.1 encryption tape. Keeping every distribution and shape
/// explicit prevents the cascade PRF from silently replacing Gaussian coins
/// by ternary or centered-binomial samples.
#[derive(Clone)]
pub(crate) struct PrivatePrfeEncryptionCoins {
    pub s_bar: Mat,
    pub e_b: Mat,
    pub seed: Vec<Mat>,
    pub a_bar_fhe: Mat,
    pub e_fhe: Mat,
    pub r: Mat,
    pub e_att: Vec<Mat>,
}

#[derive(Clone)]
pub(crate) struct PrivatePrfeFunctionKeyWire {
    pub preimage: Mat,
    pub output_count: usize,
}

#[derive(Debug, Error)]
pub(crate) enum PrivatePrfeGraphError {
    #[error(transparent)]
    Config(#[from] Aky24ConfigError),
    #[error(transparent)]
    Attribute(#[from] AttributeEvaluationError),
    #[error("the private-prFE attribute layout is inconsistent")]
    AttributeLayout,
    #[error("the private-prFE matrix-valued circuit layout is inconsistent")]
    CircuitLayout,
    #[error("the private-prFE coefficient conversion is not representable")]
    CoefficientConversion,
    #[error("the private-prFE arithmetic circuit cannot be normalized: {0}")]
    CircuitNormalization(String),
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

impl PrivatePrfeLayerWires {
    pub fn with_public_matrices(
        &self,
        b_public: Mat,
        attribute_public: Vec<Mat>,
    ) -> Result<Self, PrivatePrfeGraphError> {
        if !matrix_has_shape(&b_public, SECRET_ROWS, SECRET_ROWS * (self.config.digit_count + 2)) ||
            attribute_public.len() != self.attribute_public.len() ||
            attribute_public.iter().any(|matrix| {
                !matrix_has_shape(matrix, SECRET_ROWS, SECRET_ROWS * self.config.digit_count)
            })
        {
            return Err(PrivatePrfeGraphError::AttributeLayout);
        }
        let mut layer = self.clone();
        layer.b_public = b_public;
        layer.attribute_public = attribute_public;
        Ok(layer)
    }

    pub fn public_key_scalar_bits(&self) -> Result<Vec<Mat>, PrivatePrfeGraphError> {
        let mut bits = Vec::new();
        self.serialize_matrix_scalar_bits(
            self.b_public.clone(),
            SECRET_ROWS,
            SECRET_ROWS * (self.config.digit_count + 2),
            &mut bits,
        )?;
        for attribute in &self.attribute_public {
            self.serialize_matrix_scalar_bits(
                attribute.clone(),
                SECRET_ROWS,
                SECRET_ROWS * self.config.digit_count,
                &mut bits,
            )?;
        }
        Ok(bits)
    }

    pub fn ciphertext_from_components(
        &self,
        c_b: Mat,
        x: Mat,
        vectors: Vec<Mat>,
    ) -> Result<PrivatePrfeCiphertextWire, PrivatePrfeGraphError> {
        if vectors.len() != self.attribute_public.len() {
            return Err(PrivatePrfeGraphError::AttributeLayout);
        }
        let attributes = std::iter::once(self.compiler.ring.identity(1))
            .chain(self.serialize_x_bits(x.clone())?)
            .zip(self.attribute_public.iter().cloned())
            .zip(vectors)
            .map(|((attribute, public_matrix), vector)| AttributeEncodingWire {
                vector,
                public_matrix,
                attribute,
            })
            .collect();
        let ciphertext = PrivatePrfeCiphertextWire { c_b, x, attributes };
        // Reuse the canonical serializer's complete shape validation.
        let _ = self.serialize_ciphertext(&ciphertext)?;
        Ok(ciphertext)
    }

    /// Appendix B.1 Setup. The public GSW ciphertext contains the message and
    /// the `sd` seed, one binary-GSW block per bit. The leading attribute one
    /// is added here.
    pub fn setup(
        config: &Aky24IoConfig,
        plaintext_bits: usize,
    ) -> Result<Self, PrivatePrfeGraphError> {
        config.validate()?;
        if plaintext_bits == 0 {
            return Err(PrivatePrfeGraphError::AttributeLayout);
        }
        let dimensions = PrivatePrfeLayerDimensions::new(config, plaintext_bits)?;
        let coefficient_bits = dimensions.coefficient_bits;
        let x_rows = SECRET_ROWS;
        let x_columns = dimensions.x_columns;
        // HLL23 MakeHEval receives one scalar attribute for every canonical
        // coefficient bit of X. Derived GSW values can therefore be gadget
        // decomposed gate-by-gate without normalizing the logical circuit to
        // an exponentially large sparse polynomial.
        let attribute_bits = x_rows
            .checked_mul(x_columns)
            .and_then(|count| count.checked_mul(config.ring_dimension))
            .and_then(|count| count.checked_mul(coefficient_bits))
            .ok_or(PrivatePrfeGraphError::AttributeLayout)?;
        let ring = Ring::new(config.modulus.clone(), config.ring_dimension);
        let block_columns = SECRET_ROWS
            .checked_mul(config.digit_count)
            .ok_or(PrivatePrfeGraphError::AttributeLayout)?;
        debug_assert_eq!(attribute_bits + 1, dimensions.attribute_count);
        let attribute_public = (0..dimensions.attribute_count)
            .map(|_| ring.uniform((SECRET_ROWS, block_columns)))
            .collect();
        let b_trapdoor = ring.sample_trapdoor(
            SECRET_ROWS,
            config.trapdoor_sigma.clone(),
            config.gadget_base.clone(),
            config.digit_count,
        );
        let b_public = b_trapdoor.public_matrix();
        Ok(Self {
            b_trapdoor,
            b_public,
            attribute_public,
            compiler: AttributeEncodingCompiler {
                ring,
                gadget_base: config.gadget_base.clone().into(),
                digit_count: config.digit_count.into(),
            },
            config: config.clone(),
            x_rows,
            x_columns,
            coefficient_bits,
            plaintext_bits,
        })
    }

    /// Appendix B.1 Enc. `X`, `c_B`, and `c_att` deliberately share the same
    /// secret `s = (s_bar, -1)`.
    pub fn encrypt(
        &self,
        message: Vec<Mat>,
        seed: Vec<Mat>,
    ) -> Result<PrivatePrfeCiphertextWire, PrivatePrfeGraphError> {
        let ring = &self.compiler.ring;
        let gsw_columns = SECRET_ROWS * self.config.digit_count;
        let b_columns = self.config.digit_count + 2;
        self.encrypt_with_coins(
            message,
            PrivatePrfeEncryptionCoins {
                s_bar: ring.gaussian((1, SECRET_ROWS - 1), self.config.secret_sigma.clone()),
                e_b: ring.gaussian((1, SECRET_ROWS * b_columns), self.config.b_error_sigma.clone()),
                seed,
                a_bar_fhe: ring.uniform((SECRET_ROWS - 1, gsw_columns)),
                e_fhe: ring.gaussian((1, gsw_columns), self.config.fhe_error_sigma.clone()),
                r: ring.uniform_in((gsw_columns, self.x_columns), 0, 1),
                e_att: (0..self.attribute_public.len())
                    .map(|_| {
                        ring.gaussian(
                            (1, SECRET_ROWS * self.config.digit_count),
                            self.config.attribute_error_sigma.clone(),
                        )
                    })
                    .collect(),
            },
        )
    }

    /// Appendix B.1 Enc with a fully prescribed random tape. Cascade
    /// functions call this algebra after deriving `coins` from their PRF.
    pub fn encrypt_with_coins(
        &self,
        message: Vec<Mat>,
        coins: PrivatePrfeEncryptionCoins,
    ) -> Result<PrivatePrfeCiphertextWire, PrivatePrfeGraphError> {
        let seed = coins.seed;
        if message.len() != self.plaintext_bits ||
            seed.len() != self.config.security_parameter_bits ||
            message.iter().chain(&seed).any(|bit| {
                bit.matrix_type().rows != IntExpr::constant(1) ||
                    bit.matrix_type().columns != IntExpr::constant(1)
            })
        {
            return Err(PrivatePrfeGraphError::AttributeLayout);
        }
        let ring = &self.compiler.ring;
        let gsw_columns = SECRET_ROWS * self.config.digit_count;
        let b_columns = self.config.digit_count + 2;
        if !matrix_has_shape(&coins.s_bar, 1, SECRET_ROWS - 1) ||
            !matrix_has_shape(&coins.e_b, 1, SECRET_ROWS * b_columns) ||
            !matrix_has_shape(&coins.a_bar_fhe, SECRET_ROWS - 1, gsw_columns) ||
            !matrix_has_shape(&coins.e_fhe, 1, gsw_columns) ||
            !matrix_has_shape(&coins.r, gsw_columns, self.x_columns) ||
            coins.e_att.len() != self.attribute_public.len() ||
            coins
                .e_att
                .iter()
                .any(|error| !matrix_has_shape(error, 1, SECRET_ROWS * self.config.digit_count))
        {
            return Err(PrivatePrfeGraphError::AttributeLayout);
        }
        let s_bar = coins.s_bar;
        let minus_one = -ring.identity(1);
        let secret = Mat::concat(ConcatAxis::Columns, vec![s_bar.clone(), minus_one]);
        let a_bar = coins.a_bar_fhe;
        let fhe_error = coins.e_fhe;
        let a_fhe = Mat::concat(ConcatAxis::Rows, vec![a_bar.clone(), s_bar * a_bar + fhe_error]);
        let gadget =
            ring.gadget(SECRET_ROWS, self.config.gadget_base.clone(), self.config.digit_count);
        let plaintext_gadget = Mat::concat(
            ConcatAxis::Columns,
            message.into_iter().chain(seed).map(|bit| gadget.clone() * bit).collect(),
        );
        let x = a_fhe * coins.r - plaintext_gadget;
        let x_bits = self.serialize_x_bits(x.clone())?;
        let c_b = secret.clone() * self.b_public.clone() + coins.e_b;
        let attributes = std::iter::once(ring.identity(1))
            .chain(x_bits)
            .zip(self.attribute_public.iter().cloned())
            .zip(coins.e_att)
            .map(|((attribute, public_matrix), error)| AttributeEncodingWire {
                vector: secret.clone() *
                    (public_matrix.clone() - gadget.clone() * attribute.clone()) +
                    error,
                public_matrix,
                attribute,
            })
            .collect();
        Ok(PrivatePrfeCiphertextWire { c_b, x, attributes })
    }

    /// Full prescribed-randomness layout for this layer. Counts are over
    /// scalar polynomial coefficients, not merely matrix entries.
    pub fn prescribed_coin_layout(&self) -> Result<PrescribedCoinLayout, PrivatePrfeGraphError> {
        let ring_dimension = self.config.ring_dimension;
        let gsw_columns = SECRET_ROWS
            .checked_mul(self.config.digit_count)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let r_bits = gsw_columns
            .checked_mul(self.x_columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let binary_bits = self
            .config
            .security_parameter_bits
            .checked_add(r_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let uniform_coefficients = (SECRET_ROWS - 1)
            .checked_mul(gsw_columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let b_error_coefficients = SECRET_ROWS
            .checked_mul(self.config.digit_count + 2)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let fhe_error_coefficients = gsw_columns
            .checked_mul(ring_dimension)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let attribute_error_coefficients = self
            .attribute_public
            .len()
            .checked_mul(gsw_columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let evaluate_sigma = |sigma: &mxx_ir_core::RealExpr| {
            sigma
                .evaluate_f64(&ParamEnv::default())
                .map_err(|error| PrivatePrfeGraphError::CircuitNormalization(error.to_string()))
        };
        let secret_coefficients = (SECRET_ROWS - 1)
            .checked_mul(ring_dimension)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        Ok(PrescribedCoinLayout {
            binary_bits,
            uniform_coefficients,
            uniform_sample_bits: self
                .coefficient_bits
                .checked_add(self.config.uniform_statistical_bits)
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
            gaussian_sample_bits: self.config.gaussian_sample_bits,
            gaussian_groups: vec![
                PrescribedGaussianGroup {
                    coefficients: secret_coefficients,
                    sigma: evaluate_sigma(&self.config.secret_sigma)?,
                },
                PrescribedGaussianGroup {
                    coefficients: b_error_coefficients,
                    sigma: evaluate_sigma(&self.config.b_error_sigma)?,
                },
                PrescribedGaussianGroup {
                    coefficients: fhe_error_coefficients,
                    sigma: evaluate_sigma(&self.config.fhe_error_sigma)?,
                },
                PrescribedGaussianGroup {
                    coefficients: attribute_error_coefficients,
                    sigma: evaluate_sigma(&self.config.attribute_error_sigma)?,
                },
            ],
        })
    }

    /// `MakeVEvalCkt` for the linear projection `C(x, sd) = scale * x_i`.
    /// This is also the base ciphertext operation from which general Boolean
    /// GSW evaluation is built.
    #[cfg(test)]
    pub fn projection_veval<P: Poly>(
        &self,
        message_index: usize,
        scale: num_bigint::BigUint,
    ) -> Result<PolyCircuit<P>, PrivatePrfeGraphError> {
        if message_index >= self.plaintext_bits {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let mut logical = PolyCircuit::new();
        let inputs = logical.input(self.plaintext_bits + self.config.security_parameter_bits);
        logical.output([inputs.at(message_index)]);
        self.arithmetic_veval(&logical, 0, scale)
    }

    /// `MakeVEvalCkt` for the supported Goldreich local-predicate family.
    ///
    /// This is a small logical-circuit front end to the same gate-by-gate
    /// HLL23 MakeHEval construction used by [`Self::arithmetic_veval`].
    #[cfg(test)]
    pub fn goldreich_veval<P: Poly>(
        &self,
        graph: &GoldreichGraph,
        scale: num_bigint::BigUint,
    ) -> Result<PolyCircuit<P>, PrivatePrfeGraphError> {
        if graph.input_size > self.plaintext_bits || graph.edges.is_empty() {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let mut logical = PolyCircuit::new();
        let inputs = logical.input(self.plaintext_bits + self.config.security_parameter_bits);
        let outputs = graph
            .edges
            .iter()
            .map(|edge| {
                let and =
                    logical.and_gate(inputs.at(edge.and_inputs[0]), inputs.at(edge.and_inputs[1]));
                let first =
                    logical.xor_gate(inputs.at(edge.xor_inputs[0]), inputs.at(edge.xor_inputs[1]));
                let second = logical.xor_gate(first, inputs.at(edge.xor_inputs[2]));
                logical.xor_gate(second, and).as_single_wire()
            })
            .collect::<Vec<_>>();
        logical.output(outputs);
        self.arithmetic_veval(&logical, 0, scale)
    }

    /// Builds `MakeVEvalCkt` for an arbitrary arithmetic circuit over the
    /// plaintext and `sd` inputs encrypted in `X`.
    ///
    /// The returned circuit receives scalar `bits(X)` attributes first and
    /// `public_input_count` dynamic public bits second. It represents every
    /// logical GSW value by the canonical bits of all matrix coefficients.
    /// Consequently a multiplication can build `-L * G^-1(R)` even when `R`
    /// is derived by earlier gates. This is the polynomial-size MakeHEval
    /// construction from HLL23 Lemma 5, not sparse-polynomial normalization.
    pub fn arithmetic_veval<P: Poly>(
        &self,
        plaintext_circuit: &PolyCircuit<P>,
        public_input_count: usize,
        scale: num_bigint::BigUint,
    ) -> Result<PolyCircuit<P>, PrivatePrfeGraphError> {
        let encoded_input_count = self
            .plaintext_bits
            .checked_add(self.config.security_parameter_bits)
            .ok_or(PrivatePrfeGraphError::CircuitLayout)?;
        if plaintext_circuit.num_input() !=
            encoded_input_count
                .checked_add(public_input_count)
                .ok_or(PrivatePrfeGraphError::CircuitLayout)? ||
            plaintext_circuit.num_output() == 0
        {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let mut circuit = PolyCircuit::new();
        let attribute_count = self.attribute_public.len() - 1;
        let inputs = circuit.input(attribute_count + public_input_count);
        let attribute_bits = inputs.clone().slice(0..attribute_count);
        let original = (0..encoded_input_count)
            .map(|index| self.original_gsw_block(&attribute_bits, index))
            .collect::<Vec<_>>();
        let public = (0..public_input_count)
            .map(|index| {
                let bit = inputs.at(attribute_count + index).as_single_wire();
                gsw_from_bit(
                    bit,
                    &mut circuit,
                    self.config.ring_dimension,
                    &self.config.modulus.to_biguint().expect("validated positive modulus"),
                    self.config.digit_count,
                )
            })
            .collect::<Vec<_>>();
        let one_gate = circuit.const_one_gate().as_single_wire();
        let one = gsw_from_bit(
            one_gate,
            &mut circuit,
            self.config.ring_dimension,
            &self.config.modulus.to_biguint().expect("validated positive modulus"),
            self.config.digit_count,
        );
        let mut lowering = MakeHevalLowering {
            circuit: &mut circuit,
            modulus: self.config.modulus.to_biguint().expect("validated positive modulus"),
            ring_dimension: self.config.ring_dimension,
            digit_count: self.config.digit_count,
        };
        let evaluated = lower_circuit(
            plaintext_circuit,
            one,
            original.into_iter().chain(public),
            &mut lowering,
        )
        .map_err(map_arithmetic_normalization_error)?;
        let outputs = evaluated
            .iter()
            .map(|value| {
                project_gsw(
                    value,
                    &scale,
                    lowering.circuit,
                    &lowering.modulus,
                    lowering.digit_count,
                )
            })
            .collect::<Vec<_>>();
        circuit.output(
            outputs.iter().map(|output| output[0]).chain(outputs.iter().map(|output| output[1])),
        );
        Ok(circuit)
    }

    fn original_gsw_block(
        &self,
        bits: &mxx_gadgets::circuit::BatchedWire,
        block: usize,
    ) -> CanonicalMatrix {
        let width = SECRET_ROWS * self.config.digit_count;
        let block_start = block * width;
        let entry_bits = self.config.ring_dimension * self.coefficient_bits;
        let canonical = (0..SECRET_ROWS)
            .flat_map(|row| {
                (0..width).flat_map(move |column| {
                    let offset = (row * self.x_columns + block_start + column) * entry_bits;
                    (0..entry_bits).map(move |bit| bits.at(offset + bit).as_single_wire())
                })
            })
            .collect::<Vec<_>>();
        CanonicalMatrix::from_canonical_bits(
            &canonical,
            SECRET_ROWS,
            width,
            self.config.ring_dimension,
            &self.config.modulus.to_biguint().expect("validated positive modulus"),
        )
    }

    /// Appendix B.1 KeyGen for already-constructed `VEval_high` and
    /// `VEval_low`. Their outputs are row-major `(n+1) x ell` matrices.
    pub fn keygen<P: Poly>(
        &self,
        veval_high: &PolyCircuit<P>,
        veval_low: &PolyCircuit<P>,
        public_inputs: &[Mat],
    ) -> Result<PrivatePrfeFunctionKeyWire, PrivatePrfeGraphError> {
        let output_count = self.validate_veval_pair(veval_high, veval_low, public_inputs.len())?;
        let high = self.compiler.evaluate_public_matrix_mixed(
            veval_high,
            self.attribute_public[0].clone(),
            self.attribute_public[1..].iter().cloned(),
            public_inputs.iter().cloned(),
            SECRET_ROWS,
        )?;
        let low = self.compiler.evaluate_public_matrix_mixed(
            veval_low,
            self.attribute_public[0].clone(),
            self.attribute_public[1..].iter().cloned(),
            public_inputs.iter().cloned(),
            SECRET_ROWS,
        )?;
        let target = self.combine_high_low(high, low, SECRET_ROWS, output_count)?;
        let b_rows = IntExpr::constant(SECRET_ROWS * (self.config.digit_count + 2));
        let preimage = self
            .b_trapdoor
            .sample_preimage(target, (b_rows, IntExpr::constant(output_count)))
            .as_mat();
        Ok(PrivatePrfeFunctionKeyWire { preimage, output_count })
    }

    /// Appendix B.1 Dec, including the high/low modulus split and final
    /// threshold decision.
    pub fn decrypt<P: Poly>(
        &self,
        veval_high: &PolyCircuit<P>,
        veval_low: &PolyCircuit<P>,
        ciphertext: &PrivatePrfeCiphertextWire,
        key: &PrivatePrfeFunctionKeyWire,
        public_inputs: &[Mat],
    ) -> Result<Vec<Bool>, PrivatePrfeGraphError> {
        let output_count = self.validate_veval_pair(veval_high, veval_low, public_inputs.len())?;
        if output_count != key.output_count ||
            ciphertext.attributes.len() != self.attribute_public.len() ||
            ciphertext.x.matrix_type().rows.clone().canonicalize() !=
                IntExpr::constant(self.x_rows) ||
            ciphertext.x.matrix_type().columns.clone().canonicalize() !=
                IntExpr::constant(self.x_columns)
        {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let high = self.evaluate_ciphertext_matrix(veval_high, ciphertext, public_inputs)?;
        let low = self.evaluate_ciphertext_matrix(veval_low, ciphertext, public_inputs)?;
        let evaluated = self.combine_high_low(high.vector, low.vector, 1, output_count)?;
        let z = ciphertext.c_b.clone() * key.preimage.clone() - evaluated;
        Ok(z.threshold_decode_bools(2, output_count))
    }

    /// Canonical bit serialization of an Appendix B.1 ciphertext.
    ///
    /// Matrices are ordered `c_B`, `X`, then the attribute vectors.  Within
    /// each matrix entries are row-major; each polynomial is coefficient-major
    /// and little-endian within a coefficient.  Public attribute matrices and
    /// the attributes themselves are deliberately omitted: they are recovered
    /// from this layer's public key and the reconstructed `X`.
    pub fn serialize_ciphertext(
        &self,
        ciphertext: &PrivatePrfeCiphertextWire,
    ) -> Result<Vec<Bool>, PrivatePrfeGraphError> {
        if ciphertext.attributes.len() != self.attribute_public.len() ||
            !matrix_has_shape(&ciphertext.c_b, 1, SECRET_ROWS * (self.config.digit_count + 2)) ||
            !matrix_has_shape(&ciphertext.x, self.x_rows, self.x_columns) ||
            ciphertext.attributes.iter().any(|attribute| {
                !matrix_has_shape(&attribute.vector, 1, SECRET_ROWS * self.config.digit_count)
            })
        {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let expected =
            PrivatePrfeLayerDimensions::new(&self.config, self.plaintext_bits)?.ciphertext_bits;
        let mut bits = Vec::with_capacity(expected);
        self.serialize_matrix_coefficients(
            ciphertext.c_b.clone(),
            1,
            SECRET_ROWS * (self.config.digit_count + 2),
            &mut bits,
        )?;
        self.serialize_matrix_coefficients(
            ciphertext.x.clone(),
            self.x_rows,
            self.x_columns,
            &mut bits,
        )?;
        for attribute in &ciphertext.attributes {
            self.serialize_matrix_coefficients(
                attribute.vector.clone(),
                1,
                SECRET_ROWS * self.config.digit_count,
                &mut bits,
            )?;
        }
        debug_assert_eq!(bits.len(), expected);
        Ok(bits)
    }

    /// Reconstructs exactly the ciphertext serialized by
    /// [`Self::serialize_ciphertext`].
    pub fn deserialize_ciphertext(
        &self,
        bits: &[Bool],
    ) -> Result<PrivatePrfeCiphertextWire, PrivatePrfeGraphError> {
        let expected =
            PrivatePrfeLayerDimensions::new(&self.config, self.plaintext_bits)?.ciphertext_bits;
        if bits.len() != expected {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let mut cursor = 0;
        let c_b = self.deserialize_matrix_coefficients(
            bits,
            &mut cursor,
            1,
            SECRET_ROWS * (self.config.digit_count + 2),
        )?;
        let x =
            self.deserialize_matrix_coefficients(bits, &mut cursor, self.x_rows, self.x_columns)?;
        let attributes = std::iter::once(self.compiler.ring.identity(1))
            .chain(self.serialize_x_bits(x.clone())?)
            .zip(self.attribute_public.iter().cloned())
            .map(|(attribute, public_matrix)| {
                let vector = self.deserialize_matrix_coefficients(
                    bits,
                    &mut cursor,
                    1,
                    SECRET_ROWS * self.config.digit_count,
                )?;
                Ok(AttributeEncodingWire { vector, public_matrix, attribute })
            })
            .collect::<Result<Vec<_>, PrivatePrfeGraphError>>()?;
        if cursor != bits.len() {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        Ok(PrivatePrfeCiphertextWire { c_b, x, attributes })
    }

    fn serialize_matrix_coefficients(
        &self,
        matrix: Mat,
        rows: usize,
        columns: usize,
        output: &mut Vec<Bool>,
    ) -> Result<(), PrivatePrfeGraphError> {
        if !matrix_has_shape(&matrix, rows, columns) {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        for row in 0..rows {
            for column in 0..columns {
                let scalar = matrix.clone().slice(
                    Some(IndexRange { start: row.into(), end: (row + 1).into() }),
                    Some(IndexRange { start: column.into(), end: (column + 1).into() }),
                );
                let bits = scalar.canonical_coefficient_bits(
                    self.config.ring_dimension,
                    self.coefficient_bits,
                )?;
                output.extend(
                    (0..self.config.ring_dimension * self.coefficient_bits)
                        .map(|index| bits.get_static(index)),
                );
            }
        }
        Ok(())
    }

    fn serialize_matrix_scalar_bits(
        &self,
        matrix: Mat,
        rows: usize,
        columns: usize,
        output: &mut Vec<Mat>,
    ) -> Result<(), PrivatePrfeGraphError> {
        if !matrix_has_shape(&matrix, rows, columns) {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        let zero = self.compiler.ring.zero((1, 1));
        let one = self.compiler.ring.identity(1);
        for row in 0..rows {
            for column in 0..columns {
                let scalar = matrix.clone().slice(
                    Some(IndexRange { start: row.into(), end: (row + 1).into() }),
                    Some(IndexRange { start: column.into(), end: (column + 1).into() }),
                );
                let bits = scalar.canonical_coefficient_bits(
                    self.config.ring_dimension,
                    self.coefficient_bits,
                )?;
                output.extend((0..self.config.ring_dimension * self.coefficient_bits).map(|bit| {
                    bits.get_static(bit)
                        .to_int()
                        .select(vec![zero.clone(), one.clone()])
                        .expect("binary scalar selector has two branches")
                }));
            }
        }
        Ok(())
    }

    fn deserialize_matrix_coefficients(
        &self,
        bits: &[Bool],
        cursor: &mut usize,
        rows: usize,
        columns: usize,
    ) -> Result<Mat, PrivatePrfeGraphError> {
        let polynomial_bits = self
            .config
            .ring_dimension
            .checked_mul(self.coefficient_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let mut output_rows = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut output_columns = Vec::with_capacity(columns);
            for _ in 0..columns {
                let end = cursor
                    .checked_add(polynomial_bits)
                    .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
                let coefficient_bits =
                    bits.get(*cursor..end).ok_or(PrivatePrfeGraphError::CircuitLayout)?;
                *cursor = end;
                output_columns.push(self.compiler.ring.pack_polynomial_coefficients(
                    Family::pack_bools(coefficient_bits.to_vec())?,
                    self.coefficient_bits,
                ));
            }
            output_rows.push(Mat::concat(ConcatAxis::Columns, output_columns));
        }
        Ok(Mat::concat(ConcatAxis::Rows, output_rows))
    }

    fn evaluate_ciphertext_matrix<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        ciphertext: &PrivatePrfeCiphertextWire,
        public_inputs: &[Mat],
    ) -> Result<AttributeMatrixEvaluation, PrivatePrfeGraphError> {
        self.compiler
            .evaluate_encoded_matrix_mixed(
                circuit,
                ciphertext.attributes[0].clone(),
                ciphertext.attributes[1..].iter().cloned(),
                public_inputs.iter().cloned(),
                SECRET_ROWS,
            )
            .map_err(Into::into)
    }

    fn validate_veval_pair<P: Poly>(
        &self,
        high: &PolyCircuit<P>,
        low: &PolyCircuit<P>,
        public_input_count: usize,
    ) -> Result<usize, PrivatePrfeGraphError> {
        let input_count = self
            .attribute_public
            .len()
            .checked_sub(1)
            .and_then(|count| count.checked_add(public_input_count))
            .ok_or(PrivatePrfeGraphError::CircuitLayout)?;
        if high.num_input() != input_count ||
            low.num_input() != input_count ||
            high.num_output() != low.num_output() ||
            high.num_output() == 0 ||
            !high.num_output().is_multiple_of(SECRET_ROWS)
        {
            return Err(PrivatePrfeGraphError::CircuitLayout);
        }
        Ok(high.num_output() / SECRET_ROWS)
    }

    fn serialize_x_bits(&self, x: Mat) -> Result<Vec<Mat>, PrivatePrfeGraphError> {
        let mut output = Vec::with_capacity(self.attribute_public.len() - 1);
        self.serialize_matrix_scalar_bits(x, self.x_rows, self.x_columns, &mut output)?;
        Ok(output)
    }

    fn combine_high_low(
        &self,
        high: Mat,
        low: Mat,
        rows: usize,
        columns: usize,
    ) -> Result<Mat, PrivatePrfeGraphError> {
        let high = self.round_divide_matrix(high, rows, columns)?;
        let low = self.round_divide_matrix(low, rows, columns)?;
        let scale =
            self.compiler.ring.polynomial([IntExpr::constant(self.config.modulus_split.clone())]);
        Ok(high * scale + low)
    }

    /// Coefficient-wise centered `round(value / M)`. This is intentionally a
    /// private composition of ordinary integer and bit-pack nodes, not a
    /// generic ModDown/ModUp operation.
    fn round_divide_matrix(
        &self,
        matrix: Mat,
        rows: usize,
        columns: usize,
    ) -> Result<Mat, PrivatePrfeGraphError> {
        let modulus = self.config.modulus.clone();
        let divisor = self.config.modulus_split.clone();
        let coefficient_bits = modulus
            .to_biguint()
            .map(|value| value.bits() as usize)
            .filter(|bits| *bits > 0)
            .ok_or(PrivatePrfeGraphError::CoefficientConversion)?;
        let mut output_rows = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut output_columns = Vec::with_capacity(columns);
            for column in 0..columns {
                let scalar = matrix.clone().slice(
                    Some(IndexRange { start: row.into(), end: (row + 1).into() }),
                    Some(IndexRange { start: column.into(), end: (column + 1).into() }),
                );
                let mut bits = Vec::with_capacity(self.config.ring_dimension * coefficient_bits);
                for coefficient in 0..self.config.ring_dimension {
                    let residue = scalar.clone().extract_coefficient(coefficient);
                    let centered = centered_residue(residue, &modulus);
                    let quotient = rounded_quotient(centered, &divisor);
                    let canonical = canonical_residue(quotient, &modulus);
                    bits.extend((0..coefficient_bits).map(|bit| canonical.clone().bit(bit)));
                }
                let bits = Family::<Bool>::pack_bools(bits)?;
                output_columns
                    .push(self.compiler.ring.pack_polynomial_coefficients(bits, coefficient_bits));
            }
            output_rows.push(Mat::concat(ConcatAxis::Columns, output_columns));
        }
        Ok(Mat::concat(ConcatAxis::Rows, output_rows))
    }
}

fn gsw_from_bit<P: Poly>(
    bit: GateId,
    circuit: &mut PolyCircuit<P>,
    ring_dimension: usize,
    modulus: &num_bigint::BigUint,
    digit_count: usize,
) -> CanonicalMatrix {
    let coefficient_bits = modulus.bits() as usize;
    let zero_gate = circuit.const_zero_gate().as_single_wire();
    let bit_polynomial = CanonicalPolynomial::from_coefficients(
        (0..ring_dimension)
            .map(|coefficient| {
                let mut bits = vec![zero_gate; coefficient_bits];
                if coefficient == 0 {
                    bits[0] = bit;
                }
                CanonicalResidue::from_canonical_bits(bits, modulus)
            })
            .collect(),
    );
    let zero = CanonicalPolynomial::zero(circuit, ring_dimension, modulus);
    let width = SECRET_ROWS * digit_count;
    let mut entries = Vec::with_capacity(SECRET_ROWS * width);
    for row in 0..SECRET_ROWS {
        for column in 0..width {
            if column / digit_count == row {
                let digit = column % digit_count;
                let weight = (num_bigint::BigUint::from(1u8) << digit) % modulus;
                let negative = if weight == num_bigint::BigUint::from(0u8) {
                    weight
                } else {
                    modulus - weight
                };
                entries.push(bit_polynomial.scale_constant(&negative, circuit, modulus));
            } else {
                entries.push(zero.clone());
            }
        }
    }
    CanonicalMatrix::from_entries(SECRET_ROWS, width, entries)
}

fn gadget_decompose<P: Poly>(
    value: &CanonicalMatrix,
    circuit: &mut PolyCircuit<P>,
    modulus: &num_bigint::BigUint,
    digit_count: usize,
) -> CanonicalMatrix {
    let coefficient_bits = modulus.bits() as usize;
    let ring_dimension = value.entry(0, 0).ring_dimension();
    let zero_gate = circuit.const_zero_gate().as_single_wire();
    let mut entries = Vec::with_capacity(SECRET_ROWS * digit_count * value.columns());
    for decomposition_row in 0..SECRET_ROWS * digit_count {
        let source_row = decomposition_row / digit_count;
        let digit = decomposition_row % digit_count;
        for column in 0..value.columns() {
            let source = value.entry(source_row, column);
            let coefficients = (0..ring_dimension)
                .map(|coefficient| {
                    let selected = source
                        .coefficient(coefficient)
                        .bits()
                        .get(digit)
                        .copied()
                        .unwrap_or(zero_gate);
                    let mut bits = vec![zero_gate; coefficient_bits];
                    bits[0] = selected;
                    CanonicalResidue::from_canonical_bits(bits, modulus)
                })
                .collect();
            entries.push(CanonicalPolynomial::from_coefficients(coefficients));
        }
    }
    CanonicalMatrix::from_entries(SECRET_ROWS * digit_count, value.columns(), entries)
}

fn normalized_public_polynomial(
    coefficients: impl IntoIterator<Item = BigInt>,
    ring_dimension: usize,
    modulus: &BigInt,
) -> Vec<num_bigint::BigUint> {
    let mut normalized = vec![BigInt::from(0); ring_dimension];
    for (index, coefficient) in coefficients.into_iter().enumerate() {
        let target = index % ring_dimension;
        let signed =
            if (index / ring_dimension).is_multiple_of(2) { coefficient } else { -coefficient };
        normalized[target] = ((normalized[target].clone() + signed) % modulus + modulus) % modulus;
    }
    normalized
        .into_iter()
        .map(|value| value.to_biguint().expect("canonical coefficient is nonnegative"))
        .collect()
}

fn scale_gsw<P: Poly>(
    value: &CanonicalMatrix,
    coefficients: Vec<num_bigint::BigUint>,
    circuit: &mut PolyCircuit<P>,
    modulus: &num_bigint::BigUint,
) -> CanonicalMatrix {
    let polynomial = CanonicalPolynomial::from_coefficients(
        coefficients
            .into_iter()
            .map(|coefficient| CanonicalResidue::constant(circuit, &coefficient, modulus))
            .collect(),
    );
    let mut entries = Vec::with_capacity(value.rows() * value.columns());
    for row in 0..value.rows() {
        for column in 0..value.columns() {
            entries.push(value.entry(row, column).mul_negacyclic(&polynomial, circuit, modulus));
        }
    }
    CanonicalMatrix::from_entries(value.rows(), value.columns(), entries)
}

fn emit_polynomial<P: Poly>(value: &CanonicalPolynomial, circuit: &mut PolyCircuit<P>) -> GateId {
    let ring_dimension = value.ring_dimension();
    let mut terms = Vec::new();
    for coefficient in 0..ring_dimension {
        for (bit, gate) in value.coefficient(coefficient).bits().iter().copied().enumerate() {
            let mut scalar = vec![num_bigint::BigUint::from(0u8); ring_dimension];
            scalar[coefficient] = num_bigint::BigUint::from(1u8) << bit;
            terms.push(circuit.large_scalar_mul(gate, &scalar).as_single_wire());
        }
    }
    if terms.is_empty() {
        circuit.const_zero_gate().as_single_wire()
    } else {
        add_gate_tree(circuit, terms)
    }
}

fn project_gsw<P: Poly>(
    value: &CanonicalMatrix,
    scale: &num_bigint::BigUint,
    circuit: &mut PolyCircuit<P>,
    modulus: &num_bigint::BigUint,
    digit_count: usize,
) -> [GateId; SECRET_ROWS] {
    std::array::from_fn(|row| {
        let mut sum =
            CanonicalPolynomial::zero(circuit, value.entry(0, 0).ring_dimension(), modulus);
        for digit in 0..digit_count {
            if scale.bit(digit as u64) {
                sum = sum.add(value.entry(row, digit_count + digit), circuit, modulus);
            }
        }
        emit_polynomial(&sum, circuit)
    })
}

struct MakeHevalLowering<'a, P: Poly> {
    circuit: &'a mut PolyCircuit<P>,
    modulus: num_bigint::BigUint,
    ring_dimension: usize,
    digit_count: usize,
}

impl<P: Poly> CircuitLoweringTypes for MakeHevalLowering<'_, P> {
    type Wire = CanonicalMatrix;
    type Error = PrivatePrfeGraphError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for MakeHevalLowering<'_, P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &CanonicalMatrix,
        rhs: &CanonicalMatrix,
        gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(lhs.add(rhs, self.circuit, &self.modulus)),
            PolyGateKind::Sub => {
                let negative = rhs.negate(self.circuit, &self.modulus);
                Ok(lhs.add(&negative, self.circuit, &self.modulus))
            }
            PolyGateKind::Mul => {
                let decomposed =
                    gadget_decompose(rhs, self.circuit, &self.modulus, self.digit_count);
                let product =
                    lhs.mul(&decomposed, self.circuit, self.ring_dimension, &self.modulus);
                Ok(product.negate(self.circuit, &self.modulus))
            }
            _ => Err(PrivatePrfeGraphError::CircuitNormalization(format!(
                "gate {} is not arithmetic",
                gate.local_gate().index()
            ))),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &CanonicalMatrix,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        let modulus = BigInt::from(self.modulus.clone());
        let coefficients = normalized_public_polynomial(
            scalar.iter().copied().map(BigInt::from),
            self.ring_dimension,
            &modulus,
        );
        Ok(scale_gsw(input, coefficients, self.circuit, &self.modulus))
    }

    fn large_scalar_mul(
        &mut self,
        input: &CanonicalMatrix,
        scalar: &[num_bigint::BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        let modulus = BigInt::from(self.modulus.clone());
        let coefficients = normalized_public_polynomial(
            scalar.iter().cloned().map(BigInt::from),
            self.ring_dimension,
            &modulus,
        );
        Ok(scale_gsw(input, coefficients, self.circuit, &self.modulus))
    }
}

impl<P: Poly> PublicLookupLowering<P> for MakeHevalLowering<'_, P> {
    fn public_lookup(
        &mut self,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &CanonicalMatrix,
        gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        Err(PrivatePrfeGraphError::CircuitNormalization(format!(
            "gate {} uses a public lookup",
            gate.local_gate().index()
        )))
    }
}

impl<P: Poly> SlotOperationLowering<P> for MakeHevalLowering<'_, P> {
    fn slot_transfer(
        &mut self,
        _input: &CanonicalMatrix,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        Err(PrivatePrfeGraphError::CircuitNormalization(format!(
            "gate {} uses slot transfer",
            gate.local_gate().index()
        )))
    }

    fn slot_reduce(
        &mut self,
        _inputs: &[CanonicalMatrix],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<CanonicalMatrix, Self::Error> {
        Err(PrivatePrfeGraphError::CircuitNormalization(format!(
            "gate {} uses slot reduction",
            gate.local_gate().index()
        )))
    }
}

fn map_arithmetic_normalization_error(
    error: CircuitLowerError<PrivatePrfeGraphError>,
) -> PrivatePrfeGraphError {
    match error {
        CircuitLowerError::Operation { source, .. } => source,
        other => PrivatePrfeGraphError::CircuitNormalization(other.to_string()),
    }
}

fn add_gate_tree<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    mut layer: Vec<mxx_gadgets::circuit::gate::GateId>,
) -> mxx_gadgets::circuit::gate::GateId {
    assert!(!layer.is_empty(), "matrix reconstruction requires at least one term");
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut pairs = layer.chunks_exact(2);
        for pair in &mut pairs {
            next.push(circuit.add_gate(pair[0], pair[1]).as_single_wire());
        }
        next.extend_from_slice(pairs.remainder());
        layer = next;
    }
    layer[0]
}

fn matrix_has_shape(matrix: &Mat, rows: usize, columns: usize) -> bool {
    matrix.matrix_type().rows.clone().canonicalize() == IntExpr::constant(rows) &&
        matrix.matrix_type().columns.clone().canonicalize() == IntExpr::constant(columns)
}

fn centered_residue(residue: Int, modulus: &BigInt) -> Int {
    let upper = Int::constant((modulus + 1) / 2).less_equal(residue.clone()).to_int();
    residue.sub(upper.mul(Int::constant(modulus.clone())))
}

fn rounded_quotient(value: Int, divisor: &BigInt) -> Int {
    let negative = value.clone().less_equal(Int::constant(-1)).to_int();
    let sign = Int::constant(1).sub(negative.clone().mul(Int::constant(2)));
    let absolute = value.mul(sign.clone());
    let rounded = absolute.add(Int::constant(divisor / 2)).div(Int::constant(divisor.clone()));
    rounded.mul(sign)
}

fn canonical_residue(value: Int, modulus: &BigInt) -> Int {
    let negative = value.clone().less_equal(Int::constant(-1)).to_int();
    value.add(negative.mul(Int::constant(modulus.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aky24::config::Aky24GoldreichPrf;
    use mxx_dsl::DslContext;
    use mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::GoldreichGraphGeneration;
    use mxx_ir_core::{ParamEnv, RealExpr};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly as ConcretePoly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        Backend, RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn config(parameters: &DCRTPolyParams) -> Aky24IoConfig {
        Aky24IoConfig {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            input_size: 8,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count: parameters.modulus_digits(),
            modulus_split: 1.into(),
            trapdoor_sigma: RealExpr::from_integer(4),
            secret_sigma: RealExpr::from_integer(2),
            b_error_sigma: RealExpr::from_integer(1),
            fhe_error_sigma: RealExpr::from_integer(1),
            attribute_error_sigma: RealExpr::from_integer(1),
            security_parameter_bits: 1,
            cascade_randomness_bits: 16,
            gaussian_sample_bits: 16,
            uniform_statistical_bits: 16,
            function: Aky24GoldreichPrf { output_bits: 1, graph_seed: [41; 32] },
        }
    }

    fn zero_veval(input_count: usize) -> PolyCircuit<DCRTPoly> {
        let mut circuit = PolyCircuit::new();
        let _inputs = circuit.input(input_count);
        let zero = circuit.const_zero_gate().as_single_wire();
        circuit.output([zero, zero]);
        circuit
    }

    #[test]
    fn private_prfe_decodes_with_nonzero_sampling_noise() {
        let parameters = DCRTPolyParams::new(2, 1, 6, 1);
        let mut config = config(&parameters);
        config.security_parameter_bits = 1;
        let test_error_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.b_error_sigma = test_error_sigma.clone();
        config.fhe_error_sigma = test_error_sigma.clone();
        config.attribute_error_sigma = test_error_sigma;
        let layer = PrivatePrfeLayerWires::setup(&config, 1).unwrap();
        let ciphertext = layer
            .encrypt(
                vec![layer.compiler.ring.identity(1)],
                vec![layer.compiler.ring.uniform_in((1, 1), 0, 1)],
            )
            .unwrap();
        let input_count = layer.attribute_public.len() - 1;
        let high = layer
            .projection_veval(0, ((parameters.modulus().as_ref() + 1u8) / 2u8).clone())
            .unwrap();
        let low = zero_veval(input_count);
        let key = layer.keygen(&high, &low, &[]).unwrap();
        let decoded = layer.decrypt(&high, &low, &ciphertext, &key, &[]).unwrap();
        let graph = DslContext::new("aky24-private-prfe-nonzero-noise")
            .bool_output("decoded", decoded[0].clone())
            .unwrap()
            .build()
            .unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &validated,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        assert!(matches!(result.outputs["decoded"], RuntimeValue::Bool(true)));
    }

    #[test]
    fn ciphertext_canonical_bits_roundtrip_at_runtime() {
        // This graph serializes every coefficient of every private-prFE
        // attribute. Rust's test harness gives each test thread a small stack,
        // while dropping the resulting deeply shared immutable DAG needs the
        // same stack headroom as a normal application thread.
        std::thread::Builder::new()
            .name("aky24-canonical-ciphertext-roundtrip".to_owned())
            .stack_size(16 * 1024 * 1024)
            .spawn(ciphertext_canonical_bits_roundtrip_at_runtime_inner)
            .unwrap()
            .join()
            .unwrap();
    }

    fn ciphertext_canonical_bits_roundtrip_at_runtime_inner() {
        let parameters = DCRTPolyParams::new(2, 1, 6, 1);
        let mut config = config(&parameters);
        config.security_parameter_bits = 1;
        let layer = PrivatePrfeLayerWires::setup(&config, 1).unwrap();
        let ciphertext = layer
            .encrypt(
                vec![layer.compiler.ring.identity(1)],
                vec![layer.compiler.ring.uniform_in((1, 1), 0, 1)],
            )
            .unwrap();
        let bits = layer.serialize_ciphertext(&ciphertext).unwrap();
        let reconstructed = layer.deserialize_ciphertext(&bits).unwrap();
        let original_attributes = Mat::concat(
            ConcatAxis::Columns,
            ciphertext.attributes.iter().map(|attribute| attribute.vector.clone()).collect(),
        );
        let reconstructed_attributes = Mat::concat(
            ConcatAxis::Columns,
            reconstructed.attributes.iter().map(|attribute| attribute.vector.clone()).collect(),
        );
        let graph = DslContext::new("aky24-private-prfe-ciphertext-bit-roundtrip")
            .output("c-b", ciphertext.c_b)
            .unwrap()
            .output("c-b-roundtrip", reconstructed.c_b)
            .unwrap()
            .output("x", ciphertext.x)
            .unwrap()
            .output("x-roundtrip", reconstructed.x)
            .unwrap()
            .output("attributes", original_attributes)
            .unwrap()
            .output("attributes-roundtrip", reconstructed_attributes)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        assert_runtime_matrices_equal(&result.outputs, "c-b", "c-b-roundtrip");
        assert_runtime_matrices_equal(&result.outputs, "x", "x-roundtrip");
        assert_runtime_matrices_equal(&result.outputs, "attributes", "attributes-roundtrip");
    }

    fn assert_runtime_matrices_equal<B: Backend<Matrix = DCRTPolyMatrix>>(
        outputs: &BTreeMap<String, RuntimeValue<B>>,
        lhs: &str,
        rhs: &str,
    ) {
        let RuntimeValue::Matrix(lhs) = &outputs[lhs] else { panic!("{lhs} must be a matrix") };
        let RuntimeValue::Matrix(rhs) = &outputs[rhs] else { panic!("{rhs} must be a matrix") };
        assert_eq!(lhs.to_compact_bytes(), rhs.to_compact_bytes());
    }

    #[test]
    #[ignore = "canonical-bit GSW execution exceeds six minutes even at the minimum ring size"]
    fn private_prfe_evaluates_a_goldreich_predicate_with_gsw_multiplication() {
        let parameters = DCRTPolyParams::new(1, 1, 2, 1);
        let mut config = config(&parameters);
        config.security_parameter_bits = 1;
        // Keep every Appendix B.1 error tape nonzero while leaving a clear
        // correctness margin for this tiny one-tower runtime test. Production
        // sigmas/moduli are selected by the IR noise search, not this fixture.
        let test_error_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.b_error_sigma = test_error_sigma.clone();
        config.fhe_error_sigma = test_error_sigma.clone();
        config.attribute_error_sigma = test_error_sigma;
        let message = [true, false, true, true, false];
        let graph = GoldreichGraph::generate(
            message.len(),
            1,
            [73; 32],
            GoldreichGraphGeneration::default(),
        );
        let edge = &graph.edges[0];
        let expected = message[edge.xor_inputs[0]] ^
            message[edge.xor_inputs[1]] ^
            message[edge.xor_inputs[2]] ^
            (message[edge.and_inputs[0]] & message[edge.and_inputs[1]]);
        let layer = PrivatePrfeLayerWires::setup(&config, message.len()).unwrap();
        let encoded_message = message
            .into_iter()
            .map(|bit| {
                if bit { layer.compiler.ring.identity(1) } else { layer.compiler.ring.zero((1, 1)) }
            })
            .collect::<Vec<_>>();
        let ciphertext = layer
            .encrypt(encoded_message, vec![layer.compiler.ring.uniform_in((1, 1), 0, 1)])
            .unwrap();
        let input_count = layer.attribute_public.len() - 1;
        let high = layer
            .goldreich_veval(&graph, ((parameters.modulus().as_ref() + 1u8) / 2u8).clone())
            .unwrap();
        let low = zero_veval(input_count);
        let key = layer.keygen(&high, &low, &[]).unwrap();
        let decoded = layer.decrypt(&high, &low, &ciphertext, &key, &[]).unwrap();
        let executable = DslContext::new("aky24-private-prfe-goldreich-gsw")
            .bool_output("decoded", decoded[0].clone())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &executable,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        assert!(
            matches!(&result.outputs["decoded"], RuntimeValue::Bool(actual) if *actual == expected),
            "GSW VEval must match the plaintext Goldreich predicate"
        );
    }

    #[test]
    fn arithmetic_veval_builds_products_with_two_derived_operands() {
        let parameters = DCRTPolyParams::new(2, 1, 6, 1);
        let mut config = config(&parameters);
        config.modulus = BigInt::from(3);
        config.ring_dimension = 1;
        config.gadget_base = BigInt::from(2);
        config.digit_count = 2;
        config.security_parameter_bits = 1;
        let message_bits = 5;
        let layer = PrivatePrfeLayerWires::setup(&config, message_bits).unwrap();

        let mut logical = PolyCircuit::<DCRTPoly>::new();
        let inputs = logical.input(message_bits + 1);
        let left = logical.mul_gate(inputs.at(0), inputs.at(1));
        let right = logical.mul_gate(inputs.at(2), inputs.at(3));
        let product = logical.mul_gate(left, right);
        logical.output([product]);

        let veval = layer.arithmetic_veval(&logical, 0, 2u8.into()).unwrap();
        assert_eq!(veval.num_input(), layer.attribute_public.len() - 1);
        assert_eq!(veval.num_output(), SECRET_ROWS);
        assert!(veval.num_gates() > veval.num_input());
    }

    #[test]
    fn modulus_split_rounds_positive_and_negative_centered_coefficients() {
        let parameters = DCRTPolyParams::new(8, 2, 20, 1);
        let mut config = config(&parameters);
        let split = BigInt::from(parameters.to_crt().0[0]);
        config.modulus_split = split.clone();
        config.validate().unwrap();
        let layer = PrivatePrfeLayerWires::setup(&config, 1).unwrap();
        let input = layer.compiler.ring.input("input", (1, 1));
        let output = layer.round_divide_matrix(input, 1, 1).unwrap();
        let graph = DslContext::new("aky24-private-prfe-modulus-split")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        let modulus = parameters.modulus().as_ref().clone();
        let split = split.to_biguint().unwrap();
        let positive = &split * 3u8 + &split / 4u8;
        let negative_magnitude = &split * 2u8 + &split / 4u8;
        let input = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_biguints(&parameters, &[positive, &modulus - negative_magnitude])],
        );
        let result = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(input))]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(output) = &result.outputs["output"] else {
            panic!("modulus split output must be a matrix")
        };
        let expected = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_biguints(&parameters, &[3u8.into(), &modulus - 2u8])],
        );
        assert_eq!(output.as_ref(), &expected);
    }
}
