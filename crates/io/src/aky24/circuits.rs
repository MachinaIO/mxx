use super::{
    cascade::CascadeLayerPayload,
    config::{Aky24ConfigError, Aky24IoConfig},
};
use digest::Digest;
use keccak_asm::Keccak256;
use mxx_gadgets::{
    Poly,
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    circuit_gadgets::fhe_prg::goldreich::{GoldreichGraph, GoldreichGraphGeneration},
};

/// Fixed-length one-hot encoding of the public Goldreich graph consumed by the
/// specialized universal circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoldreichCircuitDescription {
    pub bits: Vec<bool>,
    pub input_size: usize,
    pub output_bits: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PrescribedGaussianGroup {
    pub coefficients: usize,
    pub sigma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PrescribedCoinLayout {
    /// Direct binary coins: the Appendix B.1 seed `sd` and every coefficient
    /// of the binary GSW randomizer `R`.
    pub binary_bits: usize,
    /// Ring coefficients sampled by interpreting a pseudorandom bit string
    /// modulo `q`, used for `A_bar`.
    pub uniform_coefficients: usize,
    /// PRG bits consumed for each uniform ring coefficient. Callers use the
    /// modulus bit length plus a statistical-security margin.
    pub uniform_sample_bits: usize,
    /// Bits consumed by the explicit bounded inverse-CDF sampler for each
    /// discrete-Gaussian coefficient.
    pub gaussian_sample_bits: usize,
    /// Ordered tapes for `s_bar`, `e_B`, `e_fhe`, and `e_att`.
    pub gaussian_groups: Vec<PrescribedGaussianGroup>,
}

/// Randomized circuit-compatible SKE ciphertext used by Section 3.2.
///
/// The nonce is public and the mask is `GoldreichPRF(key || nonce)`.  Keeping
/// the ciphertext as bits lets the hardwired `F_i` circuit decrypt it with the
/// same Boolean gates that are evaluated by private prFE.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) struct GoldreichSkeCiphertext {
    pub nonce: Vec<bool>,
    pub masked_payload: Vec<bool>,
}

impl PrescribedCoinLayout {
    pub(crate) fn uniform_bits(&self) -> Result<usize, Aky24ConfigError> {
        let uniform = self
            .uniform_coefficients
            .checked_mul(self.uniform_sample_bits)
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let gaussian = self.gaussian_groups.iter().try_fold(0usize, |total, group| {
            group
                .coefficients
                .checked_mul(self.gaussian_sample_bits)
                .and_then(|count| total.checked_add(count))
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)
        })?;
        self.binary_bits
            .checked_add(uniform)
            .and_then(|count| count.checked_add(gaussian))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)
    }

    #[cfg(test)]
    pub(crate) fn output_count(&self) -> Result<usize, Aky24ConfigError> {
        self.gaussian_groups.iter().try_fold(
            self.binary_bits
                .checked_add(self.uniform_coefficients)
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
            |total, group| {
                total.checked_add(group.coefficients).ok_or(Aky24ConfigError::CascadeLayoutOverflow)
            },
        )
    }
}

/// Builds the fixed public Goldreich graph used by the supported AKY24 iO
/// function family.
pub fn goldreich_graph(config: &Aky24IoConfig) -> Result<GoldreichGraph, Aky24ConfigError> {
    config.validate()?;
    Ok(GoldreichGraph::generate(
        config.input_size,
        config.function.output_bits,
        config.function.graph_seed,
        GoldreichGraphGeneration::default(),
    ))
}

pub fn encode_goldreich_circuit(
    config: &Aky24IoConfig,
) -> Result<GoldreichCircuitDescription, Aky24ConfigError> {
    let graph = goldreich_graph(config)?;
    let mut bits = Vec::with_capacity(config.function.output_bits * 5 * config.input_size);
    for edge in &graph.edges {
        for selected in edge.xor_inputs.into_iter().chain(edge.and_inputs) {
            bits.extend((0..config.input_size).map(|index| index == selected));
        }
    }
    Ok(GoldreichCircuitDescription {
        bits,
        input_size: config.input_size,
        output_bits: config.function.output_bits,
    })
}

/// Builds the fixed-size universal circuit used by Section 5.2 for the
/// supported Goldreich family.
///
/// The circuit description is encrypted in the final prMIFE slot. Each of the
/// five source indices of a Goldreich edge is encoded as a one-hot vector. This
/// avoids a generic gate interpreter while still making the topology an input
/// to `U`, rather than hardwiring the obfuscated function into the evaluator
/// key.
pub fn build_goldreich_universal_circuit<P: Poly>(
    config: &Aky24IoConfig,
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    config.validate()?;
    let mut circuit = PolyCircuit::new();
    let description_len = config.function.output_bits * 5 * config.input_size;
    let wires = circuit.input(config.input_size + description_len + config.security_parameter_bits);
    let message = wires.clone().slice(0..config.input_size);
    let description = wires.slice(config.input_size..config.input_size + description_len);
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(config.function.output_bits);
    for _ in 0..config.function.output_bits {
        let mut selected = Vec::with_capacity(5);
        for _ in 0..5 {
            let selector = description.clone().slice(offset..offset + config.input_size);
            offset += config.input_size;
            let terms = (0..config.input_size)
                .map(|index| {
                    circuit.and_gate(message.at(index), selector.at(index)).as_single_wire()
                })
                .collect();
            selected.push(xor_tree(&mut circuit, terms));
        }
        let and = circuit.and_gate(selected[3], selected[4]).as_single_wire();
        outputs.push(xor_tree(&mut circuit, vec![selected[0], selected[1], selected[2], and]));
    }
    circuit.output(outputs);
    Ok(circuit)
}

fn xor_tree<P: Poly>(circuit: &mut PolyCircuit<P>, mut layer: Vec<GateId>) -> GateId {
    assert!(!layer.is_empty(), "a selector must contain at least one candidate");
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut pairs = layer.chunks_exact(2);
        for pair in &mut pairs {
            next.push(circuit.xor_gate(pair[0], pair[1]).as_single_wire());
        }
        next.extend_from_slice(pairs.remainder());
        layer = next;
    }
    layer[0]
}

/// Builds the prescribed-randomness circuit used inside a cascade function
/// `F_i`. The Goldreich output stream is mapped deterministically to the two
/// distributions needed by the private prFE encryption arithmetic:
///
/// - direct binary coins for `sd` and `R`,
/// - modular uniform coefficients for `A_bar`, and
/// - separate bounded inverse-CDF discrete-Gaussian tapes for `s_bar`, `e_B`, `e_fhe`, and `e_att`.
///
/// Both mappings are explicit circuit arithmetic, so key generation and the
/// reverse cascade use identical coins without an out-of-graph sampler.
#[cfg(test)]
pub(crate) fn build_prescribed_coin_circuit<P: Poly>(
    seed_bits: usize,
    layout: &PrescribedCoinLayout,
    graph_seed: [u8; 32],
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    if seed_bits < 5 ||
        layout.uniform_sample_bits == 0 ||
        layout.gaussian_sample_bits == 0 ||
        layout.gaussian_sample_bits > 52 ||
        layout.gaussian_groups.iter().any(|group| group.coefficients == 0 || group.sigma <= 0.0)
    {
        return Err(Aky24ConfigError::NonPositiveParameter);
    }
    let tape_bits = layout.uniform_bits()?;
    if tape_bits == 0 {
        return Err(Aky24ConfigError::NonPositiveParameter);
    }
    let mut circuit = PolyCircuit::new();
    let seeds = circuit.input(seed_bits);
    let uniform = generate_goldreich_stream_from_gates(
        &mut circuit,
        &seeds.gate_ids().collect::<Vec<_>>(),
        tape_bits,
        graph_seed,
    )?;
    let mut cursor = 0;
    let mut outputs = Vec::with_capacity(layout.output_count()?);
    outputs.extend_from_slice(&uniform[..layout.binary_bits]);
    cursor += layout.binary_bits;
    for _ in 0..layout.uniform_coefficients {
        let terms = (0..layout.uniform_sample_bits)
            .map(|bit| {
                let value = uniform[cursor + bit];
                circuit
                    .large_scalar_mul(value, &[num_bigint::BigUint::from(1u8) << bit])
                    .as_single_wire()
            })
            .collect::<Vec<_>>();
        cursor += layout.uniform_sample_bits;
        outputs.push(add_tree(&mut circuit, terms));
    }
    for group in &layout.gaussian_groups {
        for _ in 0..group.coefficients {
            let random_bits = &uniform[cursor..cursor + layout.gaussian_sample_bits];
            cursor += layout.gaussian_sample_bits;
            outputs.push(discrete_gaussian_inverse_cdf(&mut circuit, random_bits, group.sigma));
        }
    }
    assert_eq!(cursor, uniform.len(), "prescribed coin tape must be consumed exactly");
    circuit.output(outputs);
    Ok(circuit)
}

#[cfg(test)]
fn add_tree<P: Poly>(circuit: &mut PolyCircuit<P>, mut layer: Vec<GateId>) -> GateId {
    assert!(!layer.is_empty(), "a CBD half must contain at least one bit");
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

/// Samples one explicitly truncated discrete Gaussian from a fixed random-bit
/// tape. The support is the same 6.5-sigma high-probability envelope used by
/// the simulator. Quantizing its inverse CDF to `2^random_bits.len()` cells
/// makes the mapping deterministic and circuit-evaluable without replacing
/// the Gaussian by a ternary or centered-binomial distribution.
#[cfg(test)]
fn discrete_gaussian_inverse_cdf<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    random_bits: &[GateId],
    sigma: f64,
) -> GateId {
    assert!(!random_bits.is_empty() && random_bits.len() <= 52);
    let (support, thresholds) = discrete_gaussian_thresholds(random_bits.len(), sigma);
    let one = circuit.const_one_gate();
    let mut sample = circuit.large_scalar_mul(one, &[num_bigint::BigUint::from(support as u64)]);
    for threshold in thresholds {
        let below = unsigned_bits_less_than_constant(circuit, random_bits, threshold);
        sample = circuit.sub_gate(sample, below);
    }
    sample.as_single_wire()
}

fn discrete_gaussian_thresholds(random_bits: usize, sigma: f64) -> (i64, Vec<u64>) {
    let support = (6.5 * sigma).ceil() as i64;
    let weights = (-support..=support)
        .map(|value| (-(value as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let cells = 1u64 << random_bits;
    let mut cumulative = 0.0;
    let thresholds = weights[..weights.len() - 1]
        .iter()
        .map(|weight| {
            cumulative += weight / total;
            ((cumulative * cells as f64).round() as u64).clamp(1, cells - 1)
        })
        .collect::<Vec<_>>();
    (support, thresholds)
}

fn unsigned_bits_less_than_constant<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    little_endian_bits: &[GateId],
    threshold: u64,
) -> GateId {
    let mut less = circuit.const_zero_gate();
    let mut equal = circuit.const_one_gate();
    for bit_index in (0..little_endian_bits.len()).rev() {
        let bit = little_endian_bits[bit_index];
        let not_bit = circuit.not_gate(bit);
        if ((threshold >> bit_index) & 1) == 1 {
            let becomes_less = circuit.and_gate(equal, not_bit);
            // `less` and `becomes_less` are mutually exclusive.
            less = circuit.add_gate(less, becomes_less);
            equal = circuit.and_gate(equal, bit);
        } else {
            equal = circuit.and_gate(equal, not_bit);
        }
    }
    less.as_single_wire()
}

/// Canonical little-endian residue used by the coefficient-level cascade
/// circuit. Values stay reduced to `[0, q)`, so the final ciphertext bits are
/// exact and require no non-arithmetic extraction oracle.
#[derive(Clone)]
pub(crate) struct CanonicalResidue {
    bits: Vec<GateId>,
}

impl CanonicalResidue {
    pub(crate) fn from_canonical_bits(
        bits: impl IntoIterator<Item = GateId>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let bits = bits.into_iter().collect::<Vec<_>>();
        assert_eq!(bits.len(), modulus.bits() as usize);
        Self { bits }
    }

    pub(crate) fn bits(&self) -> &[GateId] {
        &self.bits
    }

    pub(crate) fn zero<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let zero = circuit.const_zero_gate().as_single_wire();
        Self { bits: vec![zero; modulus.bits() as usize] }
    }

    pub(crate) fn constant<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        value: &num_bigint::BigUint,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let value = value % modulus;
        let zero = circuit.const_zero_gate().as_single_wire();
        let one = circuit.const_one_gate().as_single_wire();
        Self {
            bits: constant_bits(&value, modulus.bits() as usize)
                .into_iter()
                .map(|bit| if bit { one } else { zero })
                .collect(),
        }
    }

    fn from_unreduced_bits<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        bits: &[GateId],
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let mut value = Self::zero(circuit, modulus);
        let mut power = Self::constant(circuit, &num_bigint::BigUint::from(1u8), modulus);
        for bit in bits {
            let selected = Self {
                bits: power
                    .bits
                    .iter()
                    .map(|value| circuit.and_gate(*bit, *value).as_single_wire())
                    .collect(),
            };
            value = value.add(&selected, circuit, modulus);
            power = power.add(&power, circuit, modulus);
        }
        value
    }

    fn select<P: Poly>(
        condition: GateId,
        if_true: &Self,
        if_false: &Self,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self {
            bits: if_true
                .bits
                .iter()
                .zip(&if_false.bits)
                .map(|(if_true, if_false)| select_bit(circuit, condition, *if_true, *if_false))
                .collect(),
        }
    }

    pub(crate) fn add<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        assert_eq!(self.bits.len(), rhs.bits.len());
        let zero = circuit.const_zero_gate().as_single_wire();
        let (sum, carry) = add_bits(circuit, &self.bits, &rhs.bits, zero);
        Self { bits: reduce_once(circuit, sum, carry, modulus) }
    }

    pub(crate) fn negate<P: Poly>(
        &self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let zero = circuit.const_zero_gate().as_single_wire();
        let modulus_bits = constant_bits(modulus, self.bits.len());
        let (difference, _) = subtract_constant_bits(circuit, &modulus_bits, &self.bits, zero);
        let nonzero = or_reduce(circuit, &self.bits, zero);
        Self {
            bits: difference
                .into_iter()
                .map(|bit| select_bit(circuit, nonzero, bit, zero))
                .collect(),
        }
    }

    pub(crate) fn mul<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let mut result = Self::zero(circuit, modulus);
        let mut doubled = self.clone();
        for bit in &rhs.bits {
            let selected = Self {
                bits: doubled
                    .bits
                    .iter()
                    .map(|value| circuit.and_gate(*bit, *value).as_single_wire())
                    .collect(),
            };
            result = result.add(&selected, circuit, modulus);
            doubled = doubled.add(&doubled, circuit, modulus);
        }
        result
    }
}

fn discrete_gaussian_residue<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    random_bits: &[GateId],
    sigma: f64,
    modulus: &num_bigint::BigUint,
) -> CanonicalResidue {
    let (support, thresholds) = discrete_gaussian_thresholds(random_bits.len(), sigma);
    let mut sample =
        CanonicalResidue::constant(circuit, &num_bigint::BigUint::from(support as u64), modulus);
    let minus_one = CanonicalResidue::constant(circuit, &(modulus - 1u8), modulus);
    let zero = CanonicalResidue::zero(circuit, modulus);
    for threshold in thresholds {
        let below = unsigned_bits_less_than_constant(circuit, random_bits, threshold);
        let decrement = CanonicalResidue::select(below, &minus_one, &zero, circuit);
        sample = sample.add(&decrement, circuit, modulus);
    }
    sample
}

#[derive(Clone)]
pub(crate) struct CanonicalPolynomial {
    coefficients: Vec<CanonicalResidue>,
}

impl CanonicalPolynomial {
    pub(crate) fn from_coefficients(coefficients: Vec<CanonicalResidue>) -> Self {
        Self { coefficients }
    }

    pub(crate) fn from_canonical_bits(
        bits: &[GateId],
        ring_dimension: usize,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let width = modulus.bits() as usize;
        assert_eq!(bits.len(), ring_dimension * width);
        Self {
            coefficients: bits
                .chunks_exact(width)
                .map(|coefficient| {
                    CanonicalResidue::from_canonical_bits(coefficient.iter().copied(), modulus)
                })
                .collect(),
        }
    }

    pub(crate) fn bits(&self) -> impl Iterator<Item = GateId> + '_ {
        self.coefficients.iter().flat_map(|coefficient| coefficient.bits().iter().copied())
    }

    pub(crate) fn ring_dimension(&self) -> usize {
        self.coefficients.len()
    }

    pub(crate) fn coefficient(&self, index: usize) -> &CanonicalResidue {
        &self.coefficients[index]
    }

    pub(crate) fn zero<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        ring_dimension: usize,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        Self {
            coefficients: (0..ring_dimension)
                .map(|_| CanonicalResidue::zero(circuit, modulus))
                .collect(),
        }
    }

    pub(crate) fn constant<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        value: &num_bigint::BigUint,
        ring_dimension: usize,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let mut coefficients = Vec::with_capacity(ring_dimension);
        coefficients.push(CanonicalResidue::constant(circuit, value, modulus));
        coefficients.extend((1..ring_dimension).map(|_| CanonicalResidue::zero(circuit, modulus)));
        Self { coefficients }
    }

    pub(crate) fn add<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        Self {
            coefficients: self
                .coefficients
                .iter()
                .zip(&rhs.coefficients)
                .map(|(lhs, rhs)| lhs.add(rhs, circuit, modulus))
                .collect(),
        }
    }

    pub(crate) fn negate<P: Poly>(
        &self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        Self {
            coefficients: self
                .coefficients
                .iter()
                .map(|coefficient| coefficient.negate(circuit, modulus))
                .collect(),
        }
    }

    pub(crate) fn mul_negacyclic<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let ring_dimension = self.coefficients.len();
        assert_eq!(ring_dimension, rhs.coefficients.len());
        let mut output = (0..ring_dimension)
            .map(|_| CanonicalResidue::zero(circuit, modulus))
            .collect::<Vec<_>>();
        for (lhs_index, lhs) in self.coefficients.iter().enumerate() {
            for (rhs_index, rhs) in rhs.coefficients.iter().enumerate() {
                let raw_index = lhs_index + rhs_index;
                let index = raw_index % ring_dimension;
                let product = lhs.mul(rhs, circuit, modulus);
                let term = if raw_index >= ring_dimension {
                    product.negate(circuit, modulus)
                } else {
                    product
                };
                output[index] = output[index].add(&term, circuit, modulus);
            }
        }
        Self { coefficients: output }
    }

    pub(crate) fn scale_constant<P: Poly>(
        &self,
        scalar: &num_bigint::BigUint,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let scalar = CanonicalResidue::constant(circuit, scalar, modulus);
        Self {
            coefficients: self
                .coefficients
                .iter()
                .map(|coefficient| coefficient.mul(&scalar, circuit, modulus))
                .collect(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct CanonicalMatrix {
    rows: usize,
    columns: usize,
    entries: Vec<CanonicalPolynomial>,
}

impl CanonicalMatrix {
    pub(crate) fn from_entries(
        rows: usize,
        columns: usize,
        entries: Vec<CanonicalPolynomial>,
    ) -> Self {
        assert_eq!(entries.len(), rows * columns);
        Self { rows, columns, entries }
    }

    pub(crate) fn from_canonical_bits(
        bits: &[GateId],
        rows: usize,
        columns: usize,
        ring_dimension: usize,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        let polynomial_bits = ring_dimension * modulus.bits() as usize;
        assert_eq!(bits.len(), rows * columns * polynomial_bits);
        Self {
            rows,
            columns,
            entries: bits
                .chunks_exact(polynomial_bits)
                .map(|polynomial| {
                    CanonicalPolynomial::from_canonical_bits(polynomial, ring_dimension, modulus)
                })
                .collect(),
        }
    }

    pub(crate) fn bits(&self) -> impl Iterator<Item = GateId> + '_ {
        self.entries.iter().flat_map(CanonicalPolynomial::bits)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn entry(&self, row: usize, column: usize) -> &CanonicalPolynomial {
        &self.entries[row * self.columns + column]
    }

    pub(crate) fn add<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        assert_eq!((self.rows, self.columns), (rhs.rows, rhs.columns));
        Self {
            rows: self.rows,
            columns: self.columns,
            entries: self
                .entries
                .iter()
                .zip(&rhs.entries)
                .map(|(lhs, rhs)| lhs.add(rhs, circuit, modulus))
                .collect(),
        }
    }

    pub(crate) fn negate<P: Poly>(
        &self,
        circuit: &mut PolyCircuit<P>,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        Self {
            rows: self.rows,
            columns: self.columns,
            entries: self.entries.iter().map(|entry| entry.negate(circuit, modulus)).collect(),
        }
    }

    pub(crate) fn mul<P: Poly>(
        &self,
        rhs: &Self,
        circuit: &mut PolyCircuit<P>,
        ring_dimension: usize,
        modulus: &num_bigint::BigUint,
    ) -> Self {
        assert_eq!(self.columns, rhs.rows);
        let mut entries = Vec::with_capacity(self.rows * rhs.columns);
        for row in 0..self.rows {
            for column in 0..rhs.columns {
                let mut sum = CanonicalPolynomial::zero(circuit, ring_dimension, modulus);
                for inner in 0..self.columns {
                    let product = self.entry(row, inner).mul_negacyclic(
                        rhs.entry(inner, column),
                        circuit,
                        modulus,
                    );
                    sum = sum.add(&product, circuit, modulus);
                }
                entries.push(sum);
            }
        }
        Self { rows: self.rows, columns: rhs.columns, entries }
    }

    fn concat_rows(values: &[Self]) -> Self {
        assert!(!values.is_empty());
        let columns = values[0].columns;
        assert!(values.iter().all(|value| value.columns == columns));
        Self {
            rows: values.iter().map(|value| value.rows).sum(),
            columns,
            entries: values.iter().flat_map(|value| value.entries.iter().cloned()).collect(),
        }
    }

    fn concat_columns(values: &[Self]) -> Self {
        assert!(!values.is_empty());
        let rows = values[0].rows;
        assert!(values.iter().all(|value| value.rows == rows));
        let columns = values.iter().map(|value| value.columns).sum();
        let mut entries = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for value in values {
                entries.extend((0..value.columns).map(|column| value.entry(row, column).clone()));
            }
        }
        Self { rows, columns, entries }
    }
}

/// Exact coefficient-level Appendix B.1 encryption used inside one cascade
/// function. `message` and `prf_domain` are hidden circuit inputs. The two
/// public slices are canonical coefficient bits of the target layer's `B`
/// and ordered `A_att` matrices, supplied through mixed public inputs.
#[cfg(test)]
pub(crate) fn build_private_prfe_encryption_circuit<P: Poly>(
    config: &Aky24IoConfig,
    plaintext_bits: usize,
    prf_input_bits: usize,
    coin_layout: &PrescribedCoinLayout,
    graph_seed: [u8; 32],
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    let modulus = config
        .modulus
        .to_biguint()
        .filter(|modulus| *modulus > num_bigint::BigUint::from(1u8))
        .ok_or(Aky24ConfigError::NonPositiveParameter)?;
    let coefficient_bits = modulus.bits() as usize;
    let polynomial_bits = config
        .ring_dimension
        .checked_mul(coefficient_bits)
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let gsw_columns =
        2usize.checked_mul(config.digit_count).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let x_columns = gsw_columns
        .checked_mul(
            plaintext_bits
                .checked_add(config.security_parameter_bits)
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
        )
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let attribute_count = 2usize
        .checked_mul(x_columns)
        .and_then(|count| count.checked_mul(config.ring_dimension))
        .and_then(|count| count.checked_mul(coefficient_bits))
        .and_then(|count| count.checked_add(1))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let b_public_bits = 2usize
        .checked_mul(2 * (config.digit_count + 2))
        .and_then(|count| count.checked_mul(polynomial_bits))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let attribute_public_bits = attribute_count
        .checked_mul(2 * gsw_columns)
        .and_then(|count| count.checked_mul(polynomial_bits))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let mut circuit = PolyCircuit::new();
    let message = circuit.input(plaintext_bits);
    let prf_domain = circuit.input(prf_input_bits);
    let b_public = circuit.input(b_public_bits);
    let attribute_public = circuit.input(attribute_public_bits);
    let output = encrypt_private_prfe_in_circuit(
        &mut circuit,
        config,
        &message.gate_ids().collect::<Vec<_>>(),
        &prf_domain.gate_ids().collect::<Vec<_>>(),
        &b_public.gate_ids().collect::<Vec<_>>(),
        &attribute_public.gate_ids().collect::<Vec<_>>(),
        coin_layout,
        graph_seed,
    )?;
    circuit.output(output);
    Ok(circuit)
}

pub(crate) fn encrypt_private_prfe_in_circuit<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    config: &Aky24IoConfig,
    message: &[GateId],
    prf_domain: &[GateId],
    b_public_bits: &[GateId],
    attribute_public_bits: &[GateId],
    coin_layout: &PrescribedCoinLayout,
    graph_seed: [u8; 32],
) -> Result<Vec<GateId>, Aky24ConfigError> {
    let modulus = config
        .modulus
        .to_biguint()
        .filter(|modulus| *modulus > num_bigint::BigUint::from(1u8))
        .ok_or(Aky24ConfigError::NonPositiveParameter)?;
    let coefficient_bits = modulus.bits() as usize;
    let ring_dimension = config.ring_dimension;
    let gsw_columns =
        2usize.checked_mul(config.digit_count).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let x_columns = gsw_columns
        .checked_mul(
            message
                .len()
                .checked_add(config.security_parameter_bits)
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
        )
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let attribute_count = 2usize
        .checked_mul(x_columns)
        .and_then(|count| count.checked_mul(ring_dimension))
        .and_then(|count| count.checked_mul(coefficient_bits))
        .and_then(|count| count.checked_add(1))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let polynomial_bits = ring_dimension
        .checked_mul(coefficient_bits)
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let b_columns = 2usize
        .checked_mul(config.digit_count + 2)
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let expected_binary_bits = config
        .security_parameter_bits
        .checked_add(
            gsw_columns
                .checked_mul(x_columns)
                .and_then(|count| count.checked_mul(ring_dimension))
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
        )
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let checked_coefficients = |rows: usize, columns: usize| {
        rows.checked_mul(columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)
    };
    let expected_gaussian_coefficients = [
        ring_dimension,
        checked_coefficients(1, b_columns)?,
        checked_coefficients(1, gsw_columns)?,
        checked_coefficients(attribute_count, gsw_columns)?,
    ];
    if b_public_bits.len() != 2 * b_columns * polynomial_bits ||
        attribute_public_bits.len() != attribute_count * 2 * gsw_columns * polynomial_bits ||
        coin_layout.binary_bits != expected_binary_bits ||
        coin_layout.uniform_coefficients != checked_coefficients(1, gsw_columns)? ||
        coin_layout.gaussian_groups.len() != 4 ||
        !coin_layout
            .gaussian_groups
            .iter()
            .map(|group| group.coefficients)
            .eq(expected_gaussian_coefficients)
    {
        return Err(Aky24ConfigError::CascadeLayoutOverflow);
    }

    let tape = generate_goldreich_stream_from_gates(
        circuit,
        prf_domain,
        coin_layout.uniform_bits()?,
        graph_seed,
    )?;
    let mut cursor = 0usize;
    let seed_end = cursor
        .checked_add(config.security_parameter_bits)
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let seed = tape.get(cursor..seed_end).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?.to_vec();
    cursor = seed_end;
    let r_polynomials =
        gsw_columns.checked_mul(x_columns).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let r_bits =
        r_polynomials.checked_mul(ring_dimension).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let r_end = cursor.checked_add(r_bits).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let r = binary_polynomial_matrix(
        circuit,
        tape.get(cursor..r_end).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
        gsw_columns,
        x_columns,
        ring_dimension,
        &modulus,
    );
    cursor = r_end;
    let a_bar = uniform_polynomial_matrix(
        circuit,
        &tape,
        &mut cursor,
        1,
        gsw_columns,
        ring_dimension,
        coin_layout.uniform_sample_bits,
        &modulus,
    )?;
    let gaussian_shapes =
        [(1, 1), (1, b_columns), (1, gsw_columns), (attribute_count, gsw_columns)];
    let mut gaussian = coin_layout
        .gaussian_groups
        .iter()
        .zip(gaussian_shapes)
        .map(|(group, (rows, columns))| {
            gaussian_polynomial_matrix(
                circuit,
                &tape,
                &mut cursor,
                rows,
                columns,
                ring_dimension,
                coin_layout.gaussian_sample_bits,
                group.sigma,
                &modulus,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cursor != tape.len() {
        return Err(Aky24ConfigError::CascadeLayoutOverflow);
    }
    let e_att = gaussian.pop().expect("four Gaussian groups");
    let e_fhe = gaussian.pop().expect("four Gaussian groups");
    let e_b = gaussian.pop().expect("four Gaussian groups");
    let s_bar = gaussian.pop().expect("four Gaussian groups");

    let b_public =
        CanonicalMatrix::from_canonical_bits(b_public_bits, 2, b_columns, ring_dimension, &modulus);
    let public_attributes = attribute_public_bits
        .chunks_exact(2 * gsw_columns * polynomial_bits)
        .map(|bits| {
            CanonicalMatrix::from_canonical_bits(bits, 2, gsw_columns, ring_dimension, &modulus)
        })
        .collect::<Vec<_>>();
    let minus_one =
        CanonicalPolynomial::constant(circuit, &(&modulus - 1u8), ring_dimension, &modulus);
    let secret = CanonicalMatrix::concat_columns(&[
        s_bar.clone(),
        CanonicalMatrix::from_entries(1, 1, vec![minus_one]),
    ]);
    let lower_a =
        s_bar.mul(&a_bar, circuit, ring_dimension, &modulus).add(&e_fhe, circuit, &modulus);
    let a_fhe = CanonicalMatrix::concat_rows(&[a_bar, lower_a]);
    let logical_bits = message.iter().copied().chain(seed).collect::<Vec<_>>();
    let plaintext_gadget =
        binary_message_gadget(circuit, &logical_bits, config.digit_count, ring_dimension, &modulus);
    let x = a_fhe.mul(&r, circuit, ring_dimension, &modulus).add(
        &plaintext_gadget.negate(circuit, &modulus),
        circuit,
        &modulus,
    );
    let c_b = secret.mul(&b_public, circuit, ring_dimension, &modulus).add(&e_b, circuit, &modulus);
    let attributes = std::iter::once(CanonicalPolynomial::constant(
        circuit,
        &num_bigint::BigUint::from(1u8),
        ring_dimension,
        &modulus,
    ))
    .chain(decompose_canonical_matrix(circuit, &x, coefficient_bits, &modulus))
    .zip(public_attributes)
    .zip(e_att.entries.chunks_exact(gsw_columns))
    .map(|((attribute, public), error_entries)| {
        let gadget =
            polynomial_gadget(circuit, &attribute, config.digit_count, ring_dimension, &modulus);
        let shifted_public = public.add(&gadget.negate(circuit, &modulus), circuit, &modulus);
        let error = CanonicalMatrix::from_entries(1, gsw_columns, error_entries.to_vec());
        secret
            .mul(&shifted_public, circuit, ring_dimension, &modulus)
            .add(&error, circuit, &modulus)
    })
    .collect::<Vec<_>>();

    Ok(c_b
        .bits()
        .chain(x.bits())
        .chain(attributes.iter().flat_map(CanonicalMatrix::bits))
        .collect())
}

/// Builds the exact hardwired cascade function `F_i` from Figure 1.
///
/// Logical inputs are ordered as the hidden layer-`i+1` payload followed by
/// public `SKE.ct_i.nonce`, `SKE.ct_i.masked_payload`, `r_i`, and the exact
/// canonical bits of `prFE_i.mpk`. The output is the canonical serialization
/// of `prFE_i.ct`.
pub(crate) fn build_cascade_function_circuit<P: Poly>(
    config: &Aky24IoConfig,
    source: &CascadeLayerPayload,
    target: &CascadeLayerPayload,
    target_public_key_bits: usize,
    coin_layout: &PrescribedCoinLayout,
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    if source.layer != target.layer + 1 || target.layer == 0 {
        return Err(Aky24ConfigError::CascadeLayoutOverflow);
    }
    let lambda = config.cascade_randomness_bits;
    let public_bits = lambda
        .checked_add(1)
        .and_then(|count| count.checked_add(lambda))
        .and_then(|count| count.checked_add(target_public_key_bits))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let mut circuit = PolyCircuit::new();
    let hidden_all = circuit.input(source.bit_len + config.security_parameter_bits);
    let hidden = hidden_all.slice(0..source.bit_len);
    let public = circuit.input(public_bits);
    let nonce = public.clone().slice(0..lambda);
    let masked_payload = public.clone().slice(lambda..lambda + 1);
    let randomness = public.clone().slice(lambda + 1..2 * lambda + 1);
    let public_key = public.slice(2 * lambda + 1..public_bits);
    let ske_key_range = source.ske_key.clone().ok_or(Aky24ConfigError::SkeLayout)?;
    let recovered = goldreich_ske_decrypt_wires(
        &mut circuit,
        hidden.clone().slice(ske_key_range),
        nonce,
        masked_payload,
        config.function.graph_seed,
    )?;
    if recovered.len() != 1 {
        return Err(Aky24ConfigError::SkeLayout);
    }

    let mut message = vec![None; target.bit_len];
    let mut assign = |range: std::ops::Range<usize>, values: Vec<GateId>| {
        if range.len() != values.len() || range.end > message.len() {
            return Err(Aky24ConfigError::CascadeLayoutOverflow);
        }
        for (slot, value) in message[range].iter_mut().zip(values) {
            *slot = Some(value);
        }
        Ok(())
    };
    if let Some(target_ske) = &target.ske_key {
        let source_ske = source.ske_key.clone().ok_or(Aky24ConfigError::SkeLayout)?;
        assign(target_ske.clone(), hidden.clone().slice(source_ske).gate_ids().collect())?;
    }
    for (slot, target_range) in &target.slot_values {
        if *slot == target.layer {
            assign(target_range.clone(), recovered.clone())?;
        } else {
            let source_range = source
                .slot_values
                .iter()
                .find_map(|(candidate, range)| (*candidate == *slot).then(|| range.clone()))
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
            assign(target_range.clone(), hidden.clone().slice(source_range).gate_ids().collect())?;
        }
    }
    for (slot, target_range) in &target.randomness {
        if *slot == target.layer {
            assign(target_range.clone(), randomness.gate_ids().collect())?;
        } else {
            let source_range = source
                .randomness
                .iter()
                .find_map(|(candidate, range)| (*candidate == *slot).then(|| range.clone()))
                .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
            assign(target_range.clone(), hidden.clone().slice(source_range).gate_ids().collect())?;
        }
    }
    for (key, target_range) in &target.prf_keys {
        let source_range = source
            .prf_keys
            .iter()
            .find_map(|(candidate, range)| (*candidate == *key).then(|| range.clone()))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        assign(target_range.clone(), hidden.clone().slice(source_range).gate_ids().collect())?;
    }
    let message = message
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;

    let key_range = source
        .prf_keys
        .iter()
        .find_map(|(key, range)| (*key == target.layer).then(|| range.clone()))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let mut prf_domain = hidden.clone().slice(key_range).gate_ids().collect::<Vec<_>>();
    prf_domain.extend(randomness.gate_ids());
    for (slot, range) in &source.randomness {
        if *slot > target.layer {
            prf_domain.extend(hidden.clone().slice(range.clone()).gate_ids());
        }
    }

    let modulus_bits = config.modulus.bits() as usize;
    let polynomial_bits = config
        .ring_dimension
        .checked_mul(modulus_bits)
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let b_public_bits = 2usize
        .checked_mul(2 * (config.digit_count + 2))
        .and_then(|count| count.checked_mul(polynomial_bits))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    if target_public_key_bits < b_public_bits {
        return Err(Aky24ConfigError::CascadeLayoutOverflow);
    }
    let output = encrypt_private_prfe_in_circuit(
        &mut circuit,
        config,
        &message,
        &prf_domain,
        &public_key.clone().slice(0..b_public_bits).gate_ids().collect::<Vec<_>>(),
        &public_key.slice(b_public_bits..target_public_key_bits).gate_ids().collect::<Vec<_>>(),
        coin_layout,
        config.function.graph_seed,
    )?;
    circuit.output(output);
    Ok(circuit)
}

fn binary_polynomial_matrix<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    bits: &[GateId],
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    modulus: &num_bigint::BigUint,
) -> CanonicalMatrix {
    assert_eq!(bits.len(), rows * columns * ring_dimension);
    let width = modulus.bits() as usize;
    let zero = circuit.const_zero_gate().as_single_wire();
    CanonicalMatrix::from_entries(
        rows,
        columns,
        bits.chunks_exact(ring_dimension)
            .map(|coefficients| CanonicalPolynomial {
                coefficients: coefficients
                    .iter()
                    .map(|bit| {
                        let mut bits = vec![zero; width];
                        bits[0] = *bit;
                        CanonicalResidue::from_canonical_bits(bits, modulus)
                    })
                    .collect(),
            })
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn uniform_polynomial_matrix<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    tape: &[GateId],
    cursor: &mut usize,
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    sample_bits: usize,
    modulus: &num_bigint::BigUint,
) -> Result<CanonicalMatrix, Aky24ConfigError> {
    sampled_polynomial_matrix(
        circuit,
        tape,
        cursor,
        rows,
        columns,
        ring_dimension,
        sample_bits,
        |circuit, bits| CanonicalResidue::from_unreduced_bits(circuit, bits, modulus),
    )
}

#[allow(clippy::too_many_arguments)]
fn gaussian_polynomial_matrix<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    tape: &[GateId],
    cursor: &mut usize,
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    sample_bits: usize,
    sigma: f64,
    modulus: &num_bigint::BigUint,
) -> Result<CanonicalMatrix, Aky24ConfigError> {
    sampled_polynomial_matrix(
        circuit,
        tape,
        cursor,
        rows,
        columns,
        ring_dimension,
        sample_bits,
        |circuit, bits| discrete_gaussian_residue(circuit, bits, sigma, modulus),
    )
}

#[allow(clippy::too_many_arguments)]
fn sampled_polynomial_matrix<P: Poly, F>(
    circuit: &mut PolyCircuit<P>,
    tape: &[GateId],
    cursor: &mut usize,
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    sample_bits: usize,
    mut sample: F,
) -> Result<CanonicalMatrix, Aky24ConfigError>
where
    F: FnMut(&mut PolyCircuit<P>, &[GateId]) -> CanonicalResidue,
{
    let sample_count = rows
        .checked_mul(columns)
        .and_then(|count| count.checked_mul(ring_dimension))
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let end = cursor
        .checked_add(
            sample_count.checked_mul(sample_bits).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?,
        )
        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    let source = tape.get(*cursor..end).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    *cursor = end;
    Ok(CanonicalMatrix::from_entries(
        rows,
        columns,
        source
            .chunks_exact(sample_bits * ring_dimension)
            .map(|polynomial| CanonicalPolynomial {
                coefficients: polynomial
                    .chunks_exact(sample_bits)
                    .map(|bits| sample(circuit, bits))
                    .collect(),
            })
            .collect(),
    ))
}

fn binary_message_gadget<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    bits: &[GateId],
    digit_count: usize,
    ring_dimension: usize,
    modulus: &num_bigint::BigUint,
) -> CanonicalMatrix {
    let zero = CanonicalPolynomial::zero(circuit, ring_dimension, modulus);
    let mut entries = Vec::with_capacity(2 * bits.len() * 2 * digit_count);
    for row in 0..2 {
        for bit in bits {
            for gadget_row in 0..2 {
                for digit in 0..digit_count {
                    if row == gadget_row {
                        let scalar = CanonicalResidue::from_canonical_bits(
                            std::iter::once(*bit).chain(std::iter::repeat_n(
                                circuit.const_zero_gate().as_single_wire(),
                                modulus.bits() as usize - 1,
                            )),
                            modulus,
                        );
                        let scalar = scalar.mul(
                            &CanonicalResidue::constant(
                                circuit,
                                &(num_bigint::BigUint::from(1u8) << digit),
                                modulus,
                            ),
                            circuit,
                            modulus,
                        );
                        let mut coefficients = Vec::with_capacity(ring_dimension);
                        coefficients.push(scalar);
                        coefficients.extend(
                            (1..ring_dimension).map(|_| CanonicalResidue::zero(circuit, modulus)),
                        );
                        entries.push(CanonicalPolynomial { coefficients });
                    } else {
                        entries.push(zero.clone());
                    }
                }
            }
        }
    }
    CanonicalMatrix::from_entries(2, bits.len() * 2 * digit_count, entries)
}

fn polynomial_gadget<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    value: &CanonicalPolynomial,
    digit_count: usize,
    ring_dimension: usize,
    modulus: &num_bigint::BigUint,
) -> CanonicalMatrix {
    let zero = CanonicalPolynomial::zero(circuit, ring_dimension, modulus);
    let mut entries = Vec::with_capacity(4 * digit_count);
    for row in 0..2 {
        for gadget_row in 0..2 {
            for digit in 0..digit_count {
                entries.push(if row == gadget_row {
                    value.scale_constant(
                        &(num_bigint::BigUint::from(1u8) << digit),
                        circuit,
                        modulus,
                    )
                } else {
                    zero.clone()
                });
            }
        }
    }
    CanonicalMatrix::from_entries(2, 2 * digit_count, entries)
}

fn decompose_canonical_matrix<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    matrix: &CanonicalMatrix,
    coefficient_bits: usize,
    modulus: &num_bigint::BigUint,
) -> Vec<CanonicalPolynomial> {
    let zero = circuit.const_zero_gate().as_single_wire();
    let mut entries = Vec::with_capacity(
        matrix.entries.len() * matrix.entries[0].coefficients.len() * coefficient_bits,
    );
    for entry in &matrix.entries {
        for coefficient in &entry.coefficients {
            for bit in 0..coefficient_bits {
                let mut scalar_bits = vec![zero; coefficient_bits];
                scalar_bits[0] = coefficient.bits[bit];
                let mut coefficients = Vec::with_capacity(entry.coefficients.len());
                coefficients.push(CanonicalResidue::from_canonical_bits(scalar_bits, modulus));
                coefficients.extend(
                    (1..entry.coefficients.len()).map(|_| CanonicalResidue::zero(circuit, modulus)),
                );
                entries.push(CanonicalPolynomial { coefficients });
            }
        }
    }
    entries
}

fn reduce_once<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    value: Vec<GateId>,
    carry: GateId,
    modulus: &num_bigint::BigUint,
) -> Vec<GateId> {
    let zero = circuit.const_zero_gate().as_single_wire();
    let one = circuit.const_one_gate().as_single_wire();
    let modulus_bits = constant_bits(modulus, value.len());
    let less = bits_less_than_constant(circuit, &value, &modulus_bits, zero, one);
    let ge_without_carry = circuit.not_gate(less).as_single_wire();
    let reduce = or_bit(circuit, carry, ge_without_carry);
    let (difference, _) = subtract_bits_constant(circuit, &value, &modulus_bits, zero);
    value
        .into_iter()
        .zip(difference)
        .map(|(original, reduced)| select_bit(circuit, reduce, reduced, original))
        .collect()
}

fn constant_bits(value: &num_bigint::BigUint, width: usize) -> Vec<bool> {
    (0..width).map(|bit| value.bit(bit as u64)).collect()
}

fn add_bits<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    lhs: &[GateId],
    rhs: &[GateId],
    mut carry: GateId,
) -> (Vec<GateId>, GateId) {
    let mut output = Vec::with_capacity(lhs.len());
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let lhs_xor_rhs = circuit.xor_gate(lhs, rhs).as_single_wire();
        output.push(circuit.xor_gate(lhs_xor_rhs, carry).as_single_wire());
        let both = circuit.and_gate(lhs, rhs).as_single_wire();
        let carry_one = circuit.and_gate(carry, lhs_xor_rhs).as_single_wire();
        carry = or_bit(circuit, both, carry_one);
    }
    (output, carry)
}

fn subtract_constant_bits<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    lhs_constant: &[bool],
    rhs: &[GateId],
    mut borrow: GateId,
) -> (Vec<GateId>, GateId) {
    let zero = circuit.const_zero_gate().as_single_wire();
    let one = circuit.const_one_gate().as_single_wire();
    let mut output = Vec::with_capacity(rhs.len());
    for (&lhs, &rhs) in lhs_constant.iter().zip(rhs) {
        let lhs = if lhs { one } else { zero };
        let lhs_xor_rhs = circuit.xor_gate(lhs, rhs).as_single_wire();
        output.push(circuit.xor_gate(lhs_xor_rhs, borrow).as_single_wire());
        let not_lhs = circuit.not_gate(lhs).as_single_wire();
        let first = circuit.and_gate(not_lhs, rhs).as_single_wire();
        let equal = circuit.not_gate(lhs_xor_rhs).as_single_wire();
        let second = circuit.and_gate(equal, borrow).as_single_wire();
        borrow = or_bit(circuit, first, second);
    }
    (output, borrow)
}

fn subtract_bits_constant<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    lhs: &[GateId],
    rhs_constant: &[bool],
    mut borrow: GateId,
) -> (Vec<GateId>, GateId) {
    let zero = circuit.const_zero_gate().as_single_wire();
    let one = circuit.const_one_gate().as_single_wire();
    let mut output = Vec::with_capacity(lhs.len());
    for (&lhs, &rhs) in lhs.iter().zip(rhs_constant) {
        let rhs = if rhs { one } else { zero };
        let lhs_xor_rhs = circuit.xor_gate(lhs, rhs).as_single_wire();
        output.push(circuit.xor_gate(lhs_xor_rhs, borrow).as_single_wire());
        let not_lhs = circuit.not_gate(lhs).as_single_wire();
        let first = circuit.and_gate(not_lhs, rhs).as_single_wire();
        let equal = circuit.not_gate(lhs_xor_rhs).as_single_wire();
        let second = circuit.and_gate(equal, borrow).as_single_wire();
        borrow = or_bit(circuit, first, second);
    }
    (output, borrow)
}

fn bits_less_than_constant<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    value: &[GateId],
    constant: &[bool],
    mut less: GateId,
    mut equal: GateId,
) -> GateId {
    for (&bit, &constant) in value.iter().zip(constant).rev() {
        let not_bit = circuit.not_gate(bit).as_single_wire();
        if constant {
            let becomes_less = circuit.and_gate(equal, not_bit).as_single_wire();
            less = or_bit(circuit, less, becomes_less);
            equal = circuit.and_gate(equal, bit).as_single_wire();
        } else {
            equal = circuit.and_gate(equal, not_bit).as_single_wire();
        }
    }
    less
}

fn or_bit<P: Poly>(circuit: &mut PolyCircuit<P>, lhs: GateId, rhs: GateId) -> GateId {
    let xor = circuit.xor_gate(lhs, rhs).as_single_wire();
    let both = circuit.and_gate(lhs, rhs).as_single_wire();
    circuit.xor_gate(xor, both).as_single_wire()
}

fn or_reduce<P: Poly>(circuit: &mut PolyCircuit<P>, bits: &[GateId], zero: GateId) -> GateId {
    bits.iter().fold(zero, |result, bit| or_bit(circuit, result, *bit))
}

fn select_bit<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    condition: GateId,
    if_true: GateId,
    if_false: GateId,
) -> GateId {
    let difference = circuit.xor_gate(if_true, if_false).as_single_wire();
    let selected = circuit.and_gate(condition, difference).as_single_wire();
    circuit.xor_gate(if_false, selected).as_single_wire()
}

/// Encrypts one fixed-length bit string with the circuit-compatible SKE used
/// by the AKY24 cascade.
#[cfg(test)]
pub(crate) fn goldreich_ske_encrypt(
    key: &[bool],
    nonce: Vec<bool>,
    plaintext: &[bool],
    graph_seed: [u8; 32],
) -> Result<GoldreichSkeCiphertext, Aky24ConfigError> {
    if key.is_empty() || nonce.len() != key.len() || plaintext.is_empty() {
        return Err(Aky24ConfigError::SkeLayout);
    }
    let seed_len = key.len().checked_add(nonce.len()).ok_or(Aky24ConfigError::SkeLayout)?;
    if seed_len < 5 ||
        !mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::goldreich_output_bound_holds(
            seed_len,
            plaintext.len(),
        )
    {
        return Err(Aky24ConfigError::GoldreichOutputBound);
    }
    let graph = GoldreichGraph::generate(
        seed_len,
        plaintext.len(),
        graph_seed,
        GoldreichGraphGeneration::default(),
    );
    let seed = key.iter().copied().chain(nonce.iter().copied()).collect::<Vec<_>>();
    let mask = evaluate_goldreich_bits(&graph, &seed);
    let masked_payload = plaintext.iter().zip(mask).map(|(bit, mask)| *bit ^ mask).collect();
    Ok(GoldreichSkeCiphertext { nonce, masked_payload })
}

#[cfg(test)]
fn evaluate_goldreich_bits(graph: &GoldreichGraph, input: &[bool]) -> Vec<bool> {
    assert_eq!(graph.input_size, input.len(), "Goldreich input layout must be exact");
    graph
        .edges
        .iter()
        .map(|edge| {
            input[edge.xor_inputs[0]] ^
                input[edge.xor_inputs[1]] ^
                input[edge.xor_inputs[2]] ^
                (input[edge.and_inputs[0]] & input[edge.and_inputs[1]])
        })
        .collect()
}

pub(crate) fn goldreich_ske_decrypt_wires<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    key: BatchedWire,
    nonce: BatchedWire,
    masked_payload: BatchedWire,
    graph_seed: [u8; 32],
) -> Result<Vec<GateId>, Aky24ConfigError> {
    if key.is_empty() || nonce.len() != key.len() || masked_payload.is_empty() {
        return Err(Aky24ConfigError::SkeLayout);
    }
    let seed_len = key.len().checked_add(nonce.len()).ok_or(Aky24ConfigError::SkeLayout)?;
    if !mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::goldreich_output_bound_holds(
        seed_len,
        masked_payload.len(),
    ) {
        return Err(Aky24ConfigError::GoldreichOutputBound);
    }
    let mut seed = key.gate_ids().collect::<Vec<_>>();
    seed.extend(nonce.gate_ids());
    let mask =
        generate_goldreich_wires_from_gates(circuit, &seed, masked_payload.len(), graph_seed);
    Ok(masked_payload
        .gate_ids()
        .zip(mask)
        .map(|(ciphertext_bit, mask)| circuit.xor_gate(ciphertext_bit, mask).as_single_wire())
        .collect())
}

/// Circuit form of SKE decryption used by cascade functions. Inputs are the
/// hidden key bits, followed by dynamic public nonce bits and dynamic public
/// masked-payload bits. Sampled values are never embedded in the graph spec.
#[cfg(test)]
pub(crate) fn build_goldreich_ske_decryption_circuit<P: Poly>(
    key_bits: usize,
    plaintext_bits: usize,
    graph_seed: [u8; 32],
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    if key_bits == 0 || plaintext_bits == 0 {
        return Err(Aky24ConfigError::SkeLayout);
    }
    let mut circuit = PolyCircuit::new();
    let key = circuit.input(key_bits);
    let nonce = circuit.input(key_bits);
    let masked_payload = circuit.input(plaintext_bits);
    let plaintext =
        goldreich_ske_decrypt_wires(&mut circuit, key, nonce, masked_payload, graph_seed)?;
    circuit.output(plaintext);
    Ok(circuit)
}

fn generate_goldreich_wires_from_gates<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    seeds: &[GateId],
    output_bits: usize,
    graph_seed: [u8; 32],
) -> Vec<GateId> {
    let graph = GoldreichGraph::generate(
        seeds.len(),
        output_bits,
        graph_seed,
        GoldreichGraphGeneration::default(),
    );
    graph
        .edges
        .iter()
        .map(|edge| {
            let and = circuit
                .and_gate(seeds[edge.and_inputs[0]], seeds[edge.and_inputs[1]])
                .as_single_wire();
            xor_tree(
                circuit,
                vec![
                    seeds[edge.xor_inputs[0]],
                    seeds[edge.xor_inputs[1]],
                    seeds[edge.xor_inputs[2]],
                    and,
                ],
            )
        })
        .collect()
}

fn goldreich_stream_round_seed(base_seed: [u8; 32], round: usize) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(b"mxx/AKY24/iterated-Goldreich-stream/v1");
    hasher.update(base_seed);
    hasher.update(round.to_le_bytes());
    hasher.finalize().into()
}

/// Expands an arbitrary-length tape while keeping every individual Goldreich
/// graph inside `m < n^1.4`. Each round maps `d` state bits to `2d` bits,
/// retains the first half as the next state, and appends the second half to
/// the output tape. AKY24 cascade domains have at least six bits.
fn generate_goldreich_stream_from_gates<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    initial_state: &[GateId],
    output_bits: usize,
    graph_seed: [u8; 32],
) -> Result<Vec<GateId>, Aky24ConfigError> {
    if initial_state.len() < 6 || output_bits == 0 {
        return Err(Aky24ConfigError::GoldreichOutputBound);
    }
    let state_bits = initial_state.len();
    if !mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::goldreich_output_bound_holds(
        state_bits,
        2 * state_bits,
    ) {
        return Err(Aky24ConfigError::GoldreichOutputBound);
    }
    let mut state = initial_state.to_vec();
    let mut output = Vec::with_capacity(output_bits);
    let mut round = 0usize;
    while output.len() < output_bits {
        let expanded = generate_goldreich_wires_from_gates(
            circuit,
            &state,
            2 * state_bits,
            goldreich_stream_round_seed(graph_seed, round),
        );
        state = expanded[..state_bits].to_vec();
        let remaining = output_bits - output.len();
        output.extend_from_slice(&expanded[state_bits..state_bits + remaining.min(state_bits)]);
        round = round.checked_add(1).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
    }
    Ok(output)
}

/// Specializes the universal-circuit payload to the one public function family
/// currently supported by mxx.
///
/// This only removes generic circuit parsing from the executable circuit.  The
/// Section 5.2 protocol still requires a distinct final prMIFE ciphertext that
/// binds this circuit payload; the cascade must not omit that last slot.
#[cfg(test)]
pub fn build_goldreich_function_circuit<P: Poly>(
    config: &Aky24IoConfig,
) -> Result<PolyCircuit<P>, Aky24ConfigError> {
    let graph = goldreich_graph(config)?;
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(config.input_size);
    let outputs = graph
        .edges
        .iter()
        .map(|edge| {
            let and =
                circuit.and_gate(inputs.at(edge.and_inputs[0]), inputs.at(edge.and_inputs[1]));
            let left =
                circuit.xor_gate(inputs.at(edge.xor_inputs[0]), inputs.at(edge.xor_inputs[1]));
            let right = circuit.xor_gate(inputs.at(edge.xor_inputs[2]), and.as_single_wire());
            circuit.xor_gate(left, right).as_single_wire()
        })
        .collect::<Vec<_>>();
    circuit.output(outputs);
    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aky24::{cascade::CascadeLayerPayload, prfe::PrivatePrfeLayerWires};
    use mxx_gadgets::circuit::PolyGateType;
    use mxx_ir_core::RealExpr;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn config() -> Aky24IoConfig {
        Aky24IoConfig {
            modulus: 257.into(),
            ring_dimension: 8,
            input_size: 8,
            gadget_base: 2.into(),
            digit_count: 9,
            modulus_split: 1.into(),
            trapdoor_sigma: RealExpr::from_integer(4),
            secret_sigma: RealExpr::from_integer(2),
            b_error_sigma: RealExpr::from_integer(1),
            fhe_error_sigma: RealExpr::from_integer(1),
            attribute_error_sigma: RealExpr::from_integer(1),
            security_parameter_bits: 128,
            cascade_randomness_bits: 128,
            gaussian_sample_bits: 16,
            uniform_statistical_bits: 16,
            function: super::super::config::Aky24GoldreichPrf {
                output_bits: 2,
                graph_seed: [7; 32],
            },
        }
    }

    #[test]
    fn specialized_circuit_keeps_the_public_goldreich_shape() {
        let circuit = build_goldreich_function_circuit::<DCRTPoly>(&config()).unwrap();
        assert_eq!(circuit.num_input(), 8);
        assert_eq!(circuit.num_output(), 2);
    }

    #[test]
    fn universal_circuit_consumes_the_encrypted_one_hot_description() {
        let config = config();
        let description = encode_goldreich_circuit(&config).unwrap();
        let circuit = build_goldreich_universal_circuit::<DCRTPoly>(&config).unwrap();
        assert_eq!(description.bits.len(), config.function.output_bits * 5 * config.input_size);
        assert_eq!(
            circuit.num_input(),
            config.input_size + description.bits.len() + config.security_parameter_bits,
        );
        assert_eq!(circuit.num_output(), config.function.output_bits);
        for selector in description.bits.chunks_exact(config.input_size) {
            assert_eq!(selector.iter().filter(|bit| **bit).count(), 1);
        }
    }

    #[test]
    fn prescribed_coin_circuit_exposes_distinct_binary_uniform_and_gaussian_outputs() {
        let layout = PrescribedCoinLayout {
            binary_bits: 5,
            uniform_coefficients: 2,
            uniform_sample_bits: 12,
            gaussian_sample_bits: 8,
            gaussian_groups: vec![
                PrescribedGaussianGroup { coefficients: 3, sigma: 2.0 },
                PrescribedGaussianGroup { coefficients: 2, sigma: 4.0 },
            ],
        };
        let circuit = build_prescribed_coin_circuit::<DCRTPoly>(64, &layout, [11; 32]).unwrap();
        assert_eq!(circuit.num_input(), 64);
        assert_eq!(circuit.num_output(), layout.output_count().unwrap());
    }

    #[test]
    fn iterated_goldreich_stream_matches_native_for_arbitrary_output_length() {
        let seed = [true, false, true, true, false, false, true, false];
        let output_bits = 257;
        let graph_seed = [13; 32];
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(seed.len());
        let outputs = generate_goldreich_stream_from_gates(
            &mut circuit,
            &inputs.gate_ids().collect::<Vec<_>>(),
            output_bits,
            graph_seed,
        )
        .unwrap();
        circuit.output(outputs);

        let mut state = seed.to_vec();
        let mut expected = Vec::with_capacity(output_bits);
        let mut round = 0;
        while expected.len() < output_bits {
            let graph = GoldreichGraph::generate(
                state.len(),
                2 * state.len(),
                goldreich_stream_round_seed(graph_seed, round),
                GoldreichGraphGeneration::default(),
            );
            let expanded = evaluate_goldreich_bits(&graph, &state);
            state = expanded[..seed.len()].to_vec();
            let remaining = output_bits - expected.len();
            expected
                .extend_from_slice(&expanded[seed.len()..seed.len() + remaining.min(seed.len())]);
            round += 1;
        }
        assert_eq!(evaluate_boolean_circuit(&circuit, &seed), expected);
    }

    #[test]
    fn cascade_function_matches_exact_target_encryption_bit_for_bit() {
        let mut config = config();
        config.modulus = 3.into();
        config.ring_dimension = 1;
        config.gadget_base = 2.into();
        config.digit_count = 2;
        config.security_parameter_bits = 3;
        config.cascade_randomness_bits = 3;
        config.gaussian_sample_bits = 3;
        config.uniform_statistical_bits = 1;
        config.secret_sigma = RealExpr::from_integer(1);
        config.b_error_sigma = RealExpr::from_integer(1);
        config.fhe_error_sigma = RealExpr::from_integer(1);
        config.attribute_error_sigma = RealExpr::from_integer(1);
        let target = CascadeLayerPayload {
            layer: 1,
            bit_len: 2,
            ske_key: None,
            slot_values: vec![(1, 0..1), (2, 1..2)],
            randomness: Vec::new(),
            prf_keys: Vec::new(),
        };
        let source = CascadeLayerPayload {
            layer: 2,
            bit_len: 7,
            ske_key: Some(0..3),
            slot_values: vec![(2, 3..4)],
            randomness: Vec::new(),
            prf_keys: vec![(1, 4..7)],
        };
        let target_layer = PrivatePrfeLayerWires::setup(&config, target.bit_len).unwrap();
        let coin_layout = target_layer.prescribed_coin_layout().unwrap();
        let public_key_bits = target_layer.public_key_scalar_bits().unwrap().len();
        let function = build_cascade_function_circuit::<DCRTPoly>(
            &config,
            &source,
            &target,
            public_key_bits,
            &coin_layout,
        )
        .unwrap();
        let reference = build_private_prfe_encryption_circuit::<DCRTPoly>(
            &config,
            target.bit_len,
            6,
            &coin_layout,
            config.function.graph_seed,
        )
        .unwrap();

        let ske_key = [true, false, true];
        let x2 = false;
        let k1 = [false, true, true];
        let r1 = [true, true, false];
        let nonce = vec![false, true, false];
        let x1 = true;
        let ske = goldreich_ske_encrypt(&ske_key, nonce.clone(), &[x1], config.function.graph_seed)
            .unwrap();
        let public_key = (0..public_key_bits).map(|index| index % 3 == 1).collect::<Vec<_>>();
        let function_inputs = ske_key
            .into_iter()
            .chain([x2])
            .chain(k1)
            .chain([false, true, false])
            .chain(nonce)
            .chain(ske.masked_payload)
            .chain(r1)
            .chain(public_key.iter().copied())
            .collect::<Vec<_>>();
        let reference_inputs =
            [x1, x2].into_iter().chain(k1).chain(r1).chain(public_key).collect::<Vec<_>>();
        assert_eq!(
            evaluate_boolean_circuit(&function, &function_inputs),
            evaluate_boolean_circuit(&reference, &reference_inputs),
        );
    }

    #[test]
    fn nonce_goldreich_ske_decrypts_inside_the_same_poly_circuit() {
        let key = [true, false, true, true, false, false, true, false];
        let nonce = vec![false, true, false, true, true, false, false, true];
        let plaintext = [true, false, true, false, true];
        let graph_seed = [19; 32];
        let ciphertext =
            goldreich_ske_encrypt(&key, nonce, &plaintext, graph_seed).expect("SKE encryption");
        let circuit = build_goldreich_ske_decryption_circuit::<DCRTPoly>(
            key.len(),
            plaintext.len(),
            graph_seed,
        )
        .expect("SKE decryption circuit");
        let inputs = key
            .into_iter()
            .chain(ciphertext.nonce)
            .chain(ciphertext.masked_payload)
            .collect::<Vec<_>>();
        assert_eq!(evaluate_boolean_circuit(&circuit, &inputs), plaintext);
    }

    #[test]
    fn canonical_residue_arithmetic_matches_integer_modulus() {
        use rand::Rng;

        let modulus = num_bigint::BigUint::from(97u8);
        let width = modulus.bits() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(width * 2);
        let lhs = CanonicalResidue::from_canonical_bits(
            inputs.clone().slice(0..width).gate_ids(),
            &modulus,
        );
        let rhs = CanonicalResidue::from_canonical_bits(
            inputs.slice(width..width * 2).gate_ids(),
            &modulus,
        );
        let sum = lhs.add(&rhs, &mut circuit, &modulus);
        let product = lhs.mul(&rhs, &mut circuit, &modulus);
        let negated = lhs.negate(&mut circuit, &modulus);
        circuit.output(sum.bits().iter().chain(product.bits()).chain(negated.bits()).copied());

        let mut rng = rand::rng();
        for _ in 0..32 {
            let lhs = rng.random_range(0..97u64);
            let rhs = rng.random_range(0..97u64);
            let input = (0..width)
                .map(|bit| ((lhs >> bit) & 1) == 1)
                .chain((0..width).map(|bit| ((rhs >> bit) & 1) == 1))
                .collect::<Vec<_>>();
            let output = evaluate_boolean_circuit(&circuit, &input);
            let decode = |bits: &[bool]| {
                bits.iter()
                    .enumerate()
                    .fold(0u64, |value, (bit, set)| if *set { value | (1 << bit) } else { value })
            };
            assert_eq!(decode(&output[..width]), (lhs + rhs) % 97);
            assert_eq!(decode(&output[width..width * 2]), (lhs * rhs) % 97);
            assert_eq!(decode(&output[width * 2..]), (97 - lhs) % 97);
        }
    }

    #[test]
    fn canonical_polynomial_multiplication_is_negacyclic() {
        use rand::Rng;

        let modulus = num_bigint::BigUint::from(97u8);
        let width = modulus.bits() as usize;
        let ring_dimension = 2;
        let polynomial_bits = ring_dimension * width;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(polynomial_bits * 2);
        let lhs = CanonicalPolynomial::from_canonical_bits(
            &inputs.clone().slice(0..polynomial_bits).gate_ids().collect::<Vec<_>>(),
            ring_dimension,
            &modulus,
        );
        let rhs = CanonicalPolynomial::from_canonical_bits(
            &inputs.slice(polynomial_bits..polynomial_bits * 2).gate_ids().collect::<Vec<_>>(),
            ring_dimension,
            &modulus,
        );
        let product = lhs.mul_negacyclic(&rhs, &mut circuit, &modulus);
        circuit.output(product.bits());

        let mut rng = rand::rng();
        for _ in 0..16 {
            let lhs = [rng.random_range(0..97u64), rng.random_range(0..97u64)];
            let rhs = [rng.random_range(0..97u64), rng.random_range(0..97u64)];
            let input = lhs
                .into_iter()
                .chain(rhs)
                .flat_map(|value| (0..width).map(move |bit| ((value >> bit) & 1) == 1))
                .collect::<Vec<_>>();
            let output = evaluate_boolean_circuit(&circuit, &input);
            let decode = |coefficient: usize| {
                output[coefficient * width..(coefficient + 1) * width]
                    .iter()
                    .enumerate()
                    .fold(0u64, |value, (bit, set)| if *set { value | (1 << bit) } else { value })
            };
            let expected_constant = (lhs[0] * rhs[0] + 97 - (lhs[1] * rhs[1]) % 97) % 97;
            let expected_linear = (lhs[0] * rhs[1] + lhs[1] * rhs[0]) % 97;
            assert_eq!(decode(0), expected_constant);
            assert_eq!(decode(1), expected_linear);
        }
    }

    #[test]
    fn canonical_matrix_multiplication_uses_negacyclic_entries() {
        let modulus = num_bigint::BigUint::from(97u8);
        let width = modulus.bits() as usize;
        let ring_dimension = 2;
        let polynomial_bits = ring_dimension * width;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(polynomial_bits * 4);
        let input_gates = inputs.gate_ids().collect::<Vec<_>>();
        let lhs = CanonicalMatrix::from_canonical_bits(
            &input_gates[..polynomial_bits * 2],
            1,
            2,
            ring_dimension,
            &modulus,
        );
        let rhs = CanonicalMatrix::from_canonical_bits(
            &input_gates[polynomial_bits * 2..],
            2,
            1,
            ring_dimension,
            &modulus,
        );
        let product = lhs.mul(&rhs, &mut circuit, ring_dimension, &modulus);
        circuit.output(product.bits());

        let polynomials = [[3u64, 5], [7, 11], [13, 17], [19, 23]];
        let input = polynomials
            .into_iter()
            .flatten()
            .flat_map(|value| (0..width).map(move |bit| ((value >> bit) & 1) == 1))
            .collect::<Vec<_>>();
        let output = evaluate_boolean_circuit(&circuit, &input);
        let decode = |coefficient: usize| {
            output[coefficient * width..(coefficient + 1) * width]
                .iter()
                .enumerate()
                .fold(0u64, |value, (bit, set)| if *set { value | (1 << bit) } else { value })
        };
        let expected_constant = (3 * 13 + 97 - 5 * 17 % 97 + 7 * 19 + 97 - 11 * 23 % 97) % 97;
        let expected_linear = (3 * 17 + 5 * 13 + 7 * 23 + 11 * 19) % 97;
        assert_eq!([decode(0), decode(1)], [expected_constant, expected_linear]);
    }

    #[test]
    fn coefficient_encryption_circuit_has_exact_ciphertext_width() {
        let mut config = config();
        config.modulus = BigInt::from(5);
        config.ring_dimension = 1;
        config.gadget_base = BigInt::from(2);
        config.digit_count = 3;
        config.security_parameter_bits = 1;
        config.cascade_randomness_bits = 1;
        config.gaussian_sample_bits = 1;
        config.uniform_statistical_bits = 1;
        config.trapdoor_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.secret_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.b_error_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.fhe_error_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        config.attribute_error_sigma = RealExpr::from_f64_exact(0.125).unwrap();
        let plaintext_bits = 1;
        let dimensions =
            super::super::prfe::PrivatePrfeLayerDimensions::new(&config, plaintext_bits).unwrap();
        let gsw_columns = 2 * config.digit_count;
        let x_columns = gsw_columns * (plaintext_bits + config.security_parameter_bits);
        let attribute_count = 2 * x_columns * dimensions.coefficient_bits + 1;
        let layout = PrescribedCoinLayout {
            binary_bits: config.security_parameter_bits + gsw_columns * x_columns,
            uniform_coefficients: gsw_columns,
            uniform_sample_bits: dimensions.coefficient_bits + config.uniform_statistical_bits,
            gaussian_sample_bits: config.gaussian_sample_bits,
            gaussian_groups: vec![
                PrescribedGaussianGroup { coefficients: 1, sigma: 0.125 },
                PrescribedGaussianGroup {
                    coefficients: 2 * (config.digit_count + 2),
                    sigma: 0.125,
                },
                PrescribedGaussianGroup { coefficients: gsw_columns, sigma: 0.125 },
                PrescribedGaussianGroup {
                    coefficients: attribute_count * gsw_columns,
                    sigma: 0.125,
                },
            ],
        };
        let circuit = build_private_prfe_encryption_circuit::<DCRTPoly>(
            &config,
            plaintext_bits,
            96,
            &layout,
            [29; 32],
        )
        .unwrap();
        assert_eq!(circuit.num_output(), dimensions.ciphertext_bits);
    }

    fn evaluate_boolean_circuit(circuit: &PolyCircuit<DCRTPoly>, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(circuit.num_input(), inputs.len());
        let mut values = BTreeMap::<usize, BigInt>::from([(0, BigInt::from(1))]);
        let mut next_input = 0;
        for (gate_id, gate) in circuit.gates_in_id_order() {
            if gate_id.index() == 0 {
                continue;
            }
            let value = match &gate.gate_type {
                PolyGateType::Input => {
                    let value = BigInt::from(inputs[next_input]);
                    next_input += 1;
                    value
                }
                PolyGateType::Add => {
                    &values[&gate.input_gates[0].index()] + &values[&gate.input_gates[1].index()]
                }
                PolyGateType::Sub => {
                    &values[&gate.input_gates[0].index()] - &values[&gate.input_gates[1].index()]
                }
                PolyGateType::Mul => {
                    &values[&gate.input_gates[0].index()] * &values[&gate.input_gates[1].index()]
                }
                other => panic!("unexpected Boolean SKE gate: {other:?}"),
            };
            values.insert(gate_id.index(), value);
        }
        circuit
            .output_gate_ids()
            .iter()
            .map(|gate| match &values[&gate.index()] {
                value if value == &BigInt::from(0) => false,
                value if value == &BigInt::from(1) => true,
                value => panic!("Boolean circuit produced {value}"),
            })
            .collect()
    }
}
