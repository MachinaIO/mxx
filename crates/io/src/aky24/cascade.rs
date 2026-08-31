use super::{
    artifacts::Aky24ArtifactNames,
    circuits::{
        build_cascade_function_circuit, build_goldreich_universal_circuit, encode_goldreich_circuit,
    },
    config::{Aky24ConfigError, Aky24IoConfig},
    prfe::{
        PrivatePrfeFunctionKeyWire, PrivatePrfeGraphError, PrivatePrfeLayerDimensions,
        PrivatePrfeLayerWires,
    },
};
use mxx_dsl::{BuiltGraph, DslContext, Family, Mat, Preimage, Ring};
use mxx_gadgets::{
    circuit::PolyCircuit,
    circuit_gadgets::fhe_prg::goldreich::{
        GoldreichGraph, GoldreichGraphGeneration, goldreich_output_bound_holds,
    },
};
use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId};
use mxx_primitives::poly::dcrt::poly::DCRTPoly;
use std::ops::Range;
use thiserror::Error;

/// Exact bit slices consumed by one Section 3.2 private-prFE layer.
///
/// Layer one contains only `x_1 || ... || x_N`. For layer `i >= 2`, the
/// ordering is exactly
///
/// ```text
/// SKE.sk || (x_i,r_i) || ... || (x_{N-1},r_{N-1}) || x_N || K_1 || ... || K_{i-1}.
/// ```
///
/// The final slot `x_N` is the padded Goldreich circuit description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CascadeLayerPayload {
    pub layer: usize,
    pub bit_len: usize,
    pub ske_key: Option<Range<usize>>,
    pub slot_values: Vec<(usize, Range<usize>)>,
    pub randomness: Vec<(usize, Range<usize>)>,
    pub prf_keys: Vec<(usize, Range<usize>)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24CascadeLayout {
    pub arity: usize,
    /// Length `Lambda` of the SKE key, cascade randomness, and PRF keys.
    /// This is distinct from the private-prFE seed length `lambda` stored in
    /// [`Aky24IoConfig::security_parameter_bits`].
    pub cascade_randomness_bits: usize,
    pub function_description_bits: usize,
    pub layers: Vec<CascadeLayerPayload>,
    pub(crate) dimensions: Vec<PrivatePrfeLayerDimensions>,
}

pub struct Aky24IoPreprocessingGraph {
    pub graph: BuiltGraph,
    pub layout: Aky24CascadeLayout,
}

pub struct Aky24IoEvaluationGraph {
    pub graph: BuiltGraph,
    pub layout: Aky24CascadeLayout,
}

#[derive(Debug, Error)]
pub enum Aky24CascadeGraphError {
    #[error(transparent)]
    Config(#[from] Aky24ConfigError),
    #[error("private-prFE graph construction failed: {0}")]
    Prfe(String),
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

/// Executable DSL state for every private-prFE layer in the Section 3.2
/// reverse cascade. Constructing this value creates the setup wires used by
/// preprocessing and later key generation.
pub struct Aky24CascadeCompiler {
    pub config: Aky24IoConfig,
    pub layout: Aky24CascadeLayout,
    layers: Vec<PrivatePrfeLayerWires>,
}

impl From<PrivatePrfeGraphError> for Aky24CascadeGraphError {
    fn from(error: PrivatePrfeGraphError) -> Self {
        Self::Prfe(error.to_string())
    }
}

impl Aky24CascadeCompiler {
    pub fn new(config: Aky24IoConfig) -> Result<Self, Aky24CascadeGraphError> {
        config.validate()?;
        let layout = Aky24CascadeLayout::select(&config)?;
        let mut selected = config;
        // Layout selection may raise Lambda so every prescribed PRF stream is
        // long enough. The private-prFE seed length lambda remains fixed.
        selected.cascade_randomness_bits = layout.cascade_randomness_bits;
        let layers = layout
            .layers
            .iter()
            .map(|payload| PrivatePrfeLayerWires::setup(&selected, payload.bit_len))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { config: selected, layout, layers })
    }

    pub fn config(&self) -> &Aky24IoConfig {
        &self.config
    }

    pub fn layout(&self) -> &Aky24CascadeLayout {
        &self.layout
    }

    pub fn build_preprocessing(&self) -> Result<Aky24IoPreprocessingGraph, Aky24CascadeGraphError> {
        let ring = self.ring();
        let lambda = self.layout.cascade_randomness_bits;
        let ske_key = sample_scalar_bits(&ring, lambda)?;
        let prf_keys = (0..self.layout.arity - 1)
            .map(|_| sample_scalar_bits(&ring, lambda))
            .collect::<Result<Vec<_>, _>>()?;
        let q_half =
            self.config.modulus.to_biguint().ok_or(Aky24ConfigError::NonPositiveParameter)? / 2u8;

        let final_payload = &self.layout.layers[self.layout.arity - 1];
        let mut final_message = vec![None; final_payload.bit_len];
        assign_mats(
            &mut final_message,
            final_payload.ske_key.clone().ok_or(Aky24ConfigError::SkeLayout)?,
            ske_key.clone(),
        )?;
        let description = encode_goldreich_circuit(&self.config)?;
        let description_range = final_payload
            .slot_values
            .iter()
            .find_map(|(slot, range)| (*slot == self.layout.arity).then(|| range.clone()))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        assign_mats(
            &mut final_message,
            description_range,
            description.bits.into_iter().map(|bit| bit_mat(&ring, bit)).collect(),
        )?;
        for (key, range) in &final_payload.prf_keys {
            assign_mats(&mut final_message, range.clone(), prf_keys[*key - 1].clone())?;
        }
        let final_message = final_message
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let final_seed = sample_scalar_bits(&ring, self.config.security_parameter_bits)?;
        let final_ciphertext =
            self.layers[self.layout.arity - 1].encrypt(final_message, final_seed)?;

        let universal = build_goldreich_universal_circuit::<DCRTPoly>(&self.config)?;
        let universal_high = self.layers[0].arithmetic_veval(&universal, 0, q_half.clone())?;
        let universal_low =
            zero_veval(self.layers[0].attribute_public.len() - 1, universal.num_output());
        let universal_key = self.layers[0].keygen(&universal_high, &universal_low, &[])?;

        struct BranchFamilies {
            preimages: Family<Preimage>,
            nonces: Vec<Family<Mat>>,
            masked_payloads: Family<Mat>,
            randomness: Vec<Family<Mat>>,
        }
        let mut branches = Vec::with_capacity(self.layout.arity - 1);
        for input in 0..self.layout.arity - 1 {
            let source = &self.layout.layers[input + 1];
            let target = &self.layout.layers[input];
            let target_public = self.layers[input].public_key_scalar_bits()?;
            let coin_layout = self.layers[input].prescribed_coin_layout()?;
            let function = build_cascade_function_circuit::<DCRTPoly>(
                &self.config,
                source,
                target,
                target_public.len(),
                &coin_layout,
            )?;
            let public_count = 2 * lambda + 1 + target_public.len();
            let high =
                self.layers[input + 1].arithmetic_veval(&function, public_count, q_half.clone())?;
            let low = zero_veval(
                self.layers[input + 1].attribute_public.len() - 1 + public_count,
                function.num_output(),
            );
            let choices = Family::pack(vec![ring.zero((1, 1)), ring.identity(1)])?;
            let layer = self.layers[input + 1].clone();
            let ring = ring.clone();
            let ske_key = ske_key.clone();
            let graph_seed = self.config.function.graph_seed;
            let (preimages, nonces, (masked_payloads, randomness)) =
                choices.parallel_map_values(move |_, bit| {
                    let randomness = sample_scalar_bits(&ring, lambda)
                        .expect("validated scalar-bit sampler in AKY24 parallel branch");
                    let nonce = sample_scalar_bits(&ring, lambda)
                        .expect("validated scalar-bit sampler in AKY24 parallel branch");
                    let mask = goldreich_mat_stream(
                        ske_key.iter().cloned().chain(nonce.iter().cloned()).collect(),
                        1,
                        graph_seed,
                        &ring,
                    )
                    .expect("validated Goldreich stream in AKY24 parallel branch");
                    let masked_payload = xor_mat(bit, mask[0].clone());
                    let public_inputs = nonce
                        .iter()
                        .cloned()
                        .chain(std::iter::once(masked_payload.clone()))
                        .chain(randomness.iter().cloned())
                        .chain(target_public.iter().cloned())
                        .collect::<Vec<_>>();
                    let key = layer
                        .keygen(&high, &low, &public_inputs)
                        .expect("validated private-prFE keygen in AKY24 parallel branch");
                    (key.preimage, nonce, (masked_payload, randomness))
                })?;
            branches.push(BranchFamilies { preimages, nonces, masked_payloads, randomness });
        }

        let mut context = DslContext::new("aky24-io-preprocessing");
        for (index, layer) in self.layers.iter().enumerate() {
            let paper_layer = index + 1;
            context = context
                .public_output(
                    Aky24ArtifactNames::layer_b_public(paper_layer),
                    layer.b_public.clone(),
                )?
                .public_family_output(
                    Aky24ArtifactNames::layer_attribute_public(paper_layer),
                    Family::pack(layer.attribute_public.clone())?,
                )?;
        }
        context = context
            .public_preimage_output(Aky24ArtifactNames::FINAL_KEY_PREIMAGE, universal_key.preimage)?
            .public_output(Aky24ArtifactNames::FUNCTION_CIPHERTEXT_C_B, final_ciphertext.c_b)?
            .public_output(Aky24ArtifactNames::FUNCTION_CIPHERTEXT_X, final_ciphertext.x)?
            .public_family_output(
                Aky24ArtifactNames::FUNCTION_CIPHERTEXT_ATTRIBUTE_VECTORS,
                Family::pack(
                    final_ciphertext.attributes.into_iter().map(|value| value.vector).collect(),
                )?,
            )?;
        for (input, branch) in branches.iter().enumerate() {
            for (choice, bit) in [false, true].into_iter().enumerate() {
                context = context
                    .public_preimage_output(
                        Aky24ArtifactNames::input_ciphertext_preimage(input, bit),
                        branch.preimages.get_static(choice),
                    )?
                    .public_family_output(
                        Aky24ArtifactNames::input_ske_nonce(input, bit),
                        Family::pack(
                            branch.nonces.iter().map(|family| family.get_static(choice)).collect(),
                        )?,
                    )?
                    .public_output(
                        Aky24ArtifactNames::input_ske_masked_payload(input, bit),
                        branch.masked_payloads.get_static(choice),
                    )?
                    .public_family_output(
                        Aky24ArtifactNames::input_randomness(input, bit),
                        Family::pack(
                            branch
                                .randomness
                                .iter()
                                .map(|family| family.get_static(choice))
                                .collect(),
                        )?,
                    )?;
            }
        }
        Ok(Aky24IoPreprocessingGraph { graph: context.build()?, layout: self.layout.clone() })
    }

    pub fn build_evaluation(
        &self,
        input: &[bool],
        production: ProductionId,
    ) -> Result<Aky24IoEvaluationGraph, Aky24CascadeGraphError> {
        if input.len() != self.config.input_size {
            return Err(Aky24ConfigError::CascadeLayoutOverflow.into());
        }
        let ring = self.ring();
        let lambda = self.layout.cascade_randomness_bits;
        let public_layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let paper_layer = index + 1;
                let b = ring.artifact_input(
                    production.clone(),
                    Aky24ArtifactNames::layer_b_public(paper_layer),
                    (2, 2 * (self.config.digit_count + 2)),
                    ArtifactConfidentiality::Public,
                );
                let attributes = ring.family_artifact_input(
                    production.clone(),
                    Aky24ArtifactNames::layer_attribute_public(paper_layer),
                    self.layout.dimensions[index].attribute_count,
                    (2, 2 * self.config.digit_count),
                    ArtifactConfidentiality::Public,
                );
                layer.with_public_matrices(
                    b,
                    (0..self.layout.dimensions[index].attribute_count)
                        .map(|attribute| attributes.get_static(attribute))
                        .collect(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let last = self.layout.arity - 1;
        let vectors = ring.family_artifact_input(
            production.clone(),
            Aky24ArtifactNames::FUNCTION_CIPHERTEXT_ATTRIBUTE_VECTORS,
            self.layout.dimensions[last].attribute_count,
            (1, 2 * self.config.digit_count),
            ArtifactConfidentiality::Public,
        );
        let mut ciphertext = public_layers[last].ciphertext_from_components(
            ring.artifact_input(
                production.clone(),
                Aky24ArtifactNames::FUNCTION_CIPHERTEXT_C_B,
                (1, 2 * (self.config.digit_count + 2)),
                ArtifactConfidentiality::Public,
            ),
            ring.artifact_input(
                production.clone(),
                Aky24ArtifactNames::FUNCTION_CIPHERTEXT_X,
                (2, self.layout.dimensions[last].x_columns),
                ArtifactConfidentiality::Public,
            ),
            (0..self.layout.dimensions[last].attribute_count)
                .map(|attribute| vectors.get_static(attribute))
                .collect(),
        )?;
        let q_half = self.config.modulus.to_biguint().unwrap() / 2u8;
        for source_index in (1..self.layout.arity).rev() {
            let input_index = source_index - 1;
            let selected = input[input_index];
            let target_public = public_layers[input_index].public_key_scalar_bits()?;
            let nonce = ring.family_artifact_input(
                production.clone(),
                Aky24ArtifactNames::input_ske_nonce(input_index, selected),
                lambda,
                (1, 1),
                ArtifactConfidentiality::Public,
            );
            let randomness = ring.family_artifact_input(
                production.clone(),
                Aky24ArtifactNames::input_randomness(input_index, selected),
                lambda,
                (1, 1),
                ArtifactConfidentiality::Public,
            );
            let public_inputs = (0..lambda)
                .map(|index| nonce.get_static(index))
                .chain(std::iter::once(ring.artifact_input(
                    production.clone(),
                    Aky24ArtifactNames::input_ske_masked_payload(input_index, selected),
                    (1, 1),
                    ArtifactConfidentiality::Public,
                )))
                .chain((0..lambda).map(|index| randomness.get_static(index)))
                .chain(target_public.iter().cloned())
                .collect::<Vec<_>>();
            let function = build_cascade_function_circuit::<DCRTPoly>(
                &self.config,
                &self.layout.layers[source_index],
                &self.layout.layers[input_index],
                target_public.len(),
                &public_layers[input_index].prescribed_coin_layout()?,
            )?;
            let high = public_layers[source_index].arithmetic_veval(
                &function,
                public_inputs.len(),
                q_half.clone(),
            )?;
            let low = zero_veval(
                public_layers[source_index].attribute_public.len() - 1 + public_inputs.len(),
                function.num_output(),
            );
            let key = PrivatePrfeFunctionKeyWire {
                preimage: ring.preimage_artifact_input(
                    production.clone(),
                    Aky24ArtifactNames::input_ciphertext_preimage(input_index, selected),
                    (
                        2 * (self.config.digit_count + 2),
                        self.layout.dimensions[input_index].ciphertext_bits,
                    ),
                    ArtifactConfidentiality::Public,
                ),
                output_count: self.layout.dimensions[input_index].ciphertext_bits,
            };
            let bits = public_layers[source_index].decrypt(
                &high,
                &low,
                &ciphertext,
                &key,
                &public_inputs,
            )?;
            ciphertext = public_layers[input_index].deserialize_ciphertext(&bits)?;
        }
        let universal = build_goldreich_universal_circuit::<DCRTPoly>(&self.config)?;
        let high = public_layers[0].arithmetic_veval(&universal, 0, q_half)?;
        let low = zero_veval(public_layers[0].attribute_public.len() - 1, universal.num_output());
        let key = PrivatePrfeFunctionKeyWire {
            preimage: ring.preimage_artifact_input(
                production,
                Aky24ArtifactNames::FINAL_KEY_PREIMAGE,
                (2 * (self.config.digit_count + 2), self.config.function.output_bits),
                ArtifactConfidentiality::Public,
            ),
            output_count: self.config.function.output_bits,
        };
        let outputs = public_layers[0].decrypt(&high, &low, &ciphertext, &key, &[])?;
        let context = DslContext::new("aky24-io-evaluation")
            .bool_family_output(Aky24ArtifactNames::OUTPUT, Family::pack_bools(outputs)?)?;
        Ok(Aky24IoEvaluationGraph { graph: context.build()?, layout: self.layout.clone() })
    }

    fn ring(&self) -> Ring {
        Ring::new(self.config.modulus.clone(), self.config.ring_dimension)
    }

    /// Exact PRF domain for the prescribed encryption of target layer `i`.
    /// It is `K_i || r_i || ... || r_{N-1}` and contains no implicit padding.
    pub fn prescribed_prf_domain_bits(&self, target_layer: usize) -> Option<usize> {
        if target_layer == 0 || target_layer >= self.layout.arity {
            return None;
        }
        self.layout
            .arity
            .checked_sub(target_layer)
            .and_then(|remaining_r| remaining_r.checked_add(1))
            .and_then(|blocks| blocks.checked_mul(self.layout.cascade_randomness_bits))
    }
}

impl Aky24CascadeLayout {
    pub fn select(config: &Aky24IoConfig) -> Result<Self, Aky24ConfigError> {
        // The smallest prescribed domain is K_{N-1} || r_{N-1}, hence 2*Lambda.
        // Lambda >= 3 gives the six state bits required by the iterated  d -> 2d
        // Goldreich stream. Tape length no longer changes the payload layout.
        Self::for_cascade_randomness(config, config.cascade_randomness_bits.max(3))
    }

    fn for_cascade_randomness(
        config: &Aky24IoConfig,
        cascade_randomness_bits: usize,
    ) -> Result<Self, Aky24ConfigError> {
        let arity = config.prmife_arity()?;
        let function_description_bits = config
            .function
            .output_bits
            .checked_mul(5)
            .and_then(|count| count.checked_mul(config.input_size))
            .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
        let mut layers = Vec::with_capacity(arity);
        let mut first_cursor = 0usize;
        let mut first_slots = Vec::with_capacity(arity);
        for slot in 1..=arity {
            let width = if slot == arity { function_description_bits } else { 1 };
            let end =
                first_cursor.checked_add(width).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
            first_slots.push((slot, first_cursor..end));
            first_cursor = end;
        }
        layers.push(CascadeLayerPayload {
            layer: 1,
            bit_len: first_cursor,
            ske_key: None,
            slot_values: first_slots,
            randomness: Vec::new(),
            prf_keys: Vec::new(),
        });
        for layer in 2..=arity {
            let mut cursor = cascade_randomness_bits;
            let ske_key = Some(0..cursor);
            let mut slot_values = Vec::with_capacity(arity - layer + 1);
            let mut randomness = Vec::with_capacity(arity - layer);
            for slot in layer..=arity {
                let width = if slot == arity { function_description_bits } else { 1 };
                let value_end =
                    cursor.checked_add(width).ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
                slot_values.push((slot, cursor..value_end));
                cursor = value_end;
                if slot < arity {
                    let randomness_end = cursor
                        .checked_add(cascade_randomness_bits)
                        .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
                    randomness.push((slot, cursor..randomness_end));
                    cursor = randomness_end;
                }
            }
            let mut prf_keys = Vec::with_capacity(layer - 1);
            for key in 1..layer {
                let end = cursor
                    .checked_add(cascade_randomness_bits)
                    .ok_or(Aky24ConfigError::CascadeLayoutOverflow)?;
                prf_keys.push((key, cursor..end));
                cursor = end;
            }
            layers.push(CascadeLayerPayload {
                layer,
                bit_len: cursor,
                ske_key,
                slot_values,
                randomness,
                prf_keys,
            });
        }
        let mut selected = config.clone();
        selected.cascade_randomness_bits = cascade_randomness_bits;
        let dimensions = layers
            .iter()
            .map(|payload| PrivatePrfeLayerDimensions::new(&selected, payload.bit_len))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { arity, cascade_randomness_bits, function_description_bits, layers, dimensions })
    }

    #[cfg(test)]
    fn prf_domains_support_iterated_streams(&self) -> bool {
        (1..self.arity).all(|target_layer| {
            let Some(blocks) = self.arity.checked_sub(target_layer).and_then(|n| n.checked_add(1))
            else {
                return false;
            };
            let Some(domain_bits) = blocks.checked_mul(self.cascade_randomness_bits) else {
                return false;
            };
            domain_bits >= 6 && goldreich_output_bound_holds(domain_bits, 2 * domain_bits)
        })
    }
}

fn zero_veval(input_count: usize, output_count: usize) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::new();
    let _inputs = circuit.input(input_count);
    let zero = circuit.const_zero_gate().as_single_wire();
    circuit.output(std::iter::repeat_n(zero, 2 * output_count));
    circuit
}

fn bit_mat(ring: &Ring, bit: bool) -> Mat {
    if bit { ring.identity(1) } else { ring.zero((1, 1)) }
}

fn sample_scalar_bits(ring: &Ring, count: usize) -> Result<Vec<Mat>, Aky24CascadeGraphError> {
    (0..count)
        .map(|_| {
            let sample = ring.uniform_interval((1, 1), 0, 1);
            Ok(sample
                .extract_coefficient(0)
                .bit(0)
                .to_int()
                .select(vec![ring.zero((1, 1)), ring.identity(1)])?)
        })
        .collect()
}

fn xor_mat(lhs: Mat, rhs: Mat) -> Mat {
    let product = lhs.clone() * rhs.clone();
    lhs + rhs - (product.clone() + product)
}

fn goldreich_mat_stream(
    state: Vec<Mat>,
    output_bits: usize,
    graph_seed: [u8; 32],
    _ring: &Ring,
) -> Result<Vec<Mat>, Aky24CascadeGraphError> {
    if state.len() < 5 || !goldreich_output_bound_holds(state.len(), output_bits) {
        return Err(Aky24ConfigError::GoldreichOutputBound.into());
    }
    let graph = GoldreichGraph::generate(
        state.len(),
        output_bits,
        graph_seed,
        GoldreichGraphGeneration::default(),
    );
    Ok(graph
        .edges
        .iter()
        .map(|edge| {
            let and = state[edge.and_inputs[0]].clone() * state[edge.and_inputs[1]].clone();
            let first =
                xor_mat(state[edge.xor_inputs[0]].clone(), state[edge.xor_inputs[1]].clone());
            xor_mat(xor_mat(first, state[edge.xor_inputs[2]].clone()), and)
        })
        .collect())
}

fn assign_mats(
    output: &mut [Option<Mat>],
    range: Range<usize>,
    values: Vec<Mat>,
) -> Result<(), Aky24CascadeGraphError> {
    if range.len() != values.len() || range.end > output.len() {
        return Err(Aky24ConfigError::CascadeLayoutOverflow.into());
    }
    for (slot, value) in output[range].iter_mut().zip(values) {
        *slot = Some(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aky24::config::Aky24GoldreichPrf;
    use mxx_ir_core::{
        ParamEnv, RealExpr,
        artifact::{SpecHash, export_validated_manifest},
        node::NodeKind,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn config() -> Aky24IoConfig {
        Aky24IoConfig {
            modulus: 257.into(),
            ring_dimension: 1,
            input_size: 5,
            gadget_base: 2.into(),
            digit_count: 9,
            modulus_split: 1.into(),
            trapdoor_sigma: RealExpr::from_integer(4),
            secret_sigma: RealExpr::from_integer(2),
            b_error_sigma: RealExpr::from_integer(1),
            fhe_error_sigma: RealExpr::from_integer(1),
            attribute_error_sigma: RealExpr::from_integer(1),
            security_parameter_bits: 8,
            cascade_randomness_bits: 8,
            gaussian_sample_bits: 8,
            uniform_statistical_bits: 8,
            function: Aky24GoldreichPrf { output_bits: 1, graph_seed: [5; 32] },
        }
    }

    #[test]
    fn exact_layer_payloads_interleave_values_and_randomness_without_padding() {
        let layout = Aky24CascadeLayout::select(&config()).unwrap();
        assert_eq!(layout.arity, 6);
        assert_eq!(layout.layers[0].slot_values.len(), 6);
        let layer_two = &layout.layers[1];
        assert_eq!(layer_two.ske_key, Some(0..layout.cascade_randomness_bits));
        assert_eq!(
            layer_two.slot_values.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
            vec![2, 3, 4, 5, 6]
        );
        assert_eq!(
            layer_two.randomness.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
            vec![2, 3, 4, 5]
        );
        assert_eq!(layer_two.prf_keys.iter().map(|(key, _)| *key).collect::<Vec<_>>(), vec![1]);
        assert_eq!(layer_two.prf_keys.last().unwrap().1.end, layer_two.bit_len);
        let last = &layout.layers[5];
        assert_eq!(last.slot_values.len(), 1);
        assert!(last.randomness.is_empty());
        assert_eq!(last.prf_keys.len(), 5);
    }

    #[test]
    fn selected_cascade_randomness_covers_tapes_without_changing_prfe_seed_length() {
        let mut selected = config();
        selected.security_parameter_bits = 2;
        selected.cascade_randomness_bits = 8;
        let layout = Aky24CascadeLayout::select(&selected).unwrap();
        assert_ne!(selected.security_parameter_bits, layout.cascade_randomness_bits);
        assert!(layout.prf_domains_support_iterated_streams());
        for target_layer in 1..layout.arity {
            let domain = (layout.arity - target_layer + 1) * layout.cascade_randomness_bits;
            assert!(goldreich_output_bound_holds(domain, 2 * domain));
        }
        let compiler = Aky24CascadeCompiler::new(selected).unwrap();
        assert_eq!(compiler.config.security_parameter_bits, 2);
        assert_eq!(
            compiler.config.cascade_randomness_bits,
            compiler.layout.cascade_randomness_bits,
        );
    }

    #[test]
    #[ignore = "expands the complete AKY24 private-prFE preprocessing graph"]
    fn preprocessing_exports_the_complete_public_schema() {
        let mut config = config();
        config.modulus = 3.into();
        config.ring_dimension = 1;
        config.gadget_base = 2.into();
        config.digit_count = 2;
        config.security_parameter_bits = 3;
        config.cascade_randomness_bits = 3;
        config.gaussian_sample_bits = 3;
        config.uniform_statistical_bits = 1;
        config.input_size = 5;
        let compiler = Aky24CascadeCompiler::new(config).unwrap();
        let preprocessing = compiler.build_preprocessing().unwrap();
        let validated = preprocessing.graph.validate(&ParamEnv::default()).unwrap();
        let public = Aky24ArtifactNames::all_public_names(compiler.config.input_size);
        assert!(public.iter().all(|name| validated.source.outputs().contains_key(name)));
    }

    #[test]
    #[ignore = "expands the complete AKY24 private-prFE cascade"]
    fn evaluation_manifest_imports_only_selected_public_branches() {
        let mut selected = config();
        selected.modulus = 3.into();
        selected.digit_count = 2;
        selected.security_parameter_bits = 3;
        selected.cascade_randomness_bits = 3;
        selected.gaussian_sample_bits = 3;
        selected.uniform_statistical_bits = 1;
        let compiler = Aky24CascadeCompiler::new(selected).unwrap();
        let producer = compiler.build_preprocessing().unwrap();
        let validated_producer = producer.graph.validate(&ParamEnv::default()).unwrap();
        let production = ProductionId { spec_hash: SpecHash([17; 32]), execution_nonce: [23; 32] };
        let manifest = export_validated_manifest(production.clone(), &validated_producer).unwrap();
        assert!(
            manifest
                .artifacts
                .values()
                .all(|artifact| artifact.confidentiality == ArtifactConfidentiality::Public)
        );
        let input = [false, true, false, true, true];
        let consumer = compiler.build_evaluation(&input, production.clone()).unwrap();
        consumer
            .graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();

        let imported = consumer
            .graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter_map(|node| match node.kind() {
                NodeKind::Input { artifact: Some(artifact), .. } => {
                    Some(artifact.artifact_name.clone())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        for (index, selected_bit) in input.into_iter().enumerate() {
            for name in [
                Aky24ArtifactNames::input_ciphertext_preimage(index, selected_bit),
                Aky24ArtifactNames::input_ske_nonce(index, selected_bit),
                Aky24ArtifactNames::input_ske_masked_payload(index, selected_bit),
                Aky24ArtifactNames::input_randomness(index, selected_bit),
            ] {
                assert!(imported.contains(&name), "selected artifact {name} was not imported");
            }
            let opposite = !selected_bit;
            assert!(
                !imported
                    .contains(&Aky24ArtifactNames::input_ciphertext_preimage(index, opposite,))
            );
            assert!(!imported.contains(&Aky24ArtifactNames::input_ske_nonce(index, opposite)));
            assert!(
                !imported.contains(&Aky24ArtifactNames::input_ske_masked_payload(index, opposite,))
            );
            assert!(!imported.contains(&Aky24ArtifactNames::input_randomness(index, opposite)));
        }
        assert!(
            imported.is_subset(&Aky24ArtifactNames::all_public_names(compiler.config.input_size,))
        );
    }
}
