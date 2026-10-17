//! A closed protocol that evaluates a layered Boolean circuit on BGG+ encodings and decrypts its
//! output bit.
//!
//! Every graph declares the circuit parameters of [`BooleanCircuitFamilyParams`], and every loop
//! is a DSL family, so the graphs do not grow with the circuit's depth or width:
//!
//! 1. `keygen` samples a ternary `s_bar`; the secret is `s = (s_bar, 1)`.
//! 2. `encrypt` encodes every input bit as `c_i = s (A_i - x_i G) + e_i` and the constant one as
//!    `c_0 = s (A_0 - G) + e_0`, with public matrices expanded from a public hash key and one
//!    parallel loop over the `max_layer_width` input slots ([`BggEncodingSampler::sample_family`]).
//! 3. `evaluate` evaluates the layers with [`evaluate_boolean_encoding_layers`] from the encodings,
//!    their public matrices, and the public input bits, selects the output slot, and applies the
//!    public decomposition `D = G^(-1)(floor(q / 2) e_2)`: `c' = c D`, `A' = A D`.
//! 4. `decrypt` rounds `2 (c' - s A') / q` modulo two. Since `s G D = floor(q / 2)`, `c' - s A' =
//!    -x floor(q / 2) + e D` for the output bit `x`.
//!
//! The ideal functionality evaluates the same circuit on the input bits
//! ([`evaluate_boolean_family`]), and the requirement is [`boolean_circuit_validity_predicate`].
//! [`crate::noise_simulate::boolean`] bounds `e D` term by term along these graphs.

use crate::{
    bgg::{
        BggEncodingSampler, BggEncodingWire, BggPublicKeyFamily, BggPublicKeySampler,
        BggPublicKeyWire, BggSamplerLayout, CircuitEncoding, evaluate_boolean_encoding_layers,
        public_key::BggPublicKeyCompiler,
    },
    circuit::{
        BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, boolean_circuit_validity_predicate,
        evaluate_boolean_family, select_boolean_output,
    },
};
use mxx_dsl::{
    BuiltGraph, Bytes, DslContext, Family, IdealSpec, Int, Mat, MatType, Ring, parallel,
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, Rational, RealExpr,
    node::ConcatAxis,
    protocol::{
        ArtifactBinding, ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec,
        EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OutputRef, ParameterDecl, ParameterKind,
        ProtocolDecl, ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId,
        ProtocolPreconditionSpec, ProtocolStage, StageId, StageInputName, Workflow,
    },
};
use num_bigint::{BigInt, BigUint};
use std::collections::BTreeMap;

/// The secret `s = (s_bar, 1)` has two components.
pub const SECRET_DIMENSION: usize = 2;
pub const HASH_KEY_BYTES: usize = 32;

/// The BGG+ ring, gadget, and encoding errors of the protocol.
#[derive(Clone, Debug)]
pub struct BooleanBggProtocolParameters {
    pub ring: Ring,
    pub base_bits: usize,
    /// The gadget digits of one secret component.
    pub digit_count: usize,
    /// The encoding error width and coefficient cutoff; `None` samples no error.
    pub error: Option<(Rational, BigUint)>,
}

impl BooleanBggProtocolParameters {
    pub fn layout(&self) -> BggSamplerLayout {
        BggSamplerLayout {
            ring: self.ring.clone(),
            secret_dimension: SECRET_DIMENSION,
            digit_count: self.digit_count,
            gadget_base: self.base(),
        }
    }

    pub fn base(&self) -> IntExpr {
        IntExpr::constant(BigInt::from(1u8) << self.base_bits)
    }

    /// The columns `m` of an encoding.
    pub fn columns(&self) -> usize {
        SECRET_DIMENSION * self.digit_count
    }

    /// `s = (s_bar, 1)`.
    fn secret(&self, s_bar: Mat) -> Mat {
        Mat::concat(ConcatAxis::Columns, vec![s_bar, self.ring.identity(1)])
    }

    /// The constant-one public matrix and the family of the input public matrices.
    fn public_keys(
        &self,
        hash_key: Bytes,
        count: IntExpr,
    ) -> (BggPublicKeyWire, BggPublicKeyFamily) {
        let sampler = BggPublicKeySampler { layout: self.layout() };
        let one = sampler.sample(hash_key.clone(), b"boolean-bgg/one".as_slice(), &[]).remove(0);
        let inputs = sampler
            .sample_family(hash_key, b"boolean-bgg/inputs".as_slice(), count, self.columns())
            .expect("a packed public-key family");
        (one, inputs)
    }

    fn plaintexts(&self, bits: &Family<Int>, count: IntExpr) -> Family<Mat> {
        let matrix_type = self.ring.matrix_type((1, 1)).clone();
        parallel(count, |slot| Ok(bits.at(&slot).lift_to_constant_polynomial(matrix_type.clone())))
            .expect("a plaintext family")
    }
}

/// A graph context declaring the circuit parameters, which every graph of the protocol shares.
fn context(name: &str) -> (DslContext, BooleanCircuitFamilyParams) {
    BooleanCircuitFamilyParams::declare(DslContext::new(name))
}

/// The four stage graphs of the protocol.
pub struct BooleanBggProtocolGraphs {
    pub keygen: BuiltGraph,
    pub encrypt: BuiltGraph,
    pub evaluate: BuiltGraph,
    pub decrypt: BuiltGraph,
}

impl BooleanBggProtocolGraphs {
    pub fn new(parameters: &BooleanBggProtocolParameters) -> Self {
        Self {
            keygen: keygen_graph(parameters),
            encrypt: encrypt_graph(parameters),
            evaluate: evaluate_graph(parameters),
            decrypt: decrypt_graph(parameters),
        }
    }
}

/// `secret = s_bar`, a ternary constant-shape ring element.
fn keygen_graph(parameters: &BooleanBggProtocolParameters) -> BuiltGraph {
    let (context, _) = context("boolean-bgg-keygen");
    let secret = parameters.ring.uniform_interval((1, 1), -1, 1);
    context.output("secret", secret).unwrap().build().unwrap()
}

/// Inputs `secret`, `public_hash_key`, `input_bits`; outputs the vectors `one` and `encodings`.
fn encrypt_graph(parameters: &BooleanBggProtocolParameters) -> BuiltGraph {
    let (context, params) = context("boolean-bgg-encrypt");
    let width = params.max_layer_width.clone();
    let secret = parameters.secret(parameters.ring.input("secret", (1, 1)));
    let hash_key = parameters.ring.bytes_input("public_hash_key", HASH_KEY_BYTES);
    let bits = context.int_family_input("input_bits", width.clone());
    let (one_key, keys) = parameters.public_keys(hash_key, width.clone());
    let sampler = BggEncodingSampler {
        layout: parameters.layout(),
        gaussian_sigma: parameters
            .error
            .as_ref()
            .map(|(sigma, _)| RealExpr::Rational(sigma.clone())),
        gaussian_max_coefficient_bound: parameters
            .error
            .as_ref()
            .map(|(_, cutoff)| IntExpr::constant(BigInt::from(cutoff.clone()))),
    };
    let one = sampler.sample(secret.clone(), None, &[one_key], &[]).unwrap().remove(0);
    let encodings =
        sampler.sample_family(secret, &keys, &parameters.plaintexts(&bits, width)).unwrap();
    context
        .output("one", one.vector)
        .unwrap()
        .output("encodings", encodings.field(|encoding| encoding.vector).unwrap())
        .unwrap()
        .build()
        .unwrap()
}

/// Inputs `one`, `encodings`, `public_hash_key`, `input_bits`, and the circuit families; outputs
/// `vector = c D` and `public_key = A D` of the selected output slot.
fn evaluate_graph(parameters: &BooleanBggProtocolParameters) -> BuiltGraph {
    evaluation(parameters, "boolean-bgg-evaluate", true)
}

/// The `evaluate` graph without its final decomposition: it outputs the output slot's encoding
/// `c` and public matrix `A` as the layers leave them, before `D = G^(-1)(floor(q / 2) e_2)`. It is
/// not a protocol stage; cost accounting uses it for the circuit evaluation alone.
pub fn evaluate_layers_graph(parameters: &BooleanBggProtocolParameters) -> BuiltGraph {
    evaluation(parameters, "boolean-bgg-evaluate-layers", false)
}

/// The circuit evaluation of `evaluate`, with the final decomposition when `decompose`.
fn evaluation(
    parameters: &BooleanBggProtocolParameters,
    name: &str,
    decompose: bool,
) -> BuiltGraph {
    let (context, params) = context(name);
    let width = params.max_layer_width.clone();
    let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
    let ring = &parameters.ring;
    let one = ring.input("one", (1, parameters.columns()));
    let encodings = ring.input_family("encodings", width.clone(), (1, parameters.columns()));
    let hash_key = ring.bytes_input("public_hash_key", HASH_KEY_BYTES);
    let bits = context.int_family_input("input_bits", width.clone());
    let (one_key, keys) = parameters.public_keys(hash_key, width.clone());
    let plaintexts = parameters.plaintexts(&bits, width.clone());
    let preceding = parallel(width, |slot| {
        Ok(CircuitEncoding {
            vector: encodings.at(&slot),
            public_key: keys.at(&slot).matrix,
            plaintext: plaintexts.at(&slot),
        })
    })
    .unwrap();
    let digit_count = IntExpr::constant(parameters.digit_count);
    let compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: parameters.base(),
        digit_count: digit_count.clone(),
    };
    let layers = evaluate_boolean_encoding_layers(
        &params,
        circuit.clone(),
        preceding,
        BggEncodingWire { vector: one, plaintext: Some(ring.identity(1)) },
        one_key,
        compiler,
    )
    .unwrap();
    let output = layers.at(circuit.output_source());
    if !decompose {
        return context
            .output("vector", output.vector)
            .unwrap()
            .output("public_key", output.public_key)
            .unwrap()
            .build()
            .unwrap();
    }
    // D = G^(-1)(floor(q / 2) e_2), so that s G D = floor(q / 2) for s = (s_bar, 1).
    let target = Mat::concat(
        ConcatAxis::Rows,
        vec![ring.zero((1, 1)), ring.polynomial([ring.modulus().floor_div(2)])],
    );
    let decomposition = target.decompose(parameters.base(), digit_count);
    context
        .output("vector", output.vector.mul_small_rhs(decomposition.clone()))
        .unwrap()
        .output("public_key", output.public_key.mul_small_rhs(decomposition))
        .unwrap()
        .build()
        .unwrap()
}

/// Inputs `secret`, `vector`, `public_key`; outputs `bit = round(2 (c' - s A') / q) mod 2`.
fn decrypt_graph(parameters: &BooleanBggProtocolParameters) -> BuiltGraph {
    let (context, _) = context("boolean-bgg-decrypt");
    let ring = &parameters.ring;
    let secret = parameters.secret(ring.input("secret", (1, 1)));
    let vector = ring.input("vector", (1, 1));
    let public_key = ring.input("public_key", (SECRET_DIMENSION, 1));
    let bit = (vector - &secret * &public_key).threshold_decode_bools(2, 1).remove(0);
    context.output("bit", bit).unwrap().build().unwrap()
}

/// Inputs `input_bits` and the circuit families; outputs the circuit's output `bit`.
fn ideal_graph() -> Graph {
    let (context, params) = context("boolean-bgg-ideal");
    let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
    let bits = context.int_family_input("input_bits", params.max_layer_width.clone());
    let inputs =
        parallel(params.max_layer_width.clone(), |slot| Ok(bits.at(&slot).equal(1))).unwrap();
    let layers = evaluate_boolean_family(&params, circuit.clone(), inputs).unwrap();
    let bit = select_boolean_output(&circuit, &layers);
    context.output("bit", bit).unwrap().build().unwrap().graph
}

/// The circuit parameters of `depth` layers of at most `max_layer_width` gates over
/// `instance_width` input bits.
pub fn circuit_bindings(instance_width: u128, depth: usize, max_layer_width: u128) -> ParamEnv {
    ParamEnv {
        integers: BTreeMap::from([
            (
                BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(),
                instance_width.into(),
            ),
            (BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(), 0.into()),
            (BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), depth.into()),
            (
                BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(),
                max_layer_width.into(),
            ),
        ]),
        ..ParamEnv::default()
    }
}

/// The bindings feeding the consumer input `consumer` from the producer output. Every linked
/// input is one matrix or one matrix family, a single leaf.
fn link(consumer: &str, producer: &StageId, output: &str, leaf: &MatType) -> Vec<ArtifactBinding> {
    mxx_dsl::artifact_bindings(leaf, consumer, producer, output).unwrap()
}

/// The closed protocol of the four stages under the circuit parameters `bindings`: the decrypted
/// bit equals the ideal evaluation of the circuit on the input bits, except with probability at
/// most `2^-128` over the sampled values, for every input meeting the contracts and the
/// requirement.
pub fn boolean_bgg_protocol(
    graphs: &BooleanBggProtocolGraphs,
    parameters: &BooleanBggProtocolParameters,
    bindings: &ParamEnv,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let (keygen, encrypt, evaluate, decrypt) =
        (id("keygen"), id("encrypt"), id("evaluate"), id("decrypt"));
    let leaf = MatType::new(parameters.ring.matrix_type((1, 1)));
    let stage = |id: &StageId, graph: &BuiltGraph, bindings: Vec<Vec<ArtifactBinding>>| {
        ProtocolStage { id: id.clone(), graph: graph.graph.clone(), bindings: bindings.concat() }
    };
    let stages = vec![
        stage(&keygen, &graphs.keygen, vec![]),
        stage(&encrypt, &graphs.encrypt, vec![link("secret", &keygen, "secret", &leaf)]),
        stage(
            &evaluate,
            &graphs.evaluate,
            vec![
                link("one", &encrypt, "one", &leaf),
                link("encodings", &encrypt, "encodings", &leaf),
            ],
        ),
        stage(
            &decrypt,
            &graphs.decrypt,
            vec![
                link("secret", &keygen, "secret", &leaf),
                link("vector", &evaluate, "vector", &leaf),
                link("public_key", &evaluate, "public_key", &leaf),
            ],
        ),
    ];

    let var = |name: &str| IntExpr::Var(name.to_owned());
    let width = var(BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER);
    let depth = var(BooleanCircuitFamilyParams::DEPTH_PARAMETER);
    let range = |lower: IntExpr, upper: IntExpr| InputValueContract::IntegerRange { lower, upper };
    let family = |count: IntExpr, element: InputValueContract| InputValueContract::Family {
        count,
        element: Box::new(element),
    };
    let slot = range(0.into(), &width - 1);
    let to_stage = |stage: &StageId, input: &str| ProtocolInputDestination::WorkflowStage {
        stage: stage.clone(),
        input: StageInputName(input.to_owned()),
    };
    let to_ideal = |input: &str| ProtocolInputDestination::Ideal { input: input.to_owned() };
    let circuit_input = |input: &str| {
        vec![
            to_stage(&evaluate, input),
            to_ideal(input),
            ProtocolInputDestination::Requirement { requirement: 0, input: input.to_owned() },
        ]
    };
    let flattened = &depth * &width;
    let (contracts, input_bindings): (Vec<_>, Vec<_>) = [
        (
            "public_hash_key",
            InputValueContract::Bytes { length: HASH_KEY_BYTES.into() },
            vec![to_stage(&encrypt, "public_hash_key"), to_stage(&evaluate, "public_hash_key")],
        ),
        (
            "input_bits",
            family(width.clone(), range(0.into(), 1.into())),
            vec![
                to_stage(&encrypt, "input_bits"),
                to_stage(&evaluate, "input_bits"),
                to_ideal("input_bits"),
            ],
        ),
        (
            "circuit-active-gate-count",
            family(depth.clone(), range(0.into(), width.clone())),
            circuit_input("circuit-active-gate-count"),
        ),
        (
            "circuit-gate-kind",
            family(flattened.clone(), range(0.into(), 5.into())),
            circuit_input("circuit-gate-kind"),
        ),
        (
            "circuit-left-source",
            family(flattened.clone(), slot.clone()),
            circuit_input("circuit-left-source"),
        ),
        (
            "circuit-right-source",
            family(flattened, slot.clone()),
            circuit_input("circuit-right-source"),
        ),
        ("circuit-output-source", family(1.into(), slot), circuit_input("circuit-output-source")),
    ]
    .into_iter()
    .map(|(name, value, destinations)| {
        (
            InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
            ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
        )
    })
    .unzip();
    let requirement =
        boolean_circuit_validity_predicate(DslContext::new("boolean-bgg-circuit-valid")).unwrap();
    let endpoint = EndpointSpecId::Exact;
    ProtocolDecl::new(ProtocolDecl {
        params: [
            BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER,
            BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER,
            BooleanCircuitFamilyParams::DEPTH_PARAMETER,
            BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER,
        ]
        .into_iter()
        .map(|name| ParameterDecl { name: name.to_owned(), kind: ParameterKind::Integer })
        .collect(),
        bindings: bindings.clone(),
        failure_probability_log2: Some(128),
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: decrypt.clone() },
            ideal: IdealSpec::new(ideal_graph()).unwrap(),
            requirements: vec![requirement],
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "bit".to_owned(),
                    ideal_input: "bit".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef { stage: decrypt, output: "bit".to_owned() },
                    ideal_output: "bit".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings,
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        },
    })
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bgg::test_utils::matrix_output,
        circuit::{
            BooleanCircuitData, BooleanCircuitShape, BooleanGateData, BooleanGateKind, PolyCircuit,
            lower_to_boolean_layers,
        },
        circuit_gadgets::{
            arith::{CrtWindow, DecomposeArithmeticGadget, MontgomeryPoly, MontgomeryPolyContext},
            fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        },
        noise_simulate::{BooleanBggParameterRequest, select_boolean_parameters},
        ring_from_params,
    };
    use mxx_backends::{
        ExecutionConfig, ExecutionResult, RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::cpu_backend,
        execute,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::bounds::hard_cutoff_from_sigma_bound,
        transcript::SamplingMode,
        utils::gen_biguint_for_modulus,
    };
    use rand::Rng;
    use std::sync::Arc;

    /// One execution of the protocol for one output of `circuit`.
    struct Run {
        bit: bool,
        secret: DCRTPolyMatrix,
        vector: DCRTPolyMatrix,
        public_key: DCRTPolyMatrix,
    }

    /// Executes one stage; stages share `store`, which holds the outputs they stage.
    fn execute_stage(
        graph: &BuiltGraph,
        bgg: &DCRTPolyParams,
        bindings: &ParamEnv,
        store: &mut MemoryArtifactStore,
        inputs: BTreeMap<String, RuntimeValue>,
    ) -> ExecutionResult {
        execute(
            &graph.validate(bindings).expect("a valid stage graph"),
            &mut cpu_backend([bgg.clone()]),
            inputs,
            store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .expect("stage execution")
    }

    fn input_bits(bits: &[bool], width: usize) -> RuntimeValue {
        RuntimeValue::integer_values(
            (0..width)
                .map(|slot| BigInt::from(u8::from(bits.get(slot).copied().unwrap_or(false))))
                .collect(),
        )
    }

    fn run_protocol(
        parameters: &BooleanBggProtocolParameters,
        bgg: &DCRTPolyParams,
        shape: &BooleanCircuitShape,
        circuit: &BooleanCircuitData,
        bits: &[bool],
    ) -> Run {
        let graphs = BooleanBggProtocolGraphs::new(parameters);
        let bindings = circuit_bindings(
            shape.instance_width as u128,
            shape.depth,
            shape.max_layer_width as u128,
        );
        let hash_key =
            RuntimeValue::Bytes(rand::rng().random::<[u8; HASH_KEY_BYTES]>().to_vec().into());
        let mut store = MemoryArtifactStore::default();
        let keys = execute_stage(&graphs.keygen, bgg, &bindings, &mut store, BTreeMap::new());
        let secret = keys.outputs["secret"].clone();
        let encrypted = execute_stage(
            &graphs.encrypt,
            bgg,
            &bindings,
            &mut store,
            BTreeMap::from([
                ("secret".to_owned(), secret.clone()),
                ("public_hash_key".to_owned(), hash_key.clone()),
                ("input_bits".to_owned(), input_bits(bits, shape.max_layer_width)),
            ]),
        );
        let mut inputs = BooleanCircuitFamilyInputs::runtime_values(circuit, shape);
        inputs.insert("one".to_owned(), encrypted.outputs["one"].clone());
        inputs.insert("encodings".to_owned(), encrypted.outputs["encodings"].clone());
        inputs.insert("public_hash_key".to_owned(), hash_key);
        inputs.insert("input_bits".to_owned(), input_bits(bits, shape.max_layer_width));
        let evaluated = execute_stage(&graphs.evaluate, bgg, &bindings, &mut store, inputs);
        let decrypted = execute_stage(
            &graphs.decrypt,
            bgg,
            &bindings,
            &mut store,
            BTreeMap::from([
                ("secret".to_owned(), secret),
                ("vector".to_owned(), evaluated.outputs["vector"].clone()),
                ("public_key".to_owned(), evaluated.outputs["public_key"].clone()),
            ]),
        );
        let RuntimeValue::Bool(bit) = decrypted.outputs["bit"] else {
            panic!("the decrypted bit is a Boolean")
        };
        Run {
            bit,
            secret: matrix_output(&keys, "secret").clone(),
            vector: matrix_output(&evaluated, "vector").clone(),
            public_key: matrix_output(&evaluated, "public_key").clone(),
        }
    }

    /// The constant coefficient of `c' - s A' + x floor(q / 2)`, centered modulo `q`.
    fn output_error(run: &Run, bgg: &DCRTPolyParams, bit: bool) -> BigUint {
        let modulus = bgg.modulus();
        let secret = DCRTPolyMatrix::from_poly_vec_row(
            bgg,
            vec![run.secret.entry(0, 0).clone(), DCRTPoly::const_one(bgg)],
        );
        let phase = run.vector.clone() - &(secret * &run.public_key);
        let half = modulus.as_ref() / 2u32;
        let value = (phase.entry(0, 0).coeffs_biguints()[0].clone() +
            if bit { half } else { BigUint::ZERO }) %
            modulus.as_ref();
        value.clone().min(modulus.as_ref() - value)
    }

    #[test]
    fn noiseless_protocol_decrypts_ring_gsw_product_bits() {
        let fhe = DCRTPolyParams::new(2, 1, 3, 1, None, None);
        let window = CrtWindow::full(1);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let arithmetic = Arc::new(MontgomeryPolyContext::setup(&mut circuit, &fhe, 1, false));
        let context = Arc::new(RingGswContext::from_arith_context(
            &mut circuit,
            &fhe,
            2,
            arithmetic.clone(),
            window,
        ));
        let width = context.width();
        let left = RingGswCiphertext::<DCRTPoly, MontgomeryPoly<DCRTPoly>>::input(
            context.clone(),
            None,
            &mut circuit,
        );
        let right = RingGswCiphertext::input(context, None, &mut circuit);
        let product = left.mul(&right, &mut circuit);
        circuit.output(
            product
                .rows
                .iter()
                .flatten()
                .flat_map(|entry| entry.value[0].iter().flat_map(|residue| residue.limbs.clone()))
                .collect::<Vec<_>>(),
        );
        let layering = lower_to_boolean_layers(&circuit);

        let modulus = fhe.modulus();
        let radix = BigUint::from(1u8) << arithmetic.num_limbs;
        let mut rng = rand::rng();
        let mut entries = || {
            (0..2 * width)
                .map(|_| {
                    (0..2)
                        .map(|_| gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let (left_entries, right_entries) = (entries(), entries());
        let montgomery_bits = |coefficient: &BigUint| {
            let residue = (coefficient * &radix) % modulus.as_ref();
            (0..arithmetic.num_limbs).map(move |bit| residue.bit(bit as u64))
        };
        let bits = left_entries
            .iter()
            .chain(&right_entries)
            .flatten()
            .flat_map(montgomery_bits)
            .collect::<Vec<_>>();
        let native = |entries: &[Vec<BigUint>]| {
            DCRTPolyMatrix::from_poly_vec(
                &fhe,
                entries
                    .chunks(width)
                    .map(|row| {
                        row.iter()
                            .map(|coefficients| DCRTPoly::from_biguints(&fhe, coefficients))
                            .collect()
                    })
                    .collect(),
            )
        };
        let expected = native(&left_entries) *
            MontgomeryPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
                &fhe,
                &arithmetic,
                &native(&right_entries),
                window,
            );
        let expected_bits = (0..2 * width)
            .flat_map(|entry| expected.entry(entry / width, entry % width).coeffs_biguints())
            .flat_map(|coefficient| montgomery_bits(&coefficient).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let bgg = DCRTPolyParams::new(8, 1, 40, 20, None, None);
        let parameters = BooleanBggProtocolParameters {
            ring: ring_from_params(&bgg),
            base_bits: 20,
            digit_count: 2,
            error: None,
        };
        {
            let output = rng.random_range(0..layering.output_slots.len());
            let run = run_protocol(
                &parameters,
                &bgg,
                &layering.shape,
                &layering.with_output(output),
                &bits,
            );
            assert_eq!(run.bit, expected_bits[output], "output {output} of the Ring-GSW product");
            assert_eq!(output_error(&run, &bgg, run.bit), BigUint::ZERO, "noiseless phase");
        }
    }

    #[test]
    fn noisy_protocol_errors_stay_within_the_simulated_bound() {
        let (input_width, depth, width) = (6, 3, 8);
        let mut rng = rand::rng();
        let layers = (0..depth)
            .map(|layer| {
                let preceding = if layer == 0 { input_width } else { width };
                (0..width)
                    .map(|_| {
                        let kind = [
                            BooleanGateKind::Copy,
                            BooleanGateKind::Not,
                            BooleanGateKind::And,
                            BooleanGateKind::Xor,
                        ][rng.random_range(0..4)];
                        let binary = matches!(kind, BooleanGateKind::And | BooleanGateKind::Xor);
                        BooleanGateData {
                            kind,
                            left: rng.random_range(0..preceding),
                            right: if binary { rng.random_range(0..preceding) } else { 0 },
                        }
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        let shape = BooleanCircuitShape {
            instance_width: input_width,
            witness_width: 0,
            depth,
            max_layer_width: width,
        };
        let sigma = bigdecimal::BigDecimal::from(3);
        let request = BooleanBggParameterRequest {
            depth,
            secret_dimension: SECRET_DIMENSION,
            crt_bits: 40,
            base_bits: (1..=20).collect(),
            ring_dimensions: vec![16],
            max_crt_depth: 12,
            fresh_error_bound: hard_cutoff_from_sigma_bound(&sigma),
            security_bits: 0,
        };
        let selection = select_boolean_parameters(&request, |_, _| Ok(1)).unwrap();
        let bgg = DCRTPolyParams::new(
            16,
            selection.crt_depth(),
            40,
            selection.base_bits as u32,
            None,
            None,
        );
        assert_eq!(bgg.to_crt().0, selection.crt_moduli, "the protocol ring is the selected one");
        let parameters = BooleanBggProtocolParameters {
            ring: ring_from_params(&bgg),
            base_bits: selection.base_bits,
            digit_count: selection.noise.digit_count,
            error: Some((Rational::from_integer(3.into()), request.fresh_error_bound.clone())),
        };
        let bits = (0..input_width).map(|_| rng.random::<bool>()).collect::<Vec<_>>();
        for output in [0, width - 1] {
            let circuit = BooleanCircuitData { layers: layers.clone(), output_source: output };
            let expected = circuit.evaluate(&shape, &bits, &[]).unwrap();
            let run = run_protocol(&parameters, &bgg, &shape, &circuit, &bits);
            assert_eq!(run.bit, expected, "output {output}");
            let error = output_error(&run, &bgg, expected);
            assert!(
                error <= selection.report.output_error,
                "the measured output error {error} exceeds the simulated {}",
                selection.report.output_error
            );
        }
    }

    /// The GPU counts of the evaluate graph, derived by hand from `evaluate_graph` and
    /// `evaluate_boolean_encoding_layers` for moduli of at least 31 bits and `inverse`
    /// multiplies per one-limb inverse NTT.
    fn expected_evaluate_counts(
        n: u128,
        crt_depth: u128,
        m: u128,
        depth: u128,
        width: u128,
        inverse: u128,
    ) -> BTreeMap<String, u128> {
        let limb_coefficients = n * crt_depth;
        let wide_mod_calls = |inner: u128| {
            (0..4).map(|slice| inner.saturating_sub(slice).div_ceil(4) / 63 + 1).sum::<u128>()
        };
        let slots = depth * width;
        // Root: the one-key slice (2 x m) and the packed key family (2 x mW) leave the hash
        // samples, each lane lifts its plaintext, and the decoding target concatenates two
        // constants and multiplies two m-digit decompositions. Every slot transforms the
        // m x m digits of G^(-1)(A_right) twice and its constant 2; every layer transforms the
        // constant-one plaintext once.
        let forward = 2 * m + 2 * m * width + width + slots * (2 * m * m + 1) + depth + 2 + 2 * m;
        // Every slot decomposes A_right (2 x m); the decoding target is 2 x 1.
        let inverse_polys = slots * 2 * m + 2;
        let shoup =
            (forward * (n / 2 * u128::from(n.ilog2()) + n) + inverse_polys * inverse) * crt_depth;
        // Per slot: c_L D (1 x m x m), A_L D (2 x m x m), and two 1 x 1 plaintext products;
        // the decoding products are 1 x m x 1 and 2 x m x 1.
        let wide_mul = (slots * (3 * m * m + 2) + 3 * m) * limb_coefficients;
        let wide_mod = (slots * (3 * m * wide_mod_calls(m) + 2 * wide_mod_calls(1)) +
            3 * wide_mod_calls(m)) *
            limb_coefficients;
        // Per slot: c_R x_L, the doubled vector, and the doubled public key (1x1 scalars).
        let mulmod = slots * 4 * m * limb_coefficients;
        let digit_polynomials = slots * m * m + m;
        // Matrix-family sequential carries are streamed by the GPU runtime. They add no
        // arithmetic operations and are no longer an unsupported-operation diagnostic.
        BTreeMap::from([
            ("gadget_decompositions".to_owned(), inverse_polys),
            ("gadget_decomposed_words".to_owned(), inverse_polys * n * crt_depth),
            ("gadget_digit_coefficients".to_owned(), digit_polynomials * n),
            ("gadget_digit_polynomials".to_owned(), digit_polynomials),
            ("mul64_instructions".to_owned(), wide_mul + 3 * shoup + mulmod),
            ("mulmod_u64".to_owned(), mulmod),
            ("ntt_forward".to_owned(), forward * crt_depth),
            ("ntt_inverse".to_owned(), inverse_polys * crt_depth),
            ("shoup_mul".to_owned(), shoup),
            ("wide_mod_calls".to_owned(), wide_mod),
            ("wide_mul".to_owned(), wide_mul),
        ])
    }

    fn generated_parameters(
        ring_dimension: u32,
        crt_depth: usize,
        crt_bits: usize,
        base_bits: usize,
    ) -> BooleanBggProtocolParameters {
        let moduli = mxx_ir_core::generate_crt_basis(ring_dimension, crt_depth, crt_bits).unwrap();
        BooleanBggProtocolParameters {
            ring: Ring::from_crt_moduli(
                moduli.into_iter().map(Into::into).collect(),
                ring_dimension,
            ),
            base_bits,
            digit_count: crt_depth * crt_bits.div_ceil(base_bits),
            error: None,
        }
    }

    #[test]
    fn gpu_operation_counts_of_the_evaluate_graph_match_the_per_slot_work() {
        let parameters = generated_parameters(8, 2, 33, 2);
        let graphs = BooleanBggProtocolGraphs::new(&parameters);
        let (depth, width) = (2, 3);
        let counts = mxx_backends::gpu_cost_model::gpu_operation_counts(
            &graphs.evaluate.graph,
            &circuit_bindings(4, depth, width),
            None,
            4,
        )
        .unwrap();
        let m = parameters.columns() as u128;
        // n = 8 runs one whole-ring blocked inverse transform: (n / 2) log2 n + 2n.
        let expected = expected_evaluate_counts(8, 2, m, depth as u128, width, 4 * 3 + 16);
        assert_eq!(counts, expected);
    }

    /// The selected row (n = 2^19, 186 x 44-bit moduli, base 4, depth 240) is counted in closed
    /// form: the walk visits each scope a constant number of times, whatever the loop counts.
    #[test]
    fn selected_row_gpu_operation_counts_are_closed_form() {
        let parameters = generated_parameters(1 << 19, 186, 44, 2);
        let graphs = BooleanBggProtocolGraphs::new(&parameters);
        let (depth, width) = (240, 20_072_706_095_024u128);
        let mut counts = mxx_backends::gpu_cost_model::gpu_operation_counts(
            &graphs.evaluate.graph,
            &circuit_bindings(31_719_424, depth, width),
            None,
            4,
        )
        .unwrap();
        let n = 1u128 << 19;
        assert!(counts.remove("unsupported(ring_dimension_above_131072)").unwrap() > 0);
        // n / 1024 = 512 > 32 tiles: the wide-stage inverse, 10n + (n / 2) log2(512 / 32) + 2n.
        let expected = expected_evaluate_counts(
            n,
            186,
            parameters.columns() as u128,
            depth as u128,
            width,
            10 * n + n / 2 * 4 + 2 * n,
        );
        assert_eq!(counts, expected);
    }
}
