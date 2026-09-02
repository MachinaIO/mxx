//! GPU runtime coverage for the setup-fixed, flat Exponent-LUT evaluator.
//!
//! These tests deliberately build the private encoding and public projection
//! through separate compilers.  The expected vector is the concrete BGG+
//! relation `s_mask * A - mu * s_payload * G`; it is not obtained from the
//! private evaluator's output.

use crate::{
    ExponentLutEncodingCompiler, ExponentLutPublicKeyCompiler,
    encoding::{
        ExponentLutEncodingSampler, FlatLutHelper, FlatLutHelperSet, FlatLutMaskBank,
        FlatLutPublicMaskBank,
    },
    program::{ExponentLutProgramBuilder, LutTable},
    public_key::{ExponentLutPublicKeySampler, FlatLutPublicHelper, FlatLutPublicHelperSet},
    rhs::ExponentRhsPackage,
};
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_ir_core::{ParamEnv, node::ConstantMatrix};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    RuntimeValue, artifact::MemoryArtifactStore, backend::poly::gpu::gpu_backend, execute,
    transcript::SamplingMode,
};
use num_bigint::BigInt;
use serial_test::serial;
use std::{collections::BTreeMap, env};

const RING_DIMENSION_ENV: &str = "MXX_EXPONENT_LUT_GPU_RING_DIMENSION";
const CRT_BITS_ENV: &str = "MXX_EXPONENT_LUT_GPU_CRT_BITS";
const BASE_BITS_ENV: &str = "MXX_EXPONENT_LUT_GPU_BASE_BITS";

fn parameter(name: &str, default: u32) -> u32 {
    env::var(name)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer")))
        .unwrap_or(default)
}

struct Fixture {
    parameters: DCRTPolyParams,
    gpu_parameters: GpuDCRTPolyParams,
    ring: Ring,
    sampler: ExponentLutEncodingSampler,
    public_sampler: ExponentLutPublicKeySampler,
    compiler: ExponentLutEncodingCompiler,
    public_compiler: ExponentLutPublicKeyCompiler,
    mask_secret: Mat,
    payload_secret: Mat,
    hash_key: mxx_dsl::Bytes,
}

struct Relation {
    encoded: BggEncodingWire,
    expected_public: Mat,
    mu: Mat,
}

impl Fixture {
    fn new() -> Self {
        let ring_dimension = parameter(RING_DIMENSION_ENV, 4);
        let crt_bits = parameter(CRT_BITS_ENV, 17);
        let base_bits = parameter(BASE_BITS_ENV, 9);
        assert!(ring_dimension.is_power_of_two() && ring_dimension >= 2);
        assert!(base_bits > 0 && base_bits < 63 && crt_bits > base_bits);
        let parameters = DCRTPolyParams::new(ring_dimension, 1, crt_bits as usize, base_bits);
        assert_eq!(parameters.modulus_digits(), 2, "GPU fixture requires digit_count=2");
        let (moduli, _, crt_depth) = parameters.to_crt();
        assert_eq!(crt_depth, 1);
        let gpu_parameters = GpuDCRTPolyParams::new(ring_dimension, moduli, parameters.base_bits());
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus.clone(), ring_dimension as usize);
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: modulus.into(),
            ring_dimension: (ring_dimension as usize).into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: BigInt::from(1u64 << base_bits).into(),
        };
        let bgg = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: layout.gadget_base.clone(),
            digit_count: layout.digit_count.into(),
        };
        Self {
            parameters,
            gpu_parameters,
            ring: ring.clone(),
            sampler: ExponentLutEncodingSampler {
                layout: layout.clone(),
                gaussian_sigma: None,
                gaussian_max_coefficient_bound: None,
            },
            public_sampler: ExponentLutPublicKeySampler { layout },
            compiler: ExponentLutEncodingCompiler::from_public_key(bgg.clone()),
            public_compiler: ExponentLutPublicKeyCompiler::new(bgg),
            mask_secret: ring.input("exponent-lut-gpu-mask-secret", (1, 2)),
            payload_secret: ring.input("exponent-lut-gpu-payload-secret", (1, 2)),
            hash_key: ring.bytes_input("exponent-lut-gpu-hash-key", 32),
        }
    }

    fn rotation(&self, exponent: usize) -> Mat {
        self.ring.constant((1, 1), ConstantMatrix::Rotation { exponent: exponent.into() })
    }

    fn input(&self, tag: impl Into<mxx_dsl::HashTag>, exponent: usize) -> BggEncodingWire {
        let mut values = self
            .sampler
            .sample_input_encodings(
                self.mask_secret.clone(),
                Some(self.payload_secret.clone()),
                self.hash_key.clone(),
                tag,
                &[self.rotation(exponent)],
            )
            .expect("ciphertext-only input sample");
        values.pop().expect("one input encoding")
    }

    fn public_input(&self, tag: impl Into<mxx_dsl::HashTag>) -> BggPublicKeyWire {
        let mut values = self
            .public_sampler
            .sample_input_keys(self.hash_key.clone(), tag, 1)
            .expect("ciphertext-only public input sample");
        values.pop().expect("one public input key")
    }

    fn helpers(
        &self,
        table: &[usize],
        tag: impl Into<mxx_dsl::HashTag>,
    ) -> (Vec<FlatLutHelper>, Vec<FlatLutPublicHelper>) {
        let tag = tag.into();
        let table =
            LutTable::unary(table.len(), self.parameters.ring_dimension() as usize, table.to_vec())
                .expect("flat LUT table");
        let private_bank = self
            .sampler
            .sample_flat_mask_bank(
                self.mask_secret.clone(),
                self.hash_key.clone(),
                table.values().len(),
                tag.clone(),
            )
            .expect("flat mask bank");
        let public_bank = self
            .public_sampler
            .sample_flat_mask_bank(self.hash_key.clone(), table.values().len(), tag.clone())
            .expect("public flat mask bank");
        self.helpers_with_banks(&table, tag, private_bank.as_ref(), &public_bank)
    }

    fn helpers_with_banks(
        &self,
        table: &LutTable,
        tag: impl Into<mxx_dsl::HashTag>,
        private_bank: &FlatLutMaskBank,
        public_bank: &std::sync::Arc<FlatLutPublicMaskBank>,
    ) -> (Vec<FlatLutHelper>, Vec<FlatLutPublicHelper>) {
        let tag = tag.into();
        let private = self
            .sampler
            .sample_flat_helpers_for_lut(
                self.mask_secret.clone(),
                Some(self.payload_secret.clone()),
                self.hash_key.clone(),
                table,
                private_bank,
                tag,
            )
            .expect("flat setup helpers");
        let public = private
            .iter()
            .map(|helper| {
                FlatLutPublicHelper::with_mask_bank(
                    helper.sigma(),
                    helper.switch().public_projection(),
                    public_bank.clone(),
                )
                .expect("public flat helper mask branch")
            })
            .collect();
        (private, public)
    }

    fn rhs(&self, tag: impl Into<mxx_dsl::HashTag>, exponent: usize) -> ExponentRhsPackage {
        self.sampler
            .sample_cross_secret_rhs(
                self.payload_secret.clone(),
                self.payload_secret.clone(),
                self.rotation(exponent),
                self.hash_key.clone(),
                tag,
            )
            .expect("C-only RHS package")
    }

    fn execute(&self, name: &str, relations: Vec<Relation>) {
        let gadget = self.ring.gadget(2, self.compiler.bgg.public_key.base.clone(), 2);
        let mut context = DslContext::new(name);
        for (index, relation) in relations.iter().enumerate() {
            assert!(relation.encoded.plaintext.is_none());
            assert!(!relation.encoded.pubkey.reveal_plaintext);
            let expected_vector = self.mask_secret.clone() * relation.expected_public.clone() -
                relation.mu.clone() * (self.payload_secret.clone() * gadget.clone());
            context = context
                .output(format!("vector-{index}"), relation.encoded.vector.clone())
                .unwrap()
                .output(format!("public-{index}"), relation.encoded.pubkey.matrix.clone())
                .unwrap()
                .output(format!("expected-vector-{index}"), expected_vector)
                .unwrap()
                .output(format!("expected-public-{index}"), relation.expected_public.clone())
                .unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let mask = concrete_secret(&self.parameters, [2, 1]);
        let payload = concrete_secret(&self.parameters, [1, 2]);
        let result = execute(
            &graph,
            &mut gpu_backend([self.gpu_parameters.clone()]),
            BTreeMap::from([
                (
                    "exponent-lut-gpu-mask-secret".to_owned(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                        &self.gpu_parameters,
                        &mask,
                    )),
                ),
                (
                    "exponent-lut-gpu-payload-secret".to_owned(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                        &self.gpu_parameters,
                        &payload,
                    )),
                ),
                ("exponent-lut-gpu-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x91; 32])),
            ]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for index in 0..relations.len() {
            let RuntimeValue::Matrix(vector) = &result.outputs[&format!("vector-{index}")] else {
                panic!("vector output")
            };
            let RuntimeValue::Matrix(expected_vector) =
                &result.outputs[&format!("expected-vector-{index}")]
            else {
                panic!("expected vector output")
            };
            let RuntimeValue::Matrix(public) = &result.outputs[&format!("public-{index}")] else {
                panic!("public output")
            };
            let RuntimeValue::Matrix(expected_public) =
                &result.outputs[&format!("expected-public-{index}")]
            else {
                panic!("expected public output")
            };
            assert_eq!(vector.as_ref(), expected_vector.as_ref(), "relation {index}: vector");
            assert_eq!(
                public.as_ref(),
                expected_public.as_ref(),
                "relation {index}: public matrix"
            );
        }
    }
}

fn concrete_secret(parameters: &DCRTPolyParams, coefficients: [u32; 2]) -> DCRTPolyMatrix {
    let mut values = vec![0u32; parameters.ring_dimension() as usize];
    values[..coefficients.len()].copy_from_slice(&coefficients);
    DCRTPolyMatrix::from_poly_vec(
        parameters,
        vec![vec![
            DCRTPoly::from_u32s(parameters, &values),
            DCRTPoly::from_usize_to_constant(parameters, 1),
        ]],
    )
}

fn unary_table() -> Vec<usize> {
    vec![1, 0]
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_flat_unary_exhaustive_inputs_distinct_secrets() {
    let fixture = Fixture::new();
    let table = unary_table();
    let (helpers, public_helpers) = fixture.helpers(&table, b"unary-helpers".as_slice());
    let public_input = fixture.public_input(b"unary-input".as_slice());
    let mut relations = Vec::new();
    for exponent in 0..table.len() {
        let encoded = fixture
            .compiler
            .single_input_lut(&fixture.input(b"unary-input".as_slice(), exponent), &table, &helpers)
            .unwrap();
        let expected_public = fixture
            .public_compiler
            .single_input_lut(&public_input.matrix, &table, &public_helpers)
            .unwrap();
        relations.push(Relation {
            encoded,
            expected_public,
            mu: fixture.rotation(table[exponent]),
        });
    }
    fixture.execute("exponent-lut-flat-unary-gpu", relations);
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_flat_binary_all_u_plus_bv_pairs() {
    let fixture = Fixture::new();
    let table = LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap();
    let (helpers, public_helpers) = fixture.helpers(table.values(), b"binary-helpers".as_slice());
    let public_lhs = fixture.public_input(b"binary-lhs".as_slice());
    let mut relations = Vec::new();
    for lhs_exponent in 0..2 {
        for rhs_exponent in 0..2 {
            let rhs =
                fixture.rhs(format!("binary-rhs-{rhs_exponent}").into_bytes(), 2 * rhs_exponent);
            let encoded = fixture
                .compiler
                .two_input_lut(
                    &fixture.input(b"binary-lhs".as_slice(), lhs_exponent),
                    &rhs,
                    2,
                    2,
                    table.values(),
                    &helpers,
                )
                .unwrap();
            let expected_public = fixture
                .public_compiler
                .two_input_lut(&public_lhs.matrix, &rhs, 2, 2, table.values(), &public_helpers)
                .unwrap();
            let index = lhs_exponent + 2 * rhs_exponent;
            relations.push(Relation {
                encoded,
                expected_public,
                mu: fixture.rotation(table.values()[index]),
            });
        }
    }
    fixture.execute("exponent-lut-flat-binary-gpu", relations);
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_flat_two_stage_lut_chain() {
    let fixture = Fixture::new();
    let first_table = vec![0, 1];
    let second_table = unary_table();
    let private_bank = fixture
        .sampler
        .sample_flat_mask_bank(
            fixture.mask_secret.clone(),
            fixture.hash_key.clone(),
            2,
            b"chain-mask-bank".as_slice(),
        )
        .expect("chain private mask bank");
    let public_bank = fixture
        .public_sampler
        .sample_flat_mask_bank(fixture.hash_key.clone(), 2, b"chain-mask-bank".as_slice())
        .expect("chain public mask bank");
    let (first_helpers, first_public_helpers) = fixture.helpers_with_banks(
        &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, first_table.clone())
            .unwrap(),
        b"chain-first".as_slice(),
        private_bank.as_ref(),
        &public_bank,
    );
    let (second_helpers, second_public_helpers) = fixture.helpers_with_banks(
        &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, second_table.clone())
            .unwrap(),
        b"chain-second".as_slice(),
        private_bank.as_ref(),
        &public_bank,
    );
    let input = fixture.input(b"chain-input".as_slice(), 1);
    let public_input = fixture.public_input(b"chain-input".as_slice());
    let first = fixture.compiler.single_input_lut(&input, &first_table, &first_helpers).unwrap();
    let first_public = fixture
        .public_compiler
        .single_input_lut(&public_input.matrix, &first_table, &first_public_helpers)
        .unwrap();
    let second = fixture.compiler.single_input_lut(&first, &second_table, &second_helpers).unwrap();
    let second_public = fixture
        .public_compiler
        .single_input_lut(&first_public, &second_table, &second_public_helpers)
        .unwrap();
    fixture.execute(
        "exponent-lut-flat-chain-gpu",
        vec![Relation { encoded: second, expected_public: second_public, mu: fixture.rotation(0) }],
    );
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_flat_generic_exponent_lut_program() {
    let fixture = Fixture::new();
    let table = LutTable::unary(2, 2, unary_table()).unwrap();
    let mut builder = ExponentLutProgramBuilder::new();
    let input_id = builder.input(2).unwrap();
    let lut_id = builder.lut(table.clone()).unwrap();
    let output_id = builder.unary(builder.input_wire(input_id).unwrap(), lut_id).unwrap();
    builder.output(output_id).unwrap();
    let program = builder.build().unwrap();
    let (private_helpers, public_helpers) =
        fixture.helpers(table.values(), b"program-helpers".as_slice());
    let private_set = FlatLutHelperSet::new(&table, private_helpers).unwrap();
    let public_set = FlatLutPublicHelperSet::new(&table, public_helpers).unwrap();
    let encoded = fixture.input(b"program-input".as_slice(), 1);
    let public_input = fixture.public_input(b"program-input".as_slice());
    let actual = fixture
        .compiler
        .compile_program(
            &program,
            &BTreeMap::from([(input_id, encoded)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::from([(lut_id, private_set)]),
        )
        .unwrap();
    let expected = fixture
        .public_compiler
        .compile_program(
            &program,
            &BTreeMap::from([(input_id, public_input)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::from([(lut_id, public_set)]),
        )
        .unwrap();
    fixture.execute(
        "exponent-lut-flat-program-gpu",
        vec![Relation {
            encoded: actual[&output_id].clone(),
            expected_public: expected[&output_id].matrix.clone(),
            mu: fixture.rotation(0),
        }],
    );
}
