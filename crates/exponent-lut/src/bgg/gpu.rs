// GPU runtime coverage for the setup-fixed native Exponent-LUT evaluator.
//
// These tests deliberately build the private encoding and public projection
// through separate compilers. The expected vector is the concrete BGG+
// relation `s_mask * A - mu * s_payload * G`; it is not obtained from the
// private evaluator's output.

use crate::{
    bgg::{
        encoding::{
            ExponentLutEncodingCompiler, ExponentLutEncodingSampler, LutHelper, LutHelperSet,
            LutMaskBank, LutPublicMaskBank,
        },
        public_key::{
            ExponentLutPublicKeyCompiler, ExponentLutPublicKeySampler, LutPublicHelper,
            LutPublicHelperSet,
        },
    },
    program::{AttributeKnowledge, ExponentLutProgramBuilder, LutTable, ProgramWireType},
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
    fn new(width_log2: u32) -> Self {
        let ring_dimension = parameter(RING_DIMENSION_ENV, 4);
        let crt_bits = parameter(CRT_BITS_ENV, 18);
        let base_bits = parameter(BASE_BITS_ENV, 9);
        assert!(ring_dimension.is_power_of_two() && ring_dimension >= 2);
        assert!(base_bits > 0 && base_bits <= crt_bits / 2);
        let parameters =
            DCRTPolyParams::new(ring_dimension, 1, crt_bits as usize, base_bits, None, None);
        assert_eq!(parameters.modulus_digits(), 2, "GPU fixture requires digit_count=2");
        let (moduli, _, crt_depth) = parameters.to_crt();
        assert_eq!(crt_depth, 1);
        let gpu_parameters =
            GpuDCRTPolyParams::new(ring_dimension, moduli, parameters.base_bits(), None);
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
                exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2 },
                gaussian_sigma: None,
                gaussian_max_coefficient_bound: None,
            },
            public_sampler: ExponentLutPublicKeySampler {
                layout,
                exponent_layout: crate::layout::ExpLayout { stride_log2: 0, width_log2 },
            },
            compiler: ExponentLutEncodingCompiler::with_layout(
                bgg.clone(),
                crate::layout::ExpLayout { stride_log2: 0, width_log2 },
            ),
            public_compiler: ExponentLutPublicKeyCompiler::with_layout(
                bgg,
                crate::layout::ExpLayout { stride_log2: 0, width_log2 },
            ),
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
    ) -> (Vec<LutHelper>, Vec<LutPublicHelper>) {
        let tag = tag.into();
        let table =
            LutTable::unary(table.len(), self.parameters.ring_dimension() as usize, table.to_vec())
                .expect("native LUT table");
        let private_bank = self
            .sampler
            .sample_mask_bank(
                self.mask_secret.clone(),
                self.hash_key.clone(),
                table.values().len(),
                tag.clone(),
            )
            .expect("native mask bank");
        let public_bank = self
            .public_sampler
            .sample_mask_bank(self.hash_key.clone(), table.values().len(), tag.clone())
            .expect("public native mask bank");
        self.helpers_with_banks(&table, tag, private_bank.as_ref(), &public_bank, None)
    }

    fn helpers_with_banks(
        &self,
        table: &LutTable,
        tag: impl Into<mxx_dsl::HashTag>,
        private_bank: &LutMaskBank,
        public_bank: &std::sync::Arc<LutPublicMaskBank>,
        rhs: Option<Mat>,
    ) -> (Vec<LutHelper>, Vec<LutPublicHelper>) {
        let tag = tag.into();
        let private = self
            .sampler
            .sample_helpers_for_lut(
                self.mask_secret.clone(),
                Some(self.payload_secret.clone()),
                self.hash_key.clone(),
                table,
                private_bank,
                tag,
                rhs,
            )
            .expect("native setup helpers");
        let public = private
            .iter()
            .map(|helper| {
                LutPublicHelper::with_mask_bank(
                    helper.sigma(),
                    helper.switch().public_projection(),
                    public_bank.clone(),
                )
                .expect("public native helper mask branch")
            })
            .collect();
        (private, public)
    }

    fn execute(&self, name: &str, relations: Vec<Relation>) {
        let gadget = self.ring.gadget(
            2,
            self.compiler.gadget_base.clone(),
            self.compiler.digit_count.clone(),
        );
        let mut context = DslContext::new(name);
        for (index, relation) in relations.iter().enumerate() {
            assert!(relation.encoded.plaintext.is_none());
            let expected_vector = self.mask_secret.clone() * relation.expected_public.clone() -
                relation.mu.clone() * (self.payload_secret.clone() * gadget.clone());
            context = context
                .output(format!("vector-{index}"), relation.encoded.vector.clone())
                .unwrap()
                .output(format!("expected-vector-{index}"), expected_vector)
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
                    RuntimeValue::matrix(
                        GpuDCRTPolyMatrix::from_cpu_matrix(&self.gpu_parameters, &mask).into(),
                    ),
                ),
                (
                    "exponent-lut-gpu-payload-secret".to_owned(),
                    RuntimeValue::matrix(
                        GpuDCRTPolyMatrix::from_cpu_matrix(&self.gpu_parameters, &payload).into(),
                    ),
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
            assert_eq!(vector.as_ref(), expected_vector.as_ref(), "relation {index}: vector");
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
fn test_gpu_native_unary_exhaustive_inputs_distinct_secrets() {
    let fixture = Fixture::new(1);
    let table = unary_table();
    let (helpers, public_helpers) = fixture.helpers(&table, b"unary-helpers".as_slice());
    let public_input = fixture.public_input(b"unary-input".as_slice());
    let cache = fixture
        .public_compiler
        .build_lut_cache(
            &public_input.matrix,
            &LutTable::unary(
                table.len(),
                fixture.parameters.ring_dimension() as usize,
                table.clone(),
            )
            .unwrap(),
            &public_helpers,
        )
        .unwrap();
    let mut relations = Vec::new();
    for exponent in 0..table.len() {
        let encoded = fixture
            .compiler
            .test_single_input_lut(
                &fixture.input(b"unary-input".as_slice(), exponent),
                &table,
                &helpers,
                &cache,
            )
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
    fixture.execute("exponent-lut-native-unary-gpu", relations);
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_native_binary_all_u_plus_bv_pairs() {
    let fixture = Fixture::new(2);
    let table = LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap();
    let private_bank = fixture
        .sampler
        .sample_mask_bank(
            fixture.mask_secret.clone(),
            fixture.hash_key.clone(),
            table.values().len(),
            b"binary-bank".as_slice(),
        )
        .unwrap();
    let public_bank = fixture
        .public_sampler
        .sample_mask_bank(fixture.hash_key.clone(), table.values().len(), b"binary-bank".as_slice())
        .unwrap();
    let public_lhs = fixture.public_input(b"binary-lhs".as_slice());
    let mut relations = Vec::new();
    for rhs_exponent in 0..2 {
        let (helpers, public_helpers) = fixture.helpers_with_banks(
            &table,
            format!("binary-helpers-{rhs_exponent}").into_bytes(),
            private_bank.as_ref(),
            &public_bank,
            Some(fixture.rotation(2 * rhs_exponent)),
        );
        // Each branch encrypts D_k*sigma_k(rhs) freshly. There is no
        // preliminary external product on either the key or encoding path.
        let cache = fixture
            .public_compiler
            .build_lut_cache(&public_lhs.matrix, &table, &public_helpers)
            .unwrap();
        let expected_public = fixture
            .public_compiler
            .single_input_lut(&public_lhs.matrix, table.values(), &public_helpers)
            .unwrap();
        for lhs_exponent in 0..2 {
            let encoded = fixture
                .compiler
                .single_input_lut_table_with_cache(
                    &fixture.input(b"binary-lhs".as_slice(), lhs_exponent),
                    &table,
                    &helpers,
                    Some(&cache),
                )
                .unwrap();
            relations.push(Relation {
                encoded,
                expected_public: expected_public.clone(),
                mu: fixture.rotation(table.values()[lhs_exponent + 2 * rhs_exponent]),
            });
        }
    }
    fixture.execute("exponent-lut-native-binary-gpu", relations);
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_native_two_stage_lut_chain() {
    let fixture = Fixture::new(1);
    let first_table = vec![0, 1];
    let second_table = unary_table();
    let private_bank = fixture
        .sampler
        .sample_mask_bank(
            fixture.mask_secret.clone(),
            fixture.hash_key.clone(),
            2,
            b"chain-mask-bank".as_slice(),
        )
        .expect("chain private mask bank");
    let public_bank = fixture
        .public_sampler
        .sample_mask_bank(fixture.hash_key.clone(), 2, b"chain-mask-bank".as_slice())
        .expect("chain public mask bank");
    let (first_helpers, first_public_helpers) = fixture.helpers_with_banks(
        &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, first_table.clone())
            .unwrap(),
        b"chain-first".as_slice(),
        private_bank.as_ref(),
        &public_bank,
        None,
    );
    let (second_helpers, second_public_helpers) = fixture.helpers_with_banks(
        &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, second_table.clone())
            .unwrap(),
        b"chain-second".as_slice(),
        private_bank.as_ref(),
        &public_bank,
        None,
    );
    let input = fixture.input(b"chain-input".as_slice(), 1);
    let public_input = fixture.public_input(b"chain-input".as_slice());
    let first_cache = fixture
        .public_compiler
        .build_lut_cache(
            &public_input.matrix,
            &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, first_table.clone())
                .unwrap(),
            &first_public_helpers,
        )
        .unwrap();
    let first = fixture
        .compiler
        .test_single_input_lut(&input, &first_table, &first_helpers, &first_cache)
        .unwrap();
    let first_public = fixture
        .public_compiler
        .single_input_lut(&public_input.matrix, &first_table, &first_public_helpers)
        .unwrap();
    let second_cache = fixture
        .public_compiler
        .build_lut_cache(
            &first_public,
            &LutTable::unary(2, fixture.parameters.ring_dimension() as usize, second_table.clone())
                .unwrap(),
            &second_public_helpers,
        )
        .unwrap();
    let second = fixture
        .compiler
        .test_single_input_lut(&first, &second_table, &second_helpers, &second_cache)
        .unwrap();
    let second_public = fixture
        .public_compiler
        .single_input_lut(&first_public, &second_table, &second_public_helpers)
        .unwrap();
    fixture.execute(
        "exponent-lut-native-chain-gpu",
        vec![Relation { encoded: second, expected_public: second_public, mu: fixture.rotation(0) }],
    );
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_native_generic_exponent_lut_program() {
    let fixture = Fixture::new(1);
    let table = LutTable::unary(2, 2, unary_table()).unwrap();
    let mut builder = ExponentLutProgramBuilder::new();
    builder
        .set_layout(
            crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
            fixture.parameters.ring_dimension() as usize,
        )
        .unwrap();
    let input_id = builder
        .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
        .unwrap();
    let lut_id = builder.register_lut(table.clone()).unwrap();
    let output_id = builder.lut(builder.input_wire(input_id).unwrap(), lut_id, None).unwrap();
    builder.output(output_id).unwrap();
    let program = builder.build().unwrap();
    let (private_helpers, public_helpers) =
        fixture.helpers(table.values(), b"program-helpers".as_slice());
    let private_set = LutHelperSet::new(&table, private_helpers).unwrap();
    let public_set = LutPublicHelperSet::new(&table, public_helpers).unwrap();
    let encoded = fixture.input(b"program-input".as_slice(), 1);
    let public_input = fixture.public_input(b"program-input".as_slice());
    let ranges = crate::program::ProgramFamilyBindings::new();
    let public_inputs = BTreeMap::from([(input_id, public_input)]);
    let public_sets = BTreeMap::from([(lut_id, public_set)]);
    let preprocessing = fixture
        .public_compiler
        .compile_program_with_preprocessing(
            &program,
            &public_inputs,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &ranges,
            &public_sets,
        )
        .unwrap();
    let actual = fixture
        .compiler
        .compile_program_with_ranges_and_cache(
            &program,
            &BTreeMap::from([(input_id, encoded)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &ranges,
            &BTreeMap::from([(lut_id, private_set)]),
            &preprocessing.cache,
        )
        .unwrap();
    let expected = fixture
        .public_compiler
        .compile_program(
            &program,
            &public_inputs,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &public_sets,
        )
        .unwrap();
    fixture.execute(
        "exponent-lut-native-program-gpu",
        vec![Relation {
            encoded: actual[&output_id].clone(),
            expected_public: expected[&output_id].matrix.clone(),
            mu: fixture.rotation(0),
        }],
    );
}

mod refresh_decoder_tests {
    // GPU execution of the production switched decoder graph.

    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        },
    };
    use mxx_runtime::{
        MemoryArtifactStore, RuntimeValue, backend::poly_gpu::gpu_backend_on,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    #[test]
    #[serial_test::serial(dcrt_runtime)]
    fn test_gpu_switched_decoder_numeric_round_trip_with_shared_fresh_error() {
        let (graph, parameters, bounds) =
            crate::bgg::encoding::refresh_tests::switched_decoder_numeric_fixture();
        let source = &parameters[0];
        let root = GpuDCRTPolyParams::new(
            source.ring_dimension(),
            source.to_crt().0,
            source.base_bits(),
            None,
        );
        let selected = parameters
            .iter()
            .skip(1)
            .map(|p| root.select_modulus(p.modulus().as_ref()).unwrap())
            .collect::<Vec<_>>();
        let mut rings = vec![root];
        rings.extend(selected);
        let mut backend = gpu_backend_on(rings, detected_gpu_device_ids());
        for _execution in 0..2 {
            let result = mxx_runtime::execute(
                &graph,
                &mut backend,
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let mut observations = BTreeMap::new();
            for (name, value) in result.outputs {
                let RuntimeValue::Matrix(value) = value else {
                    panic!("matrix output expected");
                };
                // Host copies occur only after the entire GPU graph, at the assertion boundary.
                let matrix = backend.gather_matrix_for_host(&value).unwrap();
                let mut coefficients = Vec::new();
                for row in 0..matrix.row_size() {
                    for column in 0..matrix.col_size() {
                        coefficients.extend(matrix.entry(row, column).coeffs_biguints());
                    }
                }
                observations
                    .insert(name, (matrix.params().modulus().as_ref().clone(), coefficients));
            }
            crate::bgg::encoding::refresh_tests::assert_numeric_refresh_outputs(
                observations,
                &bounds,
            );
        }
    }
}
