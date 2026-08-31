//! GPU-only Power-LUT regression-test module.
//!
//! Heavy sampled-ciphertext tests belong in this file so enabling the `gpu`
//! feature cannot accidentally place GPU-facing code in a CPU-only module.
//! The concrete tests are kept beside the encoding fixtures and are enabled
//! only when a CUDA-capable runtime is selected.

use crate::{PowerLutEncodingCompiler, PowerLutPublicKeyCompiler, encoding::AutomorphismHelper};
use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_ir_core::ParamEnv;
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
use std::collections::BTreeMap;

struct Relation {
    encoded: BggEncodingWire,
    expected_public: Mat,
    expected_mu: Mat,
}

/// Runtime fixture for the migrated GPU tests. Setup and expected values use
/// the CPU polynomial type, while the single execution below evaluates every
/// relation through the GPU backend.
struct GpuNoiselessFixture {
    parameters: DCRTPolyParams,
    gpu_parameters: GpuDCRTPolyParams,
    ring: Ring,
    compiler: PowerLutEncodingCompiler,
    public_compiler: PowerLutPublicKeyCompiler,
    sampler: crate::encoding::PowerLutEncodingSampler,
    public_sampler: crate::public_key::PowerLutPublicKeySampler,
    secret: Mat,
    hash_key: mxx_dsl::Bytes,
}

impl GpuNoiselessFixture {
    fn new() -> Self {
        Self::with_dimension(4)
    }

    fn with_dimension(ring_dimension: u32) -> Self {
        let parameters = DCRTPolyParams::new(ring_dimension, 1, 17, 9);
        assert_eq!(parameters.crt_depth(), 1);
        assert_eq!(parameters.crt_bits().div_ceil(9), 2);
        let (moduli, crt_bits, crt_depth) = parameters.to_crt();
        assert_eq!(crt_depth, 1);
        assert_eq!(crt_bits.div_ceil(9), 2);
        let gpu_parameters =
            GpuDCRTPolyParams::new(parameters.ring_dimension(), moduli, parameters.base_bits());
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus.clone(), ring_dimension as usize);
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: modulus.into(),
            ring_dimension: (ring_dimension as usize).into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 512.into(),
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
            compiler: PowerLutEncodingCompiler::from_public_key(bgg.clone()),
            public_compiler: PowerLutPublicKeyCompiler::new(bgg),
            sampler: crate::encoding::PowerLutEncodingSampler {
                layout: layout.clone(),
                gaussian_sigma: None,
                gaussian_max_coefficient_bound: None,
            },
            public_sampler: crate::public_key::PowerLutPublicKeySampler { layout },
            secret: ring.input("noiseless-secret", (1, 2)),
            hash_key: ring.bytes_input("noiseless-hash", 32),
        }
    }

    fn rotation(&self, exponent: usize) -> Mat {
        self.ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::Rotation { exponent: exponent.into() },
        )
    }

    fn input(&self, tag: &[u8], exponent: usize) -> BggEncodingWire {
        self.sampler
            .sample_input_encoding(
                self.secret.clone(),
                self.hash_key.clone(),
                tag,
                self.rotation(exponent),
            )
            .expect("noiseless sampled input")
    }

    fn public_input(&self, tag: &[u8]) -> BggPublicKeyWire {
        self.public_sampler
            .sample_input_key(self.hash_key.clone(), tag)
            .expect("noiseless public input")
    }

    fn helpers(&self, tag: &[u8], width: usize) -> Vec<AutomorphismHelper> {
        self.sampler
            .sample_automorphism_helpers(self.secret.clone(), self.hash_key.clone(), tag, width)
            .expect("noiseless sampled helpers")
    }

    fn public_helpers(
        &self,
        tag: &[u8],
        width: usize,
    ) -> Vec<crate::public_key::AutomorphismPublicHelper> {
        self.public_sampler
            .sample_automorphism_helpers(self.hash_key.clone(), tag, width)
            .expect("noiseless public helpers")
    }

    fn execute_relations(&self, name: &str, relations: Vec<Relation>) {
        let gadget = self.ring.gadget(2, 512, 2);
        let mut context = DslContext::new(name);
        for (index, relation) in relations.iter().enumerate() {
            let expected_vector = self.secret.clone() * relation.expected_public.clone() -
                relation.expected_mu.clone() * (self.secret.clone() * gadget.clone());
            context = context
                .output(format!("relation-{index}-vector"), relation.encoded.vector.clone())
                .unwrap()
                .output(format!("relation-{index}-public"), relation.encoded.pubkey.matrix.clone())
                .unwrap()
                .output(
                    format!("relation-{index}-expected-public"),
                    relation.expected_public.clone(),
                )
                .unwrap()
                .output(format!("relation-{index}-expected-vector"), expected_vector)
                .unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let mut secret_coefficients = vec![0; self.parameters.ring_dimension() as usize];
        secret_coefficients[0] = 2;
        secret_coefficients[1] = 1;
        let secret = DCRTPolyMatrix::from_poly_vec(
            &self.parameters,
            vec![vec![
                DCRTPoly::from_u32s(&self.parameters, &secret_coefficients),
                DCRTPoly::from_usize_to_constant(&self.parameters, 1),
            ]],
        );
        let result = execute(
            &graph,
            &mut gpu_backend([self.gpu_parameters.clone()]),
            BTreeMap::from([
                (
                    "noiseless-secret".to_owned(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                        &self.gpu_parameters,
                        &secret,
                    )),
                ),
                ("noiseless-hash".to_owned(), RuntimeValue::Bytes(vec![0x91; 32])),
            ]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for index in 0..relations.len() {
            let RuntimeValue::Matrix(vector) = &result.outputs[&format!("relation-{index}-vector")]
            else {
                panic!("encoded vector")
            };
            let RuntimeValue::Matrix(expected_vector) =
                &result.outputs[&format!("relation-{index}-expected-vector")]
            else {
                panic!("expected vector")
            };
            let RuntimeValue::Matrix(public) = &result.outputs[&format!("relation-{index}-public")]
            else {
                panic!("encoded public")
            };
            let RuntimeValue::Matrix(expected_public) =
                &result.outputs[&format!("relation-{index}-expected-public")]
            else {
                panic!("expected public")
            };
            assert_eq!(public, expected_public, "relation {index}: public key mismatch");
            assert_eq!(vector, expected_vector, "relation {index}: noiseless BGG relation");
        }
    }
}

#[test]
#[serial(dcrt_runtime)]
fn noiseless_clear_coeff_and_exhaustive_single_lut_use_independent_public_keys() {
    let fixture = GpuNoiselessFixture::new();
    let helpers = fixture.helpers(b"clear-helper", 4);
    let public_helpers = fixture.public_helpers(b"clear-helper", 4);
    let input = fixture.input(b"clear-input", 1);
    let public_input = fixture.public_input(b"clear-input");
    let clear = fixture.compiler.clear_coeff(&input, 4, &helpers).unwrap();
    let clear_public =
        fixture.public_compiler.clear_coeff(&public_input.matrix, 4, &public_helpers).unwrap();
    let mut expected_mu = fixture.rotation(1);
    for helper in [5usize, 3] {
        expected_mu = expected_mu.clone() + expected_mu.clone().ring_automorphism(helper);
    }
    let mut relations =
        vec![Relation { encoded: clear, expected_public: clear_public, expected_mu }];
    let single_public_input = fixture.public_input(b"single-input");
    for (width, table) in [(2usize, vec![1usize, 0]), (4, vec![3, 1, 0, 2])] {
        let helpers = fixture.helpers(b"single-helper", width);
        let public_helpers = fixture.public_helpers(b"single-helper", width);
        for input_exponent in 0..width {
            let encoded = fixture
                .compiler
                .single_input_lut(&fixture.input(b"single-input", input_exponent), &table, &helpers)
                .unwrap();
            let public = fixture
                .public_compiler
                .single_input_lut(&single_public_input.matrix, &table, &public_helpers)
                .unwrap();
            relations.push(Relation {
                encoded,
                expected_public: public,
                expected_mu: fixture.rotation(table[input_exponent]),
            });
        }
    }
    fixture.execute_relations("power-lut-noiseless-clear-and-single-gpu", relations);
}

#[test]
#[serial(dcrt_runtime)]
fn noiseless_two_stage_program_uses_shared_public_projection() {
    let fixture = GpuNoiselessFixture::new();
    let helpers = fixture.helpers(b"stages-helper", 2);
    let public_helpers = fixture.public_helpers(b"stages-helper", 2);
    let first = fixture
        .compiler
        .single_input_lut(&fixture.input(b"stages-input", 1), &[0, 1], &helpers)
        .unwrap();
    let first_public = fixture
        .public_compiler
        .single_input_lut(&fixture.public_input(b"stages-input").matrix, &[0, 1], &public_helpers)
        .unwrap();
    let second = fixture.compiler.single_input_lut(&first, &[1, 0], &helpers).unwrap();
    let second_public =
        fixture.public_compiler.single_input_lut(&first_public, &[1, 0], &public_helpers).unwrap();
    fixture.execute_relations(
        "power-lut-noiseless-two-stage-gpu",
        vec![Relation {
            encoded: second,
            expected_public: second_public,
            expected_mu: fixture.rotation(0),
        }],
    );
}

#[test]
#[serial(dcrt_runtime)]
fn noiseless_generic_program_matches_independent_public_lowering() {
    let fixture = GpuNoiselessFixture::new();
    let mut builder = crate::program::PowerLutProgramBuilder::new();
    let input_id = builder.input(2).unwrap();
    let lut = builder.lut(crate::program::LutTable::unary(2, 2, vec![1, 0]).unwrap()).unwrap();
    let output_id = builder.unary(builder.input_wire(input_id).unwrap(), lut).unwrap();
    builder.output(output_id).unwrap();
    let program = builder.build().unwrap();
    let private = fixture.input(b"program-input", 1);
    let public = fixture.public_input(b"program-input");
    let helpers = fixture.helpers(b"program-helper", 2);
    let public_helpers = fixture.public_helpers(b"program-helper", 2);
    let encoded = fixture
        .compiler
        .compile_program(
            &program,
            &BTreeMap::from([(input_id, private)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &helpers,
        )
        .unwrap()[&output_id]
        .clone();
    let public = fixture
        .public_compiler
        .compile_program(
            &program,
            &BTreeMap::from([(input_id, public)]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &public_helpers,
        )
        .unwrap()[&output_id]
        .clone();
    fixture.execute_relations(
        "power-lut-noiseless-generic-program-gpu",
        vec![Relation {
            encoded,
            expected_public: public.matrix,
            expected_mu: fixture.rotation(0),
        }],
    );
}

#[test]
#[serial(dcrt_runtime)]
fn noiseless_exhaustive_two_input_pairs_use_independent_public_keys() {
    let fixture = GpuNoiselessFixture::with_dimension(16);
    let mut relations = Vec::new();
    for width in [2usize, 4] {
        let flattened_width = width * width;
        let helpers = fixture.helpers(b"binary-helper", flattened_width);
        let public_helpers = fixture.public_helpers(b"binary-helper", flattened_width);
        let table =
            (0..flattened_width).map(|value| (3 * value + 1) % flattened_width).collect::<Vec<_>>();
        let lhs_public = fixture.public_input(format!("binary-lhs-{width}").as_bytes());
        assert!(!lhs_public.reveal_plaintext);
        let rhs_packages = (0..width)
            .map(|rhs_exponent| {
                let tag = format!("binary-rhs-{width}-{rhs_exponent}");
                fixture
                    .sampler
                    .sample_cross_secret_rhs(
                        fixture.secret.clone(),
                        fixture.secret.clone(),
                        fixture.rotation(width * rhs_exponent),
                        fixture.hash_key.clone(),
                        tag.clone().into_bytes(),
                    )
                    .expect("noiseless binary RHS")
            })
            .collect::<Vec<_>>();
        let rhs_public = (0..width)
            .map(|rhs_exponent| {
                fixture
                    .public_sampler
                    .sample_rhs_public(
                        fixture.hash_key.clone(),
                        format!("binary-rhs-{width}-{rhs_exponent}").into_bytes(),
                    )
                    .expect("noiseless public binary RHS")
            })
            .collect::<Vec<_>>();
        for lhs_exponent in 0..width {
            let lhs = fixture.input(format!("binary-lhs-{width}").as_bytes(), lhs_exponent);
            assert!(lhs.plaintext.is_none());
            assert!(!lhs.pubkey.reveal_plaintext);
            for rhs_exponent in 0..width {
                let rhs = &rhs_packages[rhs_exponent];
                let encoded = fixture
                    .compiler
                    .two_input_lut(&lhs, rhs, width, width, &table, &helpers)
                    .unwrap_or_else(|error| {
                        panic!("two-input width {width} ({lhs_exponent},{rhs_exponent}): {error:?}")
                    });
                assert!(encoded.plaintext.is_none());
                assert!(!encoded.pubkey.reveal_plaintext);
                let public = fixture
                    .public_compiler
                    .two_input_lut(
                        &lhs_public.matrix,
                        &rhs_public[rhs_exponent],
                        width,
                        width,
                        &table,
                        &public_helpers,
                    )
                    .expect("binary LUT public projection");
                let k = crate::flattened_lut_index(lhs_exponent, rhs_exponent, width, width)
                    .expect("binary LUT index");
                relations.push(Relation {
                    encoded,
                    expected_public: public,
                    expected_mu: fixture.rotation(table[k]),
                });
            }
        }
    }
    fixture.execute_relations("power-lut-noiseless-two-input-gpu", relations);
}
