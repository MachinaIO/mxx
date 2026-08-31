//! GPU acceptance coverage for the setup-fixed Section-7 refresh path.
//!
//! This test owns a small local fixture so the runtime acceptance path crosses
//! the same selector export, preprocessing export/import, refresh, and
//! verification boundaries used by production.  The fixture intentionally
//! contains only the current fixed-C selector family and flat LUT helpers.

use crate::{
    PowerLutEncodingCompiler, PowerLutPublicKeyCompiler,
    encoding::{EncodingSelectorFamily, FlatLutHelperSet, PowerLutEncodingSampler},
    pbc::{
        PbcActiveCellIndex, PbcEncodedPublicVector, PbcParameters, PbcPublicVectorFamilyBinding,
        PbcRootSeed, PbcSelectorArtifactNames, PbcSelectorArtifacts, PbcTrustedSelectorBits,
        build_structural_selector_families, clear_pbc_inner_product, derive_lwr_vector,
        generate_key_layout,
    },
    prf::{RefreshPrfBatchInputs, SparseLwrPrfProfile, SparseLwrPrfProgram},
    program::{FamilyRange, LutTable, ProgramFamilyRanges},
    public_key::{
        FlatLutPublicHelper, FlatLutPublicHelperSet, PowerLutPublicKeySampler, PublicSelectorFamily,
    },
    refresh::{RefreshFreshErrorPrfOutput, RefreshMaskPrfOutput, RefreshPrfLabel},
    refresh_setup::{
        ImportedRefreshSetup, RefreshPreprocessingProducer, RefreshPreprocessingRequest,
        RefreshPrfInputs, RefreshSetupParameters, build_refresh_verification,
    },
};
use mxx_bgg::{BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout};
use mxx_dsl::{DslContext, Family, Mat, Sequential};
use mxx_ir_core::{
    ParamEnv, artifact::ArtifactConfidentiality, encoding::spec_hash, node::ConcatAxis,
};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    RuntimeValue,
    artifact::{ArtifactStore, MemoryArtifactStore},
    backend::{poly::gpu::gpu_backend, poly_gpu::GpuDcrtBackend},
    execute_in_session_with_config, execute_with_config,
    executor::ExecutionConfig,
    transcript::SamplingMode,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serial_test::serial;
use std::{collections::BTreeMap, ffi::OsStr, num::NonZeroUsize};
use tracing::info;
use tracing_subscriber::EnvFilter;

const RING_DIMENSION_ENV: &str = "MXX_POWER_LUT_REFRESH_SETUP_GPU_RING_DIMENSION";
const CRT_BITS_ENV: &str = "MXX_POWER_LUT_REFRESH_SETUP_GPU_CRT_BITS";
const BASE_BITS_ENV: &str = "MXX_POWER_LUT_REFRESH_SETUP_GPU_BASE_BITS";
const MAX_PARALLEL_INSTANCES_ENV: &str = "MXX_POWER_LUT_REFRESH_SETUP_GPU_MAX_PARALLEL_INSTANCES";
const DEFAULT_RING_DIMENSION: u32 = 4;
const DEFAULT_CRT_BITS: u32 = 17;
const DEFAULT_BASE_BITS: u32 = 16;
const DEFAULT_MAX_PARALLEL_INSTANCES: usize = 4;

fn positive_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .map(|value| {
            let parsed =
                value.parse().unwrap_or_else(|_| panic!("{name} must be a positive integer"));
            assert!(parsed > 0, "{name} must be a positive integer");
            parsed
        })
        .unwrap_or(default)
}

fn max_parallel_instances() -> NonZeroUsize {
    let value = std::env::var_os(MAX_PARALLEL_INSTANCES_ENV)
        .unwrap_or_else(|| OsStr::new(&DEFAULT_MAX_PARALLEL_INSTANCES.to_string()).to_os_string());
    let value =
        value.to_str().unwrap_or_else(|| panic!("{MAX_PARALLEL_INSTANCES_ENV} must be Unicode"));
    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{MAX_PARALLEL_INSTANCES_ENV} must be a positive integer"));
    NonZeroUsize::new(parsed)
        .unwrap_or_else(|| panic!("{MAX_PARALLEL_INSTANCES_ENV} must be positive"))
}

fn execution_config() -> ExecutionConfig {
    ExecutionConfig {
        max_parallel_instances: max_parallel_instances(),
        preimage_progress: None,
        release_fence_interval: None,
    }
}

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_power_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

struct Fixture {
    dcrt: DCRTPolyParams,
    gpu: GpuDCRTPolyParams,
    parameters: RefreshSetupParameters,
    compiler: PowerLutEncodingCompiler,
    request: RefreshPreprocessingRequest,
    selector_graph: mxx_dsl::BuiltGraph,
    selector_production: mxx_ir_core::artifact::ProductionId,
    selector_artifacts: PbcSelectorArtifacts,
    selector_bits: PbcTrustedSelectorBits,
    mask_secret_reference: DCRTPolyMatrix,
    payload_secret_reference: DCRTPolyMatrix,
    mask_secret: Mat,
    payload_secret: Mat,
    selected_prf_vector: Mat,
    selected_prf_public: Mat,
    selected_prf_public_independent: Mat,
    expected_scalar: Mat,
    independent_public_values_name: String,
    independent_public_values: Vec<u64>,
    public_values_name: String,
    public_values: Vec<u64>,
    expected_plaintext: Mat,
}

impl Fixture {
    fn new() -> Self {
        let ring_dimension = positive_u32(RING_DIMENSION_ENV, DEFAULT_RING_DIMENSION);
        let crt_bits = positive_u32(CRT_BITS_ENV, DEFAULT_CRT_BITS);
        let base_bits = positive_u32(BASE_BITS_ENV, DEFAULT_BASE_BITS);
        assert!(ring_dimension.is_power_of_two() && ring_dimension >= 2);
        assert!(base_bits <= crt_bits && base_bits < 63);

        let dcrt = DCRTPolyParams::new(ring_dimension, 2, crt_bits as usize, base_bits);
        let (crt_moduli, _, crt_depth) = dcrt.to_crt();
        assert_eq!(crt_depth, 2);
        let modulus = BigInt::from(dcrt.modulus().as_ref().clone());
        let layout = BggSamplerLayout {
            modulus: modulus.clone().into(),
            ring_dimension: (ring_dimension as usize).into(),
            secret_dimension: 2,
            digit_count: dcrt.modulus_digits(),
            gadget_base: (BigInt::from(1u8) << base_bits as usize).into(),
        };
        let ring = layout.ring();
        let sampler = PowerLutEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let public_sampler = PowerLutPublicKeySampler { layout: layout.clone() };
        // Use distinct deterministic augmented secrets: each has a ternary
        // prefix followed by the mandatory final coordinate one.
        let mask_secret =
            Mat::concat(ConcatAxis::Columns, vec![ring.polynomial([1.into()]), ring.identity(1)]);
        let payload_secret = Mat::concat(
            ConcatAxis::Columns,
            vec![ring.polynomial([BigInt::from(-1).into()]), ring.identity(1)],
        );
        let mask_secret_reference = DCRTPolyMatrix::from_poly_vec(
            &dcrt,
            vec![vec![
                DCRTPoly::from_usize_to_constant(&dcrt, 1),
                DCRTPoly::from_usize_to_constant(&dcrt, 1),
            ]],
        );
        let payload_secret_reference = DCRTPolyMatrix::from_poly_vec(
            &dcrt,
            vec![vec![
                DCRTPoly::from_biguint_to_constant(&dcrt, dcrt.modulus().as_ref().clone() - 1u8),
                DCRTPoly::from_usize_to_constant(&dcrt, 1),
            ]],
        );
        let hash_key = ring.bytes_input("refresh-setup-gpu-hash-key", 32);

        let pbc_parameters = PbcParameters::custom(1, 1, 2, 2, 1, None);
        let generated = generate_key_layout(&pbc_parameters, PbcRootSeed([0x27; 32]), &[0])
            .expect("small PBC fixture layout");
        let selector_bits = PbcTrustedSelectorBits::from_schedule(&generated, &ring, [0x4b; 32])
            .expect("trusted selector bits");
        let selector_names = PbcSelectorArtifactNames::canonicalize_schema(
            &generated.public_layout,
            [0x4b; 32],
            layout.secret_dimension,
            layout.public_key_columns(),
        )
        .expect("selector artifact names");
        let selector_artifacts = PbcSelectorArtifacts::from_structural(
            &generated.public_layout,
            [0x4b; 32],
            selector_names.clone(),
            &sampler.layout,
            [0x4b; 32],
            &sampler.layout,
            [0x4b; 32],
        )
        .expect("selector artifacts");
        let structural = build_structural_selector_families(
            &sampler,
            selector_bits.family().clone(),
            payload_secret.clone(),
            payload_secret.clone(),
            hash_key.clone(),
            &generated.public_layout,
            [0x4b; 32],
        )
        .expect("fixed-C selector family");
        let selector_graph = selector_artifacts
            .add_structural_family_outputs(
                DslContext::new("refresh-setup-gpu-selector"),
                &generated.public_layout,
                structural,
            )
            .expect("selector outputs")
            .private_output("refresh-setup-gpu-payload-secret", payload_secret.clone())
            .expect("selector payload secret output")
            .private_output("refresh-setup-gpu-mask-secret", mask_secret.clone())
            .expect("selector mask secret output")
            .build()
            .expect("selector graph");
        let selector_spec_hash =
            spec_hash(&selector_graph.graph, &ParamEnv::default()).expect("selector graph hash");
        let selector_production = mxx_ir_core::artifact::ProductionId {
            spec_hash: selector_spec_hash,
            execution_nonce: [0x52; 32],
        };
        let package_count = generated.public_layout.parameters.universe_size *
            generated.public_layout.parameters.hash_count +
            generated.public_layout.parameters.bucket_count;
        assert_eq!(package_count, 4);
        let mask_secret_input = ring.artifact_input(
            selector_production.clone(),
            "refresh-setup-gpu-mask-secret",
            (1, layout.secret_dimension),
            ArtifactConfidentiality::Private,
        );
        let gsw_name = crate::pbc::selector_family_artifact_name(&selector_artifacts, "gsw");
        let gsw = ring.family_artifact_input(
            selector_production.clone(),
            gsw_name,
            package_count,
            (layout.secret_dimension, layout.public_key_columns()),
            ArtifactConfidentiality::Public,
        );
        let public_selector =
            PublicSelectorFamily::new(gsw.clone()).expect("fixed-C public selector binding");
        let selectors = EncodingSelectorFamily::new(gsw).expect("fixed-C selector binding");
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: layout.gadget_base.clone(),
            digit_count: layout.digit_count.into(),
        });
        let refresh_id = [0x09; 32];
        // X^0=1 is the canonical nonzero fixture attribute.  Keep this
        // attribute identical across the sampled encodings and verification
        // target so a zero-valued PRF/input path cannot pass accidentally.
        let unit_attribute = ring.polynomial([1.into()]);
        let input_encodings = sampler
            .sample_input_encodings(
                mask_secret.clone(),
                Some(payload_secret.clone()),
                hash_key.clone(),
                b"refresh-setup-gpu-inputs".as_slice(),
                &[unit_attribute.clone(), unit_attribute.clone()],
            )
            .expect("sample state and fresh encodings");
        let state = input_encodings[0].clone();
        let fresh = input_encodings[1].clone();
        let reduction_table = (0..4).map(|value| value % 2).collect::<Vec<_>>();
        let reduction_lut = LutTable::unary(4, 4, reduction_table.clone()).expect("reduction LUT");
        let mask_bank = sampler
            .sample_flat_mask_bank(
                mask_secret.clone(),
                hash_key.clone(),
                4,
                b"refresh-setup-gpu-mask-bank".as_slice(),
            )
            .expect("sample shared mask bank");
        let public_mask_bank = public_sampler
            .sample_flat_mask_bank(hash_key.clone(), 4, b"refresh-setup-gpu-mask-bank".as_slice())
            .expect("sample independent public mask bank");
        let helpers = BTreeMap::from([(
            crate::program::LutId::from_index(0),
            FlatLutHelperSet::new(
                &reduction_lut,
                sampler
                    .sample_flat_helpers_for_lut(
                        mask_secret.clone(),
                        Some(payload_secret.clone()),
                        hash_key.clone(),
                        &reduction_lut,
                        mask_bank.as_ref(),
                        b"refresh-setup-gpu-reduce".as_slice(),
                    )
                    .expect("sample reduction helpers"),
            )
            .expect("reduction helper set"),
        )]);

        let labels = crate::refresh::RefreshPrfLabelIndex::new(
            refresh_id,
            crt_moduli.len(),
            layout.secret_dimension,
            1,
            layout.digit_count,
        )
        .expect("refresh labels");
        let refresh = crate::refresh::RefreshCompiler {
            full_modulus: modulus.clone().into(),
            crt_plaintext_moduli: crt_moduli
                .iter()
                .map(|value| BigInt::from(*value).into())
                .collect(),
            reconstruction_coefficients: dcrt
                .reconst_coeffs()
                .into_iter()
                .map(|value| BigInt::from_biguint(num_bigint::Sign::Plus, value).into())
                .collect(),
        };
        let profile = SparseLwrPrfProfile::new(2, 2, 4, ring_dimension as usize)
            .expect("small sparse-LWR profile");
        let parameters = RefreshSetupParameters::new(
            refresh_id,
            profile.p(),
            layout.secret_dimension,
            1,
            layout.digit_count,
            4,
            layout.clone(),
            refresh,
            1.into(),
            "refresh-setup-gpu",
        );
        let program =
            SparseLwrPrfProgram::new(profile.clone(), generated.public_layout.bucket_width)
                .expect("sparse-LWR program");
        let rounding_lut = program
            .rounding_program()
            .lut(crate::program::LutId::from_index(0))
            .expect("rounding LUT");
        // Sample helpers from the declared terminal scalar LUT using the
        // explicitly shared bank; the table preserves its scalar output form.
        let rounding_helpers = sampler
            .sample_flat_helpers_for_lut(
                mask_secret.clone(),
                Some(payload_secret.clone()),
                hash_key.clone(),
                rounding_lut,
                mask_bank.as_ref(),
                b"refresh-setup-gpu-rounding".as_slice(),
            )
            .expect("sample rounding helpers");
        let rounding_helpers =
            FlatLutHelperSet::new(rounding_lut, rounding_helpers).expect("rounding helper set");
        let batch = RefreshPrfBatchInputs::new(&generated.public_layout, profile.clone(), &labels)
            .expect("refresh PRF batch");
        let public_values =
            batch.materialize_public_values(&labels).expect("materialize public PBC values");
        assert!(public_values.iter().any(|value| *value != 0));
        // Independently derive the scalar represented by the selected PBC
        // output.  This is the clear schedule oracle followed by the
        // specified floor(p*z/Q_L) rounding rule; it does not inspect any
        // encoded output.
        // The batch payload above is active-cell, label-major storage for the
        // runtime family.  The clear oracle deliberately receives the dense
        // universe vector for the selected label, derived by the same
        // label/domain hash function used by `PbcEncodedPublicVector`.
        let selected_label = labels.label(0).expect("selected PRF label").canonical_bytes();
        let oracle_public_vector = derive_lwr_vector(
            generated.public_layout.layout_id,
            &selected_label,
            generated.public_layout.parameters.universe_size,
            profile.q_l(),
        )
        .expect("derive selected clear PBC public vector");
        let oracle_public_vector_u64 =
            oracle_public_vector.iter().map(|value| *value as u64).collect::<Vec<_>>();
        let z = clear_pbc_inner_product(
            &generated.public_layout,
            generated.private_schedule(),
            &oracle_public_vector_u64,
            profile.q_l() as u64,
        )
        .expect("clear PBC inner-product oracle");
        let expected_scalar = ring.polynomial([BigInt::from(
            (profile.p() as u128 * z as u128 / profile.q_l() as u128) as u64,
        )
        .into()]);
        let public_helpers = BTreeMap::from([(
            crate::program::LutId::from_index(0),
            FlatLutPublicHelperSet::new(
                &reduction_lut,
                helpers[&crate::program::LutId::from_index(0)]
                    .iter()
                    .map(|helper| {
                        FlatLutPublicHelper::with_mask_bank(
                            helper.sigma(),
                            helper.switch().public_projection(),
                            public_mask_bank.clone(),
                        )
                        .expect("reduction public mask branch")
                    })
                    .collect(),
            )
            .expect("reduction public helper set"),
        )]);
        let public_rounding_helpers = rounding_helpers
            .iter()
            .map(|helper| {
                FlatLutPublicHelper::with_mask_bank(
                    helper.sigma(),
                    helper.switch().public_projection(),
                    public_mask_bank.clone(),
                )
                .expect("rounding public mask branch")
            })
            .collect::<Vec<_>>();
        let public_rounding_helpers =
            FlatLutPublicHelperSet::new(rounding_lut, public_rounding_helpers)
                .expect("public rounding helper set");
        let public_values_name = batch.public_input_name();
        let encoded_oracle_vector = PbcEncodedPublicVector::route_usize(
            &generated.public_layout,
            &oracle_public_vector,
            profile.q_l(),
        )
        .expect("route clear PBC public vector");
        let independent_public_values = PbcPublicVectorFamilyBinding::from_encoded(
            &generated.public_layout,
            &encoded_oracle_vector,
        )
        .expect("bind independent public values")
        .values_u64()
        .to_vec();
        let independent_public_values_name =
            "refresh-setup-gpu-independent-public-values".to_owned();
        let mask_count = crt_moduli.len() * layout.secret_dimension * layout.digit_count;
        let fresh_count = layout.secret_dimension * layout.digit_count;
        let total = mask_count + fresh_count;
        let outputs = program
            .compile_pbc_encoding_family_typed_with_batch_and_rounding_helpers(
                &compiler,
                Family::pack(
                    (0..total)
                        .map(|index| {
                            if index < mask_count {
                                state.vector.clone()
                            } else {
                                fresh.vector.clone()
                            }
                        })
                        .collect(),
                )
                .expect("state/fresh vector family"),
                Family::pack(
                    (0..total)
                        .map(|index| {
                            if index < mask_count {
                                state.pubkey.matrix.clone()
                            } else {
                                fresh.pubkey.matrix.clone()
                            }
                        })
                        .collect(),
                )
                .expect("state/fresh public family"),
                &batch,
                selectors,
                &helpers,
                &rounding_helpers,
            )
            .expect("batched PBC lowering");
        let selected = outputs.project(0).expect("selected PBC output");
        let selected_prf_vector = selected.encoding().vector.clone();
        let selected_prf_public = selected.encoding().pubkey.matrix.clone();
        let independent_public_family = ring.input_family(
            independent_public_values_name.clone(),
            independent_public_values.len(),
            (1, 1),
        );
        let active = PbcActiveCellIndex::build(&generated.public_layout).expect("active PBC cells");
        let active_widths = active.bucket_active_widths().collect::<Vec<_>>();
        let active_count = active.len();
        let bucket_count = generated.public_layout.parameters.bucket_count;
        assert!(
            active_widths.iter().all(|width| *width == generated.public_layout.bucket_width),
            "fixed PBC fixture must have a full active width per bucket"
        );
        let bucket_width = generated.public_layout.bucket_width;
        assert_eq!(
            independent_public_values.len(),
            active_count,
            "independent public family must contain exactly the routed active-cell count"
        );
        assert_eq!(
            active_count,
            bucket_count * bucket_width,
            "fixed PBC active-cell count must equal bucket count times bucket width"
        );
        let selector_count = public_selector
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .expect("public selector count must be concrete");
        let public_family_count = independent_public_family
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .expect("independent public family count must be concrete");
        assert_eq!(
            selector_count, public_family_count,
            "public selector and routed value families must have equal cardinality"
        );
        let helper_shapes = public_helpers
            .iter()
            .map(|(lut, helpers)| (lut.index(), helpers.as_slice().len()))
            .collect::<Vec<_>>();
        let input_type = state.pubkey.matrix.matrix_type();
        let input_shape = (
            input_type.rows.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize()),
            input_type
                .columns
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_usize()),
        );
        let public_compiler = PowerLutPublicKeyCompiler::new(compiler.bgg.public_key.clone());
        // Normal executable families cannot be captured by a nested
        // sequential body.  Carry the routed value family as an explicit
        // invariant, matching the production batched lowerer's construction;
        // the body then applies the bucket range exactly once.
        let independent_bucket_state = Sequential::range(bucket_count)
            .scan(
                BggPublicKeyWire {
                    matrix: state.pubkey.matrix.clone(),
                    reveal_plaintext: false,
                },
                (public_selector.gsw().clone(), independent_public_family.clone()),
                |bucket, state, (selector_family, public_values)| {
                    let selectors = PublicSelectorFamily::new(selector_family)
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    let start_expression = mxx_ir_core::IntExpr::Mul(
                        Box::new(bucket.expression()),
                        Box::new(mxx_ir_core::IntExpr::constant(bucket_width)),
                    )
                    .canonicalize();
                    let range = FamilyRange::bounded(
                        start_expression,
                        mxx_ir_core::IntExpr::constant(bucket_width),
                        bucket_width,
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?;
                    let mut ranges = ProgramFamilyRanges::new();
                    ranges.selector(program.selector_family, range.clone());
                    ranges.public_values(program.public_value_family, range);
                    let outputs = public_compiler
                        .compile_program_with_ranges(
                            &program.program,
                            &BTreeMap::from([(program.input, state)]),
                            &BTreeMap::new(),
                            &BTreeMap::from([(program.selector_family, selectors)]),
                            &BTreeMap::from([(program.public_value_family, public_values)]),
                            &ranges,
                            &public_helpers,
                        )
                        .map_err(|_| mxx_dsl::DslError::Schema)?;
                    outputs.get(&program.output).cloned().ok_or(mxx_dsl::DslError::Schema)
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "independent public PBC bucket lowering failed: {error:?}; bucket_count={bucket_count}, bucket_width={bucket_width}, active_count={active_count}, selector_count={selector_count}, public_family_count={public_family_count}, helper_shapes={helper_shapes:?}, input_shape={input_shape:?}"
                )
            });
        let rounded_outputs = public_compiler
            .compile_program(
                program.rounding_program(),
                &BTreeMap::from([(program.rounding_input(), independent_bucket_state)]),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::from([(
                    crate::program::LutId::from_index(0),
                    public_rounding_helpers.clone(),
                )]),
            )
            .expect("independent public PBC rounding");
        let selected_public_independent = rounded_outputs
            .get(&program.terminal_output_wire())
            .cloned()
            .expect("independent public PBC terminal output")
            .matrix;
        let mut masks = (0..crt_moduli.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for index in 0..mask_count {
            let RefreshPrfLabel::Mask { slot, component, coefficient, digit, .. } =
                labels.label(index).expect("mask label")
            else {
                panic!("mask label index entered fresh-error group")
            };
            masks[slot].push(
                RefreshMaskPrfOutput::from_pbc_evaluation(
                    outputs.project(index).expect("mask output"),
                    refresh_id,
                    slot,
                    component,
                    coefficient,
                    digit,
                )
                .expect("mask output descriptor"),
            );
        }
        let fresh_error = (0..fresh_count)
            .map(|offset| {
                let index = mask_count + offset;
                let RefreshPrfLabel::FreshError { component, coefficient, digit, .. } =
                    labels.label(index).expect("fresh label")
                else {
                    panic!("fresh label index entered mask group")
                };
                RefreshFreshErrorPrfOutput::from_pbc_evaluation(
                    outputs.project(index).expect("fresh output"),
                    refresh_id,
                    component,
                    coefficient,
                    digit,
                )
                .expect("fresh output descriptor")
            })
            .collect();
        let prf = RefreshPrfInputs::from_pbc_outputs(&parameters, &program, masks, fresh_error)
            .expect("refresh PRF inputs");
        let request = RefreshPreprocessingRequest {
            parameters: parameters.clone(),
            prf,
            compiler: PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
                ring: ring.clone(),
                base: layout.gadget_base.clone(),
                digit_count: layout.digit_count.into(),
            }),
            state,
            secret: mask_secret_input,
            hash_key,
        };
        let expected_plaintext = unit_attribute;
        let gpu = GpuDCRTPolyParams::new(ring_dimension, dcrt.to_crt().0, dcrt.base_bits());
        Fixture {
            dcrt,
            gpu,
            parameters,
            compiler,
            request,
            selector_graph,
            selector_production,
            selector_artifacts,
            selector_bits,
            mask_secret_reference,
            payload_secret_reference,
            mask_secret,
            payload_secret,
            selected_prf_vector,
            selected_prf_public,
            selected_prf_public_independent: selected_public_independent,
            expected_scalar,
            independent_public_values_name,
            independent_public_values,
            public_values_name,
            public_values,
            expected_plaintext,
        }
    }
}

fn runtime_matrix(fixture: &Fixture, value: &DCRTPolyMatrix) -> RuntimeValue<GpuDcrtBackend> {
    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&fixture.gpu, value))
}

fn runtime_constant(fixture: &Fixture, value: u64) -> RuntimeValue<GpuDcrtBackend> {
    let matrix = DCRTPolyMatrix::from_poly_vec(
        &fixture.dcrt,
        vec![vec![mxx_primitives::poly::dcrt::poly::DCRTPoly::from_usize_to_constant(
            &fixture.dcrt,
            value as usize,
        )]],
    );
    runtime_matrix(fixture, &matrix)
}

/// Binds a public LUT value as the ring monomial `X^a`.  Host PBC vectors stay
/// residue-valued; only the DSL public-family representation uses monomials.
fn runtime_monomial(fixture: &Fixture, value: u64) -> RuntimeValue<GpuDcrtBackend> {
    let exponent = value as usize;
    assert!(
        exponent < fixture.dcrt.ring_dimension() as usize,
        "public monomial exponent exceeds N"
    );
    let matrix = DCRTPolyMatrix::from_poly_vec(
        &fixture.dcrt,
        vec![vec![DCRTPoly::const_rotate_poly(&fixture.dcrt, exponent)]],
    );
    runtime_matrix(fixture, &matrix)
}

fn request_clone(request: &RefreshPreprocessingRequest) -> RefreshPreprocessingRequest {
    RefreshPreprocessingRequest {
        parameters: request.parameters.clone(),
        prf: request.prf.clone(),
        compiler: PowerLutEncodingCompiler::from_public_key(
            request.compiler.bgg.public_key.clone(),
        ),
        state: request.state.clone(),
        secret: request.secret.clone(),
        hash_key: request.hash_key.clone(),
    }
}

#[test]
#[serial(dcrt_runtime)]
fn test_gpu_refresh_setup_producer_export_import_refresh_verify() {
    install_tracing();
    info!("refresh setup GPU fixture construction started");
    let fixture = Fixture::new();
    let config = execution_config();
    info!(
        ring_dimension = fixture.gpu.ring_dimension(),
        crt_count = fixture.parameters.refresh.crt_plaintext_moduli.len(),
        max_parallel_instances = config.max_parallel_instances.get(),
        "refresh setup GPU fixture constructed"
    );
    let verification_mask_secret = fixture.request.secret.clone();
    let verification_payload_secret = fixture.payload_secret.clone();
    let mut store = MemoryArtifactStore::default();

    let selector =
        fixture.selector_graph.validate(&ParamEnv::default()).expect("selector validation");
    info!("refresh setup GPU selector execution started");
    let selector_result = execute_in_session_with_config(
        &selector,
        &mut gpu_backend([fixture.gpu.clone()]),
        BTreeMap::from([
            (
                fixture.selector_bits.input_name().to_owned(),
                RuntimeValue::IndexedFamily(
                    fixture
                        .selector_bits
                        .runtime_bits()
                        .iter()
                        .map(|bit| runtime_constant(&fixture, u64::from(*bit)))
                        .collect(),
                ),
            ),
            ("refresh-setup-gpu-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
        ]),
        &mut store,
        fixture.selector_production.execution_nonce,
        config,
    )
    .expect("selector GPU execution");
    info!("refresh setup GPU selector execution completed");
    assert_eq!(selector_result.production_id.as_ref(), Some(&fixture.selector_production));
    let RuntimeValue::Matrix(selector_payload_secret) = selector_result
        .outputs
        .get("refresh-setup-gpu-payload-secret")
        .expect("selector payload secret output")
    else {
        panic!("selector payload secret must be a matrix")
    };
    assert_eq!(selector_payload_secret.to_cpu_matrix(), fixture.payload_secret_reference);
    assert_ne!(
        selector_payload_secret.to_cpu_matrix(),
        DCRTPolyMatrix::zero(&fixture.dcrt, 1, fixture.parameters.layout.secret_dimension),
        "exported selector payload secret regressed to zero"
    );
    let RuntimeValue::Matrix(selector_mask_secret) = selector_result
        .outputs
        .get("refresh-setup-gpu-mask-secret")
        .expect("selector mask secret output")
    else {
        panic!("selector mask secret must be a matrix")
    };
    assert_eq!(selector_mask_secret.to_cpu_matrix(), fixture.mask_secret_reference);
    let mut selector_manifest =
        store.manifest(&fixture.selector_production).cloned().expect("selector manifest export");
    fixture
        .selector_artifacts
        .finalize_export_manifest(&mut selector_manifest)
        .expect("selector family metadata");
    mxx_ir_core::artifact::validate_manifest(&selector_manifest).expect("selector manifest valid");
    let producer = RefreshPreprocessingProducer::build(request_clone(&fixture.request))
        .expect("build producer");

    let producer_graph = producer
        .built()
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(fixture.selector_production.clone(), selector_manifest.clone())]),
        )
        .expect("producer manifest validation");
    info!("refresh setup GPU preprocessing producer execution started");
    let producer_result = execute_with_config(
        &producer_graph,
        &mut gpu_backend([fixture.gpu.clone()]),
        BTreeMap::from([
            ("refresh-setup-gpu-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
            (
                fixture.public_values_name.clone(),
                RuntimeValue::IndexedFamily(
                    fixture
                        .public_values
                        .iter()
                        .map(|value| runtime_monomial(&fixture, *value))
                        .collect(),
                ),
            ),
        ]),
        &mut store,
        SamplingMode::Fresh,
        config,
    )
    .expect("preprocessing producer GPU execution");
    info!("refresh setup GPU preprocessing producer execution completed");

    // Confirm that the materialized PRF encoding is not the degenerate zero
    // path.  Check both private vector and public matrix projections without
    // logging their values; the later residual oracle independently checks
    // the nonzero X^0 plaintext relation.
    let public_key_columns = fixture.parameters.layout.public_key_columns();
    let first_mask = &producer.declaration().names.masks[0];
    let RuntimeValue::Matrix(mask_vector) =
        producer_result.outputs.get(&first_mask.vector).expect("producer mask vector output")
    else {
        panic!("producer mask vector output must be a matrix")
    };
    let RuntimeValue::Matrix(mask_public) = producer_result
        .outputs
        .get(&first_mask.public_matrix)
        .expect("producer mask public output")
    else {
        panic!("producer mask public output must be a matrix")
    };
    let mask_vector_nonzero =
        mask_vector.to_cpu_matrix() != DCRTPolyMatrix::zero(&fixture.dcrt, 1, public_key_columns);
    let mask_public_nonzero = mask_public.to_cpu_matrix() !=
        DCRTPolyMatrix::zero(
            &fixture.dcrt,
            fixture.parameters.component_count,
            public_key_columns,
        );
    assert!(mask_vector_nonzero || mask_public_nonzero, "materialized PRF mask encoding is zero");

    let production_id = producer_result.production_id.clone().expect("producer production id");
    let mut manifest = store.load_manifest(&production_id).expect("producer manifest export");
    producer.finalize_export_manifest(&mut manifest).expect("finalize producer manifest");
    mxx_ir_core::artifact::validate_manifest(&manifest).expect("producer manifest valid");

    let imported = ImportedRefreshSetup::import(
        production_id,
        fixture.parameters.clone(),
        producer.declaration().clone(),
        producer.attestation(),
        &manifest,
    )
    .expect("import preprocessing setup");
    let refresh_setup = fixture
        .parameters
        .refresh
        .bind_imported_setup(&fixture.compiler, &imported)
        .expect("bind imported setup");
    info!("refresh setup CPU refresh binding started");
    let refreshed = fixture
        .parameters
        .refresh
        .refresh(&fixture.compiler, &refresh_setup)
        .expect("refresh imported setup");
    info!("refresh setup CPU refresh binding completed");
    let verification = build_refresh_verification(
        refreshed.encoding(),
        &verification_mask_secret,
        &verification_payload_secret,
        &fixture.expected_plaintext,
        fixture.parameters.layout.gadget_base.clone(),
        fixture.parameters.digit_count,
        fixture.parameters.base_p,
        1,
    )
    .expect("build verification");
    assert!(
        fixture
            .parameters
            .refresh
            .full_modulus
            .evaluate(&ParamEnv::default())
            .expect("full modulus evaluation") >
            BigInt::from(1u8)
    );
    let ring = fixture.parameters.layout.ring();
    let gadget = ring.gadget(
        fixture.parameters.layout.secret_dimension,
        fixture.parameters.layout.gadget_base.clone(),
        fixture.parameters.digit_count,
    );
    let selected_expected_vector = fixture.mask_secret.clone() *
        fixture.selected_prf_public.clone() -
        fixture.expected_scalar.clone() * (fixture.payload_secret.clone() * gadget.clone());
    let mut relation_context = DslContext::new("refresh-setup-gpu-bk-oracle");
    relation_context = relation_context
        .public_output("imported_state_public", imported.state.pubkey.matrix.clone())
        .expect("state oracle output")
        .public_output("imported_fresh_public", imported.fresh.pubkey.matrix.clone())
        .expect("fresh oracle output")
        .public_output("imported_a_prime", imported.a_prime.clone())
        .expect("a-prime oracle output")
        .public_output("imported_public_b", imported.public_b.clone())
        .expect("B oracle output")
        .private_output("selected_prf_vector", fixture.selected_prf_vector.clone())
        .expect("selected PRF vector output")
        .public_output("selected_prf_public", fixture.selected_prf_public.clone())
        .expect("selected PRF public output")
        .public_output(
            "selected_prf_public_independent",
            fixture.selected_prf_public_independent.clone(),
        )
        .expect("independent public PRF output")
        .private_output("selected_prf_expected_vector", selected_expected_vector)
        .expect("selected PRF relation output");
    for slot in 0..fixture.parameters.refresh.crt_plaintext_moduli.len() {
        let alpha = ring.polynomial([fixture
            .parameters
            .refresh
            .scale_expression(slot)
            .expect("slot scale")]);
        let actual = imported.public_b.clone() * imported.preimages[slot].clone();
        let scaled_state = fixture.compiler.bgg.large_scalar_mul(&imported.state, &alpha);
        let scaled_fresh = fixture.compiler.bgg.large_scalar_mul(&imported.fresh, &alpha);
        let target = scaled_state.pubkey.matrix +
            imported.masks[slot].pubkey.matrix.clone() +
            scaled_fresh.pubkey.matrix -
            alpha.clone() * imported.a_prime.clone();
        relation_context = relation_context
            .public_output(
                format!("imported_mask_public_{slot}"),
                imported.masks[slot].pubkey.matrix.clone(),
            )
            .expect("mask oracle output")
            .public_output(format!("imported_preimage_{slot}"), imported.preimages[slot].clone())
            .expect("preimage oracle output")
            .public_output(format!("actual_bk_{slot}"), actual)
            .expect("B*K oracle output")
            .public_output(format!("target_{slot}"), target)
            .expect("target oracle output");
    }
    let graph = verification
        .add_outputs(relation_context, "residual", "decoded")
        .expect("verification outputs")
        .build()
        .expect("verification graph")
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([
                (fixture.selector_production.clone(), selector_manifest),
                (manifest.production_id.clone(), manifest),
            ]),
        )
        .expect("verification manifest validation");
    info!("refresh setup GPU verification execution started");
    let result = execute_with_config(
        &graph,
        &mut gpu_backend([fixture.gpu.clone()]),
        BTreeMap::from([
            ("refresh-setup-gpu-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
            (
                fixture.public_values_name.clone(),
                RuntimeValue::IndexedFamily(
                    fixture
                        .public_values
                        .iter()
                        .map(|value| runtime_monomial(&fixture, *value))
                        .collect(),
                ),
            ),
            (
                fixture.independent_public_values_name.clone(),
                RuntimeValue::IndexedFamily(
                    fixture
                        .independent_public_values
                        .iter()
                        .map(|value| runtime_monomial(&fixture, *value))
                        .collect(),
                ),
            ),
        ]),
        &mut store,
        SamplingMode::Fresh,
        execution_config(),
    )
    .expect("verification GPU execution");
    info!("refresh setup GPU verification execution completed");
    for (imported_name, producer_name) in [
        ("imported_state_public", &producer.declaration().names.state_public_matrix),
        ("imported_fresh_public", &producer.declaration().names.fresh_public_matrix),
        ("imported_a_prime", &producer.declaration().names.a_prime),
        ("imported_public_b", &producer.declaration().names.public_matrix_b),
    ] {
        let RuntimeValue::Matrix(imported_value) = &result.outputs[imported_name] else {
            panic!("imported oracle output must be a matrix")
        };
        let RuntimeValue::Matrix(producer_value) =
            producer_result.outputs.get(producer_name).expect("producer oracle output")
        else {
            panic!("producer oracle output must be a matrix")
        };
        assert_eq!(imported_value, producer_value, "artifact changed for {imported_name}");
    }
    for slot in 0..fixture.parameters.refresh.crt_plaintext_moduli.len() {
        let imported_name = format!("imported_mask_public_{slot}");
        let RuntimeValue::Matrix(imported_value) = &result.outputs[&imported_name] else {
            panic!("imported mask oracle output must be a matrix")
        };
        let RuntimeValue::Matrix(producer_value) = producer_result
            .outputs
            .get(&producer.declaration().names.masks[slot].public_matrix)
            .expect("producer mask oracle output")
        else {
            panic!("producer mask oracle output must be a matrix")
        };
        assert_eq!(imported_value, producer_value, "mask artifact changed at slot {slot}");
        let preimage_name = format!("imported_preimage_{slot}");
        let RuntimeValue::Matrix(imported_value) = &result.outputs[&preimage_name] else {
            panic!("imported preimage oracle output must be a matrix")
        };
        let RuntimeValue::Matrix(producer_value) = producer_result
            .outputs
            .get(&producer.declaration().names.preimages[slot])
            .expect("producer preimage oracle output")
        else {
            panic!("producer preimage oracle output must be a matrix")
        };
        assert_eq!(imported_value, producer_value, "preimage artifact changed at slot {slot}");
    }
    let RuntimeValue::Matrix(residual) = &result.outputs["residual"] else {
        panic!("verification residual must be a matrix")
    };
    assert_eq!(residual.nrow, 1);
    assert_eq!(residual.ncol, public_key_columns);
    let residual_cpu = residual.to_cpu_matrix();
    // This fixture uses the unit X^0 plaintext and zero Gaussian noise, so
    // `c - sA + mu*(tG)` must be exactly zero.
    assert_eq!(
        residual_cpu,
        DCRTPolyMatrix::zero(&fixture.dcrt, 1, public_key_columns),
        "noiseless fixed-C refresh residual is not zero"
    );
    for slot in 0..fixture.parameters.refresh.crt_plaintext_moduli.len() {
        let RuntimeValue::Matrix(actual) = &result.outputs[&format!("actual_bk_{slot}")] else {
            panic!("B*K oracle output must be a matrix")
        };
        let RuntimeValue::Matrix(target) = &result.outputs[&format!("target_{slot}")] else {
            panic!("target oracle output must be a matrix")
        };
        let expected_shape = (fixture.parameters.component_count, public_key_columns);
        assert_eq!((actual.nrow, actual.ncol), expected_shape);
        assert_eq!((target.nrow, target.ncol), expected_shape);
        assert!(
            actual == target,
            "B*K differs from independently reconstructed target at CRT slot {slot}"
        );
    }
    let RuntimeValue::Matrix(selected_vector) = &result.outputs["selected_prf_vector"] else {
        panic!("selected PRF vector output must be a matrix")
    };
    let RuntimeValue::Matrix(selected_public) = &result.outputs["selected_prf_public"] else {
        panic!("selected PRF public output must be a matrix")
    };
    let RuntimeValue::Matrix(independent_public) =
        &result.outputs["selected_prf_public_independent"]
    else {
        panic!("independent public PRF output must be a matrix")
    };
    let RuntimeValue::Matrix(expected_vector) = &result.outputs["selected_prf_expected_vector"]
    else {
        panic!("selected PRF expected vector output must be a matrix")
    };
    // The producer's slot mask is an aggregate over every component,
    // coefficient, and digit label.  It is intentionally not compared with
    // this one selected sparse-LWR output; the relation below is the exact
    // selected output check.
    assert!(selected_public == independent_public, "private/public PRF matrix mismatch");
    assert!(selected_vector == expected_vector, "split-secret PRF BGG relation mismatch");
    for column in 0..verification.decoded().len() {
        assert!(
            matches!(result.outputs[&format!("decoded_{column}")], RuntimeValue::Bool(false)),
            "fixed-C decoded output {column} is not Bool(false)"
        );
    }
}
