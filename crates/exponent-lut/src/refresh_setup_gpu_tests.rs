//! GPU acceptance coverage for the setup-fixed Section-7 refresh path.
//!
//! This test owns a small local fixture so the runtime acceptance path crosses
//! the same selector export, preprocessing export/import, refresh, and
//! verification boundaries used by production.  The fixture intentionally
//! contains only the current fixed-C selector family and flat LUT helpers.

use crate::{
    ExponentLutEncodingCompiler,
    encoding::{EncodingSelectorFamily, ExponentLutEncodingSampler, FlatLutHelperSet},
    pbc::{
        PbcParameters, PbcRootSeed, PbcSelectorArtifactNames, PbcSelectorArtifacts,
        PbcTrustedSelectorBits, build_structural_selector_families, generate_key_layout,
    },
    prf::{
        RefreshPrfBatchInputs, SparseLwrPrfHelperBundle, SparseLwrPrfProfile, SparseLwrPrfProgram,
        SparseLwrReductionHelpers, SparseLwrTerminalHelpers,
    },
    program::LutTable,
    refresh_setup::{
        RefreshPreprocessingProducer, RefreshPreprocessingRequest, RefreshPrfInputs,
        RefreshSetupParameters,
    },
};
use mxx_bgg::{BggPublicKeyCompiler, BggSamplerLayout};
use mxx_dsl::{DslContext, Int, Mat, Parallel};
use mxx_ir_core::{
    ParamEnv,
    artifact::ArtifactConfidentiality,
    encoding::spec_hash,
    node::{ConcatAxis, NodeKind},
};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
#[cfg(feature = "gpu")]
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly,
        dcrt::{gpu::GpuDCRTPolyParams, poly::DCRTPoly},
    },
};
use num_bigint::BigInt;
use std::collections::BTreeMap;

const RING_DIMENSION_ENV: &str = "MXX_EXPONENT_LUT_REFRESH_SETUP_GPU_RING_DIMENSION";
const CRT_BITS_ENV: &str = "MXX_EXPONENT_LUT_REFRESH_SETUP_GPU_CRT_BITS";
const BASE_BITS_ENV: &str = "MXX_EXPONENT_LUT_REFRESH_SETUP_GPU_BASE_BITS";
const DEFAULT_RING_DIMENSION: u32 = 4;
const DEFAULT_CRT_BITS: u32 = 17;
const DEFAULT_BASE_BITS: u32 = 16;

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

struct Fixture {
    #[cfg(feature = "gpu")]
    dcrt: DCRTPolyParams,
    #[cfg(feature = "gpu")]
    gpu: GpuDCRTPolyParams,
    parameters: RefreshSetupParameters,
    #[cfg(feature = "gpu")]
    compiler: ExponentLutEncodingCompiler,
    request: RefreshPreprocessingRequest,
    #[cfg(feature = "gpu")]
    selector_graph: mxx_dsl::BuiltGraph,
    #[cfg(feature = "gpu")]
    selector_production: mxx_ir_core::artifact::ProductionId,
    #[cfg(feature = "gpu")]
    selector_artifacts: PbcSelectorArtifacts,
    #[cfg(feature = "gpu")]
    selector_bits: PbcTrustedSelectorBits,
    #[cfg(feature = "gpu")]
    mask_secret_reference: DCRTPolyMatrix,
    #[cfg(feature = "gpu")]
    payload_secret_reference: DCRTPolyMatrix,
    #[cfg(feature = "gpu")]
    payload_secret: Mat,
    #[cfg(feature = "gpu")]
    expected_plaintext: Mat,
}

impl Fixture {
    fn new(mask_base_p_digit_count: usize, fresh_error_base_p_digit_count: usize) -> Self {
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
        let sampler = ExponentLutEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: Some(1.into()),
            gaussian_max_coefficient_bound: Some(6.into()),
        };
        // Use distinct deterministic augmented secrets: each has a ternary
        // prefix followed by the mandatory final coordinate one.
        let mask_secret =
            Mat::concat(ConcatAxis::Columns, vec![ring.polynomial([1.into()]), ring.identity(1)]);
        let payload_secret = Mat::concat(
            ConcatAxis::Columns,
            vec![ring.polynomial([BigInt::from(-1).into()]), ring.identity(1)],
        );
        #[cfg(feature = "gpu")]
        let mask_secret_reference = DCRTPolyMatrix::from_poly_vec(
            &dcrt,
            vec![vec![
                DCRTPoly::from_usize_to_constant(&dcrt, 1),
                DCRTPoly::from_usize_to_constant(&dcrt, 1),
            ]],
        );
        #[cfg(feature = "gpu")]
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
            .expect("selector bits");
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
        let selectors = EncodingSelectorFamily::new(gsw).expect("fixed-C selector binding");
        let compiler = ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
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
        let profile = SparseLwrPrfProfile::new(2, 2, 4, ring_dimension as usize)
            .expect("small sparse-LWR profile");
        let program = SparseLwrPrfProgram::new(
            profile.clone(),
            generated.public_layout.bucket_width,
            generated.public_layout.parameters.bucket_count,
        )
        .expect("sparse-LWR program");
        let reduction_width = program.plan.lut_width();
        let reduction_table = (0..reduction_width).map(|value| value % 2).collect::<Vec<_>>();
        let reduction_lut =
            LutTable::unary(reduction_width, reduction_width, reduction_table.clone())
                .expect("reduction LUT");
        let mask_bank = sampler
            .sample_flat_mask_bank(
                mask_secret.clone(),
                hash_key.clone(),
                reduction_width,
                b"refresh-setup-gpu-mask-bank".as_slice(),
            )
            .expect("sample shared mask bank");
        let mut helpers = BTreeMap::from([(
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

        let mask_statistical_security_bits = 32;
        let labels = crate::refresh::RefreshPrfLabelIndex::new(
            refresh_id,
            crt_moduli.len(),
            layout.public_key_columns(),
            ring_dimension as usize,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
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
        let parameters = RefreshSetupParameters::new(
            refresh_id,
            profile.p(),
            layout.secret_dimension,
            ring_dimension as usize,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
            mask_statistical_security_bits,
            4,
            layout.clone(),
            refresh,
            1.into(),
            1.into(),
            "refresh-setup-gpu",
        );
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
        let helper_bundle = SparseLwrPrfHelperBundle {
            reduction: SparseLwrReductionHelpers::new(
                helpers
                    .remove(&crate::program::LutId::from_index(0))
                    .expect("reduction helper set"),
            ),
            terminal: SparseLwrTerminalHelpers::new(rounding_helpers),
        };
        let batch = RefreshPrfBatchInputs::new(
            &generated.public_layout,
            profile.clone(),
            &labels,
            &program,
        )
        .expect("refresh PRF batch");
        let mask_count = crt_moduli.len() *
            layout.public_key_columns() *
            ring_dimension as usize *
            mask_base_p_digit_count;
        let fresh_count =
            layout.public_key_columns() * ring_dimension as usize * fresh_error_base_p_digit_count;
        let total = mask_count + fresh_count;
        let selector = Parallel::range(total)
            .map_values(|index| index.as_int().less_equal(Int::constant(mask_count - 1)).to_int())
            .expect("state/fresh selector family");
        let state_vectors = Parallel::range(total)
            .map_values({
                let state_vector = state.vector.clone();
                move |_| state_vector.clone()
            })
            .expect("state vector family");
        let fresh_vectors = Parallel::range(total)
            .map_values({
                let fresh_vector = fresh.vector.clone();
                move |_| fresh_vector.clone()
            })
            .expect("fresh vector family");
        let input_vectors = selector
            .clone()
            .parallel_select_mats(vec![fresh_vectors, state_vectors])
            .expect("state/fresh vector selection");
        let state_public_keys = Parallel::range(total)
            .map_values({
                let state_public_key = state.pubkey.matrix.clone();
                move |_| state_public_key.clone()
            })
            .expect("state public family");
        let fresh_public_keys = Parallel::range(total)
            .map_values({
                let fresh_public_key = fresh.pubkey.matrix.clone();
                move |_| fresh_public_key.clone()
            })
            .expect("fresh public family");
        let input_public_keys = selector
            .parallel_select_mats(vec![fresh_public_keys, state_public_keys])
            .expect("state/fresh public selection");
        let outputs = program
            .compile_pbc_encoding_family_typed_with_batch_and_helpers(
                &compiler,
                input_vectors,
                input_public_keys,
                &batch,
                selectors,
                &helper_bundle,
            )
            .expect("batched PBC lowering");
        let prf =
            RefreshPrfInputs::from_pbc_family_outputs(&parameters, &program, &batch, &outputs)
                .expect("refresh PRF inputs");
        let request = RefreshPreprocessingRequest {
            parameters: parameters.clone(),
            prf,
            compiler: ExponentLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
                ring: ring.clone(),
                base: layout.gadget_base.clone(),
                digit_count: layout.digit_count.into(),
            }),
            state,
            secret: mask_secret_input,
            hash_key,
        };
        #[cfg(feature = "gpu")]
        let expected_plaintext = unit_attribute;
        #[cfg(feature = "gpu")]
        let gpu = GpuDCRTPolyParams::new(ring_dimension, dcrt.to_crt().0, dcrt.base_bits());
        Fixture {
            #[cfg(feature = "gpu")]
            dcrt,
            #[cfg(feature = "gpu")]
            gpu,
            parameters,
            #[cfg(feature = "gpu")]
            compiler,
            request,
            #[cfg(feature = "gpu")]
            selector_graph,
            #[cfg(feature = "gpu")]
            selector_production,
            #[cfg(feature = "gpu")]
            selector_artifacts,
            #[cfg(feature = "gpu")]
            selector_bits,
            #[cfg(feature = "gpu")]
            mask_secret_reference,
            #[cfg(feature = "gpu")]
            payload_secret_reference,
            #[cfg(feature = "gpu")]
            payload_secret,
            #[cfg(feature = "gpu")]
            expected_plaintext,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct StructuredRefreshGraphSummary {
    route_body_count: usize,
    family_pack_count: usize,
    family_static_read_count: usize,
    gather_grid_count: usize,
    add_grid_count: usize,
}

/// Summarizes graph-level evidence for the structured family boundary. The
/// two `PowerOfBase` constants are the single mask/fresh routing bodies.
fn structured_refresh_graph_summary(
    graph: &mxx_dsl::BuiltGraph,
    slot_count: usize,
) -> StructuredRefreshGraphSummary {
    let mut route_scopes = Vec::new();
    let mut family_pack_count = 0;
    let family_static_reads = graph
        .graph
        .scopes()
        .values()
        .flat_map(|scope| scope.nodes())
        .filter(|node| matches!(node.kind(), NodeKind::FamilyGetStatic { .. }))
        .count();
    let mut gather_grid_count = 0;
    let mut add_grid_count = 0;
    for (scope_id, scope) in graph.graph.scopes() {
        family_pack_count += scope
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::FamilyPack { .. }))
            .count();
        let power_routes = scope
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.kind(),
                    NodeKind::ConstantMatrix {
                        value: mxx_ir_core::node::ConstantMatrix::PowerOfBase { .. },
                        ..
                    }
                )
            })
            .count();
        if power_routes != 0 {
            route_scopes.push((scope_id, power_routes));
            assert_eq!(power_routes, 1, "each symbolic route body has one digit scale");
            assert!(
                scope.nodes().iter().all(|node| {
                    !matches!(
                        node.kind(),
                        NodeKind::FamilyPack { .. } | NodeKind::FamilyGetStatic { .. }
                    )
                }),
                "route bodies must not contain label-axis family packing or static reads"
            );
        }
        for node in scope.nodes() {
            let NodeKind::ParallelGrid(_) = node.kind() else { continue };
            let child_id = mxx_ir_core::FrozenGraphScopeId::ParallelBody {
                parent: Box::new(scope_id.clone()),
                owner: scope.node_id(node).unwrap(),
            };
            let Some(child) = graph.graph.scope(&child_id) else { continue };
            if child
                .nodes()
                .iter()
                .any(|child_node| matches!(child_node.kind(), NodeKind::FamilyGetDynamic { .. }))
            {
                gather_grid_count += 1;
            }
            if child.nodes().iter().any(|child_node| {
                matches!(
                    child_node.kind(),
                    NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add)
                )
            }) {
                add_grid_count += 1;
            }
        }
    }
    assert!(slot_count >= 2, "graph-shape fixture must exercise CRT slot reduction");
    StructuredRefreshGraphSummary {
        route_body_count: route_scopes.len(),
        family_pack_count,
        family_static_read_count: family_static_reads,
        gather_grid_count,
        add_grid_count,
    }
}

fn balanced_sum_level_count(count: usize) -> usize {
    let mut levels = 0;
    let mut width = 1;
    while width < count {
        width *= 2;
        levels += 1;
    }
    levels
}

fn expected_refresh_reduction_grid_counts(
    component_count: usize,
    coefficient_count: usize,
    mask_digits: usize,
    fresh_digits: usize,
) -> (usize, usize, usize) {
    let mask_group = component_count * coefficient_count * mask_digits;
    let fresh_group = component_count * coefficient_count * fresh_digits;
    let levels = balanced_sum_level_count(mask_group) + balanced_sum_level_count(fresh_group);
    // Each balanced sum is built independently for vectors and public keys;
    // each level has two gathers and one pairwise-add grid.
    let gather_grids = 4 * levels;
    let add_grids = 2 * levels;
    // Mask and direct fresh slot segmentation each add one outer grid for
    // each of the vector/public families; these are fixed across profiles.
    let segmented_grids = 4;
    (gather_grids, add_grids, segmented_grids + gather_grids + add_grids)
}

#[test]
fn test_structured_refresh_graph_shape_cpu_fixture() {
    // Exercise the real whole-family constructor and producer without
    // entering the GPU runtime. The second profile changes both mask/fresh
    // digit cardinalities and therefore the label-family cardinality.
    let mut summaries = Vec::new();
    for (mask_digits, fresh_digits) in [(1, 1), (2, 3)] {
        let fixture = Fixture::new(mask_digits, fresh_digits);
        let slot_count = fixture.parameters.refresh.crt_plaintext_moduli.len();
        let component_count = fixture.parameters.prf_component_count();
        let coefficient_count = fixture.parameters.coefficient_count;
        let producer = RefreshPreprocessingProducer::build(fixture.request)
            .unwrap_or_else(|error| panic!("build structured refresh producer: {error:?}"));
        summaries.push((
            structured_refresh_graph_summary(producer.built(), slot_count),
            component_count,
            coefficient_count,
            mask_digits,
            fresh_digits,
        ));
    }
    assert_eq!(summaries[0].0.route_body_count, 2);
    assert_eq!(summaries[1].0.route_body_count, 2);
    assert_eq!(summaries[0].0.family_pack_count, summaries[1].0.family_pack_count);
    assert_eq!(summaries[0].0.family_static_read_count, summaries[1].0.family_static_read_count);
    assert_eq!(summaries[0].1, summaries[1].1);
    assert_eq!(summaries[0].2, summaries[1].2);
    let first_expected = expected_refresh_reduction_grid_counts(
        summaries[0].1,
        summaries[0].2,
        summaries[0].3,
        summaries[0].4,
    );
    let second_expected = expected_refresh_reduction_grid_counts(
        summaries[1].1,
        summaries[1].2,
        summaries[1].3,
        summaries[1].4,
    );
    let observed_gather_delta = summaries[1].0.gather_grid_count - summaries[0].0.gather_grid_count;
    let observed_add_delta = summaries[1].0.add_grid_count - summaries[0].0.add_grid_count;
    assert_eq!(
        observed_gather_delta,
        second_expected.0 - first_expected.0,
        "vector/public reduction gathers follow ceil(log2(group))"
    );
    assert_eq!(
        observed_add_delta,
        second_expected.1 - first_expected.1,
        "vector/public reduction additions follow ceil(log2(group))"
    );
    assert_eq!(
        summaries[0].0.gather_grid_count - first_expected.0,
        summaries[1].0.gather_grid_count - second_expected.0,
        "non-reduction gather scopes remain fixed"
    );
    assert_eq!(
        summaries[0].0.add_grid_count - first_expected.1,
        summaries[1].0.add_grid_count - second_expected.1,
        "non-reduction addition scopes remain fixed"
    );
    assert_eq!(
        second_expected.2 - first_expected.2,
        observed_gather_delta + observed_add_delta,
        "only logarithmic reduction depth changes"
    );
}

/// GPU builds validate the same compact graph contract without materializing
/// a label-by-label runtime family. Runtime family payloads are supplied by
/// the external artifact provider, so this fixture only checks structural
/// cardinality and provenance metadata.
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_refresh_setup_graph_contract_is_compact() {
    let fixture = Fixture::new(1, 1);
    let slot_count = fixture.parameters.refresh.crt_plaintext_moduli.len();
    let producer = RefreshPreprocessingProducer::build(fixture.request)
        .expect("build compact GPU refresh producer");
    let summary = structured_refresh_graph_summary(producer.built(), slot_count);
    assert_eq!(summary.route_body_count, 2);
    assert!(summary.gather_grid_count > 0);
}
