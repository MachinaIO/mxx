//! Exponent-LUT adapters for mechanical export of the frozen implementation.

pub mod certify;
pub mod check;
mod native_program_proof;
mod numeric;
mod root_cache_alignment;
mod root_decoder_setup;
mod root_inputs;
mod root_online_readout;
mod root_private;
mod root_program;
mod root_program_context;
mod root_program_inputs;
mod root_program_keys;
mod root_program_setup;
mod root_readout;
mod root_readout_alignment;
mod root_readout_function;
mod root_readout_proof;
mod root_readout_public;
mod root_readout_stage;
mod root_reduction;
mod root_refresh_correctness;
mod root_refresh_crt;
mod root_refresh_decode;
mod root_refresh_digits;
mod root_refresh_finish;
mod root_refresh_high_links;
mod root_refresh_high_source;
mod root_refresh_normalize;
mod root_refresh_route_public;
mod root_refresh_route_sums;
mod root_refresh_route_targets;
mod root_refresh_source;
mod root_refresh_source_proof;
mod root_refresh_state;
mod root_refresh_state_source;
mod root_seeded_keys;
mod scopes;
mod trace;

use std::{collections::BTreeMap, fs, path::Path};

use mxx_ir_core::{
    ParamEnv,
    lean::{ExportOptions, LeanArtifact, export},
    validate_with_manifests,
};
use mxx_runtime::lean::LeanBackendArtifact;

use crate::noise_refresh::setup::RefreshParameterSimulationBundle;

pub struct ExportedRefreshGraphs {
    /// Authoritative bytes of every emitted Lean module, keyed by module name.
    pub sources: BTreeMap<String, String>,
    pub preprocessing: LeanArtifact,
    pub verification: LeanArtifact,
    pub ideal: LeanArtifact,
}

/// Extract the actual preprocessing and verification executions for this bundle.
/// The selector graph aliases preprocessing and must not introduce a second
/// independent sampler execution. The linked claim concludes that the refreshed
/// residual has an integer lift bounded by the fresh-error support. Exporting
/// this proposition does not prove it or accept a parameter candidate.
pub fn export_refresh_graphs(
    bundle: &RefreshParameterSimulationBundle,
    bindings: &ParamEnv,
    backend: &LeanBackendArtifact,
    directory: &Path,
) -> Result<ExportedRefreshGraphs, Box<dyn std::error::Error>> {
    if directory.exists() && fs::read_dir(directory)?.next().is_some() {
        return Err(
            "refresh Lean export directory must be empty; checked sources are immutable".into()
        );
    }
    let preprocessing = bundle
        .preprocessing_graph
        .as_ref()
        .ok_or("proof export requires the simulation preprocessing graph")?;
    let verification = bundle
        .verification_graph
        .as_ref()
        .ok_or("proof export requires the simulation verification graph")?;
    let preprocessing =
        validate_with_manifests(&preprocessing.graph, bindings, bundle.preprocessing_manifests())?;
    let verification =
        validate_with_manifests(&verification.graph, bindings, bundle.verification_manifests())?;
    let options = |name: &str| ExportOptions {
        namespace: name.into(),
        module_name: name.into(),
        backend_layouts: backend.exporter_bindings(),
        ..ExportOptions::default()
    };
    // Finish both extractions before writing any file, so unsupported graph
    // operations cannot leave a directory appearing to contain a complete export.
    let preprocessing_artifact = export(&preprocessing, &options("RefreshPreprocessing"))?;
    let (setup_proofs, setup_scope_count) =
        scopes::selector_setup_proofs(&preprocessing, &preprocessing_artifact, bindings);
    let (root_proofs, _) =
        scopes::root_decomposition_proofs(&preprocessing, &preprocessing_artifact);
    let (private_proofs, _) = root_private::root_private(&preprocessing, &preprocessing_artifact)?;
    let readout_public_proofs =
        root_readout_public::root_readout_public(&preprocessing, &preprocessing_artifact)?;
    let readout_function_proofs =
        root_readout_function::root_readout_function(&preprocessing, &preprocessing_artifact)?;
    let (readout_proofs, _) = root_readout::root_readout(&preprocessing, &preprocessing_artifact)?;
    if setup_scope_count == 0 {
        return Err("refresh implementation has no recognized selector-setup scope".into());
    }
    let (alignment_proofs, _) =
        root_readout_alignment::root_readout_alignment(&preprocessing, &preprocessing_artifact)?;
    let program_setup_proofs =
        root_program_setup::root_program_setup(&preprocessing, &preprocessing_artifact)?;
    let cache_alignment_proofs =
        root_cache_alignment::root_cache_alignment(&preprocessing, &preprocessing_artifact)?;
    let (program_key_proofs, program_key_stages) =
        root_program_keys::root_program_keys(&preprocessing, &preprocessing_artifact)?;
    let readout_operand_proofs =
        root_readout_stage::pre_readout_operands(&preprocessing, &preprocessing_artifact)?;
    let verification_artifact = export(&verification, &options("RefreshVerification"))?;
    let seeded_key_proofs = root_seeded_keys::root_seeded_keys(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let readout_stage_proofs = root_readout_stage::root_readout_stage(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let decoder_setup_proofs = root_decoder_setup::root_decoder_setup(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let readout_proof = root_readout_proof::root_readout_proof(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let refresh_source_proofs = root_refresh_source::root_refresh_source(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let refresh_crt_proofs =
        root_refresh_crt::root_refresh_crt(&verification, &verification_artifact)?;
    let validated_preprocessing = &preprocessing;
    let refresh_decode_proofs = root_refresh_decode::root_refresh_decode(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.simulate_noise()?,
    )?;
    let refresh_digit_proofs = root_refresh_digits::root_refresh_digits(
        &preprocessing,
        &preprocessing_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_normalize_proofs = root_refresh_normalize::root_refresh_normalize(
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_finish_proofs = root_refresh_finish::root_refresh_finish(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
        &bundle.simulate_noise()?,
    )?;
    let refresh_high_links_proofs = root_refresh_high_links::root_refresh_high_links(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
    )?;
    let refresh_high_source_proofs = root_refresh_high_source::root_refresh_high_source(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
        &bundle.simulate_noise()?,
    )?;
    let refresh_correctness =
        root_refresh_correctness::root_refresh_correctness(&bundle.simulate_noise()?);
    let refresh_state_proofs = root_refresh_state::root_refresh_state(
        &preprocessing,
        &preprocessing_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_state_source = root_refresh_state_source::root_refresh_state_source(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_route_targets = root_refresh_route_targets::root_refresh_route_targets(
        &preprocessing,
        &preprocessing_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_route_sums = root_refresh_route_sums::root_refresh_route_sums(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_source_proof = root_refresh_source_proof::root_refresh_source_proof(
        &preprocessing,
        &preprocessing_artifact,
        &verification,
        &verification_artifact,
        &bundle.noise_snapshot,
    )?;
    let refresh_route_public = root_refresh_route_public::root_refresh_route_public(
        &preprocessing,
        &preprocessing_artifact,
        &bundle.noise_snapshot,
    )?;
    let preprocessing = preprocessing_artifact;
    let online_readout_proofs =
        root_online_readout::root_online_readout(&verification, &verification_artifact)?;
    let root_input_proofs = root_inputs::root_inputs(&verification, &verification_artifact)?;
    let program_input_proofs =
        root_program_inputs::root_program_inputs(&verification, &verification_artifact)?;

    let (program_proofs, program_stages) =
        root_program::root_program(&verification, &verification_artifact)?;
    let cache_wire = verification_artifact
        .root
        .inputs
        .get(&program_stages[0].private_name)
        .ok_or("missing native cache input")?
        .wire;
    let (program_owner, _) = verification
        .source
        .root_scope()
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| {
            matches!(node.kind(), NodeKind::ParallelLoop(_)) &&
                verification
                    .source
                    .root_scope()
                    .arguments(node)
                    .is_some_and(|args| args.contains(&cache_wire))
        })
        .ok_or("missing native execution owner")?;
    let program_owner = mxx_ir_core::types::NodeId(program_owner as u64);
    let program_execution_path = ".2".repeat(
        verification_artifact.scope_proofs[&mxx_ir_core::FrozenGraphScopeId::Root].node_constraints
            [&program_owner]
            .clone()
            .last()
            .ok_or("missing native execution fact")?,
    );
    let mxx_ir_core::types::ConcreteWireType::IndexedFamily { count: program_lane_count, .. } =
        &verification.scopes[&mxx_ir_core::FrozenGraphScopeId::Root].wire_types
            [&mxx_ir_core::types::WireRef {
                node: program_owner,
                port: mxx_ir_core::types::Port(0),
            }]
    else {
        return Err("native execution is not a family".into());
    };
    let program_lane_count = *program_lane_count;
    let program_context_proofs = root_program_context::root_program_context(
        &verification,
        &verification_artifact,
        &program_stages,
    )?;
    let reduction_proofs = root_reduction::root_reduction(&verification, &verification_artifact)?;
    let program_manifest = program_stages.iter().map(|stage| format!("stage {}: {} -> {}; {} members; current {:?}; private {:?}; public {:?}; input {:?}; message {:?}; output {:?}\n",stage.index,stage.private_name,stage.public_name,stage.members,stage.current,stage.private_cache,stage.public_cache,stage.public_ciphertext,stage.public_message,stage.output)).collect::<String>();
    let (action_proofs, action_scope_count) =
        scopes::matrix_action_proofs(&verification, &verification_artifact);
    let (input_proofs, input_scope_count) =
        scopes::encoding_input_proofs(&verification, &verification_artifact, bindings);
    if input_scope_count == 0 {
        return Err("refresh implementation has no recognized public-input encoding scope".into());
    }
    if action_scope_count == 0 {
        return Err("refresh implementation has no recognized matrix-action scopes".into());
    }
    let validated_verification = &verification;
    let verification = verification_artifact;
    use mxx_ir_core::{
        lean::claim::{
            self, ClaimBackend, ClaimRoot, ClaimSemantics, Endpoint, ExternalInput, InputContract,
            Link, LinkedClaim, Port,
        },
        node::NodeKind,
        types::ConcreteWireType,
    };
    let residual = verification.root.outputs.get("residual").ok_or("missing refresh residual")?;
    let ConcreteWireType::Matrix(matrix) = &residual.wire_type else {
        return Err("refresh residual is not a matrix".into());
    };
    let zero = mxx_dsl::Ring::new(matrix.modulus.clone(), matrix.ring_dimension)
        .zero((matrix.rows, matrix.columns));
    let ideal_graph =
        mxx_dsl::DslContext::new("refresh-zero-residual").public_output("zero", zero)?.build()?;
    let ideal =
        export(&mxx_ir_core::validate(&ideal_graph.graph, bindings)?, &options("RefreshIdeal"))?;
    let producer_graph = &bundle.preprocessing_graph().graph;
    let verification_graph = &bundle.verification_graph().graph;
    let mut externals = BTreeMap::<String, ExternalInput>::new();
    let mut links = Vec::new();
    for (root, graph, artifact) in
        [(0, producer_graph, &preprocessing), (1, verification_graph, &verification)]
    {
        for (name, value) in &artifact.root.inputs {
            let node =
                graph.root_scope().node(value.wire.node).ok_or("missing exported input node")?;
            let NodeKind::Input { artifact: imported, .. } = node.kind() else {
                return Err("exported root input is not an input node".into());
            };
            let destination = Port { root, name: name.clone() };
            if let Some(imported) = imported {
                if root != 1 || imported.production_id.spec_hash != preprocessing.spec_hash {
                    return Err("refresh artifact is not produced by the one linked preprocessing execution".into());
                }
                links.push(Link {
                    producer: Port { root: 0, name: imported.artifact_name.clone() },
                    consumer: destination,
                });
                continue;
            }
            let contract = if name == "refresh-parameter-search-hash-key" {
                InputContract::Bytes { length: 32.into() }
            } else if root == 0 {
                InputContract::OneHotPolynomialFamily {
                    block_lengths: bundle.noise_snapshot.family_active_counts.clone(),
                }
            } else if name == "refresh-parameter-search-native-monomials" {
                let ConcreteWireType::IndexedFamily { count, .. } = value.wire_type else {
                    return Err("native monomial input is not a family".into());
                };
                InputContract::Family {
                    count: count.into(),
                    element: Box::new(InputContract::SignedMonomial {
                        exponent_stride: bundle.noise_snapshot.layout.stride(),
                    }),
                }
            } else {
                return Err(format!("unexpected refresh external input: {name}").into());
            };
            externals
                .entry(name.clone())
                .or_insert_with(|| ExternalInput { contract, destinations: Vec::new() })
                .destinations
                .push(destination);
        }
    }
    let fresh_digits = u32::try_from(bundle.noise_snapshot.refresh.fresh_digit_count)?;
    let bound = bundle.noise_snapshot.refresh.base_p.pow(fresh_digits) - 1u8;
    let linked = LinkedClaim {
        roots: vec![
            ClaimRoot {
                graph: producer_graph,
                artifact: &preprocessing,
                field: "preprocessing".into(),
            },
            ClaimRoot {
                graph: verification_graph,
                artifact: &verification,
                field: "verification".into(),
            },
            ClaimRoot { graph: &ideal_graph.graph, artifact: &ideal, field: "ideal".into() },
        ],
        externals: externals.into_values().collect(),
        links,
        requirements: Vec::new(),
        actual: Port { root: 1, name: "residual".into() },
        ideal: Port { root: 2, name: "zero".into() },
        endpoint: Endpoint::MatrixApprox { bound },
    };
    let program_scope = mxx_ir_core::FrozenGraphScopeId::ParallelBody {
        parent: Box::new(mxx_ir_core::FrozenGraphScopeId::Root),
        owner: program_owner,
    };
    let program_matrix = validated_verification.scopes[&program_scope].wire_types
        [&program_stages[0].current]
        .matrix_type()
        .ok_or("native program state is not a matrix")?;
    let layout_index = verification
        .backend_layouts
        .iter()
        .position(|layout| {
            layout.modulus == program_matrix.modulus &&
                layout.ring_dimension == program_matrix.ring_dimension &&
                2 * layout.regular_digits == program_matrix.columns
        })
        .ok_or("native program backend layout is missing")?;
    let native_program_proofs = native_program_proof::native_program_proof(
        validated_preprocessing,
        validated_verification,
        &preprocessing,
        &verification,
        &program_stages,
        &program_key_stages,
        layout_index,
        &linked.externals,
        &bundle.noise_snapshot,
    )?;
    let claim = claim::assemble_claim(
        &linked,
        bindings,
        &ClaimBackend {
            module_name: backend.module_name(),
            context_name: backend.context_name(),
            layouts: &backend.exporter_bindings(),
        },
        &ClaimSemantics {
            imports: &[],
            hash_model_type: "MxxRuntime.HashModel",
            // MatrixApprox does not use Boolean decoder semantics.
            centered_lift: "",
            message_center: "",
            decoder_radius: "",
        },
    )?;
    let mut execution_witnesses = String::from(
        "import Claim\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedExecution\nopen GeneratedClaim\n\n",
    );
    for (name, artifact, run) in [
        ("preprocessing", &preprocessing, "runs.2.1"),
        ("verification", &verification, "runs.2.2.1"),
    ] {
        execution_witnesses.push_str(&format!("def {name}Inputs {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : {}}} {{outputs : {}}}\n    (_runs : {} backend hashModel params inputs outputs) := inputs\n\n",artifact.root.parameter_type,artifact.root.input_type,artifact.root.output_type,artifact.root.relation));
        let output_path = ".2"
            .repeat(artifact.scope_proofs[&mxx_ir_core::FrozenGraphScopeId::Root].constraint_count);
        execution_witnesses.push_str(&format!("noncomputable def {name}Witness {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) := Classical.choose ({run})\n\nnoncomputable def {name}Facts {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) := Classical.choose_spec ({run})\n\nnoncomputable def {name}Output {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) := ({name}Facts runs){output_path}\n\n"));
    }
    execution_witnesses.push_str(&format!("noncomputable def programWitness {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) (lane : Fin {program_lane_count}) := Classical.choose ((verificationFacts runs){program_execution_path}.1 lane)\n\nnoncomputable def programFacts {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) (lane : Fin {program_lane_count}) := Classical.choose_spec ((verificationFacts runs){program_execution_path}.1 lane)\n\n"));
    execution_witnesses.push_str("end ExponentGeneratedExecution\n");
    let noise_report = bundle.simulate_noise()?;
    let numeric_proofs = numeric::refresh_bounds(&bundle.noise_snapshot, &noise_report)?;
    let trace_proofs = trace::trace_indices(
        &bundle.noise_snapshot.layout,
        bundle
            .noise_snapshot
            .refresh
            .transcript_topology
            .as_ref()
            .ok_or("missing transcript topology")?
            .coefficient_count,
    )?;
    fs::create_dir_all(directory)?;
    let mut sources = BTreeMap::new();
    sources.insert(backend.module_name().to_owned(), backend.source().to_owned());
    sources.insert("RefreshPreprocessing".into(), preprocessing.source.to_owned());
    sources.insert("RefreshVerification".into(), verification.source.to_owned());
    sources.insert("RefreshIdeal".into(), ideal.source.to_owned());
    sources.insert("Claim".into(), claim);
    sources.insert("ExecutionWitnesses".into(), execution_witnesses);
    sources.insert("MatrixActionScopes".into(), action_proofs);
    sources.insert("SelectorSetupScopes".into(), setup_proofs);
    sources.insert("EncodingInputScopes".into(), input_proofs);
    sources.insert("RootInputs".into(), root_input_proofs);
    sources.insert("RootOnlineReadout".into(), online_readout_proofs);
    sources.insert("RootSeededKeys".into(), seeded_key_proofs);
    sources.insert("RootReadoutPublic".into(), readout_public_proofs);
    sources.insert("RootDecoderSetup".into(), decoder_setup_proofs);
    sources.insert("RootRefreshDecode".into(), refresh_decode_proofs);
    sources.insert("RootRefreshDigits".into(), refresh_digit_proofs);
    sources.insert("RootRefreshState".into(), refresh_state_proofs);
    sources.insert("RootRefreshStateSource".into(), refresh_state_source);
    sources.insert("RootRefreshNormalize".into(), refresh_normalize_proofs);
    sources.insert("RootRefreshFinish".into(), refresh_finish_proofs);
    sources.insert("RootRefreshHighSource".into(), refresh_high_source_proofs);
    if let Some(correctness) = refresh_correctness {
        sources.insert("RefreshCorrectness".into(), correctness);
    }
    sources.insert("RootRefreshHighLinks".into(), refresh_high_links_proofs);
    sources.insert("RootRefreshSource".into(), refresh_source_proofs);
    sources.insert("RootRefreshRouteTargets".into(), refresh_route_targets);
    sources.insert("RootRefreshRoutePublic".into(), refresh_route_public);
    sources.insert("RootRefreshSourceProof".into(), refresh_source_proof);
    sources.insert("RootRefreshRouteSums".into(), refresh_route_sums);
    sources.insert("RootRefreshCrt".into(), refresh_crt_proofs);
    sources.insert("RootReadoutFunction".into(), readout_function_proofs);
    sources.insert("RootReadoutStage".into(), readout_stage_proofs);
    sources.insert("RootReadoutProof".into(), readout_proof);
    sources.insert("RootReadoutOperands".into(), readout_operand_proofs);
    sources.insert("RootProgram".into(), program_proofs);
    sources.insert("ExponentNativeProgramProof".into(), native_program_proofs);
    fs::write(directory.join("program-proof-bindings.txt"), program_manifest)?;
    sources.insert("RootProgramContext".into(), program_context_proofs);
    sources.insert("RootProgramInputs".into(), program_input_proofs);
    sources.insert("RootReduction".into(), reduction_proofs);
    sources.insert("RootProgramSetup".into(), program_setup_proofs);
    sources.insert("RootProgramKeys".into(), program_key_proofs);
    fs::write(directory.join("ProgramKeyBindings.txt"), program_key_stages.iter().map(|(name, stage)| format!(
        "{name}: {} output={:?} public_inputs={:?} public_cache={:?} lanes={} members={} offset={} raw_count={}\n",
        stage.theorem, stage.output, stage.public_inputs, stage.public_cache, stage.lane_count, stage.member_count, stage.offset, stage.raw_count
    )).collect::<String>())?;
    sources.insert("RootCacheAlignment".into(), cache_alignment_proofs);
    sources.insert("RootPrivate".into(), private_proofs);
    sources.insert("RootReadout".into(), readout_proofs);
    sources.insert("RootReadoutAlignment".into(), alignment_proofs);
    sources.insert("RootDecomposition".into(), root_proofs);
    sources.insert("NumericBounds".into(), numeric_proofs);
    sources.insert("TraceIndices".into(), trace_proofs);
    for (name, source) in &sources {
        fs::write(directory.join(format!("{name}.lean")), source)?;
    }
    Ok(ExportedRefreshGraphs { sources, preprocessing, verification, ideal })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise_refresh::{
        pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
        prf::SparseLwrPrfProfile,
        setup::{RefreshParameterSimulationRequest, tests::parameters_for_numeric_refresh},
    };
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};

    #[test]
    fn test_export_concrete_refresh_implementation() {
        let parameter = |name, default| {
            std::env::var(name).ok().map(|value| value.parse::<u32>().unwrap()).unwrap_or(default)
        };
        let parameters = DCRTPolyParams::new(
            parameter("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32),
            parameter("MXX_PRIMITIVE_TEST_CRT_DEPTH", 3) as usize,
            parameter("MXX_PRIMITIVE_TEST_CRT_BITS", 40) as usize,
            parameter("MXX_PRIMITIVE_TEST_BASE_BITS", 4),
            None,
        );
        let mut setup = parameters_for_numeric_refresh(&parameters);
        let profile = SparseLwrPrfProfile::new(
            crate::layout::ExpLayout {
                stride_log2: 1,
                width_log2: parameters.ring_dimension().ilog2() - 1,
            },
            2,
        )
        .unwrap();
        setup.lut_width = profile.layout.width();
        let expected = setup.layout.ring().polynomial([1.into()]);
        let generated = generate_key_layout(
            &PbcParameters::paper_evaluation(10, 4),
            PbcRootSeed(rand::random()),
            &[0, 1, 2, 3],
        )
        .unwrap();
        let bundle = RefreshParameterSimulationRequest::new(
            setup.clone(),
            profile,
            generated,
            rand::random(),
            setup.decoder_sigma.clone(),
            expected,
            1,
        )
        .unwrap()
        .build()
        .unwrap();
        let rings = bundle.ring_parameters(parameters).unwrap();
        let layouts = mxx_runtime::lean::export_dcrt_layouts(&rings).unwrap();
        let backend =
            mxx_runtime::lean::render_backend_context(&layouts, "Backend", "RefreshBackend")
                .unwrap();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/exponent_lut_lean_export")
            .join(
                bundle
                    .public_identity()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            );
        let exported =
            export_refresh_graphs(&bundle, &ParamEnv::default(), &backend, &directory).unwrap();
        assert_eq!(
            exported.preprocessing.spec_hash,
            mxx_ir_core::encoding::spec_hash(
                &bundle.preprocessing_graph().graph,
                &ParamEnv::default()
            )
            .unwrap()
        );
        assert_eq!(
            exported.verification.spec_hash,
            mxx_ir_core::encoding::spec_hash(
                &bundle.verification_graph().graph,
                &ParamEnv::default()
            )
            .unwrap()
        );
    }

    pub(super) fn accepted_worst_case_fixture()
    -> (RefreshParameterSimulationBundle, mxx_runtime::lean::LeanBackendArtifact) {
        let parameter = |name, default| {
            std::env::var(name).ok().map(|value| value.parse::<u32>().unwrap()).unwrap_or(default)
        };
        let n = parameter("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32);
        let max_depth = parameter("MXX_PRIMITIVE_TEST_CRT_DEPTH", 8) as usize;
        let bits = parameter("MXX_PRIMITIVE_TEST_CRT_BITS", 40) as usize;
        let base_bits = parameter("MXX_PRIMITIVE_TEST_BASE_BITS", 4);
        let profile = SparseLwrPrfProfile::new(
            crate::layout::ExpLayout { stride_log2: 1, width_log2: n.ilog2() - 1 },
            2,
        )
        .unwrap();
        let generated = generate_key_layout(
            &PbcParameters::paper_evaluation(10, 4),
            PbcRootSeed(rand::random()),
            &[0, 1, 2, 3],
        )
        .unwrap();
        let key_id = rand::random();
        let mut selected = None;
        // The first accepted tuple determines the next step. No graph is
        // materialized for rejected numerical candidates.
        'depths: for depth in 2..=max_depth {
            let parameters = DCRTPolyParams::new(n, depth, bits, base_bits, None);
            let mut setup = parameters_for_numeric_refresh(&parameters);
            let primes = parameters.to_crt().0;
            let full_modulus = parameters.modulus().as_ref().clone();
            setup.switch_moduli = (0..depth)
                .map(|slot| {
                    &full_modulus /
                        num_bigint::BigUint::from(
                            primes[if slot + 1 == depth { 0 } else { depth - 1 }],
                        )
                })
                .collect();
            setup.lut_width = profile.layout.width();
            // The explicit high-ring mask target includes the switching lift;
            // constrain its support before constructing a decomposition plan.
            let maximum_digits =
                (setup.switch_moduli.iter().map(|modulus| modulus.bits()).min().unwrap() as usize -
                    2) /
                    2;
            for digits in 1..=maximum_digits {
                setup.mask_base_p_digit_counts.fill(digits);
                let expected = setup.layout.ring().polynomial([1.into()]);
                let snapshot = RefreshParameterSimulationRequest::new(
                    setup.clone(),
                    profile.clone(),
                    generated.clone(),
                    key_id,
                    setup.decoder_sigma.clone(),
                    expected,
                    1,
                )
                .unwrap()
                .build_noise_snapshot_only()
                .unwrap();
                let report = snapshot.simulate().unwrap();
                if report.refresh.accepted {
                    selected = Some((parameters, setup, report));
                    break 'depths;
                }
            }
        }
        let (parameters, setup, report) =
            selected.expect("a worst-case noise candidate in the configured search range");
        assert!(report.refresh.slots.iter().all(|slot| slot.lift > 1u8.into()));
        let expected = setup.layout.ring().polynomial([1.into()]);
        let bundle = RefreshParameterSimulationRequest::new(
            setup.clone(),
            profile,
            generated,
            key_id,
            setup.decoder_sigma.clone(),
            expected,
            1,
        )
        .unwrap()
        .build()
        .unwrap();
        assert_eq!(bundle.simulate_noise().unwrap(), report);
        let rings = bundle.ring_parameters(parameters).unwrap();
        let layouts = mxx_runtime::lean::export_dcrt_layouts(&rings).unwrap();
        let backend =
            mxx_runtime::lean::render_backend_context(&layouts, "Backend", "RefreshBackend")
                .unwrap();
        (bundle, backend)
    }

    #[test]
    #[ignore = "searches a worst-case noise fixture and exports its full implementation"]
    fn test_export_accepted_worst_case_candidate() {
        let (bundle, backend) = accepted_worst_case_fixture();
        let report = bundle.simulate_noise().unwrap();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/exponent_lut_lean_accepted")
            .join(
                bundle
                    .public_identity()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            );
        export_refresh_graphs(&bundle, &ParamEnv::default(), &backend, &directory).unwrap();
        fs::write(directory.join("noise-report.json"), serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
    }
}
