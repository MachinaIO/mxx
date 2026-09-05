//! Exponent-LUT adapters for mechanical export of the frozen implementation.

pub mod check;
mod numeric;
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
    if setup_scope_count == 0 {
        return Err("refresh implementation has no recognized selector-setup scope".into());
    }
    let preprocessing = preprocessing_artifact;
    let verification_artifact = export(&verification, &options("RefreshVerification"))?;
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
    fs::write(directory.join(format!("{}.lean", backend.module_name())), backend.source())?;
    fs::write(directory.join("RefreshPreprocessing.lean"), &preprocessing.source)?;
    fs::write(directory.join("RefreshVerification.lean"), &verification.source)?;
    fs::write(directory.join("RefreshIdeal.lean"), &ideal.source)?;
    fs::write(directory.join("Claim.lean"), claim)?;
    fs::write(directory.join("MatrixActionScopes.lean"), action_proofs)?;
    fs::write(directory.join("SelectorSetupScopes.lean"), setup_proofs)?;
    fs::write(directory.join("EncodingInputScopes.lean"), input_proofs)?;
    fs::write(directory.join("RootDecomposition.lean"), root_proofs)?;
    fs::write(directory.join("NumericBounds.lean"), numeric_proofs)?;
    fs::write(directory.join("TraceIndices.lean"), trace_proofs)?;
    Ok(ExportedRefreshGraphs { preprocessing, verification, ideal })
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

    #[test]
    #[ignore = "searches a worst-case noise fixture and exports its full implementation"]
    fn test_export_accepted_worst_case_candidate() {
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
