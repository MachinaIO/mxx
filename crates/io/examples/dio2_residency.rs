//! Production-shape residency diagnostic, without executing an obfuscation or a round trip.
//!
//! Reports the former helper-family payload and PRF input sizes. Helpers and large inputs
//! are now consumed by member. This is not a measurement of a complete GPU plan's peak. Parameters
//! use `MXX_IO_DIO2_ESTIMATE_*` overrides.

use bigdecimal::BigDecimal;
use mxx_backends::{MemoryArtifactStore, RuntimeValue};
use mxx_exponent_lut::{
    ExpLayout,
    api::{ApiError, StageExecution, StageOutputs},
    refresh::PrfSettings,
};
use mxx_io::dio2::{Dio2Parameters, StreamingFunction, step::StepPlan};
use mxx_ir_core::{ValidatedGraph, artifact::ProductionId};
use std::collections::BTreeMap;

fn setting(name: &str, default: usize) -> usize {
    std::env::var(format!("MXX_IO_DIO2_ESTIMATE_{name}"))
        .map_or(default, |value| value.parse().expect("integer estimate parameter"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let n = setting("N", 1 << 16) as u32;
    let crt_bits = setting("CRT_BITS", 60);
    let parameters = Dio2Parameters {
        ring_dimension: n,
        crt_bits,
        base_bits: setting("BASE_BITS", 6) as u32,
        primes: mxx_ir_core::generate_crt_basis(n, setting("PRIMES", 27), crt_bits)?,
        levels: setting("LEVELS", 16),
        prf: PrfSettings {
            lwr_dimension: setting("LWR_DIMENSION", 1024),
            support_weight: setting("SUPPORT_WEIGHT", 31),
            input_modulus: setting("INPUT_MODULUS", 256),
            output_modulus: setting("OUTPUT_MODULUS", 32),
            bucket_width: 0,
        },
        smudge_digits: setting("SMUDGE_DIGITS", 306),
        trapdoor_sigma: 4,
        encoding_error_sigma: BigDecimal::from(4),
    };
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 })
            .with_step_row(vec![1, 0, 1, 0])?
            .with_step_row(vec![0, 1, 1, 0])?;
    let plan = StepPlan::new(&parameters, &function)?;
    let row_bytes = u128::from(n) * parameters.primes.len() as u128 * plan.columns as u128 * 8;
    let helper_row_bytes = row_bytes * plan.automorphisms.len() as u128;
    println!(
        "n={n} primes={} w0={} labels={}x{} parts={} helpers={}",
        parameters.primes.len(),
        plan.columns,
        plan.label_rows,
        plan.label_width + 1,
        2 * plan.full_target_count,
        plan.automorphisms.len()
    );
    println!("full_row_bytes={row_bytes}");
    let preimage_bound = mxx_io::dio2::preimage_bound(&parameters)?;
    let preimage_coefficient_bytes = 1 + preimage_bound.bits().div_ceil(8);
    let preimage_bytes = 2 *
        (plan.columns as u128 + 2) *
        plan.columns as u128 *
        u128::from(n) *
        u128::from(preimage_coefficient_bytes);
    println!(
        "preimage_bound_bits={} preimage_coefficient_bytes={preimage_coefficient_bytes} one_preimage_bytes={preimage_bytes} two_preimage_import_buffers_bytes={}",
        preimage_bound.bits(),
        2 * preimage_bytes
    );

    // Computational anchors retain covariance leaves and one stacked [R; E] allocation.
    // R/E are views, including after dynamic family selection. These are payload terms,
    // not a complete planner peak (operation workspaces and other live rows are additional).
    let d = mxx_exponent_lut::decoder::unmask::UNMASK_ROWS as u128;
    let polynomial_bytes = u128::from(n) * parameters.primes.len() as u128 * 8;
    let old_secret = (4 * d * d * plan.columns as u128 + 3 * d * d) * polynomial_bytes;
    let shared_secret = (2 * d * d * plan.columns as u128 + 3 * d * d) * polynomial_bytes;
    let public = d * d * (plan.columns as u128 + 2) * polynomial_bytes;
    let anchors = function.stream_length() as u128;
    println!(
        "anchor_secret_bytes_before_sharing={old_secret} anchor_secret_bytes_after_sharing={shared_secret} anchor_public_bytes={public}"
    );
    println!(
        "anchor_families_plus_one_preimage_bytes_before_sharing={} anchor_families_plus_one_preimage_bytes_after_sharing={}",
        anchors * (old_secret + public) + preimage_bytes,
        anchors * (shared_secret + public) + preimage_bytes
    );

    println!("before_fix_one_row_per_helper_bytes={helper_row_bytes}");
    println!("after_fix_simultaneous_helper_rows=1");
    let shape = |role: &str, members| parameters.prf.declared_shape(1, role, members).unwrap();
    let keys = mxx_io::dio2::step::ChainKeys {
        sigma: Some(shape("sigma", plan.ring_dimension)),
        cbd: Some(shape("cbd", plan.cbd_members())),
        smudge: shape("smudge", plan.smudge_members()),
    };
    let program =
        mxx_io::dio2::step::ChainProgram::refreshing(&parameters, &function, &plan, &keys)?;
    for (role, prf) in program
        .sigma
        .iter()
        .map(|s| ("sigma", s.prf()))
        .chain(program.cbd.iter().map(|s| ("cbd", s.prf())))
        .chain(std::iter::once(("smudge", &program.smudge)))
    {
        let selectors: u128 = prf.selector_families().iter().map(|s| s.members as u128).sum();
        println!(
            "{role} integer_input_members_per_chunk={selectors} bytes_per_2_chunks_2_families_at_16_bytes={}",
            selectors * 2 * 2 * 16
        );
    }
    if std::env::var_os("MXX_IO_DIO2_ESTIMATE_GRAPHS").is_none() {
        return Ok(());
    }
    let mut dio2 = mxx_io::dio2::Dio2::new(
        parameters,
        RecordGraphs,
        mxx_backends::MemoryArtifactStore::default(),
    )?
    .with_inputs_by_shape();
    use mxx_io::Obfuscation;
    let input = mxx_io::dio2::Dio2Function {
        function: function.clone(),
        y: vec![vec![false]; function.stream_length()],
        initial_state: 0,
    };
    let obfuscation = dio2.obfuscate(&input)?;
    dio2.evaluate(&obfuscation, &vec![vec![false]; function.stream_length()])?;
    Ok(())
}

/// Record shapes only. This executor performs no cryptographic operations.
struct RecordGraphs;
fn inspect(graph: &ValidatedGraph, stage: &str) {
    use mxx_ir_core::{
        node::NodeKind,
        types::{ConcreteWireType as T, NodeId, Port, WireRef},
    };
    let types = graph.root_scope();
    let mut total = 0u128;
    for (index, node) in graph.source.root_scope().nodes().iter().enumerate() {
        if let NodeKind::Input { name, artifact, .. } = node.kind() {
            if let Some(T::IndexedFamily { element, count }) =
                types.wire_types.get(&WireRef { node: NodeId(index as u64), port: Port(0) })
            {
                if matches!(element.as_ref(), T::Int | T::Bool) {
                    let bytes = count * 8;
                    if artifact.is_none() {
                        total += bytes;
                    }
                    println!(
                        "{stage} integer_input={name} streamed={} count={count} declared_bytes={bytes}",
                        artifact.is_some()
                    );
                }
            }
        }
    }
    println!("{stage} root_integer_input_bytes={total}");
    if std::env::var_os("MXX_IO_DIO2_ESTIMATE_LIVENESS").is_some() {
        inspect_sampler_liveness(graph, stage);
    }
}
/// List logical live matrices at each sampler and its enclosing loop call sites.
/// This deliberately does not sum them into a VRAM estimate: parent/child inputs,
/// views and packed families can alias, and backend scratch is not represented here.
fn inspect_sampler_liveness(graph: &ValidatedGraph, stage: &str) {
    use mxx_ir_core::{graph::FrozenGraphScopeId, node::NodeKind, types::ConcreteWireType};
    for (scope_id, scope) in graph.source.scopes() {
        for (index, node) in scope.nodes().iter().enumerate() {
            if !matches!(node.kind(), NodeKind::PreimageSample { .. }) {
                continue;
            }
            println!("{stage} sampler_scope={scope_id:?} sampler_node={index}");
            let mut current = scope_id;
            let mut position = index;
            loop {
                let validated = &graph.scopes[current];
                let source = graph.source.scope(current).unwrap();
                for (wire, ty) in &validated.wire_types {
                    if wire.node.0 as usize >= position ||
                        (!validated.liveness.retained.contains(wire) &&
                            validated
                                .liveness
                                .last_use
                                .get(wire)
                                .is_none_or(|last| *last < position))
                    {
                        continue;
                    }
                    // Families are reported by member shape, never multiplied by their
                    // symbolic cardinality: streamed families need not be resident.
                    let (element, members) = match ty {
                        ConcreteWireType::IndexedFamily { element, count } => {
                            (element.as_ref(), Some(*count))
                        }
                        _ => (ty, None),
                    };
                    if let Some(matrix) = element.matrix_type() {
                        let bytes = matrix.rows as u128 *
                            matrix.columns as u128 *
                            u128::from(matrix.ring.ring_dimension()) *
                            matrix.ring.crt_depth() as u128 *
                            8;
                        println!(
                            "{stage} sampler_live scope={current:?} call={position} wire={wire:?} producer={:?} rows={} columns={} full_member_bytes={bytes} family_members={members:?} artifact_input={}",
                            std::mem::discriminant(source.node(wire.node).unwrap().kind()),
                            matrix.rows,
                            matrix.columns,
                            validated.artifact_inputs.contains_key(wire)
                        );
                    }
                }
                match current {
                    FrozenGraphScopeId::ParallelBody { parent, owner } |
                    FrozenGraphScopeId::SequentialBody { parent, owner } => {
                        current = parent;
                        position = owner.0 as usize;
                    }
                    _ => break,
                }
            }
        }
    }
}

impl StageExecution<MemoryArtifactStore> for RecordGraphs {
    fn produce(
        &mut self,
        graph: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut MemoryArtifactStore,
        nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        inspect(&graph, "obfuscate");
        Ok(mxx_ir_core::artifact::production_id(
            mxx_ir_core::encoding::spec_hash(&graph.source, &graph.bindings).unwrap(),
            nonce,
        ))
    }
    fn evaluate(
        &mut self,
        graph: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut MemoryArtifactStore,
        _: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        inspect(&graph, "evaluate");
        Ok(StageOutputs {
            outputs: graph
                .source
                .outputs()
                .keys()
                .map(|name| (name.clone(), RuntimeValue::Int(0.into())))
                .collect(),
        })
    }
    fn download_matrix(
        &self,
        _: &RuntimeValue,
    ) -> Result<mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix, ApiError> {
        unreachable!("graph shape recording never downloads a matrix")
    }
    fn download_integers(&self, _: &RuntimeValue) -> Result<Vec<num_bigint::BigInt>, ApiError> {
        Ok(vec![0.into()])
    }
}
