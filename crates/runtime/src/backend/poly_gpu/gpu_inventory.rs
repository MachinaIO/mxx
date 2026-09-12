//! Derive a complete prepared inventory from a validated graph.
//!
//! Every root-scope node whose kind has a compiled admitted runner contributes
//! its retained output owners, its fixed-input preparation slots, and the
//! exact native scratch its runner claims at the full output width. Artifact
//! inputs and exported outputs contribute the codec's staging and readback
//! resources. Unsupported kinds fail here, before any node executes. Scratch
//! is sized by the same native queries the admitted runners use; it is never
//! a bytes-per-column estimate.

use super::{
    gpu_compiled::{PreimageClaimPlan, PreimagePlanKey, TrapdoorPlanKey},
    *,
};
use mxx_ir_core::{
    ValidatedGraph,
    graph::FrozenGraphScopeId,
    node::{ConcatAxis, HashVariant, NodeKind},
    types::{ConcreteWireType, Port, WireRef},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnData,
        gpu_dcrt_poly::{
            GpuCompactTransferKind, GpuCpuStagingLayout, GpuDCRTPolyMatrix, GpuMatrixCrtOperation,
            GpuPreparedSlotKind, GpuPreparedStorage, GpuPreparedWorkspaceLayout, GpuTracedClaim,
            trace_native_claims,
        },
    },
    sampler::{PolyTrapdoorSampler, trapdoor::gpu::GpuDCRTPolyTrapdoorSampler},
};
use std::collections::BTreeMap;

/// Native demand of one parameter context on one device.
#[derive(Default)]
struct ContextDemand {
    matrices: Vec<(usize, usize)>,
    layouts: Vec<GpuPreparedWorkspaceLayout>,
}

fn stream() -> GpuPreparedWorkspaceLayout {
    GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::SubmissionStream,
        bytes: 0,
        alignment: 1,
    }
}

fn event() -> GpuPreparedWorkspaceLayout {
    GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::CompletionEvent,
        bytes: 0,
        alignment: 1,
    }
}

fn compact_payload(
    parameters: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bound: &num_bigint::BigUint,
) -> Result<GpuPreparedWorkspaceLayout, PolyBackendError> {
    Ok(GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::CompactPayload,
        bytes: GpuSmallMatrix::allocation_bytes(parameters, rows, columns, bound)
            .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?,
        alignment: 256,
    })
}

fn pinned_payload(
    ty: &ConcreteMatrixType,
    bound: &num_bigint::BigUint,
) -> GpuPreparedWorkspaceLayout {
    GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::PinnedHost,
        bytes: ty.rows *
            ty.columns *
            ty.ring_dimension *
            (usize::try_from(bound.bits().div_ceil(8)).unwrap_or(usize::MAX).max(1) + 1),
        alignment: 1,
    }
}

fn bound_of(wire_type: &ConcreteWireType) -> Option<(&ConcreteMatrixType, num_bigint::BigUint)> {
    match wire_type {
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
        ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
            Some((matrix, max_coefficient_bound.to_biguint()?))
        }
        _ => None,
    }
}

/// Add traced claims as inventory demand: matrix owners become backing
/// matrices of the traced shape, other kinds become typed workspace layouts.
fn push_traced(
    demand: &mut BTreeMap<(String, usize), (ConcreteMatrixType, ContextDemand)>,
    ty: &ConcreteMatrixType,
    parameters: &GpuDCRTPolyParams,
    claims: &[GpuTracedClaim],
    add_matrix: &impl Fn(
        &mut BTreeMap<(String, usize), (ConcreteMatrixType, ContextDemand)>,
        &ConcreteMatrixType,
        usize,
        usize,
    ),
    add_layouts: &impl Fn(
        &mut BTreeMap<(String, usize), (ConcreteMatrixType, ContextDemand)>,
        &ConcreteMatrixType,
        Vec<GpuPreparedWorkspaceLayout>,
    ),
) {
    let _ = parameters;
    for claim in claims {
        if claim.kind() == GpuPreparedSlotKind::Matrix {
            add_matrix(demand, ty, claim.rows(), claim.columns());
        } else if let Some(layout) = claim.layout() {
            add_layouts(demand, ty, vec![layout]);
        }
    }
}

impl GpuDcrtBackend {
    /// Trace the exact claims of sampling one trapdoor class and preparing its
    /// preimage covariance cache, on the still-open domains of device 0.
    fn trace_trapdoor_sampling(
        &mut self,
        matrix: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<Vec<GpuTracedClaim>, PolyBackendError> {
        let key = TrapdoorPlanKey {
            modulus: matrix.modulus.to_string(),
            ring_dimension: matrix.ring_dimension,
            rows: matrix.rows,
            columns: matrix.columns,
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.to_string(),
            digit_count,
        };
        if let Some(claims) = self.trapdoor_plans.get(&key) {
            return Ok(claims.clone());
        }
        let params = self.devices[0].1.parameters(matrix)?.clone();
        let device = &mut self.devices[0].1;
        let (sampled, mut claims) =
            trace_native_claims(|| device.sample_trapdoor(matrix, sigma, gadget_base, digit_count))
                .map_err(PolyBackendError::GpuSubmission)?;
        let (_, trapdoor) = sampled?;
        let sampler = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::new(&params, sigma);
        let (_, cache) =
            trace_native_claims(|| sampler.prepare_preimage_cache(&params, &trapdoor, matrix.rows))
                .map_err(PolyBackendError::GpuSubmission)?;
        claims.extend(cache);
        self.trapdoor_plans.insert(key, claims.clone());
        Ok(claims)
    }

    /// Trace the exact claims of scalar polynomial value readback for `ty` in
    /// the requested domain, for both possible input formats (the consumed
    /// matrix's format is only known at execution).
    fn trace_polynomial_values(
        &mut self,
        ty: &ConcreteMatrixType,
        evaluation: bool,
    ) -> Result<Vec<GpuTracedClaim>, PolyBackendError> {
        let params = self.devices[0].1.parameters(ty)?.clone();
        let mut all = Vec::new();
        for input_ntt in [false, true] {
            let key = (
                params.modulus().to_string(),
                params.ring_dimension() as usize,
                evaluation,
                input_ntt,
            );
            if let Some(claims) = self.polynomial_value_plans.get(&key) {
                all.extend(claims.iter().cloned());
                continue;
            }
            let mut probe = <GpuDCRTPolyMatrix as PolyMatrix>::zero(&params, 1, 1);
            if input_ntt {
                probe.ntt_all_in_place();
            }
            let device = &mut self.devices[0].1;
            let (values, claims) =
                trace_native_claims(|| device.polynomial_values(&probe, evaluation))
                    .map_err(PolyBackendError::GpuSubmission)?;
            values?;
            all.extend(claims.iter().cloned());
            self.polynomial_value_plans.insert(key, claims);
        }
        Ok(all)
    }

    /// Trace the exact claims of one preimage class: the destination's
    /// hard-cutoff plan, the staged target tile and one candidate attempt at
    /// the single-column pilot width and the full output width.
    fn trace_preimage_plan(
        &mut self,
        trapdoor_matrix: &ConcreteMatrixType,
        public: &ConcreteMatrixType,
        ty: &ConcreteMatrixType,
        bound: &num_bigint::BigUint,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<PreimageClaimPlan, PolyBackendError> {
        let key = PreimagePlanKey {
            modulus: ty.modulus.to_string(),
            ring_dimension: ty.ring_dimension,
            rows: ty.rows,
            columns: ty.columns,
            public_rows: public.rows,
            bound: bound.to_string(),
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.to_string(),
            digit_count,
        };
        if let Some(plan) = self.preimage_plans.get(&key) {
            return Ok(plan.clone());
        }
        let trapdoor_claims =
            self.trace_trapdoor_sampling(trapdoor_matrix, sigma, gadget_base, digit_count)?;
        let params = self.devices[0].1.parameters(ty)?.clone();
        let (public_matrix, trapdoor) =
            self.devices[0].1.sample_trapdoor(trapdoor_matrix, sigma, gadget_base, digit_count)?;
        let sampler = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::new(&params, sigma);
        sampler.prepare_preimage_cache(&params, &trapdoor, public.rows);
        let (destination, destination_claims) = trace_native_claims(|| {
            GpuDCRTPolyTrapdoorSampler::preimage_destination(
                &params,
                ty.rows,
                ty.columns,
                bound.clone(),
            )
        })
        .map_err(PolyBackendError::GpuSubmission)?;
        let mut destination = destination?;
        if destination_claims.first().map(|c| c.kind()) != Some(GpuPreparedSlotKind::CompactPayload)
        {
            return Err(PolyBackendError::GpuSubmission(
                "preimage destination trace does not start with its payload".into(),
            ));
        }
        let level = params.crt_depth() - 1;
        let bytes_per_poly = GpuDCRTPolyMatrix::cpu_staging_layout(
            &params,
            &GpuDCRTPolyMatrix::zero(&params, 1, 1).into_cpu_staging_bytes(),
        )
        .map_err(PolyBackendError::GpuCalibration)?
        .bytes_per_poly;
        let staging = GpuCpuStagingLayout {
            rows: public.rows,
            columns: ty.columns,
            level,
            is_ntt: true,
            bytes_per_poly,
        }
        .zero_bytes(&params)
        .map_err(PolyBackendError::GpuCalibration)?;
        let target = PolyMatrixColumnData::<GpuFleetMatrix>::staged(
            &params,
            Arc::new(staging),
            0,
            ty.columns,
        );
        let mut tile = std::collections::BTreeMap::new();
        let mut attempt = std::collections::BTreeMap::new();
        let seed: [u8; 32] = rand::random();
        for width in [1usize, ty.columns] {
            if width == 0 || tile.contains_key(&width) {
                continue;
            }
            let (materialized, tile_claims) = trace_native_claims(|| {
                super::gpu_compiled::materialize_preimage_tile(
                    &params,
                    &target,
                    public.rows,
                    0,
                    width,
                )
            })
            .map_err(PolyBackendError::GpuSubmission)?;
            let materialized = materialized.map_err(PolyBackendError::GpuSubmission)?;
            let (attempted, attempt_claims) = trace_native_claims(|| {
                sampler.preimage_attempt(
                    &params,
                    &trapdoor,
                    &public_matrix,
                    &materialized,
                    &mut destination,
                    0,
                    0,
                    0,
                    seed,
                )
            })
            .map_err(PolyBackendError::GpuSubmission)?;
            attempted?;
            tile.insert(width, tile_claims);
            attempt.insert(width, attempt_claims);
        }
        let plan = PreimageClaimPlan {
            destination: destination_claims[1..].to_vec(),
            tile,
            attempt,
            trapdoor: trapdoor_claims,
            attempts: mxx_primitives::env::gpu_preimage_max_tile_attempts()
                .map_err(PolyBackendError::GpuSubmission)?,
            bytes_per_poly,
        };
        self.preimage_plans.insert(key, plan.clone());
        Ok(plan)
    }

    /// Provision the complete prepared inventory for `validated` and accept it
    /// through `prepare_memory`. Returns immediately when a ledger is already
    /// installed. Kinds without a compiled admitted runner are rejected before
    /// any GPU work; the caller asserts exclusive device observation.
    pub fn prepare_graph_admission(
        &mut self,
        validated: &ValidatedGraph,
    ) -> Result<(), PolyBackendError> {
        if self.prepared_ledger.is_some() {
            return Ok(());
        }
        let scope = validated
            .source
            .scope(&FrozenGraphScopeId::Root)
            .ok_or_else(|| PolyBackendError::GpuSubmission("graph has no root scope".into()))?;
        let checked = validated.root_scope();
        let unsupported = |kind: &NodeKind| {
            PolyBackendError::GpuSubmission(format!(
                "prepared GPU admission has no compiled runner for {kind:?}"
            ))
        };
        // Demand per parameter context, keyed by the context's matrix type
        // identity (modulus and ring dimension); replicated on every device.
        let mut demand = BTreeMap::<(String, usize), (ConcreteMatrixType, ContextDemand)>::new();
        let add_matrix = |demand: &mut BTreeMap<_, (ConcreteMatrixType, ContextDemand)>,
                          ty: &ConcreteMatrixType,
                          rows: usize,
                          columns: usize| {
            demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), ContextDemand::default()))
                .1
                .matrices
                .push((rows, columns));
        };
        let add_layouts = |demand: &mut BTreeMap<_, (ConcreteMatrixType, ContextDemand)>,
                           ty: &ConcreteMatrixType,
                           layouts: Vec<GpuPreparedWorkspaceLayout>| {
            demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), ContextDemand::default()))
                .1
                .layouts
                .extend(layouts);
        };
        let parameters =
            |backend: &Self, ty: &ConcreteMatrixType| backend.devices[0].1.parameters(ty).cloned();
        for (position, handle) in checked.execution_order.iter().enumerate() {
            let id = mxx_ir_core::types::NodeId(position as u64);
            let arguments = scope
                .arguments(handle)
                .ok_or_else(|| PolyBackendError::GpuSubmission("missing node arguments".into()))?;
            let argument_types =
                arguments.iter().map(|wire| checked.wire_types[wire].clone()).collect::<Vec<_>>();
            let output_types = (0..handle.output_types().len())
                .map(|port| {
                    checked.wire_types[&WireRef { node: id, port: Port(port as u32) }].clone()
                })
                .collect::<Vec<_>>();
            let kind = handle.kind();
            // Scalars and structural family plumbing hold no native owners.
            match kind {
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::BitExtract { .. } |
                NodeKind::IntToReal |
                NodeKind::BoolToInt |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt => continue,
                NodeKind::Input { .. } => {}
                NodeKind::ConstantMatrix { .. } |
                NodeKind::UniformResidueSample { .. } |
                NodeKind::UniformIntervalSample { .. } |
                NodeKind::GaussianSample { .. } |
                NodeKind::HashSample { .. } |
                NodeKind::GadgetDecompose { .. } |
                NodeKind::MatrixBinary(_) |
                NodeKind::MatrixMulAccumulate { .. } |
                NodeKind::MatrixMulSmallRhs |
                NodeKind::MatrixNegate |
                NodeKind::MatrixScale { .. } |
                NodeKind::RingAutomorphism { .. } |
                NodeKind::ModulusSwitch { .. } |
                NodeKind::ModulusReduce { .. } |
                NodeKind::CenteredRebase { .. } |
                NodeKind::CenteredExtend { .. } |
                NodeKind::BlockModSwitch { .. } |
                NodeKind::RnsModUp { .. } |
                NodeKind::RnsModDown { .. } |
                NodeKind::CrtRecompose { .. } |
                NodeKind::Transpose |
                NodeKind::Slice { .. } |
                NodeKind::Tensor |
                NodeKind::Concat { .. } |
                NodeKind::PolynomialValues { .. } |
                NodeKind::TrapdoorSample { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::PreimageSample { .. } => {}
                other => return Err(unsupported(other)),
            }
            // Retained outputs.
            for output in &output_types {
                match output {
                    ConcreteWireType::Matrix(ty) => {
                        add_matrix(&mut demand, ty, ty.rows, ty.columns)
                    }
                    ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                        let (ty, bound) =
                            bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
                        let params = parameters(self, ty)?;
                        let layout = compact_payload(&params, ty.rows, ty.columns, &bound)?;
                        add_layouts(&mut demand, ty, vec![layout]);
                    }
                    ConcreteWireType::Bytes { .. } | ConcreteWireType::Trapdoor { .. } => {}
                    // Scalar value families (polynomial readback) hold no owners.
                    ConcreteWireType::IndexedFamily { element, .. }
                        if matches!(**element, ConcreteWireType::Int) => {}
                    _ => return Err(unsupported(kind)),
                }
            }
            // Fixed-input preparation: one full-shape owner per ordinary matrix
            // argument covers replicas and format normalization fragments.
            for argument in &argument_types {
                if let ConcreteWireType::Matrix(ty) = argument {
                    add_matrix(&mut demand, ty, ty.rows, ty.columns);
                }
            }
            // Kind-specific native scratch at the full output width.
            match kind {
                NodeKind::Input { .. } => {
                    for output in &output_types {
                        match output {
                            ConcreteWireType::Matrix(ty) => {
                                // Artifact import: coefficient staging, codec
                                // stream and load workspace for the widest payload.
                                let params = parameters(self, ty)?;
                                let level = params.crt_depth() - 1;
                                let bits = u16::try_from(params.modulus().bits())
                                    .map_err(|_| PolyBackendError::InvalidInteger)?;
                                let load = params
                                    .compact_transfer_workspace(
                                        level,
                                        ty.rows,
                                        ty.columns,
                                        GpuCompactTransferKind::Load { max_coefficient_bits: bits },
                                    )
                                    .map_err(PolyBackendError::GpuSubmission)?;
                                let transfer = params
                                    .rns_transfer_workspace(level, ty.rows, ty.columns)
                                    .map_err(PolyBackendError::GpuSubmission)?;
                                add_matrix(&mut demand, ty, ty.rows, ty.columns);
                                add_layouts(
                                    &mut demand,
                                    ty,
                                    vec![
                                        stream(),
                                        load,
                                        GpuPreparedWorkspaceLayout {
                                            kind: GpuPreparedSlotKind::PinnedHost,
                                            bytes: transfer.bytes,
                                            alignment: 1,
                                        },
                                        transfer,
                                        event(),
                                    ],
                                );
                            }
                            ConcreteWireType::SmallMatrix { .. } |
                            ConcreteWireType::Preimage { .. } => {
                                let (ty, bound) =
                                    bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
                                let params = parameters(self, ty)?;
                                let staging =
                                    compact_payload(&params, ty.rows, ty.columns, &bound)?;
                                add_layouts(
                                    &mut demand,
                                    ty,
                                    vec![staging, pinned_payload(ty, &bound), event()],
                                );
                            }
                            _ => {}
                        }
                    }
                }
                NodeKind::GadgetDecompose { digit_count, .. } |
                NodeKind::HashSample {
                    variant: HashVariant::Decomposed | HashVariant::SmallDecomposed,
                    digit_count: Some(digit_count),
                    ..
                } => {
                    let (ty, _) =
                        bound_of(&output_types[0]).ok_or(PolyBackendError::InvalidInteger)?;
                    let digits = digit_count
                        .evaluate(&validated.bindings)
                        .ok()
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|digits| *digits != 0)
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    let params = parameters(self, ty)?;
                    let level = params.crt_depth() - 1;
                    let source_rows = ty.rows / digits;
                    let layout = params
                        .compact_decomposition_layout(
                            match kind {
                                NodeKind::GadgetDecompose { small, .. } => *small,
                                _ => matches!(
                                    kind,
                                    NodeKind::HashSample {
                                        variant: HashVariant::SmallDecomposed,
                                        ..
                                    }
                                ),
                            },
                            Some(digits),
                        )
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    // Hashed source and correction copy in coefficient format.
                    add_matrix(&mut demand, ty, source_rows, ty.columns);
                    add_matrix(&mut demand, ty, source_rows, ty.columns);
                    let correction = params
                        .matrix_gadget_correction_workspace_bytes(
                            level,
                            source_rows,
                            1,
                            layout.dropped_moduli,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    if correction.additional_bytes != 0 {
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![GpuPreparedWorkspaceLayout {
                                kind: GpuPreparedSlotKind::TransformWorkspace,
                                bytes: correction.additional_bytes,
                                alignment: correction.alignment,
                            }],
                        );
                    }
                }
                NodeKind::MatrixMulSmallRhs => {
                    let ConcreteWireType::Matrix(ty) = &output_types[0] else {
                        return Err(unsupported(kind));
                    };
                    let inner = match argument_types.first() {
                        Some(ConcreteWireType::Matrix(left)) => left.columns,
                        _ => return Err(unsupported(kind)),
                    };
                    let params = parameters(self, ty)?;
                    let workspaces = params
                        .small_rhs_workspaces(params.crt_depth() - 1, inner, ty.columns)
                        .map_err(PolyBackendError::GpuSubmission)?;
                    add_layouts(&mut demand, ty, workspaces);
                }
                NodeKind::PolynomialValues { evaluation } => {
                    let Some(ConcreteWireType::Matrix(ty)) = argument_types.first() else {
                        return Err(unsupported(kind));
                    };
                    let claims = self.trace_polynomial_values(ty, *evaluation)?;
                    let params = parameters(self, ty)?;
                    push_traced(&mut demand, ty, &params, &claims, &add_matrix, &add_layouts);
                }
                NodeKind::CrtRecompose { plaintext_moduli, .. } => {
                    let ConcreteWireType::Matrix(ty) = &output_types[0] else {
                        return Err(unsupported(kind));
                    };
                    let params = parameters(self, ty)?;
                    let layout = params
                        .matrix_crt_workspace_bytes(
                            params.crt_depth() - 1,
                            1,
                            1,
                            GpuMatrixCrtOperation::Recompose,
                            1,
                            plaintext_moduli.len(),
                            true,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    if layout.additional_bytes != 0 {
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![GpuPreparedWorkspaceLayout {
                                kind: GpuPreparedSlotKind::TransformWorkspace,
                                bytes: layout.additional_bytes,
                                alignment: layout.alignment,
                            }],
                        );
                    }
                }
                NodeKind::RnsModUp { source_moduli, .. } |
                NodeKind::RnsModDown { source_moduli, .. } => {
                    let ConcreteWireType::Matrix(ty) = &output_types[0] else {
                        return Err(unsupported(kind));
                    };
                    let params = parameters(self, ty)?;
                    let layout = params
                        .matrix_crt_workspace_bytes(
                            params.crt_depth() - 1,
                            1,
                            1,
                            GpuMatrixCrtOperation::RnsConversion,
                            source_moduli.len(),
                            1,
                            true,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    if layout.additional_bytes != 0 {
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![GpuPreparedWorkspaceLayout {
                                kind: GpuPreparedSlotKind::TransformWorkspace,
                                bytes: layout.additional_bytes,
                                alignment: layout.alignment,
                            }],
                        );
                    }
                }
                NodeKind::TrapdoorSample { .. } => {
                    // Port 0 is the public matrix, port 1 the trapdoor.
                    let Some(ConcreteWireType::Trapdoor {
                        matrix,
                        sigma,
                        gadget_base,
                        digit_count,
                        ..
                    }) = output_types.get(1)
                    else {
                        return Err(unsupported(kind));
                    };
                    let sigma = sigma
                        .evaluate_f64(&validated.bindings)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    // Prepared trapdoor/preimage sampling is single-device; refuse
                    // here, before any domain is sealed, rather than at the node.
                    if self.devices.len() != 1 {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    let claims =
                        self.trace_trapdoor_sampling(matrix, sigma, gadget_base, *digit_count)?;
                    let params = parameters(self, matrix)?;
                    push_traced(&mut demand, matrix, &params, &claims, &add_matrix, &add_layouts);
                }
                NodeKind::PreimageSample { .. } => {
                    let (
                        Some(ConcreteWireType::Matrix(public)),
                        Some(ConcreteWireType::Trapdoor {
                            matrix: trapdoor_matrix,
                            sigma,
                            gadget_base,
                            digit_count,
                            ..
                        }),
                    ) = (argument_types.first(), argument_types.get(1))
                    else {
                        return Err(unsupported(kind));
                    };
                    let (ty, bound) =
                        bound_of(&output_types[0]).ok_or(PolyBackendError::InvalidInteger)?;
                    let sigma = sigma
                        .evaluate_f64(&validated.bindings)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    if self.devices.len() != 1 {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    let plan = self.trace_preimage_plan(
                        trapdoor_matrix,
                        public,
                        ty,
                        &bound,
                        sigma,
                        gadget_base,
                        *digit_count,
                    )?;
                    let params = parameters(self, ty)?;
                    let mut claims = plan.destination.clone();
                    claims.extend(plan.trapdoor.iter().cloned());
                    for list in plan.tile.values().chain(plan.attempt.values()) {
                        claims.extend(list.iter().cloned());
                    }
                    push_traced(&mut demand, ty, &params, &claims, &add_matrix, &add_layouts);
                }
                NodeKind::Concat { axis: ConcatAxis::Rows } => {
                    // Fused row sums reduce large groups through one-row
                    // intermediates; provision one per concatenated block.
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        for _ in &argument_types {
                            add_matrix(&mut demand, ty, 1, ty.columns);
                        }
                    }
                }
                NodeKind::Tensor => {
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        add_matrix(&mut demand, ty, 1, ty.columns);
                    }
                }
                _ => {}
            }
        }
        // Exports: the codec clones the owner, opens its private stream and
        // uses the store workspace; compact exports record one completion.
        for (_, output) in validated.source.outputs() {
            match &checked.wire_types[&output.value] {
                ConcreteWireType::Matrix(ty) => {
                    let params = parameters(self, ty)?;
                    let store = params
                        .compact_transfer_workspace(
                            params.crt_depth() - 1,
                            ty.rows,
                            ty.columns,
                            GpuCompactTransferKind::Store,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    add_layouts(&mut demand, ty, vec![stream(), store]);
                }
                other if bound_of(other).is_some() => {
                    let (ty, _) = bound_of(other).unwrap();
                    add_layouts(&mut demand, ty, vec![event()]);
                }
                _ => {}
            }
        }
        // Build one storage per context per device before any domain closes.
        let mut prepared = Vec::new();
        for (device, (_, backend)) in self.devices.iter().enumerate() {
            for (ty, context) in demand.values() {
                let params = backend.parameters(ty)?;
                let matrices = context
                    .matrices
                    .iter()
                    .map(|&(rows, columns)| {
                        GpuDCRTPolyMatrix::zero(params, rows.max(1), columns.max(1))
                    })
                    .collect::<Vec<_>>();
                let matrices = if matrices.is_empty() {
                    vec![GpuDCRTPolyMatrix::zero(params, 1, 1)]
                } else {
                    matrices
                };
                let storage = GpuPreparedStorage::new(matrices, Some(&context.layouts))
                    .map_err(PolyBackendError::GpuSubmission)?;
                prepared.push((device, Arc::new(storage)));
            }
        }
        self.prepare_memory(prepared, true)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, MemoryArtifactStore, RuntimeValue,
        executor::{ExecutionConfig, execute_with_config},
        transcript::SamplingMode,
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly as _, PolyParams,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use std::collections::BTreeMap;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_execution_derives_complete_prepared_inventory() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let a = ring.input("a", (2, 3));
        let b = ring.input("b", (2, 3));
        let sum = a.clone() + b;
        let reconstructed = a.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let coefficients = ring.input("c", (1, 1)).coefficients();
        let graph = DslContext::new("prepared-graph-admission")
            .output("sum", sum)
            .unwrap()
            .output("reconstructed", reconstructed)
            .unwrap()
            .output("coefficients", coefficients)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let sampler = DCRTPolyUniformSampler::new();
        let left = sampler.sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let right = sampler.sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected_sum =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&left + &right)).to_compact_bytes();
        let expected_left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).to_compact_bytes();
        let scalar = sampler.sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let expected_coefficients = scalar
            .entry(0, 0)
            .coeffs_biguints()
            .into_iter()
            .map(num_bigint::BigInt::from)
            .collect::<Vec<_>>();
        // One resident input and one host-staged artifact input.
        let inputs = BTreeMap::from([
            (
                "a".to_string(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).into()),
            ),
            (
                "b".to_string(),
                RuntimeValue::HostMatrix {
                    matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                        modulus: num_bigint::BigInt::from(params.modulus().as_ref().clone()),
                        ring_dimension: n as usize,
                        rows: 2,
                        columns: 3,
                    },
                    bytes: std::sync::Arc::new(
                        GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right)
                            .into_cpu_staging_bytes(),
                    ),
                },
            ),
            (
                "c".to_string(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &scalar).into()),
            ),
        ]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let mut store = MemoryArtifactStore::default();
        let config = ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() };
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            config,
        )
        .unwrap();
        assert!(backend.prepared_ledger.is_some());
        for (name, expected) in [("sum", &expected_sum), ("reconstructed", &expected_left)] {
            let RuntimeValue::Matrix(output) =
                result.materialize_output(name, &mut backend, &mut store).unwrap()
            else {
                panic!("matrix output")
            };
            assert_eq!(&backend.matrix_to_bytes(&output).unwrap(), expected, "{name}");
        }
        // Scalar polynomial readback runs inside its traced claim boundary.
        let RuntimeValue::IndexedFamily(values) =
            result.materialize_output("coefficients", &mut backend, &mut store).unwrap()
        else {
            panic!("coefficient family output")
        };
        let values = values
            .into_iter()
            .map(|value| match value {
                RuntimeValue::Int(value) => value.clone(),
                _ => panic!("integer coefficient"),
            })
            .collect::<Vec<_>>();
        assert_eq!(values, expected_coefficients);

        // A kind without a compiled runner is rejected before any node runs and
        // before any domain is sealed. Trapdoor sampling is admitted since
        // Step 3, so use a runtime polynomial import, which has no runner.
        let values = mxx_dsl::Family::pack(
            (0..n).map(|value| mxx_dsl::Int::constant(value as u64)).collect(),
        )
        .unwrap();
        let trapdoor_graph = DslContext::new("prepared-graph-admission-unsupported")
            .output("polynomial", ring.from_coefficients(&values))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut fresh = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let config = ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() };
        assert!(
            execute_with_config(
                &trapdoor_graph,
                &mut fresh,
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
                config,
            )
            .is_err()
        );
        assert!(fresh.prepared_ledger.is_none());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_execution_admits_trapdoor_and_preimage_sampling() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let trapdoor = ring.sample_trapdoor(1, 5, 16, digits, 100_000_000);
        let target = ring.input("t", (1, 3));
        let preimage = trapdoor.sample_preimage(target, (digits + 2, 3));
        let check = preimage.mul_small_rhs(trapdoor.public_matrix());
        let graph = DslContext::new("prepared-graph-preimage")
            .output("check", check)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let sampler = DCRTPolyUniformSampler::new();
        let target_value = sampler.sample_uniform(&cpu, 1, 3, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &target_value).to_compact_bytes();
        let inputs = BTreeMap::from([(
            "t".to_string(),
            RuntimeValue::HostMatrix {
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    modulus: num_bigint::BigInt::from(params.modulus().as_ref().clone()),
                    ring_dimension: n as usize,
                    rows: 1,
                    columns: 3,
                },
                bytes: std::sync::Arc::new(
                    GpuDCRTPolyMatrix::from_cpu_matrix(&params, &target_value)
                        .into_cpu_staging_bytes(),
                ),
            },
        )]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let mut store = MemoryArtifactStore::default();
        let config = ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() };
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            config,
        )
        .unwrap();
        assert!(backend.prepared_ledger.is_some());
        let RuntimeValue::Matrix(output) =
            result.materialize_output("check", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output")
        };
        // A · G^{-1}-free trapdoor preimage x satisfies A x = t exactly.
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }
}
