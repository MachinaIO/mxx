//! Derive a complete prepared inventory from a validated graph.
//!
//! Root-scope liveness assigns retained outputs and transient operation demand
//! to reusable shape/class slots. A slot can serve a later node as soon as its
//! Rust owners have died; native reader events still order physical reuse. Artifact
//! inputs and exported outputs contribute the codec's staging and readback
//! resources. Unsupported kinds fail here, before any node executes. Scratch
//! is sized by the same native queries the admitted runners use; it is never
//! a bytes-per-column estimate.

use super::{
    gpu_compiled::{PreimageClaimPlan, PreimagePlanKey, PreparedMatrixOperation, TrapdoorPlanKey},
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
            GpuCompactTransferKind, GpuCpuStagingLayout, GpuDCRTPolyMatrix, GpuGraphAdmissionGuard,
            GpuMatrixCrtOperation, GpuPreparedSlotKind, GpuPreparedStorage,
            GpuPreparedWorkspaceLayout, GpuTracedClaim, trace_native_claims,
        },
    },
    sampler::{PolyTrapdoorSampler, trapdoor::gpu::GpuDCRTPolyTrapdoorSampler},
};
use std::collections::BTreeMap;

/// Possible source boundaries and a descending list of containing owner widths.
/// Capacities are not execution intervals. For alternative family layouts, rank
/// k contains the kth largest fragment of every possible selected member.
#[derive(Clone, Default)]
struct ColumnInventoryLayout {
    cuts: Vec<usize>,
    capacities: Vec<usize>,
    alternatives: bool,
    // May contain descriptors with no native owner until materialization.
    // Alternatives use OR: eager members retain their separately counted owners.
    lazy: bool,
    packed_family: bool,
}

impl ColumnInventoryLayout {
    fn new(mut cuts: Vec<usize>) -> Self {
        cuts.sort_unstable();
        cuts.dedup();
        let mut capacities = cuts.windows(2).map(|range| range[1] - range[0]).collect::<Vec<_>>();
        capacities.sort_unstable_by(|a, b| b.cmp(a));
        Self { cuts, capacities, alternatives: false, lazy: false, packed_family: false }
    }

    fn include_alternative(&mut self, other: Self) {
        if self.cuts.is_empty() {
            *self = other;
            return;
        }
        self.lazy |= other.lazy;
        self.packed_family |= other.packed_family;
        self.alternatives |= other.alternatives || self.cuts != other.cuts;
        self.cuts.extend(other.cuts);
        self.cuts.sort_unstable();
        self.cuts.dedup();
        self.capacities.resize(self.capacities.len().max(other.capacities.len()), 0);
        for (capacity, alternative) in self.capacities.iter_mut().zip(other.capacities) {
            *capacity = (*capacity).max(alternative);
        }
    }

    fn include_input(
        &mut self,
        value: &crate::backend::RuntimeValue<GpuDcrtBackend>,
        columns: usize,
    ) {
        use crate::backend::RuntimeValue;
        let cuts = match value {
            RuntimeValue::Matrix(matrix) | RuntimeValue::Trapdoor { public: matrix, .. } => matrix
                .shards()
                .iter()
                .flat_map(|shard| {
                    [shard.global_column_start, shard.global_column_start + shard.value.size().1]
                })
                .collect(),
            RuntimeValue::SmallMatrix(matrix) => matrix
                .shards()
                .iter()
                .flat_map(|shard| {
                    [shard.global_column_start, shard.global_column_start + shard.value.size().1]
                })
                .collect(),
            RuntimeValue::IndexedFamily(members) => {
                // Accumulate one bounded summary; do not retain a layout per
                // member or a traversal queue proportional to family cardinality.
                for member in members {
                    self.include_input(member, columns);
                }
                self.packed_family = true;
                return;
            }
            // Host/lazy members materialize as complete fresh matrices.
            _ => vec![0, columns],
        };
        let mut layout = Self::new(cuts);
        layout.lazy = matches!(
            value,
            RuntimeValue::LazyArtifact { .. } |
                RuntimeValue::StagedArtifact { .. } |
                RuntimeValue::HostMatrix { .. } |
                RuntimeValue::LazyArtifactFamily { .. } |
                RuntimeValue::StagedArtifactFamily { .. }
        );
        self.include_alternative(layout);
    }
}

/// Native demand of one parameter context on one device.
#[derive(Default)]
struct ContextDemand {
    matrices: Vec<(usize, usize)>,
    matrix_until: Vec<usize>,
    layouts: Vec<GpuPreparedWorkspaceLayout>,
    layout_until: Vec<usize>,
}

impl ContextDemand {
    fn matrix(&mut self, begin: usize, end: usize, rows: usize, columns: usize) {
        use super::gpu_prepare::matrix_capacity_class;
        let class = matrix_capacity_class(rows, columns);
        let mut shape = (rows, columns);
        let mut available = None;
        for (index, &(r, c)) in self.matrices.iter().enumerate() {
            if matrix_capacity_class(r, c) == class {
                shape.0 = shape.0.max(r);
                shape.1 = shape.1.max(c);
                if self.matrix_until[index] < begin {
                    available = Some(index);
                }
            }
        }
        // Every slot of a class has the same containing shape, making the
        // native smallest-fit order irrelevant within that class. Dimensions
        // use observed maxima, not power-of-two padded allocations.
        for slot in &mut self.matrices {
            if matrix_capacity_class(slot.0, slot.1) == class {
                *slot = shape;
            }
        }
        if let Some(index) = available {
            self.matrix_until[index] = end;
        } else {
            self.matrices.push(shape);
            self.matrix_until.push(end);
        }
    }

    fn missing(
        &self,
        params: &GpuDCRTPolyParams,
        inventory: &[(usize, Arc<GpuPreparedStorage>)],
        device: usize,
        used: &mut HashSet<u64>,
    ) -> Result<(Vec<(usize, usize)>, Vec<GpuPreparedWorkspaceLayout>), PolyBackendError> {
        let broker = super::gpu_compiled::PreparedClaimBroker::new(
            params,
            inventory
                .iter()
                .filter(|(owner, _)| *owner == device)
                .map(|(_, storage)| storage.clone())
                .collect(),
        );
        let claims = self
            .matrices
            .iter()
            .map(|&(rows, columns)| {
                GpuTracedClaim::matrix(rows.max(1), columns.max(1), params.crt_depth() - 1, true)
            })
            .chain(self.layouts.iter().copied().map(GpuTracedClaim::workspace));
        let mut matrices = Vec::new();
        let mut layouts = Vec::new();
        for claim in claims {
            if let Some(slot) =
                broker.select(&claim, used).map_err(PolyBackendError::GpuSubmission)?
            {
                used.insert(slot.slot_id());
            } else if claim.kind() == GpuPreparedSlotKind::Matrix {
                matrices.push((claim.rows(), claim.columns()));
            } else {
                layouts.push(claim.layout().unwrap());
            }
        }
        Ok((matrices, layouts))
    }

    fn workspace(&mut self, begin: usize, end: usize, mut layout: GpuPreparedWorkspaceLayout) {
        use super::gpu_prepare::capacity_class;
        let matches = |previous: &GpuPreparedWorkspaceLayout| {
            previous.kind == layout.kind &&
                previous.alignment == layout.alignment &&
                capacity_class(previous.bytes) == capacity_class(layout.bytes)
        };
        let indices = self
            .layouts
            .iter()
            .enumerate()
            .filter_map(|(index, previous)| matches(previous).then_some(index))
            .collect::<Vec<_>>();
        let available = indices.iter().copied().find(|&index| self.layout_until[index] < begin);
        for &index in &indices {
            layout.bytes = layout.bytes.max(self.layouts[index].bytes);
        }
        for index in indices {
            self.layouts[index].bytes = layout.bytes;
        }
        // Zero-byte events and streams still need one slot per simultaneous
        // owner; kind and alignment remain distinct native resource domains.
        if let Some(index) = available {
            self.layout_until[index] = end;
        } else {
            self.layouts.push(layout);
            self.layout_until.push(end);
        }
    }
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

// Families own their leaves; structural selections only alias those owners.
fn family_leaves(ty: &ConcreteWireType) -> Box<dyn Iterator<Item = &ConcreteWireType> + '_> {
    match ty {
        ConcreteWireType::IndexedFamily { element, count } => {
            Box::new((0..*count).flat_map(move |_| family_leaves(element)))
        }
        _ => Box::new(std::iter::once(ty)),
    }
}

fn import_inventory(
    parameters: &GpuDCRTPolyParams,
    output: &ConcreteWireType,
    scratch_columns: usize,
) -> Result<(Vec<(usize, usize)>, Vec<GpuPreparedWorkspaceLayout>), PolyBackendError> {
    let mut matrices = Vec::new();
    let mut layouts = Vec::new();
    match output {
        ConcreteWireType::Matrix(ty) => {
            // Artifact import: coefficient staging, codec
            // stream and load workspace for the widest payload.
            let params = parameters;
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
                .rns_transfer_workspace(level, ty.rows, ty.columns.min(scratch_columns))
                .map_err(PolyBackendError::GpuSubmission)?;
            matrices.push((ty.rows, ty.columns.min(scratch_columns)));
            layouts.extend(vec![
                stream(),
                load,
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::PinnedHost,
                    bytes: transfer.bytes,
                    alignment: 1,
                },
                transfer,
                event(),
            ]);
        }
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
            let (ty, bound) = bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
            let params = parameters;
            let staging =
                compact_payload(&params, ty.rows, ty.columns.min(scratch_columns), &bound)?;
            layouts.extend(vec![
                staging,
                pinned_payload(
                    &ConcreteMatrixType { columns: ty.columns.min(scratch_columns), ..ty.clone() },
                    &bound,
                ),
                event(),
            ]);
        }
        _ => {}
    }
    Ok((matrices, layouts))
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
        warm_up: bool,
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
        if !warm_up {
            return Err(PolyBackendError::GpuSubmission(format!(
                "explicit graph warmup required: missing trace_trapdoor_sampling resource plan for {key:?}"
            )));
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
        warm_up: bool,
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
            if !warm_up {
                return Err(PolyBackendError::GpuSubmission(format!(
                    "explicit graph warmup required: missing trace_polynomial_values resource plan for {key:?}"
                )));
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
    /// hard-cutoff plan, the staged target tile and one single-column candidate.
    /// Native layout queries specialize the traced claim order for other widths.
    fn trace_preimage_plan(
        &mut self,
        trapdoor_matrix: &ConcreteMatrixType,
        public: &ConcreteMatrixType,
        ty: &ConcreteMatrixType,
        bound: &num_bigint::BigUint,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
        warm_up: bool,
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
        if !warm_up {
            return Err(PolyBackendError::GpuSubmission(format!(
                "explicit graph warmup required: missing trace_preimage_plan resource plan for {key:?}"
            )));
        }
        let trapdoor_claims = self.trace_trapdoor_sampling(
            trapdoor_matrix,
            sigma,
            gadget_base,
            digit_count,
            warm_up,
        )?;
        let params = self.devices[0].1.parameters(ty)?.clone();
        let (public_matrix, trapdoor) =
            self.devices[0].1.sample_trapdoor(trapdoor_matrix, sigma, gadget_base, digit_count)?;
        let sampler = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::new(&params, sigma);
        sampler.prepare_preimage_cache(&params, &trapdoor, public.rows);
        let (destination, destination_claims) = trace_native_claims(|| {
            GpuDCRTPolyTrapdoorSampler::preimage_destination(&params, ty.rows, 1, bound.clone())
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
            columns: 1,
            level,
            is_ntt: true,
            bytes_per_poly,
        }
        .zero_bytes(&params)
        .map_err(PolyBackendError::GpuCalibration)?;
        let target =
            PolyMatrixColumnData::<GpuFleetMatrix>::staged(&params, Arc::new(staging), 0, 1);
        let seed: [u8; 32] = rand::random();
        let (materialized, tile) = trace_native_claims(|| {
            super::gpu_compiled::materialize_preimage_tile(&params, &target, public.rows, 0, 1)
        })
        .map_err(PolyBackendError::GpuSubmission)?;
        let materialized = materialized.map_err(PolyBackendError::GpuSubmission)?;
        let (attempted, attempt) = trace_native_claims(|| {
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
    /// through `prepare_memory`. Explicitly installed ledgers retain their
    /// caller-managed lifecycle; automatic inventories reuse free backing and
    /// add only missing demand. The caller asserts exclusive device observation.
    /// Explicit resource warmup uses `warm_up = true` and discards the returned
    /// guard before production. Reuse this backend for every warmed graph.
    /// Production always uses false: cache misses fail before discovery trials.
    pub fn prepare_graph_admission(
        &mut self,
        validated: &ValidatedGraph,
        capture_trace: bool,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        warm_up: bool,
    ) -> Result<Option<GpuGraphAdmissionGuard>, PolyBackendError> {
        if self.prepared_ledger.is_some() && !self.graph_prepared {
            return Ok(None);
        }
        let guard = GpuGraphAdmissionGuard::new(self.device_parameters())
            .map_err(PolyBackendError::GpuSubmission)?;
        // Caller-owned fragments remain distinct in the compiled runner, even
        // when they occupy the same device. Preserve their actual boundaries.
        let input_columns = validated
            .root_scope()
            .execution_order
            .par_iter()
            .enumerate()
            .filter_map(|(position, handle)| {
                let NodeKind::Input { name, artifact, .. } = handle.kind() else { return None };
                let wire =
                    WireRef { node: mxx_ir_core::types::NodeId(position as u64), port: Port(0) };
                let mut layout = ColumnInventoryLayout::default();
                let mut ty = &validated.root_scope().wire_types[&wire];
                while let ConcreteWireType::IndexedFamily { element, .. } = ty {
                    ty = element;
                }
                if let Some(ty) = ty.matrix_type() {
                    if let Some(value) = inputs.get(name) {
                        layout.include_input(value, ty.columns);
                    }
                    if layout.cuts.is_empty() {
                        layout = ColumnInventoryLayout::new(vec![0, ty.columns]);
                    }
                }
                layout.lazy |= artifact.is_some();
                Some((wire, layout))
            })
            .collect::<BTreeMap<_, _>>();
        let maximum = validated
            .root_scope()
            .wire_types
            .values()
            .filter_map(ConcreteWireType::matrix_type)
            .map(|ty| ty.columns)
            .max()
            .unwrap_or(1)
            .max(1);
        let parameters = self.device_parameters();
        let available = parameters
            .par_iter()
            .map(|params| {
                let memory = mxx_primitives::poly::dcrt::gpu::gpu_device_memory_usage(
                    params.device_ids()[0],
                )
                .map_err(PolyBackendError::GpuCalibration)?;
                Ok(params.vram_budget_bytes().saturating_sub(memory.resident))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let inventory = self
            .prepared_ledger
            .as_ref()
            .map(|ledger| {
                ledger
                    .prepared_inventory()
                    .map(|(device, storage)| (device, storage.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut width = maximum;
        let demand = loop {
            let demand = self
                .graph_admission_demand(
                    validated,
                    &FrozenGraphScopeId::Root,
                    &validated.bindings,
                    capture_trace,
                    Some(&input_columns),
                    width,
                    warm_up,
                )?
                .0;
            let requested = self
                .devices
                .par_iter()
                .enumerate()
                .map(|(device, (_, backend))| -> Result<usize, PolyBackendError> {
                    let mut bytes = 0usize;
                    let mut used = HashSet::new();
                    for (ty, context) in demand.values() {
                        let params = backend.parameters(ty)?;
                        let (matrices, layouts) =
                            context.missing(params, &inventory, device, &mut used)?;
                        for (rows, columns) in matrices {
                            let layout = params
                                .matrix_allocation_bytes(
                                    params.crt_depth() - 1,
                                    rows.max(1),
                                    columns.max(1),
                                    true,
                                )
                                .map_err(PolyBackendError::GpuCalibration)?;
                            bytes = bytes
                                .checked_add(layout.data_bytes)
                                .and_then(|bytes| bytes.checked_add(layout.aux_bytes))
                                .ok_or(PolyBackendError::InvalidInteger)?;
                        }
                        for layout in layouts {
                            if !matches!(
                                layout.kind,
                                GpuPreparedSlotKind::PinnedHost |
                                    GpuPreparedSlotKind::CompletionEvent |
                                    GpuPreparedSlotKind::SubmissionStream
                            ) {
                                bytes = bytes
                                    .checked_add(layout.bytes)
                                    .ok_or(PolyBackendError::InvalidInteger)?;
                            }
                        }
                    }
                    Ok(bytes)
                })
                .collect::<Result<Vec<_>, PolyBackendError>>()?;
            if requested.iter().zip(&available).all(|(requested, available)| requested <= available)
            {
                break demand;
            }
            if width == 1 {
                return Err(PolyBackendError::GpuSubmission(format!(
                    "retained graph owners and one-column scratch exceed the setup memory budget: requested bytes per device {requested:?}, available bytes per device {available:?}"
                )));
            }
            width = width.div_ceil(2);
        };
        self.prepare_graph_storage(demand)?;
        self.graph_prepared = true;
        Ok(Some(guard))
    }

    fn prepare_graph_storage(
        &mut self,
        demand: BTreeMap<(String, usize), (ConcreteMatrixType, ContextDemand)>,
    ) -> Result<(), PolyBackendError> {
        // Build one storage per context per device before any domain closes.
        let mut prepared = self
            .prepared_ledger
            .as_ref()
            .map(|ledger| {
                ledger
                    .prepared_inventory()
                    .map(|(device, storage)| (device, storage.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let original_count = prepared.len();
        let mut used = HashSet::new();
        for (device, (_, backend)) in self.devices.iter().enumerate() {
            for (ty, context) in demand.values() {
                let params = backend.parameters(ty)?;
                let (missing_matrices, missing_layouts) =
                    context.missing(params, &prepared, device, &mut used)?;
                if missing_matrices.is_empty() && missing_layouts.is_empty() {
                    continue;
                }
                let matrices = missing_matrices
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
                let storage = GpuPreparedStorage::new(matrices, Some(&missing_layouts))
                    .map_err(PolyBackendError::GpuSubmission)?;
                prepared.push((device, Arc::new(storage)));
            }
        }
        if prepared.is_empty() {
            return Ok(());
        }
        if original_count != 0 && prepared.len() == original_count {
            // Re-enter the graph seal without adding backing or imposing a new
            // initial-budget check on already accepted capacity. Output leases
            // and their asynchronous reader/release edges remain untouched.
            for device in 0..self.devices.len() {
                let stores = prepared
                    .iter()
                    .filter(|(owner, _)| *owner == device)
                    .map(|(_, storage)| storage.as_ref())
                    .collect::<Vec<_>>();
                GpuPreparedStorage::finish_setup(&stores)
                    .map_err(PolyBackendError::GpuSubmission)?;
            }
            return Ok(());
        }
        let previous = self.prepared_ledger.take();
        self.prepared_required = false;
        match self.prepare_memory(prepared, true) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.prepared_required = previous.is_some();
                self.prepared_ledger = previous;
                Err(error)
            }
        }
    }
    /// Size candidates using native layout queries, before backing allocation.
    /// Diagnostic residency chooses a range cap; it is not admission evidence.
    /// The completed inventory is still accepted against coherent native receipts.
    fn graph_admission_demand(
        &mut self,
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        bindings: &ParamEnv,
        capture_trace: bool,
        input_columns: Option<&BTreeMap<WireRef, ColumnInventoryLayout>>,
        scratch_columns: usize,
        warm_up: bool,
    ) -> Result<
        (
            BTreeMap<(String, usize), (ConcreteMatrixType, ContextDemand)>,
            Vec<ColumnInventoryLayout>,
        ),
        PolyBackendError,
    > {
        let scope = validated.source.scope(scope_id).ok_or_else(|| {
            PolyBackendError::GpuSubmission("graph has no requested scope".into())
        })?;
        let checked = validated.scope(scope_id).unwrap();
        let optimizer = if *scope_id == FrozenGraphScopeId::Root {
            crate::executor::gpu_plan::inventory_plan(validated, capture_trace)
        } else {
            crate::executor::gpu_plan::InventoryPlan::default()
        };
        let mut column_layouts = BTreeMap::<WireRef, ColumnInventoryLayout>::new();
        if let Some(inputs) = input_columns {
            column_layouts.extend(inputs.iter().map(|(wire, cuts)| (*wire, cuts.clone())));
        }
        let mut owner_until = checked.liveness.last_use.clone();
        for wire in &checked.liveness.retained {
            owner_until.insert(*wire, usize::MAX);
        }
        for (&wire, &end) in &optimizer.borrowed_until {
            owner_until
                .entry(wire)
                .and_modify(|previous| *previous = (*previous).max(end))
                .or_insert(end);
        }
        // Propagate alias lifetimes backwards, including nested packs/selections.
        // Dynamic selections retain every candidate until the selection dies.
        for (position, handle) in checked.execution_order.iter().enumerate().rev() {
            for port in 0..handle.output_types().len() {
                let wire = WireRef {
                    node: mxx_ir_core::types::NodeId(position as u64),
                    port: Port(port as u32),
                };
                if let Some(source) = optimizer.aliases.get(&wire) {
                    let end = owner_until.get(&wire).copied().unwrap_or(position);
                    owner_until
                        .entry(*source)
                        .and_modify(|previous| *previous = (*previous).max(end))
                        .or_insert(end);
                }
            }
            if matches!(
                handle.kind(),
                NodeKind::FamilyPack { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. }
            ) {
                let output =
                    WireRef { node: mxx_ir_core::types::NodeId(position as u64), port: Port(0) };
                let end = owner_until.get(&output).copied().unwrap_or(position);
                for wire in scope.arguments(handle).unwrap() {
                    owner_until
                        .entry(wire)
                        .and_modify(|until| *until = (*until).max(end))
                        .or_insert(end);
                }
            }
            if let NodeKind::SequentialLoop(body) = handle.kind() {
                let count =
                    body.count.evaluate(bindings).map_err(|_| PolyBackendError::InvalidInteger)?;
                if count == BigInt::from(0) {
                    for (port, wire) in scope
                        .arguments(handle)
                        .unwrap()
                        .into_iter()
                        .take(body.carried_count)
                        .enumerate()
                    {
                        let output = WireRef {
                            node: mxx_ir_core::types::NodeId(position as u64),
                            port: Port(port as u32),
                        };
                        let end = owner_until.get(&output).copied().unwrap_or(position);
                        owner_until
                            .entry(wire)
                            .and_modify(|until| *until = (*until).max(end))
                            .or_insert(end);
                    }
                }
            }
        }

        let unsupported = |kind: &NodeKind| {
            PolyBackendError::GpuSubmission(format!(
                "prepared GPU admission has no compiled runner for {kind:?}"
            ))
        };
        // Demand per parameter context, keyed by the context's matrix type
        // identity (modulus and ring dimension); replicated on every device.
        let mut demand = BTreeMap::<(String, usize), (ConcreteMatrixType, ContextDemand)>::new();
        let position = std::cell::Cell::new(0usize);
        let until = std::cell::Cell::new(0usize);
        let add_matrix = |demand: &mut BTreeMap<_, (ConcreteMatrixType, ContextDemand)>,
                          ty: &ConcreteMatrixType,
                          rows: usize,
                          columns: usize| {
            demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), ContextDemand::default()))
                .1
                .matrix(position.get(), until.get(), rows, columns);
        };
        let add_layouts = |demand: &mut BTreeMap<_, (ConcreteMatrixType, ContextDemand)>,
                           ty: &ConcreteMatrixType,
                           layouts: Vec<GpuPreparedWorkspaceLayout>| {
            let context = &mut demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), ContextDemand::default()))
                .1;
            for layout in layouts {
                context.workspace(position.get(), until.get(), layout);
            }
        };
        let parameters =
            |backend: &Self, ty: &ConcreteMatrixType| backend.devices[0].1.parameters(ty).cloned();
        for (node_position, handle) in checked.execution_order.iter().enumerate() {
            position.set(node_position);
            until.set(node_position);
            let id = mxx_ir_core::types::NodeId(node_position as u64);
            let retained_until = |port: usize| {
                let wire = WireRef { node: id, port: Port(port as u32) };
                let mut end = if capture_trace {
                    usize::MAX
                } else {
                    owner_until.get(&wire).copied().unwrap_or(node_position)
                };
                if matches!(handle.kind(), NodeKind::Select { .. }) {
                    // Select materializes and caches only the chosen candidate.
                    // Its owner can outlive the selection output through a later
                    // use of that candidate; reserve one possible owner until
                    // the latest candidate use without importing all candidates.
                    for candidate in scope.arguments(handle).unwrap().iter().skip(1) {
                        end = end.max(owner_until.get(candidate).copied().unwrap_or(node_position));
                    }
                }
                // execute_instances_batch borrows the complete input map. A
                // child input materialized from a staged family therefore stays
                // alive until the body returns, even after its last wire use.
                if matches!(handle.kind(), NodeKind::Input { .. }) {
                    end.max(checked.execution_order.len())
                } else {
                    end
                }
            };
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
            // Stored fragments survive column-wise operations. Each interval
            // needs its own retained destination, even on the same device.
            for (port, output) in output_types.iter().enumerate() {
                let wire = WireRef { node: id, port: Port(port as u32) };
                if let Some(source) = optimizer.aliases.get(&wire) {
                    if let Some(boundaries) = column_layouts.get(source).cloned() {
                        column_layouts.insert(wire, boundaries);
                        continue;
                    }
                }
                let mut leaf = output;
                while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
                    leaf = element;
                }
                let Some(ty) = leaf.matrix_type() else { continue };
                if matches!(kind, NodeKind::Input { .. }) && column_layouts.contains_key(&wire) {
                    continue;
                }
                let mut cuts = vec![0, ty.columns];
                let mut sources = Vec::new();
                match kind {
                    NodeKind::Input { .. } |
                    NodeKind::ConstantMatrix { .. } |
                    NodeKind::UniformResidueSample { .. } |
                    NodeKind::UniformIntervalSample { .. } |
                    NodeKind::GaussianSample { .. } |
                    NodeKind::HashSample { .. } |
                    NodeKind::PolynomialFromValues { .. } |
                    NodeKind::Transpose |
                    NodeKind::Tensor |
                    NodeKind::TrapdoorSample { .. } |
                    NodeKind::PreimageSample { .. } => {}
                    NodeKind::Concat { axis } if *axis != ConcatAxis::Rows => {
                        let mut offset = 0;
                        for (argument, argument_type) in arguments.iter().zip(&argument_types) {
                            if let Some(input) = argument_type.matrix_type() {
                                if let Some(layout) = column_layouts.get(argument) {
                                    cuts.extend(layout.cuts.iter().map(|column| offset + column));
                                    sources.push(layout);
                                }
                                offset += input.columns;
                            }
                        }
                    }
                    NodeKind::MatrixMulSmallRhs |
                    NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply) => {
                        let left = argument_types[0].matrix_type().unwrap();
                        let right = argument_types[1].matrix_type().unwrap();
                        let scales_left = matches!(kind, NodeKind::MatrixBinary(_)) &&
                            crate::gpu_calibration::gpu_matrix_multiply_scales_left(
                                left.rows,
                                left.columns,
                                right.rows,
                                right.columns,
                            );
                        if let Some(layout) =
                            column_layouts.get(&arguments[usize::from(!scales_left)])
                        {
                            cuts.extend(&layout.cuts);
                            sources.push(layout);
                        }
                    }
                    NodeKind::Slice { columns: Some(range), .. } => {
                        let start = range
                            .start
                            .evaluate(bindings)
                            .ok()
                            .and_then(|v| v.to_usize())
                            .ok_or(PolyBackendError::InvalidInteger)?;
                        if let Some(layout) = column_layouts.get(&arguments[0]) {
                            cuts.extend(
                                layout
                                    .cuts
                                    .iter()
                                    .filter_map(|column| column.checked_sub(start))
                                    .filter(|column| *column < ty.columns),
                            );
                            sources.push(layout);
                        }
                    }
                    _ => {
                        for (argument, argument_type) in arguments.iter().zip(&argument_types) {
                            let mut input = argument_type;
                            while let ConcreteWireType::IndexedFamily { element, .. } = input {
                                input = element;
                            }
                            if input.matrix_type().is_some_and(|input| input.columns == ty.columns)
                            {
                                if let Some(layout) = column_layouts.get(argument) {
                                    cuts.extend(&layout.cuts);
                                    sources.push(layout);
                                }
                            }
                        }
                    }
                }
                let mut layout = ColumnInventoryLayout::new(cuts);
                if matches!(kind, NodeKind::FamilyPack { .. } | NodeKind::Select { .. }) &&
                    !sources.is_empty()
                {
                    // A selection takes one partition, rather than refining
                    // every member together. Retain rank-wise peak capacities.
                    layout = ColumnInventoryLayout::default();
                    for source in sources {
                        layout.include_alternative(source.clone());
                    }
                } else if sources.iter().any(|source| source.alternatives) {
                    layout.alternatives = true;
                    let limit = layout.cuts.len().saturating_sub(1);
                    let mut largest = std::collections::BinaryHeap::with_capacity(limit);
                    // Bound temporary storage as well as the final inventory:
                    // a many-input refinement never collects every capacity list.
                    for &width in sources.iter().flat_map(|source| &source.capacities) {
                        if largest.len() < limit {
                            largest.push(std::cmp::Reverse(width));
                        } else if let Some(mut smallest) = largest.peek_mut() {
                            if width > smallest.0 {
                                *smallest = std::cmp::Reverse(width);
                            }
                        }
                    }
                    layout.capacities = largest.into_iter().map(|width| width.0).collect();
                    layout.capacities.sort_unstable_by(|a, b| b.cmp(a));
                    // Each common-refinement interval starts at an operand
                    // interval's left endpoint, so distinct operand intervals
                    // contain them. A slice can only shorten those intervals.
                    // The kth largest nonoverlapping output is also <= C/k.
                    // Together these bounds avoid retaining every family layout.
                    for (index, width) in layout.capacities.iter_mut().enumerate() {
                        *width = (*width).min(ty.columns / (index + 1));
                    }
                }
                layout.lazy = match kind {
                    NodeKind::Input { artifact, .. } => artifact.is_some(),
                    NodeKind::FamilyPack { .. } => arguments
                        .iter()
                        .any(|wire| column_layouts.get(wire).is_some_and(|source| source.lazy)),
                    NodeKind::FamilyGetStatic { .. } => {
                        column_layouts.get(&arguments[0]).is_some_and(|source| source.lazy)
                    }
                    _ => false,
                };
                layout.packed_family = matches!(kind, NodeKind::FamilyPack { .. }) ||
                    (matches!(
                        kind,
                        NodeKind::FamilyGetStatic { .. } |
                            NodeKind::FamilyGetDynamic |
                            NodeKind::Select { .. }
                    ) && matches!(output, ConcreteWireType::IndexedFamily { .. }) &&
                        arguments.iter().any(|wire| {
                            column_layouts.get(wire).is_some_and(|source| source.packed_family)
                        }));
                if matches!(output, ConcreteWireType::IndexedFamily { .. }) &&
                    matches!(kind, NodeKind::FamilyGetDynamic | NodeKind::Select { .. })
                {
                    layout.lazy |= arguments
                        .iter()
                        .any(|wire| column_layouts.get(wire).is_some_and(|source| source.lazy));
                }
                column_layouts.insert(wire, layout);
            }
            // Only materializing consumers create owners for lazy inputs. Packs
            // and static selection copy descriptors; loop Zip placement imports
            // one member into the borrowed child input map instead.
            let structural = matches!(
                kind,
                NodeKind::Input { .. } |
                    NodeKind::FamilyPack { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. } |
                    NodeKind::SubgraphCall(_) |
                    NodeKind::SequentialLoop(_)
            );
            let materialized = arguments
                .iter()
                .enumerate()
                .filter_map(|(index, wire)| {
                    let layout = column_layouts.get(wire)?;
                    if !layout.lazy || structural {
                        return None;
                    }
                    let end = if let NodeKind::ParallelLoop(body) = kind {
                        if !matches!(
                            body.input_modes[index],
                            mxx_ir_core::node::LoopInputMode::Broadcast
                        ) || (matches!(
                            argument_types[index],
                            ConcreteWireType::IndexedFamily { .. }
                        ) && !layout.packed_family)
                        {
                            return None;
                        }
                        node_position
                    } else {
                        owner_until.get(wire).copied().unwrap_or(node_position)
                    };
                    Some((*wire, argument_types[index].clone(), end))
                })
                .collect::<Vec<_>>();
            let mut imports = Vec::new();
            for (wire, ty, end) in materialized {
                until.set(if capture_trace { usize::MAX } else { end });
                for leaf in family_leaves(&ty) {
                    if let ConcreteWireType::Matrix(ty) = leaf {
                        add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    } else if let Some((ty, bound)) = bound_of(leaf) {
                        let params = parameters(self, ty)?;
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                        );
                    }
                    imports.push(leaf.clone());
                }
                // Placement imports clone the descriptor; ordinary materialize
                // replaces this wire's cached value in the executor map.
                if !matches!(kind, NodeKind::ParallelLoop(_)) {
                    column_layouts.get_mut(&wire).unwrap().lazy = false;
                }
            }
            if matches!(
                kind,
                NodeKind::Input { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. }
            ) {
                for (port, ty) in output_types.iter().enumerate() {
                    let wire = WireRef { node: id, port: Port(port as u32) };
                    if !column_layouts.get(&wire).is_some_and(|layout| layout.lazy) &&
                        !matches!(ty, ConcreteWireType::IndexedFamily { .. })
                    {
                        imports.extend(family_leaves(ty).cloned());
                    }
                }
            }
            until.set(node_position);
            for output in imports {
                if let Some(ty) = output.matrix_type() {
                    let params = parameters(self, ty)?;
                    let (matrices, layouts) = import_inventory(&params, &output, scratch_columns)?;
                    for (rows, columns) in matrices {
                        add_matrix(&mut demand, ty, rows, columns);
                    }
                    add_layouts(&mut demand, ty, layouts);
                }
            }
            // A fleet executes one sibling body at a time, with parallelism
            // inside each admitted primitive. Summarize child peak slots once;
            // never replicate the body's scratch for the total iteration count.
            if let Some(child_id) = validated.source.child_scope_id(scope_id, id) {
                let (child_bindings, loop_slot, count) = match kind {
                    NodeKind::SubgraphCall(call) => (&call.bindings, None, 1),
                    NodeKind::ParallelLoop(body) => (
                        &body.bindings,
                        Some(body.index_slot),
                        body.count
                            .evaluate(bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or(PolyBackendError::InvalidInteger)?,
                    ),
                    NodeKind::SequentialLoop(body) => (
                        &body.bindings,
                        Some(body.index_slot),
                        body.count
                            .evaluate(bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or(PolyBackendError::InvalidInteger)?,
                    ),
                    _ => unreachable!("validated child scope"),
                };
                if count == 0 {
                    if let NodeKind::SequentialLoop(body) = kind {
                        // No body or invariant participates in a zero-count
                        // result: each output is its corresponding carried alias.
                        for (port, source) in arguments.iter().take(body.carried_count).enumerate()
                        {
                            if let Some(layout) = column_layouts.get(source).cloned() {
                                column_layouts
                                    .insert(WireRef { node: id, port: Port(port as u32) }, layout);
                            }
                        }
                    }
                    continue;
                }
                let mut child_env = bindings.clone();
                if let Some(slot) = loop_slot {
                    child_env.loop_indices.insert(slot, BigInt::from(0));
                }
                let expression_env = child_env.clone();
                for (name, expression) in child_bindings {
                    child_env.integers.insert(
                        name.clone(),
                        expression
                            .evaluate(&expression_env)
                            .map_err(|_| PolyBackendError::InvalidInteger)?,
                    );
                }
                let child_inputs = arguments
                    .iter()
                    .map(|wire| column_layouts.get(wire).cloned().unwrap_or_default())
                    .zip(validated.source.scope(&child_id).unwrap().inputs().iter().copied())
                    .map(|(mut cuts, wire)| {
                        if matches!(kind, NodeKind::ParallelLoop(_)) &&
                            !matches!(
                                validated.scope(&child_id).unwrap().wire_types[&wire],
                                ConcreteWireType::IndexedFamily { .. }
                            )
                        {
                            cuts.lazy = false;
                        }
                        (wire, cuts)
                    })
                    .collect::<BTreeMap<_, _>>();
                let (child, child_outputs) = self.graph_admission_demand(
                    validated,
                    &child_id,
                    &child_env,
                    capture_trace,
                    Some(&child_inputs),
                    scratch_columns,
                    warm_up,
                )?;
                let staged = matches!(kind, NodeKind::ParallelLoop(_)) &&
                    crate::executor::stage_matrix_family_output(
                        scope_id,
                        count,
                        self.parallel_wave_size(1),
                        scope.outputs().iter().any(|wire| wire.node == id),
                    );
                let output_end = if staged {
                    node_position
                } else {
                    (0..output_types.len()).map(retained_until).max().unwrap_or(node_position)
                };
                if !staged {
                    for (port, boundaries) in child_outputs.into_iter().enumerate() {
                        column_layouts
                            .insert(WireRef { node: id, port: Port(port as u32) }, boundaries);
                    }
                } else {
                    // Staged members are imported as fresh complete matrices;
                    // their previous device fragments do not survive storage.
                    for (port, output) in output_types.iter().enumerate() {
                        let mut leaf = output;
                        while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
                            leaf = element;
                        }
                        if let Some(ty) = leaf.matrix_type() {
                            column_layouts.insert(WireRef { node: id, port: Port(port as u32) }, {
                                let mut layout = ColumnInventoryLayout::new(vec![0, ty.columns]);
                                layout.lazy = true;
                                layout
                            });
                        }
                    }
                }
                for (_, (ty, child)) in child {
                    for ((rows, columns), end) in child.matrices.into_iter().zip(child.matrix_until)
                    {
                        until.set(if end == usize::MAX { output_end } else { node_position });
                        add_matrix(&mut demand, &ty, rows, columns);
                    }
                    for (layout, end) in child.layouts.into_iter().zip(child.layout_until) {
                        until.set(if end == usize::MAX { output_end } else { node_position });
                        add_layouts(&mut demand, &ty, vec![layout]);
                    }
                }
                if matches!(kind, NodeKind::SequentialLoop(_)) {
                    // The preceding iteration's carried outputs remain live
                    // while the next body computes their replacements.
                    until.set(node_position);
                    for ty in output_types.iter().flat_map(family_leaves) {
                        if let ConcreteWireType::Matrix(ty) = ty {
                            add_matrix(&mut demand, ty, ty.rows, ty.columns);
                        } else if let Some((ty, bound)) = bound_of(ty) {
                            let params = parameters(self, ty)?;
                            add_layouts(
                                &mut demand,
                                ty,
                                vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                            );
                        }
                    }
                }
                continue;
            }
            // Scalars create no native owners; family packs preserve aliases.
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
                NodeKind::RealSqrt |
                NodeKind::FamilyPack { .. } => continue,
                NodeKind::Input { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. } => {}
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
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::PolynomialValues { .. } |
                NodeKind::TrapdoorSample { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::PreimageSample { .. } => {}
                other => return Err(unsupported(other)),
            }
            // Family inputs are collections of existing owners or lazy member
            // descriptors. Merely binding the family imports no members; the
            // selecting node accounts for each materialized member below.
            if matches!(kind, NodeKind::Input { .. }) &&
                output_types
                    .iter()
                    .all(|ty| matches!(ty, ConcreteWireType::IndexedFamily { .. }))
            {
                continue;
            }
            // Retained outputs.
            for (port, output) in output_types
                .iter()
                .enumerate()
                .flat_map(|(port, ty)| family_leaves(ty).map(move |leaf| (port, leaf)))
                .filter(|_| !optimizer.omitted.contains(&id))
                .filter(|(port, _)| {
                    !column_layouts
                        .get(&WireRef { node: id, port: Port(*port as u32) })
                        .is_some_and(|layout| layout.lazy)
                })
            {
                until.set(retained_until(port));
                match output {
                    ConcreteWireType::Matrix(ty) => {
                        for &columns in &column_layouts
                            [&WireRef { node: id, port: Port(port as u32) }]
                            .capacities
                        {
                            add_matrix(&mut demand, ty, ty.rows, columns);
                        }
                    }
                    ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                        let (ty, bound) =
                            bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
                        let params = parameters(self, ty)?;
                        for &columns in &column_layouts
                            [&WireRef { node: id, port: Port(port as u32) }]
                            .capacities
                        {
                            let layout = compact_payload(&params, ty.rows, columns, &bound)?;
                            add_layouts(&mut demand, ty, vec![layout]);
                        }
                    }
                    ConcreteWireType::Bytes { .. } |
                    ConcreteWireType::Trapdoor { .. } |
                    ConcreteWireType::Int |
                    ConcreteWireType::Real |
                    ConcreteWireType::Bool => {}
                    // Scalar value families (polynomial readback) hold no owners.
                    ConcreteWireType::IndexedFamily { element, .. }
                        if matches!(**element, ConcreteWireType::Int) => {}
                    _ => return Err(unsupported(kind)),
                }
            }
            if let Some(outputs) = optimizer.outputs.get(&id) {
                for (ty, aliases) in outputs {
                    let end = aliases
                        .iter()
                        .map(|wire| {
                            if checked.liveness.retained.contains(wire) {
                                usize::MAX
                            } else {
                                owner_until.get(wire).copied().unwrap_or(node_position)
                            }
                        })
                        .max()
                        .unwrap_or(node_position);
                    until.set(end);
                    for &columns in &column_layouts[&WireRef { node: id, port: Port(0) }].capacities
                    {
                        add_matrix(&mut demand, ty, ty.rows, columns);
                    }
                }
            }
            until.set(node_position);
            // Replication may need a full owner; normalization preserves every
            // input fragment and keeps all normalized fragments live together.
            for (wire, argument) in arguments.iter().zip(&argument_types) {
                if column_layouts.get(wire).is_some_and(|layout| layout.lazy) {
                    continue;
                }
                if let ConcreteWireType::Matrix(ty) = argument {
                    add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    if let Some(layout) =
                        column_layouts.get(wire).filter(|layout| layout.capacities.len() > 1)
                    {
                        for &columns in &layout.capacities {
                            add_matrix(&mut demand, ty, ty.rows, columns);
                        }
                    }
                }
            }
            // Kind-specific native scratch at the candidate range width.
            match kind {
                NodeKind::PolynomialFromValues { .. } | NodeKind::ConstantMatrix { .. } => {
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        if ty.rows == 1 && ty.columns == 1 {
                            let params = parameters(self, ty)?;
                            let operation = PreparedMatrixOperation::Polynomial {
                                ty: ty.clone(),
                                coefficients: Vec::new(),
                                evaluation: false,
                            };
                            add_layouts(
                                &mut demand,
                                ty,
                                operation.fixed_workspaces(&params, params.crt_depth() - 1)?,
                            );
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
                        .evaluate(bindings)
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
                    add_matrix(&mut demand, ty, source_rows, ty.columns.min(scratch_columns));
                    add_matrix(&mut demand, ty, source_rows, ty.columns.min(scratch_columns));
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
                        .small_rhs_workspaces(
                            params.crt_depth() - 1,
                            inner,
                            ty.columns.min(scratch_columns),
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    for _ in 0..optimizer.outputs.get(&id).map_or(1, Vec::len) {
                        add_layouts(&mut demand, ty, workspaces.clone());
                    }
                }
                NodeKind::PolynomialValues { evaluation } => {
                    let Some(ConcreteWireType::Matrix(ty)) = argument_types.first() else {
                        return Err(unsupported(kind));
                    };
                    let claims = self.trace_polynomial_values(ty, *evaluation, warm_up)?;
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
                        .evaluate_f64(bindings)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    // Prepared trapdoor/preimage sampling is single-device; refuse
                    // here, before any domain is sealed, rather than at the node.
                    if self.devices.len() != 1 {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    until.set(retained_until(0).max(retained_until(1)));
                    let claims = self.trace_trapdoor_sampling(
                        matrix,
                        sigma,
                        gadget_base,
                        *digit_count,
                        warm_up,
                    )?;
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
                        .evaluate_f64(bindings)
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
                        warm_up,
                    )?;
                    let params = parameters(self, ty)?;
                    // The hard-cutoff plan is owned by the compact output,
                    // including its acceptance word. It outlives this node.
                    until.set(retained_until(0));
                    push_traced(
                        &mut demand,
                        ty,
                        &params,
                        &plan.destination,
                        &add_matrix,
                        &add_layouts,
                    );
                    until.set(node_position);
                    let mut claims = plan.trapdoor.clone();
                    let width = ty.columns.min(scratch_columns);
                    claims.extend(
                        plan.tile_claims(&params, public.rows, width)
                            .map_err(PolyBackendError::GpuSubmission)?,
                    );
                    claims.extend(
                        plan.attempt_claims(&params, public.rows, width, ty.rows, &bound)
                            .map_err(PolyBackendError::GpuSubmission)?,
                    );
                    push_traced(&mut demand, ty, &params, &claims, &add_matrix, &add_layouts);
                }
                NodeKind::Concat { axis: ConcatAxis::Rows } => {
                    // Fused row sums reduce large groups through one-row
                    // intermediates; provision one per concatenated block.
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        for _ in &argument_types {
                            add_matrix(&mut demand, ty, 1, ty.columns.min(scratch_columns));
                        }
                    }
                }
                NodeKind::Tensor => {
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        add_matrix(&mut demand, ty, 1, ty.columns.min(scratch_columns));
                    }
                }
                _ => {}
            }
        }
        position.set(checked.execution_order.len());
        until.set(position.get());
        // Exports: the codec clones the owner, opens its private stream and
        // uses the store workspace; compact exports record one completion.
        for output in scope.outputs() {
            let mut output_type = &checked.wire_types[output];
            while let ConcreteWireType::IndexedFamily { element, .. } = output_type {
                output_type = element;
            }
            if *scope_id == FrozenGraphScopeId::Root &&
                column_layouts.get(output).is_some_and(|layout| layout.lazy)
            {
                // materialize_output recursively loads the entire requested
                // family. Every imported member remains owned by the result;
                // only its transfer scratch can be reused between members.
                until.set(usize::MAX);
                for leaf in family_leaves(&checked.wire_types[output]) {
                    if let ConcreteWireType::Matrix(ty) = leaf {
                        add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    } else if let Some((ty, bound)) = bound_of(leaf) {
                        let params = parameters(self, ty)?;
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                        );
                    }
                }
                until.set(position.get());
                if let Some(ty) = output_type.matrix_type() {
                    let params = parameters(self, ty)?;
                    let (matrices, layouts) =
                        import_inventory(&params, output_type, scratch_columns)?;
                    for (rows, columns) in matrices {
                        add_matrix(&mut demand, ty, rows, columns);
                    }
                    add_layouts(&mut demand, ty, layouts);
                }
            }
            match output_type {
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
                    // Family staging pipelines at most two full shard snapshots.
                    let transfer = params
                        .rns_transfer_workspace(params.crt_depth() - 1, ty.rows, ty.columns)
                        .map_err(PolyBackendError::GpuSubmission)?;
                    for _ in 0..2 {
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![
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
                }
                other if bound_of(other).is_some() => {
                    let (ty, _) = bound_of(other).unwrap();
                    add_layouts(&mut demand, ty, vec![event()]);
                }
                _ => {}
            }
        }
        let outputs = scope
            .outputs()
            .iter()
            .map(|wire| column_layouts.remove(wire).unwrap_or_default())
            .collect();
        Ok((demand, outputs))
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
    use mxx_ir_core::{ParamEnv, graph::FrozenGraphScopeId, types::ConcreteMatrixType};
    use mxx_primitives::{
        matrix::{
            PolyMatrix,
            gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuPreparedSlotKind},
        },
        poly::{
            Poly as _, PolyParams,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use std::collections::BTreeMap;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_mixed_family_layouts_survive_selection_and_child_calls() {
        use super::super::{GpuColumnShard, GpuFleetMatrix};
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let fragments = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(9)
            .max(9);
        let columns = fragments * 4;
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let evaluation = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let host_bytes = std::sync::Arc::new(evaluation.clone().into_cpu_staging_bytes());
        let expected = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&original + &original))
            .to_compact_bytes();
        let members = [4usize, 1]
            .into_iter()
            .map(|width| {
                let shards = (0..columns / width)
                    .into_par_iter()
                    .map(|index| {
                        let start = index * width;
                        let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params,
                            &original.slice_columns(start, start + width),
                        );
                        value.intt_all_in_place();
                        GpuColumnShard { device_id: device, global_column_start: start, value }
                    })
                    .collect();
                RuntimeValue::matrix(GpuFleetMatrix::new(2, columns, shards))
            })
            .chain(std::iter::once(RuntimeValue::HostMatrix {
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    rows: 2,
                    columns,
                    ring_dimension: n as usize,
                    modulus: params.modulus().as_ref().clone().into(),
                },
                bytes: host_bytes,
            }))
            .collect::<Vec<_>>();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for mode in ["direct", "child", "zero"] {
            let context = DslContext::new("mixed-family-layout-inventory");
            let choice = context.int_family_input("choice", 1).at(0);
            let family = ring.input_family("members", members.len(), (2, columns));
            let selected = family.at(choice);
            let left = ring.input("a", (2, columns));
            let output = match mode {
                "child" => mxx_dsl::iterate(1, selected, |_, value| Ok(left + value)).unwrap(),
                "zero" => {
                    let coarse = ring.input("coarse", (2, columns));
                    let fine = ring.input("fine", (2, columns));
                    left + mxx_dsl::iterate(0, coarse, |_, value| Ok(value + fine)).unwrap()
                }
                _ => left + selected,
            };
            let graph = context
                .output("sum", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            for (choice, width) in [4usize, 1, columns]
                .into_iter()
                .enumerate()
                .take(if mode == "zero" { 1 } else { 3 })
            {
                let mut store = MemoryArtifactStore::default();
                let mut inputs =
                    BTreeMap::from([("a".into(), RuntimeValue::matrix(evaluation.clone().into()))]);
                if mode == "zero" {
                    inputs.insert("coarse".into(), members[0].clone());
                    inputs.insert("fine".into(), members[1].clone());
                } else {
                    inputs.insert("members".into(), RuntimeValue::IndexedFamily(members.clone()));
                    inputs.insert(
                        "choice".into(),
                        RuntimeValue::IndexedFamily(vec![RuntimeValue::Int(choice.into())]),
                    );
                }
                let mut result = execute_with_config(
                    &graph,
                    &mut backend,
                    inputs,
                    &mut store,
                    SamplingMode::Fresh,
                    ExecutionConfig::default(),
                )
                .unwrap();
                let RuntimeValue::Matrix(output) =
                    result.materialize_output("sum", &mut backend, &mut store).unwrap()
                else {
                    panic!("matrix output")
                };
                assert_eq!(
                    output.shards().len(),
                    columns / width,
                    "capacity summaries must not change execution partitions"
                );
                assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_fused_blocks_retain_family_aliases() {
        use mxx_dsl::{Family, Mat};
        use mxx_ir_core::node::{ConcatAxis, IndexRange};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
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
        let input =
            Mat::concat(ConcatAxis::Rows, vec![ring.input("a", (1, 3)), ring.input("b", (1, 3))]);
        let left = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("x", (1, 2 * digits)), ring.input("y", (1, 2 * digits))],
        );
        let product = input.decompose(16, digits).mul_small_rhs(left);
        let blocks = Family::pack(
            (0..2usize)
                .map(|row| {
                    product
                        .clone()
                        .slice(Some(IndexRange { start: row.into(), end: (row + 1).into() }), None)
                })
                .collect(),
        )
        .unwrap();
        let graph = DslContext::new("fused-block-family-lifetimes")
            .output("blocks", blocks)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let plan = crate::executor::gpu_plan::inventory_plan(&graph, false);
        assert_eq!(plan.outputs.len(), 1, "fixture must select the fused product");
        assert_eq!(plan.outputs.values().next().unwrap().len(), 2);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let (demand, _) = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &ParamEnv::default(),
                false,
                None,
                3,
                true,
            )
            .unwrap();
        let retained_blocks = demand
            .values()
            .flat_map(|(_, demand)| demand.matrices.iter().zip(&demand.matrix_until))
            .filter(|(shape, until)| **shape == (1, 3) && **until == usize::MAX)
            .count();
        assert_eq!(
            retained_blocks, 2,
            "packing aliases into a returned family must retain both fused owners"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_family_demand_depends_on_selected_members() {
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = mxx_ir_core::types::ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut inventories = Vec::new();
        for count in [2usize, 65_536] {
            let manifest = Manifest {
                ir_version: mxx_ir_core::encoding::IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([(
                    "members".into(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(ty.clone()),
                        family_count: Some(count),
                        confidentiality: ArtifactConfidentiality::Private,
                        content_hash: None,
                        layout: None,
                    },
                )]),
            };
            let members = ring.family_artifact_input(
                production.clone(),
                "members",
                count,
                (2, 3),
                ArtifactConfidentiality::Private,
            );
            let graph = DslContext::new("lazy-family-inventory")
                .output("selected", -members.at(0))
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest)]),
                )
                .unwrap();
            let (demand, _) = backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &ParamEnv::default(),
                    false,
                    None,
                    3,
                    true,
                )
                .unwrap();
            inventories.push(
                demand
                    .into_values()
                    .map(|(_, demand)| (demand.matrices, demand.layouts))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(inventories[0], inventories[1]);
        assert!(!inventories[0][0].0.is_empty(), "selected member still needs native owners");
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_scalar_packs_match_lazy_families() {
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(128)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for (compact, select_candidates) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut inventories = Vec::new();
            for count in [2, count] {
                for packed in [false, true] {
                    let artifact_type = if compact {
                        ArtifactType::Preimage {
                            matrix: ty.clone(),
                            max_coefficient_bound: 7.into(),
                        }
                    } else {
                        ArtifactType::Matrix(ty.clone())
                    };
                    let names = if packed {
                        (0..count).map(|index| format!("member-{index}")).collect::<Vec<_>>()
                    } else {
                        vec!["members".into()]
                    };
                    let manifest = Manifest {
                        ir_version: mxx_ir_core::encoding::IR_VERSION,
                        production_id: production.clone(),
                        artifacts: names
                            .iter()
                            .map(|name| {
                                (
                                    name.clone(),
                                    ManifestArtifact {
                                        artifact_type: artifact_type.clone(),
                                        family_count: (!packed).then_some(count),
                                        confidentiality: ArtifactConfidentiality::Private,
                                        content_hash: None,
                                        layout: None,
                                    },
                                )
                            })
                            .collect(),
                    };
                    let result = if compact {
                        let members = if packed {
                            mxx_dsl::Family::pack(
                                names
                                    .iter()
                                    .map(|name| {
                                        ring.preimage_artifact_input(
                                            production.clone(),
                                            name.clone(),
                                            (2, 3),
                                            7,
                                            ArtifactConfidentiality::Private,
                                        )
                                    })
                                    .collect(),
                            )
                            .unwrap()
                        } else {
                            ring.preimage_family_artifact_input(
                                production.clone(),
                                "members",
                                count,
                                (2, 3),
                                7,
                                ArtifactConfidentiality::Private,
                            )
                        };
                        let selected = if select_candidates {
                            mxx_dsl::select(0, (0..count).map(|index| members.at(index)).collect())
                                .unwrap()
                        } else {
                            members.at(0)
                        };
                        selected.mul_small_rhs(ring.input("left", (1, 2)))
                    } else {
                        let members = if packed {
                            mxx_dsl::Family::pack(
                                names
                                    .iter()
                                    .map(|name| {
                                        ring.artifact_input(
                                            production.clone(),
                                            name.clone(),
                                            (2, 3),
                                            ArtifactConfidentiality::Private,
                                        )
                                    })
                                    .collect(),
                            )
                            .unwrap()
                        } else {
                            ring.family_artifact_input(
                                production.clone(),
                                "members",
                                count,
                                (2, 3),
                                ArtifactConfidentiality::Private,
                            )
                        };
                        let selected = if select_candidates {
                            mxx_dsl::select(0, (0..count).map(|index| members.at(index)).collect())
                                .unwrap()
                        } else {
                            members.at(0)
                        };
                        -selected
                    };
                    let graph = DslContext::new("scalar-artifact-pack-inventory")
                        .output("result", result)
                        .unwrap()
                        .build()
                        .unwrap()
                        .validate_with_manifests(
                            &ParamEnv::default(),
                            &BTreeMap::from([(production.clone(), manifest)]),
                        )
                        .unwrap();
                    let (demand, _) = backend
                        .graph_admission_demand(
                            &graph,
                            &FrozenGraphScopeId::Root,
                            &ParamEnv::default(),
                            false,
                            None,
                            3,
                            true,
                        )
                        .unwrap();
                    inventories.push(
                        demand
                            .into_values()
                            .map(|(_, demand)| (demand.matrices, demand.layouts))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            if !compact {
                // A mixed packed family is eagerly placed when broadcast,
                // including lazy members not selected by the child body.
                let names = (0..count).map(|index| format!("mixed-{index}")).collect::<Vec<_>>();
                let manifest = Manifest {
                    ir_version: mxx_ir_core::encoding::IR_VERSION,
                    production_id: production.clone(),
                    artifacts: names
                        .iter()
                        .map(|name| {
                            (
                                name.clone(),
                                ManifestArtifact {
                                    artifact_type: ArtifactType::Matrix(ty.clone()),
                                    family_count: None,
                                    confidentiality: ArtifactConfidentiality::Private,
                                    content_hash: None,
                                    layout: None,
                                },
                            )
                        })
                        .collect(),
                };
                let members = mxx_dsl::Family::pack(
                    std::iter::once(ring.input("eager", (2, 3)))
                        .chain(names.iter().map(|name| {
                            ring.artifact_input(
                                production.clone(),
                                name.clone(),
                                (2, 3),
                                ArtifactConfidentiality::Private,
                            )
                        }))
                        .collect(),
                )
                .unwrap();
                // Direct and offset indices lower to Zip/ZipOffset. A runtime
                // remainder keeps the whole family as a broadcast input.
                let result = mxx_dsl::parallel(2, |index| Ok(-members.at(index % 2))).unwrap();
                let graph = DslContext::new("mixed-family-broadcast")
                    .output("result", result)
                    .unwrap()
                    .build()
                    .unwrap()
                    .validate_with_manifests(
                        &ParamEnv::default(),
                        &BTreeMap::from([(production.clone(), manifest)]),
                    )
                    .unwrap();
                assert!(graph.root_scope().execution_order.iter().any(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(body)
                        if body.input_modes.contains(&mxx_ir_core::node::LoopInputMode::Broadcast))
                }));
                let (demand, _) = backend
                    .graph_admission_demand(
                        &graph,
                        &FrozenGraphScopeId::Root,
                        &ParamEnv::default(),
                        false,
                        None,
                        3,
                        true,
                    )
                    .unwrap();
                let owners = demand
                    .values()
                    .flat_map(|(_, demand)| &demand.matrices)
                    .filter(|&&(rows, columns)| rows >= 2 && columns >= 3)
                    .count();
                assert!(
                    owners >= count,
                    "broadcast placement must reserve every lazy member of a mixed packed family: {owners} < {count}"
                );
            }
            for inventory in &inventories[1..] {
                assert_eq!(
                    &inventories[0], inventory,
                    "lazy scalar packs must reserve only selected members; compact={compact}"
                );
            }
            assert!(
                inventories[0].iter().any(|(_, layouts)| layouts
                    .iter()
                    .any(|layout| layout.kind == GpuPreparedSlotKind::PinnedHost)),
                "materialization still requires native import staging"
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_input_cached_across_intervening_operations() {
        use crate::artifact::{ArtifactKey, ArtifactPayload, ArtifactStore};
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let bytes = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).to_compact_bytes();
        let mut store = MemoryArtifactStore::default();
        store
            .store(
                ArtifactKey { production: production.clone(), name: "input".into(), index: None },
                &ArtifactType::Matrix(ty.clone()),
                ArtifactConfidentiality::Private,
                None,
                ArtifactPayload::Matrix(bytes.clone()),
            )
            .unwrap();
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "input".into(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Matrix(ty),
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        store.store_manifest(manifest.clone()).unwrap();
        let input = ring.artifact_input(
            production.clone(),
            "input",
            (2, 3),
            ArtifactConfidentiality::Private,
        );
        let mut intermediate = -mxx_dsl::select(0, vec![input.clone(), input.clone()]).unwrap();
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(8)
            .max(2);
        for _ in 0..count {
            intermediate = -(-intermediate);
        }
        // The second access must read the cached imported owner after temporary
        // slots have been reused repeatedly. Trusted arithmetic predicts input.
        let output = (intermediate + input.clone()) + input;
        let graph = DslContext::new("cached-lazy-input-lifetime")
            .output("result", output)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        drop(backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), true).unwrap());
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), bytes);
        drop(result);
        drop(backend);

        // Returned lazy descriptors must also be importable when the caller
        // requests the whole output family after graph execution.
        let input = ring.artifact_input(
            production.clone(),
            "input",
            (2, 3),
            ArtifactConfidentiality::Private,
        );
        let family = mxx_dsl::Family::pack(vec![input; count]).unwrap();
        let graph = DslContext::new("retained-lazy-output-family")
            .output("result", family)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        drop(backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), true).unwrap());
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::IndexedFamily(members) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("family result")
        };
        assert_eq!(members.len(), count);
        for member in members {
            let RuntimeValue::Matrix(matrix) = member else { panic!("matrix member") };
            assert_eq!(backend.matrix_to_bytes(matrix).unwrap(), bytes);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_zero_iteration_outputs_keep_carried_owners() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected = [&original, &(-original.clone())]
            .map(|value| GpuDCRTPolyMatrix::from_cpu_matrix(&params, value).to_compact_bytes());
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(9)
            .max(5);
        let mut context = DslContext::new("zero-iteration-alias-inventory");
        let mut value = ring.input("a", (2, 3));
        for index in 0..count {
            value = -value;
            let alias = mxx_dsl::iterate(0, value.clone(), |_, carried| Ok(-carried)).unwrap();
            context = context.output(format!("value-{index:04}"), alias).unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(input.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        for index in 0..count {
            let RuntimeValue::Matrix(output) = result
                .materialize_output(&format!("value-{index:04}"), &mut backend, &mut store)
                .unwrap()
            else {
                panic!("matrix output")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected[(index + 1) % 2]);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_inventory_preserves_fragmented_coefficient_inputs() {
        use super::super::{GpuColumnShard, GpuFleetMatrix};
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(17)
            .max(9);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let graph = DslContext::new("fragmented-input-inventory")
            .output("result", ring.input("a", (2, columns)) + ring.input("b", (2, columns)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&input + &input)).to_compact_bytes();
        let evaluation = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let shards = (0..columns)
            .into_par_iter()
            .map(|start| {
                let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                    &params,
                    &input.slice_columns(start, start + 1),
                );
                value.intt_all_in_place();
                GpuColumnShard { device_id: device, global_column_start: start, value }
            })
            .collect();
        let input = GpuFleetMatrix::new(2, columns, shards);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([
                ("a".into(), RuntimeValue::matrix(evaluation.into())),
                ("b".into(), RuntimeValue::matrix(input.clone())),
            ]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result")
        };
        assert_eq!(output.shards().len(), columns);
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
        assert!(input.shards().iter().all(|shard| !shard.value.is_ntt()));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_loop_inventory_scales_with_live_wave_not_iteration_count() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(-input.clone())).to_compact_bytes();
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut capacities = Vec::new();
        let mut diagnostic_plan_counts = Vec::new();
        for count in [2usize, 65] {
            let matrix = ring.input("matrix", (2, 3));
            let outputs = mxx_dsl::parallel(count, |_| Ok(-matrix.clone())).unwrap();
            let graph = DslContext::new("bounded-loop-inventory")
                .output("result", outputs.at(count - 1))
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let inputs =
                BTreeMap::from([("matrix".into(), RuntimeValue::matrix(input.clone().into()))]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(output) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            diagnostic_plan_counts.push(backend.admitted_plan_log().len());
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
        }
        assert_eq!(
            capacities[0], capacities[1],
            "loop count must not multiply GPU scratch backing"
        );
        assert!(diagnostic_plan_counts[0] > 0);
        assert_eq!(
            diagnostic_plan_counts[0], diagnostic_plan_counts[1],
            "repeated loop members must not accumulate identical diagnostic plans"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_polynomial_values_preserve_both_domains() {
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
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let expected =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let expected_bytes =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &expected).to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for evaluation in [false, true] {
            let context = DslContext::new("admitted-polynomial-values");
            let values = context.int_family_input("values", n as usize);
            let output = if evaluation {
                ring.from_evaluations(&values)
            } else {
                ring.from_coefficients(&values)
            };
            let graph = context
                .output("result", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let polynomial = expected.entry(0, 0);
            let values =
                if evaluation { polynomial.evals_biguints() } else { polynomial.coeffs_biguints() };
            let inputs = BTreeMap::from([(
                "values".into(),
                RuntimeValue::IndexedFamily(
                    values.into_iter().map(|value| RuntimeValue::Int(value.into())).collect(),
                ),
            )]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(output) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected_bytes);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_range_scratch_retains_complete_outputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns >= 3);
        let width = columns.div_ceil(3);
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
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let digits = params.modulus_digits();
        let value = ring.input("a", (2, columns));
        let result = value.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let graph = DslContext::new("prepared-range-scratch")
            .output("result", result)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let expected = matrix.to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        // Exercise the same demand/provisioning passes with a deliberately
        // smaller scratch envelope. This avoids allocating a VRAM-sized test.
        let demand = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &graph.bindings,
                false,
                None,
                width,
                true,
            )
            .unwrap()
            .0;
        backend.prepare_graph_storage(demand).unwrap();
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        let product = backend
            .admitted_plan_log()
            .iter()
            .map(|(_, invocation)| invocation)
            .find(|invocation| invocation.operation == "MultiplyCompact")
            .unwrap();
        assert_eq!(product.plan.columns, columns);
        assert_eq!(product.plan.widths, vec![width]);
        assert_eq!(product.plan.wave_count, columns.div_ceil(width));
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_measured_ranges_reuse_representatives_and_measure_the_tail() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns >= 3);
        let width = columns.div_ceil(3);
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
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let digits = params.modulus_digits();
        let value = ring.input("a", (2, columns));
        let result = value.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let graph = DslContext::new("prepared-range-scratch")
            .output("result", result)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let expected = matrix.to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        // Exercise the same demand/provisioning passes with a deliberately
        // smaller scratch envelope. This avoids allocating a VRAM-sized test.
        let demand = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &graph.bindings,
                false,
                None,
                width,
                true,
            )
            .unwrap()
            .0;
        backend.prepare_graph_storage(demand).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
            sender.send(event).unwrap();
        })));
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        backend.set_admitted_measurement_sink(None);
        let mut measured = Vec::new();
        let mut invocation = Vec::new();
        for event in receiver {
            match event {
                crate::gpu_measurement::GpuAdmittedMeasurement::Invocation { .. } => {
                    if !invocation.is_empty() {
                        measured.push(std::mem::take(&mut invocation));
                    }
                }
                crate::gpu_measurement::GpuAdmittedMeasurement::Wave { jobs, measured, .. } => {
                    invocation.push((jobs, measured));
                }
                _ => {}
            }
        }
        if !invocation.is_empty() {
            measured.push(invocation);
        }
        let product = backend
            .admitted_plan_log()
            .iter()
            .map(|(_, invocation)| invocation)
            .find(|invocation| invocation.operation == "MultiplyCompact")
            .unwrap();
        assert_eq!(product.plan.columns, columns);
        assert_eq!(product.plan.widths, vec![width]);
        assert_eq!(product.plan.wave_count, columns.div_ceil(width));
        let repeated = measured
            .iter()
            .filter(|waves| waves.len() == columns.div_ceil(width))
            .collect::<Vec<_>>();
        assert!(!repeated.is_empty());
        for waves in repeated {
            let mut classes = std::collections::BTreeSet::new();
            for (jobs, measured) in waves {
                let shape = jobs
                    .iter()
                    .map(|job| (job.device, job.source_interval, job.end - job.start))
                    .collect::<Vec<_>>();
                assert_eq!(
                    *measured,
                    classes.insert(shape),
                    "only the first wave of each range/width class is sampled"
                );
            }
            if columns > width * 2 {
                assert!(waves.iter().any(|(_, measured)| !measured));
            }
            assert!(
                waves.last().unwrap().1 || columns % width == 0,
                "a new tail width needs its own measurement"
            );
        }
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_inventory_reuses_dead_values_in_long_chains() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 5, DistType::FinRingDist);
        let mut capacities = Vec::new();
        for length in [8, 128] {
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
            let mut value = ring.input("a", (2, 5));
            for _ in 0..length {
                value = -value;
            }
            let graph = DslContext::new("prepared-live-chain")
                .output("value", value)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
            let expected = matrix.to_compact_bytes();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
            )
            .unwrap();
            let RuntimeValue::Matrix(output) =
                result.materialize_output("value", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result");
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
        }
        assert_eq!(
            capacities[0], capacities[1],
            "sequential depth must not increase prepared backing"
        );
    }

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
        let error = backend
            .prepare_graph_admission(&graph, false, &inputs, false)
            .err()
            .expect("polynomial readback requires explicit resource warmup");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.polynomial_value_plans.is_empty());
        drop(backend.prepare_graph_admission(&graph, false, &inputs, true).unwrap());
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
        run_preimage_graph(None);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preimage_range_scratch_handles_tail() {
        run_preimage_graph(Some(2));
    }

    fn run_preimage_graph(scratch_columns: Option<usize>) {
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
        let error = execute_with_config(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .err()
        .expect("production cannot discover missing resource plans");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.trapdoor_plans.is_empty());
        assert!(backend.preimage_plans.is_empty());
        if let Some(width) = scratch_columns {
            // Warm discovery plans without installing wider automatic storage:
            // this fixture supplies its own width-limited inventory below.
            backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &graph.bindings,
                    false,
                    None,
                    width,
                    true,
                )
                .unwrap();
        } else {
            drop(backend.prepare_graph_admission(&graph, false, &inputs, true).unwrap());
        }
        let warmed = (backend.trapdoor_plans.len(), backend.preimage_plans.len());
        assert!(warmed.0 > 0 && warmed.1 > 0);
        let saved = std::mem::take(&mut backend.preimage_plans);
        let error = backend
            .prepare_graph_admission(&graph, false, &inputs, false)
            .err()
            .expect("lost warmup plan must not trigger a replacement trial");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.preimage_plans.is_empty());
        backend.preimage_plans = saved;
        if let Some(width) = scratch_columns {
            let demand = backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &graph.bindings,
                    false,
                    None,
                    width,
                    true,
                )
                .unwrap()
                .0;
            backend.prepare_graph_storage(demand).unwrap();
        }
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
        assert_eq!((backend.trapdoor_plans.len(), backend.preimage_plans.len()), warmed);
        assert!(backend.prepared_ledger.is_some());
        if let Some(width) = scratch_columns {
            let preimage = backend
                .admitted_plan_log()
                .iter()
                .map(|(_, invocation)| invocation)
                .find(|invocation| invocation.operation == "Preimage")
                .unwrap();
            assert_eq!(preimage.plan.widths, vec![width]);
            assert_eq!(preimage.plan.wave_count, 3usize.div_ceil(width));
        }

        let RuntimeValue::Matrix(output) =
            result.materialize_output("check", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output")
        };
        // A · G^{-1}-free trapdoor preimage x satisfies A x = t exactly.
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }
}
