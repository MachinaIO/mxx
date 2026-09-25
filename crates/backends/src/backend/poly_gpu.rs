use crate::{
    backend::RuntimeValue,
    gpu_execution_plan::{GpuPlanContract, PhysicalEncoding},
};
use mxx_ir_core::{
    ValidatedGraph,
    encoding::{hash_canonical, spec_hash},
    node::NodeKind,
    types::{ConcreteMatrixType, NodeId, Port, WireRef},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};

mod fleet;
pub use fleet::GpuDcrtBackend;
pub(crate) use fleet::{
    GpuPreparedNativeResources, PhysicalExport, emit_compiled_gpu_op,
    emit_compiled_monomial_difference, emit_compiled_small_rhs_sum, emit_compiled_subgraph_kernel,
    physical_raw_matrix_view, prepare_compiled_gpu_program, transcode_raw_artifact,
};

impl GpuDcrtBackend {
    /// Freeze the actual device and input-shape contract for a direct physical
    /// plan. Pointer addresses and scalar contents are deliberately omitted.
    pub(crate) fn physical_plan_contract(
        &self,
        validated: &ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<GpuPlanContract, String> {
        let mut shape_descriptor = Vec::new();
        let mut expected_names = BTreeSet::new();
        for (position, handle) in validated.root_scope().execution_order.iter().enumerate() {
            let NodeKind::Input { name, artifact, .. } = handle.kind() else {
                continue;
            };
            expected_names.insert(name.as_str());
            let wire = WireRef { node: NodeId(position as u64), port: Port(0) };
            let ty = validated
                .root_scope()
                .wire_types
                .get(&wire)
                .ok_or_else(|| format!("input {name} has no validated wire type"))?;
            match inputs.get(name) {
                Some(value) if value.matches_wire_type(ty) => {
                    shape_descriptor.push((name.clone(), ty.clone(), physical_input_shape(value)));
                }
                Some(_) => return Err(format!("GPU input {name} differs from its validated type")),
                None if artifact.is_some() => {
                    shape_descriptor.push((
                        name.clone(),
                        ty.clone(),
                        "artifact-on-demand".to_owned(),
                    ));
                }
                None => return Err(format!("missing GPU input {name}")),
            }
        }
        if let Some(extra) = inputs.keys().find(|name| !expected_names.contains(name.as_str())) {
            return Err(format!("unexpected GPU input {extra}"));
        }
        let graph_specification_hash =
            spec_hash(&validated.source, &validated.bindings).map_err(|error| error.to_string())?.0;
        let shape_contract_hash =
            hash_canonical(&shape_descriptor).map_err(|error| error.to_string())?;
        let logical_to_physical_devices = self
            .physical_device_ids()
            .into_iter()
            .map(|device| {
                usize::try_from(device).map_err(|_| "negative physical GPU id".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let contract = GpuPlanContract {
            graph_specification_hash,
            backend_identity: self.runtime_backend_identity().map_err(|error| error.to_string())?,
            logical_to_physical_devices,
            device_budgets: self.runtime_device_budgets().map_err(|error| error.to_string())?,
            shape_contract_hash,
            backend_revision: env!("CARGO_PKG_VERSION").to_owned(),
        };
        contract.validate().map_err(|error| error.to_string())?;
        Ok(contract)
    }

    pub(crate) fn physical_matrix_parameters(
        &self,
        ty: &ConcreteMatrixType,
        physical_device: i32,
    ) -> Result<GpuDCRTPolyParams, String> {
        self.parameters_on_physical_device(physical_device, ty)
            .cloned()
            .map_err(|error| error.to_string())
    }

    /// Select a registered device context for scalar-only control values,
    /// which have no matrix ring from which to choose parameters.
    pub(crate) fn control_parameters_on_device(
        &self,
        physical_device: i32,
    ) -> Result<GpuDCRTPolyParams, String> {
        self.parameters_on_device(physical_device).cloned()
    }

    /// Allocate the planned full-matrix representation on its selected device.
    /// The physical descriptor remains authoritative for type and encoding;
    /// callers bind the returned owner through `BoundStorage::from_matrix_data`.
    pub(crate) fn allocate_physical_matrix(
        &self,
        ty: &ConcreteMatrixType,
        physical_device: i32,
        encoding: PhysicalEncoding,
    ) -> Result<Arc<GpuDCRTPolyMatrix>, String> {
        if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) {
            return Err("physical matrix allocation requires a full-matrix encoding".into());
        }
        let parameters = self.physical_matrix_parameters(ty, physical_device)?;
        let owner = GpuDCRTPolyMatrix::zero_with_state(&parameters, ty.rows, ty.columns)
            .map_err(|error| error.to_string())?;
        Ok(Arc::new(owner))
    }

    /// Fill one already planned import owner from the selected canonical
    /// artifact payload. The caller reaches this only at that import's first
    /// consumer, after joining every prior GPU region and artifact reader.
    ///
    /// # Safety
    /// The caller must serialize execution of this plan and ensure no GPU or
    /// I/O operation still reads or writes `owner`. The native method preserves
    /// the allocation address and the validated shape, ring, and domain.
    pub(crate) unsafe fn upload_physical_matrix_import_after_completion(
        &self,
        owner: &Arc<GpuDCRTPolyMatrix>,
        ty: &ConcreteMatrixType,
        canonical_bytes: &[u8],
    ) -> Result<(), String> {
        let components = owner.binding_components().map_err(|error| error.to_string())?;
        let [component] = components.as_ref() else {
            return Err("import destination requires one physical matrix component".into());
        };
        if owner.size() != (ty.rows, ty.columns) ||
            owner.params().ring_dimension() != ty.ring.ring_dimension() ||
            owner.params().moduli() != ty.ring.crt_moduli()
        {
            return Err("import destination owner differs from its concrete matrix type".into());
        }
        self.physical_matrix_parameters(ty, component.physical_device)?;
        unsafe {
            owner.load_compact_bytes_in_place_after_completion(
                canonical_bytes,
                ty.ring.crt_moduli(),
                ty.ring.ring_dimension(),
            )
        }
        .map_err(|error| error.to_string())
    }
}

fn physical_input_shape(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::Matrix(matrix) => match matrix.as_gpu() {
            Some(resident) => format!("gpu:{}", physical_descriptor_shape(resident)),
            None if matrix.as_cpu_full().is_some() => "cpu-full".to_owned(),
            None if matrix.as_cpu_compact().is_some() => "cpu-compact".to_owned(),
            None if matrix.encoded_bytes().is_some() => "encoded-matrix".to_owned(),
            None => "unknown-matrix-storage".to_owned(),
        },
        RuntimeValue::Resident(resident) => {
            format!("resident:{}", physical_descriptor_shape(resident))
        }
        RuntimeValue::IndexedFamily { element_type, values } => format!(
            "host-family:{element_type:?}:{}:{:?}",
            values.len(),
            values.iter().map(physical_input_shape).collect::<Vec<_>>()
        ),
        RuntimeValue::Trapdoor(trapdoor) => {
            format!(
                "host-trapdoor:{:?}:{}",
                trapdoor.wire_type(),
                physical_input_shape(&RuntimeValue::Matrix(trapdoor.public_matrix().clone()))
            )
        }
        RuntimeValue::LazyArtifact { descriptor, .. } |
        RuntimeValue::LazyArtifactFamily { descriptor, .. } |
        RuntimeValue::StagedArtifact { descriptor, .. } |
        RuntimeValue::StagedArtifactFamily { descriptor, .. } => {
            format!(
                "artifact:{:?}:{:?}:{:?}",
                descriptor.artifact_type, descriptor.family_count, descriptor.layout
            )
        }
        RuntimeValue::Composite(leaves) => {
            format!("composite:{:?}", leaves.iter().map(physical_input_shape).collect::<Vec<_>>())
        }
        RuntimeValue::Int(_) => "int".to_owned(),
        RuntimeValue::Real(_) => "real".to_owned(),
        RuntimeValue::Bool(_) => "bool".to_owned(),
        RuntimeValue::Bytes(bytes) => format!("bytes:{}", bytes.len()),
        RuntimeValue::TypedBlob { type_name, schema_hash, .. } => {
            format!("blob:{type_name}:{schema_hash:?}")
        }
    }
}

fn physical_descriptor_shape(resident: &super::GpuResidentValue) -> String {
    let physical = resident.physical();
    let mut canonical_storage = BTreeMap::new();
    let parts = physical
        .parts
        .iter()
        .map(|part| {
            let next = canonical_storage.len();
            let storage = *canonical_storage.entry(part.storage).or_insert(next);
            let allocation_bytes = resident.storage(part.storage).map(|bound| bound.bytes);
            (part.leaf, storage, part.device, allocation_bytes, &part.view)
        })
        .collect::<Vec<_>>();
    format!("{:?}:{:?}:{:?}:{parts:?}", physical.ty, physical.encodings, physical.integer_ranges)
}

pub fn gpu_backend(parameters: impl IntoIterator<Item = GpuDCRTPolyParams>) -> GpuDcrtBackend {
    let parameters = parameters.into_iter().collect::<Vec<_>>();
    let device_ids = detected_gpu_device_ids();
    gpu_backend_on(parameters, device_ids)
}

/// Builds a GPU backend restricted to the requested detected devices.
pub fn gpu_backend_on(
    parameters: impl IntoIterator<Item = GpuDCRTPolyParams>,
    device_ids: impl IntoIterator<Item = i32>,
) -> GpuDcrtBackend {
    let parameters = parameters.into_iter().collect::<Vec<_>>();
    let mut device_ids = device_ids.into_iter().collect::<Vec<_>>();
    device_ids.sort_unstable();
    assert!(!device_ids.is_empty(), "mxx-backends GPU backend requires at least one detected GPU");
    let placements = device_ids
        .into_iter()
        .map(|device_id| {
            let mut placement = Vec::new();
            for parameters in &parameters {
                let local = parameters.params_for_device(device_id, placement.first());
                placement.push(local);
            }
            placement
        })
        .collect();
    GpuDcrtBackend::new(placements)
}
