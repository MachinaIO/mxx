use super::poly::{CrtRecomposeMatrix, PolyBackendError};
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
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

impl CrtRecomposeMatrix for GpuDCRTPolyMatrix {
    fn crt_recompose_levels(
        levels: &[Self],
        plaintext_moduli: &[num_bigint::BigInt],
        reconstruction_coefficients: &[num_bigint::BigInt],
        destination: &GpuDCRTPolyParams,
    ) -> Result<Self, PolyBackendError> {
        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        if levels.len() != plaintext_moduli.len() ||
            levels.len() != reconstruction_coefficients.len() ||
            levels.iter().any(|level| {
                level.params().ring_dimension() != destination.ring_dimension() ||
                    level.row_size() != 1 ||
                    level.col_size() != first.col_size()
            })
        {
            return Err(PolyBackendError::InvalidInteger);
        }
        let plaintext_moduli = plaintext_moduli
            .iter()
            .map(|modulus| modulus.to_u64().filter(|modulus| *modulus != 0))
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::InvalidInteger)?;
        let ring_moduli = destination.moduli();
        let reconstruction_residues = reconstruction_coefficients
            .iter()
            .flat_map(|coefficient| {
                ring_moduli.iter().map(move |modulus| {
                    let modulus = num_bigint::BigInt::from(*modulus);
                    (((coefficient % &modulus) + &modulus) % &modulus).to_u64()
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::InvalidInteger)?;
        Ok(GpuDCRTPolyMatrix::crt_recompose_levels(
            levels,
            &plaintext_moduli,
            &reconstruction_residues,
            destination,
        ))
    }
}
use crate::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};

mod fleet;
pub use fleet::GpuDcrtBackend;
pub(crate) use fleet::{
    GpuPreparedNativeResources, PhysicalExport, emit_compiled_gpu_op, physical_raw_matrix_view,
    prepare_compiled_gpu_program, transcode_raw_artifact,
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
        let is_ntt = match encoding {
            PhysicalEncoding::FullCoeff => false,
            PhysicalEncoding::FullEval => true,
            _ => return Err("physical matrix allocation requires a full-matrix encoding".into()),
        };
        let parameters = self.physical_matrix_parameters(ty, physical_device)?;
        let owner = GpuDCRTPolyMatrix::zero_with_state(&parameters, ty.rows, ty.columns, is_ntt)
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

#[cfg(test)]
pub(super) fn wait_for_gpu_test_context_quiescence(device: i32) {
    use crate::poly::dcrt::gpu::{gpu_device_memory_usage, gpu_device_sync};
    use std::time::{Duration, Instant};

    // The named serial-test lock protects test bodies, but event-ordered owners
    // from the preceding body can outlive the lock briefly.  Drain completed
    // device work at this test-only boundary and wait for those owners to drop
    // before a calibration test creates its context.
    gpu_device_sync();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let usage = gpu_device_memory_usage(device).expect("query GPU test context state");
        if usage.live_contexts == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "previous GPU test contexts did not quiesce: device={device}, live_contexts={}",
            usage.live_contexts
        );
        std::thread::sleep(Duration::from_millis(1));
    }
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

#[cfg(test)]
mod crt_tests {
    use super::*;
    use crate::{
        matrix::PolyMatrix,
        poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPoly},
    };
    use num_bigint::{BigInt, BigUint, Sign};

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_crt_recompose_matches_direct_residues_five_times() {
        let device = detected_gpu_device_ids()[0];
        wait_for_gpu_test_context_quiescence(device);
        // Fixed primes avoid constructing an OpenFHE parameter object (and its
        // process-global transform/cache state) in this GPU correctness oracle.
        // Each prime is 1 mod 2N for N = 32.
        let moduli = vec![131_009, 130_817, 129_793, 129_281, 128_833];
        let gpu_parameters = GpuDCRTPolyParams::new(32, moduli.clone(), 8, None);
        let q = moduli.iter().map(|modulus| BigUint::from(*modulus)).product::<BigUint>();
        assert!(q.bits() > 64, "test must exercise a multi-word ring modulus");
        let q_minus_one = &q - BigUint::from(1u8);
        let half_q_low = (&q / BigUint::from(2u8)).to_u64_digits()[0];
        let overflowing_plaintext_modulus = (2u64..10_000)
            .find(|modulus| {
                let scaled_low = (&q_minus_one * *modulus).to_u64_digits()[0];
                scaled_low.overflowing_add(half_q_low).1
            })
            .expect("test parameters must produce a low-word carry while adding Q/2");
        let plaintext_moduli = vec![BigInt::from(overflowing_plaintext_modulus), BigInt::from(19)];
        let reconstruction_coefficients = vec![BigInt::from(-23), BigInt::from(29)];
        let level_coefficients = plaintext_moduli
            .iter()
            .enumerate()
            .map(|(level, plaintext_modulus)| {
                let plaintext_modulus = plaintext_modulus.to_biguint().unwrap();
                (0..5)
                    .map(|column| {
                        (0..gpu_parameters.ring_dimension() as usize)
                            .map(|index| {
                                let ordinal = level + column + index;
                                match ordinal % 5 {
                                    0 => BigUint::from(0u8),
                                    1 => &q - BigUint::from(1u8),
                                    2 => &q / BigUint::from(2u8),
                                    3 => {
                                        let bucket = BigUint::from((ordinal % 7) + 1);
                                        (&q * (BigUint::from(2u8) * bucket + 1u8)) /
                                            (BigUint::from(2u8) * &plaintext_modulus)
                                    }
                                    _ => {
                                        (BigUint::from(7919usize * (ordinal + 1)) +
                                            BigUint::from(104729usize * (column + 1))) %
                                            &q
                                    }
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let q_signed = BigInt::from_biguint(Sign::Plus, q.clone());
        let expected_coefficients = (0..5)
            .map(|column| {
                (0..gpu_parameters.ring_dimension() as usize)
                    .map(|index| {
                        let accumulated = level_coefficients
                            .iter()
                            .zip(&plaintext_moduli)
                            .zip(&reconstruction_coefficients)
                            .fold(BigInt::from(0u8), |acc, ((level, modulus), reconstruction)| {
                                let value =
                                    BigInt::from_biguint(Sign::Plus, level[column][index].clone());
                                let rounded =
                                    ((modulus * value + &q_signed / 2u8) / &q_signed) % modulus;
                                acc + rounded * reconstruction
                            });
                        ((accumulated % &q_signed) + &q_signed) % &q_signed
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut expected_rns_bytes = Vec::new();
        for polynomial in &expected_coefficients {
            for modulus in &moduli {
                let modulus = BigInt::from(*modulus);
                for coefficient in polynomial {
                    let residue = ((coefficient % &modulus) + &modulus) % &modulus;
                    expected_rns_bytes.extend_from_slice(
                        &u64::try_from(residue).expect("residue must fit in u64").to_le_bytes(),
                    );
                }
            }
        }

        for _ in 0..5 {
            let gpu_levels = level_coefficients
                .iter()
                .map(|level| {
                    GpuDCRTPolyMatrix::from_poly_vec_row(
                        &gpu_parameters,
                        level
                            .iter()
                            .map(|coefficients| {
                                GpuDCRTPoly::from_biguints(&gpu_parameters, coefficients)
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            let actual = <GpuDCRTPolyMatrix as CrtRecomposeMatrix>::crt_recompose_levels(
                &gpu_levels,
                &plaintext_moduli,
                &reconstruction_coefficients,
                &gpu_parameters,
            )
            .unwrap();
            let snapshot = actual.to_coefficient_rns_snapshot_for_test();
            assert_eq!((snapshot.nrow(), snapshot.ncol()), (1, 5));
            assert_eq!(snapshot.level(), moduli.len() - 1);
            assert!(!snapshot.is_ntt());
            assert_eq!(snapshot.bytes(), expected_rns_bytes);
        }
    }
}
