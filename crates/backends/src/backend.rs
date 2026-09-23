use crate::{
    matrix::{
        CpuSmallMatrix, PolyMatrix as PrimitivePolyMatrix, PolyMatrixColumnSource,
        dcrt_poly::DCRTPolyMatrix,
    },
    poly::PolyParams,
    sampler::trapdoor::DCRTTrapdoor,
    transcript::DrawSite,
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactType, SmallMatrixSemanticKind},
    types::{
        CoefficientBoundDomain, ConcreteMatrixType, ConcreteWireType, InstantiationFrame, WireRef,
    },
};
use num_bigint::BigInt;
#[cfg(feature = "gpu")]
use std::collections::BTreeMap;
use std::{
    fmt::{self, Debug},
    sync::Arc,
};

pub mod poly;
#[cfg(feature = "gpu")]
pub mod poly_gpu;

/// The public matrix value owns immutable semantic type information alongside
/// its native storage. A shallow clone shares storage, including GPU storage.
#[derive(Clone)]
pub struct PolyMatrix {
    ty: ConcreteWireType,
    storage: Arc<MatrixStorage>,
}

enum MatrixStorage {
    CpuFull(Arc<DCRTPolyMatrix>),
    CpuCompact(Arc<CpuSmallMatrix<DCRTPolyMatrix>>),
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuResidentValue>),
    Encoded(Arc<[u8]>),
}

impl PolyMatrix {
    fn wire_type_from_cpu(
        value: &DCRTPolyMatrix,
        compact: Option<(BigInt, SmallMatrixSemanticKind, CoefficientBoundDomain)>,
    ) -> ConcreteWireType {
        let params = value.params();
        let ring = mxx_ir_core::ring::RingRef::new(mxx_ir_core::ring::RingExpr::Explicit {
            crt_moduli: params
                .moduli()
                .iter()
                .copied()
                .map(mxx_ir_core::IntExpr::constant)
                .collect(),
            ring_dimension: params.ring_dimension(),
        })
        .resolve(&ParamEnv::default(), |_, _, _, basis| {
            basis.ok_or_else(|| "trusted CPU matrix has no explicit CRT basis".into())
        })
        .expect("CPU matrix parameters carry a validated CRT basis");
        let (rows, columns) = value.size();
        let matrix = ConcreteMatrixType { ring, rows, columns };
        match compact {
            Some((max_coefficient_bound, SmallMatrixSemanticKind::Generic, bound_domain)) => {
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain }
            }
            Some((max_coefficient_bound, SmallMatrixSemanticKind::Preimage, bound_domain)) => {
                ConcreteWireType::Preimage { matrix, max_coefficient_bound, bound_domain }
            }
            None => ConcreteWireType::Matrix(matrix),
        }
    }

    pub(crate) fn from_backend_cpu_full(value: DCRTPolyMatrix) -> Self {
        let ty = Self::wire_type_from_cpu(&value, None);
        Self::cpu_full(ty, value).expect("backend CPU matrix matches its own validated parameters")
    }

    pub(crate) fn from_backend_cpu_compact(
        value: CpuSmallMatrix<DCRTPolyMatrix>,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Self {
        let ty = Self::wire_type_from_cpu(
            value.value(),
            Some((
                BigInt::from(value.max_coefficient_bound().clone()),
                semantic_kind,
                value.bound_domain(),
            )),
        );
        Self::cpu_compact(ty, value)
            .expect("backend compact matrix matches its own validated parameters")
    }

    fn new(ty: ConcreteWireType, storage: MatrixStorage) -> Result<Self, &'static str> {
        let matrix = ty.matrix_type().ok_or("value is not a matrix wire type")?;
        if !matches!(
            ty,
            ConcreteWireType::Matrix(_) |
                ConcreteWireType::SmallMatrix { .. } |
                ConcreteWireType::Preimage { .. }
        ) {
            return Err("value is not a matrix wire type");
        }
        match &storage {
            MatrixStorage::CpuFull(value) => {
                if !matches!(ty, ConcreteWireType::Matrix(_)) ||
                    !Self::cpu_shape_matches(matrix, value)
                {
                    return Err("full CPU matrix does not match its wire type");
                }
            }
            MatrixStorage::CpuCompact(value) => {
                let expected_bound = match &ty {
                    ConcreteWireType::SmallMatrix {
                        max_coefficient_bound, bound_domain, ..
                    } |
                    ConcreteWireType::Preimage { max_coefficient_bound, bound_domain, .. } => {
                        max_coefficient_bound.to_biguint().map(|bound| (bound, *bound_domain))
                    }
                    _ => None,
                };
                if expected_bound.as_ref() !=
                    Some(&(value.max_coefficient_bound().clone(), value.bound_domain())) ||
                    !Self::cpu_shape_matches(matrix, value.value())
                {
                    return Err("compact CPU matrix does not match its wire type");
                }
            }
            #[cfg(feature = "gpu")]
            MatrixStorage::Gpu(value) if value.wire_type() != &ty => {
                return Err("GPU matrix does not match its wire type");
            }
            #[cfg(feature = "gpu")]
            MatrixStorage::Gpu(_) => {}
            MatrixStorage::Encoded(_) => {}
        }
        Ok(Self { ty, storage: Arc::new(storage) })
    }

    fn cpu_shape_matches(ty: &ConcreteMatrixType, value: &DCRTPolyMatrix) -> bool {
        let params = value.params();
        value.size() == (ty.rows, ty.columns) &&
            params.ring_dimension() == ty.ring.ring_dimension() &&
            params.moduli() == ty.ring.crt_moduli()
    }

    pub fn cpu_full(ty: ConcreteWireType, value: DCRTPolyMatrix) -> Result<Self, &'static str> {
        Self::new(ty, MatrixStorage::CpuFull(Arc::new(value)))
    }

    pub(crate) fn from_shared_cpu_full(
        ty: ConcreteWireType,
        value: Arc<DCRTPolyMatrix>,
    ) -> Result<Self, &'static str> {
        Self::new(ty, MatrixStorage::CpuFull(value))
    }

    pub fn cpu_compact(
        ty: ConcreteWireType,
        value: CpuSmallMatrix<DCRTPolyMatrix>,
    ) -> Result<Self, &'static str> {
        Self::new(ty, MatrixStorage::CpuCompact(Arc::new(value)))
    }

    pub fn encoded(ty: ConcreteWireType, bytes: Arc<[u8]>) -> Result<Self, &'static str> {
        Self::new(ty, MatrixStorage::Encoded(bytes))
    }

    #[cfg(feature = "gpu")]
    pub fn gpu(ty: ConcreteWireType, value: Arc<GpuResidentValue>) -> Result<Self, &'static str> {
        Self::new(ty, MatrixStorage::Gpu(value))
    }

    pub fn wire_type(&self) -> &ConcreteWireType {
        &self.ty
    }

    pub fn matrix_type(&self) -> &ConcreteMatrixType {
        self.ty.matrix_type().expect("matrix value has a matrix wire type")
    }

    pub fn as_cpu_full(&self) -> Option<&DCRTPolyMatrix> {
        match self.storage.as_ref() {
            MatrixStorage::CpuFull(value) => Some(value),
            _ => None,
        }
    }

    pub fn cpu_full_arc(&self) -> Option<Arc<DCRTPolyMatrix>> {
        match self.storage.as_ref() {
            MatrixStorage::CpuFull(value) => Some(Arc::clone(value)),
            _ => None,
        }
    }

    pub fn as_cpu_compact(&self) -> Option<&CpuSmallMatrix<DCRTPolyMatrix>> {
        match self.storage.as_ref() {
            MatrixStorage::CpuCompact(value) => Some(value),
            _ => None,
        }
    }

    pub fn cpu_compact_arc(&self) -> Option<Arc<CpuSmallMatrix<DCRTPolyMatrix>>> {
        match self.storage.as_ref() {
            MatrixStorage::CpuCompact(value) => Some(Arc::clone(value)),
            _ => None,
        }
    }

    pub fn encoded_bytes(&self) -> Option<&[u8]> {
        match self.storage.as_ref() {
            MatrixStorage::Encoded(value) => Some(value),
            _ => None,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn as_gpu(&self) -> Option<&Arc<GpuResidentValue>> {
        match self.storage.as_ref() {
            MatrixStorage::Gpu(value) => Some(value),
            _ => None,
        }
    }
}

impl Debug for PolyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolyMatrix").field("ty", &self.ty).finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct TrapdoorValue {
    ty: ConcreteWireType,
    public: PolyMatrix,
    secret: Option<Arc<DCRTTrapdoor>>,
}

impl TrapdoorValue {
    pub fn wire_type(&self) -> &ConcreteWireType {
        &self.ty
    }

    pub fn public_matrix(&self) -> &PolyMatrix {
        &self.public
    }

    pub fn cpu_secret(&self) -> Option<&DCRTTrapdoor> {
        self.secret.as_deref()
    }

    pub(crate) fn cpu_secret_arc(&self) -> Option<Arc<DCRTTrapdoor>> {
        self.secret.as_ref().map(Arc::clone)
    }

    pub fn new(
        ty: ConcreteWireType,
        public: PolyMatrix,
        secret: Option<Arc<DCRTTrapdoor>>,
    ) -> Result<Self, &'static str> {
        if secret.is_none() {
            return Err("external trapdoor value requires secret matrices");
        }
        Self::checked(ty, public, secret)
    }

    pub(crate) fn public_gadget(
        ty: ConcreteWireType,
        public: PolyMatrix,
    ) -> Result<Self, &'static str> {
        Self::checked(ty, public, None)
    }

    fn checked(
        ty: ConcreteWireType,
        public: PolyMatrix,
        secret: Option<Arc<DCRTTrapdoor>>,
    ) -> Result<Self, &'static str> {
        let ConcreteWireType::Trapdoor { matrix, .. } = &ty else {
            return Err("value is not a trapdoor wire type");
        };
        if public.wire_type() != &ConcreteWireType::Matrix(matrix.clone()) {
            return Err("trapdoor public matrix does not match its wire type");
        }
        if secret.as_ref().is_some_and(|secret| {
            !secret.matches_ordered_ring(matrix.ring.crt_moduli(), matrix.ring.ring_dimension())
        }) {
            return Err("trapdoor secret matrices do not match its ordered ring");
        }
        Ok(Self { ty, public, secret })
    }
}

impl Debug for TrapdoorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrapdoorValue").field("ty", &self.ty).finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub(crate) struct BoundStorage {
    pub device: i32,
    pub address: u64,
    pub bytes: u64,
    pub owner: Arc<dyn std::any::Any + Send + Sync>,
}

#[cfg(feature = "gpu")]
impl BoundStorage {
    pub(crate) fn from_matrix_data(
        owner: Arc<crate::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix>,
        component_index: usize,
    ) -> Result<Self, String> {
        let components = owner.binding_components().map_err(|error| error.to_string())?;
        let component = components
            .get(component_index)
            .ok_or_else(|| "GPU matrix component index is out of bounds".to_string())?;
        let bytes = u64::try_from(component.data_bytes)
            .map_err(|_| "GPU matrix allocation length overflows u64".to_string())?;
        if component.data_address == 0 || bytes == 0 {
            return Err("GPU matrix has no bound data allocation".into());
        }
        Ok(Self {
            device: component.physical_device,
            address: component.data_address,
            bytes,
            owner,
        })
    }

    pub(crate) fn from_signed_values(
        owner: Arc<crate::poly::dcrt::gpu::GpuSignedValues>,
    ) -> Result<Self, String> {
        let binding = owner.binding_descriptor().map_err(|error| error.to_string())?;
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU signed-value allocation length overflows u64".to_string())?;
        if binding.device_address == 0 || bytes == 0 {
            return Err("GPU signed values have no bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address: binding.device_address, bytes, owner })
    }

    pub(crate) fn from_small_matrix_payload(
        owner: Arc<crate::matrix::gpu_dcrt_poly::GpuSmallMatrix>,
    ) -> Result<Self, String> {
        let binding = owner.binding_descriptor().map_err(|error| error.to_string())?;
        let bytes = u64::try_from(binding.payload_bytes)
            .map_err(|_| "GPU compact payload length overflows u64".to_string())?;
        if binding.payload_address == 0 || bytes == 0 {
            return Err("GPU compact matrix has no bound payload allocation".into());
        }
        Ok(Self { device: binding.physical_device, address: binding.payload_address, bytes, owner })
    }

    pub(crate) fn from_device_seed(
        owner: Arc<crate::poly::dcrt::gpu::GpuDeviceSeed>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU seed allocation length overflows u64".to_string())?;
        if address == 0 || bytes == 0 {
            return Err("GPU seed has no bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }

    pub(crate) fn from_device_bytes(
        owner: Arc<crate::poly::dcrt::gpu::GpuDeviceBytes>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.allocation_bytes())
            .map_err(|_| "GPU bytes allocation length overflows u64".to_string())?;
        if address == 0 || bytes == 0 {
            return Err("GPU bytes have no bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }

    pub(crate) fn from_export_status(
        owner: Arc<crate::poly::dcrt::gpu::GpuExportStatus>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU export status allocation length overflows u64".to_string())?;
        if address == 0 || bytes == 0 {
            return Err("GPU export status has no bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }

    pub(crate) fn from_preimage_attempt(
        owner: Arc<crate::poly::dcrt::gpu::GpuPreimageAttempt>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU preimage attempt allocation length overflows u64".to_string())?;
        if address == 0 || bytes != 8 {
            return Err("GPU preimage attempt has no eight-byte bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }

    pub(crate) fn from_preimage_status(
        owner: Arc<crate::poly::dcrt::gpu::GpuPreimageStatus>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU preimage status allocation length overflows u64".to_string())?;
        if address == 0 || bytes != 16 {
            return Err("GPU preimage status has no sixteen-byte bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }

    pub(crate) fn from_device_real(
        owner: Arc<crate::poly::dcrt::gpu_real::GpuDeviceReal>,
    ) -> Result<Self, String> {
        let address = owner.device_address();
        let bytes = u64::try_from(owner.byte_len())
            .map_err(|_| "GPU real allocation length overflows u64".to_string())?;
        if address == 0 || bytes != 8 {
            return Err("GPU real has no eight-byte bound allocation".into());
        }
        Ok(Self { device: owner.physical_device(), address, bytes, owner })
    }
}

#[cfg(feature = "gpu")]
pub struct GpuResidentValue {
    physical: Arc<crate::gpu_execution_plan::PhysicalValue>,
    storage: BTreeMap<crate::gpu_execution_plan::StorageRef, BoundStorage>,
    ready: Box<[Arc<crate::poly::dcrt::gpu::GpuNativeEvent>]>,
}

#[cfg(feature = "gpu")]
impl GpuResidentValue {
    pub fn wire_type(&self) -> &ConcreteWireType {
        &self.physical.ty
    }

    pub(crate) fn physical(&self) -> &Arc<crate::gpu_execution_plan::PhysicalValue> {
        &self.physical
    }

    pub(crate) fn storage(
        &self,
        slot: crate::gpu_execution_plan::StorageRef,
    ) -> Option<&BoundStorage> {
        self.storage.get(&slot)
    }

    pub(crate) fn ready_events(&self) -> &[Arc<crate::poly::dcrt::gpu::GpuNativeEvent>] {
        &self.ready
    }

    pub(crate) fn with_physical_view(
        &self,
        physical: Arc<crate::gpu_execution_plan::PhysicalValue>,
    ) -> Result<Self, &'static str> {
        Self::new(physical, self.storage.clone(), self.ready.clone())
    }

    pub(crate) fn new(
        physical: Arc<crate::gpu_execution_plan::PhysicalValue>,
        storage: BTreeMap<crate::gpu_execution_plan::StorageRef, BoundStorage>,
        ready: Box<[Arc<crate::poly::dcrt::gpu::GpuNativeEvent>]>,
    ) -> Result<Self, &'static str> {
        for part in physical.parts.iter() {
            let bound = storage.get(&part.storage).ok_or("physical storage binding is missing")?;
            if part.view.element_bytes == 0 ||
                bound.address == 0 ||
                bound.bytes == 0 ||
                bound.address.checked_add(bound.bytes).is_none() ||
                bound.address % u64::from(part.view.element_bytes) != 0
            {
                return Err("physical storage binding has an invalid device address");
            }
        }
        physical.validate(|slot| storage.get(&slot).map(|bound| (bound.device, bound.bytes)))?;
        Ok(Self { physical, storage, ready })
    }
}

#[cfg(feature = "gpu")]
impl Debug for GpuResidentValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuResidentValue").field("ty", self.wire_type()).finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeValue {
    Int(BigInt),
    Real(f64),
    Bool(bool),
    Bytes(Arc<[u8]>),
    TypedBlob {
        type_name: String,
        schema_hash: [u8; 32],
        bytes: Arc<[u8]>,
    },
    Matrix(PolyMatrix),
    Trapdoor(TrapdoorValue),
    #[cfg(feature = "gpu")]
    Resident(Arc<GpuResidentValue>),
    IndexedFamily {
        element_type: ConcreteWireType,
        values: Arc<[RuntimeValue]>,
    },
    LazyArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: Option<usize>,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    LazyArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: usize,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
}

impl RuntimeValue {
    /// Bind an already uploaded signed GPU family to its exact semantic type.
    /// The owner supplies the device address, encoding, and allocation length.
    #[cfg(feature = "gpu")]
    pub fn gpu_signed_family(
        ty: ConcreteWireType,
        owner: Arc<crate::poly::dcrt::gpu::GpuSignedValues>,
    ) -> Result<Self, String> {
        use crate::{
            gpu_execution_plan::{
                PhysicalEncoding, PhysicalPart, PhysicalValue, PhysicalView, StorageRef,
            },
            poly::dcrt::gpu::GpuSignedValuesEncoding,
        };

        let ConcreteWireType::IndexedFamily { element, count } = &ty else {
            return Err("GPU signed family requires an indexed-family wire type".into());
        };
        if *count == 0 || *count != owner.count() {
            return Err("GPU signed family count differs from its owner".into());
        }
        let encoding = match (element.as_ref(), owner.encoding()) {
            (ConcreteWireType::Int, encoding) => PhysicalEncoding::Signed(encoding),
            (ConcreteWireType::Bool, GpuSignedValuesEncoding::SignedI64) => {
                PhysicalEncoding::BoolI64
            }
            _ => return Err("GPU signed family element type or encoding is unsupported".into()),
        };
        let words = owner.encoding().words_per_value();
        let bytes_per_value = words
            .checked_mul(8)
            .ok_or_else(|| "GPU signed family word width overflows".to_string())?;
        let family_stride = u64::try_from(bytes_per_value)
            .map_err(|_| "GPU signed family stride overflows u64".to_string())?;
        let count = u64::try_from(*count)
            .map_err(|_| "GPU signed family count overflows u64".to_string())?;
        let (origin, extent, byte_strides): (Box<[u64]>, Box<[u64]>, Box<[u64]>) =
            if matches!(encoding, PhysicalEncoding::BoolI64) {
                (
                    vec![0, 0].into_boxed_slice(),
                    vec![count, 1].into_boxed_slice(),
                    vec![8, 8].into_boxed_slice(),
                )
            } else {
                (
                    vec![0, 0, 0].into_boxed_slice(),
                    vec![count, 1, words as u64].into_boxed_slice(),
                    vec![family_stride, family_stride, 8].into_boxed_slice(),
                )
            };
        let slot = StorageRef::Input(0);
        let physical = PhysicalValue {
            ty,
            encodings: Box::new([encoding]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage: slot,
                device: owner.physical_device(),
                view: PhysicalView {
                    byte_offset: 0,
                    origin,
                    extent,
                    byte_strides,
                    element_bytes: 8,
                },
            }]),
            integer_ranges: BTreeMap::new(),
        };
        let resident = GpuResidentValue::new(
            Arc::new(physical),
            BTreeMap::from([(slot, BoundStorage::from_signed_values(owner)?)]),
            Box::new([]),
        )
        .map_err(str::to_owned)?;
        Ok(Self::Resident(Arc::new(resident)))
    }

    #[cfg(feature = "gpu")]
    pub fn gpu_matrix(
        ty: ConcreteWireType,
        owner: Arc<crate::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix>,
    ) -> Result<Self, String> {
        use crate::gpu_execution_plan::{PhysicalEncoding, StorageRef};

        let ConcreteWireType::Matrix(matrix) = &ty else {
            return Err("GPU matrix value requires a matrix wire type".into());
        };
        if owner.size() != (matrix.rows, matrix.columns) ||
            owner.params().ring_dimension() != matrix.ring.ring_dimension() ||
            owner.params().moduli() != matrix.ring.crt_moduli()
        {
            return Err("GPU matrix owner disagrees with its concrete wire type".into());
        }
        let encoding =
            if owner.is_ntt() { PhysicalEncoding::FullEval } else { PhysicalEncoding::FullCoeff };
        let (_, resident) = crate::gpu_physical_lowering::physical_matrix(
            matrix,
            encoding,
            StorageRef::Input(0),
            owner,
        )?;
        Ok(Self::Matrix(PolyMatrix::gpu(ty, resident).map_err(str::to_owned)?))
    }

    pub fn matrix(value: DCRTPolyMatrix) -> Self {
        Self::Matrix(PolyMatrix::from_backend_cpu_full(value))
    }

    pub fn small_matrix(value: CpuSmallMatrix<DCRTPolyMatrix>) -> Self {
        Self::Matrix(PolyMatrix::from_backend_cpu_compact(value, SmallMatrixSemanticKind::Generic))
    }

    pub fn preimage(value: CpuSmallMatrix<DCRTPolyMatrix>) -> Self {
        Self::Matrix(PolyMatrix::from_backend_cpu_compact(value, SmallMatrixSemanticKind::Preimage))
    }

    pub fn integer_values(values: Vec<BigInt>) -> Self {
        Self::IndexedFamily {
            element_type: ConcreteWireType::Int,
            values: values.into_iter().map(Self::Int).collect::<Vec<_>>().into(),
        }
    }

    pub fn indexed_family(
        element_type: ConcreteWireType,
        values: Vec<RuntimeValue>,
    ) -> Result<Self, &'static str> {
        if !values.iter().all(|value| value.matches_wire_type(&element_type)) {
            return Err("indexed family contains a value with the wrong element type");
        }
        Ok(Self::IndexedFamily { element_type, values: values.into() })
    }

    pub fn matches_wire_type(&self, ty: &ConcreteWireType) -> bool {
        match (self, ty) {
            (Self::Int(_), ConcreteWireType::Int | ConcreteWireType::ConstantInt) |
            (Self::Real(_), ConcreteWireType::Real | ConcreteWireType::ConstantReal) |
            (Self::Bool(_), ConcreteWireType::Bool | ConcreteWireType::ConstantBool) => true,
            (Self::Bytes(bytes), ConcreteWireType::Bytes { length }) => bytes.len() == *length,
            (
                Self::TypedBlob { type_name, schema_hash, .. },
                ConcreteWireType::TypedBlob {
                    type_name: expected_name,
                    schema_hash: expected_hash,
                },
            ) => type_name == expected_name && schema_hash == expected_hash,
            (Self::Matrix(matrix), _) => matrix.wire_type() == ty,
            (Self::Trapdoor(trapdoor), _) => trapdoor.wire_type() == ty,
            #[cfg(feature = "gpu")]
            (Self::Resident(resident), _) => resident.wire_type() == ty,
            (
                Self::IndexedFamily { element_type, values },
                ConcreteWireType::IndexedFamily { element, count },
            ) => {
                values.len() == *count &&
                    element_type == element.as_ref() &&
                    values.iter().all(|value| value.matches_wire_type(element))
            }
            (
                Self::LazyArtifactFamily { descriptor, .. } |
                Self::StagedArtifactFamily { descriptor, .. },
                ConcreteWireType::IndexedFamily { element, count },
            ) => {
                descriptor.family_count == Some(*count) &&
                    ArtifactType::from_wire_type(element).as_ref() ==
                        Some(&descriptor.artifact_type)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod concrete_value_tests {
    use super::*;
    use crate::poly::dcrt::params::DCRTPolyParams;

    #[test]
    fn cpu_matrix_clone_shares_storage_and_checks_ordered_ring() {
        let parameters = DCRTPolyParams::new(8, 2, 17, 2, None, None);
        let value = RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, 1, 1));
        let RuntimeValue::Matrix(matrix) = value else {
            panic!("matrix constructor returned another value kind")
        };
        let cloned = matrix.clone();
        assert!(std::ptr::eq(matrix.as_cpu_full().unwrap(), cloned.as_cpu_full().unwrap()));
        assert_eq!(matrix.matrix_type().ring.crt_moduli(), parameters.to_crt().0);
        assert!(
            RuntimeValue::indexed_family(
                ConcreteWireType::Int,
                vec![RuntimeValue::Int(BigInt::from(1)), RuntimeValue::Bool(true)],
            )
            .is_err()
        );
    }
}

#[derive(Clone, Debug)]
pub struct PreimageRequest<M, T> {
    pub instance_slot: usize,
    /// Fixed-plan site/layout contract for the sampled instance.  Legacy
    /// callers leave this absent; fixed executor paths always populate it.
    pub fixed_metadata: Option<PlannedNodeBatchRequest>,
    pub matrix_type: ConcreteMatrixType,
    pub sigma: f64,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
    pub trapdoor: Arc<T>,
    pub public: Arc<M>,
    pub target: Arc<dyn PolyMatrixColumnSource<M>>,
    /// Seed for deterministic GPU sampling with a fixed column schedule.
    pub randomness_seed: [u8; 32],
}

/// Metadata passed to a backend immediately before a fixed node batch is
/// submitted.  It deliberately contains no matrix owner or native pointer;
/// those are bound by the backend to the current runtime values.  The
/// executor keeps the original instance slots so a backend cannot accidentally
/// use a compressed batch index as a sampling or output coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedNodeBatchRequest {
    pub site: u64,
    pub shape_class: u64,
    pub instance_class: u64,
    pub operation_identity: [u8; 32],
    pub implementation_variant: String,
    pub output_layouts: Vec<crate::gpu_execution_plan::LayoutId>,
    /// Explicit port-to-layout bindings. `output_layouts` is retained as the
    /// compact plan representation, while this field prevents a backend from
    /// accidentally treating a reordered fused output as positional-only.
    pub output_port_layouts: Vec<(usize, crate::gpu_execution_plan::LayoutId)>,
    /// Concrete source layouts checked at submit time.  `layout_id` is absent
    /// when a source is a runtime value rather than a planned output.
    pub source_layouts: Vec<PlannedLayoutMetadata>,
    /// Effective operands after lowering fused nodes.  A compact product's
    /// logical concat is represented by its origin wires and lowered row
    /// blocks, while its compact RHS remains a distinct typed operand.
    /// `source_layouts` therefore describes only the physical block sources
    /// for that operation, never an ambiguous logical `(concat, rhs)` pair.
    pub effective_operands: Vec<PlannedOperandMetadata>,
    pub output_layout_metadata: Vec<PlannedLayoutMetadata>,
    pub columns_per_job: Vec<usize>,
    pub instance_slots: Vec<usize>,
    pub instance_paths: Vec<Vec<InstantiationFrame>>,
    pub draw_sites: Vec<Option<DrawSite>>,
    pub randomness_seeds: Vec<Option<[u8; 32]>>,
    pub output_ports: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlannedOperandMetadata {
    RowBlock {
        logical_operand: usize,
        block_index: usize,
        origin: WireRef,
        layout: PlannedLayoutMetadata,
    },
    CompactRhs {
        logical_operand: usize,
        origin: WireRef,
        layout: PlannedLayoutMetadata,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedLayoutMetadata {
    pub layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: String,
}

impl PlannedLayoutMetadata {
    pub fn for_type(
        ty: &ConcreteWireType,
        layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    ) -> Self {
        let matrix = ty.matrix_type();
        Self {
            layout_id,
            rows: matrix.map_or(0, |matrix| matrix.rows),
            columns: matrix.map_or(0, |matrix| matrix.columns),
            ring_dimension: matrix.map_or(0, |matrix| matrix.ring.ring_dimension() as usize),
            representation: format!("{ty:?}"),
        }
    }
}

impl PlannedNodeBatchRequest {
    /// Canonical fixed metadata for a lowered operation. Warmup supplies its
    /// candidate output layouts; execution supplies the frozen plan layouts.
    /// Instance paths and random draws are bound by the respective caller.
    pub fn for_lowered_operation(
        key: crate::gpu_execution_plan::GpuExecutionSiteKey,
        operation_identity: [u8; 32],
        implementation_variant: String,
        inputs: GpuEffectiveInputs,
        outputs: Vec<PlannedLayoutMetadata>,
        columns_per_job: Vec<usize>,
    ) -> Self {
        let output_layouts =
            outputs.iter().filter_map(|output| output.layout_id).collect::<Vec<_>>();
        Self {
            site: key.site,
            shape_class: key.shape_class,
            instance_class: key.instance_class,
            operation_identity,
            implementation_variant,
            output_port_layouts: output_layouts
                .iter()
                .enumerate()
                .map(|(port, layout)| (port, *layout))
                .collect(),
            output_layouts,
            source_layouts: inputs.source_layouts,
            effective_operands: inputs.operands,
            output_ports: outputs.len(),
            output_layout_metadata: outputs,
            columns_per_job,
            instance_slots: vec![0],
            instance_paths: vec![Vec::new()],
            draw_sites: vec![None],
            randomness_seeds: vec![None],
        }
    }
}

/// Dynamic fused primitives retain the backend's ordinary CPU-compatible
/// fallback semantics. They are a distinct type so dynamic execution cannot
/// manufacture a fixed request with absent plan metadata.
pub enum DynamicFusedBatchRequest<M, S> {
    RowSum { source: Arc<M>, right: Option<Arc<M>>, rows: Vec<Vec<usize>> },
    TensorRowSums { source: Arc<M>, right: Arc<M>, rows: Vec<Vec<Vec<usize>>> },
    Decompose { blocks: Vec<Arc<M>>, small: bool, digits: usize },
    SmallProduct { blocks: Vec<Arc<M>>, rhs: Arc<S> },
    Add { blocks: Vec<Arc<M>>, right: Arc<M> },
}

pub enum FusedBatchOutput<M, S> {
    Matrices(Vec<M>),
    Small(S),
}

/// Full logical preimage target whose expanded columns are loaded on demand.
/// The staged constructor owns host bytes, allowing a GPU owner to be dropped
/// before sampling while preserving the original matrix dimensions.
pub struct PreimageTarget<M> {
    rows: usize,
    columns: usize,
    loader: Arc<dyn Fn(usize, usize) -> M + Send + Sync>,
}

impl<M> Clone for PreimageTarget<M> {
    fn clone(&self) -> Self {
        Self { rows: self.rows, columns: self.columns, loader: self.loader.clone() }
    }
}

impl<M> Debug for PreimageTarget<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreimageTarget")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl<M> PreimageTarget<M>
where
    M: PrimitivePolyMatrix + 'static,
{
    pub fn staged(
        params: <M::P as crate::poly::Poly>::Params,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Self {
        let loader = Arc::new(move |start: usize, end: usize| {
            M::from_cpu_staging_columns(&params, bytes.as_slice(), start, end)
        });
        Self { rows, columns, loader }
    }

    pub fn resident(value: Arc<M>) -> Self {
        let rows = value.row_size();
        let columns = value.col_size();
        let loader = Arc::new(move |start: usize, end: usize| value.slice_columns(start, end));
        Self { rows, columns, loader }
    }
}

impl<M> PolyMatrixColumnSource<M> for PreimageTarget<M>
where
    M: PrimitivePolyMatrix + 'static,
{
    fn row_size(&self) -> usize {
        self.rows
    }

    fn col_size(&self) -> usize {
        self.columns
    }

    fn load_columns(&self, start: usize, end: usize) -> M {
        assert!(start <= end && end <= self.columns, "invalid preimage target column interval");
        (self.loader)(start, end)
    }
}

#[derive(Clone, Debug)]
pub struct MatrixMulAccumulateRequest<M> {
    pub products: Vec<(BigInt, Arc<M>, Arc<M>)>,
    pub bias: Option<Arc<M>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleRange {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// Shared, value-only lowering contract for measurement and fixed dispatch.
/// Origins are ordered exactly as the concrete backend request; logical
/// concatenations that execution elides never appear as replacement inputs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuEffectiveInputs {
    pub origins: Vec<WireRef>,
    pub source_layouts: Vec<PlannedLayoutMetadata>,
    pub operands: Vec<PlannedOperandMetadata>,
    pub row_groups: Vec<Vec<usize>>,
    /// Complete row-group inventory for a shared TensorRowSums dispatch.
    /// Empty means the ordinary single-output fused row sum contract.
    pub row_sum_groups: Vec<Vec<Vec<usize>>>,
}
