//! Small, value-only metadata for a frozen GPU execution plan.
//!
//! A plan records choices made during warmup. It never owns GPU buffers,
//! pointers, command objects, or runtime owners. Input ranges, physical shard
//! views, and transfer actions remain derived data produced by the shared
//! column-policy lowering.

#[cfg(feature = "gpu")]
use crate::poly::dcrt::gpu::{GpuHashTagPart, GpuSignedValuesEncoding};
use crate::{
    gpu_column_policy::{
        ColumnCapability, EffectiveGpuOperation, capability_for_effective_operation,
    },
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule, GpuScheduleError},
};
#[cfg(feature = "gpu")]
use mxx_ir_core::types::ConcreteWireType;
#[cfg(feature = "gpu")]
use num_bigint::BigInt;
#[cfg(any(feature = "gpu", test))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "gpu")]
use std::ops::RangeInclusive;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// An index into the physical-value table of one compiled plan. IDs must never
/// be moved between plans without their table and allocation bindings.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg(feature = "gpu")]
pub(crate) struct PhysicalValueId(pub u32);

/// HashSample's frozen framing and replayed scalar operands. The operand
/// addresses are resolved from current resident bindings at launch time.
#[derive(Clone, Debug)]
#[cfg(feature = "gpu")]
pub(crate) struct GpuHashResourceSpec {
    pub parts: Arc<[GpuHashTagPart]>,
    pub operands: Box<[(PhysicalValueId, u32, u32)]>,
}

/// An allocation binding slot local to a plan or a resident value. The number
/// is an allocation slot, not an operand number.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg(feature = "gpu")]
pub(crate) enum StorageRef {
    Input(u32),
    Output(u32),
    Constant(u32),
    Scratch(u32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) enum PhysicalEncoding {
    FullCoeff,
    FullEval,
    /// Public-only gadget trapdoor represented by its one public matrix.
    PublicGadgetEval,
    CompactCoeff {
        magnitude_bytes: usize,
    },
    /// Signed magnitudes are distinct for each ordered CRT limb.
    CompactCoeffPerCrtLimb {
        magnitude_bytes: usize,
    },
    Signed(GpuSignedValuesEncoding),
    BoolI64,
    RealF64,
    Bytes,
    /// Opaque payload with an in-band u64 length and fixed plan capacity.
    TypedBlobLengthPrefixed,
}

/// Coordinates name logical elements. For a coordinate j, the corresponding
/// address is base + byte_offset + sum((j[a] - origin[a]) * byte_strides[a]).
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct PhysicalView {
    pub byte_offset: u64,
    pub origin: Box<[u64]>,
    pub extent: Box<[u64]>,
    pub byte_strides: Box<[u64]>,
    pub element_bytes: u32,
}

#[cfg(feature = "gpu")]
impl PhysicalView {
    /// Check every addressable element, including the last byte of the last
    /// element, before binding a native pointer. An empty semantic leaf has no
    /// parts and must not be encoded as a zero-extent view.
    pub(crate) fn validate_in_allocation(
        &self,
        allocation_bytes: u64,
        alignment: u64,
    ) -> Result<(), &'static str> {
        if self.origin.len() != self.extent.len() ||
            self.origin.len() != self.byte_strides.len() ||
            self.origin.is_empty() ||
            self.element_bytes == 0 ||
            alignment == 0 ||
            self.extent.contains(&0)
        {
            return Err("invalid physical view dimensions");
        }
        if self.byte_offset % alignment != 0 || u64::from(self.element_bytes) % alignment != 0 {
            return Err("physical view is misaligned");
        }
        let mut last_byte = self.byte_offset;
        for (&origin, (&extent, &stride)) in
            self.origin.iter().zip(self.extent.iter().zip(self.byte_strides.iter()))
        {
            origin.checked_add(extent).ok_or("physical coordinate overflow")?;
            let displacement =
                (extent - 1).checked_mul(stride).ok_or("physical stride overflow")?;
            last_byte = last_byte.checked_add(displacement).ok_or("physical address overflow")?;
        }
        last_byte = last_byte
            .checked_add(u64::from(self.element_bytes) - 1)
            .ok_or("physical address overflow")?;
        if last_byte >= allocation_bytes {
            return Err("physical view exceeds allocation");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct PhysicalPart {
    pub leaf: u32,
    pub storage: StorageRef,
    pub device: i32,
    pub view: PhysicalView,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct PhysicalValue {
    pub ty: ConcreteWireType,
    pub encodings: Box<[PhysicalEncoding]>,
    pub parts: Box<[PhysicalPart]>,
    pub integer_ranges: BTreeMap<u32, RangeInclusive<BigInt>>,
}

#[cfg(feature = "gpu")]
impl PhysicalValue {
    /// Validate the compact template shape without expanding a family by its
    /// element count. Allocation lengths come from the actual bound owners.
    pub(crate) fn validate(
        &self,
        mut allocation: impl FnMut(StorageRef) -> Option<(i32, u64)>,
    ) -> Result<(), &'static str> {
        fn collect_leaves(
            ty: &ConcreteWireType,
            family_axes: usize,
            public_gadget: bool,
            out: &mut Vec<(usize, u8)>,
        ) {
            match ty {
                ConcreteWireType::IndexedFamily { element, .. } => {
                    collect_leaves(element, family_axes + 1, public_gadget, out)
                }
                ConcreteWireType::Trapdoor { .. } => {
                    if public_gadget {
                        out.push((family_axes + 4, 5));
                    } else {
                        out.extend(std::iter::repeat_n((family_axes + 4, 0), 6));
                    }
                }
                ConcreteWireType::Matrix(_) |
                ConcreteWireType::SmallMatrix { .. } |
                ConcreteWireType::Preimage { .. } => out.push((family_axes + 4, 0)),
                ConcreteWireType::Int | ConcreteWireType::ConstantInt => {
                    out.push((family_axes + 2, 1));
                }
                ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                    out.push((family_axes + 1, 2));
                }
                ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                    out.push((family_axes + 1, 3));
                }
                ConcreteWireType::Bytes { .. } => {
                    out.push((family_axes + 1, 4));
                }
                ConcreteWireType::TypedBlob { .. } => out.push((family_axes + 1, 6)),
            }
        }
        let mut semantic_leaf = &self.ty;
        while let ConcreteWireType::IndexedFamily { element, .. } = semantic_leaf {
            semantic_leaf = element;
        }
        let public_gadget = matches!(semantic_leaf, ConcreteWireType::Trapdoor { .. }) &&
            matches!(self.encodings.as_ref(), [PhysicalEncoding::PublicGadgetEval]);
        let mut leaves = Vec::new();
        collect_leaves(&self.ty, 0, public_gadget, &mut leaves);
        if self.encodings.len() != leaves.len() {
            return Err("physical encoding count does not match semantic leaves");
        }
        for (encoding, &(_, kind)) in self.encodings.iter().zip(&leaves) {
            let compatible = matches!(
                (kind, encoding),
                (
                    0,
                    PhysicalEncoding::FullCoeff |
                        PhysicalEncoding::FullEval |
                        PhysicalEncoding::CompactCoeff { .. } |
                        PhysicalEncoding::CompactCoeffPerCrtLimb { .. }
                ) | (1, PhysicalEncoding::Signed(_)) |
                    (2, PhysicalEncoding::BoolI64) |
                    (3, PhysicalEncoding::RealF64) |
                    (4, PhysicalEncoding::Bytes) |
                    (6, PhysicalEncoding::TypedBlobLengthPrefixed) |
                    (5, PhysicalEncoding::PublicGadgetEval)
            );
            if !compatible {
                return Err("physical encoding does not match semantic leaf");
            }
        }
        for (&leaf, range) in &self.integer_ranges {
            if leaves.get(leaf as usize).is_none_or(|(_, kind)| *kind != 1) ||
                range.start() > range.end()
            {
                return Err("invalid physical integer range");
            }
        }
        for part in &self.parts {
            let (expected_axes, _) =
                leaves.get(part.leaf as usize).ok_or("physical part references an unknown leaf")?;
            let encoding = &self.encodings[part.leaf as usize];
            let expected_axes = *expected_axes +
                usize::from(matches!(encoding, PhysicalEncoding::CompactCoeffPerCrtLimb { .. }));
            if part.view.origin.len() != expected_axes {
                return Err("physical view has the wrong axis count");
            }
            let (device, bytes) =
                allocation(part.storage).ok_or("physical part has no allocation binding")?;
            if part.device != device {
                return Err("physical part device differs from allocation device");
            }
            let alignment = match encoding {
                PhysicalEncoding::FullCoeff |
                PhysicalEncoding::FullEval |
                PhysicalEncoding::PublicGadgetEval => {
                    if !matches!(part.view.element_bytes, 4 | 8) {
                        return Err("full residue word must occupy 4 or 8 bytes");
                    }
                    u64::from(part.view.element_bytes)
                }
                PhysicalEncoding::Signed(_) |
                PhysicalEncoding::BoolI64 |
                PhysicalEncoding::RealF64 => 8,
                PhysicalEncoding::CompactCoeff { .. } |
                PhysicalEncoding::CompactCoeffPerCrtLimb { .. } |
                PhysicalEncoding::Bytes |
                PhysicalEncoding::TypedBlobLengthPrefixed => 1,
            };
            if !matches!(
                encoding,
                PhysicalEncoding::FullCoeff |
                    PhysicalEncoding::FullEval |
                    PhysicalEncoding::PublicGadgetEval
            ) && part.view.element_bytes != alignment as u32
            {
                return Err("physical element width does not match encoding");
            }
            let mut family_bounds = Vec::new();
            let mut leaf_ty = &self.ty;
            while let ConcreteWireType::IndexedFamily { element, count } = leaf_ty {
                family_bounds.push(*count as u64);
                leaf_ty = element;
            }
            if matches!(encoding, PhysicalEncoding::CompactCoeff { .. }) &&
                matches!(
                    leaf_ty,
                    ConcreteWireType::SmallMatrix {
                        bound_domain: mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb,
                        ..
                    } | ConcreteWireType::Preimage {
                        bound_domain: mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb,
                        ..
                    }
                )
            {
                return Err("per-CRT-limb bounded values cannot use global compact encoding");
            }
            if matches!(encoding, PhysicalEncoding::CompactCoeffPerCrtLimb { .. }) &&
                !matches!(
                    leaf_ty,
                    ConcreteWireType::SmallMatrix {
                        bound_domain: mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb,
                        ..
                    } | ConcreteWireType::Preimage {
                        bound_domain: mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb,
                        ..
                    }
                )
            {
                return Err("per-CRT-limb compact encoding needs a per-limb bounded type");
            }
            let leaf_bounds = match (leaf_ty, encoding) {
                (
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. },
                    PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval,
                ) => Some(
                    [
                        matrix.rows as u64,
                        matrix.columns as u64,
                        matrix.ring.crt_depth() as u64,
                        u64::from(matrix.ring.ring_dimension()),
                    ]
                    .as_slice()
                    .to_vec(),
                ),
                (ConcreteWireType::Trapdoor { matrix, .. }, PhysicalEncoding::PublicGadgetEval) => {
                    Some(vec![
                        matrix.rows as u64,
                        matrix.columns as u64,
                        matrix.ring.crt_depth() as u64,
                        u64::from(matrix.ring.ring_dimension()),
                    ])
                }
                (
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. },
                    PhysicalEncoding::CompactCoeff { magnitude_bytes },
                ) => {
                    let payload_bytes = magnitude_bytes
                        .checked_add(1)
                        .ok_or("compact coefficient width overflows")?;
                    Some(vec![
                        matrix.rows as u64,
                        matrix.columns as u64,
                        u64::from(matrix.ring.ring_dimension()),
                        payload_bytes as u64,
                    ])
                }
                (
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. },
                    PhysicalEncoding::CompactCoeffPerCrtLimb { magnitude_bytes },
                ) => {
                    let payload_bytes = magnitude_bytes
                        .checked_add(1)
                        .ok_or("compact coefficient width overflows")?;
                    Some(vec![
                        matrix.rows as u64,
                        matrix.columns as u64,
                        u64::from(matrix.ring.ring_dimension()),
                        matrix.ring.crt_depth() as u64,
                        payload_bytes as u64,
                    ])
                }
                (
                    ConcreteWireType::Int | ConcreteWireType::ConstantInt,
                    PhysicalEncoding::Signed(integer),
                ) => Some(vec![1, integer.words_per_value() as u64]),
                (
                    ConcreteWireType::Bool |
                    ConcreteWireType::ConstantBool |
                    ConcreteWireType::Real |
                    ConcreteWireType::ConstantReal,
                    _,
                ) => Some(vec![1]),
                (ConcreteWireType::Bytes { length }, _) => Some(vec![*length as u64]),
                (ConcreteWireType::TypedBlob { .. }, PhysicalEncoding::TypedBlobLengthPrefixed) => {
                    if part.view.extent[0] < 8 {
                        return Err("length-prefixed typed blob capacity is below its header");
                    }
                    Some(vec![part.view.extent[0]])
                }
                (ConcreteWireType::Trapdoor { .. }, _) => None,
                _ => return Err("physical encoding and semantic leaf disagree"),
            };
            if let Some(leaf_bounds) = leaf_bounds {
                family_bounds.extend(leaf_bounds);
                if part.view.origin.len() != family_bounds.len() ||
                    part.view.origin.iter().zip(&part.view.extent).zip(&family_bounds).any(
                        |((&origin, &extent), &bound)| {
                            origin.checked_add(extent).is_none_or(|end| end > bound)
                        },
                    )
                {
                    return Err("physical view exceeds semantic shape");
                }
            }
            part.view.validate_in_allocation(bytes, alignment)?;
        }
        Ok(())
    }
}

/// The native implementation registry is process-local. Persistent cache keys
/// also require the registry schema and build identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg(feature = "gpu")]
pub(crate) struct GpuImplId(pub u32);

/// The selected implementation and its native emitter are registered together.
/// An ID is meaningful only with the registry that allocated it; serialized
/// plans must resolve it again against the current binary's registry schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[cfg(feature = "gpu")]
pub(crate) enum GpuNativePrimitive {
    Copy,
    Zero,
    ForwardNtt,
    InverseNtt,
    GadgetDecomposeCoeff,
    GadgetDecomposeSmallBalanced,
    MatrixAdd,
    MatrixSub,
    MatrixMul,
    MatrixMulAccumulate,
    ExpandCompact,
    MatrixCopyView,
    MatrixTranspose,
    MatrixTensor,
    MatrixScale,
    MatrixScaleDynamic,
    MatrixIndexedCopy,
    MatrixSliceDynamic,
    RingAutomorphism,
    LiftIntegerConstant,
    MatrixMulTransposeRhs,
    IdentityFill,
    GadgetFill,
    BranchIf,
    LoopWhile,
    IntegerOperation,
    RealOperation,
    ModulusSwitch,
    CrtConvert,
    CenteredRoundDivide,
    CenteredRoundDivideDynamic,
    RnsModUp,
    RnsModDown,
    BlockModSwitch,
    CrtRecomposeLevel,
    CompactPack,
    CompactPackPerCrtLimb,
    HashSample,
    SampleUniform,
    SampleBit,
    SampleGaussian,
    PolynomialFromValues,
    PolynomialValues,
    ExtractCoefficient,
    PackPolynomialCoefficients,
    ThresholdDecode,
    P1CovarianceRefresh,
    P1Sample,
    GqSample,
    PreimageCorrection,
    PreimageCutoff,
    PreimagePublish,
    PreimageDeriveAttemptSeed,
    ExportCopy,
    ExportPublish,
    ExportDynamic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) enum GpuArgumentKind {
    Value,
    OptionalValue,
    U32,
    OptionalBinding,
    U64,
    U64List,
    I64,
    F64,
    RuntimeScalar,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct GpuImplementation {
    pub primitive: GpuNativePrimitive,
    pub argument_kinds: Box<[GpuArgumentKind]>,
    pub output_count: usize,
}

#[cfg(feature = "gpu")]
impl GpuImplementation {
    pub(crate) fn copy() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::Copy,
            argument_kinds: Box::new([Value, U32, Value, U32, U64, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn zero() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::Zero,
            argument_kinds: Box::new([Value, U32, U64, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn ntt(inverse: bool) -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: if inverse {
                GpuNativePrimitive::InverseNtt
            } else {
                GpuNativePrimitive::ForwardNtt
            },
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_add_sub(subtract: bool) -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: if subtract {
                GpuNativePrimitive::MatrixSub
            } else {
                GpuNativePrimitive::MatrixAdd
            },
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_mul(accumulate: bool) -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: if accumulate {
                GpuNativePrimitive::MatrixMulAccumulate
            } else {
                GpuNativePrimitive::MatrixMul
            },
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn gadget_decompose_coeff() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::GadgetDecomposeCoeff,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn gadget_decompose_small_balanced() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::GadgetDecomposeSmallBalanced,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn modulus_switch() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::ModulusSwitch,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn crt_convert() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::CrtConvert,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn centered_round_divide() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::CenteredRoundDivide,
            argument_kinds: Box::new([Value, U32, Value, U32, U64List, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn centered_round_divide_dynamic() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::CenteredRoundDivideDynamic,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn rns_mod_up() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::RnsModUp,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn rns_mod_down() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::RnsModDown,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, U64List, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn block_mod_switch() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::BlockModSwitch,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, U64List, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn crt_recompose_level() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::CrtRecomposeLevel,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, U64List, U64List, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn compact_pack() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::CompactPack,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, U64List, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn compact_pack_per_crt_limb() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::CompactPackPerCrtLimb,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U64, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn hash_sample() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::HashSample,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_transpose() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixTranspose,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_tensor() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixTensor,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_scale() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixScale,
            argument_kinds: Box::new([Value, U32, Value, U32, U64List, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_scale_dynamic() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixScaleDynamic,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_indexed_copy() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixIndexedCopy,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_slice_dynamic() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixSliceDynamic,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, Value, U32, Value, U32, Value, U32,
                U32, U32, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn ring_automorphism() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::RingAutomorphism,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn lift_integer_constant() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::LiftIntegerConstant,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn polynomial_values() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::PolynomialValues,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn extract_coefficient() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::ExtractCoefficient,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn pack_polynomial_coefficients() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::PackPolynomialCoefficients,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn threshold_decode() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::ThresholdDecode,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, Value, U32, Value, U32, U32, U32,
                U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn sample(gaussian: bool) -> Self {
        use GpuArgumentKind::{F64, U32, U64, Value};
        Self {
            primitive: if gaussian {
                GpuNativePrimitive::SampleGaussian
            } else {
                GpuNativePrimitive::SampleUniform
            },
            argument_kinds: Box::new([Value, U32, Value, U32, F64, U64, U64, U64, U64, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn sample_bit() -> Self {
        let mut sample = Self::sample(false);
        sample.primitive = GpuNativePrimitive::SampleBit;
        sample
    }

    pub(crate) fn polynomial_from_values() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::PolynomialFromValues,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn real_operation() -> Self {
        use GpuArgumentKind::{F64, OptionalBinding, OptionalValue, U32, Value};
        Self {
            primitive: GpuNativePrimitive::RealOperation,
            argument_kinds: Box::new([
                U32,
                Value,
                U32,
                OptionalValue,
                U32,
                OptionalValue,
                U32,
                Value,
                U32,
                F64,
                U32,
                OptionalBinding,
                OptionalBinding,
                U32,
            ]),
            output_count: 1,
        }
    }

    /// Plan-global resource ID precedes all physical arguments. The P1
    /// workspace is prepared before Graph construction and shared by the
    /// refresh and sample operations in one retry body.
    pub(crate) fn p1_covariance_refresh() -> Self {
        use GpuArgumentKind::{F64, U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::P1CovarianceRefresh,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, F64, F64, F64, U64, U32, U32, U32, U32,
                U32, U32, U32, U32,
            ]),
            output_count: 0,
        }
    }

    pub(crate) fn p1_sample() -> Self {
        use GpuArgumentKind::{F64, U32, Value};
        Self {
            primitive: GpuNativePrimitive::P1Sample,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, F64, F64, U32, U32, U32, U32, U32, U32,
                U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn gq_sample() -> Self {
        use GpuArgumentKind::{F64, U32, Value};
        Self {
            primitive: GpuNativePrimitive::GqSample,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, U32, F64, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn preimage_correction() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::PreimageCorrection,
            argument_kinds: Box::new([
                Value, U32, Value, U32, Value, U32, Value, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn preimage_cutoff() -> Self {
        use GpuArgumentKind::{U32, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::PreimageCutoff,
            argument_kinds: Box::new([
                U32, Value, U32, Value, U32, Value, U32, U64List, U32, U32, U32, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn preimage_publish() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::PreimagePublish,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn preimage_derive_attempt_seed() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::PreimageDeriveAttemptSeed,
            argument_kinds: Box::new([Value, U32, Value, U32, U64, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    /// Expand one compact sign/magnitude fragment into a preallocated
    /// coefficient-form matrix. Its NTT and multiplication are separate ops.
    pub(crate) fn expand_compact() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::ExpandCompact,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_copy_view() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixCopyView,
            argument_kinds: Box::new([Value, U32, Value, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn matrix_mul_transpose_rhs() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixMulTransposeRhs,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U32, U32, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn identity_fill() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::IdentityFill,
            argument_kinds: Box::new([Value, U32, U64, U64, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn gadget_fill() -> Self {
        use GpuArgumentKind::{U32, U64, U64List, Value};
        Self {
            primitive: GpuNativePrimitive::GadgetFill,
            argument_kinds: Box::new([Value, U32, U64, U32, U64List, U64, U32]),
            output_count: 1,
        }
    }

    pub(crate) fn branch_if() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::BranchIf,
            argument_kinds: Box::new([Value, U32, U32]),
            output_count: 0,
        }
    }

    pub(crate) fn loop_while() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::LoopWhile,
            argument_kinds: Box::new([Value, U32, Value, U32, Value, U32, U64, U32, U32, U32]),
            output_count: 0,
        }
    }

    pub(crate) fn integer_operation() -> Self {
        use GpuArgumentKind::{OptionalBinding, OptionalValue, U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::IntegerOperation,
            argument_kinds: Box::new([
                U32,
                Value,
                U32,
                Value,
                U32,
                OptionalValue,
                U32,
                OptionalValue,
                U32,
                Value,
                U32,
                U64,
                U32,
                U32,
                OptionalBinding,
                OptionalBinding,
                U32,
            ]),
            output_count: 1,
        }
    }

    /// Copy one contiguous physical part into one mapped slot. Binding IDs
    /// are plan-global; the native builder patches both addresses on replay.
    pub(crate) fn export_copy() -> Self {
        use GpuArgumentKind::{U32, U64, Value};
        Self {
            primitive: GpuNativePrimitive::ExportCopy,
            argument_kinds: Box::new([Value, U32, U32, U64, U32, U32]),
            output_count: 0,
        }
    }

    /// Publish readiness only after the copy node has completed. The offset
    /// names a raw staging file range, not a canonical artifact byte offset.
    pub(crate) fn export_publish() -> Self {
        use GpuArgumentKind::{U32, U64};
        Self {
            primitive: GpuNativePrimitive::ExportPublish,
            argument_kinds: Box::new([U32, U64, U64, U64, U32, U32, U32]),
            output_count: 0,
        }
    }

    pub(crate) fn export_dynamic() -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::ExportDynamic,
            argument_kinds: Box::new([U32, Value, U32, Value, U32, U32, U32, U32, U32, U32, U32]),
            output_count: 0,
        }
    }
}

/// A process-local table used by both candidate selection and native lowering.
/// Keeping the operation kind with the ID prevents a replay from silently
/// dispatching an opaque number through a different emitter catalogue.
#[derive(Clone, Debug, Default)]
#[cfg(feature = "gpu")]
pub(crate) struct GpuImplementationRegistry {
    entries: Vec<GpuImplementation>,
    by_kind: BTreeMap<GpuNativePrimitive, GpuImplId>,
}

#[cfg(feature = "gpu")]
impl GpuImplementationRegistry {
    /// Select the native emitter from the same registry used at Graph build.
    /// Unsupported effective operations are a plan error, never a request to
    /// run a legacy dispatcher or a host fallback.
    pub(crate) fn select_effective(
        &mut self,
        operation: EffectiveGpuOperation,
    ) -> Result<GpuImplId, &'static str> {
        let implementation = match operation {
            EffectiveGpuOperation::MatrixAdd => GpuImplementation::matrix_add_sub(false),
            EffectiveGpuOperation::MatrixSubtract => GpuImplementation::matrix_add_sub(true),
            EffectiveGpuOperation::MatrixMultiply => GpuImplementation::matrix_mul(false),
            EffectiveGpuOperation::MatrixMulAccumulate => GpuImplementation::matrix_mul(true),
            EffectiveGpuOperation::GadgetDecompose => GpuImplementation::gadget_decompose_coeff(),
            EffectiveGpuOperation::ModulusSwitch => GpuImplementation::modulus_switch(),
            EffectiveGpuOperation::UniformResidueSample |
            EffectiveGpuOperation::UniformIntervalSample => GpuImplementation::sample(false),
            EffectiveGpuOperation::GaussianSample => GpuImplementation::sample(true),
            _ => return Err("effective GPU operation has no direct native implementation"),
        };
        self.register(implementation)
    }

    pub(crate) fn register(
        &mut self,
        implementation: GpuImplementation,
    ) -> Result<GpuImplId, &'static str> {
        if let Some(&id) = self.by_kind.get(&implementation.primitive) {
            if self.entries[id.0 as usize] != implementation {
                return Err("GPU primitive registered with conflicting argument schema");
            }
            return Ok(id);
        }
        let id = GpuImplId(
            u32::try_from(self.entries.len()).map_err(|_| "too many GPU native implementations")?,
        );
        self.by_kind.insert(implementation.primitive, id);
        self.entries.push(implementation);
        Ok(id)
    }

    pub(crate) fn resolve(&self, id: GpuImplId) -> Result<&GpuImplementation, &'static str> {
        self.entries.get(id.0 as usize).ok_or("GPU implementation ID is not registered")
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) enum KernelArg {
    Value(PhysicalValueId),
    OptionalValue(Option<PhysicalValueId>),
    U32(u32),
    OptionalBinding(Option<u32>),
    U64(u64),
    U64List(Box<[u64]>),
    I64(i64),
    F64(f64),
    RuntimeScalar(u32),
}

/// A fixed native launch and its direct dependency edges within one region.
/// A Value argument expands through the implementation registry at plan time.
#[derive(Clone, Debug, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct CompiledGpuOp {
    pub implementation: GpuImplId,
    pub arguments: Box<[KernelArg]>,
    pub outputs: Box<[PhysicalValueId]>,
    pub device: i32,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub predecessors: Box<[u32]>,
    /// Direct nodes in the CUDA conditional body. Dependencies use indices
    /// local to this body; nested bodies follow the same schema recursively.
    pub body: Option<Box<[CompiledGpuOp]>>,
}

#[cfg(feature = "gpu")]
impl CompiledGpuOp {
    pub(crate) fn validate_in_sequence(
        &self,
        index: usize,
        value_count: usize,
        registry: &GpuImplementationRegistry,
    ) -> Result<(), &'static str> {
        let implementation = registry.resolve(self.implementation)?;
        let control = matches!(
            implementation.primitive,
            GpuNativePrimitive::BranchIf | GpuNativePrimitive::LoopWhile
        );
        if control != self.body.is_some() {
            return Err("GPU conditional operation has no matching body");
        }
        if self.arguments.len() != implementation.argument_kinds.len() ||
            self.outputs.len() != implementation.output_count ||
            self.arguments.iter().zip(&implementation.argument_kinds).any(|(argument, kind)| {
                !matches!(
                    (argument, kind),
                    (KernelArg::Value(_), GpuArgumentKind::Value) |
                        (KernelArg::OptionalValue(_), GpuArgumentKind::OptionalValue) |
                        (KernelArg::U32(_), GpuArgumentKind::U32) |
                        (KernelArg::OptionalBinding(_), GpuArgumentKind::OptionalBinding) |
                        (KernelArg::U64(_), GpuArgumentKind::U64) |
                        (KernelArg::U64List(_), GpuArgumentKind::U64List) |
                        (KernelArg::I64(_), GpuArgumentKind::I64) |
                        (KernelArg::F64(_), GpuArgumentKind::F64) |
                        (KernelArg::RuntimeScalar(_), GpuArgumentKind::RuntimeScalar)
                )
            })
        {
            return Err("GPU operation arguments do not match registered implementation");
        }
        let native_builtin = matches!(
            implementation.primitive,
            GpuNativePrimitive::Copy |
                GpuNativePrimitive::Zero |
                GpuNativePrimitive::ExportCopy |
                GpuNativePrimitive::ExportPublish |
                GpuNativePrimitive::ExportDynamic |
                GpuNativePrimitive::BranchIf |
                GpuNativePrimitive::LoopWhile
        );
        if !native_builtin && (self.grid.contains(&0) || self.block.contains(&0)) {
            return Err("empty GPU launch geometry");
        }
        if self.predecessors.iter().any(|predecessor| *predecessor as usize >= index) {
            return Err("GPU operation has a cyclic or out-of-range predecessor");
        }
        if let Some(body) = &self.body {
            for (body_index, nested) in body.iter().enumerate() {
                nested.validate_in_sequence(body_index, value_count, registry)?;
            }
        }
        if self.arguments.iter().any(|argument| match argument {
            KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) => {
                id.0 as usize >= value_count
            }
            _ => false,
        }) || self.outputs.iter().any(|id| id.0 as usize >= value_count)
        {
            return Err("GPU operation references an unknown physical value");
        }
        Ok(())
    }
}

/// One immutable direct-node region. The runtime supplies fresh bound owners
/// for each sequential execute; only physical metadata and implementation IDs
/// are frozen here. No native pointer or mapped write slot is cached in a plan.
#[derive(Clone, Debug)]
#[cfg(feature = "gpu")]
pub(crate) struct CompiledGpuProgram {
    pub values: Box<[PhysicalValue]>,
    pub implementations: GpuImplementationRegistry,
    pub operations: Box<[CompiledGpuOp]>,
    /// Dense, plan-global binding indices used by native patch records.
    pub bindings: Box<[GpuBindingSource]>,
    pub export_slots: Box<[GpuExportSlotRange]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) enum GpuBindingSource {
    PhysicalPart {
        value: PhysicalValueId,
        part: u32,
        /// A matrix limb within this part, or zero for a flat allocation.
        limb: u32,
    },
    ExportSlotPayload {
        slot: usize,
    },
    ExportSlotHeader {
        slot: usize,
    },
    PreparedWorkspace {
        kind: GpuPreparedWorkspaceKind,
        resource_id: u32,
        component: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[cfg(feature = "gpu")]
pub(crate) enum GpuPreparedWorkspaceKind {
    P1,
    Gq,
    Cutoff,
    DynamicExport,
}

#[cfg(feature = "gpu")]
impl CompiledGpuProgram {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        let mut export_copies = BTreeMap::<u32, usize>::new();
        let mut export_publications = BTreeSet::<u32>::new();
        for (index, operation) in self.operations.iter().enumerate() {
            operation.validate_in_sequence(index, self.values.len(), &self.implementations)?;
            let kind = self.implementations.resolve(operation.implementation)?.primitive;
            let binding = |id: u32| {
                self.bindings
                    .get(id as usize)
                    .copied()
                    .ok_or("GPU operation binding index is not planned")
            };
            match (kind, operation.arguments.as_ref()) {
                (
                    GpuNativePrimitive::ExportCopy,
                    [
                        KernelArg::Value(value),
                        KernelArg::U32(part),
                        KernelArg::U32(slot),
                        KernelArg::U64(_),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(slot_binding),
                    ],
                ) => {
                    if binding(*source_binding)? !=
                        (GpuBindingSource::PhysicalPart { value: *value, part: *part, limb: 0 }) ||
                        binding(*slot_binding)? !=
                            (GpuBindingSource::ExportSlotPayload { slot: *slot as usize })
                    {
                        return Err("GPU export copy binding does not match its operands");
                    }
                    if export_copies.insert(*slot, index).is_some() {
                        return Err("GPU export slot has more than one copy producer");
                    }
                }
                (
                    GpuNativePrimitive::ExportPublish,
                    [
                        KernelArg::U32(slot),
                        KernelArg::U64(_),
                        KernelArg::U64(_),
                        KernelArg::U64(_),
                        KernelArg::U32(_),
                        KernelArg::U32(_),
                        KernelArg::U32(header_binding),
                    ],
                ) => {
                    if binding(*header_binding)? !=
                        (GpuBindingSource::ExportSlotHeader { slot: *slot as usize })
                    {
                        return Err("GPU export publication binding does not match its slot");
                    }
                    let copy = export_copies
                        .get(slot)
                        .ok_or("GPU export publication has no preceding copy")?;
                    let copy =
                        u32::try_from(*copy).map_err(|_| "GPU export copy index exceeds u32")?;
                    if !operation.predecessors.contains(&copy) || !export_publications.insert(*slot)
                    {
                        return Err("GPU export publication is unordered or duplicated");
                    }
                }
                (
                    GpuNativePrimitive::ExportDynamic,
                    [
                        KernelArg::U32(resource_id),
                        KernelArg::Value(source),
                        KernelArg::U32(source_part),
                        KernelArg::Value(occurrence),
                        KernelArg::U32(occurrence_part),
                        KernelArg::U32(table_binding),
                        KernelArg::U32(claims_binding),
                        KernelArg::U32(claim_result_binding),
                        KernelArg::U32(occurrence_binding),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(status_binding),
                    ],
                ) => {
                    let prepared = |component| GpuBindingSource::PreparedWorkspace {
                        kind: GpuPreparedWorkspaceKind::DynamicExport,
                        resource_id: *resource_id,
                        component,
                    };
                    if binding(*table_binding)? != prepared(0) ||
                        binding(*claims_binding)? != prepared(1) ||
                        binding(*claim_result_binding)? != prepared(2) ||
                        binding(*status_binding)? != prepared(3) ||
                        binding(*occurrence_binding)? !=
                            (GpuBindingSource::PhysicalPart {
                                value: *occurrence,
                                part: *occurrence_part,
                                limb: 0,
                            }) ||
                        binding(*source_binding)? !=
                            (GpuBindingSource::PhysicalPart {
                                value: *source,
                                part: *source_part,
                                limb: 0,
                            })
                    {
                        return Err("GPU dynamic export bindings disagree with its operands");
                    }
                }
                _ => {}
            }
        }
        if export_copies.keys().copied().collect::<BTreeSet<_>>() != export_publications {
            return Err("GPU export copy has no publication node");
        }
        // A later writer may reuse an export source allocation after its
        // device copy has completed. No disk completion edge is required.
        for (copy_index, copy_op) in self.operations.iter().enumerate() {
            let kind = self.implementations.resolve(copy_op.implementation)?.primitive;
            if !matches!(kind, GpuNativePrimitive::ExportCopy | GpuNativePrimitive::ExportDynamic) {
                continue;
            }
            let (source, part) = match (kind, copy_op.arguments.as_ref()) {
                (
                    GpuNativePrimitive::ExportCopy,
                    [KernelArg::Value(source), KernelArg::U32(part), ..],
                ) => (source, part),
                (
                    GpuNativePrimitive::ExportDynamic,
                    [KernelArg::U32(_), KernelArg::Value(source), KernelArg::U32(part), ..],
                ) => (source, part),
                _ => return Err("GPU export copy has no physical source"),
            };
            let storage = self.values[source.0 as usize]
                .parts
                .get(*part as usize)
                .ok_or("GPU export source part is out of range")?
                .storage;
            for writer in self.operations.iter().skip(copy_index + 1) {
                let reuses_storage = writer.outputs.iter().any(|id| {
                    self.values[id.0 as usize].parts.iter().any(|output| output.storage == storage)
                });
                if !reuses_storage {
                    continue;
                }
                let mut pending = writer.predecessors.to_vec();
                let mut visited = BTreeSet::new();
                let mut ordered = false;
                while let Some(predecessor) = pending.pop() {
                    if predecessor as usize == copy_index {
                        ordered = true;
                        break;
                    }
                    if visited.insert(predecessor) {
                        pending
                            .extend_from_slice(&self.operations[predecessor as usize].predecessors);
                    }
                }
                if !ordered {
                    return Err("GPU allocation is reused before export copy completes");
                }
            }
        }
        for source in &self.bindings {
            if let GpuBindingSource::PhysicalPart { value, part, limb } = *source {
                let physical = self
                    .values
                    .get(value.0 as usize)
                    .ok_or("GPU binding references an unknown physical value")?;
                let physical_part = physical
                    .parts
                    .get(part as usize)
                    .ok_or("GPU binding references an unknown physical part")?;
                if physical_part.view.origin.len() >= 4 &&
                    u64::from(limb) >= physical_part.view.extent[2]
                {
                    return Err("GPU binding references an unknown CRT limb");
                }
                if physical_part.view.origin.len() < 4 && limb != 0 {
                    return Err("GPU nonmatrix binding has a CRT limb");
                }
            } else if let GpuBindingSource::PreparedWorkspace { kind, component, .. } = *source {
                let valid = match kind {
                    GpuPreparedWorkspaceKind::P1 => component < 5,
                    GpuPreparedWorkspaceKind::Gq | GpuPreparedWorkspaceKind::Cutoff => {
                        component == 0
                    }
                    GpuPreparedWorkspaceKind::DynamicExport => component < 4,
                };
                if !valid {
                    return Err("GPU prepared workspace binding has an invalid component");
                }
            }
        }
        let mut next = 0usize;
        let mut sites = BTreeSet::new();
        for range in &self.export_slots {
            if !sites.insert(range.site) ||
                range.start != next ||
                range.occurrence_count == 0 ||
                range.fragments_per_occurrence == 0
            {
                return Err("compiled export slot ranges are invalid");
            }
            let expected = range
                .occurrence_count
                .checked_mul(range.fragments_per_occurrence)
                .and_then(|count| usize::try_from(count).ok())
                .and_then(|count| range.start.checked_add(count))
                .ok_or("compiled export slot count overflows")?;
            if range.end != expected {
                return Err("compiled export slot range has the wrong length");
            }
            next = range.end;
        }
        if self.bindings.iter().any(|binding| {
            matches!(binding,
            GpuBindingSource::ExportSlotPayload { slot } |
                GpuBindingSource::ExportSlotHeader { slot } if *slot >= next)
        }) {
            return Err("GPU binding references an unknown export slot");
        }
        Ok(())
    }
}

/// A fixed range of mapped write slots for one export site. Each occurrence
/// and physical fragment has its own slot, so a producer never waits for disk
/// completion or reuses a slot within an execute call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "gpu")]
pub(crate) struct GpuExportSlotRange {
    pub site: u32,
    pub occurrence_count: u64,
    pub fragments_per_occurrence: u64,
    pub start: usize,
    pub end: usize,
}

#[cfg(feature = "gpu")]
impl GpuExportSlotRange {
    pub(crate) fn index(&self, occurrence: u64, fragment: u64) -> Result<usize, &'static str> {
        if occurrence >= self.occurrence_count || fragment >= self.fragments_per_occurrence {
            return Err("GPU export occurrence or fragment exceeds its planned range");
        }
        let local = occurrence
            .checked_mul(self.fragments_per_occurrence)
            .and_then(|base| base.checked_add(fragment))
            .ok_or("GPU export slot index overflows")?;
        let index = self
            .start
            .checked_add(usize::try_from(local).map_err(|_| "GPU export slot index exceeds usize")?)
            .ok_or("GPU export slot index overflows")?;
        (index < self.end).then_some(index).ok_or("GPU export slot index exceeds reserved range")
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn reserve_gpu_export_slots(
    sites: impl IntoIterator<Item = (u32, u64, u64)>,
) -> Result<Box<[GpuExportSlotRange]>, &'static str> {
    let mut ranges = Vec::new();
    let mut seen = BTreeSet::new();
    let mut next = 0usize;
    for (site, occurrence_count, fragments_per_occurrence) in sites {
        if !seen.insert(site) || occurrence_count == 0 || fragments_per_occurrence == 0 {
            return Err("GPU export site is duplicate or has an empty slot range");
        }
        let count = occurrence_count
            .checked_mul(fragments_per_occurrence)
            .ok_or("GPU export slot count overflows")?;
        let count = usize::try_from(count).map_err(|_| "GPU export slot count exceeds usize")?;
        let end = next.checked_add(count).ok_or("GPU export slot count overflows")?;
        ranges.push(GpuExportSlotRange {
            site,
            occurrence_count,
            fragments_per_occurrence,
            start: next,
            end,
        });
        next = end;
    }
    Ok(ranges.into_boxed_slice())
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_export_slot_tests {
    use super::*;

    fn byte_value() -> PhysicalValue {
        PhysicalValue {
            ty: ConcreteWireType::Bytes { length: 8 },
            encodings: Box::new([PhysicalEncoding::Bytes]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage: StorageRef::Input(0),
                device: 0,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: Box::new([0]),
                    extent: Box::new([8]),
                    byte_strides: Box::new([1]),
                    element_bytes: 1,
                },
            }]),
            integer_ranges: BTreeMap::new(),
        }
    }

    #[test]
    fn slots_are_unique_across_sites_occurrences_and_fragments() {
        let ranges = reserve_gpu_export_slots([(7, 3, 2), (11, 1, 4)]).unwrap();
        let mut indices = std::collections::BTreeSet::new();
        for range in &ranges {
            for occurrence in 0..range.occurrence_count {
                for fragment in 0..range.fragments_per_occurrence {
                    assert!(indices.insert(range.index(occurrence, fragment).unwrap()));
                }
            }
        }
        assert_eq!(indices, (0..10).collect());
        assert!(ranges[0].index(3, 0).is_err());
        assert!(ranges[0].index(0, 2).is_err());
    }

    #[test]
    fn reservation_rejects_overflow_before_slot_allocation() {
        assert!(reserve_gpu_export_slots([(0, u64::MAX, 2)]).is_err());
        assert!(reserve_gpu_export_slots([(0, 1, 1), (0, 1, 1)]).is_err());
    }

    #[test]
    fn export_copy_precedes_ready_and_allocation_reuse() {
        let mut implementations = GpuImplementationRegistry::default();
        let copy = implementations.register(GpuImplementation::export_copy()).unwrap();
        let publish = implementations.register(GpuImplementation::export_publish()).unwrap();
        let zero = implementations.register(GpuImplementation::zero()).unwrap();
        let operations = vec![
            CompiledGpuOp {
                implementation: copy,
                arguments: Box::new([
                    KernelArg::Value(PhysicalValueId(0)),
                    KernelArg::U32(0),
                    KernelArg::U32(0),
                    KernelArg::U64(8),
                    KernelArg::U32(0),
                    KernelArg::U32(1),
                ]),
                outputs: Box::new([]),
                device: 0,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: Box::new([]),
                body: None,
            },
            CompiledGpuOp {
                implementation: publish,
                arguments: Box::new([
                    KernelArg::U32(0),
                    KernelArg::U64(0),
                    KernelArg::U64(0),
                    KernelArg::U64(8),
                    KernelArg::U32(7),
                    KernelArg::U32(1),
                    KernelArg::U32(2),
                ]),
                outputs: Box::new([]),
                device: 0,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: Box::new([0]),
                body: None,
            },
            CompiledGpuOp {
                implementation: zero,
                arguments: Box::new([
                    KernelArg::Value(PhysicalValueId(0)),
                    KernelArg::U32(0),
                    KernelArg::U64(8),
                    KernelArg::U32(0),
                ]),
                outputs: Box::new([PhysicalValueId(0)]),
                device: 0,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: Box::new([0]),
                body: None,
            },
        ];
        let program = CompiledGpuProgram {
            values: Box::new([byte_value()]),
            implementations,
            operations: operations.clone().into_boxed_slice(),
            bindings: Box::new([
                GpuBindingSource::PhysicalPart { value: PhysicalValueId(0), part: 0, limb: 0 },
                GpuBindingSource::ExportSlotPayload { slot: 0 },
                GpuBindingSource::ExportSlotHeader { slot: 0 },
            ]),
            export_slots: reserve_gpu_export_slots([(7, 1, 1)]).unwrap(),
        };
        assert!(program.validate().is_ok());
        let mut unordered = program.clone();
        unordered.operations[2].predecessors = Box::new([]);
        assert_eq!(
            unordered.validate(),
            Err("GPU allocation is reused before export copy completes")
        );
        let mut early_ready = program;
        early_ready.operations[1].predecessors = Box::new([]);
        assert_eq!(
            early_ready.validate(),
            Err("GPU export publication is unordered or duplicated")
        );
    }
}

/// Shared bounded dispatcher: all jobs returned for one owner are consumed,
/// while only distinct owners execute concurrently. The callback owns the
/// backend-specific completion/release rule.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dispatch_column_batch<D: Send, T: Send, E: Send>(
    devices: &mut [D],
    schedules: &[&GpuColumnSchedule],
    operation: impl Fn(&mut D, usize, GpuColumnJob) -> Result<T, E> + Sync,
) -> Result<Vec<(usize, GpuColumnJob, T)>, E> {
    let mut results = Vec::new();
    for wave in GpuColumnSchedule::batch_waves(schedules) {
        let by_device = devices
            .par_iter_mut()
            .enumerate()
            .map(|(device, state)| {
                wave.iter()
                    .filter(|(_, job)| job.device == device)
                    .map(|(instance, job)| {
                        operation(state, *instance, *job).map(|value| (*instance, *job, value))
                    })
                    .collect::<Result<Vec<_>, E>>()
            })
            .collect::<Result<Vec<_>, E>>()?;
        results.extend(by_device.into_iter().flatten());
    }
    results.sort_by_key(|(instance, job, _)| (*instance, job.start, job.end, job.device));
    Ok(results)
}

/// One output-port's view of a fused union job. The clipped range is in the
/// port's global column coordinates; `source_interval` and `owner_device`
/// retain the schedule's physical ownership without making it the union
/// dispatch device.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionPortJob {
    pub port: usize,
    pub clipped_range: Option<crate::gpu_column_policy::ColumnRange>,
    pub source_interval: Option<usize>,
    pub owner_device: Option<usize>,
}

/// A single primitive invocation covering one union range. Ports may be absent
/// from a range when their output is narrower; present ports retain their own
/// owner interval and clipped range.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionJob {
    pub instance: usize,
    pub logical_wave: usize,
    pub device: usize,
    pub range: crate::gpu_column_policy::ColumnRange,
    pub port_jobs: Vec<GpuFusedUnionPortJob>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuFusedUnionError {
    #[error("fused union requires at least one output port")]
    EmptyPorts,
    #[error("fused union port schedule count does not match instance count")]
    InstanceCountMismatch,
    #[error("fused union instance {instance} has no schedulable output ranges")]
    EmptyInstance { instance: usize },
    #[error("fused union ranges do not cover a contiguous global domain for instance {instance}")]
    InvalidCoverage { instance: usize },
    #[error("fused union range arithmetic overflow")]
    ArithmeticOverflow,
}

/// A compressed run of source logical waves. Its size is bounded by stored
/// schedule intervals and owner/width changes, not by the number of columns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionWaveClass {
    pub instance: usize,
    pub first_wave: usize,
    pub multiplicity: usize,
    pub active_ports: Vec<usize>,
}

/// One stored schedule interval's lazy job cursor.
struct FusedUnionIntervalCursor {
    device: usize,
    source_interval: usize,
    next_start: usize,
    end: usize,
    width: usize,
    wave_start: usize,
    emitted: usize,
}

/// Cursor over one port's jobs in global-column order. Only stored intervals
/// and one active job are retained, so width-one schedules do not allocate one
/// descriptor per column.
struct FusedUnionPortCursor<'a> {
    _schedule: &'a GpuColumnSchedule,
    intervals: Vec<FusedUnionIntervalCursor>,
    next_interval: usize,
    current: Option<(usize, GpuColumnJob)>,
}

impl<'a> FusedUnionPortCursor<'a> {
    fn new(schedule: &'a GpuColumnSchedule) -> Self {
        let mut wave_starts = vec![0; schedule.widths().len()];
        let intervals = schedule
            .intervals()
            .iter()
            .enumerate()
            .map(|(source_interval, interval)| {
                let width = schedule.widths()[interval.device];
                let count = (interval.end - interval.start).div_ceil(width);
                let wave_start = wave_starts[interval.device];
                wave_starts[interval.device] += count;
                FusedUnionIntervalCursor {
                    device: interval.device,
                    source_interval,
                    next_start: interval.start,
                    end: interval.end,
                    width,
                    wave_start,
                    emitted: 0,
                }
            })
            .collect();
        let mut cursor = Self { _schedule: schedule, intervals, next_interval: 0, current: None };
        cursor.pull_next();
        cursor
    }

    fn pull_next(&mut self) {
        loop {
            let Some(interval) = self.intervals.get_mut(self.next_interval) else {
                self.current = None;
                return;
            };
            if interval.next_start >= interval.end {
                self.next_interval += 1;
                continue;
            }
            let start = interval.next_start;
            let width = interval.width.min(interval.end - start);
            interval.next_start += width;
            let wave = interval.wave_start + interval.emitted;
            interval.emitted += 1;
            self.current = Some((
                wave,
                GpuColumnJob {
                    device: interval.device,
                    source_interval: interval.source_interval,
                    start,
                    end: start + width,
                },
            ));
            return;
        }
    }

    fn advance_if_ended(&mut self, end: usize) {
        while self.current.is_some_and(|(_, job)| job.end <= end) {
            self.pull_next();
        }
    }
}

/// Lazy pure union-job producer for production dispatch, warmup, and
/// estimation. It retains one active wave per output port and emits one union
/// segment at a time.
pub struct GpuFusedUnionJobStream<'a> {
    schedules_by_port: &'a [Vec<GpuColumnSchedule>],
    instance_end: usize,
    instance: usize,
    cursors: Vec<FusedUnionPortCursor<'a>>,
    boundary: Option<usize>,
    prepared: bool,
    finished: bool,
}

impl<'a> GpuFusedUnionJobStream<'a> {
    fn prepare_instance(&mut self) {
        while self.instance < self.instance_end {
            self.cursors = self
                .schedules_by_port
                .iter()
                .map(|port| FusedUnionPortCursor::new(&port[self.instance]))
                .collect();
            self.prepared = true;
            self.boundary = None;
            if self.cursors.iter().any(|cursor| cursor.current.is_some()) {
                return;
            }
            // Every port is a valid zero-column/host-control output.
            self.instance += 1;
            self.prepared = false;
        }
        self.finished = true;
    }

    fn next_job(&mut self) -> Result<Option<GpuFusedUnionJob>, GpuFusedUnionError> {
        if self.finished {
            return Ok(None);
        }
        if !self.prepared {
            self.prepare_instance();
            if self.finished {
                return Ok(None);
            }
        }
        let instance = self.instance;
        let start = self.boundary.unwrap_or_else(|| {
            self.cursors
                .iter()
                .filter_map(|cursor| cursor.current.map(|(_, job)| job.start))
                .min()
                .unwrap_or(0)
        });
        let end = self
            .cursors
            .iter()
            .filter_map(|cursor| {
                cursor.current.map(|(_, job)| if job.start > start { job.start } else { job.end })
            })
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        if start >= end {
            return Err(GpuFusedUnionError::InvalidCoverage { instance });
        }
        let containing = self
            .cursors
            .iter()
            .map(|cursor| {
                cursor.current.and_then(|(wave, job)| {
                    (job.start <= start && end <= job.end).then_some((wave, job))
                })
            })
            .collect::<Vec<_>>();
        let device = containing
            .iter()
            .filter_map(|job| job.map(|(_, job)| job.device))
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let logical_wave = containing
            .iter()
            .filter_map(|job| job.map(|(wave, _)| wave))
            .max()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let port_jobs = containing
            .into_iter()
            .enumerate()
            .map(|(port, job)| match job {
                Some((_, job)) => GpuFusedUnionPortJob {
                    port,
                    clipped_range: Some(crate::gpu_column_policy::ColumnRange { start, end }),
                    source_interval: Some(job.source_interval),
                    owner_device: Some(job.device),
                },
                None => GpuFusedUnionPortJob {
                    port,
                    clipped_range: None,
                    source_interval: None,
                    owner_device: None,
                },
            })
            .collect();
        for cursor in &mut self.cursors {
            cursor.advance_if_ended(end);
        }
        self.boundary = Some(end);
        if self.cursors.iter().all(|cursor| cursor.current.is_none()) {
            self.instance += 1;
            self.boundary = None;
            self.prepared = false;
        }
        Ok(Some(GpuFusedUnionJob {
            instance,
            logical_wave,
            device,
            range: crate::gpu_column_policy::ColumnRange { start, end },
            port_jobs,
        }))
    }
}

impl Iterator for GpuFusedUnionJobStream<'_> {
    type Item = Result<GpuFusedUnionJob, GpuFusedUnionError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_job() {
            Ok(Some(job)) => Some(Ok(job)),
            Ok(None) => None,
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

fn jobs_over_ranges(
    schedule: &GpuColumnSchedule,
    ranges: &[(usize, usize)],
) -> Vec<(usize, GpuColumnJob)> {
    let mut wave_starts = vec![0; schedule.widths().len()];
    let mut jobs = Vec::new();
    for (source_interval, interval) in schedule.intervals().iter().enumerate() {
        let width = schedule.widths()[interval.device];
        let wave_start = wave_starts[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        wave_starts[interval.device] += count;
        for &(range_start, range_end) in ranges {
            if range_start >= range_end ||
                interval.end <= range_start ||
                interval.start >= range_end
            {
                continue;
            }
            let offset = range_start.saturating_sub(interval.start) / width * width;
            let mut start = interval.start + offset;
            while start < range_end && start < interval.end {
                let end = start + width.min(interval.end - start);
                if end > range_start {
                    jobs.push((
                        wave_start + (start - interval.start) / width,
                        GpuColumnJob { device: interval.device, source_interval, start, end },
                    ));
                }
                if end == interval.end {
                    break;
                }
                start = end;
            }
        }
    }
    jobs.sort_unstable_by_key(|(_, job)| (job.start, job.end, job.device, job.source_interval));
    jobs.dedup();
    jobs
}

pub(crate) fn fused_union_jobs_for_wave(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instance: usize,
    logical_wave: usize,
) -> Result<Vec<GpuFusedUnionJob>, GpuFusedUnionError> {
    let mut candidate_ranges = schedules_by_port
        .iter()
        .flat_map(|port| port[instance].wave_jobs(logical_wave))
        .map(|job| (job.start, job.end))
        .filter(|(start, end)| start < end)
        .collect::<Vec<_>>();
    if candidate_ranges.is_empty() {
        return Ok(Vec::new());
    }
    candidate_ranges.sort_unstable();
    candidate_ranges.dedup();
    let port_jobs = schedules_by_port
        .iter()
        .map(|port| jobs_over_ranges(&port[instance], &candidate_ranges))
        .collect::<Vec<_>>();
    let mut boundaries = port_jobs
        .iter()
        .flat_map(|jobs| jobs.iter().flat_map(|(_, job)| [job.start, job.end]))
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut output = Vec::new();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if start >= end {
            continue;
        }
        let containing = port_jobs
            .iter()
            .map(|jobs| jobs.iter().find(|(_, job)| job.start <= start && end <= job.end).copied())
            .collect::<Vec<_>>();
        let max_wave = containing.iter().filter_map(|job| job.map(|(wave, _)| wave)).max();
        if max_wave != Some(logical_wave) {
            continue;
        }
        let device = containing
            .iter()
            .filter_map(|job| job.map(|(_, job)| job.device))
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let port_jobs = containing
            .into_iter()
            .enumerate()
            .map(|(port, job)| match job {
                Some((_, job)) => GpuFusedUnionPortJob {
                    port,
                    clipped_range: Some(crate::gpu_column_policy::ColumnRange { start, end }),
                    source_interval: Some(job.source_interval),
                    owner_device: Some(job.device),
                },
                None => GpuFusedUnionPortJob {
                    port,
                    clipped_range: None,
                    source_interval: None,
                    owner_device: None,
                },
            })
            .collect();
        output.push(GpuFusedUnionJob {
            instance,
            logical_wave,
            device,
            range: crate::gpu_column_policy::ColumnRange { start, end },
            port_jobs,
        });
    }
    Ok(output)
}

/// Construct a lazy union stream. Validation is eager; job generation is
/// deferred until the caller advances the iterator.
pub fn fused_union_jobs_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<GpuFusedUnionJobStream<'_>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    Ok(GpuFusedUnionJobStream {
        schedules_by_port,
        instance_end: instances,
        instance: 0,
        cursors: Vec::new(),
        boundary: None,
        prepared: false,
        finished: false,
    })
}

/// Lazy logical-wave producer. Each iterator step returns one logical wave
/// across all instances. It reconstructs only the requested wave from stored
/// interval cursors, preserving fleet cross-instance grouping without treating
/// global-column order as logical-wave order or materializing all union jobs.
pub struct GpuFusedUnionWavesLazy<'a> {
    schedules_by_port: &'a [Vec<GpuColumnSchedule>],
    instances: usize,
    next_wave: usize,
    wave_count: usize,
}

impl Iterator for GpuFusedUnionWavesLazy<'_> {
    type Item = Result<Vec<GpuFusedUnionJob>, GpuFusedUnionError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_wave >= self.wave_count {
            return None;
        }
        let logical_wave = self.next_wave;
        self.next_wave += 1;
        let mut wave = Vec::new();
        for instance in 0..self.instances {
            match fused_union_jobs_for_wave(self.schedules_by_port, instance, logical_wave) {
                Ok(mut jobs) => wave.append(&mut jobs),
                Err(error) => return Some(Err(error)),
            }
        }
        Some(Ok(wave))
    }
}

/// Construct a lazy cross-instance logical-wave iterator.
pub fn fused_union_waves_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<GpuFusedUnionWavesLazy<'_>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    let wave_count = schedules_by_port
        .iter()
        .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
        .max()
        .unwrap_or(0);
    Ok(GpuFusedUnionWavesLazy { schedules_by_port, instances, next_wave: 0, wave_count })
}

/// Return compressed source-wave classes for warmup and estimation. This does
/// not inspect individual column tiles.
pub fn fused_union_wave_classes(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<Vec<GpuFusedUnionWaveClass>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    let mut classes = Vec::new();
    for instance in 0..instances {
        let source =
            schedules_by_port.iter().map(|port| port[instance].wave_classes()).collect::<Vec<_>>();
        let mut boundaries = Vec::new();
        for port in &source {
            for class in port {
                let end = class
                    .first_wave
                    .checked_add(class.multiplicity)
                    .ok_or(GpuFusedUnionError::ArithmeticOverflow)?;
                boundaries.extend([class.first_wave, end]);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        for pair in boundaries.windows(2) {
            let (first_wave, end_wave) = (pair[0], pair[1]);
            let active_ports = source
                .iter()
                .enumerate()
                .filter_map(|(port, classes)| {
                    classes
                        .iter()
                        .any(|class| {
                            let end = class.first_wave.saturating_add(class.multiplicity);
                            class.first_wave <= first_wave && first_wave < end
                        })
                        .then_some(port)
                })
                .collect::<Vec<_>>();
            if !active_ports.is_empty() {
                classes.push(GpuFusedUnionWaveClass {
                    instance,
                    first_wave,
                    multiplicity: end_wave - first_wave,
                    active_ports,
                });
            }
        }
    }
    Ok(classes)
}

/// Build the sole pure representation of a fused multi-output union for small
/// or debugging callers. Production paths should use
/// [`fused_union_waves_lazy`] or [`fused_union_jobs_lazy`] so generated jobs do
/// not accumulate in memory.
pub fn build_fused_union_jobs(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<Vec<Vec<GpuFusedUnionJob>>, GpuFusedUnionError> {
    let mut jobs_by_instance = vec![Vec::new(); instances];
    for job in fused_union_jobs_lazy(schedules_by_port, instances)? {
        let job = job?;
        jobs_by_instance[job.instance].push(job);
    }
    Ok(jobs_by_instance)
}

/// Count primitive invocations without retaining the generated union jobs.
/// This is intended for warmup/estimation paths that need an exact count but
/// must remain bounded by stored intervals and active cursors in memory.
pub fn fused_union_invocation_count_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<usize, GpuFusedUnionError> {
    let mut count = 0usize;
    for job in fused_union_jobs_lazy(schedules_by_port, instances)? {
        job?;
        count = count.checked_add(1).ok_or(GpuFusedUnionError::ArithmeticOverflow)?;
    }
    Ok(count)
}

/// Group union jobs by their source logical wave. Jobs on distinct devices in
/// one returned wave may run concurrently; callers execute jobs sharing a
/// device sequentially in the returned order.
pub fn fused_union_waves(
    jobs_by_instance: &[Vec<GpuFusedUnionJob>],
) -> Vec<Vec<(usize, GpuFusedUnionJob)>> {
    let mut waves = Vec::<Vec<(usize, GpuFusedUnionJob)>>::new();
    for (instance, jobs) in jobs_by_instance.iter().enumerate() {
        for job in jobs {
            if waves.len() <= job.logical_wave {
                waves.resize_with(job.logical_wave + 1, Vec::new);
            }
            waves[job.logical_wave].push((instance, job.clone()));
        }
    }
    waves
}

pub fn fused_union_invocation_count(jobs_by_instance: &[Vec<GpuFusedUnionJob>]) -> usize {
    jobs_by_instance.iter().map(Vec::len).sum()
}

pub type LayoutId = u32;

/// Compact lexical scope identity shared by warmup and execution. Concrete
/// instance shapes are checked separately; loop indices are never plan keys.
pub fn scope_shape_class(
    validated: &mxx_ir_core::ValidatedGraph,
    scope: &mxx_ir_core::graph::FrozenGraphScopeId,
) -> Result<u64, GpuPlanError> {
    validated
        .scopes
        .keys()
        .position(|candidate| candidate == scope)
        .map(|index| index as u64)
        .ok_or(GpuPlanError::ContractMismatch)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuDeviceBudget {
    pub device: usize,
    pub device_bytes: u64,
    pub pinned_host_bytes: u64,
    pub host_bytes: u64,
}

/// Resolve a physical owner in one unambiguous logical device ordering.
pub fn logical_device_for_physical(devices: &[i32], physical: i32) -> Result<usize, String> {
    let mut seen = BTreeSet::new();
    if devices.iter().any(|device| *device < 0 || !seen.insert(*device)) {
        return Err("invalid or duplicate physical GPU owner".into());
    }
    devices
        .iter()
        .position(|device| *device == physical)
        .ok_or_else(|| format!("physical GPU {physical} is absent from the device mapping"))
}

/// Project fleet budgets into a worker's physical-owner order. Budget device
/// fields are logical indices in their respective fleet, never CUDA ordinals.
pub fn project_gpu_device_budgets(
    fleet: &[i32],
    worker: &[i32],
    budgets: &[GpuDeviceBudget],
) -> Result<Vec<GpuDeviceBudget>, String> {
    if fleet.is_empty() || worker.is_empty() || budgets.len() != fleet.len() {
        return Err("GPU budget and device mapping lengths disagree".into());
    }
    logical_device_for_physical(fleet, fleet[0])?;
    logical_device_for_physical(worker, worker[0])?;
    let mut by_logical = vec![None; fleet.len()];
    for budget in budgets {
        let entry = by_logical
            .get_mut(budget.device)
            .ok_or_else(|| "GPU budget logical index is out of range".to_owned())?;
        if entry.replace(budget).is_some() {
            return Err("duplicate GPU budget logical index".into());
        }
    }
    worker
        .iter()
        .enumerate()
        .map(|(local, physical)| {
            let global = logical_device_for_physical(fleet, *physical)?;
            let mut budget =
                by_logical[global].ok_or_else(|| "missing GPU budget".to_owned())?.clone();
            budget.device = local;
            Ok(budget)
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuPlanContract {
    pub graph_specification_hash: [u8; 32],
    pub backend_identity: String,
    pub logical_to_physical_devices: Vec<usize>,
    pub device_budgets: Vec<GpuDeviceBudget>,
    pub shape_contract_hash: [u8; 32],
    pub backend_revision: String,
}

impl GpuPlanContract {
    pub fn validate(&self) -> Result<(), GpuPlanError> {
        if self.logical_to_physical_devices.is_empty() {
            return Err(GpuPlanError::EmptyDeviceMapping);
        }
        if self.device_budgets.len() != self.logical_to_physical_devices.len() {
            return Err(GpuPlanError::DeviceBudgetCount {
                devices: self.logical_to_physical_devices.len(),
                budgets: self.device_budgets.len(),
            });
        }
        let mut physical = BTreeSet::new();
        for (logical, (&mapped, budget)) in
            self.logical_to_physical_devices.iter().zip(&self.device_budgets).enumerate()
        {
            if !physical.insert(mapped) {
                return Err(GpuPlanError::DuplicatePhysicalDevice(mapped));
            }
            if budget.device != logical {
                return Err(GpuPlanError::BudgetDeviceMismatch {
                    logical,
                    budget_device: budget.device,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuLayout {
    pub id: LayoutId,
    pub columns: usize,
    pub rows: usize,
    pub ring_dimension: usize,
    pub representation: String,
    /// Deterministic rotation for sibling instance classes. Zero preserves
    /// resident input ownership; newly placed small values may use one.
    pub instance_device_stride: usize,
    /// Logical owner intervals; values are intentionally not physical shards.
    pub owner_intervals: Vec<GpuColumnInterval>,
}

impl GpuLayout {
    pub fn schedule(
        &self,
        widths: &[usize],
        instance: usize,
    ) -> Result<GpuColumnSchedule, GpuScheduleError> {
        if widths.is_empty() {
            return Err(GpuScheduleError::EmptyFleet);
        }
        let offset = ((instance as u128 * self.instance_device_stride as u128) %
            widths.len() as u128) as usize;
        // Layouts loaded from older/value-only plans may omit explicit
        // ownership.  Production uses the deterministic balanced mapper for
        // that representation; schedule construction must use the same map
        // instead of handing an empty interval list to `GpuColumnSchedule`.
        // Keeping this lowering here also makes every caller (warmup,
        // fixed dispatch, and tests) observe identical owner geometry.
        let owner_intervals = if self.owner_intervals.is_empty() && self.columns > 0 {
            let base = self.columns / widths.len();
            let remainder = self.columns % widths.len();
            let mut start = 0usize;
            (0..widths.len())
                .filter_map(|device| {
                    let length = base + usize::from(device < remainder);
                    let interval = (length > 0).then_some(GpuColumnInterval {
                        device,
                        start,
                        end: start + length,
                    });
                    start += length;
                    interval
                })
                .collect::<Vec<_>>()
        } else {
            self.owner_intervals.clone()
        };
        let owners = owner_intervals
            .iter()
            .map(|interval| GpuColumnInterval {
                device: (interval.device + offset) % widths.len(),
                ..*interval
            })
            .collect();
        GpuColumnSchedule::new_for_instance(self.columns, widths.to_vec(), owners, instance, offset)
    }
    fn validate(&self, device_count: usize) -> Result<(), GpuPlanError> {
        if self.representation.is_empty() {
            return Err(GpuPlanError::EmptyLayoutRepresentation { layout: self.id });
        }
        let widths = vec![usize::MAX; device_count];
        GpuColumnSchedule::new(self.columns, widths, self.owner_intervals.clone())
            .map(|_| ())
            .map_err(GpuPlanError::InvalidLayoutSchedule)
    }
}

/// Shape and provenance contract for one output port of a fused operation.
/// Ports are validated independently: a multi-output operation may legitimately
/// produce different shapes and representations in one job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuOutputPortContract {
    pub port: u32,
    pub layout: LayoutId,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: String,
    /// The logical source layout inherited by this output, when it is a view or
    /// a mapped result. This is metadata only; it does not retain a runtime
    /// owner or pointer.
    pub source_layout: Option<LayoutId>,
}

impl FrozenGpuPlan {
    /// Validate per-port shape, representation, and source-layout metadata for
    /// a node. This helper is intentionally separate from owner scheduling so a
    /// fused operation is charged once while every output port is checked.
    pub fn validate_output_ports(
        &self,
        key: GpuExecutionSiteKey,
        ports: &[GpuOutputPortContract],
    ) -> Result<(), GpuPlanError> {
        let index = FrozenGpuPlanIndex::build(self)?;
        let node = index.node_choice(self, key).ok_or(GpuPlanError::MissingNodeSite(key))?;
        if ports.len() != node.output_layouts.len() {
            return Err(GpuPlanError::OutputPortCount {
                site: key,
                expected: node.output_layouts.len(),
                actual: ports.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for port in ports {
            if !seen.insert(port.port) {
                return Err(GpuPlanError::DuplicateOutputPort { site: key, port: port.port });
            }
            let port_index = usize::try_from(port.port)
                .map_err(|_| GpuPlanError::InvalidOutputPort { site: key, port: port.port })?;
            let expected_layout = node
                .output_layouts
                .get(port_index)
                .ok_or(GpuPlanError::InvalidOutputPort { site: key, port: port.port })?;
            if *expected_layout != port.layout {
                return Err(GpuPlanError::OutputPortLayoutMismatch {
                    site: key,
                    port: port.port,
                    expected: *expected_layout,
                    actual: port.layout,
                });
            }
            let layout = index
                .layout(self, port.layout)
                .ok_or(GpuPlanError::UnknownLayout { site: key, layout: port.layout })?;
            if (layout.rows, layout.columns, layout.ring_dimension) !=
                (port.rows, port.columns, port.ring_dimension)
            {
                return Err(GpuPlanError::OutputPortShapeMismatch { site: key, port: port.port });
            }
            if port.representation.is_empty() || port.representation != layout.representation {
                return Err(GpuPlanError::OutputPortRepresentationMismatch {
                    site: key,
                    port: port.port,
                });
            }
            if let Some(source) = port.source_layout {
                let source_layout =
                    index.layout(self, source).ok_or(GpuPlanError::UnknownSourceLayout {
                        site: key,
                        port: port.port,
                        source_layout: source,
                    })?;
                if source_layout.representation.is_empty() {
                    return Err(GpuPlanError::InvalidSourceLayout { site: key, port: port.port });
                }
            }
        }
        if seen.len() != node.output_layouts.len() {
            return Err(GpuPlanError::InvalidOutputPorts { site: key });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct GpuLoopSiteKey {
    pub site: u64,
    pub shape_class: u64,
    /// Dense plan-local ordinal of this lexical site's finite invocation path.
    /// Distinct invocations may share one scope and semantic shape class while
    /// binding different outer loop indices.
    pub instance_class: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuLoopChoice {
    pub key: GpuLoopSiteKey,
    pub loop_count: usize,
    pub wave_instances: usize,
    pub tail_instances: usize,
}

impl GpuLoopChoice {
    fn validate(&self) -> Result<(), GpuPlanError> {
        if self.wave_instances == 0 {
            return Err(GpuPlanError::ZeroLoopWave { site: self.key });
        }
        if self.loop_count == 0 {
            if self.tail_instances != 0 {
                return Err(GpuPlanError::InvalidTail { site: self.key, tail: self.tail_instances });
            }
        } else if self.tail_instances >= self.wave_instances {
            return Err(GpuPlanError::InvalidTail { site: self.key, tail: self.tail_instances });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct GpuExecutionSiteKey {
    pub site: u64,
    pub shape_class: u64,
    pub instance_class: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuNodeChoice {
    pub key: GpuExecutionSiteKey,
    pub loop_site: Option<GpuLoopSiteKey>,
    pub operation_identity: [u8; 32],
    pub effective_operation: EffectiveGpuOperation,
    pub column_capability: ColumnCapability,
    pub output_layouts: Vec<LayoutId>,
    /// Fixed local tile width for each logical device. Zero means inactive.
    pub columns_per_job: Vec<usize>,
    pub implementation_variant: String,
    /// Frozen retry bound for preimage operations. Other operations leave this
    /// unset; runtime execution must not rediscover it from available memory.
    pub preimage_max_attempts: Option<usize>,
}

impl GpuNodeChoice {
    fn validate(
        &self,
        contract: &GpuPlanContract,
        layouts: &BTreeSet<LayoutId>,
    ) -> Result<(), GpuPlanError> {
        if self.effective_operation == EffectiveGpuOperation::Unsupported {
            return Err(GpuPlanError::UnsupportedOperation { site: self.key });
        }
        if self.columns_per_job.len() != contract.logical_to_physical_devices.len() {
            return Err(GpuPlanError::WidthCount {
                site: self.key,
                expected: contract.logical_to_physical_devices.len(),
                actual: self.columns_per_job.len(),
            });
        }
        if self.output_layouts.is_empty() {
            return Err(GpuPlanError::NoOutputLayout { site: self.key });
        }
        if let Some(layout) = self.output_layouts.iter().find(|layout| !layouts.contains(layout)) {
            return Err(GpuPlanError::UnknownLayout { site: self.key, layout: *layout });
        }
        if self.implementation_variant.is_empty() {
            return Err(GpuPlanError::EmptyImplementationVariant { site: self.key });
        }
        if self.column_capability !=
            capability_for_effective_operation(self.effective_operation, &[])
        {
            return Err(GpuPlanError::CapabilityMismatch { site: self.key });
        }
        if self.preimage_max_attempts == Some(0) {
            return Err(GpuPlanError::InvalidPreimageMaxAttempts { site: self.key });
        }
        if self.effective_operation == EffectiveGpuOperation::PreimageSample &&
            self.preimage_max_attempts.is_none()
        {
            return Err(GpuPlanError::MissingPreimageMaxAttempts { site: self.key });
        }
        if self.effective_operation != EffectiveGpuOperation::PreimageSample &&
            self.preimage_max_attempts.is_some()
        {
            return Err(GpuPlanError::UnexpectedPreimageMaxAttempts { site: self.key });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenGpuPlan {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub loops: Vec<GpuLoopChoice>,
    pub nodes: Vec<GpuNodeChoice>,
}

/// Preparation/execution-local indexes for a frozen plan. The serialized plan
/// intentionally remains the public ordered vectors; callers that perform
/// repeated lookups build this immutable index once and share it locally.
#[derive(Clone, Debug)]
pub(crate) struct FrozenGpuPlanIndex {
    layouts: BTreeMap<LayoutId, usize>,
    loops: BTreeMap<GpuLoopSiteKey, usize>,
    nodes: BTreeMap<GpuExecutionSiteKey, usize>,
}

impl FrozenGpuPlanIndex {
    pub(crate) fn build(plan: &FrozenGpuPlan) -> Result<Self, GpuPlanError> {
        let mut layouts = BTreeMap::new();
        for (index, layout) in plan.layouts.iter().enumerate() {
            if layouts.insert(layout.id, index).is_some() {
                return Err(GpuPlanError::DuplicateLayout(layout.id));
            }
        }
        let mut loops = BTreeMap::new();
        for (index, choice) in plan.loops.iter().enumerate() {
            if loops.insert(choice.key, index).is_some() {
                return Err(GpuPlanError::DuplicateLoopSite(choice.key));
            }
        }
        let mut nodes = BTreeMap::new();
        for (index, choice) in plan.nodes.iter().enumerate() {
            if nodes.insert(choice.key, index).is_some() {
                return Err(GpuPlanError::DuplicateNodeSite(choice.key));
            }
        }
        Ok(Self { layouts, loops, nodes })
    }

    fn validate_for(&self, plan: &FrozenGpuPlan) -> Result<(), GpuPlanError> {
        let layouts_valid = self.layouts.len() == plan.layouts.len() &&
            self.layouts.values().all(|index| {
                plan.layouts
                    .get(*index)
                    .is_some_and(|layout| self.layouts.get(&layout.id) == Some(index))
            });
        let loops_valid = self.loops.len() == plan.loops.len() &&
            self.loops.values().all(|index| {
                plan.loops
                    .get(*index)
                    .is_some_and(|choice| self.loops.get(&choice.key) == Some(index))
            });
        let nodes_valid = self.nodes.len() == plan.nodes.len() &&
            self.nodes.values().all(|index| {
                plan.nodes
                    .get(*index)
                    .is_some_and(|choice| self.nodes.get(&choice.key) == Some(index))
            });
        if !layouts_valid || !loops_valid || !nodes_valid {
            return Err(GpuPlanError::StalePlanIndex);
        }
        Ok(())
    }

    pub(crate) fn layout<'a>(
        &self,
        plan: &'a FrozenGpuPlan,
        id: LayoutId,
    ) -> Option<&'a GpuLayout> {
        self.layouts.get(&id).and_then(|index| plan.layouts.get(*index))
    }

    pub(crate) fn loop_choice<'a>(
        &self,
        plan: &'a FrozenGpuPlan,
        key: GpuLoopSiteKey,
    ) -> Option<&'a GpuLoopChoice> {
        self.loops.get(&key).and_then(|index| plan.loops.get(*index))
    }

    pub(crate) fn node_choice<'a>(
        &self,
        plan: &'a FrozenGpuPlan,
        key: GpuExecutionSiteKey,
    ) -> Option<&'a GpuNodeChoice> {
        self.nodes.get(&key).and_then(|index| plan.nodes.get(*index))
    }

    /// Return the first frozen job selected for one node instance.
    ///
    /// The job is the only authoritative placement for a resident operation
    /// whose lowering does not itself carry a native request.  In particular,
    /// callers must not derive a device from the vector position or fall back
    /// to logical GPU zero when a control operation is adjacent to a resident
    /// producer/consumer.
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn selected_job(
        &self,
        plan: &FrozenGpuPlan,
        key: GpuExecutionSiteKey,
        instance: usize,
    ) -> Result<Option<GpuColumnJob>, GpuPlanError> {
        let Some(choice) = self.node_choice(plan, key) else { return Ok(None) };
        let mut selected = None;
        for layout_id in &choice.output_layouts {
            let layout = self
                .layout(plan, *layout_id)
                .ok_or(GpuPlanError::UnknownLayout { site: key, layout: *layout_id })?;
            let schedule = layout
                .schedule(&choice.columns_per_job, instance)
                .map_err(GpuPlanError::InvalidLayoutSchedule)?;
            if let Some(job) = schedule.waves().flatten().next() {
                selected = Some(job);
                break;
            }
        }
        Ok(selected)
    }
}

impl FrozenGpuPlan {
    pub fn new(
        contract: GpuPlanContract,
        layouts: Vec<GpuLayout>,
        loops: Vec<GpuLoopChoice>,
        nodes: Vec<GpuNodeChoice>,
    ) -> Result<Self, GpuPlanError> {
        let plan = Self { contract, layouts, loops, nodes };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), GpuPlanError> {
        let index = FrozenGpuPlanIndex::build(self)?;
        self.validate_with_index(&index)
    }

    pub(crate) fn validate_with_index(
        &self,
        index: &FrozenGpuPlanIndex,
    ) -> Result<(), GpuPlanError> {
        self.contract.validate()?;
        let device_count = self.contract.logical_to_physical_devices.len();
        index.validate_for(self)?;
        let layouts = index.layouts.keys().copied().collect::<BTreeSet<_>>();
        for layout in &self.layouts {
            layout.validate(device_count)?;
        }
        for choice in &self.loops {
            choice.validate()?;
        }
        for choice in &self.nodes {
            choice.validate(&self.contract, &layouts)?;
            for id in &choice.output_layouts {
                let layout = index
                    .layout(self, *id)
                    .ok_or(GpuPlanError::UnknownLayout { site: choice.key, layout: *id })?;
                let classes = if layout.instance_device_stride == 0 { 1 } else { device_count };
                for instance in 0..classes {
                    layout
                        .schedule(&choice.columns_per_job, instance)
                        .map_err(GpuPlanError::InvalidLayoutSchedule)?;
                }
            }
        }
        Ok(())
    }

    pub fn layout(&self, id: LayoutId) -> Option<&GpuLayout> {
        self.layouts.iter().find(|layout| layout.id == id)
    }

    pub fn loop_choice(&self, key: GpuLoopSiteKey) -> Option<&GpuLoopChoice> {
        self.loops.iter().find(|choice| choice.key == key)
    }

    pub fn node_choice(&self, key: GpuExecutionSiteKey) -> Option<&GpuNodeChoice> {
        self.nodes.iter().find(|choice| choice.key == key)
    }

    /// Validate a runtime contract before dispatch; no fallback or re-planning
    /// is performed when the frozen contract does not match.
    pub fn validate_runtime_contract(
        &self,
        contract: &GpuPlanContract,
    ) -> Result<(), GpuPlanError> {
        self.validate()?;
        contract.validate()?;
        if self.contract != *contract {
            return Err(GpuPlanError::ContractMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuPlanError {
    #[error("frozen GPU plan has no logical devices")]
    EmptyDeviceMapping,
    #[error("GPU plan has {devices} devices but {budgets} budgets")]
    DeviceBudgetCount { devices: usize, budgets: usize },
    #[error("physical GPU {0} is mapped more than once")]
    DuplicatePhysicalDevice(usize),
    #[error("budget entry {budget_device} is not for logical GPU {logical}")]
    BudgetDeviceMismatch { logical: usize, budget_device: usize },
    #[error("layout {0} occurs more than once")]
    DuplicateLayout(LayoutId),
    #[error("frozen GPU plan index does not match its plan")]
    StalePlanIndex,
    #[error("layout schedule is invalid: {0}")]
    InvalidLayoutSchedule(GpuScheduleError),
    #[error("layout {layout} has an empty representation")]
    EmptyLayoutRepresentation { layout: LayoutId },
    #[error("loop site {0:?} occurs more than once")]
    DuplicateLoopSite(GpuLoopSiteKey),
    #[error("loop site {site:?} has zero wave width")]
    ZeroLoopWave { site: GpuLoopSiteKey },
    #[error("loop site {site:?} has invalid tail size {tail}")]
    InvalidTail { site: GpuLoopSiteKey, tail: usize },
    #[error("node site {0:?} occurs more than once")]
    DuplicateNodeSite(GpuExecutionSiteKey),
    #[error("node site {site:?} has {actual} widths; expected {expected}")]
    WidthCount { site: GpuExecutionSiteKey, expected: usize, actual: usize },
    #[error("node site {site:?} has no output layout")]
    NoOutputLayout { site: GpuExecutionSiteKey },
    #[error("node site {site:?} refers to unknown layout {layout}")]
    UnknownLayout { site: GpuExecutionSiteKey, layout: LayoutId },
    #[error("node site {site:?} has an empty implementation variant")]
    EmptyImplementationVariant { site: GpuExecutionSiteKey },
    #[error("node site {site:?} has a capability inconsistent with its effective operation")]
    CapabilityMismatch { site: GpuExecutionSiteKey },
    #[error("node site {site:?} uses an unsupported GPU operation")]
    UnsupportedOperation { site: GpuExecutionSiteKey },
    #[error("node site {site:?} has zero preimage max_attempts")]
    InvalidPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("preimage node site {site:?} has no frozen max_attempts")]
    MissingPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("non-preimage node site {site:?} carries preimage max_attempts")]
    UnexpectedPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("node site {0:?} is missing from the frozen plan")]
    MissingNodeSite(GpuExecutionSiteKey),
    #[error("node site {site:?} has {actual} output ports; expected {expected}")]
    OutputPortCount { site: GpuExecutionSiteKey, expected: usize, actual: usize },
    #[error("node site {site:?} has duplicate output port {port}")]
    DuplicateOutputPort { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} has invalid output port {port}")]
    InvalidOutputPort { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} uses layout {actual}; expected {expected}")]
    OutputPortLayoutMismatch {
        site: GpuExecutionSiteKey,
        port: u32,
        expected: LayoutId,
        actual: LayoutId,
    },
    #[error("node site {site:?} output port {port} has an incompatible shape")]
    OutputPortShapeMismatch { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} has an incompatible representation")]
    OutputPortRepresentationMismatch { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} has unknown source layout {source_layout}")]
    UnknownSourceLayout { site: GpuExecutionSiteKey, port: u32, source_layout: LayoutId },
    #[error("node site {site:?} output port {port} has an invalid source layout")]
    InvalidSourceLayout { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} has invalid output port metadata")]
    InvalidOutputPorts { site: GpuExecutionSiteKey },
    #[error("contract does not match the frozen plan")]
    ContractMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_budget_projection_preserves_physical_identity_and_local_order() {
        let budgets = vec![
            GpuDeviceBudget { device: 0, device_bytes: 100, host_bytes: 11, pinned_host_bytes: 12 },
            GpuDeviceBudget { device: 1, device_bytes: 200, host_bytes: 21, pinned_host_bytes: 22 },
        ];
        for worker in [vec![7], vec![2], vec![7, 2], vec![2, 7]] {
            let projected = project_gpu_device_budgets(&[7, 2], &worker, &budgets).unwrap();
            for (local, (physical, budget)) in worker.iter().zip(projected).enumerate() {
                let mut expected =
                    budgets[logical_device_for_physical(&[7, 2], *physical).unwrap()].clone();
                expected.device = local;
                assert_eq!(budget, expected);
            }
        }
        // Replicated workers independently project the same fleet budgets.
        assert_eq!(project_gpu_device_budgets(&[7, 2], &[7, 2], &budgets).unwrap(), budgets);
        for (fleet, worker) in
            [(vec![7, 7], vec![7]), (vec![7, 2], vec![2, 2]), (vec![7, 2], vec![9])]
        {
            assert!(project_gpu_device_budgets(&fleet, &worker, &budgets).is_err());
        }
        assert!(project_gpu_device_budgets(&[7, 2], &[7], &budgets[..1]).is_err());
        assert!(
            project_gpu_device_budgets(&[7, 2], &[7], &[budgets[0].clone(), budgets[0].clone()])
                .is_err()
        );
    }

    #[test]
    fn tiny_columns_start_on_four_devices_before_any_finishes() {
        use std::{
            sync::{Arc, Condvar, Mutex, mpsc},
            time::Duration,
        };
        let schedules = (0..4)
            .map(|device| {
                GpuColumnSchedule::new(
                    1,
                    vec![1; 4],
                    vec![GpuColumnInterval { device, start: 0, end: 1 }],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let (started, receiver) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let coordinator_gate = gate.clone();
        let coordinator = std::thread::spawn(move || {
            let mut devices = BTreeSet::new();
            for _ in 0..4 {
                devices.insert(
                    receiver
                        .recv_timeout(Duration::from_secs(5))
                        .expect("all devices must start concurrently"),
                );
            }
            assert_eq!(devices, BTreeSet::from([0, 1, 2, 3]));
            *coordinator_gate.0.lock().unwrap() = true;
            coordinator_gate.1.notify_all();
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let results = pool
            .install(|| {
                dispatch_column_batch(
                    &mut [0usize; 4],
                    &schedules.iter().collect::<Vec<_>>(),
                    |active, instance, job| {
                        assert_eq!(*active, 0);
                        *active += 1;
                        started.send(job.device).unwrap();
                        let (released, timeout) = gate
                            .1
                            .wait_timeout_while(
                                gate.0.lock().unwrap(),
                                Duration::from_secs(5),
                                |released| !*released,
                            )
                            .unwrap();
                        assert!(
                            *released && !timeout.timed_out(),
                            "dispatcher serialized independent devices"
                        );
                        *active -= 1;
                        Ok::<_, ()>(vec![(instance, 0), (instance, 1)])
                    },
                )
            })
            .unwrap();
        coordinator.join().unwrap();
        assert_eq!(
            results.iter().map(|(instance, _, ports)| (*instance, ports.len())).collect::<Vec<_>>(),
            vec![(0, 2), (1, 2), (2, 2), (3, 2)]
        );
    }

    #[test]
    fn every_interval_instance_and_output_group_is_dispatched_once() {
        let schedule = GpuColumnSchedule::new(
            5,
            vec![1, 2],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 4 },
                GpuColumnInterval { device: 0, start: 4, end: 5 },
            ],
        )
        .unwrap();
        let results = dispatch_column_batch(
            &mut [0usize; 2],
            &[&schedule, &schedule],
            |calls, instance, job| {
                *calls += 1;
                Ok::<_, ()>((instance, job.start..job.end))
            },
        )
        .unwrap();
        let coordinates = results
            .into_iter()
            .flat_map(|(instance, _, (_, range))| range.map(move |column| (instance, column)))
            .collect::<Vec<_>>();
        assert_eq!(coordinates.len(), 10);
        assert_eq!(
            coordinates.into_iter().collect::<BTreeSet<_>>(),
            (0..2).flat_map(|instance| (0..5).map(move |column| (instance, column))).collect()
        );
    }

    fn contract(devices: usize) -> GpuPlanContract {
        GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "fake-gpu".into(),
            logical_to_physical_devices: (0..devices).collect(),
            device_budgets: (0..devices)
                .map(|device| GpuDeviceBudget {
                    device,
                    device_bytes: 1024,
                    pinned_host_bytes: 1024,
                    host_bytes: 1024,
                })
                .collect(),
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        }
    }

    fn layout() -> GpuLayout {
        GpuLayout {
            id: 1,
            columns: 8,
            rows: 1,
            ring_dimension: 1,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: vec![
                GpuColumnInterval { device: 0, start: 0, end: 4 },
                GpuColumnInterval { device: 1, start: 4, end: 8 },
            ],
        }
    }

    fn second_layout() -> GpuLayout {
        GpuLayout {
            id: 2,
            columns: 3,
            rows: 2,
            ring_dimension: 4,
            representation: "small-matrix".into(),
            instance_device_stride: 0,
            owner_intervals: vec![GpuColumnInterval { device: 0, start: 0, end: 3 }],
        }
    }

    #[test]
    fn empty_layout_ownership_uses_the_balanced_production_mapper() {
        let layout = GpuLayout {
            id: 9,
            columns: 5,
            rows: 1,
            ring_dimension: 1,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: Vec::new(),
        };
        let schedule = layout.schedule(&[2, 2], 0).unwrap();
        assert_eq!(
            schedule.intervals(),
            &[
                GpuColumnInterval { device: 0, start: 0, end: 3 },
                GpuColumnInterval { device: 1, start: 3, end: 5 },
            ]
        );
    }

    #[test]
    fn same_profile_different_site_widths_and_owner_independence() {
        let key0 = GpuExecutionSiteKey { site: 7, shape_class: 0, instance_class: 0 };
        let key1 = GpuExecutionSiteKey { site: 8, shape_class: 0, instance_class: 0 };
        let node = |key, width| GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [3; 32],
            effective_operation: EffectiveGpuOperation::MatrixMulSmallRhs,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1],
            columns_per_job: vec![width, width],
            implementation_variant: "matmul".into(),
            preimage_max_attempts: None,
        };
        let plan = FrozenGpuPlan::new(
            contract(2),
            vec![layout()],
            vec![],
            vec![node(key0, 2), node(key1, 4)],
        )
        .unwrap();
        assert_eq!(plan.node_choice(key0).unwrap().columns_per_job, vec![2, 2]);
        assert_eq!(plan.node_choice(key1).unwrap().columns_per_job, vec![4, 4]);
        assert_eq!(plan.layout(1).unwrap().owner_intervals[0].start, 0);
    }

    #[test]
    fn frozen_plan_index_matches_ordered_lookup_and_rejects_duplicates() {
        let key = GpuExecutionSiteKey { site: 17, shape_class: 2, instance_class: 1 };
        let loop_key = GpuLoopSiteKey { site: 18, shape_class: 2, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: Some(loop_key),
            operation_identity: [9; 32],
            effective_operation: EffectiveGpuOperation::MatrixAdd,
            column_capability: ColumnCapability::SameColumns,
            output_layouts: vec![1],
            columns_per_job: vec![2, 2],
            implementation_variant: "indexed".into(),
            preimage_max_attempts: None,
        };
        let plan = FrozenGpuPlan {
            contract: contract(2),
            layouts: vec![layout()],
            loops: vec![GpuLoopChoice {
                key: loop_key,
                loop_count: 4,
                wave_instances: 2,
                tail_instances: 0,
            }],
            nodes: vec![node],
        };
        let index = FrozenGpuPlanIndex::build(&plan).expect("unique plan entries");
        assert_eq!(index.layout(&plan, 1), plan.layout(1));
        assert_eq!(index.loop_choice(&plan, loop_key), plan.loop_choice(loop_key));
        assert_eq!(index.node_choice(&plan, key), plan.node_choice(key));

        let mut duplicate_layout = plan.clone();
        duplicate_layout.layouts.push(layout());
        assert!(matches!(
            FrozenGpuPlanIndex::build(&duplicate_layout),
            Err(GpuPlanError::DuplicateLayout(1))
        ));
        let mut duplicate_loop = plan.clone();
        duplicate_loop.loops.push(duplicate_loop.loops[0].clone());
        assert!(matches!(
            FrozenGpuPlanIndex::build(&duplicate_loop),
            Err(GpuPlanError::DuplicateLoopSite(key)) if key == loop_key
        ));
        let mut duplicate_node = plan;
        duplicate_node.nodes.push(duplicate_node.nodes[0].clone());
        assert!(matches!(
            FrozenGpuPlanIndex::build(&duplicate_node),
            Err(GpuPlanError::DuplicateNodeSite(site)) if site == key
        ));
    }

    #[test]
    fn selected_job_uses_frozen_output_layout_owner() {
        let key = GpuExecutionSiteKey { site: 19, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [7; 32],
            effective_operation: EffectiveGpuOperation::HostOrControl,
            column_capability: ColumnCapability::HostOrControl,
            output_layouts: vec![1],
            columns_per_job: vec![2, 2],
            implementation_variant: "resident-family".into(),
            preimage_max_attempts: None,
        };
        let plan = FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![node]).unwrap();
        let index = FrozenGpuPlanIndex::build(&plan).unwrap();
        let job = index.selected_job(&plan, key, 0).unwrap().unwrap();
        assert_eq!(job.device, 0);
        assert_eq!(job.start, 0);
        assert_eq!(job.end, 2);
    }

    #[test]
    fn selected_job_reports_empty_zero_width_schedule_without_a_fallback_device() {
        let key = GpuExecutionSiteKey { site: 20, shape_class: 0, instance_class: 0 };
        let empty_layout = GpuLayout {
            id: 1,
            columns: 0,
            rows: 1,
            ring_dimension: 1,
            representation: "empty".into(),
            instance_device_stride: 0,
            owner_intervals: Vec::new(),
        };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [8; 32],
            effective_operation: EffectiveGpuOperation::HostOrControl,
            column_capability: ColumnCapability::HostOrControl,
            output_layouts: vec![1],
            columns_per_job: vec![0, 0],
            implementation_variant: "empty-family".into(),
            preimage_max_attempts: None,
        };
        let plan = FrozenGpuPlan::new(contract(2), vec![empty_layout], vec![], vec![node]).unwrap();
        let index = FrozenGpuPlanIndex::build(&plan).unwrap();
        assert!(index.selected_job(&plan, key, 0).unwrap().is_none());
    }

    #[test]
    fn missing_or_mismatched_plan_is_rejected() {
        let key = GpuExecutionSiteKey { site: 1, shape_class: 0, instance_class: 0 };
        let plan = FrozenGpuPlan::new(
            contract(2),
            vec![layout()],
            vec![],
            vec![GpuNodeChoice {
                key,
                loop_site: None,
                operation_identity: [0; 32],
                effective_operation: EffectiveGpuOperation::MatrixAdd,
                column_capability: ColumnCapability::SameColumns,
                output_layouts: vec![1],
                columns_per_job: vec![1, 1],
                implementation_variant: "add".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        let mut changed = contract(2);
        changed.shape_contract_hash = [9; 32];
        assert!(matches!(
            plan.validate_runtime_contract(&changed),
            Err(GpuPlanError::ContractMismatch)
        ));
        assert!(
            plan.node_choice(GpuExecutionSiteKey { site: 2, shape_class: 0, instance_class: 0 })
                .is_none()
        );
    }

    #[test]
    fn large_loops_use_compact_plans() {
        let key = GpuLoopSiteKey { site: 5, shape_class: 2, instance_class: 0 };
        let choice =
            GpuLoopChoice { key, loop_count: usize::MAX, wave_instances: 4, tail_instances: 3 };
        let plan = FrozenGpuPlan::new(contract(2), vec![layout()], vec![choice], vec![]).unwrap();
        assert_eq!(plan.loops.len(), 1);
        assert_eq!(plan.loops[0].loop_count, usize::MAX);
    }

    #[test]
    fn tail_empty_and_zero_capacity_cases_are_validated() {
        let key = GpuLoopSiteKey { site: 5, shape_class: 2, instance_class: 0 };
        let choice = GpuLoopChoice { key, loop_count: 8, wave_instances: 4, tail_instances: 0 };
        assert!(FrozenGpuPlan::new(contract(2), vec![layout()], vec![choice], vec![]).is_ok());
        assert!(GpuColumnSchedule::new(0, vec![0, 0], vec![]).is_ok());
        let invalid = GpuLoopChoice { key, loop_count: 5, wave_instances: 4, tail_instances: 4 };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![invalid], vec![]),
            Err(GpuPlanError::InvalidTail { .. })
        ));
    }

    #[test]
    fn logical_and_physical_device_ids_are_distinct() {
        let mut plan_contract = contract(2);
        plan_contract.logical_to_physical_devices = vec![2, 5];
        assert!(plan_contract.validate().is_ok());
        assert_eq!(plan_contract.logical_to_physical_devices, vec![2, 5]);
    }

    #[test]
    fn preimage_retry_bound_is_frozen_and_nonzero() {
        let key = GpuExecutionSiteKey { site: 9, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [4; 32],
            effective_operation: EffectiveGpuOperation::PreimageSample,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1],
            columns_per_job: vec![2, 2],
            implementation_variant: "preimage".into(),
            preimage_max_attempts: Some(7),
        };
        let plan = FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![node]).unwrap();
        assert_eq!(plan.node_choice(key).unwrap().preimage_max_attempts, Some(7));
        let invalid = GpuNodeChoice {
            preimage_max_attempts: Some(0),
            ..plan.node_choice(key).unwrap().clone()
        };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![invalid]),
            Err(GpuPlanError::InvalidPreimageMaxAttempts { .. })
        ));
        let missing =
            GpuNodeChoice { preimage_max_attempts: None, ..plan.node_choice(key).unwrap().clone() };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![missing]),
            Err(GpuPlanError::MissingPreimageMaxAttempts { .. })
        ));
    }

    #[test]
    fn multi_output_ports_validate_independent_shapes_and_sources() {
        let key = GpuExecutionSiteKey { site: 12, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [5; 32],
            effective_operation: EffectiveGpuOperation::MatrixMulAccumulate,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1, 2],
            columns_per_job: vec![1, 1],
            implementation_variant: "fused-two-port".into(),
            preimage_max_attempts: None,
        };
        let plan =
            FrozenGpuPlan::new(contract(2), vec![layout(), second_layout()], vec![], vec![node])
                .unwrap();
        let ports = vec![
            GpuOutputPortContract {
                port: 0,
                layout: 1,
                rows: 1,
                columns: 8,
                ring_dimension: 1,
                representation: "matrix".into(),
                source_layout: Some(1),
            },
            GpuOutputPortContract {
                port: 1,
                layout: 2,
                rows: 2,
                columns: 3,
                ring_dimension: 4,
                representation: "small-matrix".into(),
                source_layout: Some(1),
            },
        ];
        assert!(plan.validate_output_ports(key, &ports).is_ok());
        let mut wrong_shape = ports.clone();
        wrong_shape[1].rows = 1;
        assert!(matches!(
            plan.validate_output_ports(key, &wrong_shape),
            Err(GpuPlanError::OutputPortShapeMismatch { port: 1, .. })
        ));
        let mut wrong_source = ports;
        wrong_source[1].source_layout = Some(99);
        assert!(matches!(
            plan.validate_output_ports(key, &wrong_source),
            Err(GpuPlanError::UnknownSourceLayout { port: 1, .. })
        ));
    }

    #[test]
    fn unsupported_effective_operation_is_rejected_before_dispatch() {
        let key = GpuExecutionSiteKey { site: 13, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [6; 32],
            effective_operation: EffectiveGpuOperation::Unsupported,
            column_capability: ColumnCapability::Unsupported,
            output_layouts: vec![1],
            columns_per_job: vec![1, 1],
            implementation_variant: "unknown".into(),
            preimage_max_attempts: None,
        };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![node]),
            Err(GpuPlanError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn fused_union_jobs_preserve_port_owners_and_tail_coverage() {
        let port0 = vec![
            GpuColumnSchedule::new(
                7,
                vec![2, 2],
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 3 },
                    GpuColumnInterval { device: 1, start: 3, end: 7 },
                ],
            )
            .unwrap(),
        ];
        let port1 = vec![
            GpuColumnSchedule::new(
                5,
                vec![3, 1],
                vec![
                    GpuColumnInterval { device: 1, start: 0, end: 1 },
                    GpuColumnInterval { device: 0, start: 1, end: 5 },
                ],
            )
            .unwrap(),
        ];
        let jobs = build_fused_union_jobs(&[port0, port1], 1).unwrap();
        assert_eq!(jobs[0].first().unwrap().range.start, 0);
        assert_eq!(jobs[0].last().unwrap().range.end, 7);
        assert!(jobs[0].windows(2).all(|jobs| jobs[0].range.end == jobs[1].range.start));
        assert!(jobs[0].iter().any(|job| job.port_jobs[1].clipped_range.is_none()));
        assert!(jobs[0].iter().any(|job| {
            let left = job.port_jobs[0].owner_device;
            let right = job.port_jobs[1].owner_device;
            left.is_some() && right.is_some() && left != right
        }));
        let first = &jobs[0][0];
        assert_eq!(first.port_jobs[0].source_interval, Some(0));
        assert_eq!(first.port_jobs[0].clipped_range.unwrap().start, 0);
    }

    #[test]
    fn fused_union_waves_keep_gpu_parallelism_and_primitive_count() {
        let schedule = |device| {
            GpuColumnSchedule::new(
                2,
                vec![1, 1],
                vec![GpuColumnInterval { device, start: 0, end: 2 }],
            )
            .unwrap()
        };
        let jobs = build_fused_union_jobs(&[vec![schedule(0), schedule(1)]], 2).unwrap();
        let waves = fused_union_waves(&jobs);
        assert_eq!(waves.len(), 2);
        assert_eq!(waves.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
        assert_eq!(fused_union_invocation_count(&jobs), 4);
        assert_eq!(
            fused_union_invocation_count_lazy(&[vec![schedule(0), schedule(1)]], 2).unwrap(),
            4
        );
        assert_eq!(waves[0].iter().map(|(_, job)| job.device).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(waves[0].iter().map(|(_, job)| job.instance).collect::<Vec<_>>(), vec![0, 1]);
        let lazy_waves = fused_union_waves_lazy(&[vec![schedule(0), schedule(1)]], 2)
            .unwrap()
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(lazy_waves.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
    }

    #[test]
    fn fused_union_lazy_waves_merge_nonmonotonic_global_wave_ids() {
        let schedule = GpuColumnSchedule::new(
            4,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 4 },
            ],
        )
        .unwrap();
        let schedules = vec![vec![schedule]];
        let materialized = build_fused_union_jobs(&schedules, 1).unwrap();
        assert_eq!(
            materialized[0].iter().map(|job| job.logical_wave).collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
        let expected = fused_union_waves(&materialized);
        let actual =
            fused_union_waves_lazy(&schedules, 1).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            actual,
            expected
                .into_iter()
                .map(|wave| wave.into_iter().map(|(_, job)| job).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        assert_eq!(actual[0].iter().map(|job| job.range.start).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(actual[1].iter().map(|job| job.range.start).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn fused_union_rejects_empty_ports_and_mismatched_instances() {
        assert!(matches!(build_fused_union_jobs(&[], 1), Err(GpuFusedUnionError::EmptyPorts)));
        let schedule = GpuColumnSchedule::new(
            1,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: 1 }],
        )
        .unwrap();
        assert!(matches!(
            build_fused_union_jobs(&[vec![schedule]], 2),
            Err(GpuFusedUnionError::InstanceCountMismatch)
        ));
    }

    #[test]
    fn fused_union_accepts_zero_column_ports_and_all_empty_instances() {
        let empty = GpuColumnSchedule::new(0, vec![0, 0], vec![]).unwrap();
        let work = GpuColumnSchedule::new(
            2,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 2 }],
        )
        .unwrap();
        let jobs = build_fused_union_jobs(&[vec![empty.clone()], vec![work]], 1).unwrap();
        assert_eq!(jobs[0].len(), 1);
        assert!(jobs[0][0].port_jobs[0].clipped_range.is_none());
        let all_empty = build_fused_union_jobs(&[vec![empty.clone()], vec![empty]], 1).unwrap();
        assert_eq!(all_empty, vec![Vec::new()]);
    }

    #[test]
    fn fused_union_lazy_stream_matches_materialized_api() {
        let schedule = GpuColumnSchedule::new(
            5,
            vec![2, 3],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 5 },
            ],
        )
        .unwrap();
        let schedules = vec![vec![schedule.clone()], vec![schedule]];
        let expected = build_fused_union_jobs(&schedules, 1).unwrap();
        let actual =
            fused_union_jobs_lazy(&schedules, 1).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(actual, expected[0]);
        let classes = fused_union_wave_classes(&schedules, 1).unwrap();
        assert!(!classes.is_empty());
        assert!(classes.iter().all(|class| class.multiplicity > 0));
    }

    #[test]
    fn fused_union_lazy_stream_does_not_materialize_width_one_tiles() {
        let huge = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let schedules = vec![vec![huge]];
        let first_two = fused_union_jobs_lazy(&schedules, 1)
            .unwrap()
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(first_two.len(), 2);
        assert_eq!(first_two[0].range, crate::gpu_column_policy::ColumnRange { start: 0, end: 1 });
        assert_eq!(first_two[1].range, crate::gpu_column_policy::ColumnRange { start: 1, end: 2 });
        let classes = fused_union_wave_classes(&schedules, 1).unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].multiplicity, usize::MAX);
    }

    #[test]
    fn fused_union_lazy_waves_keep_only_one_wave_for_huge_inputs() {
        let huge = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let schedules = vec![vec![huge.clone(), huge.clone()], vec![huge.clone(), huge]];
        let mut waves = fused_union_waves_lazy(&schedules, 2).unwrap();
        let first = waves.next().unwrap().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].range.start, 0);
        let arbitrary = waves.nth(1023).unwrap().unwrap();
        assert_eq!(arbitrary.len(), 2);
        assert_eq!(arbitrary[0].range.start, 1024);
        let classes = fused_union_wave_classes(&schedules, 2).unwrap();
        assert_eq!(classes.len(), 2);
        assert!(classes.iter().all(|class| class.multiplicity == usize::MAX));
    }
}
