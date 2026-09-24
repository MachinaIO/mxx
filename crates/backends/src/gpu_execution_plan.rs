//! Small, value-only metadata for a frozen GPU execution plan.
//!
//! A plan records choices made during warmup. It never owns GPU buffers,
//! pointers, command objects, or runtime owners. Input ranges, physical shard
//! views, and transfer actions remain derived data produced by the shared
//! column-policy lowering.

use crate::gpu_schedule::{GpuColumnInterval, GpuColumnSchedule, GpuScheduleError};
#[cfg(feature = "gpu")]
use crate::poly::dcrt::gpu::{GpuHashTagPart, GpuSignedValuesEncoding};

/// A half-open global column range.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ColumnRange {
    pub start: usize,
    pub end: usize,
}

/// A lowered read range. `operand` is the operation argument index, not a
/// device or an instance index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputColumnRange {
    pub operand: usize,
    pub range: ColumnRange,
}
#[cfg(feature = "gpu")]
use mxx_ir_core::types::ConcreteWireType;
#[cfg(feature = "gpu")]
use num_bigint::BigInt;
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
    MultiplyMonomial,
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
    SampleInterval,
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

    pub(crate) fn multiply_monomial() -> Self {
        Self { primitive: GpuNativePrimitive::MultiplyMonomial, ..Self::ring_automorphism() }
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
        use GpuArgumentKind::{F64, I64, U32, U64, Value};
        Self {
            primitive: if gaussian {
                GpuNativePrimitive::SampleGaussian
            } else {
                GpuNativePrimitive::SampleUniform
            },
            argument_kinds: Box::new([
                Value, U32, Value, U32, F64, U64, U64, I64, I64, U64, U64, U32, U32,
            ]),
            output_count: 1,
        }
    }

    pub(crate) fn sample_interval() -> Self {
        let mut sample = Self::sample(false);
        sample.primitive = GpuNativePrimitive::SampleInterval;
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
        Self::matrix_copy_views(1)
    }

    /// `count` independent view copies in one operation: each copy is a
    /// (source, part, destination, part, source binding, destination binding)
    /// argument group with its destination as the matching output.
    pub(crate) fn matrix_copy_views(count: usize) -> Self {
        use GpuArgumentKind::{U32, Value};
        Self {
            primitive: GpuNativePrimitive::MatrixCopyView,
            argument_kinds: [Value, U32, Value, U32, U32, U32].repeat(count).into_boxed_slice(),
            output_count: count,
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
}

#[cfg(feature = "gpu")]
impl GpuImplementationRegistry {
    pub(crate) fn register(
        &mut self,
        implementation: GpuImplementation,
    ) -> Result<GpuImplId, &'static str> {
        // One primitive may have several argument schemas, e.g. a view copy
        // of any number of windows; each distinct schema is one entry.
        if let Some(index) = self.entries.iter().position(|entry| *entry == implementation) {
            return Ok(GpuImplId(index as u32));
        }
        let id = GpuImplId(
            u32::try_from(self.entries.len()).map_err(|_| "too many GPU native implementations")?,
        );
        self.entries.push(implementation);
        Ok(id)
    }

    pub(crate) fn resolve(&self, id: GpuImplId) -> Result<&GpuImplementation, &'static str> {
        self.entries.get(id.0 as usize).ok_or("GPU implementation ID is not registered")
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
                        (KernelArg::F64(_), GpuArgumentKind::F64)
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
        if self.preimage_max_attempts == Some(0) {
            return Err(GpuPlanError::InvalidPreimageMaxAttempts { site: self.key });
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
    #[error("node site {site:?} has zero preimage max_attempts")]
    InvalidPreimageMaxAttempts { site: GpuExecutionSiteKey },
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
    }
}
