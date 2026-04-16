use num_bigint::BigUint;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, fmt::Display};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct GateId(pub(crate) usize);

impl Display for GateId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolyGateKind {
    Input,
    Add,
    Sub,
    Mul,
    SmallScalarMul,
    LargeScalarMul,
    SlotTransfer,
    PubLut,
    SubCircuitOutput,
    SummedSubCircuitOutput,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateParamSource<T> {
    Const(T),
    Param(usize),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubCircuitParamKind {
    SmallScalarMul,
    LargeScalarMul,
    SlotTransfer,
    PubLut,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotTransferSpec {
    Explicit(Vec<(u32, Option<u32>)>),
    Rotation { diagonal: u32, num_slots: u32 },
    Repeated { src_slot: u32, num_slots: u32, prefix_len: u32, prefix_scalar: Option<u32> },
}

impl SlotTransferSpec {
    pub fn explicit(values: Vec<(u32, Option<u32>)>) -> Self {
        Self::Explicit(values)
    }

    pub fn rotation(diagonal: usize, num_slots: usize) -> Self {
        Self::Rotation {
            diagonal: u32::try_from(diagonal).expect("rotation diagonal must fit in u32"),
            num_slots: u32::try_from(num_slots).expect("rotation slot count must fit in u32"),
        }
    }

    pub fn repeated(
        src_slot: usize,
        num_slots: usize,
        prefix_len: usize,
        prefix_scalar: Option<u32>,
    ) -> Self {
        Self::Repeated {
            src_slot: u32::try_from(src_slot).expect("repeated source slot must fit in u32"),
            num_slots: u32::try_from(num_slots).expect("repeated slot count must fit in u32"),
            prefix_len: u32::try_from(prefix_len).expect("repeated prefix length must fit in u32"),
            prefix_scalar,
        }
    }

    pub fn materialize(&self) -> Vec<(u32, Option<u32>)> {
        match self {
            Self::Explicit(values) => values.clone(),
            Self::Rotation { diagonal, num_slots } => {
                let diagonal = usize::try_from(*diagonal)
                    .expect("rotation diagonal must fit in usize on this platform");
                let num_slots = usize::try_from(*num_slots)
                    .expect("rotation slot count must fit in usize on this platform");
                if num_slots >= 1024 {
                    (0..num_slots)
                        .into_par_iter()
                        .map(|dst_slot| {
                            let src_slot =
                                (dst_slot + num_slots - (diagonal % num_slots)) % num_slots;
                            (
                                u32::try_from(src_slot)
                                    .expect("rotation source slot index must fit in u32"),
                                None,
                            )
                        })
                        .collect()
                } else {
                    (0..num_slots)
                        .map(|dst_slot| {
                            let src_slot =
                                (dst_slot + num_slots - (diagonal % num_slots)) % num_slots;
                            (
                                u32::try_from(src_slot)
                                    .expect("rotation source slot index must fit in u32"),
                                None,
                            )
                        })
                        .collect()
                }
            }
            Self::Repeated { src_slot, num_slots, prefix_len, prefix_scalar } => {
                let num_slots = usize::try_from(*num_slots)
                    .expect("repeated slot count must fit in usize on this platform");
                let prefix_len = usize::try_from(*prefix_len)
                    .expect("repeated prefix length must fit in usize on this platform");
                if num_slots >= 1024 {
                    (0..num_slots)
                        .into_par_iter()
                        .map(|dst_slot| {
                            let scalar = if dst_slot < prefix_len { *prefix_scalar } else { None };
                            (*src_slot, scalar)
                        })
                        .collect()
                } else {
                    (0..num_slots)
                        .map(|dst_slot| {
                            let scalar = if dst_slot < prefix_len { *prefix_scalar } else { None };
                            (*src_slot, scalar)
                        })
                        .collect()
                }
            }
        }
    }

    pub fn as_explicit_slice(&self) -> Option<&[(u32, Option<u32>)]> {
        match self {
            Self::Explicit(values) => Some(values.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubCircuitParamValue {
    SmallScalarMul(Vec<u32>),
    LargeScalarMul(Vec<BigUint>),
    SlotTransfer(SlotTransferSpec),
    PubLut(usize),
}

impl SubCircuitParamValue {
    pub fn kind(&self) -> SubCircuitParamKind {
        match self {
            Self::SmallScalarMul(_) => SubCircuitParamKind::SmallScalarMul,
            Self::LargeScalarMul(_) => SubCircuitParamKind::LargeScalarMul,
            Self::SlotTransfer(_) => SubCircuitParamKind::SlotTransfer,
            Self::PubLut(_) => SubCircuitParamKind::PubLut,
        }
    }
}

impl GateParamSource<Vec<u32>> {
    pub fn resolve_small_scalar<'a>(&'a self, bindings: &'a [SubCircuitParamValue]) -> &'a [u32] {
        match self {
            Self::Const(value) => value.as_slice(),
            Self::Param(param_id) => match bindings.get(*param_id) {
                Some(SubCircuitParamValue::SmallScalarMul(value)) => value.as_slice(),
                Some(other) => panic!(
                    "sub-circuit parameter kind mismatch for SmallScalarMul: expected {:?}, got {:?}",
                    SubCircuitParamKind::SmallScalarMul,
                    other.kind()
                ),
                None => panic!("missing sub-circuit parameter binding {param_id}"),
            },
        }
    }
}

impl GateParamSource<Vec<BigUint>> {
    pub fn resolve_large_scalar<'a>(
        &'a self,
        bindings: &'a [SubCircuitParamValue],
    ) -> &'a [BigUint] {
        match self {
            Self::Const(value) => value.as_slice(),
            Self::Param(param_id) => match bindings.get(*param_id) {
                Some(SubCircuitParamValue::LargeScalarMul(value)) => value.as_slice(),
                Some(other) => panic!(
                    "sub-circuit parameter kind mismatch for LargeScalarMul: expected {:?}, got {:?}",
                    SubCircuitParamKind::LargeScalarMul,
                    other.kind()
                ),
                None => panic!("missing sub-circuit parameter binding {param_id}"),
            },
        }
    }
}

impl GateParamSource<SlotTransferSpec> {
    pub fn resolve_slot_transfer<'a>(
        &'a self,
        bindings: &'a [SubCircuitParamValue],
    ) -> Cow<'a, [(u32, Option<u32>)]> {
        match self {
            Self::Const(value) => value
                .as_explicit_slice()
                .map(Cow::Borrowed)
                .unwrap_or_else(|| Cow::Owned(value.materialize())),
            Self::Param(param_id) => match bindings.get(*param_id) {
                Some(SubCircuitParamValue::SlotTransfer(value)) => value
                    .as_explicit_slice()
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(value.materialize())),
                Some(other) => panic!(
                    "sub-circuit parameter kind mismatch for SlotTransfer: expected {:?}, got {:?}",
                    SubCircuitParamKind::SlotTransfer,
                    other.kind()
                ),
                None => panic!("missing sub-circuit parameter binding {param_id}"),
            },
        }
    }
}

impl GateParamSource<usize> {
    pub fn resolve_public_lookup(&self, bindings: &[SubCircuitParamValue]) -> usize {
        match self {
            Self::Const(value) => *value,
            Self::Param(param_id) => match bindings.get(*param_id) {
                Some(SubCircuitParamValue::PubLut(value)) => *value,
                Some(other) => panic!(
                    "sub-circuit parameter kind mismatch for PubLut: expected {:?}, got {:?}",
                    SubCircuitParamKind::PubLut,
                    other.kind()
                ),
                None => panic!("missing sub-circuit parameter binding {param_id}"),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyGate {
    pub gate_id: GateId,
    pub gate_type: PolyGateType,
    pub input_gates: Vec<GateId>,
}

impl PolyGate {
    pub fn new(gate_id: GateId, gate_type: PolyGateType, input_gates: Vec<GateId>) -> Self {
        Self { gate_id, gate_type, input_gates }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum PolyGateType {
    Input,
    Add,
    Sub,
    Mul,
    SmallScalarMul { scalar: GateParamSource<Vec<u32>> },
    LargeScalarMul { scalar: GateParamSource<Vec<BigUint>> },
    SlotTransfer { src_slots: GateParamSource<SlotTransferSpec> },
    PubLut { lut_id: GateParamSource<usize> },
    SubCircuitOutput { call_id: usize, output_idx: usize, num_inputs: usize },
    SummedSubCircuitOutput { summed_call_id: usize, output_idx: usize, num_inputs: usize },
}

impl PolyGateType {
    pub fn num_input(&self) -> usize {
        match self {
            PolyGateType::Input => 0,
            PolyGateType::SmallScalarMul { .. } |
            PolyGateType::LargeScalarMul { .. } |
            PolyGateType::SlotTransfer { .. } |
            PolyGateType::PubLut { .. } => 1,
            PolyGateType::SubCircuitOutput { num_inputs, .. } |
            PolyGateType::SummedSubCircuitOutput { num_inputs, .. } => *num_inputs,
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => 2,
        }
    }

    pub fn kind(&self) -> PolyGateKind {
        match self {
            PolyGateType::Input => PolyGateKind::Input,
            PolyGateType::Add => PolyGateKind::Add,
            PolyGateType::Sub => PolyGateKind::Sub,
            PolyGateType::Mul => PolyGateKind::Mul,
            PolyGateType::SmallScalarMul { .. } => PolyGateKind::SmallScalarMul,
            PolyGateType::LargeScalarMul { .. } => PolyGateKind::LargeScalarMul,
            PolyGateType::SlotTransfer { .. } => PolyGateKind::SlotTransfer,
            PolyGateType::PubLut { .. } => PolyGateKind::PubLut,
            PolyGateType::SubCircuitOutput { .. } => PolyGateKind::SubCircuitOutput,
            PolyGateType::SummedSubCircuitOutput { .. } => PolyGateKind::SummedSubCircuitOutput,
        }
    }
}
