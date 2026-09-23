//! Run-local storage for typed, device-resident control owners.
//!
//! A resident frame retains the concrete GPU owner described by the compiled
//! slot contract.  It never turns a family or matrix into a host value.  The
//! only integer-specific operation is the explicit [`ResidentOwner::integer`]
//! accessor used by scalar integer/bool control operations.

use crate::{
    backend::poly_gpu::{
        GpuCaptureOwnedOwner, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSignedValues,
        GpuFleetSmallMatrix, GpuFleetTrapdoor,
    },
    gpu_compiled::{NativeIntegerEncoding, ResidentInstructionId, ResidentSlotType, ValueSlot},
};
use mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding;
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentControlFrameError {
    #[error("resident control owner allocation failed: {0}")]
    Allocation(String),
    #[error("resident control slot {slot:?} received an owner incompatible with {expected}")]
    TypeMismatch { slot: ValueSlot, expected: &'static str },
    #[error("resident control slot {0:?} is not an integer or bool owner")]
    NotInteger(ValueSlot),
    #[error("resident control slot {0:?} is already allocated")]
    Duplicate(ValueSlot),
    #[error("resident control slot {0:?} is missing")]
    Missing(ValueSlot),
    #[error("resident control slot {0:?} is not an indexed family")]
    NotFamily(ValueSlot),
    #[error(
        "resident indexed-family slot {slot:?} index {index} is out of bounds for count {count}"
    )]
    FamilyIndexOutOfBounds { slot: ValueSlot, index: usize, count: usize },
    #[error("resident carried arena {0:?} is missing")]
    MissingCarriedArena(ResidentInstructionId),
    #[error("resident carried index {index} of instruction {instruction:?} is missing")]
    MissingCarriedIndex { instruction: ResidentInstructionId, index: usize },
    #[error("resident carried slots {current:?} and {next:?} must be distinct")]
    CarriedSlotCollision { current: ValueSlot, next: ValueSlot },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentArenaKind {
    FullWave,
    Sequential(ResidentInstructionId),
}

/// The concrete GPU owner retained by a resident frame.
#[derive(Clone, Debug)]
pub(crate) enum ResidentOwner {
    Matrix(Arc<GpuFleetMatrix>),
    SmallMatrix(Arc<GpuFleetSmallMatrix>),
    Trapdoor {
        public: Arc<GpuFleetMatrix>,
        secret: Arc<GpuFleetTrapdoor>,
    },
    Integer(Arc<GpuFleetSignedValues>),
    IndexedFamily {
        element_type: Box<ResidentSlotType>,
        elements: Arc<[ResidentOwner]>,
        packed_integer: Option<Arc<GpuFleetSignedValues>>,
    },
}

impl ResidentOwner {
    pub(crate) fn from_capture(owner: GpuCaptureOwnedOwner) -> Self {
        match owner {
            GpuCaptureOwnedOwner::Matrix(value) => Self::Matrix(Arc::new(value)),
            GpuCaptureOwnedOwner::SmallMatrix(value) => Self::SmallMatrix(Arc::new(value)),
            GpuCaptureOwnedOwner::TrapdoorPair { public, secret } => {
                Self::Trapdoor { public: Arc::new(public), secret: Arc::new(secret) }
            }
            GpuCaptureOwnedOwner::IntegerValues(value) => Self::Integer(Arc::new(value)),
        }
    }

    pub(crate) fn from_integer(value: GpuFleetSignedValues) -> Self {
        Self::Integer(Arc::new(value))
    }

    pub(crate) fn from_integer_for_type(
        value: GpuFleetSignedValues,
        slot: ValueSlot,
        ty: &ResidentSlotType,
    ) -> Result<Self, ResidentControlFrameError> {
        match ty {
            ResidentSlotType::IndexedFamily { element, count, .. }
                if matches!(
                    element.as_ref(),
                    ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
                ) =>
            {
                Self::from_packed_integer_family(value, element, *count, slot)
            }
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. } => {
                Ok(Self::from_integer(value))
            }
            _ => Err(ResidentControlFrame::type_mismatch(slot, ty)),
        }
    }

    pub(crate) fn from_elements(
        element_type: &ResidentSlotType,
        elements: Vec<ResidentOwner>,
        slot: ValueSlot,
    ) -> Result<Self, ResidentControlFrameError> {
        if !elements
            .iter()
            .all(|owner| ResidentControlFrame::owner_matches_type(owner, element_type))
        {
            return Err(ResidentControlFrameError::TypeMismatch {
                slot,
                expected: "indexed family element owner",
            });
        }
        Ok(Self::IndexedFamily {
            element_type: Box::new(element_type.clone()),
            elements: elements.into(),
            packed_integer: None,
        })
    }

    fn from_packed_integer_family(
        value: GpuFleetSignedValues,
        element_type: &ResidentSlotType,
        count: usize,
        slot: ValueSlot,
    ) -> Result<Self, ResidentControlFrameError> {
        if !matches!(
            element_type,
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
        ) {
            return Err(ResidentControlFrameError::TypeMismatch {
                slot,
                expected: "non-integer indexed family owner",
            });
        }
        if value.count() != count {
            return Err(ResidentControlFrameError::TypeMismatch {
                slot,
                expected: "indexed family with the frozen element count",
            });
        }
        let backing = Arc::new(value);
        Ok(Self::IndexedFamily {
            element_type: Box::new(element_type.clone()),
            // A flat KSK family can contain tens of millions of integers.
            // Keep its packed GPU owner authoritative and materialize scalar
            // views only when an indexed operation actually requests one.
            elements: Arc::from([]),
            packed_integer: Some(backing),
        })
    }

    pub(crate) fn family_element(
        &self,
        slot: ValueSlot,
        index: usize,
    ) -> Result<ResidentOwner, ResidentControlFrameError> {
        let Self::IndexedFamily { elements, packed_integer, .. } = self else {
            return Err(ResidentControlFrameError::NotFamily(slot));
        };
        if let Some(packed) = packed_integer {
            if index >= packed.count() {
                return Err(ResidentControlFrameError::FamilyIndexOutOfBounds {
                    slot,
                    index,
                    count: packed.count(),
                });
            }
            let view = packed
                .slice(index..index + 1)
                .map_err(|error| ResidentControlFrameError::Allocation(error.to_string()))?;
            return Ok(Self::Integer(Arc::new(view)));
        }
        elements.get(index).cloned().ok_or(ResidentControlFrameError::FamilyIndexOutOfBounds {
            slot,
            index,
            count: elements.len(),
        })
    }

    pub(crate) fn packed_integer(
        &self,
        slot: ValueSlot,
    ) -> Result<&GpuFleetSignedValues, ResidentControlFrameError> {
        let Self::IndexedFamily { packed_integer: Some(value), .. } = self else {
            return Err(ResidentControlFrameError::NotInteger(slot));
        };
        Ok(value.as_ref())
    }

    /// Borrow the native integer family at the scalar integer/bool boundary.
    pub(crate) fn integer(
        &self,
        slot: ValueSlot,
    ) -> Result<&GpuFleetSignedValues, ResidentControlFrameError> {
        match self {
            Self::Integer(value) => Ok(value.as_ref()),
            Self::Matrix(_) |
            Self::SmallMatrix(_) |
            Self::Trapdoor { .. } |
            Self::IndexedFamily { .. } => Err(ResidentControlFrameError::NotInteger(slot)),
        }
    }
}

#[derive(Default)]
struct ResidentOwnerArena {
    owners: BTreeMap<ValueSlot, ResidentOwner>,
}

struct ResidentCarriedPair {
    current_slot: ValueSlot,
    next_slot: ValueSlot,
    current: ResidentOwnerArena,
    next: ResidentOwnerArena,
}

/// Current/next storage for every carried index of one sequential instruction.
/// One atomic bit controls the metadata swap for all indices, so replay never
/// observes a partially swapped carried state.
struct ResidentCarriedArena {
    pairs: BTreeMap<usize, ResidentCarriedPair>,
    current_is_next: AtomicBool,
}

impl ResidentCarriedArena {
    fn new() -> Self {
        Self { pairs: BTreeMap::new(), current_is_next: AtomicBool::new(false) }
    }

    fn current_is_next(&self) -> bool {
        self.current_is_next.load(Ordering::Acquire)
    }

    fn current(&self, index: usize) -> Option<&ResidentOwnerArena> {
        let pair = self.pairs.get(&index)?;
        Some(if self.current_is_next() { &pair.next } else { &pair.current })
    }

    fn swap(&self) {
        self.current_is_next.fetch_xor(true, Ordering::AcqRel);
    }
}

pub(crate) struct ResidentControlFrame {
    full_wave: ResidentOwnerArena,
    carried: BTreeMap<ResidentInstructionId, ResidentCarriedArena>,
    active: ResidentArenaKind,
}

impl ResidentControlFrame {
    pub(crate) fn new() -> Self {
        Self {
            full_wave: ResidentOwnerArena::default(),
            carried: BTreeMap::new(),
            active: ResidentArenaKind::FullWave,
        }
    }

    pub(crate) fn select_arena(
        &mut self,
        arena: ResidentArenaKind,
    ) -> Result<(), ResidentControlFrameError> {
        if let ResidentArenaKind::Sequential(id) = arena {
            if !self.carried.contains_key(&id) {
                return Err(ResidentControlFrameError::MissingCarriedArena(id));
            }
        }
        self.active = arena;
        Ok(())
    }

    fn allocation_encoding(ty: &ResidentSlotType) -> Option<GpuSignedValuesEncoding> {
        ty.integer_encoding()
            .or_else(|| {
                matches!(ty, ResidentSlotType::Boolean { .. })
                    .then_some(NativeIntegerEncoding::UnsignedWord)
            })
            .map(|encoding| match encoding {
                NativeIntegerEncoding::SignedWords(words) => {
                    GpuSignedValuesEncoding::SignedWords(words)
                }
                NativeIntegerEncoding::SignedWord => GpuSignedValuesEncoding::SignedI64,
                NativeIntegerEncoding::UnsignedWord => GpuSignedValuesEncoding::CanonicalU64,
            })
    }

    fn allocate_owner(
        backend: &GpuDcrtBackend,
        slot: ValueSlot,
        physical_device: i32,
        ty: &ResidentSlotType,
        physical_capacity: NonZeroUsize,
    ) -> Result<ResidentOwner, ResidentControlFrameError> {
        // A lane-parallel matrix/compact/trapdoor value is a table of
        // semantic owners, not a matrix with its columns multiplied by the
        // lane count.  Keep the semantic type in the schema and represent
        // the physical wave as an indexed owner table.  Fleet's lane
        // selector consumes this table using the scoped physical layout.
        if physical_capacity.get() > 1 &&
            matches!(
                ty,
                ResidentSlotType::Matrix { .. } |
                    ResidentSlotType::SmallMatrix { .. } |
                    ResidentSlotType::Preimage { .. } |
                    ResidentSlotType::Trapdoor { .. }
            )
        {
            let one = NonZeroUsize::new(1).expect("one is non-zero");
            let elements = (0..physical_capacity.get())
                .map(|_| Self::allocate_owner(backend, slot, physical_device, ty, one))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(ResidentOwner::IndexedFamily {
                element_type: Box::new(ty.clone()),
                elements: elements.into(),
                packed_integer: None,
            });
        }
        if let ResidentSlotType::IndexedFamily { element, count, .. } = ty {
            if !matches!(
                element.as_ref(),
                ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
            ) {
                let mut elements = Vec::with_capacity(*count);
                for _ in 0..*count {
                    elements.push(Self::allocate_owner(
                        backend,
                        slot,
                        physical_device,
                        element,
                        physical_capacity,
                    )?);
                }
                return Ok(ResidentOwner::IndexedFamily {
                    element_type: element.clone(),
                    elements: elements.into(),
                    packed_integer: None,
                });
            }
        }
        // Scalar Int/Bool owners use the same one-element native family
        // representation as the GPU integer primitives. This is the typed
        // device representation used by the native integer primitives.
        // Scalar values are semantically one value, but a parallel resident
        // phase needs one physical lane value per active wave slot.  Keep the
        // DSL type unchanged and widen only the native integer owner.  The
        // native control ABI already treats this owner as a vector, so this
        // does not turn a scalar into an indexed-family value in the schema.
        let physical_count = match ty {
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. } => {
                physical_capacity.get()
            }
            _ => 1,
        };
        let wire_type = match ty {
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. } => {
                mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                    element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                    count: physical_count,
                }
            }
            _ => ty.wire_type().clone(),
        };
        let owner = backend
            .allocate_capture_owner_on_device(
                &wire_type,
                physical_device,
                Self::allocation_encoding(ty),
            )
            .map_err(|error| ResidentControlFrameError::Allocation(error.to_string()))?;
        let owner = match ty {
            ResidentSlotType::IndexedFamily { element, count, .. }
                if matches!(
                    element.as_ref(),
                    ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
                ) =>
            {
                let GpuCaptureOwnedOwner::IntegerValues(value) = owner else {
                    return Err(Self::type_mismatch(slot, ty));
                };
                ResidentOwner::from_packed_integer_family(value, element, *count, slot)?
            }
            _ => ResidentOwner::from_capture(owner),
        };
        if !Self::owner_matches_type(&owner, ty) {
            return Err(Self::type_mismatch(slot, ty));
        }
        Ok(owner)
    }

    pub(crate) fn owner_matches_type(owner: &ResidentOwner, ty: &ResidentSlotType) -> bool {
        match ty {
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. } => {
                matches!(owner, ResidentOwner::Integer(_))
            }
            ResidentSlotType::Matrix { .. } => {
                matches!(owner, ResidentOwner::Matrix(_)) ||
                    Self::physical_family_matches(owner, ty)
            }
            ResidentSlotType::SmallMatrix { .. } | ResidentSlotType::Preimage { .. } => {
                matches!(owner, ResidentOwner::SmallMatrix(_)) ||
                    Self::physical_family_matches(owner, ty)
            }
            ResidentSlotType::Trapdoor { .. } => {
                matches!(owner, ResidentOwner::Trapdoor { .. }) ||
                    Self::physical_family_matches(owner, ty)
            }
            ResidentSlotType::IndexedFamily { element, count, .. } => {
                let ResidentOwner::IndexedFamily { element_type, elements, packed_integer } = owner
                else {
                    return false;
                };
                if element_type.as_ref() != element.as_ref() {
                    return false;
                }
                if let Some(packed) = packed_integer {
                    return matches!(
                        element.as_ref(),
                        ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
                    ) && packed.count() == *count;
                }
                elements.len() == *count &&
                    elements.iter().all(|owner| Self::owner_matches_type(owner, element))
            }
            ResidentSlotType::Real { .. } |
            ResidentSlotType::Bytes { .. } |
            ResidentSlotType::TypedBlob { .. } => false,
        }
    }

    fn physical_family_matches(owner: &ResidentOwner, element_type: &ResidentSlotType) -> bool {
        let ResidentOwner::IndexedFamily { element_type: owner_type, elements, .. } = owner else {
            return false;
        };
        owner_type.as_ref() == element_type &&
            !elements.is_empty() &&
            elements.iter().all(|element| match element_type {
                ResidentSlotType::Matrix { .. } => matches!(element, ResidentOwner::Matrix(_)),
                ResidentSlotType::SmallMatrix { .. } | ResidentSlotType::Preimage { .. } => {
                    matches!(element, ResidentOwner::SmallMatrix(_))
                }
                ResidentSlotType::Trapdoor { .. } => {
                    matches!(element, ResidentOwner::Trapdoor { .. })
                }
                _ => false,
            })
    }

    fn type_mismatch(slot: ValueSlot, ty: &ResidentSlotType) -> ResidentControlFrameError {
        ResidentControlFrameError::TypeMismatch {
            slot,
            expected: match ty {
                ResidentSlotType::Integer { .. } |
                ResidentSlotType::Boolean { .. } |
                ResidentSlotType::IndexedFamily { .. } => "integer family",
                ResidentSlotType::Matrix { .. } => "matrix",
                ResidentSlotType::SmallMatrix { .. } => "small matrix",
                ResidentSlotType::Preimage { .. } => "preimage",
                ResidentSlotType::Trapdoor { .. } => "trapdoor",
                ResidentSlotType::Real { .. } => "real",
                ResidentSlotType::Bytes { .. } => "bytes",
                ResidentSlotType::TypedBlob { .. } => "typed blob",
            },
        }
    }

    fn insert_into(
        owners: &mut ResidentOwnerArena,
        slot: ValueSlot,
        owner: ResidentOwner,
    ) -> Result<(), ResidentControlFrameError> {
        if owners.owners.contains_key(&slot) {
            return Err(ResidentControlFrameError::Duplicate(slot));
        }
        owners.owners.insert(slot, owner);
        Ok(())
    }

    pub(crate) fn insert(
        &mut self,
        slot: ValueSlot,
        owner: ResidentOwner,
    ) -> Result<(), ResidentControlFrameError> {
        Self::insert_into(&mut self.full_wave, slot, owner)
    }

    /// Allocate a run-local owner with the physical wave capacity required by
    /// the compiled layout.  `ResidentSlotType` remains the semantic DSL
    /// type; only the backing native storage is sized for the wave.
    pub(crate) fn allocate_physical(
        &mut self,
        backend: &GpuDcrtBackend,
        slot: ValueSlot,
        physical_device: i32,
        ty: &ResidentSlotType,
        physical_capacity: NonZeroUsize,
    ) -> Result<(), ResidentControlFrameError> {
        let owner = Self::allocate_owner(backend, slot, physical_device, ty, physical_capacity)?;
        Self::insert_into(&mut self.full_wave, slot, owner)
    }

    pub(crate) fn allocate_carried_pair_physical(
        &mut self,
        backend: &GpuDcrtBackend,
        instruction: ResidentInstructionId,
        carried_index: usize,
        current_slot: ValueSlot,
        next_slot: ValueSlot,
        physical_device: i32,
        current_type: &ResidentSlotType,
        next_type: &ResidentSlotType,
        physical_capacity: NonZeroUsize,
    ) -> Result<(), ResidentControlFrameError> {
        if current_slot == next_slot {
            return Err(ResidentControlFrameError::CarriedSlotCollision {
                current: current_slot,
                next: next_slot,
            });
        }
        let current = Self::allocate_owner(
            backend,
            current_slot,
            physical_device,
            current_type,
            physical_capacity,
        )?;
        let next = Self::allocate_owner(
            backend,
            next_slot,
            physical_device,
            next_type,
            physical_capacity,
        )?;
        self.insert_carried_pair(instruction, carried_index, current_slot, next_slot, current, next)
    }

    pub(crate) fn allocate_carried_pair_with_current_physical(
        &mut self,
        backend: &GpuDcrtBackend,
        instruction: ResidentInstructionId,
        carried_index: usize,
        current_slot: ValueSlot,
        next_slot: ValueSlot,
        current: ResidentOwner,
        physical_device: i32,
        current_type: &ResidentSlotType,
        next_type: &ResidentSlotType,
        physical_capacity: NonZeroUsize,
    ) -> Result<(), ResidentControlFrameError> {
        if current_slot == next_slot {
            return Err(ResidentControlFrameError::CarriedSlotCollision {
                current: current_slot,
                next: next_slot,
            });
        }
        let next = Self::allocate_owner(
            backend,
            next_slot,
            physical_device,
            next_type,
            physical_capacity,
        )?;
        if !Self::owner_matches_type(&current, current_type) {
            return Err(Self::type_mismatch(current_slot, current_type));
        }
        self.insert_carried_pair(instruction, carried_index, current_slot, next_slot, current, next)
    }

    fn insert_carried_pair(
        &mut self,
        instruction: ResidentInstructionId,
        carried_index: usize,
        current_slot: ValueSlot,
        next_slot: ValueSlot,
        current: ResidentOwner,
        next: ResidentOwner,
    ) -> Result<(), ResidentControlFrameError> {
        let arena = self.carried.entry(instruction).or_insert_with(ResidentCarriedArena::new);
        if arena.pairs.contains_key(&carried_index) {
            return Err(ResidentControlFrameError::Duplicate(current_slot));
        }
        arena.pairs.insert(
            carried_index,
            ResidentCarriedPair {
                current_slot,
                next_slot,
                current: ResidentOwnerArena {
                    owners: [(current_slot, current)].into_iter().collect(),
                },
                next: ResidentOwnerArena { owners: [(next_slot, next)].into_iter().collect() },
            },
        );
        Ok(())
    }

    /// Swap every carried index of one sequential instruction atomically.
    pub(crate) fn swap_carried(
        &self,
        instruction: ResidentInstructionId,
    ) -> Result<(), ResidentControlFrameError> {
        self.carried
            .get(&instruction)
            .ok_or(ResidentControlFrameError::MissingCarriedArena(instruction))?
            .swap();
        Ok(())
    }

    pub(crate) fn carried_slots(
        &self,
        instruction: ResidentInstructionId,
        carried_index: usize,
    ) -> Result<(ValueSlot, ValueSlot), ResidentControlFrameError> {
        let arena = self
            .carried
            .get(&instruction)
            .ok_or(ResidentControlFrameError::MissingCarriedArena(instruction))?;
        let pair = arena.pairs.get(&carried_index).ok_or(
            ResidentControlFrameError::MissingCarriedIndex { instruction, index: carried_index },
        )?;
        if arena.current_is_next() {
            Ok((pair.next_slot, pair.current_slot))
        } else {
            Ok((pair.current_slot, pair.next_slot))
        }
    }

    pub(crate) fn owner(
        &self,
        slot: ValueSlot,
    ) -> Result<&ResidentOwner, ResidentControlFrameError> {
        let owner = match self.active {
            ResidentArenaKind::FullWave => self.full_wave.owners.get(&slot),
            ResidentArenaKind::Sequential(instruction) => {
                self.carried.get(&instruction).and_then(|arena| {
                    arena
                        .current(arena.pairs.iter().find_map(|(index, pair)| {
                            (pair.current_slot == slot || pair.next_slot == slot).then_some(*index)
                        })?)
                        .and_then(|owners| owners.owners.get(&slot))
                })
            }
        };
        owner
            .or_else(|| self.full_wave.owners.get(&slot))
            .ok_or(ResidentControlFrameError::Missing(slot))
    }

    pub(crate) fn integer_owner(
        &self,
        slot: ValueSlot,
    ) -> Result<&GpuFleetSignedValues, ResidentControlFrameError> {
        self.owner(slot)?.integer(slot)
    }

    pub(crate) fn owner_any(
        &self,
        slot: ValueSlot,
    ) -> Result<&ResidentOwner, ResidentControlFrameError> {
        self.iter_all()
            .into_iter()
            .find(|(candidate, _)| *candidate == slot)
            .map(|(_, owner)| owner)
            .ok_or(ResidentControlFrameError::Missing(slot))
    }

    pub(crate) fn integer_owner_any(
        &self,
        slot: ValueSlot,
    ) -> Result<&GpuFleetSignedValues, ResidentControlFrameError> {
        self.owner_any(slot)?.integer(slot)
    }

    pub(crate) fn family_element(
        &self,
        slot: ValueSlot,
        index: usize,
    ) -> Result<ResidentOwner, ResidentControlFrameError> {
        self.owner_any(slot)?.family_element(slot, index)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (ValueSlot, &ResidentOwner)> {
        let mut entries: BTreeMap<ValueSlot, &ResidentOwner> = BTreeMap::new();
        entries.extend(self.full_wave.owners.iter().map(|(slot, owner)| (*slot, owner)));
        if let ResidentArenaKind::Sequential(instruction) = self.active {
            if let Some(arena) = self.carried.get(&instruction) {
                for pair in arena.pairs.values() {
                    let owners = if arena.current_is_next() { &pair.next } else { &pair.current };
                    entries.extend(owners.owners.iter().map(|(slot, owner)| (*slot, owner)));
                }
            }
        }
        entries.into_iter()
    }

    pub(crate) fn iter_all(&self) -> Vec<(ValueSlot, &ResidentOwner)> {
        let mut entries = Vec::new();
        entries.extend(self.full_wave.owners.iter().map(|(slot, owner)| (*slot, owner)));
        for arena in self.carried.values() {
            for pair in arena.pairs.values() {
                entries.extend(pair.current.owners.iter().map(|(slot, owner)| (*slot, owner)));
                entries.extend(pair.next.owners.iter().map(|(slot, owner)| (*slot, owner)));
            }
        }
        entries
    }
}
