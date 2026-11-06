//! Shared host descriptors for physical GPU plans. Allocation bindings remain
//! value-local; immutable layouts and view shapes are interned per lowering.
//! Weak entries cannot keep rejected candidate metadata alive. Equality is
//! checked after hashing, so a hash collision cannot alias distinct layouts.

use super::{PhysicalEncoding, PhysicalPart};
use mxx_ir_core::types::ConcreteWireType;
use num_bigint::BigInt;
use std::{
    collections::{BTreeMap, HashMap},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut, RangeInclusive},
    sync::{Arc, Weak},
};

struct InternPool<T> {
    entries: HashMap<u64, Vec<Weak<T>>>,
}

impl<T> Default for InternPool<T> {
    fn default() -> Self {
        Self { entries: HashMap::new() }
    }
}

impl<T: Eq + Hash> InternPool<T> {
    fn intern(&mut self, value: T) -> Arc<T> {
        self.intern_arc(Arc::new(value))
    }

    fn intern_arc(&mut self, value: Arc<T>) -> Arc<T> {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hash);
        let bucket = self.entries.entry(hash.finish()).or_default();
        let mut found = None;
        bucket.retain(|entry| {
            let Some(existing) = entry.upgrade() else {
                return false;
            };
            if *existing == *value {
                found = Some(existing);
            }
            true
        });
        if let Some(existing) = found {
            return existing;
        }
        bucket.push(Arc::downgrade(&value));
        value
    }
}

/// One pool per candidate, shared by nested lowering through PhysicalOwners.
/// No device allocations, pointers, or cross-candidate bindings are cached.
#[derive(Default)]
pub(crate) struct PhysicalMetadataPool {
    layouts: InternPool<PhysicalValueLayout>,
    shapes: InternPool<PhysicalViewShape>,
}

/// Constructor input; runtime descriptors store the shape separately from the
/// offset, so independent lane allocations can share every dimension/stride.
pub(crate) struct PhysicalViewData {
    pub byte_offset: u64,
    pub origin: Box<[u64]>,
    pub extent: Box<[u64]>,
    pub byte_strides: Box<[u64]>,
    pub element_bytes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PhysicalViewShape {
    pub origin: Box<[u64]>,
    pub extent: Box<[u64]>,
    pub byte_strides: Box<[u64]>,
    pub element_bytes: u32,
}

/// Coordinates name logical elements. The address is `base + byte_offset +
/// sum((j[a] - origin[a]) * byte_strides[a])`. Shape edits detach just the shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalView {
    pub byte_offset: u64,
    shape: Arc<PhysicalViewShape>,
}

impl PhysicalView {
    pub(crate) fn new(data: PhysicalViewData) -> Self {
        Self {
            byte_offset: data.byte_offset,
            shape: Arc::new(PhysicalViewShape {
                origin: data.origin,
                extent: data.extent,
                byte_strides: data.byte_strides,
                element_bytes: data.element_bytes,
            }),
        }
    }
}

impl Deref for PhysicalView {
    type Target = PhysicalViewShape;
    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}
impl DerefMut for PhysicalView {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.shape)
    }
}

pub(crate) struct PhysicalValueData {
    pub ty: ConcreteWireType,
    pub encodings: Box<[PhysicalEncoding]>,
    pub parts: Box<[PhysicalPart]>,
    pub integer_ranges: BTreeMap<u32, RangeInclusive<BigInt>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PhysicalValueLayout {
    pub ty: ConcreteWireType,
    pub encodings: Box<[PhysicalEncoding]>,
    pub integer_ranges: BTreeMap<u32, RangeInclusive<BigInt>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalValueStorage {
    layout: Arc<PhysicalValueLayout>,
    pub parts: Box<[PhysicalPart]>,
}
impl Deref for PhysicalValueStorage {
    type Target = PhysicalValueLayout;
    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}
impl DerefMut for PhysicalValueStorage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.layout)
    }
}

/// The plan table and resident owner share the complete descriptor. Different
/// lanes share layouts and shapes while retaining separate storage references.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalValue(Arc<PhysicalValueStorage>);

impl PhysicalValue {
    pub(crate) fn new(data: PhysicalValueData) -> Self {
        Self::from_layout(data, None)
    }

    pub(crate) fn intern(data: PhysicalValueData, pool: &mut PhysicalMetadataPool) -> Self {
        Self::from_layout(data, Some(pool))
    }

    pub(crate) fn from_layout(
        mut data: PhysicalValueData,
        pool: Option<&mut PhysicalMetadataPool>,
    ) -> Self {
        let layout = PhysicalValueLayout {
            ty: data.ty,
            encodings: data.encodings,
            integer_ranges: data.integer_ranges,
        };
        let layout = if let Some(pool) = pool {
            for part in &mut data.parts {
                part.view.shape = pool.shapes.intern_arc(Arc::clone(&part.view.shape));
            }
            pool.layouts.intern(layout)
        } else {
            Arc::new(layout)
        };
        Self(Arc::new(PhysicalValueStorage { layout, parts: data.parts }))
    }

    pub(crate) fn to_data(&self) -> PhysicalValueData {
        PhysicalValueData {
            ty: self.ty.clone(),
            encodings: self.encodings.clone(),
            parts: self.parts.clone(),
            integer_ranges: self.integer_ranges.clone(),
        }
    }
}
impl Deref for PhysicalValue {
    type Target = PhysicalValueStorage;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for PhysicalValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_execution_plan::StorageRef;

    fn lane_data(lane: u32) -> PhysicalValueData {
        PhysicalValueData {
            ty: ConcreteWireType::Bytes { length: 128 },
            encodings: Box::new([PhysicalEncoding::Bytes]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage: StorageRef::Scratch(lane),
                device: 0,
                view: PhysicalView::new(super::PhysicalViewData {
                    byte_offset: u64::from(lane) * 128,
                    origin: Box::new([0]),
                    extent: Box::new([128]),
                    byte_strides: Box::new([1]),
                    element_bytes: 1,
                }),
            }]),
            integer_ranges: BTreeMap::new(),
        }
    }

    #[test]
    fn test_gpu_metadata_lanes_share_shapes_without_aliasing_bindings() {
        let mut pool = PhysicalMetadataPool::default();
        let lanes: Vec<_> =
            (0..64).map(|lane| PhysicalValue::intern(lane_data(lane), &mut pool)).collect();
        for (lane, value) in lanes.iter().enumerate() {
            assert!(Arc::ptr_eq(&lanes[0].layout, &value.layout));
            assert!(Arc::ptr_eq(&lanes[0].parts[0].view.shape, &value.parts[0].view.shape));
            assert_eq!(value.parts[0].storage, StorageRef::Scratch(lane as u32));
            assert_eq!(value.parts[0].view.byte_offset, lane as u64 * 128);
            value.validate(|_| Some((0, 8192))).unwrap();
        }
        let mut edited = lanes[7].clone();
        assert!(Arc::ptr_eq(&edited.0, &lanes[7].0));
        edited.parts[0].storage = StorageRef::Output(0);
        edited.parts[0].view.byte_offset = 0;
        edited.parts[0].view.extent[0] = 64;
        edited.ty = ConcreteWireType::Bytes { length: 64 };
        edited.integer_ranges.insert(0, BigInt::from(1)..=BigInt::from(2));
        edited.encodings[0] = PhysicalEncoding::BoolI64;
        assert_eq!(lanes[7].ty, ConcreteWireType::Bytes { length: 128 });
        assert_eq!(lanes[7].parts[0].view.extent[0], 128);
        assert_eq!(lanes[7].parts[0].storage, StorageRef::Scratch(7));
        assert_eq!(lanes[7].parts[0].view.byte_offset, 7 * 128);
        assert!(lanes[7].integer_ranges.is_empty());
        assert_eq!(lanes[7].encodings[0], PhysicalEncoding::Bytes);
    }

    #[test]
    fn test_gpu_metadata_pool_does_not_retain_discarded_candidates() {
        let mut pool = PhysicalMetadataPool::default();
        let value = PhysicalValue::intern(lane_data(0), &mut pool);
        let layout = Arc::downgrade(&value.layout);
        let shape = Arc::downgrade(&value.parts[0].view.shape);
        drop(value);
        assert!(layout.upgrade().is_none());
        assert!(shape.upgrade().is_none());
        let next = PhysicalValue::intern(lane_data(1), &mut pool);
        assert_eq!(pool.layouts.entries.values().map(Vec::len).sum::<usize>(), 1);
        assert_eq!(pool.shapes.entries.values().map(Vec::len).sum::<usize>(), 1);
        assert_eq!(next.parts[0].storage, StorageRef::Scratch(1));
    }

    #[test]
    fn test_gpu_metadata_hash_collisions_preserve_distinct_descriptors() {
        #[derive(Eq, PartialEq)]
        struct Colliding(u32);
        impl Hash for Colliding {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0u8.hash(state);
            }
        }
        let mut pool = InternPool::default();
        let first = pool.intern(Colliding(1));
        let second = pool.intern(Colliding(2));
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first, &pool.intern(Colliding(1))));
    }
}
