//! Artifacts kept in GPU memory.
//!
//! A store that keeps artifacts on the GPU receives each export as the raw
//! physical bytes the Graph copied, without leaving device memory and
//! without transcoding. An import whose destination has the same physical
//! layout is a device-to-device copy. A host reader, or an import into a
//! different layout, downloads the raw bytes and transcodes them into the
//! canonical payload on demand.

use crate::{
    artifact::{ArtifactKey, ArtifactPayload, decode_stored_payload},
    backend::poly_gpu::{PhysicalExport, transcode_raw_artifact},
    poly::dcrt::gpu::GpuDeviceMemory,
};
use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType};
use std::{collections::BTreeMap, sync::Arc};

/// One finished artifact in GPU memory.
pub struct DeviceArtifact {
    pub(crate) artifact_type: ArtifactType,
    pub(crate) availability: ArtifactAvailability,
    pub(crate) layout: Option<String>,
    pub(crate) payload_kind: u8,
    pub(crate) memory: GpuDeviceMemory,
    pub(crate) export: Arc<PhysicalExport>,
    host_reads: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for DeviceArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceArtifact")
            .field("artifact_type", &self.artifact_type)
            .field("availability", &self.availability)
            .field("raw_bytes", &self.raw_bytes())
            .field("physical_device", &self.memory.physical_device())
            .finish()
    }
}

impl DeviceArtifact {
    pub fn raw_bytes(&self) -> usize {
        self.memory.len()
    }

    /// How many times the raw bytes were downloaded for a host reader.
    pub fn host_reads(&self) -> usize {
        self.host_reads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The canonical payload: download the raw bytes and transcode them.
    pub fn host_payload(&self) -> Result<ArtifactPayload, String> {
        self.host_reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut raw = vec![0u8; self.memory.len()];
        self.memory.download(0, &mut raw).map_err(|error| error.to_string())?;
        let mut canonical = Vec::new();
        transcode_raw_artifact(&self.export, &mut std::io::Cursor::new(raw), &mut canonical)?;
        decode_stored_payload(self.payload_kind, &canonical)
    }
}

struct DeviceRawStage {
    memory: GpuDeviceMemory,
    ranges: BTreeMap<u64, u64>,
    written_bytes: u64,
}

/// Staged and finished artifacts of a store that keeps them in GPU memory.
#[derive(Clone, Default)]
pub struct DeviceArtifacts {
    stages: BTreeMap<ArtifactKey, Arc<std::sync::Mutex<DeviceRawStage>>>,
    entries: BTreeMap<ArtifactKey, Arc<DeviceArtifact>>,
}

impl std::fmt::Debug for DeviceArtifacts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceArtifacts")
            .field("staged", &self.stages.len())
            .field("artifacts", &self.entries.len())
            .field("bytes", &self.entries.values().map(|entry| entry.raw_bytes()).sum::<usize>())
            .finish()
    }
}

impl DeviceArtifacts {
    /// Copy one published chunk of `key` from device memory into its stage,
    /// allocated on the exporting device at the first chunk. Returns whether
    /// the stage is complete.
    pub(crate) fn stage_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        physical_device: i32,
        source: u64,
        bytes: usize,
    ) -> Result<bool, String> {
        let total = usize::try_from(total_raw_bytes).map_err(|_| "device artifact is too large")?;
        let end = offset
            .checked_add(bytes as u64)
            .filter(|end| *end <= total_raw_bytes)
            .ok_or("device artifact chunk exceeds its artifact")?;
        let stage = match self.stages.get(&key) {
            Some(stage) => Arc::clone(stage),
            None => {
                let memory = GpuDeviceMemory::allocate(physical_device, total)
                    .map_err(|error| error.to_string())?;
                let stage = Arc::new(std::sync::Mutex::new(DeviceRawStage {
                    memory,
                    ranges: BTreeMap::new(),
                    written_bytes: 0,
                }));
                self.stages.insert(key, Arc::clone(&stage));
                stage
            }
        };
        let mut stage = stage.lock().map_err(|_| "device artifact stage is poisoned")?;
        if stage.memory.len() != total ||
            (offset < end &&
                (stage
                    .ranges
                    .range(..=offset)
                    .next_back()
                    .is_some_and(|(_, previous_end)| *previous_end > offset) ||
                    stage.ranges.range(offset..end).next().is_some()))
        {
            return Err("device artifact chunk overlaps or changes its length".into());
        }
        stage
            .memory
            .copy_from_device(offset as usize, source, bytes)
            .map_err(|error| error.to_string())?;
        stage.ranges.insert(offset, end);
        stage.written_bytes += bytes as u64;
        Ok(stage.written_bytes == total_raw_bytes)
    }

    /// Turn the complete stage of `key` into a finished artifact.
    pub(crate) fn finish(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        export: Arc<PhysicalExport>,
    ) -> Result<(), String> {
        let stage = self.stages.remove(&key).ok_or("device artifact was never staged")?;
        let stage = Arc::try_unwrap(stage)
            .map_err(|_| "device artifact stage is still shared")?
            .into_inner()
            .map_err(|_| "device artifact stage is poisoned")?;
        if stage.written_bytes != stage.memory.len() as u64 ||
            export.raw_total_bytes != stage.written_bytes
        {
            return Err("device artifact stage is incomplete".into());
        }
        if self.entries.contains_key(&key) {
            return Err("device artifact already exists".into());
        }
        self.entries.insert(
            key,
            Arc::new(DeviceArtifact {
                artifact_type: artifact_type.clone(),
                availability,
                layout: layout.map(str::to_owned),
                payload_kind,
                memory: stage.memory,
                export,
                host_reads: Default::default(),
            }),
        );
        Ok(())
    }

    pub fn get(&self, key: &ArtifactKey) -> Option<&Arc<DeviceArtifact>> {
        self.entries.get(key)
    }

    pub(crate) fn remove(&mut self, key: &ArtifactKey) {
        self.stages.remove(key);
        self.entries.remove(key);
    }

    /// Total raw bytes of the finished artifacts.
    pub fn resident_bytes(&self) -> usize {
        self.entries.values().map(|entry| entry.raw_bytes()).sum()
    }
}
