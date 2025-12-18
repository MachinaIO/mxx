use crate::{matrix::PolyMatrix, poly::Poly, sampler::PolyTrapdoorSampler};
use dashmap::{DashMap, DashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrapdoorId(String);

impl TrapdoorId {
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TrapdoorId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TrapdoorId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl Hash for TrapdoorId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SeenKey {
    trapdoor_id: TrapdoorId,
    id: String,
}

struct PreimageRequest<M: PolyMatrix> {
    id: String,
    target: M,
    ncol: usize,
}

#[derive(Clone)]
struct TrapdoorCtx<TS>
where
    TS: PolyTrapdoorSampler,
{
    sampler: Arc<TS>,
    trapdoor: Arc<TS::Trapdoor>,
    public_matrix: Arc<TS::M>,
}

#[derive(Clone, Debug)]
pub struct FlushedPreimageBatch {
    /// Directory written for this trapdoor: `dir_path/<trapdoor_id>/`.
    pub trapdoor_dir: PathBuf,
    /// Bytes file written for this trapdoor.
    pub bytes_path: PathBuf,
    /// Metadata file written for this trapdoor.
    pub meta_path: PathBuf,
    /// Sorted request ids written under `trapdoor_dir` as:
    /// - a single `<bytes_path>` containing all compact bytes concatenated
    /// - a single `<meta_path>` containing the per-id byte ranges
    pub ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreimageSliceMetadata {
    pub start: usize,
    pub end: usize, // exclusive
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreimageBatchMetadata {
    pub format_version: u32,
    pub trapdoor_id: String,
    pub total_bytes: usize,
    /// Map: `id -> (start,end)` within `preimages.bin`.
    pub entries: HashMap<String, PreimageSliceMetadata>,
}

/// A thread-safe request pool that batches many preimage computations by concatenating target
/// matrices column-wise, computing one preimage, then slicing the result back into per-request
/// chunks.
///
/// Notes:
/// - This batching changes the joint distribution of sampled preimages across columns (they become
///   correlated). If you require independent samples per target, do not batch.
/// - `flush()` must be called after all enqueues, typically after circuit evaluation completes.
pub struct PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
{
    dir_path: PathBuf,
    contexts: DashMap<TrapdoorId, TrapdoorCtx<TS>>,
    queues: DashMap<TrapdoorId, Vec<PreimageRequest<TS::M>>>,
    seen: DashSet<SeenKey>,
    max_batch_cols: Option<usize>,
}

impl<TS> Default for PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
    TS::M: PolyMatrix,
{
    fn default() -> Self {
        Self::new(PathBuf::from("."), None)
    }
}

impl<TS> PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
    TS::M: PolyMatrix,
{
    pub fn new<P: AsRef<Path>>(dir_path: P, max_batch_cols: Option<usize>) -> Self {
        Self {
            dir_path: dir_path.as_ref().to_path_buf(),
            contexts: DashMap::new(),
            queues: DashMap::new(),
            seen: DashSet::new(),
            max_batch_cols,
        }
    }

    /// Register a (sampler, trapdoor, public matrix) tuple for a specific trapdoor id.
    /// Returns true iff it was newly inserted.
    pub fn register_trapdoor(
        &self,
        trapdoor_id: TrapdoorId,
        sampler: Arc<TS>,
        trapdoor: Arc<TS::Trapdoor>,
        public_matrix: Arc<TS::M>,
    ) -> bool {
        self.contexts
            .insert(trapdoor_id.clone(), TrapdoorCtx { sampler, trapdoor, public_matrix })
            .is_none()
    }

    /// Enqueue a target matrix to be preimaged under `trapdoor_id`.
    ///
    /// `id` identifies which segment within the eventual batched preimage corresponds to this
    /// target matrix. During `flush`, requests are sorted by `id` (ascending), targets are
    /// concatenated in that order, and the corresponding preimage slice is written to:
    /// `dir_path/<trapdoor_id>/preimages.bin` (indexed by `preimages.json`).
    ///
    /// Returns true if the request was newly enqueued (deduplicated by trapdoor_id + id).
    pub fn enqueue<S: Into<String>>(&self, trapdoor_id: &TrapdoorId, id: S, target: TS::M) -> bool {
        let id = id.into();
        let (nrow, ncol) = target.size();
        debug_assert!(nrow > 0, "target must have at least one row");
        debug_assert!(ncol > 0, "target must have at least one column");

        let seen_key = SeenKey { trapdoor_id: trapdoor_id.clone(), id: id.clone() };
        if !self.seen.insert(seen_key) {
            return false;
        }

        self.queues.entry(trapdoor_id.clone()).or_insert_with(Vec::new).push(PreimageRequest {
            id,
            target,
            ncol,
        });
        true
    }

    /// Flush all queued requests, returning a map:
    ///   `trapdoor_id -> { trapdoor_dir, ids }`.
    ///
    /// File writes are scheduled concurrently, and this function does not return until all writes
    /// (across all trapdoors) complete.
    pub async fn flush(
        &self,
        params: &<<TS::M as PolyMatrix>::P as Poly>::Params,
    ) -> HashMap<TrapdoorId, FlushedPreimageBatch> {
        let mut out: HashMap<TrapdoorId, FlushedPreimageBatch> = HashMap::new();
        let mut write_handles: Vec<tokio::task::JoinHandle<Result<(), String>>> = Vec::new();

        std::fs::create_dir_all(&self.dir_path).unwrap_or_else(|e| {
            panic!("failed to create preimage pool dir {:?}: {e}", self.dir_path)
        });

        // Drain queues. We snapshot keys first to avoid holding a map lock across expensive work.
        let trapdoor_ids: Vec<TrapdoorId> = self.queues.iter().map(|e| e.key().clone()).collect();

        for trapdoor_id in trapdoor_ids {
            // Clone context out of the DashMap guard so we can `await` later without holding locks.
            let (sampler, trapdoor, public_matrix) = {
                let ctx = self.contexts.get(&trapdoor_id).unwrap_or_else(|| {
                    panic!("trapdoor not registered for {}", trapdoor_id.as_str())
                });
                (
                    Arc::clone(&ctx.sampler),
                    Arc::clone(&ctx.trapdoor),
                    Arc::clone(&ctx.public_matrix),
                )
            };

            let pending = self.queues.remove(&trapdoor_id).map(|(_, v)| v).unwrap_or_default();
            if pending.is_empty() {
                continue;
            }

            let mut reqs = pending;
            reqs.sort_by(|a, b| a.id.cmp(&b.id));

            let total_cols: usize = reqs.iter().map(|r| r.ncol).sum();
            if let Some(limit) = self.max_batch_cols {
                assert!(
                    total_cols <= limit,
                    "preimage batch for {} has {total_cols} cols > limit {limit}; consider increasing max_batch_cols or flushing more frequently",
                    trapdoor_id.as_str()
                );
            }

            let targets_refs: Vec<&TS::M> = reqs.iter().map(|r| &r.target).collect();
            let target_all = targets_refs[0].concat_columns(&targets_refs[1..]);
            let preimage_all =
                sampler.preimage(params, trapdoor.as_ref(), public_matrix.as_ref(), &target_all);

            // 1) Enumerate per-id column ranges in a deterministic serial pass.
            let mut spans: Vec<(String, Range<usize>)> = Vec::with_capacity(reqs.len());
            let mut col_offset = 0usize;
            for req in &reqs {
                let col_start = col_offset;
                let col_end = col_offset + req.ncol;
                col_offset = col_end;
                spans.push((req.id.to_string(), col_start..col_end));
            }
            debug_assert_eq!(col_offset, total_cols, "batched column accounting mismatch");

            // 2) Convert each slice into compact bytes in parallel (order-preserving collect).
            let slice_bytes: Vec<Vec<u8>> = spans
                .par_iter()
                .map(|(_, col_range)| {
                    let slice = preimage_all.slice_columns(col_range.start, col_range.end);
                    slice.to_compact_bytes()
                })
                .collect();

            // Pack slices into a single bytes buffer and compute per-id byte ranges.
            let ids: Vec<String> = spans.iter().map(|(id, _)| id.clone()).collect();
            let total_bytes: usize = slice_bytes.iter().map(|b| b.len()).sum();
            let mut bytes: Vec<u8> = Vec::with_capacity(total_bytes);
            let mut entries: HashMap<String, PreimageSliceMetadata> =
                HashMap::with_capacity(ids.len());

            let mut offset = 0usize;
            for ((id, _), b) in spans.into_iter().zip(slice_bytes.into_iter()) {
                let start = offset;
                bytes.extend_from_slice(&b);
                offset += b.len();
                entries.insert(id, PreimageSliceMetadata { start, end: offset });
            }
            debug_assert_eq!(offset, total_bytes);

            // Write as one file per trapdoor:
            //   dir_path/<trapdoor_id>/preimages.bin
            //   dir_path/<trapdoor_id>/preimages.json
            let trapdoor_dir = self.dir_path.join(trapdoor_id.as_str());
            std::fs::create_dir_all(&trapdoor_dir).unwrap_or_else(|e| {
                panic!("failed to create trapdoor output dir {:?}: {e}", trapdoor_dir)
            });

            let bytes_path = trapdoor_dir.join("preimages.bin");
            let meta_path = trapdoor_dir.join("preimages.json");
            let bytes_path_task = bytes_path.clone();
            let meta_path_task = meta_path.clone();

            let trapdoor_id_str = trapdoor_id.as_str().to_string();
            let meta = PreimageBatchMetadata {
                format_version: 1,
                trapdoor_id: trapdoor_id_str,
                total_bytes: bytes.len(),
                entries,
            };
            let meta_bytes =
                serde_json::to_vec_pretty(&meta).expect("failed to serialize preimage metadata");

            write_handles.push(tokio::spawn(async move {
                let (w1, w2) = tokio::join!(
                    tokio::fs::write(&bytes_path_task, bytes),
                    tokio::fs::write(&meta_path_task, meta_bytes)
                );
                if let Err(e) = w1 {
                    return Err(format!(
                        "failed to write preimage bytes to {:?}: {e}",
                        bytes_path_task
                    ));
                }
                if let Err(e) = w2 {
                    return Err(format!(
                        "failed to write preimage metadata to {:?}: {e}",
                        meta_path_task
                    ));
                }
                Ok(())
            }));

            out.insert(
                trapdoor_id,
                FlushedPreimageBatch { trapdoor_dir, bytes_path, meta_path, ids },
            );
        }

        for h in write_handles {
            match h.await {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => panic!("{msg}"),
                Err(e) => panic!("preimage write task failed: {e}"),
            }
        }

        // Reset dedup state after flushing.
        self.seen.clear();

        out
    }

    /// Read a previously flushed preimage from:
    /// - `dir_path/<trapdoor_id>/preimages.json` (metadata)
    /// - `dir_path/<trapdoor_id>/preimages.bin` (compact bytes)
    pub fn read_preimage<S: AsRef<str>>(
        &self,
        params: &<<TS::M as PolyMatrix>::P as Poly>::Params,
        trapdoor_id: &TrapdoorId,
        id: S,
    ) -> TS::M {
        let id = id.as_ref();
        let trapdoor_dir = self.dir_path.join(trapdoor_id.as_str());

        let meta_path = trapdoor_dir.join("preimages.json");
        let meta_bytes = std::fs::read(&meta_path).unwrap_or_else(|e| {
            panic!("failed to read preimage metadata from {:?}: {e}", meta_path)
        });
        let meta: PreimageBatchMetadata = serde_json::from_slice(&meta_bytes).unwrap_or_else(|e| {
            panic!("failed to deserialize preimage metadata from {:?}: {e}", meta_path)
        });
        assert_eq!(meta.format_version, 1, "unsupported preimage format_version");
        assert_eq!(meta.trapdoor_id, trapdoor_id.as_str(), "preimage trapdoor_id mismatch");

        let entry = meta.entries.get(id).unwrap_or_else(|| {
            panic!("preimage id not found in metadata: {}/{}", trapdoor_id.as_str(), id)
        });
        assert!(
            entry.start <= entry.end && entry.end <= meta.total_bytes,
            "invalid byte slice range in metadata"
        );

        let bytes_path = trapdoor_dir.join("preimages.bin");
        let mut f = std::fs::File::open(&bytes_path)
            .unwrap_or_else(|e| panic!("failed to open preimage bytes file {:?}: {e}", bytes_path));
        let len = entry.end - entry.start;
        let mut buf = vec![0u8; len];
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(entry.start as u64)).unwrap_or_else(|e| {
            panic!("failed to seek preimage bytes file {:?} to {}: {e}", bytes_path, entry.start)
        });
        f.read_exact(&mut buf).unwrap_or_else(|e| {
            panic!(
                "failed to read preimage bytes file {:?} [{}, {}): {e}",
                bytes_path, entry.start, entry.end
            )
        });

        TS::M::from_compact_bytes(params, &buf)
    }
}
