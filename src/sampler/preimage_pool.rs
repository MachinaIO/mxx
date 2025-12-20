use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    sampler::PolyTrapdoorSampler,
    storage::{
        read::read_matrix_from_multi_batch,
        write::{add_lookup_buffer, get_lookup_buffer},
    },
    utils::log_mem,
};
use dashmap::{DashMap, DashSet};
use digest::Digest;
use keccak_asm::Keccak256;
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
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
    pub trapdoor_id: TrapdoorId,
    pub id_prefix: String,
    pub ids: Vec<String>,
    pub indices: Vec<usize>,
}

fn request_index(id: &str) -> usize {
    let digest = Keccak256::digest(id.as_bytes());
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(first8) as usize
}

/// A thread-safe request pool that batches many preimage computations by concatenating target
/// matrices column-wise, computing one preimage, then slicing the result back into per-request
/// chunks.
///
/// Storage format:
/// - One lookup table per `trapdoor_id`, stored via `storage::write` using:
///   - `id_prefix = trapdoor_id`
///   - `target_k = keccak64(request.id)`
///
/// Notes:
/// - This batching changes the joint distribution of sampled preimages across columns (they become
///   correlated). If you require independent samples per target, do not batch.
/// - `flush()` must be called after all enqueues, typically after circuit evaluation completes.
/// - The storage system must be initialized via `storage::write::init_storage_system()`.
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

impl<TS> Debug for PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreimagePool").field("dir_path", &self.dir_path).finish()
    }
}

impl<TS> Default for PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
    TS::M: PolyMatrix + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new(PathBuf::from("."), None)
    }
}

impl<TS> PreimagePool<TS>
where
    TS: PolyTrapdoorSampler + Send + Sync,
    TS::M: PolyMatrix + Send + Sync + 'static,
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

    pub fn dir_path(&self) -> &Path {
        &self.dir_path
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

    /// Compute all queued preimages and enqueue them into the storage system as lookup tables.
    pub fn flush(
        &self,
        params: &<<TS::M as PolyMatrix>::P as Poly>::Params,
    ) -> HashMap<TrapdoorId, FlushedPreimageBatch> {
        let mut out: HashMap<TrapdoorId, FlushedPreimageBatch> = HashMap::new();

        // Drain queues. We snapshot keys first to avoid holding a map lock across expensive work.
        let trapdoor_ids: Vec<TrapdoorId> = self.queues.iter().map(|e| e.key().clone()).collect();

        for trapdoor_id in trapdoor_ids {
            // Clone context out of the DashMap guard so we can do expensive work without locks.
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

            // Keep large temporaries (requests/targets/preimage matrices) scoped so they can be
            // dropped before we enqueue the (much smaller) serialized lookup buffer for I/O.
            let (buffer, ids, indices) = {
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
                log_mem(format!(
                    "start preimage sampling for {} with {} cols",
                    trapdoor_id.as_str(),
                    total_cols,
                ));
                let start = std::time::Instant::now();
                let preimage_all = sampler.preimage(
                    params,
                    trapdoor.as_ref(),
                    public_matrix.as_ref(),
                    &target_all,
                );
                log_mem(format!(
                    "finished preimage sampling for {} with {} cols in {:?}",
                    trapdoor_id.as_str(),
                    total_cols,
                    start.elapsed()
                ));

                // No longer need the concatenated target.
                drop(target_all);
                drop(targets_refs);

                // Slice back into per-request matrices.
                let mut col_offset = 0usize;
                let mut used_indices: HashMap<usize, String> = HashMap::with_capacity(reqs.len());
                let mut ids: Vec<String> = Vec::with_capacity(reqs.len());
                let mut preimages: Vec<(usize, TS::M)> = Vec::with_capacity(reqs.len());
                for req in &reqs {
                    let col_start = col_offset;
                    let col_end = col_offset + req.ncol;
                    col_offset = col_end;

                    let idx = request_index(&req.id);
                    if let Some(prev) = used_indices.insert(idx, req.id.clone()) {
                        panic!(
                            "keccak index collision for trapdoor {}: {} and {} both map to {}",
                            trapdoor_id.as_str(),
                            prev,
                            req.id,
                            idx
                        );
                    }
                    ids.push(req.id.clone());

                    let slice = preimage_all.slice_columns(col_start, col_end);
                    preimages.push((idx, slice));
                }
                debug_assert_eq!(col_offset, total_cols, "batched column accounting mismatch");

                // Drop large data before serializing to bytes (reduces peak memory usage).
                drop(used_indices);
                drop(preimage_all);
                drop(reqs);

                let buffer = get_lookup_buffer(preimages, trapdoor_id.as_str());
                let indices = buffer.indices.clone();
                (buffer, ids, indices)
            };

            let ok = add_lookup_buffer(&self.dir_path, buffer);
            assert!(
                ok,
                "failed to add lookup buffer for {}; did you call init_storage_system()?",
                trapdoor_id.as_str()
            );

            let id_prefix = trapdoor_id.as_str().to_string();
            out.insert(
                trapdoor_id.clone(),
                FlushedPreimageBatch { trapdoor_id: trapdoor_id.clone(), id_prefix, ids, indices },
            );
        }

        // Reset dedup state after flushing.
        self.seen.clear();

        out
    }

    /// Read a previously flushed preimage from the storage directory using:
    /// - `id_prefix = trapdoor_id`
    /// - `target_k = keccak64(id)`
    pub fn read_preimage<S: AsRef<str>>(
        &self,
        params: &<<TS::M as PolyMatrix>::P as Poly>::Params,
        trapdoor_id: &TrapdoorId,
        id: S,
    ) -> TS::M {
        let id = id.as_ref();
        let k = request_index(id);
        read_matrix_from_multi_batch::<TS::M>(params, &self.dir_path, trapdoor_id.as_str(), k)
            .unwrap_or_else(|| {
                panic!(
                    "preimage not found; trapdoor_id: {}, id: {}, target_k: {}",
                    trapdoor_id.as_str(),
                    id,
                    k
                )
            })
    }
}
