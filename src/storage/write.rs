use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    utils::{block_size, debug_mem, log_mem},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use tokio::task::JoinHandle;
use tracing::info;

#[derive(Debug)]
pub struct SerializedMatrix {
    pub id: String,
    pub filename: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct BatchLookupBuffer {
    pub data: Vec<u8>,
    pub num_matrices: usize,
    pub bytes_per_matrix: usize,
    pub indices: Vec<usize>,
    pub id_prefix: String,
}

impl BatchLookupBuffer {
    /// Return the raw matrix payload (no inner header/indices).
    /// Layout of `data` produced by `get_lookup_buffer`:
    ///   [ u64 num_matrices | u64 bytes_per_matrix | u64[idx]*num_matrices | payload ]
    /// We expose only `payload`, whose length is `num_matrices * bytes_per_matrix`.
    pub fn payload(&self) -> &[u8] {
        let header_size = 16 + 8 * self.num_matrices;
        let payload_len = self.num_matrices * self.bytes_per_matrix;
        &self.data[header_size..header_size + payload_len]
    }

    /// Split this buffer into multiple smaller buffers that fit within the given size limit.
    /// Each split buffer will have a suffix added to its id_prefix to distinguish them.
    pub fn split_by_size(&self, size_limit: usize) -> Vec<BatchLookupBuffer> {
        if self.data.len() <= size_limit {
            return vec![self.clone()];
        }
        let mut result = Vec::new();
        let header_size = 16 + 8 * self.num_matrices;
        let available_data_size = size_limit.saturating_sub(header_size);

        if available_data_size == 0 {
            return vec![self.clone()];
        }
        let matrices_per_chunk = available_data_size / self.bytes_per_matrix;
        if matrices_per_chunk == 0 {
            return vec![self.clone()];
        }

        let total_matrices = self.num_matrices;
        let num_chunks = total_matrices.div_ceil(matrices_per_chunk);

        for chunk_idx in 0..num_chunks {
            let start_matrix = chunk_idx * matrices_per_chunk;
            let end_matrix = ((chunk_idx + 1) * matrices_per_chunk).min(total_matrices);
            let chunk_size = end_matrix - start_matrix;
            let chunk_indices = self.indices[start_matrix..end_matrix].to_vec();
            let chunk_header_size = 16 + 8 * chunk_size;
            let chunk_data_size = chunk_size * self.bytes_per_matrix;
            let mut chunk_data = vec![0u8; chunk_header_size + chunk_data_size];

            // Write header
            chunk_data[0..8].copy_from_slice(&(chunk_size as u64).to_le_bytes());
            chunk_data[8..16].copy_from_slice(&(self.bytes_per_matrix as u64).to_le_bytes());
            for (i, &idx) in chunk_indices.iter().enumerate() {
                let offset = 16 + i * 8;
                chunk_data[offset..offset + 8].copy_from_slice(&(idx as u64).to_le_bytes());
            }

            // matrix data
            let original_data_start = 16 + 8 * self.num_matrices;
            let chunk_data_start = chunk_header_size;
            for i in 0..chunk_size {
                let src_offset = original_data_start + (start_matrix + i) * self.bytes_per_matrix;
                let dst_offset = chunk_data_start + i * self.bytes_per_matrix;
                chunk_data[dst_offset..dst_offset + self.bytes_per_matrix]
                    .copy_from_slice(&self.data[src_offset..src_offset + self.bytes_per_matrix]);
            }

            let id_suffix =
                if num_chunks > 1 { format!("_part{}", chunk_idx) } else { String::new() };

            result.push(BatchLookupBuffer {
                data: chunk_data,
                num_matrices: chunk_size,
                bytes_per_matrix: self.bytes_per_matrix,
                indices: chunk_indices,
                id_prefix: format!("{}{}", self.id_prefix, id_suffix),
            });
        }

        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableIndexEntry {
    pub file_index: usize,
    pub file_offset: u64,
    pub num_matrices: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalTableIndex {
    pub entries: HashMap<String, TableIndexEntry>,
}

#[derive(Debug, Clone)]
pub struct MultiBatchLookupBuffer {
    pub lookup_tables: Vec<BatchLookupBuffer>,
    pub total_size: usize,
    pub bytes_limit: Option<usize>,
}

static WRITE_HANDLES: OnceLock<Arc<Mutex<Vec<JoinHandle<()>>>>> = OnceLock::new();
static RUNTIME_HANDLE: OnceLock<tokio::runtime::Handle> = OnceLock::new();

#[derive(Debug, Default)]
struct DirIndexState {
    next_file_index: usize,
    global_index: GlobalTableIndex,
}

static DIR_INDEX_STATES: OnceLock<Arc<Mutex<HashMap<PathBuf, DirIndexState>>>> = OnceLock::new();

fn lut_bytes_limit() -> Option<usize> {
    std::env::var("LUT_BYTES_LIMIT").ok().and_then(|s| s.parse::<usize>().ok())
}

/// Initialize the storage system with an optional byte limit for batched lookup tables.
pub fn init_storage_system() {
    let _ = WRITE_HANDLES.get_or_init(|| Arc::new(Mutex::new(Vec::new())));
    let _ = RUNTIME_HANDLE.get_or_init(|| tokio::runtime::Handle::current());
    let _ = DIR_INDEX_STATES.get_or_init(|| Arc::new(Mutex::new(HashMap::new())));
    info!("LUT_BYTES_LIMIT={:?}", lut_bytes_limit());
}

fn preprocess_matrix_for_storage<M>(matrix: M, id: &str) -> SerializedMatrix
where
    M: PolyMatrix + Send + 'static,
{
    let start = Instant::now();
    let block_size_val = block_size();
    let (nrow, ncol) = matrix.size();
    let row_range = 0..nrow;
    let col_range = 0..ncol;
    let entries = matrix.block_entries(row_range.clone(), col_range.clone());
    let entries_bytes: Vec<Vec<Vec<u8>>> = entries
        .par_iter()
        .map(|row| row.par_iter().map(|poly| poly.to_compact_bytes()).collect())
        .collect();
    let data = bincode::encode_to_vec(&entries_bytes, bincode::config::standard())
        .expect("Failed to serialize matrix");
    let filename = format!(
        "{}_{}_{}.{}_{}.{}.matrix",
        id, block_size_val, row_range.start, row_range.end, col_range.start, col_range.end
    );
    let elapsed = start.elapsed();
    debug_mem(format!("Serialized matrix {} len {} bytes in {elapsed:?}", id, data.len()));
    SerializedMatrix { id: id.to_string(), filename, data }
}

impl Default for MultiBatchLookupBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiBatchLookupBuffer {
    pub fn new() -> Self {
        Self { lookup_tables: Vec::new(), total_size: 0, bytes_limit: None }
    }

    pub fn with_limit(bytes_limit: usize) -> Self {
        Self { lookup_tables: Vec::new(), total_size: 0, bytes_limit: Some(bytes_limit) }
    }

    /// Add a batch lookup buffer to the collection.
    pub fn add_batch(&mut self, batch: BatchLookupBuffer) {
        self.total_size += batch.data.len() + 32;
        self.lookup_tables.push(batch);
    }

    /// Serialize all batches into a single buffer with an index for O(1) access.
    /// Format:
    /// - 8 bytes: file format version (u64) - set to 2 for indexed format
    /// - 8 bytes: number of lookup tables (u64)
    /// - 8 bytes: index section size (u64) Index section (for each lookup table):
    ///   - 8 bytes: id_prefix length (u64)
    ///   - N bytes: id_prefix string
    ///   - 8 bytes: file offset to table data (u64)
    ///
    ///     Data section (for each lookup table):
    ///   - 8 bytes: data length (u64)
    ///   - 8 bytes: num_matrices (u64)
    ///   - 8 bytes: bytes_per_matrix (u64)
    ///   - 8 bytes: indices length (u64)
    ///   - M*8 bytes: indices data
    ///   - K bytes: matrix payload (no inner header)
    pub fn to_bytes(&self) -> Vec<u8> {
        const FORMAT_VERSION: u64 = 2;

        // index size: [u64 id_len | id_bytes | u64 file_offset] per table
        let index_size: usize = self.lookup_tables.iter().map(|b| 8 + b.id_prefix.len() + 8).sum();
        let header_size = 8 + 8 + 8;
        let data_start = header_size + index_size;
        let mut current_offset = data_start;
        let mut offsets = Vec::with_capacity(self.lookup_tables.len());
        for batch in &self.lookup_tables {
            offsets.push(current_offset);
            let payload_len = batch.num_matrices * batch.bytes_per_matrix;
            let table_size = 8 + 8 + 8 + 8 + batch.indices.len() * 8 + payload_len;
            current_offset += table_size;
        }

        // allocate final buffer
        let mut result = Vec::with_capacity(current_offset);

        // header
        result.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        result.extend_from_slice(&(self.lookup_tables.len() as u64).to_le_bytes());
        result.extend_from_slice(&(index_size as u64).to_le_bytes());

        // index
        for (batch, &offset) in self.lookup_tables.iter().zip(offsets.iter()) {
            let id_bytes = batch.id_prefix.as_bytes();
            result.extend_from_slice(&(id_bytes.len() as u64).to_le_bytes());
            result.extend_from_slice(id_bytes);
            result.extend_from_slice(&(offset as u64).to_le_bytes());
        }

        // data section (metadata + indices + payload)
        for batch in &self.lookup_tables {
            let payload = batch.payload();
            result.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            result.extend_from_slice(&(batch.num_matrices as u64).to_le_bytes());
            result.extend_from_slice(&(batch.bytes_per_matrix as u64).to_le_bytes());
            result.extend_from_slice(&(batch.indices.len() as u64).to_le_bytes());
            for &idx in &batch.indices {
                result.extend_from_slice(&(idx as u64).to_le_bytes());
            }
            result.extend_from_slice(payload);
        }

        result
    }

    /// Create a future that will write all batches to a single file.
    pub async fn write_to_file(
        self,
        dir: std::path::PathBuf,
        filename: String,
    ) -> Result<(), std::io::Error> {
        let data = self.to_bytes();
        let path = dir.join(&filename);
        tokio::fs::write(&path, &data).await?;
        log_mem(format!(
            "Multi-batch lookup table written to {} ({} bytes, {} tables)",
            filename,
            data.len(),
            self.lookup_tables.len()
        ));
        Ok(())
    }
}

/// Add a batch lookup buffer to the global collection.
/// If the buffer exceeds the limit, it will be split into smaller chunks.
/// Returns true if at least one buffer was added successfully.
pub fn add_lookup_buffer(dir_path: &Path, buffer: BatchLookupBuffer) -> bool {
    let Some(dir_index_states) = DIR_INDEX_STATES.get() else {
        eprintln!("Warning: Storage system not initialized, cannot store lookup buffer");
        return false;
    };

    if let Err(e) = std::fs::create_dir_all(dir_path) {
        eprintln!(
            "Warning: failed to create storage dir {}: {}",
            dir_path.display(),
            e
        );
        return false;
    }

    let buffers = if let Some(limit) = lut_bytes_limit() {
        buffer.split_by_size(limit)
    } else {
        vec![buffer]
    };

    for split_buffer in buffers {
        let mut multi_buffer = MultiBatchLookupBuffer::new();
        multi_buffer.add_batch(split_buffer);

        let (file_index, filename, index_entries) = {
            let mut guard = dir_index_states.lock().unwrap();
            let state = guard.entry(dir_path.to_path_buf()).or_default();
            let file_index = state.next_file_index;
            state.next_file_index += 1;
            let entries = build_index_for_file(&multi_buffer, file_index);
            (file_index, format!("lookup_tables_batch_{}.bin", file_index), entries)
        };

        {
            let mut guard = dir_index_states.lock().unwrap();
            let state = guard.entry(dir_path.to_path_buf()).or_default();
            for (id_prefix, entry) in index_entries {
                state.global_index.entries.insert(id_prefix, entry);
            }
        }

        if let (Some(handles), Some(rt_handle)) = (WRITE_HANDLES.get(), RUNTIME_HANDLE.get()) {
            let dir = dir_path.to_path_buf();
            let write_handle = rt_handle.spawn(async move {
                if let Err(e) = multi_buffer.write_to_file(dir.clone(), filename.clone()).await {
                    eprintln!(
                        "Failed to write batched lookup tables {} in {}: {}",
                        filename,
                        dir.display(),
                        e
                    );
                } else {
                    debug_mem(format!(
                        "Finished write for lookup_tables_batch_{}.bin ({} tables)",
                        file_index, 1usize
                    ));
                }
            });
            handles.lock().unwrap().push(write_handle);
        } else {
            eprintln!("Warning: Storage system not initialized, falling back to blocking write");
            let path = dir_path.join(&filename);
            let data = multi_buffer.to_bytes();
            if let Err(e) = std::fs::write(&path, &data) {
                eprintln!("Failed to write {}: {}", path.display(), e);
            }
        }
    }

    true
}

/// Batch serialize and store multiple matrices in a single file.
/// Returns a BatchLookupBuffer containing the serialized data and metadata.
pub fn get_lookup_buffer<M>(preimages: Vec<(usize, M)>, id_prefix: &str) -> BatchLookupBuffer
where
    M: PolyMatrix + Send + 'static,
{
    let start = Instant::now();
    let mut sorted_preimages = preimages;
    sorted_preimages.sort_by_key(|(k, _)| *k);
    let num_matrices = sorted_preimages.len();
    let indices: Vec<usize> = sorted_preimages.iter().map(|(k, _)| *k).collect();
    let (matrix_data, max_bytes_per_matrix): (Vec<(usize, Vec<u8>)>, usize) = sorted_preimages
        .into_par_iter()
        .enumerate()
        .map(|(i, (_, matrix))| {
            let matrix_bytes = matrix.to_compact_bytes();
            let byte_len = matrix_bytes.len();
            drop(matrix);
            ((i, matrix_bytes), byte_len)
        })
        .fold(
            || (Vec::new(), 0usize),
            |(mut vec, max_len), ((i, bytes), len)| {
                vec.push((i, bytes));
                (vec, max_len.max(len))
            },
        )
        .reduce(
            || (Vec::new(), 0usize),
            |(mut vec1, max1), (vec2, max2)| {
                vec1.extend(vec2);
                (vec1, max1.max(max2))
            },
        );

    let max_bytes_per_matrix = max_bytes_per_matrix.saturating_add(16);

    // Header format:
    // - 8 bytes: number of matrices (u64)
    // - 8 bytes: bytes per matrix (u64)
    // - 8 * num_matrices bytes: indices (u64 each)
    let header_size = 16 + 8 * num_matrices;
    let total_size = header_size + max_bytes_per_matrix * num_matrices;
    let mut encoded = vec![0u8; total_size];
    encoded[0..8].copy_from_slice(&(num_matrices as u64).to_le_bytes());
    encoded[8..16].copy_from_slice(&(max_bytes_per_matrix as u64).to_le_bytes());
    for (i, &idx) in indices.iter().enumerate() {
        let offset = 16 + i * 8;
        encoded[offset..offset + 8].copy_from_slice(&(idx as u64).to_le_bytes());
    }
    for (i, matrix_bytes) in matrix_data {
        let offset = header_size + i * max_bytes_per_matrix;
        encoded[offset..offset + matrix_bytes.len()].copy_from_slice(&matrix_bytes);
    }
    let elapsed = start.elapsed();
    log_mem(format!(
        "Serialized {} matrices for {} ({} bytes, {} bytes per matrix) in {elapsed:?}",
        num_matrices, id_prefix, total_size, max_bytes_per_matrix
    ));

    // Return the buffer with metadata.
    BatchLookupBuffer {
        data: encoded,
        num_matrices,
        bytes_per_matrix: max_bytes_per_matrix,
        indices,
        id_prefix: id_prefix.to_string(),
    }
}

/// CPU preprocessing (blocking) + background I/O (non-blocking)
pub fn store_and_drop_matrix<M>(matrix: M, dir: &Path, id: &str)
where
    M: PolyMatrix + Send + 'static,
{
    let dir = dir.to_path_buf();
    let id = id.to_owned();

    let serialized_matrix = preprocess_matrix_for_storage(matrix, &id);
    // Prepare async write task
    let dir_async = dir.clone();
    let filename_async = serialized_matrix.filename.clone();
    let id_async = serialized_matrix.id.clone();
    let data_async = serialized_matrix.data.clone();
    let write_task = async move {
        let path = dir_async.join(&filename_async);
        match tokio::fs::write(&path, &data_async).await {
            Ok(_) => {
                log_mem(format!(
                    "Matrix {} written to {} ({} bytes)",
                    id_async,
                    filename_async,
                    data_async.len()
                ));
            }
            Err(e) => {
                eprintln!("Failed to write {}: {}", path.display(), e);
            }
        }
    };

    // Spawn on the captured runtime handle when available; otherwise fallback to blocking write
    if let (Some(handles), Some(rt_handle)) = (WRITE_HANDLES.get(), RUNTIME_HANDLE.get()) {
        let write_handle = rt_handle.spawn(write_task);
        handles.lock().unwrap().push(write_handle);
    } else {
        eprintln!("Warning: Storage system not initialized, falling back to blocking write");
        let path = dir.join(&serialized_matrix.filename);
        if let Err(e) = std::fs::write(&path, &serialized_matrix.data) {
            eprintln!("Failed to write {}: {}", path.display(), e);
        } else {
            log_mem(format!(
                "Matrix {} written to {} ({} bytes)",
                serialized_matrix.id,
                serialized_matrix.filename,
                serialized_matrix.data.len()
            ));
        }
    }
}

#[cfg(feature = "debug")]
pub fn store_and_drop_poly<P: Poly>(poly: P, dir: &Path, id: &str) {
    log_mem(format!("Storing {id}"));
    poly.write_to_file(dir, id);
    drop(poly);
    log_mem(format!("Stored {id}"));
}

fn build_index_for_file(
    multi_buffer: &MultiBatchLookupBuffer,
    file_index: usize,
) -> HashMap<String, TableIndexEntry> {
    let mut entries = HashMap::new();

    // header: [version | num_tables | index_size]
    let header_size = 8 + 8 + 8;

    // index size: [u64 id_len | id_bytes | u64 file_offset] per table
    let index_size: usize =
        multi_buffer.lookup_tables.iter().map(|b| 8 + b.id_prefix.len() + 8).sum();

    // first data offset
    let data_start = header_size + index_size;
    let mut current_offset = data_start;

    for batch in &multi_buffer.lookup_tables {
        entries.insert(
            batch.id_prefix.clone(),
            TableIndexEntry {
                file_index,
                file_offset: current_offset as u64,
                num_matrices: batch.num_matrices,
            },
        );

        // table size uses payload length (no inner header)
        let payload_len = batch.num_matrices * batch.bytes_per_matrix;
        let table_size = 8 + 8 + 8 + 8 +           // metadata
            batch.indices.len() * 8 + // indices
            payload_len; // payload only
        current_offset += table_size;
    }

    entries
}

/// Write the global index to a file.
async fn write_global_index(index: &GlobalTableIndex, path: &Path) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(index)?;
    tokio::fs::write(path, json.as_bytes()).await
}

/// Wait for all pending writes to complete and write batched lookup tables
pub async fn wait_for_all_writes(
    dir_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let handles = WRITE_HANDLES.get().ok_or("Storage system not initialized")?;
    let handles_vec: Vec<JoinHandle<()>> = {
        let mut guard = handles.lock().unwrap();
        std::mem::take(guard.as_mut())
    };
    log_mem(format!("Waiting for {} pending writes to complete", handles_vec.len()));
    for handle in handles_vec {
        if let Err(e) = handle.await {
            eprintln!("Write task failed: {e}");
        }
    }

    if let Some(dir_index_states) = DIR_INDEX_STATES.get() {
        let index = {
            let mut guard = dir_index_states.lock().unwrap();
            guard.remove(&dir_path).map(|s| s.global_index)
        };

        if let Some(global_index) = index {
            let index_path = dir_path.join("lookup_tables.index");
            if let Err(e) = write_global_index(&global_index, &index_path).await {
                eprintln!("Failed to write global index: {}", e);
            } else {
                log_mem(format!(
                    "Global index written with {} entries",
                    global_index.entries.len()
                ));
            }
        }
    }

    log_mem("All writes completed");
    Ok(())
}
