use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    utils::{block_size, debug_mem, log_mem},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use tokio::task::JoinHandle;

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

#[derive(Debug, Clone)]
pub struct MultiBatchLookupBuffer {
    pub lookup_tables: Vec<BatchLookupBuffer>,
    pub total_size: usize,
}

static WRITE_HANDLES: OnceLock<Arc<Mutex<Vec<JoinHandle<()>>>>> = OnceLock::new();
static RUNTIME_HANDLE: OnceLock<tokio::runtime::Handle> = OnceLock::new();
static LOOKUP_BUFFERS: OnceLock<Arc<Mutex<MultiBatchLookupBuffer>>> = OnceLock::new();

/// Initialize the storage system
pub fn init_storage_system() {
    WRITE_HANDLES
        .set(Arc::new(Mutex::new(Vec::new())))
        .expect("Storage system already initialized");
    RUNTIME_HANDLE
        .set(tokio::runtime::Handle::current())
        .expect("Storage system already initialized");
    LOOKUP_BUFFERS
        .set(Arc::new(Mutex::new(MultiBatchLookupBuffer::new())))
        .expect("Storage system already initialized");
}

/// Wait for all pending writes to complete and write batched lookup tables
pub async fn wait_for_all_writes(
    dir_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(buffers) = LOOKUP_BUFFERS.get() {
        let multi_buffer = {
            let mut guard = buffers.lock().unwrap();
            std::mem::replace(&mut *guard, MultiBatchLookupBuffer::new())
        };

        if !multi_buffer.lookup_tables.is_empty() {
            log_mem(format!("Writing {} batched lookup tables", multi_buffer.lookup_tables.len()));
            let filename = "lookup_tables_combined.batch";
            if let Err(e) = multi_buffer.write_to_file(dir_path, filename.to_string()).await {
                eprintln!("Failed to write batched lookup tables: {}", e);
            }
        }
    }
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
    log_mem("All writes completed");
    Ok(())
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

impl MultiBatchLookupBuffer {
    pub fn new() -> Self {
        Self { lookup_tables: Vec::new(), total_size: 0 }
    }

    /// Add a batch lookup buffer to the collection.
    pub fn add_batch(&mut self, batch: BatchLookupBuffer) {
        self.total_size += batch.data.len() + 32; // Extra space for metadata.
        self.lookup_tables.push(batch);
    }

    /// Serialize all batches into a single buffer.
    /// Format:
    /// - 8 bytes: number of lookup tables (u64)
    /// - For each lookup table:
    ///   - 8 bytes: id_prefix length (u64)
    ///   - N bytes: id_prefix string
    ///   - 8 bytes: data length (u64)
    ///   - 8 bytes: num_matrices (u64)
    ///   - 8 bytes: bytes_per_matrix (u64)
    ///   - 8 bytes: indices length (u64)
    ///   - M*8 bytes: indices data
    ///   - K bytes: matrix data
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.total_size + 8);
        result.extend_from_slice(&(self.lookup_tables.len() as u64).to_le_bytes());

        for batch in &self.lookup_tables {
            let id_bytes = batch.id_prefix.as_bytes();
            result.extend_from_slice(&(id_bytes.len() as u64).to_le_bytes());
            result.extend_from_slice(id_bytes);
            result.extend_from_slice(&(batch.data.len() as u64).to_le_bytes());
            result.extend_from_slice(&(batch.num_matrices as u64).to_le_bytes());
            result.extend_from_slice(&(batch.bytes_per_matrix as u64).to_le_bytes());
            result.extend_from_slice(&(batch.indices.len() as u64).to_le_bytes());
            for &idx in &batch.indices {
                result.extend_from_slice(&(idx as u64).to_le_bytes());
            }
            result.extend_from_slice(&batch.data);
        }

        result
    }

    /// Create a future that will write all batches to a single file.
    pub fn write_to_file(
        self,
        dir: std::path::PathBuf,
        filename: String,
    ) -> impl std::future::Future<Output = Result<(), std::io::Error>> {
        async move {
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

    /// Write all batches to a single file synchronously.
    pub fn write_to_file_sync(&self, dir: &Path, filename: &str) -> std::io::Result<()> {
        let data = self.to_bytes();
        let path = dir.join(filename);
        std::fs::write(&path, &data)?;
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
pub fn add_lookup_buffer(buffer: BatchLookupBuffer) {
    if let Some(buffers) = LOOKUP_BUFFERS.get() {
        let mut guard = buffers.lock().unwrap();
        guard.add_batch(buffer);
    } else {
        eprintln!("Warning: Storage system not initialized, cannot store lookup buffer");
    }
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
    debug_mem(format!(
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
