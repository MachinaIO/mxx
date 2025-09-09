use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    utils::{block_size, debug_mem, log_mem},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

#[derive(Debug)]
pub struct SerializedMatrix {
    pub id: String,
    pub filename: String,
    pub data: Vec<u8>,
}

static WRITE_HANDLES: OnceLock<Arc<Mutex<Vec<JoinHandle<()>>>>> = OnceLock::new();
static RUNTIME_HANDLE: OnceLock<tokio::runtime::Handle> = OnceLock::new();

#[derive(Clone)]
struct BatchedLookupTable {
    lookup_id: String,
    matrices: Vec<(usize, Vec<u8>)>,
    total_bytes: usize,
    dir_path: PathBuf,
}

struct BatchCommand {
    lookup_table: BatchedLookupTable,
    response_tx: oneshot::Sender<String>,
}
static PENDING_LOOKUPS: OnceLock<Arc<Mutex<HashMap<String, String>>>> = OnceLock::new();
static BATCH_CHANNEL: OnceLock<mpsc::UnboundedSender<BatchCommand>> = OnceLock::new();
static BATCH_WORKER_HANDLE: OnceLock<JoinHandle<()>> = OnceLock::new();

/// Initialize the storage system with optional batch threshold (in bytes).
/// Default threshold is 100MB.
pub fn init_storage_system() {
    init_storage_system_with_threshold(100 * 1024 * 1024) // 100MB default.
}

/// Initialize the storage system with a custom batch threshold.
pub fn init_storage_system_with_threshold(threshold_bytes: usize) {
    WRITE_HANDLES
        .set(Arc::new(Mutex::new(Vec::new())))
        .expect("Storage system already initialized");
    RUNTIME_HANDLE
        .set(tokio::runtime::Handle::current())
        .expect("Storage system already initialized");
    PENDING_LOOKUPS
        .set(Arc::new(Mutex::new(HashMap::new())))
        .expect("Pending lookups already initialized");

    // Initialize the batch channel and worker.
    let (tx, mut rx) = mpsc::unbounded_channel::<BatchCommand>();
    BATCH_CHANNEL.set(tx).expect("Batch channel already initialized");

    let rt_handle = tokio::runtime::Handle::current();
    let pending_lookups = PENDING_LOOKUPS.get().unwrap().clone();

    let worker_handle = rt_handle.spawn(async move {
        let mut buffer: Vec<BatchedLookupTable> = Vec::new();
        let mut buffer_bytes = 0usize;
        let mut batch_counter = 0usize;
        let mut pending_responses: Vec<(String, oneshot::Sender<String>)> = Vec::new();

        loop {
            match rx.recv().await {
                Some(cmd) => {
                    let lookup_id = cmd.lookup_table.lookup_id.clone();
                    let lookup_bytes = cmd.lookup_table.total_bytes;
                    log_mem(format!(
                        "Received lookup table '{}' with {} bytes",
                        lookup_id, lookup_bytes
                    ));
                    buffer.push(cmd.lookup_table);
                    buffer_bytes += lookup_bytes;
                    pending_responses.push((lookup_id.clone(), cmd.response_tx));
                    log_mem(format!(
                        "Buffer now has {} bytes, threshold is {}",
                        buffer_bytes, threshold_bytes
                    ));

                    // Check if we should flush.
                    if buffer_bytes >= threshold_bytes {
                        let filename =
                            flush_batch(&mut buffer, &mut buffer_bytes, &mut batch_counter).await;

                        // Update all pending lookups with the batch filename.
                        let mut pending = pending_lookups.lock().unwrap();
                        for (id, tx) in pending_responses.drain(..) {
                            pending.insert(id.clone(), filename.clone());
                            let _ = tx.send(filename.clone());
                        }
                    }
                }
                None => {
                    // Channel closed, flush any remaining data.
                    if !buffer.is_empty() {
                        let filename =
                            flush_batch(&mut buffer, &mut buffer_bytes, &mut batch_counter).await;

                        // Update remaining pending lookups.
                        let mut pending = pending_lookups.lock().unwrap();
                        for (id, tx) in pending_responses.drain(..) {
                            pending.insert(id.clone(), filename.clone());
                            let _ = tx.send(filename.clone());
                        }
                    }
                    break;
                }
            }
        }
    });

    BATCH_WORKER_HANDLE.set(worker_handle).expect("Batch worker already initialized");
}

/// Flush a batch of lookup tables to disk.
async fn flush_batch(
    buffer: &mut Vec<BatchedLookupTable>,
    buffer_bytes: &mut usize,
    batch_counter: &mut usize,
) -> String {
    if buffer.is_empty() {
        return String::new();
    }

    let batch_id = *batch_counter;
    *batch_counter += 1;
    let start = Instant::now();

    // Multi-lookup table batch format:
    // Header:
    // - 8 bytes: number of lookup tables (u64)
    // - For each lookup table:
    //   - Variable: lookup_id string length (u64) + lookup_id bytes
    //   - 8 bytes: number of matrices in this lookup table (u64)
    //   - 8 bytes: offset to this lookup table's data (u64)
    //   - 8 bytes: size of this lookup table's data (u64)
    // Data section:
    // - For each lookup table:
    //   - For each matrix:
    //     - 8 bytes: matrix index (u64)
    //     - 8 bytes: matrix size (u64)
    //     - Variable: matrix bytes

    let num_lookups = buffer.len();
    let mut header = Vec::new();
    let mut data = Vec::new();
    let mut current_data_offset = 0u64;

    // Write number of lookup tables.
    header.extend_from_slice(&(num_lookups as u64).to_le_bytes());

    // Write headers and collect data for each lookup table.
    for lookup in buffer.iter() {
        // Lookup ID.
        let id_bytes = lookup.lookup_id.as_bytes();
        header.extend_from_slice(&(id_bytes.len() as u64).to_le_bytes());
        header.extend_from_slice(id_bytes);

        // Number of matrices.
        header.extend_from_slice(&(lookup.matrices.len() as u64).to_le_bytes());

        // Data offset (will be updated later).
        let offset_position = header.len();
        header.extend_from_slice(&current_data_offset.to_le_bytes());

        // Calculate and store lookup table data.
        let mut lookup_data = Vec::new();
        for (idx, matrix_bytes) in &lookup.matrices {
            lookup_data.extend_from_slice(&(*idx as u64).to_le_bytes());
            lookup_data.extend_from_slice(&(matrix_bytes.len() as u64).to_le_bytes());
            lookup_data.extend_from_slice(matrix_bytes);
        }

        // Data size.
        let data_size = lookup_data.len() as u64;
        header.extend_from_slice(&data_size.to_le_bytes());

        // Update the actual offset in header.
        let actual_offset = current_data_offset;
        header[offset_position..offset_position + 8].copy_from_slice(&actual_offset.to_le_bytes());

        current_data_offset += data_size;
        data.extend(lookup_data);
    }

    // Combine header and data.
    let mut full_data = header;
    full_data.extend(data);

    let filename = format!("multi_lookup_batch_{}.mlb", batch_id);
    let elapsed = start.elapsed();

    // Use the directory from the first lookup table (they should all be the same).
    let dir_path = buffer.first().map(|lt| lt.dir_path.clone()).unwrap_or_default();
    let full_path = dir_path.join(&filename);

    log_mem(format!(
        "Flushing batch {} with {} lookup tables, {} total bytes to {} in {:?}",
        batch_id,
        buffer.len(),
        full_data.len(),
        full_path.display(),
        elapsed
    ));

    // Write to disk asynchronously.
    let write_result = tokio::fs::write(&full_path, &full_data).await;
    if let Err(e) = write_result {
        eprintln!("Failed to write batch file {}: {}", full_path.display(), e);
    }

    buffer.clear();
    *buffer_bytes = 0;

    filename
}

/// Wait for all pending writes to complete
pub async fn wait_for_all_writes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // First, close the batch channel to signal the worker to finish.
    if let Some(sender) = BATCH_CHANNEL.get() {
        let _ = sender; // This will cause the receiver to return None.
    }

    // Wait for the batch worker to finish.
    if let Some(_worker_handle) = BATCH_WORKER_HANDLE.get() {
        // Note: We can't take ownership of the handle from OnceLock,
        // so this would need refactoring for production use.
        log_mem("Waiting for batch worker to finish");
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

/// Batch serialize and store multiple matrices using the new multi-lookup table batching system.
/// This will buffer the lookup table until the threshold is reached.
pub fn store_and_drop_matrices_batched<M>(
    preimages: Vec<(usize, M)>,
    dir: PathBuf,
    id_prefix: String,
) -> impl std::future::Future<Output = String>
where
    M: PolyMatrix + Send + 'static,
{
    async move {
        // Check if the batching system is initialized.
        let channel = match BATCH_CHANNEL.get() {
            Some(channel) => channel,
            None => {
                // Batching system not initialized, fall back to old method.
                return store_and_drop_matrices(preimages, &dir, &id_prefix);
            }
        };

        // Serialize all matrices in parallel using rayon.
        let start = Instant::now();
        let matrices: Vec<(usize, Vec<u8>)> = preimages
            .into_par_iter()
            .map(|(idx, matrix)| (idx, matrix.to_compact_bytes()))
            .collect();

        let serialize_time = start.elapsed();

        // Calculate total bytes for this lookup table.
        let total_bytes = matrices
            .iter()
            .map(|(_, bytes)| 16 + bytes.len()) // 8 bytes index + 8 bytes size + data
            .sum::<usize>();

        log_mem(format!(
            "Serialized {} matrices for lookup '{}' in {:?} ({} bytes total)",
            matrices.len(),
            id_prefix,
            serialize_time,
            total_bytes
        ));

        let lookup_table =
            BatchedLookupTable { lookup_id: id_prefix, matrices, total_bytes, dir_path: dir };
        let lookup_id = lookup_table.lookup_id.clone();
        let (response_tx, response_rx) = oneshot::channel();
        let batch_command = BatchCommand { lookup_table, response_tx };
        if let Err(_) = channel.send(batch_command) {
            return format!("{}_batch.matrices", lookup_id);
        }

        match response_rx.await {
            Ok(filename) => filename,
            Err(_) => {
                format!("{}_batch.matrices", lookup_id)
            }
        }
    }
}

/// Batch serialize and store multiple matrices in a single file.
/// Returns the base filename that was created.
/// This is the original implementation for backward compatibility.
pub fn store_and_drop_matrices<M>(preimages: Vec<(usize, M)>, dir: &Path, id_prefix: &str) -> String
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
    let filename = format!("{}_batch.matrices", id_prefix);
    let elapsed = start.elapsed();
    debug_mem(format!(
        "Serialized {} matrices into fixed-size batch file {} ({} bytes, {} bytes per matrix) in {elapsed:?}",
        num_matrices, filename, total_size, max_bytes_per_matrix
    ));

    // Write asynchronously.
    let dir_async = dir.to_path_buf();
    let filename_async = filename.clone();
    let encoded_async = encoded.clone();
    let write_task = async move {
        let path = dir_async.join(&filename_async);
        match tokio::fs::write(&path, &encoded_async).await {
            Ok(_) => {
                log_mem(format!(
                    "Batch matrices written to {} ({} bytes)",
                    filename_async,
                    encoded_async.len()
                ));
            }
            Err(e) => {
                eprintln!("Failed to write {}: {}", path.display(), e);
            }
        }
    };

    // Spawn on the captured runtime handle when available; otherwise fallback to blocking write.
    if let (Some(handles), Some(rt_handle)) = (WRITE_HANDLES.get(), RUNTIME_HANDLE.get()) {
        let write_handle = rt_handle.spawn(write_task);
        handles.lock().unwrap().push(write_handle);
    } else {
        eprintln!("Warning: Storage system not initialized, falling back to blocking write");
        let path = dir.join(&filename);
        if let Err(e) = std::fs::write(&path, &encoded) {
            eprintln!("Failed to write {}: {}", path.display(), e);
        } else {
            log_mem(format!("Batch matrices written to {} ({} bytes)", filename, encoded.len()));
        }
    }

    filename
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

/// Read a specific matrix from a multi-lookup table batch file.
/// First checks if the lookup is in a multi-lookup batch, then falls back to single batch.
pub fn read_single_matrix_from_multi_batch<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    lookup_id: &str,
    target_k: usize,
) -> Option<M>
where
    M: PolyMatrix,
{
    // Check if we have a pending lookups mapping.
    if let Some(pending) = PENDING_LOOKUPS.get() {
        let pending_map = pending.lock().unwrap();
        log_mem(format!(
            "Looking for lookup_id '{}' in pending map with {} entries",
            lookup_id,
            pending_map.len()
        ));
        if let Some(batch_filename) = pending_map.get(lookup_id) {
            // Read from the multi-lookup batch file.
            let path = dir.join(batch_filename);
            if let Ok(mut file) = File::open(&path) {
                let mut header_buf = vec![0u8; 8];
                if file.read_exact(&mut header_buf).is_ok() {
                    let num_lookups = u64::from_le_bytes(header_buf.try_into().unwrap()) as usize;

                    // Find the lookup table in the batch.
                    let mut found_lookup = None;
                    let mut header_offset = 8;

                    for _ in 0..num_lookups {
                        // Read lookup ID.
                        let mut id_len_buf = vec![0u8; 8];
                        if file.read_exact(&mut id_len_buf).is_err() {
                            break;
                        }
                        let id_len = u64::from_le_bytes(id_len_buf.try_into().unwrap()) as usize;

                        let mut id_buf = vec![0u8; id_len];
                        if file.read_exact(&mut id_buf).is_err() {
                            break;
                        }
                        let id = String::from_utf8_lossy(&id_buf);

                        // Read num matrices, data offset, and data size.
                        let mut lookup_info_buf = vec![0u8; 24];
                        if file.read_exact(&mut lookup_info_buf).is_err() {
                            break;
                        }
                        let num_matrices =
                            u64::from_le_bytes(lookup_info_buf[0..8].try_into().unwrap()) as usize;
                        let data_offset =
                            u64::from_le_bytes(lookup_info_buf[8..16].try_into().unwrap()) as usize;
                        let data_size =
                            u64::from_le_bytes(lookup_info_buf[16..24].try_into().unwrap())
                                as usize;

                        if id == lookup_id {
                            found_lookup = Some((num_matrices, data_offset, data_size));
                            break;
                        }

                        header_offset += 8 + id_len + 24;
                    }

                    if let Some((_num_matrices, data_offset, data_size)) = found_lookup {
                        // Seek to the data section.
                        let header_end = header_offset;
                        let absolute_offset = header_end + data_offset;

                        if file.seek(SeekFrom::Start(absolute_offset as u64)).is_ok() {
                            // Read matrices until we find the target index.
                            let mut current_pos = 0;
                            while current_pos < data_size {
                                let mut idx_buf = vec![0u8; 8];
                                if file.read_exact(&mut idx_buf).is_err() {
                                    break;
                                }
                                let idx = u64::from_le_bytes(idx_buf.try_into().unwrap()) as usize;

                                let mut size_buf = vec![0u8; 8];
                                if file.read_exact(&mut size_buf).is_err() {
                                    break;
                                }
                                let matrix_size =
                                    u64::from_le_bytes(size_buf.try_into().unwrap()) as usize;

                                if idx == target_k {
                                    let mut matrix_buf = vec![0u8; matrix_size];
                                    if file.read_exact(&mut matrix_buf).is_ok() {
                                        return Some(M::from_compact_bytes(params, &matrix_buf));
                                    }
                                }

                                // Skip this matrix.
                                if file.seek(SeekFrom::Current(matrix_size as i64)).is_err() {
                                    break;
                                }

                                current_pos += 16 + matrix_size;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fall back to single batch format.
    log_mem(format!("Falling back to single batch format for lookup_id '{}'", lookup_id));
    read_single_matrix_from_batch(params, dir, lookup_id, target_k)
}

/// Read a specific matrix from a batch file by its index.
/// This is more efficient than loading all matrices when only one is needed.
pub fn read_single_matrix_from_batch<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
) -> Option<M>
where
    M: PolyMatrix,
{
    let filename = format!("{}_batch.matrices", id_prefix);
    let path = dir.join(&filename);

    let start = Instant::now();
    let mut file = File::open(&path)
        .unwrap_or_else(|_| panic!("Failed to open batch matrices file {:?}", path));
    let mut header_buf = vec![0u8; 16];
    file.read_exact(&mut header_buf).expect("Failed to read header");
    let num_matrices = u64::from_le_bytes(header_buf[0..8].try_into().unwrap()) as usize;
    let bytes_per_matrix = u64::from_le_bytes(header_buf[8..16].try_into().unwrap()) as usize;
    let mut indices_buf = vec![0u8; 8 * num_matrices];
    file.read_exact(&mut indices_buf).expect("Failed to read indices");

    let mut matrix_position = None;
    for i in 0..num_matrices {
        let offset = i * 8;
        let idx = u64::from_le_bytes(indices_buf[offset..offset + 8].try_into().unwrap()) as usize;
        if idx == target_k {
            matrix_position = Some(i);
            break;
        }
    }

    match matrix_position {
        Some(pos) => {
            let header_size = 16 + 8 * num_matrices;
            let matrix_offset = header_size + pos * bytes_per_matrix;
            file.seek(SeekFrom::Start(matrix_offset as u64)).expect("Failed to seek to matrix");
            let mut matrix_buf = vec![0u8; bytes_per_matrix];
            file.read_exact(&mut matrix_buf).expect("Failed to read matrix data");
            let matrix = M::from_compact_bytes(params, &matrix_buf);
            let elapsed = start.elapsed();
            log_mem(format!(
                "Loaded matrix {} from batch file {} at position {} in {elapsed:?}",
                target_k, filename, pos
            ));
            Some(matrix)
        }
        None => {
            log_mem(format!("Matrix {} not found in batch file {}", target_k, filename));
            None
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
