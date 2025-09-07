use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    utils::{block_size, debug_mem, log_mem},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
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

static WRITE_HANDLES: OnceLock<Arc<Mutex<Vec<JoinHandle<()>>>>> = OnceLock::new();
static RUNTIME_HANDLE: OnceLock<tokio::runtime::Handle> = OnceLock::new();

/// Initialize the storage system
pub fn init_storage_system() {
    WRITE_HANDLES
        .set(Arc::new(Mutex::new(Vec::new())))
        .expect("Storage system already initialized");
    RUNTIME_HANDLE
        .set(tokio::runtime::Handle::current())
        .expect("Storage system already initialized");
}

/// Wait for all pending writes to complete
pub async fn wait_for_all_writes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

/// Batch serialize and store multiple matrices in a single file.
/// Returns the base filename that was created.
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
