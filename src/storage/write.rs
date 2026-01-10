use crate::{matrix::PolyMatrix, utils::log_mem};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, mpsc},
    thread,
    time::Instant,
};
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

    /// Split this buffer into chunks that each keep the payload under the given limit.
    pub fn split_by_payload_limit(&self, payload_limit: usize) -> Vec<BatchLookupBuffer> {
        let payload_len = self.num_matrices * self.bytes_per_matrix;
        if payload_len <= payload_limit || payload_limit == 0 {
            return vec![self.clone()];
        }

        let matrices_per_chunk = payload_limit / self.bytes_per_matrix;
        if matrices_per_chunk == 0 {
            return vec![self.clone()];
        }

        let total_matrices = self.num_matrices;
        let num_chunks = total_matrices.div_ceil(matrices_per_chunk);
        let mut result = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start_matrix = chunk_idx * matrices_per_chunk;
            let end_matrix = ((chunk_idx + 1) * matrices_per_chunk).min(total_matrices);
            let chunk_size = end_matrix - start_matrix;
            let chunk_indices = self.indices[start_matrix..end_matrix].to_vec();
            let chunk_header_size = 16 + 8 * chunk_size;
            let chunk_data_size = chunk_size * self.bytes_per_matrix;
            let mut chunk_data = vec![0u8; chunk_header_size + chunk_data_size];

            chunk_data[0..8].copy_from_slice(&(chunk_size as u64).to_le_bytes());
            chunk_data[8..16].copy_from_slice(&(self.bytes_per_matrix as u64).to_le_bytes());
            for (i, &idx) in chunk_indices.iter().enumerate() {
                let offset = 16 + i * 8;
                chunk_data[offset..offset + 8].copy_from_slice(&(idx as u64).to_le_bytes());
            }

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
    pub bytes_per_matrix: usize,
    pub indices: Vec<usize>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct GlobalTableIndex {
    pub entries: HashMap<String, TableIndexEntry>,
}

enum WriteCommand {
    Append(BatchLookupBuffer),
    Flush(mpsc::Sender<Result<(), io::Error>>),
    Reset { dir_path: PathBuf, bytes_limit: Option<usize>, reply: mpsc::Sender<()> },
}

struct WriterState {
    dir_path: PathBuf,
    bytes_limit: Option<usize>,
    current_file_index: usize,
    current_file_size: usize,
    file: Option<File>,
    last_error: Option<io::Error>,
}

impl WriterState {
    fn new(dir_path: PathBuf, bytes_limit: Option<usize>) -> Self {
        Self {
            dir_path,
            bytes_limit,
            current_file_index: 0,
            current_file_size: 0,
            file: None,
            last_error: None,
        }
    }

    fn reset(&mut self, dir_path: PathBuf, bytes_limit: Option<usize>) {
        self.dir_path = dir_path;
        self.bytes_limit = bytes_limit;
        self.current_file_index = 0;
        self.current_file_size = 0;
        self.file = None;
        self.last_error = None;
    }

    fn rotate_file(&mut self) {
        self.current_file_index += 1;
        self.current_file_size = 0;
        self.file = None;
    }

    fn ensure_file(&mut self) -> Result<(), io::Error> {
        if self.file.is_none() {
            let filename = format!("lookup_tables_batch_{}.bin", self.current_file_index);
            let path = self.dir_path.join(&filename);
            let file = OpenOptions::new().create(true).write(true).truncate(true).open(&path)?;
            self.file = Some(file);
        }
        Ok(())
    }

    fn record_error(&mut self, err: io::Error) {
        self.last_error = Some(err);
    }
}

#[derive(Debug, Clone)]
pub struct MultiBatchLookupBuffer {
    pub lookup_tables: Vec<BatchLookupBuffer>,
    pub total_size: usize,
    pub bytes_limit: Option<usize>,
}

static WRITE_SENDER: OnceLock<mpsc::Sender<WriteCommand>> = OnceLock::new();
static METADATA: OnceLock<Arc<Mutex<GlobalTableIndex>>> = OnceLock::new();

/// Initialize the storage system with an optional byte limit for batched lookup tables.
pub fn init_storage_system(dir_path: PathBuf) {
    let bytes_limit = std::env::var("LUT_BYTES_LIMIT").ok().and_then(|s| s.parse::<usize>().ok());
    info!("LUT_BYTES_LIMIT={:?}", bytes_limit);
    let metadata = METADATA.get_or_init(|| Arc::new(Mutex::new(GlobalTableIndex::default())));

    if WRITE_SENDER.get().is_none() {
        let (sender, receiver) = mpsc::channel();
        let _ = WRITE_SENDER.set(sender);
        let metadata = metadata.clone();
        let dir_for_thread = dir_path.clone();
        thread::spawn(move || writer_thread_loop(receiver, metadata, dir_for_thread, bytes_limit));
    }

    if let Some(meta) = METADATA.get() {
        meta.lock().unwrap().entries.clear();
    }

    if let Some(sender) = WRITE_SENDER.get() {
        let (reply_tx, reply_rx) = mpsc::channel();
        if sender.send(WriteCommand::Reset { dir_path, bytes_limit, reply: reply_tx }).is_ok() {
            let _ = reply_rx.recv();
        }
    }
}

fn writer_thread_loop(
    receiver: mpsc::Receiver<WriteCommand>,
    metadata: Arc<Mutex<GlobalTableIndex>>,
    dir_path: PathBuf,
    bytes_limit: Option<usize>,
) {
    let mut state = WriterState::new(dir_path, bytes_limit);

    while let Ok(command) = receiver.recv() {
        match command {
            WriteCommand::Append(buffer) => {
                if let Some(limit) = state.bytes_limit {
                    let id_prefix = buffer.id_prefix.clone();
                    let chunks = buffer.split_by_payload_limit(limit);
                    if chunks.len() > 1 {
                        log_mem(format!(
                            "Split oversized lookup buffer {} into {} parts",
                            id_prefix,
                            chunks.len()
                        ));
                    }
                    for chunk in chunks {
                        write_buffer(&mut state, chunk, &metadata);
                    }
                } else {
                    write_buffer(&mut state, buffer, &metadata);
                }
            }
            WriteCommand::Flush(reply) => {
                if let Some(file) = state.file.as_mut() {
                    if let Err(err) = file.flush() {
                        state.record_error(err);
                    }
                }
                let result = state.last_error.take().map_or(Ok(()), Err);
                let _ = reply.send(result);
            }
            WriteCommand::Reset { dir_path, bytes_limit, reply } => {
                state.reset(dir_path, bytes_limit);
                let _ = reply.send(());
            }
        }
    }
}

fn write_buffer(
    state: &mut WriterState,
    buffer: BatchLookupBuffer,
    metadata: &Arc<Mutex<GlobalTableIndex>>,
) {
    let payload = buffer.payload();
    let payload_len = payload.len();

    if let Some(limit) = state.bytes_limit {
        if state.current_file_size + payload_len > limit && state.current_file_size > 0 {
            state.rotate_file();
        }
    }

    if let Err(err) = state.ensure_file() {
        state.record_error(err);
        return;
    }

    let file_offset = state.current_file_size as u64;
    let file_index = state.current_file_index;

    if let Some(file) = state.file.as_mut() {
        if let Err(err) = file.write_all(payload) {
            state.record_error(err);
            return;
        }
    }

    state.current_file_size += payload_len;

    let entry = TableIndexEntry {
        file_index,
        file_offset,
        num_matrices: buffer.num_matrices,
        bytes_per_matrix: buffer.bytes_per_matrix,
        indices: buffer.indices.clone(),
    };
    metadata.lock().unwrap().entries.insert(buffer.id_prefix, entry);
}

// /// Split a MultiBatchLookupBuffer into multiple buffers based on byte limit.
// #[allow(dead_code)]
// fn split_buffers_by_limit(
//     multi_buffer: &MultiBatchLookupBuffer,
//     bytes_limit: usize,
// ) -> Vec<MultiBatchLookupBuffer> {
//     let mut result = Vec::new();
//     let mut current_buffer = MultiBatchLookupBuffer::with_limit(bytes_limit);

//     for batch in &multi_buffer.lookup_tables {
//         let batch_size = batch.data.len();

//         // If this batch would exceed the limit and we already have some batches, start a new
// buffer         if current_buffer.would_exceed_limit(batch_size) &&
// !current_buffer.lookup_tables.is_empty()         {
//             result.push(current_buffer);
//             current_buffer = MultiBatchLookupBuffer::with_limit(bytes_limit);
//         }

//         // Add the batch to the current buffer
//         current_buffer.add_batch(batch.clone());
//     }

//     // Add the last buffer if it has content
//     if !current_buffer.lookup_tables.is_empty() {
//         result.push(current_buffer);
//     }

//     result
// }

// fn preprocess_matrix_for_storage<M>(matrix: M, id: &str) -> SerializedMatrix
// where
//     M: PolyMatrix + Send + 'static,
// {
//     let start = Instant::now();
//     let block_size_val = block_size();
//     let (nrow, ncol) = matrix.size();
//     let row_range = 0..nrow;
//     let col_range = 0..ncol;
//     let entries = matrix.block_entries(row_range.clone(), col_range.clone());
//     let entries_bytes: Vec<Vec<Vec<u8>>> = entries
//         .par_iter()
//         .map(|row| row.par_iter().map(|poly| poly.to_compact_bytes()).collect())
//         .collect();
//     let data = bincode::encode_to_vec(&entries_bytes, bincode::config::standard())
//         .expect("Failed to serialize matrix");
//     let filename = format!(
//         "{}_{}_{}.{}_{}.{}.matrix",
//         id, block_size_val, row_range.start, row_range.end, col_range.start, col_range.end
//     );
//     let elapsed = start.elapsed();
//     debug_mem(format!("Serialized matrix {} len {} bytes in {elapsed:?}", id, data.len()));
//     SerializedMatrix { id: id.to_string(), filename, data }
// }

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

    /// Check if adding a batch would exceed the byte limit.
    pub fn would_exceed_limit(&self, batch_size: usize) -> bool {
        if let Some(limit) = self.bytes_limit {
            self.total_size + batch_size + 32 > limit
        } else {
            false
        }
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
pub fn add_lookup_buffer(buffer: BatchLookupBuffer) -> bool {
    let sender = match WRITE_SENDER.get() {
        Some(sender) => sender,
        None => {
            eprintln!("Warning: Storage system not initialized, cannot store lookup buffer");
            return false;
        }
    };

    if sender.send(WriteCommand::Append(buffer)).is_ok() { true } else { false }
}

/// Batch serialize and store multiple matrices in a single file.
/// Returns a BatchLookupBuffer containing the serialized data and metadata.
pub fn get_lookup_buffer<M>(preimages: Vec<(usize, M)>, id_prefix: &str) -> BatchLookupBuffer
where
    M: PolyMatrix + Send,
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
// pub fn store_and_drop_matrix<M>(matrix: M, dir: &Path, id: &str)
// where
//     M: PolyMatrix + Send + 'static,
// {
//     let dir = dir.to_path_buf();
//     let id = id.to_owned();

//     let serialized_matrix = preprocess_matrix_for_storage(matrix, &id);
//     // Prepare async write task
//     let dir_async = dir.clone();
//     let filename_async = serialized_matrix.filename.clone();
//     let id_async = serialized_matrix.id.clone();
//     let data_async = serialized_matrix.data.clone();
//     let write_task = async move {
//         let path = dir_async.join(&filename_async);
//         match tokio::fs::write(&path, &data_async).await {
//             Ok(_) => {
//                 log_mem(format!(
//                     "Matrix {} written to {} ({} bytes)",
//                     id_async,
//                     filename_async,
//                     data_async.len()
//                 ));
//             }
//             Err(e) => {
//                 eprintln!("Failed to write {}: {}", path.display(), e);
//             }
//         }
//     };

//     // Spawn on the captured runtime handle when available; otherwise fallback to blocking write
//     if let (Some(handles), Some(rt_handle)) = (WRITE_HANDLES.get(), RUNTIME_HANDLE.get()) {
//         let write_handle = rt_handle.spawn(write_task);
//         handles.lock().unwrap().push(write_handle);
//     } else {
//         eprintln!("Warning: Storage system not initialized, falling back to blocking write");
//         let path = dir.join(&serialized_matrix.filename);
//         if let Err(e) = std::fs::write(&path, &serialized_matrix.data) {
//             eprintln!("Failed to write {}: {}", path.display(), e);
//         } else {
//             log_mem(format!(
//                 "Matrix {} written to {} ({} bytes)",
//                 serialized_matrix.id,
//                 serialized_matrix.filename,
//                 serialized_matrix.data.len()
//             ));
//         }
//     }
// }

// #[cfg(feature = "debug")]
// pub fn store_and_drop_poly<P: Poly>(poly: P, dir: &Path, id: &str) {
//     log_mem(format!("Storing {id}"));
//     poly.write_to_file(dir, id);
//     drop(poly);
//     log_mem(format!("Stored {id}"));
// }

#[allow(dead_code)]
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
                bytes_per_matrix: batch.bytes_per_matrix,
                indices: batch.indices.clone(),
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
    let sender = WRITE_SENDER.get().ok_or("Storage system not initialized")?;
    let (reply_tx, reply_rx) = mpsc::channel();
    sender.send(WriteCommand::Flush(reply_tx))?;
    match reply_rx.recv()? {
        Ok(()) => {}
        Err(e) => return Err(Box::new(e)),
    }

    if let Some(metadata) = METADATA.get() {
        let snapshot = { metadata.lock().unwrap().clone() };
        let total_indices: usize = snapshot.entries.values().map(|entry| entry.indices.len()).sum();
        let approx_bytes = total_indices.saturating_mul(std::mem::size_of::<usize>());
        log_mem(format!(
            "Metadata entries={}, total_indices={}, approx_indices_bytes={}",
            snapshot.entries.len(),
            total_indices,
            approx_bytes
        ));
        let index_path = dir_path.join("lookup_tables.index");
        if let Err(e) = write_global_index(&snapshot, &index_path).await {
            eprintln!("Failed to write global index: {}", e);
        } else {
            log_mem(format!("Global index written with {} entries", snapshot.entries.len()));
        }
        metadata.lock().unwrap().entries.clear();
    }

    log_mem("All writes completed");
    Ok(())
}
