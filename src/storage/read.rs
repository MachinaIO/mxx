use std::{
    fs::{File, read_to_string},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    storage::write::GlobalTableIndex,
    utils::{debug_mem, log_mem},
};

/// Read a specific matrix from split batch files by its ID prefix and index.
/// Uses indexed format for O(1) access to lookup tables.
pub fn read_matrix_from_multi_batch<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
) -> Option<M>
where
    M: PolyMatrix,
{
    let start = Instant::now();
    let index_path = dir.join("lookup_tables.index");
    if let Ok(index_data) = read_to_string(&index_path)
        && let Ok(global_index) = serde_json::from_str::<GlobalTableIndex>(&index_data)
            && let Some(entry) = global_index.entries.get(id_prefix) {
                let filename = format!("lookup_tables_batch_{}.bin", entry.file_index);
                let path = dir.join(&filename);
                if let Ok(mut file) = File::open(&path)
                    && let Some(matrix) = read_matrix_from_indexed_file_with_hint(
                        &mut file,
                        params,
                        id_prefix,
                        target_k,
                        &filename,
                        start,
                        Some(entry.file_offset),
                    ) {
                        return Some(matrix);
                    }
            }
    None
}

/// Read a specific matrix from an indexed file with an optional offset hint.
/// Optimized path when we already know the table's byte offset.
fn read_matrix_from_indexed_file_with_hint<M>(
    file: &mut File,
    params: &<M::P as Poly>::Params,
    id_prefix: &str,
    target_k: usize,
    filename: &str,
    start: Instant,
    offset_hint: Option<u64>,
) -> Option<M>
where
    M: PolyMatrix,
{
    if let Some(offset) = offset_hint
        && file.seek(SeekFrom::Start(offset)).is_ok() {
            // Read table metadata: [u64 data_len | u64 num_matrices | u64 bytes_per_matrix | u64
            // indices_len]
            let mut header = [0u8; 32];
            if file.read_exact(&mut header).is_ok() {
                let _data_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
                let _num_matrices = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
                let bytes_per_matrix =
                    u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
                let indices_len = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

                // Read indices (u64 * indices_len)
                let mut indices_data = vec![0u8; indices_len * 8];
                if file.read_exact(&mut indices_data).is_ok() {
                    // Find target_k position
                    for i in 0..indices_len {
                        let idx = u64::from_le_bytes(
                            indices_data[i * 8..(i + 1) * 8].try_into().unwrap(),
                        ) as usize;
                        if idx == target_k {
                            // Seek to the matrix payload position and read exactly bytes_per_matrix
                            let matrix_offset = i * bytes_per_matrix;
                            if file.seek(SeekFrom::Current(matrix_offset as i64)).is_ok() {
                                let mut matrix_data = vec![0u8; bytes_per_matrix];
                                if file.read_exact(&mut matrix_data).is_ok() {
                                    let matrix = M::from_compact_bytes(params, &matrix_data);
                                    log_mem(format!(
                                        "Found matrix {} with id_prefix {} in {} (using index hint) in {:?}",
                                        target_k,
                                        id_prefix,
                                        filename,
                                        start.elapsed()
                                    ));
                                    return Some(matrix);
                                }
                            }
                            return None;
                        }
                    }
                }
            }
        }

    // Fallback to full indexed read if hint path fails
    file.rewind().ok()?;
    read_matrix_from_indexed_file(file, params, id_prefix, target_k, filename, start)
}

/// Helper: read a matrix from an indexed batch file (combined or split).
/// Layout:
///   Header: [u64 version=2 | u64 num_tables | u64 index_size]
///   Index:  repeat num_tables times: [u64 id_len | id_bytes | u64 file_offset]
///   Data @ file_offset:
///     [u64 data_len | u64 num_matrices | u64 bytes_per_matrix | u64 indices_len
///      u64[idx]*indices_len | payload (num_matrices * bytes_per_matrix)]
fn read_matrix_from_indexed_file<M>(
    file: &mut File,
    params: &<M::P as Poly>::Params,
    id_prefix: &str,
    target_k: usize,
    filename: &str,
    start_time: Instant,
) -> Option<M>
where
    M: PolyMatrix,
{
    // Read global header
    let mut header_buf = [0u8; 24];
    file.read_exact(&mut header_buf).ok()?;
    let version = u64::from_le_bytes(header_buf[0..8].try_into().unwrap());
    let num_tables = u64::from_le_bytes(header_buf[8..16].try_into().unwrap()) as usize;
    let index_size = u64::from_le_bytes(header_buf[16..24].try_into().unwrap()) as usize;

    if version != 2 {
        log_mem(format!("Unsupported file version: {} in {}", version, filename));
        return None;
    }

    // Read full index
    let mut index_data = vec![0u8; index_size];
    file.read_exact(&mut index_data).ok()?;

    // Locate table offset by id
    let table_file_offset = find_table_offset_in_index(&index_data, num_tables, id_prefix)?;

    // Seek to table data and read metadata
    file.seek(SeekFrom::Start(table_file_offset)).ok()?;
    let mut metadata_buf = [0u8; 32];
    file.read_exact(&mut metadata_buf).ok()?;
    let _data_len = u64::from_le_bytes(metadata_buf[0..8].try_into().unwrap()) as usize;
    let _num_matrices = u64::from_le_bytes(metadata_buf[8..16].try_into().unwrap()) as usize;
    let bytes_per_matrix = u64::from_le_bytes(metadata_buf[16..24].try_into().unwrap()) as usize;
    let indices_len = u64::from_le_bytes(metadata_buf[24..32].try_into().unwrap()) as usize;

    // Read indices
    let mut indices_buf = vec![0u8; indices_len * 8];
    file.read_exact(&mut indices_buf).ok()?;

    // Position of the target matrix in payload
    let matrix_position = find_matrix_position(&indices_buf, indices_len, target_k)?;

    // Seek to payload + offset(i * bytes_per_matrix)
    let matrix_offset_within_data = matrix_position * bytes_per_matrix;
    file.seek(SeekFrom::Current(matrix_offset_within_data as i64)).ok()?;

    // Read matrix bytes and reconstruct
    let mut matrix_buf = vec![0u8; bytes_per_matrix];
    file.read_exact(&mut matrix_buf).ok()?;
    let matrix = M::from_compact_bytes(params, &matrix_buf);

    debug_mem(format!(
        "Loaded matrix {} from indexed batch file {} (table: {}) in {:?} [O(1) access]",
        target_k,
        filename,
        id_prefix,
        start_time.elapsed()
    ));
    Some(matrix)
}

/// Parse the index section to find the file offset for a specific table ID.
fn find_table_offset_in_index(
    index_data: &[u8],
    num_tables: usize,
    target_id: &str,
) -> Option<u64> {
    let mut offset = 0usize;
    for _ in 0..num_tables {
        if offset + 8 > index_data.len() {
            break;
        }
        let id_len =
            u64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + id_len + 8 > index_data.len() {
            break;
        }
        let id = String::from_utf8(index_data[offset..offset + id_len].to_vec()).ok()?;
        offset += id_len;
        let file_offset = u64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
        offset += 8;
        if id == target_id {
            return Some(file_offset);
        }
    }
    None
}

/// Return the position of `target_k` within the indices list.
fn find_matrix_position(indices_buf: &[u8], indices_len: usize, target_k: usize) -> Option<usize> {
    for i in 0..indices_len {
        let o = i * 8;
        let idx = u64::from_le_bytes(indices_buf[o..o + 8].try_into().unwrap()) as usize;
        if idx == target_k {
            return Some(i);
        }
    }
    None
}
