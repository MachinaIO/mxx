use std::{
    fs::{File, read_to_string},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

use crate::{matrix::PolyMatrix, poly::Poly, storage::write::GlobalTableIndex};
use tracing::{debug, warn};

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
    if let Ok(index_data) = read_to_string(&index_path) &&
        let Ok(global_index) = serde_json::from_str::<GlobalTableIndex>(&index_data) &&
        !global_index.entries.is_empty()
    {
        debug!(
            "read_matrix_from_multi_batch: lookup_tables.index loaded with {} entries",
            global_index.entries.len()
        );
        if let Some(entry) = global_index.entries.get(id_prefix) {
            if let Some(matrix) =
                read_matrix_from_entry(params, dir, id_prefix, target_k, entry, start)
            {
                return Some(matrix);
            } else {
                debug!(
                    "read_matrix_from_multi_batch: matrix not found in entry for id_prefix {}",
                    id_prefix
                );
            }
        }
        let part_prefix = format!("{}_part", id_prefix);
        for (key, entry) in &global_index.entries {
            if key.starts_with(&part_prefix) {
                if let Some(matrix) =
                    read_matrix_from_entry(params, dir, key, target_k, entry, start)
                {
                    return Some(matrix);
                }
            }
        }
    }
    None
}

pub fn read_bytes_from_multi_batch(
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
) -> Option<Vec<u8>> {
    let index_path = dir.join("lookup_tables.index");
    if let Ok(index_data) = read_to_string(&index_path) &&
        let Ok(global_index) = serde_json::from_str::<GlobalTableIndex>(&index_data) &&
        !global_index.entries.is_empty()
    {
        if let Some(entry) = global_index.entries.get(id_prefix) {
            if let Some(bytes) = read_bytes_from_entry(dir, id_prefix, target_k, entry) {
                return Some(bytes);
            }
        }
        let part_prefix = format!("{}_part", id_prefix);
        for (key, entry) in &global_index.entries {
            if key.starts_with(&part_prefix) {
                if let Some(bytes) = read_bytes_from_entry(dir, key, target_k, entry) {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

fn read_matrix_from_entry<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
    entry: &crate::storage::write::TableIndexEntry,
    start: Instant,
) -> Option<M>
where
    M: PolyMatrix,
{
    let matrix_position = find_matrix_position(&entry.indices, target_k)?;
    let filename = format!("lookup_tables_batch_{}.bin", entry.file_index);
    let path = dir.join(&filename);
    let mut file = File::open(&path).ok()?;
    debug!("read_matrix_from_entry: opened file {} for id_prefix {}", filename, id_prefix);
    let offset = entry.file_offset + (matrix_position * entry.bytes_per_matrix) as u64;
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut matrix_data = vec![0u8; entry.bytes_per_matrix];
    file.read_exact(&mut matrix_data).ok()?;
    let matrix = M::from_compact_bytes(params, &matrix_data);
    debug!(
        "{}",
        format!(
            "Loaded matrix {} from batch file {} (table: {}) in {:?}",
            target_k,
            filename,
            id_prefix,
            start.elapsed()
        )
    );
    Some(matrix)
}

fn read_bytes_from_entry(
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
    entry: &crate::storage::write::TableIndexEntry,
) -> Option<Vec<u8>> {
    let matrix_position = find_matrix_position(&entry.indices, target_k)?;
    let filename = format!("lookup_tables_batch_{}.bin", entry.file_index);
    let path = dir.join(&filename);
    let mut file = File::open(&path).ok()?;
    if entry.bytes_per_matrix == 0 {
        warn!("Invalid bytes_per_matrix=0 for {} (table: {})", filename, id_prefix);
        return None;
    }
    let offset = entry.file_offset + (matrix_position * entry.bytes_per_matrix) as u64;
    let file_len = file.metadata().ok()?.len();
    let end = offset + entry.bytes_per_matrix as u64;
    if end > file_len {
        warn!(
            "Bytes out of bounds for {} (table: {}): offset {} + size {} > file_len {}",
            filename, id_prefix, offset, entry.bytes_per_matrix, file_len
        );
        return None;
    }
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut data = vec![0u8; entry.bytes_per_matrix];
    file.read_exact(&mut data).ok()?;
    Some(data)
}

/// Return the position of `target_k` within the indices list.
fn find_matrix_position(indices: &[usize], target_k: usize) -> Option<usize> {
    indices.iter().position(|&idx| idx == target_k)
}
