use std::{
    fs::{File, read_to_string},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

use crate::{matrix::PolyMatrix, poly::Poly, storage::write::GlobalTableIndex};
use tracing::info;

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
        if let Some(entry) = global_index.entries.get(id_prefix) {
            if let Some(matrix) =
                read_matrix_from_entry(params, dir, id_prefix, target_k, entry, start)
            {
                return Some(matrix);
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
    let offset = entry.file_offset + (matrix_position * entry.bytes_per_matrix) as u64;
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut matrix_data = vec![0u8; entry.bytes_per_matrix];
    file.read_exact(&mut matrix_data).ok()?;
    let matrix = M::from_compact_bytes(params, &matrix_data);
    info!("{}", format!(
        "Loaded matrix {} from batch file {} (table: {}) in {:?}",
        target_k,
        filename,
        id_prefix,
        start.elapsed()
    ));
    Some(matrix)
}

/// Return the position of `target_k` within the indices list.
fn find_matrix_position(indices: &[usize], target_k: usize) -> Option<usize> {
    indices.iter().position(|&idx| idx == target_k)
}
