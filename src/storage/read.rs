use std::{
    collections::HashMap,
    fs::{File, metadata, read_to_string},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, RwLock},
    time::{Instant, SystemTime},
};

use crate::{matrix::PolyMatrix, poly::Poly, storage::write::GlobalTableIndex};
use tracing::{debug, warn};

#[derive(Clone)]
struct CachedGlobalTableIndex {
    modified: Option<SystemTime>,
    len: u64,
    index: Arc<GlobalTableIndex>,
}

fn global_table_index_cache() -> &'static RwLock<HashMap<PathBuf, CachedGlobalTableIndex>> {
    static CACHE: OnceLock<RwLock<HashMap<PathBuf, CachedGlobalTableIndex>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn load_global_table_index(dir: &Path) -> Option<Arc<GlobalTableIndex>> {
    let index_path = dir.join("lookup_tables.index");
    let metadata = metadata(&index_path).ok()?;
    let modified = metadata.modified().ok();
    let len = metadata.len();

    if let Some(cached) = global_table_index_cache()
        .read()
        .expect("global table index cache read lock poisoned")
        .get(&index_path)
        .cloned() &&
        cached.modified == modified &&
        cached.len == len
    {
        debug!(
            "read_matrix_from_multi_batch: lookup_tables.index cache hit with {} entries",
            cached.index.entries.len()
        );
        return Some(cached.index);
    }

    let index_data = read_to_string(&index_path).ok()?;
    let global_index = serde_json::from_str::<GlobalTableIndex>(&index_data).ok()?;
    if global_index.entries.is_empty() {
        return None;
    }

    debug!(
        "read_matrix_from_multi_batch: lookup_tables.index loaded with {} entries",
        global_index.entries.len()
    );
    let global_index = Arc::new(global_index);
    global_table_index_cache()
        .write()
        .expect("global table index cache write lock poisoned")
        .insert(
            index_path,
            CachedGlobalTableIndex { modified, len, index: Arc::clone(&global_index) },
        );
    Some(global_index)
}

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
    if let Some(global_index) = load_global_table_index(dir) {
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
    if let Some(global_index) = load_global_table_index(dir) {
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

#[cfg(test)]
mod tests {
    use super::load_global_table_index;
    use crate::storage::write::{GlobalTableIndex, TableIndexEntry};
    use std::{collections::HashMap, fs, sync::Arc, thread, time::Duration};
    use tempfile::tempdir;

    fn write_index(
        dir: &std::path::Path,
        entries: HashMap<String, TableIndexEntry>,
    ) -> GlobalTableIndex {
        let index = GlobalTableIndex { entries };
        fs::write(
            dir.join("lookup_tables.index"),
            serde_json::to_vec(&index).expect("index json should serialize"),
        )
        .expect("index file should be written");
        index
    }

    #[test]
    fn load_global_table_index_reuses_cache_until_file_changes() {
        let dir = tempdir().expect("temporary directory should be created");
        let initial_index = write_index(
            dir.path(),
            HashMap::from([(
                "first".to_string(),
                TableIndexEntry {
                    file_index: 0,
                    file_offset: 0,
                    num_matrices: 1,
                    bytes_per_matrix: 16,
                    indices: vec![0],
                },
            )]),
        );

        let first = load_global_table_index(dir.path()).expect("initial index should load");
        let second = load_global_table_index(dir.path()).expect("cached index should load");
        assert!(Arc::ptr_eq(&first, &second), "unchanged file should reuse cached Arc");
        assert_eq!(first.entries.len(), initial_index.entries.len());

        thread::sleep(Duration::from_millis(20));
        write_index(
            dir.path(),
            HashMap::from([
                (
                    "first".to_string(),
                    TableIndexEntry {
                        file_index: 0,
                        file_offset: 0,
                        num_matrices: 1,
                        bytes_per_matrix: 16,
                        indices: vec![0],
                    },
                ),
                (
                    "second".to_string(),
                    TableIndexEntry {
                        file_index: 1,
                        file_offset: 32,
                        num_matrices: 1,
                        bytes_per_matrix: 24,
                        indices: vec![0],
                    },
                ),
            ]),
        );

        let updated = load_global_table_index(dir.path()).expect("updated index should load");
        assert!(!Arc::ptr_eq(&first, &updated), "changed file should invalidate the cached Arc");
        assert!(updated.entries.contains_key("second"));
    }
}
