use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

use crate::{
    matrix::PolyMatrix,
    poly::Poly,
    utils::{debug_mem, log_mem},
};

/// Read a specific matrix from a multi-batch file by its ID prefix and index.
/// This reads from the combined batch file that contains multiple lookup tables.
pub fn read_matrix_from_multi_batch<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    target_k: usize,
) -> Option<M>
where
    M: PolyMatrix,
{
    let filename = format!("lookup_tables_combined.batch");
    let path = dir.join(filename);
    let start = Instant::now();
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => {
            return read_single_matrix_from_batch(params, dir, id_prefix, target_k);
        }
    };

    // Read number of lookup tables
    let mut num_tables_buf = [0u8; 8];
    file.read_exact(&mut num_tables_buf).expect("Failed to read number of tables");
    let num_tables = u64::from_le_bytes(num_tables_buf) as usize;

    // Search for the matching lookup table
    for _ in 0..num_tables {
        // Read id_prefix length and string
        let mut id_len_buf = [0u8; 8];
        file.read_exact(&mut id_len_buf).expect("Failed to read id length");
        let id_len = u64::from_le_bytes(id_len_buf) as usize;

        let mut id_buf = vec![0u8; id_len];
        file.read_exact(&mut id_buf).expect("Failed to read id");
        let current_id = String::from_utf8(id_buf).expect("Invalid UTF-8");

        // Read metadata
        let mut metadata_buf = [0u8; 32]; // data_len, num_matrices, bytes_per_matrix, indices_len
        file.read_exact(&mut metadata_buf).expect("Failed to read metadata");
        let data_len = u64::from_le_bytes(metadata_buf[0..8].try_into().unwrap()) as usize;
        let num_matrices = u64::from_le_bytes(metadata_buf[8..16].try_into().unwrap()) as usize;
        let bytes_per_matrix =
            u64::from_le_bytes(metadata_buf[16..24].try_into().unwrap()) as usize;
        let indices_len = u64::from_le_bytes(metadata_buf[24..32].try_into().unwrap()) as usize;

        // Read indices
        let mut indices_buf = vec![0u8; indices_len * 8];
        file.read_exact(&mut indices_buf).expect("Failed to read indices");

        if current_id == id_prefix {
            // Found the right lookup table, now find the matrix
            let mut matrix_position = None;
            for i in 0..indices_len {
                let offset = i * 8;
                let idx = u64::from_le_bytes(indices_buf[offset..offset + 8].try_into().unwrap())
                    as usize;
                if idx == target_k {
                    matrix_position = Some(i);
                    break;
                }
            }

            match matrix_position {
                Some(pos) => {
                    // Calculate offset within the data section
                    let header_size = 16 + 8 * num_matrices; // Original batch header
                    let matrix_offset = header_size + pos * bytes_per_matrix;

                    // Seek to the matrix within the data
                    file.seek(SeekFrom::Current(matrix_offset as i64))
                        .expect("Failed to seek to matrix");
                    let mut matrix_buf = vec![0u8; bytes_per_matrix];
                    file.read_exact(&mut matrix_buf).expect("Failed to read matrix data");

                    let matrix = M::from_compact_bytes(params, &matrix_buf);
                    let elapsed = start.elapsed();
                    debug_mem(format!(
                        "Loaded matrix {} from multi-batch file (table: {}) in {elapsed:?}",
                        target_k, id_prefix
                    ));
                    return Some(matrix);
                }
                None => {
                    debug_mem(format!(
                        "Matrix {} not found in lookup table {}",
                        target_k, id_prefix
                    ));
                    return None;
                }
            }
        } else {
            // Skip this lookup table's data
            file.seek(SeekFrom::Current(data_len as i64)).expect("Failed to skip data");
        }
    }

    log_mem(format!("Lookup table {} not found in multi-batch file", id_prefix));
    None
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
